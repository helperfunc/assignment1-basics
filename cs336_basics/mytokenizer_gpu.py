import regex as re
from collections import defaultdict
from typing import List, Dict, Tuple
import os
from typing import BinaryIO
import multiprocessing as mp
import json
import base64
import cProfile
import pstats
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
TOKEN_RE = re.compile(PAT)
SINGLE_BYTES = [bytes([i]) for i in range(256)]

def find_chunk_boundaries(file, desired_num_chunks: int, split_special_token: bytes) -> List[int]:
    assert isinstance(split_special_token, bytes)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    if desired_num_chunks <= 1 or file_size == 0:
        return [0, file_size]
    chunk_size = max(1, file_size // desired_num_chunks)
    boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    boundaries[-1] = file_size
    mini_chunk_size = 4096
    for bi in range(1, len(boundaries) - 1):
        initial = boundaries[bi]
        file.seek(initial)
        while True:
            buf = file.read(mini_chunk_size)
            if buf == b"":
                boundaries[bi] = file_size
                break
            pos = buf.find(split_special_token)
            if pos != -1:
                boundaries[bi] = initial + pos
                break
            initial += mini_chunk_size
    return sorted(set(boundaries))

def _process_file_chunk(args):
    start, end, input_path, special_tokens = args
    with open(input_path, 'rb') as f:
        f.seek(start)
        data = f.read(end - start)
    text = data.decode("utf-8", errors="ignore")
    # Normalize Windows newlines so parallel path matches serial path snapshot expectations
    if '\r' in text:
        text = text.replace('\r\n', '\n').replace('\r', '\n')
    return _init_sequences(text, special_tokens)

def _init_sequences_parallel(input_path: str, special_tokens: List[str], num_processes: int) -> List[List[bytes]]:
    if num_processes <= 1:
        with open(input_path, "r", encoding="utf-8") as f:
            corpus = f.read()
        return _init_sequences(corpus, special_tokens)
    split_token = None
    if special_tokens:
        if "<|endoftext|>" in special_tokens:
            split_token = "<|endoftext|>"
        else:
            split_token = special_tokens[0]
    if split_token is None:
        return _init_sequences_parallel(input_path, special_tokens, 1)
    with open(input_path, "rb") as fb:
        boundaries = find_chunk_boundaries(fb, num_processes, split_token.encode("utf-8"))
    if len(boundaries) <= 2:
        return _init_sequences_parallel(input_path, special_tokens, 1)
    tasks = [(start, end, input_path, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:]) if end > start]
    with mp.Pool(processes=min(num_processes, len(tasks))) as pool:
        parts = pool.map(_process_file_chunk, tasks)
    out = []
    for p in parts:
        out.extend(p)
    return out

def _pre_tokenize(text: str):
    for m in TOKEN_RE.finditer(text):
        yield m.group()

def _compile_special_pattern(special_tokens: List[str]) -> re.Pattern | None:
    if not special_tokens:
        return None
    parts = [re.escape(t) for t in sorted(special_tokens, key=len, reverse=True)]
    return re.compile("(" + "|".join(parts) + ")")

def _init_sequences(corpus: str, special_tokens: List[str]) -> List[List[bytes]]:
    sequences = []
    special_pat = _compile_special_pattern(special_tokens)
    if special_pat:
        parts = special_pat.split(corpus)
        for i, part in enumerate(parts):
            if i % 2 == 1:
                continue
            if not part:
                continue
            for tok in _pre_tokenize(part):
                b = tok.encode('utf-8')
                sequences.append([SINGLE_BYTES[x] for x in b])
    else:
        for tok in _pre_tokenize(corpus):
            b = tok.encode('utf-8')
            sequences.append([SINGLE_BYTES[x] for x in b])
    return sequences

class GPUBPETrainer:
    def __init__(self, device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def bytes_to_int(self, byte_seq: bytes) -> int:
        """Convert bytes to integer for tensor operations"""
        return int.from_bytes(byte_seq, byteorder='big', signed=False)
    
    def int_to_bytes(self, int_val: int, length: int) -> bytes:
        """Convert integer back to bytes"""
        return int_val.to_bytes(length, byteorder='big', signed=False)
    
    def prepare_sequences_gpu(self, sequences: List[List[bytes]], freqs: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert sequences to GPU tensors for fast processing"""
        # Convert bytes to integers for GPU processing
        max_seq_len = max(len(seq) for seq in sequences) if sequences else 0
        
        # Create tensors
        seq_tensors = []
        seq_lengths = []
        
        for seq in sequences:
            # Convert each byte to its integer value (0-255)
            int_seq = [byte_seq[0] if len(byte_seq) == 1 else self.bytes_to_int(byte_seq) for byte_seq in seq]
            seq_tensors.append(torch.tensor(int_seq, dtype=torch.long))
            seq_lengths.append(len(int_seq))
        
        # Pad sequences to same length
        if seq_tensors:
            padded_sequences = pad_sequence(seq_tensors, batch_first=True, padding_value=-1)
        else:
            padded_sequences = torch.empty((0, 0), dtype=torch.long)
            
        seq_lengths_tensor = torch.tensor(seq_lengths, dtype=torch.long)
        freqs_tensor = torch.tensor(freqs, dtype=torch.long)
        
        # Move to GPU
        padded_sequences = padded_sequences.to(self.device)
        seq_lengths_tensor = seq_lengths_tensor.to(self.device)
        freqs_tensor = freqs_tensor.to(self.device)
        
        return padded_sequences, seq_lengths_tensor, freqs_tensor
    
    def count_pairs_gpu(self, sequences: torch.Tensor, lengths: torch.Tensor, freqs: torch.Tensor) -> Dict[Tuple[int, int], int]:
        """GPU-accelerated pair counting"""
        pair_counts = defaultdict(int)
        
        # Process in batches to manage memory
        batch_size = min(1000, sequences.size(0))
        
        for i in range(0, sequences.size(0), batch_size):
            batch_end = min(i + batch_size, sequences.size(0))
            batch_seqs = sequences[i:batch_end]
            batch_lengths = lengths[i:batch_end]
            batch_freqs = freqs[i:batch_end]
            
            # Extract pairs for each sequence in batch
            for j in range(batch_seqs.size(0)):
                seq = batch_seqs[j]
                length = batch_lengths[j]
                freq = batch_freqs[j]
                
                if length < 2:
                    continue
                    
                # Get consecutive pairs
                for k in range(length - 1):
                    if seq[k] != -1 and seq[k+1] != -1:  # Skip padding
                        pair = (seq[k].item(), seq[k+1].item())
                        pair_counts[pair] += freq.item()
        
        return pair_counts
    
    def merge_sequences_gpu(self, sequences: torch.Tensor, lengths: torch.Tensor, 
                           old_pair: Tuple[int, int], new_token: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """GPU-accelerated sequence merging"""
        a, b = old_pair
        new_sequences = []
        new_lengths = []
        
        # Process each sequence
        for i in range(sequences.size(0)):
            seq = sequences[i]
            length = lengths[i]
            
            if length < 2:
                new_sequences.append(seq)
                new_lengths.append(length)
                continue
            
            # Find and merge pairs
            new_seq = []
            j = 0
            while j < length:
                if j < length - 1 and seq[j] == a and seq[j+1] == b:
                    # Merge the pair
                    new_seq.append(new_token)
                    j += 2  # Skip both tokens
                else:
                    new_seq.append(seq[j].item())
                    j += 1
            
            # Pad to original length
            while len(new_seq) < seq.size(0):
                new_seq.append(-1)
                
            new_sequences.append(torch.tensor(new_seq, dtype=torch.long, device=self.device))
            new_lengths.append(len([x for x in new_seq if x != -1]))
        
        if new_sequences:
            new_sequences_tensor = torch.stack(new_sequences)
        else:
            new_sequences_tensor = torch.empty((0, 0), dtype=torch.long, device=self.device)
            
        new_lengths_tensor = torch.tensor(new_lengths, dtype=torch.long, device=self.device)
        
        return new_sequences_tensor, new_lengths_tensor

def BPE_tokenizer_training_gpu(input_path: str, vocab_size: int, special_tokens: List[str],
                              num_processes: int = 8, device: str = 'cuda:0') -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """GPU-accelerated BPE tokenizer training"""
    
    base_size = len(special_tokens) + 256
    if vocab_size < base_size:
        raise ValueError(f"vocab_size {vocab_size} < minimum required {base_size} (special tokens + 256 byte symbols)")
    
    # Initialize trainer
    trainer = GPUBPETrainer(device)
    
    vocab = {}
    merges = []
    
    # Vocabulary initialization
    next_id = 0
    
    # Special tokens
    for s in special_tokens:
        vocab[next_id] = s.encode('utf-8')
        next_id += 1
    
    # 256 byte symbols
    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1
    
    print("Loading and preprocessing sequences...")
    sequences = _init_sequences_parallel(input_path, special_tokens, num_processes)
    
    # Deduplicate sequences and get frequencies
    seq_freq_map = defaultdict(int)
    for seq in sequences:
        seq_freq_map[tuple(seq)] += 1
    
    sequences = [list(t) for t in seq_freq_map.keys()]
    freqs = list(seq_freq_map.values())
    
    print(f"Preparing {len(sequences)} unique sequences for GPU processing...")
    
    # Prepare GPU tensors
    gpu_sequences, gpu_lengths, gpu_freqs = trainer.prepare_sequences_gpu(sequences, freqs)
    
    print("Starting GPU-accelerated BPE training...")
    
    # Create mapping between byte objects and integers for merging
    byte_to_int = {}
    int_to_byte = {}
    current_int = 256  # Start after single byte values
    
    # Map existing single bytes
    for i in range(256):
        byte_to_int[bytes([i])] = i
        int_to_byte[i] = bytes([i])
    
    iteration = 0
    while len(vocab) < vocab_size:
        iteration += 1
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, vocab size: {len(vocab)}")
        
        # Count pairs on GPU
        pair_counts = trainer.count_pairs_gpu(gpu_sequences, gpu_lengths, gpu_freqs)
        
        if not pair_counts:
            break
        
        # Find most frequent pair
        (a_int, b_int), freq = max(pair_counts.items(), key=lambda x: (x[1], x[0][0], x[0][1]))
        
        if freq < 1:
            break
        
        # Convert back to bytes for merge recording
        a_bytes = int_to_byte.get(a_int, bytes([a_int]))
        b_bytes = int_to_byte.get(b_int, bytes([b_int]))
        
        merges.append((a_bytes, b_bytes))
        new_token_bytes = a_bytes + b_bytes
        vocab[next_id] = new_token_bytes
        
        # Add to mapping
        byte_to_int[new_token_bytes] = current_int
        int_to_byte[current_int] = new_token_bytes
        
        # Merge sequences on GPU
        gpu_sequences, gpu_lengths = trainer.merge_sequences_gpu(
            gpu_sequences, gpu_lengths, (a_int, b_int), current_int
        )
        
        next_id += 1
        current_int += 1
    
    print(f"Training completed. Final vocab size: {len(vocab)}, Merges: {len(merges)}")
    return vocab, merges

def serialize_save_vocab(vocab: Dict[int, bytes], saved_path: str) -> None:
    json_data = {k: base64.b64encode(v).decode('utf-8') for k, v in vocab.items()}
    serialized = json.dumps(json_data)
    with open(saved_path, 'w') as f:
        f.write(serialized)

def serialize_save_merges(merges: List[Tuple[bytes, bytes]], saved_path: str) -> None:
    json_data = [[base64.b64encode(a).decode('utf-8'), base64.b64encode(b).decode('utf-8')] 
                 for a, b in merges]
    serialized = json.dumps(json_data)
    with open(saved_path, 'w') as f:
        f.write(serialized)

def train_bpe_tinystories_gpu(input_path: str, vocab_size: int, special_tokens: List[str],
                              num_processes: int = 8, device: str = 'cuda:0'):
    """GPU-accelerated BPE training for TinyStories"""
    vocab, merges = BPE_tokenizer_training_gpu(input_path, vocab_size, special_tokens, num_processes, device)
    base_dir = os.getcwd()
    serialize_save_vocab(vocab, os.path.join(base_dir, 'tinystories_vocab_gpu'))
    serialize_save_merges(merges, os.path.join(base_dir, 'tinystories_merges_gpu'))
    return vocab, merges

# 使用示例
if __name__ == "__main__":
    # GPU加速训练示例
    input_path = "TinyStories/TinyStories-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    
    # 选择GPU设备 (你有4个GPU可选择: cuda:0, cuda:1, cuda:2, cuda:3)
    device = 'cuda:0'  # 使用第一个GPU，根据nvidia-smi的信息，这个GPU内存使用较少
    
    print("Starting GPU-accelerated BPE training...")
    
    # 开始profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    vocab, merges = train_bpe_tinystories_gpu(input_path, vocab_size, special_tokens, 
                                              num_processes=8, device=device)
    
    profiler.disable()
    profiler.dump_stats('train_bpe_gpu_profile.prof')
    
    # 显示profiling结果
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(30)
    
    print(f"Training completed! Vocab size: {len(vocab)}, Merges: {len(merges)}")