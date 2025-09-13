import regex as re
from collections import defaultdict
from typing import List, Dict, Tuple
import os
from typing import BinaryIO
import multiprocessing as mp

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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
        parts = pool.mp(_process_file_chunk, tasks)
    out = []
    for p in parts:
        out.extend(p)
    return out

def _pre_tokenize(text: str):
    for m in re.finditer(PAT, text):
        yield m.group()

def _compile_special_pattern(special_tokens: List[str]) -> re.Pattern | None:
    '''
    import re
    special_tokens = ["<|endoftext|>", "<pad>"]
    pat = re.compile("(" + "|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True)) + ")")
    corpus = "Hello<|endoftext|>World<pad>Byehellohello"
    parts = pat.split(corpus)
    print(parts)
    ['Hello', '<|endoftext|>', 'World', '<pad>', 'Byehellohello']
    '''
    if not special_tokens:
        return None
    # order according to the length
    parts = [re.escape(t) for t in sorted(special_tokens, key=len, reverse=True)]
    return re.compile("(" + "|".join(parts) + ")")

def _init_sequences(corpus: str, special_tokens: List[str]) -> List[List[bytes]]:
    # every token is a sequence of utf-8 bytes
    sequences = []
    special_pat = _compile_special_pattern(special_tokens)
    if special_pat:
        parts = special_pat.split(corpus)
        for i, part in enumerate(parts):
            # odd indices are special tokens
            if i % 2 == 1:
                continue
            if not part:
                continue
            for tok in _pre_tokenize(part):
                b = tok.encode('utf-8')
                sequences.append([bytes([x]) for x in b])
    else:
        for tok in _pre_tokenize(corpus):
            b = tok.encode('utf-8')
            sequences.append([bytes([x]) for x in b])
    return sequences

def _count_all_pairs(sequences: List[List[bytes]]) -> Dict[Tuple[bytes, bytes], int]:
    counts = defaultdict(int)
    for seq in sequences:
        for i in range(len(seq) - 1):
            counts[(seq[i], seq[i+1])] += 1
    return counts

def _dec(counts: Dict[Tuple[bytes, bytes], int], pair: Tuple[bytes, bytes]):
    v = counts.get(pair)
    if v is None:
        return 
    if v <= 1:
        del counts[pair]
    else:
        counts[pair] = v - 1

def _inc(counts: Dict[Tuple[bytes, bytes], int], pair: Tuple[bytes, bytes]):
    counts[pair] = counts.get(pair, 0) + 1

def _merge_sequence_incremental(seq: List[bytes], a: bytes, b: bytes, new_tok: bytes, 
                                pair_counts: Dict[Tuple[bytes, bytes], int]) -> None:
    i = 0
    while i < len(seq) - 1:
        if seq[i] == a and seq[i+1] == b:
            prev_tok = seq[i-1] if i - 1 >= 0 else None
            next_tok = seq[i+2] if i + 2 < len(seq) else None # token after b
            if prev_tok is not None:
                _dec(pair_counts, (prev_tok, a))
            _dec(pair_counts, (a, b))
            if next_tok is not None:
                _dec(pair_counts, (b, next_tok))
            seq[i:i+2] = [new_tok]
            if prev_tok is not None:
                _inc(pair_counts, (prev_tok, new_tok))
            if next_tok is not None:
                _inc(pair_counts, (new_tok, next_tok))
            i += 1 # advance past the merged token
        else:
            i += 1

def BPE_tokenizer_training(input_path: str, vocab_size: int, special_tokens: List[str],
                           num_processes: int = 8) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    '''
    input params:
        input_path: str Path to a text file with BPE tokenizer training data.
        vocab_size: int A positive integer that defines the maximum final vocabulary size 
                    (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
        special_tokens: list[str] A list of strings to add to the vocabulary. 
                        These special tokens do not otherwise affect BPE training.
    returns:
        vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
        merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item is 
                a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with <token2>. 
                The merges should be ordered by order of creation.
    '''
    base_size = len(special_tokens) + 256
    if vocab_size < base_size:
        raise ValueError(f"vocab_size {vocab_size} < minimum required {base_size} (special tokens + 256 byte symbols)")
    vocab = {} # id:bytes
    merges = []
    # vocabulary initialization
    next_id = 0
    # special tokens
    for s in special_tokens:
        vocab[next_id] = s.encode('utf-8')
        next_id += 1
    # 256 byte
    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1
    
    sequences = _init_sequences_parallel(input_path, special_tokens, num_processes)
    pair_counts = _count_all_pairs(sequences)
    
    while len(vocab) < vocab_size and pair_counts:
        # select the most frequent
        (a, b), freq = max(pair_counts.items(), key=lambda x: (x[1], x[0][0], x[0][1]))
        if freq < 1:
            break
        merges.append((a, b))
        new_token = a + b
        vocab[next_id] = new_token
        next_id += 1

        # merge the (a, b) on all the sequences
        for seq in sequences:
            _merge_sequence_incremental(seq, a, b, new_token, pair_counts)
        
    return vocab, merges

# vocab, merges = BPE_tokenizer_training(r"tests\fixtures\tinystories_sample.txt", 300, ["<|endoftext|>"])
# print("Final vocab size:", len(vocab))
# print("First merges:", merges[:20])    
