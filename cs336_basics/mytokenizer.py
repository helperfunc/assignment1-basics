import regex as re
from collections import defaultdict
from typing import List, Dict, Tuple

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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

def _init_sequences(corpus: List[str], special_tokens: List[str]) -> List[List[bytes]]:
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

def _merge_in_sequence(seq: List[bytes], a: bytes, b: bytes) -> None:
    i = 0
    while i < len(seq) - 1:
        if seq[i] == a and seq[i+1] == b:
            seq[i:i+2] = [a+b]
        else:
            i += 1

def BPE_tokenizer_training(input_path: str, vocab_size: int, special_tokens: List[str]
                            ) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
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

    with open(input_path, 'r', encoding='utf-8') as f:
        corpus = f.read()
    
    sequences = _init_sequences(corpus, special_tokens)
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
            _merge_in_sequence(seq, a, b)
        
        # count all the pairs
        if len(vocab) < vocab_size:
            pair_counts = _count_all_pairs(sequences)
    return vocab, merges

# vocab, merges = BPE_tokenizer_training(r"tests\fixtures\tinystories_sample.txt", 300, ["<|endoftext|>"])
# print("Final vocab size:", len(vocab))
# print("First merges:", merges[:20])    
