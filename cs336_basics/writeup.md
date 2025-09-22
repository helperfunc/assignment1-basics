## Problem (unicode1): Understanding Unicode (1 point)

### (a) What Unicode character does chr(0) return?
Deliverable: A one-sentence response.
> `\x00`

### (b) How does this character’s string representation (__repr__()) differ from its printed representation?
Deliverable: A one-sentence response.
> `__repr__()` provides an unambiguous, developer-centric representation of an object, if pasted back into a Python interpreter, could ideally recreate the object. Printed representation is intended for human-readable, user friendly output. For example, `__repr__()` includes quotes explicityly printed representation doesn't have.

### (c) What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:
```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```
Deliverable: A one-sentence response.
> printed representation of `chr(0)` is empty. `__repr__()` of `chr(0)` is `'\x00'`

## Problem (unicode2): Unicode Encodings (3 points)
### (a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.
Deliverable: A one-to-two sentence response.
```python
def byte_values(s):
    utf8_encoded = s.encode('utf-8')
    utf16_encoded = s.encode('utf-16')
    utf32_encoded = s.encode('utf-32')
    utf8_byte = list(utf8_encoded)
    utf16_byte = list(utf16_encoded)
    utf32_byte = list(utf32_encoded)
    return utf8_byte, utf16_byte, utf32_byte

slist = ['Hello', '你好！', '中文', '数学', '大语言模型', 'LLM', '这菜很好吃！', 'It is delicious!']
for s in slist:
    print(byte_values(s))
```
> `Hello` in UTF-8 is 5 bytes, but in UTF-16 and UTF-32 it is 12 and 20 bytes respectively, making UTF-8 more efficient and simpler for tokenization.


### (b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.
```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```
**Deliverable**: An example input byte string for which decode_utf8_bytes_to_str_wrong produces incorrect output, with a one-sentence explanation of why the function is incorrect.
> `好`, will raise `UnicodeDecodeError`, because the function tries to decode each byte separately, but multi-byte UTF-8 characters must be decoded together. The function is incorrect because it does not handle multi-byte UTF-8 sequences properly.

### (c) Give a two byte sequence that does not decode to any Unicode character(s).
Deliverable: An example, with a one-sentence explanation.
```python
def decode_bytes(bytes_str):
    return bytes_str.decode('utf-8')
print(decode_bytes(b'\xc0\xaf'))
```
> To encode `/` (U+002F = 0b00101111, 00000000 00101111). In UTF-8, the format 110xxxxx 10xxxxxx is used for multi-byte character, prevent confusion with single-byte characters. When we split the bits 00000000 00101111 to fit the two-byte UTF-8, we have 11000000 10101111 (`b'\xc0\xaf'`), which is a overlong encoding of `/`. Decoding `b'\xc0\xaf'` will get `UnicodeDecodeError`.

### Problem (train_bpe): BPE Tokenizer Training (15 points)
Deliverable: Write a function that, given a path to an input text file, trains a (byte-level) BPE
tokenizer. Your BPE training function should handle (at least) the following input parameters:
```
input_path: str Path to a text file with BPE tokenizer training data.
vocab_size: int A positive integer that defines the maximum final vocabulary size (including the
initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not
otherwise affect BPE training.
```
Your BPE training function should return the resulting vocabulary and merges:
```
vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
<token2>. The merges should be ordered by order of creation.
```
To test your BPE training function against our provided tests, you will first need to implement the
test adapter at `[adapters.run_train_bpe]`. Then, run `uv run pytest tests/test_train_bpe.py`.
Your implementation should be able to pass all tests. Optionally (this could be a large time-investment),
you can implement the key parts of your training method using some systems language, for instance
C++ (consider cppyy for this) or Rust (using PyO3). If you do this, be aware of which operations
require copying vs reading directly from Python memory, and make sure to leave build instructions, or
make sure it builds using only pyproject.toml. Also note that the GPT-2 regex is not well-supported
in most regex engines and will be too slow in most that do. We have verified that Oniguruma is
reasonably fast and supports negative lookahead, but the regex package in Python is, if anything,
even faster.

> Optimizing the merging step: 7.5s -> 2.6s, count the number of times (token1, token2) appear, add or minus the number of appearences of the tokens pairs: 2.6s->0.6s.

```python
import regex as re
from collections import defaultdict
from typing import List, Dict, Tuple
import os
from typing import BinaryIO
import multiprocessing as mp

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
                sequences.append([SINGLE_BYTES[x] for x in b])
    else:
        for tok in _pre_tokenize(corpus):
            b = tok.encode('utf-8')
            sequences.append([SINGLE_BYTES[x] for x in b])
    return sequences

def _count_all_pairs_weighted(sequences: List[List[bytes]], freqs: List[int]) -> Dict[Tuple[bytes, bytes], int]:
    counts = defaultdict(int)
    for seq, f in zip(sequences, freqs):
        if len(seq) < 2:
            continue
        for i in range(len(seq) - 1):
            counts[(seq[i], seq[i+1])] += f
    return counts

def _dec_by(counts: Dict[Tuple[bytes, bytes], int], pair: Tuple[bytes, bytes], delta: int):
    v = counts.get(pair)
    if v is None:
        return 
    if v <= delta:
        counts.pop(pair, None)
    else:
        counts[pair] = v - delta

def _inc_by(counts: Dict[Tuple[bytes, bytes], int], pair: Tuple[bytes, bytes], delta: int):
    counts[pair] = counts.get(pair, 0) + delta

def _merge_sequence_incremental(seq: List[bytes], a: bytes, b: bytes, new_tok: bytes, 
                                pair_counts: Dict[Tuple[bytes, bytes], int], freq: int) -> None:
    i = 0
    while i < len(seq) - 1:
        if seq[i] == a and seq[i+1] == b:
            prev_tok = seq[i-1] if i - 1 >= 0 else None
            next_tok = seq[i+2] if i + 2 < len(seq) else None # token after b
            if prev_tok is not None:
                _dec_by(pair_counts, (prev_tok, a), freq)
            _dec_by(pair_counts, (a, b), freq)
            if next_tok is not None:
                _dec_by(pair_counts, (b, next_tok), freq)
            seq[i:i+2] = [new_tok]
            if prev_tok is not None:
                _inc_by(pair_counts, (prev_tok, new_tok), freq)
            if next_tok is not None:
                _inc_by(pair_counts, (new_tok, next_tok), freq)
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
    
    # 
    seq_freq_map = defaultdict(int)
    for seq in sequences:
        seq_freq_map[tuple(seq)] += 1
    sequences = [list(t) for t in seq_freq_map.keys()]
    freqs = list(seq_freq_map.values())

    pair_counts = _count_all_pairs_weighted(sequences, freqs)
    
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
        for seq, f in zip(sequences, freqs):
            _merge_sequence_incremental(seq, a, b, new_token, pair_counts, f)
        
    return vocab, merges

# vocab, merges = BPE_tokenizer_training(r"tests\fixtures\tinystories_sample.txt", 300, ["<|endoftext|>"])
# print("Final vocab size:", len(vocab))
# print("First merges:", merges[:20])    
```

### Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)
#### (a) Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size of 10,000. Make sure to add the TinyStories <|endoftext|> special token to the vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How many hours and memory did training take? What is the longest token in the vocabulary? Does it make sense?
**Resource requirements**: ≤ 30 minutes (no GPUs), ≤ 30GB RAM
**Hint** You should be able to get under 2 minutes for BPE training using multiprocessing during pretokenization and the following two facts:
(a) The <|endoftext|> token delimits documents in the data files.
(b) The <|endoftext|> token is handled as a special case before the BPE merges are applied.
**Deliverable**: A one-to-two sentence response.
> 3202327539 function calls (3202327389 primitive calls) in 1032.077 seconds

#### (b) Profile your code. What part of the tokenizer training process takes the most time?
**Deliverable**: A one-to-two sentence response
```
3202327539 function calls (3202327389 primitive calls) in 1032.077 seconds

   Ordered by: cumulative time
   List reduced from 395 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.005    0.005 1032.077 1032.077 /chronos_data/huixu/assignment1-basics/cs336_basics/mytokenizer.py:312(train_bpe_tinystories)
        1  141.878  141.878 1031.988 1031.988 /chronos_data/huixu/assignment1-basics/cs336_basics/mytokenizer.py:229(BPE_tokenizer_training)
665700218  430.827    0.000  548.776    0.000 /chronos_data/huixu/assignment1-basics/cs336_basics/mytokenizer.py:208(_merge_sequence_incremental)
        1    0.291    0.291  190.681  190.681 /chronos_data/huixu/assignment1-basics/cs336_basics/mytokenizer.py:103(_init_sequences_parallel)
       10    0.000    0.000  190.281   19.028 /chronos_data/conda_envs/irt/lib/python3.10/threading.py:288(wait)
       43  190.281    4.425  190.281    4.425 {method 'acquire' of '_thread.lock' objects}
        9    0.000    0.000  190.281   21.142 /chronos_data/conda_envs/irt/lib/python3.10/multiprocessing/pool.py:853(next)
     9780   77.476    0.008  150.444    0.015 {built-in method builtins.max}
2121014153  117.114    0.000  117.114    0.000 {built-in method builtins.len}
413043097   72.968    0.000   72.968    0.000 /chronos_data/huixu/assignment1-basics/cs336_basics/mytokenizer.py:266(<lambda>)
   766533    0.390    0.000    0.550    0.000 /chronos_data/huixu/assignment1-basics/cs336_basics/mytokenizer.py:196(_dec_by)
   444011    0.203    0.000    0.300    0.000 /chronos_data/huixu/assignment1-basics/cs336_basics/mytokenizer.py:205(_inc_by)
  1210617    0.244    0.000    0.244    0.000 {method 'get' of 'dict' objects}
        1    0.184    0.184    0.194    0.194 /chronos_data/huixu/assignment1-basics/cs336_basics/mytokenizer.py:187(_count_all_pairs_weighted)
        1    0.045    0.045    0.045    0.045 /chronos_data/huixu/assignment1-basics/cs336_basics/mytokenizer.py:142(<listcomp>)
        1    0.000    0.000    0.044    0.044 /chronos_data/huixu/assignment1-basics/cs336_basics/mytokenizer.py:297(serialize_save_merges)
        1    0.000    0.000    0.039    0.039 /chronos_data/huixu/assignment1-basics/cs336_basics/mytokenizer.py:284(serialize_save_vocab)
        9    0.000    0.000    0.039    0.004 /chronos_data/conda_envs/irt/lib/python3.10/multiprocessing/util.py:205(__call__)
        1    0.000    0.000    0.039    0.039 /chronos_data/conda_envs/irt/lib/python3.10/multiprocessing/pool.py:738(__exit__)
        1    0.000    0.000    0.039    0.039 /chronos_data/conda_envs/irt/lib/python3.10/multiprocessing/pool.py:654(terminate)
```

### Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)
#### (a) Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. What is the longest token in the vocabulary? Does it make sense?
Resource requirements: ≤ 12 hours (no GPUs), ≤ 100GB RAM
Deliverable: A one-to-two sentence response.

#### (b) Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.
Deliverable: A one-to-two sentence response.