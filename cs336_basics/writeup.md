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

## Problem (train_bpe_tinystories): BPE Training on TinyStories
(a) pretokenization takes 111.482s. The longest token in the vocabulary from `get_longest_token.py` is `accomplishment`. It make sense.
(b) `_collect_word_freqs_parallel` takes the most of the time.
```
$ python cs336_basics/train_corpus_tokenizer.py 
         22225266 function calls (22225130 primitive calls) in 127.566 seconds                        

   Ordered by: cumulative time
   List reduced from 489 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.071    0.071  127.566  127.566 /chronos_data/huixu/assignment1-basics/cs336_basics/train_corpus_tokenizer.py:25(train_bpe_tinystories)
        1    5.042    5.042  127.435  127.435 /chronos_data/huixu/assignment1-basics/cs336_basics/mytokenizer.py:179(BPE_tokenizer_training)
        1    0.380    0.380  111.482  111.482 /chronos_data/huixu/assignment1-basics/cs336_basics/mytokenizer.py:101(_collect_word_freqs_parallel)
       20    0.000    0.000  110.464    5.523 /chronos_data/conda_envs/irt/lib/python3.10/threading.py:288(wait)
       17    0.000    0.000  110.464    6.498 /chronos_data/conda_envs/irt/lib/python3.10/multiprocessing/pool.py:853(next)
       83  110.464    1.331  110.464    1.331 {method 'acquire' of '_thread.lock' objects}
     9743    0.130    0.000    3.207    0.000 /chronos_data/conda_envs/irt/lib/python3.10/site-packages/tqdm/std.py:1402(set_postfix)
     9881    0.023    0.000    3.014    0.000 /chronos_data/conda_envs/irt/lib/python3.10/site-packages/tqdm/std.py:1325(refresh)
     9883    0.027    0.000    2.942    0.000 /chronos_data/conda_envs/irt/lib/python3.10/site-packages/tqdm/std.py:1464(display)
   923569    2.702    0.000    2.702    0.000 {built-in method _heapq.heappop}
     9881    0.043    0.000    1.758    0.000 /chronos_data/conda_envs/irt/lib/python3.10/site-packages/tqdm/std.py:1150(__str__)
  1180814    1.164    0.000    1.728    0.000 /chronos_data/huixu/assignment1-basics/cs336_basics/mytokenizer.py:237(_push)
    29645    0.034    0.000    1.661    0.000 /chronos_data/conda_envs/irt/lib/python3.10/site-packages/tqdm/utils.py:378(disp_len)
     9881    0.255    0.000    1.641    0.000 /chronos_data/conda_envs/irt/lib/python3.10/site-packages/tqdm/std.py:464(format_meter)
    29645    0.024    0.000    1.606    0.000 /chronos_data/conda_envs/irt/lib/python3.10/site-packages/tqdm/utils.py:374(_text_width)
    29645    0.376    0.000    1.582    0.000 {built-in method builtins.sum}
   317920    0.940    0.000    1.233    0.000 /chronos_data/huixu/assignment1-basics/cs336_basics/mytokenizer.py:159(_merge_word_all_with_positions)
  2775922    0.833    0.000    1.206    0.000 /chronos_data/conda_envs/irt/lib/python3.10/site-packages/tqdm/utils.py:375(<genexpr>)
     9883    0.033    0.000    1.153    0.000 /chronos_data/conda_envs/irt/lib/python3.10/site-packages/tqdm/std.py:457(print_status)
   524104    0.687    0.000    0.774    0.000 /chronos_data/huixu/assignment1-basics/cs336_basics/mytokenizer.py:147(_merge_positions_nonoverlap)
```

## Problem (train_bpe_expts_owt): BPE Training on OpenWebText
(a) The longest token is '----------------------------------------------------------------'. It does not make sense.
(b) 
```
$ python cs336_basics/compare_tokenizers.py 
TinyStories:
  vocab size:  10000
  merges:       9743
  avg bytes/token: 5.85
  max bytes/token: 15
  longest token (utf-8, errors ignored): ' accomplishment'
  first 5 merges: [('h', 'e'), (' ', 't'), (' ', 'a'), (' ', 's'), (' ', 'w')]
  last 5 merges: [(' pound', 'ing'), (' pl', 'umber'), (' Stan', 'ley'), ('el', 've'), (' tap', 's')]

OpenWebText:
  vocab size:  32000
  merges:      31743
  avg bytes/token: 6.33
  max bytes/token: 64
  longest token (utf-8, errors ignored): '----------------------------------------------------------------'
  first 5 merges: [(' ', 't'), (' ', 'a'), ('h', 'e'), ('i', 'n'), ('r', 'e')]
  last 5 merges: [(' o', 'y'), (' comp', 'ounded'), (' Ass', 'uming'), ('ow', 'an'), (' sa', 'p')]
```

```
  2072150409 function calls (2072150268 primitive calls) in 3684.086 seconds

   Ordered by: cumulative time
   List reduced from 498 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1   20.297   20.297 3684.086 3684.086 /chronos_data/huixu/assignment1-basics/cs336_basics/train_corpus_tokenizer.py:34(train_bpe_expts_owt)
        1  763.489  763.489 3663.396 3663.396 /chronos_data/huixu/assignment1-basics/cs336_basics/mytokenizer.py:179(BPE_tokenizer_training)
113443969 1258.440    0.000 1258.440    0.000 {built-in method _heapq.heappop}
        1   31.540   31.540  788.843  788.843 /chronos_data/huixu/assignment1-basics/cs336_basics/mytokenizer.py:101(_collect_word_freqs_parallel)
        5    0.000    0.000  756.692  151.338 /data/backup/anaconda3/lib/python3.9/threading.py:280(wait)
       23  756.692   32.900  756.692   32.900 {method 'acquire' of '_thread.lock' objects}
       17    0.000    0.000  756.692   44.511 /data/backup/anaconda3/lib/python3.9/multiprocessing/pool.py:850(next)
157404951  147.785    0.000  256.793    0.000 /chronos_data/huixu/assignment1-basics/cs336_basics/mytokenizer.py:237(_push)
 36921495  152.140    0.000  212.215    0.000 /chronos_data/huixu/assignment1-basics/cs336_basics/mytokenizer.py:159(_merge_word_all_with_positions)
 70974164  143.804    0.000  158.811    0.000 /chronos_data/huixu/assignment1-basics/cs336_basics/mytokenizer.py:147(_merge_positions_nonoverlap)
428766427  121.979    0.000  121.979    0.000 {method 'get' of 'dict' objects}
157424682   80.452    0.000   80.452    0.000 {built-in method _heapq.heappush}
350978321   59.267    0.000   59.267    0.000 {method 'append' of 'list' objects}
277623076   58.555    0.000   58.555    0.000 {method 'add' of 'set' objects}
466628924   50.779    0.000   50.779    0.000 {built-in method builtins.len}
    40114    0.208    0.000   16.027    0.000 /data/backup/anaconda3/lib/python3.9/site-packages/tqdm/std.py:1326(refresh)
    40116    0.184    0.000   15.277    0.000 /data/backup/anaconda3/lib/python3.9/site-packages/tqdm/std.py:1465(display)
        1   14.965   14.965   14.965   14.965 /chronos_data/huixu/assignment1-basics/cs336_basics/mytokenizer.py:201(<listcomp>)
    31743    0.658    0.000   13.536    0.000 /data/backup/anaconda3/lib/python3.9/site-packages/tqdm/std.py:1403(set_postfix)
    40116    0.185    0.000    9.952    0.000 /data/backup/anaconda3/lib/python3.9/site-packages/tqdm/std.py:460(print_status)
```

## Problem (tokenizer): Implementing the tokenizer
`uv run pytest tests/test_tokenizer.py` all test passed.

## Problem (tokenizer_experiments): Experiments with tokenizers
```
$ python cs336_basics/tokenizer_analysis.py --tinystories TinyStories/TinyStories-train.txt --openwebtext openwebtext/owt_corpus.txt --n-docs 10
TinyStories tokenizer on TinyStories: 4.181 bytes/token
OpenWebText tokenizer on OpenWebText: 4.557 bytes/token
TinyStories tokenizer on OpenWebText: 3.320 bytes/token
Tokenizer throughput (OpenWebText tokenizer): 0.39 MB/s
Estimated time for 825GB: 623.97 hours
```
`uint16` stores integers up to `65,535`, which comfortably covers both vocabularies (`10k` and `32k` tokens), while halving the storage footprint compared to `uint32` and keeping NumPy operations fast.
 
## Problem (transformer_accounting): Transformer LM resource accounting 
### (a) Consider GPT-2 XL, which has the following configuration:
```
vocab_size : 50,257
context_length : 1,024
num_layers : 48
d_model : 1,600
num_heads : 25
d_ff : 6,400  # 4× multiplier of d_model
```
Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?

Parameter Breakdown
For a Transformer language model, the trainable parameters include:

1. Token Embeddings
(number of vocab, d_model) embedding_mat[vocab_ind]
Parameters: 
vocab_size x d_model = 50,257 x 1,600 = 80,411,200

2. Position Embeddings
Using RoPE no position embeddings parameters.
k = d_k // 2  d_k (query, key dimention)
(token1, token2) -> \theta_1
(token3, token4) -> \theta_2
...

3. Transformer Layers (x48)
For each layer:
```
self.rms_norm_1 = RMSNorm(d_model, device=device, dtype=dtype)
d_model = 1,600
```

Multi-Head Self-Attention
```
self.W_qkv = Linear(d_model, 3*d_model, device=device, dtype=dtype)
# 1,600 X 4,800 = 7,680,000

self.W_o = Linear(d_model, d_model, device=device, dtype=dtype)
# 1,600 X 1,600 = 2,560,000
```

```
self.rms_norm_2 = RMSNorm(d_model, device=device, dtype=dtype)
d_model = 1,600
```

```
self.point_wise_ff = SwiGLU(d_model)
SwiGLU parameters:
self.glu = GLU(d_model) 
self.lin = Linear(d_ff, d_model)
d_model = 1,600
d_ff = 6,400
1600*6400 + 20,480,000 = 30,720,000

GLU parameters:
self.lin1 = Linear(d_model, d_ff)
self.lin2 = Linear(d_model, d_ff)
2*1600*6400 = 20,480,000
```

Every Transformer Block
```
self.rms_norm_1 = RMSNorm(d_model, device=device, dtype=dtype)
self.multihead_self_attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
self.rms_norm_2 = RMSNorm(d_model, device=device, dtype=dtype)
self.point_wise_ff = SwiGLU(d_model, d_ff)

RMSNorm_1:             1,600
Attention (W_qkv):  7,680,000
Attention (W_o):    2,560,000
RMSNorm_2:             1,600
SwiGLU:            30,720,000
──────────────────────────────
Total:             40,963,200
```

Transformer lm parameters
```
Token Embeddings:        80,411,200  (50,257 × 1,600)
Position Embeddings:              0  (RoPE - no parameters)
48 × Transformer Blocks: 1,966,233,600  (48 × 40,963,200)
Final RMSNorm:                 1,600  (1,600)
LM Head:                  80,411,200  (50,257 × 1,600) not sharing with token embeddings！
─────────────────────────────────────
Total:                   2,127,057,600
```

Single-precision floating point (FP32) 
4 bytes per parameter
```
Memory = 2,127,057,600 × 4 bytes = 8,508,230,400 bytes
8,508,230,400 bytes ÷ 1,000,000,000 = 8.508 GB ≈ 8.51 GB
8,508,230,400 bytes ÷ 1024^3 = 7.926 GiB ≈ 7.93 GiB


This is the minimum memory required to just load the model weights. During training or inference, additional memory is needed for:
- Activations (intermediate outputs from each layer)
- Gradients (during training)
- Optimizer states (e.g., Adam requires ~2× model size for momentum and variance)
- Input/output buffers
```

### (b) Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has context_length tokens.
Deliverable: A list of matrix multiplies (with descriptions), and the total number of FLOPs
1. Token Embedding:
not matrix multiplication
just lookup

2. Every Transformer Block
RMSNorm is element-wise operations not matrix multiplication
```
self.multihead_self_attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
```
include:
```
QKV Projection: x @ W_qkv
# Input: (b, seq_len, d_model) = (1, 1024, 1600)
# W_qkv: (d_model, 3*d_model) = (1600, 4800)
# Output: (b, seq_len, 3*d_model) = (1, 1024, 4800)
x ∈ R^(seq_len × d_model), W_qkv ∈ R^(d_model × 3*d_model)
m = seq_len = 1,024
n = d_model = 1,600
p = 3*d_model = 4,800

FLOPs = 2 * 1024 * 1600 * 4800 = 15728640000

Q @ K^T (Attention scores)
# Q: (b, num_heads, seq_len, d_k) = (1, 25, 1024, 64)  d_k = d_model // num_heads = 1600 // 25 = 64
# K^T: (b, num_heads, d_k, seq_len) = (1, 25, 64, 1024)
# Score: (b, num_heads, seq_len, seq_len) = (1, 25, 1024, 1024)
For every head,
A ∈ R^(seq_len × d_k), B ∈ R^(d_k × seq_len)
m = seq_len = 1,024
n = d_k = 64
p = seq_len = 1,024

FLOPS per head = 2 * 1024 * 64 * 1024 = 134217728
Total FLOPs = 25 * 134217728 = 3355443200


Attention @ V
# Attention: (b, num_heads, seq_len, seq_len) = (1, 25, 1024, 1024)
# V: (b, num_heads, seq_len, d_k) = (1, 25, 1024, 64)
# Output: (b, num_heads, seq_len, d_k) = (1, 25, 1024, 64)
A ∈ R^(seq_len × seq_len), B ∈ R^(seq_len × d_k)
m = seq_len = 1,024
n = seq_len = 1,024
p = d_k = 64
FLOPs per head = 2 * 1024 * 1024 * 64 = 134217728
Total FLOPs = 25 * 134217728 = 3355443200

Attention output: x @ W_o
# Input: (b, seq_len, d_model) = (1, 1024, 1600)
# W_o: (d_model, d_model) = (1600, 1600)
# Output: (b, seq_len, d_model) = (1, 1024, 1600)
A ∈ R^(seq_len × d_model), B ∈ R^(d_model × d_model)
m = seq_len = 1,024
n = d_model = 1,600
p = d_model = 1,600

FLOPs = 2 * 1024 * 1600 * 1600 = 5242880000

Attention total FLOPs per layer = 15728640000 + 3355443200 + 3355443200 + 5242880000 = 27682406400
Attention: 48 * 27682406400 = 1328755507200 (0.29440647731422626)
```

RoPEs on Q, K are element-wise operations, not matrix multiplication.


Point-wise feed-forward
```
ff_output = self.point_wise_ff(self.rms_norm_2(atten_res_output))
```
contains
```
SwiGLU W1: x @ W1
# Input: (b, seq_len, d_model) = (1, 1024, 1600)
# W3: (d_model, d_ff) = (1600, 6400)
# Output: (b, seq_len, d_ff) = (1, 1024, 6400)
A ∈ R^(seq_len × d_model), B ∈ R^(d_model × d_ff)
m = seq_len = 1,024
n = d_model = 1,600
p = d_ff = 6,400

FLOPs = 2 * 1024 * 1600 * 6400 = 20971520000

SwiGLU W3: x @ W3
# Input: (b, seq_len, d_model) = (1, 1024, 1600)
# W3: (d_model, d_ff) = (1600, 6400)
# Output: (b, seq_len, d_ff) = (1, 1024, 6400)

A ∈ R^(seq_len × d_model), B ∈ R^(d_model × d_ff)
m = seq_len = 1,024
n = d_model = 1,600
p = d_ff = 6,400

FLOPs = 2 * 1024 * 1600 * 6400 = 20971520000

SwiGLU W2: x @ W2
# Input: (b, seq_len, d_ff) = (1, 1024, 6400)
# W2: (d_ff, d_model) = (6400, 1600)
# Output: (b, seq_len, d_model) = (1, 1024, 1600)

A ∈ R^(seq_len × d_ff), B ∈ R^(d_ff × d_model)
m = seq_len = 1,024
n = d_ff = 6,400
p = d_model = 1,600

FLOPs = 2 * 1024 * 6400 * 1600 = 20971520000
```
SwiGLU Total FLOPs per layer:
20971520000 + 20971520000 + 20971520000 = 62914560000

48 * 62914560000 = 3019898880000  (0.6691056302596051)

Every Transformer Block Total FLOPS:
27682406400 + 62914560000 = 90596966400

```
self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
LM Head: x @ W_lm

# Input: (b, seq_len, d_model) = (1, 1024, 1600)
# W_lm: (d_model, vocab_size) = (1600, 50257)
# Output: (b, seq_len, vocab_size) = (1, 1024, 50257)

A ∈ R^(seq_len × d_model), B ∈ R^(d_model × vocab_size)
m = seq_len = 1,024
n = d_model = 1,600
p = vocab_size = 50,257

FLOPs = 2 * 1024 * 1600 * 50257 = 164682137600
```

Total FLOPS:
```
Transformer_blocks = 1328755507200 + 3019898880000 = 4348654387200

LM head = 164682137600  (0.036487892426168594)

Total FLOPs = 4348654387200 + 164682137600 = 4513336524800 = 4.51 * 10^12 FLOPs = 4.51 TFLOPs



1. Token Embedding: 0 FLOPs (lookup)
2. 48 × Transformer Blocks:
   - Attention (QKV proj): 48 × 15,728,640,000 = 754,974,720,000
   - Attention (Q@K^T): 48 × 3,355,443,200 = 161,061,273,600
   - Attention (Attn@V): 48 × 3,355,443,200 = 161,061,273,600
   - Attention (Output): 48 × 5,242,880,000 = 251,658,240,000
   - SwiGLU (Gate): 48 × 20,971,520,000 = 1,006,632,960,000
   - SwiGLU (Up): 48 × 20,971,520,000 = 1,006,632,960,000
   - SwiGLU (Down): 48 × 20,971,520,000 = 1,006,632,960,000
3. LM Head: 164,681,318,400
───────────────────────────────────────────────
Total: 4,513,335,705,600 FLOPs
```

### Based on your analysis above, which parts of the model require the most FLOPs?
Total FLOPs: 4,513,335,705,600 (≈4.51 TFLOPs)

1. SwiGLU Feed-Forward: 3,019,898,880,000 FLOPs (66.9%)
   - Gate projection (W1): 1,006,632,960,000 FLOPs
   - Up projection (W3):   1,006,632,960,000 FLOPs
   - Down projection (W2): 1,006,632,960,000 FLOPs

2. Attention Mechanism: 1,328,755,507,200 FLOPs (29.4%)
   - QKV projection:         754,974,720,000 FLOPs (16.7%)
   - Q @ K^T (scores):       161,061,273,600 FLOPs (3.6%)
   - Attention @ V:          161,061,273,600 FLOPs (3.6%)
   - Output projection:      251,658,240,000 FLOPs (5.6%)

3. LM Head: 164,681,318,400 FLOPs (3.6%)

### Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24 layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?
Deliverable: For each model, provide a breakdown of model components and its associated FLOPs (as a proportion of the total FLOPs required for a forward pass). In addition, provide a one-to-two sentence description of how varying the model size changes the proportional FLOPs of each component.

FLOPs calculations
vocab_size = 50,257
context_length = seq_len = 1,024

GPT-2 Small (12 layers, 768 d_model, 12 heads)
```
12 x Transformer Blocks
   - Attention (QKV proj): 2 * seq_len * d_model * 3 * d_model = 2 * 1024 * 768 * 3 * 768 = 3623878656  (0.01036488535195174)
   - Attention (Q@K^T): heads * (2 * seq_len * (d_model // heads) * seq_len) = 12*(2 * 1024 * (768//12) * 1024) = 1610612736  (0.0046066157119785504)
   - Attention (Attn@V): heads * (2 * seq_len * seq_len * (d_model // heads)) = 12 * (2 * 1024 * 1024 * (768//12)) = 1610612736 (0.0046066157119785504)
   - Attention (Output): 2 * seq_len * d_model * d_model = 2 * 1024 * 768 * 768 = 1207959552  (0.003454961783983913)

  Attention: 12 * (3623878656 + 1610612736 * 2 + 1207959552) = 96636764160  (0.276396942718713)
  
   - SwiGLU (Gate): 2 * seq_len * d_model * d_ff = 2 * 1024 * 768 * 4 * 768 = 4831838208  (0.013819847135935651)
   - SwiGLU (Up): 2 * seq_len * d_model * d_ff = 2 * 1024 * 768 * 4 * 768 = 4831838208
   - SwiGLU (Down): 2 * seq_len * d_ff * d_model = 2 * 4 * 768 * 1024 * 768 = 4831838208
  SwiGLU: 12 * (4831838208 * 3) = 173946175488  (0.49751449689368343)

  Total FLOPs = 96636764160 + 173946175488 = 270582939648

LM head
2 * seq_len * d_model * vocab = 2 * 1024 * 768 * 50257 = 79047426048  (0.22608856038760353)

Total = 270582939648 + 79047426048 = 349630365696
349630365696 = 0.3496 * 10^12 = 0.3496 TFLOPs

```

GPT-2 Medium (24 layers, 1,024 d_model, 16 heads)

```
24 x Transformer Blocks
   - Attention (QKV proj): 2 * seq_len * d_model * 3 * d_model = 2 * 1024 * 1024 * 3 * 1024 = 6442450944
   - Attention (Q@K^T): heads * (2 * seq_len * (d_model // heads) * seq_len) = 16 * (2 * 1024 * (1024//24) * 1024) = 1409286144
   - Attention (Attn@V): heads * (2 * seq_len * seq_len * (d_model // heads)) = 16 * (2 * 1024 * 1024 * (1024//24)) = 1409286144
   - Attention (Output): 2 * seq_len * d_model * d_model = 2 * 1024 * 1024 * 1024 = 2147483648
  Attention: 24 * (6442450944 + 1409286144 * 2 + 2147483648) = 273804165120 (0.2744419617050884)
  
   - SwiGLU (Gate): 2 * seq_len * d_model * d_ff = 2 * 1024 * 1024 * 4 * 1024 = 8589934592
   - SwiGLU (Up): 2 * seq_len * d_model * d_ff = 2 * 1024 * 1024 * 4 * 1024 = 8589934592
   - SwiGLU (Down): 2 * seq_len * d_ff * d_model = 2 * 4 * 1024 * 1024 * 1024 = 8589934592
  SwiGLU: 24 * (8589934592 * 3) = 618475290624  (0.6199159605573762)

  Total FLOPs = 273804165120 + 618475290624 = 892279455744

LM head
2 * seq_len * d_model * vocab = 2 * 1024 * 1024 * 50257 = 105396568064  (0.10564207773753545)

Total = 892279455744 + 105396568064 = 997676023808
997676023808 = 0.998 * 10^12 = 0.998 TFLOPs

```

GPT-2 Large (36 layers, 1,280 d_model, 20 heads)
```
36 x Transformer Blocks
   - Attention (QKV proj): 2 * seq_len * d_model * 3 * d_model = 2 * 1024 * 1280 * 3 * 1280 = 10066329600
   - Attention (Q@K^T): heads * (2 * seq_len * (d_model // heads) * seq_len) = 20 * (2 * 1024 * (1280//36) * 1024) = 1468006400
   - Attention (Attn@V): heads * (2 * seq_len * seq_len * (d_model // heads)) = 20 * (2 * 1024 * 1024 * (1280//36)) = 1468006400
   - Attention (Output): 2 * seq_len * d_model * d_model = 2 * 1024 * 1280 * 1280 = 3355443200
  Attention: 36 * (10066329600 + 1468006400 * 2 + 3355443200) = 588880281600 (0.2713512116222971) 

   - SwiGLU (Gate): 2 * seq_len * d_model * d_ff = 2 * 1024 * 1280 * 4 * 1280 = 13421772800
   - SwiGLU (Up): 2 * seq_len * d_model * d_ff = 2 * 1024 * 1280 * 4 * 1280 = 13421772800
   - SwiGLU (Down): 2 * seq_len * d_ff * d_model = 2 * 4 * 1280 * 1024 * 1280 = 13421772800
  SwiGLU: 36 * (13421772800 * 3) = 1449551462400  (0.6679414439933467)

  Total FLOPs = 588880281600 + 1449551462400 = 2038431744000

LM head
2 * seq_len * d_model * vocab = 2 * 1024 * 1280 * 50257 = 131745710080  (0.06070734438435624)

Total = 2038431744000 + 131745710080 = 2170177454080
2170177454080 = 2.17 * 10^12 = 2.17 TFLOPs

```

Attention takes up proportionally more (0.276396942718713 -> 0.2744419617050884 -> 0.2713512116222971)
SwiGLU takes up proportionally more (0.49751449689368343 -> 0.6199159605573762 -> 0.6679414439933467)
LM Head takes up proportionally less (0.22608856038760353 -> 0.10564207773753545 -> 0.06070734438435624)

### (e) Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change? How do the relative contribution of FLOPs of the model components change?
Deliverable: A one-to-two sentence response.
context length = 16,384
(48 layers, 1,600 d_model, 25 heads) 
```
48 x Transformer Blocks
   - Attention (QKV proj): 2 * seq_len * d_model * 3 * d_model = 2 * 16384 * 1600 * 3 * 1600 = 251658240000
   - Attention (Q@K^T): heads * (2 * seq_len * (d_model // heads) * seq_len) = 25 * (2 * 16384 * (1600//48) * 16384) = 442918502400
   - Attention (Attn@V): heads * (2 * seq_len * seq_len * (d_model // heads)) = 25 * (2 * 16384 * 16384 * (1600//48)) = 442918502400
   - Attention (Output): 2 * seq_len * d_model * d_model = 2 * 16384 * 1600 * 1600 = 83886080000
  Attention: 48 * (251658240000 + 442918502400 * 2 + 83886080000) = 58626303590400  (0.5350111120946)

   - SwiGLU (Gate): 2 * seq_len * d_model * d_ff = 2 * 16384 * 1600 * 4 * 1600 = 335544320000
   - SwiGLU (Up): 2 * seq_len * d_model * d_ff = 2 * 16384 * 1600 * 4 * 1600 = 335544320000
   - SwiGLU (Down): 2 * seq_len * d_ff * d_model = 2 * 4 * 1600 * 16384 * 1600 = 335544320000
  SwiGLU: 48 * (335544320000 * 3) = 48318382080000  (0.4409432242537912)

  Total FLOPs = 58626303590400 + 48318382080000 = 106944685670400

LM head
2 * seq_len * d_model * vocab = 2 * 16384 * 1600 * 50257 = 2634914201600   (0.02404566365160892)

Total = 106944685670400 + 2634914201600 = 109579599872000 
109579599872000 = 109.58 * 10^12 = 109.585 TFLOPs
          GPT-2 XL            GPT-2 XL
          4.51 TFLOPs         109.58 TFLOPs
Attention 0.29440647731422626  0.5350111120946
SwiGLU  0.6691056302596051  0.4409432242537912
LM head 0.036487892426168594 0.02404566365160892
```

## Problem (learning_rate_tuning): Tuning the learning rate (1 point)
As we will see, one of the hyperparameters that affects training the most is the learning rate. Let’s
see that in practice in our toy example. Run the SGD example above with three other values for the
learning rate: 1e1, 1e2, and 1e3, for just 10 training iterations. What happens with the loss for each
of these learning rates? Does it decay faster, slower, or does it diverge (i.e., increase over the course of
training)?

```
lr = 1e1  decay slower
28.021129608154297
17.933523178100586
13.219829559326172
10.343104362487793
8.377913475036621
6.946251392364502
5.858241081237793
5.006033420562744
4.323103904724121
3.765903949737549

lr = 1e2 decay faster
21.00170135498047
21.001699447631836
3.6033217906951904
0.08623562008142471
1.6875389762544143e-16
1.880866851470532e-18
6.333541004559537e-20
3.7729339481548824e-21
3.2366662542985406e-22
3.596296013411913e-23

lr = 1e3 diverge
22.911476135253906
8271.04296875
1428539.25
158909728.0
12871688192.0
812350963712.0
41703440384000.0
1794260437303296.0
6.61326136386519e+16
2.123591672645288e+18
```