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

### Problem (train_bpe_tinystories): BPE Training on TinyStories
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

### Problem (train_bpe_expts_owt): BPE Training on OpenWebText
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

### Problem (tokenizer): Implementing the tokenizer
`uv run pytest tests/test_tokenizer.py` all test passed.

### Problem (tokenizer_experiments): Experiments with tokenizers
```
$ python cs336_basics/tokenizer_analysis.py --tinystories TinyStories/TinyStories-train.txt --openwebtext openwebtext/owt_corpus.txt --n-docs 10
TinyStories tokenizer on TinyStories: 4.181 bytes/token
OpenWebText tokenizer on OpenWebText: 4.557 bytes/token
TinyStories tokenizer on OpenWebText: 3.320 bytes/token
Tokenizer throughput (OpenWebText tokenizer): 0.39 MB/s
Estimated time for 825GB: 623.97 hours
```
`uint16` stores integers up to `65,535`, which comfortably covers both vocabularies (`10k` and `32k` tokens), while halving the storage footprint compared to `uint32` and keeping NumPy operations fast.
