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

