import json
import base64
from typing import Iterable, Iterator, Optional, Dict, List, Tuple

def deserialize_load_vocab(saved_path: str) -> Dict[int, bytes]:
    with open(saved_path, 'r') as f:
        json_data = json.loads(f.read())
    return {int(k): base64.b64decode(v.encode('utf-8')) for k, v in json_data.items()}

def get_longest_token(vocab_file):
    vocab = deserialize_load_vocab(vocab_file)
    # longest_token = max(vocab.values(), key=len)
    longest_token = sorted(vocab.values(), key=lambda x: len(x), reverse=True)[0]
    longest_token_str = longest_token.decode('utf-8', errors='ignore')
    return longest_token_str

# vocab_file = r'tinystories_vocab'
vocab_file = r'expts_owt_vocab'
longest_token_str = get_longest_token(vocab_file)
print(longest_token_str)