import json
import base64
from typing import Dict, List, Tuple
from mytokenizer import BPE_tokenizer_training
import cProfile
import os
import pstats

def serialize_save_vocab(vocab: Dict[int, bytes], saved_path: str) -> None:
    # save the serialized vocab to saved_path
    json_data = {k: base64.b64encode(v).decode('utf-8') for k, v in vocab.items()}
    serialized = json.dumps(json_data)
    with open(saved_path, 'w') as f:
        f.write(serialized)

def deserialize_load_vocab(saved_path: str) -> Dict[int, bytes]:
    with open(saved_path, 'r') as f:
        json_data = json.loads(f.read())
    loaded_json = json.loads(json_data)
    return {int(k): base64.b64decode(v.encode('utf-8')) for k, v in loaded_json.items()}

def serialize_save_merges(merges: List[Tuple[bytes, bytes]], saved_path: str) -> None:
    # save the serialized merges to saved_path
    # Convert each tuple of bytes to a list of base64-encoded strings
    json_data = [[base64.b64encode(a).decode('utf-8'), base64.b64encode(b).decode('utf-8')] 
                 for a, b in merges]
    serialized = json.dumps(json_data)
    with open(saved_path, 'w') as f:
        f.write(serialized)

def deserialize_load_merges(saved_path: str) -> List[Tuple[bytes, bytes]]:
    with open(saved_path, 'r') as f:
        json_data = json.loads(f.read())
    return [(base64.b64decode(a.encode('utf-8')), base64.b64decode(b.encode('utf-8'))) 
            for a, b in json_data]

def train_bpe_tinystories(input_path: str, vocab_size: int, special_tokens: List[str],
                          num_processes: int = 8):
    # Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size of 10,000.
    vocab, merges = BPE_tokenizer_training(input_path, vocab_size, special_tokens, num_processes)
    base_dir = os.getcwd()
    serialize_save_vocab(vocab, os.path.join(base_dir, 'tinystories_vocab'))
    serialize_save_merges(merges, os.path.join(base_dir, 'tinystories_merges'))


def train_bpe_expts_owt(input_path: str, vocab_size: int, special_tokens: List[str],
                          num_processes: int = 8):
    # Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size of 10,000.
    vocab, merges = BPE_tokenizer_training(input_path, vocab_size, special_tokens, num_processes)
    base_dir = os.getcwd()
    serialize_save_vocab(vocab, os.path.join(base_dir, 'expts_owt_vocab'))
    serialize_save_merges(merges, os.path.join(base_dir, 'expts_owt_merges'))


if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    # input_path = r"TinyStories/TinyStories-train.txt"
    # vocab_size = 10000
    # special_tokens = ["<|endoftext|>"]
    # # Allow overriding num processes via env; default 8
    # try:
    #     env_proc = int(os.environ.get("BPE_PROCESSES", "16"))
    # except Exception:
    #     env_proc = 16
    # train_bpe_tinystories(input_path, vocab_size, special_tokens, num_processes=env_proc)
    # profiler.disable()
    # profiler.dump_stats('train_bpe_profile.prof')
    # stats = pstats.Stats(profiler)
    # stats.sort_stats('cumulative')
    # stats.print_stats(20)


    profiler = cProfile.Profile()
    profiler.enable()
    input_path = r"openwebtext/owt_corpus.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]
    train_bpe_expts_owt(input_path, vocab_size, special_tokens, num_processes=16)
    profiler.disable()
    profiler.dump_stats('train_bpe_expts_owt_profile.prof')
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)