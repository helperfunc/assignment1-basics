import base64
import json
from typing import Dict, List, Tuple

def load_vocab(path: str) -> Dict[int, bytes]:
    with open(path, "r") as f:
        payload = json.load(f)
    return {int(k): base64.b64decode(v.encode("utf-8")) for k, v in payload.items()}

def load_merges(path: str) -> List[Tuple[bytes, bytes]]:
    with open(path, "r") as f:
        payload = json.load(f)
    return [
        (base64.b64decode(a.encode("utf-8")), base64.b64decode(b.encode("utf-8")))
        for a, b in payload
    ]

def summarize(name: str, vocab_path: str, merges_path: str) -> None:
    vocab = load_vocab(vocab_path)
    merges = load_merges(merges_path)

    tokens = list(vocab.values())
    lengths = [len(tok) for tok in tokens]
    longest = max(tokens, key=len)

    print(f"{name}:")
    print(f"  vocab size: {len(vocab):>6}")
    print(f"  merges:     {len(merges):>6}")
    print(f"  avg bytes/token: {sum(lengths) / len(lengths):.2f}")
    print(f"  max bytes/token: {len(longest)}")
    print(f"  longest token (utf-8, errors ignored): {longest.decode('utf-8', 'ignore')!r}")
    print(f"  first 5 merges: {[ (a.decode('utf-8', 'ignore'), b.decode('utf-8', 'ignore')) for a, b in merges[:5] ]}")
    print(f"  last 5 merges: {[ (a.decode('utf-8', 'ignore'), b.decode('utf-8', 'ignore')) for a, b in merges[-5:] ]}")
    print()

if __name__ == "__main__":
    summarize(
        "TinyStories",
        "tinystories_vocab",
        "tinystories_merges",
    )
    summarize(
        "OpenWebText",
        "expts_owt_vocab",
        "expts_owt_merges",
    )