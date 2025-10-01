import argparse
import base64
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

from mytokenizer_encode_decode import Tokenizer

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

def build_tokenizer(prefix: str, special: List[str]) -> Tokenizer:
    vocab = load_vocab(f"{prefix}_vocab")
    merges = load_merges(f"{prefix}_merges")
    return Tokenizer(vocab, merges, special)

def read_samples(path: Path, n_docs: int, separator: str = "<|endoftext|>") -> List[str]:
    docs: List[str] = []
    buf: List[str] = []
    sep = separator.strip()
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip() == sep:
                if buf:
                    docs.append("".join(buf))
                    buf.clear()
                if len(docs) >= n_docs:
                    break
            else:
                buf.append(line)
    if buf and len(docs) < n_docs:
        docs.append("".join(buf))
    return docs[:n_docs]

def compression_ratio(tokenizer: Tokenizer, docs: Iterable[str]) -> float:
    total_bytes = 0
    total_tokens = 0
    for doc in docs:
        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(tokenizer.encode(doc))
    return total_bytes / max(total_tokens, 1)

def tokenize_corpus(tokenizer: Tokenizer, corpus: Iterator[str], chunk_bytes: int = 1_048_576) -> Iterator[int]:
    for chunk in corpus:
        yield from tokenizer.encode(chunk)

def benchmark(tokenizer: Tokenizer, text: str, repeats: int = 3) -> float:
    payload = text.encode("utf-8")
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        tokenizer.encode(text)
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)
    return len(payload) / best if best > 0 else 0.0

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tinystories", type=Path, required=True, help="TinyStories text file")
    parser.add_argument("--openwebtext", type=Path, required=True, help="OpenWebText text file")
    parser.add_argument("--n-docs", type=int, default=10)
    args = parser.parse_args()

    ts_tok = build_tokenizer("tinystories", ["<|endoftext|>"])
    owt_tok = build_tokenizer("expts_owt", ["<|endoftext|>"])

    ts_docs = read_samples(args.tinystories, args.n_docs)
    owt_docs = read_samples(args.openwebtext, args.n_docs)

    ts_ratio = compression_ratio(ts_tok, ts_docs)
    owt_ratio = compression_ratio(owt_tok, owt_docs)
    cross_ratio = compression_ratio(ts_tok, owt_docs)

    sample_text = "\n".join(ts_docs + owt_docs)
    throughput = benchmark(owt_tok, sample_text)

    print(f"TinyStories tokenizer on TinyStories: {ts_ratio:.3f} bytes/token")
    print(f"OpenWebText tokenizer on OpenWebText: {owt_ratio:.3f} bytes/token")
    print(f"TinyStories tokenizer on OpenWebText: {cross_ratio:.3f} bytes/token")
    print(f"Tokenizer throughput (OpenWebText tokenizer): {throughput/1e6:.2f} MB/s")

    pile_seconds = (825 * 1024**3) / max(throughput, 1)
    print(f"Estimated time for 825GB: {pile_seconds/3600:.2f} hours")

if __name__ == "__main__":
    main()