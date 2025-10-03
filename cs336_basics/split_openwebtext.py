import argparse
from pathlib import Path

def split_corpus(corpus_path: Path, train_path: Path, val_path: Path, split: float = 0.9) -> None:
    with corpus_path.open("r", encoding="utf-8") as src:
        docs = src.read().split("<|endoftext|>\n")
    docs = [d for d in docs if d.strip()]
    cut = int(len(docs) * split)
    train_docs, val_docs = docs[:cut], docs[cut:]

    train_path.write_text("\n<|endoftext|>\n".join(train_docs) + "\n<|endoftext|>\n", encoding="utf-8")
    val_path.write_text("\n<|endoftext|>\n".join(val_docs) + "\n<|endoftext|>\n", encoding="utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--val-out", type=Path, required=True)
    parser.add_argument("--split", type=float, default=0.9)
    args = parser.parse_args()
    split_corpus(args.corpus, args.train_out, args.val_out, args.split)