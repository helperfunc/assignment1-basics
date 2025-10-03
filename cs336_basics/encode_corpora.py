import argparse
import array
import base64
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import concurrent.futures
import math

import numpy as np

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

def encode_file(tokenizer: Tokenizer, input_path: Path) -> np.ndarray:
    ids = array.array("H")  # uint16
    with input_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            ids.extend(tokenizer.encode(line))
    return np.frombuffer(ids, dtype=np.uint16)

def resolve_inputs(input_path: Path) -> List[Path]:
    if input_path.is_dir():
        return sorted(p for p in input_path.iterdir() if p.is_file())
    return [input_path]

def encode_file_range(
    tokenizer: Tokenizer,
    input_path: Path,
    start: int,
    end: int,
) -> np.ndarray:
    ids = array.array("H")
    with input_path.open("rb") as f:
        f.seek(start)
        if start:
            f.readline()
        while True:
            pos = f.tell()
            if pos >= end:
                break
            line = f.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="ignore")
            ids.extend(tokenizer.encode(decoded))
    return np.frombuffer(ids, dtype=np.uint16)

def partition_file_offsets(input_path: Path, shards: int) -> List[Tuple[int, int]]:
    size = input_path.stat().st_size
    if size == 0:
        return [(0, 0)]
    chunk = max(1, math.ceil(size / shards))
    offsets = []
    start = 0
    while start < size:
        end = min(size, start + chunk)
        offsets.append((start, end))
        start = end
    return offsets

def encode_shard_process(
    prefix: str,
    specials: List[str],
    input_path: str,
    shard_out: str,
    offset_start: Optional[int] = None,
    offset_end: Optional[int] = None,
) -> Tuple[str, int]:
    tokenizer = build_tokenizer(prefix, specials)
    if offset_start is None or offset_end is None:
        arr = encode_file(tokenizer, Path(input_path))
    else:
        arr = encode_file_range(tokenizer, Path(input_path), offset_start, offset_end)
    np.save(shard_out, arr)
    return shard_out, arr.size

def encode_dataset(
    prefix: str,
    specials: List[str],
    inputs: List[Path],
    out_path: Path,
    workers: int,
    tmp_dir: Path,
    keep_shards: bool,
) -> None:
    if not inputs:
        raise ValueError(f"No input files found for {out_path.name}")
    if len(inputs) == 1 and workers == 1:
        tokenizer = build_tokenizer(prefix, specials)
        np.save(out_path, encode_file(tokenizer, inputs[0]))
        return

    tmp_dir.mkdir(parents=True, exist_ok=True)
    shard_template = f"{out_path.stem}_shard"
    shard_records: List[Tuple[int, Path, int]] = []

    if workers == 1:
        tokenizer = build_tokenizer(prefix, specials)
        for idx, input_path in enumerate(inputs):
            shard_out = tmp_dir / f"{shard_template}_{idx:05d}.npy"
            arr = encode_file(tokenizer, input_path)
            np.save(shard_out, arr)
            shard_records.append((idx, shard_out, arr.size))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {}
            if len(inputs) == 1:
                input_path = inputs[0]
                for idx, (start, end) in enumerate(partition_file_offsets(input_path, workers)):
                    if start >= end:
                        continue
                    shard_out = tmp_dir / f"{shard_template}_{idx:05d}.npy"
                    future = executor.submit(
                        encode_shard_process,
                        prefix,
                        specials,
                        str(input_path),
                        str(shard_out),
                        start,
                        end,
                    )
                    future_to_idx[future] = idx
            else:
                for idx, input_path in enumerate(inputs):
                    shard_out = tmp_dir / f"{shard_template}_{idx:05d}.npy"
                    future = executor.submit(
                        encode_shard_process,
                        prefix,
                        specials,
                        str(input_path),
                        str(shard_out),
                        None,
                        None,
                    )
                    future_to_idx[future] = idx
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                shard_path, shard_len = future.result()
                shard_records.append((idx, Path(shard_path), shard_len))

    shard_records.sort(key=lambda item: item[0])
    total = sum(length for _, _, length in shard_records)
    final = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.uint16, shape=(total,)
    )
    offset = 0
    for _, shard_path, shard_len in shard_records:
        shard_view = np.load(shard_path, mmap_mode="r")
        final[offset : offset + shard_len] = shard_view
        offset += shard_len
        del shard_view
    final.flush()
    del final

    if not keep_shards:
        for _, shard_path, _ in shard_records:
            shard_path.unlink(missing_ok=True)
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owt-train", type=Path, required=True)
    parser.add_argument("--owt-dev", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--keep-shards", action="store_true")
    parser.add_argument("--tmp-dir", type=Path, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_inputs = resolve_inputs(args.owt_train)
    dev_inputs = resolve_inputs(args.owt_dev)

    tmp_root = args.tmp_dir or args.out_dir / "tmp_encode"

    encode_dataset(
        "expts_owt",
        ["<|endoftext|>"],
        train_inputs,
        args.out_dir / "owt_train_ids.npy",
        args.workers,
        tmp_root / "train",
        args.keep_shards,
    )
    encode_dataset(
        "expts_owt",
        ["<|endoftext|>"],
        dev_inputs,
        args.out_dir / "owt_dev_ids.npy",
        args.workers,
        tmp_root / "dev",
        args.keep_shards,
    )

    if not args.keep_shards and args.tmp_dir is None and tmp_root.exists():
        try:
            tmp_root.rmdir()
        except OSError:
            pass

if __name__ == "__main__":
    main()