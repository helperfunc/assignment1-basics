#!/usr/bin/env python3
import argparse
import io
import lzma
import os
import re
import tarfile
from typing import Iterator, Optional, Tuple


def _decode_text(data: bytes, max_bytes: Optional[int]) -> str:
    if not data:
        return ""
    if max_bytes is not None:
        data = data[:max_bytes]
    text = data.decode("utf-8", errors="ignore")
    if "\x00" in text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n+", "\n", text).strip()


def _iter_xz_payloads(member_name: str, fh, max_bytes: Optional[int]) -> Iterator[Tuple[str, str]]:
    compressed = fh.read()
    try:
        payload = lzma.decompress(compressed)
    except lzma.LZMAError:
        return
    bio = io.BytesIO(payload)
    try:
        with tarfile.open(fileobj=bio, mode="r:*") as inner_tar:
            for inner in inner_tar.getmembers():
                if not inner.isfile():
                    continue
                inner_fh = inner_tar.extractfile(inner)
                if inner_fh is None:
                    continue
                text = _decode_text(inner_fh.read(), max_bytes)
                if text:
                    yield f"{member_name}/{inner.name}", text
    except tarfile.ReadError:
        text = _decode_text(payload, max_bytes)
        if text:
            base = member_name[:-3] if member_name.endswith(".xz") else member_name
            yield base, text


def iter_openwebtext_tar(tar_path: str, max_bytes: Optional[int]) -> Iterator[Tuple[str, str]]:
    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            name_lower = member.name.lower()
            if name_lower.endswith(".xz"):
                yield from _iter_xz_payloads(member.name, fh, max_bytes)
            else:
                text = _decode_text(fh.read(), max_bytes)
                if text:
                    yield member.name, text


def iter_openwebtext_dir(root: str, max_bytes: Optional[int]) -> Iterator[Tuple[str, str]]:
    for dirpath, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            if not filename.lower().endswith(".tar"):
                continue
            tar_path = os.path.join(dirpath, filename)
            yield from iter_openwebtext_tar(tar_path, max_bytes)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract OpenWebText tar contents to a single text file.")
    ap.add_argument("--input-dir", default="openwebtext/subsets", help="directory containing OpenWebText tar archives")
    ap.add_argument("--output", required=True, help="output text file path")
    ap.add_argument("--max-bytes", type=int, default=None, help="optional cap on decoded bytes per document")
    ap.add_argument("--separator", default="\n<|endoftext|>\n", help="separator inserted between documents")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    count = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for _, text in iter_openwebtext_dir(args.input_dir, args.max_bytes):
            if count > 0:
                out.write(args.separator)
            out.write(text)
            count += 1
    print(f"[done] wrote {count} documents to {args.output}")


if __name__ == "__main__":
    main()