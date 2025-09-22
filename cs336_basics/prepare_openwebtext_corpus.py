import argparse
import os
import tarfile
import json
from typing import Iterator, Optional

SPECIAL = "<|endoftext|>"


def _normalize_text(t: str) -> str:
    # Normalize newlines and strip trailing spaces; keep internal spacing
    if "\r" in t:
        t = t.replace("\r\n", "\n").replace("\r", "\n")
    return t


def _read_member_text(tf: tarfile.TarFile, member: tarfile.TarInfo) -> Optional[str]:
    if not member.isfile() or member.size == 0:
        return None
    f = tf.extractfile(member)
    if f is None:
        return None
    data = f.read()
    if not data:
        return None
    # Try JSON first (many OWT variants store json lines or json files)
    name_lower = member.name.lower()
    text: Optional[str] = None
    if name_lower.endswith(".json") or name_lower.endswith(".jsonl"):
        try:
            # Try parse as a single json object
            obj = json.loads(data.decode("utf-8", errors="ignore"))
            if isinstance(obj, dict) and "text" in obj and isinstance(obj["text"], str):
                text = obj["text"]
            else:
                # Fallback: maybe json lines
                lines = data.decode("utf-8", errors="ignore").splitlines()
                buf = []
                for line in lines:
                    try:
                        rec = json.loads(line)
                        if isinstance(rec, dict) and isinstance(rec.get("text"), str):
                            buf.append(rec["text"])
                    except Exception:
                        continue
                if buf:
                    text = "\n\n".join(buf)
        except Exception:
            # Not valid json, fall back to raw text
            pass
    if text is None:
        # Treat as plain text
        text = data.decode("utf-8", errors="ignore")
    text = _normalize_text(text)
    text = text.strip()
    if not text:
        return None
    return text


def iter_openwebtext(root: str) -> Iterator[str]:
    """
    Yield documents as strings from an OpenWebText folder that contains .tar/.tar.gz/.tgz/.tar.xz files
    or plain .txt files. Each yielded string is one document.
    """
    # Walk directory; collect tar files and text files
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            path = os.path.join(dirpath, fn)
            lower = fn.lower()
            # Tarballs
            if lower.endswith(('.tar', '.tar.gz', '.tgz', '.tar.xz', '.tar.bz2')):
                try:
                    with tarfile.open(path, mode='r:*') as tf:
                        for member in tf.getmembers():
                            txt = _read_member_text(tf, member)
                            if txt:
                                yield txt
                except Exception as e:
                    print(f"[warn] Skipping tar {path}: {e}")
            # Plain text files
            elif lower.endswith(('.txt', '.text')):
                try:
                    with open(path, 'rb') as f:
                        data = f.read()
                    txt = data.decode('utf-8', errors='ignore')
                    txt = _normalize_text(txt).strip()
                    if txt:
                        yield txt
                except Exception as e:
                    print(f"[warn] Skipping text {path}: {e}")
            # JSON or JSONL files
            elif lower.endswith(('.json', '.jsonl')):
                try:
                    with open(path, 'rb') as f:
                        data = f.read()
                    # Try parse json or json lines and collect 'text'
                    try:
                        obj = json.loads(data.decode('utf-8', errors='ignore'))
                        if isinstance(obj, dict) and isinstance(obj.get('text'), str):
                            txt = _normalize_text(obj['text']).strip()
                            if txt:
                                yield txt
                        else:
                            # Not a single dict; maybe a list
                            if isinstance(obj, list):
                                for rec in obj:
                                    if isinstance(rec, dict) and isinstance(rec.get('text'), str):
                                        txt = _normalize_text(rec['text']).strip()
                                        if txt:
                                            yield txt
                    except Exception:
                        # Try json lines
                        ok = False
                        for line in data.decode('utf-8', errors='ignore').splitlines():
                            try:
                                rec = json.loads(line)
                                if isinstance(rec, dict) and isinstance(rec.get('text'), str):
                                    txt = _normalize_text(rec['text']).strip()
                                    if txt:
                                        ok = True
                                        yield txt
                            except Exception:
                                continue
                        if not ok:
                            # Fallback: treat as text
                            txt = data.decode('utf-8', errors='ignore')
                            txt = _normalize_text(txt).strip()
                            if txt:
                                yield txt
                except Exception as e:
                    print(f"[warn] Skipping json {path}: {e}")
            else:
                # Unsupported extension - skip silently
                continue


def build_corpus(input_dir: str, output_path: str, special_token: str = SPECIAL, flush_every: int = 1000) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    count = 0
    with open(output_path, 'w', encoding='utf-8') as out:
        for doc in iter_openwebtext(input_dir):
            out.write(doc)
            out.write('\n')
            out.write(special_token)
            out.write('\n')
            count += 1
            if count % flush_every == 0:
                out.flush()
                os.fsync(out.fileno())
                print(f"[info] Wrote {count} docs so far to {output_path}")
    print(f"[done] Wrote {count} documents to {output_path}")


def main():
    ap = argparse.ArgumentParser(description='Prepare OpenWebText corpus as a single text file')
    ap.add_argument('--input', required=True, help='Path to OpenWebText root (folder containing tar files or text/json)')
    ap.add_argument('--output', required=True, help='Output text file to write corpus to')
    ap.add_argument('--special', default=SPECIAL, help='Special token separator to insert between docs')
    args = ap.parse_args()
    build_corpus(args.input, args.output, args.special)


if __name__ == '__main__':
    '''
    python cs336_basics/prepare_openwebtext_corpus.py \
    --input /path/to/OpenWebText/root_or_tar_folder \
    --output /path/to/owt_corpus.txt \
    --special "<|endoftext|>"
    '''
    main()
