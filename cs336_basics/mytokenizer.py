import os
import regex as re
import multiprocessing as mp
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, BinaryIO

# ----------------------------
# Public helpers expected elsewhere
# ----------------------------

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
TOKEN_RE = re.compile(PAT)
SINGLE_BYTES = [bytes([i]) for i in range(256)]

def _compile_special_pattern(special_tokens: List[str]) -> re.Pattern | None:
    if not special_tokens:
        return None
    parts = [re.escape(t) for t in sorted(special_tokens, key=len, reverse=True)]
    return re.compile("(" + "|".join(parts) + ")")

# ----------------------------
# Efficient preprocessing
# ----------------------------

def _choose_split_token(special_tokens: List[str]) -> Optional[str]:
    if not special_tokens:
        return None
    if "<|endoftext|>" in special_tokens:
        return "<|endoftext|>"
    return special_tokens[0]

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> List[int]:
    assert isinstance(split_special_token, bytes)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if desired_num_chunks <= 1 or file_size == 0:
        return [0, file_size]

    chunk_size = max(1, file_size // desired_num_chunks)
    boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    boundaries[-1] = file_size

    mini = 4 * 1024 * 1024  # 4MB probe to reduce syscalls
    for i in range(1, len(boundaries) - 1):
        pos = boundaries[i]
        file.seek(pos)
        scan = pos
        while True:
            buf = file.read(mini)
            if not buf:
                boundaries[i] = file_size
                break
            k = buf.find(split_special_token)
            if k != -1:
                boundaries[i] = scan + k
                break
            scan += len(buf)
    return sorted(set(boundaries))

_WORD_PAT = re.compile(PAT)

def _split_by_special(text: str, special_tokens: List[str]) -> List[str]:
    if not special_tokens:
        return [text]
    toks = sorted(special_tokens, key=len, reverse=True)
    pat = re.compile("|".join(re.escape(t) for t in toks))
    parts = pat.split(text)
    # exclude the special tokens themselves
    return [p for p in parts if p]

def _pre_tokenize_into_word_byte_ids(s: str) -> List[List[int]]:
    # list of words; each word -> list[int] of raw bytes (0..255)
    words = []
    for m in _WORD_PAT.finditer(s):
        w = m.group(0).encode("utf-8")
        words.append(list(w))
    return words

def _read_chunk(input_path: str, start: int, end: int) -> str:
    with open(input_path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    # normalize newlines for deterministic results
    t = data.decode("utf-8", errors="ignore")
    if "\r" in t:
        t = t.replace("\r\n", "\n").replace("\r", "\n")
    return t

def _count_words_in_text(text: str, special_tokens: List[str]) -> Dict[Tuple[int, ...], int]:
    local = defaultdict(int)
    for chunk in _split_by_special(text, special_tokens):
        for word_ids in _pre_tokenize_into_word_byte_ids(chunk):
            if word_ids:
                local[tuple(word_ids)] += 1
    return local

def _wordcount_worker(args) -> Dict[Tuple[int, ...], int]:
    input_path, start, end, special_tokens = args
    text = _read_chunk(input_path, start, end)
    return _count_words_in_text(text, special_tokens)

def _collect_word_freqs_parallel(
    input_path: str,
    special_tokens: List[str],
    num_processes: int,
) -> Dict[Tuple[int, ...], int]:
    # build chunk boundaries using a known split token if we have one
    split = _choose_split_token(special_tokens)
    if split is None:
        with open(input_path, "rb") as fb:
            size = fb.seek(0, os.SEEK_END)
        tasks = [(input_path, 0, size, special_tokens)]
    else:
        with open(input_path, "rb") as fb:
            bounds = find_chunk_boundaries(fb, max(1, num_processes), split.encode("utf-8"))
        tasks = [(input_path, s, e, special_tokens) for s, e in zip(bounds[:-1], bounds[1:]) if e > s]

    if not tasks:
        return {}

    wf = defaultdict(int)
    worker_count = min(max(1, num_processes), len(tasks), 16)
    try:
        with mp.Pool(processes=worker_count, maxtasksperchild=64) as pool:
            for local in pool.imap_unordered(_wordcount_worker, tasks, chunksize=1):
                for k, v in local.items():
                    wf[k] += v
    except (BrokenPipeError, OSError):
        for t in tasks:
            local = _wordcount_worker(t)
            for k, v in local.items():
                wf[k] += v
    return wf

# ----------------------------
# Training core (deterministic tie-break on BYTES)
# ----------------------------

def _pairs_of(word: List[int]) -> List[Tuple[int, int]]:
    if len(word) < 2:
        return []
    return list(zip(word, word[1:]))

def _merge_word_all(word: List[int], a: int, b: int, new_id: int) -> List[int]:
    out = []
    i = 0
    L = len(word)
    while i < L:
        if i + 1 < L and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


def BPE_tokenizer_training(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
    num_processes: int = 8,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    base_size = 256 + len(special_tokens)
    if vocab_size < base_size:
        raise ValueError(
            f"vocab_size {vocab_size} < minimum required {base_size} (256 bytes + {len(special_tokens)} specials)"
        )

    # ---- CORRECT ID SPACE ----
    # IDs 0..255 == raw bytes 0..255
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    # specials come AFTER the 256 bytes (so they don't clash with word byte IDs)
    for s in special_tokens:
        vocab[next_id] = s.encode("utf-8")
        next_id += 1

    # words: list[list[int]] (byte ids), freqs aligned
    word_freqs = _collect_word_freqs_parallel(input_path, special_tokens, num_processes)
    words: List[List[int]] = [list(w) for w in word_freqs.keys()]
    freqs: List[int] = [word_freqs[w] for w in word_freqs.keys()]
    del word_freqs

    # initial counts + inverted index
    pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    pair_to_words: Dict[Tuple[int, int], set] = defaultdict(set)
    for wi, (w, f) in enumerate(zip(words, freqs)):
        if len(w) < 2:
            continue
        for p in zip(w, w[1:]):
            pair_counts[p] += f
            pair_to_words[p].add(wi)

    merges: List[Tuple[bytes, bytes]] = []

    # ---- EXACT reference selection: max by (count, (vocab[a], vocab[b])) ----
    while len(vocab) < vocab_size and pair_counts:
        best_pair = max(
            pair_counts,
            key=lambda p: (pair_counts[p], (vocab[p[0]], vocab[p[1]])),
        )
        best_count = pair_counts[best_pair]
        if best_count < 1:
            break

        a, b = best_pair
        new_index = next_id
        next_id += 1

        # record merge in BYTES and register new token bytes
        merges.append((vocab[a], vocab[b]))
        vocab[new_index] = vocab[a] + vocab[b]

        # only words that had (a,b)
        affected = list(pair_to_words.get(best_pair, ()))

        for wi in affected:
            w = words[wi]
            f = freqs[wi]

            # remove old pairs
            if len(w) >= 2:
                old_pairs = list(zip(w, w[1:]))
                for p in old_pairs:
                    cur = pair_counts.get(p)
                    if cur is not None:
                        cur -= f
                        if cur <= 0:
                            pair_counts.pop(p, None)
                        else:
                            pair_counts[p] = cur
                    s = pair_to_words.get(p)
                    if s is not None:
                        s.discard(wi)

            # merge all (a,b) -> new_index
            w[:] = _merge_word_all(w, a, b, new_index)

            # add new pairs
            if len(w) >= 2:
                new_pairs = list(zip(w, w[1:]))
                for p in new_pairs:
                    pair_counts[p] = pair_counts.get(p, 0) + f
                    pair_to_words[p].add(wi)

        # (a,b) gone
        pair_to_words.pop(best_pair, None)

    return vocab, merges

