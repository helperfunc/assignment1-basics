import os
import regex as re
import multiprocessing as mp
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, BinaryIO, Set
import heapq
import mmap
from tqdm import tqdm

# ----------------------------
# Public helpers expected elsewhere
# ----------------------------

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
TOKEN_RE = re.compile(PAT)
_WORD_PAT = re.compile(PAT)
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
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            data = mm[start:end]
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

    pbar = tqdm(total=len(tasks), desc="Counting chunks", unit="chunk", mininterval=0.1, leave=False)
    try:
        try:
            with mp.Pool(processes=worker_count, maxtasksperchild=64) as pool:
                chunksize = max(1, len(tasks) // (worker_count * 4))
                for local in pool.imap_unordered(_wordcount_worker, tasks, chunksize=chunksize):
                    for k, v in local.items():
                        wf[k] += v
                    pbar.update(1)
        except (BrokenPipeError, OSError):
            # fallback path
            for t in tasks:
                local = _wordcount_worker(t)
                for k, v in local.items():
                    wf[k] += v
                pbar.update(1)
    finally:
        pbar.close()

    return wf

# ----------------------------
# Training core (deterministic tie-break on BYTES)
# ----------------------------

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

def _merge_positions_nonoverlap(w: List[int], a: int, b: int) -> List[int]:
    """Positions of non-overlapping (a,b) matches, matching the merge pass."""
    pos = []
    i, L = 0, len(w)
    while i + 1 < L:
        if w[i] == a and w[i+1] == b:
            pos.append(i)
            i += 2
        else:
            i += 1
    return pos

def _merge_word_all_with_positions(w: List[int], a: int, b: int, new_id: int) -> Tuple[List[int], List[int]]:
    """Merge all (a,b)->new_id and also return indices where new_id was inserted."""
    out = []
    new_pos = []
    i, L = 0, len(w)
    while i < L:
        if i + 1 < L and w[i] == a and w[i+1] == b:
            new_pos.append(len(out))
            out.append(new_id)
            i += 2
        else:
            out.append(w[i])
            i += 1
    return out, new_pos

def _desc_lex_key(b: bytes) -> tuple[int, ...]:
    """Key for descending lexicographic compare on bytes (GPT-2 tie-break).
    Compare (255-x) elementwise and append 256 so longer strings > prefixes."""
    return tuple(255 - x for x in b) + (256,)

def BPE_tokenizer_training(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
    num_processes: int = 8,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    # ---- sanity on size ----
    base_size = 256 + len(special_tokens)
    if vocab_size < base_size:
        raise ValueError(
            f"vocab_size {vocab_size} < minimum required {base_size} (256 bytes + {len(special_tokens)} specials)"
        )

    # ---- initialize vocab: raw bytes then specials ----
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for s in special_tokens:
        vocab[next_id] = s.encode("utf-8")
        next_id += 1

    # ---- collect word frequencies (byte-ids per word) ----
    word_freqs = _collect_word_freqs_parallel(input_path, special_tokens, num_processes)
    words: List[List[int]] = [list(w) for w in word_freqs.keys()]
    freqs: List[int] = [word_freqs[w] for w in word_freqs.keys()]
    del word_freqs

    # ---- initial pair counts + inverted index (pair -> set(word_idx)) ----
    Pair = Tuple[int, int]
    pair_counts: Dict[Pair, int] = defaultdict(int)
    pair_to_words: Dict[Pair, Set[int]] = defaultdict(set)

    for wi, (w, f) in enumerate(zip(words, freqs)):
        if len(w) < 2:
            continue
        for i in range(len(w) - 1):
            p = (w[i], w[i + 1])
            pair_counts[p] += f
            pair_to_words[p].add(wi)

    merges: List[Tuple[bytes, bytes]] = []

    # ---- tie-break key cache for every token id ----
    # ids are contiguous and monotonically increasing, so list-index == token id
    lex_key: List[tuple[int, ...]] = [None] * (256 + len(special_tokens))
    for i in range(256):
        lex_key[i] = _desc_lex_key(bytes([i]))
    nid0 = 256
    for s in special_tokens:
        lex_key[nid0] = _desc_lex_key(s.encode("utf-8"))
        nid0 += 1

    # ---- single heap with lazy invalidation ----
    # store: (-count, lex_key[a], lex_key[b], a, b)
    heap: List[tuple[int, tuple[int,...], tuple[int,...], int, int]] = []
    for (a, b), c in pair_counts.items():
        if c > 0:
            heapq.heappush(heap, (-c, lex_key[a], lex_key[b], a, b))

    def _push(a: int, b: int):
        c = pair_counts.get((a, b), 0)
        if c > 0:
            heapq.heappush(heap, (-c, lex_key[a], lex_key[b], a, b))

    # ---- progress bar setup ----
    merges_target = vocab_size - len(vocab)
    pbar = tqdm(total=max(0, merges_target), desc="BPE merges", unit="merge", mininterval=0.1, leave=False)

    try:
        # ---- main training loop ----
        while len(vocab) < vocab_size and pair_counts and heap:
            # pop stale entries until the top reflects current count
            while heap:
                negc, _, _, a, b = heap[0]
                cur = pair_counts.get((a, b), 0)
                if cur > 0 and -negc == cur:
                    best_pair = (a, b)
                    best_count = cur
                    break
                heapq.heappop(heap)  # stale
            else:
                break

            if best_count < 1:
                break

            a, b = best_pair
            new_index = next_id
            next_id += 1

            # record merge (in bytes) & register new token
            merges.append((vocab[a], vocab[b]))
            vocab[new_index] = vocab[a] + vocab[b]
            # add tie-break key for the newly formed token
            lex_key.append(_desc_lex_key(vocab[new_index]))

            # progress
            pbar.update(1)
            pbar.set_postfix(top_count=best_count)

            # words affected by (a,b)
            affected = list(pair_to_words.get(best_pair, ()))

            for wi in affected:
                w = words[wi]
                f = freqs[wi]
                if len(w) < 2:
                    continue

                # 1) non-overlapping positions of (a,b)
                pos = _merge_positions_nonoverlap(w, a, b)
                if not pos:
                    continue

                # 2) decrement ONLY neighborhoods in OLD word: {s-1, s, s+1}
                old_aff_starts = set()
                for s in pos:
                    if s - 1 >= 0: old_aff_starts.add(s - 1)
                    old_aff_starts.add(s)
                    if s + 1 < len(w) - 1: old_aff_starts.add(s + 1)

                for i0 in old_aff_starts:
                    if 0 <= i0 < len(w) - 1:
                        p = (w[i0], w[i0 + 1])
                        old_c = pair_counts.get(p)
                        if old_c is not None:
                            new_c = old_c - f
                            if new_c <= 0:
                                pair_counts.pop(p, None)
                                # safe to free entire set when the pair disappears globally
                                pair_to_words.pop(p, None)
                            else:
                                pair_counts[p] = new_c
                                _push(p[0], p[1])
                        # NOTE: do NOT discard wi from pair_to_words[p] here.
                        # Over-inclusion is safe and preserves correctness.

                # 3) merge and collect where new_id was inserted
                new_w, new_pos = _merge_word_all_with_positions(w, a, b, new_index)

                # 4) increment ONLY neighborhoods in NEW word: {j-1, j}
                new_aff_starts = set()
                for j in new_pos:
                    if j - 1 >= 0: new_aff_starts.add(j - 1)
                    if j < len(new_w) - 1: new_aff_starts.add(j)

                for i1 in new_aff_starts:
                    if 0 <= i1 < len(new_w) - 1:
                        p = (new_w[i1], new_w[i1 + 1])
                        new_c = pair_counts.get(p, 0) + f
                        pair_counts[p] = new_c
                        pair_to_words[p].add(wi)  # over-inclusive is OK
                        _push(p[0], p[1])

                # 5) commit the new word
                w[:] = new_w

            # (a,b) no longer valid
            pair_to_words.pop(best_pair, None)

    finally:
        pbar.close()

    return vocab, merges
