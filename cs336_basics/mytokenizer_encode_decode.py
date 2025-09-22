from __future__ import annotations

from typing import Iterable, Iterator, Optional, Dict, List, Tuple
import collections

# Import helpers from our training module
from .mytokenizer import (
    deserialize_load_vocab,
    deserialize_load_merges,
    _compile_special_pattern,
    TOKEN_RE,
    SINGLE_BYTES,
)

class Tokenizer:
    '''
    given a vocabulary and a list of merges, encodes text into integer IDs and decodes integer IDs into text. 
    '''
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ):
        '''
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens. 
        This function should accept the following parameters:
            vocab: dict[int, bytes]
            merges: list[tuple[bytes, bytes]]
            special_tokens: list[str] | None = None
        '''
        self.vocab: Dict[int, bytes] = vocab
        self.merges: List[Tuple[bytes, bytes]] = merges
        self.special_tokens: List[str] | None = special_tokens or None

        # Build fast lookup tables
        self.bytes_to_id: Dict[bytes, int] = {b: i for i, b in self.vocab.items()}
        # Merge ranks: earlier merges have lower rank (higher priority)
        self.rank: Dict[Tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }
        # Precompile special token pattern (longest first) if provided
        self._special_pat = _compile_special_pattern(self.special_tokens or [])
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None,
    ) -> "Tokenizer":
        '''
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens. This method should accept the following additional parameters:
            vocab_filepath: str
            merges_filepath: str
            special_tokens: list[str] | None = None
        '''
        vocab = deserialize_load_vocab(vocab_filepath)
        merges = deserialize_load_merges(merges_filepath)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        '''
        Encode an input text into a sequence of token IDs.
        '''
        out: List[int] = []
        if text == "":
            return out

        # If we have special tokens, split the text such that specials are preserved as single units
        if self._special_pat is not None:
            parts = self._special_pat.split(text)
            for i, part in enumerate(parts):
                if part == "":
                    continue
                if i % 2 == 1:
                    # This is a special token occurrence
                    tok_id = self._lookup_special(part)
                    if tok_id is not None:
                        out.append(tok_id)
                    else:
                        # If it's somehow not in vocab, fall back to normal encoding of the literal text
                        out.extend(self._encode_plain(part))
                else:
                    out.extend(self._encode_plain(part))
        else:
            out.extend(self._encode_plain(text))

        return out

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        '''
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        '''
        for chunk in iterable:
            if not chunk:
                continue
            # stream out ids without holding more than necessary
            for _id in self.encode(chunk):
                yield _id
        
    def decode(self, ids: list[int]) -> str:
        '''
        Decode a sequence of token IDs into text.
        '''
        if not ids:
            return ""
        try:
            return b"".join(self.vocab[i] for i in ids).decode("utf-8")
        except KeyError as e:
            raise ValueError(f"Unknown token id during decode: {e}")
        except UnicodeDecodeError:
            # Fallback for single-token decodes that may split a multi-byte UTF-8 sequence.
            # Full-sequence decodes should not hit this path since concatenated bytes are valid UTF-8.
            return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="ignore")

    # -----------------
    # Internal helpers
    # -----------------
    def _lookup_special(self, token_str: str) -> Optional[int]:
        if token_str is None:
            return None
        b = token_str.encode("utf-8")
        return self.bytes_to_id.get(b)

    def _encode_plain(self, text: str) -> List[int]:
        """
        Encode text that contains no special tokens: GPT-2 style byte-level BPE.
        """
        ids: List[int] = []
        if text == "":
            return ids
        # Tokenize into GPT-2 regex tokens first
        for m in TOKEN_RE.finditer(text):
            token = m.group()
            if token == "":
                continue
            # Start from raw UTF-8 bytes split into single-byte symbols
            b = token.encode("utf-8")
            seq: List[bytes] = [SINGLE_BYTES[x] for x in b]
            if not seq:
                continue
            # Repeatedly merge the best-ranked adjacent pair
            seq = self._bpe_merge(seq)
            # Map each bytes token to ID
            for t in seq:
                try:
                    ids.append(self.bytes_to_id[t])
                except KeyError:
                    # Should not happen if vocab includes base-256 and merges, but guard anyway
                    raise ValueError(f"Unknown token bytes not in vocab: {t!r}")
        return ids

    def _bpe_merge(self, seq: List[bytes]) -> List[bytes]:
        if len(seq) < 2:
            return seq
        # Build initial list of pairs with ranks
        while True:
            best_rank = None
            best_idx = -1
            # scan adjacent pairs to find the best (lowest rank)
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                r = self.rank.get(pair)
                if r is None:
                    continue
                if (best_rank is None) or (r < best_rank):
                    best_rank = r
                    best_idx = i
            if best_rank is None:
                # no more applicable merges
                break
            # Merge all non-overlapping occurrences of the selected pair in a single left-to-right pass
            i = 0
            new_seq: List[bytes] = []
            a, b = seq[best_idx], seq[best_idx + 1]
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    new_seq.append(a + b)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            seq = new_seq
            if len(seq) < 2:
                break
        return seq

