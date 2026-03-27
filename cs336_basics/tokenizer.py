from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator
from tqdm import tqdm

import regex as re


def bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1)) +
        list(range(ord("¡"), ord("¬") + 1)) +
        list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

BYTE_TO_UNICODE = bytes_to_unicode()
UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}

GPT2_PRETOKENIZER_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

def pretokenize(text: str) -> list[str]:
    return GPT2_PRETOKENIZER_PATTERN.findall(text)

def pretokenize_excluding_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return pretokenize(text)
    
    escaped = sorted((re.escape(st) for st in special_tokens), key=len, reverse=True)
    pattern = re.compile('|'.join(escaped))
    
    pieces: list[str] = []
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            pieces.extend(pretokenize(text[pos:m.start()]))
        pos = m.end()

    if pos < len(text):
        pieces.extend(pretokenize(text[pos:]))

    return pieces

def word_to_initial_ids(piece: str) -> tuple[int, ...]:
    return tuple(piece.encode('utf-8'))

def iter_pairs(word: tuple[int, ...]):
    for i in range(len(word) - 1):
        yield (word[i], word[i + 1])

def merge_word_ids(word: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    a, b = pair
    new_word = []
    i = 0
    n = len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            new_word.append(new_id)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)

def train_bpe(input_path: str | Path, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    text = Path(input_path).read_text(encoding='utf-8')

    word_freq: Counter[tuple[int, ...]] = Counter()

    pbar = tqdm(total=len(text), desc="Counting word frequencies")

    if not special_tokens:
        pos = 0
        for m in GPT2_PRETOKENIZER_PATTERN.finditer(text):
            word_freq[word_to_initial_ids(m.group(0))] += 1
            pbar.update(m.end() - pos)
            pos = m.end()
        pbar.update(len(text) - pos)
    else:
        escaped = sorted((re.escape(st) for st in special_tokens), key=len, reverse=True)
        pattern = re.compile('|'.join(escaped))
        
        pos = 0
        for m in pattern.finditer(text):
            start = m.start()
            if start > pos:
                chunk = text[pos:start]
                for tm in GPT2_PRETOKENIZER_PATTERN.finditer(chunk):
                    word_freq[word_to_initial_ids(tm.group(0))] += 1
            pbar.update(m.end() - pos)
            pos = m.end()
        
        if pos < len(text):
            chunk = text[pos:]
            for tm in GPT2_PRETOKENIZER_PATTERN.finditer(chunk):
                word_freq[word_to_initial_ids(tm.group(0))] += 1
            pbar.update(len(text) - pos)

    pbar.close()

    id_to_token: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges_ids: list[tuple[int, int]] = []

    target_num_merges = vocab_size - 256 - len(special_tokens)
    if target_num_merges < 0:
        raise ValueError(f"Vocab size must be at least {256 + len(special_tokens)}")

    pair_counts: Counter[tuple[int, int]] = Counter()
    pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)

    for word, freq in word_freq.items():
        for pair in iter_pairs(word):
            pair_counts[pair] += freq
            pair_to_words[pair].add(word)

    for _ in tqdm(range(target_num_merges), desc="Training BPE"):
        if not pair_counts:
            break

        best_pair = max(pair_counts.items(), key=lambda kv: (kv[1], id_to_token[kv[0][0]], id_to_token[kv[0][1]]))[0]

        a, b = best_pair
        new_id = len(id_to_token)
        id_to_token[new_id] = id_to_token[a] + id_to_token[b]
        merges_ids.append(best_pair)

        affected_words = list(pair_to_words.get(best_pair, set()))
        if not affected_words:
            continue

        new_entries: Counter[tuple[int, ...]] = Counter()

        for word in affected_words:
            freq = word_freq[word]
            if freq == 0:
                continue

            for pair in iter_pairs(word):
                pair_counts[pair] -= freq
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
                words_set = pair_to_words.get(pair)
                if words_set is not None:
                    words_set.discard(word)
                    if not words_set:
                        del pair_to_words[pair]
            
            merged_word = merge_word_ids(word, best_pair, new_id)
            new_entries[merged_word] += freq
        
        for new_word, freq in new_entries.items():
            word_freq[new_word] = freq
            for pair in iter_pairs(new_word):
                pair_counts[pair] += freq
                pair_to_words[pair].add(new_word)

    vocab: dict[int, bytes] = {token_id: token for token_id, token in id_to_token.items()}
    for st in special_tokens:
        vocab[len(vocab)] = st.encode('utf-8')

    merges: list[tuple[bytes, bytes]] = [(id_to_token[a], id_to_token[b]) for a, b in merges_ids]

    return vocab, merges


def gpt2_token_str_to_bytes(token_str: str) -> bytes:
    return bytes([UNICODE_TO_BYTE[c] for c in token_str])

class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ) -> None:
        self.id_to_token = vocab
        self.token_to_id = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merge_rank: dict[tuple[int, int], int] = {}
        self.merge_result: dict[tuple[int, int], int] = {}

        for rank, (a, b) in enumerate(merges):
            a_id = self.token_to_id[a]
            b_id = self.token_to_id[b]
            new_token = a + b
            new_id = self.token_to_id[new_token]

            self.merge_rank[(a_id, b_id)] = rank
            self.merge_result[(a_id, b_id)] = new_id

        self.byte_value_to_token_id: dict[int, int] = {b: self.token_to_id[bytes([b])] for b in range(256)}

        self.special_tokens = special_tokens or []
        self.special_str_to_id: dict[str, int] = {st: self.token_to_id[st.encode('utf-8')] for st in self.special_tokens}

        if self.special_tokens:
            escaped = sorted((re.escape(st) for st in self.special_tokens), key=len, reverse=True)
            self.special_pattern = re.compile('|'.join(escaped))
        else:
            self.special_pattern = None

    def _encode_piece_into(self, piece: str, out: list[int]) -> None:
        symbols = [self.byte_value_to_token_id[b] for b in piece.encode('utf-8')]

        while True:
            best_rank = None
            best_pair = None
            best_idx = -1

            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.merge_rank.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair
                    best_idx = i

            if best_idx < 0:
                break

            merged_id = self.merge_result[best_pair]

            new_symbols = []
            i = 0
            n = len(symbols)
            while i < n:
                if i == best_idx:
                    new_symbols.append(merged_id)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        out.extend(symbols)

    def _encode_ordinary_into(self, text: str, start: int, end: int, out: list[int]) -> None:
        for m in GPT2_PRETOKENIZER_PATTERN.finditer(text, start, end):
            self._encode_piece_into(m.group(0), out)

    def encode(self, text: str) -> list[int]:
        out: list[int] = []

        if self.special_pattern is None:
            self._encode_ordinary_into(text, 0, len(text), out)
            return out
        
        pos = 0
        for m in self.special_pattern.finditer(text):
            start, end = m.span()

            if start > pos:
                self._encode_ordinary_into(text, pos, start, out)

            out.append(self.special_str_to_id[m.group(0)])
            pos = end

        if pos < len(text):
            self._encode_ordinary_into(text, pos, len(text), out)

        return out

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in tqdm(iterable, desc="Encoding"):
            yield from self.encode(chunk)

    def decode(self, token_ids: Iterable[int]) -> str:
        bs = b"".join(self.id_to_token[token_id] for token_id in token_ids)
        return bs.decode('utf-8', errors='replace')

    @classmethod
    def from_files(
        cls,
        vocab_path: str | Path,
        merges_path: str | Path,
        special_tokens: list[str] | None = None
    ) -> BPETokenizer:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)

        vocab: dict[int, bytes] = {}
        for token_str, token_id in vocab_json.items():
            vocab[int(token_id)] = gpt2_token_str_to_bytes(token_str)

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                a, b = line.split()
                merges.append((gpt2_token_str_to_bytes(a), gpt2_token_str_to_bytes(b)))

        return cls(vocab, merges, special_tokens)