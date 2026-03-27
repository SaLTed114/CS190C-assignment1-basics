#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import (
    BPETokenizer,
    BYTE_TO_UNICODE,
    train_bpe,
)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def pick_dtype(vocab_size: int) -> np.dtype:
    if vocab_size <= np.iinfo(np.uint16).max + 1:
        return np.uint16
    return np.uint32


def bytes_to_gpt2_token_str(bs: bytes) -> str:
    return "".join(BYTE_TO_UNICODE[b] for b in bs)


def save_vocab_json(vocab: dict[int, bytes], path: Path) -> None:
    vocab_json: dict[str, int] = {}
    for token_id in sorted(vocab.keys()):
        token_bytes = vocab[token_id]
        token_str = bytes_to_gpt2_token_str(token_bytes)
        vocab_json[token_str] = token_id

    path.write_text(
        json.dumps(vocab_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_merges_txt(merges: list[tuple[bytes, bytes]], path: Path) -> None:
    lines = []
    for a, b in merges:
        a_str = bytes_to_gpt2_token_str(a)
        b_str = bytes_to_gpt2_token_str(b)
        lines.append(f"{a_str} {b_str}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_txt", type=Path, required=True)
    parser.add_argument("--valid_txt", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument(
        "--special_tokens",
        type=str,
        nargs="*",
        default=[],
    )
    parser.add_argument(
        "--save_pickle",
        action="store_true",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] reading train text: {args.train_txt}")
    train_text = read_text(args.train_txt)
    print(f"train chars: {len(train_text):,}")

    print(f"[2/6] reading valid text: {args.valid_txt}")
    valid_text = read_text(args.valid_txt)
    print(f"valid chars: {len(valid_text):,}")

    print(f"[3/6] training BPE tokenizer, vocab_size={args.vocab_size}")
    vocab, merges = train_bpe(
        input_path=args.train_txt,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
    )
    tokenizer = BPETokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=args.special_tokens,
    )

    actual_vocab_size = len(vocab)
    dtype = pick_dtype(actual_vocab_size)

    print(f"actual vocab size: {actual_vocab_size:,}")
    print(f"token dtype: {np.dtype(dtype)}")

    print("[4/6] encoding train / valid")
    train_ids = np.fromiter(tokenizer.encode_iterable(train_text.splitlines(keepends=True)), dtype=dtype)
    valid_ids = np.fromiter(tokenizer.encode_iterable(valid_text.splitlines(keepends=True)), dtype=dtype)

    print(f"train tokens: {len(train_ids):,}")
    print(f"valid tokens: {len(valid_ids):,}")
    print(f"train tokens/chars: {len(train_ids) / max(len(train_text), 1):.4f}")
    print(f"valid tokens/chars: {len(valid_ids) / max(len(valid_text), 1):.4f}")

    print("[5/6] saving arrays")
    np.save(out_dir / "train.npy", train_ids)
    np.save(out_dir / "valid.npy", valid_ids)

    print("[6/6] saving tokenizer artifacts")
    save_vocab_json(vocab, out_dir / "vocab.json")
    save_merges_txt(merges, out_dir / "merges.txt")

    tokenizer_cfg = {
        "type": "bpe",
        "vocab_path": str((out_dir / "vocab.json").resolve()),
        "merges_path": str((out_dir / "merges.txt").resolve()),
        "special_tokens": args.special_tokens,
        "actual_vocab_size": actual_vocab_size,
    }
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.save_pickle:
        with open(out_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)

    meta = {
        "train_txt": str(args.train_txt),
        "valid_txt": str(args.valid_txt),
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": actual_vocab_size,
        "special_tokens": args.special_tokens,
        "train_num_tokens": int(len(train_ids)),
        "valid_num_tokens": int(len(valid_ids)),
        "dtype": str(np.dtype(dtype)),
        "train_npy": str((out_dir / "train.npy").resolve()),
        "valid_npy": str((out_dir / "valid.npy").resolve()),
        "vocab_json": str((out_dir / "vocab.json").resolve()),
        "merges_txt": str((out_dir / "merges.txt").resolve()),
        "tokenizer_config_json": str((out_dir / "tokenizer_config.json").resolve()),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    data_config = {
        "train_npy": meta["train_npy"],
        "valid_npy": meta["valid_npy"],
        "tokenizer_path": str((out_dir / "tokenizer.pkl").resolve()) if args.save_pickle else "",
        "meta_path": str((out_dir / "meta.json").resolve()),
    }
    (out_dir / "data_config.json").write_text(
        json.dumps(data_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"saved to: {out_dir}")
    print("files:")
    print(f"  - {out_dir / 'train.npy'}")
    print(f"  - {out_dir / 'valid.npy'}")
    print(f"  - {out_dir / 'vocab.json'}")
    print(f"  - {out_dir / 'merges.txt'}")
    print(f"  - {out_dir / 'tokenizer_config.json'}")
    print(f"  - {out_dir / 'meta.json'}")
    print(f"  - {out_dir / 'data_config.json'}")
    if args.save_pickle:
        print(f"  - {out_dir / 'tokenizer.pkl'}")


if __name__ == "__main__":
    main()