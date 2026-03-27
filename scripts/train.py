from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import torch

from cs336_basics.config import ConfigIO
from cs336_basics.logging_utils import JsonlLogger
from cs336_basics.optimizer import AdamW
from cs336_basics.trainer import LMTrainer
from cs336_basics.transformer import TransformerLM

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Transformer language model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment config JSON file.")
    parser.add_argument("--resume", action="store_true", help="Whether to resume training from the latest checkpoint.")
    args = parser.parse_args()

    cfg = ConfigIO.load(args.config)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ConfigIO.save(cfg, out_dir / "config.json")

    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)

    logger = JsonlLogger(out_dir)
    logger.log_message(f"Loaded config from {args.config}")
    logger.log_message(f"Using device: {device}")

    train_data = np.load(cfg.data.train_npy, mmap_mode="r")
    valid_data = np.load(cfg.data.valid_npy, mmap_mode="r")

    logger.log_message(f"train tokens: {len(train_data)}")
    logger.log_message(f"valid tokens: {len(valid_data)}")
    logger.log_message(f"vocab size: {cfg.model.vocab_size}")

    model = TransformerLM(
        vocab_size=cfg.model.vocab_size,
        context_length=cfg.model.context_length,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        theta=cfg.model.theta,
        eps=cfg.model.eps,
        device=device,
        dtype=torch.float32,
    )

    if cfg.train.compile_model and hasattr(torch, "compile"):
        logger.log_message("Compiling the model with torch.compile()")
        model = torch.compile(model)

    optimizer = AdamW(
        params=model.parameters(),
        lr=cfg.optim.lr,
        betas=cfg.optim.betas,
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay,
    )

    trainer = LMTrainer(
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        valid_data=valid_data,
        cfg=cfg,
        logger=logger,
        device=device,
    )
    trainer.train(resume=args.resume)

if __name__ == "__main__":
    main()