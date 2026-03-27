from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import time

import numpy as np
import torch
from tqdm import tqdm

from cs336_basics.config import ExperimentConfig
from cs336_basics.data import get_batch
from cs336_basics.logging_utils import JsonlLogger
from cs336_basics.nn import cross_entropy
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping

@dataclass(slots=True)
class TrainState:
    step: int = 0
    best_val_loss: float = float('inf')
    tokens_seen: int = 0

class LMTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: AdamW,
        train_data: np.ndarray,
        valid_data: np.ndarray,
        cfg: ExperimentConfig,
        logger: JsonlLogger,
        device: torch.device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_data = train_data
        self.valid_data = valid_data
        self.cfg = cfg
        self.logger = logger
        self.device = device
        
        self.state = TrainState()

        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.latest_ckpt = self.out_dir / "latest.pt"
        self.best_ckpt = self.out_dir / "best.pt"

        self._mask_cache: dict[int, torch.Tensor] = {}

    def _casual_mask(self, seq_len: int) -> torch.Tensor:
        mask = self._mask_cache.get(seq_len)
        if mask is None:
            mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.device))
            self._mask_cache[seq_len] = mask
        return mask
    
    def _set_lr(self, step: int) -> float:
        lr = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=self.cfg.optim.lr,
            min_learning_rate=self.cfg.optim.min_lr,
            warmup_iters=self.cfg.optim.warmup_steps,
            cosine_cycle_iters=self.cfg.train.max_steps
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr
    
    def _make_batch(self, dataset: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        return get_batch(
            dataset=dataset,
            batch_size=self.cfg.train.batch_size,
            context_size=self.cfg.model.context_length,
            device=self.device,
        )
    
    def _forward_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mask = self._casual_mask(x.shape[1])
        logits = self.model(x, mask=mask)
        loss = cross_entropy(logits, y)
        return loss
    
    def _train_step(self) -> dict:
        self.model.train()

        lr = self._set_lr(self.state.step)

        x, y = self._make_batch(self.train_data)
        loss = self._forward_loss(x, y)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(list(self.model.parameters()), self.cfg.optim.grad_clip)
        self.optimizer.step()

        self.state.step += 1
        self.state.tokens_seen += int(x.numel())

        return {
            "step": self.state.step,
            "split": "train",
            "loss": float(loss.item()),
            "lr": float(lr),
            "tokens_seen": int(self.state.tokens_seen),
        }
    
    @torch.no_grad()
    def evaluate(self, split: str, eval_steps: int | None = None) -> dict:
        if split not in ("train", "valid"):
            raise ValueError(f"Invalid split: {split}")
        
        dataset = self.train_data if split == "train" else self.valid_data
        num_steps = eval_steps if eval_steps is not None else self.cfg.train.eval_steps

        self.model.eval()
        losses: list[float] = []

        for _ in range(num_steps):
            x, y = self._make_batch(dataset)
            loss = self._forward_loss(x, y)
            losses.append(float(loss.item()))

        mean_loss = sum(losses) / len(losses)
        ppl = math.exp(min(mean_loss, 20.0))

        return {
            "step": self.state.step,
            "split": split,
            "loss": mean_loss,
            "ppl": ppl,
            "tokens_seen": int(self.state.tokens_seen),
        }
    
    def save_checkpoint(self, path: str | Path) -> None:
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_state": asdict(self.state),
            "config": self.cfg.to_dict(),
        }
        torch.save(ckpt, path)
        
    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        state_raw = ckpt.get("train_state", {})
        self.state = TrainState(
            step=state_raw.get("step", 0),
            best_val_loss=state_raw.get("best_val_loss", float('inf')),
            tokens_seen=state_raw.get("tokens_seen", 0),
        )

    def train(self, resume: bool = False) -> None:
        if resume and self.latest_ckpt.exists():
            self.load_checkpoint(self.latest_ckpt)
            self.logger.log_message(f"Resumed from {self.latest_ckpt} at step={self.state.step}")

        pbar = tqdm(total=self.cfg.train.max_steps, desc="Training", initial=self.state.step, dynamic_ncols=True)
        train_start_time = time.time()

        while self.state.step < self.cfg.train.max_steps:
            train_metrics = self._train_step()

            if self.state.step % self.cfg.train.log_every == 0 or self.state.step == 1:
                self.logger.log_metrics(train_metrics)
                pbar.set_postfix(
                    loss=f'{train_metrics["loss"]:.4f}',
                    lr=f'{train_metrics["lr"]:.2e}',
                    best=f'{self.state.best_val_loss:.4f}'
                    if math.isfinite(self.state.best_val_loss) else "inf"
                )

            if self.state.step % self.cfg.train.eval_every == 0 or self.state.step == self.cfg.train.max_steps:
                train_eval = self.evaluate(split="train")
                valid_eval = self.evaluate(split="valid")

                self.logger.log_metrics(train_eval)
                self.logger.log_metrics(valid_eval)

                self.logger.log_message(
                    f"[eval] step={self.state.step} | "
                    f"train loss {train_eval['loss']:.4f} ppl {train_eval['ppl']:.2f} | "
                    f"valid loss {valid_eval['loss']:.4f} ppl {valid_eval['ppl']:.2f}"
                )

                self.save_checkpoint(self.latest_ckpt)

                if valid_eval["loss"] < self.state.best_val_loss:
                    self.state.best_val_loss = valid_eval["loss"]
                    self.save_checkpoint(self.best_ckpt)
                    self.logger.log_message(f"New best checkpoint saved at step {self.state.step} with val loss {valid_eval['loss']:.4f}")

            elif self.state.step % self.cfg.train.save_every == 0:
                self.save_checkpoint(self.latest_ckpt)
                self.logger.log_message(f"Checkpoint saved at step {self.state.step}")

            pbar.update(1)

        pbar.close()

        elapsed_time = time.time() - train_start_time
        self.logger.save_summary({
            "experiment_name": self.cfg.experiment_name,
            "final_steps": self.state.step,
            "best_val_loss": self.state.best_val_loss,
            "tokens_seen": self.state.tokens_seen,
            "eslapsed_sec": elapsed_time,
            "latest_checkpoint": str(self.latest_ckpt),
            "best_checkpoint": str(self.best_ckpt),
        })
        self.logger.log_message(f"Training completed in {elapsed_time:.2f} seconds")