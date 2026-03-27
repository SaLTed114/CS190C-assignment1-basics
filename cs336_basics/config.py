from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass
class DataConfig:
    train_npy: str
    valid_npy: str
    tokenizer_path: str
    meta_path: str

@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    theta: float = 10000.0
    eps: float = 1e-5

@dataclass
class OptimConfig:
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8

@dataclass
class TrainConfig:
    batch_size: int = 16
    max_steps: int = 1000
    eval_every: int = 100
    eval_steps: int = 20
    log_every: int = 20
    save_every: int = 200
    device: str = "auto"
    seed: int = 42
    compile_model: bool = False
    num_workers: int = 0

@dataclass
class ExperimentConfig:
    experiment_name: str
    out_dir: str
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig
    train: TrainConfig

    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> ExperimentConfig:
        return cls(
            experiment_name=d["experiment_name"],
            out_dir=d["out_dir"],
            data=DataConfig(**d["data"]),
            model=ModelConfig(**d["model"]),
            optim=OptimConfig(
                lr=d["optim"].get("lr", 3e-4),
                min_lr=d["optim"].get("min_lr", 3e-5),
                warmup_steps=d["optim"].get("warmup_steps", 100),
                weight_decay=d["optim"].get("weight_decay", 0.1),
                grad_clip=d["optim"].get("grad_clip", 1.0),
                betas=tuple(d["optim"].get("betas", (0.9, 0.95))),
                eps=d["optim"].get("eps", 1e-8)
            ),
            train=TrainConfig(**d["train"])
        )
    
class ConfigIO:
    @staticmethod
    def save(config: ExperimentConfig, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=4)

    @staticmethod
    def load(path: str | Path) -> ExperimentConfig:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        return ExperimentConfig.from_dict(d)