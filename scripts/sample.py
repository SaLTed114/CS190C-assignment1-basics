from __future__ import annotations

import argparse
import pickle
import json
from pathlib import Path

import torch

from cs336_basics.config import ConfigIO, ExperimentConfig
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.transformer import TransformerLM

def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)

def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None
) -> torch.Tensor:
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1)
    
    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k)
        kth = values[..., [-1]]
        logits = torch.where(logits < kth, torch.full_like(logits, float('-inf')), logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    device: torch.device,
    temperature: float = 1.0,
    top_k: int | None = None
) -> list[int]:
    model.eval()
    ids = torch.tensor(prompt_ids, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        idx_cond = ids[:, -model.context_length:]
        seq_len = idx_cond.size(1)

        mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0)

        logits = model(idx_cond, mask=mask)
        next_token_logits = logits[:, -1, :]

        next_token = sample_next_token(next_token_logits, temperature=temperature, top_k=top_k)
        ids = torch.cat((ids, next_token.unsqueeze(0)), dim=1)

    return ids.squeeze(0).tolist()

def load_cfg(checkpoint_path: Path, config_path: Path | None) -> ExperimentConfig:
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "config" in ckpt:
        return ExperimentConfig.from_dict(ckpt["config"])

    if config_path is None:
        raise ValueError("Checkpoint does not contain config and no config path provided.")
    return ConfigIO.load(config_path)

def load_tokenizer(config_path: Path) -> BPETokenizer:
    tokenizer_cfg_path = config_path / "tokenizer_config.json"
    if tokenizer_cfg_path.exists():
        with open(tokenizer_cfg_path, "r", encoding="utf-8") as f:
            tokenizer_cfg = json.load(f)
        
        return BPETokenizer.from_files(
            vocab_path=tokenizer_cfg["vocab_path"],
            merges_path=tokenizer_cfg["merges_path"],
            special_tokens=tokenizer_cfg.get("special_tokens", [])
        )
    
    pkl_path = config_path / "tokenizer.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
        
    raise FileNotFoundError("No tokenizer configuration found in data directory.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained Transformer language model.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--config", type=Path, help="Path to the config JSON file (if not included in checkpoint).")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Initial prompt for generation.")
    parser.add_argument("--max_new_tokens", type=int, default=120, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter.")
    parser.add_argument("--device", type=str, default="auto", help="Device to run the model on (e.g., 'cpu', 'cuda', or 'auto').")
    args = parser.parse_args()

    cfg = load_cfg(args.checkpoint, args.config)
    device = resolve_device(args.device)

    tokenizer = load_tokenizer(args.config)

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
        dtype=torch.float32
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)
    if len(prompt_ids) == 0:
        raise ValueError("Prompt must contain at least one token.")
    
    output_ids = generate(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k
    )
    output_text = tokenizer.decode(output_ids)

    print("===== PROMPT =====")
    print(args.prompt, end="\n\n")
    print("===== OUTPUT =====")
    print(output_text)

if __name__ == "__main__":
    main()