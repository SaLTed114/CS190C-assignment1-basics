from __future__ import annotations

import torch
from torch import Tensor
from torch import nn

from cs336_basics.nn import RMSNorm, SwiGLU, Embedding, Linear
from cs336_basics.attention import MultiHeadSelfAttentionWithRoPE

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        
        self.ln1 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)

        self.attn = MultiHeadSelfAttentionWithRoPE(
            d_in=d_model,
            d_model=d_model,
            num_heads=num_heads,
            d_k_total=d_model,
            d_v_total=d_model,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype
        )

        self.ln2 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)

        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_size, seq_len, _ = x.shape

        token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        token_positions = token_positions.unsqueeze(0).expand(batch_size, seq_len)

        x = x + self.attn(self.ln1(x), token_positions=token_positions, mask=mask)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=theta,
                eps=eps,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, bias=False, device=device, dtype=dtype)

    def forward(self, token_ids: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_size, seq_len = token_ids.shape

        if seq_len > self.context_length:
            raise ValueError(f"Sequence length {seq_len} exceeds context length {self.context_length}")

        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits