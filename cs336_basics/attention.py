from __future__ import annotations

import math
import torch
from torch import Tensor
from torch import nn

from cs336_basics.nn import Linear, SiLU, softmax

def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = softmax(scores, dim=-1)
    output = attn_weights @ V
    return output

class RoPE(nn.Module):
    def __init__(
        self,
        d_k: int,
        theta: float = 10000.0,
        max_seq_len: int = 4096,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE")
        
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        half_d = d_k // 2
        idx = torch.arange(half_d, device=device)
        inv_freq = theta ** (-2 * idx / d_k)

        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, x: Tensor, token_position: Tensor) -> Tensor:
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]

        angles = token_position.unsqueeze(-1) * self.inv_freq
        
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        out_even = x_even * cos - x_odd * sin
        out_odd  = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int,
        num_heads: int,
        d_k_total: int,
        d_v_total: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        if d_k_total % num_heads != 0:
            raise ValueError("d_k_total must be divisible by num_heads")
        if d_v_total % num_heads != 0:
            raise ValueError("d_v_total must be divisible by num_heads")

        self.d_in = d_in
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k_total = d_k_total
        self.d_v_total = d_v_total
        self.head_dim_qk = d_k_total // num_heads
        self.head_dim_v  = d_v_total // num_heads

        self.q_proj = Linear(d_in, d_k_total, bias=False, device=device, dtype=dtype)
        self.k_proj = Linear(d_in, d_k_total, bias=False, device=device, dtype=dtype)
        self.v_proj = Linear(d_in, d_v_total, bias=False, device=device, dtype=dtype)
        self.output_proj = Linear(d_v_total, d_model, bias=False, device=device, dtype=dtype)

    def _split_heads_qk(self, x: Tensor) -> Tensor:
        *prefix, seq_len, _ = x.shape
        x = x.reshape(*prefix, seq_len, self.num_heads, self.head_dim_qk)
        return x.transpose(-3, -2)
    
    def _split_heads_v(self, x: Tensor) -> Tensor:
        *prefix, seq_len, _ = x.shape
        x = x.reshape(*prefix, seq_len, self.num_heads, self.head_dim_v)
        return x.transpose(-3, -2)
    
    def _merge_heads(self, x: Tensor) -> Tensor:
        *prefix, num_heads, seq_len, head_dim = x.shape
        x = x.transpose(-3, -2)
        x = x.contiguous().reshape(*prefix, seq_len, num_heads * head_dim)
        return x
    
    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        Q = self._split_heads_qk(self.q_proj(x))
        K = self._split_heads_qk(self.k_proj(x))
        V = self._split_heads_v(self.v_proj(x))

        if mask is None:
            seq_len = x.shape[-2]
            casual_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))
            mask = casual_mask

        attn_output = scaled_dot_product_attention(Q, K, V, mask)
        attn_output_merged = self._merge_heads(attn_output)
        output = self.output_proj(attn_output_merged)
        return output


class MultiHeadSelfAttentionWithRoPE(MultiHeadSelfAttention):
    def __init__(
        self,
        d_in: int,
        d_model: int,
        num_heads: int,
        d_k_total: int,
        d_v_total: int,
        theta: float = 10000.0,
        max_seq_len: int = 4096,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__(d_in, d_model, num_heads, d_k_total, d_v_total, device=device, dtype=dtype)
        self.rope = RoPE(d_k=self.head_dim_qk, theta=theta, max_seq_len=max_seq_len, device=device)

    def forward(self, x: Tensor, token_positions: Tensor, mask: Tensor | None = None) -> Tensor:
        Q = self._split_heads_qk(self.q_proj(x))
        K = self._split_heads_qk(self.k_proj(x))
        V = self._split_heads_v(self.v_proj(x))

        rope_positions = token_positions.unsqueeze(-2)
        Q = self.rope(Q, rope_positions)
        K = self.rope(K, rope_positions)

        if mask is None:
            seq_len = x.shape[-2]
            casual_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))
            mask = casual_mask

        attn_output = scaled_dot_product_attention(Q, K, V, mask)
        attn_output_merged = self._merge_heads(attn_output)
        output = self.output_proj(attn_output_merged)
        return output