from __future__ import annotations

import math
import torch
from torch import Tensor
from torch import nn

def softmax(x: Tensor, dim: int) -> Tensor:
    max_x = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_x)
    sum_exp_x = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp_x

def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    if logits.shape[:-1] != targets.shape:
        raise ValueError("Logits and targets shapes are incompatible")

    vocab_size = logits.shape[-1]
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    max_logits = torch.max(logits_flat, dim=-1, keepdim=True).values
    shifted_logits = logits_flat - max_logits
    logsumexp = max_logits.squeeze(-1) + torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))

    target_logits = logits_flat.gather(dim=-1, index=targets_flat.unsqueeze(1)).squeeze(1)

    loss = logsumexp - target_logits
    return loss.mean()


class Linear(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        
        self.weight = nn.Parameter(torch.empty((d_out, d_in), device=device, dtype=dtype))

        if bias:
            self.bias = nn.Parameter(torch.empty((d_out,), device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.d_in)
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.uniform_(-bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.weight.transpose(0, 1)
        if self.bias is not None:
            y = y + self.bias
        return y


class SiLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, bias=False, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        self.silu = SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))


class Embedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.weight = nn.Parameter(torch.empty((vocab_size, d_model), device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.normal_(mean=0.0, std=1.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-6,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.empty((d_model,), device=device, dtype=dtype))

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.fill_(1.0)

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
