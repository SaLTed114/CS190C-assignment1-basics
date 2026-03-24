from __future__ import annotations

import math
import torch
from torch import Tensor
from torch.optim import Optimizer


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int
) -> float:
    if warmup_iters < 0:
        raise ValueError(f"Invalid warmup_iters: {warmup_iters}")
    if cosine_cycle_iters < warmup_iters:
        raise ValueError(f"cosine_cycle_iters must be greater than or equal to warmup_iters: {cosine_cycle_iters} < {warmup_iters}")

    if warmup_iters > 0 and it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    if it > cosine_cycle_iters:
        return min_learning_rate
    
    if cosine_cycle_iters == warmup_iters:
        return min_learning_rate
    
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + (max_learning_rate - min_learning_rate) * coeff

@torch.no_grad()
def gradient_clipping(parameters: list[Tensor], max_l2_norm: float) -> None:
    if max_l2_norm <= 0.0:
        raise ValueError(f"Invalid max_l2_norm: {max_l2_norm}")

    params = [p for p in parameters if p.grad is not None]
    if not params:
        return
    
    total_sq_norm = torch.zeros(())
    for p in params:
        total_sq_norm = total_sq_norm + torch.sum(p.grad ** 2)
        
    total_norm = math.sqrt(total_sq_norm)

    if total_norm <= max_l2_norm or total_norm == 0.0:
        return
    
    scale = max_l2_norm / total_norm
    for p in params:
        p.grad.mul_(scale)


class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError('AdamW does not support sparse gradients')
            
            state = self.state[p]

            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

            state['step'] += 1
            step = state['step']

            if weight_decay != 0:
                p.mul_(1 - lr * weight_decay)

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            exp_avg_hat = exp_avg / bias_correction1
            exp_avg_sq_hat = exp_avg_sq / bias_correction2

            denom = exp_avg_sq_hat.sqrt().add_(eps)
            p.addcdiv_(exp_avg_hat, denom, value=-lr)

        return loss