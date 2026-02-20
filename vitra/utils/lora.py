"""
lora.py — Minimal LoRA (Low-Rank Adaptation) for nn.Linear modules.

Usage:
    from vitra.utils.lora import LoRALinear, apply_lora, lora_params

    apply_lora(model.language_model, ["q_proj", "v_proj"], rank=16, alpha=32)
    # LoRA B is zero-initialized → no initial contribution to output
    # Freeze/unfreeze LoRA params per training phase
"""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Drop-in nn.Linear replacement with low-rank adaptation.

    output = Linear(x) + (x @ A @ B) * (alpha / rank)

    A is kaiming-initialized, B is zero-initialized,
    so the LoRA contribution starts at zero.
    """

    def __init__(self, base_linear: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.linear = base_linear  # original weights, already loaded
        in_f = base_linear.in_features
        out_f = base_linear.out_features
        self.lora_A = nn.Parameter(torch.empty(in_f, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_f))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A @ self.lora_B) * self.scale


def apply_lora(
    model: nn.Module,
    target_modules: list[str],
    rank: int = 16,
    alpha: float = 32.0,
) -> int:
    """Replace matching nn.Linear modules with LoRALinear in-place.

    Returns the number of replaced modules.
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        if not any(name.endswith(t) for t in target_modules):
            continue
        if not isinstance(module, nn.Linear):
            continue
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRALinear(module, rank, alpha))
        replaced += 1
    print(f"  LoRA: {replaced} modules wrapped (rank={rank}, alpha={alpha})")
    return replaced


def lora_params(model: nn.Module):
    """Yield only LoRA A/B parameters from the model."""
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            yield param


def lora_param_count(model: nn.Module) -> int:
    """Count total LoRA parameters."""
    return sum(p.numel() for p in lora_params(model))
