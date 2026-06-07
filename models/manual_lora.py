"""
Manual LoRA implementation that avoids HuggingFace PEFT.

Why: PEFT's get_peft_model() wraps the entire model in PeftModel, which adds
ModulesToSaveWrapper, adapter management hooks, and complex parameter iteration.
This causes DeepSpeed broadcast desync across nodes (different ranks enumerate
parameters in different order → NCCL timeout).

This module provides simple LoRA injection: replace nn.Linear in-place with
LoRALinear. No model wrapper, no hooks, clean parameter traversal.
"""

import math
import logging
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """
    LoRA-augmented Linear layer.
    
    Replaces a frozen nn.Linear with: output = frozen_linear(x) + scale * B(A(dropout(x)))
    
    Following the original LoRA paper (Hu et al., 2021):
    - A is initialized with Kaiming uniform
    - B is initialized with zeros (so LoRA output starts at 0)
    - Original weights are frozen
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()

        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r = r
        self.scaling = lora_alpha / r

        # Keep original linear frozen
        self.linear = original_linear
        for param in self.linear.parameters():
            param.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)

        # Dropout
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Match dtype of original
        dtype = next(original_linear.parameters()).dtype
        self.lora_A.weight.data = self.lora_A.weight.data.to(dtype)
        self.lora_B.weight.data = self.lora_B.weight.data.to(dtype)

    def forward(self, x):
        # Original frozen path
        result = self.linear(x)
        # LoRA path
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return result + lora_out * self.scaling

    def extra_repr(self):
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'r={self.r}, scaling={self.scaling:.2f}')


def _get_submodule(model: nn.Module, target: str) -> nn.Module:
    """Get a submodule by dot-separated path."""
    atoms = target.split('.')
    mod = model
    for atom in atoms:
        if hasattr(mod, atom):
            mod = getattr(mod, atom)
        elif atom.isdigit():
            mod = mod[int(atom)]
        else:
            raise AttributeError(f"Module {model.__class__.__name__} has no attribute {atom}")
    return mod


def _set_submodule(model: nn.Module, target: str, new_module: nn.Module):
    """Set a submodule by dot-separated path."""
    atoms = target.split('.')
    parent = _get_submodule(model, '.'.join(atoms[:-1])) if len(atoms) > 1 else model
    setattr(parent, atoms[-1], new_module)


def apply_lora_to_model(
    model: nn.Module,
    target_module_names: List[str],
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    exclude_patterns: Optional[List[str]] = None,
) -> int:
    """
    Apply LoRA to all nn.Linear modules whose name contains any of target_module_names.
    
    Args:
        model: The model to modify (in-place)
        target_module_names: List of substrings to match, e.g. ["q_proj", "v_proj"]
        r: LoRA rank
        lora_alpha: LoRA alpha (scaling = alpha/r)
        lora_dropout: Dropout rate
        exclude_patterns: List of substrings to exclude, e.g. ["vision_model"]
    
    Returns:
        Number of modules replaced
    """
    exclude_patterns = exclude_patterns or []
    replaced = 0
    lora_params = 0

    # Collect all (name, module) pairs first to avoid modifying during iteration
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # Check if name matches any target
        if not any(t in name for t in target_module_names):
            continue
        # Check exclusions
        if any(e in name for e in exclude_patterns):
            continue
        targets.append((name, module))

    for name, module in targets:
        lora_module = LoRALinear(
            original_linear=module,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        _set_submodule(model, name, lora_module)
        replaced += 1
        lora_params += r * module.in_features + r * module.out_features

    logger.info(f"Applied LoRA to {replaced} modules, added {lora_params:,} LoRA parameters "
                f"(r={r}, alpha={lora_alpha})")
    return replaced
