"""
Merge manual LoRA weights back into original Linear layers.

Use this before save_pretrained to get a clean model that can be loaded
with standard AutoModelForImageTextToText.from_pretrained().

Usage:
    from models.merge_lora import merge_lora_weights
    merge_lora_weights(model.base_model)
    model.save_pretrained(output_dir)
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def merge_lora_weights(model: nn.Module) -> int:
    """
    Merge all LoRALinear modules back into nn.Linear (in-place).

    After merging, the model has no LoRA modules and can be saved/loaded
    with standard HuggingFace methods.

    W_merged = W_original + (lora_alpha / r) * B @ A

    Args:
        model: Model containing LoRALinear modules

    Returns:
        Number of modules merged
    """
    # Import here to avoid circular imports; works whether called as
    # models.merge_lora or utils.merge_lora
    try:
        from .manual_lora import LoRALinear
    except ImportError:
        from manual_lora import LoRALinear

    merged = 0
    targets = []

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            targets.append((name, module))

    for name, lora_module in targets:
        # Compute merged weight: W + scaling * B @ A
        with torch.no_grad():
            merged_weight = (
                lora_module.linear.weight.data
                + lora_module.scaling
                * lora_module.lora_B.weight.data @ lora_module.lora_A.weight.data
            )

        # Create clean nn.Linear
        new_linear = nn.Linear(
            lora_module.in_features,
            lora_module.out_features,
            bias=lora_module.linear.bias is not None,
            device=merged_weight.device,
            dtype=merged_weight.dtype,
        )
        new_linear.weight.data = merged_weight
        if lora_module.linear.bias is not None:
            new_linear.bias.data = lora_module.linear.bias.data.clone()

        # Replace in model tree
        atoms = name.split('.')
        parent = model
        for atom in atoms[:-1]:
            if atom.isdigit():
                parent = parent[int(atom)]
            else:
                parent = getattr(parent, atom)
        setattr(parent, atoms[-1], new_linear)

        merged += 1

    logger.info(f"Merged {merged} LoRA modules back into Linear layers")
    return merged
