"""
Post-training script: merge LoRA and export clean model.

Handles:
- DeepSpeed ZeRO-2 checkpoint format (global_step* directories)
- State dict key prefix mismatches (module. prefix from DDP/DeepSpeed)
- Manual LoRA → merged nn.Linear conversion
- Correct model architecture preservation (SmolVLM / FastVLM)

Usage:
    # SmolVLM (default)
    python scripts/merge_and_export.py \
        --checkpoint_dir /path/to/output/checkpoint-1000 \
        --output_dir /path/to/merged_model \
        --model_name_or_path /projects/bffz/yzou1/models/SmolVLM2-2.2B-Instruct

    # FastVLM
    python scripts/merge_and_export.py \
        --model_type fastvlm \
        --checkpoint_dir /path/to/output/checkpoint-1000 \
        --output_dir /path/to/merged_model \
        --model_name_or_path /path/to/FastVLM-0.5B
"""

import argparse
import glob
import os
import sys
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_checkpoint_state_dict(checkpoint_dir: str) -> dict:
    """
    Load state dict from various checkpoint formats.

    Handles:
    - pytorch_model.bin (standard HF format)
    - model.safetensors (safetensors format)
    - DeepSpeed ZeRO checkpoint (global_step* with mp_rank files)
    """
    # Try pytorch_model.bin
    pt_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.exists(pt_path):
        logger.info(f"Loading from {pt_path}")
        return torch.load(pt_path, map_location="cpu")

    # Try safetensors
    st_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(st_path):
        logger.info(f"Loading from {st_path}")
        from safetensors.torch import load_file
        return load_file(st_path)

    # Try DeepSpeed ZeRO format
    ds_files = glob.glob(os.path.join(checkpoint_dir, "global_step*", "mp_rank_*_model_states.pt"))
    if not ds_files:
        ds_files = glob.glob(os.path.join(checkpoint_dir, "mp_rank_*_model_states.pt"))

    if ds_files:
        try:
            from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
            logger.info(f"Converting DeepSpeed ZeRO checkpoint from {checkpoint_dir}")
            return get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)
        except Exception as e:
            logger.error(f"Failed to convert DeepSpeed checkpoint: {e}")
            raise

    raise FileNotFoundError(
        f"No checkpoint found in {checkpoint_dir}. "
        f"Expected pytorch_model.bin, model.safetensors, or DeepSpeed checkpoint files."
    )


def fix_state_dict_keys(state_dict: dict, model) -> dict:
    """
    Fix state dict key mismatches between checkpoint and model.
    (unchanged — generic logic)
    """
    model_keys = set(model.state_dict().keys())
    sd_keys = set(state_dict.keys())

    if model_keys == sd_keys:
        logger.info("State dict keys match perfectly")
        return state_dict

    # Try removing 'module.' prefix
    if all(k.startswith('module.') for k in sd_keys):
        logger.info("Removing 'module.' prefix from state dict keys")
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        sd_keys = set(state_dict.keys())

    missing = model_keys - sd_keys
    unexpected = sd_keys - model_keys

    if missing:
        logger.warning(f"Missing {len(missing)} keys in checkpoint (first 5):")
        for k in sorted(missing)[:5]:
            logger.warning(f"  {k}")

    if unexpected:
        logger.warning(f"Unexpected {len(unexpected)} keys in checkpoint (first 5):")
        for k in sorted(unexpected)[:5]:
            logger.warning(f"  {k}")

    matched = model_keys & sd_keys
    logger.info(f"Matched {len(matched)}/{len(model_keys)} model keys")

    return state_dict


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights and export clean model")

    # ── [NEW] model type ──
    parser.add_argument("--model_type", type=str, default="smolvlm",
                        choices=["smolvlm", "fastvlm"],
                        help="Model type: smolvlm or fastvlm")

    parser.add_argument("--checkpoint_dir", required=True,
                        help="Training checkpoint directory")
    parser.add_argument("--output_dir", required=True,
                        help="Where to save the merged clean model")
    parser.add_argument("--model_name_or_path", required=True,
                        help="Original base model path (for architecture)")
    parser.add_argument("--reg_max", type=int, default=32)
    parser.add_argument("--num_time_layers", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_target_modules", type=str, nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"])
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    # ── [CHANGED] dynamic import by model_type ──
    if args.model_type == "smolvlm":
        from models.smolvlm_distime import SmolVLMDisTime, DisTimeConfig
        ModelClass = SmolVLMDisTime
    elif args.model_type == "fastvlm":
        from models.fastvlm_distime import FastVLMDisTime, DisTimeConfig
        ModelClass = FastVLMDisTime
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    from models.merge_lora import merge_lora_weights

    # ================================================================
    # Step 1: Create model with same architecture as training
    # ── [CHANGED] 用 ModelClass 替代硬编码 SmolVLMDisTime ──
    # ================================================================
    logger.info(f"Creating {args.model_type} model from {args.model_name_or_path}")
    distime_config = DisTimeConfig(
        reg_max=args.reg_max,
        num_time_layers=args.num_time_layers,
    )
    model = ModelClass(
        model_name_or_path=args.model_name_or_path,
        distime_config=distime_config,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    # ================================================================
    # Step 2: Apply LoRA to create identical structure as training
    # (unchanged — setup_training 是模型自己的方法)
    # ================================================================
    logger.info("Applying LoRA structure (to match training checkpoint keys)...")
    model.setup_training(
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        freeze_vision=True,
    )

    # ================================================================
    # Step 3: Load trained weights (unchanged)
    # ================================================================
    logger.info(f"Loading checkpoint from {args.checkpoint_dir}")
    state_dict = load_checkpoint_state_dict(args.checkpoint_dir)
    state_dict = fix_state_dict_keys(state_dict, model)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded: {len(missing)} missing, {len(unexpected)} unexpected keys")

    if missing:
        critical_missing = [k for k in missing
                           if 'lora_' in k or 'time_encoder' in k or 'time_decoder' in k]
        if critical_missing:
            logger.error("CRITICAL: Missing trained weight keys!")
            for k in critical_missing[:10]:
                logger.error(f"  {k}")
            raise RuntimeError("Cannot proceed with missing critical weights")

    # ================================================================
    # Step 4: Merge LoRA weights (unchanged — generic)
    # W_merged = W_original + (alpha/r) * B @ A
    # ================================================================
    logger.info("Merging LoRA weights...")
    num_merged = merge_lora_weights(model.base_model)
    logger.info(f"Merged {num_merged} LoRA modules")

    # ================================================================
    # Step 5: Save (unchanged)
    # ================================================================
    os.makedirs(args.output_dir, exist_ok=True)

    # Save base model (clean nn.Linear, loadable by AutoModel)
    logger.info(f"Saving merged base model to {args.output_dir}")
    model.base_model.save_pretrained(args.output_dir)

    # Save DisTime modules
    distime_state = {
        'time_encoder': model.time_encoder.state_dict(),
        'time_decoder': model.time_decoder.state_dict(),
        'config': model.distime_config.__dict__,
    }
    torch.save(distime_state, os.path.join(args.output_dir, 'distime_modules.pt'))

    # Save processor (tokenizer with TIME_STAMP/FRAME_TIME tokens)
    model.processor.save_pretrained(args.output_dir)

    # ================================================================
    # Verify
    # ================================================================
    logger.info("=" * 60)
    logger.info("Export complete!")
    logger.info(f"Output: {args.output_dir}")
    for f in sorted(os.listdir(args.output_dir)):
        size = os.path.getsize(os.path.join(args.output_dir, f))
        logger.info(f"  {f:40s} {size / 1e6:.1f} MB")
    logger.info("=" * 60)
    # ── [CHANGED] 日志提示也用通用类名 ──
    logger.info(f"Load with: {ModelClass.__name__}.from_pretrained('{args.output_dir}')")


if __name__ == "__main__":
    main()
