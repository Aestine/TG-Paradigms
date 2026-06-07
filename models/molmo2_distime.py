"""
Molmo2-DisTime: Molmo2 with DisTime temporal grounding capabilities.

Molmo2 architecture (Allen AI / Ai2):
- Vision encoder: SigLIP 2 (So400m/14), 378×378, built into Molmo2VisionBackbone
- Connector: MLP projector + multi-headed attention pooling (3×3 windows)
- Language model: Qwen3-4B (decoder-only Transformer)

Key differences from SmolVLM/FastVLM:
- Vision attribute: model.vision_backbone (not vision_model or vision_tower)
- LLM attribute: model.transformer (not text_model or language_model)
- Embedding: Molmo2Embedding (custom, with .embedding + .new_embedding params)
- LoRA targets: att_proj, attn_out, ff_proj, ff_out (Molmo2's fused QKV naming)
- Forward needs: pixel_values_videos, video_token_pooling, video_grids (from video_processor)
- Internal vision merge: build_input_embeddings() adds vision features at image_patch_id
- XOR constraint: cannot pass both input_ids AND inputs_embeds
- Requires trust_remote_code=True
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoConfig,
)

from .time_modules import TimeEncoder, TimeDecoder

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


@dataclass
class DisTimeConfig:
    """Configuration for DisTime modules."""
    reg_max: int = 32
    num_time_layers: int = 3
    hidden_size: int = 2560  # Molmo2-4B default (Qwen3-4B)
    time_loss_weight: float = 1.0
    iou_loss_weight: float = 1.0
    sigma: float = 1.0  # Gaussian sigma for time encoding


# Special tokens
TIME_STAMP_TOKEN = "<TIME_STAMP>"
FRAME_TIME_TOKEN = "<FRAME_TIME>"


class Molmo2DisTime(nn.Module):
    """
    Molmo2 with DisTime temporal grounding.

    Architecture:
    - Base: Molmo2 (SigLIP2 vision backbone + Qwen3-4B LLM)
    - Added: TimeEncoder, TimeDecoder for temporal grounding

    Training:
    - Freeze vision backbone (SigLIP2 + connector)
    - Apply LoRA to LLM (Qwen3 layers)
    - Train DisTime modules end-to-end

    Key differences from FastVLMDisTime:
    - Molmo2 uses vision_backbone (contains ViT + pooling + projector all-in-one)
    - LLM is model.transformer (Molmo2TextModel, not Qwen2 language_model)
    - Embedding is Molmo2Embedding (custom class, not nn.Embedding)
    - LoRA targets: att_proj, attn_out, ff_proj, ff_out (fused QKV + gated MLP naming)
    - Forward requires extra inputs: pixel_values_videos, video_token_pooling, video_grids
    - Molmo2 handles vision-text merge internally via build_input_embeddings()
    """

    def __init__(
        self,
        model_name_or_path: str = "allenai/Molmo2-4B",
        distime_config: Optional[DisTimeConfig] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        use_flash_attention: bool = False,
    ):
        super().__init__()

        self.distime_config = distime_config or DisTimeConfig()

        # Load base model
        logger.info(f"Loading Molmo2 base model from {model_name_or_path}")

        # Molmo2 的 modeling_molmo2.py 声明 _supports_flash_attn=True
        # transformers 在 super().__init__(config) 里会根据这个类属性自动选 flash_attention_2
        # 即使在 config 上设了 sdpa，deep copy 或内部逻辑也可能丢掉
        # 最可靠的做法：直接 monkey-patch 模型类的 _supports_flash_attn
        attn_impl = "flash_attention_2" if use_flash_attention else "sdpa"

        if not use_flash_attention:
            # 先触发 trust_remote_code 的模型类解析和缓存
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                model_name_or_path, trust_remote_code=True,
            )
            # 通过 config.auto_map 找到实际模型类并 patch
            auto_map = getattr(config, 'auto_map', {})
            if auto_map:
                from transformers.dynamic_module_utils import get_class_from_dynamic_module
                for key, class_ref in auto_map.items():
                    try:
                        cls_resolved = get_class_from_dynamic_module(
                            class_ref, model_name_or_path
                        )
                        if hasattr(cls_resolved, '_supports_flash_attn'):
                            cls_resolved._supports_flash_attn = False
                            logger.info(f"Patched {cls_resolved.__name__}._supports_flash_attn = False")
                    except Exception:
                        pass

        self.base_model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )

        # Force use_frame_special_tokens=True so build_batched_videos
        # counts videos by <frame_end> token (matching our prompt format)
        self.base_model.config.use_frame_special_tokens = True
        logger.info("Forced config.use_frame_special_tokens = True")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )

        # Detect hidden size
        config = self.base_model.config
        if hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
            hidden_size = config.text_config.hidden_size
        elif hasattr(config, 'hidden_size'):
            hidden_size = config.hidden_size
        else:
            hidden_size = 2560  # Molmo2-4B default (Qwen3-4B)
            logger.warning(f"Could not detect hidden_size, using default {hidden_size}")

        self.distime_config.hidden_size = hidden_size
        logger.info(f"Detected hidden_size: {hidden_size}")

        # Store image_patch_id for vision merge
        self.image_patch_id = getattr(config, 'image_patch_id', None)
        if self.image_patch_id is None:
            logger.warning("Could not find image_patch_id in config, vision merge may fail")

        # Add special tokens
        self._add_special_tokens()

        # Initialize DisTime modules
        self._init_distime_modules()

        # Store token IDs
        self.time_stamp_token_id = self.processor.tokenizer.convert_tokens_to_ids(TIME_STAMP_TOKEN)
        self.frame_time_token_id = self.processor.tokenizer.convert_tokens_to_ids(FRAME_TIME_TOKEN)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the base model."""
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def _add_special_tokens(self):
        """Add special tokens for DisTime.

        Molmo2 uses Molmo2Embedding (custom class with .embedding + .new_embedding),
        not nn.Embedding. resize_token_embeddings may not work correctly with this,
        so we handle embedding extension manually if needed.
        """
        special_tokens = {
            'additional_special_tokens': [TIME_STAMP_TOKEN, FRAME_TIME_TOKEN]
        }

        num_added = self.processor.tokenizer.add_special_tokens(special_tokens)
        logger.info(f"Added {num_added} special tokens: {TIME_STAMP_TOKEN}, {FRAME_TIME_TOKEN}")

        if num_added > 0:
            new_vocab_size = len(self.processor.tokenizer)

            # Get current embedding layer
            embed_layer = self.base_model.get_input_embeddings()

            # Check if it's Molmo2Embedding (custom) or nn.Embedding
            if hasattr(embed_layer, 'embedding') and hasattr(embed_layer, 'new_embedding'):
                # Molmo2Embedding: has .embedding (vocab_size, D) + .new_embedding (additional, D)
                current_total = embed_layer.embedding.shape[0] + embed_layer.new_embedding.shape[0]

                if new_vocab_size > current_total:
                    # Need to extend new_embedding
                    extra_needed = new_vocab_size - current_total
                    D = embed_layer.new_embedding.shape[1]

                    with torch.no_grad():
                        # Compute mean of existing embeddings for initialization
                        all_embeds = torch.cat([embed_layer.embedding.data,
                                                embed_layer.new_embedding.data], dim=0)
                        embed_avg = all_embeds.mean(dim=0, keepdim=True)

                        # Extend new_embedding
                        new_extra = embed_avg.expand(extra_needed, -1).clone()
                        new_new_embedding = torch.cat([embed_layer.new_embedding.data, new_extra], dim=0)
                        embed_layer.new_embedding = nn.Parameter(new_new_embedding)

                    logger.info(f"Extended Molmo2Embedding.new_embedding by {extra_needed} "
                               f"(total: {current_total} -> {new_vocab_size})")
                else:
                    logger.info(f"Molmo2Embedding already has capacity for {current_total} tokens "
                               f"(need {new_vocab_size}), no extension needed")

                # Initialize new token positions to mean of existing
                with torch.no_grad():
                    all_embeds = torch.cat([embed_layer.embedding.data,
                                            embed_layer.new_embedding.data], dim=0)
                    embed_avg = all_embeds[:-num_added].mean(dim=0, keepdim=True)
                    # The new tokens are at the end
                    total_now = embed_layer.embedding.shape[0] + embed_layer.new_embedding.shape[0]
                    new_start = total_now - num_added
                    # They live in new_embedding
                    base_size = embed_layer.embedding.shape[0]
                    new_start_in_new = new_start - base_size
                    embed_layer.new_embedding.data[new_start_in_new:] = embed_avg
                    logger.info(f"Initialized new token embeddings to mean of existing")

            else:
                # Standard nn.Embedding - use resize_token_embeddings
                self.base_model.resize_token_embeddings(new_vocab_size, mean_resizing=False)

                input_embeddings = self.base_model.get_input_embeddings()
                if input_embeddings is not None:
                    with torch.no_grad():
                        embed_weight = input_embeddings.weight.data
                        embed_avg = embed_weight[:-num_added].mean(dim=0, keepdim=True)
                        embed_weight[-num_added:] = embed_avg
                        logger.info(f"Initialized new input embeddings to mean of existing")

            # Extend lm_head if needed
            lm_head = self.base_model.get_output_embeddings()
            if lm_head is not None and hasattr(lm_head, 'weight'):
                current_out = lm_head.weight.shape[0]
                if new_vocab_size > current_out:
                    extra_needed = new_vocab_size - current_out
                    in_features = lm_head.weight.shape[1]
                    has_bias = lm_head.bias is not None

                    with torch.no_grad():
                        out_avg = lm_head.weight.data.mean(dim=0, keepdim=True)
                        new_weight = torch.cat([lm_head.weight.data,
                                                out_avg.expand(extra_needed, -1).clone()], dim=0)
                        lm_head.weight = nn.Parameter(new_weight)

                        if has_bias:
                            new_bias = torch.cat([lm_head.bias.data,
                                                  torch.zeros(extra_needed, device=lm_head.bias.device,
                                                              dtype=lm_head.bias.dtype)], dim=0)
                            lm_head.bias = nn.Parameter(new_bias)

                    logger.info(f"Extended lm_head by {extra_needed} "
                               f"(total: {current_out} -> {new_vocab_size})")

            # Sync vocab_size to config
            config = self.base_model.config
            if hasattr(config, 'text_config') and hasattr(config.text_config, 'vocab_size'):
                config.text_config.vocab_size = new_vocab_size
                logger.info(f"Updated config.text_config.vocab_size to {new_vocab_size}")
            # Molmo2Config.vocab_size 是只读 property（从 text_config 派生），不能直接赋值
            try:
                config.vocab_size = new_vocab_size
                logger.info(f"Updated config.vocab_size to {new_vocab_size}")
            except (AttributeError, TypeError):
                logger.info(f"config.vocab_size is read-only property, skipped (derives from text_config)")

            # Molmo2 has model.transformer as the LLM
            if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'transformer'):
                if hasattr(self.base_model.model.transformer, 'config'):
                    try:
                        self.base_model.model.transformer.config.vocab_size = new_vocab_size
                        logger.info(f"Updated model.transformer.config.vocab_size to {new_vocab_size}")
                    except (AttributeError, TypeError):
                        logger.info(f"transformer.config.vocab_size is read-only, skipped")

            logger.info(f"Vocab size after adding special tokens: {new_vocab_size}")

    def _init_distime_modules(self):
        """Initialize TimeEncoder, TimeDecoder, and Loss functions."""
        cfg = self.distime_config

        self.time_encoder = TimeEncoder(
            hidden_size=cfg.hidden_size,
            reg_max=cfg.reg_max,
            num_layers=cfg.num_time_layers,
            sigma=cfg.sigma,
        )

        self.time_decoder = TimeDecoder(
            hidden_size=cfg.hidden_size,
            reg_max=cfg.reg_max,
            num_layers=cfg.num_time_layers,
        )

        # Initialize loss functions
        from utils.losses import DistributionFocalLoss, DIoULoss
        self.dfl_loss_fn = DistributionFocalLoss(reg_max=cfg.reg_max)
        self.diou_loss_fn = DIoULoss()

        logger.info(f"Initialized DisTime modules with reg_max={cfg.reg_max}, "
                    f"num_layers={cfg.num_time_layers}, hidden_size={cfg.hidden_size}")

    def freeze_vision_encoder(self):
        """Freeze the vision backbone (SigLIP2 ViT + connector + projector).

        Molmo2 structure:
            Molmo2ForConditionalGeneration
              └── model (Molmo2Model)
                    └── vision_backbone (Molmo2VisionBackbone)
                          ├── image_vit (Molmo2VisionTransformer)
                          ├── image_pooling_2d (ViTMultiHeadDotProductAttention)
                          └── image_projector (ImageProjectorMLP)
        """
        if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'vision_backbone'):
            vision_backbone = self.base_model.model.vision_backbone
            if vision_backbone is not None:
                for param in vision_backbone.parameters():
                    param.requires_grad = False
                logger.info("Froze vision_backbone (SigLIP2 ViT + connector + projector)")
            else:
                logger.warning("vision_backbone is None")
        elif hasattr(self.base_model, 'vision_backbone'):
            # Molmo2ForConditionalGeneration has a @property for this
            for param in self.base_model.vision_backbone.parameters():
                param.requires_grad = False
            logger.info("Froze vision_backbone via property")
        else:
            logger.warning("Could not find vision_backbone to freeze")

    def setup_training(
        self,
        use_lora: bool = True,
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules=None,
        freeze_vision: bool = True,
    ):
        # Step 1: Freeze everything first
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Step 2: Freeze vision encoder (redundant but explicit)
        if freeze_vision:
            self.freeze_vision_encoder()

        # Step 3: Apply manual LoRA (NO PeftModel wrapper!)
        if use_lora:
            from .manual_lora import apply_lora_to_model

            # If passed as comma-separated string (e.g. from command line), split into list
            if isinstance(lora_target_modules, str):
                lora_target_modules = [m.strip() for m in lora_target_modules.split(',')]
                logger.info(f"Parsed lora_target_modules from string: {lora_target_modules}")

            if lora_target_modules is None:
                # Molmo2 (Qwen3) uses fused QKV naming:
                #   att_proj (fused Q/K/V), attn_out (output proj)
                #   ff_proj (fused gate+up), ff_out (down proj)
                lora_target_modules = [
                    "att_proj", "attn_out",
                    "ff_proj", "ff_out"
                ]

            num_replaced = apply_lora_to_model(
                model=self.base_model,
                target_module_names=lora_target_modules,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                exclude_patterns=["vision_backbone", "image_vit",
                                  "image_pooling", "image_projector",
                                  "lm_head", "wte"],
            )
            logger.info(f"Applied LoRA with r={lora_r}, alpha={lora_alpha}, replaced {num_replaced} modules")

        # DEBUG: check if lm_head got wrapped by LoRA
        lm_head_check = self.base_model.lm_head
        logger.info(f"[DEBUG] lm_head type after LoRA: {type(lm_head_check).__name__}")
        if hasattr(lm_head_check, 'linear'):
            logger.info(f"[DEBUG] lm_head IS LoRALinear! inner linear out_features={lm_head_check.linear.out_features}, lora_B out={lm_head_check.out_features}")

        # Step 4: Unfreeze embed_tokens + lm_head
        embed_layer = self.base_model.get_input_embeddings()
        if embed_layer is not None:
            for param in embed_layer.parameters():
                param.requires_grad = True
            logger.info("Unfroze embedding layer (Molmo2Embedding)")

        lm_head = self.base_model.get_output_embeddings()
        if lm_head is not None and lm_head is not embed_layer:
            for param in lm_head.parameters():
                param.requires_grad = True
            logger.info("Unfroze lm_head")
        elif lm_head is embed_layer:
            logger.info("lm_head shares weights with embed (tie_word_embeddings=True), already unfrozen")

        # Step 5: Enable input require grads
        def _enable_input_require_grads(module, input, output):
            output.requires_grad_(True)

        embed = self.base_model.get_input_embeddings()
        if embed is not None:
            embed.register_forward_hook(_enable_input_require_grads)

        # Step 6: Ensure DisTime modules are trainable
        for param in self.time_encoder.parameters():
            param.requires_grad = True
        for param in self.time_decoder.parameters():
            param.requires_grad = True

        # Step 7: Print trainable parameters
        trainable = 0
        total = 0
        lora_count = 0
        for name, param in self.named_parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
                if 'lora_' in name:
                    lora_count += param.numel()

        logger.info(f"trainable params: {trainable:,} || all params: {total:,} "
                    f"|| trainable%: {100*trainable/total:.4f}")
        logger.info(f"  of which LoRA params: {lora_count:,}")

    def get_trainable_parameters(self) -> Tuple[int, int]:
        """Get number of trainable and total parameters."""
        trainable = 0
        total = 0
        for param in self.parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
        return trainable, total

    def print_training_setup(self):
        """Print detailed training setup for verification."""
        logger.info("="*60)
        logger.info("Molmo2 DisTime Training Setup")
        logger.info("="*60)

        # 1. Vision Backbone
        vision_trainable = 0
        vision_total = 0
        for name, param in self.named_parameters():
            if 'vision_backbone' in name or 'image_vit' in name:
                vision_total += param.numel()
                if param.requires_grad:
                    vision_trainable += param.numel()
        logger.info(f"[FREEZE] Vision Backbone: {vision_trainable:,}/{vision_total:,} trainable")

        # 2. LLM Embedding
        embed_trainable = 0
        embed_total = 0
        for name, param in self.named_parameters():
            if 'wte' in name or ('embed' in name and 'lora' not in name
                                  and 'vision' not in name):
                embed_total += param.numel()
                if param.requires_grad:
                    embed_trainable += param.numel()
        logger.info(f"[TRAIN ] LLM Token Embedding: {embed_trainable:,}/{embed_total:,} trainable")

        # 3. LLM Head
        head_trainable = 0
        head_total = 0
        for name, param in self.named_parameters():
            if 'lm_head' in name:
                head_total += param.numel()
                if param.requires_grad:
                    head_trainable += param.numel()
        logger.info(f"[TRAIN ] LLM Head: {head_trainable:,}/{head_total:,} trainable")

        # 4. LoRA
        lora_trainable = 0
        for name, param in self.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                lora_trainable += param.numel()
        logger.info(f"[LORA  ] LoRA adapters: {lora_trainable:,} trainable")

        # 5. Time Modules
        time_trainable = 0
        time_total = 0
        for name, param in self.named_parameters():
            if 'time_encoder' in name or 'time_decoder' in name:
                time_total += param.numel()
                if param.requires_grad:
                    time_trainable += param.numel()
        logger.info(f"[TRAIN ] Time Encoder/Decoder: {time_trainable:,}/{time_total:,} trainable")

        # Summary
        trainable, total = self.get_trainable_parameters()
        logger.info("-"*60)
        logger.info(f"Total: {trainable:,} / {total:,} trainable ({100*trainable/total:.2f}%)")
        logger.info("="*60)

    def _build_inputs_with_vision(
        self,
        input_ids: torch.Tensor,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_token_pooling: Optional[torch.Tensor] = None,
        video_grids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build inputs_embeds by calling Molmo2's internal vision merge (video path).

        Reuses Molmo2Model.merge_visual_inputs() + build_input_embeddings()
        to correctly handle video frame processing and vision token pooling.

        Uses the VIDEO path (pixel_values_videos), not IMAGE path (pixel_values),
        because our framework processes videos as sequences of frames with
        <frame_start>/<frame_end> tokens matching Molmo2's video format.

        Returns:
            inputs_embeds: (batch_size, seq_len, hidden_size) with vision features merged
        """
        inner_model = self.base_model.model  # Molmo2Model

        # Step 1: Merge visual inputs via video path
        images, token_pooling = inner_model.merge_visual_inputs(
            input_ids=input_ids,
            pixel_values_videos=pixel_values_videos,
            video_token_pooling=video_token_pooling,
            video_grids=video_grids,
        )

        # Step 2: Build input embeddings with vision features merged
        # This calls wte(input_ids), then adds vision features at image_patch_id positions
        inputs_embeds, _ = inner_model.build_input_embeddings(
            input_ids=input_ids,
            images=images,
            token_pooling=token_pooling,
        )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_token_pooling: Optional[torch.Tensor] = None,
        video_grids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        time_gt: Optional[torch.Tensor] = None,
        num_events: Optional[torch.Tensor] = None,
        duration: Optional[torch.Tensor] = None,
        frame_times: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ):
        """
        Forward pass with DisTime temporal grounding.

        Key design decisions (following InternVL reference):
        1. P0 FIX: Time decoder uses SHIFTED mask (position p-1) to avoid information
           leakage from GT embeddings injected at position p during training.
        2. ZeRO-3 FIX: All ranks always execute the same forward path through
           time_encoder and time_decoder to keep NCCL collectives synchronized.

        Molmo2-specific:
        - Vision merge is handled by Molmo2's internal build_input_embeddings()
        - After vision merge, we inject time embeddings, then pass to transformer directly
        - XOR constraint: we bypass Molmo2Model.forward() and call transformer directly
          with inputs_embeds (no input_ids)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        reg_max = self.distime_config.reg_max

        # ================================================================
        # Build inputs_embeds with vision features merged (Molmo2 video path)
        # ================================================================
        inputs_embeds = self._build_inputs_with_vision(
            input_ids=input_ids,
            pixel_values_videos=pixel_values_videos,
            video_token_pooling=video_token_pooling,
            video_grids=video_grids,
        )

        # IMPORTANT: clone() to avoid in-place modification issues with autograd
        inputs_embeds = inputs_embeds.clone()

        # ================================================================
        # [P0 FIX] Create TWO masks for TIME_STAMP
        # ================================================================
        time_stamp_mask_enc = (input_ids == self.time_stamp_token_id)

        time_stamp_mask_dec = torch.cat([
            (input_ids[:, 1:] == self.time_stamp_token_id),
            torch.zeros((batch_size, 1), dtype=torch.bool, device=device),
        ], dim=1)

        has_time_stamps = time_stamp_mask_enc.any()
        frame_time_mask = (input_ids == self.frame_time_token_id)
        has_frame_times = frame_time_mask.any() and frame_times is not None

        # ================================================================
        # FRAME_TIME encoding
        # ================================================================
        all_ft_starts = []
        all_ft_ends = []
        ft_positions = []

        if has_frame_times:
            if duration is not None:
                frame_times_norm = frame_times / duration.unsqueeze(-1).clamp(min=1e-6) * reg_max
            else:
                frame_times_norm = frame_times / frame_times.max().clamp(min=1e-6) * reg_max

            for b in range(batch_size):
                mask_b = frame_time_mask[b]
                if not mask_b.any():
                    continue

                frame_token_indices = mask_b.nonzero(as_tuple=True)[0]
                num_frame_tokens = frame_token_indices.shape[0]

                sample_frame_times = frame_times_norm[b]
                num_frames = (sample_frame_times >= 0).sum().item() or sample_frame_times.shape[0]
                num_to_replace = min(num_frame_tokens, num_frames)

                if num_to_replace > 0:
                    times_to_encode = sample_frame_times[:num_to_replace]
                    for i in range(num_to_replace):
                        all_ft_starts.append(times_to_encode[i])
                        all_ft_ends.append(times_to_encode[i])
                        ft_positions.append((b, frame_token_indices[i]))

        if all_ft_starts:
            ft_starts_tensor = torch.stack(all_ft_starts)
            ft_ends_tensor = torch.stack(all_ft_ends)
            ft_embeds = self.time_encoder(ft_starts_tensor, ft_ends_tensor)
            for idx, (b, pos) in enumerate(ft_positions):
                inputs_embeds[b, pos] = ft_embeds[idx]
        else:
            dummy_t = torch.zeros(1, device=device, dtype=inputs_embeds.dtype)
            _ = self.time_encoder(dummy_t, dummy_t)

        # ================================================================
        # TIME_STAMP GT encoding
        # ================================================================
        all_gt_starts = []
        all_gt_ends = []
        gt_positions = []

        if self.training and time_gt is not None and has_time_stamps:
            for b in range(batch_size):
                mask_b = time_stamp_mask_enc[b]
                if not mask_b.any():
                    continue

                indices = mask_b.nonzero(as_tuple=True)[0]
                n_stamps = indices.shape[0]

                n_events_b = num_events[b].item() if num_events is not None else time_gt.shape[1]
                if n_events_b == 0:
                    continue

                sample_gt = time_gt[b, :n_events_b] * reg_max
                n_to_encode = min(n_stamps, n_events_b)

                for i in range(n_to_encode):
                    all_gt_starts.append(sample_gt[i, 0])
                    all_gt_ends.append(sample_gt[i, 1])
                    gt_positions.append((b, indices[i]))

        if all_gt_starts:
            gt_starts_tensor = torch.stack(all_gt_starts)
            gt_ends_tensor = torch.stack(all_gt_ends)
            gt_embeds = self.time_encoder(gt_starts_tensor, gt_ends_tensor)
            for idx, (b, pos) in enumerate(gt_positions):
                inputs_embeds[b, pos] = gt_embeds[idx]
        else:
            dummy_t = torch.zeros(1, device=device, dtype=inputs_embeds.dtype)
            _ = self.time_encoder(dummy_t, dummy_t)

        # ================================================================
        # Forward through Molmo2 transformer (bypass Molmo2Model to avoid double merge)
        # ================================================================
        # We call the inner transformer directly since we've already done embedding + vision merge
        transformer = self.base_model.model.transformer  # Molmo2TextModel

        outputs = transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state

        # Apply lm_head to get logits
        logits = self.base_model.lm_head(hidden_states)

        # Compute LM loss manually (since we bypassed the high-level forward)
        lm_loss = torch.tensor(0.0, device=device)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Get vocab size from lm_head
            vocab_size = logits.shape[-1]
            lm_loss = loss_fct(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1).to(shift_logits.device)
            )

        # ================================================================
        # Time decoding (using shifted mask at p-1)
        # ================================================================
        if hidden_states is not None and time_stamp_mask_dec.any():
            time_logits, pred_times = self.time_decoder(hidden_states, time_stamp_mask_dec)
        else:
            if hidden_states is not None:
                dummy_hidden = torch.zeros_like(hidden_states)
            else:
                dummy_hidden = torch.zeros(
                    batch_size, 1, self.distime_config.hidden_size,
                    device=device, dtype=inputs_embeds.dtype
                )
            dummy_mask = torch.zeros(
                dummy_hidden.shape[0], dummy_hidden.shape[1],
                dtype=torch.bool, device=device
            )
            dummy_mask[0, 0] = True
            time_logits, pred_times = self.time_decoder(dummy_hidden, dummy_mask)
            time_logits = time_logits * 0
            pred_times = pred_times * 0

        # ================================================================
        # Compute DisTime losses
        # ================================================================
        time_loss = (time_logits * 0).sum()
        iou_loss = (pred_times * 0).sum()

        if self.training and time_gt is not None and time_stamp_mask_dec.any() and pred_times.shape[0] > 0:
            num_stamps_per_sample = time_stamp_mask_dec.sum(dim=1)
            target_expanded = []

            for b in range(batch_size):
                n_stamps = num_stamps_per_sample[b].item()
                if n_stamps == 0:
                    continue

                n_events_b = num_events[b].item() if num_events is not None else time_gt.shape[1]
                if n_events_b == 0:
                    continue

                sample_gt = time_gt[b, :n_events_b] * reg_max

                if n_stamps <= n_events_b:
                    target_expanded.append(sample_gt[:n_stamps])
                else:
                    pad = sample_gt[-1:].expand(n_stamps - n_events_b, -1)
                    target_expanded.append(torch.cat([sample_gt, pad], dim=0))

            if target_expanded:
                target_times = torch.cat(target_expanded, dim=0)
                target_times = target_times.clamp(min=0, max=reg_max - 1e-3)

                time_loss = self.dfl_loss_fn(time_logits, target_times)
                iou_loss = self.diou_loss_fn(pred_times, target_times)

        # Total loss
        cfg = self.distime_config
        total_loss = lm_loss + cfg.time_loss_weight * time_loss + cfg.iou_loss_weight * iou_loss

        # Return outputs
        if return_dict:
            return {
                'loss': total_loss,
                'lm_loss': lm_loss,
                'time_loss': time_loss,
                'iou_loss': iou_loss,
                'pred_times': pred_times,
                'time_logits': time_logits,
                'logits': logits,
            }
        else:
            return total_loss

    @torch.no_grad()
    def generate(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            pixel_values_videos: Optional[torch.Tensor] = None,
            video_token_pooling: Optional[torch.Tensor] = None,
            video_grids: Optional[torch.Tensor] = None,
            duration: Optional[torch.Tensor] = None,
            frame_times: Optional[torch.Tensor] = None,
            max_new_tokens: int = 512,
            do_sample: bool = False,
            temperature: float = 1.0,
            **generate_kwargs
    ) -> dict:
        """
        Generate with DisTime temporal grounding - full autoregressive loop.
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]
        reg_max = self.distime_config.reg_max

        # EOS tokens (Qwen3 style: <|endoftext|> + <|im_end|>)
        default_eos = self.processor.tokenizer.eos_token_id
        im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != default_eos:
            default_eos_list = [default_eos, im_end_id]
        else:
            default_eos_list = [default_eos]
        eos_token_ids = generate_kwargs.pop('eos_token_id', default_eos_list)
        if not isinstance(eos_token_ids, list):
            eos_token_ids = [eos_token_ids]

        # Build initial embeddings with vision features (video path)
        inputs_embeds = self._build_inputs_with_vision(
            input_ids=input_ids,
            pixel_values_videos=pixel_values_videos,
            video_token_pooling=video_token_pooling,
            video_grids=video_grids,
        ).clone()

        # Inject FRAME_TIME embeddings
        frame_time_mask = (input_ids == self.frame_time_token_id)
        if frame_time_mask.any() and frame_times is not None:
            frame_times_norm = frame_times / duration.unsqueeze(-1).clamp(min=1e-6) * reg_max
            for b in range(batch_size):
                mask_b = frame_time_mask[b]
                if not mask_b.any():
                    continue
                indices = mask_b.nonzero(as_tuple=True)[0]
                times_b = frame_times_norm[b]
                n = min(indices.shape[0], times_b.shape[0])
                if n > 0:
                    ft_embeds = self.time_encoder(times_b[:n], times_b[:n])
                    for i in range(n):
                        inputs_embeds[b, indices[i]] = ft_embeds[i]

        # Prefill through transformer (bypass Molmo2Model)
        transformer = self.base_model.model.transformer

        outputs = transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
        )

        past_key_values = outputs.past_key_values
        last_hidden = outputs.last_hidden_state
        next_token_logits = self.base_model.lm_head(last_hidden[:, -1:, :]).squeeze(1)

        generated_ids = []
        pred_times = []
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)

        embed_layer = self.base_model.get_input_embeddings()

        for step in range(max_new_tokens):
            if do_sample and temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            generated_ids.append(next_tokens.clone())

            if eos_token_ids:
                for eid in eos_token_ids:
                    unfinished = unfinished & (next_tokens != eid)
                if not unfinished.any():
                    break

            attention_mask = torch.cat([
                attention_mask,
                torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=device)
            ], dim=1)

            is_time_stamp = (next_tokens == self.time_stamp_token_id)

            if is_time_stamp.any():
                decode_hidden = last_hidden[:, -1:, :]
                decode_mask = torch.ones(
                    batch_size, 1, dtype=torch.bool, device=device
                )
                _, pred_time_step = self.time_decoder(decode_hidden, decode_mask)
                pred_times.append(pred_time_step)

                time_embeds = self.time_encoder(
                    pred_time_step[:, 0], pred_time_step[:, 1]
                )
                next_input_embeds = time_embeds.unsqueeze(1)

                outputs = transformer(
                    inputs_embeds=next_input_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=True,
                )
            else:
                next_input_embeds = embed_layer(next_tokens.unsqueeze(1))
                outputs = transformer(
                    inputs_embeds=next_input_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            last_hidden = outputs.last_hidden_state
            next_token_logits = self.base_model.lm_head(last_hidden[:, -1:, :]).squeeze(1)

        if generated_ids:
            gen_ids = torch.stack(generated_ids, dim=1)
            output_ids = torch.cat([input_ids, gen_ids], dim=1)
        else:
            gen_ids = torch.empty(batch_size, 0, dtype=torch.long, device=device)
            output_ids = input_ids

        generated_text = self.processor.tokenizer.batch_decode(
            gen_ids, skip_special_tokens=False
        )

        pred_times_list = [None] * batch_size
        pred_times_normalized_list = [None] * batch_size

        if pred_times:
            all_preds = torch.cat(pred_times, dim=0)
            pred_normalized = all_preds / reg_max

            if batch_size == 1:
                pred_times_normalized_list[0] = pred_normalized
                if duration is not None:
                    d = duration[0].item() if duration.dim() > 0 else duration.item()
                    pred_times_list[0] = pred_normalized * d
                else:
                    pred_times_list[0] = pred_normalized

        return {
            'generated_ids': output_ids,
            'generated_text': generated_text,
            'pred_times': pred_times_list,
            'pred_times_normalized': pred_times_normalized_list,
        }

    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            dist.barrier()
            return

        os.makedirs(save_directory, exist_ok=True)

        self.base_model.save_pretrained(save_directory)

        distime_state = {
            'time_encoder': self.time_encoder.state_dict(),
            'time_decoder': self.time_decoder.state_dict(),
            'config': self.distime_config.__dict__,
        }
        torch.save(distime_state, os.path.join(save_directory, 'distime_modules.pt'))

        # 兼容新版 transformers save_pretrained 中访问的属性
        for attr in ('audio_tokenizer', 'chat_template', '_auto_class'):
            if not hasattr(self.processor, attr):
                setattr(self.processor, attr, None)
        self.processor.save_pretrained(save_directory)

        logger.info(f"Saved model to {save_directory}")

        if dist.is_initialized():
            dist.barrier()

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load model from directory."""
        distime_path = os.path.join(model_path, 'distime_modules.pt')
        if os.path.exists(distime_path):
            distime_state = torch.load(distime_path, map_location='cpu')
            distime_config = DisTimeConfig(**distime_state['config'])
        else:
            distime_config = DisTimeConfig()

        model = cls(
            model_name_or_path=model_path,
            distime_config=distime_config,
            **kwargs
        )

        if os.path.exists(distime_path):
            model.time_encoder.load_state_dict(distime_state['time_encoder'])
            model.time_decoder.load_state_dict(distime_state['time_decoder'])
            logger.info(f"Loaded DisTime modules from {distime_path}")

        return model
