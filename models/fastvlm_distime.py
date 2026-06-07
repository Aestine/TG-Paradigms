"""
FastVLM-DisTime: FastVLM with DisTime temporal grounding capabilities.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoConfig,
    FastVlmForConditionalGeneration,
)

from .time_modules import TimeEncoder, TimeDecoder

logger = logging.getLogger(__name__)


@dataclass
class DisTimeConfig:
    """Configuration for DisTime modules."""
    reg_max: int = 32
    num_time_layers: int = 3
    hidden_size: int = 1536  # <<<< 改动1: FastVLM-0.5B 默认 1536 (Qwen2-1.5B), SmolVLM 是 2048
    time_loss_weight: float = 1.0
    iou_loss_weight: float = 1.0
    sigma: float = 1.0  # Gaussian sigma for time encoding
    vision_pool_stride: int = 2  # 视觉 token 空间压缩步长: 1=不压缩(256), 2=4x压缩(64), 4=16x压缩(16)


# Special tokens
TIME_STAMP_TOKEN = "<TIME_STAMP>"
FRAME_TIME_TOKEN = "<FRAME_TIME>"


class FastVLMDisTime(nn.Module):  # <<<< 改动2: 类名 SmolVLMDisTime -> FastVLMDisTime
    """
    FastVLM with DisTime temporal grounding.

    Architecture:
    - Base: FastVLM (FastViTHD/fastvit_mci3 vision encoder + Qwen2-1.5B LLM)  # <<<< 改动3: 描述
    - Added: TimeEncoder, TimeDecoder for temporal grounding

    Training:
    - Freeze vision encoder
    - Apply LoRA to LLM
    - Train DisTime modules end-to-end

    Key differences from SmolVLMDisTime:
    - FastVLM inherits from LlavaForConditionalGeneration (modular), 展开后为独立类
    - Vision encoder is model.vision_tower (not model.vision_model)
    - LLM is model.language_model (not model.text_model)
    - LoRA exclude pattern is "vision_tower" (not "vision_model")
    """

    def __init__(
        self,
        model_name_or_path: str = "KamilaMila/FastVLM-0.5B",  # <<<< 改动4: transformers 内置版
        distime_config: Optional[DisTimeConfig] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        use_flash_attention: bool = True,
    ):
        super().__init__()

        self.distime_config = distime_config or DisTimeConfig()

        # Load base model
        logger.info(f"Loading base model from {model_name_or_path}")

        # FastVLM 的 vision tower (TimmWrapperModel) 不支持 sdpa/flash_attention_2,
        # 不传 attn_implementation，让 transformers 自动选择:
        #   - Qwen2 LLM → 默认 sdpa
        #   - TimmWrapper vision tower → 默认 eager
        self.base_model = FastVlmForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

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
            hidden_size = 1536  # <<<< 改动5: 默认值 2048 -> 1536 (FastVLM-0.5B Qwen2-1.5B)
            logger.warning(f"Could not detect hidden_size, using default {hidden_size}")

        self.distime_config.hidden_size = hidden_size
        logger.info(f"Detected hidden_size: {hidden_size}")

        # Add special tokens
        self._add_special_tokens()

        # Initialize DisTime modules
        self._init_distime_modules()

        # Install vision token pooling (compress 256 tokens/frame → fewer)
        self._install_vision_pooling()

        # Store token IDs
        self.time_stamp_token_id = self.processor.tokenizer.convert_tokens_to_ids(TIME_STAMP_TOKEN)
        self.frame_time_token_id = self.processor.tokenizer.convert_tokens_to_ids(FRAME_TIME_TOKEN)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the base model."""
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def _add_special_tokens(self):
        """Add special tokens for DisTime with proper initialization (following InternVL)."""
        special_tokens = {
            'additional_special_tokens': [TIME_STAMP_TOKEN, FRAME_TIME_TOKEN]
        }

        num_added = self.processor.tokenizer.add_special_tokens(special_tokens)
        logger.info(f"Added {num_added} special tokens: {TIME_STAMP_TOKEN}, {FRAME_TIME_TOKEN}")

        if num_added > 0:
            new_vocab_size = len(self.processor.tokenizer)

            # 1. Resize embeddings
            self.base_model.resize_token_embeddings(new_vocab_size, mean_resizing=False)

            # 2. Initialize new token embeddings to mean of existing (following InternVL)
            input_embeddings = self.base_model.get_input_embeddings()
            if input_embeddings is not None:
                with torch.no_grad():
                    embed_weight = input_embeddings.weight.data
                    embed_avg = embed_weight[:-num_added].mean(dim=0, keepdim=True)
                    embed_weight[-num_added:] = embed_avg
                    logger.info(f"Initialized new input embeddings to mean of existing")

            # Also initialize output embeddings (lm_head) if different from input
            output_embeddings = self.base_model.get_output_embeddings()
            if output_embeddings is not None and output_embeddings.weight is not input_embeddings.weight:
                with torch.no_grad():
                    out_weight = output_embeddings.weight.data
                    out_avg = out_weight[:-num_added].mean(dim=0, keepdim=True)
                    out_weight[-num_added:] = out_avg
                    logger.info(f"Initialized new output embeddings to mean of existing")

            # 3. 同步 vocab_size 到 config
            config = self.base_model.config

            if hasattr(config, 'text_config') and hasattr(config.text_config, 'vocab_size'):
                config.text_config.vocab_size = new_vocab_size
                logger.info(f"Updated config.text_config.vocab_size to {new_vocab_size}")

            if hasattr(config, 'vocab_size'):
                config.vocab_size = new_vocab_size
                logger.info(f"Updated config.vocab_size to {new_vocab_size}")

            # <<<< 改动6: FastVLM 的 LLM 子模块叫 language_model (继承自 LLaVA)
            # SmolVLM 这里也有这段代码 (第170行), 但 SmolVLM 实际走不到
            # (因为 SmolVLM 的子模块叫 text_model, 没有 language_model)
            # FastVLM 会走到这里, 因为它确实有 model.language_model
            if hasattr(self.base_model, 'language_model') and hasattr(self.base_model.language_model, 'config'):
                self.base_model.language_model.config.vocab_size = new_vocab_size
                logger.info(f"Updated language_model.config.vocab_size to {new_vocab_size}")
            # 注意: FastVLM 里 language_model 在 self.base_model.model.language_model
            # 而不是 self.base_model.language_model, 需要确认
            # 如果上面走不到, 用下面这个:
            if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'language_model'):
                if hasattr(self.base_model.model.language_model, 'config'):
                    self.base_model.model.language_model.config.vocab_size = new_vocab_size
                    logger.info(f"Updated model.language_model.config.vocab_size to {new_vocab_size}")

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

        # Initialize loss functions (reused across forward passes)
        from utils.losses import DistributionFocalLoss, DIoULoss
        self.dfl_loss_fn = DistributionFocalLoss(reg_max=cfg.reg_max)
        self.diou_loss_fn = DIoULoss()

        logger.info(f"Initialized DisTime modules with reg_max={cfg.reg_max}, "
                   f"num_layers={cfg.num_time_layers}, hidden_size={cfg.hidden_size}")

    def _install_vision_pooling(self):
        """
        Monkey-patch base_model.model.get_image_features 来压缩视觉 token。

        FastVLM 的 vision tokens 是 2D 空间网格 (16×16 = 256 tokens):
          - crop_size=1024, patch_size=64 → grid = 1024/64 = 16
          - 256 tokens = 16 × 16

        stride=1: 不压缩, 256 tokens (16×16)
        stride=2: 4x 压缩, 64 tokens (8×8)     ← 推荐，32帧×64=2048，塞进 4096
        stride=4: 16x 压缩, 16 tokens (4×4)

        transformers 里 FastVlmModel.forward 使用方式:
          image_features = self.get_image_features(..., return_dict=True).pooler_output
          image_features = torch.cat(image_features, dim=0)  # list of (S, D) → (N*S, D)
        所以我们 patch pooler_output 即可。
        """
        stride = self.distime_config.vision_pool_stride
        if stride <= 1:
            logger.info("Vision pooling disabled (stride=1), keeping 256 tokens/frame")
            return

        original_tokens = 256  # (1024 // 64)^2
        grid_size = int(original_tokens ** 0.5)  # 16

        # AvgPool2d: 在 16×16 空间网格上做 stride=2 → 8×8 = 64 tokens
        pool2d = nn.AvgPool2d(kernel_size=stride, stride=stride)

        # 保存原始方法
        inner_model = self.base_model.model  # FastVlmModel
        original_get_image_features = inner_model.get_image_features

        def pooled_get_image_features(pixel_values, **kwargs):
            """get_image_features with 2D spatial pooling on vision tokens."""
            outputs = original_get_image_features(pixel_values, **kwargs)
            # pooler_output: list of (num_tokens, hidden_size) tensors
            # 每个 feat: (256, 1536) = (16*16, D)
            pooled_features = []
            for feat in outputs.pooler_output:
                S, D = feat.shape  # (256, 1536)
                # reshape 到 2D 空间网格: (1, D, 16, 16)
                f = feat.permute(1, 0).reshape(1, D, grid_size, grid_size)
                f = pool2d(f)  # (1, D, 8, 8) when stride=2
                new_S = f.shape[2] * f.shape[3]  # 64
                f = f.reshape(D, new_S).permute(1, 0)  # (64, D)
                pooled_features.append(f)
            outputs.pooler_output = pooled_features
            return outputs

        inner_model.get_image_features = pooled_get_image_features

        new_tokens = original_tokens // (stride * stride)
        logger.info(f"Installed vision pooling: AvgPool2d stride={stride}, "
                   f"{grid_size}×{grid_size}={original_tokens} → "
                   f"{grid_size//stride}×{grid_size//stride}={new_tokens} tokens/frame "
                   f"(32 frames = {32 * new_tokens} total, fits in 4096 seq)")

    def freeze_vision_encoder(self):
        """Freeze the vision encoder.

        <<<< 改动7: FastVLM 的 vision encoder 叫 vision_tower (继承自 LLaVA)
        SmolVLM 叫 vision_model (继承自 Idefics3)
        结构: FastVLM -> base_model.model.vision_tower
              SmolVLM -> base_model.model.vision_model
        """
        # FastVLM: model.vision_tower (LLaVA 架构)
        if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'vision_tower'):
            for param in self.base_model.model.vision_tower.parameters():
                param.requires_grad = False
            logger.info("Froze vision_tower")
        elif hasattr(self.base_model, 'vision_tower'):
            for param in self.base_model.vision_tower.parameters():
                param.requires_grad = False
            logger.info("Froze vision_tower")
        # fallback: 兼容其他可能的命名
        elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'vision_model'):
            for param in self.base_model.model.vision_model.parameters():
                param.requires_grad = False
            logger.info("Froze vision_model")
        else:
            logger.warning("Could not find vision encoder to freeze")

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

            if lora_target_modules is None:
                lora_target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ]

            num_replaced = apply_lora_to_model(
                model=self.base_model,
                target_module_names=lora_target_modules,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                exclude_patterns=["vision_tower"],  # <<<< 改动8: "vision_model" -> "vision_tower"
            )
            logger.info(f"Applied LoRA with r={lora_r}, alpha={lora_alpha}, replaced {num_replaced} modules")

        # Step 4: 解冻 embed_tokens + lm_head (mixed 方案)
        # 不同于 DisTime 原版 (冻结), 这里解冻 embed/head 以训练新 special tokens
        embed_layer = self.base_model.get_input_embeddings()
        if embed_layer is not None:
            for param in embed_layer.parameters():
                param.requires_grad = True
            logger.info("Unfroze embed_tokens")

        lm_head = self.base_model.get_output_embeddings()
        if lm_head is not None and lm_head is not embed_layer:
            for param in lm_head.parameters():
                param.requires_grad = True
            logger.info("Unfroze lm_head")
        elif lm_head is embed_layer:
            logger.info("lm_head shares weights with embed_tokens (tie_word_embeddings=True), already unfrozen")

        # Step 5: Enable input require grads (replaces PEFT's enable_input_require_grads)
        def _enable_input_require_grads(module, input, output):
            output.requires_grad_(True)

        if hasattr(self.base_model, 'get_input_embeddings'):
            embed = self.base_model.get_input_embeddings()
        elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'embed_tokens'):
            embed = self.base_model.model.embed_tokens
        else:
            embed = None

        if embed is not None:
            embed.register_forward_hook(_enable_input_require_grads)

        # Step 6: Ensure DisTime modules are trainable
        for param in self.time_encoder.parameters():
            param.requires_grad = True
        for param in self.time_decoder.parameters():
            param.requires_grad = True

        # Step 7: Print trainable parameters (replaces PEFT's print_trainable_parameters)
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
        """Print detailed training setup for verification against DisTime paper."""
        logger.info("="*60)
        logger.info("DisTime Training Setup (should match paper)")
        logger.info("="*60)

        # 1. Vision Encoder
        vision_trainable = 0
        vision_total = 0
        for name, param in self.named_parameters():
            if 'vision' in name.lower():
                vision_total += param.numel()
                if param.requires_grad:
                    vision_trainable += param.numel()
        logger.info(f"[FREEZE] Vision Encoder: {vision_trainable:,}/{vision_total:,} trainable")

        # 2. LLM Embedding
        embed_trainable = 0
        embed_total = 0
        for name, param in self.named_parameters():
            if 'embed_tokens' in name or 'wte' in name or ('embed' in name and 'lora' not in name):
                embed_total += param.numel()
                if param.requires_grad:
                    embed_trainable += param.numel()
        logger.info(f"[TRAIN ] LLM Token Embedding: {embed_trainable:,}/{embed_total:,} trainable")

        # 3. LLM Head
        head_trainable = 0
        head_total = 0
        for name, param in self.named_parameters():
            if 'lm_head' in name or ('head' in name and 'time' not in name):
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        time_gt: Optional[torch.Tensor] = None,
        num_events: Optional[torch.Tensor] = None,
        duration: Optional[torch.Tensor] = None,
        frame_times: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ):
        # <<<< 改动9: 去掉 image_flags 参数
        # SmolVLM forward 有 image_flags 参数, FastVLM (LLaVA架构) 不需要

        # ============ DEBUG ============
        import torch.distributed as dist
        if dist.is_initialized():
            rank = dist.get_rank()
            ft_mask = (input_ids == self.frame_time_token_id)
            ts_mask = (input_ids == self.time_stamp_token_id)
            img_token_id = getattr(self.base_model.config, 'image_token_id', None)
            img_count = (input_ids == img_token_id).sum().item() if img_token_id else -1

            print(
                f"[Rank {rank}] batch_size={input_ids.shape[0]}, seq_len={input_ids.shape[1]}, "
                f"pixel_values={pixel_values.shape if pixel_values is not None else None}, "
                f"image_tokens={img_count}, "
                f"frame_time_tokens={ft_mask.sum().item()}, "
                f"time_stamp_tokens={ts_mask.sum().item()}, "
                f"frame_times={'not None' if frame_times is not None else None}, "
                f"time_gt={'not None' if time_gt is not None else None}, "
                f"num_events={num_events}",
                flush=True
            )
        # ============ END DEBUG ============
        """
        Forward pass with DisTime temporal grounding.

        Key design decisions (following InternVL reference):
        1. P0 FIX: Time decoder uses SHIFTED mask (position p-1) to avoid information
           leakage from GT embeddings injected at position p during training.
        2. ZeRO-3 FIX: All ranks always execute the same forward path through
           time_encoder and time_decoder to keep NCCL collectives synchronized.
           When no TIME_STAMP/FRAME_TIME tokens exist, dummy inputs are used.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        reg_max = self.distime_config.reg_max

        # Get input embeddings
        if hasattr(self.base_model, 'get_input_embeddings'):
            embed_layer = self.base_model.get_input_embeddings()
        else:
            embed_layer = self.base_model.model.embed_tokens

        # IMPORTANT: clone() to avoid in-place modification issues with autograd
        inputs_embeds = embed_layer(input_ids).clone()

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
        # Forward through base model
        # ================================================================
        # FastVLM (LLaVA架构):
        #   1. 不接受 image_flags 参数
        #   2. 不允许同时传 input_ids 和 inputs_embeds (XOR 校验)
        #      → 只传 inputs_embeds (已注入 time embeddings), 不传 input_ids
        #      → get_placeholder_mask 会用 embedding 比较找 <image> 位置
        #   3. pixel_values 必须是 4D [N, C, H, W], dataset 返回 5D [batch, frames, C, H, W]
        #      → 需要 reshape, 用 image_flags 过滤 padding 帧
        if pixel_values is not None and pixel_values.dim() == 5:
            # [batch, num_frames, C, H, W] → [N, C, H, W]
            # 用 image_flags 过滤 padding 帧 (如果有)
            B, F, C, H, W = pixel_values.shape
            if 'image_flags' in kwargs:
                image_flags = kwargs.pop('image_flags')
                # image_flags: [batch, max_frames], 1=real, 0=padding
                valid_mask = image_flags.bool().view(-1)  # [B*F]
                pixel_values_flat = pixel_values.view(B * F, C, H, W)
                pixel_values = pixel_values_flat[valid_mask]  # [N_real, C, H, W]
            else:
                pixel_values = pixel_values.view(B * F, C, H, W)
        else:
            # 已经是 4D 或 None, 不需要处理
            kwargs.pop('image_flags', None)

        outputs = self.base_model(
            input_ids=None,          # ← 不传 input_ids (FastVLM XOR 校验)
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            pixel_values=pixel_values,
        )

        # Get LM loss
        lm_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=device)

        # Get last layer hidden states
        if outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = None

        # ================================================================
        # Time decoding
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
                'logits': outputs.logits if hasattr(outputs, 'logits') else None,
            }
        else:
            return total_loss

    @torch.no_grad()
    def generate(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            pixel_values: Optional[torch.Tensor] = None,
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
        input_len = input_ids.shape[1]
        reg_max = self.distime_config.reg_max

        # FastVLM generation_config 只有 <|endoftext|> (151643) 作为 eos
        # 但 ChatML 格式中 <|im_end|> (151645) 才是真正的 turn 结束标记
        # 需要两个都作为 stop token
        default_eos = self.processor.tokenizer.eos_token_id  # <|endoftext|>
        im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != default_eos:
            default_eos_list = [default_eos, im_end_id]
        else:
            default_eos_list = [default_eos]
        eos_token_ids = generate_kwargs.pop('eos_token_id', default_eos_list)
        if not isinstance(eos_token_ids, list):
            eos_token_ids = [eos_token_ids]

        embed_layer = self.base_model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids).clone()

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

        # FastVLM: pixel_values 必须是 4D [N, C, H, W]
        if pixel_values is not None and pixel_values.dim() == 5:
            B, F, C, H, W = pixel_values.shape
            pixel_values = pixel_values.view(B * F, C, H, W)

        # FastVLM: 不允许同时传 input_ids 和 inputs_embeds
        outputs = self.base_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
        )

        past_key_values = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1]
        next_token_logits = outputs.logits[:, -1, :]

        generated_ids = []
        pred_times = []
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)

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

                outputs = self.base_model(
                    inputs_embeds=next_input_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=True,
                )
            else:
                outputs = self.base_model(
                    input_ids=next_tokens.unsqueeze(1),
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1]
            next_token_logits = outputs.logits[:, -1, :]

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
        print(f"[DEBUG] base_model.save_pretrained done, now saving distime_modules", flush=True)
        distime_state = {
            'time_encoder': self.time_encoder.state_dict(),
            'time_decoder': self.time_decoder.state_dict(),
            'config': self.distime_config.__dict__,
        }
        torch.save(distime_state, os.path.join(save_directory, 'distime_modules.pt'))

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
