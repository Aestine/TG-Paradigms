"""
SmolVLM-DisTime: SmolVLM with DisTime temporal grounding capabilities.
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
)

from .time_modules import TimeEncoder, TimeDecoder

logger = logging.getLogger(__name__)


@dataclass
class DisTimeConfig:
    """Configuration for DisTime modules."""
    reg_max: int = 32
    num_time_layers: int = 3
    hidden_size: int = 2048  # Will be auto-detected from model
    time_loss_weight: float = 1.0
    iou_loss_weight: float = 1.0
    sigma: float = 1.0  # Gaussian sigma for time encoding


# Special tokens
TIME_STAMP_TOKEN = "<TIME_STAMP>"
FRAME_TIME_TOKEN = "<FRAME_TIME>"


class SmolVLMDisTime(nn.Module):
    """
    SmolVLM with DisTime temporal grounding.

    Architecture:
    - Base: SmolVLM2 (vision encoder + LLM)
    - Added: TimeEncoder, TimeDecoder for temporal grounding

    Training:
    - Freeze vision encoder
    - Apply LoRA to LLM
    - Train DisTime modules end-to-end
    """

    def __init__(
        self,
        model_name_or_path: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        distime_config: Optional[DisTimeConfig] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        use_flash_attention: bool = True,
    ):
        super().__init__()

        self.distime_config = distime_config or DisTimeConfig()

        # Load base model
        logger.info(f"Loading base model from {model_name_or_path}")

        # attn_implementation = "flash_attention_2" if use_flash_attention else "eager"
        import os
        attn_impl = os.environ.get("ATTN_IMPLEMENTATION", None)
        if attn_impl:
            attn_implementation = attn_impl
        elif use_flash_attention:
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "sdpa"

        logger.info(f"Using attention implementation: {attn_implementation}")

        self.base_model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
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
            hidden_size = 2048  # Default for SmolVLM2-2.2B
            logger.warning(f"Could not detect hidden_size, using default {hidden_size}")

        self.distime_config.hidden_size = hidden_size
        logger.info(f"Detected hidden_size: {hidden_size}")

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

            # 3. 关键！同步 vocab_size 到 config（按照 InternVL 的做法）
            # 这样 forward 时 logits 的维度才会和 labels 对齐
            config = self.base_model.config

            # SmolVLM 的 config 结构：config.text_config.vocab_size
            if hasattr(config, 'text_config') and hasattr(config.text_config, 'vocab_size'):
                config.text_config.vocab_size = new_vocab_size
                logger.info(f"Updated config.text_config.vocab_size to {new_vocab_size}")

            # 也更新顶层 vocab_size（如果存在）
            if hasattr(config, 'vocab_size'):
                config.vocab_size = new_vocab_size
                logger.info(f"Updated config.vocab_size to {new_vocab_size}")

            # 同步内部 language_model 的 config（如果是嵌套结构）
            if hasattr(self.base_model, 'language_model') and hasattr(self.base_model.language_model, 'config'):
                self.base_model.language_model.config.vocab_size = new_vocab_size
                logger.info(f"Updated language_model.config.vocab_size to {new_vocab_size}")

            # 打印确认
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

    def freeze_vision_encoder(self):
        """Freeze the vision encoder."""
        if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'vision_model'):
            for param in self.base_model.model.vision_model.parameters():
                param.requires_grad = False
            logger.info("Froze vision_model")
        elif hasattr(self.base_model, 'vision_model'):
            for param in self.base_model.vision_model.parameters():
                param.requires_grad = False
            logger.info("Froze vision_model")
        elif hasattr(self.base_model, 'vision_tower'):
            for param in self.base_model.vision_tower.parameters():
                param.requires_grad = False
            logger.info("Froze vision_tower")
        else:
            logger.warning("Could not find vision encoder to freeze")

    # def setup_training(
    #     self,
    #     use_lora: bool = True,
    #     lora_r: int = 64,
    #     lora_alpha: int = 16,
    #     lora_dropout: float = 0.05,
    #     lora_target_modules: Optional[List[str]] = None,
    #     freeze_vision: bool = True,
    # ):
    #     """
    #     Setup model for training.
    #
    #     Args:
    #         use_lora: Whether to use LoRA
    #         lora_r: LoRA rank
    #         lora_alpha: LoRA alpha
    #         lora_dropout: LoRA dropout
    #         lora_target_modules: Target modules for LoRA
    #         freeze_vision: Whether to freeze vision encoder
    #     """
    #     # Freeze vision encoder
    #     if freeze_vision:
    #         self.freeze_vision_encoder()
    #
    #     # Apply LoRA
    #     if use_lora:
    #         try:
    #             from peft import LoraConfig, get_peft_model
    #
    #             if lora_target_modules is None:
    #                 # Default target modules for typical LLM
    #                 lora_target_modules = [
    #                     "q_proj", "k_proj", "v_proj", "o_proj",
    #                     "gate_proj", "up_proj", "down_proj"
    #                 ]
    #
    #             lora_config = LoraConfig(
    #                 r=lora_r,
    #                 lora_alpha=lora_alpha,
    #                 lora_dropout=lora_dropout,
    #                 target_modules=lora_target_modules,
    #                 # exclude_modules=["vision_model"],
    #                 # modules_to_save=["embed_tokens", "lm_head"],
    #                 bias="none",
    #                 task_type="CAUSAL_LM",
    #             )
    #
    #             self.base_model = get_peft_model(self.base_model, lora_config)
    #             self.base_model.enable_input_require_grads()
    #
    #             # 手动 unfreeze embed_tokens 和 lm_head (按原始 DisTime 的做法)
    #             for name, param in self.base_model.named_parameters():
    #                 if any(x in name for x in ["embed_tokens", "lm_head"]):
    #                     param.requires_grad = True
    #             # ====== 添加这段：手动冻结 vision 相关的 LoRA 参数 ======
    #             frozen_count = 0
    #             for name, param in self.base_model.named_parameters():
    #                 if 'vision_model' in name and 'lora' in name.lower():
    #                     param.requires_grad = False
    #                     frozen_count += param.numel()
    #             logger.info(f"Froze {frozen_count:,} LoRA params in vision_model")
    #             # ====================================================
    #             self.base_model.print_trainable_parameters()
    #
    #             logger.info(f"Applied LoRA with r={lora_r}, alpha={lora_alpha}")
    #
    #         except ImportError:
    #             logger.warning("PEFT not installed, skipping LoRA")
    #
    #     # Ensure DisTime modules are trainable
    #     for param in self.time_encoder.parameters():
    #         param.requires_grad = True
    #     for param in self.time_decoder.parameters():
    #         param.requires_grad = True

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
                exclude_patterns=["vision_model"],
            )
            logger.info(f"Applied LoRA with r={lora_r}, alpha={lora_alpha}, replaced {num_replaced} modules")

        # Step 4: Unfreeze embed_tokens + lm_head (following original DisTime)
        for name, param in self.base_model.named_parameters():
            if any(x in name for x in ["embed_tokens", "lm_head"]):
                param.requires_grad = True

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
        image_flags: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        time_gt: Optional[torch.Tensor] = None,  # (batch, max_events, 2) normalized [0, 1]
        num_events: Optional[torch.Tensor] = None,  # (batch,) 每个样本的实际事件数
        duration: Optional[torch.Tensor] = None,  # (batch,) video duration
        frame_times: Optional[torch.Tensor] = None,  # (batch, num_frames) frame timestamps
        return_dict: bool = True,
        **kwargs
    ):
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
        # [P0 FIX] Create TWO masks for TIME_STAMP:
        #   - time_stamp_mask_enc: at position p (where TIME_STAMP is), for GT embedding injection
        #   - time_stamp_mask_dec: at position p-1 (shifted left), for time decoding
        #
        # Rationale (following InternVL modeling_internvl_chat_distime.py:321-328):
        #   During training, position p's embedding is REPLACED with GT time embedding.
        #   If time_decoder reads hidden_states[p], it sees information derived from
        #   the GT itself → information leakage, and train/inference mismatch.
        #   Position p-1's hidden state was computed BEFORE seeing the GT embedding,
        #   so it's safe for prediction and consistent with autoregressive inference.
        # ================================================================
        time_stamp_mask_enc = (input_ids == self.time_stamp_token_id)  # (B, seq_len) at position p

        # Shifted mask: position p-1 is True when input_ids[p] == TIME_STAMP
        # input_ids[:, 1:] shifts the comparison left by 1, then pad a False at the end
        time_stamp_mask_dec = torch.cat([
            (input_ids[:, 1:] == self.time_stamp_token_id),
            torch.zeros((batch_size, 1), dtype=torch.bool, device=device),
        ], dim=1)  # (B, seq_len) at position p-1

        has_time_stamps = time_stamp_mask_enc.any()
        frame_time_mask = (input_ids == self.frame_time_token_id)
        has_frame_times = frame_time_mask.any() and frame_times is not None

        # ================================================================
        # FRAME_TIME encoding: batched single call to time_encoder
        # (ZeRO-3 FIX: all ranks execute exactly ONE time_encoder forward for FRAME_TIME)
        # ================================================================
        # Collect all frame time values and their positions across the batch
        all_ft_starts = []
        all_ft_ends = []
        ft_positions = []  # list of (batch_idx, seq_pos)

        if has_frame_times:
            # Normalize frame times to [0, reg_max]
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
            # Batched single call
            ft_starts_tensor = torch.stack(all_ft_starts)  # (N_ft,)
            ft_ends_tensor = torch.stack(all_ft_ends)      # (N_ft,)
            ft_embeds = self.time_encoder(ft_starts_tensor, ft_ends_tensor)  # (N_ft, hidden)
            # Scatter back
            for idx, (b, pos) in enumerate(ft_positions):
                inputs_embeds[b, pos] = ft_embeds[idx]
        else:
            # Dummy forward: ensure time_encoder parameters are gathered on all ranks
            dummy_t = torch.zeros(1, device=device, dtype=inputs_embeds.dtype)
            _ = self.time_encoder(dummy_t, dummy_t)  # result discarded

        # ================================================================
        # TIME_STAMP GT encoding: batched single call to time_encoder
        # (ZeRO-3 FIX: all ranks execute exactly ONE time_encoder forward for TIME_STAMP)
        # ================================================================
        all_gt_starts = []
        all_gt_ends = []
        gt_positions = []  # list of (batch_idx, seq_pos)

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
            # Batched single call
            gt_starts_tensor = torch.stack(all_gt_starts)  # (N_gt,)
            gt_ends_tensor = torch.stack(all_gt_ends)      # (N_gt,)
            gt_embeds = self.time_encoder(gt_starts_tensor, gt_ends_tensor)  # (N_gt, hidden)
            # Scatter back: replace TIME_STAMP embeddings with GT time embeddings
            for idx, (b, pos) in enumerate(gt_positions):
                inputs_embeds[b, pos] = gt_embeds[idx]
        else:
            # Dummy forward: ensure time_encoder parameters are gathered on all ranks
            # (This is the SECOND call; FRAME_TIME was the first. Both always execute.)
            dummy_t = torch.zeros(1, device=device, dtype=inputs_embeds.dtype)
            _ = self.time_encoder(dummy_t, dummy_t)

        # ================================================================
        # Forward through base model
        # ================================================================
        outputs = self.base_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            pixel_values=pixel_values,
            image_flags=image_flags,
            **kwargs
        )

        # Get LM loss
        lm_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=device)

        # Get last layer hidden states
        if outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]  # (B, seq_len, hidden)
        else:
            hidden_states = None

        # ================================================================
        # Time decoding: ALWAYS call time_decoder forward
        # (ZeRO-3 FIX: all ranks execute exactly ONE time_decoder forward)
        # Uses SHIFTED mask (time_stamp_mask_dec at position p-1) — P0 FIX
        # ================================================================
        if hidden_states is not None and time_stamp_mask_dec.any():
            # Real forward: decode time from hidden states at position p-1
            time_logits, pred_times = self.time_decoder(hidden_states, time_stamp_mask_dec)
        else:
            # Dummy forward: pass through time_decoder to keep BROADCAST ops in sync
            # Following InternVL (modeling_internvl_chat_distime.py:472-477)
            if hidden_states is not None:
                dummy_hidden = torch.zeros_like(hidden_states)
            else:
                # Fallback: create minimal dummy
                dummy_hidden = torch.zeros(
                    batch_size, 1, self.distime_config.hidden_size,
                    device=device, dtype=inputs_embeds.dtype
                )
            dummy_mask = torch.zeros(
                dummy_hidden.shape[0], dummy_hidden.shape[1],
                dtype=torch.bool, device=device
            )
            dummy_mask[0, 0] = True  # Need at least one True for forward to work
            time_logits, pred_times = self.time_decoder(dummy_hidden, dummy_mask)
            # Zero out dummy outputs so they don't affect loss
            time_logits = time_logits * 0
            pred_times = pred_times * 0

        # ================================================================
        # Compute DisTime losses
        # ================================================================
        time_loss = (time_logits * 0).sum()  # default: zero loss connected to decoder params
        iou_loss = (pred_times * 0).sum()

        if self.training and time_gt is not None and time_stamp_mask_dec.any() and pred_times.shape[0] > 0:
            # Multi-event loss alignment: each TIME_STAMP prediction matches its GT event
            num_stamps_per_sample = time_stamp_mask_dec.sum(dim=1)  # (batch,)
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
                target_times = torch.cat(target_expanded, dim=0)  # (N, 2)
                target_times = target_times.clamp(min=0, max=reg_max - 1e-3)

                # DFL loss
                time_loss = self.dfl_loss_fn(time_logits, target_times)

                # DIoU loss
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

    # ================================================================
    # 方案B: 完整自回归闭环 (对标 InternVL _sample)
    # ================================================================
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

        Mirrors InternVL's generate_distime() + _sample() architecture:
        1. Prefill: inject FRAME_TIME embeddings + process vision features
        2. Autoregressive loop: when TIME_STAMP is generated,
           decode time → encode back → inject as next input embedding
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]
        input_len = input_ids.shape[1]
        reg_max = self.distime_config.reg_max

        # EOS 处理: 支持多个 stop token (与 fastvlm_distime.py 对齐)
        # SmolVLM 可能同时需要 <end_of_utterance> 和 <|im_end|> 作为 stop token
        default_eos = self.processor.tokenizer.eos_token_id
        eos_token_ids = generate_kwargs.pop('eos_token_id', [default_eos])
        if not isinstance(eos_token_ids, list):
            eos_token_ids = [eos_token_ids]

        # ================================================================
        # Step 1: Build inputs_embeds with FRAME_TIME injection
        # (Same as forward() lines 492-571)
        # ================================================================
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
                    ft_embeds = self.time_encoder(times_b[:n], times_b[:n])  # (n, hidden)
                    for i in range(n):
                        inputs_embeds[b, indices[i]] = ft_embeds[i]

        # ================================================================
        # Step 2: Prefill - forward through base_model to get KV cache
        # base_model handles vision (pixel_values) + text (inputs_embeds)
        # input_ids is passed so base_model can locate <image> positions
        # ================================================================
        outputs = self.base_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
        )

        past_key_values = outputs.past_key_values
        # Last hidden state from prefill - needed for time decoding if
        # the very first generated token is TIME_STAMP
        last_hidden = outputs.hidden_states[-1]  # (B, prefill_len, hidden)
        next_token_logits = outputs.logits[:, -1, :]  # (B, vocab_size)

        # ================================================================
        # Step 3: Autoregressive loop with TIME_STAMP feedback
        # (Mirrors InternVL _sample lines 915-1013)
        #
        # Key: when TIME_STAMP is generated, we:
        #   1. Decode time from hidden_state at p-1 (the step that predicted TIME_STAMP)
        #   2. Encode predicted time back via time_encoder → embedding
        #   3. Feed this embedding (not the raw TIME_STAMP token embedding) as next input
        # This matches training where TIME_STAMP positions always have time_encoder output.
        # ================================================================
        generated_ids = []
        pred_times = []
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)

        for step in range(max_new_tokens):
            # --- Token selection ---
            if do_sample and temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)  # (B,)

            generated_ids.append(next_tokens.clone())

            # --- Check EOS (支持多个 stop token) ---
            if eos_token_ids:
                for eid in eos_token_ids:
                    unfinished = unfinished & (next_tokens != eid)
                if not unfinished.any():
                    break

            # --- Extend attention mask for new token ---
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=device)
            ], dim=1)

            # --- Check if TIME_STAMP was generated ---
            is_time_stamp = (next_tokens == self.time_stamp_token_id)

            if is_time_stamp.any():
                # ---- TIME_STAMP feedback loop ----
                # Decode time from hidden state at position p-1
                # last_hidden[:, -1] is the hidden state from the forward pass
                # that produced the logits predicting TIME_STAMP — exactly p-1
                # (Same as InternVL _sample lines 975-982)
                decode_hidden = last_hidden[:, -1:, :]  # (B, 1, hidden)
                decode_mask = torch.ones(
                    batch_size, 1, dtype=torch.bool, device=device
                )
                _, pred_time_step = self.time_decoder(decode_hidden, decode_mask)
                # pred_time_step: (B, 2) in [0, reg_max] scale
                pred_times.append(pred_time_step)

                # Encode predicted time back to embedding
                # (Same as InternVL _sample lines 987-991)
                time_embeds = self.time_encoder(
                    pred_time_step[:, 0], pred_time_step[:, 1]
                )  # (B, hidden)
                next_input_embeds = time_embeds.unsqueeze(1)  # (B, 1, hidden)

                # Forward with time embedding (not token embedding!)
                # (Same as InternVL _sample lines 927-929)
                outputs = self.base_model(
                    inputs_embeds=next_input_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=True,
                )
            else:
                # ---- Normal token: use standard embedding ----
                outputs = self.base_model(
                    input_ids=next_tokens.unsqueeze(1),  # (B, 1)
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=True,
                )

            # Update state for next iteration
            past_key_values = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1]  # (B, 1, hidden)
            next_token_logits = outputs.logits[:, -1, :]

        # ================================================================
        # Step 4: Assemble results
        # ================================================================
        if generated_ids:
            gen_ids = torch.stack(generated_ids, dim=1)  # (B, gen_len)
            output_ids = torch.cat([input_ids, gen_ids], dim=1)
        else:
            gen_ids = torch.empty(batch_size, 0, dtype=torch.long, device=device)
            output_ids = input_ids

        # Decode only generated part (not prompt)
        generated_text = self.processor.tokenizer.batch_decode(
            gen_ids, skip_special_tokens=False
        )

        # Process predicted times
        pred_times_list = [None] * batch_size
        pred_times_normalized_list = [None] * batch_size

        if pred_times:
            all_preds = torch.cat(pred_times, dim=0)  # (total_stamps, 2)
            pred_normalized = all_preds / reg_max  # → [0, 1]

            # Split per sample (for batch_size=1, simple case)
            # TODO: for batch>1, need to track which TIME_STAMPs belong to which sample
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
        # Only save on main process in distributed training
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            dist.barrier()  # Wait for main process to finish saving
            return

        os.makedirs(save_directory, exist_ok=True)

        # Save base model (with LoRA keys in state_dict for checkpoint compatibility)
        self.base_model.save_pretrained(save_directory)

        # Save DisTime modules
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

        # Sync all processes after saving
        if dist.is_initialized():
            dist.barrier()

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load model from directory."""
        # Load DisTime modules
        distime_path = os.path.join(model_path, 'distime_modules.pt')
        if os.path.exists(distime_path):
            distime_state = torch.load(distime_path, map_location='cpu')
            distime_config = DisTimeConfig(**distime_state['config'])
        else:
            distime_config = DisTimeConfig()

        # Create model
        model = cls(
            model_name_or_path=model_path,
            distime_config=distime_config,
            **kwargs
        )

        # Load DisTime weights
        if os.path.exists(distime_path):
            model.time_encoder.load_state_dict(distime_state['time_encoder'])
            model.time_decoder.load_state_dict(distime_state['time_decoder'])
            logger.info(f"Loaded DisTime modules from {distime_path}")

        return model