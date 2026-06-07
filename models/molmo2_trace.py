"""
Molmo2-TRACE: Molmo2 with TRACE-style causal event modeling.

TRACE paradigm (Guo et al. 2024): Structured token generation with separate
encoder-decoder pairs for text, time, and score modalities. Events are decoded
autoregressively through a head-switching state machine:
    text -> <sync> -> time_chars -> <time_sync> -> score_chars -> <score_sync> -> text

This file mirrors FastVLMTrace but uses Molmo2 as the VLM backbone.
Key differences from FastVLMTrace:
    - Base model: AutoModelForImageTextToText + trust_remote_code=True
    - Vision module: vision_backbone (not vision_tower)
    - LLM: model.transformer (not model.language_model)
    - Embedding: Molmo2Embedding (custom class with .embedding + .new_embedding)
    - LoRA targets: att_proj, attn_out, ff_proj, ff_out (Molmo2/Qwen3 fused naming)
    - Forward requires: image_token_pooling, image_grids, image_num_crops
    - Internal vision merge via build_input_embeddings()
    - XOR constraint: cannot pass both input_ids AND inputs_embeds

Token ID Layout (identical to SmolVLMTrace/FastVLMTrace):
    [0, vocab_size)                              : text tokens (LLM native)
    vocab_size                                   : sync token (text -> time)
    [vocab_size+1, vocab_size+1+13)              : time character tokens (local 0-12)
    [vocab_size+1+13, vocab_size+1+13+13)        : score character tokens (local 0-12)
"""

import os
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)

from .trace_modules import (
    TimeTower, ScoreTower, SyncTower,
    TimeTokenizer, ScoreTokenizer,
    decode_trace_tokens,
)

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


@dataclass
class TraceConfig:
    """
    Configuration for TRACE modules.
    Parallel to DisTimeConfig — controls the TRACE-specific components only.
    """
    time_vocab_size: int = 13
    score_vocab_size: int = 13
    hidden_size: int = 2560       # Molmo2-4B default (Qwen3-4B)


class Molmo2Trace(nn.Module):
    """
    Molmo2 with TRACE-style causal event modeling.

    Architecture:
    - Base: Molmo2 (SigLIP2 vision backbone + Qwen3-4B LLM)
    - Added: TimeTower, ScoreTower, SyncTower           [TRACE-specific embedding]
    - Added: time_head, score_head, sync_head            [TRACE-specific output heads]
    - Loss: text_CE + time_CE + score_CE                 [vs DisTime's LM + DFL + DIoU]

    Token ID Layout (extended beyond LLM vocab):
        [0, vocab_size)                              : text tokens (LLM native)
        vocab_size                                   : sync token (text -> time)
        [vocab_size+1, vocab_size+1+time_vocab)      : time character tokens
        [vocab_size+1+time_vocab, ...+score_vocab)   : score character tokens

    Head-switching state machine (inference):
        head 0 (text) --[sync]--> head 1 (time) --[time_sync]--> head 2 (score) --[score_sync]--> head 0
    """

    def __init__(
        self,
        model_name_or_path: str = "allenai/Molmo2-4B",
        trace_config: Optional[TraceConfig] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        use_flash_attention: bool = False,
    ):
        super().__init__()

        self.trace_config = trace_config or TraceConfig()

        # ================================================================
        # Load base model (Molmo2-specific)
        # ================================================================
        logger.info(f"Loading Molmo2 base model from {model_name_or_path}")

        # Molmo2 的 modeling_molmo2.py 声明 _supports_flash_attn=True
        # transformers 在 super().__init__(config) 里会根据这个类属性自动选 flash_attention_2
        # 即使在 config 上设了 sdpa，deep copy 或内部逻辑也可能丢掉
        # 最可靠的做法：直接 monkey-patch 模型类的 _supports_flash_attn
        attn_impl = "flash_attention_2" if use_flash_attention else "sdpa"

        if not use_flash_attention:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                model_name_or_path, trust_remote_code=True,
            )
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

        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )

        # ================================================================
        # Detect hidden size
        # ================================================================
        config = self.base_model.config
        if hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
            hidden_size = config.text_config.hidden_size
        elif hasattr(config, 'hidden_size'):
            hidden_size = config.hidden_size
        else:
            hidden_size = 2560  # Molmo2-4B default (Qwen3-4B)
            logger.warning(f"Could not detect hidden_size, using default {hidden_size}")

        self.trace_config.hidden_size = hidden_size
        logger.info(f"Detected hidden_size: {hidden_size}")

        # Store image_patch_id for vision merge
        self.image_patch_id = getattr(config, 'image_patch_id', None)

        # ================================================================
        # Vocab layout: DO NOT resize token embeddings.
        # TRACE uses separate towers for time/score/sync, not the LLM embedding.
        # vocab_size must match lm_head output dimension
        # ================================================================
        lm_head = self.base_model.get_output_embeddings()
        if lm_head is not None and hasattr(lm_head, 'out_features'):
            self.vocab_size = lm_head.out_features
        elif lm_head is not None and hasattr(lm_head, 'weight'):
            self.vocab_size = lm_head.weight.shape[0]
        elif hasattr(config, 'text_config') and hasattr(config.text_config, 'vocab_size'):
            self.vocab_size = config.text_config.vocab_size
        elif hasattr(config, 'vocab_size'):
            self.vocab_size = config.vocab_size
        else:
            self.vocab_size = len(self.processor.tokenizer)

        # Molmo2 has additional_vocab_size, total = vocab_size + additional_vocab_size
        if hasattr(config, 'text_config'):
            additional = getattr(config.text_config, 'additional_vocab_size', 0)
            base_vocab = getattr(config.text_config, 'vocab_size', 0)
            if additional > 0 and base_vocab > 0:
                total_expected = base_vocab + additional
                # Use lm_head size as ground truth
                if lm_head is not None and hasattr(lm_head, 'weight'):
                    self.vocab_size = lm_head.weight.shape[0]
                else:
                    self.vocab_size = total_expected

        logger.info(f"vocab_size (from lm_head/config): {self.vocab_size}")

        # Token ID boundaries (extended above vocab_size)
        self.sync_token_id = self.vocab_size                                     # 1 ID
        self.time_start_id = self.vocab_size + 1                                 # 13 IDs
        self.time_end_id = self.time_start_id + self.trace_config.time_vocab_size
        self.score_start_id = self.time_end_id                                   # 13 IDs
        self.score_end_id = self.score_start_id + self.trace_config.score_vocab_size

        logger.info(f"Token ID layout: sync={self.sync_token_id}, "
                     f"time=[{self.time_start_id}, {self.time_end_id}), "
                     f"score=[{self.score_start_id}, {self.score_end_id})")

        # ================================================================
        # Initialize TRACE modules
        # ================================================================
        self._init_trace_modules()

        # ================================================================
        # Head-switching map for autoregressive inference
        # ================================================================
        self.swap_tokens = {
            self.sync_token_id: 1,       # text <sync>  -> switch to time head
            self.time_start_id: 2,       # time <sync> (local 0) -> switch to score head
            self.score_start_id: 0,      # score <sync> (local 0) -> switch to text head
        }

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the base model."""
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def _init_trace_modules(self):
        """Initialize TRACE-specific embedding towers and output heads."""
        cfg = self.trace_config

        # Tokenizers (stateless, no parameters)
        self.time_tokenizer = TimeTokenizer()
        self.score_tokenizer = ScoreTokenizer()

        # Embedding towers
        self.time_tower = TimeTower(self.time_tokenizer, cfg.hidden_size)
        self.score_tower = ScoreTower(self.score_tokenizer, cfg.hidden_size)
        self.sync_tower = SyncTower(cfg.hidden_size)

        # Output heads (separate from lm_head)
        self.time_head = nn.Linear(cfg.hidden_size, cfg.time_vocab_size, bias=False)
        self.score_head = nn.Linear(cfg.hidden_size, cfg.score_vocab_size, bias=False)
        self.sync_head = nn.Linear(cfg.hidden_size, 1, bias=False)

        logger.info(f"Initialized TRACE modules: "
                     f"time_vocab={cfg.time_vocab_size}, "
                     f"score_vocab={cfg.score_vocab_size}, "
                     f"hidden_size={cfg.hidden_size}")

    # ================================================================
    # Vision merge helper (shared with DisTime)
    # ================================================================

    def _build_inputs_with_vision(
        self,
        input_ids: torch.Tensor,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_token_pooling: Optional[torch.Tensor] = None,
        video_grids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build inputs_embeds by calling Molmo2's internal vision merge (video path).
        Returns inputs_embeds with vision features merged at image_patch_id positions.
        """
        inner_model = self.base_model.model  # Molmo2Model

        images, token_pooling = inner_model.merge_visual_inputs(
            input_ids=input_ids,
            pixel_values_videos=pixel_values_videos,
            video_token_pooling=video_token_pooling,
            video_grids=video_grids,
        )

        inputs_embeds, _ = inner_model.build_input_embeddings(
            input_ids=input_ids,
            images=images,
            token_pooling=token_pooling,
        )

        return inputs_embeds

    # ================================================================
    # Training setup
    # ================================================================

    def freeze_vision_encoder(self):
        """Freeze the vision backbone (SigLIP2 ViT + connector + projector).

        Molmo2 structure:
            base_model.model.vision_backbone
              ├── image_vit
              ├── image_pooling_2d
              └── image_projector
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
        """
        Setup model for training.
        Molmo2-specific LoRA targets and vision_backbone exclusion.
        """
        # Step 1: Freeze everything first
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Step 2: Freeze vision encoder
        if freeze_vision:
            self.freeze_vision_encoder()

        # Step 3: Apply manual LoRA (NO PeftModel wrapper!)
        if use_lora:
            from .manual_lora import apply_lora_to_model

            # If passed as comma-separated string (e.g. from command line), split into list
            if isinstance(lora_target_modules, str):
                lora_target_modules = [m.strip() for m in lora_target_modules.split(',')]

            if lora_target_modules is None:
                # Molmo2 (Qwen3) uses fused QKV naming:
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

        # Step 4: Unfreeze embed_tokens + lm_head
        embed_layer = self.base_model.get_input_embeddings()
        if embed_layer is not None:
            for param in embed_layer.parameters():
                param.requires_grad = True

        lm_head = self.base_model.get_output_embeddings()
        if lm_head is not None and lm_head is not embed_layer:
            for param in lm_head.parameters():
                param.requires_grad = True
        elif lm_head is embed_layer:
            logger.info("lm_head shares weights with embed (tie_word_embeddings=True)")

        # Step 5: Enable input require grads
        def _enable_input_require_grads(module, input, output):
            output.requires_grad_(True)

        embed = self.base_model.get_input_embeddings()
        if embed is not None:
            embed.register_forward_hook(_enable_input_require_grads)

        # Step 6: Ensure TRACE modules are trainable
        for module in [self.time_tower, self.score_tower, self.sync_tower,
                       self.time_head, self.score_head, self.sync_head]:
            for param in module.parameters():
                param.requires_grad = True

        # Step 7: Print trainable parameters
        trainable = 0
        total = 0
        lora_count = 0
        trace_count = 0
        for name, param in self.named_parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
                if 'lora_' in name:
                    lora_count += param.numel()
                if any(x in name for x in ['time_tower', 'score_tower', 'sync_tower',
                                           'time_head', 'score_head', 'sync_head']):
                    trace_count += param.numel()

        logger.info(f"trainable params: {trainable:,} || all params: {total:,} "
                     f"|| trainable%: {100 * trainable / total:.4f}")
        logger.info(f"  of which LoRA params: {lora_count:,}")
        logger.info(f"  of which TRACE params: {trace_count:,}")

    def get_trainable_parameters(self) -> Tuple[int, int]:
        """Get number of trainable and total parameters."""
        trainable = 0
        total = 0
        for param in self.parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
        return trainable, total

    # ================================================================
    # Forward pass
    # ================================================================

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_token_pooling: Optional[torch.Tensor] = None,
        video_grids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        time_labels: Optional[torch.Tensor] = None,
        score_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ):
        """
        Forward pass with TRACE multi-head temporal modeling.

        Molmo2-specific:
        1. Vision merge via build_input_embeddings() (handles image_token_pooling etc.)
        2. After vision merge, route extended tokens to their towers
        3. Bypass Molmo2Model.forward(), call transformer directly (XOR constraint)
        4. Manual lm_head + multi-head logit computation
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # ================================================================
        # Step 1: Build inputs_embeds with vision features merged
        # ================================================================
        # Clamp extended IDs to valid range for Molmo2 embedding
        # (extended IDs >= vocab_size would cause index errors in wte)
        safe_ids = input_ids.clone()
        safe_ids[safe_ids >= self.vocab_size] = 0  # placeholder, will be overwritten

        inputs_embeds = self._build_inputs_with_vision(
            input_ids=safe_ids,
            pixel_values_videos=pixel_values_videos,
            video_token_pooling=video_token_pooling,
            video_grids=video_grids,
        ).clone()

        # ================================================================
        # Step 1b: Replace extended token embeddings with tower outputs
        # ================================================================
        sync_mask = (input_ids == self.sync_token_id)
        time_mask = (input_ids >= self.time_start_id) & (input_ids < self.time_end_id)
        score_mask = (input_ids >= self.score_start_id) & (input_ids < self.score_end_id)

        if sync_mask.any():
            sync_embeds = self.sync_tower(input_ids[sync_mask])
            inputs_embeds[sync_mask] = sync_embeds

        if time_mask.any():
            time_local_ids = input_ids[time_mask] - self.time_start_id
            time_embeds = self.time_tower(time_local_ids)
            inputs_embeds[time_mask] = time_embeds

        if score_mask.any():
            score_local_ids = input_ids[score_mask] - self.score_start_id
            score_embeds = self.score_tower(score_local_ids)
            inputs_embeds[score_mask] = score_embeds

        # ================================================================
        # ZeRO-3 FIX: dummy forwards to keep NCCL collectives synchronized
        # ================================================================
        if not sync_mask.any():
            _ = self.sync_tower(torch.zeros(1, dtype=torch.long, device=device))
        if not time_mask.any():
            _ = self.time_tower(torch.zeros(1, dtype=torch.long, device=device))
        if not score_mask.any():
            _ = self.score_tower(torch.zeros(1, dtype=torch.long, device=device))

        # Dummy forwards for output heads
        dummy_h = torch.zeros(1, 1, self.trace_config.hidden_size,
                              device=device, dtype=inputs_embeds.dtype)
        if not time_mask.any():
            _ = self.time_head(dummy_h)
        if not score_mask.any():
            _ = self.score_head(dummy_h)
        if not sync_mask.any():
            _ = self.sync_head(dummy_h)

        # ================================================================
        # Step 2: Forward through transformer (bypass Molmo2Model)
        # ================================================================
        transformer = self.base_model.model.transformer

        outputs = transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state

        # ================================================================
        # Step 3: Multi-head logit computation (TRACE core)
        # ================================================================
        text_logits = self.base_model.lm_head(hidden_states)      # (B, seq, vocab_size)

        sync_logits = self.sync_head(hidden_states)               # (B, seq, 1)
        text_sync_logits = torch.cat([text_logits, sync_logits], dim=-1).float()
        del text_logits, sync_logits

        time_logits = self.time_head(hidden_states).float()       # (B, seq, time_vocab_size)
        score_logits = self.score_head(hidden_states).float()     # (B, seq, score_vocab_size)
        del hidden_states

        # ================================================================
        # Step 4: Loss computation — sum of three CE losses
        # ================================================================
        loss = None
        text_loss = torch.tensor(0.0, device=device)
        time_loss = torch.tensor(0.0, device=device)
        score_loss = torch.tensor(0.0, device=device)

        if labels is not None:
            loss_fct = CrossEntropyLoss()

            # Text + sync CE loss
            shift_logits = text_sync_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            text_loss = loss_fct(
                shift_logits.view(-1, self.vocab_size + 1),
                shift_labels.view(-1).to(shift_logits.device)
            )

            # Time CE loss
            if time_labels is not None:
                shift_time_logits = time_logits[..., :-1, :].contiguous()
                shift_time_labels = time_labels[..., 1:].contiguous()
                time_loss = loss_fct(
                    shift_time_logits.view(-1, self.trace_config.time_vocab_size),
                    shift_time_labels.view(-1).to(shift_time_logits.device)
                )

            # Score CE loss
            if score_labels is not None:
                shift_score_logits = score_logits[..., :-1, :].contiguous()
                shift_score_labels = score_labels[..., 1:].contiguous()
                score_loss = loss_fct(
                    shift_score_logits.view(-1, self.trace_config.score_vocab_size),
                    shift_score_labels.view(-1).to(shift_score_logits.device)
                )

            loss = text_loss + time_loss + score_loss

        logits = torch.cat([text_sync_logits, time_logits, score_logits], dim=-1)

        if return_dict:
            return {
                'loss': loss,
                'text_loss': text_loss,
                'time_loss': time_loss,
                'score_loss': score_loss,
                'logits': logits,
            }
        else:
            return loss

    # ================================================================
    # Generation with head-switching state machine
    # ================================================================

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_token_pooling: Optional[torch.Tensor] = None,
        video_grids: Optional[torch.Tensor] = None,
        duration: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        **generate_kwargs
    ) -> dict:
        """
        Generate with TRACE head-switching state machine.
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]

        eos_token_id = generate_kwargs.pop('eos_token_id',
                                           self.processor.tokenizer.eos_token_id)
        # Normalize to list for uniform EOS checking
        if eos_token_id is None:
            eos_token_ids = []
        elif isinstance(eos_token_id, (list, tuple)):
            eos_token_ids = list(eos_token_id)
        else:
            eos_token_ids = [eos_token_id]

        # ================================================================
        # Step 1: Build inputs_embeds with tower routing + vision merge
        # ================================================================
        safe_ids = input_ids.clone()
        safe_ids[safe_ids >= self.vocab_size] = 0

        inputs_embeds = self._build_inputs_with_vision(
            input_ids=safe_ids,
            pixel_values_videos=pixel_values_videos,
            video_token_pooling=video_token_pooling,
            video_grids=video_grids,
        ).clone()

        # Route extended tokens to towers
        sync_mask = (input_ids == self.sync_token_id)
        time_mask = (input_ids >= self.time_start_id) & (input_ids < self.time_end_id)
        score_mask = (input_ids >= self.score_start_id) & (input_ids < self.score_end_id)

        if sync_mask.any():
            inputs_embeds[sync_mask] = self.sync_tower(input_ids[sync_mask])
        if time_mask.any():
            inputs_embeds[time_mask] = self.time_tower(input_ids[time_mask] - self.time_start_id)
        if score_mask.any():
            inputs_embeds[score_mask] = self.score_tower(input_ids[score_mask] - self.score_start_id)

        # ================================================================
        # Step 2: Prefill -> KV cache (through transformer directly)
        # ================================================================
        transformer = self.base_model.model.transformer

        outputs = transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
        )

        past_key_values = outputs.past_key_values
        hidden_states = outputs.last_hidden_state

        # Get embed layer and lm_head
        embed_layer = self.base_model.get_input_embeddings()
        lm_head = self.base_model.lm_head

        # Initial full logits
        last_h = hidden_states[:, -1:, :]
        text_logits = lm_head(last_h).squeeze(1)
        sync_logits = self.sync_head(last_h).squeeze(1)
        time_logits = self.time_head(last_h).squeeze(1)
        score_logits = self.score_head(last_h).squeeze(1)
        full_logits = torch.cat([text_logits, sync_logits, time_logits, score_logits], dim=-1)

        # ================================================================
        # Step 3: Autoregressive loop with head switching
        # ================================================================
        heads = [0] * batch_size

        vocab_ranges = [
            (0, self.vocab_size + 1),
            (self.vocab_size + 1,
             self.vocab_size + 1 + self.trace_config.time_vocab_size),
            (self.vocab_size + 1 + self.trace_config.time_vocab_size,
             self.vocab_size + 1 + self.trace_config.time_vocab_size
             + self.trace_config.score_vocab_size),
        ]

        generated_ids = []
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)

        for step in range(max_new_tokens):
            # Mask logits to active head's vocab range
            masked_logits = torch.full_like(full_logits, -float('inf'))
            for b in range(batch_size):
                start, end = vocab_ranges[heads[b]]
                masked_logits[b, start:end] = full_logits[b, start:end]

            # Sample / argmax
            if do_sample and temperature > 0:
                probs = torch.softmax(masked_logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(masked_logits, dim=-1)

            generated_ids.append(next_tokens.clone())

            # Check EOS
            if eos_token_ids:
                is_eos = torch.zeros_like(unfinished)
                for eid in eos_token_ids:
                    is_eos = is_eos | (next_tokens == eid)
                unfinished = unfinished & (~is_eos)
                if not unfinished.any():
                    break

            # Head switching
            for b in range(batch_size):
                token_id = next_tokens[b].item()
                if token_id in self.swap_tokens:
                    heads[b] = self.swap_tokens[token_id]

            # Build next input embeddings from appropriate tower
            next_embeds_list = []
            for b in range(batch_size):
                token_id = next_tokens[b].item()
                if token_id < self.vocab_size:
                    emb = embed_layer(next_tokens[b:b + 1])
                elif token_id == self.sync_token_id:
                    emb = self.sync_tower(torch.zeros(1, dtype=torch.long, device=device))
                elif token_id < self.time_end_id:
                    local_id = torch.tensor([token_id - self.time_start_id], device=device)
                    emb = self.time_tower(local_id)
                else:
                    local_id = torch.tensor([token_id - self.score_start_id], device=device)
                    emb = self.score_tower(local_id)
                next_embeds_list.append(emb)

            next_embeds = torch.stack(next_embeds_list, dim=0)

            # Extend attention mask
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=device)
            ], dim=1)

            # Forward step (through transformer directly)
            outputs = transformer(
                inputs_embeds=next_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True,
                return_dict=True,
                use_cache=True,
            )

            past_key_values = outputs.past_key_values
            hidden_states = outputs.last_hidden_state

            # Compute full logits for next step
            last_h = hidden_states[:, -1:, :]
            text_logits = lm_head(last_h).squeeze(1)
            sync_logits = self.sync_head(last_h).squeeze(1)
            time_logits = self.time_head(last_h).squeeze(1)
            score_logits = self.score_head(last_h).squeeze(1)
            full_logits = torch.cat([text_logits, sync_logits, time_logits, score_logits], dim=-1)

        # ================================================================
        # Step 4: Assemble results
        # ================================================================
        if generated_ids:
            gen_ids = torch.stack(generated_ids, dim=1)
            output_ids = torch.cat([input_ids, gen_ids], dim=1)
        else:
            gen_ids = torch.empty(batch_size, 0, dtype=torch.long, device=device)
            output_ids = input_ids

        # Decode text tokens
        text_gen_ids = gen_ids.clone()
        text_gen_ids[text_gen_ids >= self.vocab_size] = self.processor.tokenizer.pad_token_id or 0
        generated_text = self.processor.tokenizer.batch_decode(
            text_gen_ids, skip_special_tokens=False
        )

        # Extract predicted times and scores
        pred_times_list = [None] * batch_size
        pred_times_normalized_list = [None] * batch_size

        for b in range(batch_size):
            sample_ids = gen_ids[b].tolist()
            result = decode_trace_tokens(
                token_ids=sample_ids,
                time_tokenizer=self.time_tokenizer,
                score_tokenizer=self.score_tokenizer,
                time_start_id=self.time_start_id,
                time_end_id=self.time_end_id,
                score_start_id=self.score_start_id,
                score_end_id=self.score_end_id,
                sync_token_id=self.sync_token_id,
            )

            if result['times']:
                times_tensor = torch.tensor(result['times'], device=device)
                pred_times_list[b] = times_tensor
                if duration is not None:
                    d = duration[b].item() if duration.dim() > 0 else duration.item()
                    pred_times_normalized_list[b] = times_tensor / max(d, 1e-6)

        return {
            'generated_ids': output_ids,
            'generated_text': generated_text,
            'pred_times': pred_times_list,
            'pred_times_normalized': pred_times_normalized_list,
        }

    # ================================================================
    # Save / Load
    # ================================================================

    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            dist.barrier()
            return

        os.makedirs(save_directory, exist_ok=True)

        self.base_model.save_pretrained(save_directory)

        trace_state = {
            'time_tower': self.time_tower.state_dict(),
            'score_tower': self.score_tower.state_dict(),
            'sync_tower': self.sync_tower.state_dict(),
            'time_head': self.time_head.state_dict(),
            'score_head': self.score_head.state_dict(),
            'sync_head': self.sync_head.state_dict(),
            'config': asdict(self.trace_config),
        }
        torch.save(trace_state, os.path.join(save_directory, 'trace_modules.pt'))

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
        trace_path = os.path.join(model_path, 'trace_modules.pt')
        if os.path.exists(trace_path):
            trace_state = torch.load(trace_path, map_location='cpu')
            trace_config = TraceConfig(**trace_state['config'])
        else:
            trace_config = TraceConfig()

        model = cls(
            model_name_or_path=model_path,
            trace_config=trace_config,
            **kwargs
        )

        if os.path.exists(trace_path):
            model.time_tower.load_state_dict(trace_state['time_tower'])
            model.score_tower.load_state_dict(trace_state['score_tower'])
            model.sync_tower.load_state_dict(trace_state['sync_tower'])
            model.time_head.load_state_dict(trace_state['time_head'])
            model.score_head.load_state_dict(trace_state['score_head'])
            model.sync_head.load_state_dict(trace_state['sync_head'])
            logger.info(f"Loaded TRACE modules from {trace_path}")

        return model

    # ================================================================
    # Utilities
    # ================================================================

    def print_training_setup(self):
        """Print detailed training setup for verification."""
        logger.info("=" * 60)
        logger.info("Molmo2 TRACE Training Setup")
        logger.info("=" * 60)

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
                                  and 'vision' not in name and 'tower' not in name):
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

        # 5. TRACE Modules
        trace_trainable = 0
        trace_total = 0
        for name, param in self.named_parameters():
            if any(x in name for x in ['time_tower', 'score_tower', 'sync_tower',
                                       'time_head', 'score_head', 'sync_head']):
                trace_total += param.numel()
                if param.requires_grad:
                    trace_trainable += param.numel()
        logger.info(f"[TRAIN ] TRACE Towers+Heads: {trace_trainable:,}/{trace_total:,} trainable")

        logger.info("-" * 60)
        trainable, total = self.get_trainable_parameters()
        logger.info(f"Total: {trainable:,} / {total:,} trainable ({100*trainable/total:.2f}%)")
        logger.info("=" * 60)
