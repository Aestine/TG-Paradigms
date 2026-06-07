"""
SmolVLM-TRACE: SmolVLM with TRACE-style causal event modeling.

TRACE paradigm (Guo et al. 2024): Structured token generation with separate
encoder-decoder pairs for text, time, and score modalities. Events are decoded
autoregressively through a head-switching state machine:
    text -> <sync> -> time_chars -> <time_sync> -> score_chars -> <score_sync> -> text

This file is designed to share the EXACT same backbone and training infrastructure
as SmolVLMDisTime, with the ONLY difference being the temporal output formulation:
    - DisTime: continuous distribution-based decoding (Gaussian -> bins -> DFL+DIoU)
    - TRACE:   discrete character-level token prediction (multi-head CrossEntropy)

Interface contract (compatible with SmolVLMDisTime):
    - Same __init__ pattern: model_name_or_path, config, dtype, device_map
    - Same setup_training(): freeze vision, apply LoRA, unfreeze embed/head
    - Same generate() return format: {generated_ids, generated_text, pred_times, ...}
    - Same save_pretrained() / from_pretrained() pattern

Token ID Layout (extended beyond LLM vocab):
    [0, vocab_size)                              : text tokens (LLM native)
    vocab_size                                   : sync token (text -> time)
    [vocab_size+1, vocab_size+1+13)              : time character tokens (local 0-12)
    [vocab_size+1+13, vocab_size+1+13+13)        : score character tokens (local 0-12)

TRACE character tokenizer vocab (13 tokens):
    0: <sync>   - end-of-sequence marker
    1: <sep>    - separator between start and end timestamps
    2-11: '0'-'9' - digit characters
    12: '.'     - decimal point
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
    All backbone/training settings remain shared via args.py.
    """
    time_vocab_size: int = 13       # TimeTokenizer vocab: <sync>, <sep>, 0-9, '.'
    score_vocab_size: int = 13      # ScoreTokenizer vocab: same structure
    hidden_size: int = 2048         # Auto-detected from backbone


class SmolVLMTrace(nn.Module):
    """
    SmolVLM with TRACE-style causal event modeling.

    Architecture:
    - Base: SmolVLM2 (vision encoder + LLM)       [IDENTICAL to DisTime]
    - Added: TimeTower, ScoreTower, SyncTower      [TRACE-specific embedding]
    - Added: time_head, score_head, sync_head      [TRACE-specific output heads]
    - Loss: text_CE + time_CE + score_CE           [vs DisTime's LM + DFL + DIoU]

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
        model_name_or_path: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        trace_config: Optional[TraceConfig] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        use_flash_attention: bool = True,
    ):
        super().__init__()

        self.trace_config = trace_config or TraceConfig()

        # ================================================================
        # Load base model (IDENTICAL to SmolVLMDisTime)
        # ================================================================
        logger.info(f"Loading base model from {model_name_or_path}")

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

        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )

        # ================================================================
        # Detect hidden size (IDENTICAL to SmolVLMDisTime)
        # ================================================================
        config = self.base_model.config
        if hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
            hidden_size = config.text_config.hidden_size
        elif hasattr(config, 'hidden_size'):
            hidden_size = config.hidden_size
        else:
            hidden_size = 2048
            logger.warning(f"Could not detect hidden_size, using default {hidden_size}")

        self.trace_config.hidden_size = hidden_size
        logger.info(f"Detected hidden_size: {hidden_size}")

        # ================================================================
        # Vocab layout: DO NOT resize token embeddings.
        # TRACE uses separate towers for time/score/sync, not the LLM embedding.
        # The lm_head stays at original vocab_size for text-only output.
        # ================================================================
        self.vocab_size = len(self.processor.tokenizer)
        logger.info(f"Original vocab_size: {self.vocab_size}")

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
        # Key: token ID that triggers switch, Value: new head index
        #   head 0 = text, head 1 = time, head 2 = score
        # ================================================================
        self.swap_tokens = {
            self.sync_token_id: 1,       # text <sync>  -> switch to time head
            self.time_start_id: 2,       # time <sync> (local 0, global=time_start_id) -> switch to score head
            self.score_start_id: 0,      # score <sync> (local 0, global=score_start_id) -> switch to text head
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
    # Training setup (IDENTICAL pattern to SmolVLMDisTime)
    # ================================================================

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
        IDENTICAL to SmolVLMDisTime.setup_training(), except Step 6 ensures
        TRACE modules (towers + heads) are trainable instead of DisTime modules.
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

        # Step 4: Unfreeze embed_tokens + lm_head (following original TRACE & DisTime)
        for name, param in self.base_model.named_parameters():
            if any(x in name for x in ["embed_tokens", "lm_head"]):
                param.requires_grad = True

        # Step 5: Enable input require grads
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
        pixel_values: Optional[torch.Tensor] = None,
        image_flags: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,          # text labels (sync pos -> vocab_size, time/score pos -> IGNORE)
        time_labels: Optional[torch.Tensor] = None,      # time head labels (non-time pos -> IGNORE)
        score_labels: Optional[torch.Tensor] = None,     # score head labels (non-score pos -> IGNORE)
        return_dict: bool = True,
        **kwargs
    ):
        """
        Forward pass with TRACE multi-head temporal modeling.

        Key differences from SmolVLMDisTime.forward():
        1. Embedding routing: tokens above vocab_size go through respective towers
        2. Multi-head output: text+sync, time, score heads computed separately
        3. Loss: sum of three CrossEntropy losses (one per head)
        4. No P0 FIX needed: TRACE is purely autoregressive, no GT injection

        Args:
            input_ids: (B, seq_len) token IDs. Extended IDs (>= vocab_size) are
                       routed to TRACE towers during embedding.
            attention_mask: (B, seq_len)
            pixel_values: Optional vision inputs (handled by SmolVLM natively)
            image_flags: Optional frame validity mask
            labels: (B, seq_len) text+sync labels. sync positions have value
                    `vocab_size`, time/score positions have IGNORE_INDEX.
            time_labels: (B, seq_len) time head labels. LOCAL token IDs (0-12)
                         at time positions, IGNORE_INDEX elsewhere.
            score_labels: (B, seq_len) score head labels. LOCAL token IDs (0-12)
                          at score positions, IGNORE_INDEX elsewhere.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # ================================================================
        # Step 1: Build inputs_embeds by routing tokens to different towers
        #
        # TRACE key difference: time/score tokens use separate embedding
        # tables (towers), NOT the LLM's embedding layer.
        # ================================================================
        if hasattr(self.base_model, 'get_input_embeddings'):
            embed_layer = self.base_model.get_input_embeddings()
        else:
            embed_layer = self.base_model.model.embed_tokens

        # Clamp to valid text range for base embedding layer
        safe_ids = torch.clamp(input_ids, min=0, max=self.vocab_size - 1)
        inputs_embeds = embed_layer(safe_ids).clone()

        # Identify extended token positions
        sync_mask = (input_ids == self.sync_token_id)
        time_mask = (input_ids >= self.time_start_id) & (input_ids < self.time_end_id)
        score_mask = (input_ids >= self.score_start_id) & (input_ids < self.score_end_id)

        # Replace with tower embeddings
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
        # Same pattern as SmolVLMDisTime, ensuring all ranks call all towers.
        # ================================================================
        if not sync_mask.any():
            _ = self.sync_tower(torch.zeros(1, dtype=torch.long, device=device))
        if not time_mask.any():
            _ = self.time_tower(torch.zeros(1, dtype=torch.long, device=device))
        if not score_mask.any():
            _ = self.score_tower(torch.zeros(1, dtype=torch.long, device=device))

        # Dummy forwards for output heads (ensure parameters gathered)
        dummy_h = torch.zeros(1, 1, self.trace_config.hidden_size,
                              device=device, dtype=inputs_embeds.dtype)
        if not time_mask.any():
            _ = self.time_head(dummy_h)
        if not score_mask.any():
            _ = self.score_head(dummy_h)
        if not sync_mask.any():
            _ = self.sync_head(dummy_h)

        # ================================================================
        # Step 2: Forward through base model
        #
        # Sanitize input_ids: replace extended tokens with pad_token_id
        # so SmolVLM can still find <image> positions from input_ids,
        # but won't crash on out-of-vocab IDs.
        # ================================================================
        model_input_ids = input_ids.clone()
        extended_mask = input_ids >= self.vocab_size
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        model_input_ids[extended_mask] = pad_id

        # DO NOT pass labels: TRACE computes all losses manually
        # from hidden_states via multi-head projection
        outputs = self.base_model(
            input_ids=model_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            pixel_values=pixel_values,
            image_flags=image_flags,
            **kwargs
        )

        # Get last-layer hidden states + logits, then free outputs to save memory
        if outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]  # (B, seq_len, hidden)
        else:
            raise RuntimeError("hidden_states is None; ensure output_hidden_states=True")

        # ================================================================
        # Step 3: Multi-head logit computation (TRACE core)
        #
        # 关键优化: 直接用 outputs.logits, 不重复调用 lm_head(hidden_states)
        # 避免在显存中同时存两份 (B, seq, vocab_size) 的 logits tensor
        # ================================================================
        text_logits = outputs.logits                             # (B, seq, vocab_size)
        del outputs  # 释放 hidden_states 缓存 (所有层) 和其他中间变量

        sync_logits = self.sync_head(hidden_states)             # (B, seq, 1)
        text_sync_logits = torch.cat([text_logits, sync_logits], dim=-1).float()  # (B, seq, vocab_size+1)
        del text_logits, sync_logits  # text_sync_logits 已包含两者

        time_logits = self.time_head(hidden_states).float()     # (B, seq, time_vocab_size)
        score_logits = self.score_head(hidden_states).float()   # (B, seq, score_vocab_size)
        del hidden_states  # 后续不再需要

        # ================================================================
        # Step 4: Loss computation — sum of three CE losses
        #
        # TRACE key difference from DisTime:
        #   DisTime: LM_loss + DFL_loss + DIoU_loss (continuous regression)
        #   TRACE:   text_CE + time_CE + score_CE   (discrete classification)
        # ================================================================
        loss = None
        text_loss = torch.tensor(0.0, device=device)
        time_loss = torch.tensor(0.0, device=device)
        score_loss = torch.tensor(0.0, device=device)

        if labels is not None:
            loss_fct = CrossEntropyLoss()  # ignore_index=-100 by default

            # Text + sync CE loss
            # labels: text positions have normal text token IDs,
            #         sync positions have vocab_size,
            #         time/score positions have IGNORE_INDEX
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

        # Full logits for return (concatenated for compatibility)
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
        pixel_values: Optional[torch.Tensor] = None,
        duration: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        **generate_kwargs
    ) -> dict:
        """
        Generate with TRACE head-switching state machine.

        Return format matches SmolVLMDisTime.generate():
            {generated_ids, generated_text, pred_times, ...}

        TRACE inference flow:
        1. Prefill: inject tower embeddings + process vision
        2. Autoregressive loop:
           - Mask logits to active head's vocab range
           - Sample/argmax next token
           - Check if token triggers head switch (via swap_tokens)
           - Build next embedding from appropriate tower
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
        # Step 1: Build inputs_embeds with tower routing
        # ================================================================
        embed_layer = self.base_model.get_input_embeddings()
        safe_ids = torch.clamp(input_ids, min=0, max=self.vocab_size - 1)
        inputs_embeds = embed_layer(safe_ids).clone()

        sync_mask = (input_ids == self.sync_token_id)
        time_mask = (input_ids >= self.time_start_id) & (input_ids < self.time_end_id)
        score_mask = (input_ids >= self.score_start_id) & (input_ids < self.score_end_id)

        if sync_mask.any():
            inputs_embeds[sync_mask] = self.sync_tower(input_ids[sync_mask])
        if time_mask.any():
            inputs_embeds[time_mask] = self.time_tower(input_ids[time_mask] - self.time_start_id)
        if score_mask.any():
            inputs_embeds[score_mask] = self.score_tower(input_ids[score_mask] - self.score_start_id)

        # Sanitize input_ids for base model
        model_input_ids = input_ids.clone()
        model_input_ids[input_ids >= self.vocab_size] = self.processor.tokenizer.pad_token_id or 0

        # ================================================================
        # Step 2: Prefill -> KV cache
        # ================================================================
        outputs = self.base_model(
            input_ids=model_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
        )

        past_key_values = outputs.past_key_values
        hidden_states = outputs.hidden_states[-1]

        # Get lm_head reference
        if hasattr(self.base_model, 'lm_head'):
            lm_head = self.base_model.lm_head
        else:
            lm_head = self.base_model.get_output_embeddings()

        # Initial full logits
        last_h = hidden_states[:, -1:, :]  # (B, 1, hidden)
        text_logits = lm_head(last_h).squeeze(1)           # (B, vocab_size)
        sync_logits = self.sync_head(last_h).squeeze(1)     # (B, 1)
        time_logits = self.time_head(last_h).squeeze(1)     # (B, time_vocab)
        score_logits = self.score_head(last_h).squeeze(1)   # (B, score_vocab)
        full_logits = torch.cat([text_logits, sync_logits, time_logits, score_logits], dim=-1)

        # ================================================================
        # Step 3: Autoregressive loop with head switching
        # ================================================================
        # 支持外部传入初始 heads (原始 TRACE eval 传 heads=[1] 从 time head 开始)
        initial_heads = generate_kwargs.pop('heads', None)
        if initial_heads is not None:
            heads = list(initial_heads) if not isinstance(initial_heads, list) else initial_heads
            # 确保长度匹配 batch_size
            if len(heads) < batch_size:
                heads = heads + [0] * (batch_size - len(heads))
            heads = heads[:batch_size]
        else:
            heads = [0] * batch_size  # 0=text, 1=time, 2=score
        # Vocab ranges for masking (start_inclusive, end_exclusive)
        vocab_ranges = [
            (0, self.vocab_size + 1),                              # head 0: text + sync
            (self.vocab_size + 1,
             self.vocab_size + 1 + self.trace_config.time_vocab_size),  # head 1: time
            (self.vocab_size + 1 + self.trace_config.time_vocab_size,
             self.vocab_size + 1 + self.trace_config.time_vocab_size
             + self.trace_config.score_vocab_size),                     # head 2: score
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
                next_tokens = torch.argmax(masked_logits, dim=-1)  # (B,)

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
                    # Text token -> LLM embedding
                    emb = embed_layer(next_tokens[b:b + 1])  # (1, hidden)
                elif token_id == self.sync_token_id:
                    # Sync token -> SyncTower
                    emb = self.sync_tower(torch.zeros(1, dtype=torch.long, device=device))  # (1, hidden)
                elif token_id < self.time_end_id:
                    # Time token -> TimeTower
                    local_id = torch.tensor([token_id - self.time_start_id], device=device)
                    emb = self.time_tower(local_id)  # (1, hidden)
                else:
                    # Score token -> ScoreTower
                    local_id = torch.tensor([token_id - self.score_start_id], device=device)
                    emb = self.score_tower(local_id)  # (1, hidden)
                next_embeds_list.append(emb)

            next_embeds = torch.stack(next_embeds_list, dim=0)  # (B, 1, hidden)

            # Extend attention mask
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=device)
            ], dim=1)

            # Forward step
            outputs = self.base_model(
                inputs_embeds=next_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True,
                return_dict=True,
                use_cache=True,
            )

            past_key_values = outputs.past_key_values
            hidden_states = outputs.hidden_states[-1]

            # Compute full logits for next step
            last_h = hidden_states[:, -1:, :]
            text_logits = lm_head(last_h).squeeze(1)
            sync_logits = self.sync_head(last_h).squeeze(1)
            time_logits = self.time_head(last_h).squeeze(1)
            score_logits = self.score_head(last_h).squeeze(1)
            full_logits = torch.cat([text_logits, sync_logits, time_logits, score_logits], dim=-1)

        # ================================================================
        # Step 4: Assemble results (same format as SmolVLMDisTime.generate)
        # ================================================================
        if generated_ids:
            gen_ids = torch.stack(generated_ids, dim=1)  # (B, gen_len)
            output_ids = torch.cat([input_ids, gen_ids], dim=1)
        else:
            gen_ids = torch.empty(batch_size, 0, dtype=torch.long, device=device)
            output_ids = input_ids

        # Decode text tokens — 只保留 text head 生成的 token, 跳过 time/score/sync
        text_only_ids = []
        for b in range(batch_size):
            ids = [t.item() for t in gen_ids[b] if t.item() < self.vocab_size]
            text_only_ids.append(torch.tensor(ids, dtype=torch.long, device=device))
        # pad to same length for batch_decode
        max_text_len = max(len(t) for t in text_only_ids) if text_only_ids else 0
        if max_text_len > 0:
            padded = torch.full((batch_size, max_text_len),
                                self.processor.tokenizer.pad_token_id or 0,
                                dtype=torch.long, device=device)
            for b in range(batch_size):
                padded[b, :len(text_only_ids[b])] = text_only_ids[b]
            generated_text = self.processor.tokenizer.batch_decode(
                padded, skip_special_tokens=True
            )
        else:
            generated_text = [''] * batch_size

        # 兼容旧逻辑: 也保留 full decode (用于 debug)
        text_gen_ids = gen_ids.clone()
        text_gen_ids[text_gen_ids >= self.vocab_size] = self.processor.tokenizer.pad_token_id or 0
        generated_text_raw = self.processor.tokenizer.batch_decode(
            text_gen_ids, skip_special_tokens=False
        )

        # Extract predicted times and scores
        pred_times_list = [None] * batch_size
        pred_times_normalized_list = [None] * batch_size
        pred_scores_list = [None] * batch_size

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
                times_tensor = torch.tensor(result['times'], device=device)  # (N, 2)
                pred_times_list[b] = times_tensor
                # Normalize by duration if provided
                if duration is not None:
                    d = duration[b].item() if duration.dim() > 0 else duration.item()
                    pred_times_normalized_list[b] = times_tensor / max(d, 1e-6)

            if result['scores']:
                pred_scores_list[b] = result['scores']  # list of floats

        return {
            'generated_ids': output_ids,
            'generated_text': generated_text,
            'pred_times': pred_times_list,
            'pred_times_normalized': pred_times_normalized_list,
            'pred_scores': pred_scores_list,
        }

    # ================================================================
    # Save / Load (same pattern as SmolVLMDisTime)
    # ================================================================

    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            dist.barrier()
            return

        os.makedirs(save_directory, exist_ok=True)

        # Save base model (with LoRA keys in state_dict for checkpoint compatibility)
        self.base_model.save_pretrained(save_directory)

        # Save TRACE modules
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

        logger.info(f"Saved TRACE modules + processor to {save_directory}")

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
        logger.info("TRACE Training Setup")
        logger.info("=" * 60)

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
            if 'embed_tokens' in name or ('embed' in name and 'lora' not in name
                                          and 'tower' not in name):
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

        # Summary
        trainable, total = self.get_trainable_parameters()
        logger.info("-" * 60)
        logger.info(f"Total: {trainable:,} / {total:,} trainable ({100 * trainable / total:.2f}%)")
        logger.info("=" * 60)
