"""Shared mixin for the Text Numeral output paradigm.

The Text Numeral paradigm (VTimeLLM-style) is the simplest of the three studied
in the paper: timestamps are emitted as plain-text numerals using the LLM's
native vocabulary and language-modeling head, with **no extra output modules**
and the standard next-token cross-entropy loss (see paper Appendix:
"The Text paradigm adds no modules and uses the standard next-token loss.").

Implementation strategy
------------------------
Each backbone already has a fully-working DisTime wrapper (`*_distime.py`) that
loads the base VLM, resizes embeddings, freezes the vision encoder and applies
LoRA. The Text models reuse all of that by subclassing the DisTime class and:

  * `_init_distime_modules`  -> creates NO TimeEncoder/TimeDecoder (paper: 0 extra params)
  * `setup_training`         -> same LoRA setup, minus the time-module step
  * `forward`                -> backbone-specific, returns ONLY the LM loss
  * `generate`               -> plain autoregressive text generation

The frame prompt (and thus the visual input) is identical to the other
paradigms, so the controlled-comparison assumption holds. `<TIME_STAMP>` is
never used in Text targets; only `<FRAME_TIME>` appears (as in the other
paradigms) and is treated as an ordinary input marker.

NOTE: verified at the integration / data-construction level. Full training and
`generate()` should be validated on real backbone weights and GPUs; per-backbone
generation quirks (especially Molmo2's custom video path) may need adjustment.
"""

from typing import Optional
import torch


class TextParadigmMixin:
    """Mixin overriding the DisTime-specific parts for the Text Numeral paradigm."""

    def _init_distime_modules(self):
        # Text Numeral paradigm: no temporal output modules at all.
        self.time_encoder = None
        self.time_decoder = None

    def setup_training(
        self,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules=None,
        freeze_vision: bool = True,
    ):
        """Freeze base, (optionally) freeze vision, apply LoRA, unfreeze embed/lm_head.

        Identical to the DisTime version except it does NOT enable any time
        modules (there are none in this paradigm).
        """
        import logging
        logger = logging.getLogger(__name__)

        # Step 1: freeze everything
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Step 2: freeze vision encoder
        if freeze_vision:
            self.freeze_vision_encoder()

        # Step 3: apply manual LoRA (no PeftModel wrapper, matching DisTime/TRACE)
        if use_lora:
            from .manual_lora import apply_lora_to_model
            if lora_target_modules is None:
                lora_target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ]
            num_replaced = apply_lora_to_model(
                model=self.base_model,
                target_module_names=lora_target_modules,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                exclude_patterns=["vision_model"],
            )
            logger.info(f"[Text] Applied LoRA r={lora_r}, alpha={lora_alpha}, "
                        f"replaced {num_replaced} modules")

        # Step 4: unfreeze embed_tokens + lm_head
        for name, param in self.base_model.named_parameters():
            if any(x in name for x in ["embed_tokens", "lm_head"]):
                param.requires_grad = True

        # Step 5: enable input require grads (needed with gradient checkpointing)
        def _enable_input_require_grads(module, inp, out):
            out.requires_grad_(True)

        if hasattr(self.base_model, 'get_input_embeddings'):
            embed = self.base_model.get_input_embeddings()
        elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'embed_tokens'):
            embed = self.base_model.model.embed_tokens
        else:
            embed = None
        if embed is not None:
            embed.register_forward_hook(_enable_input_require_grads)

        # (No Step 6 — there are no time modules in the Text paradigm.)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"[Text] trainable params: {trainable:,} / {total:,} "
                    f"({100 * trainable / max(total, 1):.4f}%)")

    @staticmethod
    def _lm_return(loss, logits=None):
        """Uniform return dict, compatible with the DisTimeTrainer logging keys."""
        zero = loss.detach() * 0 if loss is not None else None
        return {
            'loss': loss,
            'lm_loss': loss.detach() if loss is not None else None,
            'time_loss': zero,
            'iou_loss': zero,
            'logits': logits,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        **generate_kwargs,
    ):
        """Plain text generation via the base VLM's own `.generate`.

        Returns ONLY the newly generated token ids (prompt stripped), shape
        (batch, gen_len) — consistent with the FastVLM / Molmo2 text models so
        the evaluator can decode and parse uniformly. Backbones whose base
        `.generate` needs `inputs_embeds` instead of `input_ids` (e.g. FastVLM)
        override this method.
        """
        self.eval()
        # strip kwargs the base model does not accept
        for k in ("image_flags", "time_gt", "num_events", "duration",
                  "frame_times", "position_ids"):
            generate_kwargs.pop(k, None)
        input_len = input_ids.shape[1]
        out = self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **generate_kwargs,
        )
        # HF .generate returns prompt+generated when input_ids are given -> strip prompt
        return out[:, input_len:]
