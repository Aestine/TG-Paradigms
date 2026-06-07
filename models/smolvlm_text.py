"""SmolVLM2 — Text Numeral paradigm.

Subclasses SmolVLMDisTime to reuse base-model loading / embedding resize /
vision freezing, but uses ONLY the standard next-token LM loss (no time modules).
See models/text_paradigm_base.py for the shared logic.
"""

from typing import Optional
import torch

from .smolvlm_distime import SmolVLMDisTime, DisTimeConfig
from .text_paradigm_base import TextParadigmMixin

# Text paradigm reuses the same lightweight config object as DisTime
# (only `hidden_size`, lora settings etc. are read; time-specific fields unused).
TextConfig = DisTimeConfig


class SmolVLMText(TextParadigmMixin, SmolVLMDisTime):
    """SmolVLM2 with plain-text timestamp generation."""

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_flags: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,  # absorbs time_gt / num_events / duration / frame_times from collate
    ):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_flags=image_flags,
            return_dict=True,
        )
        loss = outputs.loss
        result = self._lm_return(loss, getattr(outputs, 'logits', None))
        return result if return_dict else loss
