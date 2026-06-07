"""FastVLM-1.5B — Text Numeral paradigm.

Subclasses FastVLMDisTime. FastVLM's base model enforces an XOR between
`input_ids` and `inputs_embeds`, so (mirroring fastvlm_distime) we pass
`input_ids=None` together with `inputs_embeds`.
"""

from typing import Optional
import torch

from .fastvlm_distime import FastVLMDisTime, DisTimeConfig
from .text_paradigm_base import TextParadigmMixin

TextConfig = DisTimeConfig


class FastVLMText(TextParadigmMixin, FastVLMDisTime):
    """FastVLM with plain-text timestamp generation."""

    def _embed(self, input_ids):
        if hasattr(self.base_model, 'get_input_embeddings'):
            return self.base_model.get_input_embeddings()(input_ids)
        return self.base_model.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_flags: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        inputs_embeds = self._embed(input_ids)
        outputs = self.base_model(
            input_ids=None,                 # FastVLM XOR constraint
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            return_dict=True,
        )
        loss = outputs.loss
        result = self._lm_return(loss, getattr(outputs, 'logits', None))
        return result if return_dict else loss

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
        self.eval()
        for k in ("image_flags", "time_gt", "num_events", "duration",
                  "frame_times", "position_ids"):
            generate_kwargs.pop(k, None)
        inputs_embeds = self._embed(input_ids)
        return self.base_model.generate(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **generate_kwargs,
        )
