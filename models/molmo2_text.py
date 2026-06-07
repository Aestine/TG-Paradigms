"""Molmo2 (4B / 8B) — Text Numeral paradigm.

Subclasses Molmo2DisTime to reuse its custom video vision-merge
(`_build_inputs_with_vision`) and inner-transformer forward, but computes only
the standard next-token LM loss (no time modules).

NOTE: Molmo2 uses a bespoke video path and an inner `transformer` + `lm_head`
rather than a high-level HF `.generate`. The `forward` here mirrors
molmo2_distime exactly (minus time decoding). `generate` is a simple greedy
loop (no KV cache) that is correct but unoptimized — validate / speed it up on
real weights before large-scale evaluation.
"""

from typing import Optional
import torch
from torch.nn import CrossEntropyLoss

from .molmo2_distime import Molmo2DisTime, DisTimeConfig
from .text_paradigm_base import TextParadigmMixin

TextConfig = DisTimeConfig


class Molmo2Text(TextParadigmMixin, Molmo2DisTime):
    """Molmo2 with plain-text timestamp generation."""

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_token_pooling: Optional[torch.Tensor] = None,
        video_grids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        device = input_ids.device

        # Vision merge (Molmo2 video path) — reused from Molmo2DisTime
        inputs_embeds = self._build_inputs_with_vision(
            input_ids,
            pixel_values_videos=pixel_values_videos,
            video_token_pooling=video_token_pooling,
            video_grids=video_grids,
        )

        transformer = self.base_model.model.transformer
        outputs = transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.base_model.lm_head(hidden_states)

        loss = torch.tensor(0.0, device=device)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            vocab_size = logits.shape[-1]
            loss = loss_fct(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1).to(shift_logits.device),
            )

        result = self._lm_return(loss, logits)
        return result if return_dict else loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_token_pooling: Optional[torch.Tensor] = None,
        video_grids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
        eos_token_id=None,
        **kwargs,
    ):
        """Autoregressive decoding with KV cache (batch-aware, multi-EOS).

        Mirrors molmo2_distime.generate's prefill + cached loop, but WITHOUT the
        <TIME_STAMP> decoding branch and WITHOUT FRAME_TIME injection (the Text
        paradigm has no time modules). Returns the generated token ids only
        (prompt excluded), shape (batch, gen_len).
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]
        transformer = self.base_model.model.transformer
        embed_layer = self.base_model.get_input_embeddings()

        # EOS handling (Qwen3 style: <|endoftext|> + <|im_end|>), same as distime
        default_eos = self.processor.tokenizer.eos_token_id
        im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != default_eos:
            default_eos_list = [default_eos, im_end_id]
        else:
            default_eos_list = [default_eos]
        eos_token_ids = eos_token_id if eos_token_id is not None else default_eos_list
        if not isinstance(eos_token_ids, list):
            eos_token_ids = [eos_token_ids]

        # Vision merge (no FRAME_TIME injection — no time encoder in Text paradigm)
        inputs_embeds = self._build_inputs_with_vision(
            input_ids,
            pixel_values_videos=pixel_values_videos,
            video_token_pooling=video_token_pooling,
            video_grids=video_grids,
        ).clone()

        # Prefill -> KV cache
        outputs = transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token_logits = self.base_model.lm_head(outputs.last_hidden_state[:, -1:, :]).squeeze(1)

        generated_ids = []
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            if do_sample and temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            generated_ids.append(next_tokens.clone())

            for eid in eos_token_ids:
                unfinished = unfinished & (next_tokens != eid)
            if not unfinished.any():
                break

            attention_mask = torch.cat(
                [attention_mask, torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=device)],
                dim=1,
            )
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
            next_token_logits = self.base_model.lm_head(outputs.last_hidden_state[:, -1:, :]).squeeze(1)

        if generated_ids:
            return torch.stack(generated_ids, dim=1)  # (batch, gen_len)
        return torch.empty((batch_size, 0), dtype=torch.long, device=device)
