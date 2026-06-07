#!/usr/bin/env python
"""Evaluate the Text Numeral paradigm on Charades-STA / QVHighlights / YouCook2.

The model generates plain text; timestamps are parsed with a "from X to Y
seconds" pattern (paper Appendix). Inference prompt templates are taken verbatim
from the paper's Appendix (Table `tab:prompts`).

Reuses the proven video/frame/prompt construction (`prepare_inference_inputs`)
and metric code (`compute_metrics`, `compute_iou`) from `evaluate_charades.py`.

Usage:
    python eval/evaluate_text.py \
        --task charades --model_type smolvlm \
        --model_name_or_path /path/to/SmolVLM2-2.2B-Instruct \
        --checkpoint_dir /path/to/text_checkpoint \
        --data_file /path/to/charades_sta_test.json \
        --video_folder /path/to/videos --num_frames 32

NOTE: integration-level implementation. Validate `generate()` and parsing on
real backbone weights; Molmo2 uses a simple greedy loop (see molmo2_text.py).
"""

import os
import re
import json
import argparse
import logging

import torch
from tqdm import tqdm

# Reuse the proven helpers from the DisTime evaluator
from evaluate_charades import (
    load_charades_data,
    prepare_inference_inputs,
    compute_metrics,
    get_eos_token_id,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Inference prompt templates — verbatim from the paper Appendix (Table tab:prompts)
# ---------------------------------------------------------------------------
TEXT_PROMPTS = {
    "charades": (
        "Localize the visual content described by the given textual query "
        "'{query}' in the video, and output the start and end timestamps in seconds."
    ),
    "qvh": (
        "Please find the highlight contents in the video described by a sentence "
        "query, determining the highlight timestamps and its saliency score on a "
        "scale from 1 to 5. Now I will give you the sentence query: '{query}'. "
        "Please return the query-based highlight timestamps and salient scores."
    ),
    "youcook2": (
        "Scrutinize the video and determine multiple occurrences, providing their "
        "initial and final timestamps as well as a summary of each action."
    ),
}

# Matches "from 52.0 to 63.0 seconds", "from 52 to 63 s", etc.
_TS_PATTERN = re.compile(
    r"from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*(?:seconds|second|secs|sec|s)\b",
    re.IGNORECASE,
)
# Fallback: any "X to Y"
_TS_FALLBACK = re.compile(r"(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)")


def parse_timestamps(text):
    """Extract all (start, end) pairs from generated text. Returns list of [s, e]."""
    spans = [[float(a), float(b)] for a, b in _TS_PATTERN.findall(text)]
    if not spans:
        spans = [[float(a), float(b)] for a, b in _TS_FALLBACK.findall(text)]
    return spans


def load_text_model(args):
    """Build a Text-paradigm model and load LoRA + embed/lm_head weights.

    Mirrors evaluate_charades.load_model but uses the *_text classes and skips
    the DisTime-module loading step.
    """
    mt = args.model_type
    if mt == "smolvlm":
        from models.smolvlm_text import SmolVLMText as ModelClass
    elif mt == "fastvlm":
        from models.fastvlm_text import FastVLMText as ModelClass
    elif mt == "molmo2":
        from models.molmo2_text import Molmo2Text as ModelClass
    else:
        raise ValueError(f"Unknown model_type: {mt}")

    model = ModelClass(
        model_name_or_path=args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        use_flash_attention=False,
    )
    model.setup_training(
        use_lora=True, lora_r=args.lora_r, lora_alpha=args.lora_alpha, freeze_vision=True,
    )

    if args.checkpoint_dir:
        import glob
        from safetensors import safe_open
        sf = os.path.join(args.checkpoint_dir, "model.safetensors")
        sf_idx = os.path.join(args.checkpoint_dir, "model.safetensors.index.json")
        state_dict = {}
        if os.path.exists(sf):
            with safe_open(sf, framework="pt") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
        elif os.path.exists(sf_idx):
            for shard in sorted(glob.glob(os.path.join(args.checkpoint_dir, "model-*.safetensors"))):
                with safe_open(shard, framework="pt") as f:
                    for k in f.keys():
                        state_dict[k] = f.get_tensor(k)
        else:
            pt = os.path.join(args.checkpoint_dir, "pytorch_model.bin")
            if os.path.exists(pt):
                state_dict = torch.load(pt, map_location="cpu")
            else:
                from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
                state_dict = get_fp32_state_dict_from_zero_checkpoint(args.checkpoint_dir)

        remapped = {}
        for k, v in state_dict.items():
            remapped[k if k.startswith("base_model.") else f"base_model.{k}"] = v
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        logger.info(f"Loaded checkpoint: {len(missing)} missing, {len(unexpected)} unexpected")

        # Re-tie lm_head if the checkpoint omitted it (tie_word_embeddings backbones)
        if "base_model.lm_head.weight" in missing:
            emb = model.base_model.get_input_embeddings().weight
            head = model.base_model.get_output_embeddings()
            if head is not None and emb.shape == head.weight.shape:
                head.weight = emb

    model.to("cuda").eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["charades", "qvh", "youcook2"])
    ap.add_argument("--model_type", required=True, choices=["smolvlm", "fastvlm", "molmo2"])
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--checkpoint_dir", default=None)
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--video_root", default="", help="folder containing the videos (joined with video_id + ext)")
    ap.add_argument("--num_frames", type=int, default=32)
    ap.add_argument("--image_size", type=int, default=384)
    ap.add_argument("--image_seq_len_override", type=int, default=None)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--output_file", default="text_predictions.json")
    args = ap.parse_args()

    if args.task in ("qvh", "youcook2"):
        logger.warning(
            "evaluate_text computes moment-retrieval metrics (R@1 / mIoU) only. "
            "QVHighlights highlight detection (mAP / HIT@1) is not applicable to the "
            "Text paradigm by design, and YouCook2 dense-captioning metrics "
            "(CIDEr / SODA_c / F1) require the dedicated scorer in evaluate_youcook2.py "
            "applied to the parsed multi-event text output."
        )

    model = load_text_model(args)
    processor = model.processor
    eos_ids = get_eos_token_id(processor, args.model_type)

    # load_charades_data returns {video_id, query, gt_start, gt_end}; it also
    # parses QVHighlights-style moment-retrieval and flat formats.
    data = load_charades_data(args.data_file)
    prompt_tpl = TEXT_PROMPTS[args.task]

    predictions = []
    errors = 0
    for item in tqdm(data, desc=f"text-eval/{args.task}"):
        video_id = item["video_id"]
        raw_query = item["query"]
        prompt = prompt_tpl.format(query=raw_query) if "{query}" in prompt_tpl else prompt_tpl

        # resolve video path (mirror evaluate_charades)
        video_path = None
        for ext in (".mp4", ".avi", ".mkv", ".webm", ""):
            cand = os.path.join(args.video_root, video_id + ext)
            if os.path.exists(cand):
                video_path = cand
                break

        pred_start, pred_end, text = 0.0, 0.0, ""
        if video_path is None:
            errors += 1
        else:
            try:
                inputs = prepare_inference_inputs(
                    processor=processor, video_path=video_path, query=prompt,
                    num_frames=args.num_frames, device="cuda",
                    image_size=args.image_size, model_type=args.model_type,
                    image_seq_len_override=args.image_seq_len_override,
                )
                duration = inputs.get("duration", None)
                gen_kwargs = dict(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=eos_ids,
                )
                if "pixel_values" in inputs:
                    gen_kwargs["pixel_values"] = inputs["pixel_values"]
                for k in ("pixel_values_videos", "video_token_pooling", "video_grids"):
                    if k in inputs:
                        gen_kwargs[k] = inputs[k]

                out_ids = model.generate(**gen_kwargs)
                text = processor.tokenizer.decode(out_ids[0], skip_special_tokens=True)
                spans = parse_timestamps(text)
                if spans:
                    pred_start, pred_end = spans[0]
                    if pred_start > pred_end:
                        pred_start, pred_end = pred_end, pred_start
                    if duration is not None:
                        pred_start = max(0.0, min(pred_start, float(duration)))
                        pred_end = max(0.0, min(pred_end, float(duration)))
            except Exception as e:
                logger.warning(f"{video_id}: {e}")
                errors += 1

        predictions.append({
            "video_id": video_id, "query": raw_query,
            "pred_start": pred_start, "pred_end": pred_end,
            "gt_start": item["gt_start"], "gt_end": item["gt_end"],
            "raw_text": text,
        })

    metrics = compute_metrics(predictions)
    logger.info(f"Metrics ({args.task}): {metrics}  (errors={errors}/{len(data)})")

    with open(args.output_file, "w") as f:
        json.dump({"metrics": metrics, "errors": errors, "predictions": predictions}, f, indent=2)
    logger.info(f"Saved -> {args.output_file}")


if __name__ == "__main__":
    main()
