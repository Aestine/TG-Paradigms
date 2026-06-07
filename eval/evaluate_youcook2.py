"""
YouCook2 Dense Video Captioning (DVC) evaluation for DisTime (SmolVLM / FastVLM).

Supports both model backends via --model_type.

DVC task: Given a video, generate multiple (timestamp, caption) pairs.
DisTime outputs: <TIME_STAMP> → TimeDecoder → [start, end], then text caption until next <TIME_STAMP>.

Metrics: CIDEr, METEOR, SODA_c, F1 (event-level precision/recall with IoU threshold).

Data format (COCO-style):
  {
    "annotations": [
      {"image_id": "video.mp4", "caption": "...", "timestamp": [s, e], "duration": D},
      ...
    ]
  }

Usage:
    python evaluate_youcook2.py \
        --model_type smolvlm \
        --model_name_or_path /projects/bffz/yzou1/models/SmolVLM2-2.2B-Instruct \
        --checkpoint_dir /path/to/checkpoint \
        --anno_file /path/to/val.caption_coco_format.json \
        --video_root /path/to/videos \
        --num_frames 64
"""

import os
import sys
import json
import logging
import argparse
import re
import tempfile
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.multiprocessing as mp
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

FRAME_TIME_TOKEN = "<FRAME_TIME>"
TIME_STAMP_TOKEN = "<TIME_STAMP>"

MODEL_DEFAULT_IMAGE_SIZE = {
    "smolvlm": 384,
    "fastvlm": 1024,
    "molmo2": 378,
}

MODEL_DEFAULT_NORMALIZE = {
    "smolvlm": "siglip",
    "fastvlm": "none",
    "molmo2": "siglip",
}


# ============================================================================
# Data loading
# ============================================================================

def load_youcook2_data(anno_file):
    """
    Load YouCook2 annotations. Auto-detects format:

    Format 1 - TRACE COCO (annotations list with segments + pure_cap):
      {"annotations": [{"image_id": "vid.mp4", "segments": [[s,e],...], "pure_cap": "...", "duration": D}]}

    Format 2 - List with timestamps + captions:
      [{"video": "path/vid.mp4", "timestamps": [[s,e],...], "captions": ["..."], "duration": D}]

    Format 3 - Database format:
      {"database": {"vid": {"duration": D, "annotations": [{"segment": [s,e], "sentence": "..."}]}}}

    Returns:
        video_annotations: dict[video_id] = {
            "duration": float,
            "events": [{"caption": str, "start": float, "end": float}, ...]
        }
    """
    with open(anno_file, 'r') as f:
        raw = json.load(f)

    video_annotations = defaultdict(lambda: {"duration": 0, "events": []})

    if isinstance(raw, dict) and "annotations" in raw:
        sample = raw["annotations"][0] if raw["annotations"] else {}

        # TRACE COCO format: has "segments" + "pure_cap" fields
        if "segments" in sample and "pure_cap" in sample:
            logger.info("Detected TRACE COCO format (segments + pure_cap)")
            for ann in raw["annotations"]:
                vid = ann["image_id"].replace(".mp4", "").split("/")[-1]
                dur = ann.get("duration", 0)
                segments = ann.get("segments", [])
                pure_cap = ann.get("pure_cap", "")

                # pure_cap 是用 ". " 分隔的多个 caption
                captions = [c.strip() for c in pure_cap.split(". ") if c.strip()]
                # 去掉末尾可能残留的 "."
                captions = [c.rstrip(".") for c in captions]

                video_annotations[vid]["duration"] = max(video_annotations[vid]["duration"], dur)
                for i, seg in enumerate(segments):
                    cap = captions[i] if i < len(captions) else ""
                    video_annotations[vid]["events"].append({
                        "caption": cap,
                        "start": float(seg[0]),
                        "end": float(seg[1]),
                    })

        # Standard COCO format: has "timestamp" field with real values
        elif "timestamp" in sample:
            logger.info("Detected standard COCO caption format")
            for ann in raw["annotations"]:
                vid = ann["image_id"].replace(".mp4", "").split("/")[-1]
                ts = ann.get("timestamp", [0, 0])
                dur = ann.get("duration", 0)
                video_annotations[vid]["duration"] = max(video_annotations[vid]["duration"], dur)
                video_annotations[vid]["events"].append({
                    "caption": ann.get("pure_cap", ann.get("caption", "")),
                    "start": float(ts[0]),
                    "end": float(ts[1]),
                })
        else:
            raise ValueError(f"Unknown COCO annotation format, keys: {list(sample.keys())}")

    elif isinstance(raw, dict) and "database" in raw:
        logger.info("Detected database format")
        for vid, info in raw["database"].items():
            video_annotations[vid]["duration"] = info.get("duration", 0)
            for ann in info.get("annotations", []):
                seg = ann["segment"]
                video_annotations[vid]["events"].append({
                    "caption": ann.get("sentence", ann.get("label", "")),
                    "start": float(seg[0]),
                    "end": float(seg[1]),
                })

    elif isinstance(raw, list):
        sample = raw[0] if raw else {}

        # Format: {"video": "...", "timestamps": [[s,e],...], "captions": ["..."]}
        if "timestamps" in sample and "captions" in sample:
            logger.info("Detected list format (timestamps + captions)")
            for item in raw:
                vid = item.get("video", item.get("video_id", ""))
                vid = vid.replace(".mp4", "").split("/")[-1]
                dur = item.get("duration", 0)
                video_annotations[vid]["duration"] = max(video_annotations[vid]["duration"], dur)

                timestamps = item.get("timestamps", [])
                captions = item.get("captions", [])
                for i, cap in enumerate(captions):
                    t = timestamps[i] if i < len(timestamps) else [0, 0]
                    video_annotations[vid]["events"].append({
                        "caption": cap,
                        "start": float(t[0]),
                        "end": float(t[1]),
                    })
        else:
            # Generic list format
            logger.info("Detected generic list format")
            for item in raw:
                vid = item.get("video", item.get("video_id", ""))
                vid = vid.replace(".mp4", "").split("/")[-1]
                dur = item.get("duration", 0)
                video_annotations[vid]["duration"] = max(video_annotations[vid]["duration"], dur)

                times = item.get("times", item.get("timestamps", []))
                captions = item.get("captions", [])
                for i, cap in enumerate(captions):
                    t = times[i] if i < len(times) else [0, 0]
                    video_annotations[vid]["events"].append({
                        "caption": cap,
                        "start": float(t[0]),
                        "end": float(t[1]),
                    })
    else:
        raise ValueError(f"Unknown annotation format in {anno_file}")

    # Sort events by start time
    for vid in video_annotations:
        video_annotations[vid]["events"].sort(key=lambda e: e["start"])

    n_videos = len(video_annotations)
    n_events = sum(len(v["events"]) for v in video_annotations.values())
    logger.info(f"Loaded {n_events} events from {n_videos} videos")
    return dict(video_annotations)


# ============================================================================
# Model loading (same as evaluate_charades.py)
# ============================================================================

def load_model(args):
    """Load DisTime model for inference."""
    model_type = args.model_type

    if model_type == "smolvlm":
        from models.smolvlm_distime import SmolVLMDisTime, DisTimeConfig
        ModelClass = SmolVLMDisTime
    elif model_type == "fastvlm":
        from models.fastvlm_distime import FastVLMDisTime, DisTimeConfig
        ModelClass = FastVLMDisTime
    elif model_type == "molmo2":
        from models.molmo2_distime import Molmo2DisTime, DisTimeConfig
        ModelClass = Molmo2DisTime
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    logger.info(f"Creating {model_type} model from {args.model_name_or_path}")
    config_kwargs = dict(
        reg_max=args.reg_max,
        num_time_layers=args.num_time_layers,
    )
    if model_type == "fastvlm" and hasattr(args, 'vision_pool_stride'):
        config_kwargs['vision_pool_stride'] = args.vision_pool_stride

    distime_config = DisTimeConfig(**config_kwargs)

    model = ModelClass(
        model_name_or_path=args.model_name_or_path,
        distime_config=distime_config,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        use_flash_attention=False,
    )

    model.setup_training(
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        freeze_vision=True,
    )

    if args.checkpoint_dir:
        logger.info(f"Loading checkpoint from {args.checkpoint_dir}")

        import glob
        from safetensors import safe_open
        sf_path = os.path.join(args.checkpoint_dir, "model.safetensors")
        sf_index_path = os.path.join(args.checkpoint_dir, "model.safetensors.index.json")

        if os.path.exists(sf_path):
            state_dict = {}
            with safe_open(sf_path, framework="pt") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
            logger.info(f"Checkpoint has {len(state_dict)} keys (single safetensors)")
        elif os.path.exists(sf_index_path):
            shard_files = sorted(glob.glob(os.path.join(args.checkpoint_dir, "model-*.safetensors")))
            state_dict = {}
            for shard_file in shard_files:
                logger.info(f"Loading shard: {os.path.basename(shard_file)}")
                with safe_open(shard_file, framework="pt") as f:
                    for k in f.keys():
                        state_dict[k] = f.get_tensor(k)
            logger.info(f"Checkpoint has {len(state_dict)} keys ({len(shard_files)} shards)")
        else:
            pt_path = os.path.join(args.checkpoint_dir, "pytorch_model.bin")
            if os.path.exists(pt_path):
                state_dict = torch.load(pt_path, map_location="cpu")
            else:
                try:
                    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
                    state_dict = get_fp32_state_dict_from_zero_checkpoint(args.checkpoint_dir)
                except Exception as e:
                    raise FileNotFoundError(f"No loadable checkpoint in {args.checkpoint_dir}: {e}")

        remapped = {}
        for k, v in state_dict.items():
            if k.startswith('time_encoder.') or k.startswith('time_decoder.'):
                remapped[k] = v
            elif k.startswith('base_model.'):
                remapped[k] = v
            else:
                remapped[f'base_model.{k}'] = v

        missing, unexpected = model.load_state_dict(remapped, strict=False)
        logger.info(f"Loaded: {len(missing)} missing, {len(unexpected)} unexpected")

        # Re-tie embeddings if needed
        if 'base_model.lm_head.weight' in missing:
            embed_weight = model.base_model.get_input_embeddings().weight
            lm_head = model.base_model.get_output_embeddings()
            if lm_head is not None and embed_weight.shape == lm_head.weight.shape:
                lm_head.weight = embed_weight
                logger.info("Re-tied lm_head.weight → embed_tokens.weight")

        # Load DisTime modules
        distime_path = os.path.join(args.checkpoint_dir, 'distime_modules.pt')
        if os.path.exists(distime_path):
            distime_state = torch.load(distime_path, map_location='cpu')
            model.time_encoder.load_state_dict(distime_state['time_encoder'])
            model.time_decoder.load_state_dict(distime_state['time_decoder'])
            model.time_encoder.to(torch.bfloat16)
            model.time_decoder.to(torch.bfloat16)
            logger.info(f"Loaded DisTime modules from {distime_path}")
        else:
            logger.error(f"distime_modules.pt NOT FOUND in {args.checkpoint_dir}")

    device = args.device
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded (model_type={model_type})")
    return model


# ============================================================================
# Input preparation
# ============================================================================

def build_transform(input_size=384, normalize_type='siglip'):
    from torchvision import transforms
    SIGLIP_MEAN, SIGLIP_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    transform_list = [
        transforms.Resize(
            (input_size, input_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
    ]
    if normalize_type == "siglip":
        transform_list.append(transforms.Normalize(mean=SIGLIP_MEAN, std=SIGLIP_STD))
    return transforms.Compose(transform_list)


def get_eos_token_id(processor, model_type):
    if model_type in ("fastvlm", "molmo2"):
        # FastVLM (Qwen2) / Molmo2 (Qwen3): dual EOS [<|endoftext|>, <|im_end|>]
        default_eos = processor.tokenizer.eos_token_id
        im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != default_eos:
            return [default_eos, im_end_id]
        return [default_eos]
    else:
        # SmolVLM: 同时加 <end_of_utterance> 和 <|im_end|> 作为 stop token
        default_eos = processor.tokenizer.eos_token_id
        eou_id = processor.tokenizer.convert_tokens_to_ids("<end_of_utterance>")
        im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_ids = set()
        if default_eos is not None:
            eos_ids.add(default_eos)
        if eou_id is not None:
            eos_ids.add(eou_id)
        if im_end_id is not None:
            eos_ids.add(im_end_id)
        return list(eos_ids)


def _process_frames_molmo2(processor, frames, device="cuda"):
    """Process video frames using Molmo2's video_processor (same as dataset.py)."""
    from transformers.image_utils import SizeDict
    video_proc = processor.video_processor
    frames_np = [f.numpy() for f in frames]
    size = SizeDict(**video_proc.size) if isinstance(video_proc.size, dict) else video_proc.size
    video_inputs = video_proc._preprocess(
        videos=[frames_np],
        size=size,
        resample=video_proc.resample,
        image_mean=video_proc.image_mean,
        image_std=video_proc.image_std,
        patch_size=video_proc.patch_size,
        pooling_size=video_proc.pooling_size,
        return_tensors="pt",
    )
    return {
        'pixel_values_videos': video_inputs['pixel_values_videos'].to(device=device, dtype=torch.bfloat16),
        'video_token_pooling': video_inputs['video_token_pooling'].to(device),
        'video_grids': video_inputs['video_grids'].to(device),
    }


def _get_molmo2_image_token(processor):
    """Get Molmo2's image patch token string."""
    model_config = getattr(processor, 'config', None) or getattr(processor, 'image_processor', None)
    image_patch_id = None
    if model_config is not None:
        image_patch_id = getattr(model_config, 'image_patch_id', None)
    if image_patch_id is None:
        image_patch_id = 151938
    token = processor.tokenizer.convert_ids_to_tokens(image_patch_id)
    if token is None:
        token = f"<image_patch_{image_patch_id}>"
    return token, image_patch_id


def _get_molmo2_grid_size(processor):
    """Get Molmo2 video pooling grid size."""
    video_proc = getattr(processor, 'video_processor', None)
    if video_proc and hasattr(video_proc, 'pooling_size'):
        ps = video_proc.pooling_size
        raw_per_dim = 378 // 14
        return raw_per_dim // ps[0], raw_per_dim // ps[1]
    return 9, 9


def prepare_dvc_inputs(processor, video_path, num_frames=64,
                       device="cuda", image_size=384, model_type="smolvlm",
                       image_seq_len_override=None,
                       query_template="Describe the events in this video with their timestamps."):
    """Prepare inputs for DVC inference."""
    from utils.mm_utils import load_video

    frames, frame_times, duration = load_video(video_path, num_frames=num_frames)

    # Process images
    molmo2_video_inputs = None
    if model_type == "molmo2":
        molmo2_video_inputs = _process_frames_molmo2(processor, frames, device=device)
        pixel_values = None
    else:
        pil_frames = [Image.fromarray(f.numpy()) for f in frames]
        normalize_type = MODEL_DEFAULT_NORMALIZE.get(model_type, "siglip")
        transform = build_transform(image_size, normalize_type=normalize_type)
        pixel_values = torch.stack([transform(f) for f in pil_frames])

    # Calculate image_seq_len
    if model_type == "molmo2":
        image_token, _ = _get_molmo2_image_token(processor)
        grid_h, grid_w = _get_molmo2_grid_size(processor)
        image_seq_len = grid_h * grid_w
        if image_seq_len_override is not None:
            image_seq_len = image_seq_len_override
    elif model_type == "fastvlm":
        image_token = getattr(processor, 'image_token', '<image>')
        if hasattr(processor, 'patch_size') and processor.patch_size is not None:
            crop_h = image_size
            if hasattr(processor, 'image_processor') and hasattr(processor.image_processor, 'crop_size'):
                crop_h = processor.image_processor.crop_size.get('height', image_size)
            image_seq_len = (crop_h // processor.patch_size) ** 2
        else:
            image_seq_len = 256
        if image_seq_len_override is not None:
            image_seq_len = image_seq_len_override
    else:
        image_token = getattr(processor, 'image_token', '<image>')
        image_seq_len = getattr(processor, 'image_seq_len', 81)
        if image_seq_len_override is not None:
            image_seq_len = image_seq_len_override

    # Build frame prompt
    if model_type == "smolvlm":
        fake_token = '<fake_token_around_image>'
        global_token = '<global-img>'
        frame_strings = []
        for i in range(len(frames)):
            img_tokens = image_token * image_seq_len
            frame_strings.append(f"{FRAME_TIME_TOKEN}: {fake_token}{global_token}{img_tokens}{fake_token}")
    elif model_type == "fastvlm":
        frame_strings = []
        for i in range(len(frames)):
            img_tokens = image_token * image_seq_len
            frame_strings.append(f"{FRAME_TIME_TOKEN}: {img_tokens}")
    elif model_type == "molmo2":
        all_patch_tokens = image_token * image_seq_len
        frame_strings = []
        for i in range(len(frames)):
            frame_strings.append(f"{FRAME_TIME_TOKEN}: <frame_start>{all_patch_tokens}<frame_end>")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    frame_prompt = "\n".join(frame_strings)
    user_text = f"{frame_prompt}\n{query_template}"

    if model_type == "smolvlm":
        conversation = f"<|im_start|>User: {user_text}<end_of_utterance>\nAssistant:"
    elif model_type in ("fastvlm", "molmo2"):
        conversation = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"

    encoded = processor.tokenizer(
        conversation, return_tensors="pt", add_special_tokens=True,
    )

    result = {
        'input_ids': encoded.input_ids.to(device),
        'attention_mask': encoded.attention_mask.to(device),
        'frame_times': torch.tensor(frame_times, dtype=torch.float32).unsqueeze(0).to(device),
        'duration': duration,
    }

    if molmo2_video_inputs is not None:
        result.update(molmo2_video_inputs)
    else:
        result['pixel_values'] = pixel_values.unsqueeze(0).to(device=device, dtype=torch.bfloat16)

    return result


# ============================================================================
# Parse DVC predictions from model output
# ============================================================================

def parse_dvc_output(generated_text, pred_times, duration):
    """
    Parse DisTime output into event list for DVC evaluation.

    DisTime generates: text <TIME_STAMP> text <TIME_STAMP> text ...
    Each <TIME_STAMP> corresponds to a pred_times entry.

    For DVC, each (timestamp, caption_before_it) is one event.

    Returns:
        events: [{"caption": str, "start": float, "end": float}, ...]
    """
    events = []

    if pred_times is None or pred_times.shape[0] == 0:
        return events

    # Split generated text by TIME_STAMP token
    parts = generated_text.split(TIME_STAMP_TOKEN)

    for i in range(min(len(parts), pred_times.shape[0])):
        # Caption is the text BEFORE or AFTER this timestamp
        # Convention: caption comes AFTER the timestamp marker
        if i + 1 < len(parts):
            caption = parts[i + 1].strip()
        elif i < len(parts):
            caption = parts[i].strip()
        else:
            caption = ""

        # Clean caption: remove special tokens, leading punctuation, and extra whitespace
        caption = caption.replace("<|im_end|>", "").replace("<end_of_utterance>", "")
        caption = caption.replace("<|im_start|>", "").replace("<|endoftext|>", "")
        caption = re.sub(r'\s+', ' ', caption).strip()
        # Strip leading commas, periods, semicolons etc. that may remain after splitting
        caption = caption.lstrip('.,;: ').strip()

        if not caption:
            continue

        start = max(0.0, min(float(pred_times[i, 0].item()), duration))
        end = max(0.0, min(float(pred_times[i, 1].item()), duration))
        if start > end:
            start, end = end, start

        events.append({
            "caption": caption,
            "start": round(start, 2),
            "end": round(end, 2),
        })

    return events


# ============================================================================
# DVC Metrics
# ============================================================================

def compute_temporal_iou(pred, gt):
    """Compute temporal IoU between two [start, end] segments."""
    inter_start = max(pred[0], gt[0])
    inter_end = min(pred[1], gt[1])
    inter = max(0, inter_end - inter_start)
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return inter / max(union, 1e-8)


def compute_f1(pred_events, gt_events, iou_threshold=0.5):
    """
    Compute F1 score for event detection.
    An event is a true positive if IoU >= threshold AND caption is matched.

    For DVC, we use IoU-based matching (no caption matching for F1).
    """
    if len(pred_events) == 0 and len(gt_events) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_events) == 0:
        return 0.0, 0.0, 0.0
    if len(gt_events) == 0:
        return 0.0, 0.0, 0.0

    matched_gt = set()
    tp = 0

    for pred in pred_events:
        best_iou = 0
        best_gt_idx = -1
        for j, gt in enumerate(gt_events):
            if j in matched_gt:
                continue
            iou = compute_temporal_iou(
                [pred["start"], pred["end"]],
                [gt["start"], gt["end"]]
            )
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)

    precision = tp / len(pred_events) if len(pred_events) > 0 else 0
    recall = tp / len(gt_events) if len(gt_events) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def compute_avg_f1(all_pred_events, all_gt_events, tious=(0.3, 0.5, 0.7, 0.9)):
    """
    Compute average F1 across tIoU thresholds (standard DVC protocol).
    Returns: avg_precision, avg_recall, avg_f1, and per-threshold details.
    """
    per_tiou = {}
    for t in tious:
        precisions, recalls, f1s = [], [], []
        for vid in all_gt_events:
            p, r, f1 = compute_f1(
                all_pred_events.get(vid, []),
                all_gt_events[vid],
                iou_threshold=t,
            )
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
        per_tiou[t] = {
            "P": np.mean(precisions),
            "R": np.mean(recalls),
            "F1": np.mean(f1s),
        }

    avg_p = np.mean([v["P"] for v in per_tiou.values()])
    avg_r = np.mean([v["R"] for v in per_tiou.values()])
    avg_f1 = np.mean([v["F1"] for v in per_tiou.values()])

    return avg_p, avg_r, avg_f1, per_tiou


def compute_avg_cider(all_pred_events, all_gt_events, tious=(0.3, 0.5, 0.7, 0.9)):
    """
    Compute CIDEr averaged across tIoU thresholds (standard DVC protocol).
    At each tIoU, match pred→gt by IoU, then compute CIDEr on matched pairs.
    """
    try:
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    except ImportError:
        logger.warning("pycocoevalcap not installed, cannot compute CIDEr")
        return -1.0, {}

    per_tiou = {}
    for t in tious:
        eval_preds = {}
        eval_gts = {}
        pair_id = 0
        matched_count = 0

        for vid in all_gt_events:
            gt_events = all_gt_events[vid]
            pred_events = all_pred_events.get(vid, [])
            matched_gt_indices = set()

            for pred in pred_events:
                best_iou, best_j = 0, -1
                for j, gt in enumerate(gt_events):
                    if j in matched_gt_indices:
                        continue
                    iou = compute_temporal_iou(
                        [pred["start"], pred["end"]],
                        [gt["start"], gt["end"]]
                    )
                    if iou > best_iou:
                        best_iou, best_j = iou, j

                if best_iou >= t and best_j >= 0:
                    eval_preds[str(pair_id)] = [pred["caption"]]
                    eval_gts[str(pair_id)] = [gt_events[best_j]["caption"]]
                    matched_gt_indices.add(best_j)
                    matched_count += 1
                    pair_id += 1

            # 标准 DVC 评估 (densevid_eval) 对未匹配的 GT 用空 prediction 配对,
            # 这样未检测到的 GT event 贡献 CIDEr=0, 惩罚模型的漏检.
            # 这保证了 CIDEr 随 tIoU 递增而单调递减 (符合论文报告惯例).
            for j in range(len(gt_events)):
                if j not in matched_gt_indices:
                    eval_preds[str(pair_id)] = [""]
                    eval_gts[str(pair_id)] = [gt_events[j]["caption"]]
                    pair_id += 1

        if pair_id == 0:
            per_tiou[t] = 0.0
        else:
            # PTB tokenization: 小写化 + 标点分离, 与学术标准对齐
            # PTBTokenizer 要求 {id: [{"caption": str}]} 格式
            tokenizer = PTBTokenizer()
            ptb_gts = {k: [{"caption": c} for c in v] for k, v in eval_gts.items()}
            ptb_preds = {k: [{"caption": c} for c in v] for k, v in eval_preds.items()}
            eval_gts_tok = tokenizer.tokenize(ptb_gts)
            eval_preds_tok = tokenizer.tokenize(ptb_preds)

            cider_scorer = Cider()
            score, _ = cider_scorer.compute_score(eval_gts_tok, eval_preds_tok)
            per_tiou[t] = score

        logger.info(f"  CIDEr@tIoU={t}: {per_tiou[t]:.4f} "
                    f"({matched_count} matched / {pair_id} total)")

    avg_cider = np.mean(list(per_tiou.values()))
    return avg_cider, per_tiou


class NLTKMeteorWrapper:
    """
    Wraps nltk's METEOR to match pycocoevalcap's Meteor interface.
    This allows SODA to use METEOR without Java.
    """
    def __init__(self):
        import nltk
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        from nltk.translate.meteor_score import meteor_score
        self._meteor = meteor_score

    def compute_score(self, gts, res):
        """
        gts: {id: [ref_sentence, ...]}
        res: {id: [hyp_sentence]}
        Returns: (avg_score, per_item_scores)
        """
        scores = []
        for key in sorted(gts.keys()):
            refs = [r.split() for r in gts[key]]
            hyp = res[key][0].split()
            score = self._meteor(refs, hyp)
            scores.append(score)
        avg = np.mean(scores) if scores else 0.0
        return avg, scores

    def method(self):
        return "METEOR"


def compute_soda_c(all_pred_events, all_gt_events, soda_path=None):
    """
    Compute SODA_c using the official SODA tool (fujiso/SODA).
    Uses nltk METEOR as scorer (no Java needed).

    SODA expects:
      preds: {vid: [{"sentence": str, "timestamp": [s,e]}, ...]}
      gts:   [{vid: {"timestamps": [[s,e],...], "sentences": [str,...]}}]
      gt_vids: set of video ids
    """
    # ---- Step 1: Find and import SODA ----
    search_paths = []
    if soda_path:
        search_paths.append(soda_path)
    search_paths += [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "SODA"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval", "SODA"),
        os.path.expanduser("~/smolvlm_chat/eval/SODA"),
    ]

    soda_cls = None
    anet_cls = None
    soda_dir = None
    for p in search_paths:
        if not (p and os.path.isdir(p) and os.path.exists(os.path.join(p, "soda.py"))):
            continue
        soda_dir = p
        # Temporarily manipulate sys.path and module cache
        # to avoid conflict with project's utils/ package
        old_path = sys.path.copy()
        cached_utils = sys.modules.pop('utils', None)
        cached_soda = sys.modules.pop('soda', None)
        cached_dataset = sys.modules.pop('dataset', None)
        sys.path.insert(0, p)
        try:
            from soda import SODA as _S
            from dataset import ANETCaptions as _A
            soda_cls = _S
            anet_cls = _A
            logger.info(f"Loaded official SODA from {p}")
            break
        except Exception as e:
            logger.warning(f"Failed to import SODA from {p}: {e}")
            import traceback; traceback.print_exc()
        finally:
            sys.path = old_path
            if cached_utils is not None:
                sys.modules['utils'] = cached_utils
            if cached_soda is not None and soda_cls is None:
                sys.modules['soda'] = cached_soda
            if cached_dataset is not None and anet_cls is None:
                sys.modules['dataset'] = cached_dataset

    if soda_cls is None or anet_cls is None:
        logger.warning("Official SODA tool not found.")
        logger.warning("Install: git clone https://github.com/fujiso/SODA.git")
        return -1.0

    # ---- Step 2: Build data in SODA's expected format ----
    # preds: {vid: [{"sentence": str, "timestamp": [s,e]}, ...]}
    preds = {}
    for vid in all_gt_events:
        events = all_pred_events.get(vid, [])
        preds[vid] = sorted(
            [{"sentence": e["caption"], "timestamp": [e["start"], e["end"]]} for e in events],
            key=lambda x: x["timestamp"][0]
        )

    # gts: [{vid: {"timestamps": [...], "sentences": [...]}}]  (list of 1 reference)
    gt = {}
    gt_vids = set()
    for vid in all_gt_events:
        gt_vids.add(vid)
        events = sorted(all_gt_events[vid], key=lambda e: e["start"])
        gt[vid] = {
            "timestamps": [[e["start"], e["end"]] for e in events],
            "sentences": [e["caption"] for e in events],
        }

    # Only evaluate videos that appear in both pred and gt,
    # AND have at least one prediction (SODA crashes on empty pred lists)
    common_vids = gt_vids & set(preds.keys())
    n_before = len(common_vids)
    common_vids = {v for v in common_vids if len(preds.get(v, [])) > 0}
    n_skipped = n_before - len(common_vids)
    if n_skipped > 0:
        logger.warning(f"SODA: skipped {n_skipped} videos with 0 predictions")

    # ---- Step 3: Monkey-patch Meteor to use nltk (no Java) ----
    # Patch at EVERY level to ensure SODA's internal imports also get it:
    #   1) Module attribute: pycocoevalcap.meteor.meteor.Meteor
    #   2) Any already-imported references in SODA's scorer resolution
    _orig_meteor = None
    _patched_modules = []
    try:
        import pycocoevalcap.meteor.meteor as _meteor_mod
        _orig_meteor = getattr(_meteor_mod, 'Meteor', None)
        _meteor_mod.Meteor = NLTKMeteorWrapper
        _patched_modules.append(('pycocoevalcap.meteor.meteor', _meteor_mod, _orig_meteor))

        # Also patch pycocoevalcap.meteor if it re-exports Meteor
        try:
            import pycocoevalcap.meteor as _meteor_pkg
            if hasattr(_meteor_pkg, 'Meteor'):
                _patched_modules.append(('pycocoevalcap.meteor', _meteor_pkg,
                                         getattr(_meteor_pkg, 'Meteor')))
                _meteor_pkg.Meteor = NLTKMeteorWrapper
        except ImportError:
            pass

        # Patch the SODA module's scorer lookup — SODA may have already
        # imported Meteor at module level via `from pycocoevalcap.meteor.meteor import Meteor`
        # so we need to patch it directly in the soda module's namespace
        soda_module = sys.modules.get('soda')
        if soda_module and hasattr(soda_module, 'Meteor'):
            _patched_modules.append(('soda', soda_module,
                                     getattr(soda_module, 'Meteor')))
            soda_module.Meteor = NLTKMeteorWrapper

        logger.info("Patched pycocoevalcap.Meteor -> NLTKMeteorWrapper (no Java)")
    except ImportError:
        logger.warning("pycocoevalcap.meteor not found, SODA may fail")

    # Create ANETCaptions data object and preprocess
    data = anet_cls(preds, [gt], list(common_vids), verbose=False)
    data.preprocess()  # tokenize and transform preds/gts into {timestamps, sentences} format

    # ---- Step 4: Run SODA evaluation ----
    try:
        # Try passing NLTKMeteorWrapper instance directly first (bypasses
        # SODA's internal import). Fall back to string if SODA doesn't accept it.
        try:
            soda_evaluator = soda_cls(data, soda_type='c', scorer=NLTKMeteorWrapper())
        except (TypeError, Exception):
            soda_evaluator = soda_cls(data, soda_type='c', scorer='Meteor')
        scores = soda_evaluator.evaluate()

        if isinstance(scores, dict):
            for key, val in scores.items():
                if isinstance(val, (list, tuple)) and len(val) >= 3:
                    logger.info(f"SODA_c ({key}): P={val[0]:.4f} R={val[1]:.4f} F1={val[2]:.4f}")
                    return val[2]  # F1
        return 0.0
    except Exception as e:
        logger.error(f"SODA evaluation failed: {e}")
        import traceback; traceback.print_exc()
        return -1.0
    finally:
        # Restore all patched Meteor references
        for mod_name, mod_obj, orig_cls in _patched_modules:
            try:
                mod_obj.Meteor = orig_cls
            except Exception:
                pass


# ============================================================================
# Main
# ============================================================================

def find_video_path(video_root, vid):
    """Find video file given video_root and video id."""
    for ext in ['.mp4', '.avi', '.mkv', '.webm', '']:
        candidate = os.path.join(video_root, vid + ext)
        if os.path.exists(candidate):
            return candidate
        # Try with subdirectory (YouCook2 sometimes has subfolder structure)
        if os.path.isdir(video_root):
            for sub in os.listdir(video_root):
                subdir = os.path.join(video_root, sub)
                if os.path.isdir(subdir):
                    candidate = os.path.join(subdir, vid + ext)
                    if os.path.exists(candidate):
                        return candidate
    return None


def worker_inference(worker_id, gpu_id, video_ids, video_annotations, args,
                     image_size, tmp_dir):
    """
    Worker process: load model onto assigned GPU, run inference on video subset,
    save results to a temp JSON file.
    """
    # 确保子进程能看到 GPU
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(8))
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    wlog = logging.getLogger(f"worker-{worker_id}")
    wlog.setLevel(logging.INFO)
    if not wlog.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            f'%(asctime)s - W{worker_id}/GPU{gpu_id} - %(message)s'))
        wlog.addHandler(handler)

    wlog.info(f"Starting: {len(video_ids)} videos on GPU {gpu_id}")

    # Load model on this GPU
    args_copy = argparse.Namespace(**vars(args))
    args_copy.device = device
    model = load_model(args_copy)
    processor = model.processor

    # Auto-detect image_size from processor if not overridden
    if args.image_size is None and args.model_type in ("smolvlm", "molmo2"):
        detected = None
        if hasattr(processor, 'image_processor'):
            ip = processor.image_processor
            if hasattr(ip, 'size') and isinstance(ip.size, dict):
                detected = ip.size.get('height', ip.size.get('shortest_edge', None))
            elif hasattr(ip, 'crop_size') and isinstance(ip.crop_size, dict):
                detected = ip.crop_size.get('height', None)
        if detected and detected != image_size:
            wlog.info(f"Auto-detected image_size={detected} from processor "
                      f"(was {image_size})")
            image_size = detected

    # <<<< 计算 image_seq_len override
    # SmolVLM: CLI --image_seq_len 覆盖 (500M=64, 2.2B=81)
    if args.image_seq_len is not None:
        image_seq_len_override = args.image_seq_len
        wlog.info(f"Using CLI image_seq_len override: {image_seq_len_override}")
    elif args.model_type == "fastvlm" and args.vision_pool_stride > 1:
        base_image_seq_len = (image_size // 64) ** 2
        image_seq_len_override = base_image_seq_len // (args.vision_pool_stride ** 2)
    else:
        image_seq_len_override = None

    proc_isl = getattr(processor, 'image_seq_len', 'N/A')
    wlog.info(f"processor.image_seq_len={proc_isl}, override={image_seq_len_override}")

    eos_token_id = get_eos_token_id(processor, args.model_type)

    pred_events_dict = {}
    results_list = []
    errors = 0

    for idx, vid in enumerate(tqdm(video_ids, desc=f"W{worker_id}/GPU{gpu_id}",
                                    position=worker_id)):
        gt_info = video_annotations[vid]
        video_path = find_video_path(args.video_root, vid)

        if video_path is None:
            wlog.warning(f"Video not found: {vid}")
            errors += 1
            pred_events_dict[vid] = []
            continue

        try:
            inputs = prepare_dvc_inputs(
                processor=processor,
                video_path=video_path,
                num_frames=args.num_frames,
                device=device,
                image_size=image_size,
                model_type=args.model_type,
                image_seq_len_override=image_seq_len_override,
                query_template=args.query_template,
            )
            duration = inputs['duration']

            with torch.no_grad():
                gen_kwargs = dict(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    duration=torch.tensor([duration], device=device),
                    frame_times=inputs['frame_times'],
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=eos_token_id,
                )
                if args.model_type == "molmo2":
                    gen_kwargs['pixel_values_videos'] = inputs['pixel_values_videos']
                    gen_kwargs['video_token_pooling'] = inputs['video_token_pooling']
                    gen_kwargs['video_grids'] = inputs['video_grids']
                else:
                    gen_kwargs['pixel_values'] = inputs['pixel_values']
                outputs = model.generate(**gen_kwargs)

            generated_text = outputs.get('generated_text', [''])[0]
            pred_times = outputs.get('pred_times', [None])[0]

            pred_events = parse_dvc_output(generated_text, pred_times, duration)
            pred_events_dict[vid] = pred_events

            results_list.append({
                "video_id": vid,
                "duration": duration,
                "pred_events": pred_events,
                "gt_events": gt_info["events"],
                "generated_text": generated_text[:500],
                "num_pred_timestamps": pred_times.shape[0] if pred_times is not None else 0,
            })

        except Exception as e:
            wlog.warning(f"Error {vid}: {e}")
            pred_events_dict[vid] = []
            errors += 1

        # Free GPU memory periodically
        if (idx + 1) % 20 == 0:
            torch.cuda.empty_cache()

    # Save worker results to temp file
    out_path = os.path.join(tmp_dir, f"worker_{worker_id}.json")
    with open(out_path, 'w') as f:
        json.dump({
            "pred_events": pred_events_dict,
            "results": results_list,
            "errors": errors,
        }, f, ensure_ascii=False)
    wlog.info(f"Done: {len(video_ids)} videos, {errors} errors -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="DisTime YouCook2 DVC Evaluation")

    # Model
    parser.add_argument("--model_type", type=str, default="smolvlm",
                        choices=["smolvlm", "fastvlm", "molmo2"])
    parser.add_argument("--model_name_or_path", type=str, default=None,
                        help="Base model path (not required if --load_predictions is used)")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--reg_max", type=int, default=32)
    parser.add_argument("--num_time_layers", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--vision_pool_stride", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=None,
                        help="Override image size (default: auto-detect from processor, "
                             "384 for SmolVLM-2.2B, 512 for SmolVLM-0.5B)")
    parser.add_argument("--image_seq_len", type=int, default=None,
                        help="Override image_seq_len (SmolVLM). If None, read from processor. "
                             "SmolVLM-0.5B=64, SmolVLM-2.2B=81")

    # Data
    parser.add_argument("--anno_file", type=str, default=None,
                        help="YouCook2 annotations (COCO caption format). "
                             "Not required if --load_predictions is used")
    parser.add_argument("--video_root", type=str, default=None,
                        help="Not required if --load_predictions is used")
    parser.add_argument("--num_frames", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    # DVC specific
    parser.add_argument("--query_template", type=str,
                        default="Scrutinize the video and determine multiple occurrences, "
                                "providing their initial and final timestamps as well as "
                                "a summary of each action.",
                        help="DVC query template")
    parser.add_argument("--soda_path", type=str, default=None,
                        help="Path to fujiso/SODA repo (for official SODA_c)")

    # Parallel inference
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs to use for parallel inference")
    parser.add_argument("--workers_per_gpu", type=int, default=1,
                        help="Number of parallel workers per GPU")

    # Output
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load_predictions", type=str, default=None,
                        help="Path to predictions.json to skip inference and only recompute metrics")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Max number of videos to evaluate (for debugging)")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Shard index for parallel eval (0-indexed)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of shards for parallel eval")
    parser.add_argument("--flash_attn", action="store_true", default=True,
                        help="Use flash attention (default: True)")
    parser.add_argument("--no_flash_attn", action="store_true",
                        help="Disable flash attention")

    args = parser.parse_args()
    if args.no_flash_attn:
        args.flash_attn = False

    # Validate: --anno_file always required (for GT in metrics).
    # --model_name_or_path and --video_root only required for inference.
    if not args.anno_file:
        parser.error("--anno_file is always required (needed for GT annotations)")
    if not args.load_predictions:
        missing = []
        if not args.model_name_or_path:
            missing.append("--model_name_or_path")
        if not args.video_root:
            missing.append("--video_root")
        if missing:
            parser.error(f"The following arguments are required when not using "
                         f"--load_predictions: {', '.join(missing)}")

    if args.out_dir is None:
        args.out_dir = f"./results/YouCook2/{args.model_type}"
    os.makedirs(args.out_dir, exist_ok=True)

    # Determine image_size: user override > auto-detect from processor > fallback
    if args.image_size is not None:
        image_size = args.image_size
    else:
        image_size = MODEL_DEFAULT_IMAGE_SIZE[args.model_type]
    logger.info(f"model_type={args.model_type}, image_size={image_size} "
                f"(override={args.image_size is not None})")

    # ================================================================
    # 1. Load annotations
    # ================================================================
    video_annotations = load_youcook2_data(args.anno_file)
    video_ids = sorted(video_annotations.keys())
    if args.max_videos:
        video_ids = video_ids[:args.max_videos]

    # Shard for cross-node parallel evaluation (SLURM array jobs)
    if args.num_shards > 1:
        total = len(video_ids)
        shard_size = (total + args.num_shards - 1) // args.num_shards
        start = args.shard_id * shard_size
        end = min(start + shard_size, total)
        video_ids = video_ids[start:end]
        logger.info(f"Shard {args.shard_id}/{args.num_shards}: videos [{start}, {end})")

    # ================================================================
    # 快捷模式: 从已有的 predictions.json 直接重算 metrics
    # ================================================================
    if args.load_predictions:
        logger.info(f"Loading predictions from {args.load_predictions}")
        with open(args.load_predictions, 'r') as f:
            all_results = json.load(f)

        all_pred_events = {}
        all_gt_events = {}
        for r in all_results:
            vid = r["video_id"]
            all_pred_events[vid] = r["pred_events"]
        for vid in video_ids:
            all_gt_events[vid] = video_annotations[vid]["events"]

        # 只保留有 GT 的 video
        all_pred_events = {v: all_pred_events.get(v, []) for v in all_gt_events}

        logger.info(f"Loaded {len(all_pred_events)} videos, computing metrics...")
        errors = 0
        total_workers = 0
    else:
        total_workers = args.num_gpus * args.workers_per_gpu
        logger.info(f"Evaluating {len(video_ids)} videos with {total_workers} workers "
                    f"({args.num_gpus} GPUs x {args.workers_per_gpu} workers/GPU)")

    # ================================================================
    # 2. Run inference (single-process or multi-process)
    # ================================================================
    if args.load_predictions:
        pass  # 跳过推理
    elif total_workers <= 1:
        # ---- Single-process mode (backward compatible) ----
        model = load_model(args)
        processor = model.processor

        # Auto-detect image_size from processor if not overridden
        if args.image_size is None and args.model_type in ("smolvlm", "molmo2"):
            detected = None
            if hasattr(processor, 'image_processor'):
                ip = processor.image_processor
                if hasattr(ip, 'size') and isinstance(ip.size, dict):
                    detected = ip.size.get('height', ip.size.get('shortest_edge', None))
                elif hasattr(ip, 'crop_size') and isinstance(ip.crop_size, dict):
                    detected = ip.crop_size.get('height', None)
            if detected and detected != image_size:
                logger.info(f"Auto-detected image_size={detected} from processor "
                            f"(was {image_size})")
                image_size = detected

        # <<<< 计算 image_seq_len override
        if args.image_seq_len is not None:
            image_seq_len_override = args.image_seq_len
            logger.info(f"Using CLI image_seq_len override: {image_seq_len_override}")
        elif args.model_type == "fastvlm" and args.vision_pool_stride > 1:
            base_image_seq_len = (image_size // 64) ** 2
            image_seq_len_override = base_image_seq_len // (args.vision_pool_stride ** 2)
        else:
            image_seq_len_override = None

        proc_isl = getattr(processor, 'image_seq_len', 'N/A')
        logger.info(f"processor.image_seq_len={proc_isl}, override={image_seq_len_override}")

        eos_token_id = get_eos_token_id(processor, args.model_type)

        all_pred_events = {}
        all_gt_events = {}
        all_results = []
        errors = 0

        for idx, vid in enumerate(tqdm(video_ids, desc="YouCook2 DVC")):
            gt_info = video_annotations[vid]
            all_gt_events[vid] = gt_info["events"]

            video_path = find_video_path(args.video_root, vid)
            if video_path is None:
                logger.warning(f"[{idx}] Video not found: {vid}")
                errors += 1
                all_pred_events[vid] = []
                continue

            try:
                inputs = prepare_dvc_inputs(
                    processor=processor,
                    video_path=video_path,
                    num_frames=args.num_frames,
                    device=args.device,
                    image_size=image_size,
                    model_type=args.model_type,
                    image_seq_len_override=image_seq_len_override,
                    query_template=args.query_template,
                )
                duration = inputs['duration']

                with torch.no_grad():
                    gen_kwargs = dict(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        duration=torch.tensor([duration], device=args.device),
                        frame_times=inputs['frame_times'],
                        max_new_tokens=args.max_new_tokens,
                        eos_token_id=eos_token_id,
                    )
                    if args.model_type == "molmo2":
                        gen_kwargs['pixel_values_videos'] = inputs['pixel_values_videos']
                        gen_kwargs['video_token_pooling'] = inputs['video_token_pooling']
                        gen_kwargs['video_grids'] = inputs['video_grids']
                    else:
                        gen_kwargs['pixel_values'] = inputs['pixel_values']
                    outputs = model.generate(**gen_kwargs)

                generated_text = outputs.get('generated_text', [''])[0]
                pred_times = outputs.get('pred_times', [None])[0]

                pred_events = parse_dvc_output(generated_text, pred_times, duration)
                all_pred_events[vid] = pred_events

                result = {
                    "video_id": vid,
                    "duration": duration,
                    "pred_events": pred_events,
                    "gt_events": gt_info["events"],
                    "generated_text": generated_text[:500],
                    "num_pred_timestamps": pred_times.shape[0] if pred_times is not None else 0,
                }
                all_results.append(result)

                if idx < 5:
                    logger.info(f"[{idx}] {vid}: {len(pred_events)} pred events, "
                                f"{len(gt_info['events'])} gt events")
                    for e in pred_events[:3]:
                        logger.info(f"  pred: [{e['start']:.1f}, {e['end']:.1f}] {e['caption'][:80]}")
                    for e in gt_info["events"][:3]:
                        logger.info(f"  gt:   [{e['start']:.1f}, {e['end']:.1f}] {e['caption'][:80]}")

            except Exception as e:
                logger.warning(f"[{idx}] Error {vid}: {e}")
                import traceback
                traceback.print_exc()
                all_pred_events[vid] = []
                errors += 1

            if (idx + 1) % 50 == 0:
                logger.info(f"[{idx+1}/{len(video_ids)}] errors={errors}")

    else:
        # ---- Multi-process mode ----
        mp.set_start_method('spawn', force=True)

        # Split video_ids across workers (round-robin for load balance)
        worker_video_ids = [[] for _ in range(total_workers)]
        for i, vid in enumerate(video_ids):
            worker_video_ids[i % total_workers].append(vid)

        # Create temp dir for worker results
        tmp_dir = tempfile.mkdtemp(prefix="youcook2_eval_")
        logger.info(f"Worker results dir: {tmp_dir}")

        # Spawn workers
        processes = []
        for w in range(total_workers):
            gpu_id = w // args.workers_per_gpu  # worker 0,1 -> GPU 0; worker 2,3 -> GPU 1; ...
            p = mp.Process(
                target=worker_inference,
                args=(w, gpu_id, worker_video_ids[w], video_annotations,
                      args, image_size, tmp_dir),
            )
            p.start()
            processes.append(p)
            logger.info(f"Spawned worker {w} on GPU {gpu_id} ({len(worker_video_ids[w])} videos)")

        # Wait for all workers
        for p in processes:
            p.join()

        # Check for failed workers
        for i, p in enumerate(processes):
            if p.exitcode != 0:
                logger.error(f"Worker {i} exited with code {p.exitcode}")

        # Merge results from all workers
        all_pred_events = {}
        all_gt_events = {}
        all_results = []
        errors = 0

        for w in range(total_workers):
            result_path = os.path.join(tmp_dir, f"worker_{w}.json")
            if not os.path.exists(result_path):
                logger.error(f"Worker {w} result file missing: {result_path}")
                continue
            with open(result_path, 'r') as f:
                worker_data = json.load(f)
            all_pred_events.update(worker_data["pred_events"])
            all_results.extend(worker_data["results"])
            errors += worker_data["errors"]

        # Fill gt events
        for vid in video_ids:
            all_gt_events[vid] = video_annotations[vid]["events"]

        logger.info(f"Merged results: {len(all_pred_events)} videos, {errors} errors")

    # ================================================================
    # 3. Compute metrics (standard DVC protocol)
    #    All metrics averaged across tIoU = {0.3, 0.5, 0.7, 0.9}
    # ================================================================
    logger.info("Computing metrics...")
    tious = (0.3, 0.5, 0.7, 0.9)

    # F1 (averaged across tIoU thresholds)
    avg_p, avg_r, avg_f1, f1_per_tiou = compute_avg_f1(
        all_pred_events, all_gt_events, tious=tious)
    logger.info(f"F1: P={avg_p:.4f} R={avg_r:.4f} F1={avg_f1:.4f}")
    for t, v in f1_per_tiou.items():
        logger.info(f"  F1@tIoU={t}: P={v['P']:.4f} R={v['R']:.4f} F1={v['F1']:.4f}")

    # CIDEr (averaged across tIoU thresholds)
    avg_cider, cider_per_tiou = compute_avg_cider(
        all_pred_events, all_gt_events, tious=tious)
    logger.info(f"CIDEr (avg): {avg_cider:.4f}")

    # SODA_c (official tool)
    soda = compute_soda_c(
        all_pred_events, all_gt_events,
        soda_path=args.soda_path,
    )

    # Aggregate
    metrics = {
        "F1": round(avg_f1 * 100, 2),
        "Precision": round(avg_p * 100, 2),
        "Recall": round(avg_r * 100, 2),
        "CIDEr": round(avg_cider * 100, 2),
        "SODA_c": round(soda * 100, 2) if soda >= 0 else "N/A (install fujiso/SODA)",
    }
    # Per-threshold details
    for t in tious:
        metrics[f"F1@{t}"] = round(f1_per_tiou[t]["F1"] * 100, 2)
    for t in tious:
        cval = cider_per_tiou.get(t, -1)
        metrics[f"CIDEr@{t}"] = round(cval * 100, 2) if cval >= 0 else "N/A"
    metrics.update({
        "num_videos": len(video_ids),
        "num_errors": errors,
        "num_workers": total_workers,
        "avg_pred_events": round(np.mean([
            len(all_pred_events.get(vid, [])) for vid in video_ids
        ]), 2),
        "avg_gt_events": round(np.mean([
            len(all_gt_events.get(vid, [])) for vid in video_ids
        ]), 2),
    })

    # Print results
    print(f"\n{'='*60}")
    print(f"  YouCook2 DVC Results ({len(video_ids)} videos, model_type={args.model_type})")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")

    # ================================================================
    # 4. Save results
    # ================================================================
    metrics_file = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")

    pred_file = os.path.join(args.out_dir, "predictions.json")
    with open(pred_file, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Predictions saved to {pred_file}")

    # Save in COCO eval format (for external evaluation tools)
    coco_pred = {"annotations": []}
    ann_id = 0
    for vid, events in all_pred_events.items():
        for e in events:
            coco_pred["annotations"].append({
                "image_id": vid,
                "caption": e["caption"],
                "timestamp": [e["start"], e["end"]],
                "id": ann_id,
            })
            ann_id += 1
    coco_file = os.path.join(args.out_dir, "coco_format_predictions.json")
    with open(coco_file, 'w') as f:
        json.dump(coco_pred, f, indent=2, ensure_ascii=False)
    logger.info(f"COCO format predictions saved to {coco_file}")

    print(f"\nDone: {len(video_ids)} videos, {errors} errors")


if __name__ == "__main__":
    main()
