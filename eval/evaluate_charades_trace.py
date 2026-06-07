"""
Charades-STA evaluation for TRACE models (SmolVLM-TRACE & FastVLM-TRACE).

Supports both model backends via --model_type flag.

Input construction is aligned with DisTime evaluate_charades.py:
  1. build_transform 处理图像 (model-specific normalization)
  2. 手动拼 frame prompt + conversation (与 dataset._build_frame_prompt / _build_conversation 一致)
  3. processor.tokenizer() 做 tokenization (不用 apply_chat_template)
  4. image_seq_len_override 支持 vision pooling (与 train.py 一致)

The TRACE paradigm generates multiple events per query, each with:
  caption + [start, end] timestamp + saliency score
For TVG (Temporal Video Grounding), we follow the original TRACE reformat_tvg.py
logic: extract the first valid (len==2) timestamp pair as the prediction.

Output files:
  - raw_trace_output.json: Raw TRACE format {video, id, timestamps, scores, captions}
  - tvg_predictions.json:  TVG-reformatted {qid: {query, timestamp, vid}}
  - predictions.json:      Full predictions with metrics info

Usage (SmolVLM-TRACE):
    python evaluate_charades_trace.py \
        --model_name_or_path /path/to/SmolVLM2-2.2B-Instruct \
        --model_type smolvlm \
        --checkpoint_dir /path/to/smolvlm_trace_checkpoint \
        --data_file /path/to/charades_sta_test.json \
        --video_root /path/to/Charades/videos \
        --num_frames 32

Usage (FastVLM-TRACE):
    python evaluate_charades_trace.py \
        --model_name_or_path /path/to/FastVLM-0.5B \
        --model_type fastvlm \
        --checkpoint_dir /path/to/fastvlm_trace_checkpoint \
        --data_file /path/to/charades_sta_test.json \
        --video_root /path/to/Charades/videos \
        --num_frames 32 \
        --vision_pool_stride 2
"""

import os
import sys
import json
import logging
import argparse
import tempfile
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

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

FRAME_TIME_TOKEN = "<FRAME_TIME>"

# 每种模型的默认 image_size (与 train.py 对齐)
MODEL_DEFAULT_IMAGE_SIZE = {
    "smolvlm": 384,    # SmolVLM (SigLIP vision encoder)
    "fastvlm": 1024,   # FastVLM (fastvit_mci3, crop_size=1024)
}

# 每种模型的默认 normalize 类型 (与 dataset.py 对齐)
MODEL_DEFAULT_NORMALIZE = {
    "smolvlm": "siglip",
    "fastvlm": "none",
}


# ============================================================================
# Data loading — auto-detect format
# ============================================================================

def load_charades_data(data_file):
    """
    Auto-detect and load Charades-STA annotations.
    Returns list of {"video_id", "query", "gt_start", "gt_end", "id"}.
    """
    with open(data_file, 'r') as f:
        raw = json.load(f)

    items = []

    # TRACE coco format
    if isinstance(raw, dict) and "annotations" in raw:
        logger.info("Detected TRACE caption_coco_format")
        for ann in raw["annotations"]:
            vid = ann["image_id"].replace(".mp4", "").split("/")[-1]
            ts = ann.get("timestamp", [0, 0])
            items.append({
                "video_id": vid, "query": ann["caption"],
                "gt_start": ts[0], "gt_end": ts[1],
                "id": ann.get("id", len(items)),
            })

    # DisTime database format
    elif isinstance(raw, dict) and "database" in raw:
        logger.info("Detected DisTime database format")
        for vid, info in raw["database"].items():
            for ann in info.get("annotations", []):
                seg = ann["segment"]
                items.append({
                    "video_id": vid, "query": ann["label"],
                    "gt_start": seg[0], "gt_end": seg[1],
                    "id": len(items),
                })

    # Flat list formats
    elif isinstance(raw, list) and len(raw) > 0:
        sample = raw[0]
        if "video_id" in sample and "sentence" in sample:
            logger.info("Detected flat format (video_id + sentence)")
            for i, item in enumerate(raw):
                items.append({
                    "video_id": item["video_id"], "query": item["sentence"],
                    "gt_start": item.get("start", 0), "gt_end": item.get("end", 0),
                    "id": item.get("id", i),
                })
        elif "video" in sample and "query" in sample:
            logger.info("Detected DisTime test_float format")
            for i, item in enumerate(raw):
                items.append({
                    "video_id": item["video"], "query": item["query"],
                    "gt_start": item.get("start", 0), "gt_end": item.get("end", 0),
                    "id": item.get("id", i),
                })
        else:
            raise ValueError(f"Unknown list format, keys: {list(sample.keys())}")
    else:
        raise ValueError(f"Cannot parse {data_file}")

    logger.info(f"Loaded {len(items)} annotations ({len(set(it['video_id'] for it in items))} videos)")
    return items


# ============================================================================
# Model loading — supports both SmolVLM-TRACE and FastVLM-TRACE
# ============================================================================

def load_model(args):
    """
    Load TRACE model for inference.
    Dynamically selects SmolVLMTrace or FastVLMTrace based on --model_type.
    """
    if args.model_type == "smolvlm":
        from models.smolvlm_trace import SmolVLMTrace, TraceConfig
        ModelClass = SmolVLMTrace
    elif args.model_type == "fastvlm":
        from models.fastvlm_trace import FastVLMTrace, TraceConfig
        ModelClass = FastVLMTrace
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    # Step 1: Create TraceConfig
    logger.info(f"Creating {args.model_type} TRACE model from {args.model_name_or_path}")
    trace_config_kwargs = {}
    if args.model_type == "fastvlm" and hasattr(args, 'vision_pool_stride'):
        trace_config_kwargs['vision_pool_stride'] = args.vision_pool_stride
    trace_config = TraceConfig(**trace_config_kwargs)

    # Step 2: Create model
    model_kwargs = {
        'model_name_or_path': args.model_name_or_path,
        'trace_config': trace_config,
        'torch_dtype': torch.bfloat16,
        'device_map': "cpu",
    }
    if args.model_type == "smolvlm":
        model_kwargs['use_flash_attention'] = False

    model = ModelClass(**model_kwargs)

    # Step 3: Apply LoRA structure (to match checkpoint keys)
    model.setup_training(
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        freeze_vision=True,
    )

    # Step 4: Load checkpoint weights
    if args.checkpoint_dir:
        logger.info(f"Loading checkpoint from {args.checkpoint_dir}")

        from safetensors import safe_open
        sf_path = os.path.join(args.checkpoint_dir, "model.safetensors")

        if os.path.exists(sf_path):
            state_dict = {}
            with safe_open(sf_path, framework="pt") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
            logger.info(f"Checkpoint has {len(state_dict)} keys")
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

        # Remap keys — TRACE uses base_model prefix
        remapped = {}
        for k, v in state_dict.items():
            if any(k.startswith(prefix) for prefix in [
                'time_tower.', 'score_tower.', 'sync_tower.',
                'time_head.', 'score_head.', 'sync_head.',
            ]):
                remapped[k] = v
            elif k.startswith('model.') or k.startswith('lm_head.'):
                remapped[f'base_model.{k}'] = v
            else:
                remapped[k] = v

        missing, unexpected = model.load_state_dict(remapped, strict=False)
        logger.info(f"Loaded: {len(missing)} missing, {len(unexpected)} unexpected")

        # Step 5: Load TRACE modules separately
        trace_path = os.path.join(args.checkpoint_dir, 'trace_modules.pt')
        if os.path.exists(trace_path):
            trace_state = torch.load(trace_path, map_location='cpu')
            model.time_tower.load_state_dict(trace_state['time_tower'])
            model.score_tower.load_state_dict(trace_state['score_tower'])
            model.sync_tower.load_state_dict(trace_state['sync_tower'])
            model.time_head.load_state_dict(trace_state['time_head'])
            model.score_head.load_state_dict(trace_state['score_head'])
            model.sync_head.load_state_dict(trace_state['sync_head'])
            for module in [model.time_tower, model.score_tower, model.sync_tower,
                           model.time_head, model.score_head, model.sync_head]:
                module.to(torch.bfloat16)
            logger.info(f"Loaded TRACE modules from {trace_path}")
        else:
            logger.error(f"trace_modules.pt NOT FOUND in {args.checkpoint_dir}")
            logger.error("TRACE towers/heads will use random init weights!")

        critical = [k for k in missing
                    if ('lora_' in k or 'tower' in k or 'head' in k)
                    and not os.path.exists(trace_path)]
        if critical:
            logger.error(f"CRITICAL missing ({len(critical)}):")
            for k in critical[:10]:
                logger.error(f"  {k}")
        else:
            logger.info("All LoRA and TRACE keys loaded successfully")

    model = model.to(args.device)
    model.eval()
    logger.info("Model loaded and ready")
    return model


# ============================================================================
# Input construction — matches training dataset exactly (与 evaluate_charades.py 对齐)
# ============================================================================

def build_transform(input_size=384, normalize_type='siglip'):
    """Build image transform – same as dataset.build_transform (eval mode)."""
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
    elif normalize_type == "none":
        pass  # FastVLM: no normalization
    else:
        raise ValueError(f"Unknown normalize_type: {normalize_type}")

    return transforms.Compose(transform_list)


def prepare_inference_inputs(processor, video_path, query, num_frames=32,
                             device="cuda", image_size=384, model_type="smolvlm",
                             image_seq_len_override=None, sync_token_id=None,
                             video_cache=None):
    """
    Prepare inputs for TRACE inference, matching training dataset's __getitem__ exactly.

    跟训练完全对齐:
    1. build_transform 处理图像 (与 dataset.py 一致)
    2. 手动拼 frame prompt + conversation (与 dataset._build_frame_prompt / _build_conversation 一致)
    3. processor.tokenizer() 做 tokenization (不用 apply_chat_template)
    4. image_seq_len_override 支持 vision pooling (与 train.py 一致)
    5. 末尾加 <sync> 并替换为 model.sync_token_id (与原始 TRACE eval 一致)

    Args:
        sync_token_id: 模型的 sync_token_id (model.sync_token_id)
        video_cache: dict, 缓存 {video_path: (pixel_values, num_frames, duration)}
    """
    from utils.mm_utils import load_video

    # 1. Load video (with cache)
    if video_cache is not None and video_path in video_cache:
        pixel_values, num_frames_actual, duration = video_cache[video_path]
    else:
        frames, frame_times, duration = load_video(video_path, num_frames=num_frames)
        pil_frames = [Image.fromarray(f.numpy()) for f in frames]
        num_frames_actual = len(frames)

        # 2. Process images with build_transform (same as training)
        normalize_type = MODEL_DEFAULT_NORMALIZE.get(model_type, "siglip")
        transform = build_transform(image_size, normalize_type=normalize_type)
        pixel_values = torch.stack([transform(f) for f in pil_frames])

        if video_cache is not None:
            video_cache[video_path] = (pixel_values, num_frames_actual, duration)

    # 3. Calculate image_seq_len (same as dataset.__init__)
    image_token = getattr(processor, 'image_token', '<image>')

    if model_type == "fastvlm":
        if hasattr(processor, 'patch_size') and processor.patch_size is not None:
            crop_h = image_size
            if hasattr(processor, 'image_processor') and hasattr(processor.image_processor, 'crop_size'):
                crop_h = processor.image_processor.crop_size.get('height', image_size)
            image_seq_len = (crop_h // processor.patch_size) ** 2
        else:
            image_seq_len = 256
        # Apply pooling override (same as train.py image_seq_len_override)
        if image_seq_len_override is not None:
            image_seq_len = image_seq_len_override
    else:
        # SmolVLM
        image_seq_len = getattr(processor, 'image_seq_len', 81)

    # 4. Build frame prompt (same as dataset._build_frame_prompt)
    if model_type == "smolvlm":
        fake_token = '<fake_token_around_image>'
        global_token = '<global-img>'
        frame_strings = []
        for i in range(pixel_values.shape[0]):
            img_tokens = image_token * image_seq_len
            frame_strings.append(f"{FRAME_TIME_TOKEN}: {fake_token}{global_token}{img_tokens}{fake_token}")
    elif model_type == "fastvlm":
        frame_strings = []
        for i in range(pixel_values.shape[0]):
            img_tokens = image_token * image_seq_len
            frame_strings.append(f"{FRAME_TIME_TOKEN}: {img_tokens}")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    frame_prompt = "\n".join(frame_strings)
    user_text = f"{frame_prompt}\n{query}"

    # 5. Build conversation with generation prompt (same as dataset._build_conversation)
    #    注意: TRACE 训练数据中 assistant 回复以 <sync> 开头,
    #    原始 TRACE eval 也在 prompt 末尾加 <sync> 并传 heads=[1] 从 time head 开始
    if model_type == "smolvlm":
        conversation = f"<|im_start|>User: {user_text}<end_of_utterance>\nAssistant:<sync>"
    elif model_type == "fastvlm":
        conversation = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n<sync>"

    # 6. Tokenize (same as training: processor.tokenizer, NOT apply_chat_template)
    encoded = processor.tokenizer(
        conversation, return_tensors="pt", add_special_tokens=True,
    )
    input_ids = encoded.input_ids

    # 7. 替换 <sync> placeholder → sync_token_id (extended vocab)
    #    tokenizer 把 <sync> 编码为 tok_sync_id, 需要替换为模型的 sync_token_id
    if sync_token_id is not None:
        tok_sync_id = processor.tokenizer.convert_tokens_to_ids("<sync>")
        input_ids = input_ids.clone()
        input_ids[input_ids == tok_sync_id] = sync_token_id

    return {
        'input_ids': input_ids.to(device),
        'attention_mask': encoded.attention_mask.to(device),
        'pixel_values': pixel_values.unsqueeze(0).to(device=device, dtype=torch.bfloat16),
        'duration': duration,
    }


# ============================================================================
# EOS token helper — 按 model_type 获取正确的 eos_token_id (与 evaluate_charades.py 对齐)
# ============================================================================

def get_eos_token_id(processor, model_type):
    """
    获取 eos_token_id, 按 model_type 区分:
    - SmolVLM: 使用 tokenizer 默认的 eos_token_id
      (与 smolvlm_trace.generate 默认值一致, SmolVLM 原生 EOS 不是 <|im_end|>)
    - FastVLM: 使用 [<|endoftext|>, <|im_end|>] 双 eos (与 fastvlm_trace.generate 对齐)
    """
    if model_type == "fastvlm":
        default_eos = processor.tokenizer.eos_token_id  # <|endoftext|>
        im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != default_eos:
            return [default_eos, im_end_id]
        return [default_eos]
    else:
        # SmolVLM: 用 tokenizer 默认 eos_token_id
        return processor.tokenizer.eos_token_id


# ============================================================================
# TRACE-specific: TVG reformatting (from reformat_tvg.py)
# ============================================================================

def reformat_tvg(pred_times, pred_scores, gen_text, video_id):
    """
    Reformat TRACE output for TVG (Temporal Video Grounding) evaluation.

    Following the original TRACE reformat_tvg.py logic:
    Extract the first valid timestamp pair (len==2) as the prediction.
    A valid pair requires both start and end to be non-NaN finite values.
    """
    if pred_times is None or pred_times.shape[0] == 0:
        return {}

    for j in range(pred_times.shape[0]):
        s = float(pred_times[j, 0].item())
        e = float(pred_times[j, 1].item())
        if not (np.isfinite(s) and np.isfinite(e)):
            continue
        return {
            "query": gen_text.strip() if gen_text else "",
            "timestamp": [round(s, 2), round(e, 2)],
            "vid": video_id,
        }

    return {}


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, gt):
    """Compute temporal IoU between two segments [start, end]."""
    inter_start = max(pred[0], gt[0])
    inter_end = min(pred[1], gt[1])
    inter = max(0, inter_end - inter_start)
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return inter / max(union, 1e-8)


def compute_metrics(predictions, thresholds=[0.3, 0.5, 0.7]):
    """Compute R@1 IoU=t for each threshold and mIoU."""
    n = len(predictions)
    if n == 0:
        return {}

    ious = []
    hits = {t: 0 for t in thresholds}

    for p in predictions:
        iou = compute_iou(
            [p["pred_start"], p["pred_end"]],
            [p["gt_start"], p["gt_end"]]
        )
        ious.append(iou)
        for t in thresholds:
            if iou >= t:
                hits[t] += 1

    metrics = {}
    for t in thresholds:
        metrics[f"R@1_IoU={t}"] = hits[t] / n * 100
    metrics["mIoU"] = np.mean(ious) * 100
    return metrics


# ============================================================================
# Worker for multi-GPU inference
# ============================================================================

def worker_inference(worker_id, gpu_id, worker_data, args, image_size, tmp_dir):
    """
    Worker process: load model onto assigned GPU, run inference on data subset,
    save results to a temp JSON file.
    """
    import torch
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

    wlog.info(f"Starting: {len(worker_data)} samples on GPU {gpu_id}")

    # Load model on this GPU
    args_copy = argparse.Namespace(**vars(args))
    args_copy.device = device
    model = load_model(args_copy)
    processor = model.processor

    sync_token_id = model.sync_token_id
    if "<sync>" not in processor.tokenizer.get_vocab():
        processor.tokenizer.add_special_tokens({
            'additional_special_tokens': list(
                getattr(processor.tokenizer, 'additional_special_tokens', [])
            ) + ["<sync>", "<time>", "<score>"]
        })

    if args.model_type == "fastvlm" and args.vision_pool_stride > 1:
        base_image_seq_len = (image_size // 64) ** 2
        image_seq_len_override = base_image_seq_len // (args.vision_pool_stride ** 2)
    else:
        image_seq_len_override = None

    eos_token_id = get_eos_token_id(processor, args.model_type)

    predictions = []
    raw_trace_outputs = []
    tvg_reformatted = {}
    errors = 0
    video_cache = {}

    for idx, item in enumerate(tqdm(worker_data, desc=f"W{worker_id}/GPU{gpu_id}",
                                     position=worker_id)):
        video_id = item['video_id']
        raw_query = item['query']
        item_id = item['id']
        query = args.query_template.format(query=raw_query)

        video_path = None
        for ext in ['.mp4', '.avi', '.mkv', '.webm', '']:
            candidate = os.path.join(args.video_root, video_id + ext)
            if os.path.exists(candidate):
                video_path = candidate
                break

        if video_path is None:
            errors += 1
            raw_trace_outputs.append({
                "video": video_id, "id": item_id,
                "timestamps": [], "scores": [], "captions": [],
            })
            predictions.append({
                "video_id": video_id, "query": raw_query,
                "pred_start": 0.0, "pred_end": 0.0,
                "gt_start": item["gt_start"], "gt_end": item["gt_end"],
                "pred_score": None,
            })
            tvg_reformatted[item_id] = {
                "query": raw_query, "timestamp": [0.0, 0.0], "vid": video_id,
            }
            continue

        try:
            inputs = prepare_inference_inputs(
                processor=processor, video_path=video_path,
                query=query, num_frames=args.num_frames,
                device=device, image_size=image_size,
                model_type=args.model_type,
                image_seq_len_override=image_seq_len_override,
                sync_token_id=sync_token_id,
                video_cache=video_cache,
            )
            duration = inputs['duration']

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    pixel_values=inputs['pixel_values'],
                    duration=torch.tensor([duration], device=device),
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=eos_token_id,
                    heads=[1],
                )

            pred_times = outputs.get('pred_times', [None])[0]
            pred_scores = outputs.get('pred_scores', [None])[0]
            gen_text = outputs.get('generated_text', [''])[0]

            raw_output = {
                "video": video_id, "id": item_id,
                "timestamps": [], "scores": [], "captions": [],
            }
            if pred_times is not None and pred_times.shape[0] > 0:
                for j in range(pred_times.shape[0]):
                    s = round(float(pred_times[j, 0].item()), 2)
                    e = round(float(pred_times[j, 1].item()), 2)
                    raw_output["timestamps"].append([s, e])
                if pred_scores and len(pred_scores) > 0:
                    for sc in pred_scores:
                        raw_output["scores"].append([float(sc)])
                if gen_text:
                    raw_output["captions"].append(gen_text.strip())
            raw_trace_outputs.append(raw_output)

            tvg_item = reformat_tvg(pred_times, pred_scores, gen_text, video_id)
            if tvg_item:
                tvg_reformatted[item_id] = tvg_item
            else:
                tvg_reformatted[item_id] = {
                    "query": raw_query, "timestamp": [0.0, duration], "vid": video_id,
                }

            if pred_times is not None and pred_times.shape[0] > 0:
                start = round(float(pred_times[0, 0].item()), 2)
                end = round(float(pred_times[0, 1].item()), 2)
                if start > end:
                    start, end = end, start
                start = max(0.0, min(start, duration))
                end = max(0.0, min(end, duration))
            else:
                start, end = 0.0, duration

            saliency = pred_scores[0] if pred_scores and len(pred_scores) > 0 else None

        except Exception as e:
            wlog.warning(f"Error {video_id}: {e}")
            start, end = 0.0, 30.0
            saliency = None
            raw_trace_outputs.append({
                "video": video_id, "id": item_id,
                "timestamps": [], "scores": [], "captions": [],
            })
            tvg_reformatted[item_id] = {
                "query": raw_query, "timestamp": [0.0, 30.0], "vid": video_id,
            }
            errors += 1

        predictions.append({
            "video_id": video_id, "query": raw_query,
            "pred_start": start, "pred_end": end,
            "gt_start": item["gt_start"], "gt_end": item["gt_end"],
            "pred_score": saliency,
        })

        if (idx + 1) % 20 == 0:
            torch.cuda.empty_cache()

    out_path = os.path.join(tmp_dir, f"worker_{worker_id}.json")
    with open(out_path, 'w') as f:
        json.dump({
            "predictions": predictions,
            "raw_trace_outputs": raw_trace_outputs,
            "tvg_reformatted": tvg_reformatted,
            "errors": errors,
        }, f, ensure_ascii=False)
    wlog.info(f"Done: {len(worker_data)} samples, {errors} errors -> {out_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="TRACE Charades-STA Evaluation")

    # Model
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to base VLM model")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["smolvlm", "fastvlm"],
                        help="VLM backend type")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Path to TRACE checkpoint directory")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--vision_pool_stride", type=int, default=1,
                        help="Vision token spatial pooling stride (FastVLM only). "
                             "1=no pooling, 2=4x compression, 4=16x compression")

    # Data
    parser.add_argument("--data_file", type=str, required=True,
                        help="Charades-STA test annotations (any supported format)")
    parser.add_argument("--video_root", type=str, required=True,
                        help="Root directory of video files")
    parser.add_argument("--num_frames", type=int, default=32,
                        help="Number of frames to sample per video")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max tokens to generate (TRACE needs more for char-level)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max number of samples to evaluate (for debugging)")

    # Output
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory. Default: ./results/Charades-STA-TRACE/{model_type}")

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    # Parallel inference
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs for parallel inference")
    parser.add_argument("--workers_per_gpu", type=int, default=1,
                        help="Number of parallel workers per GPU")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Shard index for cross-node parallel eval (0-indexed)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of shards for cross-node parallel eval")

    # Query template
    parser.add_argument("--query_template", type=str,
                        default="Give you a textual query: '{query}'. When does the described content occur in the video? Please return the timestamp in seconds.",
                        help="Query template. Use {query} placeholder.")

    args = parser.parse_args()

    # 默认按 model_type 分目录
    if args.out_dir is None:
        args.out_dir = f"./results/Charades-STA-TRACE/{args.model_type}"
    os.makedirs(args.out_dir, exist_ok=True)

    # 按 model_type 选择 image_size (与 train.py 对齐)
    image_size = MODEL_DEFAULT_IMAGE_SIZE[args.model_type]
    logger.info(f"Using model_type={args.model_type}, image_size={image_size}")

    # ================================================================
    # 2. Load data
    # ================================================================
    data = load_charades_data(args.data_file)
    if args.max_samples is not None:
        data = data[:args.max_samples]
        logger.info(f"Truncated to {len(data)} samples (--max_samples)")

    # Shard for cross-node parallel evaluation (SLURM array jobs)
    if args.num_shards > 1:
        total = len(data)
        shard_size = (total + args.num_shards - 1) // args.num_shards
        start_idx = args.shard_id * shard_size
        end_idx = min(start_idx + shard_size, total)
        data = data[start_idx:end_idx]
        logger.info(f"Shard {args.shard_id}/{args.num_shards}: samples [{start_idx}, {end_idx})")

    has_gt = any(it["gt_end"] > 0 for it in data)

    # ================================================================
    # 3. Run inference (single-process or multi-process)
    # ================================================================
    num_gpus = args.num_gpus
    total_workers = num_gpus * args.workers_per_gpu

    if total_workers <= 1:
        # ---- Single-process mode ----
        model = load_model(args)
        processor = model.processor

        sync_token_id = model.sync_token_id
        logger.info(f"Model sync_token_id={sync_token_id}")

        if "<sync>" not in processor.tokenizer.get_vocab():
            processor.tokenizer.add_special_tokens({
                'additional_special_tokens': list(
                    getattr(processor.tokenizer, 'additional_special_tokens', [])
                ) + ["<sync>", "<time>", "<score>"]
            })
            logger.info("Added TRACE special tokens to tokenizer")

        if args.model_type == "fastvlm" and args.vision_pool_stride > 1:
            base_image_seq_len = (image_size // 64) ** 2
            image_seq_len_override = base_image_seq_len // (args.vision_pool_stride ** 2)
        else:
            image_seq_len_override = None

        eos_token_id = get_eos_token_id(processor, args.model_type)

        predictions = []
        raw_trace_outputs = []
        tvg_reformatted = {}
        errors = 0
        video_cache = {}

        for idx in tqdm(range(len(data)), desc=f"Charades-STA ({args.model_type}-TRACE)"):
            item = data[idx]
            video_id = item['video_id']
            raw_query = item['query']
            item_id = item['id']

            query = args.query_template.format(query=raw_query)

            video_path = None
            for ext in ['.mp4', '.avi', '.mkv', '.webm', '']:
                candidate = os.path.join(args.video_root, video_id + ext)
                if os.path.exists(candidate):
                    video_path = candidate
                    break

            if video_path is None:
                logger.warning(f"[{idx}] Video not found: {video_id}")
                errors += 1
                raw_trace_outputs.append({
                    "video": video_id, "id": item_id,
                    "timestamps": [], "scores": [], "captions": [],
                })
                predictions.append({
                    "video_id": video_id, "query": raw_query,
                    "pred_start": 0.0, "pred_end": 0.0,
                    "gt_start": item["gt_start"], "gt_end": item["gt_end"],
                    "pred_score": None,
                })
                tvg_reformatted[item_id] = {
                    "query": raw_query, "timestamp": [0.0, 0.0], "vid": video_id,
                }
                continue

            try:
                inputs = prepare_inference_inputs(
                    processor=processor,
                    video_path=video_path,
                    query=query,
                    num_frames=args.num_frames,
                    device=args.device,
                    image_size=image_size,
                    model_type=args.model_type,
                    image_seq_len_override=image_seq_len_override,
                    sync_token_id=sync_token_id,
                    video_cache=video_cache,
                )
                duration = inputs['duration']

                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        pixel_values=inputs['pixel_values'],
                        duration=torch.tensor([duration], device=args.device),
                        max_new_tokens=args.max_new_tokens,
                        eos_token_id=eos_token_id,
                        heads=[1],
                    )

                pred_times = outputs.get('pred_times', [None])[0]
                pred_scores_out = outputs.get('pred_scores', [None])[0]
                gen_text = outputs.get('generated_text', [''])[0]

                raw_output = {
                    "video": video_id, "id": item_id,
                    "timestamps": [], "scores": [], "captions": [],
                }
                if pred_times is not None and pred_times.shape[0] > 0:
                    for j in range(pred_times.shape[0]):
                        s = round(float(pred_times[j, 0].item()), 2)
                        e = round(float(pred_times[j, 1].item()), 2)
                        raw_output["timestamps"].append([s, e])
                    if pred_scores_out and len(pred_scores_out) > 0:
                        for sc in pred_scores_out:
                            raw_output["scores"].append([float(sc)])
                    if gen_text:
                        raw_output["captions"].append(gen_text.strip())
                raw_trace_outputs.append(raw_output)

                tvg_item = reformat_tvg(pred_times, pred_scores_out, gen_text, video_id)
                if tvg_item:
                    tvg_reformatted[item_id] = tvg_item
                else:
                    tvg_reformatted[item_id] = {
                        "query": raw_query, "timestamp": [0.0, duration], "vid": video_id,
                    }

                if pred_times is not None and pred_times.shape[0] > 0:
                    start = round(float(pred_times[0, 0].item()), 2)
                    end = round(float(pred_times[0, 1].item()), 2)
                    if start > end:
                        start, end = end, start
                    start = max(0.0, min(start, duration))
                    end = max(0.0, min(end, duration))
                else:
                    start, end = 0.0, duration
                    logger.warning(f"[{idx}] {video_id} no timestamp predicted")

                saliency = pred_scores_out[0] if pred_scores_out and len(pred_scores_out) > 0 else None

                if idx < 5:
                    logger.info(f"[{idx}] {video_id}: pred=[{start:.2f}, {end:.2f}] "
                                f"gt=[{item['gt_start']:.2f}, {item['gt_end']:.2f}] "
                                f"score={saliency}")
                    logger.info(f"  text: {gen_text[:200]}")

            except Exception as e:
                logger.warning(f"[{idx}] Error {video_id}: {e}")
                import traceback
                traceback.print_exc()
                start, end = 0.0, 30.0
                saliency = None
                raw_trace_outputs.append({
                    "video": video_id, "id": item_id,
                    "timestamps": [], "scores": [], "captions": [],
                })
                tvg_reformatted[item_id] = {
                    "query": raw_query, "timestamp": [0.0, 30.0], "vid": video_id,
                }
                errors += 1

            predictions.append({
                "video_id": video_id, "query": raw_query,
                "pred_start": start, "pred_end": end,
                "gt_start": item["gt_start"], "gt_end": item["gt_end"],
                "pred_score": saliency,
            })

            if (idx + 1) % 100 == 0:
                logger.info(f"[{idx+1}/{len(data)}] errors={errors}")

    else:
        # ---- Multi-process mode ----
        mp.set_start_method('spawn', force=True)

        logger.info(f"Evaluating {len(data)} samples with {total_workers} workers "
                    f"({num_gpus} GPUs x {args.workers_per_gpu} workers/GPU)")

        # Round-robin split for load balance
        worker_data_splits = [[] for _ in range(total_workers)]
        for i, item in enumerate(data):
            worker_data_splits[i % total_workers].append(item)

        tmp_dir = tempfile.mkdtemp(prefix="charades_trace_eval_")
        logger.info(f"Worker results dir: {tmp_dir}")

        # Determine GPU IDs
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            gpu_ids = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        else:
            gpu_ids = list(range(num_gpus))

        processes = []
        for w in range(total_workers):
            gpu_id = gpu_ids[w // args.workers_per_gpu] if (w // args.workers_per_gpu) < len(gpu_ids) else w // args.workers_per_gpu
            p = mp.Process(
                target=worker_inference,
                args=(w, gpu_id, worker_data_splits[w], args, image_size, tmp_dir),
            )
            p.start()
            processes.append(p)
            logger.info(f"Spawned worker {w} on GPU {gpu_id} ({len(worker_data_splits[w])} samples)")

        for p in processes:
            p.join()

        for i, p in enumerate(processes):
            if p.exitcode != 0:
                logger.error(f"Worker {i} exited with code {p.exitcode}")

        # Merge results
        predictions = []
        raw_trace_outputs = []
        tvg_reformatted = {}
        errors = 0

        for w in range(total_workers):
            result_path = os.path.join(tmp_dir, f"worker_{w}.json")
            if not os.path.exists(result_path):
                logger.error(f"Worker {w} result file missing: {result_path}")
                continue
            with open(result_path, 'r') as f:
                worker_result = json.load(f)
            predictions.extend(worker_result["predictions"])
            raw_trace_outputs.extend(worker_result["raw_trace_outputs"])
            tvg_reformatted.update(worker_result["tvg_reformatted"])
            errors += worker_result["errors"]

        logger.info(f"Merged results: {len(predictions)} samples, {errors} errors")

    # ================================================================
    # 4. Compute metrics
    # ================================================================
    if has_gt:
        metrics = compute_metrics(predictions)
        print(f"\n{'='*60}")
        print(f"  Charades-STA Results — {args.model_type}-TRACE")
        print(f"  ({len(predictions)} samples)")
        print(f"{'='*60}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.2f}")
        print(f"  Errors: {errors}")
        print(f"{'='*60}\n")

        metrics_file = os.path.join(args.out_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")
    else:
        print("\nNo GT available, skipping metric computation.")

    # ================================================================
    # 5. Save predictions
    # ================================================================
    raw_file = os.path.join(args.out_dir, "raw_trace_output.json")
    with open(raw_file, 'w') as f:
        json.dump(raw_trace_outputs, f, indent=2)
    logger.info(f"Raw TRACE output saved to {raw_file}")

    tvg_file = os.path.join(args.out_dir, "tvg_predictions.json")
    with open(tvg_file, 'w') as f:
        json.dump(tvg_reformatted, f, indent=2)
    logger.info(f"TVG format predictions saved to {tvg_file}")

    pred_file = os.path.join(args.out_dir, "predictions.json")
    with open(pred_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"Predictions saved to {pred_file}")

    print(f"\nDone: {len(predictions)} predictions, {errors} errors")


if __name__ == "__main__":
    main()
