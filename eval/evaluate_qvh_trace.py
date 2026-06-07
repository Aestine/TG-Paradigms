"""
QVHighlights evaluation for TRACE models (SmolVLM-TRACE & FastVLM-TRACE).

Supports both model backends via --model_type flag.

Evaluation includes:
  - Moment Retrieval (MR): R@1, mAP, mIoU
  - Highlight Detection (HL): mAP, HIT@1 using clip-level saliency scores

The TRACE paradigm generates multiple events per query, each with:
  caption + [start, end] timestamp + saliency score
For highlight detection, event-level scores are mapped to clip-level (2-second)
saliency scores following the original TRACE format_vhd_output logic.

Reads your COCO-format annotations:
  {"annotations": [
    {"image_id": "VID.mp4", "id": 2579, "caption": "...",
     "timestamp": [[82, 150]], "duration": 150,
     "saliency_scores": [[2,3,3], [3,4,3], ...]},
    ...
  ]}

Usage (SmolVLM-TRACE):
    python evaluate_qvh_trace.py \
        --model_name_or_path /path/to/SmolVLM2-2.2B-Instruct \
        --model_type smolvlm \
        --checkpoint_dir /path/to/smolvlm_trace_checkpoint \
        --data_file /path/to/qvhighlights/val.json \
        --video_root /path/to/qvhighlights/videos \
        --num_frames 32

Usage (FastVLM-TRACE):
    python evaluate_qvh_trace.py \
        --model_name_or_path /path/to/FastVLM-0.5B \
        --model_type fastvlm \
        --checkpoint_dir /path/to/fastvlm_trace_checkpoint \
        --data_file /path/to/qvhighlights/val.json \
        --video_root /path/to/qvhighlights/videos \
        --num_frames 32 \
        --vision_pool_stride 2
"""

import os
import sys
import json
import logging
import argparse
import tempfile
from collections import OrderedDict, defaultdict
from functools import partial
from tqdm import tqdm
import multiprocessing as mp

import torch
import torch.multiprocessing as torch_mp
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
# Data loading
# ============================================================================

def load_qvh_data(data_file):
    """
    Load QVHighlights annotations.

    Supported formats:
      1) COCO format:
         {"annotations": [{"image_id":"VID.mp4", "id":N, "caption":"...",
                           "timestamp":[[s,e],...], "duration":D,
                           "saliency_scores":[[s1,s2,s3],...]}, ...]}
      2) DisTime's val.json:
         [{"video":"VID", "qid":N, "query":"...", "relevant_windows":[[s,e],...], "duration":D}, ...]

    Returns list of:
      {"qid", "video_id", "query", "relevant_windows", "duration", "saliency_scores"}
    """
    with open(data_file, 'r') as f:
        raw = json.load(f)

    items = []

    # Format 1: COCO annotations
    if isinstance(raw, dict) and "annotations" in raw:
        logger.info("Detected COCO annotation format")
        for ann in raw["annotations"]:
            vid = ann["image_id"]
            items.append({
                "qid": ann["id"],
                "video_id": vid,
                "query": ann["caption"],
                "relevant_windows": ann["timestamp"],
                "duration": ann["duration"],
                "saliency_scores": ann.get("saliency_scores", None),
                "relevant_clip_ids": ann.get("relevant_clip_ids", None),
            })

    # Format 2: DisTime flat list
    elif isinstance(raw, list) and len(raw) > 0:
        sample = raw[0]
        if "qid" in sample or "query" in sample:
            logger.info("Detected DisTime QVH format")
            for item in raw:
                vid = item.get("video", item.get("image_id", ""))
                if not vid.endswith(".mp4"):
                    vid = vid + ".mp4"
                items.append({
                    "qid": item.get("qid", item.get("id", 0)),
                    "video_id": vid,
                    "query": item.get("query", item.get("caption", "")),
                    "relevant_windows": item.get("relevant_windows", item.get("timestamp", [])),
                    "duration": item.get("duration", 150),
                    "saliency_scores": item.get("saliency_scores", None),
                    "relevant_clip_ids": item.get("relevant_clip_ids", None),
                })
        else:
            raise ValueError(f"Unknown list format, keys: {list(sample.keys())}")
    else:
        raise ValueError(f"Cannot parse {data_file}")

    logger.info(f"Loaded {len(items)} queries ({len(set(it['video_id'] for it in items))} videos)")
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
    """
    from utils.mm_utils import load_video

    # 1. Load video
    frames, frame_times, duration = load_video(video_path, num_frames=num_frames)
    pil_frames = [Image.fromarray(f.numpy()) for f in frames]

    # 2. Process images with build_transform (same as training)
    normalize_type = MODEL_DEFAULT_NORMALIZE.get(model_type, "siglip")
    transform = build_transform(image_size, normalize_type=normalize_type)
    pixel_values = torch.stack([transform(f) for f in pil_frames])

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
        for i in range(len(frames)):
            img_tokens = image_token * image_seq_len
            frame_strings.append(f"{FRAME_TIME_TOKEN}: {fake_token}{global_token}{img_tokens}{fake_token}")
    elif model_type == "fastvlm":
        frame_strings = []
        for i in range(len(frames)):
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
# TRACE-specific: Clip-level saliency score mapping (from reformat_vhd.py)
# ============================================================================

def format_vhd_output(pred_times, pred_scores, duration):
    """
    Map event-level timestamps and scores to clip-level saliency scores.

    QVHighlights uses 2-second clips. Each event's start time is mapped to a
    clip_id, and scores are accumulated/averaged per clip.

    This follows the original TRACE reformat_vhd.py logic:
      clip_id = max(0, int(t / 2) - 1)

    Args:
        pred_times: tensor of shape [N, 2] with (start, end) pairs, or None
        pred_scores: list of float scores per event, or None
        duration: video duration in seconds

    Returns:
        clip_scores: list of float, one per 2-second clip
    """
    clip_num = max(1, int(duration / 2))
    cid2score = np.zeros(clip_num)
    cid2num = np.zeros(clip_num)

    if pred_times is not None and pred_scores is not None and len(pred_scores) > 0:
        num_events = min(pred_times.shape[0], len(pred_scores))
        for i in range(num_events):
            t = float(pred_times[i, 0].item())  # use start time of event
            s = float(pred_scores[i])
            if t > duration:
                continue
            clip_id = max(0, int(t / 2) - 1)
            clip_id = min(clip_id, clip_num - 1)  # clamp
            cid2score[clip_id] += s
            cid2num[clip_id] += 1

    clip_scores = []
    for cid in range(clip_num):
        if cid2num[cid] == 0:
            clip_scores.append(0.0)
        else:
            clip_scores.append(float(cid2score[cid] / cid2num[cid]))

    return clip_scores


# ============================================================================
# QVHighlights Metrics — Moment Retrieval
# ============================================================================

def compute_temporal_iou_batch_cross(spans1, spans2):
    """(N,2) x (M,2) -> (N,M) IoU matrix."""
    spans1 = np.array(spans1, dtype=np.float64)
    spans2 = np.array(spans2, dtype=np.float64)
    areas1 = spans1[:, 1] - spans1[:, 0]
    areas2 = spans2[:, 1] - spans2[:, 0]
    left = np.maximum(spans1[:, None, 0], spans2[None, :, 0])
    right = np.minimum(spans1[:, None, 1], spans2[None, :, 1])
    inter = np.clip(right - left, 0, None)
    union = areas1[:, None] + areas2[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union != 0)
    return iou, union


def compute_temporal_iou_batch_paired(pred_windows, gt_windows):
    """Paired IoU: (N,2) x (N,2) -> (N,)."""
    pred_windows = np.array(pred_windows, dtype=np.float64)
    gt_windows = np.array(gt_windows, dtype=np.float64)
    intersection = np.maximum(
        0,
        np.minimum(pred_windows[:, 1], gt_windows[:, 1])
        - np.maximum(pred_windows[:, 0], gt_windows[:, 0]),
    )
    union = (
        np.maximum(pred_windows[:, 1], gt_windows[:, 1])
        - np.minimum(pred_windows[:, 0], gt_windows[:, 0])
    )
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union != 0)


def interpolated_precision_recall(precision, recall):
    """VOC-style interpolated AP."""
    mprecision = np.hstack([[0], precision, [0]])
    mrecall = np.hstack([[0], recall, [1]])
    for i in range(len(mprecision) - 1)[::-1]:
        mprecision[i] = max(mprecision[i], mprecision[i + 1])
    idx = np.where(mrecall[1::] != mrecall[0:-1])[0] + 1
    ap = np.sum((mrecall[idx] - mrecall[idx - 1]) * mprecision[idx])
    return ap


def compute_average_precision_detection(ground_truth, prediction,
                                         tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Compute AP (detection task) — from DisTime dist_utils."""
    num_thresholds = len(tiou_thresholds)
    num_gts = len(ground_truth)
    num_preds = len(prediction)
    ap = np.zeros(num_thresholds)
    if num_preds == 0:
        return ap

    num_positive = float(num_gts)
    lock_gt = np.ones((num_thresholds, num_gts)) * -1
    tp = np.zeros((num_thresholds, num_preds))
    fp = np.zeros((num_thresholds, num_preds))

    ground_truth_by_videoid = {}
    for i, item in enumerate(ground_truth):
        item["index"] = i
        ground_truth_by_videoid.setdefault(item["video-id"], []).append(item)

    for idx, pred in enumerate(prediction):
        if pred["video-id"] in ground_truth_by_videoid:
            gts = ground_truth_by_videoid[pred["video-id"]]
        else:
            fp[:, idx] = 1
            continue

        _pred = np.array([[pred["t-start"], pred["t-end"]]])
        _gt = np.array([[gt["t-start"], gt["t-end"]] for gt in gts])
        tiou_arr = compute_temporal_iou_batch_cross(_pred, _gt)[0].reshape(-1)

        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for t_idx, tiou_threshold in enumerate(tiou_thresholds):
            for j_idx in tiou_sorted_idx:
                if tiou_arr[j_idx] < tiou_threshold:
                    fp[t_idx, idx] = 1
                    break
                if lock_gt[t_idx, gts[j_idx]["index"]] >= 0:
                    continue
                tp[t_idx, idx] = 1
                lock_gt[t_idx, gts[j_idx]["index"]] = idx
                break
            if fp[t_idx, idx] == 0 and tp[t_idx, idx] == 0:
                fp[t_idx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)
    recall_cumsum = tp_cumsum / num_positive
    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for t_idx in range(num_thresholds):
        ap[t_idx] = interpolated_precision_recall(
            precision_cumsum[t_idx, :], recall_cumsum[t_idx, :]
        )
    return ap


def compute_ap_wrapper(input_triple, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    qid, ground_truth, prediction = input_triple
    scores = compute_average_precision_detection(
        ground_truth, prediction, tiou_thresholds=tiou_thresholds
    )
    return qid, scores


def compute_mr_ap(submission, ground_truth,
                  iou_thds=np.linspace(0.5, 0.95, 10),
                  max_gt_windows=None, max_pred_windows=None,
                  num_workers=8, chunksize=50):
    """Compute mAP for moment retrieval."""
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]

    pred_qid2data = defaultdict(list)
    gt_qid2data = defaultdict(list)

    for d in submission:
        qid = d["qid"]
        pred_windows = (d["pred_relevant_windows"][:max_pred_windows]
                        if max_pred_windows else d["pred_relevant_windows"])
        for w in pred_windows:
            pred_qid2data[qid].append({
                "video-id": d["qid"], "t-start": w[0], "t-end": w[1],
            })

        gt_windows = (d["relevant_windows"][:max_gt_windows]
                      if max_gt_windows else d["relevant_windows"])
        for w in gt_windows:
            gt_qid2data[qid].append({
                "video-id": d["qid"], "t-start": w[0], "t-end": w[1],
            })

    data_triples = [[qid, gt_qid2data[qid], pred_qid2data[qid]] for qid in pred_qid2data]
    compute_fn = partial(compute_ap_wrapper, tiou_thresholds=iou_thds)

    qid2ap_list = {}
    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for qid, scores in pool.imap_unordered(compute_fn, data_triples, chunksize=chunksize):
                qid2ap_list[qid] = scores
    else:
        for triple in data_triples:
            qid, scores = compute_fn(triple)
            qid2ap_list[qid] = scores

    ap_array = np.array(list(qid2ap_list.values()))
    ap_thds = ap_array.mean(0)
    iou_thd2ap = dict(zip([str(e) for e in iou_thds], ap_thds))
    iou_thd2ap["average"] = np.mean(ap_thds)
    iou_thd2ap = {k: float(f"{100 * v:.2f}") for k, v in iou_thd2ap.items()}
    return iou_thd2ap


def compute_mr_r1(submission, ground_truth, iou_thds=np.linspace(0.5, 0.95, 10)):
    """Compute R@1, mIoU — from DisTime dist_utils."""
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]

    pred_qid2window = {d["qid"]: d["pred_relevant_windows"][0][:2] for d in submission}

    gt_qid2window = {}
    for d in ground_truth:
        cur_gt_windows = d["relevant_windows"]
        cur_qid = d["qid"]
        cur_max_iou_idx = 0
        if len(cur_gt_windows) > 0:
            cur_ious = compute_temporal_iou_batch_cross(
                np.array([pred_qid2window[cur_qid]]), np.array(cur_gt_windows)
            )[0]
            cur_max_iou_idx = np.argmax(cur_ious)
        gt_qid2window[cur_qid] = cur_gt_windows[cur_max_iou_idx]

    qids = list(pred_qid2window.keys())
    pred_windows = np.array([pred_qid2window[k] for k in qids]).astype(float)
    gt_windows = np.array([gt_qid2window[k] for k in qids]).astype(float)
    pred_gt_iou = compute_temporal_iou_batch_paired(pred_windows, gt_windows)

    iou_thd2recall = {}
    for thd in iou_thds:
        iou_thd2recall[str(thd)] = float(f"{np.mean(pred_gt_iou >= thd) * 100:.2f}")

    invalid_pred_num = sum(1 for p in pred_windows if -1 in p)
    r1_avg = np.mean(list(iou_thd2recall.values()))
    mIoU = float(np.mean(pred_gt_iou))

    return iou_thd2recall, r1_avg, mIoU, invalid_pred_num


# ============================================================================
# QVHighlights Metrics — Highlight Detection (matching original TRACE eval)
# ============================================================================

def mk_gt_scores(gt_data_item, clip_length=2):
    """
    Build full-video GT saliency matrix from sparse relevant_clip_ids + saliency_scores.

    Original TRACE eval: mk_gt_scores(gt_score_dict)
    QVHighlights GT format:
      - relevant_clip_ids: list of clip indices that are relevant (sparse)
      - saliency_scores: list of [s1, s2, s3] per relevant clip (3 annotators)
      - duration: video duration in seconds

    Returns:
      np.ndarray of shape (num_clips, 3), filled with 0 for non-relevant clips.
      (Original TRACE eval uses 0, NOT -1, so threshold >= 2 treats them as negatives.)
    """
    duration = gt_data_item["duration"]
    num_clips = max(1, int(duration / clip_length))

    relevant_clip_ids = gt_data_item.get("relevant_clip_ids", None)
    saliency_scores = gt_data_item.get("saliency_scores", None)

    # Initialize with 0 (non-relevant = score 0, matching original TRACE eval)
    gt_scores = np.zeros((num_clips, 3), dtype=np.float32)

    if relevant_clip_ids is not None and saliency_scores is not None:
        for idx, clip_id in enumerate(relevant_clip_ids):
            if clip_id < num_clips and idx < len(saliency_scores):
                gt_scores[clip_id] = saliency_scores[idx]
    elif saliency_scores is not None:
        # Fallback: saliency_scores covers ALL clips (no relevant_clip_ids)
        for i, scores in enumerate(saliency_scores):
            if i < num_clips:
                gt_scores[i] = scores

    return gt_scores


def _precision_recall_curve(y_true, y_predict):
    """
    Replicate sklearn.metrics.precision_recall_curve without sklearn dependency.

    Returns (precision, recall, thresholds) in the same format as sklearn:
      - Sorted by decreasing threshold
      - precision[i] = tp / (tp + fp) at threshold[i]
      - recall[i] = tp / total_positives at threshold[i]
      - Last values are precision=1.0, recall=0.0 (sentinel)
      - Tied scores are handled correctly (only one point per unique threshold)
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_predict = np.array(y_predict, dtype=np.float64)

    # Sort by descending score, stable sort for determinism
    desc_score_indices = np.argsort(-y_predict, kind='stable')
    y_predict_sorted = y_predict[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]

    # Find distinct value indices (handle ties)
    distinct_indices = np.where(np.diff(y_predict_sorted))[0]
    end = np.concatenate([distinct_indices, [len(y_true_sorted) - 1]])

    tp = np.cumsum(y_true_sorted)[end]
    fp = np.cumsum(1 - y_true_sorted)[end]

    precision = tp / (tp + fp)
    recall = tp / y_true_sorted.sum()
    thresholds = y_predict_sorted[end]

    # Add sentinel: precision=1, recall=0
    precision = np.concatenate([precision[::-1], [1.0]])
    recall = np.concatenate([recall[::-1], [0.0]])
    thresholds = thresholds[::-1]

    return precision, recall, thresholds


def get_ap(y_true, y_predict, interpolate=True, point_11=False):
    """
    Average Precision — matches original QVHighlights standalone_eval/utils.py exactly.

    ref: https://github.com/jayleicn/moment_detr/blob/main/standalone_eval/utils.py
    ref: https://github.com/gyglim/video2gif_dataset/blob/master/v2g_evaluation/__init__.py

    :param y_true: list/numpy vector of true labels in {0,1}
    :param y_predict: predicted score for each element
    :param interpolate: Use interpolation?
    :param point_11: Use 11-point approximation?
    :return: average precision
    """
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)

    assert len(y_true) == len(y_predict), \
        "Prediction and ground truth need to be of the same length"

    if len(set(y_true)) == 1:
        if y_true[0] == 0:
            return 0  # all negatives
        else:
            return 1  # all positives

    # Compute precision and recall (replicate sklearn.precision_recall_curve)
    precision, recall, _ = _precision_recall_curve(y_true, y_predict)
    recall = recall.astype(np.float32)

    if interpolate:
        for i in range(1, len(precision)):
            precision[i] = max(precision[i - 1], precision[i])

    if point_11:
        precision_11 = [precision[np.where(recall >= t)[0][-1]]
                        for t in np.arange(0, 1.01, 0.1)]
        return np.mean(precision_11)
    else:
        indices = np.where(np.diff(recall))
        return np.mean(precision[indices])


def compute_hl_hit1(qid2pred_scores, qid2gt_scores_binary, threshold=None):
    """
    Compute HIT@1 for highlight detection — matches original QVHighlights eval exactly.

    Original code from moment_detr/standalone_eval/eval.py:
      qid2max_scored_clip_idx = {k: np.argmax(v["pred_saliency_scores"]) for ...}
      hit_scores = np.zeros((len(qid2preds), 3))
      hit_scores[idx] = gt_scores_binary[pred_clip_idx]
      hit_at_one = np.mean(np.max(hit_scores, 1))

    Args:
        qid2pred_scores: {qid: list of float} clip-level predicted scores
        qid2gt_scores_binary: {qid: np.ndarray (num_clips, 3)} binary GT matrix
        threshold: if provided, binarize gt_scores_binary >= threshold first.
                   If None, assumes gt_scores_binary is already binarized.

    Returns:
        float: HIT@1 percentage
    """
    qids = list(qid2pred_scores.keys())
    hit_scores = np.zeros((len(qids), 3))

    for idx, qid in enumerate(qids):
        pred_scores = np.array(qid2pred_scores[qid])
        pred_clip_idx = np.argmax(pred_scores)

        if qid not in qid2gt_scores_binary:
            continue  # hit_scores[idx] stays [0, 0, 0]

        gt_binary = qid2gt_scores_binary[qid]
        if threshold is not None:
            gt_binary = (gt_binary >= threshold).astype(float)

        if pred_clip_idx < len(gt_binary):
            hit_scores[idx] = gt_binary[pred_clip_idx]

    hit_at_one = float(f"{100 * np.mean(np.max(hit_scores, 1)):.2f}")
    return hit_at_one


def compute_hl_ap(qid2pred_scores, qid2gt_scores_binary, num_workers=1, chunksize=50):
    """
    Compute mAP for highlight detection — matches original QVHighlights eval exactly.

    Original code from moment_detr/standalone_eval/eval.py:
      ap_scores = np.zeros((len(qid2preds), 3))
      for idx, qid in enumerate(qids):
          for w_idx in range(3):
              y_true = qid2gt_scores_binary[qid][:, w_idx]
              y_predict = np.array(qid2pred_scores[qid])
              ap = get_ap(y_true, y_predict)
              ap_scores[idx, w_idx] = ap
      mean_ap = 100 * np.mean(ap_scores)

    Note: when gt_binary is all 0 (no positives), get_ap returns 0
    and it IS included in the average. This matches original behavior.

    Args:
        qid2pred_scores: {qid: list of float} clip-level predicted scores
        qid2gt_scores_binary: {qid: np.ndarray (num_clips, 3)} binarized GT

    Returns:
        float: mAP percentage
    """
    qids = list(qid2pred_scores.keys())
    ap_scores = np.zeros((len(qids), 3))

    for idx, qid in enumerate(qids):
        pred_scores = np.array(qid2pred_scores[qid], dtype=np.float64)

        if qid not in qid2gt_scores_binary:
            continue  # ap_scores[idx] stays [0, 0, 0]

        gt_binary = qid2gt_scores_binary[qid]  # (num_clips, 3)

        # Align lengths (same as original compute_ap_from_tuple)
        for w_idx in range(3):
            y_true = gt_binary[:, w_idx]
            y_predict = pred_scores.copy()

            if len(y_true) < len(y_predict):
                y_predict = y_predict[:len(y_true)]
            elif len(y_true) > len(y_predict):
                _y_predict = np.zeros(len(y_true))
                _y_predict[:len(y_predict)] = y_predict
                y_predict = _y_predict

            ap_scores[idx, w_idx] = get_ap(y_true, y_predict)

    mean_ap = float(f"{100 * np.mean(ap_scores):.2f}")
    return mean_ap


def eval_highlight(predictions, gt_data):
    """
    Evaluate highlight detection — matches original TRACE eval_highlight().

    Evaluates at three saliency thresholds:
      - Fair: score >= 2
      - Good: score >= 3
      - VeryGood: score >= 4

    Returns dict with per-threshold and averaged metrics.
    """
    # Build qid -> full GT saliency matrix (num_clips, 3)
    qid2gt_scores_full = {}
    for item in gt_data:
        if item.get("saliency_scores") is not None:
            gt_matrix = mk_gt_scores(item)
            qid2gt_scores_full[item["qid"]] = gt_matrix

    if not qid2gt_scores_full:
        logger.warning("No GT saliency_scores found, skipping HL evaluation")
        return {}

    # Build qid -> predicted clip scores map
    qid2pred_scores = {}
    for p in predictions:
        if "pred_saliency_scores" in p and p["pred_saliency_scores"]:
            qid2pred_scores[p["qid"]] = p["pred_saliency_scores"]

    if not qid2pred_scores:
        logger.warning("No predicted saliency scores, skipping HL evaluation")
        return {}

    # Evaluate at three thresholds — matches original QVHighlights eval_highlight() exactly:
    #   qid2gt_scores_binary = {k: (v >= threshold).astype(float) for k, v in ...}
    #   hit_at_one = compute_hl_hit1(qid2preds, qid2gt_scores_binary)
    #   mean_ap = compute_hl_ap(qid2preds, qid2gt_scores_binary)
    gt_saliency_score_min_list = [2, 3, 4]
    saliency_score_names = ["Fair", "Good", "VeryGood"]

    hl_results = {}
    all_maps = []
    all_hit1s = []
    for gt_saliency_score_min, score_name in zip(gt_saliency_score_min_list,
                                                  saliency_score_names):
        # Binarize GT at this threshold (same as original)
        qid2gt_scores_binary = {
            k: (v >= gt_saliency_score_min).astype(float)
            for k, v in qid2gt_scores_full.items()
        }

        hl_map = compute_hl_ap(qid2pred_scores, qid2gt_scores_binary)
        hl_hit1 = compute_hl_hit1(qid2pred_scores, qid2gt_scores_binary)
        hl_results[f"HL-{score_name}-mAP"] = float(f"{hl_map:.2f}")
        hl_results[f"HL-{score_name}-Hit@1"] = float(f"{hl_hit1:.2f}")
        all_maps.append(hl_map)
        all_hit1s.append(hl_hit1)

    # Average across thresholds
    hl_results["HL-mAP"] = float(f"{np.mean(all_maps):.2f}")
    hl_results["HL-Hit@1"] = float(f"{np.mean(all_hit1s):.2f}")

    return hl_results


def eval_moment_retrieval(submission, ground_truth, verbose=True):
    """Full MR evaluation with mAP + R@1 + mIoU."""
    ret_metrics = {}
    for name in ["full"]:
        iou_thd2ap = compute_mr_ap(submission, ground_truth, num_workers=8, chunksize=50)
        iou_thd2recall, r1_avg, mIoU, invalid_pred_num = compute_mr_r1(submission, ground_truth)
        ret_metrics[name] = {
            "MR-mAP": iou_thd2ap,
            "MR-R1": iou_thd2recall,
            "MR-R1-avg": r1_avg,
            "MR-mIoU": mIoU,
            "MR-invalid_pred_num": invalid_pred_num,
        }
    return ret_metrics


def eval_submission(submission, ground_truth, gt_data=None, eval_tasks="mr,hl"):
    """Top-level evaluation — returns full metrics dict.

    Args:
        eval_tasks: comma-separated string of tasks to evaluate.
                    "mr" = Moment Retrieval, "hl" = Highlight Detection.
                    Default "mr,hl" runs both. Use "hl" for VHD-only (time points).
    """
    tasks = [t.strip().lower() for t in eval_tasks.split(",")]

    brief = OrderedDict()
    full_metrics = {}

    # Moment Retrieval (requires [start, end] time windows)
    if "mr" in tasks:
        eval_metrics = eval_moment_retrieval(submission, ground_truth)
        full_metrics = eval_metrics.get("full", {})
        brief["MR-mAP-avg"] = full_metrics.get("MR-mAP", {}).get("average", 0)
        brief["MR-mAP@0.5"] = full_metrics.get("MR-mAP", {}).get("0.5", 0)
        brief["MR-mAP@0.75"] = full_metrics.get("MR-mAP", {}).get("0.75", 0)
        brief["MR-R1@0.5"] = full_metrics.get("MR-R1", {}).get("0.5", 0)
        brief["MR-R1@0.7"] = full_metrics.get("MR-R1", {}).get("0.7", 0)
        brief["MR-R1-avg"] = full_metrics.get("MR-R1-avg", 0)
        brief["MR-mIoU"] = full_metrics.get("MR-mIoU", 0)
        brief["MR-invalid_pred_num"] = full_metrics.get("MR-invalid_pred_num", 0)

    # Highlight Detection (works with time points + scores)
    hl_metrics = {}
    if "hl" in tasks and gt_data is not None:
        hl_metrics = eval_highlight(submission, gt_data)
        brief.update(hl_metrics)

    return {"brief": brief, "full": full_metrics, "highlight": hl_metrics}


# ============================================================================
# Worker for multi-GPU inference
# ============================================================================

def worker_inference(worker_id, gpu_id, worker_data, args, image_size, tmp_dir,
                     passes=None, query_template=None, pass_name=""):
    """
    Worker process: load model onto assigned GPU, run inference on data subset.
    Supports multiple passes (different query templates) with a single model load.

    Args:
        passes: list of (pass_name, query_template) tuples. If provided, runs all
                passes sequentially with the same model. Overrides query_template/pass_name.
        query_template/pass_name: legacy single-pass interface (used if passes is None).
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

    # Build passes list
    if passes is None:
        if query_template is None:
            query_template = getattr(args, 'query_template', '') or ''
        passes = [(pass_name, query_template)]

    wlog.info(f"Starting: {len(worker_data)} queries x {len(passes)} passes on GPU {gpu_id}")
    for pn, pt in passes:
        wlog.info(f"  pass={pn}, template={pt[:60]}...")

    # Load model ONCE
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

    # Run all passes with the same model
    all_pass_results = {}  # pass_name -> {predictions, raw_trace_outputs, eval_entries, errors}

    for cur_pass_name, cur_query_template in passes:
        wlog.info(f"--- Pass: {cur_pass_name} ({len(worker_data)} queries) ---")

        raw_trace_outputs = []
        eval_entries = []
        predictions = []
        errors = 0
        video_cache = {}

        for idx, item in enumerate(tqdm(worker_data,
                                         desc=f"W{worker_id}/GPU{gpu_id}/{cur_pass_name}",
                                         position=worker_id)):
            qid = item['qid']
            video_id = item['video_id']
            raw_query = item['query']
            relevant_windows = item['relevant_windows']
            gt_duration = item['duration']

            try:
                query = cur_query_template.format(query=raw_query.strip())
            except (IndexError, KeyError):
                query = f"{cur_query_template} {raw_query.strip()}"

            video_path = None
            candidate = os.path.join(args.video_root, video_id)
            if os.path.exists(candidate):
                video_path = candidate
            else:
                base = video_id.replace('.mp4', '').replace('.avi', '').replace('.mkv', '')
                for ext in ['.mp4', '.avi', '.mkv', '.webm']:
                    candidate = os.path.join(args.video_root, base + ext)
                    if os.path.exists(candidate):
                        video_path = candidate
                        break

            if video_path is None:
                errors += 1
                raw_trace_outputs.append({
                    "video": video_id, "id": qid,
                    "timestamps": [], "scores": [], "captions": [],
                })
                predictions.append({
                    "qid": qid, "video_id": video_id, "query": raw_query,
                    "pred_relevant_windows": [[-1, -1, 0.0]],
                    "pred_saliency_scores": [],
                    "relevant_windows": relevant_windows,
                    "duration": gt_duration,
                })
                eval_entries.append({
                    "qid": qid,
                    "pred_relevant_windows": [[-1, -1, 0.0]],
                    "pred_saliency_scores": [],
                    "relevant_windows": relevant_windows,
                })
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
                    "video": video_id, "id": qid,
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

                if pred_times is not None and pred_times.shape[0] > 0:
                    pred_windows = []
                    event_scores = []
                    for j in range(pred_times.shape[0]):
                        s = round(float(pred_times[j, 0].item()), 2)
                        e = round(float(pred_times[j, 1].item()), 2)
                        if s > e:
                            s, e = e, s
                        s = max(0.0, min(s, gt_duration))
                        e = max(0.0, min(e, gt_duration))
                        pred_windows.append([s, e])

                    if pred_scores and len(pred_scores) > 0:
                        event_scores = [float(sc) for sc in pred_scores[:len(pred_windows)]]
                        if len(event_scores) < len(pred_windows):
                            event_scores += [0.0] * (len(pred_windows) - len(event_scores))
                        paired = list(zip(pred_windows, event_scores))
                        paired.sort(key=lambda x: x[1], reverse=True)
                        pred_windows = [p[0] for p in paired]
                        event_scores = [p[1] for p in paired]
                    else:
                        event_scores = [1.0] * len(pred_windows)

                    pred_windows_with_score = [
                        [w[0], w[1], sc] for w, sc in zip(pred_windows, event_scores)
                    ]
                else:
                    pred_windows = [[-1, -1]]
                    event_scores = [0.0]
                    pred_windows_with_score = [[-1, -1, 0.0]]

                clip_scores = format_vhd_output(pred_times, pred_scores, gt_duration)

            except Exception as e:
                wlog.warning(f"Error {video_id}: {e}")
                pred_windows = [[-1, -1]]
                event_scores = [0.0]
                pred_windows_with_score = [[-1, -1, 0.0]]
                clip_scores = []
                raw_trace_outputs.append({
                    "video": video_id, "id": qid,
                    "timestamps": [], "scores": [], "captions": [],
                })
                errors += 1

            predictions.append({
                "qid": qid, "video_id": video_id, "query": raw_query,
                "pred_relevant_windows": pred_windows_with_score,
                "pred_saliency_scores": clip_scores,
                "pred_event_scores": event_scores,
                "relevant_windows": relevant_windows,
                "duration": gt_duration,
            })
            eval_entries.append({
                "qid": qid,
                "pred_relevant_windows": pred_windows_with_score,
                "pred_saliency_scores": clip_scores,
                "relevant_windows": relevant_windows,
            })

            if (idx + 1) % 20 == 0:
                torch.cuda.empty_cache()

        all_pass_results[cur_pass_name] = {
            "predictions": predictions,
            "raw_trace_outputs": raw_trace_outputs,
            "eval_entries": eval_entries,
            "errors": errors,
        }
        wlog.info(f"Pass '{cur_pass_name}' done: {len(predictions)} preds, {errors} errors")

    # Save all pass results to one file
    out_path = os.path.join(tmp_dir, f"worker_{worker_id}.json")
    with open(out_path, 'w') as f:
        json.dump(all_pass_results, f, ensure_ascii=False)
    wlog.info(f"Done: {len(passes)} passes, {len(worker_data)} queries each -> {out_path}")

    # Explicit cleanup
    del model
    del video_cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="TRACE QVHighlights Evaluation")

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
                        help="Vision pooling stride (FastVLM only, 1=no pooling)")

    # Data
    parser.add_argument("--data_file", type=str, required=True,
                        help="QVHighlights annotations (COCO or DisTime format)")
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
                        help="Output directory. Default: ./results/QVHighlights-TRACE/{model_type}")

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

    # Query templates — MR and HL use different prompts
    parser.add_argument("--query_template", type=str, default=None,
                        help="(Legacy) Single query template for both tasks. "
                             "Overridden by --query_template_mr / --query_template_hl if set.")
    parser.add_argument("--query_template_mr", type=str,
                        default="Localize the visual content described by the given textual query '{query}' in the video, and output the start and end timestamps in seconds.",
                        help="Query template for Moment Retrieval (generates [start, end] windows).")
    parser.add_argument("--query_template_hl", type=str,
                        default="Please find the highlight contents in the video described by a sentence query, determining the highlight timestamps and its saliency score on a scale from 1 to 5. Now I will give you the sentence query: '{query}'. Please return the query-based highlight timestamps and salient scores.",
                        help="Query template for Highlight Detection / VHD (generates time points + scores).")

    # Evaluation tasks
    parser.add_argument("--eval_tasks", type=str, default="mr,hl",
                        help="Comma-separated eval tasks: 'mr' (Moment Retrieval, "
                             "needs [start,end] windows), 'hl' (Highlight Detection, "
                             "works with time points + scores). Default: 'mr,hl'. "
                             "Use 'hl' for VHD-only evaluation.")

    # Load saved predictions (skip inference)
    parser.add_argument("--load_predictions", type=str, default=None,
                        help="Path to saved predictions.json to recompute metrics "
                             "without re-running inference. Can also be a directory "
                             "containing predictions.json.")

    args = parser.parse_args()

    # 默认按 model_type 分目录
    if args.out_dir is None:
        args.out_dir = f"./results/QVHighlights-TRACE/{args.model_type}"
    os.makedirs(args.out_dir, exist_ok=True)

    # ================================================================
    # 1. Load data (GT annotations — always needed for metrics)
    # ================================================================
    data = load_qvh_data(args.data_file)

    # ================================================================
    # Fast path: recompute metrics from saved predictions
    # ================================================================
    if args.load_predictions:
        pred_path = args.load_predictions
        if os.path.isdir(pred_path):
            pred_path = os.path.join(pred_path, "predictions.json")
        logger.info(f"Loading saved predictions from {pred_path}")
        with open(pred_path, 'r') as f:
            saved_predictions = json.load(f)
        logger.info(f"Loaded {len(saved_predictions)} predictions")

        # Split predictions by _pass tag (if saved from dual-template inference)
        has_pass_tag = any("_pass" in p for p in saved_predictions)
        if has_pass_tag:
            mr_preds = [p for p in saved_predictions if p.get("_pass") == "mr"]
            hl_preds = [p for p in saved_predictions if p.get("_pass") == "hl"]
            # "mr+hl" pass (same template) goes to both
            both_preds = [p for p in saved_predictions if p.get("_pass") == "mr+hl"]
            if both_preds:
                mr_preds = both_preds
                hl_preds = both_preds
            logger.info(f"Split by _pass: {len(mr_preds)} MR, {len(hl_preds)} HL predictions")
        else:
            # No pass tag → same predictions for both tasks
            mr_preds = saved_predictions
            hl_preds = saved_predictions

        # Also try loading per-pass files if they exist
        pred_dir = os.path.dirname(pred_path)
        mr_pass_file = os.path.join(pred_dir, "predictions_mr.json")
        hl_pass_file = os.path.join(pred_dir, "predictions_hl.json")
        if os.path.exists(mr_pass_file) and os.path.exists(hl_pass_file):
            logger.info(f"Found per-pass prediction files, using those instead")
            with open(mr_pass_file, 'r') as f:
                mr_preds = json.load(f)
            with open(hl_pass_file, 'r') as f:
                hl_preds = json.load(f)
            logger.info(f"Loaded {len(mr_preds)} MR, {len(hl_preds)} HL predictions")

        def _build_eval_entries(preds):
            entries = []
            for p in preds:
                entries.append({
                    "qid": p["qid"],
                    "pred_relevant_windows": p["pred_relevant_windows"],
                    "pred_saliency_scores": p.get("pred_saliency_scores", []),
                    "relevant_windows": p["relevant_windows"],
                })
            return entries

        eval_task_list = [t.strip().lower() for t in args.eval_tasks.split(",")]
        need_mr = "mr" in eval_task_list
        need_hl = "hl" in eval_task_list

        mr_eval_entries = _build_eval_entries(mr_preds)
        hl_eval_entries = _build_eval_entries(hl_preds)

        print(f"\nRecomputing metrics from saved predictions...")
        if has_pass_tag or os.path.exists(mr_pass_file):
            print(f"  (MR: {len(mr_eval_entries)} queries, HL: {len(hl_eval_entries)} queries)")
        brief = OrderedDict()
        full_metrics = {}

        if need_mr and mr_eval_entries:
            mr_metrics = eval_moment_retrieval(mr_eval_entries, mr_eval_entries)
            full_metrics = mr_metrics.get("full", {})
            brief["MR-mAP-avg"] = full_metrics.get("MR-mAP", {}).get("average", 0)
            brief["MR-mAP@0.5"] = full_metrics.get("MR-mAP", {}).get("0.5", 0)
            brief["MR-mAP@0.75"] = full_metrics.get("MR-mAP", {}).get("0.75", 0)
            brief["MR-R1@0.5"] = full_metrics.get("MR-R1", {}).get("0.5", 0)
            brief["MR-R1@0.7"] = full_metrics.get("MR-R1", {}).get("0.7", 0)
            brief["MR-R1-avg"] = full_metrics.get("MR-R1-avg", 0)
            brief["MR-mIoU"] = full_metrics.get("MR-mIoU", 0)
            brief["MR-invalid_pred_num"] = full_metrics.get("MR-invalid_pred_num", 0)

        hl_metrics = {}
        if need_hl and hl_eval_entries:
            hl_metrics = eval_highlight(hl_eval_entries, data)
            brief.update(hl_metrics)

        all_metrics = {"brief": brief, "full": full_metrics, "highlight": hl_metrics}

        print(f"\n{'='*60}")
        print(f"  QVHighlights Results (from saved predictions)")
        print(f"  (MR={len(mr_eval_entries)} / HL={len(hl_eval_entries)} queries, tasks={args.eval_tasks})")
        print(f"{'='*60}")
        if need_mr:
            print(f"  --- Moment Retrieval ---")
            print(f"  R@1 IoU=0.5:  {brief.get('MR-R1@0.5', 0):.2f}")
            print(f"  R@1 IoU=0.7:  {brief.get('MR-R1@0.7', 0):.2f}")
            print(f"  R@1 avg:      {brief.get('MR-R1-avg', 0):.2f}")
            print(f"  mAP avg:      {brief.get('MR-mAP-avg', 0):.2f}")
            print(f"  mAP@0.5:      {brief.get('MR-mAP@0.5', 0):.2f}")
            print(f"  mAP@0.75:     {brief.get('MR-mAP@0.75', 0):.2f}")
            print(f"  mIoU:         {brief.get('MR-mIoU', 0)*100:.2f}")
            print(f"  Invalid preds: {brief.get('MR-invalid_pred_num', 0)}")
        if hl_metrics:
            print(f"  --- Highlight Detection (clip-level saliency) ---")
            print(f"  HL-mAP (avg): {hl_metrics.get('HL-mAP', 0):.2f}")
            print(f"  HL-Hit@1(avg):{hl_metrics.get('HL-Hit@1', 0):.2f}")
            for level in ["Fair", "Good", "VeryGood"]:
                m = hl_metrics.get(f'HL-{level}-mAP', 0)
                h = hl_metrics.get(f'HL-{level}-Hit@1', 0)
                print(f"    {level:10s}: mAP={m:.2f}  Hit@1={h:.2f}")
        print(f"{'='*60}\n")

        metrics_file = os.path.join(args.out_dir, "metrics_recomputed.json")
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"Recomputed metrics saved to {metrics_file}")
        return  # Done — skip inference

    # ================================================================
    # Normal path: run inference
    # ================================================================
    # 按 model_type 选择 image_size (与 train.py 对齐)
    image_size = MODEL_DEFAULT_IMAGE_SIZE[args.model_type]
    logger.info(f"Using model_type={args.model_type}, image_size={image_size}")
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
        logger.info(f"Shard {args.shard_id}/{args.num_shards}: queries [{start_idx}, {end_idx})")

    # ================================================================
    # Resolve query templates
    # ================================================================
    eval_task_list = [t.strip().lower() for t in args.eval_tasks.split(",")]

    # Legacy --query_template overrides both if explicitly set
    if args.query_template is not None:
        logger.info("Using legacy --query_template for all tasks")
        args.query_template_mr = args.query_template
        args.query_template_hl = args.query_template

    # Determine which inference passes to run
    need_mr = "mr" in eval_task_list
    need_hl = "hl" in eval_task_list
    same_template = (args.query_template_mr == args.query_template_hl)

    if same_template:
        # Same template → one pass, both tasks use same results
        passes = [("mr+hl", args.query_template_mr)]
    else:
        # Different templates → separate passes
        passes = []
        if need_mr:
            passes.append(("mr", args.query_template_mr))
        if need_hl:
            passes.append(("hl", args.query_template_hl))

    logger.info(f"Eval tasks: {args.eval_tasks}, passes: {[p[0] for p in passes]}")
    for pname, ptpl in passes:
        logger.info(f"  {pname} template: {ptpl[:80]}...")

    # ================================================================
    # 2. Run inference — one or two passes with different query templates
    # ================================================================
    num_gpus = args.num_gpus
    total_workers = num_gpus * args.workers_per_gpu

    # Results per pass
    pass_results = {}  # pass_name -> {"predictions", "eval_entries", "raw_trace_outputs", "errors"}

    if total_workers > 1:
        # ============================================================
        # Multi-GPU path: use worker_inference via torch.multiprocessing
        # Each worker loads model ONCE and runs ALL passes sequentially.
        # ============================================================
        logger.info(f"Multi-GPU mode: {num_gpus} GPUs x {args.workers_per_gpu} workers = {total_workers} workers")
        logger.info(f"Passes to run per worker: {[p[0] for p in passes]}")
        torch_mp.set_start_method('spawn', force=True)

        # Split data across workers
        worker_data_splits = [[] for _ in range(total_workers)]
        for i, item in enumerate(data):
            worker_data_splits[i % total_workers].append(item)

        for wi, wd in enumerate(worker_data_splits):
            gpu_id = wi // args.workers_per_gpu
            logger.info(f"  Worker {wi} (GPU {gpu_id}): {len(wd)} queries x {len(passes)} passes")

        # Create temp directory for worker output files
        tmp_dir = tempfile.mkdtemp(prefix="qvh_trace_multi_")
        logger.info(f"  Temp dir: {tmp_dir}")

        # Spawn workers — each worker runs ALL passes with one model load
        processes = []
        for wi in range(total_workers):
            gpu_id = wi // args.workers_per_gpu
            p = torch_mp.Process(
                target=worker_inference,
                args=(wi, gpu_id, worker_data_splits[wi], args, image_size, tmp_dir),
                kwargs={"passes": passes},
            )
            p.start()
            logger.info(f"  Started worker {wi} on GPU {gpu_id} (pid={p.pid})")
            processes.append(p)

        # Wait for all workers with timeout
        timeout = 4 * 3600  # 4 hours
        for wi, p in enumerate(processes):
            p.join(timeout=timeout)
            if p.is_alive():
                logger.error(f"Worker {wi} timed out after {timeout}s, terminating...")
                p.terminate()
                p.join(timeout=30)
            elif p.exitcode != 0:
                logger.error(f"Worker {wi} exited with code {p.exitcode}")
            else:
                logger.info(f"Worker {wi} finished successfully")

        # Merge worker results — each worker file now contains all passes
        for pn, _ in passes:
            pass_results[pn] = {
                "predictions": [], "eval_entries": [],
                "raw_trace_outputs": [], "errors": 0,
            }

        for wi in range(total_workers):
            worker_file = os.path.join(tmp_dir, f"worker_{wi}.json")
            if not os.path.exists(worker_file):
                logger.error(f"Worker {wi} output file not found: {worker_file}")
                continue
            with open(worker_file, 'r') as f:
                worker_all_passes = json.load(f)
            for pn, _ in passes:
                if pn not in worker_all_passes:
                    logger.error(f"Worker {wi} missing pass '{pn}'")
                    continue
                wr = worker_all_passes[pn]
                pass_results[pn]["predictions"].extend(wr["predictions"])
                pass_results[pn]["eval_entries"].extend(wr["eval_entries"])
                pass_results[pn]["raw_trace_outputs"].extend(wr["raw_trace_outputs"])
                pass_results[pn]["errors"] += wr["errors"]
            logger.info(f"  Merged worker {wi}")

        # Clean up temp files
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

        for pn, _ in passes:
            pr = pass_results[pn]
            logger.info(f"Pass '{pn}': {len(pr['predictions'])} predictions, {pr['errors']} errors")

    else:
        # ============================================================
        # Single-GPU path: load model once, run inference directly
        # ============================================================
        logger.info("Single-GPU mode: loading model...")

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

        for pass_name, query_template in passes:
            logger.info(f"\n{'='*40} Pass: {pass_name} {'='*40}")

            raw_trace_outputs = []
            eval_entries = []
            predictions = []
            errors = 0
            video_cache = {}

            for idx in tqdm(range(len(data)),
                            desc=f"{pass_name} ({args.model_type}-TRACE)"):
                item = data[idx]
                qid = item['qid']
                video_id = item['video_id']
                raw_query = item['query']
                relevant_windows = item['relevant_windows']
                gt_duration = item['duration']

                try:
                    query = query_template.format(query=raw_query.strip())
                except (IndexError, KeyError):
                    query = f"{query_template} {raw_query.strip()}"

                video_path = None
                candidate = os.path.join(args.video_root, video_id)
                if os.path.exists(candidate):
                    video_path = candidate
                else:
                    base = video_id.replace('.mp4', '').replace('.avi', '').replace('.mkv', '')
                    for ext in ['.mp4', '.avi', '.mkv', '.webm']:
                        candidate = os.path.join(args.video_root, base + ext)
                        if os.path.exists(candidate):
                            video_path = candidate
                            break

                if video_path is None:
                    logger.warning(f"[{idx}] Video not found: {video_id}")
                    errors += 1
                    raw_trace_outputs.append({
                        "video": video_id, "id": qid,
                        "timestamps": [], "scores": [], "captions": [],
                    })
                    predictions.append({
                        "qid": qid, "video_id": video_id, "query": raw_query,
                        "pred_relevant_windows": [[-1, -1, 0.0]],
                        "pred_saliency_scores": [],
                        "relevant_windows": relevant_windows,
                        "duration": gt_duration,
                    })
                    eval_entries.append({
                        "qid": qid,
                        "pred_relevant_windows": [[-1, -1, 0.0]],
                        "pred_saliency_scores": [],
                        "relevant_windows": relevant_windows,
                    })
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
                    pred_scores = outputs.get('pred_scores', [None])[0]
                    gen_text = outputs.get('generated_text', [''])[0]

                    raw_output = {
                        "video": video_id, "id": qid,
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

                    if pred_times is not None and pred_times.shape[0] > 0:
                        pred_windows = []
                        event_scores = []
                        for j in range(pred_times.shape[0]):
                            s = round(float(pred_times[j, 0].item()), 2)
                            e = round(float(pred_times[j, 1].item()), 2)
                            if s > e:
                                s, e = e, s
                            s = max(0.0, min(s, gt_duration))
                            e = max(0.0, min(e, gt_duration))
                            pred_windows.append([s, e])

                        if pred_scores and len(pred_scores) > 0:
                            event_scores = [float(sc) for sc in pred_scores[:len(pred_windows)]]
                            if len(event_scores) < len(pred_windows):
                                event_scores += [0.0] * (len(pred_windows) - len(event_scores))
                            paired = list(zip(pred_windows, event_scores))
                            paired.sort(key=lambda x: x[1], reverse=True)
                            pred_windows = [p[0] for p in paired]
                            event_scores = [p[1] for p in paired]
                        else:
                            event_scores = [1.0] * len(pred_windows)

                        pred_windows_with_score = [
                            [w[0], w[1], sc] for w, sc in zip(pred_windows, event_scores)
                        ]
                    else:
                        pred_windows = [[-1, -1]]
                        event_scores = [0.0]
                        pred_windows_with_score = [[-1, -1, 0.0]]
                        logger.warning(f"[{idx}] {video_id} no timestamp predicted")

                    clip_scores = format_vhd_output(pred_times, pred_scores, gt_duration)

                    if idx < 5:
                        logger.info(f"[{pass_name}][{idx}] qid={qid}: "
                                    f"pred={pred_windows_with_score[:3]} "
                                    f"event_scores={event_scores[:3]} gt={relevant_windows}")
                        logger.info(f"  text: {gen_text[:200]}")

                except Exception as e:
                    logger.warning(f"[{idx}] Error {video_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    pred_windows_with_score = [[-1, -1, 0.0]]
                    event_scores = [0.0]
                    clip_scores = []
                    raw_trace_outputs.append({
                        "video": video_id, "id": qid,
                        "timestamps": [], "scores": [], "captions": [],
                    })
                    errors += 1

                predictions.append({
                    "qid": qid, "video_id": video_id, "query": raw_query,
                    "pred_relevant_windows": pred_windows_with_score,
                    "pred_saliency_scores": clip_scores,
                    "pred_event_scores": event_scores,
                    "relevant_windows": relevant_windows,
                    "duration": gt_duration,
                })
                eval_entries.append({
                    "qid": qid,
                    "pred_relevant_windows": pred_windows_with_score,
                    "pred_saliency_scores": clip_scores,
                    "relevant_windows": relevant_windows,
                })

                if (idx + 1) % 100 == 0:
                    logger.info(f"[{pass_name}][{idx+1}/{len(data)}] errors={errors}")

            pass_results[pass_name] = {
                "predictions": predictions,
                "eval_entries": eval_entries,
                "raw_trace_outputs": raw_trace_outputs,
                "errors": errors,
            }
            logger.info(f"Pass '{pass_name}' done: {len(predictions)} predictions, {errors} errors")

        # Cleanup single-GPU model
        del model
        torch.cuda.empty_cache()

    # ================================================================
    # 3. Assemble eval data from pass results
    # ================================================================
    # When same template → single pass "mr+hl" has everything
    # When different templates → MR pass for MR eval, HL pass for HL eval
    if same_template:
        combined = pass_results["mr+hl"]
        mr_eval_entries = combined["eval_entries"]
        hl_eval_entries = combined["eval_entries"]
        all_predictions = combined["predictions"]
        all_raw = combined["raw_trace_outputs"]
        total_errors = combined["errors"]
    else:
        mr_eval_entries = pass_results.get("mr", {}).get("eval_entries", [])
        hl_eval_entries = pass_results.get("hl", {}).get("eval_entries", [])
        # Merge predictions from both passes for saving
        all_predictions = []
        all_raw = []
        total_errors = 0
        for pname in ["mr", "hl"]:
            if pname in pass_results:
                pr = pass_results[pname]
                # Tag predictions with which pass they came from
                for p in pr["predictions"]:
                    p["_pass"] = pname
                all_predictions.extend(pr["predictions"])
                all_raw.extend(pr["raw_trace_outputs"])
                total_errors += pr["errors"]

    # ================================================================
    # 4. Compute metrics
    # ================================================================
    print(f"\nComputing metrics...")
    try:
        # MR evaluation uses MR-pass results
        brief = OrderedDict()
        full_metrics = {}
        if need_mr and mr_eval_entries:
            mr_metrics = eval_moment_retrieval(mr_eval_entries, mr_eval_entries)
            full_metrics = mr_metrics.get("full", {})
            brief["MR-mAP-avg"] = full_metrics.get("MR-mAP", {}).get("average", 0)
            brief["MR-mAP@0.5"] = full_metrics.get("MR-mAP", {}).get("0.5", 0)
            brief["MR-mAP@0.75"] = full_metrics.get("MR-mAP", {}).get("0.75", 0)
            brief["MR-R1@0.5"] = full_metrics.get("MR-R1", {}).get("0.5", 0)
            brief["MR-R1@0.7"] = full_metrics.get("MR-R1", {}).get("0.7", 0)
            brief["MR-R1-avg"] = full_metrics.get("MR-R1-avg", 0)
            brief["MR-mIoU"] = full_metrics.get("MR-mIoU", 0)
            brief["MR-invalid_pred_num"] = full_metrics.get("MR-invalid_pred_num", 0)

        # HL evaluation uses HL-pass results
        hl_metrics = {}
        if need_hl and hl_eval_entries:
            hl_metrics = eval_highlight(hl_eval_entries, data)
            brief.update(hl_metrics)

        all_metrics = {"brief": brief, "full": full_metrics, "highlight": hl_metrics}

        num_queries = len(mr_eval_entries) if need_mr else len(hl_eval_entries)
        print(f"\n{'='*60}")
        print(f"  QVHighlights Results — {args.model_type}-TRACE")
        print(f"  ({num_queries} queries, tasks={args.eval_tasks})")
        if not same_template:
            print(f"  (separate inference passes for MR and HL)")
        print(f"{'='*60}")
        if need_mr and mr_eval_entries:
            print(f"  --- Moment Retrieval ---")
            print(f"  R@1 IoU=0.5:  {brief.get('MR-R1@0.5', 0):.2f}")
            print(f"  R@1 IoU=0.7:  {brief.get('MR-R1@0.7', 0):.2f}")
            print(f"  R@1 avg:      {brief.get('MR-R1-avg', 0):.2f}")
            print(f"  mAP avg:      {brief.get('MR-mAP-avg', 0):.2f}")
            print(f"  mAP@0.5:      {brief.get('MR-mAP@0.5', 0):.2f}")
            print(f"  mAP@0.75:     {brief.get('MR-mAP@0.75', 0):.2f}")
            print(f"  mIoU:         {brief.get('MR-mIoU', 0)*100:.2f}")
            print(f"  Invalid preds: {brief.get('MR-invalid_pred_num', 0)}")
        if hl_metrics:
            print(f"  --- Highlight Detection (clip-level saliency) ---")
            print(f"  HL-mAP (avg): {hl_metrics.get('HL-mAP', 0):.2f}")
            print(f"  HL-Hit@1(avg):{hl_metrics.get('HL-Hit@1', 0):.2f}")
            for level in ["Fair", "Good", "VeryGood"]:
                m = hl_metrics.get(f'HL-{level}-mAP', 0)
                h = hl_metrics.get(f'HL-{level}-Hit@1', 0)
                print(f"    {level:10s}: mAP={m:.2f}  Hit@1={h:.2f}")
        print(f"  --- Summary ---")
        print(f"  Errors:       {total_errors}")
        print(f"{'='*60}\n")

        metrics_file = os.path.join(args.out_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")
    except Exception as e:
        logger.error(f"Metrics computation failed: {e}")
        import traceback
        traceback.print_exc()

    # ================================================================
    # 5. Save predictions
    # ================================================================
    raw_file = os.path.join(args.out_dir, "raw_trace_output.json")
    with open(raw_file, 'w') as f:
        json.dump(all_raw, f, indent=2)
    logger.info(f"Raw TRACE output saved to {raw_file}")

    pred_file = os.path.join(args.out_dir, "predictions.json")
    with open(pred_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    logger.info(f"Predictions saved to {pred_file}")

    # Save per-pass predictions separately when using different templates
    if not same_template:
        for pname in ["mr", "hl"]:
            if pname in pass_results:
                pass_pred_file = os.path.join(args.out_dir, f"predictions_{pname}.json")
                with open(pass_pred_file, 'w') as f:
                    json.dump(pass_results[pname]["predictions"], f, indent=2)
                logger.info(f"Pass '{pname}' predictions saved to {pass_pred_file}")

    # Save VHD-format output (using HL-pass results for clip-level scores)
    hl_preds = pass_results.get("hl", pass_results.get("mr+hl", {})).get("predictions", [])
    vhd_output = []
    for p in hl_preds:
        vhd_output.append({
            "qid": p["qid"],
            "query": p["query"],
            "vid": p["video_id"],
            "pred_saliency_scores": p.get("pred_saliency_scores", []),
        })
    vhd_file = os.path.join(args.out_dir, "vhd_predictions.json")
    with open(vhd_file, 'w') as f:
        json.dump(vhd_output, f, indent=2)
    logger.info(f"VHD format predictions saved to {vhd_file}")

    print(f"\nDone: {len(all_predictions)} total predictions, {total_errors} errors")


if __name__ == "__main__":
    main()
