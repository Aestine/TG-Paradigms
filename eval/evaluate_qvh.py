"""
QVHighlights evaluation for DisTime (SmolVLM / FastVLM).

Supports both model backends via --model_type.

Reads your COCO-format annotations:
  {"annotations": [
    {"image_id": "VID.mp4", "id": 2579, "caption": "...",
     "timestamp": [[82, 150]], "duration": 150, ...},
    ...
  ]}

Metrics: R@1 IoU=0.5/0.7, mAP@0.5/0.75/avg, mIoU

Usage (SmolVLM):
    python evaluate_qvh.py \
        --model_type smolvlm \
        --model_name_or_path /projects/bffz/yzou1/models/SmolVLM2-2.2B-Instruct \
        --checkpoint_dir /path/to/checkpoint \
        --data_file /path/to/qvhighlights/val.json \
        --video_root /path/to/qvhighlights/videos \
        --num_frames 32

Usage (FastVLM):
    python evaluate_qvh.py \
        --model_type fastvlm \
        --model_name_or_path KamilaMila/FastVLM-0.5B \
        --checkpoint_dir /path/to/checkpoint \
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
import time as time_module
from collections import OrderedDict, defaultdict
from functools import partial
from tqdm import tqdm
import multiprocessing as mp

import torch
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

# 每种模型的默认 image_size (与 train.py 对齐)
MODEL_DEFAULT_IMAGE_SIZE = {
    "smolvlm": 384,    # SmolVLM (SigLIP vision encoder)
    "fastvlm": 1024,   # FastVLM (fastvit_mci3, crop_size=1024)
}


# ============================================================================
# Data loading
# ============================================================================

def load_qvh_data(data_file):
    """
    Load QVHighlights annotations.

    Supported formats:
      1) Your COCO format:
         {"annotations": [{"image_id":"VID.mp4", "id":N, "caption":"...",
                           "timestamp":[[s,e],...], "duration":D}, ...]}
      2) DisTime's val.json:
         [{"video":"VID", "qid":N, "query":"...", "relevant_windows":[[s,e],...], "duration":D}, ...]

    Returns list of:
      {"qid", "video_id", "query", "relevant_windows", "duration"}
    """
    with open(data_file, 'r') as f:
        raw = json.load(f)

    items = []

    # Format 1: COCO annotations
    if isinstance(raw, dict) and "annotations" in raw:
        logger.info("Detected COCO annotation format")
        for ann in raw["annotations"]:
            vid = ann["image_id"]  # e.g. "NUsG9BgSes0_210.0_360.0.mp4"
            items.append({
                "qid": ann["id"],
                "video_id": vid,
                "query": ann["caption"],
                "relevant_windows": ann["timestamp"],  # [[s,e], ...]
                "duration": ann["duration"],
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
                })
        else:
            raise ValueError(f"Unknown list format, keys: {list(sample.keys())}")
    else:
        raise ValueError(f"Cannot parse {data_file}")

    logger.info(f"Loaded {len(items)} queries ({len(set(it['video_id'] for it in items))} videos)")
    return items


# ============================================================================
# Model loading – 按 model_type 动态选择模型类 (与 train.py 对齐)
# ============================================================================

def load_model(args):
    """Load DisTime model – supports both SmolVLM and FastVLM via args.model_type."""
    model_type = args.model_type

    if model_type == "smolvlm":
        from models.smolvlm_distime import SmolVLMDisTime, DisTimeConfig
        ModelClass = SmolVLMDisTime
    elif model_type == "fastvlm":
        from models.fastvlm_distime import FastVLMDisTime, DisTimeConfig
        ModelClass = FastVLMDisTime
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Supported: smolvlm, fastvlm")

    logger.info(f"Creating {model_type} model from {args.model_name_or_path}")
    config_kwargs = dict(
        reg_max=args.reg_max,
        num_time_layers=args.num_time_layers,
    )
    # FastVLM 支持 vision_pool_stride
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

        # Remap keys
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith('time_encoder.') or k.startswith('time_decoder.'):
                remapped[k] = v
            elif k.startswith('model.') or k.startswith('lm_head.'):
                remapped[f'base_model.{k}'] = v
            else:
                remapped[k] = v

        missing, unexpected = model.load_state_dict(remapped, strict=False)
        logger.info(f"Loaded: {len(missing)} missing, {len(unexpected)} unexpected")

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

        critical = [k for k in missing
                    if ('lora_' in k or 'time_encoder' in k or 'time_decoder' in k)
                    and not os.path.exists(distime_path)]
        if critical:
            logger.error(f"CRITICAL missing ({len(critical)}):")
            for k in critical[:5]:
                logger.error(f"  {k}")
        else:
            logger.info("All LoRA and DisTime keys loaded successfully")

    model = model.to(args.device)
    model.eval()
    logger.info(f"Model loaded and ready (model_type={model_type})")
    return model


# ============================================================================
# Input construction – matches training dataset
# ============================================================================

def prepare_inference_inputs(processor, video_path, query, num_frames=32,
                             device="cuda", image_size=384):
    """Prepare inputs using apply_chat_template, same as training."""
    from utils.mm_utils import load_video

    frames, frame_times, duration = load_video(video_path, num_frames=num_frames)
    pil_frames = [Image.fromarray(f.numpy()) for f in frames]

    user_content = []
    for i, frame in enumerate(pil_frames):
        user_content.append({"type": "text", "text": f"{FRAME_TIME_TOKEN}: "})
        user_content.append({"type": "image", "image": frame})
    user_content.append({"type": "text", "text": query})

    messages = [
        {"role": "user", "content": user_content},
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        size={"longest_edge": image_size},
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pixel_values = inputs.get("pixel_values")
    if pixel_values is not None:
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(0)
        pixel_values = pixel_values.to(device)

    frame_times_tensor = torch.tensor(frame_times, dtype=torch.float32).unsqueeze(0).to(device)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'frame_times': frame_times_tensor,
        'duration': duration,
    }


# ============================================================================
# EOS token helper – 按 model_type 获取正确的 eos_token_id
# ============================================================================

def get_eos_token_id(processor, model_type):
    """
    获取 eos_token_id, 按 model_type 区分:
    - SmolVLM: 使用 <|im_end|> 作为单个 eos token
    - FastVLM: 使用 [<|endoftext|>, <|im_end|>] 双 eos (与 fastvlm_distime.generate 对齐)
    """
    if model_type == "fastvlm":
        default_eos = processor.tokenizer.eos_token_id  # <|endoftext|>
        im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != default_eos:
            return [default_eos, im_end_id]
        return [default_eos]
    else:
        # SmolVLM: 单个 eos token
        return processor.tokenizer.convert_tokens_to_ids("<|im_end|>")


# ============================================================================
# QVHighlights Metrics (from DisTime's dist_utils.py)
# ============================================================================

def compute_temporal_iou_batch_cross(spans1, spans2):
    """(N,2) x (M,2) -> (N,M) IoU matrix."""
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
    """Compute AP (detection task) – from DisTime dist_utils."""
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
    """Compute R@1, mIoU – from DisTime dist_utils."""
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


def eval_moment_retrieval(submission, ground_truth, verbose=True):
    """Full MR evaluation with mAP + R@1 + mIoU."""
    ret_metrics = {}
    for name in ["full"]:  # QVH doesn't split by duration like original
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


def eval_submission(submission, ground_truth):
    """Top-level evaluation – returns full metrics dict."""
    eval_metrics = eval_moment_retrieval(submission, ground_truth)

    brief = OrderedDict()
    full_metrics = eval_metrics.get("full", {})
    brief["MR-mAP-avg"] = full_metrics.get("MR-mAP", {}).get("average", 0)
    brief["MR-mAP@0.5"] = full_metrics.get("MR-mAP", {}).get("0.5", 0)
    brief["MR-mAP@0.75"] = full_metrics.get("MR-mAP", {}).get("0.75", 0)
    brief["MR-R1@0.5"] = full_metrics.get("MR-R1", {}).get("0.5", 0)
    brief["MR-R1@0.7"] = full_metrics.get("MR-R1", {}).get("0.7", 0)
    brief["MR-R1-avg"] = full_metrics.get("MR-R1-avg", 0)
    brief["MR-mIoU"] = full_metrics.get("MR-mIoU", 0)
    brief["MR-invalid_pred_num"] = full_metrics.get("MR-invalid_pred_num", 0)

    return {"brief": brief, "full": full_metrics}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DisTime QVHighlights Evaluation")

    # Model type – 与 train.py 对齐
    parser.add_argument("--model_type", type=str, default="smolvlm",
                        choices=["smolvlm", "fastvlm"],
                        help="VLM backend type: 'smolvlm' or 'fastvlm'")

    # Model
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--reg_max", type=int, default=32)
    parser.add_argument("--num_time_layers", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # FastVLM specific
    parser.add_argument("--vision_pool_stride", type=int, default=1,
                        help="Vision token spatial pooling stride (FastVLM only). "
                             "1=no pooling, 2=4x compression, 4=16x compression")

    # Data
    parser.add_argument("--data_file", type=str, required=True,
                        help="QVHighlights annotations (COCO or DisTime format)")
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    # Output
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory. Default: ./results/QVHighlights/{model_type}")

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    # Query template
    parser.add_argument("--query_template", type=str,
                        default="Give you a textual query: '{query}'. When does the described content occur in the video? Please return the timestamp in seconds.",
                        help="Query template. Use {query} placeholder.")

    args = parser.parse_args()

    # 默认按 model_type 分目录, 避免覆盖以前的结果
    if args.out_dir is None:
        args.out_dir = f"./results/QVHighlights/{args.model_type}"

    os.makedirs(args.out_dir, exist_ok=True)

    # 按 model_type 选择 image_size (与 train.py 对齐)
    image_size = MODEL_DEFAULT_IMAGE_SIZE[args.model_type]
    logger.info(f"Using model_type={args.model_type}, image_size={image_size}")

    # ================================================================
    # 1. Load model
    # ================================================================
    model = load_model(args)
    processor = model.processor
    eos_token_id = get_eos_token_id(processor, args.model_type)

    # ================================================================
    # 2. Load data
    # ================================================================
    data = load_qvh_data(args.data_file)

    # ================================================================
    # 3. Run inference
    # ================================================================
    predictions = []      # for saving
    eval_entries = []     # for metrics (DisTime eval_submission format)
    errors = 0
    video_cache = {}      # cache: video_id -> (duration from load_video)

    for idx in tqdm(range(len(data)), desc="QVHighlights"):
        item = data[idx]
        qid = item['qid']
        video_id = item['video_id']
        raw_query = item['query']
        relevant_windows = item['relevant_windows']
        gt_duration = item['duration']

        query = args.query_template.format(query=raw_query.strip())

        # Find video file
        video_path = None
        # Try exact name first (e.g. "NUsG9BgSes0_210.0_360.0.mp4")
        candidate = os.path.join(args.video_root, video_id)
        if os.path.exists(candidate):
            video_path = candidate
        else:
            # Try without .mp4 + extensions
            base = video_id.replace('.mp4', '').replace('.avi', '').replace('.mkv', '')
            for ext in ['.mp4', '.avi', '.mkv', '.webm']:
                candidate = os.path.join(args.video_root, base + ext)
                if os.path.exists(candidate):
                    video_path = candidate
                    break

        if video_path is None:
            logger.warning(f"[{idx}] Video not found: {video_id}")
            errors += 1
            predictions.append({
                "qid": qid, "video_id": video_id, "query": raw_query,
                "pred_relevant_windows": [[-1, -1]],
                "relevant_windows": relevant_windows,
            })
            eval_entries.append({
                "qid": qid,
                "pred_relevant_windows": [[-1, -1]],
                "relevant_windows": relevant_windows,
            })
            continue

        try:
            # Prepare inputs
            inputs = prepare_inference_inputs(
                processor=processor,
                video_path=video_path,
                query=query,
                num_frames=args.num_frames,
                device=args.device,
                image_size=image_size,
            )
            duration = inputs['duration']

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    pixel_values=inputs['pixel_values'],
                    duration=torch.tensor([duration], device=args.device),
                    frame_times=inputs['frame_times'],
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=eos_token_id,
                )

            # Parse predictions – model may predict multiple TIME_STAMPs
            pred_times = outputs.get('pred_times', [None])[0]

            if pred_times is not None and pred_times.shape[0] > 0:
                pred_windows = []
                for j in range(pred_times.shape[0]):
                    s = round(float(pred_times[j, 0].item()), 2)
                    e = round(float(pred_times[j, 1].item()), 2)
                    if s > e:
                        s, e = e, s
                    s = max(0.0, min(s, duration))
                    e = max(0.0, min(e, duration))
                    pred_windows.append([s, e])
            else:
                pred_windows = [[-1, -1]]
                logger.warning(f"[{idx}] {video_id} no TIME_STAMP predicted")

            # Log first few
            if idx < 5:
                gen_text = outputs.get('generated_text', [''])[0]
                logger.info(f"[{idx}] qid={qid}: pred={pred_windows} gt={relevant_windows}")
                logger.info(f"  text: {gen_text[:200]}")

        except Exception as e:
            logger.warning(f"[{idx}] Error {video_id}: {e}")
            import traceback
            traceback.print_exc()
            pred_windows = [[-1, -1]]
            errors += 1

        predictions.append({
            "qid": qid, "video_id": video_id, "query": raw_query,
            "pred_relevant_windows": pred_windows,
            "relevant_windows": relevant_windows,
            "duration": gt_duration,
        })

        eval_entries.append({
            "qid": qid,
            "pred_relevant_windows": pred_windows,
            "relevant_windows": relevant_windows,
        })

        if (idx + 1) % 100 == 0:
            logger.info(f"[{idx+1}/{len(data)}] errors={errors}")

    # ================================================================
    # 4. Compute metrics
    # ================================================================
    print(f"\nComputing metrics...")
    try:
        all_metrics = eval_submission(eval_entries, eval_entries)
        brief = all_metrics["brief"]

        print(f"\n{'='*60}")
        print(f"  QVHighlights Results ({len(eval_entries)} queries, model_type={args.model_type})")
        print(f"{'='*60}")
        print(f"  R@1 IoU=0.5:  {brief.get('MR-R1@0.5', 0):.2f}")
        print(f"  R@1 IoU=0.7:  {brief.get('MR-R1@0.7', 0):.2f}")
        print(f"  R@1 avg:      {brief.get('MR-R1-avg', 0):.2f}")
        print(f"  mAP avg:      {brief.get('MR-mAP-avg', 0):.2f}")
        print(f"  mAP@0.5:      {brief.get('MR-mAP@0.5', 0):.2f}")
        print(f"  mAP@0.75:     {brief.get('MR-mAP@0.75', 0):.2f}")
        print(f"  mIoU:         {brief.get('MR-mIoU', 0)*100:.2f}")
        print(f"  Invalid preds: {brief.get('MR-invalid_pred_num', 0)}")
        print(f"  Errors:       {errors}")
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
    pred_file = os.path.join(args.out_dir, "predictions.json")
    with open(pred_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"Predictions saved to {pred_file}")

    # Also save in DisTime's results.json format for cross-validation
    distime_results = []
    for p in predictions:
        distime_results.append({
            "qid": p["qid"],
            "query": p["query"],
            "prediction": str(p["pred_relevant_windows"]),
            "target": str(p["relevant_windows"]),
            "duration": p.get("duration", 150),
        })
    distime_file = os.path.join(args.out_dir, "results.json")
    with open(distime_file, 'w') as f:
        json.dump(distime_results, f, indent=2)
    logger.info(f"DisTime format results saved to {distime_file}")

    print(f"\nDone: {len(predictions)} predictions, {errors} errors")


if __name__ == "__main__":
    main()
