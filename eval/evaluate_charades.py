"""
Charades-STA evaluation for DisTime (SmolVLM / FastVLM).

Supports both model backends via --model_type.

Directly reads your annotation format:
  [{"video_id": "3MSZA", "start": 24.3, "end": 30.4, "sentence": "..."}, ...]

Usage (SmolVLM):
    python evaluate_charades.py \
        --model_type smolvlm \
        --model_name_or_path /projects/bffz/yzou1/models/SmolVLM2-2.2B-Instruct \
        --checkpoint_dir /path/to/checkpoint \
        --data_file /path/to/charades_sta_test.json \
        --video_root /path/to/videos \
        --num_frames 32

Usage (FastVLM):
    python evaluate_charades.py \
        --model_type fastvlm \
        --model_name_or_path KamilaMila/FastVLM-0.5B \
        --checkpoint_dir /path/to/checkpoint \
        --data_file /path/to/charades_sta_test.json \
        --video_root /path/to/videos \
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

# Add project root to path (same as quick_test.py)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

FRAME_TIME_TOKEN = "<FRAME_TIME>"
TIME_STAMP_TOKEN = "<TIME_STAMP>"

# 每种模型的默认 image_size (与 train.py 对齐)
MODEL_DEFAULT_IMAGE_SIZE = {
    "smolvlm": 384,    # SmolVLM (SigLIP vision encoder)
    "fastvlm": 1024,   # FastVLM (fastvit_mci3, crop_size=1024)
    "molmo2": 378,     # Molmo2 (SigLIP2 So400m/14, 378×378)
}


# ============================================================================
# Data loading – auto-detect format
# ============================================================================

def load_charades_data(data_file):
    """
    Auto-detect and load Charades-STA annotations.
    Returns list of {"video_id", "query", "gt_start", "gt_end"}.
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
                })

    # Flat list formats
    elif isinstance(raw, list) and len(raw) > 0:
        sample = raw[0]
        if "video_id" in sample and "sentence" in sample:
            logger.info("Detected flat format (video_id + sentence)")
            for item in raw:
                items.append({
                    "video_id": item["video_id"], "query": item["sentence"],
                    "gt_start": item.get("start", 0), "gt_end": item.get("end", 0),
                })
        elif "video" in sample and "query" in sample:
            logger.info("Detected DisTime test_float format")
            for item in raw:
                items.append({
                    "video_id": item["video"], "query": item["query"],
                    "gt_start": item.get("start", 0), "gt_end": item.get("end", 0),
                })
        else:
            raise ValueError(f"Unknown list format, keys: {list(sample.keys())}")
    else:
        raise ValueError(f"Cannot parse {data_file}")

    logger.info(f"Loaded {len(items)} annotations ({len(set(it['video_id'] for it in items))} videos)")
    return items


# ============================================================================
# Model loading – 按 model_type 动态选择模型类 (与 train.py 对齐)
# ============================================================================

def load_model(args):
    """
    Load DisTime model for inference.
    Supports both SmolVLM and FastVLM via args.model_type.
    """
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
        raise ValueError(f"Unknown model_type: {model_type}. Supported: smolvlm, fastvlm, molmo2")

    # Step 1: Create config
    logger.info(f"Creating {model_type} model from {args.model_name_or_path}")
    config_kwargs = dict(
        reg_max=args.reg_max,
        num_time_layers=args.num_time_layers,
    )
    # FastVLM 支持 vision_pool_stride
    if model_type == "fastvlm" and hasattr(args, 'vision_pool_stride'):
        config_kwargs['vision_pool_stride'] = args.vision_pool_stride

    distime_config = DisTimeConfig(**config_kwargs)

    # Step 2: Create model
    model = ModelClass(
        model_name_or_path=args.model_name_or_path,
        distime_config=distime_config,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        use_flash_attention=False,
    )

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

        import glob
        from safetensors import safe_open
        sf_path = os.path.join(args.checkpoint_dir, "model.safetensors")
        sf_index_path = os.path.join(args.checkpoint_dir, "model.safetensors.index.json")

        if os.path.exists(sf_path):
            # 单文件 safetensors
            state_dict = {}
            with safe_open(sf_path, framework="pt") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
            logger.info(f"Checkpoint has {len(state_dict)} keys (single safetensors)")
        elif os.path.exists(sf_index_path):
            # 分片 safetensors (model-00001-of-00002.safetensors, ...)
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

        # Remap keys: checkpoint is saved by base_model.save_pretrained(),
        # so keys lack the 'base_model.' prefix that FastVLMDisTime expects.
        # FastVLM (LLaVA) has top-level keys like multi_modal_projector.* that
        # don't start with 'model.' or 'lm_head.', so we need to add 'base_model.'
        # to ALL keys except time_encoder/time_decoder (which belong to FastVLMDisTime itself).
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith('time_encoder.') or k.startswith('time_decoder.'):
                remapped[k] = v
            elif k.startswith('base_model.'):
                # Already has prefix (e.g., from a full FastVLMDisTime checkpoint)
                remapped[k] = v
            else:
                remapped[f'base_model.{k}'] = v

        missing, unexpected = model.load_state_dict(remapped, strict=False)
        logger.info(f"Loaded: {len(missing)} missing, {len(unexpected)} unexpected")

        if missing:
            logger.info(f"Missing keys ({len(missing)}):")
            for k in sorted(missing):
                logger.info(f"  MISSING: {k}")
        if unexpected:
            logger.info(f"Unexpected keys ({len(unexpected)}):")
            for k in sorted(unexpected)[:20]:
                logger.info(f"  UNEXPECTED: {k}")

        # Step 5: Handle tied embeddings (lm_head ↔ embed_tokens)
        # FastVLM (Qwen2) uses tie_word_embeddings=True, so save_pretrained()
        # only saves embed_tokens, not lm_head. We must manually re-tie them.
        if 'base_model.lm_head.weight' in missing:
            embed_weight = model.base_model.get_input_embeddings().weight
            lm_head = model.base_model.get_output_embeddings()
            if lm_head is not None and embed_weight.shape == lm_head.weight.shape:
                lm_head.weight = embed_weight  # re-tie (same tensor)
                logger.info("Re-tied lm_head.weight → embed_tokens.weight "
                           "(checkpoint uses tie_word_embeddings)")
            else:
                logger.error(f"Cannot re-tie lm_head: shape mismatch "
                           f"embed={embed_weight.shape} vs lm_head={lm_head.weight.shape}")

        # Step 6: Load DisTime modules separately
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
            logger.error("time_encoder/time_decoder will use random init weights!")

        # Check critical keys (exclude lm_head since we re-tied it above)
        critical = [k for k in missing
                    if ('lora_' in k or 'time_encoder' in k or 'time_decoder' in k)
                    and not os.path.exists(distime_path)]
        if critical:
            logger.error(f"CRITICAL missing ({len(critical)}):")
            for k in critical[:5]:
                logger.error(f"  {k}")
        else:
            logger.info("All LoRA and DisTime keys loaded successfully")

    device = args.device
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded and ready (model_type={model_type})")
    return model


# ============================================================================
# Input construction – matches training dataset exactly
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


# 每种模型的默认 normalize 类型 (与 dataset.py 对齐)
MODEL_DEFAULT_NORMALIZE = {
    "smolvlm": "siglip",
    "fastvlm": "none",
    "molmo2": "siglip",  # SigLIP2 uses same normalization
}


def _process_frames_molmo2(processor, frames, device="cuda"):
    """Process video frames using Molmo2's video_processor (same as dataset.py).

    Returns:
        dict with pixel_values_videos, video_token_pooling, video_grids
    """
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
    """Get Molmo2's image patch token string (same as dataset.py)."""
    # Try config first
    model_config = getattr(processor, 'config', None) or getattr(processor, 'image_processor', None)
    image_patch_id = None
    if model_config is not None:
        image_patch_id = getattr(model_config, 'image_patch_id', None)
    if image_patch_id is None:
        image_patch_id = 151938  # Molmo2 default
    token = processor.tokenizer.convert_ids_to_tokens(image_patch_id)
    if token is None:
        token = f"<image_patch_{image_patch_id}>"
    return token, image_patch_id


def _get_molmo2_grid_size(processor):
    """Get Molmo2 video pooling grid size (same as dataset.py)."""
    video_proc = getattr(processor, 'video_processor', None)
    if video_proc and hasattr(video_proc, 'pooling_size'):
        ps = video_proc.pooling_size  # e.g. [3, 3]
        raw_per_dim = 378 // 14  # 27
        grid_h = raw_per_dim // ps[0]  # 9
        grid_w = raw_per_dim // ps[1]  # 9
        return grid_h, grid_w
    return 9, 9  # fallback


def prepare_inference_inputs(processor, video_path, query, num_frames=32,
                             device="cuda", image_size=384, model_type="smolvlm",
                             image_seq_len_override=None):
    """
    Prepare inputs for inference, matching training dataset's __getitem__ exactly.

    跟训练完全对齐:
    1. build_transform 处理图像 (与 dataset.py 一致)
    2. 手动拼 frame prompt + conversation (与 dataset._build_frame_prompt / _build_conversation 一致)
    3. processor.tokenizer() 做 tokenization (不用 apply_chat_template)
    4. image_seq_len_override 支持 vision pooling (与 train.py 一致)
    5. Molmo2: 使用 video_processor 处理帧, 走 video path (pixel_values_videos)
    """
    from utils.mm_utils import load_video

    # 1. Load video
    frames, frame_times, duration = load_video(video_path, num_frames=num_frames)

    # 2. Process images
    molmo2_video_inputs = None
    if model_type == "molmo2":
        # Molmo2: use video_processor for proper patches + pooling
        molmo2_video_inputs = _process_frames_molmo2(processor, frames, device=device)
        pixel_values = None  # not used for Molmo2
    else:
        pil_frames = [Image.fromarray(f.numpy()) for f in frames]
        normalize_type = MODEL_DEFAULT_NORMALIZE.get(model_type, "siglip")
        transform = build_transform(image_size, normalize_type=normalize_type)
        pixel_values = torch.stack([transform(f) for f in pil_frames])

    # 3. Calculate image_seq_len (same as dataset.__init__)
    if model_type == "molmo2":
        image_token, _ = _get_molmo2_image_token(processor)
        grid_h, grid_w = _get_molmo2_grid_size(processor)
        image_seq_len = grid_h * grid_w  # 81 (9×9)
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
        # SmolVLM
        image_token = getattr(processor, 'image_token', '<image>')
        image_seq_len = getattr(processor, 'image_seq_len', 81)
        if image_seq_len_override is not None:
            image_seq_len = image_seq_len_override

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
    elif model_type == "molmo2":
        # Molmo2: <frame_start> + <im_patch>×(grid_h*grid_w) + <frame_end>
        all_patch_tokens = image_token * image_seq_len
        frame_strings = []
        for i in range(len(frames)):
            frame_strings.append(f"{FRAME_TIME_TOKEN}: <frame_start>{all_patch_tokens}<frame_end>")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    frame_prompt = "\n".join(frame_strings)
    user_text = f"{frame_prompt}\n{query}"

    # 5. Build conversation with generation prompt (same as dataset._build_conversation)
    if model_type == "smolvlm":
        conversation = f"<|im_start|>User: {user_text}<end_of_utterance>\nAssistant:"
    elif model_type in ("fastvlm", "molmo2"):
        # Both use Qwen chat template
        conversation = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"

    # 6. Tokenize (same as training: processor.tokenizer, NOT apply_chat_template)
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
# EOS token helper – 按 model_type 获取正确的 eos_token_id
# ============================================================================

def get_eos_token_id(processor, model_type):
    """
    获取 eos_token_id, 按 model_type 区分:
    - SmolVLM: 使用 <|im_end|> 作为单个 eos token
    - FastVLM: 使用 [<|endoftext|>, <|im_end|>] 双 eos (与 fastvlm_distime.generate 对齐)
    """
    if model_type in ("fastvlm", "molmo2"):
        # FastVLM (Qwen2) / Molmo2 (Qwen3): dual EOS [<|endoftext|>, <|im_end|>]
        default_eos = processor.tokenizer.eos_token_id  # <|endoftext|>
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

    predictions = []
    distime_results = {}
    errors = 0

    for idx, item in enumerate(tqdm(worker_data, desc=f"W{worker_id}/GPU{gpu_id}",
                                     position=worker_id)):
        video_id = item['video_id']
        raw_query = item['query']
        query = args.query_template.format(query=raw_query)

        video_path = None
        for ext in ['.mp4', '.avi', '.mkv', '.webm', '']:
            candidate = os.path.join(args.video_root, video_id + ext)
            if os.path.exists(candidate):
                video_path = candidate
                break

        if video_path is None:
            errors += 1
            predictions.append({
                "video_id": video_id, "query": raw_query,
                "pred_start": 0.0, "pred_end": 0.0,
                "gt_start": item["gt_start"], "gt_end": item["gt_end"],
            })
            continue

        try:
            inputs = prepare_inference_inputs(
                processor=processor, video_path=video_path,
                query=query, num_frames=args.num_frames,
                device=device, image_size=image_size,
                model_type=args.model_type,
                image_seq_len_override=image_seq_len_override,
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

            pred_times = outputs.get('pred_times', [None])[0]

            if pred_times is not None and pred_times.shape[0] > 0:
                start = round(float(pred_times[0, 0].item()), 2)
                end = round(float(pred_times[0, 1].item()), 2)
                if start > end:
                    start, end = end, start
                start = max(0.0, min(start, duration))
                end = max(0.0, min(end, duration))
            else:
                start, end = 0.0, duration
                wlog.warning(f"[{idx}] {video_id} no TIME_STAMP predicted")

        except Exception as e:
            wlog.warning(f"Error {video_id}: {e}")
            start, end = 0.0, 30.0
            duration = 30.0
            errors += 1

        predictions.append({
            "video_id": video_id, "query": raw_query,
            "pred_start": start, "pred_end": end,
            "gt_start": item["gt_start"], "gt_end": item["gt_end"],
        })

        if video_id not in distime_results:
            distime_results[video_id] = {}
        distime_results[video_id][raw_query] = {
            "start": start, "end": end, "conf": 1,
        }

        if (idx + 1) % 20 == 0:
            torch.cuda.empty_cache()

    out_path = os.path.join(tmp_dir, f"worker_{worker_id}.json")
    with open(out_path, 'w') as f:
        json.dump({
            "predictions": predictions,
            "distime_results": distime_results,
            "errors": errors,
        }, f, ensure_ascii=False)
    wlog.info(f"Done: {len(worker_data)} samples, {errors} errors -> {out_path}")

    # Explicit cleanup to prevent CUDA context hang on shared GPUs
    del model
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DisTime Charades-STA Evaluation")

    # Model type – 与 train.py 对齐
    parser.add_argument("--model_type", type=str, default="smolvlm",
                        choices=["smolvlm", "fastvlm", "molmo2"],
                        help="VLM backend type: 'smolvlm', 'fastvlm', or 'molmo2'")

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

    # Image size override (different SmolVLM sizes use different vision encoders:
    #   SmolVLM2-2.2B: 384 (SigLIP-SO400M-patch14-384)
    #   SmolVLM2-500M: 512 (SigLIP-base-patch16-512))
    parser.add_argument("--image_size", type=int, default=None,
                        help="Override image_size for vision encoder. If None, use MODEL_DEFAULT. "
                             "SmolVLM2-2.2B=384, SmolVLM2-500M=512, FastVLM=1024")

    # SmolVLM image_seq_len override (different model sizes have different values:
    #   SmolVLM2-2.2B: 81, SmolVLM2-500M: 64)
    parser.add_argument("--image_seq_len", type=int, default=None,
                        help="Override image_seq_len (SmolVLM). If None, read from processor. "
                             "SmolVLM2-2.2B=81, SmolVLM2-500M=64")

    # Data
    parser.add_argument("--data_file", type=str, required=True,
                        help="Charades-STA test annotations (any supported format)")
    parser.add_argument("--video_root", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max number of samples to evaluate (for debugging)")

    # Output (默认按 model_type 分目录, 避免覆盖)
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory. Default: ./results/Charades-STA/{model_type}")

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

    # 默认按 model_type 分目录, 避免覆盖以前的结果
    if args.out_dir is None:
        args.out_dir = f"./results/Charades-STA/{args.model_type}"

    os.makedirs(args.out_dir, exist_ok=True)

    # 按 model_type 选择 image_size, 支持 CLI override
    if args.image_size is not None:
        image_size = args.image_size
        logger.info(f"Using CLI image_size override: {image_size}")
    else:
        image_size = MODEL_DEFAULT_IMAGE_SIZE[args.model_type]
    logger.info(f"Using model_type={args.model_type}, image_size={image_size}")

    # ================================================================
    # 1. Load data
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
    # 2. Run inference (single-process or multi-process)
    # ================================================================
    num_gpus = args.num_gpus
    total_workers = num_gpus * args.workers_per_gpu

    if total_workers <= 1:
        # ---- Single-process mode ----
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
        # SmolVLM/Molmo2: CLI --image_seq_len 覆盖
        # FastVLM: vision_pool_stride 覆盖
        if args.image_seq_len is not None:
            image_seq_len_override = args.image_seq_len
            logger.info(f"Using CLI image_seq_len override: {image_seq_len_override}")
        elif args.model_type == "fastvlm" and args.vision_pool_stride > 1:
            base_image_seq_len = (image_size // 64) ** 2  # (1024/64)^2 = 256
            image_seq_len_override = base_image_seq_len // (args.vision_pool_stride ** 2)
            logger.info(f"Vision pooling: stride={args.vision_pool_stride}, "
                        f"image_seq_len {base_image_seq_len} → {image_seq_len_override}")
        else:
            image_seq_len_override = None

        # Log actual processor value for debugging
        proc_isl = getattr(processor, 'image_seq_len', 'N/A')
        logger.info(f"processor.image_seq_len={proc_isl}, override={image_seq_len_override}")

        eos_token_id = get_eos_token_id(processor, args.model_type)

        predictions = []
        distime_results = {}  # {vid: {query: {start, end, conf}}}
        errors = 0

        for idx in tqdm(range(len(data)), desc=f"Charades-STA ({args.model_type})"):
            item = data[idx]
            video_id = item['video_id']
            raw_query = item['query']

            # Apply query template
            query = args.query_template.format(query=raw_query)

            # Find video file
            video_path = None
            for ext in ['.mp4', '.avi', '.mkv', '.webm', '']:
                candidate = os.path.join(args.video_root, video_id + ext)
                if os.path.exists(candidate):
                    video_path = candidate
                    break

            if video_path is None:
                logger.warning(f"[{idx}] Video not found: {video_id}")
                errors += 1
                predictions.append({
                    "video_id": video_id, "query": raw_query,
                    "pred_start": 0.0, "pred_end": 0.0,
                    "gt_start": item["gt_start"], "gt_end": item["gt_end"],
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

                pred_times = outputs.get('pred_times', [None])[0]

                if pred_times is not None and pred_times.shape[0] > 0:
                    start = round(float(pred_times[0, 0].item()), 2)
                    end = round(float(pred_times[0, 1].item()), 2)
                    if start > end:
                        start, end = end, start
                    start = max(0.0, min(start, duration))
                    end = max(0.0, min(end, duration))
                else:
                    start, end = 0.0, duration
                    logger.warning(f"[{idx}] {video_id} no TIME_STAMP predicted")

                if idx < 5:
                    gen_text = outputs.get('generated_text', [''])[0]
                    logger.info(f"[{idx}] {video_id}: pred=[{start:.2f}, {end:.2f}] "
                                f"gt=[{item['gt_start']:.2f}, {item['gt_end']:.2f}]")
                    logger.info(f"  text: {gen_text[:200]}")

            except Exception as e:
                logger.warning(f"[{idx}] Error {video_id}: {e}")
                import traceback
                traceback.print_exc()
                start, end = 0.0, 30.0
                duration = 30.0
                errors += 1

            predictions.append({
                "video_id": video_id, "query": raw_query,
                "pred_start": start, "pred_end": end,
                "gt_start": item["gt_start"], "gt_end": item["gt_end"],
            })

            if video_id not in distime_results:
                distime_results[video_id] = {}
            distime_results[video_id][raw_query] = {
                "start": start, "end": end, "conf": 1,
            }

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

        tmp_dir = tempfile.mkdtemp(prefix="charades_distime_eval_")
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

        for i, p in enumerate(processes):
            p.join()  # Wait indefinitely for worker to finish
            if p.exitcode is None:
                logger.warning(f"Worker {i} did not finish, terminating...")
                p.terminate()
                p.join(timeout=60)

        for i, p in enumerate(processes):
            if p.exitcode != 0:
                logger.error(f"Worker {i} exited with code {p.exitcode}")

        # Merge results
        predictions = []
        distime_results = {}
        errors = 0

        for w in range(total_workers):
            result_path = os.path.join(tmp_dir, f"worker_{w}.json")
            if not os.path.exists(result_path):
                logger.error(f"Worker {w} result file missing: {result_path}")
                continue
            with open(result_path, 'r') as f:
                worker_result = json.load(f)
            predictions.extend(worker_result["predictions"])
            # Merge distime_results dicts (nested {vid: {query: {...}}})
            for vid, queries in worker_result["distime_results"].items():
                if vid not in distime_results:
                    distime_results[vid] = {}
                distime_results[vid].update(queries)
            errors += worker_result["errors"]

        logger.info(f"Merged results: {len(predictions)} samples, {errors} errors")

    # ================================================================
    # 4. Compute metrics
    # ================================================================
    if has_gt:
        metrics = compute_metrics(predictions)
        print(f"\n{'='*60}")
        print(f"  Charades-STA Results ({len(predictions)} samples, model_type={args.model_type})")
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
    pred_file = os.path.join(args.out_dir, "predictions.json")
    with open(pred_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"Predictions saved to {pred_file}")

    # DisTime evaluator format
    distime_file = os.path.join(args.out_dir, "results.json")
    with open(distime_file, 'w') as f:
        json.dump(distime_results, f, indent=2)
    logger.info(f"DisTime format results saved to {distime_file}")

    print(f"\nDone: {len(predictions)} predictions, {errors} errors")


if __name__ == "__main__":
    main()
