"""
Efficiency benchmark: throughput, latency, peak memory for three paradigms.

Picks 10 Charades-STA videos, runs inference on each, reports averages.

Usage:
    python scripts/benchmark_efficiency.py          # runs all 3 paradigms
    python scripts/benchmark_efficiency.py --paradigm distime   # single paradigm
"""

import os
import sys
import json
import time
import argparse
import logging

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Fix decord compatibility: some versions lack decord.bridge
import decord
if not hasattr(decord, 'bridge'):
    class _FakeBridge:
        @staticmethod
        def set_bridge(*a, **kw):
            pass
    decord.bridge = _FakeBridge()

# ============================================================================
# Paths — 根据你的集群配置
# ============================================================================
DEFAULT_CONFIG = {
    "model_name_or_path": "/projects/bffz/yzou1/models/SmolVLM2-2.2B-Instruct",
    "distime_checkpoint": "/projects/bffh/yzou1/models/smolvlm_distime/smolvlm_distime_1888038_2nodes",
    "trace_checkpoint": "/work/hdd/bffh/yzou1/models/smolvlm_trace/smolvlm_trace_1909494_2nodes",
    "anno_file": "/work/hdd/bffh/yzou1/data/Charades/charades_sta_test.json",
    "video_root": "/work/hdd/bffh/yzou1/data/Charades/videos",
}

FRAME_TIME_TOKEN = "<FRAME_TIME>"
TIME_STAMP_TOKEN = "<TIME_STAMP>"
NUM_FRAMES = 32
MAX_NEW_TOKENS = 256
QUERY_TEMPLATE = "Give you a textual query: '{query}'. When does the described content occur in the video? Please return the timestamp in seconds."

# 每种模型的默认配置 (与 evaluate_charades.py / train.py 对齐)
MODEL_DEFAULT_IMAGE_SIZE = {
    "smolvlm": 384,    # SmolVLM-2.2B (SigLIP-SO400M-patch14-384)
    "fastvlm": 1024,   # FastVLM (fastvit_mci3, crop_size=1024)
}

MODEL_DEFAULT_NORMALIZE = {
    "smolvlm": "siglip",
    "fastvlm": "none",
}

# ============================================================================
# Image transform (与 evaluate_charades.py 对齐)
# ============================================================================
def build_transform(input_size=384, normalize_type="siglip"):
    from torchvision import transforms
    SIGLIP_MEAN, SIGLIP_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    transform_list = [
        transforms.Resize((input_size, input_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ]
    if normalize_type == "siglip":
        transform_list.append(transforms.Normalize(mean=SIGLIP_MEAN, std=SIGLIP_STD))
    # fastvlm: no normalization
    return transforms.Compose(transform_list)


# ============================================================================
# 加载 Charades 标注, 选 10 个不同视频
# ============================================================================
def load_samples(anno_file, video_root, n=10, seed=42):
    with open(anno_file) as f:
        data = json.load(f)

    # 按 video_id 去重, 每个视频只取一条 query
    seen = set()
    unique = []
    for item in data:
        vid = item.get("video_id") or item.get("vid")
        if vid not in seen:
            seen.add(vid)
            unique.append(item)

    rng = np.random.RandomState(seed)
    rng.shuffle(unique)

    samples = []
    for item in unique:
        if len(samples) >= n:
            break
        vid = item.get("video_id") or item.get("vid")
        query_raw = item.get("query") or item.get("sentence") or item.get("description", "")
        gt_start = item.get("gt_start") or item.get("start", 0)
        gt_end = item.get("gt_end") or item.get("end", 0)

        video_path = None
        for ext in ['.mp4', '.avi', '.mkv', '.webm', '']:
            candidate = os.path.join(video_root, vid + ext)
            if os.path.exists(candidate):
                video_path = candidate
                break
        if video_path is None:
            continue

        samples.append({
            "video_id": vid,
            "query": query_raw,
            "gt_start": gt_start,
            "gt_end": gt_end,
            "video_path": video_path,
        })

    logger.info(f"Selected {len(samples)} videos for benchmark")
    return samples


# ============================================================================
# 准备推理输入 (与 evaluate_charades.py prepare_inference_inputs 对齐)
# ============================================================================
def prepare_inputs(processor, video_path, query, device="cuda",
                    paradigm="text", sync_token_id=None,
                    model_type="smolvlm", image_size=384,
                    image_seq_len_override=None):
    """
    准备推理输入, 严格对齐各 eval 脚本:
    - Text/DisTime: evaluate_charades.py 的 prepare_inference_inputs
    - TRACE: evaluate_charades_trace.py 的 prepare_inference_inputs
    支持 SmolVLM / FastVLM 两种模型格式.
    """
    import decord
    if hasattr(decord, 'bridge'):
        decord.bridge.set_bridge('torch')
    from utils.mm_utils import load_video

    frames, frame_times, duration = load_video(video_path, num_frames=NUM_FRAMES)
    pil_frames = [Image.fromarray(f.numpy()) for f in frames]

    normalize_type = MODEL_DEFAULT_NORMALIZE.get(model_type, "siglip")
    transform = build_transform(image_size, normalize_type=normalize_type)
    pixel_values = torch.stack([transform(f) for f in pil_frames])

    image_token = getattr(processor, 'image_token', '<image>')

    # Calculate image_seq_len (与 evaluate_charades.py 一致)
    if model_type == "fastvlm":
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
        image_seq_len = getattr(processor, 'image_seq_len', 81)
        if image_seq_len_override is not None:
            image_seq_len = image_seq_len_override

    # Build frame prompt (与 evaluate_charades.py 一致)
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

    frame_prompt = "\n".join(frame_strings)
    formatted_query = QUERY_TEMPLATE.format(query=query)
    user_text = f"{frame_prompt}\n{formatted_query}"

    # Build conversation (与 evaluate_charades.py / evaluate_charades_trace.py 一致)
    if model_type == "smolvlm":
        if paradigm == "trace":
            conversation = f"<|im_start|>User: {user_text}<end_of_utterance>\nAssistant:<sync>"
        else:
            conversation = f"<|im_start|>User: {user_text}<end_of_utterance>\nAssistant:"
    elif model_type == "fastvlm":
        if paradigm == "trace":
            conversation = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n<sync>"
        else:
            conversation = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"

    encoded = processor.tokenizer(conversation, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded.input_ids

    # TRACE: 替换 <sync> placeholder → model.sync_token_id (extended vocab)
    if paradigm == "trace" and sync_token_id is not None:
        tok_sync_id = processor.tokenizer.convert_tokens_to_ids("<sync>")
        input_ids = input_ids.clone()
        input_ids[input_ids == tok_sync_id] = sync_token_id

    result = {
        'input_ids': input_ids.to(device),
        'attention_mask': encoded.attention_mask.to(device),
        'pixel_values': pixel_values.unsqueeze(0).to(device=device, dtype=torch.bfloat16),
        'duration': duration,
    }

    # DisTime 需要 frame_times, TRACE 不需要
    if paradigm != "trace":
        result['frame_times'] = torch.tensor(frame_times, dtype=torch.float32).unsqueeze(0).to(device)

    return result


# ============================================================================
# Model loading — 三个范式
# ============================================================================

def load_distime_model(config, device, model_type="smolvlm", vision_pool_stride=1):
    """Load DisTime model, supports both SmolVLM and FastVLM."""
    if model_type == "smolvlm":
        from models.smolvlm_distime import SmolVLMDisTime, DisTimeConfig
        ModelClass = SmolVLMDisTime
    elif model_type == "fastvlm":
        from models.fastvlm_distime import FastVLMDisTime, DisTimeConfig
        ModelClass = FastVLMDisTime
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    config_kwargs = dict(reg_max=32, num_time_layers=3)
    if model_type == "fastvlm" and vision_pool_stride > 1:
        config_kwargs['vision_pool_stride'] = vision_pool_stride

    distime_config = DisTimeConfig(**config_kwargs)
    model = ModelClass(
        model_name_or_path=config["model_name_or_path"],
        distime_config=distime_config,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        use_flash_attention=False,
    )
    model.setup_training(use_lora=True, lora_r=16, lora_alpha=32, freeze_vision=True)

    ckpt_dir = config["distime_checkpoint"]
    from safetensors import safe_open
    sf_path = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.exists(sf_path):
        state_dict = {}
        with safe_open(sf_path, framework="pt") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
    else:
        pt_path = os.path.join(ckpt_dir, "pytorch_model.bin")
        if os.path.exists(pt_path):
            state_dict = torch.load(pt_path, map_location="cpu")
        else:
            from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
            state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_dir)

    remapped = {}
    for k, v in state_dict.items():
        if k.startswith('time_encoder.') or k.startswith('time_decoder.'):
            remapped[k] = v
        elif k.startswith('base_model.'):
            remapped[k] = v
        else:
            remapped[f'base_model.{k}'] = v

    missing, _ = model.load_state_dict(remapped, strict=False)

    # Handle tied embeddings (FastVLM/Qwen2 uses tie_word_embeddings=True)
    if 'base_model.lm_head.weight' in missing:
        embed_weight = model.base_model.get_input_embeddings().weight
        lm_head = model.base_model.get_output_embeddings()
        if lm_head is not None and embed_weight.shape == lm_head.weight.shape:
            lm_head.weight = embed_weight
            logger.info("Re-tied lm_head.weight → embed_tokens.weight")

    # Load DisTime modules
    distime_path = os.path.join(ckpt_dir, 'distime_modules.pt')
    if os.path.exists(distime_path):
        distime_state = torch.load(distime_path, map_location='cpu')
        model.time_encoder.load_state_dict(distime_state['time_encoder'])
        model.time_decoder.load_state_dict(distime_state['time_decoder'])
        model.time_encoder.to(torch.bfloat16)
        model.time_decoder.to(torch.bfloat16)
        logger.info(f"Loaded DisTime modules from {distime_path}")
    else:
        logger.error(f"distime_modules.pt NOT FOUND in {ckpt_dir}")

    model = model.to(device).eval()
    return model


def load_trace_model(config, device):
    from models.smolvlm_trace import SmolVLMTrace, TraceConfig

    trace_config = TraceConfig()
    model = SmolVLMTrace(
        model_name_or_path=config["model_name_or_path"],
        trace_config=trace_config,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        use_flash_attention=False,
    )
    model.setup_training(use_lora=True, lora_r=16, lora_alpha=32, freeze_vision=True)

    ckpt_dir = config["trace_checkpoint"]
    from safetensors import safe_open
    sf_path = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.exists(sf_path):
        state_dict = {}
        with safe_open(sf_path, framework="pt") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
    else:
        pt_path = os.path.join(ckpt_dir, "pytorch_model.bin")
        if os.path.exists(pt_path):
            state_dict = torch.load(pt_path, map_location="cpu")
        else:
            from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
            state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_dir)

    remapped = {}
    for k, v in state_dict.items():
        if any(k.startswith(p) for p in ['time_tower.', 'score_tower.', 'sync_tower.',
                                          'time_head.', 'score_head.', 'sync_head.']):
            remapped[k] = v
        elif k.startswith('model.') or k.startswith('lm_head.'):
            remapped[f'base_model.{k}'] = v
        else:
            remapped[k] = v
    model.load_state_dict(remapped, strict=False)

    # Load TRACE modules
    trace_path = os.path.join(ckpt_dir, 'trace_modules.pt')
    if os.path.exists(trace_path):
        trace_state = torch.load(trace_path, map_location='cpu')
        model.time_tower.load_state_dict(trace_state['time_tower'])
        model.score_tower.load_state_dict(trace_state['score_tower'])
        model.sync_tower.load_state_dict(trace_state['sync_tower'])
        model.time_head.load_state_dict(trace_state['time_head'])
        model.score_head.load_state_dict(trace_state['score_head'])
        model.sync_head.load_state_dict(trace_state['sync_head'])
        for m in [model.time_tower, model.score_tower, model.sync_tower,
                  model.time_head, model.score_head, model.sync_head]:
            m.to(torch.bfloat16)

    model = model.to(device).eval()
    return model


def load_text_model(config, device):
    """Text numeral: 未微调的 SmolVLM2-2.2B-Instruct + LoRA (公平对比 throughput)"""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from models.manual_lora import apply_lora_to_model

    # 先尝试 sdpa, 某些 vision tower (TimmWrapper) 不支持则降级到 eager
    try:
        base_model = AutoModelForImageTextToText.from_pretrained(
            config["model_name_or_path"],
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
    except ValueError:
        logger.info("SDPA not supported by vision tower, falling back to eager attention")
        base_model = AutoModelForImageTextToText.from_pretrained(
            config["model_name_or_path"],
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            attn_implementation="eager",
            trust_remote_code=True,
        )
    processor = AutoProcessor.from_pretrained(
        config["model_name_or_path"], trust_remote_code=True,
    )

    # 冻结全部参数, 再加 LoRA — 与 DisTime/TRACE 保持一致
    for param in base_model.parameters():
        param.requires_grad = False

    lora_target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    apply_lora_to_model(base_model, lora_target_modules,
                        r=16, lora_alpha=32, lora_dropout=0.05)

    base_model = base_model.to(device).eval()
    return base_model, processor


# ============================================================================
# 推理 wrappers
# ============================================================================

def get_eos_token_id(processor, model_type):
    """获取 eos_token_id, 按 model_type 区分 (与 evaluate_charades.py 一致)."""
    if model_type == "fastvlm":
        default_eos = processor.tokenizer.eos_token_id
        im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != default_eos:
            return [default_eos, im_end_id]
        return [default_eos]
    else:
        # SmolVLM: 同时加 <end_of_utterance> 和 <|im_end|> 作为 stop token
        # debug 显示 2.2B DisTime 模型实际生成 <|im_end|> 而非 <end_of_utterance>
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


def run_distime_inference(model, inputs, model_type="smolvlm"):
    eos_token_id = get_eos_token_id(model.processor, model_type)
    logger.info(f"[DEBUG] eos_token_id={eos_token_id}, type={type(eos_token_id).__name__}, "
                f"default_eos={model.processor.tokenizer.eos_token_id}, "
                f"eou_id={model.processor.tokenizer.convert_tokens_to_ids('<end_of_utterance>')}")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            pixel_values=inputs['pixel_values'],
            duration=torch.tensor([inputs['duration']], device=inputs['input_ids'].device),
            frame_times=inputs['frame_times'],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=eos_token_id,
        )
    # Debug: decode generated text to see what the model actually outputs
    if isinstance(outputs, dict) and 'generated_ids' in outputs:
        gen_ids = outputs['generated_ids'][0, inputs['input_ids'].shape[-1]:]
        gen_text = model.processor.tokenizer.decode(gen_ids, skip_special_tokens=False)
        logger.info(f"[DEBUG] generated text (first 200 chars): {gen_text[:200]}")
        # Show token IDs of last 10 generated tokens
        logger.info(f"[DEBUG] last 10 token ids: {gen_ids[-10:].tolist()}")
    return outputs


def run_trace_inference(model, inputs):
    # TRACE 用 tokenizer 默认 eos_token_id, 不是 <|im_end|>
    # 与 evaluate_charades_trace.py get_eos_token_id() 一致
    eos_token_id = model.processor.tokenizer.eos_token_id
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            pixel_values=inputs['pixel_values'],
            duration=torch.tensor([inputs['duration']], device=inputs['input_ids'].device),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=eos_token_id,
            heads=[1],  # 从 time head 开始解码, 与 evaluate_charades_trace.py 一致
        )
    # 记录实际生成的 token 数, 方便 debug 延迟
    if isinstance(outputs, dict) and 'generated_ids' in outputs:
        gen_len = outputs['generated_ids'].shape[-1] - inputs['input_ids'].shape[-1]
        logger.debug(f"TRACE generated {gen_len} tokens")
    return outputs


def run_text_inference(model, inputs, eos_token_id=None, model_type="smolvlm"):
    pixel_values = inputs['pixel_values']
    # FastVLM: 原生 HF 模型的 vision pipeline 与手动 prompt 不兼容, 不传 pixel_values
    # SmolVLM (Idefics3): 原生就能处理 5D pixel_values, 不需要手动 reshape
    if model_type == "fastvlm":
        pixel_values = None
    gen_kwargs = dict(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    if pixel_values is not None:
        gen_kwargs['pixel_values'] = pixel_values
    if eos_token_id is not None:
        gen_kwargs['eos_token_id'] = eos_token_id
    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)
    return outputs


# ============================================================================
# 参数计数
# ============================================================================

def count_extra_params_distime(model):
    enc = sum(p.numel() for p in model.time_encoder.parameters())
    dec = sum(p.numel() for p in model.time_decoder.parameters())
    return enc + dec, {"time_encoder": enc, "time_decoder": dec}

def count_extra_params_trace(model):
    breakdown = {}
    total = 0
    for name in ["time_tower", "score_tower", "sync_tower",
                  "time_head", "score_head", "sync_head"]:
        n = sum(p.numel() for p in getattr(model, name).parameters())
        breakdown[name] = n
        total += n
    return total, breakdown


# ============================================================================
# Benchmark core
# ============================================================================

def benchmark_paradigm(paradigm, config, samples, device,
                        model_type="smolvlm", image_size=384,
                        image_seq_len_override=None, vision_pool_stride=1):
    """对一个范式跑完整 benchmark, 返回 results dict."""
    logger.info(f"\n{'='*60}")
    logger.info(f"BENCHMARKING: {paradigm.upper()} (model_type={model_type})")
    logger.info(f"{'='*60}")

    results = {"paradigm": paradigm, "model_type": model_type}

    # --- 1. Load model ---
    logger.info("Loading model...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    sync_token_id = None  # 仅 TRACE 使用

    if paradigm == "distime":
        model = load_distime_model(config, device, model_type=model_type,
                                    vision_pool_stride=vision_pool_stride)
        processor = model.processor
        extra, breakdown = count_extra_params_distime(model)
        infer_fn = lambda m, inp: run_distime_inference(m, inp, model_type=model_type)
    elif paradigm == "trace":
        model = load_trace_model(config, device)
        processor = model.processor
        sync_token_id = model.sync_token_id
        extra, breakdown = count_extra_params_trace(model)
        infer_fn = run_trace_inference
    elif paradigm == "text":
        model, processor = load_text_model(config, device)
        extra, breakdown = 0, {}
        text_eos = get_eos_token_id(processor, model_type)
        infer_fn = lambda m, inp: run_text_inference(m, inp, eos_token_id=text_eos, model_type=model_type)

    results["extra_params"] = extra
    results["extra_params_breakdown"] = breakdown

    if extra >= 1e6:
        extra_str = f"{extra/1e6:.1f}M"
    elif extra > 0:
        extra_str = f"{extra/1e3:.0f}K"
    else:
        extra_str = "0"
    logger.info(f"Extra params: {extra_str} ({extra:,})")

    model_mem = torch.cuda.memory_allocated(device) / 1e9
    logger.info(f"Model loaded, GPU mem: {model_mem:.2f} GB")

    # Common kwargs for prepare_inputs
    prep_kwargs = dict(
        model_type=model_type, image_size=image_size,
        image_seq_len_override=image_seq_len_override,
    )

    # --- 2. Warmup (1 sample) ---
    logger.info("Warmup...")
    warmup_inputs = prepare_inputs(processor, samples[0]["video_path"],
                                   samples[0]["query"], device=str(device),
                                   paradigm=paradigm, sync_token_id=sync_token_id,
                                   **prep_kwargs)
    for _ in range(2):
        infer_fn(model, warmup_inputs)
    torch.cuda.synchronize()
    del warmup_inputs

    # --- 3. Inference latency (10 samples) ---
    logger.info(f"Running inference on {len(samples)} videos...")
    torch.cuda.reset_peak_memory_stats()

    latencies = []
    gen_token_counts = []
    for sample in tqdm(samples, desc=f"{paradigm} inference"):
        inputs = prepare_inputs(processor, sample["video_path"],
                                sample["query"], device=str(device),
                                paradigm=paradigm, sync_token_id=sync_token_id,
                                **prep_kwargs)
        input_len = inputs['input_ids'].shape[-1]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = infer_fn(model, inputs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

        # 统计实际生成的 token 数
        if isinstance(outputs, dict) and 'generated_ids' in outputs:
            gen_len = outputs['generated_ids'].shape[-1] - input_len
        elif hasattr(outputs, 'shape'):
            gen_len = outputs.shape[-1] - input_len
        elif isinstance(outputs, dict) and 'sequences' in outputs:
            gen_len = outputs['sequences'].shape[-1] - input_len
        else:
            gen_len = -1
        gen_token_counts.append(gen_len)

        del inputs, outputs

    if gen_token_counts and gen_token_counts[0] >= 0:
        logger.info(f"Generated tokens: mean={np.mean(gen_token_counts):.0f}, "
                     f"range=[{min(gen_token_counts)}, {max(gen_token_counts)}]")

    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9

    results["latency_mean_ms"] = round(float(np.mean(latencies)), 1)
    results["latency_std_ms"] = round(float(np.std(latencies)), 1)
    results["latency_min_ms"] = round(float(np.min(latencies)), 1)
    results["latency_max_ms"] = round(float(np.max(latencies)), 1)
    results["latency_all_ms"] = [round(t, 1) for t in latencies]
    results["peak_mem_gb"] = round(peak_mem, 2)

    logger.info(f"Latency: {results['latency_mean_ms']:.1f} ± {results['latency_std_ms']:.1f} ms/query")
    logger.info(f"  range: [{results['latency_min_ms']:.1f}, {results['latency_max_ms']:.1f}] ms")
    logger.info(f"Peak GPU mem: {peak_mem:.2f} GB")

    # --- 4. Training throughput (forward + backward, 5 steps) ---
    logger.info("Measuring training throughput...")
    try:
        throughput = measure_training_throughput(model, paradigm, processor,
                                                 samples[0], device,
                                                 sync_token_id=sync_token_id,
                                                 **prep_kwargs)
        results["throughput_samp_s"] = round(throughput, 2)
        logger.info(f"Throughput: {throughput:.2f} samples/s")
    except Exception as e:
        logger.warning(f"Throughput measurement failed: {e}")
        results["throughput_samp_s"] = None

    # --- Cleanup ---
    del model
    torch.cuda.empty_cache()

    return results


def measure_training_throughput(model, paradigm, processor, sample, device,
                                 num_warmup=2, num_steps=5, sync_token_id=None,
                                 model_type="smolvlm", image_size=384,
                                 image_seq_len_override=None):
    """模拟单卡训练 forward+backward, 返回 samples/s"""
    model.train()

    inputs = prepare_inputs(processor, sample["video_path"],
                            sample["query"], device=str(device),
                            paradigm=paradigm, sync_token_id=sync_token_id,
                            model_type=model_type, image_size=image_size,
                            image_seq_len_override=image_seq_len_override)
    # HuggingFace 原生模型 (text paradigm) 不处理 5D pixel_values
    # FastVLM: 不传 pixel_values; SmolVLM (Idefics3): 原生支持 5D, 不需要 reshape
    if paradigm == "text" and model_type == "fastvlm":
        inputs['pixel_values'] = None
    seq_len = inputs['input_ids'].shape[1]
    labels = inputs['input_ids'].clone()
    labels[:, :seq_len // 2] = -100

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        logger.warning(f"No trainable params found for {paradigm}, skipping throughput")
        model.eval()
        return 0.0

    logger.info(f"Trainable params for throughput: {sum(p.numel() for p in trainable):,}")
    optimizer = torch.optim.AdamW(trainable, lr=1e-5)

    def step():
        optimizer.zero_grad()
        if paradigm == "text":
            out = model(input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        pixel_values=inputs['pixel_values'],
                        labels=labels)
            loss = out.loss
        else:
            out = model(input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        pixel_values=inputs['pixel_values'],
                        labels=labels)
            loss = out.get('loss', out.get('total_loss')) if isinstance(out, dict) else out.loss
        if loss is not None:
            loss.backward()
            optimizer.step()

    # Warmup
    for _ in range(num_warmup):
        step()
    torch.cuda.synchronize()

    # Timed
    times = []
    for _ in range(num_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    model.eval()
    return 1.0 / np.mean(times)  # batch_size=1, so samples/s = 1/avg_time


# ============================================================================
# Main
# ============================================================================

def main():
    print("[benchmark_efficiency] Starting...", flush=True)  # early diagnostic

    parser = argparse.ArgumentParser()
    parser.add_argument("--paradigm", type=str, default=None, nargs='+',
                        choices=["text", "distime", "trace"],
                        help="Run specific paradigm(s), e.g. --paradigm text distime. Default: all applicable.")
    parser.add_argument("--model_type", type=str, default="smolvlm",
                        choices=["smolvlm", "fastvlm"],
                        help="VLM backend type (smolvlm or fastvlm)")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--distime_checkpoint", type=str, default=None)
    parser.add_argument("--trace_checkpoint", type=str, default=None)
    parser.add_argument("--anno_file", type=str, default=None)
    parser.add_argument("--video_root", type=str, default=None)
    parser.add_argument("--num_videos", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_name", type=str, default="efficiency_results",
                        help="Output JSON filename (without .json). Default: efficiency_results")
    parser.add_argument("--device", type=str, default="cuda")

    # Image / vision overrides
    parser.add_argument("--image_size", type=int, default=None,
                        help="Override image_size. SmolVLM-2.2B=384, SmolVLM-500M=512, FastVLM=1024")
    parser.add_argument("--image_seq_len", type=int, default=None,
                        help="Override image_seq_len. SmolVLM-2.2B=81, SmolVLM-500M=64")
    parser.add_argument("--vision_pool_stride", type=int, default=1,
                        help="Vision token spatial pooling stride (FastVLM only)")

    args = parser.parse_args()

    model_type = args.model_type

    # Determine image_size
    if args.image_size is not None:
        image_size = args.image_size
    else:
        image_size = MODEL_DEFAULT_IMAGE_SIZE.get(model_type, 384)
    logger.info(f"model_type={model_type}, image_size={image_size}")

    # Determine image_seq_len_override
    if args.image_seq_len is not None:
        image_seq_len_override = args.image_seq_len
    elif model_type == "fastvlm" and args.vision_pool_stride > 1:
        base_isl = (image_size // 64) ** 2
        image_seq_len_override = base_isl // (args.vision_pool_stride ** 2)
        logger.info(f"Vision pooling: stride={args.vision_pool_stride}, "
                    f"image_seq_len {base_isl} → {image_seq_len_override}")
    else:
        image_seq_len_override = None

    # Merge with defaults
    config = dict(DEFAULT_CONFIG)
    for key in ["model_name_or_path", "distime_checkpoint", "trace_checkpoint",
                 "anno_file", "video_root"]:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val

    # Validate paths before proceeding
    anno_file = config["anno_file"]
    video_root = config["video_root"]
    logger.info(f"anno_file = {anno_file}")
    logger.info(f"video_root = {video_root}")
    if not os.path.exists(anno_file):
        logger.error(f"Annotation file NOT FOUND: {anno_file}")
        logger.error("Please specify --anno_file /path/to/charades_sta_test.json")
        return
    if not os.path.isdir(video_root):
        logger.error(f"Video root NOT FOUND: {video_root}")
        logger.error("Please specify --video_root /path/to/videos")
        return

    device = torch.device(args.device)

    # Load samples
    samples = load_samples(config["anno_file"], config["video_root"],
                           n=args.num_videos)
    if not samples:
        logger.error("No valid samples found! Check that video files exist in video_root.")
        return

    # Decide which paradigms to benchmark
    if args.paradigm:
        paradigms = args.paradigm  # already a list (nargs='+')
    elif model_type == "fastvlm":
        paradigms = ["text", "distime"]  # FastVLM: text baseline + distime
        logger.info("FastVLM mode: benchmarking text + distime")
    else:
        paradigms = ["text", "distime", "trace"]

    all_results = {}
    for paradigm in paradigms:
        results = benchmark_paradigm(paradigm, config, samples, device,
                                      model_type=model_type,
                                      image_size=image_size,
                                      image_seq_len_override=image_seq_len_override,
                                      vision_pool_stride=args.vision_pool_stride)
        all_results[paradigm] = results

    # ================================================================
    # Print summary in tab:efficiency format
    # ================================================================
    print(f"\n{'='*70}")
    print(f"tab:efficiency — {model_type} on single GPU")
    print(f"{'='*70}")
    print(f"{'Paradigm':<10} {'Extra Params':>14} {'Throu.(samp/s)':>16} {'Latency(ms/q)':>16} {'Mem.(GB)':>10}")
    print("-" * 70)

    labels = {"text": "Text", "distime": "Dist.", "trace": "Gen."}
    for paradigm in ["text", "distime", "trace"]:
        if paradigm not in all_results:
            continue
        r = all_results[paradigm]
        extra = r["extra_params"]
        if extra >= 1e6:
            extra_str = f"{extra/1e6:.1f}M"
        elif extra > 0:
            extra_str = f"{extra/1e3:.0f}K"
        else:
            extra_str = "0"

        throu = r.get("throughput_samp_s")
        throu_str = f"{throu:.2f}" if throu else "--"
        lat = r["latency_mean_ms"]
        mem = r["peak_mem_gb"]

        print(f"{labels[paradigm]:<10} {extra_str:>14} {throu_str:>16} {lat:>16.0f} {mem:>10.1f}")

    print(f"{'='*70}")

    # Save JSON
    output_dir = args.output_dir or "."
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{args.output_name}.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")

    # Print LaTeX snippet
    print(f"\n% LaTeX for tab:efficiency:")
    for paradigm in ["text", "distime", "trace"]:
        if paradigm not in all_results:
            continue
        r = all_results[paradigm]
        extra = r["extra_params"]
        if extra >= 1e6:
            extra_tex = f"{extra/1e6:.1f}M"
        elif extra > 0:
            extra_tex = f"{extra/1e3:.0f}K"
        else:
            extra_tex = "0"
        throu = r.get("throughput_samp_s")
        throu_tex = f"{throu:.1f}" if throu else "--"
        lat = r["latency_mean_ms"]
        mem = r["peak_mem_gb"]
        label = labels[paradigm]

        lat_suffix = "$^\\ast$" if paradigm == "trace" else ""
        print(f"{label}   & {extra_tex}  & {throu_tex}  & {lat:.0f}{lat_suffix}  & {mem:.1f}  \\\\")


if __name__ == "__main__":
    main()
