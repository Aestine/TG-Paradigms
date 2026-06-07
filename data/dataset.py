"""
InternVL 风格的 Dataset 完整实现
支持多 VLM 后端: model_type="smolvlm" | "fastvlm"
支持多时间范式: paradigm="distime" | "trace"

包含：
1. LazySupervisedDataset - InternVL 风格的 Lazy Loading (readlines)
2. PackedDataset - 动态 buffer 管理，支持多数据集采样
3. WeightedConcatDataset - 按权重采样
4. ConcatDataset - 简单拼接
5. 完整的 collate functions (distime + trace)
"""

import logging
import math
import os
import random
import re
import traceback
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Iterator, Union

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from PIL import Image
from torchvision import transforms

try:
    import orjson as json
except ImportError:
    import json

# Video loading
try:
    from utils.mm_utils import load_video
except ImportError:
    def load_video(*args, **kwargs):
        raise ImportError("mm_utils not found")

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

IGNORE_INDEX = -100
TIME_STAMP_TOKEN = "<TIME_STAMP>"
FRAME_TIME_TOKEN = "<FRAME_TIME>"
VIDEO_PLACEHOLDER = "<video>"

# TRACE placeholder tokens (in conversation text)
TRACE_SYNC_TOKEN = "<sync>"
TRACE_TIME_TOKEN = "<time>"
TRACE_SCORE_TOKEN = "<score>"

# <<<< 支持的 model_type 和 paradigm
SUPPORTED_MODEL_TYPES = ["smolvlm", "fastvlm", "molmo2"]
SUPPORTED_PARADIGMS = ["distime", "trace", "text"]

# Normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

# 每种模型的默认 normalize 类型
MODEL_DEFAULT_NORMALIZE = {
    "smolvlm": "siglip",
    "fastvlm": "none",
    "molmo2": "siglip",    # Molmo2 uses SigLIP2 vision encoder
}


# =============================================================================
# TRACE character-level encoding helpers (no nn dependency)
# =============================================================================

def encode_time_chars(time_pair):
    """
    Encode a single event's timestamps to local char IDs (0-12).

    TimeTower vocab: <sync>=0, <sep>=1, '0'=2, ..., '9'=11, '.'=12
    Format: format(t, '0>6.1f') per value, <sep> between, <sync> at end.

    Examples:
        [12.3, 45.6] -> '0012.3<sep>0045.6<sync>' -> 14 tokens
        [30.0]        -> '0030.0<sync>'             -> 7 tokens
        []            -> '<sync>'                   -> 1 token
    """
    if not time_pair:
        return [0]  # just <sync>

    result = []
    for i, t in enumerate(time_pair):
        formatted = format(float(t), '0>6.1f')  # e.g., '0012.3'
        for ch in formatted:
            if ch == '.':
                result.append(12)
            else:
                result.append(int(ch) + 2)
        if i < len(time_pair) - 1:
            result.append(1)  # <sep>
    result.append(0)  # <sync> (time sync = end of time sequence)
    return result


def encode_score_chars(score_list):
    """
    Encode a single event's scores to local char IDs (0-12).

    ScoreTower vocab: <sync>=0, <sep>=1, '0'=2, ..., '9'=11, '.'=12
    Format: format(s, '0>3.1f') per value, <sep> between, <sync> at end.

    Examples:
        [4.4]   -> '4.4<sync>'             -> 4 tokens
        []      -> '<sync>'                 -> 1 token
        [4.4, 3.2] -> '4.4<sep>3.2<sync>'  -> 8 tokens
    """
    if not score_list:
        return [0]  # just <sync>

    valid_scores = [s for s in score_list if s is not None]
    if not valid_scores:
        return [0]

    result = []
    for i, s in enumerate(valid_scores):
        formatted = format(float(s), '0>3.1f')  # e.g., '4.4'
        for ch in formatted:
            if ch == '.':
                result.append(12)
            else:
                result.append(int(ch) + 2)
        if i < len(valid_scores) - 1:
            result.append(1)  # <sep>
    result.append(0)  # <sync> (score sync = end of score sequence)
    return result


# =============================================================================
# Transform Builder (InternVL 风格)
# =============================================================================

def build_transform(
    input_size: int = 384,
    is_train: bool = True,
    normalize_type: str = 'siglip',
) -> transforms.Compose:
    if normalize_type == 'none':
        mean, std = None, None
    elif normalize_type == 'siglip':
        mean, std = SIGLIP_MEAN, SIGLIP_STD
    elif normalize_type == 'imagenet':
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == 'clip':
        mean, std = CLIP_MEAN, CLIP_STD
    else:
        raise ValueError(f"Unknown normalize_type: {normalize_type}")

    transform_list = [
        transforms.Resize(
            (input_size, input_size),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
    ]
    if mean is not None and std is not None:
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_list)


# =============================================================================
# LazySupervisedDataset (InternVL 风格)
# =============================================================================

class LazySupervisedDataset(Dataset):
    """
    InternVL 风格的 Lazy Loading Dataset

    支持两种时间范式:
    - paradigm="distime": DisTime 格式 (TIME_STAMP, Gaussian distributions)
    - paradigm="trace":   TRACE 格式 (sync/time/score tokens, char-level encoding)

    支持两种 VLM 后端:
    - model_type="smolvlm": SmolVLM2 (Idefics3)
    - model_type="fastvlm": FastVLM (LLaVA-style)
    """

    DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_FAKE_TOKEN = "<fake_token_around_image>"
    DEFAULT_GLOBAL_IMG_TOKEN = "<global-img>"
    DEFAULT_IMAGE_SEQ_LEN = 81

    def __init__(
        self,
        data_path: str,
        processor,
        model_type: str = "smolvlm",
        paradigm: str = "distime",      # <<<< 新增: 时间范式
        video_folder: str = "",
        max_frames: int = 16,
        fps: float = 1.0,
        max_length: int = 2048,
        image_size: int = 384,
        normalize_type: str = None,
        is_train: bool = True,
        repeat_time: float = 1.0,
        data_rank: int = 0,
        data_world_size: int = 1,
        force_shuffle: bool = False,
        random_seed: int = 0,
        use_packed_ds: bool = False,
        ds_name: str = "default",
        image_seq_len_override: Optional[int] = None,
        vocab_size_override: Optional[int] = None,
    ):
        super().__init__()

        # 校验 model_type
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(f"model_type must be one of {SUPPORTED_MODEL_TYPES}, got '{model_type}'")
        self.model_type = model_type

        # 校验 paradigm
        if paradigm not in SUPPORTED_PARADIGMS:
            raise ValueError(f"paradigm must be one of {SUPPORTED_PARADIGMS}, got '{paradigm}'")
        self.paradigm = paradigm

        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.video_folder = video_folder
        self.max_frames = max_frames
        self.fps = fps
        self.max_length = max_length
        self.image_size = image_size
        self.is_train = is_train
        self.ds_name = ds_name
        self.use_packed_ds = use_packed_ds

        # normalize
        if normalize_type is None:
            normalize_type = MODEL_DEFAULT_NORMALIZE[model_type]
        self.normalize_type = normalize_type

        # model_type 相关的 image token 配置
        if model_type == "smolvlm":
            self.image_seq_len = getattr(processor, 'image_seq_len', self.DEFAULT_IMAGE_SEQ_LEN)
            self.image_token = getattr(processor, 'image_token', self.DEFAULT_IMAGE_TOKEN)
            self.fake_image_token = getattr(processor, 'fake_image_token', self.DEFAULT_FAKE_TOKEN)
            self.global_image_token = getattr(processor, 'global_image_token', self.DEFAULT_GLOBAL_IMG_TOKEN)
        elif model_type == "fastvlm":
            self.image_token = getattr(processor, 'image_token', self.DEFAULT_IMAGE_TOKEN)
            self.fake_image_token = None
            self.global_image_token = None
            if hasattr(processor, 'patch_size') and processor.patch_size is not None:
                crop_h = image_size
                if hasattr(processor, 'image_processor') and hasattr(processor.image_processor, 'crop_size'):
                    crop_size = processor.image_processor.crop_size
                    crop_h = crop_size.get('height', image_size)
                self.image_seq_len = (crop_h // processor.patch_size) ** 2
            elif hasattr(processor, 'image_seq_len'):
                self.image_seq_len = processor.image_seq_len
            else:
                self.image_seq_len = 256

            if image_seq_len_override is not None:
                self.image_seq_len = image_seq_len_override

        elif model_type == "molmo2":
            # Molmo2: SigLIP2 vision encoder, 378×378, patch_size=14
            # Molmo2 用 config.image_patch_id (数字ID=151938) 标记图像 token 位置
            # 不像 SmolVLM 那样用 text token "<image>"
            self.fake_image_token = None
            self.global_image_token = None

            # 从 config 获取 image_patch_id，然后反查对应的 token string
            model_config = getattr(processor, 'config', None) or getattr(processor, 'image_processor', None)
            image_patch_id = None
            if model_config is not None:
                image_patch_id = getattr(model_config, 'image_patch_id', None)
            if image_patch_id is None:
                # 直接从 tokenizer 的 added_tokens 或 config 找
                image_patch_id = getattr(self.tokenizer, 'image_patch_id', None)
            if image_patch_id is None:
                # Hardcoded fallback for Molmo2
                image_patch_id = 151938
                logger.warning(f"Could not detect image_patch_id from processor/config, using default: {image_patch_id}")

            # 反查 token string
            self.image_token = self.tokenizer.convert_ids_to_tokens(image_patch_id)
            if self.image_token is None:
                # 如果反查失败，直接用 ID，后面会用 image_token_id
                self.image_token = f"<image_patch_{image_patch_id}>"
            self._molmo2_image_patch_id = image_patch_id
            logger.info(f"Molmo2 image_patch_id={image_patch_id}, image_token='{self.image_token}'")

            # Molmo2 视频模式: pooling_size=[3,3], 每帧 (378/14/3)^2 = 9×9 = 81 pooled patches
            # 从 video_processor 获取 pooling_size 来计算
            video_proc = getattr(processor, 'video_processor', None)
            if video_proc and hasattr(video_proc, 'pooling_size'):
                ps = video_proc.pooling_size  # e.g. [3, 3]
                raw_per_dim = 378 // 14  # 27
                pooled_h = raw_per_dim // ps[0]  # 9
                pooled_w = raw_per_dim // ps[1]  # 9
                self.image_seq_len = pooled_h * pooled_w  # 81
                self._molmo2_grid_h = pooled_h
                self._molmo2_grid_w = pooled_w
                logger.info(f"Molmo2 video pooling: {ps} -> grid {pooled_h}x{pooled_w} = {self.image_seq_len} tokens/frame")
            elif hasattr(processor, 'image_seq_len'):
                self.image_seq_len = processor.image_seq_len
                self._molmo2_grid_h = int(self.image_seq_len ** 0.5)
                self._molmo2_grid_w = self._molmo2_grid_h
            else:
                # Fallback: assume pooling_size=[3,3]
                self.image_seq_len = 81
                self._molmo2_grid_h = 9
                self._molmo2_grid_w = 9

            if image_seq_len_override is not None:
                self.image_seq_len = image_seq_len_override

        # Token IDs
        if model_type == "molmo2" and hasattr(self, '_molmo2_image_patch_id'):
            # Molmo2: 直接用 config 里的数字 ID，不经过 tokenizer 转换
            self.image_token_id = self._molmo2_image_patch_id
        else:
            self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

        # ================================================================
        # Paradigm-specific token setup
        # ================================================================
        if self.paradigm in ("distime", "text"):
            # Text Numeral paradigm reuses the same frame prompt (FRAME_TIME marker)
            # and tokenizer setup as DisTime. The only difference is the target
            # format (plain numerals, no <TIME_STAMP>) and the absence of time_gt.
            self.time_stamp_token_id = self.tokenizer.convert_tokens_to_ids(TIME_STAMP_TOKEN)
            self.frame_time_token_id = self.tokenizer.convert_tokens_to_ids(FRAME_TIME_TOKEN)
        elif self.paradigm == "trace":
            # 记录原始 vocab_size（添加 special tokens 之前）
            # 重要: 必须与模型的 vocab_size 一致！
            # Qwen2 系列 (FastVLM): len(tokenizer)=151647 ≠ config.vocab_size=151936
            # 如果不一致, TRACE extended token ID 会错位, 导致训练无效
            if vocab_size_override is not None:
                self.original_vocab_size = vocab_size_override
                logger.info(f"Using vocab_size_override={vocab_size_override} "
                            f"(len(tokenizer)={len(self.tokenizer)})")
            else:
                self.original_vocab_size = len(self.tokenizer)

            # 添加 TRACE special tokens 到 tokenizer（用于 tokenization 时识别占位符）
            # 注意: 如果 token 已存在（比如 tokenizer 之前加过），add_special_tokens 不会重复添加
            existing_special = getattr(self.tokenizer, 'additional_special_tokens', None) or []
            tokens_to_add = []
            for tok in [TRACE_SYNC_TOKEN, TRACE_TIME_TOKEN, TRACE_SCORE_TOKEN]:
                if tok not in existing_special and tok not in self.tokenizer.get_vocab():
                    tokens_to_add.append(tok)
            if tokens_to_add:
                added = self.tokenizer.add_special_tokens({
                    'additional_special_tokens': existing_special + tokens_to_add
                })
                logger.info(f"Added {added} TRACE special tokens to tokenizer")

            # 记录 tokenizer 分配的 ID（后续会被替换为 extended IDs）
            self.tok_sync_id = self.tokenizer.convert_tokens_to_ids(TRACE_SYNC_TOKEN)
            self.tok_time_id = self.tokenizer.convert_tokens_to_ids(TRACE_TIME_TOKEN)
            self.tok_score_id = self.tokenizer.convert_tokens_to_ids(TRACE_SCORE_TOKEN)

            # Extended token IDs (与 SmolVLMTrace 模型一致)
            self.sync_token_id = self.original_vocab_size       # 1 ID
            self.time_start_id = self.original_vocab_size + 1   # 13 IDs (local 0-12)
            self.score_start_id = self.original_vocab_size + 14  # 13 IDs (local 0-12)

            # FRAME_TIME 仍需要
            self.frame_time_token_id = self.tokenizer.convert_tokens_to_ids(FRAME_TIME_TOKEN)

            logger.info(f"TRACE token layout: original_vocab={self.original_vocab_size}, "
                        f"tok_sync_id={self.tok_sync_id}, tok_time_id={self.tok_time_id}, "
                        f"tok_score_id={self.tok_score_id}, "
                        f"sync={self.sync_token_id}, "
                        f"time=[{self.time_start_id}, {self.time_start_id + 13}), "
                        f"score=[{self.score_start_id}, {self.score_start_id + 13})")

        # Transform
        self.transform = build_transform(image_size, is_train, self.normalize_type)

        # === Lazy Loading ===
        logger.info(f'Loading dataset: {ds_name} from {data_path} '
                    f'(model_type={model_type}, paradigm={paradigm})')

        with open(data_path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                import json as json_module
                data_list = json_module.load(f)
                self.raw_data = [json_module.dumps(item) for item in data_list]
            else:
                self.raw_data = f.readlines()

        if repeat_time < 1:
            self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
        elif repeat_time > 1:
            self.raw_data = self.raw_data * int(repeat_time)

        self.rng = np.random.default_rng(seed=random_seed)
        if force_shuffle:
            self.rng.shuffle(self.raw_data)

        self.data_rank = data_rank
        self.data_world_size = data_world_size

        # vitt ID 映射
        self.vitt_mapping = {}
        if video_folder:
            vitt_mapping_file = os.path.join(video_folder, 'vitt/id_mapping.json')
            if os.path.exists(vitt_mapping_file):
                with open(vitt_mapping_file, 'r') as f:
                    forward_mapping = json.loads(f.read())
                    self.vitt_mapping = {yt_id: short_id for short_id, yt_id in forward_mapping.items()}

        self._precompute_lengths()

        logger.info(f'Dataset {ds_name}: {len(self.raw_data)} samples, '
                    f'model_type={model_type}, paradigm={paradigm}, '
                    f'image_seq_len={self.image_seq_len}')

    def _precompute_lengths(self):
        self.lengths = []
        for line in self.raw_data:
            try:
                item = json.loads(line)
                num_frames = min(self.max_frames, 16)
                conversations = item.get('conversations', [])
                if conversations and len(conversations) >= 2:
                    text_len = len(conversations[0].get('value', '')) + len(conversations[1].get('value', ''))
                else:
                    text_len = len(item.get('query', '')) + len(item.get('caption', ''))
                estimated_len = num_frames * self.image_seq_len + text_len // 4
                self.lengths.append(estimated_len)
            except:
                self.lengths.append(1000)

    # ================================================================
    # Molmo2 video processing
    # ================================================================

    def _process_frames_molmo2(self, frames):
        """Process video frames using Molmo2's video_processor.

        Molmo2 的视觉系统需要特殊格式的输入:
        - pixel_values_videos: [n_frames, n_patches, patch_size²×3] — 已分 patch 的帧
        - video_token_pooling: [n_pooled_patches, pool_h×pool_w] — pooling 索引
        - video_grids: [1, 3] — (num_frames, grid_h, grid_w)

        这与 SmolVLM/FastVLM 的 [N, 3, H, W] 格式完全不同。
        直接调用 Molmo2 的 video_processor._preprocess() 来得到正确格式。

        Args:
            frames: list of torch tensors (H, W, 3) uint8 from load_video

        Returns:
            pixel_values: dummy tensor (not used for Molmo2 forward)
            molmo2_video_inputs: dict with pixel_values_videos, video_token_pooling, video_grids
        """
        video_proc = self.processor.video_processor

        # Convert torch frames to numpy arrays (H, W, 3)
        frames_np = [f.numpy() for f in frames]

        # Call the video processor's internal _preprocess method
        # Input: list of videos, each video is a list of numpy frame arrays
        # video_proc.size is a dict {'height': H, 'width': W}, but _preprocess
        # expects a SizeDict with .height/.width attributes
        from transformers.image_utils import SizeDict
        size = SizeDict(**video_proc.size) if isinstance(video_proc.size, dict) else video_proc.size
        video_inputs = video_proc._preprocess(
            videos=[frames_np],  # single video = list of frames
            size=size,
            resample=video_proc.resample,
            image_mean=video_proc.image_mean,
            image_std=video_proc.image_std,
            patch_size=video_proc.patch_size,
            pooling_size=video_proc.pooling_size,
            return_tensors="pt",
        )

        pixel_values_videos = video_inputs["pixel_values_videos"]
        video_token_pooling = video_inputs["video_token_pooling"]
        video_grids = video_inputs["video_grids"]

        # Dummy pixel_values for compatibility (collate_fn uses it to detect shape)
        # Shape: [1, 3, 1, 1] — minimal dummy, won't be used in Molmo2 forward
        pixel_values = torch.zeros(1, 3, 1, 1)

        molmo2_video_inputs = {
            'pixel_values_videos': pixel_values_videos,
            'video_token_pooling': video_token_pooling,
            'video_grids': video_grids,
        }

        return pixel_values, molmo2_video_inputs

    # ================================================================
    # Frame prompt builders
    # ================================================================

    def _build_frame_prompt(self, num_frames: int) -> str:
        if self.model_type == "smolvlm":
            return self._build_frame_prompt_smolvlm(num_frames)
        elif self.model_type == "fastvlm":
            return self._build_frame_prompt_fastvlm(num_frames)
        elif self.model_type == "molmo2":
            return self._build_frame_prompt_molmo2(num_frames)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _build_frame_prompt_smolvlm(self, num_frames: int) -> str:
        frame_strings = []
        for i in range(num_frames):
            image_tokens = self.image_token * self.image_seq_len
            frame_str = (
                f"{FRAME_TIME_TOKEN}: "
                f"{self.fake_image_token}"
                f"{self.global_image_token}"
                f"{image_tokens}"
                f"{self.fake_image_token}"
            )
            frame_strings.append(frame_str)
        return "\n".join(frame_strings)

    def _build_frame_prompt_fastvlm(self, num_frames: int) -> str:
        frame_strings = []
        for i in range(num_frames):
            image_tokens = self.image_token * self.image_seq_len
            frame_str = f"{FRAME_TIME_TOKEN}: {image_tokens}"
            frame_strings.append(frame_str)
        return "\n".join(frame_strings)

    def _build_frame_prompt_molmo2(self, num_frames: int) -> str:
        """Molmo2 frame prompt: 匹配 processing_molmo2.get_video_string 的格式。

        每帧结构: <frame_start> + <im_patch>×(w*h) + <frame_end>
        其中 h=w=9 (378/14/3, video pooling_size=[3,3])

        注意: Molmo2 视频模式使用 <frame_start>/<frame_end> (不是 <im_start>/<im_end>)
        模型的 build_batched_videos 通过 <frame_end> 来计数视频帧
        """
        grid_h = getattr(self, '_molmo2_grid_h', 9)
        grid_w = getattr(self, '_molmo2_grid_w', 9)

        # 构建每帧: grid_h * grid_w 个 <im_patch>
        all_patch_tokens = self.image_token * (grid_h * grid_w)

        frame_strings = []
        for i in range(num_frames):
            frame_str = f"{FRAME_TIME_TOKEN}: <frame_start>{all_patch_tokens}<frame_end>"
            frame_strings.append(frame_str)
        return "\n".join(frame_strings)

    # ================================================================
    # Video path resolution
    # ================================================================

    def _resolve_video_path(self, video_path):
        if os.path.exists(video_path):
            return video_path

        for part in video_path.split('/'):
            if part.startswith('split_video_'):
                basename = os.path.basename(video_path)
                video_path = os.path.join('/work/hdd/bffz/yzou1/data/videos_new2', part, basename)
                break

        if 'coin/videos' in video_path:
            parts = video_path.split('/')
            if len(parts) >= 4:
                filename = parts[-1]
                if filename.startswith('video-'):
                    filename = filename[6:]
                video_path = '/'.join(parts[:-2]) + '/' + filename + '.mp4'
        elif 'queryd/QuerYD_downloader' in video_path:
            parts = video_path.split('/')
            filename = parts[-1]
            if filename.startswith('video-'):
                filename = filename[6:]
            video_path = '/'.join(parts[:-2]) + '/' + filename + '.mp4'
        elif 'yttemporal/videos' in video_path:
            parts = video_path.split('/')
            filename = parts[-1]
            if filename.startswith('video-'):
                filename = filename[6:]
            video_path = '/'.join(parts[:-1]) + '/' + filename + '.mp4'
        elif 'didemo/videos' in video_path:
            parts = video_path.split('/')
            filename = parts[-1]
            if '.' in filename:
                filename = filename.rsplit('.', 1)[0]
            video_path = '/'.join(parts[:-1]) + '/train/' + filename + '.mp4'
        elif 'vitt/videos' in video_path:
            parts = video_path.split('/')
            filename = parts[-1]
            if filename.startswith('video-'):
                youtube_id = filename[6:]
                if youtube_id in self.vitt_mapping:
                    short_id = self.vitt_mapping[youtube_id]
                    video_path = '/'.join(parts[:-1]) + '/' + short_id + '.mp4'

        if not os.path.exists(video_path):
            for ext in ['.mp4', '.mkv', '.webm', '.avi', '.mov']:
                if os.path.exists(video_path + ext):
                    return video_path + ext

        return video_path

    # ================================================================
    # Conversation builders
    # ================================================================

    def _build_conversation(self, user_text: str, assistant_text: str) -> str:
        if self.model_type == "smolvlm":
            return self._build_conversation_smolvlm(user_text, assistant_text)
        elif self.model_type == "fastvlm":
            return self._build_conversation_fastvlm(user_text, assistant_text)
        elif self.model_type == "molmo2":
            return self._build_conversation_molmo2(user_text, assistant_text)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _build_conversation_smolvlm(self, user_text: str, assistant_text: str) -> str:
        return (
            f"<|im_start|>User: {user_text}<end_of_utterance>\n"
            f"Assistant: {assistant_text}<end_of_utterance>\n"
        )

    def _build_conversation_fastvlm(self, user_text: str, assistant_text: str) -> str:
        return (
            f"<|im_start|>user\n{user_text}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_text}<|im_end|>\n"
        )

    def _build_conversation_molmo2(self, user_text: str, assistant_text: str) -> str:
        """Molmo2 uses Qwen3 chat template (same format as FastVLM's Qwen2)."""
        return (
            f"<|im_start|>user\n{user_text}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_text}<|im_end|>\n"
        )

    # ================================================================
    # Label creation
    # ================================================================

    def _create_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """创建 labels（mask user 部分）"""
        labels = input_ids.clone()

        if self.model_type == "smolvlm":
            assistant_marker = "Assistant: "
        elif self.model_type in ("fastvlm", "molmo2"):
            # FastVLM (Qwen2) and Molmo2 (Qwen3) use same chat template
            assistant_marker = "<|im_start|>assistant\n"
        else:
            assistant_marker = "Assistant: "

        assistant_marker_ids = self.tokenizer.encode(assistant_marker, add_special_tokens=False)
        input_ids_list = input_ids.tolist()
        assistant_start_idx = None

        for i in range(len(input_ids_list) - len(assistant_marker_ids) + 1):
            if input_ids_list[i:i + len(assistant_marker_ids)] == assistant_marker_ids:
                assistant_start_idx = i + len(assistant_marker_ids)
                break

        if assistant_start_idx is not None:
            labels[:assistant_start_idx] = IGNORE_INDEX
        else:
            labels[:len(labels) // 2] = IGNORE_INDEX

        return labels

    # ================================================================
    # __len__ / __getitem__
    # ================================================================

    def __len__(self) -> int:
        return len(self.raw_data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self.raw_data):
            idx = idx % len(self.raw_data)

        max_try = 10
        for try_cnt in range(max_try):
            try:
                data_item = json.loads(self.raw_data[idx])

                if self.paradigm == "distime":
                    return self._getitem_distime(data_item, idx)
                elif self.paradigm == "text":
                    return self._getitem_text(data_item, idx)
                elif self.paradigm == "trace":
                    return self._getitem_trace(data_item, idx)
                else:
                    raise ValueError(f"Unknown paradigm: {self.paradigm}")

            except Exception as e:
                if try_cnt < max_try - 1:
                    logger.warning(f"Error loading sample {idx} from {self.ds_name}: {e}")
                    idx = random.randint(0, len(self.raw_data) - 1)
                else:
                    logger.error(f"Failed to load sample after {max_try} tries: {e}")
                    traceback.print_exc()
                    raise

    # ================================================================
    # DisTime __getitem__ (原有逻辑不变)
    # ================================================================

    def _getitem_distime(self, data_item: dict, idx: int) -> Dict[str, Any]:
        """DisTime 范式的 __getitem__"""
        video_path = data_item.get('video', data_item.get('video_path', ''))
        if self.video_folder:
            video_path = os.path.join(self.video_folder, video_path)
        video_path = self._resolve_video_path(video_path)

        frames, frame_times, duration = load_video(
            video_path, num_frames=self.max_frames, fps=self.fps
        )
        num_frames = len(frames)

        # Molmo2: use video_processor to get proper patches + pooling indices
        # Other models: use plain torchvision transforms
        molmo2_video_inputs = None
        if self.model_type == "molmo2":
            pixel_values, molmo2_video_inputs = self._process_frames_molmo2(frames)
        else:
            pil_frames = [Image.fromarray(f.numpy()) for f in frames]
            pixel_values = torch.stack([self.transform(f) for f in pil_frames])

        conversations = data_item.get('conversations', [])

        if conversations and len(conversations) >= 2:
            human_msg = conversations[0].get('value', '')
            gpt_msg = conversations[1].get('value', '')
            times_raw = data_item.get('times', [])
        else:
            query = data_item.get('query', data_item.get('question', ''))
            start_time = data_item.get('start', data_item.get('start_time', 0))
            end_time = data_item.get('end', data_item.get('end_time', duration))
            caption = data_item.get('caption', data_item.get('answer', ''))
            human_msg = query
            gpt_msg = f"{TIME_STAMP_TOKEN} {caption}" if caption else caption
            times_raw = [[start_time, end_time]]

        frame_prompt = self._build_frame_prompt(num_frames)
        if VIDEO_PLACEHOLDER in human_msg:
            user_text = human_msg.replace(VIDEO_PLACEHOLDER, frame_prompt, 1)
        else:
            user_text = f"{frame_prompt}\n{human_msg}"
        assistant_text = gpt_msg

        conversation = self._build_conversation(user_text, assistant_text)

        encoded = self.tokenizer(
            conversation, return_tensors="pt", padding=False,
            truncation=True, max_length=self.max_length, add_special_tokens=True,
        )
        input_ids = encoded.input_ids.squeeze(0)
        attention_mask = encoded.attention_mask.squeeze(0)
        labels = self._create_labels(input_ids)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        epsilon = 1e-3
        if times_raw and len(times_raw) > 0:
            time_gt_list = []
            for t in times_raw:
                if isinstance(t, (int, float)):
                    s = e = max(0, min(float(t), duration - epsilon))
                elif len(t) == 1:
                    s = e = max(0, min(float(t[0]), duration - epsilon))
                else:
                    s = max(0, min(float(t[0]), duration - epsilon))
                    e = max(0, min(float(t[1]), duration - epsilon))
                time_gt_list.append([s / max(duration, 1e-6), e / max(duration, 1e-6)])
            time_gt = torch.tensor(time_gt_list, dtype=torch.float32)
        else:
            time_gt = torch.zeros(0, 2, dtype=torch.float32)

        image_flags = torch.ones(num_frames, dtype=torch.long)

        result = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'pixel_values': pixel_values,
            'image_flags': image_flags,
            'time_gt': time_gt,
            'num_events': time_gt.shape[0],
            'duration': torch.tensor(duration),
            'frame_times': torch.tensor(frame_times),
            'num_frames': num_frames,
        }

        # Add Molmo2-specific video processing outputs
        if molmo2_video_inputs is not None:
            result.update(molmo2_video_inputs)

        return result

    # ================================================================
    # Text Numeral __getitem__
    # ================================================================

    @staticmethod
    def _format_text_target(time_events, captions):
        """Build the Text Numeral target string (paper Appendix, Table `tab:format`).

        Single event (Moment Retrieval / Highlight Detection):
            "The event happens from {s:.1f} to {e:.1f} seconds."
        Multiple events (Dense Video Captioning):
            "From {s:.1f} to {e:.1f} seconds, {caption}. From ..."

        The single-event phrasing matches the paper exactly. The multi-event
        (DVC) phrasing follows the same numeral style ("from X to Y seconds")
        plus a per-event summary, since the paper's table only illustrates the
        single-event case. Eval parses timestamps via the same
        "from X to Y seconds" pattern.
        """
        def fmt(v):
            return f"{float(v):.1f}"

        if not time_events:
            return "No event found."

        if len(time_events) == 1:
            s, e = time_events[0]
            return f"The event happens from {fmt(s)} to {fmt(e)} seconds."

        parts = []
        for i, (s, e) in enumerate(time_events):
            cap = captions[i].strip() if i < len(captions) and captions[i] else ""
            if cap:
                parts.append(f"From {fmt(s)} to {fmt(e)} seconds, {cap}.")
            else:
                parts.append(f"From {fmt(s)} to {fmt(e)} seconds.")
        return " ".join(parts)

    def _getitem_text(self, data_item: dict, idx: int) -> Dict[str, Any]:
        """Text Numeral paradigm __getitem__.

        Reuses the EXACT same video loading, frame prompt, conversation template
        and label masking as DisTime. The only differences:
          * the assistant target encodes timestamps as plain text numerals
            (no <TIME_STAMP> token, no special output vocabulary);
          * no `time_gt` supervision is produced (time_gt is left empty so the
            shared collate_fn still works unchanged).

        It can consume the SAME data files as DisTime (e.g.
        combined_distime_balanced.jsonl): it reads the `times` (and, for DVC,
        the per-event captions parsed out of the original gpt message) and
        reformats them into the numeral target.
        """
        video_path = data_item.get('video', data_item.get('video_path', ''))
        if self.video_folder:
            video_path = os.path.join(self.video_folder, video_path)
        video_path = self._resolve_video_path(video_path)

        frames, frame_times, duration = load_video(
            video_path, num_frames=self.max_frames, fps=self.fps
        )
        num_frames = len(frames)

        molmo2_video_inputs = None
        if self.model_type == "molmo2":
            pixel_values, molmo2_video_inputs = self._process_frames_molmo2(frames)
        else:
            pil_frames = [Image.fromarray(f.numpy()) for f in frames]
            pixel_values = torch.stack([self.transform(f) for f in pil_frames])

        # ----- collect (human query, time events, captions) -----
        conversations = data_item.get('conversations', [])
        if conversations and len(conversations) >= 2:
            human_msg = conversations[0].get('value', '')
            gpt_msg = conversations[1].get('value', '')
            times_raw = data_item.get('times', [])
            # captions aligned with events: split original gpt msg on <TIME_STAMP>
            caption_parts = [p.strip() for p in gpt_msg.split(TIME_STAMP_TOKEN)]
            captions = [p for p in caption_parts if p]
        else:
            query = data_item.get('query', data_item.get('question', ''))
            start_time = data_item.get('start', data_item.get('start_time', 0))
            end_time = data_item.get('end', data_item.get('end_time', duration))
            caption = data_item.get('caption', data_item.get('answer', ''))
            human_msg = query
            times_raw = [[start_time, end_time]]
            captions = [caption] if caption else []

        # clean + clamp time events to [0, duration] (absolute seconds)
        epsilon = 1e-3
        time_events = []
        for t in times_raw:
            if isinstance(t, (int, float)):
                s = e = float(t)
            elif len(t) == 1:
                s = e = float(t[0])
            elif len(t) >= 2:
                s, e = float(t[0]), float(t[1])
            else:
                continue
            s = max(0.0, min(s, duration - epsilon))
            e = max(0.0, min(e, duration - epsilon))
            time_events.append([s, e])

        assistant_text = self._format_text_target(time_events, captions)

        # ----- build prompt (identical frames/template to other paradigms) -----
        frame_prompt = self._build_frame_prompt(num_frames)
        if VIDEO_PLACEHOLDER in human_msg:
            user_text = human_msg.replace(VIDEO_PLACEHOLDER, frame_prompt, 1)
        else:
            user_text = f"{frame_prompt}\n{human_msg}"

        conversation = self._build_conversation(user_text, assistant_text)

        encoded = self.tokenizer(
            conversation, return_tensors="pt", padding=False,
            truncation=True, max_length=self.max_length, add_special_tokens=True,
        )
        input_ids = encoded.input_ids.squeeze(0)
        attention_mask = encoded.attention_mask.squeeze(0)
        labels = self._create_labels(input_ids)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        # Text paradigm has NO continuous time supervision; keep empty so the
        # shared collate_fn (which pads time_gt) works without modification.
        time_gt = torch.zeros(0, 2, dtype=torch.float32)
        image_flags = torch.ones(num_frames, dtype=torch.long)

        result = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'pixel_values': pixel_values,
            'image_flags': image_flags,
            'time_gt': time_gt,
            'num_events': time_gt.shape[0],
            'duration': torch.tensor(duration),
            'frame_times': torch.tensor(frame_times),
            'num_frames': num_frames,
        }
        if molmo2_video_inputs is not None:
            result.update(molmo2_video_inputs)

        return result

    # ================================================================
    # TRACE __getitem__
    # ================================================================

    def _getitem_trace(self, data_item: dict, idx: int) -> Dict[str, Any]:
        """
        TRACE 范式的 __getitem__

        TRACE 数据格式:
            gpt: "<sync><time>×14<score>caption<sync><time>×7<score>×4caption..."
            times: [[s1, e1], [t2], ...]
            scores: [[4.4], [], [3.2], ...]

        输出 (在 DisTime 基础上新增 time_labels, score_labels):
            input_ids: 包含 extended token IDs (sync/time/score > original_vocab_size)
            labels: text+sync labels (sync pos → original_vocab_size, time/score → IGNORE)
            time_labels: local time char IDs (0-12) at time positions, IGNORE elsewhere
            score_labels: local score char IDs (0-12) at score positions, IGNORE elsewhere
        """
        # 加载视频
        video_path = data_item.get('video', data_item.get('video_path', ''))
        if self.video_folder:
            video_path = os.path.join(self.video_folder, video_path)
        video_path = self._resolve_video_path(video_path)

        frames, frame_times, duration = load_video(
            video_path, num_frames=self.max_frames, fps=self.fps
        )
        num_frames = len(frames)

        # Molmo2: use video_processor to get proper patches + pooling indices
        # Other models: use plain torchvision transforms
        molmo2_video_inputs = None
        if self.model_type == "molmo2":
            pixel_values, molmo2_video_inputs = self._process_frames_molmo2(frames)
        else:
            pil_frames = [Image.fromarray(f.numpy()) for f in frames]
            pixel_values = torch.stack([self.transform(f) for f in pil_frames])

        # 读取对话
        conversations = data_item.get('conversations', [])
        if not conversations or len(conversations) < 2:
            raise ValueError(f"Invalid conversations in TRACE sample idx={idx}")

        human_msg = conversations[0].get('value', '')
        gpt_msg = conversations[1].get('value', '')

        # 读取 times 和 scores
        times_raw = data_item.get('times', [[]])
        scores_raw = data_item.get('scores', [[]])

        # 清洗 times: [[]] → [], [[30.0], [31.0]] → [[30.0], [31.0]]
        valid_times = [t for t in times_raw if isinstance(t, list) and len(t) > 0]

        # ================================================================
        # 预编码所有事件的 time/score char IDs
        # ================================================================
        all_time_char_ids = []
        all_score_char_ids = []

        for ev_idx in range(len(valid_times)):
            tc = encode_time_chars(valid_times[ev_idx])
            all_time_char_ids.extend(tc)

            if ev_idx < len(scores_raw):
                sc = encode_score_chars(scores_raw[ev_idx])
            else:
                sc = encode_score_chars([])
            all_score_char_ids.extend(sc)

        # QA 样本: times=[[]] 但 gpt 中可能仍有 <sync><time><score>
        if not valid_times:
            n_time_in_text = gpt_msg.count(TRACE_TIME_TOKEN)
            n_score_in_text = gpt_msg.count(TRACE_SCORE_TOKEN)
            if n_time_in_text > 0:
                # 零时间: 所有 time char 设为 <sync> (local 0)
                all_time_char_ids = [0] * n_time_in_text
            if n_score_in_text > 0:
                all_score_char_ids = [0] * n_score_in_text

        # ================================================================
        # 构建对话
        # ================================================================
        frame_prompt = self._build_frame_prompt(num_frames)
        if VIDEO_PLACEHOLDER in human_msg:
            user_text = human_msg.replace(VIDEO_PLACEHOLDER, frame_prompt, 1)
        else:
            user_text = f"{frame_prompt}\n{human_msg}"
        assistant_text = gpt_msg

        conversation = self._build_conversation(user_text, assistant_text)

        # ================================================================
        # Tokenize (tokenizer 已包含 <sync>/<time>/<score> special tokens)
        # ================================================================
        encoded = self.tokenizer(
            conversation, return_tensors="pt", padding=False,
            truncation=True, max_length=self.max_length, add_special_tokens=True,
        )
        input_ids = encoded.input_ids.squeeze(0)
        attention_mask = encoded.attention_mask.squeeze(0)

        # ================================================================
        # 创建基础 labels (mask user 部分)
        # ================================================================
        labels = self._create_labels(input_ids)

        # ================================================================
        # Post-process: 替换 TRACE placeholder IDs → extended IDs
        # 并构建 time_labels, score_labels
        # ================================================================
        time_labels = torch.full_like(input_ids, IGNORE_INDEX)
        score_labels = torch.full_like(input_ids, IGNORE_INDEX)

        # 找 TRACE placeholder 位置
        sync_positions = (input_ids == self.tok_sync_id).nonzero(as_tuple=True)[0]
        time_positions = (input_ids == self.tok_time_id).nonzero(as_tuple=True)[0]
        score_positions = (input_ids == self.tok_score_id).nonzero(as_tuple=True)[0]

        # 替换 <sync> → sync_token_id (extended)
        input_ids[sync_positions] = self.sync_token_id
        # labels: sync 位置 → original_vocab_size (text+sync head 的 sync class)
        # 注意: 只有 assistant 部分的 sync 才有效，user 部分已被 IGNORE_INDEX mask
        for pos in sync_positions:
            if labels[pos] != IGNORE_INDEX:
                labels[pos] = self.original_vocab_size

        # 替换 <time> → time_start_id + local_char_id
        n_time_tokens = len(time_positions)
        n_time_chars = len(all_time_char_ids)
        if n_time_tokens != n_time_chars:
            logger.debug(
                f"[TRACE] ds={self.ds_name} idx={idx}: "
                f"<time> mismatch: text={n_time_tokens}, encoded={n_time_chars}"
            )
            if n_time_chars < n_time_tokens:
                all_time_char_ids.extend([0] * (n_time_tokens - n_time_chars))
            else:
                all_time_char_ids = all_time_char_ids[:n_time_tokens]

        for i, pos in enumerate(time_positions):
            local_id = all_time_char_ids[i]
            input_ids[pos] = self.time_start_id + local_id
            time_labels[pos] = local_id
            labels[pos] = IGNORE_INDEX  # text head 不管 time 位置

        # 替换 <score> → score_start_id + local_char_id
        n_score_tokens = len(score_positions)
        n_score_chars = len(all_score_char_ids)
        if n_score_tokens != n_score_chars:
            logger.debug(
                f"[TRACE] ds={self.ds_name} idx={idx}: "
                f"<score> mismatch: text={n_score_tokens}, encoded={n_score_chars}"
            )
            if n_score_chars < n_score_tokens:
                all_score_char_ids.extend([0] * (n_score_tokens - n_score_chars))
            else:
                all_score_char_ids = all_score_char_ids[:n_score_tokens]

        for i, pos in enumerate(score_positions):
            local_id = all_score_char_ids[i]
            input_ids[pos] = self.score_start_id + local_id
            score_labels[pos] = local_id
            labels[pos] = IGNORE_INDEX  # text head 不管 score 位置

        # ================================================================
        # Position IDs
        # ================================================================
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        # ================================================================
        # Time GT (same format as DisTime, for evaluation)
        # ================================================================
        epsilon = 1e-3
        if valid_times:
            time_gt_list = []
            for t in valid_times:
                if len(t) == 1:
                    s = e = max(0, min(float(t[0]), duration - epsilon))
                else:
                    s = max(0, min(float(t[0]), duration - epsilon))
                    e = max(0, min(float(t[1]), duration - epsilon))
                time_gt_list.append([s / max(duration, 1e-6), e / max(duration, 1e-6)])
            time_gt = torch.tensor(time_gt_list, dtype=torch.float32)
        else:
            time_gt = torch.zeros(0, 2, dtype=torch.float32)

        image_flags = torch.ones(num_frames, dtype=torch.long)

        result = {
            'input_ids': input_ids,
            'labels': labels,
            'time_labels': time_labels,         # TRACE 独有
            'score_labels': score_labels,        # TRACE 独有
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'pixel_values': pixel_values,
            'image_flags': image_flags,
            'time_gt': time_gt,
            'num_events': time_gt.shape[0],
            'duration': torch.tensor(duration),
            'frame_times': torch.tensor(frame_times),
            'num_frames': num_frames,
        }

        # Add Molmo2-specific video processing outputs
        if molmo2_video_inputs is not None:
            result.update(molmo2_video_inputs)

        return result


# =============================================================================
# PackedDataset (InternVL 风格 - 动态 Buffer)
# =============================================================================

class PackedDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        datasets: List[Dataset],
        dataset_weight: Optional[List[float]] = None,
        max_packed_tokens: int = 4096,
        num_images_expected: int = 40,
        max_buffer_size: int = 20,
        data_rank: int = 0,
        data_world_size: int = 1,
        log_freq: int = 1000,
        strict_mode: bool = True,
        replacement: bool = False,
        allow_overflow: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.datasets = datasets
        self.num_datasets = len(datasets)

        if dataset_weight is None:
            total = sum(len(d) for d in datasets)
            self.dataset_weight = [len(d) / total for d in datasets]
        else:
            self.dataset_weight = dataset_weight

        self.max_packed_tokens = max_packed_tokens
        self.num_images_expected = num_images_expected
        self.max_buffer_size = max_buffer_size
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.log_freq = log_freq
        self.strict_mode = strict_mode
        self.replacement = replacement
        self.allow_overflow = allow_overflow
        self.rng = np.random.default_rng(seed=seed + data_rank)
        self.dataset_iterators = None
        self.buffer = []
        self.total_packed = 0
        self.total_samples = sum(len(d) for d in datasets)
        avg_sample_len = 500
        self.estimated_length = self.total_samples * avg_sample_len // max_packed_tokens

    def _init_iterators(self):
        self.dataset_iterators = []
        self.dataset_indices = []
        for i, dataset in enumerate(self.datasets):
            indices = list(range(len(dataset)))
            self.rng.shuffle(indices)
            self.dataset_iterators.append(iter(indices))
            self.dataset_indices.append(indices)

    def _get_next_sample(self) -> Optional[Dict[str, Any]]:
        dataset_idx = self.rng.choice(self.num_datasets, p=self.dataset_weight)
        try:
            sample_idx = next(self.dataset_iterators[dataset_idx])
        except StopIteration:
            if self.replacement:
                self.rng.shuffle(self.dataset_indices[dataset_idx])
                self.dataset_iterators[dataset_idx] = iter(self.dataset_indices[dataset_idx])
                sample_idx = next(self.dataset_iterators[dataset_idx])
            else:
                return None
        try:
            sample = self.datasets[dataset_idx][sample_idx]
            sample['_dataset_idx'] = dataset_idx
            return sample
        except Exception as e:
            logger.warning(f"Error getting sample from dataset {dataset_idx}: {e}")
            return self._get_next_sample()

    def _can_pack(self, current_tokens: int, new_sample: Dict) -> bool:
        return (current_tokens + new_sample['input_ids'].shape[0]) <= self.max_packed_tokens

    def _pack_buffer(self) -> Dict[str, torch.Tensor]:
        if not self.buffer:
            raise ValueError("Buffer is empty")

        has_trace = 'time_labels' in self.buffer[0]
        has_molmo2_video = 'pixel_values_videos' in self.buffer[0]

        all_input_ids = []
        all_labels = []
        all_attention_mask = []
        all_pixel_values = []
        all_image_flags = []
        all_frame_times = []
        all_time_gt = []
        all_duration = []
        all_time_labels = [] if has_trace else None
        all_score_labels = [] if has_trace else None
        all_pvv = [] if has_molmo2_video else None
        all_vtp = [] if has_molmo2_video else None
        all_vg = [] if has_molmo2_video else None

        for sample in self.buffer:
            all_input_ids.append(sample['input_ids'])
            all_labels.append(sample['labels'])
            all_attention_mask.append(sample['attention_mask'])
            all_pixel_values.append(sample['pixel_values'])
            all_image_flags.append(sample['image_flags'])
            all_frame_times.append(sample['frame_times'])
            all_time_gt.append(sample['time_gt'])
            all_duration.append(sample['duration'])
            if has_trace:
                all_time_labels.append(sample['time_labels'])
                all_score_labels.append(sample['score_labels'])
            if has_molmo2_video:
                all_pvv.append(sample['pixel_values_videos'])
                all_vtp.append(sample['video_token_pooling'])
                all_vg.append(sample['video_grids'])

        packed_input_ids = torch.cat(all_input_ids, dim=0)
        packed_labels = torch.cat(all_labels, dim=0)
        packed_attention_mask = torch.cat(all_attention_mask, dim=0)
        packed_pixel_values = torch.cat(all_pixel_values, dim=0)
        packed_image_flags = torch.cat(all_image_flags, dim=0)
        packed_frame_times = torch.cat(all_frame_times, dim=0)

        max_events = max((t.shape[0] for t in all_time_gt), default=0)
        max_events = max(max_events, 1)
        num_samples_count = len(all_time_gt)
        packed_time_gt = torch.zeros(num_samples_count, max_events, 2)
        packed_num_events = torch.zeros(num_samples_count, dtype=torch.long)
        for i, t in enumerate(all_time_gt):
            n_ev = t.shape[0]
            if n_ev > 0:
                packed_time_gt[i, :n_ev] = t
            packed_num_events[i] = n_ev
        packed_duration = torch.stack(all_duration)

        position_ids = torch.zeros_like(packed_input_ids)
        current_pos = 0
        for sample in self.buffer:
            seq_len = sample['input_ids'].shape[0]
            position_ids[current_pos:current_pos + seq_len] = torch.arange(seq_len)
            current_pos += seq_len

        num_samples = len(self.buffer)
        self.buffer = []

        result = {
            'input_ids': packed_input_ids,
            'labels': packed_labels,
            'attention_mask': packed_attention_mask,
            'position_ids': position_ids,
            'pixel_values': packed_pixel_values,
            'image_flags': packed_image_flags,
            'frame_times': packed_frame_times,
            'time_gt': packed_time_gt,
            'duration': packed_duration,
            'num_samples': num_samples,
            'num_events': packed_num_events,
        }

        if has_trace:
            result['time_labels'] = torch.cat(all_time_labels, dim=0)
            result['score_labels'] = torch.cat(all_score_labels, dim=0)

        if has_molmo2_video:
            result['pixel_values_videos'] = torch.cat(all_pvv, dim=0)
            result['video_token_pooling'] = torch.cat(all_vtp, dim=0)
            result['video_grids'] = torch.cat(all_vg, dim=0)

        return result

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        self._init_iterators()
        self.buffer = []
        current_tokens = 0
        self.total_packed = 0

        while True:
            sample = self._get_next_sample()
            if sample is None:
                if self.buffer:
                    yield self._pack_buffer()
                break
            sample_tokens = sample['input_ids'].shape[0]
            if self._can_pack(current_tokens, sample):
                self.buffer.append(sample)
                current_tokens += sample_tokens
            else:
                if self.buffer:
                    self.total_packed += 1
                    if self.total_packed % self.log_freq == 0:
                        logger.info(f"Packed {self.total_packed} samples")
                    yield self._pack_buffer()
                self.buffer = [sample]
                current_tokens = sample_tokens
            if len(self.buffer) >= self.max_buffer_size:
                self.total_packed += 1
                yield self._pack_buffer()
                self.buffer = []
                current_tokens = 0

    def __len__(self) -> int:
        return self.estimated_length


# =============================================================================
# WeightedConcatDataset
# =============================================================================

class WeightedConcatDataset(Dataset):
    def __init__(self, datasets: List[Dataset], weights: Optional[List[float]] = None, seed: int = 42):
        self.datasets = datasets
        self.seed = seed
        if weights is None:
            total = sum(len(d) for d in datasets)
            self.weights = [len(d) / total for d in datasets]
        else:
            total_w = sum(weights)
            self.weights = [w / total_w for w in weights]
        self.total_length = sum(len(d) for d in datasets)
        self._build_index_map()

    def _build_index_map(self):
        self.index_map = []
        rng = np.random.default_rng(seed=self.seed)
        for dataset_idx, (dataset, weight) in enumerate(zip(self.datasets, self.weights)):
            num_samples = int(self.total_length * weight)
            dataset_len = len(dataset)
            if num_samples <= dataset_len:
                indices = rng.choice(dataset_len, num_samples, replace=False)
            else:
                indices = rng.choice(dataset_len, num_samples, replace=True)
            for local_idx in indices:
                self.index_map.append((dataset_idx, int(local_idx)))
        rng.shuffle(self.index_map)

    def reshuffle(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed = seed
        self._build_index_map()

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        dataset_idx, local_idx = self.index_map[idx]
        sample = self.datasets[dataset_idx][local_idx]
        sample['dataset_idx'] = dataset_idx
        return sample


# =============================================================================
# ConcatDataset
# =============================================================================

class ConcatDataset(Dataset):
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.cumulative_lengths = []
        cumsum = 0
        for d in datasets:
            cumsum += len(d)
            self.cumulative_lengths.append(cumsum)

    def __len__(self) -> int:
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range")
        for dataset_idx, cumlen in enumerate(self.cumulative_lengths):
            if idx < cumlen:
                local_idx = idx if dataset_idx == 0 else idx - self.cumulative_lengths[dataset_idx - 1]
                return self.datasets[dataset_idx][local_idx]
        raise IndexError(f"Index {idx} out of range")


# =============================================================================
# Collate Functions (支持 distime + trace)
# =============================================================================

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """标准 collate function，自动检测 TRACE 并处理 time_labels/score_labels

    对 Molmo2: 额外处理 pixel_values_videos, video_token_pooling, video_grids
    这些张量跨 batch 直接拼接 (concat along dim 0)，
    因为模型的 build_batched_videos 会按 <frame_end> 计数来分配到各 sample
    """
    batch = [b for b in batch if b is not None and b.get('pixel_values') is not None]
    if len(batch) == 0:
        raise ValueError("Empty batch")

    batch_size = len(batch)
    has_trace = 'time_labels' in batch[0]
    has_molmo2_video = 'pixel_values_videos' in batch[0]

    max_seq_len = max(item['input_ids'].shape[0] for item in batch)
    max_num_frames = max(item['num_frames'] for item in batch)
    max_frame_times = max(item['frame_times'].shape[0] for item in batch)
    max_events = max((item['time_gt'].shape[0] for item in batch), default=0)
    max_events = max(max_events, 1)

    input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    labels = torch.full((batch_size, max_seq_len), IGNORE_INDEX, dtype=torch.long)
    position_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)

    if has_trace:
        time_labels = torch.full((batch_size, max_seq_len), IGNORE_INDEX, dtype=torch.long)
        score_labels = torch.full((batch_size, max_seq_len), IGNORE_INDEX, dtype=torch.long)

    time_gt = torch.zeros(batch_size, max_events, 2, dtype=torch.float32)
    num_events = torch.zeros(batch_size, dtype=torch.long)
    duration = torch.stack([item['duration'] for item in batch])
    frame_times = torch.full((batch_size, max_frame_times), fill_value=-1.0)

    # For non-Molmo2 models: standard pixel_values [B, max_frames, C, H, W]
    # For Molmo2: pixel_values is a dummy, real data is in pixel_values_videos
    if not has_molmo2_video:
        sample_pv = batch[0]['pixel_values']
        C, H, W = sample_pv.shape[1:]
        pixel_values = torch.zeros(batch_size, max_num_frames, C, H, W, dtype=sample_pv.dtype)
    else:
        pixel_values = None  # Will not be used

    image_flags = torch.zeros(batch_size, max_num_frames, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item['input_ids'].shape[0]
        input_ids[i, :seq_len] = item['input_ids']
        attention_mask[i, :seq_len] = item['attention_mask']
        labels[i, :seq_len] = item['labels']
        position_ids[i, :seq_len] = item['position_ids']

        if has_trace:
            time_labels[i, :seq_len] = item['time_labels']
            score_labels[i, :seq_len] = item['score_labels']

        ft_len = item['frame_times'].shape[0]
        frame_times[i, :ft_len] = item['frame_times']

        n_ev = item['time_gt'].shape[0]
        if n_ev > 0:
            time_gt[i, :n_ev] = item['time_gt']
        num_events[i] = n_ev

        n = item['num_frames']
        if not has_molmo2_video:
            pixel_values[i, :n] = item['pixel_values']
        image_flags[i, :n] = item['image_flags']

    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'position_ids': position_ids,
        'pixel_values': pixel_values,
        'image_flags': image_flags,
        'time_gt': time_gt,
        'num_events': num_events,
        'duration': duration,
        'frame_times': frame_times,
    }

    # Molmo2 video tensors: concatenate across batch (model handles per-sample routing)
    if has_molmo2_video:
        result['pixel_values_videos'] = torch.cat(
            [item['pixel_values_videos'] for item in batch], dim=0)
        result['video_token_pooling'] = torch.cat(
            [item['video_token_pooling'] for item in batch], dim=0)
        result['video_grids'] = torch.cat(
            [item['video_grids'] for item in batch], dim=0)

    if has_trace:
        result['time_labels'] = time_labels
        result['score_labels'] = score_labels

    return result


def packed_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """PackedDataset 专用 collate function"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        raise ValueError("Empty batch")

    batch_size = len(batch)
    has_trace = 'time_labels' in batch[0]
    has_molmo2_video = 'pixel_values_videos' in batch[0]

    max_seq_len = max(item['input_ids'].shape[0] for item in batch)
    max_num_samples = max(item['num_samples'] for item in batch)
    max_frame_times = max(item['frame_times'].shape[0] for item in batch)
    max_events = max(item['time_gt'].shape[1] for item in batch)
    max_events = max(max_events, 1)

    input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    labels = torch.full((batch_size, max_seq_len), IGNORE_INDEX, dtype=torch.long)
    position_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)

    if has_trace:
        time_labels = torch.full((batch_size, max_seq_len), IGNORE_INDEX, dtype=torch.long)
        score_labels = torch.full((batch_size, max_seq_len), IGNORE_INDEX, dtype=torch.long)

    if not has_molmo2_video:
        max_frames = max(item['pixel_values'].shape[0] for item in batch)
        sample_pv = batch[0]['pixel_values']
        C, H, W = sample_pv.shape[1:]
        pixel_values = torch.zeros(batch_size, max_frames, C, H, W, dtype=sample_pv.dtype)
    else:
        pixel_values = None
        max_frames = max(item['image_flags'].shape[0] for item in batch)

    image_flags = torch.zeros(batch_size, max_frames, dtype=torch.long)
    frame_times = torch.full((batch_size, max_frame_times), fill_value=-1.0)

    time_gt = torch.zeros(batch_size, max_num_samples, max_events, 2)
    duration = torch.zeros(batch_size, max_num_samples)
    num_samples = torch.zeros(batch_size, dtype=torch.long)
    num_events = torch.zeros(batch_size, max_num_samples, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item['input_ids'].shape[0]
        input_ids[i, :seq_len] = item['input_ids']
        attention_mask[i, :seq_len] = item['attention_mask']
        labels[i, :seq_len] = item['labels']
        position_ids[i, :seq_len] = item['position_ids']

        if has_trace:
            time_labels[i, :seq_len] = item['time_labels']
            score_labels[i, :seq_len] = item['score_labels']

        if not has_molmo2_video:
            n_frames = item['pixel_values'].shape[0]
            pixel_values[i, :n_frames] = item['pixel_values']

        n_flags = item['image_flags'].shape[0]
        image_flags[i, :n_flags] = item['image_flags']

        ft_len = item['frame_times'].shape[0]
        frame_times[i, :ft_len] = item['frame_times']

        n_s = item['num_samples']
        item_tgt = item['time_gt']
        item_n_ev = item_tgt.shape[1]
        time_gt[i, :n_s, :item_n_ev] = item_tgt
        duration[i, :n_s] = item['duration']
        num_samples[i] = n_s
        if 'num_events' in item:
            num_events[i, :n_s] = item['num_events']

    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'position_ids': position_ids,
        'pixel_values': pixel_values,
        'image_flags': image_flags,
        'frame_times': frame_times,
        'time_gt': time_gt,
        'duration': duration,
        'num_samples': num_samples,
        'num_events': num_events,
    }

    if has_molmo2_video:
        result['pixel_values_videos'] = torch.cat(
            [item['pixel_values_videos'] for item in batch], dim=0)
        result['video_token_pooling'] = torch.cat(
            [item['video_token_pooling'] for item in batch], dim=0)
        result['video_grids'] = torch.cat(
            [item['video_grids'] for item in batch], dim=0)

    if has_trace:
        result['time_labels'] = time_labels
        result['score_labels'] = score_labels

    return result


# =============================================================================
# 辅助函数
# =============================================================================

def build_datasets(
    data_configs: List[Dict],
    processor,
    model_type: str = "smolvlm",
    paradigm: str = "distime",          # <<<< 新增
    video_folder: str = "",
    max_frames: int = 16,
    image_size: int = 384,
    normalize_type: str = None,
    use_data_resampling: bool = False,
    use_packed_ds: bool = False,
    max_packed_tokens: int = 4096,
    data_rank: int = 0,
    data_world_size: int = 1,
    image_seq_len_override: Optional[int] = None,
    vocab_size_override: Optional[int] = None,
) -> Dataset:
    datasets = []
    lengths = []

    for config in data_configs:
        ds = LazySupervisedDataset(
            data_path=config['data_path'],
            processor=processor,
            model_type=model_type,
            paradigm=paradigm,          # <<<< 透传
            video_folder=config.get('video_folder', video_folder),
            max_frames=max_frames,
            image_size=image_size,
            normalize_type=normalize_type,
            repeat_time=config.get('repeat_time', 1.0),
            use_packed_ds=use_packed_ds,
            ds_name=config.get('name', os.path.basename(config['data_path'])),
            data_rank=data_rank,
            data_world_size=data_world_size,
            image_seq_len_override=image_seq_len_override,
            vocab_size_override=vocab_size_override,
        )
        datasets.append(ds)

        if use_data_resampling:
            lengths.append(math.sqrt(len(ds)))
        else:
            lengths.append(len(ds))

    if use_packed_ds:
        total_length = sum(lengths)
        return PackedDataset(
            tokenizer=processor.tokenizer,
            datasets=datasets,
            dataset_weight=[l / total_length for l in lengths],
            max_packed_tokens=max_packed_tokens,
            data_rank=data_rank,
            data_world_size=data_world_size,
        )
    elif use_data_resampling:
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        return WeightedConcatDataset(datasets, weights)
    else:
        return ConcatDataset(datasets)


def prepare_inference_inputs(
    processor,
    video_path: str,
    query: str,
    model_type: str = "smolvlm",
    paradigm: str = "distime",
    num_frames: int = 16,
    image_size: int = 384,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """为推理准备输入"""
    frames, frame_times, duration = load_video(video_path, num_frames=num_frames)

    image_seq_len = getattr(processor, 'image_seq_len', 81)
    image_token = getattr(processor, 'image_token', '<image>')

    # Molmo2: use video processor for proper patches + pooling
    molmo2_video_inputs = None
    if model_type == "molmo2":
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
        molmo2_video_inputs = {
            'pixel_values_videos': video_inputs['pixel_values_videos'].to(device),
            'video_token_pooling': video_inputs['video_token_pooling'].to(device),
            'video_grids': video_inputs['video_grids'].to(device),
        }

        # Molmo2 image token from config
        config = getattr(processor, 'config', None) or getattr(processor, 'image_processor', None)
        image_patch_id = getattr(config, 'image_patch_id', None) or 151938
        image_token = processor.tokenizer.convert_ids_to_tokens(image_patch_id) or '<im_patch>'

        # Compute pooled grid size
        ps = getattr(video_proc, 'pooling_size', [3, 3])
        grid_h = (378 // 14) // ps[0]  # 9
        grid_w = (378 // 14) // ps[1]  # 9
        image_seq_len = grid_h * grid_w  # 81
        pixel_values = None  # Not used for Molmo2
    else:
        normalize_type = MODEL_DEFAULT_NORMALIZE.get(model_type, "siglip")
        transform = build_transform(image_size, is_train=False, normalize_type=normalize_type)
        pil_frames = [Image.fromarray(f.numpy()) for f in frames]
        pixel_values = torch.stack([transform(f) for f in pil_frames])

    if model_type == "smolvlm":
        fake_token = '<fake_token_around_image>'
        global_token = '<global-img>'
        frame_strings = []
        for i in range(len(frames)):
            img_tokens = image_token * image_seq_len
            frame_strings.append(f"{FRAME_TIME_TOKEN}: {fake_token}{global_token}{img_tokens}{fake_token}")
    elif model_type == "fastvlm":
        if hasattr(processor, 'patch_size') and processor.patch_size is not None:
            crop_h = image_size
            if hasattr(processor, 'image_processor') and hasattr(processor.image_processor, 'crop_size'):
                crop_h = processor.image_processor.crop_size.get('height', image_size)
            image_seq_len = (crop_h // processor.patch_size) ** 2
        frame_strings = []
        for i in range(len(frames)):
            img_tokens = image_token * image_seq_len
            frame_strings.append(f"{FRAME_TIME_TOKEN}: {img_tokens}")
    elif model_type == "molmo2":
        # Molmo2: use <frame_start>/<frame_end> for video mode
        all_patch_tokens = image_token * image_seq_len
        frame_strings = []
        for i in range(len(frames)):
            frame_strings.append(f"{FRAME_TIME_TOKEN}: <frame_start>{all_patch_tokens}<frame_end>")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    frame_prompt = "\n".join(frame_strings)

    if VIDEO_PLACEHOLDER in query:
        user_text = query.replace(VIDEO_PLACEHOLDER, frame_prompt)
    else:
        user_text = f"{frame_prompt}\n{query}"

    if model_type == "smolvlm":
        conversation = f"<|im_start|>User: {user_text}<end_of_utterance>\nAssistant:"
    elif model_type in ("fastvlm", "molmo2"):
        conversation = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    else:
        conversation = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"

    encoded = processor.tokenizer(
        conversation, return_tensors="pt", add_special_tokens=True,
    )

    result = {
        'input_ids': encoded.input_ids.to(device),
        'attention_mask': encoded.attention_mask.to(device),
        'frame_times': torch.tensor(frame_times).unsqueeze(0).to(device),
        'duration': duration,
    }

    if molmo2_video_inputs is not None:
        result.update(molmo2_video_inputs)
    else:
        result['pixel_values'] = pixel_values.unsqueeze(0).to(device)

    return result


# =============================================================================
# 兼容性别名
# =============================================================================

TemporalGroundingDataset = LazySupervisedDataset
DenseCaptioningDataset = LazySupervisedDataset
SmolVLMManualDataset = LazySupervisedDataset
collate_fn_manual = collate_fn
