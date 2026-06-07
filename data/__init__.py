"""DisTime data loading utilities."""

from .dataset import (
    LazySupervisedDataset,
    SmolVLMManualDataset,
    build_datasets,
    collate_fn,
    packed_collate_fn,
    prepare_inference_inputs,
    IGNORE_INDEX,
    TIME_STAMP_TOKEN,
    FRAME_TIME_TOKEN,
    SUPPORTED_MODEL_TYPES,
)

# 从 mm_utils 导入 load_video（如果需要）
try:
    from utils.mm_utils import load_video
except ImportError:
    load_video = None

__all__ = [
    'LazySupervisedDataset',
    'SmolVLMManualDataset',
    'build_datasets',
    'collate_fn',
    'packed_collate_fn',
    'prepare_inference_inputs',
    'load_video',
    'IGNORE_INDEX',
    'TIME_STAMP_TOKEN',
    'FRAME_TIME_TOKEN',
    'SUPPORTED_MODEL_TYPES',
]
