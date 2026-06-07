"""
Multimodal utility functions for video/image processing.

Functions:
- Video loading (decord, cv2)
- Frame sampling (uniform, fps-based)
- Image preprocessing
- Token manipulation utilities
"""

import os
import logging
from typing import List, Optional, Tuple, Union

import torch
import numpy as np

logger = logging.getLogger(__name__)

# Try to import video processing libraries
try:
    import decord
    decord.bridge.set_bridge('torch')
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False
    logger.warning("decord not available, falling back to OpenCV")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ============================================================================
# Video Loading Functions
# ============================================================================

def load_video_decord(
    video_path: str,
    num_frames: int = 16,
    fps: Optional[float] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Tuple[torch.Tensor, List[float], float]:
    """
    Load video frames using decord.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        fps: Target FPS for sampling (if None, uniform sampling)
        start_time: Start time in seconds (optional, for clip extraction)
        end_time: End time in seconds (optional, for clip extraction)
        
    Returns:
        frames: (num_frames, H, W, C) tensor of frames
        frame_times: List of timestamps for each frame in seconds
        duration: Total video duration in seconds
    """
    if not HAS_DECORD:
        raise ImportError("decord is required for load_video_decord")
    
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total_frames = len(vr)
    video_fps = vr.get_avg_fps()
    duration = total_frames / video_fps
    
    # Determine frame range
    if start_time is not None and end_time is not None:
        start_frame = int(start_time * video_fps)
        end_frame = int(end_time * video_fps)
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))
    else:
        start_frame = 0
        end_frame = total_frames
    
    frame_range = end_frame - start_frame
    
    # Sample frame indices
    if fps is not None and fps > 0:
        # Sample at target FPS
        step = video_fps / fps
        indices = [int(start_frame + i * step) for i in range(int(frame_range / step))]
        indices = indices[:num_frames]
    else:
        # Uniform sampling
        indices = uniform_sample_indices(frame_range, num_frames)
        indices = [start_frame + i for i in indices]
    
    # Ensure indices are valid
    indices = [min(i, total_frames - 1) for i in indices]
    
    # Load frames
    frames = vr.get_batch(indices)  # (N, H, W, C)
    
    # Compute frame times
    frame_times = [i / video_fps for i in indices]
    
    return frames, frame_times, duration


def load_video_cv2(
    video_path: str,
    num_frames: int = 16,
    fps: Optional[float] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Tuple[torch.Tensor, List[float], float]:
    """
    Load video frames using OpenCV (fallback when decord unavailable).
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        fps: Target FPS for sampling (if None, uniform sampling)
        start_time: Start time in seconds (optional)
        end_time: End time in seconds (optional)
        
    Returns:
        frames: (num_frames, H, W, C) tensor of frames (RGB)
        frame_times: List of timestamps for each frame in seconds
        duration: Total video duration in seconds
    """
    if not HAS_CV2:
        raise ImportError("OpenCV (cv2) is required for load_video_cv2")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    if video_fps <= 0:
        video_fps = 30.0  # Default fallback
    
    duration = total_frames / video_fps
    
    # Determine frame range
    if start_time is not None and end_time is not None:
        start_frame = int(start_time * video_fps)
        end_frame = int(end_time * video_fps)
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))
    else:
        start_frame = 0
        end_frame = total_frames
    
    frame_range = end_frame - start_frame
    
    # Sample frame indices
    if fps is not None and fps > 0:
        step = video_fps / fps
        indices = [int(start_frame + i * step) for i in range(int(frame_range / step))]
        indices = indices[:num_frames]
    else:
        indices = uniform_sample_indices(frame_range, num_frames)
        indices = [start_frame + i for i in indices]
    
    indices = [min(i, total_frames - 1) for i in indices]
    
    # Load frames
    frames = []
    frame_times = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_times.append(idx / video_fps)
        else:
            logger.warning(f"Failed to read frame {idx} from {video_path}")
    
    cap.release()
    
    if not frames:
        raise ValueError(f"No frames could be read from {video_path}")
    
    frames = torch.from_numpy(np.stack(frames))  # (N, H, W, C)
    
    return frames, frame_times, duration


def load_video(
    video_path: str,
    num_frames: int = 16,
    fps: Optional[float] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    backend: str = "auto",
) -> Tuple[torch.Tensor, List[float], float]:
    """
    Load video frames with automatic backend selection.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        fps: Target FPS for sampling
        start_time: Start time in seconds
        end_time: End time in seconds
        backend: "decord", "cv2", or "auto"
        
    Returns:
        frames: (num_frames, H, W, C) tensor
        frame_times: List of timestamps
        duration: Video duration in seconds
    """
    if backend == "auto":
        backend = "decord" if HAS_DECORD else "cv2"
    
    if backend == "decord":
        if not HAS_DECORD:
            raise ImportError("decord not available")
        return load_video_decord(video_path, num_frames, fps, start_time, end_time)
    elif backend == "cv2":
        if not HAS_CV2:
            raise ImportError("OpenCV not available")
        return load_video_cv2(video_path, num_frames, fps, start_time, end_time)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ============================================================================
# Frame Sampling Functions
# ============================================================================

def uniform_sample_indices(total: int, num_samples: int) -> List[int]:
    """
    Generate uniformly spaced indices.
    
    Args:
        total: Total number of items
        num_samples: Number of samples to select
        
    Returns:
        List of selected indices
    """
    if num_samples >= total:
        return list(range(total))
    
    # Uniform spacing
    step = total / num_samples
    indices = [int(i * step + step / 2) for i in range(num_samples)]
    
    # Ensure no duplicates and within bounds
    indices = sorted(set(min(i, total - 1) for i in indices))
    
    # Pad if needed
    while len(indices) < num_samples and indices[-1] < total - 1:
        indices.append(indices[-1] + 1)
    
    return indices[:num_samples]


def fps_sample_indices(
    total_frames: int,
    video_fps: float,
    target_fps: float,
    max_frames: Optional[int] = None,
) -> List[int]:
    """
    Sample frame indices at a target FPS.
    
    Args:
        total_frames: Total number of frames in video
        video_fps: Original video FPS
        target_fps: Target sampling FPS
        max_frames: Maximum number of frames to return
        
    Returns:
        List of frame indices
    """
    step = video_fps / target_fps
    indices = [int(i * step) for i in range(int(total_frames / step))]
    indices = [i for i in indices if i < total_frames]
    
    if max_frames is not None and len(indices) > max_frames:
        # Subsample uniformly
        indices = [indices[i] for i in uniform_sample_indices(len(indices), max_frames)]
    
    return indices


def temporal_sample_indices(
    total_frames: int,
    video_fps: float,
    start_time: float,
    end_time: float,
    num_frames: int,
) -> Tuple[List[int], List[float]]:
    """
    Sample frames within a specific time range.
    
    Args:
        total_frames: Total frames in video
        video_fps: Video FPS
        start_time: Start time in seconds
        end_time: End time in seconds
        num_frames: Number of frames to sample
        
    Returns:
        indices: List of frame indices
        times: List of corresponding timestamps
    """
    start_frame = int(start_time * video_fps)
    end_frame = int(end_time * video_fps)
    
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame + 1, min(end_frame, total_frames))
    
    frame_range = end_frame - start_frame
    local_indices = uniform_sample_indices(frame_range, num_frames)
    
    indices = [start_frame + i for i in local_indices]
    times = [i / video_fps for i in indices]
    
    return indices, times


# ============================================================================
# Image Processing Functions
# ============================================================================

def resize_image(
    image: Union[torch.Tensor, np.ndarray],
    size: Tuple[int, int],
    keep_aspect_ratio: bool = False,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Resize image to target size.
    
    Args:
        image: (H, W, C) image array/tensor
        size: (height, width) target size
        keep_aspect_ratio: Whether to maintain aspect ratio (pad if needed)
        
    Returns:
        Resized image
    """
    is_tensor = isinstance(image, torch.Tensor)
    
    if is_tensor:
        image = image.numpy()
    
    if not HAS_CV2:
        raise ImportError("OpenCV required for image resizing")
    
    h, w = image.shape[:2]
    target_h, target_w = size
    
    if keep_aspect_ratio:
        # Compute scale to fit within target size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        resized = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    else:
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    if is_tensor:
        resized = torch.from_numpy(resized)
    
    return resized


def normalize_image(
    image: torch.Tensor,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """
    Normalize image tensor.
    
    Args:
        image: (C, H, W) or (H, W, C) image tensor in [0, 1] or [0, 255]
        mean: Normalization mean per channel
        std: Normalization std per channel
        
    Returns:
        Normalized image tensor (C, H, W)
    """
    # Ensure float
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    
    # Ensure CHW format
    if image.shape[-1] == 3:  # HWC -> CHW
        image = image.permute(2, 0, 1)
    
    mean = torch.tensor(mean, device=image.device).view(-1, 1, 1)
    std = torch.tensor(std, device=image.device).view(-1, 1, 1)
    
    return (image - mean) / std


# ============================================================================
# Token Manipulation Functions
# ============================================================================

def find_token_positions(
    input_ids: torch.Tensor,
    token_id: int,
) -> torch.Tensor:
    """
    Find all positions of a specific token in input_ids.
    
    Args:
        input_ids: (batch, seq_len) token IDs
        token_id: Token ID to search for
        
    Returns:
        Boolean mask of same shape as input_ids
    """
    return input_ids == token_id


def count_tokens(
    input_ids: torch.Tensor,
    token_id: int,
) -> torch.Tensor:
    """
    Count occurrences of a token in each sequence.
    
    Args:
        input_ids: (batch, seq_len) token IDs
        token_id: Token ID to count
        
    Returns:
        (batch,) count per sequence
    """
    return (input_ids == token_id).sum(dim=-1)


def replace_token_embeddings(
    embeddings: torch.Tensor,
    mask: torch.Tensor,
    replacement: torch.Tensor,
) -> torch.Tensor:
    """
    Replace embeddings at masked positions.
    
    Args:
        embeddings: (batch, seq_len, hidden) embeddings
        mask: (batch, seq_len) boolean mask
        replacement: (N, hidden) replacement embeddings where N = mask.sum()
        
    Returns:
        Updated embeddings tensor
    """
    embeddings = embeddings.clone()
    embeddings[mask] = replacement
    return embeddings


def insert_frame_timestamps(
    text: str,
    frame_times: List[float],
    frame_token: str = "<FRAME_TIME>",
) -> str:
    """
    Insert frame timestamp tokens into text.
    
    Args:
        text: Original text
        frame_times: List of frame timestamps
        frame_token: Token to use for frame times
        
    Returns:
        Text with frame tokens prepended
    """
    frame_tokens = " ".join([frame_token] * len(frame_times))
    return f"{frame_tokens} {text}"


# ============================================================================
# Video Information Functions
# ============================================================================

def get_video_info(video_path: str) -> dict:
    """
    Get video metadata.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with duration, fps, total_frames, width, height
    """
    if HAS_DECORD:
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        # Get frame shape
        frame = vr[0]
        height, width = frame.shape[:2]
    elif HAS_CV2:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    else:
        raise ImportError("Either decord or OpenCV is required")
    
    duration = total_frames / fps if fps > 0 else 0
    
    return {
        'duration': duration,
        'fps': fps,
        'total_frames': total_frames,
        'width': width,
        'height': height,
    }


def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert HH:MM:SS.mmm or MM:SS format to seconds."""
    parts = timestamp.split(':')
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    elif len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    else:
        return float(timestamp)
