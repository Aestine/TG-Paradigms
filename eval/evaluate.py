"""
Main evaluation script for SmolVLM-DisTime.

Usage:
    python -m eval.evaluate \
        --model_path /path/to/model \
        --benchmark charades_sta \
        --data_path /path/to/data \
        --output_dir /path/to/results
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.smolvlm_distime import SmolVLMDisTime, DisTimeConfig
from utils.metrics import (
    compute_temporal_metrics,
    compute_dvc_metrics,
    compute_gvqa_metrics,
    aggregate_metrics,
)
from utils.mm_utils import get_video_info

logger = logging.getLogger(__name__)


@dataclass
class EvalArguments:
    """Arguments for evaluation."""
    model_path: str = field(metadata={"help": "Path to trained model"})
    benchmark: str = field(metadata={"help": "Benchmark to evaluate on"})
    data_path: str = field(metadata={"help": "Path to evaluation data"})
    video_folder: str = field(default="", metadata={"help": "Path to video folder"})
    output_dir: str = field(default="./eval_results", metadata={"help": "Output directory"})
    batch_size: int = field(default=1, metadata={"help": "Evaluation batch size"})
    num_frames: int = field(default=16, metadata={"help": "Number of frames to sample"})
    max_new_tokens: int = field(default=256, metadata={"help": "Max tokens to generate"})
    device: str = field(default="cuda", metadata={"help": "Device to use"})
    dtype: str = field(default="bfloat16", metadata={"help": "Data type"})
    save_predictions: bool = field(default=True, metadata={"help": "Save predictions to file"})


# ============================================================================
# Moment Retrieval Evaluation
# ============================================================================

def evaluate_moment_retrieval(
    model: SmolVLMDisTime,
    data_path: str,
    video_folder: str,
    num_frames: int = 16,
    max_new_tokens: int = 256,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Evaluate on Moment Retrieval benchmarks (Charades-STA, ANet-Caption, QVHighlights).
    
    Expected data format (JSON lines):
    {
        "video": "video_id.mp4",
        "query": "person opens a door",
        "start": 10.5,
        "end": 15.2,
        "duration": 30.0
    }
    """
    model.eval()
    
    # Load data
    with open(data_path, 'r') as f:
        if data_path.endswith('.jsonl'):
            data = [json.loads(line) for line in f]
        else:
            data = json.load(f)
    
    all_predictions = []
    all_targets = []
    all_results = []
    
    logger.info(f"Evaluating on {len(data)} samples...")
    
    for item in tqdm(data, desc="Moment Retrieval"):
        video_path = os.path.join(video_folder, item['video'])
        query = item['query']
        gt_start = item['start']
        gt_end = item['end']
        duration = item.get('duration')
        
        # Get video duration if not provided
        if duration is None:
            info = get_video_info(video_path)
            duration = info['duration']
        
        # Prepare input
        prompt = f"<video>\nWhen does '{query}' happen in this video? Answer with the start and end timestamps."
        
        try:
            # Process video
            from data.dataset_v1 import TemporalGroundingDataset
            from utils.mm_utils import load_video
            
            frames, frame_times, _ = load_video(video_path, num_frames=num_frames)
            
            # Convert to PIL for processor
            from PIL import Image
            pil_frames = [Image.fromarray(f.numpy()) for f in frames]
            
            # Process inputs
            inputs = model.processor(
                text=prompt,
                images=pil_frames,
                return_tensors="pt",
            ).to(device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pixel_values=inputs.get('pixel_values'),
                    duration=torch.tensor([duration], device=device),
                    max_new_tokens=max_new_tokens,
                )
            
            # Extract predictions
            pred_times = outputs.get('pred_times', [None])[0]
            
            if pred_times is not None:
                pred_start = pred_times[0, 0].item() * duration
                pred_end = pred_times[0, 1].item() * duration
            else:
                # Fallback: parse from text
                pred_start, pred_end = 0.0, duration
            
            # Store results
            all_predictions.append([pred_start / duration, pred_end / duration])
            all_targets.append([gt_start / duration, gt_end / duration])
            
            all_results.append({
                'video': item['video'],
                'query': query,
                'pred_start': pred_start,
                'pred_end': pred_end,
                'gt_start': gt_start,
                'gt_end': gt_end,
                'duration': duration,
            })
            
        except Exception as e:
            logger.warning(f"Error processing {video_path}: {e}")
            all_predictions.append([0.0, 1.0])
            all_targets.append([gt_start / duration, gt_end / duration])
    
    # Compute metrics
    pred_tensor = torch.tensor(all_predictions)
    target_tensor = torch.tensor(all_targets)
    
    metrics = compute_temporal_metrics(pred_tensor, target_tensor)
    
    return {
        'metrics': metrics,
        'predictions': all_results,
        'num_samples': len(data),
    }


# ============================================================================
# Dense Video Captioning Evaluation
# ============================================================================

def evaluate_dense_captioning(
    model: SmolVLMDisTime,
    data_path: str,
    video_folder: str,
    num_frames: int = 16,
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Evaluate on Dense Video Captioning benchmarks (YouCook2, ANet-Caption).
    
    Expected data format:
    {
        "video": "video_id.mp4",
        "duration": 120.0,
        "events": [
            {"start": 0.0, "end": 10.5, "caption": "person enters room"},
            {"start": 10.5, "end": 25.0, "caption": "person sits down"},
            ...
        ]
    }
    """
    model.eval()
    
    with open(data_path, 'r') as f:
        if data_path.endswith('.jsonl'):
            data = [json.loads(line) for line in f]
        else:
            data = json.load(f)
    
    all_results = []
    all_metrics = []
    
    logger.info(f"Evaluating on {len(data)} videos...")
    
    for item in tqdm(data, desc="Dense Captioning"):
        video_path = os.path.join(video_folder, item['video'])
        duration = item.get('duration')
        gt_events = item.get('events', [])
        
        if duration is None:
            info = get_video_info(video_path)
            duration = info['duration']
        
        prompt = "<video>\nDescribe all the events in this video with their timestamps."
        
        try:
            from utils.mm_utils import load_video
            from PIL import Image
            
            frames, frame_times, _ = load_video(video_path, num_frames=num_frames)
            pil_frames = [Image.fromarray(f.numpy()) for f in frames]
            
            inputs = model.processor(
                text=prompt,
                images=pil_frames,
                return_tensors="pt",
            ).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pixel_values=inputs.get('pixel_values'),
                    duration=torch.tensor([duration], device=device),
                    max_new_tokens=max_new_tokens,
                )
            
            # Parse predicted events from output
            generated_text = outputs.get('generated_text', [''])[0]
            pred_times_list = outputs.get('pred_times', [None])[0]
            
            # Simple parsing: extract events from text
            pred_events = parse_dense_caption_output(generated_text, pred_times_list, duration)
            
            # Compute per-video metrics
            video_metrics = compute_dvc_metrics(pred_events, gt_events)
            all_metrics.append(video_metrics)
            
            all_results.append({
                'video': item['video'],
                'pred_events': pred_events,
                'gt_events': gt_events,
                'metrics': video_metrics,
            })
            
        except Exception as e:
            logger.warning(f"Error processing {video_path}: {e}")
    
    # Aggregate metrics
    aggregated = aggregate_metrics(all_metrics)
    
    return {
        'metrics': aggregated,
        'predictions': all_results,
        'num_samples': len(data),
    }


def parse_dense_caption_output(
    text: str,
    pred_times: Optional[torch.Tensor],
    duration: float,
) -> List[Dict]:
    """Parse dense captioning output into event list."""
    events = []
    
    # Split by TIME_STAMP token or common separators
    parts = text.split('<TIME_STAMP>')
    
    for i, part in enumerate(parts[1:]):  # Skip first empty part
        caption = part.strip().split('\n')[0].strip()
        
        if pred_times is not None and i < len(pred_times):
            start = pred_times[i, 0].item() * duration
            end = pred_times[i, 1].item() * duration
        else:
            # Estimate from position
            start = i * duration / len(parts)
            end = (i + 1) * duration / len(parts)
        
        if caption:
            events.append({
                'start': start,
                'end': end,
                'caption': caption,
            })
    
    return events


# ============================================================================
# Grounded VQA Evaluation
# ============================================================================

def evaluate_grounded_vqa(
    model: SmolVLMDisTime,
    data_path: str,
    video_folder: str,
    num_frames: int = 16,
    max_new_tokens: int = 128,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Evaluate on Grounded Video QA benchmarks (NExT-GQA).
    
    Expected data format:
    {
        "video": "video_id.mp4",
        "question": "What did the person do after sitting down?",
        "answer": "picked up a book",
        "start": 15.0,
        "end": 25.0,
        "duration": 60.0
    }
    """
    model.eval()
    
    with open(data_path, 'r') as f:
        if data_path.endswith('.jsonl'):
            data = [json.loads(line) for line in f]
        else:
            data = json.load(f)
    
    all_pred_answers = []
    all_gt_answers = []
    all_pred_times = []
    all_gt_times = []
    all_results = []
    
    logger.info(f"Evaluating on {len(data)} samples...")
    
    for item in tqdm(data, desc="Grounded VQA"):
        video_path = os.path.join(video_folder, item['video'])
        question = item['question']
        gt_answer = item['answer']
        gt_start = item['start']
        gt_end = item['end']
        duration = item.get('duration')
        
        if duration is None:
            info = get_video_info(video_path)
            duration = info['duration']
        
        prompt = f"<video>\n{question} Also indicate when this happens in the video."
        
        try:
            from utils.mm_utils import load_video
            from PIL import Image
            
            frames, frame_times, _ = load_video(video_path, num_frames=num_frames)
            pil_frames = [Image.fromarray(f.numpy()) for f in frames]
            
            inputs = model.processor(
                text=prompt,
                images=pil_frames,
                return_tensors="pt",
            ).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pixel_values=inputs.get('pixel_values'),
                    duration=torch.tensor([duration], device=device),
                    max_new_tokens=max_new_tokens,
                )
            
            # Extract answer from generated text
            generated_text = outputs.get('generated_text', [''])[0]
            pred_answer = generated_text.split('<TIME_STAMP>')[0].strip()
            
            # Extract timestamps
            pred_times = outputs.get('pred_times', [None])[0]
            if pred_times is not None:
                pred_start = pred_times[0, 0].item()
                pred_end = pred_times[0, 1].item()
            else:
                pred_start, pred_end = 0.0, 1.0
            
            all_pred_answers.append(pred_answer)
            all_gt_answers.append(gt_answer)
            all_pred_times.append([pred_start, pred_end])
            all_gt_times.append([gt_start / duration, gt_end / duration])
            
            all_results.append({
                'video': item['video'],
                'question': question,
                'pred_answer': pred_answer,
                'gt_answer': gt_answer,
                'pred_start': pred_start * duration,
                'pred_end': pred_end * duration,
                'gt_start': gt_start,
                'gt_end': gt_end,
            })
            
        except Exception as e:
            logger.warning(f"Error processing {video_path}: {e}")
            all_pred_answers.append("")
            all_gt_answers.append(gt_answer)
            all_pred_times.append([0.0, 1.0])
            all_gt_times.append([gt_start / duration, gt_end / duration])
    
    # Compute metrics
    pred_times_tensor = torch.tensor(all_pred_times)
    gt_times_tensor = torch.tensor(all_gt_times)
    
    metrics = compute_gvqa_metrics(
        all_pred_answers, all_gt_answers,
        pred_times_tensor, gt_times_tensor
    )
    
    return {
        'metrics': metrics,
        'predictions': all_results,
        'num_samples': len(data),
    }


# ============================================================================
# Main Evaluation Entry Point
# ============================================================================

def evaluate_model(
    model: SmolVLMDisTime,
    benchmark: str,
    data_path: str,
    video_folder: str,
    **kwargs,
) -> Dict[str, Any]:
    """
    Main evaluation entry point.
    
    Args:
        model: Trained SmolVLMDisTime model
        benchmark: One of 'charades_sta', 'anet_caption', 'qvhighlights',
                   'youcook2', 'nextgqa', 'mvbench', 'videomme'
        data_path: Path to evaluation data
        video_folder: Path to videos
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with metrics and predictions
    """
    benchmark = benchmark.lower()
    
    # Moment Retrieval benchmarks
    if benchmark in ['charades_sta', 'charades', 'anet_caption', 'activitynet', 'qvhighlights']:
        return evaluate_moment_retrieval(model, data_path, video_folder, **kwargs)
    
    # Dense Video Captioning benchmarks
    elif benchmark in ['youcook2', 'anet_caption_dvc']:
        return evaluate_dense_captioning(model, data_path, video_folder, **kwargs)
    
    # Grounded VQA benchmarks
    elif benchmark in ['nextgqa', 'next_gqa']:
        return evaluate_grounded_vqa(model, data_path, video_folder, **kwargs)
    
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def load_model(
    model_path: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> SmolVLMDisTime:
    """Load trained model from checkpoint."""
    
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)
    
    logger.info(f"Loading model from {model_path}...")
    
    model = SmolVLMDisTime.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
    )
    model = model.to(device)
    model.eval()
    
    return model


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate SmolVLM-DisTime")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name")
    parser.add_argument("--data_path", type=str, required=True, help="Path to eval data")
    parser.add_argument("--video_folder", type=str, default="", help="Video folder")
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Output dir")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_frames", type=int, default=16, help="Frames to sample")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max generation tokens")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    parser.add_argument("--save_predictions", action="store_true", help="Save predictions")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        level=logging.INFO,
    )
    
    # Load model
    model = load_model(args.model_path, args.device, args.dtype)
    
    # Run evaluation
    results = evaluate_model(
        model=model,
        benchmark=args.benchmark,
        data_path=args.data_path,
        video_folder=args.video_folder,
        num_frames=args.num_frames,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    
    # Print metrics
    logger.info("=" * 50)
    logger.info(f"Benchmark: {args.benchmark}")
    logger.info(f"Samples: {results['num_samples']}")
    logger.info("Metrics:")
    for k, v in results['metrics'].items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info("=" * 50)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_file = os.path.join(args.output_dir, f"{args.benchmark}_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            'benchmark': args.benchmark,
            'model_path': args.model_path,
            'metrics': results['metrics'],
            'num_samples': results['num_samples'],
        }, f, indent=2)
    logger.info(f"Saved metrics to {output_file}")
    
    if args.save_predictions:
        pred_file = os.path.join(args.output_dir, f"{args.benchmark}_predictions.json")
        with open(pred_file, 'w') as f:
            json.dump(results['predictions'], f, indent=2)
        logger.info(f"Saved predictions to {pred_file}")


if __name__ == "__main__":
    main()
