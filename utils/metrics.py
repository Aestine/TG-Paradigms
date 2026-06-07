"""
Evaluation metrics for temporal grounding tasks.

Metrics implemented:
- mIoU: Mean Intersection over Union
- R@K: Recall at IoU threshold K (0.3, 0.5, 0.7)
- mAP: Mean Average Precision (for highlight detection)
- SODA_c, CIDEr, METEOR, F1: For dense video captioning
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def compute_iou_1d(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute 1D IoU between predicted and target time intervals.
    
    Args:
        pred: (N, 2) predicted [start, end] times
        target: (N, 2) ground truth [start, end] times
        
    Returns:
        (N,) IoU values for each sample
    """
    # Ensure proper ordering
    pred_start = pred[:, 0].clamp(min=0)
    pred_end = pred[:, 1].clamp(min=0)
    target_start = target[:, 0]
    target_end = target[:, 1]
    
    # Ensure start <= end
    pred_start, pred_end = torch.min(pred_start, pred_end), torch.max(pred_start, pred_end)
    
    # Intersection
    inter_start = torch.max(pred_start, target_start)
    inter_end = torch.min(pred_end, target_end)
    intersection = (inter_end - inter_start).clamp(min=0)
    
    # Union
    pred_len = pred_end - pred_start
    target_len = target_end - target_start
    union = pred_len + target_len - intersection
    
    # IoU
    iou = intersection / (union + 1e-8)
    return iou


def compute_temporal_metrics(
    pred_times: Union[torch.Tensor, np.ndarray, List],
    target_times: Union[torch.Tensor, np.ndarray, List],
    iou_thresholds: List[float] = [0.3, 0.5, 0.7],
) -> Dict[str, float]:
    """
    Compute standard temporal grounding metrics.
    
    Args:
        pred_times: (N, 2) predicted [start, end] times (normalized to [0, 1])
        target_times: (N, 2) ground truth [start, end] times (normalized to [0, 1])
        iou_thresholds: List of IoU thresholds for recall computation
        
    Returns:
        Dictionary with mIoU and R@K metrics
    """
    # Convert to tensors if needed
    if not isinstance(pred_times, torch.Tensor):
        pred_times = torch.tensor(pred_times, dtype=torch.float32)
    if not isinstance(target_times, torch.Tensor):
        target_times = torch.tensor(target_times, dtype=torch.float32)
    
    # Handle empty predictions
    if pred_times.numel() == 0 or target_times.numel() == 0:
        metrics = {'mIoU': 0.0}
        for thresh in iou_thresholds:
            metrics[f'R@{thresh}'] = 0.0
        return metrics
    
    # Ensure 2D
    if pred_times.dim() == 1:
        pred_times = pred_times.unsqueeze(0)
    if target_times.dim() == 1:
        target_times = target_times.unsqueeze(0)
    
    # Compute IoU
    ious = compute_iou_1d(pred_times, target_times)
    
    # Compute metrics
    metrics = {
        'mIoU': ious.mean().item(),
    }
    
    for thresh in iou_thresholds:
        recall = (ious >= thresh).float().mean().item()
        metrics[f'R@{thresh}'] = recall
    
    return metrics


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute metrics for HuggingFace Trainer.
    
    This function is designed to be passed to Trainer's compute_metrics parameter.
    
    Args:
        eval_pred: EvalPrediction object with predictions and label_ids
        
    Returns:
        Dictionary with mIoU, R@0.3, R@0.5, R@0.7
    """
    predictions, labels = eval_pred
    
    # Handle different prediction formats
    if isinstance(predictions, tuple):
        pred_times = predictions[0] if predictions[0] is not None else None
    elif isinstance(predictions, dict):
        pred_times = predictions.get('pred_times')
    else:
        pred_times = predictions
    
    # Handle different label formats
    if isinstance(labels, dict):
        target_times = labels.get('time_gt')
    else:
        target_times = labels
    
    # If no valid predictions, return zeros
    if pred_times is None or (hasattr(pred_times, '__len__') and len(pred_times) == 0):
        return {
            'mIoU': 0.0,
            'R@0.3': 0.0,
            'R@0.5': 0.0,
            'R@0.7': 0.0,
        }
    
    return compute_temporal_metrics(pred_times, target_times)


# ============================================================================
# Highlight Detection Metrics (for QVHighlights)
# ============================================================================

def compute_highlight_metrics(
    pred_saliency: torch.Tensor,
    target_saliency: torch.Tensor,
    top_k: List[int] = [1, 5],
) -> Dict[str, float]:
    """
    Compute highlight detection metrics.
    
    Args:
        pred_saliency: (N, T) predicted saliency scores
        target_saliency: (N, T) ground truth saliency scores
        top_k: List of K values for Hit@K computation
        
    Returns:
        Dictionary with mAP and Hit@K metrics
    """
    metrics = {}
    
    # Mean Average Precision
    # Simplified: treat as binary classification per clip
    pred_flat = pred_saliency.flatten()
    target_flat = (target_saliency > 0).float().flatten()
    
    if target_flat.sum() > 0:
        # Sort by predicted scores
        sorted_indices = torch.argsort(pred_flat, descending=True)
        sorted_targets = target_flat[sorted_indices]
        
        # Compute precision at each recall level
        cumsum = sorted_targets.cumsum(0)
        precision = cumsum / (torch.arange(len(sorted_targets), device=pred_flat.device) + 1)
        recall_change = sorted_targets
        ap = (precision * recall_change).sum() / target_flat.sum()
        metrics['mAP'] = ap.item()
    else:
        metrics['mAP'] = 0.0
    
    # Hit@K for each sample
    batch_size = pred_saliency.shape[0]
    for k in top_k:
        hits = 0
        for b in range(batch_size):
            top_k_indices = torch.topk(pred_saliency[b], min(k, len(pred_saliency[b]))).indices
            if target_saliency[b, top_k_indices].max() > 0:
                hits += 1
        metrics[f'Hit@{k}'] = hits / batch_size
    
    return metrics


# ============================================================================
# Dense Video Captioning Metrics
# ============================================================================

def compute_dvc_metrics(
    pred_events: List[Dict],
    target_events: List[Dict],
    iou_thresholds: List[float] = [0.3, 0.5, 0.7, 0.9],
) -> Dict[str, float]:
    """
    Compute Dense Video Captioning metrics.
    
    This computes:
    - Temporal localization: Precision, Recall, F1 at various IoU thresholds
    - Captioning quality requires external tools (pycocoevalcap)
    
    Args:
        pred_events: List of {'start': float, 'end': float, 'caption': str}
        target_events: List of {'start': float, 'end': float, 'caption': str}
        iou_thresholds: IoU thresholds for matching
        
    Returns:
        Dictionary with precision, recall, F1 at each threshold
    """
    if not pred_events or not target_events:
        return {f'F1@{t}': 0.0 for t in iou_thresholds}
    
    metrics = {}
    
    for thresh in iou_thresholds:
        # Match predictions to ground truth
        matched_pred = set()
        matched_gt = set()
        
        for i, pred in enumerate(pred_events):
            pred_interval = torch.tensor([[pred['start'], pred['end']]])
            
            for j, gt in enumerate(target_events):
                if j in matched_gt:
                    continue
                    
                gt_interval = torch.tensor([[gt['start'], gt['end']]])
                iou = compute_iou_1d(pred_interval, gt_interval).item()
                
                if iou >= thresh:
                    matched_pred.add(i)
                    matched_gt.add(j)
                    break
        
        precision = len(matched_pred) / len(pred_events) if pred_events else 0
        recall = len(matched_gt) / len(target_events) if target_events else 0
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        metrics[f'P@{thresh}'] = precision
        metrics[f'R@{thresh}'] = recall
        metrics[f'F1@{thresh}'] = f1
    
    return metrics


# ============================================================================
# Grounded VQA Metrics (for NExT-GQA)
# ============================================================================

def compute_gvqa_metrics(
    pred_answers: List[str],
    target_answers: List[str],
    pred_times: torch.Tensor,
    target_times: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute Grounded Video QA metrics.
    
    Args:
        pred_answers: List of predicted answers
        target_answers: List of ground truth answers
        pred_times: (N, 2) predicted time intervals
        target_times: (N, 2) ground truth time intervals
        
    Returns:
        Dictionary with Acc@QA, Acc@GQA, mIoP, IoP@K metrics
    """
    n = len(pred_answers)
    if n == 0:
        return {
            'Acc@QA': 0.0,
            'Acc@GQA': 0.0,
            'mIoP': 0.0,
            'IoP@0.3': 0.0,
            'IoP@0.5': 0.0,
        }
    
    # QA Accuracy (exact match, case-insensitive)
    qa_correct = sum(
        1 for p, t in zip(pred_answers, target_answers)
        if p.strip().lower() == t.strip().lower()
    )
    acc_qa = qa_correct / n
    
    # Compute IoU for temporal grounding
    ious = compute_iou_1d(pred_times, target_times)
    
    # Grounded QA Accuracy (correct answer AND IoU > 0.5)
    gqa_correct = 0
    for i, (p, t) in enumerate(zip(pred_answers, target_answers)):
        if p.strip().lower() == t.strip().lower() and ious[i] >= 0.5:
            gqa_correct += 1
    acc_gqa = gqa_correct / n
    
    # IoP metrics (Intersection over Prediction)
    # IoP = intersection / pred_length
    pred_start, pred_end = pred_times[:, 0], pred_times[:, 1]
    target_start, target_end = target_times[:, 0], target_times[:, 1]
    
    inter_start = torch.max(pred_start, target_start)
    inter_end = torch.min(pred_end, target_end)
    intersection = (inter_end - inter_start).clamp(min=0)
    pred_len = (pred_end - pred_start).clamp(min=1e-8)
    
    iop = intersection / pred_len
    
    return {
        'Acc@QA': acc_qa,
        'Acc@GQA': acc_gqa,
        'mIoP': iop.mean().item(),
        'IoP@0.3': (iop >= 0.3).float().mean().item(),
        'IoP@0.5': (iop >= 0.5).float().mean().item(),
        'mIoU': ious.mean().item(),
        'IoU@0.3': (ious >= 0.3).float().mean().item(),
        'IoU@0.5': (ious >= 0.5).float().mean().item(),
    }


# ============================================================================
# Aggregation utilities
# ============================================================================

def aggregate_metrics(
    all_metrics: List[Dict[str, float]],
    weights: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Aggregate metrics from multiple evaluation batches.
    
    Args:
        all_metrics: List of metric dictionaries
        weights: Optional weights for each batch (e.g., batch sizes)
        
    Returns:
        Aggregated metrics dictionary
    """
    if not all_metrics:
        return {}
    
    if weights is None:
        weights = [1.0] * len(all_metrics)
    
    total_weight = sum(weights)
    aggregated = {}
    
    for key in all_metrics[0].keys():
        weighted_sum = sum(
            m[key] * w for m, w in zip(all_metrics, weights)
            if key in m
        )
        aggregated[key] = weighted_sum / total_weight
    
    return aggregated
