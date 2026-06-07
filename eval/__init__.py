"""
Evaluation module for SmolVLM-DisTime.

Supports evaluation on:
- Moment Retrieval: Charades-STA, ANet-Caption, QVHighlights
- Dense Video Captioning: YouCook2, ANet-Caption
- Grounded VQA: NExT-GQA
- General Video Understanding: MVBench, Video-MME
"""

from .evaluate import (
    evaluate_model,
    evaluate_moment_retrieval,
    evaluate_dense_captioning,
    evaluate_grounded_vqa,
)

__all__ = [
    'evaluate_model',
    'evaluate_moment_retrieval',
    'evaluate_dense_captioning',
    'evaluate_grounded_vqa',
]
