"""
Training arguments for SmolVLM-DisTime.
Based on SmolVLM2 training arguments.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments as HFTrainingArguments


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    
    model_name_or_path: str = field(
        default="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    # <<<< 新增: model_type 参数
    model_type: str = field(
        default="smolvlm",
        metadata={
            "help": "VLM backend type: 'smolvlm', 'fastvlm', or 'molmo2'. "
                    "Controls model class, dataset format, normalization, etc. "
                    "Note: molmo2 requires trust_remote_code=True (handled automatically)."
        }
    )
    # <<<< 新增: paradigm 参数
    paradigm: str = field(
        default="distime",
        metadata={
            "help": "Temporal output paradigm: 'distime' (distribution-based / continuous), "
                    "'trace' (character-level temporal tokens / generative), or "
                    "'text' (plain-text numeral timestamps, VTimeLLM-style). "
                    "Controls model class, dataset branch, loss function."
        }
    )
    # DisTime specific
    reg_max: int = field(
        default=32,
        metadata={"help": "Maximum regression value for time distribution"}
    )
    num_time_layers: int = field(
        default=3,
        metadata={"help": "Number of layers in time encoder/decoder MLPs"}
    )
    time_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for distribution focal loss"}
    )
    iou_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for DIoU loss"}
    )

    # LoRA config
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for fine-tuning"}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={"help": "Target modules for LoRA (comma-separated). If None, uses default for the model type. "
                  "SmolVLM/FastVLM: 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj'. "
                  "Molmo2: 'att_proj,attn_out,ff_proj,ff_out' (fused QKV naming)."}
    )
    # Vision token pooling (FastVLM only)
    vision_pool_stride: int = field(
        default=1,
        metadata={"help": "Vision token spatial pooling stride. "
                          "1=no pooling (256 tokens/frame), 2=4x compression (64 tokens/frame), "
                          "4=16x compression (16 tokens/frame). Only affects FastVLM."}
    )
    # Freezing config
    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "Freeze vision encoder"}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Freeze LLM backbone (use with LoRA)"}
    )

    # Precision
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Torch dtype for model weights (float16, bfloat16, float32)"}
    )
    # Note: gradient_checkpointing is in TrainingArguments


@dataclass
class DataArguments:
    """Arguments for data loading."""

    data_path: str = field(
        default=None,
        metadata={"help": "Path to training data (JSON/JSONL file)"}
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to evaluation data"}
    )
    video_folder: str = field(
        default=None,
        metadata={"help": "Root folder containing videos"}
    )

    # Video processing
    max_frames: int = field(
        default=16,
        metadata={"help": "Maximum number of frames to sample from video"}
    )
    fps: float = field(
        default=0.0,
        metadata={"help": "Frames per second for video sampling"}
    )
    video_target_size: int = field(
        default=384,
        metadata={"help": "Target size for video frames"}
    )

    # Sequence length
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length"}
    )
    # Note: dataloader_num_workers and dataloader_drop_last are in TrainingArguments
    multi_dataset: bool = field(
        default=False,
        metadata={'help': 'Whether to use multi-dataset training'}
    )
    use_data_resampling: bool = field(
        default=False,
        metadata={'help': 'Whether to use weighted sampling for multi-dataset'}
    )
    use_packed_ds: bool = field(
        default=False,
        metadata={'help': 'Whether to use packed dataset for efficient training'}
    )
    max_packed_tokens: int = field(
        default=4096,
        metadata={'help': 'Maximum tokens per packed sample'}
    )

@dataclass
class TrainingArguments(HFTrainingArguments):
    """Extended training arguments with DisTime-specific options."""

    # Override defaults
    output_dir: str = field(
        default="./checkpoints/smolvlm_distime",
        metadata={"help": "Output directory for checkpoints"}
    )

    # Learning rates for different components
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "Learning rate for LoRA/LLM"}
    )
    vision_tower_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for vision tower (if unfrozen)"}
    )
    time_module_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Learning rate for DisTime modules. Defaults to learning_rate"}
    )

    # Batch size
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size per GPU"}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Gradient accumulation steps"}
    )

    # Training schedule
    num_train_epochs: float = field(
        default=1.0,
        metadata={"help": "Number of training epochs"}
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "Warmup ratio"}
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "Learning rate scheduler type"}
    )

    # Precision
    bf16: bool = field(
        default=True,
        metadata={"help": "Use bfloat16 precision"}
    )
    tf32: bool = field(
        default=True,
        metadata={"help": "Use TF32 for matmul (Ampere+ GPUs)"}
    )

    # Gradient
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum gradient norm for clipping"}
    )

    # Logging
    logging_steps: int = field(
        default=1,
        metadata={"help": "Log every N steps"}
    )
    report_to: str = field(
        default="wandb",
        metadata={"help": "Logging backend (wandb, tensorboard, none)"}
    )
    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name for the training run (wandb)"}
    )

    # Saving
    save_strategy: str = field(
        default="steps",
        metadata={"help": "Save strategy (steps, epoch, no)"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every N steps"}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "Maximum number of checkpoints to keep"}
    )

    # DeepSpeed
    deepspeed: Optional[str] = field(
        default=None,
        metadata={"help": "Path to DeepSpeed config file"}
    )

    # Misc
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Remove unused columns from dataset"}
    )