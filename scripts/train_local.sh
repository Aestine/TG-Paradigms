#!/bin/bash
# Local training script for debugging and development
# Usage: bash scripts/train_local.sh [num_gpus]

set -e

NUM_GPUS=${1:-1}  # Default to 1 GPU

# ============================================================
# Configuration - MODIFY THESE
# ============================================================
MODEL_NAME="HuggingFaceTB/SmolVLM2-2.2B-Instruct"
DATA_PATH="./data/sample_train.json"  # Use a small sample for debugging
VIDEO_FOLDER="./data/videos"
RUN_NAME="smolvlm-distime-debug"
OUTPUT_DIR="./checkpoints/${RUN_NAME}"

# ============================================================
# Environment
# ============================================================
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Adjust as needed
export WANDB_MODE="offline"  # Disable wandb for local debugging

# ============================================================
# Training
# ============================================================
if [ ${NUM_GPUS} -eq 1 ]; then
    # Single GPU training
    python scripts/train.py \
        --model_name_or_path ${MODEL_NAME} \
        --data_path ${DATA_PATH} \
        --video_folder ${VIDEO_FOLDER} \
        --output_dir ${OUTPUT_DIR} \
        --run_name ${RUN_NAME} \
        \
        --reg_max 32 \
        --num_time_layers 3 \
        --use_lora True \
        --lora_r 16 \
        --lora_alpha 32 \
        --freeze_vision_tower True \
        \
        --max_frames 8 \
        --fps 1.0 \
        --model_max_length 2048 \
        \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --num_train_epochs 1 \
        --max_steps 100 \
        --learning_rate 2e-5 \
        --warmup_steps 10 \
        \
        --bf16 True \
        --gradient_checkpointing True \
        \
        --logging_steps 1 \
        --save_strategy steps \
        --save_steps 50 \
        --save_total_limit 2 \
        \
        --dataloader_num_workers 2 \
        --report_to none \
        --seed 42

else
    # Multi-GPU training with DeepSpeed
    torchrun \
        --standalone \
        --nproc_per_node=${NUM_GPUS} \
        scripts/train.py \
        --deepspeed configs/zero2.json \
        --model_name_or_path ${MODEL_NAME} \
        --data_path ${DATA_PATH} \
        --video_folder ${VIDEO_FOLDER} \
        --output_dir ${OUTPUT_DIR} \
        --run_name ${RUN_NAME} \
        \
        --reg_max 32 \
        --num_time_layers 3 \
        --use_lora True \
        --lora_r 16 \
        --lora_alpha 32 \
        --freeze_vision_tower True \
        \
        --max_frames 8 \
        --fps 1.0 \
        --model_max_length 2048 \
        \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --num_train_epochs 1 \
        --max_steps 100 \
        --learning_rate 2e-5 \
        --warmup_steps 10 \
        \
        --bf16 True \
        --gradient_checkpointing True \
        \
        --logging_steps 1 \
        --save_strategy steps \
        --save_steps 50 \
        --save_total_limit 2 \
        \
        --dataloader_num_workers 2 \
        --report_to none \
        --seed 42
fi

echo "Debug training completed!"
