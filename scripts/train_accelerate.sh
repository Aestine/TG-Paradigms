#!/bin/bash
# Training with Accelerate launcher
# Usage: bash scripts/train_accelerate.sh

set -e

# ============================================================
# Configuration
# ============================================================
MODEL_NAME="HuggingFaceTB/SmolVLM2-2.2B-Instruct"
DATA_PATH="/path/to/train_data.json"
VIDEO_FOLDER="/path/to/videos"
RUN_NAME="smolvlm-distime-accelerate"
OUTPUT_DIR="./checkpoints/${RUN_NAME}"

# ============================================================
# Launch Training
# ============================================================
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    scripts/train.py \
    --model_name_or_path ${MODEL_NAME} \
    --data_path ${DATA_PATH} \
    --video_folder ${VIDEO_FOLDER} \
    --output_dir ${OUTPUT_DIR} \
    --run_name ${RUN_NAME} \
    \
    --reg_max 32 \
    --num_time_layers 3 \
    --use_lora True \
    --lora_r 64 \
    --lora_alpha 16 \
    --freeze_vision_tower True \
    \
    --max_frames 16 \
    --fps 1.0 \
    --model_max_length 4096 \
    \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --max_grad_norm 1.0 \
    \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --seed 42

echo "Training with accelerate completed!"
