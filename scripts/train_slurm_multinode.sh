#!/bin/bash
#SBATCH --job-name=smolvlm-distime
#SBATCH --nodes=8                      # Number of nodes
#SBATCH --ntasks-per-node=1            # Important for distributed usage
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8                   # 8 GPUs per node
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -x -e

# ============================================================
# User Configuration - MODIFY THESE
# ============================================================
# Model
MODEL_NAME="HuggingFaceTB/SmolVLM2-2.2B-Instruct"

# Data
DATA_PATH="/path/to/your/train_data.json"
VIDEO_FOLDER="/path/to/your/videos"
EVAL_DATA_PATH=""  # Optional

# Output
RUN_NAME="smolvlm-distime-v1"
OUTPUT_DIR="./checkpoints/${RUN_NAME}"

# Conda environment
CONDA_ENV="smolvlm"

# Project directory
PROJECT_DIR="/path/to/smolvlm_distime_v3"

# Cache directories
export HF_HOME="/path/to/cache/huggingface"
export WANDB_DIR="/path/to/cache/wandb"

# ============================================================
# Environment Setup
# ============================================================
# Activate conda
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# Debug prints
echo "============================================================"
echo "Job Information"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES}"
echo "GPUs per node: 8"
echo "Total GPUs: $((SLURM_NNODES * 8))"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"

# Set distributed training environment variables
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * 8))
export NODE_RANK=$SLURM_NODEID
export LOCAL_RANK=0

echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "WORLD_SIZE: ${WORLD_SIZE}"

# Set NCCL options for better performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# Create output directories
mkdir -p ${OUTPUT_DIR}
mkdir -p logs

# ============================================================
# Training Command
# ============================================================
cd ${PROJECT_DIR}

srun torchrun \
    --nproc_per_node=8 \
    --nnodes=${SLURM_NNODES} \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
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
    --time_loss_weight 1.0 \
    --iou_loss_weight 1.0 \
    \
    --use_lora True \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
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
    --weight_decay 0.0 \
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
    --dataloader_drop_last True \
    \
    --report_to wandb \
    --seed 42

echo "Training completed!"
