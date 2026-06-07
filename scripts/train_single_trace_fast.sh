#!/bin/bash

# =============================================================================
# FastVLM-TRACE 训练脚本 (单机)
# =============================================================================

# Environment Variables
WORLD_SIZE=1
NPROC_PER_NODE=1
MASTER_ADDR="127.0.0.1"
MASTER_PORT=16666
RANK=0

# Training Arguments
GLOBAL_BATCH_SIZE=10
GRADIENT_ACCUMULATION_STEPS=2
LOCAL_BATCH_SIZE=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$GRADIENT_ACCUMULATION_STEPS)]
echo "Local batch size: $LOCAL_BATCH_SIZE"

# ===================== 修改以下路径 =====================
PROJECT_ROOT=/u/yzou1/smolvlm_chat
MODEL_PATH=/projects/bffh/yzou1/models/FastVLM-1.5B
DATA_JSON=/work/hdd/bffz/yzou1/data/combined_trace_balanced.jsonl
VIDEO_FOLDER=/work/hdd/bffz/yzou1/data/
OUTP_DIR=/work/hdd/bffh/yzou1/models
# ========================================================

# 环境变量
export TRANSFORMERS_OFFLINE=0
export WANDB_PROJECT=fastvlm_trace
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=0
export NCCL_P2P_LEVEL=NVL
export CUDA_VISIBLE_DEVICES=0
export NCCL_IB_DISABLE=0
export DECORD_NUM_THREADS=4
export ATTN_IMPLEMENTATION=sdpa
export CUDA_LAUNCH_BLOCKING=1

RUN_NAME=fastvlm_trace_v1
DEEPSPEED_CONFIG=${PROJECT_ROOT}/configs/zero2.json

cd ${PROJECT_ROOT}

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    scripts/train.py \
    --deepspeed $DEEPSPEED_CONFIG \
    --model_name_or_path $MODEL_PATH \
    --model_type fastvlm \
    --paradigm trace \
    --data_path $DATA_JSON \
    --video_folder $VIDEO_FOLDER \
    --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/${RUN_NAME} \
    --torch_dtype bfloat16 \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --freeze_vision_tower True \
    --gradient_checkpointing True \
    --vision_pool_stride 2 \
    --max_frames 32 \
    --model_max_length 4096 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 5 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --dataloader_num_workers 1 \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --do_train True \
    --report_to wandb \
    --run_name $RUN_NAME \
    --seed 42