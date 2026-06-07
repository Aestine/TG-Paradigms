#!/bin/bash
# =============================================================================
# Text Numeral paradigm — single-GPU training (VTimeLLM-style)
# Timestamps are emitted as plain text ("from X to Y seconds"); LM loss only.
# Mirrors train_single.sh; the key differences are: --paradigm text and
# --model_type. Can reuse the SAME data file as DisTime (the dataset reads
# `times`/captions and reformats them into the numeral target).
# =============================================================================

WORLD_SIZE=1
NPROC_PER_NODE=1
MASTER_ADDR="127.0.0.1"
MASTER_PORT=16666
RANK=0

GLOBAL_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=2
LOCAL_BATCH_SIZE=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$GRADIENT_ACCUMULATION_STEPS)]

# ===================== EDIT THESE PATHS =====================
PROJECT_ROOT=/path/to/repo
MODEL_PATH=/path/to/SmolVLM2-2.2B-Instruct
DATA_JSON=/path/to/combined_distime_balanced.jsonl   # reused as-is for Text
VIDEO_FOLDER=/path/to/videos/
OUTP_DIR=/path/to/outputs
MODEL_TYPE=smolvlm                                   # smolvlm | fastvlm | molmo2
# ===========================================================

export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH
export WANDB_PROJECT=vtg_text
export CUDA_VISIBLE_DEVICES=0
export DECORD_NUM_THREADS=4
export ATTN_IMPLEMENTATION=sdpa

RUN_NAME=${MODEL_TYPE}_text_v1
DEEPSPEED_CONFIG=${PROJECT_ROOT}/configs/zero2.json

cd ${PROJECT_ROOT}

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    scripts/train.py \
    --deepspeed $DEEPSPEED_CONFIG \
    --model_type $MODEL_TYPE \
    --paradigm text \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_JSON \
    --video_folder $VIDEO_FOLDER \
    --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/${RUN_NAME} \
    --torch_dtype bfloat16 \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --freeze_vision_tower True \
    --gradient_checkpointing True \
    --max_frames 32 \
    --model_max_length 4096 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --save_strategy "steps" \
    --save_steps 1000 \
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
