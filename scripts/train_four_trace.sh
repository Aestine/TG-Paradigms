#!/bin/bash

# =============================================================================
# SmolVLM-TRACE 训练脚本 - 2节点各1GPU (SLURM)
# =============================================================================

# Environment Variables
WORLD_SIZE=4
NPROC_PER_NODE=1
RANK=$SLURM_NODEID

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')
echo "Head node: $head_node  |  IP: $head_ip"

# Training Arguments
GLOBAL_BATCH_SIZE=64
GRADIENT_ACCUMULATION_STEPS=4
LOCAL_BATCH_SIZE=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$GRADIENT_ACCUMULATION_STEPS)]
echo "Local batch size: $LOCAL_BATCH_SIZE"

# ===================== 路径配置 =====================
PROJECT_ROOT=/u/yzou1/smolvlm_chat
MODEL_PATH=/projects/bffz/yzou1/models/SmolVLM2-2.2B-Instruct
DATA_JSON=/work/hdd/bffz/yzou1/data/combined_trace_balanced.jsonl
VIDEO_FOLDER=/work/hdd/bffz/yzou1/data/
OUTP_DIR=/work/hdd/bffh/yzou1/models
DEEPSPEED_CONFIG=${PROJECT_ROOT}/configs/zero2.json
# ====================================================

# Environment
export TRANSFORMERS_OFFLINE=0
export WANDB_PROJECT=smolvlm_trace
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=0
export DECORD_NUM_THREADS=4
export ATTN_IMPLEMENTATION=sdpa
# export CUDA_LAUNCH_BLOCKING=1  # 调试时开启
# export NCCL_DEBUG=INFO

RUN_NAME=smolvlm_trace_2node

module load cuda/12.6.1
module load nccl 2>/dev/null
source ~/.bashrc
conda activate smolvlm_env

cd ${PROJECT_ROOT}

srun --nodes=4 --ntasks=4 --ntasks-per-node=1 \
    torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint "${head_ip}:29500" \
    scripts/train.py \
    --deepspeed $DEEPSPEED_CONFIG \
    --model_name_or_path $MODEL_PATH \
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
    --max_frames 32 \
    --model_max_length 4096 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --dataloader_num_workers 2 \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --do_train True \
    --report_to wandb \
    --run_name $RUN_NAME \
    --seed 42