#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --nodes=2
#SBATCH --mem=400G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --output=/projects/bffh/yzou1/logs/distime-%J.out
#SBATCH --partition=ghx4
#SBATCH --account=bffh-dtai-gh
#SBATCH --job-name=smolvlm_distime

date
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=$(echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l)
echo "You were assigned $NUM_GPUS gpu(s)"
echo "Number of nodes: $SLURM_NNODES"
echo "=========================================="

module load cuda/12.6.1
module load nccl 2>/dev/null
module list

# ===================== 路径配置 =====================
PROJECT_ROOT=/u/yzou1/smolvlm_chat
MODEL_PATH=/projects/bffz/yzou1/models/SmolVLM2-2.2B-Instruct
DATA_JSON=/work/hdd/bffz/yzou1/data/combined_distime_balanced.jsonl
VIDEO_FOLDER=/work/hdd/bffz/yzou1/data/
OUTP_DIR=/projects/bffh/yzou1/models
DEEPSPEED_CONFIG=${PROJECT_ROOT}/configs/zero2.json
# ====================================================

# 环境变量
export TRANSFORMERS_OFFLINE=0
export WANDB_PROJECT=smolvlm_distime
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH

# NCCL 设置 - 多节点关键配置
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=hsn
export NCCL_P2P_LEVEL=NVL
export NCCL_CUMEM_ENABLE=0
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=300

# PyTorch 和性能优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16
export TORCH_NUM_THREADS=8
export DECORD_NUM_THREADS=4
export DECORD_EOF_RETRY_MAX=40960
export ATTN_IMPLEMENTATION=sdpa

export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_PROCID}
mkdir -p $TRITON_CACHE_DIR

source ~/.bashrc
conda activate smolvlm_env

# 打印环境信息
echo "=========================================="
echo "Hostname: $(hostname)"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=========================================="

# --- 计算 torchrun 的 rendezvous 端点 ---
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')
echo "Head node: $head_node  |  IP: $head_ip"

# 多节点训练配置
WORLD_SIZE=$SLURM_NNODES
NPROC_PER_NODE=4
RANK=$SLURM_NODEID

# 批次大小配置
# 16 GPUs total: global_batch=128 / (4nodes * 4gpus * 2accum) = 4 per GPU
GLOBAL_BATCH_SIZE=160
GRADIENT_ACCUMULATION_STEPS=4
LOCAL_BATCH_SIZE=$((GLOBAL_BATCH_SIZE / (WORLD_SIZE * NPROC_PER_NODE * GRADIENT_ACCUMULATION_STEPS)))

echo "=========================================="
echo "Training Configuration:"
echo "  Number of Nodes: $WORLD_SIZE"
echo "  GPUs per Node: $NPROC_PER_NODE"
echo "  Total GPUs: $((WORLD_SIZE * NPROC_PER_NODE))"
echo "  Node Rank: $RANK"
echo "  Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "  Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo "  Local Batch Size per GPU: $LOCAL_BATCH_SIZE"
echo "  Head Node: $head_node ($head_ip)"
echo "=========================================="

# 创建输出目录
RUN_NAME=smolvlm_distime_${SLURM_JOB_ID}_${WORLD_SIZE}nodes
mkdir -p ${OUTP_DIR}/${WANDB_PROJECT}
mkdir -p /projects/bffh/yzou1/logs

# GPU显存监控 (后台)
(while true; do
    echo "=== $(date) === Node: $SLURMD_NODENAME (Rank: $RANK)" >> /projects/bffh/yzou1/logs/mem_distime_node${RANK}_${SLURM_JOB_ID}.log
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv >> /projects/bffh/yzou1/logs/mem_distime_node${RANK}_${SLURM_JOB_ID}.log
    sleep 60
done) &
MONITOR_PID=$!
echo "Memory monitoring started on $(hostname) (PID: $MONITOR_PID)"

echo "=========================================="
echo "Starting DisTime training at $(date)..."
echo "=========================================="

cd ${PROJECT_ROOT}

srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --gpus-per-task=4 \
    torchrun \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint "${head_ip}:29500" \
    scripts/train.py \
    --deepspeed $DEEPSPEED_CONFIG \
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
    --reg_max 32 \
    --num_time_layers 3 \
    --time_loss_weight 1.0 \
    --iou_loss_weight 1.0 \
    --max_frames 32 \
    --model_max_length 4096 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 3 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    --do_train True \
    --report_to wandb \
    --run_name $RUN_NAME \
    --seed 42

TRAIN_EXIT_CODE=$?

if [ ! -z "$MONITOR_PID" ]; then
    kill $MONITOR_PID 2>/dev/null
fi

echo "=========================================="
echo "Training finished at $(date)"
echo "Node $RANK exit code: $TRAIN_EXIT_CODE"
echo "=========================================="

if [ $RANK -eq 0 ]; then
    echo "Final GPU status (master node):"
    nvidia-smi

    if [ $TRAIN_EXIT_CODE -ne 0 ]; then
        echo "=========================================="
        echo "ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
        echo "Check logs: /projects/bffh/yzou1/logs/distime-${SLURM_JOB_ID}.out"
        echo "=========================================="
    fi

    if [ -d ~/wandb/offline-run-* ]; then
        echo "Syncing wandb logs..."
        wandb sync ~/wandb/offline-run-* 2>/dev/null || true
    fi
fi

exit $TRAIN_EXIT_CODE
