#!/bin/bash
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1                    # 单节点多GPU
#SBATCH --mem=200G
#SBATCH --ntasks-per-node=1          # 单进程（脚本内部用mp.Process管理多GPU）
#SBATCH --cpus-per-task=32           # 总共32核给所有GPU worker
#SBATCH --gpus-per-node=2            # 2个GPU
#SBATCH --output=/projects/bffh/yzou1/logs/eval_charades_tracefast-%J.out
#SBATCH --partition=ghx4
#SBATCH --account=bffh-dtai-gh
#SBATCH --job-name=eval_charades_tracefast

date
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=`echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l`
echo "You were assigned $NUM_GPUS gpu(s)"
echo "=========================================="

module load cuda/12.6.1
module load gcc
module list

# 环境变量
export TRANSFORMERS_OFFLINE=1

# 性能优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16
export TORCH_NUM_THREADS=8
export DECORD_NUM_THREADS=4
export DECORD_EOF_RETRY_MAX=40960

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

# ==================== 配置参数 ====================
# MODEL_NAME_OR_PATH="/projects/bffz/yzou1/models/SmolVLM2-2.2B-Instruct"
MODEL_NAME_OR_PATH="/projects/bffh/yzou1/models/FastVLM-1.5B"
# MODEL_TYPE="smolvlm"
MODEL_TYPE="fastvlm"
# CHECKPOINT_DIR="/work/hdd/bffh/yzou1/models/smolvlm_trace/smolvlm_trace_1909494_2nodes"
CHECKPOINT_DIR="/work/hdd/bffh/yzou1/models/fastvlm_trace/fastvlm_trace_1909495_2nodes"
DATA_FILE="/work/hdd/bffh/yzou1/data/Charades/charades_sta_test.json"
VIDEO_ROOT="/work/hdd/bffh/yzou1/data/Charades/videos"
OUTPUT_DIR="/u/yzou1/smolvlm_chat/eval/results/charades_trace_$(date +%Y%m%d_%H%M%S)"
QUERY_TEMPLATE="Localize the visual content described by the given textual query '{query}' in the video, and output the start and end timestamps in seconds."
NUM_FRAMES=32
NUM_GPUS=1
WORKERS_PER_GPU=1

mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "评估配置:"
echo "  Model: ${MODEL_NAME_OR_PATH}"
echo "  Model Type: ${MODEL_TYPE}"
echo "  Checkpoint: ${CHECKPOINT_DIR}"
echo "  Data: ${DATA_FILE}"
echo "  Videos: ${VIDEO_ROOT}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Num Frames: ${NUM_FRAMES}"
echo "  Query Template: ${QUERY_TEMPLATE}"
echo "  Workers/GPU: ${WORKERS_PER_GPU}"
echo "  Total Workers: $((NUM_GPUS * WORKERS_PER_GPU))"
echo "=========================================="

# ==================== 多GPU并行评估 ====================
echo "Starting Charades-STA TRACE evaluation at $(date)..."

python /u/yzou1/smolvlm_chat/eval/evaluate_charades_trace.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --model_type ${MODEL_TYPE} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --data_file ${DATA_FILE} \
    --video_root ${VIDEO_ROOT} \
    --out_dir ${OUTPUT_DIR} \
    --num_frames ${NUM_FRAMES} \
    --num_gpus ${NUM_GPUS} \
    --vision_pool_stride 2 \
    --workers_per_gpu ${WORKERS_PER_GPU} \
    --max_new_tokens 512 \
    --query_template "${QUERY_TEMPLATE}" \
    --max_samples 10

EVAL_EXIT_CODE=$?

echo "=========================================="
echo "Evaluation finished at $(date)"
echo "Exit code: $EVAL_EXIT_CODE"
echo "=========================================="

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "Charades-STA TRACE evaluation completed successfully!"
    echo "Results: ${OUTPUT_DIR}/"
    ls -la ${OUTPUT_DIR}/
    echo "=========================================="
else
    echo "=========================================="
    echo "ERROR: Evaluation failed with exit code $EVAL_EXIT_CODE"
    echo "Check logs for details:"
    echo "  /projects/bffz/yzou1/logs/eval_charades_trace_${SLURM_JOB_ID}.out"
    echo "=========================================="
fi

# 显示最终GPU状态
echo "Final GPU status:"
nvidia-smi

exit $EVAL_EXIT_CODE
