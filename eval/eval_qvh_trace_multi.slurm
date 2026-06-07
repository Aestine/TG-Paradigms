#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1                    # 单节点多GPU
#SBATCH --mem=200G
#SBATCH --ntasks-per-node=1          # 单进程（脚本内部用mp.Process管理多GPU）
#SBATCH --cpus-per-task=32           # 总共32核给所有GPU worker
#SBATCH --gpus-per-node=2            # 2个GPU
#SBATCH --output=/projects/bffh/yzou1/logs/eval_qvh_tracesfast-%J.out
#SBATCH --partition=ghx4
#SBATCH --account=bffh-dtai-gh
#SBATCH --job-name=eval_qvh_tracesfast

date
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=$SLURM_GPUS_PER_NODE
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
MODEL_NAME_OR_PATH="/projects/bffz/yzou1/models/SmolVLM2-2.2B-Instruct"
# MODEL_NAME_OR_PATH="/projects/bffh/yzou1/models/FastVLM-1.5B"
MODEL_TYPE="smolvlm"                 # smolvlm 或 fastvlm
CHECKPOINT_DIR="/work/hdd/bffh/yzou1/models/smolvlm_trace/smolvlm_trace_1909494_2nodes"
# CHECKPOINT_DIR="/work/hdd/bffh/yzou1/models/fastvlm_trace/fastvlm_trace_1909495_2nodes"
DATA_FILE="/projects/bffz/yzou1/data/TRACE/eval/vhd/qvhighlights/val.caption_coco_format.json"
VIDEO_ROOT="/work/hdd/bffh/yzou1/data/qvhighlights/videos"
OUTPUT_DIR="/u/yzou1/smolvlm_chat/eval/results/qvh_tracesfast_$(date +%Y%m%d_%H%M%S)"
NUM_FRAMES=32
NUM_GPUS=1
WORKERS_PER_GPU=1
EVAL_TASKS="mr,hl"

# MR 和 HL 用不同的 query template（TRACE 训练时区分了两个任务）
QUERY_TEMPLATE_MR="Localize the visual content described by the given textual query '{query}' in the video, and output the start and end timestamps in seconds."
QUERY_TEMPLATE_HL="Please find the highlight contents in the video described by a sentence query, determining the highlight timestamps and its saliency score on a scale from 1 to 5. Now I will give you the sentence query: '{query}'. Please return the query-based highlight timestamps and salient scores."

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
echo "  Eval Tasks: ${EVAL_TASKS}"
echo "  MR Template: ${QUERY_TEMPLATE_MR:0:60}..."
echo "  HL Template: ${QUERY_TEMPLATE_HL:0:60}..."
echo "  Workers/GPU: ${WORKERS_PER_GPU}"
echo "  Total Workers: $((NUM_GPUS * WORKERS_PER_GPU))"
echo "=========================================="

# ==================== 多GPU并行评估 ====================
echo "Starting QVHighlights TRACE evaluation at $(date)..."

python /u/yzou1/smolvlm_chat/eval/evaluate_qvh_trace.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --model_type ${MODEL_TYPE} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --data_file ${DATA_FILE} \
    --video_root ${VIDEO_ROOT} \
    --out_dir ${OUTPUT_DIR} \
    --num_frames ${NUM_FRAMES} \
    --num_gpus ${NUM_GPUS} \
    --workers_per_gpu ${WORKERS_PER_GPU} \
    --max_new_tokens 512 \
    --eval_tasks "${EVAL_TASKS}" \
    --query_template_mr "${QUERY_TEMPLATE_MR}" \
    --query_template_hl "${QUERY_TEMPLATE_HL}" \
    --max_samples 10
    --vision_pool_stride 2 \
EVAL_EXIT_CODE=$?

echo "=========================================="
echo "Evaluation finished at $(date)"
echo "Exit code: $EVAL_EXIT_CODE"
echo "=========================================="

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "QVHighlights TRACE evaluation completed successfully!"
    echo "Results: ${OUTPUT_DIR}/"
    ls -la ${OUTPUT_DIR}/
    echo "=========================================="
else
    echo "=========================================="
    echo "ERROR: Evaluation failed with exit code $EVAL_EXIT_CODE"
    echo "Check logs for details:"
    echo "  /projects/bffh/yzou1/logs/eval_qvh_trace_${SLURM_JOB_ID}.out"
    echo "=========================================="
fi

# 显示最终GPU状态
echo "Final GPU status:"
nvidia-smi

exit $EVAL_EXIT_CODE
