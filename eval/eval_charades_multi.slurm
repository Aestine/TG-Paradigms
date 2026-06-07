#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --output=/projects/bffh/yzou1/logs/eval_charades-molmo2-%J.out
#SBATCH --partition=ghx4
#SBATCH --account=bffz-dtai-gh
#SBATCH --job-name=eval_charades

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

export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16
export TORCH_NUM_THREADS=8
export DECORD_NUM_THREADS=4
export DECORD_EOF_RETRY_MAX=40960

source ~/.bashrc
conda activate trace_conda

echo "=========================================="
echo "Hostname: $(hostname)"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=========================================="

# ==============================================================================
# ★★★ 选择模型配置 ★★★
# 取消注释你要用的那一组，注释掉其他组
# ==============================================================================

# ---- SmolVLM2-2.2B (DisTime) ----
# MODEL_NAME_OR_PATH="/projects/bffz/yzou1/models/SmolVLM2-2.2B-Instruct"
# MODEL_TYPE="smolvlm"
# CHECKPOINT_DIR="/projects/bffh/yzou1/models/smolvlm_distime/smolvlm_distime_2.2b_checkpoint"
# EXTRA_ARGS=""
# RUN_TAG="smolvlm2.2b_distime"

# ---- SmolVLM2-0.5B (DisTime) ----
# MODEL_NAME_OR_PATH="/work/hdd/bffh/yzou1/models/SmolVLM2-500M-Video-Instruct"
# MODEL_TYPE="smolvlm"
# CHECKPOINT_DIR="/projects/bffh/yzou1/models/smolvlm_distime/smolvlm_distime_1928980_2nodes"
# EXTRA_ARGS="--image_size 512 --image_seq_len 64"
# RUN_TAG="smolvlm0.5b_distime"

# ---- FastVLM-1.5B (DisTime) ----
# MODEL_NAME_OR_PATH="/projects/bffh/yzou1/models/FastVLM-1.5B"
# MODEL_TYPE="fastvlm"
# CHECKPOINT_DIR="/work/hdd/bffh/yzou1/models/fastvlm_distime/fastvlm_distime_1922838_2nodes"
# EXTRA_ARGS="--vision_pool_stride 2"
# RUN_TAG="fastvlm1.5b_distime"

# ---- Molmo2-4B (DisTime) ----
# MODEL_NAME_OR_PATH="/work/hdd/bffh/yzou1/models/Molmo2-4B"
# MODEL_TYPE="molmo2"
# CHECKPOINT_DIR="/work/hdd/bffh/yzou1/models/molmo2_distime/molmo2_distime_1931241_2nodes"
# EXTRA_ARGS=""
# RUN_TAG="molmo2_4b_distime"

# ---- Molmo2-8B (DisTime) ----
MODEL_NAME_OR_PATH="/work/hdd/bffh/yzou1/models/Molmo2-8B"
MODEL_TYPE="molmo2"
CHECKPOINT_DIR="/work/hdd/bffh/yzou1/models/molmo2_8b_distime/molmo2_distime_1932583_2nodes"
EXTRA_ARGS=""
RUN_TAG="molmo2_8b_distime"

# ==============================================================================
# 通用参数（所有模型共享）
# ==============================================================================
DATA_FILE="/work/hdd/bffh/yzou1/data/Charades/charades_sta_test.json"
VIDEO_ROOT="/work/hdd/bffh/yzou1/data/Charades/videos"
OUTPUT_DIR="/u/yzou1/smolvlm_chat/eval/results/charades_${RUN_TAG}_$(date +%Y%m%d_%H%M%S)"
QUERY_TEMPLATE="Localize the visual content described by the given textual query '{query}' in the video, and output the start and end timestamps in seconds."
NUM_FRAMES=32
NUM_GPUS=2
WORKERS_PER_GPU=2
cd smolvlm_chat

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
echo "  Extra Args: ${EXTRA_ARGS}"
echo "  Run Tag: ${RUN_TAG}"
echo "  Workers/GPU: ${WORKERS_PER_GPU}"
echo "  Total Workers: $((NUM_GPUS * WORKERS_PER_GPU))"
echo "=========================================="

echo "Starting Charades-STA evaluation [${RUN_TAG}] at $(date)..."

python /u/yzou1/smolvlm_chat/eval/evaluate_charades.py \
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
    --query_template "${QUERY_TEMPLATE}" \
    ${EXTRA_ARGS}

EVAL_EXIT_CODE=$?

echo "=========================================="
echo "Evaluation finished at $(date)"
echo "Exit code: $EVAL_EXIT_CODE"
echo "=========================================="

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "Charades-STA [${RUN_TAG}] completed successfully!"
    echo "Results: ${OUTPUT_DIR}/"
    ls -la ${OUTPUT_DIR}/
else
    echo "ERROR: Evaluation failed with exit code $EVAL_EXIT_CODE"
fi

echo "Final GPU status:"
nvidia-smi

exit $EVAL_EXIT_CODE