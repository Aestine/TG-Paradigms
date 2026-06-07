#!/bin/bash
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --output=/projects/bffz/yzou1/logs/eval_youcook2_distime-%J.out
#SBATCH --partition=ghx4
#SBATCH --account=bffz-dtai-gh
#SBATCH --job-name=eval_yc2_distime

date
echo "=========================================="
echo "YouCook2 DVC Evaluation - DisTime"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

module load cuda/12.6.1
module load gcc
source ~/.bashrc
conda activate trace_conda

export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16
export TORCH_NUM_THREADS=8
export DECORD_NUM_THREADS=4
export DECORD_EOF_RETRY_MAX=40960

# ==============================================================================
# ★★★ 选择模型配置 ★★★
# 取消注释你要用的那一组，注释掉其他组
# ==============================================================================

# ---- SmolVLM2-2.2B (DisTime) ----
# MODEL_NAME_OR_PATH="/projects/bffz/yzou1/models/SmolVLM2-2.2B-Instruct"
# MODEL_TYPE="smolvlm"
# CHECKPOINT_DIR="/projects/bffh/yzou1/models/smolvlm_distime/smolvlm_distime_1888038_2nodes"
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
ANNO_FILE="/work/hdd/bffz/yzou1/data/TRACE/dense_video_caption/Youcook2/val.caption_coco_format.json"
VIDEO_ROOT="/work/hdd/bffh/yzou1/data/youcook2/videos"
NUM_FRAMES=32
NUM_GPUS=2
WORKERS_PER_GPU=2
OUTPUT_DIR="/u/yzou1/smolvlm_chat/eval/results/youcook2_${RUN_TAG}_$(date +%Y%m%d_%H%M%S)"
cd smolvlm_chat
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "评估配置:"
echo "  Model: ${MODEL_NAME_OR_PATH}"
echo "  Model Type: ${MODEL_TYPE}"
echo "  Checkpoint: ${CHECKPOINT_DIR}"
echo "  Anno: ${ANNO_FILE}"
echo "  Videos: ${VIDEO_ROOT}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Num Frames: ${NUM_FRAMES}"
echo "  Extra Args: ${EXTRA_ARGS}"
echo "  Run Tag: ${RUN_TAG}"
echo "  Workers/GPU: ${WORKERS_PER_GPU}"
echo "  Total Workers: $((NUM_GPUS * WORKERS_PER_GPU))"
echo "=========================================="

echo "Starting YouCook2 evaluation [${RUN_TAG}] at $(date)..."

python /u/yzou1/smolvlm_chat/eval/evaluate_youcook2.py \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --anno_file ${ANNO_FILE} \
    --video_root ${VIDEO_ROOT} \
    --num_frames ${NUM_FRAMES} \
    --max_new_tokens 512 \
    --num_gpus ${NUM_GPUS} \
    --workers_per_gpu ${WORKERS_PER_GPU} \
    --out_dir ${OUTPUT_DIR} \
    --query_template "Scrutinize the video and determine multiple occurrences, providing their initial and final timestamps as well as a summary of each action." \
    ${EXTRA_ARGS}

EVAL_EXIT_CODE=$?

echo "=========================================="
echo "Evaluation finished at $(date)"
echo "Exit code: $EVAL_EXIT_CODE"
echo "=========================================="

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "YouCook2 [${RUN_TAG}] completed successfully!"
    echo "Results: ${OUTPUT_DIR}/"
    ls -la ${OUTPUT_DIR}/
else
    echo "ERROR: Evaluation failed with exit code $EVAL_EXIT_CODE"
fi

echo "Final GPU status:"
nvidia-smi

exit $EVAL_EXIT_CODE
