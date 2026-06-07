#!/bin/bash
#SBATCH --time=0-2:00:00
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --output=/projects/bffh/yzou1/logs/efficiency_bench-%J.out
#SBATCH --partition=ghx4
#SBATCH --account=bffz-dtai-gh
#SBATCH --job-name=efficiency_bench

# =============================================================================
# Efficiency benchmark: SmolVLM-2.2B (3 paradigms) + SmolVLM-0.5B + FastVLM-1.5B
# 用法: sbatch run_efficiency_benchmark.sh
# =============================================================================

date
echo "=========================================="
echo "Efficiency Benchmark"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

module load cuda/12.6.1
source ~/.bashrc
conda activate smolvlm_env

PROJECT_ROOT=/u/yzou1/smolvlm_chat
RESULT_DIR=/projects/bffh/yzou1/eval_results/efficiency_benchmark
BENCH_SCRIPT=${PROJECT_ROOT}/eval/benchmarks/benchmark_efficiency.py
DATA_FILE="/work/hdd/bffh/yzou1/data/Charades/charades_sta_test.json"
VIDEO_ROOT="/work/hdd/bffh/yzou1/data/Charades/videos"

export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH
export ATTN_IMPLEMENTATION=sdpa
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DECORD_NUM_THREADS=4

mkdir -p ${RESULT_DIR}

nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=========================================="
cd smolvlm_chat
# =============================================
# 共用参数
# =============================================
COMMON_ARGS="--anno_file ${DATA_FILE} --video_root ${VIDEO_ROOT} --output_dir ${RESULT_DIR} --num_videos 5"
# =============================================
# 3) FastVLM-1.5B: distime only
#    ⚠️  请确认以下两个路径是否正确
# =============================================
echo ""
echo ">>> [3/3] FastVLM-1.5B — distime"
echo "=========================================="

FASTVLM_BASE="/projects/bffh/yzou1/models/FastVLM-1.5B"            # ← 确认此路径
FASTVLM_DISTIME="/work/hdd/bffh/yzou1/models/fastvlm_distime/fastvlm_distime_1922838_2nodes"      # ← 确认此路径

python ${BENCH_SCRIPT} \
    ${COMMON_ARGS} \
    --output_name efficiency_fastvlm_1.5b \
    --paradigm text distime \
    --model_type fastvlm \
    --model_name_or_path ${FASTVLM_BASE} \
    --distime_checkpoint ${FASTVLM_DISTIME} \
    --vision_pool_stride 2 \
    --image_size 1024
# =============================================
# 1) SmolVLM-2.2B: text + distime + trace
# =============================================
echo ""
echo ">>> [1/3] SmolVLM-2.2B — text + distime + trace"
echo "=========================================="

python ${BENCH_SCRIPT} \
    ${COMMON_ARGS} \
    --output_name efficiency_smolvlm_2.2b

# =============================================
# 2) SmolVLM-0.5B: text + distime
# =============================================
echo ""
echo ">>> [2/3] SmolVLM-0.5B — text + distime"
echo "=========================================="

python ${BENCH_SCRIPT} \
    ${COMMON_ARGS} \
    --output_name efficiency_smolvlm_0.5b \
    --paradigm text distime \
    --model_type smolvlm \
    --model_name_or_path /work/hdd/bffh/yzou1/models/SmolVLM2-500M-Video-Instruct \
    --distime_checkpoint /projects/bffh/yzou1/models/smolvlm_distime/smolvlm_distime_1928980_2nodes \
    --image_size 512 \
    --image_seq_len 64



# =============================================
echo ""
echo "=========================================="
echo "Finished at $(date)"
echo "Results:"
echo "  SmolVLM-2.2B: ${RESULT_DIR}/efficiency_smolvlm_2.2b.json"
echo "  SmolVLM-0.5B: ${RESULT_DIR}/efficiency_smolvlm_0.5b.json"
echo "  FastVLM-1.5B: ${RESULT_DIR}/efficiency_fastvlm_1.5b.json"
echo "=========================================="
