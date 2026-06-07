#!/bin/bash
#SBATCH --time=0-03:00:00
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=0
#SBATCH --output=/projects/bffz/yzou1/logs/download-%J.out
#SBATCH --partition=ghx4
#SBATCH --account=bffh-dtai-gh
#SBATCH --job-name=download_internvid

echo "$(date)"
echo "========== Download internvid-tg =========="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

source ~/.bashrc
conda activate trace_conda

DOWNLOAD_DIR="/tmp/internvid-tg-download"
DEST_DIR="/work/hdd/bffz/yzou1/data"

rm -rf ${DOWNLOAD_DIR}
mkdir -p ${DOWNLOAD_DIR} ${DEST_DIR}

echo "[$(date)] Starting download..."
huggingface-cli download yingsen/internvid-tg \
    --repo-type dataset \
    --local-dir ${DOWNLOAD_DIR} \
    --local-dir-use-symlinks False

if [ $? -ne 0 ]; then
    echo "ERROR: Download failed!"
    exit 1
fi

PART_COUNT=$(ls ${DOWNLOAD_DIR}/internvid-tg.tar.gz.part* 2>/dev/null | wc -l)
TOTAL_SIZE=$(du -sh ${DOWNLOAD_DIR} | cut -f1)
echo "[$(date)] Download complete. Parts: ${PART_COUNT}, Total size: ${TOTAL_SIZE}"

if [ "$PART_COUNT" -lt 1 ]; then
    echo "ERROR: No part files found!"
    exit 1
fi

echo "[$(date)] Starting extraction with pigz (16 cores)..."
cat ${DOWNLOAD_DIR}/internvid-tg.tar.gz.part* | pigz -dc -p 16 | tar -xf - -C ${DEST_DIR}

if [ $? -ne 0 ]; then
    echo "WARNING: pigz failed, trying plain tar..."
    cat ${DOWNLOAD_DIR}/internvid-tg.tar.gz.part* | tar -xzf - -C ${DEST_DIR}
    if [ $? -ne 0 ]; then
        echo "ERROR: Extraction failed!"
        exit 1
    fi
fi

echo "[$(date)] Copying annotations..."
cp ${DOWNLOAD_DIR}/*.jsonl ${DEST_DIR}/ 2>/dev/null

echo "[$(date)] Cleaning up..."
rm -rf ${DOWNLOAD_DIR}

echo "[$(date)] Done!"
echo "Files in ${DEST_DIR}:"
ls -lh ${DEST_DIR}/internvid* 2>/dev/null | head -20