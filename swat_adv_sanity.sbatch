#!/bin/bash
#SBATCH --job-name=adv_sanity
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dhyana.schirra@studio.unibo.it
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100_40GB:1
#SBATCH --time=00:10:00
#SBATCH --chdir=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
#SBATCH --output=slurm.adv_sanity.%j.out
#SBATCH --error=slurm.adv_sanity.%j.err

set -euo pipefail

REPO=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
IMAGE="$REPO/sip.sif"
MODEL="$REPO/outputs/fixed_cnn_55075_0/models/results/CNN-SWAT-l3-hist100-kern3-units128.h5"

echo "HOST=$(hostname)"
echo "START=$(date -Is)"

mkdir -p "$REPO/models"

echo "Ensuring SWaT scaler exists..."

if [[ ! -f "$REPO/models/SWAT_scaler.pkl" ]]; then
    /usr/bin/apptainer exec \
        --bind "$REPO:$REPO" \
        --pwd "$REPO" \
        "$IMAGE" \
        python3 -c "import data_loader; data_loader.load_train_data('SWAT', verbose=True)"
fi

ls -lh "$REPO/models/SWAT_scaler.pkl"

GPU_IDS="${CUDA_VISIBLE_DEVICES:-}"

exec /usr/bin/apptainer exec \
    --nv \
    --bind "$REPO:$REPO" \
    --pwd "$REPO" \
    --env "PYTHONPATH=$REPO" \
    --env "CUDA_VISIBLE_DEVICES=$GPU_IDS" \
    "$IMAGE" \
    python3 -u main_adversarial.py \
        --dataset SWAT \
        --model-type CNN \
        --model-path "$MODEL" \
        --history 100 \
        --target-offset 0 \
        --instance-threshold 0.9983735862433992 \
        --detection-window 1 \
        --instance-score mean_mse \
        --sanity-check
