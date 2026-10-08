#!/bin/bash
#SBATCH --job-name=per_attack
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dhyana.schirra@studio.unibo.it
#SBATCH --chdir=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
#SBATCH --output=slurm.per_attack.%j.out
#SBATCH --error=slurm.per_attack.%j.err

# Per-attack detection of the old SWAT-CLEAN detectors (shuffled split, seed 42)
# on the whole test file. Results: outputs/per_attack_old/<model>.csv and the
# printed tables in outputs/per_attack_old/summary.txt.

set -euo pipefail
REPO=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
OUT="$REPO/outputs/per_attack_old"
mkdir -p "$OUT"

/usr/bin/apptainer exec --nv \
    --bind "$REPO:$REPO" --pwd "$REPO" \
    --env "PYTHONPATH=$REPO" \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" \
    --env "TF_FORCE_GPU_ALLOW_GROWTH=true" \
    --env "PYTHONUNBUFFERED=1" \
    "$REPO/sip.sif" \
    python3 -u per_attack_detection.py --out-dir "$OUT" \
        models/swatclean_ae models/swatclean_cnn models/swatclean_dnn \
        models/swatclean_gru models/swatclean_lstm \
    2>&1 | tee "$OUT/summary.txt"
