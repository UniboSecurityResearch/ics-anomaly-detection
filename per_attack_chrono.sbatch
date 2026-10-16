#!/bin/bash
#SBATCH --job-name=per_attack
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dhyana.schirra@studio.unibo.it
#SBATCH --chdir=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
#SBATCH --output=slurm.per_attack_chrono.%j.out
#SBATCH --error=slurm.per_attack_chrono.%j.err

# Per-attack detection of the 18 chronological SWAT-CLEAN detectors (clean data):
# whole test file and final split only (rows from 134975: the 18 attacks not used
# to choose theta and the window). Rule of the original repository (an attack is
# detected if one timestep is flagged) and, as a second column, >= 10% flagged.
# Results: outputs/per_attack_chrono/<run>.csv, summary_final.csv, summary.txt.

set -euo pipefail
REPO=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
OUT="$REPO/outputs/per_attack_chrono"
mkdir -p "$OUT"
git -C "$REPO" rev-parse HEAD > "$OUT/git_commit.txt"
git -C "$REPO" diff > "$OUT/code_changes.patch"
DIRS=()
for SEED in 7 21 42; do
    for M in ae cnn dnn lstm gru lin; do
        DIRS+=("models/swatchrono_${M}_s${SEED}")
    done
done

/usr/bin/apptainer exec --nv \
    --bind "$REPO:$REPO" --pwd "$REPO" \
    --env "PYTHONPATH=$REPO" \
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" \
    --env "TF_FORCE_GPU_ALLOW_GROWTH=true" \
    --env "PYTHONUNBUFFERED=1" \
    "$REPO/sip.sif" \
    python3 -u per_attack_detection.py --out-dir "$OUT" --final-start 134975 "${DIRS[@]}" \
    2>&1 | tee "$OUT/summary.txt"
