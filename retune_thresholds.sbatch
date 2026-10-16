#!/bin/bash
#SBATCH --job-name=retune
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-17%3
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=dhyana.schirra@studio.unibo.it
#SBATCH --chdir=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
#SBATCH --output=slurm.retune.%A_%a.out
#SBATCH --error=slurm.retune.%A_%a.err

# Threshold/window choice of the 18 chronological SWAT-CLEAN detectors redone on
# the original repository's wider grid (no retraining, model files untouched).
# Task id -> run: seed = id / 6 (7, 21, 42), model = id % 6 (ae cnn dnn lstm gru lin).
# Task 0 (AE seed 7) is the check: it must reproduce the stored theta, window and F1
# (exit code 3 otherwise). Results: outputs/retune_v1/<run>.json;
# table: python3 retune_thresholds.py --summary

set -euo pipefail
REPO=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
IMAGE="$REPO/sip.sif"
RUNS=()
for SEED in 7 21 42; do
    for M in ae cnn dnn lstm gru lin; do
        RUNS+=("swatchrono_${M}_s${SEED}")
    done
done
RUN="${RUNS[$SLURM_ARRAY_TASK_ID]}"

GPU_IDS="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "$GPU_IDS" || "$GPU_IDS" == "-1" || "$GPU_IDS" == "NoDevFiles" ]]; then
    echo "ERROR: Slurm did not assign a GPU"
    exit 1
fi
mkdir -p "$REPO/outputs/retune_v1"
git -C "$REPO" rev-parse HEAD > "$REPO/outputs/retune_v1/${RUN}.git_commit.txt"
git -C "$REPO" diff > "$REPO/outputs/retune_v1/${RUN}.code_changes.patch"
echo "RUN=$RUN HOST=$(hostname) GPU=$GPU_IDS START=$(date -Is)"

/usr/bin/apptainer exec --nv \
    --bind "$REPO:$REPO" --pwd "$REPO" \
    --env "PYTHONPATH=$REPO" \
    --env "CUDA_VISIBLE_DEVICES=$GPU_IDS" \
    --env "TF_FORCE_GPU_ALLOW_GROWTH=true" \
    --env "TF_XLA_FLAGS=--tf_xla_auto_jit=0" \
    --env "PYTHONUNBUFFERED=1" \
    "$IMAGE" \
    python3 -u retune_thresholds.py "models/$RUN" 2>&1 | tee "$REPO/outputs/retune_v1/${RUN}.log"
echo "FINISHED $(date -Is)"
