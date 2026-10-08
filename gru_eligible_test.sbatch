#!/bin/bash
#SBATCH --job-name=gru_eligible
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dhyana.schirra@studio.unibo.it
#SBATCH --chdir=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
#SBATCH --output=slurm.gru_eligible.%j.out
#SBATCH --error=slurm.gru_eligible.%j.err

# GRU BATADAL eligible-set test (supervisor's request, 2026-10-08).
# Same model, threshold, window, budget, step and iterations as the September
# run (outputs/adv/gru_pgd_validation_20260905T105344Z); the ONLY change is that
# the PGD loss is computed on the clean-detected targets E (expected 1233 of
# 1968), frozen once before the attack. Per iteration: evasions and ASR over E,
# detection rate over all 1968. Checkpoints: best_loss and best_asr.

set -euo pipefail
REPO=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
OUT="$REPO/outputs/adv/gru_eligible_test_$(date +%Y%m%d)"
mkdir -p "$OUT"
git -C "$REPO" rev-parse HEAD > "$OUT/git_commit.txt"
git -C "$REPO" diff > "$OUT/code_changes.patch"

RUN=(
    /usr/bin/apptainer exec --nv
    --bind "$REPO:$REPO" --pwd "$REPO"
    --env "PYTHONPATH=$REPO"
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
    --env "TF_FORCE_GPU_ALLOW_GROWTH=true"
    --env "PYTHONUNBUFFERED=1"
    "$REPO/sip.sif"
    python3 -u main_adversarial.py
    --dataset BATADAL
    --model-type GRU
    --model-path models/gru/GRU-BATADAL-l2-hist50-units64.h5
    --history 50
    --target-offset 1
    --instance-threshold 0.54828903
    --detection-window 10
    --instance-score mean_mse
)

echo "===== SANITY CHECK (expected clean attack detection 0.6265) ====="
"${RUN[@]}" --sanity-check 2>&1 | tee "$OUT/sanity.log"

echo "===== PGD, LOSS ON ELIGIBLE SET E, 3 RESTARTS ====="
"${RUN[@]}" \
    --attack pgd_mse --goal evasion \
    --selection attack --max-targets 2048 \
    --scope both --score mean_mse \
    --epsilon 0.10 --alpha 0.01 --iterations 100 \
    --random-start --restarts 3 --seed 42 \
    --eligible-protocol --return-point best_asr \
    --save-series full \
    --output-dir "$OUT" \
    2>&1 | tee "$OUT/attack.log"

echo "FINISHED $(date -Is)"
