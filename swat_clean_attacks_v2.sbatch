#!/bin/bash
#SBATCH --job-name=swat_clean_adv
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-2%2
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dhyana.schirra@studio.unibo.it
#SBATCH --chdir=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
#SBATCH --output=slurm.swat_clean_adv.%A_%a.out
#SBATCH --error=slurm.swat_clean_adv.%A_%a.err

# Evasion benchmark (v2) on the SWAT-CLEAN detectors trained by
# swat_clean_all.sbatch: sanity check first, then every attack, including
# pgd_hinge, with the same targets, budget and detector. Targets are a block from
# the middle of every attack (8192 in total), not only the first attacks. Theta, window and history are read from the
# model's own .json, so they always match the trained detector.

set -euo pipefail

REPO=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
IMAGE="$REPO/sip.sif"

MODELS=(LSTM GRU CNN)
NAMES=(
    LSTM-SWAT-CLEAN-l1-hist50-units128
    GRU-SWAT-CLEAN-l2-hist100-units256
    CNN-SWAT-CLEAN-l3-hist100-kern3-units128
)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"
MODEL_DIR="$REPO/models/swatclean_${MODEL,,}"
OUT_DIR="$REPO/outputs/adv/SWATCLEAN_v2_${MODEL}_evasion"

GPU_IDS="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "$GPU_IDS" || "$GPU_IDS" == "-1" || "$GPU_IDS" == "NoDevFiles" ]]; then
    echo "ERROR: Slurm did not assign a GPU"
    exit 1
fi

test -r "$MODEL_DIR/$NAME.h5"
test -r "$MODEL_DIR/$NAME.json"
test -r "$REPO/models/SWAT-CLEAN_scaler.pkl"

read -r THETA WINDOW HISTORY < <(/usr/bin/apptainer exec --bind "$REPO:$REPO" "$IMAGE" python3 - "$MODEL_DIR/$NAME.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
print(repr(float(p["best_theta"])), int(p["best_window"]), int(p["history"]))
PY
)

echo "MODEL=$MODEL NAME=$NAME THETA=$THETA WINDOW=$WINDOW HISTORY=$HISTORY"
echo "HOST=$(hostname) GPU=$GPU_IDS START=$(date -Is)"
mkdir -p "$OUT_DIR"
git -C "$REPO" rev-parse HEAD > "$OUT_DIR/git_commit.txt"
git -C "$REPO" diff > "$OUT_DIR/code_changes.patch"

RUN=(
    /usr/bin/apptainer exec
    --nv
    --bind "$REPO:$REPO"
    --pwd "$REPO"
    --env "PYTHONPATH=$REPO"
    --env "CUDA_VISIBLE_DEVICES=$GPU_IDS"
    --env "TF_FORCE_GPU_ALLOW_GROWTH=true"
    --env "TF_XLA_FLAGS=--tf_xla_auto_jit=0"
    --env "MPLBACKEND=Agg"
    --env "PYTHONUNBUFFERED=1"
    "$IMAGE"
    python3 -u main_adversarial.py
    --dataset SWAT-CLEAN
    --model-type "$MODEL"
    --model-path "$MODEL_DIR/$NAME.h5"
    --history "$HISTORY"
    --target-offset 0
    --instance-threshold "$THETA"
    --detection-window "$WINDOW"
    --instance-score mean_mse
)

echo "===== SANITY CHECK ====="
"${RUN[@]}" --sanity-check 2>&1 | tee "$OUT_DIR/sanity.log"

echo "===== ATTACKS ====="
"${RUN[@]}" \
    --attack all --goal evasion \
    --selection attack --target-sampling segments --max-targets 8192 --seed 42 \
    --score mean_mse --threshold-percentile 0.995 \
    --epsilon 0.10 --alpha 0.02 --iterations 30 --random-start --top-k 5 \
    --hinge-margin 0.9 \
    --cw-confidence 0.0 --kl-reference auto \
    --corrshift-rounds 3 --corrshift-step 0.02 \
    --clip-train-range \
    --save-series none \
    --output-dir "$OUT_DIR" \
    2>&1 | tee "$OUT_DIR/attacks.log"

echo "$MODEL FINISHED $(date -Is)"
