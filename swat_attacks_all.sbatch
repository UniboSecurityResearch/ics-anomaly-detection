#!/bin/bash
#SBATCH --job-name=swat_attacks
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-19%3
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=dhyana.schirra@studio.unibo.it
#SBATCH --chdir=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
#SBATCH --output=slurm.swat_attacks.%A_%a.out
#SBATCH --error=slurm.swat_attacks.%A_%a.err

# Evasion benchmark: every existing attack (+ pgd_hinge) on every differentiable
# SWAT-CLEAN detector, old and new:
#   tasks 0-4   old models  (models/swatclean_<model>, shuffled split, seed 42)
#   tasks 5-19  new models  (models/swatchrono_<model>_s<seed>, chronological split)
# Same targets (a block from the middle of every attack, 8192 in total), budget
# and attack settings for all. Theta, window and history come from each model's
# own .json. The linear model is not included: the attack code does not support
# its input shape yet.

set -euo pipefail

REPO=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
IMAGE="$REPO/sip.sif"

RUNS=(swatclean_ae swatclean_cnn swatclean_dnn swatclean_gru swatclean_lstm)
for SEED in 7 21 42; do
    for M in ae cnn dnn gru lstm; do
        RUNS+=("swatchrono_${M}_s${SEED}")
    done
done
RUN="${RUNS[$SLURM_ARRAY_TASK_ID]}"
MODEL_DIR="$REPO/models/$RUN"
OUT_DIR="$REPO/outputs/adv/attacks_${RUN}"

GPU_IDS="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "$GPU_IDS" || "$GPU_IDS" == "-1" || "$GPU_IDS" == "NoDevFiles" ]]; then
    echo "ERROR: Slurm did not assign a GPU"
    exit 1
fi

JSON=$(ls "$MODEL_DIR"/*.json 2>/dev/null | head -1 || true)
if [[ -z "$JSON" ]]; then
    echo "ERROR: no trained model in $MODEL_DIR (did its training job fail?)"
    exit 1
fi
NAME=$(basename "$JSON" .json)
MODEL="${NAME%%-*}"
test -r "$MODEL_DIR/$NAME.h5"
test -r "$REPO/models/SWAT-CLEAN_scaler.pkl"

read -r THETA WINDOW HISTORY < <(/usr/bin/apptainer exec --bind "$REPO:$REPO" "$IMAGE" python3 - "$JSON" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
print(repr(float(p["best_theta"])), int(p["best_window"]), int(p.get("history", 0)))
PY
)

HISTORY_ARGS=()
if [[ "$MODEL" != "AE" ]]; then
    HISTORY_ARGS=(--history "$HISTORY" --target-offset 0)
fi

echo "RUN=$RUN MODEL=$MODEL NAME=$NAME THETA=$THETA WINDOW=$WINDOW HISTORY=$HISTORY"
echo "HOST=$(hostname) GPU=$GPU_IDS START=$(date -Is)"
mkdir -p "$OUT_DIR"
cp "$JSON" "$OUT_DIR/model_params.json"
git -C "$REPO" rev-parse HEAD > "$OUT_DIR/git_commit.txt"
git -C "$REPO" diff > "$OUT_DIR/code_changes.patch"

RUN_CMD=(
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
    "${HISTORY_ARGS[@]}"
    --instance-threshold "$THETA"
    --detection-window "$WINDOW"
    --instance-score mean_mse
)

echo "===== SANITY CHECK ====="
"${RUN_CMD[@]}" --sanity-check 2>&1 | tee "$OUT_DIR/sanity.log"

echo "===== ATTACKS ====="
"${RUN_CMD[@]}" \
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

echo "$RUN FINISHED $(date -Is)"
