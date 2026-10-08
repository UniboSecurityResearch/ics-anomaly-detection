#!/bin/bash
#SBATCH --job-name=swat_sweep
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-71%4
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=dhyana.schirra@studio.unibo.it
#SBATCH --chdir=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
#SBATCH --output=slurm.swat_sweep.%A_%a.out
#SBATCH --error=slurm.swat_sweep.%A_%a.err

# Evasion benchmark sweep on the chronological SWAT-CLEAN detectors.
#   task id -> model  = id / 6  (AE, CNN, DNN, LSTM x training seeds 7, 21, 42)
#              budget = id % 6
#   budgets: 0-3 range fraction 0.02 / 0.05 / 0.10 / 0.20, actuators protected (main)
#            4   standardized epsilon 0.10, actuators protected (reference)
#            5   range fraction 0.10, actuators NOT protected (reference)
# Per task: sanity check, then FGSM, PGD (step eps/10 or eps/4 x 50 or 200
# iterations), PGD top-k, all with the PGD loss on the clean-detected set E,
# 3 restarts, best-loss and best-ASR checkpoints, changed rows saved; plus CorrShift.
# Pilot first with --array=0-5 (one model, all budgets) to measure the runtime.

set -euo pipefail
REPO=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
IMAGE="$REPO/sip.sif"

RUNS=()
for SEED in 7 21 42; do
    for M in ae cnn dnn lstm; do
        RUNS+=("swatchrono_${M}_s${SEED}")
    done
done
RUN="${RUNS[$((SLURM_ARRAY_TASK_ID / 6))]}"
BUDGET_ID=$((SLURM_ARRAY_TASK_ID % 6))
case "$BUDGET_ID" in
    0) BUDGET_TAG=range0.02_protected;  BUDGET=(--epsilon-range-fraction 0.02 --protect-actuators) ;;
    1) BUDGET_TAG=range0.05_protected;  BUDGET=(--epsilon-range-fraction 0.05 --protect-actuators) ;;
    2) BUDGET_TAG=range0.10_protected;  BUDGET=(--epsilon-range-fraction 0.10 --protect-actuators) ;;
    3) BUDGET_TAG=range0.20_protected;  BUDGET=(--epsilon-range-fraction 0.20 --protect-actuators) ;;
    4) BUDGET_TAG=std0.10_protected;    BUDGET=(--epsilon 0.10 --protect-actuators) ;;
    5) BUDGET_TAG=range0.10_unprotected; BUDGET=(--epsilon-range-fraction 0.10) ;;
esac

MODEL_DIR="$REPO/models/$RUN"
OUT="$REPO/outputs/sweep/$RUN/$BUDGET_TAG"
JSON=$(ls "$MODEL_DIR"/*.json 2>/dev/null | head -1 || true)
if [[ -z "$JSON" ]]; then echo "ERROR: no model in $MODEL_DIR"; exit 1; fi
NAME=$(basename "$JSON" .json)
MODEL="${NAME%%-*}"
read -r THETA WINDOW HISTORY < <(/usr/bin/apptainer exec --bind "$REPO:$REPO" "$IMAGE" python3 - "$JSON" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
print(repr(float(p["best_theta"])), int(p["best_window"]), int(p.get("history", 0)))
PY
)
HISTORY_ARGS=()
if [[ "$MODEL" != "AE" ]]; then HISTORY_ARGS=(--history "$HISTORY" --target-offset 0); fi

mkdir -p "$OUT"
cp "$JSON" "$OUT/model_params.json"
git -C "$REPO" rev-parse HEAD > "$OUT/git_commit.txt"
git -C "$REPO" diff > "$OUT/code_changes.patch"
echo "RUN=$RUN MODEL=$MODEL BUDGET=$BUDGET_TAG THETA=$THETA WINDOW=$WINDOW HISTORY=$HISTORY"
echo "HOST=$(hostname) GPU=${CUDA_VISIBLE_DEVICES:-none} START=$(date -Is)"

BASE=(
    /usr/bin/apptainer exec --nv
    --bind "$REPO:$REPO" --pwd "$REPO"
    --env "PYTHONPATH=$REPO"
    --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
    --env "TF_FORCE_GPU_ALLOW_GROWTH=true"
    --env "TF_XLA_FLAGS=--tf_xla_auto_jit=0"
    --env "PYTHONUNBUFFERED=1"
    "$IMAGE"
    python3 -u main_adversarial.py
    --dataset SWAT-CLEAN --model-type "$MODEL" --model-path "$MODEL_DIR/$NAME.h5"
    "${HISTORY_ARGS[@]}"
    --instance-threshold "$THETA" --detection-window "$WINDOW" --instance-score mean_mse
    --selection attack --target-sampling segments --max-targets 8192
    --scope both --score mean_mse --seed 42
    "${BUDGET[@]}"
)
PROTOCOL=(--goal evasion --eligible-protocol --return-point best_asr --save-series changed)

run() {  # $1 = tag, rest = attack arguments
    local tag="$1"; shift
    echo "===== $tag $(date -Is)"
    mkdir -p "$OUT/$tag"
    "${BASE[@]}" "$@" --output-dir "$OUT/$tag" 2>&1 | tee "$OUT/$tag/run.log"
}

"${BASE[@]}" --sanity-check 2>&1 | tee "$OUT/sanity.log"

run fgsm             --attack fgsm_mse "${PROTOCOL[@]}" --restarts 1
run pgd_r0.10_it50   --attack pgd_mse  "${PROTOCOL[@]}" --alpha-relative 0.10 --iterations 50  --random-start --restarts 3
run pgd_r0.10_it200  --attack pgd_mse  "${PROTOCOL[@]}" --alpha-relative 0.10 --iterations 200 --random-start --restarts 3
run pgd_r0.25_it50   --attack pgd_mse  "${PROTOCOL[@]}" --alpha-relative 0.25 --iterations 50  --random-start --restarts 3
run pgd_r0.25_it200  --attack pgd_mse  "${PROTOCOL[@]}" --alpha-relative 0.25 --iterations 200 --random-start --restarts 3
run topk_r0.10_it200 --attack pgd_topk "${PROTOCOL[@]}" --alpha-relative 0.10 --iterations 200 --random-start --restarts 3 --top-k 5
run corrshift        --attack corrshift --goal evasion --corrshift-rounds 3 --corrshift-step 1000 --save-series changed

echo "FINISHED $(date -Is)"
