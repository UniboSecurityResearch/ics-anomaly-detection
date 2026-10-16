#!/bin/bash
#SBATCH --job-name=swat_sweep
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-89%3
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=dhyana.schirra@studio.unibo.it
#SBATCH --chdir=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
#SBATCH --output=slurm.swat_sweep.%A_%a.out
#SBATCH --error=slurm.swat_sweep.%A_%a.err

# Evasion benchmark sweep on the chronological SWAT-CLEAN detectors (90 tasks).
# GRU added after the chronological split fixed it (F1 0.852 +- 0.008); AE seed 7 stays tasks 0-5.
#   task id -> model  = id / 6  (AE, CNN, DNN, LSTM, GRU x training seeds 7, 21, 42)
#              budget = id % 6
#   budgets: 0-3 range fraction 0.02 / 0.05 / 0.10 / 0.20, actuators protected (main)
#            4   standardized epsilon 0.10, actuators protected (reference)
#            5   range fraction 0.10, actuators NOT protected (reference)
# Per task: sanity check, then FGSM, PGD (step eps/20, eps/10, eps/4; 200 iterations),
# PGD top-k, all with the PGD loss on the clean-detected set E, perturbing only the
# rows that feed that loss, 3 restarts, best-loss and best-ASR checkpoints, changed
# rows saved; plus CorrShift. Fewer iterations: read iteration_metrics.csv up to
# that iteration (same seeds, so a 50-iteration run is a prefix of the 200 one).
# v2 (outputs/sweep_v2): attack mask restricted to E; v1 results kept in outputs/sweep.
# CorrShift: score on E, shift only E rows, 30 greedy moves of eps/10 (v1 used one
# full-eps move on all target rows, which never lowered the score: 0 changed cells).

set -euo pipefail
REPO=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
IMAGE="$REPO/sip.sif"

# SWEEP_MODELS (optional) replaces the model list, e.g. SWEEP_MODELS=lin gives
# 18 tasks (0-17) for the linear model with the same budget mapping. Default: the
# original 5 models, so task ids 0-89 keep their meaning.
SWEEP_MODELS="${SWEEP_MODELS:-ae cnn dnn lstm gru}"
RUNS=()
for SEED in 7 21 42; do
    for M in $SWEEP_MODELS; do
        RUNS+=("swatchrono_${M}_s${SEED}")
    done
done
if (( SLURM_ARRAY_TASK_ID / 6 >= ${#RUNS[@]} )); then
    echo "ERROR: task ${SLURM_ARRAY_TASK_ID} has no model (models: $SWEEP_MODELS)"; exit 1
fi
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
# RANGE_FRACTION (optional) replaces the budget with that range fraction, actuators
# protected; used with ONLY for extra budgets, e.g. RANGE_FRACTION=0.01 (the
# CorrShift coefficient of FEVA-ICS). Only task id / 6 (the model) matters then.
if [[ -n "${RANGE_FRACTION:-}" ]]; then
    BUDGET_TAG="range${RANGE_FRACTION}_protected"
    BUDGET=(--epsilon-range-fraction "$RANGE_FRACTION" --protect-actuators)
fi

MODEL_DIR="$REPO/models/$RUN"
OUT="$REPO/outputs/sweep_v2/$RUN/$BUDGET_TAG"
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
# With ONLY set (a rerun of some attacks) keep the budget folder's original
# provenance; each attack folder gets its own git_commit.txt below.
if [[ -z "${ONLY:-}" || ! -e "$OUT/model_params.json" ]]; then
    # (with ONLY, only a budget folder that does not exist yet gets these files)
    cp "$JSON" "$OUT/model_params.json"
    git -C "$REPO" rev-parse HEAD > "$OUT/git_commit.txt"
    git -C "$REPO" diff > "$OUT/code_changes.patch"
fi
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
    # ONLY="<tag> [<tag> ...]" (sbatch --export=ALL,ONLY=corrshift) reruns only
    # the listed attacks.
    if [[ -n "${ONLY:-}" && " $ONLY " != *" $tag "* ]]; then return 0; fi
    echo "===== $tag $(date -Is)"
    mkdir -p "$OUT/$tag"
    git -C "$REPO" rev-parse HEAD > "$OUT/$tag/git_commit.txt"
    git -C "$REPO" diff > "$OUT/$tag/code_changes.patch"
    "${BASE[@]}" "$@" --output-dir "$OUT/$tag" 2>&1 | tee "$OUT/$tag/run.log"
}

SANITY_LOG="$OUT/sanity.log"
if [[ -n "${ONLY:-}" ]]; then SANITY_LOG="$OUT/sanity_${ONLY// /_}.log"; fi
"${BASE[@]}" --sanity-check 2>&1 | tee "$SANITY_LOG"

run fgsm             --attack fgsm_mse "${PROTOCOL[@]}" --restarts 1
run pgd_r0.05_it200  --attack pgd_mse  "${PROTOCOL[@]}" --alpha-relative 0.05 --iterations 200 --random-start --restarts 3
run pgd_r0.10_it200  --attack pgd_mse  "${PROTOCOL[@]}" --alpha-relative 0.10 --iterations 200 --random-start --restarts 3
run pgd_r0.25_it200  --attack pgd_mse  "${PROTOCOL[@]}" --alpha-relative 0.25 --iterations 200 --random-start --restarts 3
run topk_r0.10_it200 --attack pgd_topk "${PROTOCOL[@]}" --alpha-relative 0.10 --iterations 200 --random-start --restarts 3 --top-k 5
# Longer runs, only on request (not part of the 90-task sweep). Same seeds, so the
# first 200 iterations of pgd_r0.05_it500 repeat pgd_r0.05_it200. pgd_r0.025 has at
# range 0.20 the same absolute step as pgd_r0.05 at range 0.10. Used for range 0.20:
#   sbatch --array=3-87:6%3 --export=ALL,ONLY="pgd_r0.05_it500 pgd_r0.025_it500" swat_sweep.sbatch
if [[ -n "${ONLY:-}" ]]; then
    run pgd_r0.05_it500  --attack pgd_mse "${PROTOCOL[@]}" --alpha-relative 0.05  --iterations 500 --random-start --restarts 3
    run pgd_r0.025_it500 --attack pgd_mse "${PROTOCOL[@]}" --alpha-relative 0.025 --iterations 500 --random-start --restarts 3
    # CorrShift as in FEVA-ICS (Algorithm 1): one constant shift per batch of 60 rows,
    # candidate chosen by the number of targets still detected. On request only:
    #   sbatch --array=0-89%3 --export=ALL,ONLY=corrshift_feva swat_sweep.sbatch
    run corrshift_feva   --attack corrshift_feva --goal evasion --feva-batch 60 --feva-corr-source attacked --save-series changed
fi
run corrshift        --attack corrshift --goal evasion --corrshift-eligible --corrshift-rounds 30 --corrshift-step-relative 0.10 --save-series changed

echo "FINISHED $(date -Is)"
