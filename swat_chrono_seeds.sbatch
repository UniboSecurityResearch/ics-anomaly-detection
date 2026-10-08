#!/bin/bash
#SBATCH --job-name=swat_chrono
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-17%3
#SBATCH --time=03:00:00
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mail-user=dhyana.schirra@studio.unibo.it
#SBATCH --chdir=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
#SBATCH --output=slurm.swat_chrono.%A_%a.out
#SBATCH --error=slurm.swat_chrono.%A_%a.err

# All 6 detectors on SWAT-CLEAN x 3 seeds (18 runs), chronological validation
# split, same threshold grid as before. Task id -> model = id % 6, seed = id / 6.
# Each run writes training.log and <model>-feature-errors.json to outputs/<run>/.

set -euo pipefail

REPO=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
IMAGE="$REPO/sip.sif"

MODELS=(CNN AE DNN GRU LSTM LIN)
SEEDS=(7 21 42)
MODEL="${MODELS[$((SLURM_ARRAY_TASK_ID % 6))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID / 6))]}"
RUN_NAME="swatchrono_${MODEL,,}_s${SEED}"

GPU_IDS="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "$GPU_IDS" || "$GPU_IDS" == "-1" || "$GPU_IDS" == "NoDevFiles" ]]; then
    echo "ERROR: Slurm did not assign a GPU"
    exit 1
fi

case "$MODEL" in
    CNN)  MODEL_ARGS=(--cnn_model_params_layers 3 --cnn_model_params_units 128 --cnn_model_params_history 100 --cnn_model_params_kernel 3) ;;
    AE)   MODEL_ARGS=(--ae_model_params_layers 5 --ae_model_params_cf 3.0) ;;
    DNN)  MODEL_ARGS=(--dnn_model_params_layers 4 --dnn_model_params_units 64 --dnn_model_params_history 50) ;;
    GRU)  MODEL_ARGS=(--gru_model_params_layers 2 --gru_model_params_units 256 --gru_model_params_history 100) ;;
    LSTM) MODEL_ARGS=(--lstm_model_params_layers 1 --lstm_model_params_units 128 --lstm_model_params_history 50) ;;
    LIN)  MODEL_ARGS=() ;;
esac

echo "MODEL=$MODEL SEED=$SEED RUN_NAME=$RUN_NAME HOST=$(hostname) GPU=$GPU_IDS START=$(date -Is)"

mkdir -p "$REPO/outputs/$RUN_NAME" "$REPO/models/$RUN_NAME" "$REPO/plots/$RUN_NAME" "$REPO/npys/$RUN_NAME"
git -C "$REPO" rev-parse HEAD > "$REPO/outputs/$RUN_NAME/git_commit.txt"
git -C "$REPO" diff > "$REPO/outputs/$RUN_NAME/code_changes.patch"

# Refuse to run if the shuffled-split edits are still in place.
if ! grep -q "Chronological train/validation split" "$REPO/utils.py"; then
    echo "ERROR: utils.py still uses the shuffled split"
    exit 1
fi

/usr/bin/apptainer exec \
    --nv \
    --bind "$REPO:$REPO" \
    --pwd "$REPO" \
    --env "PYTHONPATH=$REPO" \
    --env "CUDA_VISIBLE_DEVICES=$GPU_IDS" \
    --env "TF_FORCE_GPU_ALLOW_GROWTH=true" \
    --env "TF_XLA_FLAGS=--tf_xla_auto_jit=0" \
    --env "MPLBACKEND=Agg" \
    --env "PYTHONUNBUFFERED=1" \
    "$IMAGE" \
    python3 -u main_train.py \
        "$MODEL" SWAT-CLEAN \
        --run_name "$RUN_NAME" \
        --gpus "$GPU_IDS" \
        --seed "$SEED" \
        --train_params_epochs 100 \
        "${MODEL_ARGS[@]}" \
        --detect_params_metrics F1 \
        --detect_params_max_fpr 0.05 \
        --detect_params_percentile 0.98 0.99 0.995 0.999 0.9995 0.9999 \
        --detect_params_windows 1 3 5 10 20 30 \
    2>&1 | tee "$REPO/outputs/$RUN_NAME/training.log"

echo "$MODEL FINISHED $(date -Is)"
