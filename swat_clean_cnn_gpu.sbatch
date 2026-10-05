#!/bin/bash
#SBATCH --job-name=swat_clean_cnn
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_40GB:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dhyana.schirra@studio.unibo.it
#SBATCH --chdir=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
#SBATCH --output=slurm.swat_clean_cnn.%j.out
#SBATCH --error=slurm.swat_clean_cnn.%j.err

# Same experiment as swat_clean_cnn.sbatch, but the GPU that Slurm allocates is
# passed through to main_train.py. Without --gpus, main_train.py hides every GPU
# and trains on CPU (about 430 s per epoch, killed by the time limit).

set -euo pipefail

REPO=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
IMAGE="$REPO/sip.sif"
RUN_NAME="swat_clean_cnn"

GPU_IDS="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "$GPU_IDS" || "$GPU_IDS" == "-1" || "$GPU_IDS" == "NoDevFiles" ]]; then
    echo "ERROR: Slurm did not assign a GPU"
    exit 1
fi

echo "HOST=$(hostname)"
echo "START=$(date -Is)"
echo "GPU_IDS=$GPU_IDS"
nvidia-smi -L

mkdir -p \
    "$REPO/outputs/$RUN_NAME" \
    "$REPO/models/$RUN_NAME" \
    "$REPO/plots/$RUN_NAME" \
    "$REPO/npys/$RUN_NAME"

git -C "$REPO" rev-parse HEAD > "$REPO/outputs/$RUN_NAME/git_commit.txt"
git -C "$REPO" status --short > "$REPO/outputs/$RUN_NAME/git_status.txt"

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
        CNN SWAT-CLEAN \
        --run_name "$RUN_NAME" \
        --gpus "$GPU_IDS" \
        --cnn_model_params_layers 3 \
        --cnn_model_params_history 100 \
        --cnn_model_params_kernel 3 \
        --cnn_model_params_units 128 \
        --train_params_epochs 100 \
        --detect_params_percentile \
            0.98 0.99 0.995 0.999 0.9995 0.9999 \
        --detect_params_windows \
            1 3 5 10 20 30 \
        --detect_params_max_fpr 0.05 \
    2>&1 | tee "$REPO/outputs/$RUN_NAME/training.log"

echo "FINISHED $(date -Is)"
