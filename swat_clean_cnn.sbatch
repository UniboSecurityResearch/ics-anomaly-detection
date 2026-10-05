#!/bin/bash
#SBATCH --job-name=swat_clean_cnn
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_40GB:1
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dhyana.schirra@studio.unibo.it
#SBATCH --chdir=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
#SBATCH --output=slurm.swat_clean_cnn.%j.out
#SBATCH --error=slurm.swat_clean_cnn.%j.err

set -euo pipefail

REPO=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
IMAGE="$REPO/sip.sif"
RUN_NAME="swat_clean_cnn"

echo "HOST=$(hostname)"
echo "START=$(date -Is)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"

echo
echo "===== HOST GPU ====="
nvidia-smi -L

echo
echo "===== CONTAINER GPU PREFLIGHT ====="

env -u CUDA_VISIBLE_DEVICES /usr/bin/apptainer exec     --nv     --bind "$REPO:$REPO"     --pwd "$REPO"     "$IMAGE"     python3 - <<'PY'
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow GPUs:", gpus)

if not gpus:
    raise SystemExit("FATAL: GPU allocated by Slurm but not visible inside container")
PY

echo
echo "===== TRAINING ====="

mkdir -p     "$REPO/outputs/$RUN_NAME"     "$REPO/models/$RUN_NAME"     "$REPO/plots/$RUN_NAME"     "$REPO/npys/$RUN_NAME"

env -u CUDA_VISIBLE_DEVICES /usr/bin/apptainer exec     --nv     --bind "$REPO:$REPO"     --pwd "$REPO"     --env "PYTHONPATH=$REPO"     "$IMAGE"     python3 -u main_train.py         CNN SWAT-CLEAN         --run_name "$RUN_NAME"         --cnn_model_params_layers 3         --cnn_model_params_history 100         --cnn_model_params_kernel 3         --cnn_model_params_units 128         --train_params_epochs 100         --detect_params_percentile             0.98 0.99 0.995 0.999 0.9995 0.9999         --detect_params_windows             1 3 5 10 20 30         --detect_params_max_fpr 0.05
