#!/bin/bash
#SBATCH --job-name=swat_lin
#SBATCH --partition=l40
#SBATCH --nodelist=slurmwkrl40.disi.unibo.it
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40:1
#SBATCH --time=04:00:00
#SBATCH --chdir=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
#SBATCH --output=slurm.swat_lin.%j.out
#SBATCH --error=slurm.swat_lin.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dhyana.schirra@studio.unibo.it

set -euo pipefail

REPO=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
IMAGE="$REPO/sip.sif"
APPTAINER=/usr/bin/apptainer

GPU_IDS="${CUDA_VISIBLE_DEVICES:-}"

if [[ -z "$GPU_IDS" || "$GPU_IDS" == "-1" || "$GPU_IDS" == "NoDevFiles" ]]; then
    echo "ERROR: Slurm did not assign a GPU"
    exit 1
fi

RUN_DIR="$REPO/outputs/tune2_linfix_${SLURM_JOB_ID}"

mkdir -p "$RUN_DIR/models/results"
mkdir -p "$RUN_DIR/plots/results"
mkdir -p "$RUN_DIR/npys/results"

ln -sfn "$REPO/data" "$RUN_DIR/data"

cp "$0" "$RUN_DIR/submitted_job.sbatch"

git -C "$REPO" rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git -C "$REPO" status --short > "$RUN_DIR/git_status.txt"

export APPTAINER_CACHEDIR="$REPO"

echo "========================================"
echo "MODEL:   LIN"
echo "HOST:    $(hostname)"
echo "GPU:     $GPU_IDS"
echo "JOB:     $SLURM_JOB_ID"
echo "START:   $(date -Is)"
echo "RUN DIR: $RUN_DIR"
echo "========================================"

"$APPTAINER" exec \
    --nv \
    --bind "$REPO:$REPO" \
    --pwd "$RUN_DIR" \
    --env "PYTHONPATH=$REPO" \
    --env "CUDA_VISIBLE_DEVICES=$GPU_IDS" \
    --env "TF_FORCE_GPU_ALLOW_GROWTH=true" \
    --env "TF_XLA_FLAGS=--tf_xla_auto_jit=0" \
    --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}" \
    --env "TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK:-8}" \
    --env "TF_NUM_INTEROP_THREADS=2" \
    --env "MPLBACKEND=Agg" \
    --env "PYTHONUNBUFFERED=1" \
    "$IMAGE" \
    python3 -u "$REPO/main_train.py" \
        LIN SWAT \
        --run_name results \
        --gpus "$GPU_IDS" \
        --seed 42 \
        --train_params_epochs 150 \
        --train_params_batch_size 256 \
        --detect_params_metrics F1 \
        --detect_params_max_fpr 0.05 \
        --detect_params_windows \
            1 3 5 10 20 30 \
        --detect_params_percentile \
            0.95 \
            0.97 \
            0.98 \
            0.99 \
            0.995 \
            0.999 \
            0.9995 \
            0.9999 \
            0.99995 \
        2>&1 | tee "$RUN_DIR/training.log"

echo
echo "========================================"
echo "LIN COMPLETED"
echo "END: $(date -Is)"
echo "OUTPUT: $RUN_DIR"
echo "========================================"
