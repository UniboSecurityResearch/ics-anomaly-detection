#!/bin/bash
#SBATCH --job-name=swat_all
#SBATCH --partition=h100sxm5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100sxm5:1
#SBATCH --array=5-6%2
#SBATCH --chdir=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
#SBATCH --output=slurm.swat_all.%A_%a.out
#SBATCH --error=slurm.swat_all.%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dhyana.schirra@studio.unibo.it

set -euo pipefail

REPO=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
IMAGE="$REPO/sip.sif"
APPTAINER=/usr/bin/apptainer

MODELS=(AE CNN DNN GRU LSTM LIN ID)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

GPU_IDS="${CUDA_VISIBLE_DEVICES:-}"

if [[ -z "$GPU_IDS" || "$GPU_IDS" == "-1" || "$GPU_IDS" == "NoDevFiles" ]]; then
    echo "ERROR: Slurm did not assign a GPU"
    exit 1
fi

RUN_DIR="$REPO/outputs/swat_${MODEL,,}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$RUN_DIR/models/results"
mkdir -p "$RUN_DIR/plots/results"
mkdir -p "$RUN_DIR/npys/results"

ln -s "$REPO/data" "$RUN_DIR/data"

cp "$0" "$RUN_DIR/submitted_job.sbatch"

git -C "$REPO" rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git -C "$REPO" status --short > "$RUN_DIR/git_status.txt"

export APPTAINER_CACHEDIR="$REPO"

APPTAINER_CMD=(
    "$APPTAINER" exec
    --nv
    --bind "$REPO:$REPO"
    --pwd "$RUN_DIR"
    --env "PYTHONPATH=$REPO"
    --env "CUDA_VISIBLE_DEVICES=$GPU_IDS"
    --env "TF_FORCE_GPU_ALLOW_GROWTH=true"
    --env "TF_XLA_FLAGS=--tf_xla_auto_jit=0"
    --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}"
    --env "TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK:-8}"
    --env "TF_NUM_INTEROP_THREADS=2"
    --env "MPLBACKEND=Agg"
    --env "PYTHONUNBUFFERED=1"
    "$IMAGE"
)

COMMON_ARGS=(
    SWAT
    --run_name results
    --gpus "$GPU_IDS"
    --seed 42
    --train_params_epochs 100
    --train_params_batch_size 256
    --detect_params_metrics F1
    --detect_params_windows 1
    --detect_params_percentile
    0.90
    0.95
    0.97
    0.98
    0.99
    0.995
    0.999
    0.9995
)

echo "========================================"
echo "MODEL:      $MODEL"
echo "HOST:       $(hostname)"
echo "GPU:        $GPU_IDS"
echo "JOB:        $SLURM_JOB_ID"
echo "ARRAY JOB:  $SLURM_ARRAY_JOB_ID"
echo "TASK:       $SLURM_ARRAY_TASK_ID"
echo "RUN DIR:    $RUN_DIR"
echo "START:      $(date -Is)"
echo "========================================"

case "$MODEL" in

    AE)
        "${APPTAINER_CMD[@]}" \
            python3 -u "$REPO/main_train.py" \
            AE "${COMMON_ARGS[@]}" \
            --ae_model_params_layers 5 \
            --ae_model_params_cf 2.5 \
            2>&1 | tee "$RUN_DIR/training.log"
        ;;

    DNN)
        "${APPTAINER_CMD[@]}" \
            python3 -u "$REPO/main_train.py" \
            DNN "${COMMON_ARGS[@]}" \
            --dnn_model_params_layers 4 \
            --dnn_model_params_units 64 \
            --dnn_model_params_history 50 \
            2>&1 | tee "$RUN_DIR/training.log"
        ;;

    GRU)
        "${APPTAINER_CMD[@]}" \
            python3 -u "$REPO/main_train.py" \
            GRU "${COMMON_ARGS[@]}" \
            --gru_model_params_layers 2 \
            --gru_model_params_units 256 \
            --gru_model_params_history 100 \
            2>&1 | tee "$RUN_DIR/training.log"
        ;;

    LSTM)
        "${APPTAINER_CMD[@]}" \
            python3 -u "$REPO/main_train.py" \
            LSTM "${COMMON_ARGS[@]}" \
            --lstm_model_params_layers 4 \
            --lstm_model_params_units 512 \
            --lstm_model_params_history 100 \
            2>&1 | tee "$RUN_DIR/training.log"
        ;;

    LIN)
        "${APPTAINER_CMD[@]}" \
            python3 -u "$REPO/main_train.py" \
            LIN "${COMMON_ARGS[@]}" \
            2>&1 | tee "$RUN_DIR/training.log"
        ;;

    ID)
        "${APPTAINER_CMD[@]}" \
            python3 -u "$REPO/main_train.py" \
            ID "${COMMON_ARGS[@]}" \
            2>&1 | tee "$RUN_DIR/training.log"
        ;;

    *)
        echo "ERROR: unsupported model $MODEL"
        exit 1
        ;;
esac

echo
echo "========================================"
echo "$MODEL COMPLETED"
echo "END: $(date -Is)"
echo "OUTPUT: $RUN_DIR"
echo "========================================"