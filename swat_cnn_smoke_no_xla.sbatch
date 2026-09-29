#!/bin/bash
#SBATCH --job-name=swat_cnn_noxla
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dhyana.schirra@studio.unibo.it
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
#SBATCH --output=slurm.swat_cnn_noxla.%j.out
#SBATCH --error=slurm.swat_cnn_noxla.%j.err

# A diagnostic run using the repository's EXISTING evaluation protocol.
# It does not fix train/validation leakage, change the prediction target,
# or implement a normal-only threshold-calibration protocol.
# First apply cnn_disable_optimizer_xla.patch to detector/cnn.py.
# Submit: sbatch swat_cnn_smoke_no_xla.sbatch
# This job does not modify repository source files or install packages.
set -euo pipefail
ERROR_LOG=""
report_error() {
    local status="$1" line="$2"
    trap - ERR
    printf 'ERROR: line %s, exit %s\n' "$line" "$status" >&2
    if [[ -n "$ERROR_LOG" && -f "$ERROR_LOG" ]]; then
        printf 'Last 100 lines of %s:\n' "$ERROR_LOG" >&2
        tail -n 100 "$ERROR_LOG" >&2 || true
    fi
    exit "$status"
}
trap 'report_error "$?" "$LINENO"' ERR
: "${SLURM_JOB_ID:?Submit this script with sbatch, not bash on the login node.}"

REPO=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
IMAGE="$REPO/sip.sif"
RUN_DIR="$REPO/outputs/swat_cnn_smoke_${SLURM_JOB_ID}"
MODEL_NAME=CNN-SWAT-l2-hist50-kern3-units64

export APPTAINER_CACHEDIR="$REPO"
command -v apptainer >/dev/null
for file in "$IMAGE" "$REPO/main_train.py" \
    "$REPO/data/SWAT/SWATv0_train.csv" \
    "$REPO/data/SWAT/SWATv0_test.csv"; do
    if [[ ! -s "$file" ]]; then
        printf 'Missing or empty file: %s\n' "$file" >&2
        exit 1
    fi
done

# Never guess GPU 0 when Slurm has not supplied a device allocation.
GPU_IDS="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "$GPU_IDS" || "$GPU_IDS" == "-1" || "$GPU_IDS" == "NoDevFiles" ]]; then
    echo "Slurm did not supply usable CUDA_VISIBLE_DEVICES. Check the GPU allocation." >&2
    exit 1
fi

# Fail rather than overwrite a previous run with the same job ID.
mkdir -p "$REPO/outputs"
mkdir "$RUN_DIR"
mkdir -p "$RUN_DIR/models/results" "$RUN_DIR/plots/results" "$RUN_DIR/npys/results"
ln -s "$REPO/data" "$RUN_DIR/data"
cp "$0" "$RUN_DIR/submitted_job.sbatch"

# The repository uses relative paths for datasets, scalers and training CSVs.
# A private working directory isolates ALL of these outputs, not just weights.
APPTAINER=(
    apptainer exec --nv
    --bind "$REPO:$REPO"
    --pwd "$RUN_DIR"
    --env "CUDA_VISIBLE_DEVICES=$GPU_IDS"
    --env "PYTHONPATH=$REPO"
    --env "TF_FORCE_GPU_ALLOW_GROWTH=true"
    --env "TF_XLA_FLAGS=--tf_xla_auto_jit=0"
    --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}"
    --env "TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK:-8}"
    --env "TF_NUM_INTEROP_THREADS=2"
    --env "MPLBACKEND=Agg"
    --env "PYTHONUNBUFFERED=1"
    "$IMAGE"
)

printf 'Started: %s\nHost: %s\nJob ID: %s\nGPU allocation: %s\nRun directory: %s\n' \
    "$(date -Is)" "$(hostname)" "$SLURM_JOB_ID" "$GPU_IDS" "$RUN_DIR"
if command -v git >/dev/null && git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$REPO" rev-parse HEAD > "$RUN_DIR/git_commit.txt"
    git -C "$REPO" status --short > "$RUN_DIR/git_status.txt"
    git -C "$REPO" diff HEAD -- main_train.py utils.py metrics.py data_loader.py detector \
        > "$RUN_DIR/tracked_code_changes.patch"
fi

echo "Checking GPU forward, backward and optimizer execution before loading SWaT..."
ERROR_LOG="$RUN_DIR/gpu_preflight.log"
"${APPTAINER[@]}" python3 -u - <<'PY' 2>&1 | tee "$ERROR_LOG"
import inspect
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
gpus = tf.config.list_physical_devices("GPU")
print("Visible GPUs:", gpus)
if not gpus:
    raise RuntimeError("TensorFlow sees no GPU. Stopping instead of training on CPU.")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# The separate optimizer-level XLA switch must also be disabled.
tf.config.optimizer.set_jit(False)

from detector.cnn import ConvNN
print("CNN source:", inspect.getfile(ConvNN))
tf.keras.utils.set_random_seed(42)
with tf.device("/GPU:0"):
    detector = ConvNN(nI=51, layers=2, units=64, history=50, kernel=3,
                      optimizer="adam", verbose=0)
    detector.create_model()
model = detector.inner
optimizer = model.optimizer
optimizer_jit = getattr(optimizer, "jit_compile", False)
model_jit = getattr(model, "jit_compile", False)
print("Optimizer:", optimizer.__class__.__name__)
print("Optimizer JIT:", optimizer_jit)
print("Model JIT:", model_jit)
if optimizer_jit or model_jit:
    raise RuntimeError(
        "XLA is still enabled. Apply cnn_disable_optimizer_xla.patch "
        "to the detector/cnn.py file printed above, then resubmit."
    )
weight = model.trainable_variables[0]
if "GPU" not in weight.device.upper():
    raise RuntimeError("CNN weights are not placed on a GPU: " + weight.device)

# These temporary weights are not saved. The real training starts in a
# separate Python process with its own seed and independently created CNN.
rng = np.random.default_rng(42)
x = rng.normal(size=(8, 50, 51)).astype(np.float32)
y = rng.normal(size=(8, 51)).astype(np.float32)
before = weight.numpy().copy()
iterations_before = int(optimizer.iterations.numpy())
losses = []
for step in range(2):
    with tf.device("/GPU:0"):
        loss = model.train_on_batch(x, y)
    loss_values = np.asarray(loss, dtype=np.float64)
    if not np.isfinite(loss_values).all():
        raise RuntimeError("Nonfinite preflight training loss.")
    losses.append(float(loss_values.reshape(-1)[0]))
iterations_after = int(optimizer.iterations.numpy())
if iterations_after != iterations_before + 2:
    raise RuntimeError("The optimizer did not complete two update steps.")
if not np.isfinite(weight.numpy()).all():
    raise RuntimeError("Preflight produced nonfinite weights.")
if np.array_equal(before, weight.numpy()):
    raise RuntimeError("CNN weights did not change during the training-step check.")
print("GPU training-step check passed: 2 optimizer updates; weights on", weight.device)
print("Preflight losses (synthetic data, NOT detection metrics):", losses)
Path("environment.json").write_text(json.dumps({
    "tensorflow": tf.__version__,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit_learn": sklearn.__version__,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "gpu_devices": [gpu.name for gpu in gpus],
    "tensorflow_build_info": tf.sysconfig.get_build_info(),
    "cnn_source": inspect.getfile(ConvNN),
    "optimizer": optimizer.__class__.__name__,
    "optimizer_jit_compile": bool(optimizer_jit),
    "model_jit_compile": bool(model_jit),
    "preflight_optimizer_steps": iterations_after - iterations_before,
    "preflight_losses_synthetic": losses,
}, indent=2, default=str) + "\n")
PY

# Up to 10 epochs; the repository's early-stopping callback stays enabled.
# Window=1 avoids the unresolved larger-window alignment/smoothing behavior.
# No adversarial perturbations are applied.
ERROR_LOG="$RUN_DIR/training.log"
echo "Training CNN and selecting its threshold using the existing repository protocol..."
"${APPTAINER[@]}" python3 -u "$REPO/main_train.py" \
    CNN SWAT \
    --run_name results \
    --gpus "$GPU_IDS" \
    --seed 42 \
    --cnn_model_params_layers 2 \
    --cnn_model_params_units 64 \
    --cnn_model_params_history 50 \
    --cnn_model_params_kernel 3 \
    --train_params_epochs 10 \
    --train_params_batch_size 256 \
    --detect_params_windows 1 \
    --detect_params_percentile 0.95 0.99 0.995 0.999 0.9995 0.99995 \
    --detect_params_metrics F1 \
    --detect_params_test_split 0.7 \
    --detect_params_save_npy \
    --detect_params_save_theta \
    2>&1 | tee "$RUN_DIR/training.log"

# Compute recall explicitly. In the current trainer, a JSON field named
# "final_recall" actually stores the chosen objective (here F1), not recall.
# This report is calculated from predictions, not that mislabeled field.
ERROR_LOG="$RUN_DIR/evaluation.log"
echo "Evaluating the saved model on the same held-out final 70%..."
"${APPTAINER[@]}" python3 -u - <<'PY' 2>&1 | tee "$RUN_DIR/evaluation.log"
import json
from pathlib import Path
import numpy as np
from data_loader import load_test_data
from main_train import load_saved_model
from utils import custom_train_test_split

model_name = "CNN-SWAT-l2-hist50-kern3-units64"
detector = load_saved_model("CNN", "results", model_name)
if int(detector.params["best_window"]) != 1:
    raise ValueError("This diagnostic evaluator requires detection window=1.")

X, labels, columns = load_test_data("SWAT")
labels = np.asarray(labels).astype(int)
if not np.isin(labels, [0, 1]).all():
    raise ValueError("Nonbinary test labels.")
_, X_final, _, y_final = custom_train_test_split(
    "SWAT", X, labels, test_size=0.7, shuffle=False
)

# Match the CURRENT code: input X[i-history:i], target X[i+1].
# Do not use this offset after independently changing the forecasting target.
history = int(detector.params["history"])
y_true = np.asarray(y_final[history + 1:]).astype(bool)
errors = np.asarray(detector.reconstruction_errors(X_final, batches=True))
if errors.shape != (len(y_true), len(columns)):
    raise ValueError("Residual/label shape mismatch: {} versus ({}, {}).".format(
        errors.shape, len(y_true), len(columns)
    ))
if not np.isfinite(errors).all():
    raise ValueError("Nonfinite model residuals.")

scores = errors.mean(axis=1)
predictions = np.asarray(detector.best_cached_detect(scores))
if predictions.shape != y_true.shape or not np.isin(predictions, [0, 1]).all():
    raise ValueError("Invalid prediction shape or nonbinary predictions.")
y_pred = predictions.astype(bool)

def pointwise_metrics(y_true, y_pred):
    tp = int(np.count_nonzero(y_true & y_pred))
    fn = int(np.count_nonzero(y_true & ~y_pred))
    fp = int(np.count_nonzero(~y_true & y_pred))
    tn = int(np.count_nonzero(~y_true & ~y_pred))
    if tp + fn == 0 or fp + tn == 0:
        raise ValueError("Final evaluation needs both attack and normal samples.")
    return {
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives": tn,
        "recall": tp / (tp + fn),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn),
        "false_positive_rate": fp / (fp + tn),
    }

report = {
    "dataset": "SWAT",
    "label_protocol": "repository-corrected attack labels",
    "model": model_name,
    "seed": 42,
    "adversarial_perturbations": False,
    "threshold_selection": "F1 on first 30% of attack/test file",
    "evaluation_split": "final 70% of attack/test file",
    "evaluation_samples_after_history": len(y_true),
    "history": history,
    "target_offset_from_input_end_exclusive": 1,
    "score": "mean squared error across features",
    "threshold": float(detector.params["best_theta"]),
    "threshold_percentile": float(detector.params["threshold_percentile"]),
    "detection_window": 1,
    "point_adjustment": False,
    **pointwise_metrics(y_true, y_pred),
}
for label, mask in (("normal", ~y_true), ("attack", y_true)):
    report[label + "_score_quantiles"] = {
        str(q): float(np.quantile(scores[mask], q))
        for q in (0.5, 0.95, 0.99, 0.999)
    }

Path("clean_metrics.json").write_text(
    json.dumps(report, indent=2, allow_nan=False) + "\n"
)
np.savez_compressed(
    "clean_predictions.npz", scores=scores, labels=y_true, predictions=y_pred
)
print(json.dumps(report, indent=2, allow_nan=False))
print("Saved clean_metrics.json and clean_predictions.npz")
PY

printf '\nFinished successfully: %s\n' "$(date -Is)"
printf 'Model: %s/models/results/%s.h5\n' "$RUN_DIR" "$MODEL_NAME"
printf 'Model parameters: %s/models/results/%s.json\n' "$RUN_DIR" "$MODEL_NAME"
printf 'Scaler: %s/models/SWAT_scaler.pkl\n' "$RUN_DIR"
printf 'Metrics: %s/clean_metrics.json\n' "$RUN_DIR"
