#!/bin/bash
#SBATCH --job-name=pgd-fgsm-smoke
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dhyana.schirra@studio.unibo.it
#SBATCH --partition=l40s
#SBATCH --nodelist=slurmwkrl40s3.disi.unibo.it
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --chdir=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

set -euo pipefail

PROJECT_DIR=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
IMAGE="$PROJECT_DIR/sip.sif"
MODEL_PATH="$PROJECT_DIR/models/cnn/CNN-BATADAL-l2-hist50-kern3-units64.h5"
OUTPUT_DIR="$PROJECT_DIR/outputs/smoke_pgd_fgsm_seed42_job${SLURM_JOB_ID}"

export APPTAINER_CACHEDIR="$PROJECT_DIR"
export PYTHONHASHSEED=42
export TF_DETERMINISTIC_OPS=1
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MODEL_PATH OUTPUT_DIR

test -f "$IMAGE" || {
    echo "ERRORE: immagine Apptainer non trovata: $IMAGE" >&2
    exit 1
}

test -e "$MODEL_PATH" || {
    echo "ERRORE: modello non trovato: $MODEL_PATH" >&2
    exit 1
}

echo "Job ID: $SLURM_JOB_ID"
echo "Nodo: $SLURMD_NODENAME"
echo "Modello: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"

apptainer exec --nv \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --pwd "$PROJECT_DIR" \
    "$IMAGE" python3 - <<'PY'
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from adversarial.attacks.fgsm_mse import FgsmMse
from adversarial.attacks.pgd_mse import PgdMse
from adversarial.build import build_context
from adversarial.cli import parse_args
from adversarial.errors import model_errors_tf
from adversarial.evaluation import evaluate_series, save_run
from adversarial.projection import build_tf_bounds


SEED = 42
ITERATIONS = 4
EPSILON = 0.10
ALPHA = 0.04
MODEL_PATH = os.environ["MODEL_PATH"]
OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])

# La soglia del detector non serve per questi test di regressione. Il comando
# usa un unico seed, 4 iterazioni, 16 target e protegge esplicitamente la
# feature 0, così viene esercitata anche la modification mask.
sys.argv = [
    "main_adversarial.py",
    "--dataset", "BATADAL",
    "--model-type", "CNN",
    "--history", "50",
    "--target-offset", "1",
    "--model-path", MODEL_PATH,
    "--attack", "pgd_mse",
    "--goal", "evasion",
    "--selection", "attack",
    "--max-targets", "16",
    "--scope", "both",
    "--seed", str(SEED),
    "--score", "mean_mse",
    "--epsilon", str(EPSILON),
    "--alpha", str(ALPHA),
    "--iterations", str(ITERATIONS),
    "--random-start",
    "--protected-cols", "0",
    "--prediction-batch-size", "256",
    "--output-dir", str(OUTPUT_DIR),
    "--save-series", "full",
]

args = parse_args()
ctx = build_context(args)
errors_before, scores_before = evaluate_series(ctx, ctx.x_test)


class RecordingPgd(PgdMse):
    """PGD reale con registrazione della loss a ogni punto visitato."""

    def __init__(self):
        self.losses = []

    def objective(self, *objective_args, **objective_kwargs):
        loss, report_score = super().objective(
            *objective_args, **objective_kwargs
        )
        self.losses.append(float(loss.numpy()))
        return loss, report_score


# ---------------------------------------------------------------------------
# PGD: best_loss, best_iteration, punto restituito/salvato, epsilon e mask.
# ---------------------------------------------------------------------------
pgd = RecordingPgd()
x_pgd, pgd_meta = pgd.run(ctx, "evasion")

# Il loop valuta la posizione iniziale (indice 0), le posizioni dopo i primi
# 1..ITERATIONS-1 update e infine quella dopo l'ultimo update.
losses = np.asarray(pgd.losses, dtype=np.float64)
assert losses.size == ITERATIONS + 1, (
    f"Attese {ITERATIONS + 1} loss, ricevute {losses.size}: {losses.tolist()}"
)
assert np.all(np.isfinite(losses)), f"Loss non finite: {losses.tolist()}"

expected_best_iteration = int(np.argmin(losses))
expected_best_loss = float(losses[expected_best_iteration])

assert pgd_meta["iterations_executed"] == ITERATIONS
assert pgd_meta["returned_point"] == "best_adv"
assert pgd_meta["best_iteration"] == expected_best_iteration, (
    f"best_iteration={pgd_meta['best_iteration']}, "
    f"attesa={expected_best_iteration}; trajectory={losses.tolist()}"
)
np.testing.assert_allclose(
    pgd_meta["best_loss"],
    expected_best_loss,
    rtol=1e-6,
    atol=1e-7,
)
assert pgd_meta["best_loss"] <= pgd_meta["final_optimization_loss"] + 1e-7, (
    f"best_loss={pgd_meta['best_loss']} > "
    f"final_loss={pgd_meta['final_optimization_loss']}"
)

pgd_summary = save_run(
    ctx,
    "pgd_mse",
    "evasion",
    x_pgd,
    pgd_meta,
    errors_before,
    scores_before,
)

pgd_dir = OUTPUT_DIR / "pgd_mse" / "evasion"
saved_pgd = np.load(pgd_dir / "x_test_adversarial_scaled.npy")
saved_delta = np.load(pgd_dir / "delta_scaled.npy")
saved_errors = np.load(pgd_dir / "feature_errors_after.npy")

# Il file deve contenere il medesimo punto restituito da run().
np.testing.assert_array_equal(saved_pgd, x_pgd)
np.testing.assert_allclose(
    saved_delta,
    x_pgd - ctx.x_test,
    rtol=0,
    atol=1e-7,
)

# Con goal=evasion e score=mean_mse la loss del punto salvato deve coincidere
# con best_loss. È ammessa una minima tolleranza fra i path TF e NumPy.
saved_point_loss = float(np.mean(saved_errors))
np.testing.assert_allclose(
    saved_point_loss,
    pgd_meta["best_loss"],
    rtol=5e-5,
    atol=1e-7,
)

delta = x_pgd - ctx.x_test
epsilon = np.broadcast_to(ctx.epsilon.reshape(1, -1), delta.shape)
raw_budget_excess = float(np.max(np.abs(delta) - epsilon))
max_budget_excess = max(0.0, raw_budget_excess)
assert raw_budget_excess <= 5e-7, (
    f"Budget epsilon violato di {raw_budget_excess}"
)

outside_mask = np.abs(delta[ctx.modification_mask == 0])
max_outside_mask = float(np.max(outside_mask)) if outside_mask.size else 0.0
assert max_outside_mask <= 1e-7, (
    f"Modification mask violata: max delta non consentito={max_outside_mask}"
)


# ---------------------------------------------------------------------------
# FGSM: confronto con il comportamento legacy, cioè un singolo passo di eps,
# senza random start, seguito dalla stessa proiezione e dalla stessa mask.
# ---------------------------------------------------------------------------
def legacy_fgsm_reference(context):
    x0 = tf.convert_to_tensor(context.x_test, dtype=tf.float32)
    mask = tf.convert_to_tensor(context.modification_mask, dtype=tf.float32)
    eps, lower, upper = build_tf_bounds(
        x0,
        context.epsilon,
        context.lower_domain,
        context.upper_domain,
    )

    reference_attack = PgdMse()
    with tf.GradientTape() as tape:
        tape.watch(x0)
        errors = model_errors_tf(
            context.adapter.keras_model,
            x0,
            context.args.model_type,
            context.target_indices,
            context.args.history,
            context.args.target_offset,
        )
        loss, _ = reference_attack.objective(
            context,
            "evasion",
            errors,
            x0,
            x0,
        )

    gradient = tape.gradient(loss, x0)
    assert gradient is not None, "Il gradiente FGSM legacy è None"
    gradient = gradient * mask
    gradient = reference_attack.process_gradient(context, gradient, mask)

    x_adv = x0 - eps * tf.sign(gradient)
    x_adv = tf.clip_by_value(x_adv, lower, upper)
    x_adv = x0 + (x_adv - x0) * mask
    return x_adv.numpy()


x_fgsm, fgsm_meta = FgsmMse().run(ctx, "evasion")
x_fgsm_legacy = legacy_fgsm_reference(ctx)
fgsm_max_abs_diff = float(np.max(np.abs(x_fgsm - x_fgsm_legacy)))

np.testing.assert_allclose(
    x_fgsm,
    x_fgsm_legacy,
    rtol=1e-6,
    atol=1e-7,
    err_msg="FGSM differisce dal comportamento legacy a un passo",
)
assert fgsm_meta["iterations_executed"] == 1
assert fgsm_meta["best_iteration"] == 1
np.testing.assert_allclose(
    fgsm_meta["best_loss"],
    fgsm_meta["final_optimization_loss"],
    rtol=1e-6,
    atol=1e-7,
)

fgsm_summary = save_run(
    ctx,
    "fgsm_mse",
    "evasion",
    x_fgsm,
    fgsm_meta,
    errors_before,
    scores_before,
)

saved_fgsm = np.load(
    OUTPUT_DIR / "fgsm_mse" / "evasion" / "x_test_adversarial_scaled.npy"
)
np.testing.assert_array_equal(saved_fgsm, x_fgsm)

# Riepilogo leggibile e report machine-readable.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame([pgd_summary, fgsm_summary]).to_csv(
    OUTPUT_DIR / "summary.csv",
    index=False,
)

report = {
    "all_checks_passed": True,
    "seed": SEED,
    "pgd_iterations": ITERATIONS,
    "epsilon": EPSILON,
    "alpha": ALPHA,
    "protected_column_index": 0,
    "protected_column_name": ctx.sensor_cols[0],
    "pgd_loss_trajectory": losses.tolist(),
    "reported_best_iteration": int(pgd_meta["best_iteration"]),
    "expected_best_iteration": expected_best_iteration,
    "reported_best_loss": float(pgd_meta["best_loss"]),
    "saved_point_loss": saved_point_loss,
    "final_optimization_loss": float(pgd_meta["final_optimization_loss"]),
    "best_loss_le_final_loss": bool(
        pgd_meta["best_loss"] <= pgd_meta["final_optimization_loss"] + 1e-7
    ),
    "max_abs_delta_scaled": float(np.max(np.abs(delta))),
    "max_budget_excess": max_budget_excess,
    "max_abs_delta_outside_mask": max_outside_mask,
    "fgsm_max_abs_diff_from_legacy": fgsm_max_abs_diff,
}

with (OUTPUT_DIR / "smoke_assertions.json").open("w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)

print()
print("============================================================")
print("TUTTI I CONTROLLI SUPERATI")
print("============================================================")
print(f"PGD loss trajectory : {losses.tolist()}")
print(f"PGD best_iteration : {pgd_meta['best_iteration']}")
print(f"PGD best_loss      : {pgd_meta['best_loss']:.10g}")
print(f"PGD final_loss     : {pgd_meta['final_optimization_loss']:.10g}")
print(f"Saved-point loss   : {saved_point_loss:.10g}")
print(f"Max budget excess : {max_budget_excess:.3e}")
print(f"Max outside mask  : {max_outside_mask:.3e}")
print(f"FGSM legacy diff  : {fgsm_max_abs_diff:.3e}")
print(f"Report             : {(OUTPUT_DIR / 'smoke_assertions.json').resolve()}")
print(f"Summary            : {(OUTPUT_DIR / 'summary.csv').resolve()}")
PY

echo "Smoke test completato correttamente."
