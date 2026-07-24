# Adversarial attack suite (modular)

White-box and black-box adversarial attacks against the reconstruction/prediction-based
ICS anomaly detectors of
[`pwwl/ics-anomaly-detection`](https://github.com/pwwl/ics-anomaly-detection)
(ESORICS 2022 paper).

Detection and attack success **always** use the paper's scalar instance-level detector
(threshold `θ` on the **mean** per-feature MSE plus a consecutive `w`-window, reproduced
from `main_eval.py`), **never** the legacy per-feature `any_j` rule. See
[ATTACKS.md](ATTACKS.md) for the semantics and [PARAMETERS.md](PARAMETERS.md) for every
flag.

## Layout

```text
ics-anomaly-detection/
├── main_adversarial.py          # entrypoint (like main_eval.py) — run from the ROOT
└── adversarial/                 # the package (like detector/)
    ├── base.py                  # Attack / WhiteBoxAttack (shared PGD loop)
    ├── attacks/                 # one implementation per attack + the ATTACKS registry
    │   ├── fgsm_mse.py  pgd_mse.py  pgd_topk.py
    │   ├── pgd_margin.py  pgd_cw.py  pgd_kl.py  corrshift.py
    │   └── __init__.py          # ATTACKS = {name: class}
    ├── context.py               # ModelAdapter + AttackContext
    ├── io_utils.py              # model / scaler / training-data / vector loading
    ├── targets.py               # labels, target selection, modification masks
    ├── errors.py                # per-feature errors + (differentiable) detector scores
    ├── projection.py            # L-inf/domain projection + top-k gradient sparsification
    ├── detector.py              # scalar theta+window detector, threshold/KL calibration
    ├── evaluation.py            # evaluate_series, save_run, run_sanity_check
    ├── build.py                 # build_context (assembles everything from CLI args)
    └── cli.py                   # parse_args
```

## Environment

Run it **from the repository root** and use the same environment you trained the models
with (TensorFlow, scikit-learn, numpy, pandas). The package imports `data_loader` and uses
the relative paths `models/…` and `data/…`.

```bash
cd /path/to/ics-anomaly-detection
source venv/bin/activate
python -c "import tensorflow as tf; print(tf.__version__)"
```

## Mental model (read this first)

There are **two distinct objects**:

| | What it is | Controlled by |
|---|---|---|
| **Optimization loss** | the differentiable function the attack pushes | `--score` (+ attack-specific params) |
| **Operative detector** | decides detected / not-detected → success, detection rate | `--instance-threshold[-percentile]` + `--detection-window` |

An attack succeeds when it **flips the detector's decision**, not when it lowers the loss.

## The 3 steps

**0) Get the official θ / percentile / window** (once per model):

```bash
python main_eval.py AE BATADAL --run_name results \
    --ae_model_params_layers 5 --ae_model_params_cf 2.0 \
    --detect_params_metrics F1 \
    --detect_params_percentile 0.95 0.99 0.995 \
    --detect_params_windows 1 3 5 10
# note the printed "Best ... percentile=..., window=..."
```

**1) Mandatory sanity check** (reproduces the detector and compares it to main_eval):

```bash
python main_adversarial.py \
    --dataset BATADAL --model-type AE \
    --model-path models/results/AE-BATADAL-l5-cf2.0.h5 \
    --instance-threshold-percentile 0.95 --detection-window 3 \
    --sanity-check
```

If the benign/attack detection rates do **not** match `main_eval.py`, stop and investigate
(wrong model/scaler, different percentile/window, wrong `--target-offset`).

**2) Attack** — anatomy of the command:

```bash
python main_adversarial.py \
  --dataset BATADAL --model-type AE --model-path models/results/AE-BATADAL-l5-cf2.0.h5 \
  --attack pgd_mse --goal evasion \
  --selection attack --max-targets 2048 --seed 42 \
  --instance-threshold-percentile 0.95 --detection-window 3 --instance-score mean_mse \
  --score mean_mse \
  --epsilon 0.10 --alpha 0.02 --iterations 30 --random-start \
  --output-dir outputs/adv/AE_pgd_evasion --save-series delta
```

**Sequence** models require `--history <training-value>` (from the model name, e.g.
`CNN-…-hist200` → `--history 200`; GRU/LSTM default 100) and `--target-offset 1`.

## Full benchmark + aggregation

One invocation per (model, goal) with `--attack all` → all attacks share identical
targets / θ+window / budget (only the method varies). Evasion on attack timesteps, false
alarm on normal timesteps:

```bash
# EVASION
python main_adversarial.py --dataset BATADAL --model-type AE \
    --model-path models/results/AE-BATADAL-l5-cf2.0.h5 \
    --attack all --goal evasion --selection attack --max-targets 2048 --seed 42 \
    --instance-threshold-percentile 0.95 --detection-window 3 --instance-score mean_mse \
    --score mean_mse --threshold-percentile 0.995 \
    --epsilon 0.10 --alpha 0.02 --iterations 30 --random-start --top-k 5 \
    --cw-confidence 0.0 --kl-reference auto --corrshift-rounds 3 --corrshift-step 0.02 \
    --output-dir outputs/adv/BATADAL_AE_evasion

# FALSE ALARM
python main_adversarial.py --dataset BATADAL --model-type AE \
    --model-path models/results/AE-BATADAL-l5-cf2.0.h5 \
    --attack all --goal false_alarm --selection normal --max-targets 2048 --seed 42 \
    --instance-threshold-percentile 0.95 --detection-window 3 --instance-score mean_mse \
    --score mean_mse --threshold-percentile 0.995 \
    --epsilon 0.10 --alpha 0.02 --iterations 30 --random-start --top-k 5 \
    --kl-reference auto --corrshift-rounds 3 --corrshift-step 0.02 \
    --output-dir outputs/adv/BATADAL_AE_false_alarm
```

`--threshold-percentile 0.995` provides the **per-feature** thresholds that `pgd_margin`/
`pgd_cw` need as a loss (NOT the detector). Repeat for LSTM/GRU/CNN, adding `--history`.

Each run writes a `summary.csv`; merge them into an `attack × model × goal` table:

```python
import glob, os, pandas as pd
rows = []
for f in glob.glob("outputs/adv/*/summary.csv"):
    df = pd.read_csv(f); df["run"] = os.path.basename(os.path.dirname(f)); rows.append(df)
table = pd.concat(rows, ignore_index=True)
cols = ["run","attack","goal","attack_success_rate","detection_rate_before",
        "detection_rate_after","linf_scaled","l2_scaled","changed_cells","runtime_seconds",
        "instance_threshold","detection_window","instance_score_kind"]
table = table[[c for c in cols if c in table.columns]]
table.to_csv("outputs/adv/master_table.csv", index=False)
print(table.to_string(index=False))
```

## Outputs

Under `--output-dir/`:

- `summary.csv` — one row per (attack, goal): `attack_success_rate`,
  `detection_rate_before/after`, `linf_scaled`, `l2_scaled`, `changed_cells`, detector
  provenance (`instance_threshold`, `detection_window`, `detection_percentile`,
  `instance_score_kind`, `optimization_score`), `runtime_seconds`, `query_count`
  (CorrShift);
- `<attack>/<goal>/attack_scores.csv` — one row per target (`score_before/after`,
  `detected_before/after_window`, `eligible`, `attack_success`);
- `<attack>/<goal>/{delta_scaled,x_test_adversarial_scaled}.npy`,
  `feature_errors_before/after.npy`, `attack_config.json`.

## Extending: add an attack

1. create `adversarial/attacks/my_attack.py`;
2. subclass `WhiteBoxAttack` (define `objective`, and optionally
   `process_gradient`/`iterations`/`use_random_start`/`step_multiplier`) or `Attack`
   (for black-box, define `run`); set `name` and the class flags
   (`requires_thresholds`, `requires_gradients`, `requires_train_data`);
3. register it in `adversarial/attacks/__init__.py`.

No `if/elif` dispatch to touch anywhere else.
