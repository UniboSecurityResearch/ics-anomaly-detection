# Parameter reference (`main_adversarial.py`)

All CLI flags, grouped as in `adversarial/cli.py`. Defaults in parentheses. See
[README.md](README.md) for the operational workflow and [ATTACKS.md](ATTACKS.md) for the
detector/attack semantics.

## Repository, model and data

| Flag | Description |
|---|---|
| `--dataset` (req.) | `BATADAL` / `SWAT` / `SWAT-CLEAN` / `WADI` / `WADI-CLEAN`. |
| `--model-type` (req.) | `AE` / `DNN` (point) or `CNN` / `GRU` / `LSTM` (sequence). |
| `--model-path` (req.) | Keras `.h5`/`.keras`/SavedModel, or a pickle exposing `predict()`. |
| `--history` (`None`) | History length for CNN/GRU/LSTM; must match training. Ignored for AE/DNN. |
| `--target-offset` (`1`) | Alignment: input `X[i-h:i]`, target `X[i+1]`. Use `0` only for a locally trained `X[i-h:i] -> X[i]` model. AE: irrelevant. |
| `--output-dir` (`outputs/adversarial_suite`) | Output directory. |
| `--seed` (`42`) | numpy/TensorFlow seed. |
| `--prediction-batch-size` (`4096`) | Batch size for model evaluation and CorrShift queries. |

## Target selection

| Flag | Description |
|---|---|
| `--selection` (`all`) | `all` / `attack` / `normal`, from the test labels. Mutually exclusive with `--start`. |
| `--start` / `--end` (`None`) | Target timestep interval `[start, end)` (both or neither). |
| `--max-targets` (`2048`) | Maximum number of targets; `<=0` disables the limit. |

## Attack configuration

| Flag | Description |
|---|---|
| `--attack` (`pgd_mse`) | `fgsm_mse`/`pgd_mse`/`pgd_topk`/`pgd_margin`/`pgd_cw`/`pgd_kl`/`corrshift` or `all`. |
| `--goal` (`evasion`) | `evasion` / `false_alarm` / `both`. |
| `--scope` (`both`) | `target` / `history` / `both` — what may be modified (sequence only; point = target). |
| `--score` (`mean_mse`) | **Optimization loss**, NOT the detector. `mean_mse` (aligned with the detector) / `max_mse` / `threshold_max` / `threshold_smooth` (the latter two = per-feature `any_j` rule, loss only). |
| `--margin-beta` (`20.0`) | Smooth-max β for `threshold_smooth` and the margin attacks. |

## Perturbation constraints

| Flag | Description |
|---|---|
| `--epsilon` (`0.10`) | L∞ budget in standardized units (StandardScaler). |
| `--epsilon-raw-file` (`None`) | Per-feature budget in physical units (`.npy/.csv/.json`); converted via `scaler.scale_`, overrides `--epsilon`. |
| `--alpha` (`0.01`) | PGD step size. |
| `--iterations` (`20`) | PGD iterations (FGSM uses 1). |
| `--random-start` | Random init inside the ε-box (ignored by FGSM). |
| `--top-k` (`5`) | `pgd_topk`: features modified per timestep/iteration. |
| `--protected-cols` (`""`) | Comma-separated feature names or 0-based indices that must NOT be modified. |
| `--clip-train-range` | Clip each feature to the standardized benign-training min/max. |

## Threshold and decision boundary

| Flag | Description |
|---|---|
| `--thresholds` (`None`) | Per-feature squared-error threshold vector (`.npy/.csv/.json`). Loss only, for `pgd_margin`/`pgd_cw`. |
| `--threshold-percentile` (`None`) | Estimate the **per-feature** thresholds from benign training errors (`0.995` or `99.5`). Loss only, not the detector. |
| `--reference-max-targets` (`10000`) | Max benign training targets used for threshold/KL estimation. |
| `--detection-window` (`1`) | Consecutive window `w` of the detector (`cached_detect`). Pass the `window` tuned with θ by `main_eval.py`. |
| `--instance-threshold` (`None`) | **Scalar θ** printed by `main_eval.py` (preferred). Takes precedence over the percentile; if both are given, they are compared with a warning. |
| `--instance-threshold-percentile` (`None`) | Recompute θ = `quantile(val_instance_errors, p)` on `main_eval.py`'s validation split (`0.95` or `95`). |
| `--instance-threshold-ref` (`validation`) | `validation` (faithful to `main_eval.py`) or `training` (legacy, training sample). |
| `--instance-score` (`mean_mse`) | Per-instance aggregation for the decision and for θ: `mean_mse` (repo) or `max_mse`. |

## Sanity check / reproducibility

| Flag | Description |
|---|---|
| `--sanity-check` | Reproduce the `main_eval.py` detector, print θ and the benign/attack detection rates on the whole series, then **exit**. Run before benchmarking. Requires `--instance-threshold[-percentile]`. |
| `--sanity-tolerance` (`0.05`) | Relative tolerance for the divergence warning between `--instance-threshold` and the reproduced θ. |

## CW-style attack

| Flag | Description |
|---|---|
| `--cw-confidence` (`0.0`) | Required margin beyond the detector boundary. |
| `--cw-l2-weight` (`0.0`) | L2 regularization weight in the CW loss. |

## Residual-KL attack (experimental)

| Flag | Description |
|---|---|
| `--kl-reference` (`auto`) | `auto` estimates a benign residual profile from training; otherwise a positive per-feature vector. |
| `--kl-temperature` (`1.0`) | Softmax temperature over the residuals. |
| `--kl-score-weight` (`1.0`) | Weight of the detector-score alignment added to the KL objective. |

## CorrShift black-box attack

| Flag | Description |
|---|---|
| `--corrshift-rounds` (`1`) | Greedy correlation-guided search rounds. |
| `--corrshift-step` (`None`) | Standardized step per round; defaults to `--alpha`. |
| `--corrshift-top-k` (`0`) | Keep only the k strongest anchor correlations; `0` = all. |
| `--corrshift-min-abs-corr` (`0.0`) | Ignore correlations with smaller absolute value. |
| `--corrshift-range-fraction` (`None`) | Alternative budget: fraction of the benign standardized range; overrides ε for CorrShift only. |

## Saved outputs

| Flag | Description |
|---|---|
| `--save-series` (`full`) | `full` (complete adversarial series) / `delta` (perturbation only) / `none`. |
| `--save-raw-csv` | Also save the adversarial series in physical units as CSV. |

## `summary.csv` columns

`attack`, `goal`, `targets`, `mean_score_before/after/change`, `linf_scaled`, `l2_scaled`,
`mean_abs_delta_scaled`, `changed_cells/rows/features`, `eligible_targets`,
`successful_targets`, `attack_success_rate`, `detection_rate_before/after`,
`instance_threshold`, `instance_threshold_source`, `detection_percentile`,
`detection_window`, `instance_score_kind`, `optimization_score`, `runtime_seconds`,
`query_count`.
