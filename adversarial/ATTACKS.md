# Attacks and detector

Reference on semantics: what each attack optimizes, how detection is decided, and the
comparison protocol. For operational usage see [README.md](README.md); for every flag see
[PARAMETERS.md](PARAMETERS.md).

## The reference detector (θ + window)

The **operative** detector — the one that decides detection and attack success — is the
paper/repo detector (`main_eval.py`, `detector/*.cached_detect`), **not** the per-feature
`any_j` rule:

1. **per-instance score** = **mean** of the per-feature MSE (`errors.mean(axis=1)`,
   `main_eval.py:298-299`); configurable via `--instance-score` (default `mean_mse`);
2. **θ** = percentile of the per-instance error on the **benign validation set**
   (`np.quantile(val_instance_errors, p)`, `main_eval.py:176`), using the split
   `train_test_split(..., test_size=0.2, random_state=42, shuffle=True)`;
3. **window** `w`: `w` **consecutive** timesteps above θ, replicating each model's
   `cached_detect` (see below), tuned jointly with θ on the `main_eval.py` grid.

Setting θ:

- `--instance-threshold <value>` — the value printed by `main_eval.py` (**preferred**);
- `--instance-threshold-percentile <p>` — recompute θ internally on the *same* validation
  split (`--instance-threshold-ref validation`, default). With `training` it uses a sample
  of the training set (legacy, does not match `main_eval.py`).

If you pass both, the two θ are compared and a divergence beyond `--sanity-tolerance`
triggers a warning.

Detection is computed on the **whole series** and only then restricted to the selected
targets, so `--selection`/`--max-targets` (which fragment the intervals) do not spuriously
reset the window counter at the borders.

### The window per model (mind GRU)

`cached_detect` is **not identical** across models:

| Model | Window rule (`w>1`) | Source |
|---|---|---|
| AE, CNN, LSTM, DNN | `np.convolve(det, ones(w), 'same') // w` (centered window) | `detector/autoencoder.py:160-170` (LSTM's backfill is commented out) |
| **GRU** | `'full'` + backfill `fill[idx-w:idx]=1`, then `Yhat[w-1:]` | `detector/gru.py:241-262` |

The package selects the correct variant from `--model-type`, so for GRU with `w>1` the
flagged region is wider (trailing + backfill).

## Optimization loss vs detector

`--score` selects **only the differentiable loss** optimized by the MSE/CorrShift attacks;
it is **not** the detector. Detection and attack success always go through the scalar
θ+window detector.

- `mean_mse` (default): mean of the per-feature MSE — **aligned** with the detector, the
  natural loss;
- `max_mse`: maximum MSE;
- `threshold_max` / `threshold_smooth`: encode the legacy per-feature
  `any_j(error_j > threshold_j)` rule, kept **only** as a research loss (useful for
  `pgd_margin`/`pgd_cw`); they require per-feature thresholds (`--thresholds` /
  `--threshold-percentile`).

## Attack catalog

| Attack | Type | Optimizes | Extra flags needed | Class |
|---|---|---|---|---|
| `fgsm_mse` | white-box, 1 step | `--score` (step = ε) | — | `FgsmMse(PgdMse)` |
| `pgd_mse` | white-box iterative | `--score` | `--alpha --iterations [--random-start]` | `PgdMse` |
| `pgd_topk` | white-box sparse | `--score`, k features/timestep | `--top-k` | `PgdTopk(PgdMse)` |
| `pgd_margin` | white-box | smooth per-feature threshold margin | `--thresholds`/`--threshold-percentile` | `PgdMargin` |
| `pgd_cw` | white-box | CW hinge on the margin | thresholds + `--cw-confidence --cw-l2-weight` | `PgdCw` |
| `pgd_kl` | white-box (experimental) | residual KL + score | `--kl-reference auto --kl-temperature --kl-score-weight` | `PgdKl` |
| `corrshift` | **black-box** | greedy correlation-guided search | `--corrshift-rounds --corrshift-step …` | `CorrShift` |

The white-box attacks share the PGD loop in `base.WhiteBoxAttack.run`; each class only
overrides `objective` (and, where needed, `process_gradient`/`iterations`/
`use_random_start`/`step_multiplier`).

## Goal and eligibility

```text
evasion:      eligible = detected_before_window
              success  = detected_before_window AND NOT detected_after_window
false_alarm:  eligible = NOT detected_before_window
              success  = NOT detected_before_window AND detected_after_window
attack_success_rate = successful_targets / eligible_targets
```

Use `--selection attack` for evasion (targets = attack timesteps) and `--selection normal`
for false alarm (targets = benign timesteps). Without θ (`--instance-threshold*`) the
detection/success metrics are disabled and only the score change is reported.

## Temporal alignment

- **AE**: point reconstruction → `--target-offset` irrelevant (input = target = `X[t]`);
- **CNN/GRU/LSTM**: `--target-offset 1` (default) reproduces the repo alignment
  (input `X[lead-h:lead]`, target `X[lead+1]`, `utils.py:79-80`); `--history` = training
  value;
- **DNN** ⚠️: the repo trains DNN as a **1-step forecaster** and `main_eval.py` scores it
  through the *sequence* branch, whereas this package treats it as a *point reconstruction*
  model → θ for DNN may not reproduce `main_eval.py`. A warning is printed: run
  `--sanity-check` and, if it diverges, treat DNN as a sequence model.

## Correct comparison protocol

Keep constant: model/checkpoint, target indices, dataset/scaler, scope, per-feature ε,
protected features, domain clipping, θ + window, seed. **Vary only the attack method** (use
`--attack all` in a single invocation). Report both success and cost (runtime, queries for
CorrShift, L∞/L2, changed cells).
