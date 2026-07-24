# Concealment / evasion attack suite

Concealment attacks against reconstruction-based ICS anomaly detectors, following
Erba et al., *"Constrained Concealment Attacks against Reconstruction-Based Anomaly
Detectors in Industrial Control Systems"* and *"On the Practical Realization of Evasion
Attacks for Industrial Control Systems"*.

Two interchangeable attacks:

- **generic replay** — baseline that substitutes anomalous observations with normal ones;
- **black-box autoencoder concealment** — an attacker-owned autoencoder trained only on
  normal data, in `unconstrained` / `partially_constrained` / `fully_constrained` modes.

Both are **strictly black-box**: they never use the detector's weights, gradients,
threshold, anomaly scores or labels. The detector is used **only** to evaluate whether
concealment succeeded.

## Threat model

The attack receives samples from an already-compromised process, `x_anom`, and produces
`x_adv` to feed the detector in their place. It always keeps, separately: the original
`x_anom`, the manipulated `x_adv`, the controllable-feature mask `m`, the actual
perturbation, and the detector's decision before/after (`AttackResult`).

## Layout

```text
ics-anomaly-detection/
├── main_concealment.py          # entrypoint (run from the ROOT)
└── concealment/
    ├── base.py                  # ConcealmentAttack (fit/transform/fit_transform) + AttackResult
    ├── replay.py                # ReplayAttack: fixed | cyclic | random | nearest_context | offline_oracle
    ├── autoencoder_attack.py    # BlackBoxAutoencoderAttack (unconstrained/partial/fully) + AutoencoderConfig
    ├── masks.py                 # controllable subset C (names/indices/percentage/k/random/file)
    ├── postprocess.py           # domain constraints (binary/discrete/continuous) from TRAIN or config
    ├── detector_eval.py         # victim-detector evaluation (reuses adversarial/: theta, cached_detect)
    ├── metrics.py               # ASR, recall, L0/L1/L2/Linf, timings, constraint violation
    ├── data.py                  # attacker normal set + test targets + leakage guards
    ├── config.py                # optional YAML/JSON config
    ├── cli.py                   # parse_args
    └── runner.py                # run one config + the 6-config comparison
```

It **reuses** `adversarial/` (model/scaler/data loading, per-instance errors, θ from the
validation split, `cached_detect`, label inference) and `data_loader`; it does **not**
modify the detector, preprocessing or the experimental pipeline.

## Data leakage (enforced)

- the attacker AE / replay buffer are built **only** from standardized **training** data
  (`load_attacker_normal`); `assert_no_leakage` guards against accidentally passing the
  test set;
- domain bounds for post-processing come from **training** (or explicit config), never
  from the test set;
- non-controllable features are asserted byte-for-byte unchanged (`assert_mask_preserved`)
  — an attack that violates this is reported `valid=False` and not counted successful.

## Configurations

| Constraint | Observes | Modifies | Vector |
|---|---|---|---|
| `unconstrained` | all features | all (`m = 1`) | `x_adv = G_A(x_anom)` |
| `partially_constrained` | all features (context) | only C | `x_adv = m ⊙ G_A(x_anom) + (1-m) ⊙ x_anom` |
| `fully_constrained` | only C | only C | `x_adv_C = G_A^C(x_anom_C)`, rest = `x_anom` |

The controllable subset C is selected with `--controlled-features` (names or indices),
`--controlled-percentage`, `--controlled-k` (optionally `--controlled-random`) or
`--controlled-file`.

## Usage

Single configuration:

```bash
python main_concealment.py \
  --dataset BATADAL --model-type AE --model-path models/results/AE-BATADAL-l5-cf2.0.h5 \
  --instance-threshold-percentile 0.95 --detection-window 3 \
  --attack autoencoder --constraint partially_constrained \
  --controlled-features PRESSURE_T7,FLOW_PU10,FLOW_PU11 \
  --postprocess --output-dir outputs/concealment/ae_partial
```

Generic replay baseline (constrained, same mask as the AE attack):

```bash
python main_concealment.py \
  --dataset BATADAL --model-type AE --model-path models/results/AE-BATADAL-l5-cf2.0.h5 \
  --instance-threshold-percentile 0.95 --detection-window 3 \
  --attack replay --replay-strategy cyclic --constraint partially_constrained \
  --controlled-k 5 --output-dir outputs/concealment/replay_k5
```

Full comparison table (the 6 required configurations, sweeping |C|):

```bash
python main_concealment.py \
  --dataset BATADAL --model-type AE --model-path models/results/AE-BATADAL-l5-cf2.0.h5 \
  --instance-threshold-percentile 0.95 --detection-window 3 \
  --compare --controlled-k-list 1 2 5 10 \
  --postprocess --output-dir outputs/concealment/compare
```

`--compare` runs: (1) no attack, (2) replay unconstrained, (3) replay constrained,
(4) autoencoder unconstrained, (5) autoencoder partially constrained, (6) autoencoder
fully constrained — reusing the same test set, anomalous targets, feature masks and seed
— and writes `comparison.csv` (one row per configuration).

Attacker autoencoder architecture is fully configurable (`--ae-hidden` or
`--ae-layers`/`--ae-compression`/`--ae-latent`, `--ae-activation`, `--ae-lr`,
`--ae-batch-size`, `--ae-epochs`, `--ae-patience`, `--ae-val-split`, `--ae-loss`,
`--ae-device`) or via a YAML/JSON `--config`:

```yaml
attack:
  type: blackbox_autoencoder
  constraint: partially_constrained
  controlled_features: [PRESSURE_T7, FLOW_PU10, FLOW_PU11]
  autoencoder: {layers: 4, compression: 2.0, activation: relu, epochs: 100, patience: 5}
```

## Evaluation

For each configuration `comparison.csv` reports: `attack_success_rate`
(`ASR = |{t: D(x_anom)=1 ∧ D(x_adv)=0}| / |{t: D(x_anom)=1}|`), `recall_before/after`,
`detected_before`, `evaded`, `constraint_violation` + `valid`, `n_controlled`,
`changed_features`, `l0/l1/l2/linf`, `attacker_train_time_s`, `gen_time_per_sample_ms`.

Detection uses the paper's scalar θ+window detector (θ from `--instance-threshold` or
reproduced from the validation split with `--instance-threshold-percentile`; window per
model's `cached_detect`) — reused from `adversarial/`.

## Amount of attacker-observed benign data

How much benign data the attacker can eavesdrop is an experimental variable: the more
normal data the AE (and replay buffer) see, the better they project a malicious input
onto the benign manifold, and the stronger the concealment. Control it with
`--attacker-train-size N` (or `--attacker-train-fraction f`); default is **all** training
data. `--attacker-train-random` samples a seeded subset instead of a contiguous
eavesdropping prefix. Every row of `comparison.csv` records `attacker_train_samples`, so
you can ablate concealment quality vs. observation budget, e.g.:

```bash
for N in 1000 5000 20000 48106; do
  python main_concealment.py --dataset BATADAL --model-type AE \
    --model-path models/results/AE-BATADAL-l5-cf2.0.h5 \
    --instance-threshold-percentile 0.95 --detection-window 3 \
    --attack autoencoder --constraint unconstrained --attacker-train-size $N \
    --output-dir outputs/concealment/ablate_N$N
done
```

## Windowing

The attacker autoencoder is **feed-forward on per-timestep vectors** (as in the reference
work), reconstructing each standardized 43-dim sample independently. The concealed series
is fed to the detector, which applies its own windowing (`--detection-window`) internally,
so temporal coherence is preserved without imposing a temporal model on the attacker.
