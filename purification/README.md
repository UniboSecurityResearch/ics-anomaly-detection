# Diffusion-based adversarial purification for time series

DiffPure (Nie et al., ICML 2022) adapted to the reconstruction/forecasting ICS
detectors of this repository, and evaluated as recommended by Lee & Kim (ICCV 2023,
*Robust Evaluation of Diffusion-Based Adversarial Purification*).

## What the defense does

A diffusion model is trained on **benign** training data only
(`main_train_diffusion.py`). At test time, before a detector scores timestep `t`,
the window of `Lw` consecutive rows containing every row the detector uses is
**purified**:

1. **forward diffusion** to timestep `t*`: `x_t* = sqrt(ab_t*) x + sqrt(1 - ab_t*) z`
   (DiffPure Eq. 3; DDPM linear schedule, T = 1000, the discretisation of DiffPure's
   VP-SDE, so DiffPure's `t* = 0.1` is `t* = 100` here);
2. **reverse denoising** from `t*` to 0 with `K` respaced ancestral DDPM steps (the
   "number of denoising steps" of Lee & Kim; `K = t*` is the full DDPM chain);
3. the **unchanged detector** scores the purified window: AE on the scored row,
   sequence models on the `h` rows before it and on the scored row as observed
   target. The observed target is purified too, so no path from the input to the
   score bypasses the purifier.

The window is `X[t+d-Lw+1 .. t+d]` with `Lw = 128` (`--context`, the length the
diffusion model is trained on) and a **look-ahead** of `d` rows. With `d = 0` (a
causal purifier) the scored row is the last row of the window, where the denoiser has
no right-hand context: on SWAT-CLEAN validation rows its reconstruction error there
is about 4-5 times the error inside the window, and one to three rows of
context remove almost all the gap. `d` is therefore chosen by a fixed rule on benign
validation rows (`check_denoiser.py`: the smallest `d <= 8` within 20% of the best
reachable error in every noise band; `d = 3` on the development model, stable across
seeds). On SWaT (1 row per second) this delays every decision by `d` seconds; the
repository's own decision rule already uses scores up to `w // 2` steps ahead
(centered window). The causal purifier is calibrated as well, for comparison.
With `t* = 0` the purifier is the identity and the defended detector *is* the
original detector, for any `d` (checked).

Because purification changes the error distribution, the defended detector gets its
own threshold `theta` and window `w`, chosen by **exactly** `main_train.py`'s
procedure (validation part of the training file, grid search on the tuning part of
the test file with the same metric and FPR cap), and is reported on the final part.

## How it is attacked (Lee & Kim's recommendation)

`main_purification.py attack` (implementation in `attack.py`):

* **PGD + EOT**, not AutoAttack (Lee & Kim, Table 2); EOT averages the gradient over
  independent purification draws;
* gradients by **direct back-propagation through the whole defense**: forward noise
  (reparameterisation), every denoising step, the detector -- *not* the adjoint
  method (Lee & Kim, Table 1); a **surrogate** process (fewer steps, same `t*`) as an
  alternative attacker; **BPDA** and a **defense-blind** attacker as weaker
  references;
* the same budget and mask conventions as the undefended benchmark
  (`swat_sweep.sbatch`): L-inf box of a fraction of each feature's training range,
  actuators protected, only rows feeding the loss may change, sign steps of
  `alpha_relative x eps`, random start; evasion loss on the targets the (defended)
  detector flags on clean data (E), false-alarm loss on normal targets;
* the stochastic defense is evaluated with **fresh randomness** over several draws
  (mean +- std); the undefended reference (`t* = 0`) uses the same code.

## Files

| file | content |
|---|---|
| `schedule.py` | DDPM schedule, forward noising, respaced DDPM / DDIM reverse chain |
| `denoiser.py` | 1-D U-Net noise predictor (plain TF, weights in `.npz`) |
| `purifier.py` | `Purifier` (full / BPDA gradients, stateless seeds) |
| `instances.py` | instance windows (context, look-ahead) and errors of the defended detector |
| `check_denoiser.py` | quality gate of the noise predictor, error profile, look-ahead rule |
| `calibration.py` | replica of `main_train.hyperparameter_search` + reproduction check |
| `attack.py` | PGD + EOT runner, evaluation over draws |
| `tests/` | analytic tests of the processes (DiffPure App. B.5) and gradient checks |
| `../main_train_diffusion.py` | training of the noise predictor |
| `../main_purification.py` | `calibrate` / `attack` entrypoint |
| `../purification_report.py` | tables from the outputs |
| `../purification.sbatch`, `../purification_phase1.sh` | cluster jobs |

## Checks built into the code

* `python -m purification.tests.test_schedule`: with the exact noise predictor of
  Gaussian data the chain reproduces its closed-form mean, variance and derivative
  (Monte Carlo), and approximates the true posterior (mean and derivative within 1%;
  DDPM's fixed per-step variance underestimates the posterior variance, as expected);
* `python -m purification.tests.test_defended`: identity defense = attack suite's
  errors and gradients exactly; gradient through the purifier = finite differences;
  BPDA backward = identity;
* `calibrate` exits with code 3 if `t* = 0` does not reproduce the stored theta,
  window and F1;
* `attack` checks that the fast (neighbourhood-only) detection equals the full
  detector on the undefended series and that every perturbation is within budget;
* `main_train_diffusion.py` exits with code 4 if the model has not learnt structure
  (validation loss on t <= 300 not below half the structure-free predictor's);
* `check_denoiser.py` (first job of every phase) re-checks the model on validation
  rows, reports the error profile along the window and chooses the look-ahead.

History: the first phase-1 run (jobs 59934-59936) trained `models/diffusion_swat_v1`
(100k steps, validation loss 2% of the structure-free predictor's) but stopped at a
second criterion that compared the mean loss over 64-row and 128-row windows (limit
1.25, measured 1.31). That ratio measures the share of edge rows in a window, not the
network (the interior loss is the same at every length), so the criterion was removed;
investigating it revealed the right-edge effect that the look-ahead addresses. The
trained model is reused unchanged.

## Selection rule for phase 2 (fixed before phase 1 runs)

All on the **tuning** split, AE seed 7:

1. *utility*: keep the `t*` whose defended tuning F1 is at least the undefended
   tuning F1 minus 0.05 (absolute);
2. *robustness*: among them, the `t*` with the highest worst-case detection rate on E
   under evasion at budget 0.05 (worst case over the full-gradient, surrogate, BPDA
   and blind attackers that were run at that `t*`, otherwise over the full-gradient
   one); ties (within 1 percentage point) go to the smaller `t*`; false-alarm
   robustness is the tie-break after that;
3. *attack strength*: EOT = the smallest value of {1, 5, 10, 20, 40} whose detection
   rate on E is within 1 percentage point of EOT 40; iterations = 200 unless the
   500-iteration run is more than 1 point stronger (then 500);
4. *denoising steps / sampler*: K = 10 DDPM unless K = 5, K = 20 or DDIM is more
   robust by more than 1 point at the same `t*` without violating 1.

If no `t*` > 0 satisfies 1, the result is that the defense costs too much clean
detection on SWaT for this detector, and phase 2 reports that rather than a tuned
defense.

## Two phases

1. `purification_phase1.sh` (AE seed 7, **tuning** split, 512 targets per attack):
   denoiser check, clean-utility curve over `t*` (with and without look-ahead),
   adaptive attacks over `t*` at budgets 0.02 / 0.05 / 0.10, attack-strength checks
   (EOT and iteration saturation, surrogate, BPDA, blind, huge budget) and defense
   variants (K, DDIM), queued in order of priority. The configuration for phase 2 is
   chosen here.
2. phase 2 (all detectors, **final** split): undefended vs defended under the
   attacks chosen in phase 1.
