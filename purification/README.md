# Diffusion-based adversarial purification for time series

DiffPure (Nie et al., ICML 2022) adapted to the reconstruction/forecasting ICS
detectors of this repository, and evaluated as recommended by Lee & Kim (ICCV 2023,
*Robust Evaluation of Diffusion-Based Adversarial Purification*).

## What the defense does

A diffusion model is trained on **benign** training data only
(`main_train_diffusion.py`). At test time, before a detector scores timestep `t`,
the window of the last `Lw` rows ending at `t` is **purified**:

1. **forward diffusion** to timestep `t*`: `x_t* = sqrt(ab_t*) x + sqrt(1 - ab_t*) z`
   (DiffPure Eq. 3; DDPM linear schedule, T = 1000, the discretisation of DiffPure's
   VP-SDE, so DiffPure's `t* = 0.1` is `t* = 100` here);
2. **reverse denoising** from `t*` to 0 with `K` respaced ancestral DDPM steps (the
   "number of denoising steps" of Lee & Kim; `K = t*` is the full DDPM chain);
3. the **unchanged detector** scores the purified window: AE on its last row,
   sequence models on its last `h` history rows and its last row as observed target.
   The observed target is purified too, so no path from the input to the score
   bypasses the purifier.

`Lw = max(context, h + 1)` (`--context`, default 64): the purifier gets temporal
context even for the AE, whose own input is one row. With `t* = 0` the purifier is
the identity and the defended detector *is* the original detector (checked).

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
| `instances.py` | instance windows and errors of the defended detector |
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
  (validation loss on t <= 300 not below half the structure-free predictor's) or fails
  at the detectors' window lengths.

## Two phases

1. `purification_phase1.sh` (AE seed 7, **tuning** split): diffusion training,
   clean-utility curve over `t*`, adaptive attacks over `t*`, attack-strength checks
   (EOT and iteration saturation, surrogate, BPDA, blind, huge budget) and defense
   variants (K, DDIM). The configuration for phase 2 is chosen here.
2. phase 2 (all detectors, **final** split): undefended vs defended under the
   attacks chosen in phase 1.
