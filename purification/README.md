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
  actuators protected, sign steps of `alpha_relative x eps`, random start; evasion
  loss on the targets the (defended) detector flags on clean data (E), false-alarm
  loss on normal targets;
* **one threat model for every arm** (`--modifiable detector`, default): the attacker
  may change the rows that feed the UNDEFENDED detector at the loss targets (target
  row, plus history rows for sequence models), whether or not a purifier is present;
  the purifier also reads unmodified context rows. `--modifiable window` lets the
  attacker change the whole purified windows (a stronger attacker that also controls
  the context the defense adds); in the first phase-1_v2 attacks this was the only
  option, which gave the defended arms far more modifiable rows than the undefended
  one -- those results are kept as that stronger-attacker reference;
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
2. *robustness*: among them, the `t*` with the highest worst-case detection rate over
   all attack targets (see the note on E below) under evasion at budget 0.05 (worst case over the full-gradient, surrogate, BPDA
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
1b. `purification_phase1b.sh`: the phase-1_v2 attacks redone with one threat model
   for every arm, the whole-range sanity check redone from the clean series with
   small steps (with steps of 10% of the range and a random start over the whole
   range, sign-PGD failed even on the undefended detector: 0% evasion; from the clean
   series with 1% steps: 96% on the development AE), and two more step sizes.
2. `purification_phase2.sh` (all 18 detectors, **final** split, 512 targets per
   attack, the phase-1b threat model): a short diagnostic on the tuning split (`p0`,
   per-target records), the calibrations (also a noise-only ablation and the causal
   purifier for AE seed 7), then one attack array in order of priority: P1 main
   budget for every detector; P2 attack strength on seed 7 (surrogate, BPDA, blind,
   EOT 1, log-score loss, 3 restarts); P3 mechanism and ablations (actuators free,
   stronger attacker, noise only, causal, whole range); P4 the `t*` curve; P5 step
   0.025 x 500 it.; P6 2048 targets; P7 the other budgets. The attack array is
   queued on hold until `p0` has been checked. `--plan` writes and counts the tasks
   without queueing. Every attack refuses to run (exit 3) if its detector's `t* = 0`
   calibration did not reproduce the stored detector (`--require-reproduction`).

Tools added for phase 2: `--loss log` (PGD on the mean log score, so targets far
above `theta` do not dominate the objective), `--sampler noise` (ablation: the
forward noise of `t*` rescaled to the input, no denoiser), `per_target.json` in every
attack folder (targets, E, per-target detection rates, scores of the scored
timesteps clean and attacked, which of them were modifiable) and
`purification_per_target.py` (decision geometry, per-block results, 95% bootstrap
intervals over blocks, paired differences between arms).
`purification.check_tasks` parses every task line with `main_purification.py`'s own
argument parser before anything is queued (the launcher runs it when TensorFlow is
importable; it always checks that `--out` equals the task folder).

### Outcome of phase 1b (AE seed 7, tuning split, 512 targets)

* All 61 attacks completed. Whole-range check from the clean series (step 0.01,
  actuators free): the undefended detector drops from 0.553 to 0.174 of the targets
  flagged (ASR 0.69): the attack machinery works (it gave 0 in phase 1_v2), though
  sign-PGD is no oracle at that budget.
* Convergence at budget 0.05: steps 0.025 (500 it.), 0.10 and 0.25 agree within
  0.3 points at `t* = 0` and 20 and within 1.1 points at `t* = 50`; EOT 1 / 20 / 40
  and 200 / 500 iterations agree within 0.7 points; full gradient ~ surrogate >
  BPDA >= blind, as expected.
* Rule 1 admits `t* <= 50`. Rule 2 (worst-case share of targets still flagged under
  evasion at budget 0.05): `t* = 0` 0.480; 5 0.408; 10 0.361; 20 0.356; 30 0.356;
  50 0.375. **No `t* > 0` improves evasion robustness: negative result.** Even
  without any attack the defended detector flags fewer targets (0.41-0.42 at
  `t* >= 20`) than the undefended one under attack. Among `t* > 0` the rule picks
  `t* = 5`. Rule 3 gives EOT 1 (within 1 point of EOT 40) and 200 iterations.
* False alarm at budget 0.05 does improve: 0.443 undefended, 0.164 at `t* = 5`,
  about 0.01 at `t* = 20-30`, 0.033 at `t* = 50`; at budget 0.10 the gain mostly
  disappears (0.441 at `t* = 20`, 0.310 at `t* = 50`, 0.737 at `t* = 10`).
* Budget-insensitive floors (`t* = 0`: 0.4805 flagged and 0.4434 false alarms at both
  0.05 and 0.10) suggest structural limits of the threat model (protected actuators;
  a false alarm needs `w` consecutive scores above `theta`, which bounds it by the
  geometry of the target blocks). Checked on the 1b targets: the share of targets
  whose window lies entirely inside the target blocks is 0.4434 (false alarm, w=20;
  0.7363 for w=10) and 0.4805 (evasion, w=20) -- exactly the undefended results.
  For false alarm this is the ceiling (blocks of ~34 rows, w consecutive rows above
  theta needed). For evasion the coincidence is not explained yet (pushing any row of
  a block below theta would also clear interior targets): phase 2 `p0` reruns these
  attacks with per-target records. Phase 2 P3 also runs with actuators free.

Evasion comparisons across `t*` use the detection rate over ALL attack targets
after the attack (`flagged`), not only over E: E is the set each detector flags on
clean data, so it differs between `t*` and DR on E is not comparable across them.
