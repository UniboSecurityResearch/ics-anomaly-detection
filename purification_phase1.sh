#!/bin/bash
# Diffusion purification, PHASE 1 (second run, October 2026): study the defense on ONE
# detector (AE, seed 7) on the TUNING part of the test file (rows before 134975), so
# that the configuration chosen here (t*, denoising steps, EOT, iterations) is never
# chosen on the final split used for the reported results (phase 2).
#
#   bash purification_phase1.sh            # checks the code state, then queues every job
#   bash purification_phase1.sh --report   # any time (also with jobs still running):
#                                          # tables of what has finished + one archive
#
# The first run (outputs/purification/phase1, jobs 59934-59936) trained the diffusion
# model models/diffusion_swat_v1 (100k steps, loss 2% of the structure-free
# predictor's) but its length check, badly designed, stopped the chain; the model is
# REUSED here unchanged (its sha256 is recorded in commit.txt).
#
# Jobs (partition l40s, at most 3 running at a time):
#   1. check       purification/check_denoiser.py on benign validation rows: quality
#                  gate and the purifier look-ahead d (exit 4 -> everything cancelled)
#   2. calibrate   AE s7, purified windows of 128 rows (the diffusion training length):
#                  t* = 0 (must reproduce the stored detector, else exit 3), 5, 10, 20,
#                  30, 50, 100, 150, 200, 300 with K = 10 DDPM steps and look-ahead d;
#                  the same with look-ahead 0 (causal purifier, for comparison);
#                  K = 5 / 20 and DDIM K = 10 at t* = 20, 50, 100
#   3. attacks     PGD, step 0.10 eps, random start, 512 targets (contiguous blocks over
#                  every attack of the tuning split), 10 evaluation draws, actuators
#                  protected; array order = priority, so the first results to arrive are
#                  the ones the decisions need:
#                  P1  budget 0.05, full-gradient PGD+EOT(20), 200 it., evasion and
#                      false alarm, t* = 0 (undefended reference), 10, 20, 50, 100, 200
#                  P2  attack-strength checks (t* = 50, budget 0.05 unless stated):
#                      whole range with actuators free at t* = 0 and 50 (an attack that
#                      cannot win there signals gradient masking); EOT 1/5/10/40;
#                      500 iterations; surrogate process with 2 / 5 steps (t* 50, 100);
#                      BPDA and defense-blind attacker (t* 20, 50, 100)
#                  P3  budgets 0.02 and 0.10, as P1
#                  P4  defense variants at t* = 50: K = 5, K = 20, DDIM
# Nothing is overwritten: every task writes its own folder under
# outputs/purification/phase1_v2 and is skipped if its DONE marker exists.
set -euo pipefail
REPO=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
cd "$REPO"
P1=outputs/purification/phase1_v2
DIFF=models/diffusion_swat_v1
MODEL=models/swatchrono_ae_s7

if [[ "${1:-}" == "--report" ]]; then
    python3 purification_report.py "$P1" | tee "$P1/report.txt"
    if [[ -f "$P1/jobs.txt" ]]; then
        IDS=$(awk '{print $2}' "$P1/jobs.txt" | paste -sd, -)
        sacct -j "$IDS" -P --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS > "$P1/sacct.txt" || true
        echo "== job states"; awk -F'|' 'NR>1 && $1 !~ /\./ {print $3}' "$P1/sacct.txt" | sort | uniq -c
        mkdir -p "$P1/failed_logs"
        for j in $(awk -F'|' 'NR>1 && $1 ~ /_[0-9]+$/ && $3 != "COMPLETED" && $3 != "RUNNING" && $3 != "PENDING" {print $1}' "$P1/sacct.txt"); do
            cp "slurm.purif.$j.out" "slurm.purif.$j.err" "$P1/failed_logs/" 2>/dev/null || true
        done
    fi
    cp "$DIFF/config.json" "$P1/diffusion_config.json" 2>/dev/null || true
    cp "$DIFF/train_log.csv" "$P1/diffusion_train_log.csv" 2>/dev/null || true
    tar czf "$HOME/purification_phase1_v2.tgz" --exclude='*.npz' -C outputs/purification phase1_v2
    echo "== archive: $HOME/purification_phase1_v2.tgz ($(du -h "$HOME/purification_phase1_v2.tgz" | cut -f1))"
    exit 0
fi

# --- checks -----------------------------------------------------------------
git fetch -q origin
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]]; then
    echo "ERROR: HEAD is not origin/main. Pull (or push) first."; exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked files have uncommitted changes:"; git status --short | grep -v '^??'; exit 1
fi
if ! ls "$MODEL"/*.h5 >/dev/null 2>&1; then echo "ERROR: no model in $MODEL"; exit 1; fi
if [[ ! -f "$DIFF/weights.npz" || ! -f "$DIFF/config.json" ]]; then
    echo "ERROR: $DIFF is not a trained diffusion model"; exit 1
fi
if [[ -n "$(squeue -u "$USER" -h)" ]]; then echo "ERROR: you still have jobs in the queue."; exit 1; fi
if [[ -e "$P1/jobs.txt" ]]; then echo "ERROR: $P1/jobs.txt exists: this phase was already queued."; exit 1; fi
mkdir -p "$P1"
{
    echo "commit $(git rev-parse --short HEAD), $(date -Is)"
    echo "diffusion $DIFF weights.npz sha256 $(sha256sum "$DIFF/weights.npz" | cut -d' ' -f1)"
} > "$P1/commit.txt"

# --- task files ----------------------------------------------------------------
TD="$P1/tasks"; mkdir -p "$TD"
DIGESTS="8e2a2c4d5bec839b70ab49c06fe4e86c859cd9d72cfb2e07c5465322925da343 a46e7134e5b3895d84205a72d0380f12d4c5cb7bca97651716998cf7744ded09 456181b4a539a65f3d9ad8f1be3b15dd348acabf6b69f186c1385a6a36e3cc3c"
CHECK=$P1/denoiser_check/check.json
echo "$P1/denoiser_check -m purification.check_denoiser --diffusion $DIFF --out $CHECK --expect-data $DIGESTS" > "$TD/check.txt"

CAL=$P1/calib
COMMON="--dataset SWAT-CLEAN --model-dir $MODEL --diffusion $DIFF --context 128"
TSTARS="0 5 10 20 30 50 100 150 200 300"
# main calibration on its own (the attacks need it); the others are comparisons
echo "$CAL/k10_ddpm main_purification.py calibrate $COMMON --lookahead $CHECK --steps 10 --sampler ddpm --t-star $TSTARS --draws 1 --out $CAL/k10_ddpm" > "$TD/calibrate_main.txt"
{
    echo "$CAL/k10_ddpm_causal main_purification.py calibrate $COMMON --lookahead 0 --steps 10 --sampler ddpm --t-star $TSTARS --draws 1 --out $CAL/k10_ddpm_causal"
    echo "$CAL/k05_ddpm main_purification.py calibrate $COMMON --lookahead $CHECK --steps 5 --sampler ddpm --t-star 20 50 100 --draws 1 --out $CAL/k05_ddpm"
    echo "$CAL/k20_ddpm main_purification.py calibrate $COMMON --lookahead $CHECK --steps 20 --sampler ddpm --t-star 20 50 100 --draws 1 --out $CAL/k20_ddpm"
    echo "$CAL/k10_ddim main_purification.py calibrate $COMMON --lookahead $CHECK --steps 10 --sampler ddim --t-star 20 50 100 --draws 1 --out $CAL/k10_ddim"
} > "$TD/calibrate.txt"

defense() {  # $1 = calib folder, $2 = t*, $3 = K, $4 = sampler -> defense json path
    local k=$3; if (( $2 == 0 )); then k=0; elif (( $2 < k )); then k=$2; fi
    printf "%s/defense_tstar%04d_k%03d_%s.json" "$1" "$2" "$k" "$4"
}
ATT=$P1/attacks
BASE="--dataset SWAT-CLEAN --model-dir $MODEL --split tune --max-targets 512 --alpha-relative 0.10 --eval-draws 10 --trace-every 10 --seed 42"
main_grid() {  # $1 = budget
    local B=$1 T G O
    for T in 0 10 20 50 100 200; do
        for G in evasion false_alarm; do
            O=$ATT/range$B/full_eot20_it200/t$T/$G
            echo "$O main_purification.py attack $BASE --epsilon-range-fraction $B --protect-actuators --defense $(defense $CAL/k10_ddpm $T 10 ddpm) --goal $G --gradient full --eot 20 --iterations 200 --out $O"
        done
    done
}
{
    # P1
    main_grid 0.05
    # P2 attack-strength checks
    B=0.05
    CHK="$BASE --epsilon-range-fraction $B --protect-actuators"
    for T in 0 50; do
        O=$ATT/range1.0_free/full_eot20_it200/t$T/evasion
        echo "$O main_purification.py attack $BASE --epsilon-range-fraction 1.0 --defense $(defense $CAL/k10_ddpm $T 10 ddpm) --goal evasion --gradient full --eot 20 --iterations 200 --out $O"
    done
    for E in 1 5 10 40; do
        O=$ATT/range$B/full_eot${E}_it200/t50/evasion
        echo "$O main_purification.py attack $CHK --defense $(defense $CAL/k10_ddpm 50 10 ddpm) --goal evasion --gradient full --eot $E --iterations 200 --out $O"
    done
    O=$ATT/range$B/full_eot20_it500/t50/evasion
    echo "$O main_purification.py attack $CHK --defense $(defense $CAL/k10_ddpm 50 10 ddpm) --goal evasion --gradient full --eot 20 --iterations 500 --out $O"
    for T in 50 100; do
        for KA in 2 5; do
            O=$ATT/range$B/surrogate_k${KA}_eot20_it200/t$T/evasion
            echo "$O main_purification.py attack $CHK --defense $(defense $CAL/k10_ddpm $T 10 ddpm) --goal evasion --gradient surrogate --attack-steps $KA --eot 20 --iterations 200 --out $O"
        done
    done
    for T in 20 50 100; do
        for GR in bpda blind; do
            for G in evasion false_alarm; do
                O=$ATT/range$B/${GR}_it200/t$T/$G
                echo "$O main_purification.py attack $CHK --defense $(defense $CAL/k10_ddpm $T 10 ddpm) --goal $G --gradient $GR --eot 20 --iterations 200 --out $O"
            done
        done
    done
    # P3 other budgets
    main_grid 0.02
    main_grid 0.10
    # P4 defense variants at t* = 50
    for CFG in "k05_ddpm 5 ddpm" "k20_ddpm 20 ddpm" "k10_ddim 10 ddim"; do
        read -r C K S <<< "$CFG"
        O=$ATT/range$B/full_eot20_it200_defense_$C/t50/evasion
        echo "$O main_purification.py attack $CHK --defense $(defense $CAL/$C 50 $K $S) --goal evasion --gradient full --eot 20 --iterations 200 --out $O"
    done
} > "$TD/attacks.txt"

N_CAL=$(wc -l < "$TD/calibrate.txt")
N_ATT=$(wc -l < "$TD/attacks.txt")

submit() {  # $1 = label, rest = sbatch arguments; prints the job id
    local label="$1"; shift
    local id
    id=$(sbatch --parsable "$@")
    id="${id%%;*}"
    echo "$label $id" >> "$P1/jobs.txt"
    echo "queued  $label  job $id" >&2
    echo "$id"
}
K=$(submit check --array=0 --time=01:00:00 --mail-type=END,FAIL \
    --export=ALL,TASKS="$REPO/$TD/check.txt" purification.sbatch)
CM=$(submit calib_main --array=0 --time=12:00:00 --mail-type=END,FAIL \
    --dependency=afterok:"$K" --kill-on-invalid-dep=yes \
    --export=ALL,TASKS="$REPO/$TD/calibrate_main.txt" purification.sbatch)
CR=$(submit calib_other --array=0-$((N_CAL - 1))%2 --time=12:00:00 --mail-type=END,FAIL \
    --dependency=afterok:"$K" --kill-on-invalid-dep=yes \
    --export=ALL,TASKS="$REPO/$TD/calibrate.txt" purification.sbatch)
# the attacks need the main calibration to succeed (t* = 0 reproduction check) and
# wait for the other calibrations to end (the last 3 attacks use their defenses)
A=$(submit attacks --array=0-$((N_ATT - 1))%3 --time=12:00:00 --mail-type=END,FAIL \
    --dependency=afterok:"$CM",afterany:"$CR" --kill-on-invalid-dep=yes \
    --export=ALL,TASKS="$REPO/$TD/attacks.txt" purification.sbatch)
echo
echo "Queued: check ($K) -> main calibration ($CM) + $N_CAL others ($CR) -> $N_ATT attacks ($A); ids in $P1/jobs.txt"
echo "Follow:  squeue -u $USER -o \"%.18i %.9T %.10M %.14R %j\""
echo "Report (also while running):  bash purification_phase1.sh --report"
