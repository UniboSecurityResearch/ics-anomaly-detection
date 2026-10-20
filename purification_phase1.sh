#!/bin/bash
# Diffusion purification, PHASE 1 (October 2026): train the diffusion model on the
# SWAT-CLEAN training file and study the defense on ONE detector (AE, seed 7) on the
# TUNING part of the test file (rows before 134975), so that the configuration
# chosen here (t*, denoising steps, EOT, iterations) is never chosen on the final
# split used for the reported results (phase 2).
#
#   bash purification_phase1.sh            # checks the code state, then queues every job
#   bash purification_phase1.sh --report   # when the queue is empty: tables + one archive
#
# Jobs (partition l40s, at most 3 running at a time):
#   1. diffusion   train the noise predictor (gate: exit 4 if it has not learnt
#                  structure or fails at the detectors' window lengths)
#   2. calibrate   AE s7: t* = 0 (must reproduce the stored detector, else exit 3),
#                  5, 10, 20, 30, 50, 100, 150, 200, 300 with K = 10 DDPM steps;
#                  K = 5 / 20 and DDIM K = 10 at t* = 20, 50, 100
#   3. attacks     PGD 200 it., step 0.10 eps, random start, up to 2048 targets,
#                  10 eval draws, actuators protected:
#                  a) full-gradient PGD+EOT(20), evasion and false alarm, budgets
#                     0.02 / 0.05 / 0.10 x training range, t* = 0 (undefended
#                     reference), 10, 20, 50, 100, 200
#                  checks b-g at budget 0.05:
#                  b) BPDA and defense-blind attacker, t* = 20, 50, 100
#                  c) surrogate process with 2 / 5 steps, t* = 50, 100 (evasion)
#                  d) EOT = 1, 5, 10, 40 at t* = 50 (evasion): EOT saturation
#                  e) 500 iterations at t* = 50 (evasion): iteration saturation
#                  f) huge budget (whole range, actuators free) at t* = 0 and 50:
#                     an attack that cannot win here would signal gradient masking
#                  g) defense with K = 5 / 20 and with DDIM at t* = 50 (evasion)
# Nothing is overwritten: every task writes its own folder under
# outputs/purification/phase1 and is skipped if its DONE marker exists.
set -euo pipefail
REPO=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
cd "$REPO"
P1=outputs/purification/phase1
DIFF=models/diffusion_swat_v1
MODEL=models/swatchrono_ae_s7

if [[ "${1:-}" == "--report" ]]; then
    python3 purification_report.py "$P1" | tee "$P1/report.txt"
    if [[ -f "$P1/jobs.txt" ]]; then
        IDS=$(awk '{print $2}' "$P1/jobs.txt" | paste -sd, -)
        sacct -j "$IDS" --format=JobID%18,JobName%10,State%11,Elapsed,MaxRSS > "$P1/sacct.txt" || true
        echo "== job states"; awk 'NR>2 {print $3}' "$P1/sacct.txt" | sort | uniq -c
    fi
    cp "$DIFF/config.json" "$P1/diffusion_config.json" 2>/dev/null || true
    cp "$DIFF/train_log.csv" "$P1/diffusion_train_log.csv" 2>/dev/null || true
    tar czf "$HOME/purification_phase1.tgz" --exclude='*.npz' -C outputs/purification phase1
    echo "== archive: $HOME/purification_phase1.tgz ($(du -h "$HOME/purification_phase1.tgz" | cut -f1))"
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
if [[ ! -f "$MODEL/$(ls "$MODEL" | grep '\.h5$' | head -1)" ]]; then
    echo "ERROR: no model in $MODEL"; exit 1
fi
if [[ -e "$P1/jobs.txt" ]]; then echo "ERROR: $P1/jobs.txt exists: phase 1 was already queued."; exit 1; fi
if [[ -e "$DIFF" ]]; then echo "ERROR: $DIFF exists; phase 1 trains it from scratch."; exit 1; fi
mkdir -p "$P1"
echo "commit $(git rev-parse --short HEAD), $(date -Is)" > "$P1/commit.txt"

# --- task files ----------------------------------------------------------------
TD="$P1/tasks"; mkdir -p "$TD"
# digests of the SWAT-CLEAN matrices as processed by data/SWAT/process_SWaT.py from
# the iTrust files whose hashes hash_SWaT.py accepts (computed in the development
# copy): the job stops (exit 5) if the cluster data differ
DIGESTS="8e2a2c4d5bec839b70ab49c06fe4e86c859cd9d72cfb2e07c5465322925da343 a46e7134e5b3895d84205a72d0380f12d4c5cb7bca97651716998cf7744ded09 456181b4a539a65f3d9ad8f1be3b15dd348acabf6b69f186c1385a6a36e3cc3c"
echo "$DIFF main_train_diffusion.py SWAT-CLEAN --out $DIFF --window 128 --steps 100000 --batch 128 --expect-data $DIGESTS" > "$TD/diffusion.txt"

CAL=$P1/calib
COMMON="--dataset SWAT-CLEAN --model-dir $MODEL --diffusion $DIFF --context 64"
{
    echo "$CAL/k10_ddpm main_purification.py calibrate $COMMON --steps 10 --sampler ddpm --t-star 0 5 10 20 30 50 100 150 200 300 --draws 5 --out $CAL/k10_ddpm"
    echo "$CAL/k05_ddpm main_purification.py calibrate $COMMON --steps 5 --sampler ddpm --t-star 20 50 100 --draws 5 --out $CAL/k05_ddpm"
    echo "$CAL/k20_ddpm main_purification.py calibrate $COMMON --steps 20 --sampler ddpm --t-star 20 50 100 --draws 5 --out $CAL/k20_ddpm"
    echo "$CAL/k10_ddim main_purification.py calibrate $COMMON --steps 10 --sampler ddim --t-star 20 50 100 --draws 5 --out $CAL/k10_ddim"
} > "$TD/calibrate.txt"

defense() {  # $1 = calib folder, $2 = t*, $3 = K, $4 = sampler -> defense json path
    local k=$3; if (( $2 == 0 )); then k=0; elif (( $2 < k )); then k=$2; fi
    printf "%s/defense_tstar%04d_k%03d_%s.json" "$1" "$2" "$k" "$4"
}
ATT=$P1/attacks
BASE="--dataset SWAT-CLEAN --model-dir $MODEL --split tune --max-targets 2048 --alpha-relative 0.10 --eval-draws 10 --trace-every 10 --seed 42"
{
    # a) full gradient, budgets 0.02 / 0.05 / 0.10 of the training range (actuators
    #    protected), every t* used for attacks (t* = 0 is the undefended reference)
    for B in 0.02 0.05 0.10; do
        for T in 0 10 20 50 100 200; do
            for G in evasion false_alarm; do
                O=$ATT/range$B/full_eot20_it200/t$T/$G
                echo "$O main_purification.py attack $BASE --epsilon-range-fraction $B --protect-actuators --defense $(defense $CAL/k10_ddpm $T 10 ddpm) --goal $G --gradient full --eot 20 --iterations 200 --out $O"
            done
        done
    done
    # checks at budget 0.05
    B=0.05
    CHK="$BASE --epsilon-range-fraction $B --protect-actuators"
    # b) BPDA and defense-blind attacker
    for T in 20 50 100; do
        for GR in bpda blind; do
            for G in evasion false_alarm; do
                O=$ATT/range$B/${GR}_it200/t$T/$G
                echo "$O main_purification.py attack $CHK --defense $(defense $CAL/k10_ddpm $T 10 ddpm) --goal $G --gradient $GR --eot 20 --iterations 200 --out $O"
            done
        done
    done
    # c) surrogate process (attacker uses fewer denoising steps)
    for T in 50 100; do
        for KA in 2 5; do
            O=$ATT/range$B/surrogate_k${KA}_eot20_it200/t$T/evasion
            echo "$O main_purification.py attack $CHK --defense $(defense $CAL/k10_ddpm $T 10 ddpm) --goal evasion --gradient surrogate --attack-steps $KA --eot 20 --iterations 200 --out $O"
        done
    done
    # d) EOT saturation (EOT = 20 is in a)
    for E in 1 5 10 40; do
        O=$ATT/range$B/full_eot${E}_it200/t50/evasion
        echo "$O main_purification.py attack $CHK --defense $(defense $CAL/k10_ddpm 50 10 ddpm) --goal evasion --gradient full --eot $E --iterations 200 --out $O"
    done
    # e) iteration saturation
    O=$ATT/range$B/full_eot20_it500/t50/evasion
    echo "$O main_purification.py attack $CHK --defense $(defense $CAL/k10_ddpm 50 10 ddpm) --goal evasion --gradient full --eot 20 --iterations 500 --out $O"
    # f) huge budget, actuators free: an attack that cannot win here would signal
    #    gradient masking
    for T in 0 50; do
        O=$ATT/range1.0_free/full_eot20_it200/t$T/evasion
        echo "$O main_purification.py attack $BASE --epsilon-range-fraction 1.0 --defense $(defense $CAL/k10_ddpm $T 10 ddpm) --goal evasion --gradient full --eot 20 --iterations 200 --out $O"
    done
    # g) other defense configurations at t* = 50
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
D=$(submit diffusion --array=0 --time=12:00:00 --export=ALL,TASKS="$REPO/$TD/diffusion.txt" purification.sbatch)
C=$(submit calibrate --array=0-$((N_CAL - 1))%3 --time=06:00:00 --dependency=afterok:"$D" --kill-on-invalid-dep=yes \
    --export=ALL,TASKS="$REPO/$TD/calibrate.txt" purification.sbatch)
A=$(submit attacks --array=0-$((N_ATT - 1))%3 --time=12:00:00 --dependency=afterok:"$C" --kill-on-invalid-dep=yes \
    --export=ALL,TASKS="$REPO/$TD/attacks.txt" purification.sbatch)
echo
echo "Queued: diffusion ($D) -> $N_CAL calibrations ($C) -> $N_ATT attacks ($A); ids in $P1/jobs.txt"
echo "Follow:  squeue -u $USER -o \"%.18i %.9T %.10M %.14R %j\""
echo "At the end:  bash purification_phase1.sh --report"
