#!/bin/bash
# Diffusion purification, PHASE 1b: the phase-1_v2 attacks redone under ONE threat
# model for every arm, plus the checks the phase-1_v2 results showed to be missing.
# AE seed 7, TUNING split, 512 targets, the defenses calibrated in phase 1_v2
# (outputs/purification/phase1_v2/calib, reused unchanged).
#
#   bash purification_phase1b.sh            # checks the code state, then queues the attacks
#   bash purification_phase1b.sh --report   # any time: tables of what has finished + archive
#
# Why (phase 1_v2, outputs/purification/phase1_v2):
#  * against a purified detector the attacker could change every row of the 128-row
#    purified windows, against the undefended one only the scored rows: the defended
#    arms had far more modifiable rows. Here every arm may change the same rows, the
#    ones feeding the undefended detector (--modifiable detector; the convention of the
#    undefended benchmark). The phase 1_v2 results stay as the "attacker also controls
#    the defense's context" reference.
#  * the whole-range sanity check failed on the UNDEFENDED detector (no evasion at all):
#    with a random start spread over the whole range and steps of 10% of the range,
#    sign-PGD cannot settle on values the autoencoder reconstructs. It is redone from the
#    clean series with small steps, where it must succeed.
#  * only one step size was used; the undefended benchmark used three. Convergence is
#    checked with steps 0.025 (500 it.) and 0.25.
#
# Array order = priority:
#  P1  budget 0.05, full-gradient PGD+EOT(20), step 0.10, 200 it., evasion and false
#      alarm, t* = 0 (undefended), 5, 10, 20, 30, 50 (the t* that pass the clean-F1 rule)
#  P2  whole range, actuators free, from the clean series, step 0.01: t* = 0, 20, 50
#  P3  step 0.025 / 500 it. and step 0.25 / 200 it.: t* = 0, 20, 50, both goals
#  P4  attack strength at t* = 20 and 50, budget 0.05, evasion: EOT 1 / 40, 500 it.,
#      surrogate 2 / 5 steps, BPDA, blind; BPDA and blind also for false alarm
#  P5  budgets 0.02 and 0.10 as P1, t* = 0, 10, 20, 50
set -euo pipefail
REPO=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
cd "$REPO"
P=outputs/purification/phase1b
CAL=outputs/purification/phase1_v2/calib/k10_ddpm
MODEL=models/swatchrono_ae_s7

if [[ "${1:-}" == "--report" ]]; then
    python3 purification_report.py "$P" | tee "$P/report.txt"
    if [[ -f "$P/jobs.txt" ]]; then
        IDS=$(awk '{print $2}' "$P/jobs.txt" | paste -sd, -)
        sacct -j "$IDS" -P --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS > "$P/sacct.txt" || true
        echo "== job states"; awk -F'|' 'NR>1 && $1 !~ /\./ {print $3}' "$P/sacct.txt" | sort | uniq -c
        mkdir -p "$P/failed_logs"
        for j in $(awk -F'|' 'NR>1 && $1 ~ /_[0-9]+$/ && $3 != "COMPLETED" && $3 != "RUNNING" && $3 != "PENDING" {print $1}' "$P/sacct.txt"); do
            cp "slurm.purif.$j.out" "slurm.purif.$j.err" "$P/failed_logs/" 2>/dev/null || true
        done
    fi
    tar czf "$HOME/purification_phase1b.tgz" --exclude='*.npz' -C outputs/purification phase1b
    echo "== archive: $HOME/purification_phase1b.tgz ($(du -h "$HOME/purification_phase1b.tgz" | cut -f1))"
    exit 0
fi

git fetch -q origin
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]]; then
    echo "ERROR: HEAD is not origin/main. Pull (or push) first."; exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked files have uncommitted changes:"; git status --short | grep -v '^??'; exit 1
fi
for T in 0 5 10 20 30 50; do
    K=10; if (( T == 0 )); then K=0; elif (( T < K )); then K=$T; fi
    F=$(printf "%s/defense_tstar%04d_k%03d_ddpm.json" "$CAL" "$T" "$K")
    if [[ ! -f "$F" ]]; then echo "ERROR: missing $F (phase 1_v2 calibration)"; exit 1; fi
done
if [[ -n "$(squeue -u "$USER" -h)" ]]; then echo "ERROR: you still have jobs in the queue."; exit 1; fi
if [[ -e "$P/jobs.txt" ]]; then echo "ERROR: $P/jobs.txt exists: this phase was already queued."; exit 1; fi
mkdir -p "$P/tasks"
echo "commit $(git rev-parse --short HEAD), $(date -Is); defenses from $CAL" > "$P/commit.txt"

defense() {  # $1 = t*
    local k=10; if (( $1 == 0 )); then k=0; elif (( $1 < k )); then k=$1; fi
    printf "%s/defense_tstar%04d_k%03d_ddpm.json" "$CAL" "$1" "$k"
}
A=$P/attacks
BASE="--dataset SWAT-CLEAN --model-dir $MODEL --split tune --max-targets 512 --eval-draws 10 --trace-every 10 --seed 42 --modifiable detector"
line() {  # $1 = output folder, rest = attack arguments
    local O=$1; shift
    echo "$O main_purification.py attack $BASE $* --out $O"
}
grid() {  # $1 = budget, rest = t* values
    local B=$1 T G; shift
    for T in "$@"; do
        for G in evasion false_alarm; do
            line "$A/range$B/full_a0.10_eot20_it200/t$T/$G" --epsilon-range-fraction "$B" --protect-actuators \
                --alpha-relative 0.10 --defense "$(defense "$T")" --goal $G --gradient full --eot 20 --iterations 200
        done
    done
}
{
    # P1
    grid 0.05 0 5 10 20 30 50
    # P2
    for T in 0 20 50; do
        line "$A/range1.0_free/full_a0.01_noRS_eot20_it200/t$T/evasion" --epsilon-range-fraction 1.0 \
            --alpha-relative 0.01 --no-random-start --defense "$(defense $T)" --goal evasion --gradient full --eot 20 --iterations 200
    done
    # P3
    for T in 0 20 50; do
        for G in evasion false_alarm; do
            line "$A/range0.05/full_a0.025_eot20_it500/t$T/$G" --epsilon-range-fraction 0.05 --protect-actuators \
                --alpha-relative 0.025 --defense "$(defense $T)" --goal $G --gradient full --eot 20 --iterations 500
            line "$A/range0.05/full_a0.25_eot20_it200/t$T/$G" --epsilon-range-fraction 0.05 --protect-actuators \
                --alpha-relative 0.25 --defense "$(defense $T)" --goal $G --gradient full --eot 20 --iterations 200
        done
    done
    # P4
    S="--epsilon-range-fraction 0.05 --protect-actuators --alpha-relative 0.10 --iterations 200"
    for T in 20 50; do
        for E in 1 40; do
            line "$A/range0.05/full_a0.10_eot${E}_it200/t$T/evasion" $S --defense "$(defense $T)" --goal evasion --gradient full --eot $E
        done
        line "$A/range0.05/full_a0.10_eot20_it500/t$T/evasion" --epsilon-range-fraction 0.05 --protect-actuators \
            --alpha-relative 0.10 --iterations 500 --defense "$(defense $T)" --goal evasion --gradient full --eot 20
        for KA in 2 5; do
            line "$A/range0.05/surrogate_k${KA}_a0.10_eot20_it200/t$T/evasion" $S --defense "$(defense $T)" --goal evasion \
                --gradient surrogate --attack-steps $KA --eot 20
        done
        for GR in bpda blind; do
            for G in evasion false_alarm; do
                line "$A/range0.05/${GR}_a0.10_eot20_it200/t$T/$G" $S --defense "$(defense $T)" --goal $G --gradient $GR --eot 20
            done
        done
    done
    # P5
    grid 0.02 0 10 20 50
    grid 0.10 0 10 20 50
} > "$P/tasks/attacks.txt"
N=$(wc -l < "$P/tasks/attacks.txt")
if [[ $(awk '{print $1}' "$P/tasks/attacks.txt" | sort | uniq -d | wc -l) -ne 0 ]]; then
    echo "ERROR: duplicate output folders in the task list"; exit 1
fi
ID=$(sbatch --parsable --array=0-$((N - 1))%3 --time=12:00:00 --mail-type=END,FAIL \
    --export=ALL,TASKS="$REPO/$P/tasks/attacks.txt" purification.sbatch)
ID="${ID%%;*}"
echo "attacks $ID" > "$P/jobs.txt"
echo "Queued $N attacks (job $ID). Report (also while running): bash purification_phase1b.sh --report"
