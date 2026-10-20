#!/bin/bash
# Diffusion purification, PHASE 2: the configuration chosen in phase 1 (on the
# tuning split, AE seed 7) applied to all 18 chronological SWAT-CLEAN detectors and
# evaluated on the FINAL split of the test file (rows from 134975).
#
# The chosen values are REQUIRED (no defaults, so phase 2 cannot run with values
# that were not chosen in phase 1):
#   T_STAR=<t*> STEPS=<K> EOT=<samples> ITERS=<PGD iterations> BUDGETS="<range fractions>" \
#       bash purification_phase2.sh
#   bash purification_phase2.sh --report
#
# Per detector: calibrate t* = 0 (must reproduce the stored detector, else exit 3 and
# that detector's attacks are cancelled) and t* = T_STAR; then, for every budget in
# BUDGETS (actuators protected), evasion and false alarm with
#   undefended  t* = 0, PGD (deterministic, EOT 1)
#   adaptive    full-gradient PGD + EOT through the defense
#   blind       PGD on the undefended detector, evaluated on the defended one
set -euo pipefail
REPO=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
cd "$REPO"
P2=outputs/purification/phase2
DIFF=models/diffusion_swat_v1

if [[ "${1:-}" == "--report" ]]; then
    python3 purification_report.py "$P2" | tee "$P2/report.txt"
    if [[ -f "$P2/jobs.txt" ]]; then
        IDS=$(awk '{print $2}' "$P2/jobs.txt" | paste -sd, -)
        sacct -j "$IDS" --format=JobID%18,JobName%10,State%11,Elapsed,MaxRSS > "$P2/sacct.txt" || true
        echo "== job states"; awk 'NR>2 {print $3}' "$P2/sacct.txt" | sort | uniq -c
    fi
    tar czf "$HOME/purification_phase2.tgz" --exclude='*.npz' -C outputs/purification phase2
    echo "== archive: $HOME/purification_phase2.tgz ($(du -h "$HOME/purification_phase2.tgz" | cut -f1))"
    exit 0
fi

: "${T_STAR:?set T_STAR (chosen in phase 1)}"
: "${STEPS:?set STEPS (chosen in phase 1)}"
: "${EOT:?set EOT (chosen in phase 1)}"
: "${ITERS:?set ITERS (chosen in phase 1)}"
: "${BUDGETS:?set BUDGETS, e.g. \"0.05 0.10\"}"
CONTEXT="${CONTEXT:-64}"
SAMPLER="${SAMPLER:-ddpm}"

git fetch -q origin
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]]; then
    echo "ERROR: HEAD is not origin/main. Pull (or push) first."; exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked files have uncommitted changes"; git status --short | grep -v '^??'; exit 1
fi
if [[ ! -f "$DIFF/weights.npz" ]]; then echo "ERROR: $DIFF not trained (phase 1)"; exit 1; fi
if [[ -e "$P2/jobs.txt" ]]; then echo "ERROR: $P2/jobs.txt exists: phase 2 was already queued."; exit 1; fi
mkdir -p "$P2/tasks"
echo "commit $(git rev-parse --short HEAD), $(date -Is); T_STAR=$T_STAR STEPS=$STEPS EOT=$EOT ITERS=$ITERS BUDGETS=$BUDGETS CONTEXT=$CONTEXT SAMPLER=$SAMPLER" > "$P2/commit.txt"

K=$STEPS; if (( T_STAR < K )); then K=$T_STAR; fi
TAG0=$(printf "tstar%04d_k%03d_%s" 0 0 "$SAMPLER")
TAG=$(printf "tstar%04d_k%03d_%s" "$T_STAR" "$K" "$SAMPLER")
RUNS=()
for SEED in 7 21 42; do for M in ae cnn dnn lstm gru lin; do RUNS+=("swatchrono_${M}_s${SEED}"); done; done

submit() {
    local label="$1"; shift
    local id; id=$(sbatch --parsable "$@"); id="${id%%;*}"
    echo "$label $id" >> "$P2/jobs.txt"; echo "queued  $label  job $id" >&2; echo "$id"
}
for RUN in "${RUNS[@]}"; do
    MODEL=models/$RUN
    CAL=$P2/$RUN/calib
    echo "$CAL main_purification.py calibrate --dataset SWAT-CLEAN --model-dir $MODEL --diffusion $DIFF --context $CONTEXT --steps $STEPS --sampler $SAMPLER --t-star 0 $T_STAR --draws 10 --out $CAL" > "$P2/tasks/$RUN.calib.txt"
    BASE="--dataset SWAT-CLEAN --model-dir $MODEL --split final --max-targets 2048 --alpha-relative 0.10 --eval-draws 10 --trace-every 10 --seed 42 --protect-actuators --iterations $ITERS"
    : > "$P2/tasks/$RUN.attacks.txt"
    for B in $BUDGETS; do
        for G in evasion false_alarm; do
            O=$P2/$RUN/range$B/undefended/$G
            echo "$O main_purification.py attack $BASE --epsilon-range-fraction $B --defense $CAL/defense_$TAG0.json --goal $G --gradient full --eot 1 --out $O" >> "$P2/tasks/$RUN.attacks.txt"
            O=$P2/$RUN/range$B/adaptive_eot$EOT/$G
            echo "$O main_purification.py attack $BASE --epsilon-range-fraction $B --defense $CAL/defense_$TAG.json --goal $G --gradient full --eot $EOT --out $O" >> "$P2/tasks/$RUN.attacks.txt"
            O=$P2/$RUN/range$B/blind/$G
            echo "$O main_purification.py attack $BASE --epsilon-range-fraction $B --defense $CAL/defense_$TAG.json --goal $G --gradient blind --eot 1 --out $O" >> "$P2/tasks/$RUN.attacks.txt"
        done
    done
    N=$(wc -l < "$P2/tasks/$RUN.attacks.txt")
    C=$(submit "calib_$RUN" --array=0 --time=08:00:00 --export=ALL,TASKS="$REPO/$P2/tasks/$RUN.calib.txt" purification.sbatch)
    submit "attacks_$RUN" --array=0-$((N - 1))%3 --time=12:00:00 --dependency=afterok:"$C" --kill-on-invalid-dep=yes \
        --export=ALL,TASKS="$REPO/$P2/tasks/$RUN.attacks.txt" purification.sbatch > /dev/null
done
echo
echo "Queued 18 calibrations and their attacks; ids in $P2/jobs.txt."
echo "Note: Slurm limits running jobs per user; the arrays run as slots free up."
echo "At the end:  bash purification_phase2.sh --report"
