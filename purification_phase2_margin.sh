#!/bin/bash
# Diffusion purification, PHASE 2 follow-up: the false-alarm attacks redone with an
# objective aligned with the detector's decision (--loss margin, purification/attack.py
# objective_terms), because the phase-2 attack-strength check found the standard
# false-alarm attack NOT saturated under purification: the log-score variant beat it by
# more than 1 point in 11 of the 18 defended seed-7 cases (up to +14.4 points), and the
# mean objective raised the mean score while the share of targets flagged stayed flat.
# Reuses the phase-2 calibrations and settings; outputs go into the same tree
# (outputs/purification/phase2/<detector>/range<B>/t<t*>/<label>/false_alarm), so
# purification_phase2.sh --report includes them.
#
#   bash purification_phase2_margin.sh --plan   # writes and counts the tasks, queues nothing
#   bash purification_phase2_margin.sh          # queues them (after the phase-2 attacks)
#   bash purification_phase2.sh --report        # tables + archive, as for phase 2
#
# Blocks, in order of priority (an array, so a block can be dropped with scancel):
#   M1  margin objective (kappa = MARGIN), all 18 detectors, t* = 0 and T_STARS -> B (false alarm)
#   M2  log-score objective, seeds 21 and 42 (seed 7 has it from phase 2), so that the
#       worst case over the objectives (mean, log, margin) exists for all 18 detectors
#   M3  margin with kappa = 0.1, seed 7, T_STARS: sensitivity to kappa
# Settings (defaults = phase 2, commit.txt): T_STARS="5 20 50" EOT=5 ITERS=200
# ALPHA=0.10 MODIFIABLE=detector BUDGET=0.05 BLOCK=64 MARGIN=0.5 MARGIN_SENS=0.1
set -euo pipefail
REPO="${REPO:-/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection}"
cd "$REPO"
P2=outputs/purification/phase2
MODE="${1:-queue}"
if [[ "$MODE" != "queue" && "$MODE" != "--plan" ]]; then echo "usage: see the header"; exit 1; fi
T_STARS="${T_STARS:-5 20 50}"
EOT="${EOT:-5}"
ITERS="${ITERS:-200}"
ALPHA="${ALPHA:-0.10}"
MODIFIABLE="${MODIFIABLE:-detector}"
BUDGET="${BUDGET:-0.05}"
BLOCK="${BLOCK:-64}"
MARGIN="${MARGIN:-0.5}"
MARGIN_SENS="${MARGIN_SENS:-0.1}"
STEPS=10
read -r -a TS <<< "$T_STARS"
ARCHS="ae cnn dnn lstm gru lin"

kof() { if (( $1 == 0 )); then echo 0; elif (( $1 < STEPS )); then echo "$1"; else echo "$STEPS"; fi; }
tag() { printf "tstar%04d_k%03d_ddpm" "$1" "$(kof "$1")"; }

if [[ "$MODE" == "queue" ]]; then
    git fetch -q origin
    if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]]; then
        echo "ERROR: HEAD is not origin/main. Pull (or push) first."; exit 1
    fi
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "ERROR: tracked files have uncommitted changes"; git status --short | grep -v '^??'; exit 1
    fi
    if [[ -n "$(squeue -u "$USER" -h)" ]]; then echo "ERROR: you still have jobs in the queue."; exit 1; fi
    if [[ ! -f "$P2/jobs.txt" ]]; then echo "ERROR: $P2/jobs.txt missing: phase 2 was not run here"; exit 1; fi
    if grep -q '^margin ' "$P2/jobs.txt"; then echo "ERROR: the margin runs were already queued (see $P2/jobs.txt)"; exit 1; fi
    TD=$P2/tasks
else
    TD=$P2/plan_margin
    rm -rf "$TD"
fi
mkdir -p "$TD"
TF_FILE=$TD/margin.txt
PR_FILE=$TD/priorities_margin.txt
: > "$TF_FILE"; : > "$PR_FILE"

# every calibration the tasks need must exist (phase 2)
for SEED in 7 21 42; do for M in $ARCHS; do
    for T in 0 "${TS[@]}"; do
        F="$P2/swatchrono_${M}_s${SEED}/calib/defense_$(tag "$T").json"
        if [[ ! -f "$F" ]]; then echo "ERROR: missing $F (phase-2 calibration)"; exit 1; fi
    done
done; done

att() {  # <run> <t*> <label suffix> <attack arguments...>
    local RUN=$1 T=$2 SUF=$3; shift 3
    local E=$EOT; (( T == 0 )) && E=1
    local C=$P2/$RUN/calib
    local O=$P2/$RUN/range$BUDGET/t$T/full_a${ALPHA}_eot${E}_it${ITERS}_$SUF/false_alarm
    if [[ -e "$O/summary.json" ]]; then echo "ERROR: $O already exists"; exit 1; fi
    echo "$O main_purification.py attack --dataset SWAT-CLEAN --model-dir models/$RUN --split final --block-length $BLOCK --eval-draws 10 --trace-every 10 --seed 42 --defense $C/defense_$(tag "$T").json --require-reproduction $C/$(tag 0).json --epsilon-range-fraction $BUDGET --goal false_alarm --protect-actuators --modifiable $MODIFIABLE --alpha-relative $ALPHA --iterations $ITERS --gradient full --eot $E $* --out $O" >> "$TF_FILE"
}
block() {
    local n; n=$(wc -l < "$TF_FILE")
    echo "$1 ${START:-0}-$((n - 1))" >> "$PR_FILE"; START=$n
}
# M1
for SEED in 7 21 42; do for M in $ARCHS; do for T in 0 "${TS[@]}"; do
    att "swatchrono_${M}_s${SEED}" "$T" "margin${MARGIN}" --loss margin --margin "$MARGIN"
done; done; done
block "M1_margin${MARGIN}_false_alarm_all18"
# M2
for SEED in 21 42; do for M in $ARCHS; do for T in 0 "${TS[@]}"; do
    att "swatchrono_${M}_s${SEED}" "$T" "log" --loss log
done; done; done
block "M2_log_false_alarm_seeds21_42"
# M3
for M in $ARCHS; do for T in "${TS[@]}"; do
    att "swatchrono_${M}_s7" "$T" "margin${MARGIN_SENS}" --loss margin --margin "$MARGIN_SENS"
done; done
block "M3_margin${MARGIN_SENS}_false_alarm_seed7"

N=$(wc -l < "$TF_FILE")
BAD=$(awk '{ ok = 0; for (i = 2; i < NF; i++) if ($i == "--out" && $(i + 1) == $1) ok = 1; if (!ok) print NR }' "$TF_FILE" | head -3)
if [[ -n "$BAD" ]]; then echo "ERROR: lines $BAD lack --out <their folder>"; exit 1; fi
if [[ $(awk '{print $1}' "$TF_FILE" | sort | uniq -d | wc -l) -ne 0 ]]; then echo "ERROR: duplicate output folders"; exit 1; fi
if command -v python3 > /dev/null && python3 -c "import tensorflow" 2> /dev/null; then
    python3 -m purification.check_tasks "$TF_FILE" || exit 1
else
    echo "note: no TensorFlow here, task arguments not parsed (the --out check passed)"
fi
CFG="T_STARS=$T_STARS EOT=$EOT ITERS=$ITERS ALPHA=$ALPHA MODIFIABLE=$MODIFIABLE BUDGET=$BUDGET BLOCK=$BLOCK MARGIN=$MARGIN MARGIN_SENS=$MARGIN_SENS"
echo "$CFG"
echo "$N false-alarm attacks; priority blocks (array task indices):"
sed 's/^/  /' "$PR_FILE"
if [[ "$MODE" == "--plan" ]]; then echo "plan only, nothing queued; task file $TF_FILE"; exit 0; fi

A=$(sbatch --parsable --array=0-$((N - 1))%3 --time=12:00:00 --mail-type=END,FAIL \
    --export=ALL,TASKS="$REPO/$TF_FILE" purification.sbatch); A="${A%%;*}"
echo "margin $A" >> "$P2/jobs.txt"
echo "margin runs, commit $(git rev-parse --short HEAD), $(date -Is); $CFG" >> "$P2/commit.txt"
echo "Queued $N attacks (job $A). Report (also while running): bash purification_phase2.sh --report"
echo "To drop a block: scancel ${A}_[<first>-<last>] with the indices above."
