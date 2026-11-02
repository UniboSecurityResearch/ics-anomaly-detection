#!/bin/bash
# Diffusion purification, follow-up of phase 2: the five checks that came out of the
# per-attack analysis of the phase-2 records (AE, CNN, DNN, LSTM, GRU, LIN; final split).
# Reuses the phase-2 calibrations and output tree, so purification_phase2.sh --report
# includes the results.
#
#   bash purification_followup.sh --plan   # writes and counts the tasks, queues nothing
#   bash purification_followup.sh          # queues them
#   bash purification_phase2.sh --report   # any time: tables + archive
#
# Three arrays, one after the other (afterany), in order of priority:
#  A  F1  a higher theta alone, no purifier: margin false alarm (kappa 0.5, 0.1) and
#         evasion at theta x THETA_FACTORS, t* = 0, all 18 detectors
#     F5  whole attacks, no attacker: per-attack detection at t* = 0 (with the theta
#         multiples, F1 on the whole final split) and T_STARS, all 18 detectors
#     F2  false alarm by an attacker who also changes the context rows of the purifier
#         (--modifiable window), t* = 20 and 50, the 17 detectors AE seed 7 does not cover
#  C  F3  calibrations of the sensors-only purifier (--keep-actuators), seed 7
#  B  F3  attacks on it (evasion; false alarm with the mean and margin objectives) and
#         per-attack detection, seed 7, SENSORS_T_STARS
#     F4  budget on the training range without the plant's start-up (--range-start),
#         seed 7, t* = 0 and SENSORS_T_STARS: evasion; false alarm with mean and margin
# Settings (defaults = phase 2): T_STARS="5 20 50" SENSORS_T_STARS="20 50" EOT=5 ITERS=200
# ALPHA=0.10 BUDGET=0.05 BLOCK=64 THETA_FACTORS="2 3 5" RANGE_START=21600
set -euo pipefail
REPO="${REPO:-/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection}"
cd "$REPO"
P2=outputs/purification/phase2
DIFF=models/diffusion_swat_v1
MODE="${1:-queue}"
if [[ "$MODE" != "queue" && "$MODE" != "--plan" ]]; then echo "usage: see the header"; exit 1; fi
T_STARS="${T_STARS:-5 20 50}"
SENSORS_T_STARS="${SENSORS_T_STARS:-20 50}"
EOT="${EOT:-5}"
ITERS="${ITERS:-200}"
ALPHA="${ALPHA:-0.10}"
BUDGET="${BUDGET:-0.05}"
BLOCK="${BLOCK:-64}"
THETA_FACTORS="${THETA_FACTORS:-2 3 5}"
RANGE_START="${RANGE_START:-21600}"
STEPS=10
CONTEXT=128
LOOKAHEAD=outputs/purification/phase1_v2/denoiser_check/check.json
read -r -a TS <<< "$T_STARS"
read -r -a TSENS <<< "$SENSORS_T_STARS"
ARCHS="ae cnn dnn lstm gru lin"
AE7=swatchrono_ae_s7
RUNS=(); for SEED in 7 21 42; do for M in $ARCHS; do RUNS+=("swatchrono_${M}_s${SEED}"); done; done
S7=(); for M in $ARCHS; do S7+=("swatchrono_${M}_s7"); done

kof() { if (( $1 == 0 )); then echo 0; elif (( $1 < STEPS )); then echo "$1"; else echo "$STEPS"; fi; }
tag() { printf "tstar%04d_k%03d_ddpm%s" "$1" "$(kof "$1")" "${2:-}"; }   # $1 t*, $2 suffix

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
    if grep -q '^followup_' "$P2/jobs.txt"; then echo "ERROR: the follow-up was already queued (see $P2/jobs.txt)"; exit 1; fi
    TD=$P2/tasks
else
    TD=$P2/plan_followup
    rm -rf "$TD"
fi
mkdir -p "$TD"
FA=$TD/followup_a.txt; FC=$TD/followup_calib.txt; FB=$TD/followup_b.txt; PR=$TD/priorities_followup.txt
: > "$FA"; : > "$FC"; : > "$FB"; : > "$PR"

# every phase-2 calibration the tasks need must exist
for RUN in "${RUNS[@]}"; do
    for T in 0 "${TS[@]}"; do
        F="$P2/$RUN/calib/defense_$(tag "$T").json"
        if [[ ! -f "$F" ]]; then echo "ERROR: missing $F (phase-2 calibration)"; exit 1; fi
    done
done
if [[ ! -f "$LOOKAHEAD" ]]; then echo "ERROR: missing $LOOKAHEAD"; exit 1; fi

# att <task file> <run> <t*> <calib folder> <tag suffix> <folder under the run> <goal> <attack arguments...>
att() {
    local FILE=$1 RUN=$2 T=$3 C=$4 SUF=$5 O=$P2/$2/$6/$7 G=$7; shift 7
    echo "$O main_purification.py attack --dataset SWAT-CLEAN --model-dir models/$RUN --split final --block-length $BLOCK --eval-draws 10 --trace-every 10 --seed 42 --defense $C/defense_$(tag "$T" "$SUF").json --require-reproduction $P2/$RUN/calib/$(tag 0).json --epsilon-range-fraction $BUDGET --goal $G --protect-actuators --alpha-relative $ALPHA --iterations $ITERS --gradient full $* --out $O" >> "$FILE"
}
block() {
    local n; n=$(wc -l < "$2")
    echo "$1 $(basename "$2" .txt) ${START:-0}-$((n - 1))" >> "$PR"; START=$n
}

# ---- array A -----------------------------------------------------------------------
LAB0=full_a${ALPHA}_eot1_it${ITERS}
for RUN in "${RUNS[@]}"; do for K in $THETA_FACTORS; do
    B=range$BUDGET/t0/thetax${K}_$LAB0
    att "$FA" "$RUN" 0 "$P2/$RUN/calib" "" "$B" evasion --modifiable detector --eot 1 --theta-factor "$K"
    for KP in 0.5 0.1; do
        att "$FA" "$RUN" 0 "$P2/$RUN/calib" "" "${B}_margin$KP" false_alarm --modifiable detector --eot 1 --theta-factor "$K" --loss margin --margin "$KP"
    done
done; done
block F1_theta_only "$FA"
for RUN in "${RUNS[@]}"; do
    C=$P2/$RUN/calib; D=()
    for T in 0 "${TS[@]}"; do D+=("$C/defense_$(tag "$T").json"); done
    O=$P2/$RUN/per_attack/standard
    echo "$O main_purification.py per-attack --dataset SWAT-CLEAN --model-dir models/$RUN --defense ${D[*]} --theta-factors 1 $THETA_FACTORS --eval-draws 5 --out $O" >> "$FA"
done
block F5_whole_attacks "$FA"
for RUN in "${RUNS[@]}"; do
    [[ "$RUN" == "$AE7" ]] && continue
    for T in 20 50; do
        att "$FA" "$RUN" "$T" "$P2/$RUN/calib" "" "range$BUDGET/t$T/rowswindow_full_a${ALPHA}_eot${EOT}_it${ITERS}" false_alarm --modifiable window --eot "$EOT"
    done
done
block F2_context_rows "$FA"

# ---- calibrations of the sensors-only purifier ---------------------------------------
START=0
for RUN in "${S7[@]}"; do
    O=$P2/$RUN/calib_sensors
    echo "$O main_purification.py calibrate --dataset SWAT-CLEAN --model-dir models/$RUN --diffusion $DIFF --context $CONTEXT --lookahead $LOOKAHEAD --steps $STEPS --sampler ddpm --keep-actuators --t-star $SENSORS_T_STARS --draws 10 --out $O" >> "$FC"
done
block F3_calibrations "$FC"

# ---- array B -----------------------------------------------------------------------
START=0
for RUN in "${S7[@]}"; do
    C=$P2/$RUN/calib_sensors; D=()
    for T in "${TSENS[@]}"; do
        LAB=range$BUDGET/t$T/sensors_full_a${ALPHA}_eot${EOT}_it${ITERS}
        att "$FB" "$RUN" "$T" "$C" _keepact "$LAB" evasion --modifiable detector --eot "$EOT"
        att "$FB" "$RUN" "$T" "$C" _keepact "$LAB" false_alarm --modifiable detector --eot "$EOT"
        att "$FB" "$RUN" "$T" "$C" _keepact "${LAB}_margin0.5" false_alarm --modifiable detector --eot "$EOT" --loss margin --margin 0.5
        D+=("$C/defense_$(tag "$T" _keepact).json")
    done
    O=$P2/$RUN/per_attack/sensors
    echo "$O main_purification.py per-attack --dataset SWAT-CLEAN --model-dir models/$RUN --defense ${D[*]} --eval-draws 5 --out $O" >> "$FB"
done
block F3_sensors_only "$FB"
for RUN in "${S7[@]}"; do
    for T in 0 "${TSENS[@]}"; do
        E=$EOT; (( T == 0 )) && E=1
        LAB=rangeop$BUDGET/t$T/full_a${ALPHA}_eot${E}_it${ITERS}
        ROP="--range-start $RANGE_START --modifiable detector --eot $E"
        att "$FB" "$RUN" "$T" "$P2/$RUN/calib" "" "$LAB" evasion $ROP
        att "$FB" "$RUN" "$T" "$P2/$RUN/calib" "" "$LAB" false_alarm $ROP
        att "$FB" "$RUN" "$T" "$P2/$RUN/calib" "" "${LAB}_margin0.5" false_alarm $ROP --loss margin --margin 0.5
    done
done
block F4_budget_without_startup "$FB"

for F in "$FA" "$FC" "$FB"; do
    BAD=$(awk '{ ok = 0; for (i = 2; i < NF; i++) if ($i == "--out" && $(i + 1) == $1) ok = 1; if (!ok) print NR }' "$F" | head -3)
    if [[ -n "$BAD" ]]; then echo "ERROR: $F lines $BAD lack --out <their folder>"; exit 1; fi
done
if [[ $(cat "$FA" "$FC" "$FB" | awk '{print $1}' | sort | uniq -d | wc -l) -ne 0 ]]; then echo "ERROR: duplicate output folders"; exit 1; fi
for F in "$FA" "$FC" "$FB"; do
    while read -r O _; do
        if [[ -e "$O/DONE" ]]; then echo "ERROR: $O already done"; exit 1; fi
    done < "$F"
done
if command -v python3 > /dev/null && python3 -c "import tensorflow" 2> /dev/null; then
    python3 -m purification.check_tasks "$FA" "$FC" "$FB" || exit 1
else
    echo "note: no TensorFlow here, task arguments not parsed (the --out check passed)"
fi
NA=$(wc -l < "$FA"); NC=$(wc -l < "$FC"); NB=$(wc -l < "$FB")
CFG="T_STARS=$T_STARS SENSORS_T_STARS=$SENSORS_T_STARS EOT=$EOT ITERS=$ITERS ALPHA=$ALPHA BUDGET=$BUDGET BLOCK=$BLOCK THETA_FACTORS=$THETA_FACTORS RANGE_START=$RANGE_START"
echo "$CFG"
echo "array A: $NA tasks; calibrations: $NC; array B: $NB tasks. Blocks (array task indices):"
sed 's/^/  /' "$PR"
if [[ "$MODE" == "--plan" ]]; then echo "plan only, nothing queued; task files in $TD"; exit 0; fi

echo "follow-up, commit $(git rev-parse --short HEAD), $(date -Is); $CFG" >> "$P2/commit.txt"
sub() { local id; id=$(sbatch --parsable "$@"); echo "${id%%;*}"; }
A=$(sub --array=0-$((NA - 1))%3 --time=12:00:00 --mail-type=END,FAIL --export=ALL,TASKS="$REPO/$FA" purification.sbatch)
echo "followup_a $A" >> "$P2/jobs.txt"
C=$(sub --array=0-$((NC - 1))%3 --time=08:00:00 --mail-type=FAIL --dependency=afterany:"$A" --export=ALL,TASKS="$REPO/$FC" purification.sbatch)
echo "followup_calib $C" >> "$P2/jobs.txt"
B=$(sub --array=0-$((NB - 1))%3 --time=12:00:00 --mail-type=END,FAIL --dependency=afterany:"$C" --export=ALL,TASKS="$REPO/$FB" purification.sbatch)
echo "followup_b $B" >> "$P2/jobs.txt"
echo "Queued: array A (job $A), calibrations (job $C, after A), array B (job $B, after the calibrations)."
echo "Report, also while running: bash purification_phase2.sh --report"
echo "To drop a block: scancel <job>_[<first>-<last>] with the indices above."
