#!/bin/bash
# Diffusion purification, PHASE 2: the configuration chosen in phase 1 / 1b (tuning
# split, AE seed 7) applied to all 18 chronological SWAT-CLEAN detectors and evaluated
# on the FINAL split of the test file (rows from 134975), under the phase-1b threat
# model (the same modifiable rows for every arm), plus the runs that test the phase-1b
# conclusion (negative for evasion, positive for false alarm at budget 0.05): attack
# strength, mechanism, ablations, sample size.
#
# The values chosen in phase 1b are REQUIRED (no defaults, so phase 2 cannot run with
# values that were not chosen):
#   T_STARS="<t* ...>" STEPS=<K> EOT=<samples> ITERS=<PGD iterations> \
#   ALPHA=<step / budget> MODIFIABLE=<detector|window> bash purification_phase2.sh
#   ... bash purification_phase2.sh --plan     # writes and counts the tasks, queues nothing
#   bash purification_phase2.sh --report       # any time: tables of what has finished + archive
# Optional: BUDGETS (default "0.05 0.10 0.02"; the FIRST is the main budget), TARGETS
# (default 512, as in phase 1), BIG_TARGETS (default 2048, sample-size check),
# CURVE_T_STARS (default "5 10 20 30 50 100 200", AE seed 7 only), HOLD (default 1:
# the attack array is queued on hold, see P0), CONTEXT, LOOKAHEAD.
#
# Jobs:
#  p0     DIAGNOSTIC, runs first (about 40 min): phase-1b attacks on the tuning split
#         redone with the per-target record (per_target.json), to explain why the
#         undefended evasion result (0.4805) equals the share of targets whose detection
#         window lies inside the target blocks. Check it, then release the attacks:
#         scontrol release <attack job id>   (or queue with HOLD=0 to skip the check).
#  calib  one task per detector (t* = 0, which must reproduce the stored detector, else
#         exit 3, plus T_STARS; AE seed 7 also CURVE_T_STARS), plus two AE seed-7
#         variants: the noise-only ablation (no denoiser) and the causal purifier (d = 0).
#  attacks  ONE array in order of priority, after every calibration has ended. An attack
#         whose detector failed the t* = 0 reproduction refuses to run (exit 3).
# Detectors are taken seed by seed (all six architectures of seed 7, then 21, then 42),
# so a block stopped early still covers every architecture.
#   P1  main budget, all 18 detectors: undefended (t* = 0) and adaptive (full gradient
#       + EOT) at every T_STARS, evasion and false alarm                -> A, B
#   P2  attack strength, seed 7, every architecture, main budget: surrogate (K/2
#       steps), BPDA, blind, EOT 1, log-score loss (also at t* = 0), 3 random restarts
#       (t* = 0 and max T_STARS)                                        -> D
#   P3  mechanism and ablations, AE seed 7, main budget: actuators free; the stronger
#       attacker (MODIFIABLE=window); noise-only ablation; causal purifier; whole range
#       from the clean series at t* = 0 for every seed-7 architecture
#   P4  t* curve, AE seed 7, main budget, CURVE_T_STARS not in T_STARS  -> C
#   P5  convergence, seed 7, every architecture: step 0.025 x 500 it. at t* = 0 and
#       max(T_STARS), evasion                                           -> D
#   P6  sample size, AE seed 7: BIG_TARGETS targets (longer blocks, so the false-alarm
#       ceiling set by the block geometry is higher) at t* = 0 and max(T_STARS)
#   P7  the other budgets, all 18 detectors, same arms as P1           -> B (budget curve)
set -euo pipefail
REPO="${REPO:-/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection}"
cd "$REPO"
P2=outputs/purification/phase2
DIFF=models/diffusion_swat_v1
CAL1=outputs/purification/phase1_v2/calib/k10_ddpm
MODE="${1:-queue}"

if [[ "$MODE" == "--report" ]]; then
    python3 purification_report.py "$P2" | tee "$P2/report.txt"
    if [[ -f "$P2/jobs.txt" ]]; then
        IDS=$(awk '{print $2}' "$P2/jobs.txt" | paste -sd, -)
        sacct -j "$IDS" -P --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS > "$P2/sacct.txt" || true
        echo "== job states"; awk -F'|' 'NR>1 && $1 !~ /\./ {print $3}' "$P2/sacct.txt" | sort | uniq -c
        mkdir -p "$P2/failed_logs"
        for j in $(awk -F'|' 'NR>1 && $1 ~ /_[0-9]+$/ && $3 != "COMPLETED" && $3 != "RUNNING" && $3 != "PENDING" {print $1}' "$P2/sacct.txt"); do
            cp "slurm.purif.$j.out" "slurm.purif.$j.err" "$P2/failed_logs/" 2>/dev/null || true
        done
        cat "$P2/jobs.txt"
        echo "== priority blocks (attack array task indices)"; cat "$P2/tasks/priorities.txt"
    fi
    tar czf "$HOME/purification_phase2.tgz" --exclude='*.npz' -C outputs/purification phase2
    echo "== archive: $HOME/purification_phase2.tgz ($(du -h "$HOME/purification_phase2.tgz" | cut -f1))"
    exit 0
fi
if [[ "$MODE" != "queue" && "$MODE" != "--plan" ]]; then echo "usage: see the header"; exit 1; fi

: "${T_STARS:?set T_STARS (chosen in phase 1b), e.g. \"5 20 50\"}"
: "${STEPS:?set STEPS (chosen in phase 1)}"
: "${EOT:?set EOT (chosen in phase 1b)}"
: "${ITERS:?set ITERS (chosen in phase 1b)}"
: "${ALPHA:?set ALPHA, the PGD step as a fraction of the budget (phase 1b)}"
: "${MODIFIABLE:?set MODIFIABLE (detector = the phase-1b threat model)}"
BUDGETS="${BUDGETS:-0.05 0.10 0.02}"
TARGETS="${TARGETS:-512}"
BIG_TARGETS="${BIG_TARGETS:-2048}"
CURVE_T_STARS="${CURVE_T_STARS:-5 10 20 30 50 100 200}"
HOLD="${HOLD:-1}"
CONTEXT="${CONTEXT:-128}"
# purifier look-ahead: the value chosen by the phase 1 denoiser check (or an integer)
LOOKAHEAD="${LOOKAHEAD:-outputs/purification/phase1_v2/denoiser_check/check.json}"
read -r -a BUD <<< "$BUDGETS"
MAIN=${BUD[0]}
read -r -a TS <<< "$T_STARS"
TMAX=$(printf "%s\n" "${TS[@]}" | sort -n | tail -1)
for T in "${TS[@]}"; do if (( T <= 0 )); then echo "ERROR: T_STARS must be > 0 (t* = 0 is always run)"; exit 1; fi; done
if [[ "$MODIFIABLE" != "detector" && "$MODIFIABLE" != "window" ]]; then echo "ERROR: MODIFIABLE=detector|window"; exit 1; fi
ARCHS="ae cnn dnn lstm gru lin"
RUNS=()
for SEED in 7 21 42; do for M in $ARCHS; do RUNS+=("swatchrono_${M}_s${SEED}"); done; done
S7=(); for M in $ARCHS; do S7+=("swatchrono_${M}_s7"); done
AE7=swatchrono_ae_s7

kof() {  # $1 t*, $2 sampler: denoising steps of the purifier (as calibrate names them)
    if (( $1 == 0 )) || [[ "$2" == noise ]]; then echo 0; elif (( $1 < STEPS )); then echo "$1"; else echo "$STEPS"; fi
}
tag() { printf "tstar%04d_k%03d_%s" "$1" "$(kof "$1" "$2")" "$2"; }   # $1 t*, $2 sampler

if [[ "$MODE" == "queue" ]]; then
    git fetch -q origin
    if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]]; then
        echo "ERROR: HEAD is not origin/main. Pull (or push) first."; exit 1
    fi
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "ERROR: tracked files have uncommitted changes"; git status --short | grep -v '^??'; exit 1
    fi
    if [[ ! -f "$DIFF/weights.npz" ]]; then echo "ERROR: $DIFF not trained (phase 1)"; exit 1; fi
    if [[ ! "$LOOKAHEAD" =~ ^[0-9]+$ && ! -f "$LOOKAHEAD" ]]; then echo "ERROR: missing $LOOKAHEAD"; exit 1; fi
    for RUN in "${RUNS[@]}"; do
        if [[ ! -d "models/$RUN" ]]; then echo "ERROR: missing models/$RUN"; exit 1; fi
    done
    for T in 0 20 50; do
        F="$CAL1/defense_$(tag $T ddpm).json"
        if [[ ! -f "$F" ]]; then echo "ERROR: missing $F (phase 1_v2 calibration, used by p0)"; exit 1; fi
    done
    if [[ -n "$(squeue -u "$USER" -h)" ]]; then echo "ERROR: you still have jobs in the queue."; exit 1; fi
    if [[ -e "$P2/jobs.txt" ]]; then echo "ERROR: $P2/jobs.txt exists: phase 2 was already queued."; exit 1; fi
    TD=$P2/tasks
else
    TD=$P2/plan
    rm -rf "$TD"
fi
mkdir -p "$TD"

# ---- p0: diagnostic on the tuning split (phase-1_v2 defenses, as phase 1b) ---------------
: > "$TD/p0.txt"
p0() {  # $1 t*, $2 budget, $3 goal, rest = extra arguments
    local T=$1 B=$2 G=$3 E=$EOT; shift 3
    (( T == 0 )) && E=1
    local O=$P2/p0_tune/range$B/t$T/full_a${ALPHA}_eot${E}_it${ITERS}$(echo "$*" | tr -d ' -')/$G
    echo "$O main_purification.py attack --dataset SWAT-CLEAN --model-dir models/$AE7 --split tune --max-targets 512 --eval-draws 10 --trace-every 10 --seed 42 --modifiable detector --protect-actuators --defense $CAL1/defense_$(tag "$T" ddpm).json --epsilon-range-fraction $B --goal $G --alpha-relative $ALPHA --iterations $ITERS --gradient full --eot $E $* --out $O" >> "$TD/p0.txt"
}
p0 0 0.05 evasion; p0 0 0.05 false_alarm; p0 0 0.02 evasion; p0 0 0.10 evasion
p0 0 0.05 evasion --loss log
p0 20 0.05 evasion; p0 50 0.05 evasion; p0 50 0.05 false_alarm

# ---- calibrations ---------------------------------------------------------------------
: > "$TD/calib.txt"
cal() {  # $1 run, $2 folder, $3 sampler, $4 look-ahead, $5 t* list
    local L; L=$(printf "%s\n" $5 | sort -n | uniq | paste -sd' ' -)
    echo "$2 main_purification.py calibrate --dataset SWAT-CLEAN --model-dir models/$1 --diffusion $DIFF --context $CONTEXT --lookahead $4 --steps $STEPS --sampler $3 --t-star $L --draws 10 --out $2" >> "$TD/calib.txt"
}
for RUN in "${RUNS[@]}"; do
    L="0 $T_STARS"; [[ "$RUN" == "$AE7" ]] && L="$L $CURVE_T_STARS"
    cal "$RUN" "$P2/$RUN/calib" ddpm "$LOOKAHEAD" "$L"
done
cal $AE7 "$P2/$AE7/calib_noise" noise "$LOOKAHEAD" "0 $T_STARS"
cal $AE7 "$P2/$AE7/calib_causal" ddpm 0 "0 $T_STARS"
NCAL=$(wc -l < "$TD/calib.txt")

# ---- attacks, in order of priority -------------------------------------------------------
: > "$TD/attacks.txt"
: > "$TD/priorities.txt"
# att <run> <calib folder> <sampler> <t*> <budget> <goal> <label> <attack arguments...>
att() {
    local RUN=$1 CAL=$2 SM=$3 T=$4 B=$5 G=$6 L=$7; shift 7
    local O=$P2/$RUN/range$B/t$T/$L/$G
    echo "$O main_purification.py attack --dataset SWAT-CLEAN --model-dir models/$RUN --split final --max-targets $TARGETS --eval-draws 10 --trace-every 10 --seed 42 --defense $CAL/defense_$(tag "$T" "$SM").json --require-reproduction $CAL/$(tag 0 "$SM").json --epsilon-range-fraction $B --goal $G $* --out $O" >> "$TD/attacks.txt"
}
THREAT="--protect-actuators --modifiable $MODIFIABLE"
std() {  # <run> <t*> <budget> <goal> [label prefix] [calib folder] [sampler] [extra...]: the P1 attack
    local RUN=$1 T=$2 B=$3 G=$4 PRE=${5:-} CAL=${6:-$P2/$1/calib} SM=${7:-ddpm}; shift 4; shift $(( $# < 3 ? $# : 3 ))
    local E=$EOT; (( T == 0 )) && E=1
    att "$RUN" "$CAL" "$SM" "$T" "$B" "$G" "${PRE}full_a${ALPHA}_eot${E}_it${ITERS}" $THREAT --alpha-relative "$ALPHA" --iterations "$ITERS" --gradient full --eot "$E" "$@"
}
arms() {  # <run> <budget>
    local T G
    for T in 0 "${TS[@]}"; do for G in evasion false_alarm; do std "$1" "$T" "$2" $G; done; done
}
block() {
    local n; n=$(wc -l < "$TD/attacks.txt")
    if (( n > ${START:-0} )); then echo "$1 ${START:-0}-$((n - 1))" >> "$TD/priorities.txt"; else echo "$1 (empty)" >> "$TD/priorities.txt"; fi
    START=$n
}

# P1
for RUN in "${RUNS[@]}"; do arms "$RUN" "$MAIN"; done
block "P1_main_budget_${MAIN}_all18"
# P2
for RUN in "${S7[@]}"; do
    C=$P2/$RUN/calib
    S="$THREAT --alpha-relative $ALPHA --iterations $ITERS"
    for G in evasion false_alarm; do
        att "$RUN" "$C" ddpm 0 "$MAIN" $G "full_a${ALPHA}_eot1_it${ITERS}_log" $S --gradient full --eot 1 --loss log
    done
    att "$RUN" "$C" ddpm 0 "$MAIN" evasion "full_a${ALPHA}_eot1_it${ITERS}_R3" $S --gradient full --eot 1 --restarts 3
    for T in "${TS[@]}"; do
        KS=$(( $(kof "$T" ddpm) / 2 )); (( KS < 1 )) && KS=1
        att "$RUN" "$C" ddpm "$T" "$MAIN" evasion "surrogate_k${KS}_a${ALPHA}_eot${EOT}_it${ITERS}" $S --gradient surrogate --attack-steps $KS --eot "$EOT"
        for GR in bpda blind; do
            for G in evasion false_alarm; do
                att "$RUN" "$C" ddpm "$T" "$MAIN" $G "${GR}_a${ALPHA}_eot${EOT}_it${ITERS}" $S --gradient $GR --eot "$EOT"
            done
        done
        if (( EOT != 1 )); then
            att "$RUN" "$C" ddpm "$T" "$MAIN" evasion "full_a${ALPHA}_eot1_it${ITERS}" $S --gradient full --eot 1
        fi
        for G in evasion false_alarm; do
            att "$RUN" "$C" ddpm "$T" "$MAIN" $G "full_a${ALPHA}_eot${EOT}_it${ITERS}_log" $S --gradient full --eot "$EOT" --loss log
        done
    done
    att "$RUN" "$C" ddpm "$TMAX" "$MAIN" evasion "full_a${ALPHA}_eot${EOT}_it${ITERS}_R3" $S --gradient full --eot "$EOT" --restarts 3
done
block "P2_attack_strength_seed7"
# P3
C=$P2/$AE7/calib
for T in 0 "${TS[@]}"; do
    E=$EOT; (( T == 0 )) && E=1
    att $AE7 "$C" ddpm "$T" "$MAIN" evasion "actfree_full_a${ALPHA}_eot${E}_it${ITERS}" --modifiable "$MODIFIABLE" --alpha-relative "$ALPHA" --iterations "$ITERS" --gradient full --eot "$E"
done
OTHER=window; [[ "$MODIFIABLE" == window ]] && OTHER=detector
for T in "${TS[@]}"; do
    for G in evasion false_alarm; do
        att $AE7 "$C" ddpm "$T" "$MAIN" $G "rows${OTHER}_full_a${ALPHA}_eot${EOT}_it${ITERS}" --protect-actuators --modifiable $OTHER --alpha-relative "$ALPHA" --iterations "$ITERS" --gradient full --eot "$EOT"
        std $AE7 "$T" "$MAIN" $G "noise_" "$P2/$AE7/calib_noise" noise
        std $AE7 "$T" "$MAIN" $G "causal_" "$P2/$AE7/calib_causal" ddpm
    done
done
for G in evasion false_alarm; do std $AE7 0 "$MAIN" $G "causal_" "$P2/$AE7/calib_causal" ddpm; done
for RUN in "${S7[@]}"; do
    att "$RUN" "$P2/$RUN/calib" ddpm 0 1.0 evasion "free_full_a0.01_noRS_eot1_it200" --modifiable "$MODIFIABLE" --alpha-relative 0.01 --no-random-start --iterations 200 --gradient full --eot 1
done
block "P3_mechanism_ablations"
# P4
for T in $CURVE_T_STARS; do
    if [[ " ${TS[*]} " == *" $T "* ]]; then continue; fi
    for G in evasion false_alarm; do std $AE7 "$T" "$MAIN" $G; done
done
block "P4_curve_ae_s7"
# P5
for RUN in "${S7[@]}"; do
    att "$RUN" "$P2/$RUN/calib" ddpm 0 "$MAIN" evasion "full_a0.025_eot1_it500" $THREAT --alpha-relative 0.025 --iterations 500 --gradient full --eot 1
    att "$RUN" "$P2/$RUN/calib" ddpm "$TMAX" "$MAIN" evasion "full_a0.025_eot${EOT}_it500" $THREAT --alpha-relative 0.025 --iterations 500 --gradient full --eot "$EOT"
done
block "P5_convergence_seed7"
# P6
for T in 0 "$TMAX"; do
    for G in evasion false_alarm; do
        std $AE7 "$T" "$MAIN" $G "n${BIG_TARGETS}_" "$P2/$AE7/calib" ddpm
        sed -i '$ s/--max-targets '"$TARGETS"' /--max-targets '"$BIG_TARGETS"' /' "$TD/attacks.txt"
    done
done
block "P6_sample_size_ae_s7"
# P7
for B in "${BUD[@]:1}"; do for RUN in "${RUNS[@]}"; do arms "$RUN" "$B"; done; done
block "P7_other_budgets_all18"

N=$(wc -l < "$TD/attacks.txt")
# every task must write where the worker expects it (first field = --out), and every
# line must be accepted by main_purification.py's argument parser
for F in p0 calib attacks; do
    BAD=$(awk '{ ok = 0; for (i = 2; i < NF; i++) if ($i == "--out" && $(i + 1) == $1) ok = 1; if (!ok) print NR }' "$TD/$F.txt" | head -3)
    if [[ -n "$BAD" ]]; then echo "ERROR: $F.txt lines $BAD lack --out <their folder>"; exit 1; fi
done
if command -v python3 > /dev/null && python3 -c "import tensorflow" 2> /dev/null; then
    python3 -m purification.check_tasks "$TD/p0.txt" "$TD/calib.txt" "$TD/attacks.txt" || exit 1
else
    echo "note: no TensorFlow here, task arguments not parsed (the --out check passed)"
fi
for F in p0 attacks; do
    if [[ $(awk '{print $1}' "$TD/$F.txt" | sort | uniq -d | wc -l) -ne 0 ]]; then
        echo "ERROR: duplicate output folders in $F"; awk '{print $1}' "$TD/$F.txt" | sort | uniq -d | head; exit 1
    fi
done
CFG="T_STARS=$T_STARS STEPS=$STEPS EOT=$EOT ITERS=$ITERS ALPHA=$ALPHA MODIFIABLE=$MODIFIABLE BUDGETS=$BUDGETS TARGETS=$TARGETS BIG_TARGETS=$BIG_TARGETS CURVE_T_STARS=$CURVE_T_STARS CONTEXT=$CONTEXT LOOKAHEAD=$LOOKAHEAD"
echo "$CFG"
echo "p0: $(wc -l < "$TD/p0.txt") diagnostic attacks; $NCAL calibrations; $N attacks; priority blocks (attack array task indices):"
sed 's/^/  /' "$TD/priorities.txt"
if [[ "$MODE" == "--plan" ]]; then echo "plan only, nothing queued; task files in $TD"; exit 0; fi

echo "commit $(git rev-parse --short HEAD), $(date -Is); $CFG HOLD=$HOLD" > "$P2/commit.txt"
sub() { local id; id=$(sbatch --parsable "$@"); echo "${id%%;*}"; }
J0=$(sub --array=0-$(( $(wc -l < "$TD/p0.txt") - 1 ))%3 --time=04:00:00 --mail-type=END,FAIL \
    --export=ALL,TASKS="$REPO/$TD/p0.txt" purification.sbatch)
echo "p0 $J0" > "$P2/jobs.txt"
C=$(sub --array=0-$((NCAL - 1))%3 --time=08:00:00 --mail-type=FAIL \
    --export=ALL,TASKS="$REPO/$TD/calib.txt" purification.sbatch)
echo "calib $C" >> "$P2/jobs.txt"
H=(); (( HOLD )) && H=(--hold)
A=$(sub "${H[@]}" --array=0-$((N - 1))%3 --time=12:00:00 --mail-type=END,FAIL \
    --dependency=afterany:"$C" --export=ALL,TASKS="$REPO/$TD/attacks.txt" purification.sbatch)
echo "attacks $A" >> "$P2/jobs.txt"
echo "Queued: diagnostic p0 (job $J0), calibrations (job $C), $N attacks (job $A, after the calibrations)."
if (( HOLD )); then
    echo "The attacks are ON HOLD. When p0 has finished: bash purification_phase2.sh --report,"
    echo "send the archive for the check, then:  scontrol release $A"
fi
echo "To drop a priority block later: scancel ${A}_[<first>-<last>] with the indices above."
