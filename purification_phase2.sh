#!/bin/bash
# Diffusion purification, PHASE 2: the configuration chosen in phase 1 / 1b (tuning
# split, AE seed 7) applied to all 18 chronological SWAT-CLEAN detectors and evaluated
# on the FINAL split of the test file (rows from 134975), under the phase-1b threat
# model (the same modifiable rows for every arm).
#
# The values chosen in phase 1b are REQUIRED (no defaults, so phase 2 cannot run with
# values that were not chosen):
#   T_STARS="<t* ...>" STEPS=<K> EOT=<samples> ITERS=<PGD iterations> \
#   ALPHA=<step / budget> MODIFIABLE=<detector|window> bash purification_phase2.sh
#   ... bash purification_phase2.sh --plan     # writes and counts the tasks, queues nothing
#   bash purification_phase2.sh --report       # any time: tables of what has finished + archive
# Optional: BUDGETS (default "0.05 0.10 0.02"; the FIRST is the main budget, the
# others are run in the order given), TARGETS (default 512, as in phase 1),
# CURVE_T_STARS (default "5 10 20 30 50 100 200": cost/robustness curve, AE seed 7
# only), CONTEXT, LOOKAHEAD, SAMPLER.
#
# One calibration array (one task per detector: t* = 0, which must reproduce the stored
# detector, else exit 3, plus T_STARS; AE seed 7 also CURVE_T_STARS), then ONE attack
# array in order of priority, started when every calibration has ended. An attack
# whose detector failed the t* = 0 reproduction refuses to run (exit 3).
# Detectors are always taken seed by seed (all six architectures of seed 7, then 21,
# then 42), so a priority block stopped early still covers every architecture.
#   P1  main budget: undefended (t* = 0, PGD) and adaptive (full gradient + EOT) at
#       every T_STARS, evasion and false alarm, all 18 detectors     -> results A, B
#   P2  the other budgets, same arms, all 18 detectors                -> B (budget curve)
#   P3  seed 7, every architecture, main budget, every T_STARS: surrogate (K/2 steps),
#       BPDA, blind, EOT 1                                            -> D
#   P4  AE seed 7, main budget, CURVE_T_STARS not in T_STARS          -> C
#   P5  seed 7, every architecture, main budget, t* = 0 and max(T_STARS), evasion:
#       step 0.025 x budget with 500 iterations                       -> D (convergence)
#   P6  diagnostics: AE seed 7 with actuators FREE (is the evasion floor the protected
#       actuators?); AE seed 7 with the stronger attacker (MODIFIABLE=window); whole
#       range from the clean series at t* = 0 for every seed-7 architecture (sanity)
set -euo pipefail
REPO="${REPO:-/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection}"
cd "$REPO"
P2=outputs/purification/phase2
DIFF=models/diffusion_swat_v1
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
        echo "== priority blocks (attack array task ranges)"; cat "$P2/tasks/priorities.txt"
    fi
    tar czf "$HOME/purification_phase2.tgz" --exclude='*.npz' -C outputs/purification phase2
    echo "== archive: $HOME/purification_phase2.tgz ($(du -h "$HOME/purification_phase2.tgz" | cut -f1))"
    exit 0
fi
if [[ "$MODE" != "queue" && "$MODE" != "--plan" ]]; then echo "usage: see the header"; exit 1; fi

: "${T_STARS:?set T_STARS (chosen in phase 1b), e.g. \"5 50\"}"
: "${STEPS:?set STEPS (chosen in phase 1)}"
: "${EOT:?set EOT (chosen in phase 1b)}"
: "${ITERS:?set ITERS (chosen in phase 1b)}"
: "${ALPHA:?set ALPHA, the PGD step as a fraction of the budget (phase 1b)}"
: "${MODIFIABLE:?set MODIFIABLE (detector = the phase-1b threat model)}"
BUDGETS="${BUDGETS:-0.05 0.10 0.02}"
TARGETS="${TARGETS:-512}"
CURVE_T_STARS="${CURVE_T_STARS:-5 10 20 30 50 100 200}"
CONTEXT="${CONTEXT:-128}"
# purifier look-ahead: the value chosen by the phase 1 denoiser check (or an integer)
LOOKAHEAD="${LOOKAHEAD:-outputs/purification/phase1_v2/denoiser_check/check.json}"
SAMPLER="${SAMPLER:-ddpm}"
read -r -a BUD <<< "$BUDGETS"
MAIN=${BUD[0]}
read -r -a TS <<< "$T_STARS"
TMAX=$(printf "%s\n" "${TS[@]}" | sort -n | tail -1)
for T in "${TS[@]}"; do if (( T <= 0 )); then echo "ERROR: T_STARS must be > 0 (t* = 0 is always run)"; exit 1; fi; done
if [[ "$MODIFIABLE" != "detector" && "$MODIFIABLE" != "window" ]]; then echo "ERROR: MODIFIABLE=detector|window"; exit 1; fi
ARCHS="ae cnn dnn lstm gru lin"
RUNS=()
for SEED in 7 21 42; do for M in $ARCHS; do RUNS+=("swatchrono_${M}_s${SEED}"); done; done

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
    if [[ -n "$(squeue -u "$USER" -h)" ]]; then echo "ERROR: you still have jobs in the queue."; exit 1; fi
    if [[ -e "$P2/jobs.txt" ]]; then echo "ERROR: $P2/jobs.txt exists: phase 2 was already queued."; exit 1; fi
    TD=$P2/tasks
else
    TD=$(mktemp -d)
fi
mkdir -p "$TD"

kof() { local k=$STEPS; if (( $1 == 0 )); then k=0; elif (( $1 < k )); then k=$1; fi; echo "$k"; }
tag() { printf "tstar%04d_k%03d_%s" "$1" "$(kof "$1")" "$SAMPLER"; }

# ---- calibrations ---------------------------------------------------------------
: > "$TD/calib.txt"
for RUN in "${RUNS[@]}"; do
    LIST="0 $T_STARS"
    if [[ "$RUN" == swatchrono_ae_s7 ]]; then LIST="$LIST $CURVE_T_STARS"; fi
    LIST=$(printf "%s\n" $LIST | sort -n | uniq | paste -sd' ' -)
    CAL=$P2/$RUN/calib
    echo "$CAL main_purification.py calibrate --dataset SWAT-CLEAN --model-dir models/$RUN --diffusion $DIFF --context $CONTEXT --lookahead $LOOKAHEAD --steps $STEPS --sampler $SAMPLER --t-star $LIST --draws 10 --out $CAL" >> "$TD/calib.txt"
done

# ---- attacks, in order of priority -------------------------------------------------
: > "$TD/attacks.txt"
: > "$TD/priorities.txt"
att() {  # $1 run, $2 t*, $3 budget, $4 goal, $5 folder label, rest = attack arguments
    local RUN=$1 T=$2 B=$3 G=$4 L=$5; shift 5
    local CAL=$P2/$RUN/calib
    local O=$P2/$RUN/range$B/t$T/$L/$G
    echo "$O main_purification.py attack --dataset SWAT-CLEAN --model-dir models/$RUN --split final --max-targets $TARGETS --eval-draws 10 --trace-every 10 --seed 42 --defense $CAL/defense_$(tag "$T").json --require-reproduction $CAL/$(tag 0).json --epsilon-range-fraction $B --goal $G $*" >> "$TD/attacks.txt"
}
THREAT="--protect-actuators --modifiable $MODIFIABLE"
arms() {  # $1 run, $2 budget: undefended + adaptive at every T_STARS, both goals
    local RUN=$1 B=$2 T G
    for T in 0 "${TS[@]}"; do
        for G in evasion false_alarm; do
            if (( T == 0 )); then
                att "$RUN" 0 "$B" $G "full_a${ALPHA}_eot1_it${ITERS}" $THREAT --alpha-relative "$ALPHA" --iterations "$ITERS" --gradient full --eot 1
            else
                att "$RUN" "$T" "$B" $G "full_a${ALPHA}_eot${EOT}_it${ITERS}" $THREAT --alpha-relative "$ALPHA" --iterations "$ITERS" --gradient full --eot "$EOT"
            fi
        done
    done
}
block() {  # $1 = label: record the task range written since the previous block
    local n; n=$(wc -l < "$TD/attacks.txt")
    if (( n > ${START:-0} )); then echo "$1 ${START:-0}-$((n - 1))" >> "$TD/priorities.txt"; else echo "$1 (empty)" >> "$TD/priorities.txt"; fi
    START=$n
}
S7=(); for M in $ARCHS; do S7+=("swatchrono_${M}_s7"); done

# P1
for RUN in "${RUNS[@]}"; do arms "$RUN" "$MAIN"; done
block "P1_main_budget_${MAIN}"
# P2
for B in "${BUD[@]:1}"; do for RUN in "${RUNS[@]}"; do arms "$RUN" "$B"; done; done
block "P2_other_budgets"
# P3
for RUN in "${S7[@]}"; do
    for T in "${TS[@]}"; do
        KS=$(( $(kof "$T") / 2 )); (( KS < 1 )) && KS=1
        S="$THREAT --alpha-relative $ALPHA --iterations $ITERS"
        att "$RUN" "$T" "$MAIN" evasion "surrogate_k${KS}_a${ALPHA}_eot${EOT}_it${ITERS}" $S --gradient surrogate --attack-steps $KS --eot "$EOT"
        for GR in bpda blind; do
            for G in evasion false_alarm; do
                att "$RUN" "$T" "$MAIN" $G "${GR}_a${ALPHA}_eot${EOT}_it${ITERS}" $S --gradient $GR --eot "$EOT"
            done
        done
        if (( EOT != 1 )); then
            att "$RUN" "$T" "$MAIN" evasion "full_a${ALPHA}_eot1_it${ITERS}" $S --gradient full --eot 1
        fi
    done
done
block "P3_attackers_seed7"
# P4
for T in $CURVE_T_STARS; do
    if [[ " ${TS[*]} " == *" $T "* ]]; then continue; fi
    for G in evasion false_alarm; do
        att swatchrono_ae_s7 "$T" "$MAIN" $G "full_a${ALPHA}_eot${EOT}_it${ITERS}" $THREAT --alpha-relative "$ALPHA" --iterations "$ITERS" --gradient full --eot "$EOT"
    done
done
block "P4_curve_ae_s7"
# P5
for RUN in "${S7[@]}"; do
    att "$RUN" 0 "$MAIN" evasion "full_a0.025_eot1_it500" $THREAT --alpha-relative 0.025 --iterations 500 --gradient full --eot 1
    att "$RUN" "$TMAX" "$MAIN" evasion "full_a0.025_eot${EOT}_it500" $THREAT --alpha-relative 0.025 --iterations 500 --gradient full --eot "$EOT"
done
block "P5_convergence_seed7"
# P6
for T in 0 "${TS[@]}"; do
    E=$EOT; (( T == 0 )) && E=1
    att swatchrono_ae_s7 "$T" "$MAIN" evasion "actfree_full_a${ALPHA}_eot${E}_it${ITERS}" --modifiable "$MODIFIABLE" --alpha-relative "$ALPHA" --iterations "$ITERS" --gradient full --eot "$E"
done
OTHER=window; [[ "$MODIFIABLE" == window ]] && OTHER=detector
for T in "${TS[@]}"; do
    for G in evasion false_alarm; do
        att swatchrono_ae_s7 "$T" "$MAIN" $G "rows${OTHER}_full_a${ALPHA}_eot${EOT}_it${ITERS}" --protect-actuators --modifiable $OTHER --alpha-relative "$ALPHA" --iterations "$ITERS" --gradient full --eot "$EOT"
    done
done
for RUN in "${S7[@]}"; do
    att "$RUN" 0 1.0 evasion "free_full_a0.01_noRS_eot1_it200" --modifiable "$MODIFIABLE" --alpha-relative 0.01 --no-random-start --iterations 200 --gradient full --eot 1
done
block "P6_diagnostics"

N=$(wc -l < "$TD/attacks.txt")
if [[ $(awk '{print $1}' "$TD/attacks.txt" | sort | uniq -d | wc -l) -ne 0 ]]; then
    echo "ERROR: duplicate output folders in the task list"; awk '{print $1}' "$TD/attacks.txt" | sort | uniq -d | head; exit 1
fi
echo "T_STARS=$T_STARS STEPS=$STEPS EOT=$EOT ITERS=$ITERS ALPHA=$ALPHA MODIFIABLE=$MODIFIABLE BUDGETS=$BUDGETS TARGETS=$TARGETS CURVE_T_STARS=$CURVE_T_STARS"
echo "18 calibrations, $N attacks; priority blocks (attack array task indices):"
sed 's/^/  /' "$TD/priorities.txt"
if [[ "$MODE" == "--plan" ]]; then echo "plan only, nothing queued; task files in $TD"; exit 0; fi

echo "commit $(git rev-parse --short HEAD), $(date -Is); T_STARS=$T_STARS STEPS=$STEPS EOT=$EOT ITERS=$ITERS ALPHA=$ALPHA MODIFIABLE=$MODIFIABLE BUDGETS=$BUDGETS TARGETS=$TARGETS CURVE_T_STARS=$CURVE_T_STARS CONTEXT=$CONTEXT LOOKAHEAD=$LOOKAHEAD SAMPLER=$SAMPLER" > "$P2/commit.txt"
C=$(sbatch --parsable --array=0-17%3 --time=08:00:00 --mail-type=FAIL \
    --export=ALL,TASKS="$REPO/$TD/calib.txt" purification.sbatch); C="${C%%;*}"
echo "calib $C" > "$P2/jobs.txt"
A=$(sbatch --parsable --array=0-$((N - 1))%3 --time=12:00:00 --mail-type=END,FAIL \
    --dependency=afterany:"$C" --export=ALL,TASKS="$REPO/$TD/attacks.txt" purification.sbatch); A="${A%%;*}"
echo "attacks $A" >> "$P2/jobs.txt"
echo "Queued calibrations (job $C) and $N attacks (job $A, starts when every calibration has ended)."
echo "Report (also while running): bash purification_phase2.sh --report"
echo "To drop a priority block later: scancel ${A}_[<first>-<last>] with the indices above."
