#!/bin/bash
# Conclusive round of experiments (October 2026). Run on the login node, from
# anywhere:
#   bash final_round.sh            # checks the code state, then queues every job
#   bash final_round.sh --report   # when all jobs are done: tables + one archive
#
# Jobs queued (partition l40s, at most 3 running at a time for the user):
#   1. retune  task 0     threshold/window choice on a wider grid, AE seed 7 only:
#                         must reproduce the stored values, otherwise step 2 is cancelled
#   2. retune  tasks 1-17 the other 17 detectors (after step 1 succeeds)
#   3. per_attack_chrono  per-attack detection of the 18 detectors (clean data)
#   4. CorrShift FEVA at 0.01 of the range (the paper's coefficient), 15 detectors
#   5. linear model: full sweep (6 budgets x 3 seeds, FGSM, PGD x 3 steps, top-k, CorrShift)
#   6. linear model: CorrShift FEVA at the 6 budgets and at 0.01 (after step 5)
#   7. linear model: 500-iteration PGD at range 0.20 (after step 5), as for the others
# Nothing already computed is overwritten: every job writes new folders.
set -euo pipefail
REPO=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
cd "$REPO"
OUT="$REPO/outputs/final_round"

if [[ "${1:-}" == "--report" ]]; then
    mkdir -p "$OUT"
    # each step runs even if another fails; errors end up in the text files
    python3 sweep_analysis.py --out "$OUT/analysis" > "$OUT/sweep_analysis.txt" 2>&1 \
        || echo "WARNING: sweep_analysis.py failed, see $OUT/sweep_analysis.txt"
    python3 retune_thresholds.py --summary > "$OUT/retune_summary.txt" 2>&1 \
        || echo "WARNING: retune summary failed, see $OUT/retune_summary.txt"
    cp slurm.retune.*.out "$OUT/" 2>/dev/null || true
    cp outputs/per_attack_chrono/summary.txt "$OUT/per_attack_summary.txt" 2>/dev/null || true
    cp outputs/per_attack_chrono/summary_final.csv "$OUT/" 2>/dev/null || true
    if [[ -f "$OUT/jobs.txt" ]]; then
        IDS=$(awk '{print $2}' "$OUT/jobs.txt" | paste -sd, -)
        sacct -j "$IDS" -X --format=JobID%16,JobName%14,State%11,Start,Elapsed > "$OUT/sacct.txt"
        echo "== job states"; awk 'NR>2 {print $3}' "$OUT/sacct.txt" | sort | uniq -c
    fi
    tar czf "$HOME/final_round.tgz" -C "$REPO/outputs" final_round
    echo "== retune"; cat "$OUT/retune_summary.txt"
    echo "== archive: $HOME/final_round.tgz ($(du -h "$HOME/final_round.tgz" | cut -f1))"
    exit 0
fi

# --- checks: the code that runs must be exactly the published commit ----------
git fetch -q origin
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]]; then
    echo "ERROR: HEAD is not origin/main. Commit and push first."; exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked files have uncommitted changes:"; git status --short | grep -v '^??'; exit 1
fi
if [[ -n "$(squeue -u "$USER" -h)" ]]; then
    echo "ERROR: you still have jobs in the queue; wait until they finish."; exit 1
fi
if [[ -e "$OUT/jobs.txt" ]]; then
    echo "ERROR: $OUT/jobs.txt exists: this round was already queued."; exit 1
fi
mkdir -p "$OUT"
echo "commit $(git rev-parse --short HEAD), $(date -Is)" > "$OUT/commit.txt"

submit() {  # $1 = label, rest = sbatch arguments; prints the job id
    local label="$1"; shift
    local id
    id=$(sbatch --parsable "$@")
    id="${id%%;*}"
    echo "$label $id" >> "$OUT/jobs.txt"
    echo "queued  $label  job $id" >&2
    echo "$id"
}

RT0=$(submit retune_check     --array=0 retune_thresholds.sbatch)
RT=$(submit  retune_rest      --array=1-17%3 --dependency=afterok:"$RT0" --kill-on-invalid-dep=yes retune_thresholds.sbatch)
PA=$(submit  per_attack       per_attack_chrono.sbatch)
F01=$(submit feva_0.01        --array=0-84:6%3 --export=ALL,ONLY=corrshift_feva,RANGE_FRACTION=0.01 swat_sweep.sbatch)
LIN=$(submit lin_sweep        --array=0-17%3 --export=ALL,SWEEP_MODELS=lin swat_sweep.sbatch)
LF=$(submit  lin_feva         --array=0-17%3 --dependency=afterany:"$LIN" --export=ALL,SWEEP_MODELS=lin,ONLY=corrshift_feva swat_sweep.sbatch)
LF01=$(submit lin_feva_0.01   --array=0,6,12 --dependency=afterany:"$LIN" --export=ALL,SWEEP_MODELS=lin,ONLY=corrshift_feva,RANGE_FRACTION=0.01 swat_sweep.sbatch)
LL=$(submit  lin_long_0.20    --array=3,9,15 --dependency=afterany:"$LIN" --export=ALL,SWEEP_MODELS=lin,ONLY="pgd_r0.05_it500 pgd_r0.025_it500" swat_sweep.sbatch)

echo
echo "All jobs queued; ids in $OUT/jobs.txt. Follow them with:"
echo "  squeue -u $USER -o \"%.18i %.9T %.10M %.14R %j\""
echo "When the queue is empty: bash final_round.sh --report"
