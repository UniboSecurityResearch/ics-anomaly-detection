#!/bin/bash
#SBATCH --job-name=purif
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=dhyana.schirra@studio.unibo.it
#SBATCH --chdir=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
#SBATCH --output=slurm.purif.%A_%a.out
#SBATCH --error=slurm.purif.%A_%a.err

# Generic worker for the diffusion-purification experiments (purification_phase*.sh
# queues it). TASKS=<file>: line number SLURM_ARRAY_TASK_ID (0-based) is
#   <output folder> <python script> <arguments...>
# The output folder gets the git commit, the uncommitted diff (must be empty: the
# launcher refuses to queue otherwise) and the full log. A task whose folder already
# has DONE is skipped, so a failed array can be resubmitted as it is.

set -euo pipefail
REPO=/scratch.hpc/dhyana.schirra/tesi/ics-anomaly-detection
IMAGE="$REPO/sip.sif"
: "${TASKS:?TASKS=<task file> is required}"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS")
if [[ -z "$LINE" ]]; then echo "ERROR: no line $SLURM_ARRAY_TASK_ID in $TASKS"; exit 1; fi
read -r OUTDIR SCRIPT ARGS <<< "$LINE"

GPU_IDS="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "$GPU_IDS" || "$GPU_IDS" == "-1" || "$GPU_IDS" == "NoDevFiles" ]]; then
    echo "ERROR: Slurm did not assign a GPU"; exit 1
fi
if [[ -e "$REPO/$OUTDIR/DONE" ]]; then echo "SKIP: $OUTDIR already done"; exit 0; fi
mkdir -p "$REPO/$OUTDIR.meta"
git -C "$REPO" rev-parse HEAD > "$REPO/$OUTDIR.meta/git_commit.txt"
git -C "$REPO" diff > "$REPO/$OUTDIR.meta/code_changes.patch"
echo "TASK $SLURM_ARRAY_TASK_ID: $LINE"
echo "HOST=$(hostname) GPU=$GPU_IDS START=$(date -Is)"

set +e
/usr/bin/apptainer exec --nv \
    --bind "$REPO:$REPO" --pwd "$REPO" \
    --env "PYTHONPATH=$REPO" \
    --env "CUDA_VISIBLE_DEVICES=$GPU_IDS" \
    --env "TF_FORCE_GPU_ALLOW_GROWTH=true" \
    --env "TF_XLA_FLAGS=--tf_xla_auto_jit=0" \
    --env "TF_CPP_MIN_LOG_LEVEL=1" \
    --env "PYTHONUNBUFFERED=1" \
    "$IMAGE" \
    python3 -u $SCRIPT $ARGS 2>&1 | tee "$REPO/$OUTDIR.meta/run.log"
STATUS=${PIPESTATUS[0]}
set -e
echo "EXIT $STATUS $(date -Is)"
if [[ $STATUS -eq 0 ]]; then mkdir -p "$REPO/$OUTDIR" && touch "$REPO/$OUTDIR/DONE"; fi
exit $STATUS
