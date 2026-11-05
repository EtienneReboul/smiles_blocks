#!/bin/bash
# Stage A on CAIUS: retrosynthetic search over all of MOSES, as a SLURM array.
#
# Prerequisite: run hpc_scripts/caius_setup_aizynth_env.sh once on a login node.
#
#   sbatch hpc_scripts/caius_aizynth_cut_bonds.sh
#
# Work unit: one array task per smiles_data part. Each part holds ~57 M
# randomised SMILES rows but only ~16 k UNIQUE reference SMILES, which is what
# gets searched:
#
#   16 000 molecules x 11.5 s           = ~51 core-hours per part
#   51 core-hours / 64 cores            = ~48 min wall per task
#   119 parts                           = ~6 100 core-hours total
#
# A GPU would not help: the MCTS itself is CPU-bound and the policy network runs
# on ONNX Runtime.
#
# NOTE the array range is 0-118, not 1-118. The parts are named part-0 ...
# part-118, so the 1-118 used by the other scripts in this directory silently
# skips part-0.

#SBATCH --job-name="aizynth_cut_bonds"
#SBATCH --array=0-118
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=0-04:00
#SBATCH --output=logs/aizynth_cut_bonds/job_%A/task_%a.out
#SBATCH --error=logs/aizynth_cut_bonds/job_%A/task_%a.err

# ============================ CONFIGURE ME ============================
# Nothing below the next block should need editing.
#
#SBATCH --account=CHANGE_ME
#SBATCH --partition=CHANGE_ME
#
# Memory: each worker loads its own copy of the expansion model, the template
# library and the stock. The stock in particular is large, so peak RSS per
# worker is the quantity that decides how many workers fit on a node.
# MEASURE IT BEFORE SUBMITTING THE FULL ARRAY -- see the pilot command in the
# deployment notes. Start conservative and tighten afterwards.
#SBATCH --mem-per-cpu=4G

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/smiles_blocks}"
ENV_PREFIX="${ENV_PREFIX:-$PROJECT_DIR/envs/aizynth}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models/aizynthfinder}"

# How to get conda on PATH on a compute node. Same value as in the setup script.
CONDA_SETUP_CMD="${CONDA_SETUP_CMD:-}"

# Reaction metadata keys used to label a disconnection, in order of preference.
# Which keys the model actually populates is NOT guaranteed by the library --
# take this from what caius_setup_aizynth_env.sh reported.
TAG_KEYS="${TAG_KEYS:-classification,template_hash,template_code,policy_name}"
# ======================================================================

set -euo pipefail

cd "$PROJECT_DIR"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-4}"

INPUT="data/processed/smiles_data/part-${TASK_ID}.parquet"
OUTPUT="data/interim/retro_cut_bonds/part-${TASK_ID}.parquet"
LOGDIR="logs/aizynth_cut_bonds/job_${SLURM_ARRAY_JOB_ID:-local}"

mkdir -p "$LOGDIR" "$(dirname "$OUTPUT")"

echo "host        : $(hostname)"
echo "task        : $TASK_ID"
echo "workers     : $NUM_WORKERS"
echo "input       : $INPUT"
echo "output      : $OUTPUT"

if [ ! -f "$INPUT" ]; then
    echo "ERROR: input part does not exist: $INPUT" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Thread pinning. ONNX Runtime and the BLAS libraries each default to one thread
# per CORE, so with one worker per core the node is 64x oversubscribed and
# throughput collapses. One thread per worker is what we want.
# ---------------------------------------------------------------------------
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ORT_NUM_THREADS=1

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
if [ -n "$CONDA_SETUP_CMD" ]; then
    # shellcheck disable=SC2086
    eval "$CONDA_SETUP_CMD"
fi

# shellcheck disable=SC1091
source activate "$ENV_PREFIX" 2>/dev/null || conda activate "$ENV_PREFIX"

python -c "import aizynthfinder, smiles_blocks; print('env OK')"

if [ ! -f "$MODEL_DIR/config.yml" ]; then
    echo "ERROR: no AiZynthFinder config at $MODEL_DIR/config.yml" >&2
    echo "Run hpc_scripts/caius_setup_aizynth_env.sh on a login node first." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Search
#
# --resume keeps whatever a previous attempt finished, so a task killed by the
# time limit can be relaunched without redoing its search. Results are flushed
# every --flush-every molecules rather than only at the end.
# ---------------------------------------------------------------------------
python jobs_scripts/aizynth_cut_bonds.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --config "$MODEL_DIR/config.yml" \
    --num-workers "$NUM_WORKERS" \
    --tag-keys "$TAG_KEYS" \
    --flush-every 200 \
    --resume \
    --logfile "$LOGDIR/task_${TASK_ID}.log"

echo "task $TASK_ID complete"
