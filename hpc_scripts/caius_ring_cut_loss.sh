#!/bin/bash
# Quantify the acyclic-only restriction over ALL of MOSES, as a SLURM array.
#
# Prerequisite: the same conda env stage A used
# (hpc_scripts/caius_setup_aizynth_env.sh, once on a login node).
#
#   sbatch hpc_scripts/caius_ring_cut_loss.sh
#
# WHY this costs a second full search. Stage A stored only the cut bonds that
# SURVIVED filtering; a cut discarded for being a ring bond leaves no trace in
# the output. Counting the loss therefore means redoing the search with the
# counters on -- there is no cheaper route, and the cost is stage A's again:
#
#   stage A, job 18554867, 118 tasks: median 3.8 h, max 10.7 h per task
#   11 785 core-hours in total, ~33 h end to end at 25 concurrent tasks
#
# This run is marginally cheaper per molecule (no parquet of randomised SMILES
# is written) but assume the same: ~12 000 core-hours, ~1.5 days wall.
#
# Unlike stage A this writes one row per MOLECULE including every discarded ring
# bond's ranks, smallest ring size and aromaticity. The search is what costs;
# recording which bonds were lost rather than only how many means ring-size and
# aromaticity questions are answered offline instead of paying a third time.
#
# The array range is 0-118, not 1-118: the parts are part-0 ... part-118.
#
# The log paths are flat on purpose. SLURM opens --output/--error BEFORE the job
# script runs, so a `job_%A/` subdirectory the script itself creates does not yet
# exist and every task dies instantly with exit code 53.

#SBATCH --job-name="ring_cut_loss"
#SBATCH --array=0-118%25
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=0-20:00
#SBATCH --output=logs/ring_cut_loss/job_%A_task_%a.out
#SBATCH --error=logs/ring_cut_loss/job_%A_task_%a.err

# ============================ CONFIGURE ME ============================
#SBATCH --account=ibmp
#SBATCH --partition=public
#
# The `public` partition has OverSubscribe=NO, so a task gets a WHOLE node
# however few CPUs it asks for. Each worker loads its own expansion model,
# template library and stock (~2.9 GB), i.e. ~69 GB at 24 workers, so --mem=0
# claims the node's memory at no extra cost rather than risking an OOM kill
# hours in under --mem-per-cpu.
#SBATCH --mem=0

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/smiles_blocks}"
ENV_PREFIX="${ENV_PREFIX:-$PROJECT_DIR/envs/aizynth}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models/aizynthfinder}"
CONDA_SETUP_CMD="${CONDA_SETUP_CMD:-module load python/Miniconda}"
# ======================================================================

set -euo pipefail

cd "$PROJECT_DIR"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-4}"

INPUT="data/interim/ref_smiles/part-${TASK_ID}.parquet"
OUTPUT="data/interim/ring_cut_loss/part-${TASK_ID}.parquet"
LOGDIR="logs/ring_cut_loss"

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

# ONNX Runtime and the BLAS libraries each default to one thread per CORE, so
# with one worker per core the node is heavily oversubscribed and throughput
# collapses. One thread per worker is what we want.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ORT_NUM_THREADS=1

# `module` is a shell function from /etc/profile, absent in sbatch's non-login
# shell, so the modules init is sourced before using it.
if ! command -v module >/dev/null 2>&1; then
    for _init in "${MODULESHOME:-}/init/bash" /usr/local/Modules/latest/init/bash; do
        if [ -r "$_init" ]; then
            # shellcheck disable=SC1091
            source "$_init"
            break
        fi
    done
fi

if [ -n "$CONDA_SETUP_CMD" ]; then
    # shellcheck disable=SC2086
    eval "$CONDA_SETUP_CMD"
fi

# shellcheck disable=SC1091
source activate "$ENV_PREFIX" 2>/dev/null || conda activate "$ENV_PREFIX"

python -c "import aizynthfinder, smiles_blocks; print('env OK')"

if [ ! -f "$MODEL_DIR/config.yml" ]; then
    echo "ERROR: no AiZynthFinder config at $MODEL_DIR/config.yml" >&2
    exit 1
fi

# --resume keeps whatever a previous attempt finished, so a task killed by the
# time limit can be relaunched without redoing its search.
python jobs_scripts/measure_ring_cut_loss.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --config "$MODEL_DIR/config.yml" \
    --num-workers "$NUM_WORKERS" \
    --flush-every 200 \
    --resume \
    --logfile "$LOGDIR/job_${SLURM_ARRAY_JOB_ID:-local}_task_${TASK_ID}.log"

echo "task $TASK_ID complete"
