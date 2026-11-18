#!/bin/bash
# Hold-out sweep on CAIUS: fit each saturation model on the first 1..100 % of the
# sampling budget of every MOSES molecule, as a SLURM array (reviewer request:
# does the exponential model converge faster towards the true plateau?).
#
#   sbatch hpc_scripts/caius_holdout_sweep.sh
#
# Each task also runs the fixed-budget / cross-replica hold-out test
# (smiles_generation_holdout.py) for its part, so one submission refreshes both.
#
# then, once all tasks are done (on CAIUS or after copying the parts back):
#
#   python jobs_scripts/aggregate_holdout_sweep.py
#   python jobs_scripts/smiles_generation_holdout.py --n_molecules 0 --summary_only
#
# Prerequisites on CAIUS:
#   - data/processed/modelling_data/part-{0..118}.parquet (576 MB) copied over,
#     e.g. rsync -av data/processed/modelling_data/ caius:<PROJECT_DIR>/data/processed/modelling_data/
#   - a Python env with numpy, scipy, pandas and pyarrow. The AiZynthFinder env
#     from caius_setup_aizynth_env.sh is used by default; smiles_blocks is put on
#     PYTHONPATH by the script itself, so nothing needs installing.
#
# Work unit: one array task per modelling_data part (~16 k molecules,
# 100 fractions x 4 models). Fits are cached on the training set (consecutive
# fractions often share one, because the sampling grid is log-spaced).
#
# Models: smiles_blocks.modelling.FINAL_MODELS (exp, weibull, hill, lomax).
# Measured: 2-parameter set ran in ~75 s per part on CAIUS (job 18610923); the
# 3-parameter Weibull/Hill fits are ~3-4x slower (1 000 molecules in ~7 s on 14
# local cores), i.e. ~3-5 min per part. A 2 h limit leaves a wide margin.
#
# Output: data/processed/holdout_sweep/part-<i>.parquet, ~32 MB each (~4 GB in
# total), one row per (molecule, fraction, function).
#
# The array range is 0-118, not 1-118: the parts are part-0 ... part-118.
#
# The log paths are flat on purpose. SLURM opens --output/--error BEFORE the job
# script runs, so a `job_%A/` subdirectory the script itself creates does not yet
# exist and every task dies instantly with exit code 53. Create logs/holdout_sweep
# before the first submission:  mkdir -p logs/holdout_sweep

#SBATCH --job-name="holdout_sweep"
#SBATCH --array=0-118%25
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=0-02:00
#SBATCH --output=logs/holdout_sweep/job_%A_task_%a.out
#SBATCH --error=logs/holdout_sweep/job_%A_task_%a.err

# ============================ CONFIGURE ME ============================
#SBATCH --account=ibmp
#SBATCH --partition=public
#
# The `public` partition has OverSubscribe=NO, so a task gets a WHOLE node
# however few CPUs it asks for; --mem=0 claims its memory at no extra cost.
# Peak use is small anyway (one part is ~2.4 M rows, a few GB in pandas).
#SBATCH --mem=0

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/smiles_blocks}"
ENV_PREFIX="${ENV_PREFIX:-$PROJECT_DIR/envs/aizynth}"
CONDA_SETUP_CMD="${CONDA_SETUP_CMD:-module load python/Miniconda}"
# ======================================================================

set -euo pipefail

cd "$PROJECT_DIR"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-4}"

INPUT="data/processed/modelling_data/part-${TASK_ID}.parquet"
OUTPUT="data/processed/holdout_sweep/part-${TASK_ID}.parquet"
LOGDIR="logs/holdout_sweep"

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

# scipy's curve_fit sits on BLAS/LAPACK, which defaults to one thread per CORE;
# with one worker process per core that oversubscribes the node.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

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

# The sweep only needs smiles_blocks.modelling; putting the repo on PYTHONPATH
# avoids depending on how (or whether) the package was installed in this env.
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if ! python -c "import numpy, scipy, pandas, pyarrow; from smiles_blocks.modelling import ModelsRegistry"; then
    echo "ERROR: env $ENV_PREFIX lacks numpy/scipy/pandas/pyarrow." >&2
    echo "Fix on a login node with:  conda activate $ENV_PREFIX && pip install scipy pandas pyarrow" >&2
    exit 1
fi

# An existing output part is skipped, so failed or timed-out tasks can simply be
# resubmitted (sbatch --array=<ids> ...). Parts are written to a .tmp file and
# renamed, so a killed task never leaves a truncated part behind.
python jobs_scripts/smiles_generation_holdout_sweep.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --nb-workers "$NUM_WORKERS" \
    --logfile "$LOGDIR/job_${SLURM_ARRAY_JOB_ID:-local}_task_${TASK_ID}.log"

# Fixed-budget and cross-replica hold-out for the same part (per-part output only;
# the summary CSVs are built once every part exists, with --summary_only).
python jobs_scripts/smiles_generation_holdout.py \
    --n_molecules 0 \
    --parts "$TASK_ID" \
    --skip_summary \
    --nb_workers "$NUM_WORKERS"

echo "task $TASK_ID complete"
