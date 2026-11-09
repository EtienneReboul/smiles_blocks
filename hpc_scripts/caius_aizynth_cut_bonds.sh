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
# Measured on a FULL part (job 18538797_0, 24 workers, node hpc-n704):
#   48.9 s per molecule per core        (the 200-molecule pilot said 25.6 s and
#                                        was wrong twice over: it read the first
#                                        200 of a SORTED list, and 8 workers do
#                                        not contend for memory bandwidth the
#                                        way 24 do)
#   16 277 molecules x 48.9 s           = ~221 core-hours per part
#   221 core-hours / 24 cores           = ~9.2 h wall per task
#   119 parts                           = ~26 300 core-hours total
#   74 % of molecules yield usable cuts, 0.8 cuts per molecule overall
#
# Per-molecule search time is long-tailed: mean 46 s but max 711 s, so the wall
# limit has to cover the tail rather than the mean.
#
# The `public` partition has OverSubscribe=NO, so a task is given a WHOLE node
# however few CPUs it asks for: the pilot requested 8 and was allocated all 24.
# Asking for 24 therefore costs nothing extra and uses the node we are charged
# for -- requesting fewer just idles cores.
#
# A GPU would not help: the MCTS itself is CPU-bound and the policy network runs
# on ONNX Runtime.
#
# NOTE the array range is 0-118, not 1-118. The parts are named part-0 ...
# part-118, so the 1-118 used by the other scripts in this directory silently
# skips part-0.
#
# NOTE the log paths below are flat. SLURM opens the --output/--error files
# BEFORE the job script runs, so a `job_%A/` subdirectory that the script itself
# creates does not yet exist and every task dies instantly with exit code 53.
# The other scripts in this directory have this same latent trap; they only work
# if their log subdirectory was created by hand beforehand.

#SBATCH --job-name="aizynth_cut_bonds"
#SBATCH --array=0-118%25
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=0-20:00
#SBATCH --output=logs/aizynth_cut_bonds/job_%A_task_%a.out
#SBATCH --error=logs/aizynth_cut_bonds/job_%A_task_%a.err

# ============================ CONFIGURE ME ============================
# Nothing below the next block should need editing.
#
#SBATCH --account=ibmp
#SBATCH --partition=public
#
# Memory: each worker loads its own expansion model, template library and stock,
# so usage scales with worker count -- the pilot measured 22.9 GB across 8
# workers, i.e. ~2.9 GB each, which is ~69 GB at 24 workers. --mem-per-cpu=3G
# would allocate only 72 GB and leave 4 % headroom before the cgroup OOM-kills
# the task hours in. Since the whole node is allocated anyway, --mem=0 claims
# all of its memory (94 GB on the smallest) at no extra cost.
#SBATCH --mem=0

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/smiles_blocks}"
ENV_PREFIX="${ENV_PREFIX:-$PROJECT_DIR/envs/aizynth}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models/aizynthfinder}"

# How to get conda on PATH on a compute node. Same value as in the setup script.
CONDA_SETUP_CMD="${CONDA_SETUP_CMD:-module load python/Miniconda}"

# Reaction metadata keys used to label a disconnection, in order of preference.
# Which keys the model actually populates is NOT guaranteed by the library --
# take this from what caius_setup_aizynth_env.sh reported.
TAG_KEYS="${TAG_KEYS:-classification,template_hash,template_code,policy_name}"
# ======================================================================

set -euo pipefail

cd "$PROJECT_DIR"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-4}"

INPUT="data/interim/ref_smiles/part-${TASK_ID}.parquet"
OUTPUT="data/interim/retro_cut_bonds/part-${TASK_ID}.parquet"
LOGDIR="logs/aizynth_cut_bonds"

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
# `module` is a shell function from /etc/profile, absent in sbatch's non-login
# shell, so the modules init is sourced before using it.
if ! command -v module >/dev/null 2>&1; then
    # Environment Modules lives under /usr/local/Modules on CAIUS and is not in
    # /etc/profile.d, so the init script is sourced by path.
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
    --logfile "$LOGDIR/job_${SLURM_ARRAY_JOB_ID:-local}_task_${TASK_ID}.log"

echo "task $TASK_ID complete"
