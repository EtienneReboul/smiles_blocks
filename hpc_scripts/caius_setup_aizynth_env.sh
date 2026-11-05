#!/bin/bash
# One-time setup of the AiZynthFinder environment on CAIUS.
#
# Run this ONCE, on a LOGIN NODE, before submitting the array job. It needs
# internet access to fetch the conda packages and the public models; compute
# nodes often have none, which is exactly why this is separate from the job
# script.
#
#   bash hpc_scripts/caius_setup_aizynth_env.sh
#
# What it does:
#   1. creates a Python 3.11 conda env on the shared filesystem
#   2. installs aizynthfinder 4.4.1 (ONNX Runtime backend: no GPU, no TensorFlow)
#   3. installs this package into it
#   4. downloads the public template models and the ZINC stock
#   5. verifies the install with a single-molecule search
#
# Why Python 3.11: aizynthfinder 4.4.1 does not support 3.12+. This env is
# therefore SEPARATE from the one used for fragmentation (which needs >=3.12).
# The two stages never run in the same process, so that is not a problem.

set -euo pipefail

# ============================ CONFIGURE ME ============================
# Shared-filesystem location for the env and the models. Must be readable from
# every compute node, and should NOT be in your home directory if home is
# small or not mounted on the nodes.
PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/smiles_blocks}"
ENV_PREFIX="${ENV_PREFIX:-$PROJECT_DIR/envs/aizynth}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models/aizynthfinder}"

# How to get a working conda/mamba on CAIUS. Replace with whatever applies:
#   module load conda        (or miniconda/anaconda/mamba)
#   source /path/to/conda/etc/profile.d/conda.sh
CONDA_SETUP_CMD="${CONDA_SETUP_CMD:-}"
# ======================================================================

# Exported for the verification heredoc at the end.
export PROJECT_DIR ENV_PREFIX MODEL_DIR

echo "== setting up AiZynthFinder env"
echo "   env prefix : $ENV_PREFIX"
echo "   model dir  : $MODEL_DIR"

if [ -n "$CONDA_SETUP_CMD" ]; then
    # shellcheck disable=SC2086
    eval "$CONDA_SETUP_CMD"
fi

if ! command -v conda >/dev/null 2>&1 && ! command -v mamba >/dev/null 2>&1; then
    echo "ERROR: no conda or mamba on PATH." >&2
    echo "Set CONDA_SETUP_CMD at the top of this script, e.g.:" >&2
    echo "  CONDA_SETUP_CMD='module load conda'" >&2
    exit 1
fi

CONDA_BIN="$(command -v mamba || command -v conda)"
echo "   using      : $CONDA_BIN"

# ---------------------------------------------------------------------------
# 1-2. Environment
# ---------------------------------------------------------------------------
if [ -d "$ENV_PREFIX" ]; then
    echo "== env already exists, reusing it"
else
    echo "== creating Python 3.11 env"
    "$CONDA_BIN" create --yes --prefix "$ENV_PREFIX" python=3.11 pip
fi

# shellcheck disable=SC1091
source activate "$ENV_PREFIX" 2>/dev/null || conda activate "$ENV_PREFIX"

echo "== installing aizynthfinder"
# pyarrow<20 is pinned because aizynthfinder 4.4.1 pulls NumPy 1.x, and pyarrow
# >=20 is built against NumPy 2.x; mixing them gives an import-time ABI error.
pip install --no-cache-dir \
    "aizynthfinder[all]==4.4.1" \
    "pyarrow<20"

echo "== installing smiles_blocks"
pip install --no-cache-dir --no-deps -e "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# 3-4. Models and stock
# ---------------------------------------------------------------------------
mkdir -p "$MODEL_DIR"
if [ -f "$MODEL_DIR/config.yml" ]; then
    echo "== models already downloaded, reusing them"
else
    echo "== downloading public models and ZINC stock (a few GB)"
    (cd "$MODEL_DIR" && download_public_data .)
fi

echo "== config written to $MODEL_DIR/config.yml"
cat "$MODEL_DIR/config.yml"

# ---------------------------------------------------------------------------
# 5. Verify
# ---------------------------------------------------------------------------
echo "== verifying with a single search (paracetamol)"
python - <<'PYEOF'
import os
# Keep ONNX single-threaded, as in the array job.
for var in ("OMP_NUM_THREADS", "ORT_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

from smiles_blocks.retro_backends.aizynth import load_finder, observed_metadata_keys, search_one

config = os.path.join(os.environ["MODEL_DIR"], "config.yml")
finder = load_finder(config)

outcome = search_one(finder, "CC(=O)Nc1ccc(O)cc1")
print(f"success={outcome.retro_success} routes={outcome.n_routes} "
      f"steps={outcome.n_steps} time={outcome.search_time_s:.1f}s "
      f"bonds={len(outcome.bonds)} error={outcome.error!r}")

# Report the reaction metadata keys this model populates. Which keys exist is a
# property of the expansion policy, not of the library, so --tag-keys has to be
# chosen from what is actually here.
trees = list(finder.routes.reaction_trees)
if trees:
    keys = sorted(observed_metadata_keys(trees[0]))
    print("reaction metadata keys:", keys)
    if "mapped_reaction_smiles" not in keys:
        print("WARNING: no mapped_reaction_smiles -> bonds cannot be traced to the target")
PYEOF

echo
echo "== setup complete"
echo "Next: set MODEL_DIR and ENV_PREFIX in hpc_scripts/caius_aizynth_cut_bonds.sh,"
echo "      then submit it."
