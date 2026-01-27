#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --job-name="env_test_smiles_blocks"
#SBATCH --output=logs/env_test_%j.out
#SBATCH --error=logs/env_test_%j.err

set -e
set -u

echo "Job started on $(hostname)"
echo "SLURM job ID: $SLURM_JOB_ID"

# Clean module environment
module purge
module load python gcc arrow rdkit

# Create virtual environment in node-local storage
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"

echo "Python version:"
python --version

# Install your package (editable, offline)
pip install -e . --no-index

# ---- ENVIRONMENT / PACKAGE TEST ----
python - << EOF
import sys
print("Python executable:", sys.executable)

import smiles_blocks
print("SUCCESS: smiles_blocks imported correctly")
print("smiles_blocks location:", smiles_blocks.__file__)
EOF

echo "Environment test completed successfully"