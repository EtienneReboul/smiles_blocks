#!/bin/bash
#SBATCH --account=<your account>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6  
#SBATCH --mem-per-cpu=3900MB    
#SBATCH --time=0-12:00
#SBATCH --output=logs/reformat_smiles_data/SLURM_log_%A.out
#SBATCH --error=logs/reformat_smiles_data/SLURM_err_%A.err
#SBATCH --job-name="reformat_smiles_data"

module load python gcc arrow rdkit
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install -e . --no-index

source $SLURM_TMPDIR/env/bin/activate

python jobs_scripts/reformat_smiles_data.py --log-file logs/reformat_smiles_data/reformat_smiles_data.log