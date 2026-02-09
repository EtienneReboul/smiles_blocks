#!/bin/bash
#SBATCH --account=<your account>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6  
#SBATCH --mem-per-cpu=3900MB    
#SBATCH --time=0-12:00
#SBATCH --output=logs/repackaging_calibration/SLURM_log_%A.out
#SBATCH --error=logs/repackaging_calibration/SLURM_err_%A.err
#SBATCH --job-name="repackaging_calibration"

module load python gcc arrow rdkit
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install -e . --no-index

source $SLURM_TMPDIR/env/bin/activate

# Run the repackaging script
python jobs_scripts/repackage_calibration_results.py --log_path logs/repackaging_calibration/repackage_calibration_results.log