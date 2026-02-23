#!/bin/bash
#SBATCH --account=<your account>
#SBATCH --cpus-per-task=10  
#SBATCH --mem-per-cpu=3000MB       
#SBATCH --time=0-12:00
#SBATCH --output=logs/smiles2block/job_%A/task_%a.out
#SBATCH --error=logs/smiles2block/job_%A/task_%a.err
#SBATCH --job-name="smiles2block"
#SBATCH --array=1-118

module load python/3.12.4 gcc arrow rdkit
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install -e . --no-index

source $SLURM_TMPDIR/env/bin/activate

python jobs_scripts/smiles2block.py --input data/processed/smiles_data/part-$SLURM_ARRAY_TASK_ID.parquet --logfile logs/smiles2block/job_$SLURM_ARRAY_JOB_ID/task_$SLURM_ARRAY_TASK_ID.log 
