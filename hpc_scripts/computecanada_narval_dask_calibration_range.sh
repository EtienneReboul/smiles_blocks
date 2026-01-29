#!/bin/bash
#SBATCH --account=<your account>
#SBATCH --cpus-per-task=64  
#SBATCH --mem-per-cpu=800MB       
#SBATCH --time=0-12:00
#SBATCH --output=logs/calibration_range/job_%A/task_%a.out
#SBATCH --error=logs/calibration_range/job_%A/task_%a.err
#SBATCH --job-name="dask_calibration_range"
#SBATCH --array=1-118

module load python gcc arrow rdkit
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install -e . --no-index

source $SLURM_TMPDIR/env/bin/activate

export DASK_SCHEDULER_ADDR=$(hostname)

export DASK_SCHEDULER_PORT=$((30000 + $RANDOM % 10000))

dask scheduler --host $DASK_SCHEDULER_ADDR --port $DASK_SCHEDULER_PORT &

dask worker "tcp://$DASK_SCHEDULER_ADDR:$DASK_SCHEDULER_PORT" --no-dashboard --nworkers=64 \
--nthreads=1  --local-directory=$SLURM_TMPDIR &

sleep 10

python jobs_scripts/dask_calibration_range.py --datafile data/processed/moses_repacked/part-$SLURM_ARRAY_TASK_ID.parquet \
--execution_mode cluster 