# smiles_blocks

[![CCDS Project template](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)

SMILES fragments that can be concatenated into a valid molecule

## Project Organization

```text
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         smiles_blocks and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
└── smiles_blocks   <- Source code for use in this project.
    │
        ├── __init__.py                  <- Makes smiles_blocks a Python module
        │
        ├── baseparams.py                <- Base parameters configuration
        │
        │
        ├── dataset.py                   <- Scripts to download or generate data
        │
        │
        ├── files.py                     <- Contains path to directories containing data and plots
        │
        ├── range_calibration.py         <- Module to models random SMILES generation
        │
        ├── smiles_fragmentation.py      <- SMILES fragmentation logic
        │
        └── utils.py                     <- General utility functions
```

## Installation

### Regular install

To get started with this project, first clone the repository to your local machine:

```bash
git clone https://github.com/EtienneReboul/smiles_blocks.git
```

Use pip to install smiles_block package using the pyproject.toml :

```bash
cd smiles_blocks/
# optional : create and active conda env
# conda create -n smiles_blocks
# conda activate smiles_blocks
pip install .
```

You should almost be ready to go !

### Cluster install (Compute Canada)

When using cluster, especially clusters with compute canada, best practice is that  the package should be installed at the beginning of each job in a virtualenv. Some package like Arrow and RDKit are loaded as module because they have cluster dependent compilation fo C/C++ code. We provide a script to test the installation  using the following command :

```bash
sbatch hpc_scripts/computecanada_narval_test_package_install.sh
```

## Download Data

To Download and convert the full MOSES dataset (train,test,scaffold) into an (py)Arrow Parquet Dataset,a module called dataset is design to that effect, here is a minimal example to do so :

```bash
# Optional : load the conda environement if you have one
# conda activate 
# conda activate smiles_blocks
python -m smiles_blocks.dataset
```

The raw data will be downloaded in data/external and then repackaged parquet files in data/processed/moses_repackaged  with the following structure:

| MOSES Dataset Split | Value |
| ------------------- | ----- |
| Number of files | 119 |
| Number of row groups per file | 397 |
| Number of rows per row group | 41 |

If you are on cluster where downloading data using the loging node is frowned upon or have very limited access to internet you can download data locally and use the follow command line to updload data:

```bash
rsync -avz --progress data/processed/moses_repacked/ user@cluster:/path/to/destination/
```

## Generate SMILES data

this step is used to both generate randomized SMILES and record the number of unique SMILES  per number of randomized SMILES generated. The randomized SMILES process is stochastic, thus by default the operation is repeated 3 times in order to account for variability.

There is two way to generate this data : local or cluster

The local mode is used for testing purposes and should not be used for production:

```bash
# Optional : load the conda environement if you have one
# conda activate 
# conda activate smiles_blocks
python jobs_scripts/dask_calibration_range.py --execution_mode local
```

The cluster mode is used for production on Compute Canada Narval Cluster, this launch a job array of 119 tasks , i.e the number of parquet files generated in the previous step, and is design to launch one dask cluster with 64 workers per task. You should customize the cluster script  and **don't forget to set the proper : #SBATCH --account=your_account**

```bash
sbatch hpc_scripts/computecanada_narval_dask_calibration_range.sh
```

## Repackage calibration results

After the calibration jobs finish, use the repackaging script to validate that each per-file job produced all expected output shards and then consolidate them into single parquet files for modelling and SMILES data. By default it reads the MOSES parquet inputs from data/processed/moses_repacked, checks the corresponding Dask outputs under data/interim/dask_calibration_range, and writes consolidated outputs to data/processed/modelling_data and data/processed/smiles_data. You can run it locally with:

```bash
# Optional : load the conda environement if you have one
# conda activate 
# conda activate smiles_blocks
python jobs_scripts/repackage_calibration_results.py \
    --input_datafolder data/processed/moses_repacked/ \
    --output_datafolder data/interim/dask_calibration_range/ \
    --log_path logs/repackage_calibration_results.log
```

Or you can run it on a SLURM cluster using the corresponding script, **don't forget to set the proper : #SBATCH --account=your_account** :

```bash
sbatch hpc_scripts/computecanada_narval_repackage_calibration_results.sh
```

### (Optional) Relaunching task

If some jobs did not complete, look at the end of the log file, the last line should have all the indices of the missing jobs. There is a relaunching optio that auto-detect the presence of output files in the generation script, so you can just relaunch task by replacing the $MISSING_IDX by one of actual mising idx.

```bash
# Optional : load the conda environement if you have one
# conda activate 
# conda activate smiles_blocks
python jobs_scripts/dask_calibration_range.py --datafile data/processed/moses_repacked/part-$MISSING_IDX.parquet --execution_mode local 
```

## Modelling randomized SMILES generation

### Model growth with scipy

To ensure that that all possible SMILES were exhaustively sampled we want to model the number of unique SMILES per the cumulative count of randomized SMILES. To that end we have  different model that are summarized in the following table :

| Model name | Formula |
| --- | --- |
| Square Root | $y = \alpha \sqrt{\beta x}$ |
| Logarithmic | $y = \alpha \log(\beta x)$ |
| Inverse | $y = \frac{\alpha}{\beta + \frac{1}{x}}$ |
| Inverse 2 | $y = \frac{\alpha}{1 + \frac{\beta}{x}}$ |
| Exponential | $y = \alpha (1 - e^{-\beta x})$ |

Modelling with scipy is relatively fast (~ 1 hour) and can be done locally , like so :

```bash
# Optional : load the conda environement if you have one
# conda activate 
# conda activate smiles_blocks
python jobs_scripts/smiles_generation_modelling.py --nb_wokers 6 
```

### Plotting modelling results

Once  the modelling is done, you can visualize the results with the following notebook. You may need to install jupyter notebook if you have not done it already. The results will be saved in the reports folder.

```bash
# Optional : load the conda environement if you have one
# conda activate 
# conda activate smiles_blocks
jupyter notebook notebooks/2-er-visualize_modeling_results.ipynb
```

--------
