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

Use pip to install smiles_block package using the pyproject.toml  :

For Linux (bash) :

```bash
cd smiles_blocks/
# optional : create and active conda env
# conda create -n smiles_blocks
# conda activate smiles_blocks
pip install .[all]
```

For MacOs (zsh):

```zsh
cd smiles_blocks/
# optional : create and active conda env
# conda create -n smiles_blocks
# conda activate smiles_blocks
pip install ".[all]"
```

The [all] option make sure that you install the optional depencies for ploting, using jupyter notebook, documentation, etc... More advanced user can look through the pyproject.toml to only install the set of dependencies they want.

You should almost be ready to go !

### Cluster install (Compute Canada)

When using cluster, especially clusters with compute canada, best practice is that  the package should be installed at the beginning of each job in a virtualenv. Some package like Arrow and RDKit are loaded as module because they have cluster dependent compilation fo C/C++ code. We provide a script to test the installation  using the following command :

```bash
sbatch hpc_scripts/computecanada_narval_test_package_install.sh
```

Note that the current install for cluster only install the minimal set of dependencies to run jobs.

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

## (Optional) Generating Canonical Control data

If you whish to reproduce experiment from preprint, you need to generate the fragmentable canonical SMILES for MOSES, i.e SMILES with all bonds explicits. You can do this locally  in ~1 minutes using the appropriate jupyter notebook :

```bash
# Optional : load the conda environement if you have one
# conda activate 
# conda activate smiles_blocks
jupyter notebook notebooks/4-er-make_canonical_control.ipynb
```

## Generate blocks and fragmented SMILES libraries

Generating blocks and blocked SMILES for the fully enumerated MOSES can be done locally within 1-2 days of continous compute with the following command :

```bash
# Optional : load the conda environement if you have one
# conda activate 
# conda activate smiles_blocks
python jobs_scripts/smiles2block.py --input data/processed/smiles_data
```

Or you can use the following SLURM script to increase the data parallelism with a job array **don't forget to set the proper : #SBATCH --account=your_account** :

```bash
sbatch hpc_scripts/computecanada_narval_smiles2block.sh
```

## (Optional) Use a retrosynthesis planner instead of R-BRICS

By default the cleavable bonds come from the R-BRICS SMARTS patterns, computed
inline inside the fragmentation workers. A retrosynthesis planner such as
AiZynthFinder can be used instead.

### Why this is a separate stage

R-BRICS costs microseconds per molecule; AiZynthFinder costs about 11.5 s. Running
a tree search inside the fragmentation workers would mean redoing it on every job
retry, and welding a CPU-bound search onto an I/O-bound pass over 65 GB of
randomised SMILES. So the search runs as its own stage and writes its results to
parquet, which the fragmentation stage reads back.

The work unit is one `smiles_data` part. Each part holds ~57 M randomised rows but
only **~16 300 unique reference SMILES**, which is what actually gets searched:

| Quantity | Value |
| --- | --- |
| Unique reference SMILES per part | ~16 300 |
| Search cost per molecule (1 core) | ~11.5 s |
| Cost per part | ~52 core-hours |
| Wall time per part on 64 cores | ~49 min |
| Full MOSES (119 parts, 1.9 M molecules) | ~6 200 core-hours |

A GPU does not help: the MCTS is CPU-bound and the policy network runs on ONNX
Runtime.

### Install AiZynthFinder

AiZynthFinder 4.4.1 needs Python 3.11, so it lives in its own environment,
separate from the one used for fragmentation. The two stages never share a
process. On a login node (internet access is needed for the models):

```bash
bash hpc_scripts/caius_setup_aizynth_env.sh
```

This creates the conda env, installs `aizynthfinder[all]==4.4.1` with `pyarrow<20`
(required because AiZynthFinder pins NumPy 1.x), downloads the public template
models and the ZINC stock, and runs a single-molecule search to verify the
install. It also reports which reaction metadata keys the model populates, which
is what `--tag-keys` should be set from.

### Run the search

Locally, on one part, for testing:

```bash
python jobs_scripts/aizynth_cut_bonds.py \
    --input data/processed/smiles_data/part-0.parquet \
    --output data/interim/retro_cut_bonds/part-0.parquet \
    --config models/aizynthfinder/config.yml \
    --num-workers 8 \
    --limit 100
```

On a SLURM cluster, as a 119-task array, **don't forget to set the proper
`#SBATCH --account=your_account` and `--partition`**:

```bash
sbatch hpc_scripts/caius_aizynth_cut_bonds.sh
```

Results are flushed every 200 molecules, so a task killed by the time limit keeps
what it finished; relaunch with `--resume` to search only what is missing. Note
that this array is `0-118`, covering all 119 parts; the older scripts in
`hpc_scripts/` use `1-118` and therefore skip `part-0`.

### Generate blocks from the precomputed bonds

Point `smiles2block.py` at the search output. Everything downstream is unchanged,
including the output schemas:

```bash
python jobs_scripts/smiles2block.py \
    --input data/processed/smiles_data/part-0.parquet \
    --retro-backend precomputed \
    --retro-bonds data/interim/retro_cut_bonds/part-0.parquet \
    --output_folder data/interim/aizynth_blocks/
```

### Routes that come back without atom mapping

The extraction above needs atom-mapped reactions. AiZynthFinder attaches that
mapping to a route's steps only when the reaction was actually mapped, and a
route without it would otherwise be discarded (`no_mapped_steps`).
`smiles_blocks/retro_backends/unmapped.py` is the fallback for those routes: it
recovers the disconnection from the molecules alone, by breaking each acyclic bond
of the product and keeping the bond when its two fragments land in *different*
precursors.

```python
from smiles_blocks.retro_backends.unmapped import UnmappedStep, unmapped_route_cut_bonds

bonds = unmapped_route_cut_bonds(
    "CC(=O)Nc1ccc(O)cc1",
    [UnmappedStep("CC(=O)Nc1ccc(O)cc1", ("CC(=O)Cl", "Nc1ccc(O)cc1"), tag="amide")],
)
```

This works through leaving groups — the acid chloride above carries a `Cl` the
product never had, yet the `CC=O` fragment still matches inside it — and it agrees
with the exact mapped extraction on the reactions tested.

Its limit: without atom maps, only the step acting on the **target** can be placed,
so deeper steps are skipped rather than guessed at. Prefer the mapped path whenever
the mapping is available; it is exact rather than inferential.

### Adding another planner

The bond extraction in `smiles_blocks/retro_backends/bond_extraction.py` works
from atom-mapped reaction SMILES alone, so a new planner needs only an adapter
that yields `RouteStep` objects — or `UnmappedStep` objects if it provides no
mapping.

Two invariants matter for any backend:

- Bond signatures are pairs of **canonical atom ranks** (`Chem.CanonicalRankAtoms`),
  not atom indices, because the fragmentation side intersects them against the
  bonds of randomised SMILES with different atom orderings.
- **Atom maps must be stripped before ranking.** With maps present,
  `CanonicalRankAtoms` returns the map numbering instead of the molecule's
  canonical ranks, so every signature would be wrong and would silently fail to
  intersect, giving `retro_bond_ratio = 0` with no error.

### (optional) Generate control blocks and fragmented SMILES libraries

If you want to reproduce the results of the preprint, you cant do so locally in couple of hours using the following command :

```bash
# Optional : load the conda environement if you have one
# conda activate 
# conda activate smiles_blocks
python jobs_scripts/smiles2block.py --num_workers 10 --input data/processed/moses_canonical_control/ --output_folder data/interim/moses_canonical_control/ --logfile logs/moses_canonical_control_smiles2block.log --rdkit-log-level CRITICAL
```

### (optional) Generate the paper figure from notebook 5

The figure used in the fragmentation results section of the paper can be regenerated from the Jupyter notebook [notebooks/5-er-smiles_fragmentation_results_analysis.ipynb](notebooks/5-er-smiles_fragmentation_results_analysis.ipynb). Open the notebook in Jupyter and run all cells from top to bottom. The notebook reads the blocked SMILES results from `data/interim/`, computes the summary statistics, and saves the figure to [reports/figures/fragmentation_results/](reports/figures/fragmentation_results/).

If you want to launch it from the command line, use:

```bash
# Optional : load the conda environement if you have one
# conda activate 
# conda activate smiles_blocks
jupyter notebook notebooks/5-er-smiles_fragmentation_results_analysis.ipynb
```

--------
