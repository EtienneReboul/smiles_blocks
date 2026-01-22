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

To get started with this project, first clone the repository to your local machine:

```bash
git clone https://github.com/EtienneReboul/smiles_blocks.git
```

Use pip to install smiles_block package using the pyproject.toml :

```bash
cd smiles_blocks/
pip install .
```

You should be ready to go !

--------
