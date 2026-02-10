"""
This scripts is used to fit models to determine the number of unique SMILES
generated as a function of the number of randomized SMILES generated.
"""

import argparse
from dataclasses import dataclass, field
from functools import partial
import logging
import multiprocessing as mp
import os
from pathlib import Path
import time
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import tqdm

from smiles_blocks.files import PROCESSED_DATA_DIR, PROJ_ROOT
from smiles_blocks.modelling import ModelsRegistry


@dataclass
class ModelFitResult:
    schema: pa.Schema = field(
        default_factory=lambda: pa.schema(
            [
                ("smiles", pa.string()),
                ("split", pa.string()),
                ("function", pa.string()),
                ("formula", pa.string()),
                ("max_nb_unique_smiles", pa.float64()),
                ("alpha", pa.float64()),
                ("beta", pa.float64()),
                ("r_squared", pa.float64()),
                ("error_log", pa.string()),
            ]
        )
    )
    compression: str = "brotli"
    compression_level: int = 5


def fit_one_smiles_generation_modelling(
    smiles: pa.StringScalar, smiles_data: pa.Table
) -> pa.Table:
    """
    Fit all models to smiles generation data.

    Parameters
    ----------
    smiles : pa.StringScalar
        The canonical SMILES string for which the models are being fit.
    smiles_data : pa.Table
        Table containing the observations to fit. Required columns:
        - 'cumulative_count': independent variable (x)
        - 'nb_unique_smiles': dependent variable (y)
    model_name : str
        Name of the model to fit, must be one of the keys in ModelsRegistry.models.
    Returns
    -------
    Tuple[numpy.ndarray, float]
        A tuple containing:
        - popt: Optimal parameters [alpha, beta] as a numpy array.
        - r_squared: Coefficient of determination for the fit.

    Raises
    ------
    ValueError
        If the specified model_name is not found in the ModelsRegistry.

    Notes
    -----
    - Uses scipy.optimize.curve_fit to perform the fit.
    - Assumes non-empty numeric x and y arrays; callers should validate / pre-filter if needed.
    """

    # instantiate local variables
    parquet_format = ModelFitResult()
    models = ModelsRegistry().models
    results = []

    # filter data for the current smiles

    mask = pc.equal(smiles_data["smiles"], smiles)
    group = smiles_data.filter(mask)

    # retrieve x and y data
    x_data = group["cumulative_count"].to_numpy()
    y_data = group["nb_unique_smiles"].to_numpy()
    max_y = y_data.max()

    # fit each function to the data
    for model_name, model in models.items():
        # loop variable
        temp_dict = {
            "smiles": group["smiles"][0].as_py(),
            "split": group["split"][0].as_py(),
            "function": model_name,
            "formula": model.formula,
            "max_nb_unique_smiles": max_y,
            "alpha": np.nan,
            "beta": np.nan,
            "r_squared": np.nan,
            "error_log": "",
        }

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                popt, _ = model.fit(x_data, y_data, p0=[max_y, 1.0], maxfev=10_000)
                if w:
                    temp_dict["error_log"] = "; ".join(str(warn.message) for warn in w)
            temp_dict["alpha"] = popt[0]
            temp_dict["beta"] = popt[1]
            temp_dict["r_squared"] = model.r2_score(x_data, y_data, *popt)

        except Exception as e:
            temp_dict["error_log"] = str(e)
            continue

        results.append(temp_dict)

    # convert results to pyarrow Table
    return pa.Table.from_pylist(results, schema=parquet_format.schema)


def fit_partition_wrapper(partition_table: pa.Table, nb_workers: int) -> pa.Table:
    """
    Fit multiple candidate models to cumulative sampling data grouped by zinc_id.

    Parameters
    ----------
    partition_table : pa.Table
        Partition table containing the observations to fit. Required columns:
        - 'smiles': canonical SMILES string
        - 'cumulative_count': independent variable (x)
        - 'nb_unique_smiles': dependent variable (y)
    fitting_functions : dict[str, callable]
        Mapping of function name -> callable, where each callable has signature
        f(x, alpha, beta) and accepts array-like x and two parameters to fit.
    formula_dict : dict[str, str], optional
        Mapping of function name -> LaTeX formula string for documentation purposes.
        Default is empty dict.

    Returns
    -------
        pa.Table
        Summary dataframe with one row per successful fit and the following columns:
        - 'smiles' (str): the canonical SMILES string for the group
        - 'split' (str): the data split (e.g., 'train', 'test', 'scaffolds') for the group
        - 'function' (str): name of the fitting function used
        - 'formula' (str): LaTeX formula of the fitting function
        - 'alpha' (float): fitted first parameter
        - 'beta' (float): fitted second parameter
        - 'r_squared' (float): coefficient of determination for the fit
        - 'error_log' (str): error message for failed fits (empty for successful fits)

    Notes
    -----
    - Uses scipy.optimize.curve_fit to perform the fits.
    - Groups the input by 'smiles' and fits each provided function to the group's
      (cumulative_count, nb_unique_smiles) data.
    - Fits that raise exceptions are skipped (their error message may be recorded).
    - The function assumes non-empty numeric x and y arrays for each smiles;
      callers should validate / pre-filter the partition if needed.

    """

    # get unique smiles and group by it
    unique_smiles = pc.unique(partition_table["smiles"])

    partial_func = partial(fit_one_smiles_generation_modelling, smiles_data=partition_table)

    with mp.Pool(processes=nb_workers) as pool:
        table_list = pool.map(partial_func, unique_smiles)

    return pa.concat_tables(table_list)


def modelling_arguments_parser() -> dict:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Model SMILES generation curves.")

    # input/output arguments
    parser.add_argument(
        "--datafolder",
        help="the input path to parquet database",
        default=str(PROCESSED_DATA_DIR / "modelling_data/"),
        type=str,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(PROCESSED_DATA_DIR / "models_metrics/"),
        help="Output path for parquet files",
    )

    parser.add_argument(
        "--logfile",
        type=str,
        default="logs/smiles_generation_modelling.log",
        help="Path to the log file",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files",
    )

    parser.add_argument(
        "--nb_workers",
        type=int,
        default=mp.cpu_count(),
        help="Number of worker processes for parallel fitting (default: number of CPU cores)",
    )

    # parse arguments
    args = parser.parse_args()

    return vars(args)


def main():
    """Main function to execute the fitting process."""

    # change to project root directory to ensure relative paths work correctly
    os.chdir(PROJ_ROOT)

    # parse arguments
    args = modelling_arguments_parser()
    parquet_format = ModelFitResult()

    # make sure log directory exists
    log_path = Path(args["logfile"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # remove existing log file if overwrite is enabled
    if log_path.exists() and args["overwrite"]:
        log_path.unlink()

    # set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path)],
    )
    logger = logging.getLogger(__name__)

    # make sure output directory exists
    output_path = Path(args["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)

    # aggregate paths to all parquet files in the input folder using pyarrow dataset
    dataset = ds.dataset(args["datafolder"], format="parquet")

    # iterate over the dataset and concatenate all partitions into a single DataFrame
    for parquet_file in tqdm.tqdm(dataset.files, desc="Processing files"):
        # cast as Path  to extract filename
        filepath = Path(parquet_file)
        filename = filepath.name

        # log the file being processed
        logger.info(f"Processing file: {filename}")
        start = time.time()

        # check if output file already exists, if so skip
        output_file = output_path / filename

        if output_file.exists() and not args["overwrite"]:
            print(f"Output file {output_file} already exists, skipping.")
            continue

        # read partition into pandas DataFrame
        partition_table = pq.read_table(parquet_file)

        # fit models and get results
        results_df = fit_partition_wrapper(partition_table, nb_workers=args["nb_workers"])

        # save results to parquet
        pq.write_table(
            results_df,
            str(output_file),
            compression=parquet_format.compression,
            compression_level=parquet_format.compression_level,
        )
        end = time.time()
        elapsed_time = end - start
        logger.info(f"Finished processing file: {filename} in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    main()
