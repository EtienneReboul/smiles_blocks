"""Dask-based parallel processing script for generating SMILES calibration range datapoints.

This script processes a parquet file containing SMILES strings and generates calibration
datapoints by creating multiple replicas with different seeds. It uses Dask for
distributed/parallel computation and can be executed either locally or on a Dask cluster.

The script reads a parquet file with SMILES data, applies the `make_datapoints` function
from the smiles_blocks.range_calibration module to each row with specified parameters
(max_power, patience, nb_replicas), and saves the resulting datapoints to a new parquet
file.

Usage
Local execution with default parameters:
    python dask_calibration_range.py

Local execution with custom parameters:
    python dask_calibration_range.py --datafile path/to/input.parquet \
        --output_path path/to/output/ --nb_replicas 5 --max_power 7 \
        --patience 10 --nb_workers 20

Cluster execution:
    python dask_calibration_range.py --execution_mode cluster \
        --datafile path/to/input.parquet

Command-line Arguments
----------------------
--datafile : str, optional
    Path to input parquet database containing SMILES data.
    Default: 'data/processed/moses_repacked/part-0.parquet'
--output_path : str, optional
    Output directory path for generated parquet files.
    Default: 'data/interim/dask_calibration_range/'
--nb_replicas : int, optional
    Number of replicas to generate for each SMILES. Default: 3
--max_power : int, optional
    Maximum power parameter for random SMILES generation. Default: 6
--patience : int, optional
    Number of iterations without improvement before early stopping. Default: 5
--execution_mode : {'local', 'cluster'}, optional
    Dask execution mode. 'local' creates a LocalCluster, 'cluster' connects to
    an existing distributed cluster. Default: 'local'
--nb_workers : int, optional
    Number of Dask workers (local mode only). Default: 10
--nb_thread_dask : int, optional
    Number of threads per Dask worker (local mode only). Default: 1
--memory_limit : str, optional
    Memory limit per Dask worker (local mode only). Default: '3GB'

Environment Variables (cluster mode)
------------------------------------
DASK_SCHEDULER_ADDR : str
    Address of the Dask scheduler
DASK_SCHEDULER_PORT : str
    Port of the Dask scheduler

Input Data Requirements
-----------------------
The input parquet file must contain at least the following columns:
- SMILES : str
    SMILES string representation of molecules
- SPLIT : str
    Dataset split identifier (e.g., 'train', 'test', 'validation')

Output Data Schema
------------------
The output parquet files contain the following columns:
- smiles : str
    Original or generated SMILES string
- nb_unique_smiles : int64
    Number of unique SMILES found
- cumulative_count : int64
    Cumulative count of SMILES variations
- nb_random_smiles : int64
    Number of random SMILES generated
- seed : int64
    Random seed used for generation
- replica : int64
    Replica identifier
- unique_smiles : str
    Unique SMILES variations found
- split : str
    Dataset split from input data

- The script automatically changes to the project root directory before execution
- Dask dashboard is available when running in local mode (URL printed to logs)
- Output files are organized in subdirectories based on input filename stem
- The script uses pyarrow as the parquet engine for reading and writing

See Also
--------
smiles_blocks.range_calibration.make_datapoints : Function that generates calibration datapoints
"""

import argparse
from dataclasses import dataclass, field
from functools import partial
import logging
import multiprocessing as mp
import os
from pathlib import Path

import dask.config as cfg
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from smiles_blocks.files import INTERIM_DATA_DIR, PROJ_ROOT
from smiles_blocks.range_calibration import make_datapoints


@dataclass
class CalibrationParquetFormat:
    """Parquet format configuration for the  calibration range output.

    This dataclass contains the parameters used to configure the Parquet file
    format when writing the  calibration range output.

    Attributes
    ----------
    schema : pa.Schema
        PyArrow schema defining the structure of the output dataset.
        Default schema contains multiple columns with appropriate types and metadata.
    compression : str
        Compression algorithm to use for the Parquet files.
        Default is "zstd".
    compression_level : int
        Compression level for the selected algorithm (higher means more compression).
        Default is 12.

    """

    schema: pa.Schema = field(
        default_factory=lambda: pa.schema(
            [
                ("smiles", pa.string()),
                ("nb_unique_smiles", pa.int64()),
                ("cumulative_count", pa.int64()),
                ("nb_random_smiles", pa.int64()),
                ("seed", pa.int64()),
                ("replica", pa.int64()),
                ("unique_smiles", pa.string()),
                ("split", pa.string()),
            ],
            metadata={
                "smiles": "SMILES string representation",
                "nb_unique_smiles": "Number of unique SMILES found",
                "cumulative_count": "Cumulative count of SMILES variations",
                "nb_random_smiles": "Number of random SMILES generated",
                "seed": "Random seed used for generation",
                "replica": "Replica identifier",
                "unique_smiles": "Unique SMILES variations found",
                "split": "Dataset split",
            },
        )
    )
    compression: str = "zstd"
    compression_level: int = 12


def process_partition(
    partition: pd.DataFrame, max_power: int, patience: int, nb_replicas: int
) -> pd.DataFrame:
    """Process a partition of a DataFrame by generating datapoints for each row.
    This function takes a pandas DataFrame partition containing SMILES strings and SPLIT
    identifiers, applies the `make_datapoints` function to each SMILES string with the
    specified parameters, and returns a concatenated DataFrame of the results.

    Parameters
    ----------
    partition : pandas.DataFrame
        Input partition containing at least the columns "smiles", "zinc_id", and "replica".
    max_power : int
        Integer parameter forwarded to make_datapoint that controls the maximum power
    patience : int
        Integer parameter forwarded to make_datapoint used for early-stopping logic
    nb_replicas : int
        Number of replicas to consider (not used directly in this function but may be relevant
        for context)
    Returns
    -------
    pandas.DataFrame
        A DataFrame produced by concatenating the individual datapoints returned by
        make_datapoint for each row in the input partition. Each row in the returned
        DataFrame will include the original "zinc_id" value and all fields produced by
        make_datapoint. The index is reset (ignore_index=True).

    Raises
    ------
    KeyError
        If the required columns "smiles", "zinc_id", or "replica" are not present in partition.
    ValueError
        If one or more objects returned by make_datapoint cannot be concatenated into a
        single DataFrame (e.g., incompatible shapes or types).

    Notes
    -----
    - make_datapoint is expected to return a pandas-compatible object (Series or DataFrame)
      for each input row. The exact schema of the returned datapoints is determined by
      make_datapoint.
    - Results are collected in memory and concatenated at the end; for very large
      partitions this may increase memory usage. Consider processing in smaller partitions
      or streaming if memory is a concern.
    - The function preserves the order of iteration over the partition as provided.
    """
    # Create a list to store the results
    results = []

    # Iterate over each row in the partition
    for smiles, split in zip(partition["SMILES"], partition["SPLIT"]):
        # Create a datapoint
        result = make_datapoints(smiles, nb_replicas, max_power, patience=patience)
        result["split"] = split

        # Append the result to the list
        results.append(result)

    # Concatenate all results into a single DataFrame
    results_df = pd.concat(results, ignore_index=True)

    return results_df


def first_run(
    configuration_dict: dict,
    parquet_format: CalibrationParquetFormat = CalibrationParquetFormat(),
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    """
    Function to run the script for the first time
    """
    # disable heartbeat because task are up to 20 minutes long
    cfg.set({"distributed.scheduler.worker-ttl": None})

    # setup the dask client
    if configuration_dict["execution_mode"] == "local":
        cluster = LocalCluster(
            n_workers=configuration_dict["nb_workers"],
            threads_per_worker=configuration_dict["nb_thread_dask"],
            memory_limit=configuration_dict["memory_limit"],
        )
        client = Client(cluster)

        logger.info("Dask dashboard available at: %s", client.dashboard_link)
    elif configuration_dict["execution_mode"] == "cluster":
        client = Client(
            f"tcp://{os.environ['DASK_SCHEDULER_ADDR']}:{os.environ['DASK_SCHEDULER_PORT']}"
        )
        logger.info("Connected to Dask cluster")

    logger.info("Results saved to: %s", configuration_dict["output_path"])

    # disable heartbeat because task are up to 20 minutes long
    # read the parquet file
    ddf = dd.read_parquet(configuration_dict["datafile"], engine="pyarrow", split_row_groups=True)
    ddf_results = ddf.map_partitions(
        process_partition,
        max_power=configuration_dict["max_power"],
        patience=configuration_dict["patience"],
        nb_replicas=configuration_dict["nb_replicas"],
        meta={
            "smiles": "str",
            "nb_unique_smiles": "int64",
            "cumulative_count": "int64",
            "nb_random_smiles": "int64",
            "seed": "int64",
            "replica": "int64",
            "unique_smiles": "str",
            "split": "str",
        },
    )

    # will trigger the computation and save the results
    ddf_results.to_parquet(
        str(configuration_dict["output_path"]),
        engine="pyarrow",
        compression=parquet_format.compression,
        compression_level=parquet_format.compression_level,
        schema=parquet_format.schema,
    )

    # terminate the dask client
    client.shutdown()


def rerun(
    configuration_dict: dict,
    parquet_format: CalibrationParquetFormat = CalibrationParquetFormat(),
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    """Rerun the calibration processing for missing partitions using multiprocessing.

    This function resumes an interrupted calibration run by processing only the
    partitions that were not completed in a previous run. Unlike the initial run
    which uses Dask, this function uses Python's multiprocessing module to process
    individual row groups from the input parquet file.

    The function reads the original parquet file row group by row group, checks if
    the corresponding output file already exists, and processes only the missing
    partitions. Each partition is processed in parallel using a multiprocessing pool,
    with each SMILES string being passed to the make_datapoints function.

    Parameters
    ----------
    configuration_dict : dict
        Configuration dictionary containing processing parameters with the following keys:

        - datafile : str or Path
            Path to the input parquet file containing SMILES data
        - output_path : Path
            Directory path where output parquet files will be saved
        - nb_replicas : int
            Number of replicas to generate for each SMILES
        - max_power : int
            Maximum power parameter for random SMILES generation
        - patience : int
            Number of iterations without improvement before early stopping
    parquet_format : CalibrationParquetFormat, optional
        Parquet format configuration object containing schema, compression settings,
        and other parquet-specific parameters. Default is a new instance of
        CalibrationParquetFormat.
    logger : logging.Logger, optional
        Logger instance for outputting processing information. Default is a logger
        for the current module.

    Returns
    -------
    None
        This function writes results directly to parquet files and does not return
        any value.

    Notes
    -----
    - This function is designed to be called when the output directory is not empty,
      indicating a previous incomplete run.
    - The use of pool map allows to allocate one SMILES processing per CPU core instead of
      one row group per core with dask,the assumption is that each SMILES processing takes
      a long time (up to 20 minutes) and this is the reason why the partition could not be
      processed with Dask directly.
    - The function uses multiprocessing with a pool size equal to the number of
      available CPU cores.
    - Output files are named as 'part.{i}.parquet' where i is the row group index.
    - Each output file corresponds to one row group from the input parquet file.
    - The function preserves the 'SPLIT' column from the input data in the output.

    See Also
    --------
    first_run : Initial run function using Dask for distributed processing
    make_datapoints : Core function that generates calibration datapoints
    CalibrationParquetFormat : Configuration class for parquet output format
    """

    # get parquet file info
    parquet_file = pq.ParquetFile(configuration_dict["datafile"])

    # get the number of partitions
    num_row_groups = parquet_file.num_row_groups

    # iterate over the partitions
    for i in range(num_row_groups):
        # check if the partition is already processed
        output_file = configuration_dict["output_path"] / f"part.{i}.parquet"

        if not output_file.exists():
            # log the processing
            logger.info(f"Processing missing partition {i}/{num_row_groups}...")
            # load the row group
            row_group_table = parquet_file.read_row_group(i)

            # convert to pandas dataframe
            partition_df = row_group_table.to_pandas()

            # gset partial function
            process_func = partial(
                make_datapoints,
                nb_replica=configuration_dict["nb_replicas"],
                max_power=configuration_dict["max_power"],
                patience=configuration_dict["patience"],
            )

            # compute using pool
            with mp.Pool(processes=mp.cpu_count()) as pool:
                result_list = pool.map(process_func, partition_df["SMILES"])

            # add split to each data frame
            for i in range(len(result_list)):
                result_list[i]["split"] = partition_df["SPLIT"][i]

            # concatenate the results
            result_df = pd.concat(result_list, ignore_index=True)

            # transform into arrow table
            result_df = pa.Table.from_pandas(result_df, schema=parquet_format.schema)

            # write results
            pq.write_table(
                result_df,
                str(output_file),
                compression=parquet_format.compression,
                compression_level=parquet_format.compression_level,
            )
    # log completion
    logger.info("Rerun completed.")


def main(configuration_dict: dict):
    """
    Main function to run the script
    """

    # setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # instantiate the parquet format configuration
    parquet_format = CalibrationParquetFormat()

    # make new output path
    configuration_dict["output_path"] = (
        Path(configuration_dict["output_path"]) / Path(configuration_dict["datafile"]).stem
    )
    # make sure the output path exists
    configuration_dict["output_path"].mkdir(parents=True, exist_ok=True)

    # check if the output path is empty
    if not os.listdir(configuration_dict["output_path"]):
        # run the first time with dask backend
        logger.info("Starting first run...")
        first_run(configuration_dict, parquet_format)
    else:
        # rerun only missing partition with multiprocessing backend
        logger.info("Found output files, restarting missing partitions...")
        rerun(configuration_dict, parquet_format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dask calibration range script")

    # input/output arguments
    parser.add_argument(
        "--datafile",
        help="the input path to parquet database",
        default=str(PROJ_ROOT / "data/processed/moses_repacked/part-0.parquet"),
        type=str,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(INTERIM_DATA_DIR / "dask_calibration_range/"),
        help="Output path for parquet files",
    )

    # computation arguments
    parser.add_argument(
        "--nb_replicas", type=int, default=3, help="Number of replicas for experiments"
    )
    parser.add_argument(
        "--max_power",
        type=int,
        default=6,
        help="Maximum power for random SMILES generation",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of iterations without improvement before stopping ",
    )

    # dask arguments
    parser.add_argument(
        "--execution_mode",
        choices=["local", "cluster"],
        default="local",
        help="Dask execution mode: 'local' for LocalCluster, 'cluster' for distributed cluster",
    )
    parser.add_argument("--nb_workers", type=int, default=10, help="Number of Dask workers")
    parser.add_argument(
        "--nb_thread_dask",
        type=int,
        default=1,
        help="Number of threads per Dask worker",
    )
    parser.add_argument(
        "--memory_limit", type=str, default="3GB", help="Memory limit per Dask worker"
    )

    args = parser.parse_args()

    params_dict = vars(args)

    # change to project directory
    os.chdir(PROJ_ROOT)

    # run the main function
    main(params_dict)
