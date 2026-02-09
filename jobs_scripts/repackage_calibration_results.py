import argparse
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import shutil
import time

from dask_calibration_range import CalibrationParquetFormat
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from smiles_blocks.files import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, PROJ_ROOT


@dataclass
class ModelingParquetFormat:
    """Dataclass for modeling parquet format parameters."""

    col_sel: list[str] = field(
        default_factory=lambda: [
            "smiles",
            "nb_unique_smiles",
            "cumulative_count",
            "nb_random_smiles",
            "seed",
            "replica",
            "split",
        ]
    )
    compression: str = "brotli"
    compression_level: int = 5


@dataclass
class SmilesParquetFormat:
    """Dataclass for smiles parquet format parameters."""

    schema: pa.Schema = field(
        default_factory=lambda: pa.schema(
            [
                ("smiles", pa.string()),
                ("unique_smiles", pa.string()),
                ("split", pa.string()),
            ]
        )
    )
    compression: str = "brotli"
    compression_level: int = 5


def convert_calibration_to_modelling(
    input_dataset: ds.Dataset,
    output_path: str | Path,
    modeling_format: ModelingParquetFormat,
) -> None:
    """Convert calibration parquet files to modeling format.

    Parameters
    ----------
    input_dataset : ds.Dataset
        Input dataset with calibration format
    output_path : str | Path
        Output directory for modeling format files
    modeling_format : ModelingParquetFormat
        Configuration for output format
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # setup dataset scanner
    scanner = input_dataset.scanner(columns=modeling_format.col_sel)

    # Calculate total number of rows across all fragments
    total_rows = scanner.count_rows()

    # Set up write options with compression
    write_options = ds.ParquetFileFormat().make_write_options(
        compression=modeling_format.compression,
        compression_level=modeling_format.compression_level,
    )

    # Write the dataset with selected columns as a single file
    ds.write_dataset(
        data=scanner,
        base_dir=output_path,
        format="parquet",
        file_options=write_options,
        existing_data_behavior="overwrite_or_ignore",
        max_rows_per_file=total_rows,
    )

    # get the number of file written
    num_files_written = len(list(output_path.glob("*.parquet")))
    assert num_files_written == 1, "Expected only one output file."

    # move the single output file to the parent directory
    old_path = list(output_path.glob("*.parquet"))[0]
    newpath = output_path.parent / f"{output_path.name}.parquet"
    old_path.rename(newpath)

    # delete the now empty output subfolder
    output_path.rmdir()


def convert_calibration_to_smiles_data(
    input_dataset: ds.Dataset,
    output_path: str | Path,
    smiles_format: SmilesParquetFormat,
) -> None:
    """Convert calibration parquet files to smiles data format.

    Parameters
    ----------
    input_dataset : ds.Dataset
        Input dataset with calibration format
    output_path : str | Path
        Output directory for smiles data format files
    smiles_format : SmilesParquetFormat
        Configuration for output format
    """

    # create temporary dir for output
    output_path = Path(output_path) if not isinstance(output_path, Path) else output_path
    output_path.mkdir(parents=True, exist_ok=True)
    temp_dir = output_path / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # read and pack data
    scanner = input_dataset.scanner(columns=[field.name for field in smiles_format.schema])
    table = scanner.to_table()

    unique_can_smiles = pc.unique(table["smiles"])

    # iterate over groups and expand unique_smiles
    for i, can_smiles in enumerate(unique_can_smiles):
        # filter the table for the current canonical smiles
        group_mask = pc.equal(table["smiles"], can_smiles)
        subset = table.filter(group_mask)

        # get split
        split = subset["split"][0].as_py()

        # split and explode unique_smiles column
        temp_smiles = pc.split_pattern(subset["unique_smiles"], "_")
        temp_smiles = list(pc.list_flatten(temp_smiles))

        # construct new table with exploded unique_smiles
        tempp_table = pa.Table.from_pydict(
            {
                "smiles": [can_smiles.as_py()] * len(temp_smiles),
                "unique_smiles": temp_smiles,
                "split": [split] * len(temp_smiles),
            },
            schema=smiles_format.schema,
        )

        # write the table to a parquet file
        output_file = temp_dir / f"part-{i}.parquet"
        pq.write_table(
            tempp_table,
            output_file,
            compression=smiles_format.compression,
            compression_level=smiles_format.compression_level,
        )

    # use datasets to read the temporary parquet files and write them as a single file
    dataset = ds.dataset(str(temp_dir), format="parquet", schema=smiles_format.schema)
    output_file = output_path / f"{output_path.name}.parquet"
    total_rows = dataset.scanner().count_rows()

    ds.write_dataset(
        data=dataset,
        base_dir=output_path,
        format="parquet",
        file_options=ds.ParquetFileFormat().make_write_options(
            compression=smiles_format.compression,
            compression_level=smiles_format.compression_level,
        ),
        existing_data_behavior="overwrite_or_ignore",
        max_rows_per_file=total_rows,
    )

    # get the number of file written
    num_files_written = len(list(output_path.glob("*.parquet")))
    assert num_files_written == 1, "Expected only one output file."

    # move the single output file to the parent directory
    old_path = list(output_path.glob("*.parquet"))[0]
    newpath = output_path.parent / f"{output_path.name}.parquet"
    old_path.rename(newpath)

    # delete temp files and directory and the now empty output subfolder
    shutil.rmtree(temp_dir)
    output_path.rmdir()


def repacking_arguments_parser() -> dict[str, str]:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Repacking results processing.")

    # input/output arguments
    parser.add_argument(
        "--input_datafolder",
        help="the input path to parquet database",
        default=str(PROJ_ROOT / "data/processed/moses_repacked/"),
        type=str,
    )
    parser.add_argument(
        "--output_datafolder",
        type=str,
        default=str(INTERIM_DATA_DIR / "dask_calibration_range/"),
        help="Output path for parquet files",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="logs/repackage_calibration_results.log",
        help="Path to the log file",
    )

    # parse arguments
    args = parser.parse_args()

    return vars(args)


def check_job_completion(
    input_file: Path, output_folder: str | Path, logger: logging.Logger
) -> bool:
    """Check if the job is completed by verifying the existence of the output file."""

    # declare local variables
    completion_status = True

    # convert output folder to Path object
    output_folder = Path(output_folder) if not isinstance(output_folder, Path) else output_folder

    # get number of row groups in the input file
    logger.info(f"Reading metadata of file: {input_file.name}")
    parquet_file = pq.ParquetFile(str(input_file))
    num_row_groups = parquet_file.num_row_groups

    # logg current check
    logger.info(
        f"Checking job completion: expecting {num_row_groups} output files in {output_folder.name}"
    )
    logger.info(f"├── folder {str(output_folder.name)}")

    # check for each expected output file
    for rg_index in range(num_row_groups):
        output_file = output_folder / f"part.{rg_index}.parquet"
        if not output_file.exists():
            logger.info(f"│   ├── Missing: {str(output_file.name)}")
            completion_status = False

    # log final status
    if completion_status:
        logger.info("└── All output files are present.")
    else:
        logger.info("└── Some output files are missing.")

    return completion_status


def calibration_range_postprocessing(input_folder, output_folder, logger):
    """Placeholder for calibration range postprocessing logic."""

    # declare local variables
    unfinished_jobs = []
    output_file_parameter = CalibrationParquetFormat()
    modelling_file_parameter = ModelingParquetFormat()

    # parse unput as Path
    input_folder = Path(input_folder) if not isinstance(input_folder, Path) else input_folder
    output_folder = Path(output_folder) if not isinstance(output_folder, Path) else output_folder

    # iterate over parquet files in the input folder
    for parquet_file in input_folder.glob("*.parquet"):
        #
        logger.info(f"Processing file: {parquet_file.name}")

        # output subfolder for the current parquet file
        output_subfolder = Path(output_folder) / parquet_file.stem

        # check the completion of the job
        completion_status = check_job_completion(parquet_file, output_subfolder, logger)

        # skip processing if job not completed
        if not completion_status:
            logger.info(f"Job not completed for file: {parquet_file.name}. Skipping processing.")
            # retrieve index of unfinished job
            index = str(parquet_file.stem).split("part-")[-1]
            unfinished_jobs.append(index)
            continue

        # perform postprocessing
        logger.info(f"Performing postprocessing for file: {parquet_file}")
        start_time = time.time()

        # use pyarrow dataset to read all parquet files in the output subfolder
        dataset = ds.dataset(str(output_subfolder), output_file_parameter.schema)

        # write modelling_file
        processed_output_folder = PROCESSED_DATA_DIR / "modelling_data" / parquet_file.stem
        processed_output_folder.mkdir(parents=True, exist_ok=True)
        convert_calibration_to_modelling(
            dataset, str(processed_output_folder), modelling_file_parameter
        )

        # process
        processed_output_path = PROCESSED_DATA_DIR / "smiles_data" / parquet_file.stem
        processed_output_path.mkdir(parents=True, exist_ok=True)
        convert_calibration_to_smiles_data(
            dataset, str(processed_output_path), SmilesParquetFormat()
        )

        # log elapsed time for postprocessing
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(
            f"Postprocessing completed for file: {parquet_file.name} in {elapsed_time:.2f} seconds."
        )

    # log unfinished jobs
    if unfinished_jobs:
        logger.info(f"Unfinished jobs indices: {','.join(unfinished_jobs)}")


def main():
    """Main function for repackaging calibration results."""

    # set the working directory
    os.chdir(PROJ_ROOT)

    # parse command line arguments
    args = repacking_arguments_parser()

    # make sure logs folder exists
    log_path = Path(args["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # delete existing log file if it exists
    if log_path.exists():
        log_path.unlink()

    # set up logging
    logging.basicConfig(
        filename=args["log_path"],
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()

    # start repackaging process
    logger.info("Starting repackaging of calibration results.")
    calibration_range_postprocessing(
        args["input_datafolder"],
        args["output_datafolder"],
        logger,
    )
    logger.info("Repackaging of calibration results completed.")


if __name__ == "__main__":
    main()
