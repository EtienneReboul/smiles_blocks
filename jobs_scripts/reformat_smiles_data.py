import argparse
from itertools import batched
import logging
import os
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.dataset as ds

from smiles_blocks.files import PROCESSED_DATA_DIR, PROJ_ROOT


def reformat_argparse() -> dict[str, str | int]:
    """
    Parse command-line arguments for reformatting SMILES data.
    Configures and returns parsed arguments for the SMILES data reformatting process,
    including input/output paths, logging, and compression settings.
    Returns
    -------
    dict[str, str | int]
        A dictionary containing the parsed command-line arguments with the following keys:
        - input_folder : str
            Path to the input folder containing SMILES data.
        - output_folder : str
            Path to the output folder where reformatted data will be saved.
        - log_file : str
            Path to the log file.
        - nb_combined_files : int
            Number of output files to generate.
        - compression : str
            Compression format for output files. Options: 'gzip', 'brotli', 'zstd', 'snappy', 'lz4'.
        - compression_level : int
            Compression level (0-11 for brotli).
        - cpu_threads : int
            Number of CPU threads for compression operations.
        - io_threads : int
            Number of I/O threads for disk operations.
    """

    parser = argparse.ArgumentParser(description="Reformat SMILES data")

    # Add arguments for input and output folders
    parser.add_argument(
        "--input_folder",
        type=str,
        default=str(PROCESSED_DATA_DIR / "smiles_data/"),
        help="Path to the input folder containing SMILES data",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=str(PROCESSED_DATA_DIR / "smiles_data_compressed/"),
        help="Path to the output folder where reformatted data will be saved",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=str(PROJ_ROOT / "logs/reformat_smiles_data.log"),
        help="Path to the log file",
    )

    # Add formatting options
    parser.add_argument(
        "--nb-combined-files",
        type=int,
        default=38,
        help="Number of output files to generate",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="brotli",
        choices=["gzip", "brotli", "zstd", "snappy", "lz4"],
        help="Compression format for output files (e.g., gzip, bz2, zip)",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=9,
        help="Compression level (0-11 for brotli)",
    )

    # Add thread configuration options
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="Number of CPU threads for compression (default: 4)",
    )
    parser.add_argument(
        "--io-threads",
        type=int,
        default=2,
        help="Number of I/O threads for disk operations (default: 2)",
    )

    return vars(parser.parse_args())


def main() -> None:
    """
    Main entry point for reformatting SMILES data files.
    This function orchestrates the process of reading parquet files from an input directory,
    repackaging smaller files into larger batches, and writing them to an output directory
    with specified compression settings.
    The workflow includes:
    1. Changing working directory to project root
    2. Parsing command-line arguments for input/output paths and compression settings
    3. Creating necessary output and log directories
    4. Setting up logging to file
    5. Sorting input parquet files by size
    6. Combining smaller files into batches of 2 files each
    7. Writing each batch to a single output parquet file with compression
    8. Logging processing time for each batch
    Returns
    -------
    None
    Notes
    -----
    - Input files are sorted by size to optimize processing
    - Output files are named as 'part-{i}.parquet' where i is the batch index
    - All rows from a batch are written to a single parquet file
    - Requires command-line arguments: input_folder, output_folder, log_file,
        nb_combined_files, compression, and compression_level
    """

    # change working directory to project root
    os.chdir(PROJ_ROOT)

    # Parse command-line arguments
    args = reformat_argparse()
    data_path = Path(args["input_folder"])
    output_path = Path(args["output_folder"])
    log_path = Path(args["log_file"])

    # Create output and log directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=log_path,
    )
    logger = logging.getLogger(__name__)

    # Configure PyArrow thread pools for optimal compression performance
    # CPU threads: for brotli compression (CPU intensive)
    # I/O threads: for disk read/write operations
    cpu_threads = args["cpu_threads"]
    io_threads = args["io_threads"]
    pa.set_cpu_count(cpu_threads)
    pa.set_io_thread_count(io_threads)
    logger.info(
        f"PyArrow thread configuration: CPU threads={cpu_threads}, I/O threads={io_threads}"
    )

    # order files by size
    parquet_list = list(data_path.glob("*.parquet"))
    parquet_list.sort(key=lambda x: x.stat().st_size)

    # repackage files into fewer, larger files
    repackaged_idx = args["nb_combined_files"]
    packed_parquet = list(batched(parquet_list[:repackaged_idx], 2))
    packed_parquet.extend(parquet_list[repackaged_idx:])

    # Set up write options with compression
    write_options = ds.ParquetFileFormat().make_write_options(
        compression=args["compression"],
        compression_level=args["compression_level"],
    )

    # Process each batch of files
    for i, batch in enumerate(packed_parquet):
        # log batch processing
        if isinstance(batch, tuple) or isinstance(batch, list):
            batch_files = ", ".join(file.name for file in batch)
        elif isinstance(batch, Path):
            batch_files = batch.name
        logger.info(f"Processing batch {i} with files: {batch_files}")

        # setup loop variables
        temp_dataset = ds.dataset(batch, format="parquet")
        temp_dir = output_path / f"batch_{i}"
        newpath = temp_dir.parent / f"part-{i}.parquet"
        start_time = time.time()

        # skip batch if output file already exists
        if newpath.exists():
            logger.info(f"File {newpath.name} already exists. Skipping batch {i}.")
            continue

        # count total number of rows in the batch to ensure all rows are written to a single file
        num_rows = temp_dataset.scanner().count_rows()

        ds.write_dataset(
            temp_dataset,
            temp_dir,
            format="parquet",
            file_options=write_options,
            existing_data_behavior="overwrite_or_ignore",
            max_rows_per_file=num_rows,  # Write all rows to a single file
        )

        # check that only one file was written
        num_files_written = len(list(temp_dir.glob("*.parquet")))
        assert num_files_written == 1, "Expected only one output file."

        # move the single output file to the parent directory
        old_path = list(temp_dir.glob("*.parquet"))[0]

        old_path.rename(newpath)

        # delete temp folder
        temp_dir.rmdir()

        # log batch completion time
        end_time = time.time()
        logger.info(f"Batch {i} completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
