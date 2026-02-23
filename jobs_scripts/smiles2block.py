import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from itertools import batched
import logging
import multiprocessing as mp
from multiprocessing import shared_memory
import os
from pathlib import Path
import re
import shutil
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from smiles_blocks.files import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, PROJ_ROOT
from smiles_blocks.retrosynthesis import retrosynthetic_analysis
from smiles_blocks.smiles_fragmentation import (
    BlockedSmilesResult,
    SmilesRegex,
    get_semantic_mem_score,
    process_set_smiles,
)
from smiles_blocks.utils import clear_shared_memory, export_to_shared_memory, retrieve_sharedmemory


@dataclass
class BlockedSmilesParquet:
    schema: pa.Schema = field(
        default_factory=lambda: pa.schema(
            [
                ("ref_smiles", pa.string()),
                ("retrosynthetic_success", pa.bool_()),
                ("fragmentation_success", pa.bool_()),
                ("smiles", pa.string()),
                ("smiles_blocked", pa.string()),
                ("mem_score", pa.float32()),
                ("unique_id_seq", pa.string()),
                ("retro_bond_ratio", pa.float32()),
                ("nb_block_cq_ok", pa.int16()),
            ]
        )
    )
    compression: str = "zstd"
    compression_level: int = 14


@dataclass
class BlockLibraryParquet:
    schema: pa.Schema = field(
        default_factory=lambda: pa.schema(
            [
                ("block", pa.string()),
                ("can_smiles", pa.string()),
                ("first_connected_can_idx", pa.int32()),
                ("last_connected_can_idx", pa.int32()),
                ("unique_id", pa.string()),
                ("begin_tag", pa.string()),
                ("end_tag", pa.string()),
                ("MolWt", pa.float32()),
                ("nHDonors", pa.int16()),
                ("nHAcceptors", pa.int16()),
                ("nRotatableBonds", pa.int16()),
                ("CrippenlogP", pa.float32()),
                ("TPSA", pa.float32()),
                ("status", pa.bool_()),
            ]
        )
    )
    compression: str = "zstd"
    compression_level: int = 14


@dataclass
class PostProcessingConfig:
    compression: str = "brotli"
    compression_level: int = 5


def smiles2block_argparse() -> dict:
    parser = argparse.ArgumentParser(description="Convert SMILES to blocks")

    # input and output arguments
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROCESSED_DATA_DIR / "smiles_data/part-0.parquet"),
        help="Path to the directory or parquet file containing the SMILES data to process",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=str(INTERIM_DATA_DIR),
        help="Path to the output folder where blocks will be saved",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default="logs/smiles2block.log",
        help="Path to the log file",
    )

    # compute arguments
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers to use for processing the dataset",
    )
    parser.add_argument(
        "--nb-batch-per-worker",
        type=int,
        default=10,
        help="Number of batches to process per worker (default: 10)",
    )

    return vars(parser.parse_args())


def process_batch(
    batch: list[str], shm: shared_memory.SharedMemory, smiles_regex: re.Pattern
) -> tuple[pa.Table]:
    """
    This function is used to process a batch of SMILES and convert them to blocks

    Args:
        batch (list): a batch of SMILES
        shm (shared_memory.SharedMemory): the shared memory object containing the partition data
        smiles_regex (re.Pattern): the regex pattern to tokenize the SMILES

    Returns:
        list: a list of blocks
    """
    # declare local variables
    batch_results = defaultdict(list)
    blocked_smiles_parquet = BlockedSmilesParquet()
    block_library_parquet = BlockLibraryParquet()

    # read the partition from shared memory
    partition_table = retrieve_sharedmemory(shm)

    for smiles in batch:
        # instantiate the result object for the current SMILES

        retro_synth_results = retrosynthetic_analysis(smiles)
        if not retro_synth_results:
            # if retrosynthetic analysis fails, we still want to record the SMILES in the blocked_smiles table with fragmentation_success = False and an empty block library
            blocked_smiles_results = vars(BlockedSmilesResult())
            blocked_smiles_results["ref_smiles"] = smiles
            blocked_smiles_results["retrosynthetic_success"] = False
            blocked_smiles_results["fragmentation_success"] = False
            batch_results["blocked_smiles"].append(blocked_smiles_results)
            continue

        # filter table to get the row corresponding to the current SMILES
        mask = pc.equal(partition_table["smiles"], smiles)
        rd_smiles = partition_table.filter(mask)["unique_smiles"].to_pylist()

        # compute semantic score
        mem_rd_smiles = [
            (smiles, get_semantic_mem_score(smiles, smiles_regex)) for smiles in rd_smiles
        ]

        status, blocked_smiles_results, block_results = process_set_smiles(
            mem_rd_smiles, retro_synth_results
        )

        # add info to blocked_smiles_results
        blocked_smiles_results["ref_smiles"] = smiles
        blocked_smiles_results["retrosynthetic_success"] = True
        blocked_smiles_results["fragmentation_success"] = status

        # record the results in the batch results
        batch_results["blocked_smiles"].append(blocked_smiles_results)
        batch_results["blocks"].extend(block_results)

    # package the batch results into a PyArrow Table and return it
    blocked_smiles_table = pa.Table.from_pylist(
        batch_results["blocked_smiles"], schema=blocked_smiles_parquet.schema
    )
    block_library_table = pa.Table.from_pylist(
        batch_results["blocks"], schema=block_library_parquet.schema
    )

    return blocked_smiles_table, block_library_table


def process_partition(
    partition: pa.Table,
    nb_workers: int,
    nb_batch_per_worker: int,
    block_library_path: Path,
    blocked_smiles_path: Path,
) -> pa.Table:
    """
    This function is used to process a partition of the dataset and convert the SMILES to blocks

    Args:
        partition (pa.Table): a partition of the dataset

    Returns:
        pa.Table: a partition of the dataset with the blocks
    """
    # declare local variables
    batch_count = 0
    blocked_smiles_parquet = BlockedSmilesParquet()
    block_library_parquet = BlockLibraryParquet()
    smiles_regex = SmilesRegex().regex

    # make sure that output directories exist
    block_library_path.mkdir(parents=True, exist_ok=True)
    blocked_smiles_path.mkdir(parents=True, exist_ok=True)

    # get the unique SMILES in the partition
    unique_smiles = pc.unique(partition["smiles"]).to_pylist()

    # batch the unique SMILES into smaller batches to avoid memory issues
    batch_size, _ = divmod(len(unique_smiles), nb_workers * nb_batch_per_worker)

    # generating the batches
    batches = list(batched(unique_smiles, batch_size))

    # free memory
    del unique_smiles

    # send partition to shared memory
    shm = export_to_shared_memory("partition", partition)

    # free memory
    del partition

    # create a partial function to process the batches with the shared memory and threshold dict
    partial_process_batch = partial(process_batch, shm=shm, smiles_regex=smiles_regex)

    # process the batches in parallel
    with mp.Pool(processes=nb_workers) as pool:
        batch_results = pool.imap_unordered(partial_process_batch, batches)

        for batch_result in batch_results:
            logging.info(f"Processed batch {batch_count}/{len(batches)}")
            # unpack the batch result
            blocked_smiles_table, block_library_table = batch_result

            # write the blocked smiles table to parquet
            pq.write_table(
                blocked_smiles_table,
                blocked_smiles_path / f"blocked_smiles_{batch_count}.parquet",
                compression=blocked_smiles_parquet.compression,
                compression_level=blocked_smiles_parquet.compression_level,
            )

            # write the block library table to parquet
            pq.write_table(
                block_library_table,
                block_library_path / f"block_library_{batch_count}.parquet",
                compression=block_library_parquet.compression,
                compression_level=block_library_parquet.compression_level,
            )

            batch_count += 1

    # clean up shared memory - close the main process handle first
    clear_shared_memory(shm)


# will be moved to utils.py and tested in test_utils.py
def files_post_processing(output_dir: str | Path) -> None:
    """
    This function is used to post-process the output files after all the batches have been processed. It can be used to aggregate the block library files and remove duplicates, and to aggregate the blocked smiles files.
    """
    # declare local variables
    postprocessing_config = PostProcessingConfig()

    # make sure that the paths are Path objects
    output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
    temp_dir = output_dir.parent / f"{output_dir.name}_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # make dataset
    dataset = ds.dataset(output_dir, format="parquet")

    # get the number of row
    scanner = dataset.scanner()
    num_rows = scanner.count_rows()

    # Set up write options with compression
    write_options = ds.ParquetFileFormat().make_write_options(
        compression=postprocessing_config.compression,
        compression_level=postprocessing_config.compression_level,
    )

    # write the dataset to a single parquet file with the specified compression and compression level
    ds.write_dataset(
        data=dataset,
        base_dir=temp_dir,
        format="parquet",
        file_options=write_options,
        existing_data_behavior="overwrite_or_ignore",
        max_rows_per_file=num_rows,  # Write all rows to a single file
        # default is ~ 1 millons rows per group, which is not happening with memory
        # limitations, adjust automatically to write one row group only
        # if less than 1 million rows,
        max_rows_per_group=num_rows if num_rows < 1024**2 else 1024**2,
    )
    # get the number of file written
    num_files_written = len(list(temp_dir.glob("*.parquet")))
    assert num_files_written == 1, "Expected only one output file."

    # move the single output file to the parent directory
    old_path = list(temp_dir.glob("*.parquet"))[0]
    newpath = output_dir.parent / f"{output_dir.name}.parquet"
    old_path.rename(newpath)

    # delete temp folder and directory and the now empty output subfolder
    temp_dir.rmdir()
    shutil.rmtree(output_dir)


def main():
    # set working directory to project root
    os.chdir(PROJ_ROOT)

    # parse the arguments
    args = smiles2block_argparse()

    # set up logging
    logging.basicConfig(
        filename=args["logfile"],
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    # create the 2 output folder if it does not exist
    block_library_path = Path(args["output_folder"]) / "block_library"
    block_library_path.mkdir(parents=True, exist_ok=True)
    blocked_smiles_path = Path(args["output_folder"]) / "blocked_smiles"
    blocked_smiles_path.mkdir(parents=True, exist_ok=True)

    # use pyarrow to aggregate all the parquet files in the input folder and save them as a single parquet file in the output file

    dataset = ds.dataset(args["input"], format="parquet")

    # iterate over the dataset and convert the SMILES to blocks

    for file in dataset.files:
        # parse the path an get the file name
        input_path = Path(file)
        file_name = input_path.name
        file_stem = input_path.stem

        logger.info(f"Processing file {file_name}...")

        # check if output file already exists, if yes, skip the file
        output_library_file = block_library_path / file_name
        output_blocked_smiles_file = blocked_smiles_path / file_name
        if output_library_file.exists() and output_blocked_smiles_file.exists():
            logger.info(f"File {file_name} already exists, skipping...")
            continue

        table = pq.read_table(file)

        # process the partition and convert the SMILES to blocks
        start = time.time()
        process_partition(
            table,
            nb_workers=args["num_workers"],
            nb_batch_per_worker=args["nb_batch_per_worker"],
            block_library_path=block_library_path / file_stem,
            blocked_smiles_path=blocked_smiles_path / file_stem,
        )
        end = time.time()
        logger.info(f"Processing file {file_name} took {end - start:.2f} seconds")

        # make the post-processing of the files to aggregate the block library files and remove duplicates, and to aggregate the blocked smiles files
        logger.info(f"Post-processing files for {file_stem}...")
        start = time.time()
        files_post_processing(block_library_path / file_stem)
        files_post_processing(blocked_smiles_path / file_stem)
        end = time.time()
        logger.info(f"Post-processing files for {file_stem} took {end - start:.2f} seconds")


if __name__ == "__main__":
    main()
