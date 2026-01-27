"""Dataset downloading and processing utilities for molecular datasets.

This module provides tools for downloading and repackaging molecular
datasets into optimized Parquet format. It currently supports the MOSES
(Molecular Sets) dataset and can be extended to support additional datasets.

The module handles:
- Dataset metadata management through registry dataclasses
- Automatic downloading with integrity verification using MD5 hashes
- Conversion from CSV to optimized Parquet format with configurable compression
- Command-line interface for dataset processing workflows

Classes
MosesRegistry
    Registry containing metadata and configuration for the MOSES dataset.
MosesParquetFormat
    Parquet format configuration for writing the MOSES dataset.
DatasetDownloader
    Handles downloading molecular datasets with integrity verification.

Download and process the MOSES dataset programmatically:

>>> print(f"Downloaded to: {path}")

Or use the command-line interface:

.. code-block:: bash

    python -m smiles_blocks.dataset --dataset moses --intermediary-dir /path/to/raw --formated-dir /path/to/processed --remove-raw

Notes
-----
The module requires the following external data directories to be configured:
- EXTERNAL_DATA_DIR: for storing raw downloaded files
- PROCESSED_DATA_DIR: for storing processed Parquet files

See Also
smiles_blocks.files : Module defining data directory paths"""

import argparse
from dataclasses import dataclass, field
import os
from pathlib import Path

import pooch
import pyarrow as pa
import pyarrow.dataset as ds

from smiles_blocks.files import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR


@dataclass
class MosesRegistry:
    """Registry for the MOSES dataset.

    This dataclass contains the metadata and configuration required to download
    and process the MOSES (Molecular Sets) dataset from GitHub.

    Attributes
    ----------
    url : str
        URL to the MOSES dataset CSV file on GitHub.
        Default is the official MOSES dataset v1 CSV.
    fname : str
        Filename for the downloaded dataset file.
        Default is "moses_dataset.csv".
    intermediary_path : Path
        Directory path where the raw dataset will be downloaded.
        Default is EXTERNAL_DATA_DIR.
    md5hash : str
        MD5 hash of the dataset file for integrity verification.
        Default is "6bdb0d9526ddf5fdeb87d6aa541df213".
    schema : pa.Schema
        PyArrow schema defining the structure of the dataset.
        Default schema contains two string columns: "SMILES" and "SPLIT".

    """

    url: str = "https://media.githubusercontent.com/media/molecularsets/moses/refs/heads/master/data/dataset_v1.csv"
    fname: str = "moses_dataset.csv"
    intermediary_path: Path = EXTERNAL_DATA_DIR
    output_path: Path = PROCESSED_DATA_DIR / "moses_repacked"
    md5hash: str = "6bdb0d9526ddf5fdeb87d6aa541df213"


@dataclass
class MosesParquetFormat:
    """Parquet format configuration for the MOSES dataset.

    This dataclass contains the parameters used to configure the Parquet file
    format when writing the MOSES dataset.

    Attributes
    ----------
    max_rows_per_file : int
        Maximum number of rows to write per Parquet file.
        Default is 16277.
    max_rows_per_group : int
        Maximum number of rows per row group within each Parquet file.
        Default is 397.
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
                ("SMILES", pa.string()),
                ("SPLIT", pa.string()),
            ],
            metadata={"SMILES": "Canonical SMILES representation", "SPLIT": "Dataset split"},
        )
    )
    max_rows_per_file: int = 16277
    max_rows_per_group: int = 41
    compression: str = "zstd"
    compression_level: int = 12


class DatasetDownloader:
    """Downloader for molecular datasets.

    This class handles downloading molecular datasets from remote sources
    using pooch for file retrieval and integrity verification.

    Parameters
    ----------
    registry : MosesRegistry
        Registry object containing dataset metadata including URL, filename,
        download path, and MD5 hash for verification.

    Attributes
    ----------
    registry : MosesRegistry
        The registry object passed during initialization.

    Methods
    -------
    download()
        Download the dataset file to the specified intermediary path.

    Examples
    --------
    >>> registry = MosesRegistry()
    >>> downloader = DatasetDownloader(registry)
    >>> path = downloader.download()
    """

    def __init__(self, registry: MosesRegistry) -> None:
        self.registry = registry

    def download(self) -> Path:
        """Download the dataset file using pooch.

        Returns:
            Path: Path to the downloaded file.
        """
        downloader = pooch.retrieve(
            url=self.registry.url,
            known_hash=f"md5:{self.registry.md5hash}",
            fname=self.registry.fname,
            path=self.registry.intermediary_path,
            progressbar=True,
        )
        return Path(downloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download molecular datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="moses",
        help="Name of the dataset to download (default: moses).",
    )
    parser.add_argument(
        "--intermediary-dir",
        type=str,
        default=str(EXTERNAL_DATA_DIR),
        help="Directory to save the downloaded raw dataset (default: EXTERNAL_DATA_DIR).",
    )
    parser.add_argument(
        "--formated-dir",
        type=str,
        default=str(PROCESSED_DATA_DIR / "moses_repacked"),
        help="Directory to save the processed dataset (default: PROCESSED_DATA_DIR).",
    )
    parser.add_argument(
        "--remove-raw",
        action="store_true",
        help="Remove the raw downloaded files after processing.",
    )
    args = parser.parse_args()

    # Select the appropriate registry based on the dataset name
    if args.dataset.lower() == "moses":
        registry = MosesRegistry(
            intermediary_path=Path(args.intermediary_dir),
            output_path=Path(args.formated_dir),
        )
        parquet_format = MosesParquetFormat()
    else:
        raise ValueError(f"Dataset '{args.dataset}' is not supported.")

    downloader = DatasetDownloader(registry)
    downloaded_path = downloader.download()
    print(f"Downloaded MOSES dataset to: {downloaded_path}")

    # repackage using pyarrow dataset
    file_options = ds.ParquetFileFormat().make_write_options(
        compression=parquet_format.compression
    )
    dataset = ds.dataset(str(downloaded_path), schema=parquet_format.schema, format="csv")
    ds.write_dataset(
        dataset,
        base_dir=args.formated_dir,
        format="parquet",
        max_rows_per_file=parquet_format.max_rows_per_file,
        file_options=file_options,
        max_rows_per_group=parquet_format.max_rows_per_group,
        existing_data_behavior="overwrite_or_ignore",
    )
    print(f"Processed dataset saved to: {args.formated_dir}")

    if args.remove_raw:
        os.remove(downloaded_path)
        print(f"Removed raw data file: {downloaded_path}")
