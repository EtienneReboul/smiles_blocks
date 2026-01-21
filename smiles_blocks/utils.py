"""
This module provides utility functions to perform common operations on dictionaries,
time parsing, and schema conversions. The functions included are:

1. parse_time_left(time_str: str) -> float:
    Parse time strings in various formats and convert to future timestamp.

2. reverse_dict(input_dict: dict) -> dict:
    Reverses the keys and values of a given dictionary.

3. dict_slicer(input_dict: dict, keys: list) -> dict:
    Slices a dictionary based on a list of keys.

4. tuple2avroschema(avro_tuple: tuple) -> dict:
    Converts an Avro schema tuple to a dictionary format.

5. tuple2parquetschema(parquet_tuple: tuple) -> pyarrow.Schema:
    Converts a ParquetSchema tuple to a pyarrow schema.

6. replace_none_values(data_dict: dict, schema: dict) -> dict:
    Replaces None or NA values in a dictionary according to Avro schema types.

7. wrapper_csv_reader(csv_path: str, line_idx: int) -> dict:
    Reads a specific line from a CSV file and converts it to a dictionary.

Functions are documented individually with detailed parameter descriptions and examples.

"""

# built-in modules
import csv
from datetime import timedelta
import logging
import os
from pathlib import Path
import time

# third-party modules
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def parse_time_left(time_str):
    """
    Parse the time string from squeue -O TimeLeft format into seconds.
    Handles 'mm:ss', 'hh:mm:ss', and 'dd-hh:mm:ss' formats.

    Args:
        time_str (str): Time string in one of the following formats:
            - 'mm:ss' for less than an hour
            - 'hh:mm:ss' for less than a day
            - 'dd-hh:mm:ss' for multiple days

    Returns:
        float: Time in seconds since epoch for the end time

    Examples:
        >>> parse_time_left("45:30")  # 45 minutes, 30 seconds
        >>> parse_time_left("02:30:00")  # 2 hours, 30 minutes
        >>> parse_time_left("2-12:30:00")  # 2 days, 12 hours, 30 minutes
    """
    time_str = time_str.strip()

    # Check for days format (dd-hh:mm:ss)
    if "-" in time_str:
        days_str, time_parts = time_str.split("-")
        days = int(days_str)
        parts = time_parts.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid days format. Expected dd-hh:mm:ss, got {time_str}")
        hours, minutes, seconds = map(int, parts)
        delta = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)

    else:
        # Handle hour or minute formats
        parts = time_str.split(":")
        if len(parts) == 2:  # mm:ss format
            minutes, seconds = map(int, parts)
            delta = timedelta(minutes=minutes, seconds=seconds)
        elif len(parts) == 3:  # hh:mm:ss format
            hours, minutes, seconds = map(int, parts)
            delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)
        else:
            raise ValueError(
                f"Unexpected time format: {time_str}. "
                "Expected formats: mm:ss, hh:mm:ss, or dd-hh:mm:ss"
            )

    # Convert to future timestamp
    end_time = time.time() + delta.total_seconds()
    return end_time


def reverse_dict(input_dict: dict) -> dict:
    """Reverse the keys and values of a dictionary.

    Args:
        input_dict (dict): The dictionary to reverse.

    Returns:
        dict: A new dictionary with keys and values reversed.
    """
    if not isinstance(input_dict, dict):
        raise ValueError(f" input is of type {type(input_dict)},Input must be a dictionary.")
    return {value: key for key, value in input_dict.items()}


def dict_slicer(input_dict: dict, keys: list) -> dict:
    """
    This function is used to slice a dictionary based on a list of keys.

    Args:
        input_dict (dict): The dictionary to slice.
        keys (list): The list of keys to keep in the sliced dictionary.

    Returns:
        dict: The sliced dictionary.
    """
    return {key: input_dict[key] for key in keys if key in input_dict}



def tuple2parquetschema(parquet_tuple: tuple) -> dict:
    """Convert a ParquetSchema tuple to a pyarrow schema.

    Args:
        parquet_tuple (tuple): ParquetSchema tuple.

    Returns:
        pyarrow.Schema: Parquet schema.
    """
    # define local variables
    fields = []
    pa_type = None

    for field_tuple in parquet_tuple:
        name, dtype = field_tuple
        if dtype == "string":
            pa_type = pa.string()
        elif dtype == "double":
            pa_type = pa.float64()
        elif dtype == "float32":
            pa_type = pa.float32()
        elif dtype == "int":
            pa_type = pa.int64()
        elif dtype == "int32":
            pa_type = pa.int32()
        elif dtype == "bool":
            pa_type = pa.bool_()
        fields.append(pa.field(name, pa_type))

    return pa.schema(fields)


def replace_none_values(data_dict: dict, schema: dict) -> dict:
    """Replace None or NA values in a dictionary according to Avro schema types.

    Args:
        data_dict (dict): Dictionary containing data that may have None values
        schema (dict): Avro schema dictionary containing field types

    Returns:
        dict: Dictionary with None values replaced by type-appropriate defaults
    """
    # local variable
    repared_dict = {}

    # Get field definitions
    fields = schema["fields"]

    # Default values by type
    defaults = {"string": "", "double": 0.0, "int": 0, "boolean": False}

    # Create type lookup from schema
    field_types = {field["name"]: field["type"] for field in fields}

    # Replace None or Nan values with defaults
    for key, value in data_dict.items():
        if value is None:
            repared_dict[key] = defaults[field_types[key]]
        elif isinstance(value, float) and np.isnan(value):
            repared_dict[key] = defaults[field_types[key]]
        else:
            repared_dict[key] = value

    return repared_dict


def wrapper_csv_reader(csv_path: str, line_idx: int) -> dict:
    """read a specific line of the csv file, convert the line into a dictionary,
    which contains the path of a single parquet file, row group number.

    Args:
        csv_path (str): Path of the instruction_file
        line_idx (int): line index in csv file

    Returns:
        instruction_dict (dict): dictionary with all the information of the parquet file.
    """
    # declare local variables
    instruction_dict = {}

    # open file
    with open(csv_path, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)

        # get header
        header = next(csv_reader)  # Assuming the first row is the header
        keys_list = header
        nb_keys = len(keys_list)

        # iterate through file and store into a dictionary
        for i, row in enumerate(csv_reader):
            if i == line_idx:
                nb_values = len(row)
                assert nb_values == nb_keys, (
                    f"Discrepancy between the number of values in row\
                    ({nb_values}) and number of columns detected in header \
                        ({nb_keys}), this happened for line : {i + 1}"
                )
                instruction_dict = {keys_list[j]: row[j] for j in range(nb_keys)}

    return instruction_dict


def read_write_setup(inout_parameters: dict, logger: logging.Logger) -> dict:
    """Setup parquet reader and CSV writer for batch processing of SMILES data.
    This function initializes a parquet reader and CSV writer for processing SMILES data in batches.
    It tracks batch processing progress through a CSV file that records batch indices
    and other metadata.
    Args:
        inout_parameters (dict): Dictionary containing:
            - output_path (str): Path where the batch index recorder CSV will be saved
            - datafile (str): Path to input parquet file
            - row_group (int): Row group to process in parquet file
            - parquet_batch_size (int): Number of rows to process per batch
        logger (logging.Logger): Logger instance for output messages
    Returns:
        dict: Dictionary containing:
            - begin_idx (int): Index to start processing from
            - csv_writer (csv.DictWriter): CSV writer for recording batch processing metadata
            - parquet_reader (ParquetReader): Iterator over parquet file batches
    Notes:
        The function handles both new processing runs and resuming from previous runs by checking
        for an existing batch index recorder file.
    """

    writer_reader_dict = {"begin_idx": 0}

    # define batch idx recorder path
    batch_idx_recorder_path = os.path.join(
        inout_parameters["output_path"], "batch_idx_recorder.csv"
    )
    fields = [
        "compute_group",
        "datafile",
        "row_group",
        "parquet_batch_size",
        "batch_idx",
        "duration",
    ]

    # check if the batch idx recorder exists
    if os.path.exists(batch_idx_recorder_path):
        # read the batch idx recorder
        with open(batch_idx_recorder_path, "r", encoding="utf-8") as f:
            csv_reader = csv.DictReader(f)
            batch_indices = [int(row["batch_idx"]) for row in csv_reader]
        writer_reader_dict["begin_idx"] = max(batch_indices) + 1 if batch_indices else 0

        # setup a new batch idx recorder in append mode
        writer_reader_dict["csv_file"] = open(
            batch_idx_recorder_path, "a", encoding="utf-8", newline=""
        )
        writer_reader_dict["csv_writer"] = csv.DictWriter(
            writer_reader_dict["csv_file"], fieldnames=fields
        )
    else:
        dir_path = Path(batch_idx_recorder_path).parent
        dir_path.mkdir(parents=True, exist_ok=True)
        # setup a new batch idx recorder in write mode
        writer_reader_dict["csv_file"] = open(
            batch_idx_recorder_path, "w", encoding="utf-8", newline=""
        )
        writer_reader_dict["csv_writer"] = csv.DictWriter(
            writer_reader_dict["csv_file"], fieldnames=fields
        )
        writer_reader_dict["csv_writer"].writeheader()

    # setup the parquet reader
    logger.info("Reading parquet file: %s", inout_parameters["datafile"])
    logger.info("Reading row group: %s", inout_parameters["row_group"])
    logger.info("Batch size: %s", inout_parameters["parquet_batch_size"])

    writer_reader_dict["parquet_reader"] = pq.ParquetFile(
        inout_parameters["datafile"]
    ).iter_batches(
        batch_size=inout_parameters["parquet_batch_size"],
        row_groups=[int(inout_parameters["row_group"])],
    )

    # Skip batches until we reach begin_idx
    if writer_reader_dict["begin_idx"] > 0:
        for _ in range(writer_reader_dict["begin_idx"]):
            next(writer_reader_dict["parquet_reader"])

    return writer_reader_dict
