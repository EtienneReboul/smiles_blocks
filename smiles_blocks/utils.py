"""
Utility functions for PyArrow tables, shared memory operations, and data processing.

This module provides utility functions to perform common operations on PyArrow tables,
shared memory management, and dictionary operations. The functions included are:

1. calculate_ipc_size(table: pa.Table) -> int:

2. export_to_shared_memory(name: str, table: pa.Table) -> shared_memory.SharedMemory:

3. retrieve_sharedmemory(name: str) -> pa.Table:

4. clear_shared_memory(name: str) -> None:

5. check_partition_not_empty(partition: pd.DataFrame) -> bool:
    Check if a Dask DataFrame partition is not empty.

6. parse_time_left(time_str: str) -> float:

7. reverse_dict(input_dict: dict) -> dict:

8. dict_slicer(input_dict: dict, keys: list) -> dict:
"""

# built-in modules
from datetime import timedelta
import logging
from multiprocessing import shared_memory
import time

# third-party modules
import pandas as pd
import pyarrow as pa


# local modules
def calculate_ipc_size(table: pa.Table) -> int:
    """
    Calculate the size of a PyArrow Table when serialized using Arrow IPC format.

    This function serializes a PyArrow Table to Arrow's IPC (Inter-Process Communication)
    streaming format and returns the size in bytes of the serialized data.

    Parameters
    ----------
    table : pa.Table
        The PyArrow Table to calculate the serialized size for.

    Returns
    -------
    int
        The size in bytes of the table when serialized in IPC streaming format.

    Examples
    --------
    >>> import pyarrow as pa
    >>> table = pa.table({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
    >>> size = calculate_ipc_size(table)
    >>> print(size)
    256
    """

    sink = pa.MockOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.size()


def export_to_shared_memory(name: str, table: pa.Table) -> shared_memory.SharedMemory:
    """
    Export a PyArrow Table to shared memory.

    Creates a new shared memory block with the specified name and writes the
    PyArrow Table data to it using the Arrow IPC (Inter-Process Communication)
    format.

    Parameters
    ----------
    name : str
        The name of the shared memory block to create. This name can be used
        by other processes to access the same shared memory.
    table : pa.Table
        The PyArrow Table object to export to shared memory.

    Returns
    -------
    shared_memory.SharedMemory
        A SharedMemory object representing the created shared memory block
        containing the serialized table data.

    Notes
    -----
    The shared memory block must be manually cleaned up using the unlink()
    method when no longer needed to avoid resource leaks.

    Examples
    --------
    >>> import pyarrow as pa
    >>> from smiles_blocks.utils import export_to_shared_memory
    >>> table = pa.table({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> shm = export_to_shared_memory('my_table', table)
    >>> # Use the shared memory block...
    >>> shm.unlink()  # Clean up when done
    """

    size = calculate_ipc_size(table)
    shm = shared_memory.SharedMemory(create=True, name=name, size=size)

    stream = pa.FixedSizeBufferWriter(pa.py_buffer(shm.buf))
    with pa.RecordBatchStreamWriter(stream, table.schema) as writer:
        writer.write_table(table)

    return shm


def retrieve_sharedmemory(table_shm: shared_memory.SharedMemory) -> pa.Table:
    """
    Retrieve a PyArrow Table from shared memory.
    This function opens a shared memory buffer by name and reconstructs a PyArrow
    Table from the serialized data stored in that buffer using the Arrow IPC
    (Inter-Process Communication) streaming format.

    Note: The returned table contains zero-copy references to the shared memory buffer.
    The SharedMemory object is not closed because the table needs access to the buffer.
    In multiprocessing scenarios, worker processes should not close their handles -
    the OS will clean them up on process exit. Only the creating process should
    close and unlink the shared memory.

    Parameters
    ----------
    table_shm : shared_memory.SharedMemory
        The shared memory object containing the serialized PyArrow Table.
    Returns
    -------
    pa.Table
        The deserialized PyArrow Table retrieved from shared memory.
    Raises
    ------
    FileNotFoundError
        If no shared memory buffer with the specified name exists.
    RuntimeError
        If the shared memory buffer cannot be read or the data cannot be deserialized.
    Examples
    --------
    >>> table = retrieve_sharedmemory(shm)
    >>> print(table.schema)
    Notes
    -----
    The shared memory buffer must have been previously created and populated
    with a serialized PyArrow Table using Arrow's IPC streaming format.
    """

    with pa.ipc.open_stream(pa.py_buffer(table_shm.buf)) as reader:
        table = reader.read_all()

    return table


def clear_shared_memory(shm: shared_memory.SharedMemory) -> None:
    """
    Clear and unlink a shared memory object.
    This function closes and unlinks a shared memory object. If any error occurs during this process, it is
    logged but does not raise an exception, allowing the function to fail gracefully.
    Parameters
    ----------
    shm : shared_memory.SharedMemory
        The shared memory object to clear and unlink.
    Returns
    -------
    None
    Raises
    ------
    None
        Errors are caught and logged rather than raised.
    Notes
    -----
    This function is non-blocking and will log any errors encountered during the
    cleanup operation without interrupting program execution.
    Examples
    --------
    >>> clear_shared_memory(shm)
    """

    try:
        shm.close()
        shm.unlink()
    except Exception as e:
        logging.error(f"Non blocking error when clearing shared memory '{shm.name}': {e}")


def check_partition_not_empty(partition: pd.DataFrame) -> bool:
    """Check if a Dask DataFrame partition is not empty.
    This function checks if the provided partition of a Dask DataFrame is not empty
    (i.e., has not zero rows).
    Parameters
    ----------
    partition : pandas.DataFrame
        A partition of a Dask DataFrame.
    Returns
    -------
    bool
        True if the partition is not empty (has rows), False if it is empty.
    """
    return partition.shape[0] != 0


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
    """
    Reverse the keys and values of a dictionary.

    Parameters
    ----------
    input_dict : dict
        The input dictionary whose keys and values will be swapped.

    Returns
    -------
    dict
        A new dictionary with the original values as keys and original keys as values.

    Raises
    ------
    ValueError
        If the input is not a dictionary.

    Examples
    --------
    >>> original = {'a': 1, 'b': 2, 'c': 3}
    >>> reverse_dict(original)
    {1: 'a', 2: 'b', 3: 'c'}
    """

    if not isinstance(input_dict, dict):
        raise ValueError(f" input is of type {type(input_dict)},Input must be a dictionary.")
    return {value: key for key, value in input_dict.items()}


def dict_slicer(input_dict: dict, keys: list) -> dict:
    """
    Create a new dictionary from selected keys of an input dictionary.

    Parameters
    ----------
    input_dict : dict
        The source dictionary from which to extract values.
    keys : list
        A list of keys to extract from the input dictionary.

    Returns
    -------
    dict
        A new dictionary containing only the key-value pairs where the keys
        are present in both the input dictionary and the keys list.

    Examples
    --------
    >>> original = {'a': 1, 'b': 2, 'c': 3}
    >>> dict_slicer(original, ['a', 'c'])
    {'a': 1, 'c': 3}

    Notes
    -----
    Keys that are in the keys list but not in the input dictionary are
    silently ignored.
    """

    return {key: input_dict[key] for key in keys if key in input_dict}
