import time

import pandas as pd
import pyarrow as pa
import pytest

from smiles_blocks.utils import (
    calculate_ipc_size,
    check_partition_not_empty,
    clear_shared_memory,
    dict_slicer,
    export_to_shared_memory,
    parse_time_left,
    retrieve_sharedmemory,
    reverse_dict,
)


class TestCalculateIpcSize:
    def test_calculate_ipc_size_simple_table(self):
        table = pa.table({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        size = calculate_ipc_size(table)
        assert isinstance(size, int)
        assert size > 0

    def test_calculate_ipc_size_empty_table(self):
        table = pa.table({"col1": [], "col2": []})
        size = calculate_ipc_size(table)
        assert isinstance(size, int)
        assert size > 0

    def test_calculate_ipc_size_large_table(self):
        table = pa.table({"col1": list(range(1000)), "col2": ["x" * 10] * 1000})
        size = calculate_ipc_size(table)
        assert size > 0


class TestExportAndRetrieveSharedMemory:
    def test_export_and_retrieve_shared_memory(self):
        table = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
        name = "test_table_1"

        try:
            shm = export_to_shared_memory(name, table)
            assert shm is not None

            retrieved_table = retrieve_sharedmemory(shm.name)
            assert retrieved_table.equals(table)
        finally:
            clear_shared_memory(shm.name)

    def test_export_with_different_data_types(self):
        table = pa.table(
            {"integers": [1, 2, 3], "floats": [1.1, 2.2, 3.3], "strings": ["a", "b", "c"]}
        )
        name = "test_table_mixed"

        try:
            shm = export_to_shared_memory(name, table)
            retrieved_table = retrieve_sharedmemory(shm.name)
            assert retrieved_table.equals(table)
        finally:
            clear_shared_memory(shm.name)

    def test_retrieve_nonexistent_shared_memory(self):
        with pytest.raises(FileNotFoundError):
            retrieve_sharedmemory("nonexistent_table")


class TestClearSharedMemory:
    def test_clear_shared_memory_success(self):
        table = pa.table({"col": [1, 2, 3]})
        name = "test_clear"

        shm = export_to_shared_memory(name, table)
        clear_shared_memory(shm.name)

        with pytest.raises(FileNotFoundError):
            retrieve_sharedmemory(shm.name)

    def test_clear_nonexistent_shared_memory(self):
        # Should not raise an exception, only log
        clear_shared_memory("nonexistent_shm")


class TestCheckPartitionNotEmpty:
    def test_partition_not_empty(self):
        partition = pd.DataFrame({"a": [1, 2, 3]})
        assert check_partition_not_empty(partition) is True

    def test_partition_empty(self):
        partition = pd.DataFrame()
        assert check_partition_not_empty(partition) is False

    def test_partition_single_row(self):
        partition = pd.DataFrame({"a": [1]})
        assert check_partition_not_empty(partition) is True


class TestParseTimeLeft:
    def test_parse_mm_ss_format(self):
        result = parse_time_left("45:30")
        expected = time.time() + 45 * 60 + 30
        assert abs(result - expected) < 1  # Allow 1 second tolerance

    def test_parse_hh_mm_ss_format(self):
        result = parse_time_left("02:30:00")
        expected = time.time() + 2 * 3600 + 30 * 60
        assert abs(result - expected) < 1

    def test_parse_dd_hh_mm_ss_format(self):
        result = parse_time_left("2-12:30:00")
        expected = time.time() + 2 * 86400 + 12 * 3600 + 30 * 60
        assert abs(result - expected) < 1

    def test_parse_time_left_with_whitespace(self):
        result = parse_time_left("  45:30  ")
        expected = time.time() + 45 * 60 + 30
        assert abs(result - expected) < 1

    def test_parse_time_left_zero_values(self):
        result = parse_time_left("00:00")
        expected = time.time()
        assert abs(result - expected) < 1

    def test_parse_time_left_invalid_format(self):
        with pytest.raises(ValueError):
            parse_time_left("invalid")

    def test_parse_time_left_invalid_days_format(self):
        with pytest.raises(ValueError):
            parse_time_left("2-30:00")

    def test_parse_time_left_invalid_separator(self):
        with pytest.raises(ValueError):
            parse_time_left("45:30:00:00")


class TestReverseDict:
    def test_reverse_dict_simple(self):
        original = {"a": 1, "b": 2, "c": 3}
        result = reverse_dict(original)
        assert result == {1: "a", 2: "b", 3: "c"}

    def test_reverse_dict_empty(self):
        original = {}
        result = reverse_dict(original)
        assert result == {}

    def test_reverse_dict_single_item(self):
        original = {"key": "value"}
        result = reverse_dict(original)
        assert result == {"value": "key"}

    def test_reverse_dict_non_dict_input(self):
        with pytest.raises(ValueError):
            reverse_dict([1, 2, 3])

    def test_reverse_dict_non_dict_string_input(self):
        with pytest.raises(ValueError):
            reverse_dict("not a dict")

    def test_reverse_dict_numeric_keys(self):
        original = {1: "a", 2: "b", 3: "c"}
        result = reverse_dict(original)
        assert result == {"a": 1, "b": 2, "c": 3}


class TestDictSlicer:
    def test_dict_slicer_basic(self):
        original = {"a": 1, "b": 2, "c": 3}
        result = dict_slicer(original, ["a", "c"])
        assert result == {"a": 1, "c": 3}

    def test_dict_slicer_empty_keys(self):
        original = {"a": 1, "b": 2, "c": 3}
        result = dict_slicer(original, [])
        assert result == {}

    def test_dict_slicer_nonexistent_keys(self):
        original = {"a": 1, "b": 2}
        result = dict_slicer(original, ["x", "y", "z"])
        assert result == {}

    def test_dict_slicer_mixed_keys(self):
        original = {"a": 1, "b": 2, "c": 3}
        result = dict_slicer(original, ["a", "x", "c"])
        assert result == {"a": 1, "c": 3}

    def test_dict_slicer_single_key(self):
        original = {"a": 1, "b": 2, "c": 3}
        result = dict_slicer(original, ["b"])
        assert result == {"b": 2}

    def test_dict_slicer_empty_dict(self):
        original = {}
        result = dict_slicer(original, ["a", "b"])
        assert result == {}

    def test_dict_slicer_all_keys(self):
        original = {"a": 1, "b": 2, "c": 3}
        result = dict_slicer(original, ["a", "b", "c"])
        assert result == original
