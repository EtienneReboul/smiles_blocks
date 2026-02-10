from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pytest

from smiles_blocks.dataset import (
    DatasetDownloader,
    MosesParquetFormat,
    MosesRegistry,
)


class TestMosesRegistry:
    """Test suite for MosesRegistry dataclass."""

    def test_default_values(self):
        """Test that MosesRegistry has correct default values."""
        registry = MosesRegistry()

        assert (
            registry.url
            == "https://media.githubusercontent.com/media/molecularsets/moses/refs/heads/master/data/dataset_v1.csv"
        )
        assert registry.fname == "moses_dataset.csv"
        assert registry.md5hash == "6bdb0d9526ddf5fdeb87d6aa541df213"
        assert isinstance(registry.intermediary_path, Path)
        assert isinstance(registry.output_path, Path)

    def test_custom_values(self):
        """Test MosesRegistry with custom values."""
        custom_path = Path("/custom/path")
        custom_output = Path("/custom/output")

        registry = MosesRegistry(
            url="https://custom.url/data.csv",
            fname="custom_dataset.csv",
            intermediary_path=custom_path,
            output_path=custom_output,
            md5hash="custom_hash_123",
        )

        assert registry.url == "https://custom.url/data.csv"
        assert registry.fname == "custom_dataset.csv"
        assert registry.intermediary_path == custom_path
        assert registry.output_path == custom_output
        assert registry.md5hash == "custom_hash_123"

    def test_path_types(self):
        """Test that paths are Path objects."""
        registry = MosesRegistry()
        assert isinstance(registry.intermediary_path, Path)
        assert isinstance(registry.output_path, Path)


class TestMosesParquetFormat:
    """Test suite for MosesParquetFormat dataclass."""

    def test_default_values(self):
        """Test that MosesParquetFormat has correct default values."""
        parquet_format = MosesParquetFormat()

        assert parquet_format.max_rows_per_file == 16277
        assert parquet_format.max_rows_per_group == 41
        assert parquet_format.compression == "zstd"
        assert parquet_format.compression_level == 12
        assert isinstance(parquet_format.schema, pa.Schema)

    def test_schema_structure(self):
        """Test that the schema has correct columns."""
        parquet_format = MosesParquetFormat()

        assert len(parquet_format.schema) == 2
        assert parquet_format.schema.field("SMILES").type == pa.string()
        assert parquet_format.schema.field("SPLIT").type == pa.string()

    def test_custom_values(self):
        """Test MosesParquetFormat with custom values."""
        custom_schema = pa.schema([("col1", pa.int32())])

        parquet_format = MosesParquetFormat(
            schema=custom_schema,
            max_rows_per_file=1000,
            max_rows_per_group=100,
            compression="gzip",
            compression_level=5,
        )

        assert parquet_format.schema == custom_schema
        assert parquet_format.max_rows_per_file == 1000
        assert parquet_format.max_rows_per_group == 100
        assert parquet_format.compression == "gzip"
        assert parquet_format.compression_level == 5

    def test_compression_options(self):
        """Test various compression options."""
        compressions = ["snappy", "gzip", "brotli", "lz4", "zstd", "none"]
        for comp in compressions:
            parquet_format = MosesParquetFormat(compression=comp)
            assert parquet_format.compression == comp

    def test_compression_level_range(self):
        """Test compression level values."""
        for level in [1, 5, 9, 12, 22]:
            parquet_format = MosesParquetFormat(compression_level=level)
            assert parquet_format.compression_level == level


class TestDatasetDownloader:
    """Test suite for DatasetDownloader class."""

    def test_initialization(self):
        """Test DatasetDownloader initialization."""
        registry = MosesRegistry()
        downloader = DatasetDownloader(registry)

        assert downloader.registry == registry

    @patch("smiles_blocks.dataset.pooch.retrieve")
    def test_download_success(self, mock_retrieve):
        """Test successful download."""
        mock_path = "/tmp/moses_dataset.csv"
        mock_retrieve.return_value = mock_path

        registry = MosesRegistry()
        downloader = DatasetDownloader(registry)

        result = downloader.download()

        assert result == Path(mock_path)
        mock_retrieve.assert_called_once_with(
            url=registry.url,
            known_hash=f"md5:{registry.md5hash}",
            fname=registry.fname,
            path=registry.intermediary_path,
            progressbar=True,
        )

    @patch("smiles_blocks.dataset.pooch.retrieve")
    def test_download_returns_path_object(self, mock_retrieve):
        """Test that download returns a Path object."""
        mock_retrieve.return_value = "/tmp/dataset.csv"

        registry = MosesRegistry()
        downloader = DatasetDownloader(registry)

        result = downloader.download()

        assert isinstance(result, Path)

    @patch("smiles_blocks.dataset.pooch.retrieve")
    def test_download_failure(self, mock_retrieve):
        """Test download failure handling."""
        mock_retrieve.side_effect = Exception("Download failed")

        registry = MosesRegistry()
        downloader = DatasetDownloader(registry)

        with pytest.raises(Exception, match="Download failed"):
            downloader.download()

    @patch("smiles_blocks.dataset.pooch.retrieve")
    def test_download_hash_mismatch(self, mock_retrieve):
        """Test hash mismatch handling."""
        mock_retrieve.side_effect = ValueError("Hash mismatch")

        registry = MosesRegistry()
        downloader = DatasetDownloader(registry)

        with pytest.raises(ValueError, match="Hash mismatch"):
            downloader.download()

    @patch("smiles_blocks.dataset.pooch.retrieve")
    def test_download_network_error(self, mock_retrieve):
        """Test network error handling."""
        mock_retrieve.side_effect = ConnectionError("Network error")

        registry = MosesRegistry()
        downloader = DatasetDownloader(registry)

        with pytest.raises(ConnectionError, match="Network error"):
            downloader.download()

    @patch("smiles_blocks.dataset.pooch.retrieve")
    def test_multiple_downloads(self, mock_retrieve):
        """Test multiple consecutive downloads."""
        mock_retrieve.return_value = "/tmp/dataset.csv"

        registry = MosesRegistry()
        downloader = DatasetDownloader(registry)

        result1 = downloader.download()
        result2 = downloader.download()

        assert result1 == result2
        assert mock_retrieve.call_count == 2


@pytest.fixture
def sample_csv_data():
    """Fixture providing sample CSV data."""
    return """SMILES,SPLIT
CCO,train
CC(C)O,test
CCCO,train
c1ccccc1,train
CCN,test
"""


@pytest.fixture
def temp_csv_file(tmp_path, sample_csv_data):
    """Fixture creating a temporary CSV file."""
    csv_file = tmp_path / "test_dataset.csv"
    csv_file.write_text(sample_csv_data)
    return csv_file


@pytest.fixture
def temp_output_dir(tmp_path):
    """Fixture creating a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


class TestIntegration:
    """Integration tests for the complete workflow."""

    @patch("smiles_blocks.dataset.pooch.retrieve")
    def test_full_workflow(self, mock_retrieve, temp_csv_file, temp_output_dir):
        """Test complete download and conversion workflow."""
        mock_retrieve.return_value = str(temp_csv_file)

        registry = MosesRegistry(output_path=temp_output_dir)
        downloader = DatasetDownloader(registry)

        downloaded_path = downloader.download()

        assert downloaded_path.exists()
        assert downloaded_path == temp_csv_file

    def test_parquet_schema_compatibility(self):
        """Test that parquet schema is compatible with expected data."""
        parquet_format = MosesParquetFormat()

        # Create sample data matching the schema
        data = {"SMILES": ["CCO", "CC(C)O"], "SPLIT": ["train", "test"]}
        table = pa.table(data, schema=parquet_format.schema)

        assert table.schema == parquet_format.schema
        assert len(table) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
