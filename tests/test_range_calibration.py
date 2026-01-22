import pandas as pd
import pytest

from smiles_blocks.range_calibration import (
    SmilesGenerationConfig,
    make_calibrationrange,
    make_datapoints,
    make_one_replica,
    smiles_set_to_col,
)


class TestSmilesGenerationConfig:
    def test_default_config(self):
        config = SmilesGenerationConfig()
        assert config.kekuleSmiles is True
        assert config.isomericSmiles is False
        assert config.allBondsExplicit is True

    def test_custom_config(self):
        config = SmilesGenerationConfig(
            kekuleSmiles=False, isomericSmiles=True, allBondsExplicit=False
        )
        assert config.kekuleSmiles is False
        assert config.isomericSmiles is True
        assert config.allBondsExplicit is False


class TestMakeCalibrationRange:
    def test_basic_range(self):
        result = list(make_calibrationrange(0, 3))
        print(result)
        assert len(result) == 30
        assert result[0] == (1, 1)
        assert result[10] == (10, 20)
        assert result[20] == (100, 210)

    def test_single_power(self):
        result = list(make_calibrationrange(0, 1))
        assert len(result) == 10
        assert all(nb == 1 for nb, _ in result)

    def test_empty_range(self):
        result = list(make_calibrationrange(5, 5))
        assert len(result) == 0


class TestMakeOneReplica:
    def test_basic_execution(self):
        smiles = "CCO"
        df, smiles_set = make_one_replica(smiles, replica=0, max_power=2, patience=10)

        assert isinstance(df, pd.DataFrame)
        assert isinstance(smiles_set, set)
        assert len(df) > 0
        assert len(smiles_set) > 0
        assert "smiles" in df.columns
        assert "nb_unique_smiles" in df.columns
        assert "cumulative_count" in df.columns
        assert "replica" in df.columns

    def test_replica_column(self):
        smiles = "CCO"
        df, _ = make_one_replica(smiles, replica=5, max_power=2, patience=10)
        assert (df["replica"] == 5).all()

    def test_custom_config(self):
        config = SmilesGenerationConfig(kekuleSmiles=False, isomericSmiles=True)
        smiles = "CCO"
        df, smiles_set = make_one_replica(
            smiles, replica=0, max_power=2, patience=10, config=config
        )
        assert len(df) > 0
        assert len(smiles_set) > 0

    def test_patience_stops_early(self):
        smiles = "C"
        df, _ = make_one_replica(smiles, replica=0, max_power=5, patience=2)
        assert len(df) < 40  # Should stop before max iterations


class TestSmilesSetToCol:
    def test_padding_when_set_smaller(self):
        smiles_set = {"CC", "CCC"}
        result = smiles_set_to_col(smiles_set, nb_row=5)
        assert len(result) == 5
        assert result.count("") == 3

    def test_exact_match(self):
        smiles_set = {"CC", "CCC", "CCCC"}
        result = smiles_set_to_col(smiles_set, nb_row=3)
        assert len(result) == 3
        assert "" not in result

    def test_empty_set_raises_error(self):
        with pytest.raises(ValueError, match="The smiles_set is empty"):
            smiles_set_to_col(set(), nb_row=5)

    def test_negative_nb_row_raises_error(self):
        with pytest.raises(ValueError, match="The nb_row must be a positive integer"):
            smiles_set_to_col({"CC"}, nb_row=0)

    def test_zero_nb_row_raises_error(self):
        with pytest.raises(ValueError, match="The nb_row must be a positive integer"):
            smiles_set_to_col({"CC"}, nb_row=-1)


class TestMakeDatapoints:
    def test_basic_execution(self):
        smiles = "CCO"
        df = make_datapoints(smiles, nb_replica=2, max_power=2, patience=5)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "smiles" in df.columns
        assert "nb_unique_smiles" in df.columns
        assert "unique_smiles" in df.columns
        assert "replica" in df.columns

    def test_multiple_replicas(self):
        smiles = "CCO"
        df = make_datapoints(smiles, nb_replica=3, max_power=2, patience=5)
        assert df["replica"].nunique() == 3

    def test_patience_validation(self):
        with pytest.raises(ValueError, match="Patience must be less than number of iterations"):
            make_datapoints("CCO", nb_replica=1, max_power=2, patience=20)

    def test_unique_smiles_column_length(self):
        smiles = "CCO"
        df = make_datapoints(smiles, nb_replica=2, max_power=2, patience=5)
        assert len(df["unique_smiles"]) == len(df)
