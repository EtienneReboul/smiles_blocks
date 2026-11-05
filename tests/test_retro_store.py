"""Tests for persistence of precomputed retrosynthetic cut bonds.

The round trip must be exact: stage A writes bonds to parquet, stage B reads them
back, and what comes back has to be byte-for-byte the same contract that
``retrosynthetic_analysis`` would have returned in-process. Anything lost here
shows up as silently reduced fragmentation rather than as an error.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rdkit import Chem

from smiles_blocks.retro_backends.store import (
    CutBondsParquet,
    bonds_to_records,
    load_cut_bonds,
    records_to_bonds,
)
from smiles_blocks.retrosynthesis import retrosynthetic_analysis

PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"


def write_table(path, rows):
    """Write rows to parquet using the production schema."""
    config = CutBondsParquet()
    pq.write_table(pa.Table.from_pylist(rows, schema=config.schema), path)
    return path


def make_row(ref_smiles, bonds, *, success=True):
    """Build a complete stage-A row for the given bond dict."""
    return {
        "ref_smiles": ref_smiles,
        "retro_success": success,
        "n_routes": 1,
        "n_steps": 1,
        "search_time_s": 1.0,
        "backend": "aizynth",
        "error": "",
        "bonds": bonds_to_records(bonds),
    }


class TestRoundTrip:
    def test_single_bond_exact(self):
        bonds = {frozenset((3, 7)): {3: {"amide"}, 7: {"amide"}}}
        assert records_to_bonds(bonds_to_records(bonds)) == bonds

    def test_multiple_tags_per_atom(self):
        bonds = {frozenset((1, 2)): {1: {"a", "b"}, 2: {"c"}}}
        assert records_to_bonds(bonds_to_records(bonds)) == bonds

    def test_several_bonds(self):
        bonds = {
            frozenset((0, 5)): {0: {"x"}, 5: {"y"}},
            frozenset((2, 9)): {2: {"z"}, 9: {"z"}},
        }
        assert records_to_bonds(bonds_to_records(bonds)) == bonds

    def test_empty(self):
        assert bonds_to_records({}) == []
        assert records_to_bonds([]) == {}
        assert records_to_bonds(None) == {}

    def test_keys_are_frozensets_of_int(self):
        bonds = {frozenset((3, 7)): {3: {"t"}, 7: {"t"}}}
        restored = records_to_bonds(bonds_to_records(bonds))
        key = next(iter(restored))
        assert isinstance(key, frozenset)
        assert all(isinstance(rank, int) for rank in key)

    def test_serialisation_is_deterministic(self):
        """begin_idx is always the lower rank, so output is stable and diffable."""
        bonds = {frozenset((9, 2)): {9: {"a"}, 2: {"b"}}}
        records = bonds_to_records(bonds)
        assert records[0]["begin_idx"] == 2
        assert records[0]["end_idx"] == 9
        assert records[0]["begin_tag"] == "b"
        assert records[0]["end_tag"] == "a"

    def test_real_rbrics_output_round_trips(self):
        """The format must carry whatever the existing backend produces."""
        bonds = retrosynthetic_analysis(PARACETAMOL)
        assert bonds, "precondition: R-BRICS finds bonds in paracetamol"
        assert records_to_bonds(bonds_to_records(bonds)) == bonds


class TestSchema:
    def test_accepts_a_real_row(self, tmp_path):
        bonds = retrosynthetic_analysis(PARACETAMOL)
        path = write_table(tmp_path / "part.parquet", [make_row(PARACETAMOL, bonds)])
        assert pq.read_table(path).num_rows == 1

    def test_ranks_fit_the_int32_column(self):
        """Canonical ranks are bounded by atom count, so int32 is ample."""
        config = CutBondsParquet()
        bond_struct = config.schema.field("bonds").type.value_type
        assert bond_struct.field("begin_idx").type == pa.int32()


class TestLoadCutBonds:
    def test_loads_keyed_by_reference_smiles(self, tmp_path):
        bonds = retrosynthetic_analysis(PARACETAMOL)
        path = write_table(tmp_path / "part.parquet", [make_row(PARACETAMOL, bonds)])

        loaded = load_cut_bonds(path)

        assert set(loaded) == {PARACETAMOL}
        assert loaded[PARACETAMOL] == bonds

    def test_failed_searches_load_as_empty_not_absent(self, tmp_path):
        """Lets the caller distinguish "search failed" from "molecule not run"."""
        path = write_table(
            tmp_path / "part.parquet",
            [make_row("CCO", {}, success=False)],
        )
        loaded = load_cut_bonds(path)

        assert "CCO" in loaded
        assert loaded["CCO"] == {}

    def test_ignores_bonds_of_unsuccessful_rows(self, tmp_path):
        """A row flagged unsuccessful must not contribute bonds."""
        row = make_row(PARACETAMOL, retrosynthetic_analysis(PARACETAMOL), success=False)
        path = write_table(tmp_path / "part.parquet", [row])
        assert load_cut_bonds(path)[PARACETAMOL] == {}

    def test_filters_to_requested_molecules(self, tmp_path):
        rows = [
            make_row(PARACETAMOL, retrosynthetic_analysis(PARACETAMOL)),
            make_row("CCOc1ccccc1", retrosynthetic_analysis("CCOc1ccccc1")),
        ]
        path = write_table(tmp_path / "part.parquet", rows)

        loaded = load_cut_bonds(path, ref_smiles={PARACETAMOL})

        assert set(loaded) == {PARACETAMOL}

    def test_reads_a_directory_of_parts(self, tmp_path):
        """Stage A writes one file per part; stage B may point at the directory."""
        folder = tmp_path / "retro_cut_bonds"
        folder.mkdir()
        write_table(folder / "part-0.parquet", [make_row(PARACETAMOL, {})])
        write_table(folder / "part-1.parquet", [make_row("CCO", {})])

        loaded = load_cut_bonds(folder)

        assert set(loaded) == {PARACETAMOL, "CCO"}

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(Exception):
            load_cut_bonds(tmp_path / "nope.parquet")


class TestEndToEnd:
    """A full stage-A -> stage-B hand-off, with R-BRICS standing in for the search."""

    def test_precomputed_bonds_fragment_identically_to_inline(self, tmp_path):
        """Routing bonds through parquet must not change the fragmentation.

        This is the guarantee that makes the two-stage split safe: the only thing
        that changes is where the bonds came from.
        """
        from smiles_blocks.smiles_fragmentation import (
            get_semantic_mem_score,
            process_set_smiles,
        )

        inline_bonds = retrosynthetic_analysis(PARACETAMOL)
        path = write_table(tmp_path / "part.parquet", [make_row(PARACETAMOL, inline_bonds)])
        reloaded_bonds = load_cut_bonds(path)[PARACETAMOL]

        randomized = set(
            Chem.MolToRandomSmilesVect(
                Chem.MolFromSmiles(PARACETAMOL),
                200,
                kekuleSmiles=True,
                isomericSmiles=False,
                allBondsExplicit=True,
                randomSeed=7,
            )
        )
        scored = [(smiles, get_semantic_mem_score(smiles)) for smiles in randomized]

        inline_result = process_set_smiles(scored, inline_bonds)
        reloaded_result = process_set_smiles(scored, reloaded_bonds)

        assert inline_result[0] is True
        assert inline_result[1] == reloaded_result[1]
        assert inline_result[2] == reloaded_result[2]
