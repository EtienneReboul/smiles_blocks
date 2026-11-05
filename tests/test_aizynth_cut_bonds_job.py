"""Tests for the stage-A retrosynthesis job script.

AiZynthFinder is not installed here (and needs Python 3.11, which this env is
not), so the searcher is stubbed. What is under test is the job's plumbing --
reading reference SMILES, parallel dispatch, the parquet schema, incremental
flushing, resume and skip -- which is where a long cluster job actually loses
work.
"""

import jobs_scripts.aizynth_cut_bonds as job
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from smiles_blocks.retro_backends.aizynth import SearchOutcome
from smiles_blocks.retro_backends.store import load_cut_bonds

PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
MOLECULES = [PARACETAMOL, "CCOc1ccccc1", "c1ccccc1", "CCN(CC)CC", "CC(C)Cc1ccccc1"]


@pytest.fixture
def smiles_part(tmp_path):
    """A miniature smiles_data part: many randomised rows, few unique references."""
    rows = []
    for molecule in MOLECULES:
        # Mimic the real layout: each reference SMILES repeats across many
        # randomised rows.
        for index in range(7):
            rows.append(
                {"smiles": molecule, "unique_smiles": f"{molecule}.{index}", "split": "train"}
            )

    # Kept in its own directory so it can never collide with a test's output path.
    folder = tmp_path / "smiles_data"
    folder.mkdir()
    path = folder / "part-0.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def stub_search(bonds_for=lambda smiles: {frozenset((0, 1)): {0: {"stub"}, 1: {"stub"}}}):
    """Build a stand-in for :func:`search_one` with deterministic results."""

    def _search(finder, smiles, **kwargs):
        bonds = bonds_for(smiles)
        return SearchOutcome(
            ref_smiles=smiles,
            retro_success=bool(bonds),
            n_routes=1 if bonds else 0,
            n_steps=1 if bonds else 0,
            search_time_s=0.01,
            error="" if bonds else "no_route",
            bonds=bonds,
        )

    return _search


@pytest.fixture
def stubbed_job(monkeypatch):
    """Replace model loading and searching; keep all the job's own logic."""
    monkeypatch.setattr(job, "load_finder", lambda *a, **k: object())
    monkeypatch.setattr(job, "search_one", stub_search())


def run_job(argv):
    """Invoke the job's main() with the given command line."""
    import sys as _sys

    original = _sys.argv
    _sys.argv = ["aizynth_cut_bonds.py", *argv]
    try:
        job.main()
    finally:
        _sys.argv = original


class TestReadReferenceSmiles:
    def test_deduplicates_to_reference_molecules(self, smiles_part):
        """35 rows, 5 unique references -- this ratio is the whole point of staging."""
        assert job.read_reference_smiles(str(smiles_part)) == sorted(MOLECULES)

    def test_sorted_for_reproducibility(self, smiles_part):
        result = job.read_reference_smiles(str(smiles_part))
        assert result == sorted(result)


class TestOutcomeToRow:
    def test_matches_the_schema(self):
        outcome = SearchOutcome(
            ref_smiles=PARACETAMOL,
            retro_success=True,
            n_routes=3,
            n_steps=2,
            search_time_s=11.5,
            bonds={frozenset((1, 4)): {1: {"amide"}, 4: {"amide"}}},
        )
        table = pa.Table.from_pylist(
            [job.outcome_to_row(outcome)], schema=job.CutBondsParquet().schema
        )
        assert table.num_rows == 1

    def test_clamps_counts_to_int16(self):
        """n_routes can exceed int16 on a wide search; clamping beats overflow."""
        row = job.outcome_to_row(SearchOutcome(ref_smiles="C", n_routes=10**6))
        assert row["n_routes"] == 32767
        pa.Table.from_pylist([row], schema=job.CutBondsParquet().schema)

    def test_failure_is_recorded_not_dropped(self):
        row = job.outcome_to_row(SearchOutcome(ref_smiles="C", error="no_route"))
        assert row["retro_success"] is False
        assert row["error"] == "no_route"
        assert row["bonds"] == []


class TestJobRun:
    def test_writes_every_molecule(self, smiles_part, tmp_path, stubbed_job):
        output = tmp_path / "out" / "part-0.parquet"
        run_job(
            [
                "--input",
                str(smiles_part),
                "--output",
                str(output),
                "--config",
                "unused.yml",
                "--num-workers",
                "2",
            ]
        )

        table = pq.read_table(output)
        assert table.num_rows == len(MOLECULES)
        assert set(table["ref_smiles"].to_pylist()) == set(MOLECULES)

    def test_output_is_loadable_by_stage_b(self, smiles_part, tmp_path, stubbed_job):
        """The real contract: stage B must be able to read what stage A wrote."""
        output = tmp_path / "part-0.parquet"
        run_job(
            [
                "--input",
                str(smiles_part),
                "--output",
                str(output),
                "--config",
                "unused.yml",
                "--num-workers",
                "2",
            ]
        )

        loaded = load_cut_bonds(output)
        assert set(loaded) == set(MOLECULES)
        assert loaded[PARACETAMOL] == {frozenset((0, 1)): {0: {"stub"}, 1: {"stub"}}}

    def test_leaves_no_partial_file_behind(self, smiles_part, tmp_path, stubbed_job):
        output = tmp_path / "part-0.parquet"
        run_job(
            [
                "--input",
                str(smiles_part),
                "--output",
                str(output),
                "--config",
                "unused.yml",
                "--num-workers",
                "2",
            ]
        )
        assert not list(tmp_path.glob("*.partial.parquet"))

    def test_flushing_does_not_lose_or_duplicate_rows(self, smiles_part, tmp_path, stubbed_job):
        """Flush every molecule, to exercise the row-group boundary logic."""
        output = tmp_path / "part-0.parquet"
        run_job(
            [
                "--input",
                str(smiles_part),
                "--output",
                str(output),
                "--config",
                "unused.yml",
                "--num-workers",
                "2",
                "--flush-every",
                "1",
            ]
        )

        written = pq.read_table(output)["ref_smiles"].to_pylist()
        assert sorted(written) == sorted(MOLECULES)

    def test_limit_truncates(self, smiles_part, tmp_path, stubbed_job):
        output = tmp_path / "part-0.parquet"
        run_job(
            [
                "--input",
                str(smiles_part),
                "--output",
                str(output),
                "--config",
                "unused.yml",
                "--num-workers",
                "2",
                "--limit",
                "2",
            ]
        )
        assert pq.read_table(output).num_rows == 2

    def test_existing_output_is_skipped(self, smiles_part, tmp_path, stubbed_job):
        """Matches smiles2block.py's relaunch behaviour."""
        output = tmp_path / "part-0.parquet"
        run_job(
            [
                "--input",
                str(smiles_part),
                "--output",
                str(output),
                "--config",
                "unused.yml",
                "--num-workers",
                "2",
                "--limit",
                "2",
            ]
        )
        before = pq.read_table(output).num_rows

        # Without --resume the job must leave the file alone.
        run_job(
            [
                "--input",
                str(smiles_part),
                "--output",
                str(output),
                "--config",
                "unused.yml",
                "--num-workers",
                "2",
            ]
        )
        assert pq.read_table(output).num_rows == before

    def test_resume_completes_without_redoing_work(self, smiles_part, tmp_path, stubbed_job):
        """A task killed by the time limit must be completable, not restarted."""
        output = tmp_path / "part-0.parquet"
        run_job(
            [
                "--input",
                str(smiles_part),
                "--output",
                str(output),
                "--config",
                "unused.yml",
                "--num-workers",
                "2",
                "--limit",
                "2",
            ]
        )
        partial = pq.read_table(output)["ref_smiles"].to_pylist()
        assert len(partial) == 2

        run_job(
            [
                "--input",
                str(smiles_part),
                "--output",
                str(output),
                "--config",
                "unused.yml",
                "--num-workers",
                "2",
                "--resume",
            ]
        )

        final = pq.read_table(output)["ref_smiles"].to_pylist()
        assert sorted(final) == sorted(MOLECULES), "resume must fill in the rest"
        assert len(final) == len(set(final)), "resume must not duplicate rows"

    def test_failures_are_persisted(self, smiles_part, tmp_path, monkeypatch):
        """A molecule with no route still gets a row, so the part file is complete."""
        monkeypatch.setattr(job, "load_finder", lambda *a, **k: object())
        monkeypatch.setattr(
            job,
            "search_one",
            stub_search(
                bonds_for=lambda smiles: (
                    {} if smiles == "c1ccccc1" else {frozenset((0, 1)): {0: {"stub"}, 1: {"stub"}}}
                )
            ),
        )

        output = tmp_path / "part-0.parquet"
        run_job(
            [
                "--input",
                str(smiles_part),
                "--output",
                str(output),
                "--config",
                "unused.yml",
                "--num-workers",
                "2",
            ]
        )

        table = pq.read_table(output).to_pylist()
        by_smiles = {row["ref_smiles"]: row for row in table}
        assert len(table) == len(MOLECULES)
        assert by_smiles["c1ccccc1"]["retro_success"] is False
        assert by_smiles["c1ccccc1"]["error"] == "no_route"
        # And stage B sees it as "searched, nothing usable" rather than missing.
        assert load_cut_bonds(output)["c1ccccc1"] == {}


class TestThreadPinning:
    def test_single_threaded_defaults_are_set_on_import(self):
        """Without this, 64 workers each spawn 64 ONNX threads and the node thrashes."""
        import os

        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "ORT_NUM_THREADS"):
            assert os.environ.get(var) == "1", f"{var} not pinned"
