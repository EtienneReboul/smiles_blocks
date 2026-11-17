"""Measure how much the acyclic-only restriction costs stage A.

Stage A keeps a route's cut only when both endpoints map back to the target, the
atoms are bonded there, and that bond is **acyclic** -- a ring bond cannot be
cleaved in a SMILES string. Ring-forming and ring-opening steps therefore
contribute nothing, and the stored output cannot say how much was lost, because a
discarded cut leaves no trace.

The only way to count the loss is to redo the search with the counters in
:func:`~smiles_blocks.retro_backends.bond_extraction.route_cut_bonds` switched
on. Two modes:

**Part mode** (``--input``), one array task per part, for the whole of MOSES.
Writes a per-MOLECULE parquet -- the counters plus every discarded ring bond's
ranks, smallest ring size and aromaticity. Since the search is the expensive
part, recording which bonds were lost rather than only how many means later
questions (ring size, aromatic vs aliphatic, correlation with molecule size) are
answered offline instead of paying for the search a third time.

    python jobs_scripts/measure_ring_cut_loss.py \\
        --input data/interim/ref_smiles/part-0.parquet \\
        --output data/interim/ring_cut_loss/part-0.parquet \\
        --config models/aizynthfinder/config.yml \\
        --num-workers 24 --resume

**Sample mode** (``--input-dir`` with ``--sample``), one node, for a quick
estimate. The sample is drawn at RANDOM across all parts. Taking the first N of a
part is not a sample: the reference SMILES are sorted, so the head is
alphabetically biased (which made two earlier cost estimates wrong by ~2x).

    python jobs_scripts/measure_ring_cut_loss.py \\
        --input-dir data/interim/ref_smiles \\
        --config models/aizynthfinder/config.yml \\
        --sample 5000 --num-workers 24 --out ring_loss.json
"""

from __future__ import annotations

import os

# Before any ONNX/BLAS import: one thread per worker, not one per core.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "ORT_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

from smiles_blocks.retro_backends.aizynth import (  # noqa: E402
    DEFAULT_TAG_KEYS,
    load_finder,
    route_steps,
)
from smiles_blocks.retro_backends.bond_extraction import (  # noqa: E402
    CUT_STATS_KEYS,
    route_cut_bonds,
)

logger = logging.getLogger(__name__)

_FINDER = None

#: One row per molecule searched. The ring-bond list is the point of part mode:
#: it records WHICH bonds the restriction threw away, not just how many.
RING_BOND_STRUCT = pa.struct(
    [
        pa.field("rank_a", pa.int32()),
        pa.field("rank_b", pa.int32()),
        pa.field("ring_size", pa.int32()),
        pa.field("aromatic", pa.bool_()),
    ]
)
SCHEMA = pa.schema(
    [
        pa.field("ref_smiles", pa.string()),
        pa.field("solved", pa.bool_()),
        *(pa.field(key, pa.int32()) for key in CUT_STATS_KEYS),
        pa.field("ring_bonds", pa.list_(RING_BOND_STRUCT)),
    ]
)
COMPRESSION = "zstd"
COMPRESSION_LEVEL = 14


def ring_loss_argparse() -> dict:
    """Parse the command line.

    Returns:
        dict: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="One part of reference SMILES; measures all of it.")
    source.add_argument("--input-dir", help="Directory of parts, to sample across.")
    parser.add_argument("--config", required=True, help="AiZynthFinder YAML config.")
    parser.add_argument("--output", help="Per-molecule parquet. Part mode only.")
    parser.add_argument("--sample", type=int, default=0, help="Sample size, with --input-dir.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--out", default=None, help="JSON summary; defaults next to --output.")
    parser.add_argument("--flush-every", type=int, default=200)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep an existing --output and measure only the molecules missing from it.",
    )
    parser.add_argument("--logfile", default=None)
    args = vars(parser.parse_args())
    if args["input"] and not args["output"]:
        parser.error("--input requires --output")
    if args["input_dir"] and not args["sample"]:
        parser.error("--input-dir requires --sample")
    return args


def read_reference_smiles(input_path: str) -> list[str]:
    """Collect one part's unique reference SMILES.

    Args:
        input_path: Parquet file or directory with a ``smiles`` column.

    Returns:
        list[str]: Unique reference SMILES, sorted so a resume is reproducible.
    """
    dataset = ds.dataset(input_path, format="parquet")
    unique: set[str] = set()
    for batch in dataset.to_batches(columns=["smiles"]):
        unique.update(pc.unique(batch["smiles"]).to_pylist())
    unique.discard(None)
    return sorted(unique)


def sample_smiles(input_dir: Path, n: int, seed: int) -> list[str]:
    """Draw a random sample of reference SMILES from across every part.

    Args:
        input_dir: Directory of ``part-*.parquet`` files holding reference SMILES.
        n: Sample size.
        seed: RNG seed, so the measurement is reproducible.

    Returns:
        list[str]: The sampled SMILES.
    """
    parts = sorted(input_dir.glob("part-*.parquet"), key=lambda p: int(p.stem.split("-")[1]))
    if not parts:
        raise SystemExit(f"no part-*.parquet in {input_dir}")

    rng = random.Random(seed)
    # Spread the draw over all parts rather than reading 1.9 M strings into memory.
    per_part = max(1, n // len(parts) + 1)
    pool: list[str] = []
    for part in parts:
        smiles = pq.read_table(part, columns=["smiles"])["smiles"].to_pylist()
        pool.extend(rng.sample(smiles, min(per_part, len(smiles))))

    rng.shuffle(pool)
    return pool[:n]


def already_done(output_path: Path) -> set[str]:
    """Read back the molecules a previous run already measured.

    Args:
        output_path: Partial output from an interrupted run.

    Returns:
        set[str]: Reference SMILES present in the file, empty if it is absent or
        unreadable. A corrupt tail is treated as "nothing done" rather than
        propagated, so a truncated file never blocks a rerun.
    """
    if not output_path.exists():
        return set()
    try:
        return set(pq.read_table(output_path, columns=["ref_smiles"])["ref_smiles"].to_pylist())
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read existing output %s (%s); restarting it", output_path, exc)
        return set()


def init_worker(config: str) -> None:
    """Load one finder per worker.

    Args:
        config: Path to the AiZynthFinder YAML config.
    """
    global _FINDER
    _FINDER = load_finder(config)

    # AiZynthFinder turns its own logger up to DEBUG when a finder is built and
    # emits a record per template application; left alone it writes gigabytes.
    for name in ("aizynthfinder", "route_distances", "rdkit", "rdApp"):
        third_party = logging.getLogger(name)
        third_party.setLevel(logging.WARNING)
        third_party.propagate = False


def measure_one(smiles: str) -> dict:
    """Search one molecule and record what the cut filter did.

    Args:
        smiles: The reference SMILES to search.

    Returns:
        dict: A row matching :data:`SCHEMA`. An unsolved molecule, or one whose
        search raised, comes back with zero counters -- it is still a row, so the
        denominator stays honest.
    """
    row: dict = {
        "ref_smiles": smiles,
        "solved": False,
        **dict.fromkeys(CUT_STATS_KEYS, 0),
        "ring_bonds": [],
    }
    try:
        _FINDER.target_smiles = smiles
        _FINDER.tree_search()
        _FINDER.build_routes()
        trees = list(_FINDER.routes.reaction_trees)
        if not trees:
            return row
        tree = trees[0]
        if not tree.is_solved:
            return row
        row["solved"] = True

        steps = route_steps(tree, DEFAULT_TAG_KEYS)
        if not steps:
            return row

        stats: dict[str, int] = {}
        dropped: list[dict] = []
        route_cut_bonds(smiles, steps, stats=stats, dropped_ring_bonds=dropped)
        for key in CUT_STATS_KEYS:
            row[key] = stats.get(key, 0)
        row["ring_bonds"] = dropped
    except Exception as exc:  # noqa: BLE001 - a failed molecule must not stop the sweep
        logger.debug("failed on %s: %s", smiles, exc)
    return row


def summarise(totals: collections.Counter, n: int, seed: int, wall_s: float) -> dict:
    """Turn the running counters into the report.

    Args:
        totals: Summed per-molecule counters.
        n: Molecules measured.
        seed: RNG seed, recorded for reproducibility.
        wall_s: Wall-clock seconds spent searching.

    Returns:
        dict: Counts, and the percentages the discussion needs. Percentages are
        omitted rather than faked when their denominator is zero.
    """
    solved = totals["solved"]
    proposed = totals["proposed"]
    report = {
        "molecules": n,
        "seed": seed,
        "solved": solved,
        "cuts_proposed": proposed,
        "cuts_kept": totals["kept"],
        "cuts_dropped_ring": totals["dropped_ring"],
        "cuts_dropped_not_bonded": totals["dropped_not_bonded"],
        "cuts_dropped_unmapped": totals["dropped_unmapped"],
        "molecules_with_a_ring_cut": totals["had_ring_cut"],
        "molecules_losing_everything_to_rings": totals["ring_only"],
        "molecules_losing_everything_any_cause": totals["lost_all"],
        "wall_min": round(wall_s / 60, 1),
    }
    if proposed:
        report["pct_cuts_lost_to_rings"] = round(100 * totals["dropped_ring"] / proposed, 2)
        report["pct_cuts_kept"] = round(100 * totals["kept"] / proposed, 2)
    if solved:
        report["pct_solved_mols_with_a_ring_cut"] = round(100 * totals["had_ring_cut"] / solved, 2)
        report["pct_solved_mols_losing_all_to_rings"] = round(
            100 * totals["ring_only"] / solved, 2
        )
    return report


def tally(row: dict, totals: collections.Counter) -> None:
    """Fold one measured molecule into the running totals.

    Args:
        row: A row from :func:`measure_one`.
        totals: Counter updated in place.
    """
    totals["solved"] += int(row["solved"])
    for key in CUT_STATS_KEYS:
        totals[key] += row[key]
    if row["dropped_ring"]:
        totals["had_ring_cut"] += 1
        if not row["kept"]:
            totals["ring_only"] += 1
    if row["solved"] and row["proposed"] and not row["kept"]:
        totals["lost_all"] += 1


def run(smiles_list: list[str], args: dict, output_path: Path | None) -> dict:
    """Measure a list of molecules, writing rows as they arrive.

    Args:
        smiles_list: Molecules still to measure.
        args: Parsed arguments.
        output_path: Per-molecule parquet, or None to keep counts only.

    Returns:
        dict: The report for the molecules measured in THIS call. In part mode
        the caller re-reads the finished file, so a resumed part is summarised
        over all of its molecules rather than only the new ones.
    """
    totals: collections.Counter = collections.Counter()
    buffer: list[dict] = []
    started = time.time()

    writer = None
    temp_path = None
    if output_path is not None:
        temp_path = output_path.with_suffix(".partial.parquet")
        writer = pq.ParquetWriter(
            temp_path, SCHEMA, compression=COMPRESSION, compression_level=COMPRESSION_LEVEL
        )

    def flush() -> None:
        """Write the buffered rows as one row group and clear the buffer."""
        if writer is None or not buffer:
            buffer.clear()
            return
        writer.write_table(pa.Table.from_pylist(buffer, schema=SCHEMA))
        buffer.clear()

    try:
        # Carry forward what an interrupted run already wrote, so the replaced
        # file is never smaller than the one it replaces.
        if writer is not None and output_path.exists():
            writer.write_table(pq.read_table(output_path, schema=SCHEMA))

        with Pool(
            args["num_workers"], initializer=init_worker, initargs=(args["config"],)
        ) as pool:
            for i, row in enumerate(pool.imap_unordered(measure_one, smiles_list, chunksize=1), 1):
                tally(row, totals)
                buffer.append(row)
                if i % args["flush_every"] == 0:
                    flush()
                    rate = i / (time.time() - started)
                    logger.info(
                        "%d/%d measured | %.2f mol/s | ETA %.1f min",
                        i,
                        len(smiles_list),
                        rate,
                        (len(smiles_list) - i) / rate / 60 if rate else 0,
                    )
        flush()
    finally:
        if writer is not None:
            writer.close()

    if temp_path is not None:
        temp_path.replace(output_path)

    return summarise(totals, len(smiles_list), args["seed"], time.time() - started)


def summarise_file(output_path: Path, seed: int, wall_s: float) -> dict:
    """Summarise a finished part from its own rows.

    Reading the file back rather than trusting the in-memory counters is what
    makes a resumed part correct: the counters only saw the molecules this run
    measured, while the file holds them all.

    Args:
        output_path: The finished per-molecule parquet.
        seed: RNG seed, recorded for reproducibility.
        wall_s: Wall-clock seconds spent searching in this run.

    Returns:
        dict: The report over every molecule in the file.
    """
    table = pq.read_table(output_path, columns=["solved", *CUT_STATS_KEYS])
    totals: collections.Counter = collections.Counter()
    for row in table.to_pylist():
        tally(row, totals)
    return summarise(totals, table.num_rows, seed, wall_s)


def main() -> None:
    """Measure one part, or a random sample, and report the ring-bond loss."""
    os.chdir(PROJ_ROOT)
    args = ring_loss_argparse()
    logging.basicConfig(
        filename=args["logfile"],
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    output_path = Path(args["output"]) if args["output"] else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if args["input"]:
        smiles_list = read_reference_smiles(args["input"])
        logger.info("%s: %d unique reference SMILES", args["input"], len(smiles_list))
        if output_path.exists():
            if not args["resume"]:
                logger.info("%s exists; nothing to do (pass --resume to continue it)", output_path)
                return
            done = already_done(output_path)
            smiles_list = [s for s in smiles_list if s not in done]
            logger.info("resuming: %d already measured, %d left", len(done), len(smiles_list))
    else:
        smiles_list = sample_smiles(Path(args["input_dir"]), args["sample"], args["seed"])
        logger.info("sampled %d molecules (seed %d)", len(smiles_list), args["seed"])

    started = time.time()
    if smiles_list:
        report = run(smiles_list, args, output_path)
    else:
        logger.info("nothing left to measure")
        report = {}

    if output_path is not None and output_path.exists():
        report = summarise_file(output_path, args["seed"], time.time() - started)

    out_path = Path(args["out"]) if args["out"] else None
    if out_path is None and output_path is not None:
        out_path = output_path.with_suffix(".summary.json")
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))

    logger.info("report: %s", json.dumps(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
