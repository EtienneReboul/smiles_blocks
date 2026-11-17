"""Re-annotate stage-A cut bonds with meaningful per-atom tags.

Stage A stored every bond tagged ``0-0-Unrecognized``, because the public USPTO
template library leaves AiZynthFinder's ``classification`` field unset on all
42 554 templates. See :mod:`smiles_blocks.retro_backends.annotate` for why no
re-run fixes that and why the annotation is derived from structure instead.

This rewrites the parquet parts in place -- bond signatures untouched, tags
replaced -- so stage B needs no change at all.

    python jobs_scripts/annotate_cut_bonds.py \\
        --input-dir data/interim/retro_cut_bonds \\
        --output-dir data/interim/retro_cut_bonds_annotated \\
        --num-workers 8
"""

from __future__ import annotations

import argparse
from functools import partial
import logging
from multiprocessing import Pool
import os
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

from smiles_blocks.retro_backends.annotate import (  # noqa: E402
    annotate_bonds,
    annotation_stats,
)
from smiles_blocks.retro_backends.store import (  # noqa: E402
    CutBondsParquet,
    bonds_to_records,
    records_to_bonds,
)

logger = logging.getLogger(__name__)

_CHEMICAL_DICT: dict | None = None


def annotate_argparse() -> dict:
    """Parse the command line.

    Returns:
        dict: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory of stage-A parts.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to write the annotated parts. May equal --input-dir to rewrite.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-parts", type=int, default=0, help="Only the first N parts.")
    parser.add_argument("--logfile", default=None)
    return vars(parser.parse_args())


def init_worker() -> None:
    """Build the R-BRICS patterns once per worker, not once per molecule."""
    global _CHEMICAL_DICT
    from smiles_blocks.retrosynthesis import prep_rbrics_data

    _CHEMICAL_DICT = prep_rbrics_data()


def annotate_part(path: Path, output_dir: Path) -> tuple[str, dict[str, int]]:
    """Re-annotate one parquet part.

    Args:
        path: The stage-A part to read.
        output_dir: Where the annotated part is written, under the same name.

    Returns:
        tuple: the part's name, and counters (``rows``, ``bonds``, ``rbrics``,
        ``fallback``).
    """
    table = pq.read_table(path)
    rows = table.to_pylist()
    counts = {"rows": len(rows), "bonds": 0, "rbrics": 0, "fallback": 0}

    for row in rows:
        records = row.get("bonds") or []
        if not records:
            continue
        bonds = records_to_bonds(records)
        annotated, from_rbrics, from_fallback = annotate_bonds(
            row["ref_smiles"], bonds, chemical_dict=_CHEMICAL_DICT
        )
        row["bonds"] = bonds_to_records(annotated)
        counts["bonds"] += len(annotated)
        counts["rbrics"] += from_rbrics
        counts["fallback"] += from_fallback

    # Write to a sibling then replace, so an interrupted run cannot leave a
    # half-written part in place of a good one.
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / path.name
    partial_path = final.with_suffix(".partial.parquet")
    schema = CutBondsParquet().schema
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        partial_path,
        compression=CutBondsParquet.compression,
        compression_level=CutBondsParquet.compression_level,
    )
    partial_path.replace(final)

    return path.name, counts


def main() -> None:
    """Re-annotate every part in the input directory."""
    os.chdir(PROJ_ROOT)
    args = annotate_argparse()

    logging.basicConfig(
        filename=args["logfile"],
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    input_dir = Path(args["input_dir"])
    output_dir = Path(args["output_dir"])
    parts = sorted(input_dir.glob("part-*.parquet"), key=lambda p: int(p.stem.split("-")[1]))
    if args["limit_parts"]:
        parts = parts[: args["limit_parts"]]
    if not parts:
        raise SystemExit(f"no part-*.parquet found in {input_dir}")

    logger.info("annotating %d parts from %s -> %s", len(parts), input_dir, output_dir)

    totals = {"rows": 0, "bonds": 0, "rbrics": 0, "fallback": 0}
    worker = partial(annotate_part, output_dir=output_dir)

    with Pool(args["num_workers"], initializer=init_worker) as pool:
        for name, counts in pool.imap_unordered(worker, parts):
            for key, value in counts.items():
                totals[key] += value
            logger.info("%s: %s", name, annotation_stats(counts))

    logger.info("TOTAL %d rows, %s", totals["rows"], annotation_stats(totals))


if __name__ == "__main__":
    main()
