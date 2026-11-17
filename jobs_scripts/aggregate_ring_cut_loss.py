"""Aggregate the per-part ring-cut-loss measurements into one report.

Each array task of :mod:`measure_ring_cut_loss` writes one per-molecule parquet.
This sums them into the figures the discussion needs, and -- because the parts
recorded WHICH ring bonds were discarded, not just how many -- also breaks the
loss down by ring size and aromaticity.

    python jobs_scripts/aggregate_ring_cut_loss.py \\
        --input-dir data/interim/ring_cut_loss \\
        --out reports/ring_cut_loss_full.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys

import pyarrow.dataset as ds

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

from smiles_blocks.retro_backends.bond_extraction import CUT_STATS_KEYS  # noqa: E402


def aggregate_argparse() -> dict:
    """Parse the command line.

    Returns:
        dict: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory of per-part parquets.")
    parser.add_argument("--out", default="reports/ring_cut_loss_full.json")
    return vars(parser.parse_args())


def wilson(hits: int, total: int, z: float = 1.96) -> list[float] | None:
    """Wilson score interval for a proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and
    behaves at the small counts some ring-size bins will have.

    Args:
        hits: Successes.
        total: Trials.
        z: Standard-normal quantile; 1.96 is 95 %.

    Returns:
        list[float] | None: Lower and upper bound in percent, or None when there
        are no trials.
    """
    if total <= 0:
        return None
    p = hits / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = z * ((p * (1 - p) / total + z**2 / (4 * total**2)) ** 0.5) / denom
    return [round(100 * max(0.0, centre - half), 3), round(100 * min(1.0, centre + half), 3)]


def main() -> None:
    """Sum every part and write the aggregate report."""
    args = aggregate_argparse()
    input_dir = Path(args["input_dir"])
    parts = sorted(input_dir.glob("part-*.parquet"), key=lambda p: int(p.stem.split("-")[1]))
    parts = [p for p in parts if ".partial" not in p.name and ".summary" not in p.name]
    if not parts:
        raise SystemExit(f"no part-*.parquet in {input_dir}")

    totals: collections.Counter = collections.Counter()
    ring_size: collections.Counter = collections.Counter()
    aromatic_rings = 0
    molecules = 0

    for part in parts:
        dataset = ds.dataset(part, format="parquet")
        for batch in dataset.to_batches(columns=["solved", "ring_bonds", *CUT_STATS_KEYS]):
            for row in batch.to_pylist():
                molecules += 1
                totals["solved"] += int(row["solved"])
                for key in CUT_STATS_KEYS:
                    totals[key] += row[key]
                if row["dropped_ring"]:
                    totals["had_ring_cut"] += 1
                    if not row["kept"]:
                        totals["ring_only"] += 1
                if row["solved"] and row["proposed"] and not row["kept"]:
                    totals["lost_all"] += 1
                for bond in row["ring_bonds"] or []:
                    ring_size[bond["ring_size"]] += 1
                    aromatic_rings += int(bond["aromatic"])

    solved = totals["solved"]
    proposed = totals["proposed"]
    dropped_ring = totals["dropped_ring"]

    report = {
        "parts": len(parts),
        "molecules": molecules,
        "solved": solved,
        "cuts_proposed": proposed,
        "cuts_kept": totals["kept"],
        "cuts_dropped_ring": dropped_ring,
        "cuts_dropped_not_bonded": totals["dropped_not_bonded"],
        "cuts_dropped_unmapped": totals["dropped_unmapped"],
        "molecules_with_a_ring_cut": totals["had_ring_cut"],
        "molecules_losing_everything_to_rings": totals["ring_only"],
        "molecules_losing_everything_any_cause": totals["lost_all"],
    }
    if proposed:
        report["pct_cuts_lost_to_rings"] = round(100 * dropped_ring / proposed, 3)
        report["pct_cuts_kept"] = round(100 * totals["kept"] / proposed, 3)
    if solved:
        report["pct_solved"] = round(100 * solved / molecules, 3) if molecules else None
        report["pct_solved_mols_with_a_ring_cut"] = round(100 * totals["had_ring_cut"] / solved, 3)
        report["pct_solved_mols_losing_all_to_rings"] = round(
            100 * totals["ring_only"] / solved, 3
        )
        # The whole population is measured, so these intervals describe sampling
        # noise in the search itself, not in a sample of molecules.
        report["ci95_pct_solved_mols_with_a_ring_cut"] = wilson(totals["had_ring_cut"], solved)
        report["ci95_pct_solved_mols_losing_all_to_rings"] = wilson(totals["ring_only"], solved)
    if dropped_ring:
        report["dropped_ring_by_size"] = {
            str(size): count for size, count in sorted(ring_size.items())
        }
        report["dropped_ring_aromatic"] = aromatic_rings
        report["pct_dropped_ring_aromatic"] = round(100 * aromatic_rings / dropped_ring, 3)

    out_path = Path(args["out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
