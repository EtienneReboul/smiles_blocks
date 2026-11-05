"""Persistence for precomputed retrosynthetic cut bonds.

An expensive retrosynthesis backend such as AiZynthFinder costs seconds to tens
of seconds per molecule, against microseconds for R-BRICS. Running them
inside the fragmentation workers would mean paying the search cost again on every
retry and coupling a CPU-bound search to an I/O-bound enumeration. So the search
runs as its own stage, writes its cut bonds here, and the fragmentation stage
reads them back.

The on-disk layout mirrors ``retrosynthesis.package_bond_info``: one row per
reference SMILES, with the bonds as a nested list of structs. Round-tripping
through :func:`bonds_to_records` and :func:`records_to_bonds` reproduces the
in-memory contract exactly, including the ``frozenset`` keys.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds

__all__ = [
    "CutBondsParquet",
    "bonds_to_records",
    "records_to_bonds",
    "load_cut_bonds",
]

#: Separator for several tags on one atom, matching the "_".join used by
#: ``smiles_fragmentation.get_annotation``.
TAG_SEPARATOR = "_"


@dataclass
class CutBondsParquet:
    """Schema and compression for the stage-A cut bond tables.

    One row per reference SMILES, including failures, so that a part file is a
    complete record of what the search was asked to do and what came back.
    """

    schema: pa.Schema = field(
        default_factory=lambda: pa.schema(
            [
                ("ref_smiles", pa.string()),
                # False when the search found no solved route, timed out, or the
                # route could not be expressed as string cuts.
                ("retro_success", pa.bool_()),
                ("n_routes", pa.int16()),
                ("n_steps", pa.int16()),
                ("search_time_s", pa.float32()),
                ("backend", pa.string()),
                ("error", pa.string()),
                (
                    "bonds",
                    pa.list_(
                        pa.struct(
                            [
                                # Canonical atom ranks, as produced by
                                # Chem.CanonicalRankAtoms on the map-free target.
                                ("begin_idx", pa.int32()),
                                ("end_idx", pa.int32()),
                                ("begin_tag", pa.string()),
                                ("end_tag", pa.string()),
                            ]
                        )
                    ),
                ),
            ]
        )
    )
    compression: str = "zstd"
    compression_level: int = 14


def bonds_to_records(bonds: dict[frozenset[int], dict[int, set[str]]]) -> list[dict]:
    """Flatten an in-memory bond dict into parquet-writable structs.

    Args:
        bonds: Bond signature (pair of canonical ranks) -> per-rank tag sets, as
            returned by ``route_cut_bonds`` or ``retrosynthetic_analysis``.

    Returns:
        list[dict]: One dict per bond with ``begin_idx``, ``end_idx``,
        ``begin_tag`` and ``end_tag``. ``begin_idx`` is always the lower rank, so
        the serialisation is deterministic and diffable.
    """
    records = []
    for signature, annotation in bonds.items():
        begin_idx, end_idx = sorted(signature)
        records.append(
            {
                "begin_idx": int(begin_idx),
                "end_idx": int(end_idx),
                "begin_tag": TAG_SEPARATOR.join(sorted(annotation.get(begin_idx, ()))),
                "end_tag": TAG_SEPARATOR.join(sorted(annotation.get(end_idx, ()))),
            }
        )
    return sorted(records, key=lambda record: (record["begin_idx"], record["end_idx"]))


def records_to_bonds(records: list[dict] | None) -> dict[frozenset[int], dict[int, set[str]]]:
    """Rebuild the in-memory bond dict from parquet structs.

    Inverse of :func:`bonds_to_records`.

    Args:
        records: Bond structs read from the ``bonds`` column; ``None`` is treated
            as no bonds.

    Returns:
        dict[frozenset[int], dict[int, set[str]]]: The contract that
        ``process_set_smiles`` expects.
    """
    bonds: dict[frozenset[int], dict[int, set[str]]] = {}
    for record in records or ():
        begin_idx = int(record["begin_idx"])
        end_idx = int(record["end_idx"])
        annotation = defaultdict(set)
        for idx, tag in ((begin_idx, record["begin_tag"]), (end_idx, record["end_tag"])):
            if tag:
                annotation[idx].update(tag.split(TAG_SEPARATOR))
        bonds[frozenset((begin_idx, end_idx))] = dict(annotation)
    return bonds


def load_cut_bonds(
    path: str | Path,
    ref_smiles: set[str] | None = None,
) -> dict[str, dict[frozenset[int], dict[int, set[str]]]]:
    """Load precomputed cut bonds, keyed by reference SMILES.

    Reads only the columns needed, and optionally only the molecules a given
    fragmentation task will actually process, so a worker never holds the whole
    1.9 M-molecule table.

    Args:
        path: Parquet file or directory written by the stage-A job.
        ref_smiles: Restrict the result to these reference SMILES. ``None`` loads
            everything.

    Returns:
        dict[str, dict]: Reference SMILES -> bond dict. Molecules whose search
        failed are present with an empty bond dict, which lets the caller tell
        "search failed" apart from "molecule absent from the table".
    """
    dataset = ds.dataset(str(path), format="parquet")

    table = dataset.to_table(columns=["ref_smiles", "retro_success", "bonds"])

    bonds_by_smiles: dict[str, dict] = {}
    for row in table.to_pylist():
        smiles = row["ref_smiles"]
        if ref_smiles is not None and smiles not in ref_smiles:
            continue
        bonds_by_smiles[smiles] = records_to_bonds(row["bonds"]) if row["retro_success"] else {}

    return bonds_by_smiles
