"""Pluggable retrosynthesis backends for SMILES block fragmentation.

Every backend produces the same structure as
:func:`smiles_blocks.retrosynthesis.retrosynthetic_analysis`::

    dict[frozenset[int], dict[int, set[str]]]

so :func:`smiles_blocks.smiles_fragmentation.process_set_smiles` consumes any of
them unchanged.

Backends split into two cost classes, which is why the pipeline is staged:

- ``rbrics``: SMARTS pattern matching, microseconds per molecule, so it runs
  inline inside the fragmentation workers.
- ``aizynth``: a tree search, seconds to tens of seconds per molecule. It runs as
  a separate precompute stage whose output is read back by the ``precomputed``
  backend.

The expensive backend lives behind a lazy import, so importing this package never
requires AiZynthFinder to be installed.
"""

from smiles_blocks.retro_backends.bond_extraction import (
    RouteStep,
    route_cut_bonds,
    sanitize_tag,
)
from smiles_blocks.retro_backends.store import (
    CutBondsParquet,
    bonds_to_records,
    load_cut_bonds,
    records_to_bonds,
)

__all__ = [
    "RouteStep",
    "route_cut_bonds",
    "sanitize_tag",
    "CutBondsParquet",
    "bonds_to_records",
    "records_to_bonds",
    "load_cut_bonds",
]
