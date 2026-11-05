"""Turn atom-mapped retrosynthesis steps into cleavable bonds on the target molecule.

This module is the backend-agnostic core of the retrosynthesis-driven fragmentation.
It consumes atom-mapped reaction SMILES and produces exactly the structure that
:func:`smiles_blocks.retrosynthesis.retrosynthetic_analysis` produces, so that
:func:`smiles_blocks.smiles_fragmentation.process_set_smiles` works unchanged::

    dict[frozenset[int], dict[int, set[str]]]
    #    ^ bond as a pair of canonical atom ranks
    #                     ^ canonical atom rank -> set of tags

Two invariants matter and are easy to get wrong:

1. Bond signatures live in *canonical atom rank* space (``Chem.CanonicalRankAtoms``),
   not atom-index space. Ranks are a canonical permutation, so they are stable
   across the different atom orderings of the randomised SMILES that the
   fragmentation side enumerates. That stability is what makes the set
   intersection in ``process_set_smiles`` meaningful.
2. **Atom map numbers corrupt canonical ranks.** With maps present,
   ``CanonicalRankAtoms`` effectively returns the map numbering itself. Maps must
   be stripped before ranking, otherwise every bond signature is wrong and the
   downstream intersection silently yields nothing. See :func:`strip_atom_maps`.

Only bonds that survive all of the following are kept, mirroring the R-BRICS rule
in ``retrosynthesis.annotate_rbond``:

- both endpoints trace back to a mapped atom of the target;
- the two atoms are actually bonded *in the target*;
- that bond is not in a ring (ring bonds cannot be cleaved in a SMILES string).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re

from rdkit import Chem

__all__ = [
    "RouteStep",
    "strip_atom_maps",
    "mapnum_bonds",
    "split_mapped_reaction",
    "step_cut_bonds",
    "build_mapnum_to_rank",
    "route_cut_bonds",
    "sanitize_tag",
]

#: Tags end up concatenated with "_" into the ``begin_tag``/``end_tag`` parquet
#: columns, and block identifiers are joined with ".", so neither character may
#: appear inside a tag.
_TAG_FORBIDDEN = re.compile(r"[^0-9A-Za-z-]+")

#: Fallback tag when a backend exposes no usable reaction label.
DEFAULT_TAG = "retro"


@dataclass(frozen=True)
class RouteStep:
    """One retrosynthetic disconnection, as an atom-mapped reaction SMILES.

    Attributes:
        mapped_reaction_smiles: Atom-mapped reaction SMILES in the **retro**
            direction, ``target>>precursors``, which is what AiZynthFinder's
            ``FixedRetroReaction.metadata["mapped_reaction_smiles"]`` contains.
            Note the product sits on the *left*.
        tag: Short label for the reaction, used to annotate both atoms of every
            bond this step cuts. Sanitised on use.
        depth: Depth of the step in the route; 0 is the step applied to the
            target itself. Only used for reporting.
    """

    mapped_reaction_smiles: str
    tag: str = DEFAULT_TAG
    depth: int = 0


def sanitize_tag(tag: str, fallback: str = DEFAULT_TAG, max_length: int = 24) -> str:
    """Make a tag safe for the ``begin_tag``/``end_tag`` columns.

    Strips every character that would collide with the ``_`` tag separator or the
    ``.`` block separator used downstream.

    Args:
        tag: Raw label from the retrosynthesis backend.
        fallback: Value returned when ``tag`` holds no usable character.
        max_length: Truncation length, to keep the parquet columns compact.

    Returns:
        str: A non-empty tag of at most ``max_length`` characters.
    """
    cleaned = _TAG_FORBIDDEN.sub("-", str(tag)).strip("-")
    return cleaned[:max_length] if cleaned else fallback


def strip_atom_maps(mol: Chem.Mol) -> Chem.Mol:
    """Return a copy of ``mol`` with every atom map number cleared.

    Required before any call to ``Chem.CanonicalRankAtoms``: atom maps are taken
    into account by the canonical ranking, so ranks computed on a mapped molecule
    do not match ranks computed on the map-free molecules used elsewhere in the
    pipeline.

    Args:
        mol: Any RDKit molecule, mapped or not. Not modified.

    Returns:
        Chem.Mol: A copy carrying no atom map numbers.
    """
    clean = Chem.Mol(mol)
    for atom in clean.GetAtoms():
        atom.SetAtomMapNum(0)
    return clean


def mapnum_bonds(mol: Chem.Mol) -> set[frozenset[int]]:
    """Collect the bonds of ``mol`` as pairs of atom map numbers.

    Bonds with an unmapped endpoint (map number 0) are skipped, since they cannot
    be traced to the target.

    Args:
        mol: An atom-mapped RDKit molecule. May hold several disconnected
            fragments, which is how the precursor side of a reaction is parsed.

    Returns:
        set[frozenset[int]]: One frozenset of two map numbers per traceable bond.
    """
    bonds = set()
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom().GetAtomMapNum()
        end = bond.GetEndAtom().GetAtomMapNum()
        if begin and end:
            bonds.add(frozenset((begin, end)))
    return bonds


def split_mapped_reaction(mapped_reaction_smiles: str) -> tuple[str, str]:
    """Split a retro-direction mapped reaction SMILES into product and precursors.

    AiZynthFinder writes these as ``target>>precursors``: the product is on the
    **left**. An optional agent field (``product>agents>precursors``) is ignored.

    The precursor side is returned as a single dot-separated SMILES rather than a
    list, because only the *union* of precursor bonds is ever needed; RDKit parses
    it directly as one multi-fragment molecule.

    Args:
        mapped_reaction_smiles: Reaction SMILES in retro direction.

    Returns:
        tuple[str, str]: ``(product_smiles, precursors_smiles)``.

    Raises:
        ValueError: If the string does not hold 2 or 3 ``>``-separated fields, or
            if either required side is empty.
    """
    fields = mapped_reaction_smiles.split(">")
    if len(fields) == 2:
        product, precursors = fields
    elif len(fields) == 3:
        product, _agents, precursors = fields
    else:
        raise ValueError(
            f"cannot parse reaction SMILES with {len(fields)} '>'-separated "
            f"fields: {mapped_reaction_smiles!r}"
        )

    if not product or not precursors:
        raise ValueError(f"reaction SMILES has an empty side: {mapped_reaction_smiles!r}")

    return product, precursors


def step_cut_bonds(mapped_reaction_smiles: str) -> set[frozenset[int]]:
    """Find the bonds a single retrosynthetic step breaks, in map-number space.

    A bond is considered broken when it is present in the product but absent from
    the precursor side. This covers both of the ways a disconnection shows up:
    the two atoms landing in different precursor molecules, and the bond simply
    being deleted.

    Ring membership is *not* filtered here; that is done in
    :func:`route_cut_bonds`, where ring information is read from the target.

    Args:
        mapped_reaction_smiles: Atom-mapped reaction SMILES, ``target>>precursors``.

    Returns:
        set[frozenset[int]]: Broken bonds as pairs of atom map numbers. Empty if
        either side fails to parse.
    """
    product_smiles, precursor_smiles = split_mapped_reaction(mapped_reaction_smiles)

    product = Chem.MolFromSmiles(product_smiles, sanitize=True)
    precursors = Chem.MolFromSmiles(precursor_smiles, sanitize=True)
    if product is None or precursors is None:
        return set()

    return mapnum_bonds(product) - mapnum_bonds(precursors)


def build_mapnum_to_rank(target_mol: Chem.Mol, mapped_product: Chem.Mol) -> dict[int, int] | None:
    """Map the atom map numbers of a route onto canonical ranks of the target.

    ``mapped_product`` must be the atom-mapped form of the same molecule as
    ``target_mol`` -- i.e. the product of the route's root step. Both molecules
    are ranked canonically with maps stripped; because the canonical ranking is a
    permutation that depends only on the molecule, rank ``r`` in one is the same
    atom as rank ``r`` in the other, whatever the atom ordering of either SMILES.
    That makes the correspondence exact without any substructure matching, which
    would be ambiguous on symmetric molecules.

    Args:
        target_mol: The reference molecule, parsed from the pipeline's canonical
            SMILES.
        mapped_product: The root step's product, carrying atom map numbers.

    Returns:
        dict[int, int] | None: Atom map number -> canonical rank in the target.
        ``None`` if the two molecules are not the same species, in which case the
        route cannot be trusted and the caller should discard it.
    """
    clean_product = strip_atom_maps(mapped_product)

    # Same-species check. Canonical SMILES of the map-free forms must agree,
    # otherwise rank-to-rank identification is meaningless.
    if Chem.MolToSmiles(clean_product) != Chem.MolToSmiles(target_mol):
        return None

    product_ranks = list(Chem.CanonicalRankAtoms(clean_product))

    mapnum_to_rank: dict[int, int] = {}
    for atom in mapped_product.GetAtoms():
        mapnum = atom.GetAtomMapNum()
        if mapnum:
            mapnum_to_rank[mapnum] = product_ranks[atom.GetIdx()]

    return mapnum_to_rank


def route_cut_bonds(
    target_smiles: str,
    steps: list[RouteStep],
) -> dict[frozenset[int], dict[int, set[str]]]:
    """Convert a whole retrosynthetic route into cleavable bonds on the target.

    The route's root step -- the one whose product is the target itself -- fixes
    the correspondence between atom map numbers and the target's canonical ranks.
    Deeper steps reuse it, which is sound because the backends propagate map
    numbers from the target down the tree.

    Bonds are discarded unless both endpoints trace to a mapped target atom, the
    atoms are bonded in the target, and that bond is acyclic. Ring-forming and
    ring-opening steps therefore contribute nothing, matching the pipeline's
    existing exclusion of ring decomposition.

    Args:
        target_smiles: The reference SMILES being fragmented.
        steps: The route's steps. Order is irrelevant; the root is located by
            comparing each step's product with the target.

    Returns:
        dict[frozenset[int], dict[int, set[str]]]: Bond signature (pair of
        canonical ranks) -> per-rank set of reaction tags. Empty when the target
        cannot be parsed, no root step is found, or no bond survives filtering.
        The shape is identical to ``retrosynthetic_analysis``'s, so it is a drop-in
        replacement.
    """
    target_mol = Chem.MolFromSmiles(target_smiles)
    if target_mol is None or not steps:
        return {}

    # Locate the root step and build the map-number -> canonical-rank table.
    mapnum_to_rank: dict[int, int] | None = None
    for step in steps:
        try:
            product_smiles, _ = split_mapped_reaction(step.mapped_reaction_smiles)
        except ValueError:
            continue
        mapped_product = Chem.MolFromSmiles(product_smiles)
        if mapped_product is None:
            continue
        candidate = build_mapnum_to_rank(target_mol, mapped_product)
        if candidate is not None:
            mapnum_to_rank = candidate
            break

    if not mapnum_to_rank:
        return {}

    # Canonical rank -> atom index, to interrogate the target's own bonds.
    rank_to_idx = {rank: idx for idx, rank in enumerate(Chem.CanonicalRankAtoms(target_mol))}

    bonds: dict[frozenset[int], dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))

    for step in steps:
        try:
            cut_mapnums = step_cut_bonds(step.mapped_reaction_smiles)
        except ValueError:
            continue
        tag = sanitize_tag(step.tag)

        for mapnum_pair in cut_mapnums:
            # A bond cut on an intermediate may involve an atom pair that is not
            # bonded in the target at all; such a cut is not expressible as a
            # string cut and is dropped.
            if len(mapnum_pair) != 2:
                continue
            ranks = [mapnum_to_rank.get(mapnum) for mapnum in mapnum_pair]
            if any(rank is None for rank in ranks):
                continue

            bond = target_mol.GetBondBetweenAtoms(rank_to_idx[ranks[0]], rank_to_idx[ranks[1]])
            if bond is None or bond.IsInRing():
                continue

            signature = frozenset(ranks)
            bonds[signature][ranks[0]].add(tag)
            bonds[signature][ranks[1]].add(tag)

    # Return plain dicts so the result pickles cleanly across process boundaries
    # (defaultdict with a lambda factory does not).
    return {signature: dict(annotation) for signature, annotation in bonds.items()}
