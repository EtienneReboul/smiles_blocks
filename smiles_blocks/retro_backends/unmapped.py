"""Cut-bond extraction for retrosynthesis steps that carry no atom mapping.

:mod:`smiles_blocks.retro_backends.bond_extraction` needs atom-mapped reaction
SMILES. That mapping is not always there: AiZynthFinder only attaches it to a
route's steps when the reaction was actually mapped, and a route that comes back
without it would otherwise be discarded as ``no_mapped_steps``. This module is
the fallback for those routes -- it recovers the disconnection from the molecules
alone, given a product and its precursors as plain SMILES.

Method
------
For each acyclic bond of the product, break it and ask where the two resulting
fragments went. A bond is a disconnection when its two fragments are each
contained in a *different* precursor::

    product:     CC(=O)Nc1ccc(O)cc1
    precursors:  CC(=O)Cl   +   Nc1ccc(O)cc1

    break the amide C-N  ->  CC=O | Nc1ccc(O)cc1
                             in precursor 0 | in precursor 1   => a cut

This works through leaving groups, which is what makes it usable on real
retrosynthesis output: the acid chloride above carries a Cl that the product
never had, yet the ``CC=O`` fragment still matches inside it, because RDKit
substructure matching does not compare implicit hydrogen counts.

Limits, and why they are acceptable
-----------------------------------
- **Only steps acting on the target itself are used.** Without atom maps there is
  no reliable way to locate an intermediate's bond back on the target, so deeper
  steps are skipped and reported rather than guessed at. With the top-route-only
  policy this costs little: the first disconnection is the one that matters, and
  deeper cuts frequently involve atom pairs that are not bonded in the target
  anyway.
- **Fragment containment is evidence, not proof.** A fragment could match a
  precursor coincidentally. Three guards keep that in check: both fragments must
  be assignable to *distinct* precursors, no single precursor may contain the
  whole product (nothing was broken then), and fragments below
  ``min_fragment_atoms`` are ignored, since a one-atom piece such as a methyl
  matches almost anything and would be a useless block regardless.

Where the atom mapping *is* available, prefer
:func:`~smiles_blocks.retro_backends.bond_extraction.route_cut_bonds`: it is
exact rather than inferential.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import permutations

from rdkit import Chem

from smiles_blocks.retro_backends.bond_extraction import DEFAULT_TAG, sanitize_tag

__all__ = [
    "UnmappedStep",
    "DEFAULT_MIN_FRAGMENT_ATOMS",
    "step_cut_bonds_unmapped",
    "unmapped_route_cut_bonds",
]

#: Smallest fragment accepted as evidence of a disconnection. One-atom fragments
#: match almost any precursor and would yield a single-atom block.
DEFAULT_MIN_FRAGMENT_ATOMS = 2


@dataclass(frozen=True)
class UnmappedStep:
    """One retrosynthetic step given as plain, unmapped molecules.

    Attributes:
        product_smiles: The molecule being disconnected.
        precursor_smiles: The molecules it is disconnected into.
        tag: Short label for the reaction, applied to both atoms of every bond
            this step cuts. Sanitised on use.
        depth: Depth in the route; 0 is the step applied to the target. Only
            steps at the target are usable -- see the module docstring.
    """

    product_smiles: str
    precursor_smiles: tuple[str, ...] = field(default=())
    tag: str = DEFAULT_TAG
    depth: int = 0


def _fragment_owners(fragment: Chem.Mol, precursors: list[Chem.Mol]) -> set[int]:
    """Find which precursors contain a fragment.

    Args:
        fragment: One side of a broken bond, used as a substructure query.
        precursors: Candidate precursor molecules.

    Returns:
        set[int]: Indices of the precursors containing ``fragment``.
    """
    return {
        index
        for index, precursor in enumerate(precursors)
        if precursor.HasSubstructMatch(fragment)
    }


def step_cut_bonds_unmapped(
    product: Chem.Mol,
    precursors: list[Chem.Mol],
    *,
    min_fragment_atoms: int = DEFAULT_MIN_FRAGMENT_ATOMS,
) -> set[frozenset[int]]:
    """Infer which bonds of ``product`` a step broke, from the molecules alone.

    Args:
        product: The molecule being disconnected.
        precursors: What it was disconnected into. At least two are needed: a
            single precursor cannot evidence a disconnection.
        min_fragment_atoms: Ignore bonds whose breaking yields a fragment smaller
            than this.

    Returns:
        set[frozenset[int]]: Broken bonds as pairs of **atom indices into**
        ``product``. Empty when no bond qualifies.
    """
    if product is None or len(precursors) < 2:
        return set()

    # If a precursor still contains the whole product, this step did not take it
    # apart (a functional-group interconversion, say), so there is nothing to find.
    if any(precursor.HasSubstructMatch(product) for precursor in precursors):
        return set()

    cuts: set[frozenset[int]] = set()

    for bond in product.GetBonds():
        # Ring bonds cannot be cleaved in a SMILES string, and breaking one does
        # not separate the molecule anyway.
        if bond.IsInRing():
            continue

        try:
            fragmented = Chem.FragmentOnBonds(product, [bond.GetIdx()], addDummies=False)
            pieces = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=True)
        except Exception:  # noqa: BLE001 - an unsplittable bond is simply not a cut
            continue

        if len(pieces) != 2:
            continue
        if any(piece.GetNumAtoms() < min_fragment_atoms for piece in pieces):
            continue

        owners = [_fragment_owners(piece, precursors) for piece in pieces]

        # The two fragments must be placeable in two *different* precursors.
        # Checking for distinct indices rules out the case where both fragments
        # merely happen to occur in the same precursor.
        if not any(
            first != second
            for first, second in permutations(range(len(precursors)), 2)
            if first in owners[0] and second in owners[1]
        ):
            continue

        cuts.add(frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))

    return cuts


def unmapped_route_cut_bonds(
    target_smiles: str,
    steps: list[UnmappedStep],
    *,
    min_fragment_atoms: int = DEFAULT_MIN_FRAGMENT_ATOMS,
) -> dict[frozenset[int], dict[int, set[str]]]:
    """Convert an unmapped route into cleavable bonds on the target.

    Produces the same structure as
    :func:`~smiles_blocks.retro_backends.bond_extraction.route_cut_bonds` and
    :func:`~smiles_blocks.retrosynthesis.retrosynthetic_analysis`, so it is a
    drop-in for the fragmentation stage.

    Only steps whose product *is* the target contribute; see the module
    docstring.

    Args:
        target_smiles: The reference SMILES being fragmented.
        steps: The route's steps, unmapped.
        min_fragment_atoms: Smallest fragment accepted as evidence of a cut.

    Returns:
        dict[frozenset[int], dict[int, set[str]]]: Bond signature (pair of
        canonical atom ranks) -> per-rank set of reaction tags. Empty if the
        target cannot be parsed or no step acts on it.
    """
    target = Chem.MolFromSmiles(target_smiles)
    if target is None or not steps:
        return {}

    target_canonical = Chem.MolToSmiles(target)
    bonds: dict[frozenset[int], dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))

    for step in steps:
        product = Chem.MolFromSmiles(step.product_smiles)
        if product is None:
            continue

        # Only the step acting on the target can be placed on the target.
        if Chem.MolToSmiles(product) != target_canonical:
            continue

        precursors = [Chem.MolFromSmiles(smiles) for smiles in step.precursor_smiles]
        precursors = [mol for mol in precursors if mol is not None]

        cuts = step_cut_bonds_unmapped(product, precursors, min_fragment_atoms=min_fragment_atoms)
        if not cuts:
            continue

        # The product is the same species as the target, so its canonical ranks
        # are the target's: the ranking is a permutation determined by the
        # molecule, not by the atom ordering of either SMILES.
        ranks = list(Chem.CanonicalRankAtoms(product))
        tag = sanitize_tag(step.tag)

        for atom_pair in cuts:
            begin_idx, end_idx = tuple(atom_pair)
            signature = frozenset((ranks[begin_idx], ranks[end_idx]))
            bonds[signature][ranks[begin_idx]].add(tag)
            bonds[signature][ranks[end_idx]].add(tag)

    return {signature: dict(annotation) for signature, annotation in bonds.items()}
