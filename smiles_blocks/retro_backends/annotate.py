"""Give retrosynthesis cut bonds a meaningful per-atom annotation.

Why this exists
---------------
A cut bond needs a tag on each of its two atoms: the fragmentation stage joins
those tags into a block's ``begin_tag``/``end_tag``, so a block is only as
informative as they are.

AiZynthFinder is asked for a reaction tag from its route metadata, preferring
``classification`` -- the reaction class, which is what a chemist would want. On
the **public** USPTO template library that field is worthless: all 42 554
templates carry the single value ``"0.0 Unrecognized"``, so every bond came back
tagged ``0-0-Unrecognized``. Populating it needs a commercially licensed
reaction classifier, and no choice of ``--tag-keys`` recovers it -- the
alternatives (``template_hash``, ``template_code``) identify *which* template
fired but say nothing a reader can interpret.

The annotation is therefore derived from structure instead of from the route,
which has a large practical advantage: it needs only the target SMILES and the
bond, both already stored, so it runs offline in minutes rather than repeating a
~12 500 core-hour search.

How a bond is annotated
-----------------------
1. **R-BRICS patterns first.** The same patterns the R-BRICS backend uses are
   matched against the target, and a cut bond that coincides with an R-BRICS bond
   inherits its two ``L``-tags. This covers ~96.5 % of cut bonds and puts them in
   the vocabulary the rest of the pipeline already speaks, so retro-derived
   blocks stay directly comparable with R-BRICS ones.
2. **A structural fallback otherwise.** The remaining ~3.5 % are real
   disconnections that R-BRICS does not encode -- aryl halides, a C-C alpha to a
   ketone -- which is precisely why a planner was used. Rather than drop them or
   leave them unlabelled, each atom gets :func:`fallback_atom_tag`, an
   ``X``-prefixed element symbol. The prefix keeps them from being mistaken for
   an R-BRICS class, and lowercase marks an aromatic atom, as in SMILES.
"""

from __future__ import annotations

from collections import defaultdict

from rdkit import Chem

from smiles_blocks.retro_backends.bond_extraction import sanitize_tag
from smiles_blocks.retrosynthesis import prep_rbrics_data, retrosynthetic_analysis

__all__ = [
    "FALLBACK_PREFIX",
    "fallback_atom_tag",
    "annotate_bonds",
    "annotation_stats",
]

#: Marks a tag that is NOT an R-BRICS chemical class, so the two vocabularies
#: stay tellable apart downstream.
FALLBACK_PREFIX = "X"


def fallback_atom_tag(atom: Chem.Atom) -> str:
    """Describe an atom for a bond that matches no R-BRICS pattern.

    Args:
        atom: The atom at one end of the cut bond.

    Returns:
        str: ``FALLBACK_PREFIX`` plus the element symbol, lowercased when the atom
        is aromatic (as in SMILES). ``XBr``, ``Xc``, ``XC``.
    """
    symbol = atom.GetSymbol()
    return FALLBACK_PREFIX + (symbol.lower() if atom.GetIsAromatic() else symbol)


def annotate_bonds(
    ref_smiles: str,
    bonds: dict[frozenset[int], dict[int, set[str]]],
    chemical_dict: dict | None = None,
) -> tuple[dict[frozenset[int], dict[int, set[str]]], int, int]:
    """Replace a molecule's cut-bond tags with structural annotations.

    The bond signatures are kept exactly as they are -- only the tags change.
    Signatures are canonical atom ranks, the same space R-BRICS uses, which is
    what lets an R-BRICS bond's tags be transferred onto a retro bond.

    Args:
        ref_smiles: The target the bonds were found on.
        bonds: Cut bonds as stage A stored them, signature -> per-rank tag set.
        chemical_dict: Prepared R-BRICS patterns from
            :func:`~smiles_blocks.retrosynthesis.prep_rbrics_data`. Pass one in
            when annotating many molecules: building it per molecule is wasteful.

    Returns:
        tuple: the re-annotated bonds, the number tagged from R-BRICS patterns,
        and the number given a structural fallback. An unparseable target returns
        the bonds untouched with both counts zero.
    """
    if not bonds:
        return {}, 0, 0

    mol = Chem.MolFromSmiles(ref_smiles)
    if mol is None:
        return dict(bonds), 0, 0

    chemical_dict = chemical_dict or prep_rbrics_data()
    rbrics = retrosynthetic_analysis(ref_smiles, chemical_dict=chemical_dict)

    # Canonical rank -> atom index, to reach the atoms a signature names. The
    # ranking is a property of the molecule, so it matches the stored signatures.
    rank_to_index = {rank: index for index, rank in enumerate(Chem.CanonicalRankAtoms(mol))}

    annotated: dict[frozenset[int], dict[int, set[str]]] = {}
    from_rbrics = from_fallback = 0

    for signature in bonds:
        if signature in rbrics:
            annotated[signature] = {rank: set(tags) for rank, tags in rbrics[signature].items()}
            from_rbrics += 1
            continue

        per_atom: dict[int, set[str]] = defaultdict(set)
        for rank in signature:
            index = rank_to_index.get(rank)
            if index is None:
                continue
            per_atom[rank].add(sanitize_tag(fallback_atom_tag(mol.GetAtomWithIdx(index))))

        # Keep the bond even if its atoms could not be located, so a signature is
        # never silently dropped; the tag set is then simply empty.
        annotated[signature] = dict(per_atom)
        from_fallback += 1

    return annotated, from_rbrics, from_fallback


def annotation_stats(counts: dict[str, int]) -> str:
    """Format annotation counters for a log line.

    Args:
        counts: Keys ``rbrics``, ``fallback`` and ``bonds``.

    Returns:
        str: A one-line summary, with percentages when any bond was annotated.
    """
    total = counts.get("bonds", 0)
    if not total:
        return "no bonds annotated"
    rbrics = counts.get("rbrics", 0)
    fallback = counts.get("fallback", 0)
    return (
        f"{total:,} bonds: {rbrics:,} R-BRICS ({100 * rbrics / total:.1f}%), "
        f"{fallback:,} fallback ({100 * fallback / total:.1f}%)"
    )
