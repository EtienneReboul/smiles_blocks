"""Tests for the backend-agnostic retrosynthesis bond extraction.

These cover the logic that converts atom-mapped reaction SMILES into bond
signatures on the target, without needing AiZynthFinder installed.
"""

import pytest
from rdkit import Chem

from smiles_blocks.retro_backends.bond_extraction import (
    DEFAULT_TAG,
    RouteStep,
    build_mapnum_to_rank,
    mapnum_bonds,
    route_cut_bonds,
    sanitize_tag,
    split_mapped_reaction,
    step_cut_bonds,
    strip_atom_maps,
)
from smiles_blocks.smiles_fragmentation import (
    ExtendedRoThreeThreshold,
    get_semantic_mem_score,
    process_set_smiles,
)

PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
# Amide C-N bond of paracetamol, as atom indices in the SMILES above.
PARACETAMOL_AMIDE = (1, 3)


def canonical_ranks(smiles: str) -> list[int]:
    """Canonical ranks of a map-free molecule, as the pipeline computes them."""
    return list(Chem.CanonicalRankAtoms(Chem.MolFromSmiles(smiles)))


def expected_signature(smiles: str, atom_pair: tuple[int, int]) -> frozenset:
    """Bond signature in canonical-rank space for a bond given by atom indices."""
    ranks = canonical_ranks(smiles)
    return frozenset((ranks[atom_pair[0]], ranks[atom_pair[1]]))


def make_mapped_retro(target: str, cut_atom_pairs: list[tuple[int, int]]) -> str:
    """Build a retro-direction mapped reaction SMILES that breaks given bonds.

    Maps atoms 1..N in index order (the same convention AiZynthFinder uses for the
    root molecule), then removes the requested bonds to form the precursor side.

    Args:
        target: Product SMILES.
        cut_atom_pairs: Bonds to break, as atom index pairs into ``target``.

    Returns:
        str: ``product>>precursors`` with atom maps on both sides.
    """
    mol = Chem.MolFromSmiles(target)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    # Kekulize before editing: breaking an aromatic ring bond would otherwise
    # leave acyclic atoms still flagged aromatic, which RDKit refuses to
    # sanitize. Aromaticity is re-perceived when the SMILES is parsed back.
    Chem.Kekulize(mol, clearAromaticFlags=True)
    product_smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)

    editable = Chem.RWMol(mol)
    for begin, end in cut_atom_pairs:
        editable.RemoveBond(begin, end)
    precursors = editable.GetMol()
    Chem.SanitizeMol(precursors)
    precursor_smiles = Chem.MolToSmiles(precursors)

    return f"{product_smiles}>>{precursor_smiles}"


class TestSanitizeTag:
    def test_strips_separators_used_downstream(self):
        # "_" joins tags and "." joins blocks/ids, so neither may survive.
        assert "_" not in sanitize_tag("amide_coupling")
        assert "." not in sanitize_tag("rxn.class.3")

    def test_keeps_alphanumerics_and_dashes(self):
        assert sanitize_tag("Amide-1") == "Amide-1"

    def test_falls_back_when_nothing_usable(self):
        assert sanitize_tag("___") == DEFAULT_TAG
        assert sanitize_tag("") == DEFAULT_TAG

    def test_truncates(self):
        assert len(sanitize_tag("a" * 100)) == 24


class TestStripAtomMaps:
    def test_does_not_mutate_input(self):
        mol = Chem.MolFromSmiles("[CH4:1]")
        strip_atom_maps(mol)
        assert mol.GetAtomWithIdx(0).GetAtomMapNum() == 1

    def test_clears_all_maps(self):
        mol = Chem.MolFromSmiles("[CH3:1][OH:2]")
        clean = strip_atom_maps(mol)
        assert all(atom.GetAtomMapNum() == 0 for atom in clean.GetAtoms())

    def test_restores_true_canonical_ranks(self):
        """The core trap: atom maps make CanonicalRankAtoms return the mapping.

        Without stripping, ranks follow the map numbers rather than the molecule,
        so every bond signature would be wrong and would silently fail to
        intersect the fragmentation side.
        """
        mapped = Chem.MolFromSmiles(
            "[CH3:1][C:2](=[O:3])[NH:4][c:5]1[cH:6][cH:7][c:8]([OH:9])[cH:10][cH:11]1"
        )
        mapped_ranks = list(Chem.CanonicalRankAtoms(mapped))
        stripped_ranks = list(Chem.CanonicalRankAtoms(strip_atom_maps(mapped)))

        assert mapped_ranks == list(range(mapped.GetNumAtoms())), (
            "expected maps to drive the ranking"
        )
        assert stripped_ranks == canonical_ranks(PARACETAMOL)
        assert stripped_ranks != mapped_ranks


class TestMapnumBonds:
    def test_skips_unmapped_endpoints(self):
        mol = Chem.MolFromSmiles("[CH3:1][CH2:2]O")
        # C1-C2 is traceable; C2-O is not, because O carries no map.
        assert mapnum_bonds(mol) == {frozenset((1, 2))}

    def test_handles_multifragment_input(self):
        mol = Chem.MolFromSmiles("[CH3:1][CH3:2].[NH2:3][CH3:4]")
        assert mapnum_bonds(mol) == {frozenset((1, 2)), frozenset((3, 4))}


class TestSplitMappedReaction:
    def test_product_is_on_the_left(self):
        """AiZynthFinder writes target>>precursors; the product leads."""
        product, precursors = split_mapped_reaction("CCO>>CC=O.[H][H]")
        assert product == "CCO"
        assert precursors == "CC=O.[H][H]"

    def test_accepts_agent_field(self):
        product, precursors = split_mapped_reaction("CCO>Pd>CC=O")
        assert product == "CCO"
        assert precursors == "CC=O"

    @pytest.mark.parametrize("bad", ["CCO", "a>b>c>d", ">>CCO", "CCO>>"])
    def test_rejects_malformed(self, bad):
        with pytest.raises(ValueError):
            split_mapped_reaction(bad)


class TestStepCutBonds:
    def test_finds_the_broken_bond(self):
        rsmi = make_mapped_retro(PARACETAMOL, [PARACETAMOL_AMIDE])
        # Map numbers are atom index + 1 by construction.
        expected = frozenset((PARACETAMOL_AMIDE[0] + 1, PARACETAMOL_AMIDE[1] + 1))
        assert step_cut_bonds(rsmi) == {expected}

    def test_no_cut_when_nothing_breaks(self):
        mol = Chem.MolFromSmiles(PARACETAMOL)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
        identity = Chem.MolToSmiles(mol)
        assert step_cut_bonds(f"{identity}>>{identity}") == set()

    def test_unparseable_side_yields_nothing(self):
        assert step_cut_bonds("not_a_molecule>>CCO") == set()

    def test_unmapped_reaction_yields_nothing(self):
        assert step_cut_bonds("CC(=O)Nc1ccccc1>>CC(=O)O.Nc1ccccc1") == set()


class TestBuildMapnumToRank:
    def test_maps_every_mapped_atom(self):
        mol = Chem.MolFromSmiles(PARACETAMOL)
        target = Chem.MolFromSmiles(PARACETAMOL)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)

        table = build_mapnum_to_rank(target, mol)

        assert table is not None
        assert len(table) == target.GetNumAtoms()
        # A permutation of the ranks, i.e. a genuine bijection.
        assert sorted(table.values()) == list(range(target.GetNumAtoms()))

    def test_is_independent_of_atom_ordering(self):
        """The same molecule written differently must give the same rank table."""
        target = Chem.MolFromSmiles(PARACETAMOL)
        reordered = Chem.MolFromSmiles(
            Chem.MolToSmiles(Chem.MolFromSmiles(PARACETAMOL), doRandom=True, canonical=False)
        )
        for atom in reordered.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)

        table = build_mapnum_to_rank(target, reordered)
        assert table is not None
        # Rank of the mapped atom must agree with the target's own ranking.
        ranks = list(Chem.CanonicalRankAtoms(strip_atom_maps(reordered)))
        assert table == {
            atom.GetAtomMapNum(): ranks[atom.GetIdx()]
            for atom in reordered.GetAtoms()
            if atom.GetAtomMapNum()
        }

    def test_rejects_a_different_molecule(self):
        target = Chem.MolFromSmiles(PARACETAMOL)
        other = Chem.MolFromSmiles("[CH3:1][CH2:2][OH:3]")
        assert build_mapnum_to_rank(target, other) is None


class TestRouteCutBonds:
    def test_recovers_an_acyclic_cut_in_rank_space(self):
        steps = [RouteStep(make_mapped_retro(PARACETAMOL, [PARACETAMOL_AMIDE]), tag="amide")]
        bonds = route_cut_bonds(PARACETAMOL, steps)

        assert set(bonds) == {expected_signature(PARACETAMOL, PARACETAMOL_AMIDE)}

    def test_tags_both_endpoints(self):
        steps = [RouteStep(make_mapped_retro(PARACETAMOL, [PARACETAMOL_AMIDE]), tag="amide")]
        bonds = route_cut_bonds(PARACETAMOL, steps)

        signature, annotation = next(iter(bonds.items()))
        assert set(annotation) == set(signature)
        assert all(tags == {"amide"} for tags in annotation.values())

    def test_shape_matches_rbrics_contract(self):
        """Must be dict[frozenset[int], dict[int, set[str]]] to be a drop-in."""
        steps = [RouteStep(make_mapped_retro(PARACETAMOL, [PARACETAMOL_AMIDE]))]
        bonds = route_cut_bonds(PARACETAMOL, steps)

        for signature, annotation in bonds.items():
            assert isinstance(signature, frozenset)
            assert all(isinstance(rank, int) for rank in signature)
            assert isinstance(annotation, dict)
            for rank, tags in annotation.items():
                assert isinstance(rank, int)
                assert isinstance(tags, set)
                assert all(isinstance(tag, str) for tag in tags)

    def test_ring_bonds_are_rejected(self):
        """Ring bonds cannot be cleaved in a SMILES string."""
        cyclohexane = "C1CCCCC1"
        steps = [RouteStep(make_mapped_retro(cyclohexane, [(0, 1)]))]
        assert route_cut_bonds(cyclohexane, steps) == {}

    def test_aromatic_ring_bonds_are_rejected(self):
        biphenyl = "c1ccc(-c2ccccc2)cc1"
        mol = Chem.MolFromSmiles(biphenyl)
        ring_bond = next(
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            for bond in mol.GetBonds()
            if bond.IsInRing()
        )
        steps = [RouteStep(make_mapped_retro(biphenyl, [ring_bond]))]
        assert route_cut_bonds(biphenyl, steps) == {}

    def test_keeps_the_exocyclic_bond_of_a_ring_system(self):
        biphenyl = "c1ccc(-c2ccccc2)cc1"
        mol = Chem.MolFromSmiles(biphenyl)
        linker = next(
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            for bond in mol.GetBonds()
            if not bond.IsInRing()
        )
        steps = [RouteStep(make_mapped_retro(biphenyl, [linker]))]
        assert set(route_cut_bonds(biphenyl, steps)) == {expected_signature(biphenyl, linker)}

    def test_wrong_direction_is_not_silently_accepted(self):
        """Reactants>>product must not be mistaken for a route on the target."""
        correct = make_mapped_retro(PARACETAMOL, [PARACETAMOL_AMIDE])
        product, precursors = correct.split(">>")
        flipped = f"{precursors}>>{product}"
        assert route_cut_bonds(PARACETAMOL, [RouteStep(flipped)]) == {}

    def test_route_for_another_molecule_is_discarded(self):
        steps = [RouteStep(make_mapped_retro("CCOc1ccccc1", [(1, 2)]))]
        assert route_cut_bonds(PARACETAMOL, steps) == {}

    def test_multiple_cuts_in_one_step(self):
        target = "CC(=O)Nc1ccc(OCC)cc1"
        mol = Chem.MolFromSmiles(target)
        amide = (1, 3)
        ether = next(
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            for bond in mol.GetBonds()
            if not bond.IsInRing()
            and {bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()} == {"O", "C"}
            and bond.GetBeginAtom().GetIdx() > 5
        )
        steps = [RouteStep(make_mapped_retro(target, [amide, ether]))]

        bonds = route_cut_bonds(target, steps)
        assert expected_signature(target, amide) in bonds
        assert expected_signature(target, ether) in bonds

    def test_multistep_route_accumulates_cuts(self):
        """A deeper step's cut must trace back through inherited map numbers."""
        target = "CC(=O)Nc1ccc(OCC)cc1"
        amide = (1, 3)

        root = make_mapped_retro(target, [amide])
        # Second step operates on the aniline precursor, cutting its ether bond,
        # keeping the map numbers the root assigned.
        aniline_frag = root.split(">>")[1].split(".")
        aniline = max(aniline_frag, key=len)
        aniline_mol = Chem.MolFromSmiles(aniline)
        ether = next(
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            for bond in aniline_mol.GetBonds()
            if not bond.IsInRing()
            and {bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()} == {"O", "C"}
        )
        editable = Chem.RWMol(aniline_mol)
        editable.RemoveBond(*ether)
        deeper = editable.GetMol()
        Chem.SanitizeMol(deeper)
        step_two = f"{aniline}>>{Chem.MolToSmiles(deeper)}"

        bonds = route_cut_bonds(
            target, [RouteStep(root, tag="amide"), RouteStep(step_two, tag="ether", depth=1)]
        )

        assert len(bonds) == 2
        tags = {
            tag for annotation in bonds.values() for tags in annotation.values() for tag in tags
        }
        assert tags == {"amide", "ether"}

    def test_empty_inputs(self):
        assert route_cut_bonds(PARACETAMOL, []) == {}
        assert route_cut_bonds("not_a_smiles", [RouteStep("CCO>>CC=O")]) == {}

    def test_result_is_picklable(self):
        """Workers receive this across process boundaries."""
        import pickle

        steps = [RouteStep(make_mapped_retro(PARACETAMOL, [PARACETAMOL_AMIDE]))]
        bonds = route_cut_bonds(PARACETAMOL, steps)
        assert pickle.loads(pickle.dumps(bonds)) == bonds


class TestFragmentationCompatibility:
    """The real proof: the output must drive the existing fragmentation code."""

    def test_drives_process_set_smiles(self):
        steps = [RouteStep(make_mapped_retro(PARACETAMOL, [PARACETAMOL_AMIDE]), tag="amide")]
        bonds = route_cut_bonds(PARACETAMOL, steps)
        assert bonds, "precondition: a cut was found"

        mol = Chem.MolFromSmiles(PARACETAMOL)
        randomized = set(
            Chem.MolToRandomSmilesVect(
                mol,
                200,
                kekuleSmiles=True,
                isomericSmiles=False,
                allBondsExplicit=True,
                randomSeed=42,
            )
        )
        scored = [(smiles, get_semantic_mem_score(smiles)) for smiles in randomized]

        status, blocked, blocks = process_set_smiles(scored, bonds)

        assert status is True
        assert blocked["retro_bond_ratio"] == 1.0, "the single cut should be recoverable"
        assert len(blocks) == 2, "one cut yields two blocks"
        assert blocked["smiles_blocked"].count(".") == 1
        # The tag assigned by the route must reach the block annotation.
        assert any("amide" in block["end_tag"] for block in blocks)

    def test_rank_space_agrees_with_rbrics(self):
        """Bonds found here must be namable in the same space as R-BRICS.

        R-BRICS independently finds the paracetamol amide bond; both backends must
        produce the identical signature, otherwise the two are not comparable.
        """
        from smiles_blocks.retrosynthesis import retrosynthetic_analysis

        rbrics = retrosynthetic_analysis(PARACETAMOL)
        steps = [RouteStep(make_mapped_retro(PARACETAMOL, [PARACETAMOL_AMIDE]))]
        retro = route_cut_bonds(PARACETAMOL, steps)

        assert set(retro).issubset(set(rbrics)), (
            f"signature mismatch: {set(retro)} not within R-BRICS {set(rbrics)}"
        )

    def test_thresholds_unchanged(self):
        """Block quality assessment is backend-independent."""
        assert ExtendedRoThreeThreshold().MolWt == 300
