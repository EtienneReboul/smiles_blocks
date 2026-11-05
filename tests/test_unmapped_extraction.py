"""Tests for cut-bond extraction from unmapped retrosynthesis steps.

This path is inferential rather than exact, so the tests concentrate on two
things: that it agrees with the mapped extraction where both apply, and that its
guards actually reject the cases it cannot know about.
"""

import pytest
from rdkit import Chem

from smiles_blocks.retro_backends.bond_extraction import RouteStep, route_cut_bonds
from smiles_blocks.retro_backends.unmapped import (
    UnmappedStep,
    step_cut_bonds_unmapped,
    unmapped_route_cut_bonds,
)

PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
PARACETAMOL_AMIDE = (1, 3)


def canonical_ranks(smiles: str) -> list[int]:
    return list(Chem.CanonicalRankAtoms(Chem.MolFromSmiles(smiles)))


def expected_signature(smiles: str, atom_pair: tuple[int, int]) -> frozenset:
    ranks = canonical_ranks(smiles)
    return frozenset((ranks[atom_pair[0]], ranks[atom_pair[1]]))


def mols(*smiles: str) -> list[Chem.Mol]:
    return [Chem.MolFromSmiles(s) for s in smiles]


class TestStepCutBondsUnmapped:
    def test_finds_amide_disconnection_through_a_leaving_group(self):
        """The acid chloride's Cl is absent from the product, yet it still matches."""
        product = Chem.MolFromSmiles(PARACETAMOL)
        precursors = mols("CC(=O)Cl", "Nc1ccc(O)cc1")

        cuts = step_cut_bonds_unmapped(product, precursors)

        assert cuts == {frozenset(PARACETAMOL_AMIDE)}

    def test_finds_amide_disconnection_from_the_free_acid(self):
        product = Chem.MolFromSmiles(PARACETAMOL)
        precursors = mols("CC(=O)O", "Nc1ccc(O)cc1")

        assert step_cut_bonds_unmapped(product, precursors) == {frozenset(PARACETAMOL_AMIDE)}

    def test_finds_a_biaryl_coupling(self):
        """Suzuki-type: boronic acid and halide, both with leaving groups."""
        biphenyl = "c1ccc(-c2ccccc2)cc1"
        product = Chem.MolFromSmiles(biphenyl)
        precursors = mols("OB(O)c1ccccc1", "Brc1ccccc1")

        cuts = step_cut_bonds_unmapped(product, precursors)

        assert len(cuts) == 1
        linker = next(
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            for bond in product.GetBonds()
            if not bond.IsInRing()
        )
        assert cuts == {frozenset(linker)}

    def test_rejects_a_single_precursor(self):
        """One precursor cannot evidence a disconnection."""
        product = Chem.MolFromSmiles(PARACETAMOL)
        assert step_cut_bonds_unmapped(product, mols("CC(=O)Nc1ccc(O)cc1")) == set()

    def test_rejects_when_a_precursor_still_holds_the_whole_product(self):
        """A functional-group interconversion takes nothing apart."""
        product = Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1")
        # Product is a substructure of the acetylated precursor: nothing broke.
        precursors = mols("CC(=O)Nc1ccc(OC(C)=O)cc1", "O")

        assert step_cut_bonds_unmapped(product, precursors) == set()

    def test_rejects_both_fragments_in_the_same_precursor(self):
        product = Chem.MolFromSmiles("CCOCC")
        # Both halves occur only in precursor 0, so no distinct assignment exists.
        precursors = mols("CCOCCOCC", "O")

        assert step_cut_bonds_unmapped(product, precursors) == set()

    def test_ring_bonds_are_never_cut(self):
        product = Chem.MolFromSmiles("C1CCCCC1")
        precursors = mols("CCCCCC", "C1CCCCC1C")

        assert step_cut_bonds_unmapped(product, precursors) == set()

    def test_min_fragment_atoms_suppresses_trivial_pieces(self):
        """A one-atom methyl matches almost anything, so it is poor evidence."""
        product = Chem.MolFromSmiles("CCc1ccccc1")
        precursors = mols("C", "Cc1ccccc1")

        permissive = step_cut_bonds_unmapped(product, precursors, min_fragment_atoms=1)
        strict = step_cut_bonds_unmapped(product, precursors, min_fragment_atoms=2)

        assert permissive, "with no size floor the methyl cut is reported"
        assert strict == set(), "with the default floor it is suppressed"

    def test_handles_none_product(self):
        assert step_cut_bonds_unmapped(None, mols("CCO", "CC")) == set()

    def test_no_bonds_to_cut(self):
        assert step_cut_bonds_unmapped(Chem.MolFromSmiles("O"), mols("CCO", "CC")) == set()


class TestUnmappedRouteCutBonds:
    def test_produces_rank_space_signatures(self):
        steps = [
            UnmappedStep(
                product_smiles=PARACETAMOL,
                precursor_smiles=("CC(=O)Cl", "Nc1ccc(O)cc1"),
                tag="amide",
            )
        ]
        bonds = unmapped_route_cut_bonds(PARACETAMOL, steps)

        assert set(bonds) == {expected_signature(PARACETAMOL, PARACETAMOL_AMIDE)}

    def test_matches_the_rbrics_contract_shape(self):
        steps = [UnmappedStep(PARACETAMOL, ("CC(=O)Cl", "Nc1ccc(O)cc1"))]
        bonds = unmapped_route_cut_bonds(PARACETAMOL, steps)

        for signature, annotation in bonds.items():
            assert isinstance(signature, frozenset)
            assert all(isinstance(rank, int) for rank in signature)
            assert set(annotation) == set(signature)
            assert all(isinstance(tags, set) for tags in annotation.values())

    def test_tolerates_a_differently_written_target(self):
        """Product and target may be the same molecule in different atom orders."""
        other_ordering = Chem.MolToSmiles(
            Chem.MolFromSmiles(PARACETAMOL), doRandom=True, canonical=False
        )
        steps = [UnmappedStep(other_ordering, ("CC(=O)Cl", "Nc1ccc(O)cc1"))]

        bonds = unmapped_route_cut_bonds(PARACETAMOL, steps)

        assert set(bonds) == {expected_signature(PARACETAMOL, PARACETAMOL_AMIDE)}

    def test_skips_steps_on_intermediates(self):
        """Without atom maps a deeper step cannot be located on the target."""
        steps = [
            UnmappedStep(
                product_smiles="Nc1ccc(OCC)cc1",  # an intermediate, not the target
                precursor_smiles=("Nc1ccc(O)cc1", "CCBr"),
            )
        ]
        assert unmapped_route_cut_bonds(PARACETAMOL, steps) == {}

    def test_uses_only_the_target_step_of_a_multistep_route(self):
        steps = [
            UnmappedStep(PARACETAMOL, ("CC(=O)Cl", "Nc1ccc(O)cc1"), tag="amide"),
            UnmappedStep("Nc1ccc(O)cc1", ("[N+](=O)([O-])c1ccc(O)cc1",), tag="nitro-red", depth=1),
        ]
        bonds = unmapped_route_cut_bonds(PARACETAMOL, steps)

        assert len(bonds) == 1
        tags = {
            tag for annotation in bonds.values() for tags in annotation.values() for tag in tags
        }
        assert tags == {"amide"}

    def test_empty_inputs(self):
        assert unmapped_route_cut_bonds(PARACETAMOL, []) == {}
        assert (
            unmapped_route_cut_bonds("not_a_smiles", [UnmappedStep(PARACETAMOL, ("C", "N"))]) == {}
        )

    def test_unparseable_precursors_are_ignored(self):
        steps = [UnmappedStep(PARACETAMOL, ("not_a_molecule", "Nc1ccc(O)cc1"))]
        # Only one usable precursor remains, which cannot evidence a cut.
        assert unmapped_route_cut_bonds(PARACETAMOL, steps) == {}

    def test_tags_are_sanitized(self):
        steps = [UnmappedStep(PARACETAMOL, ("CC(=O)Cl", "Nc1ccc(O)cc1"), tag="amide_coupling.v2")]
        bonds = unmapped_route_cut_bonds(PARACETAMOL, steps)

        tags = {
            tag for annotation in bonds.values() for tags in annotation.values() for tag in tags
        }
        assert all("_" not in tag and "." not in tag for tag in tags), tags


class TestAgreementWithMappedExtraction:
    """The unmapped path must land on the same bonds as the exact mapped path."""

    @pytest.mark.parametrize(
        "target,precursors,cut_atoms",
        [
            (PARACETAMOL, ("CC(=O)Cl", "Nc1ccc(O)cc1"), PARACETAMOL_AMIDE),
            ("CCOc1ccccc1", ("CCBr", "Oc1ccccc1"), (1, 2)),
            ("CC(=O)Nc1ccc(OCC)cc1", ("CC(=O)O", "Nc1ccc(OCC)cc1"), (1, 3)),
        ],
    )
    def test_same_signature_as_mapped_route(self, target, precursors, cut_atoms):
        """Build the same reaction twice -- mapped and unmapped -- and compare."""
        # Mapped form, as AiZynthFinder would give it.
        mol = Chem.MolFromSmiles(target)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
        mapped_product = Chem.MolToSmiles(mol)
        editable = Chem.RWMol(mol)
        editable.RemoveBond(*cut_atoms)
        broken = editable.GetMol()
        Chem.SanitizeMol(broken)
        mapped_rsmi = f"{mapped_product}>>{Chem.MolToSmiles(broken)}"

        from_mapped = route_cut_bonds(target, [RouteStep(mapped_rsmi)])
        from_unmapped = unmapped_route_cut_bonds(target, [UnmappedStep(target, precursors)])

        assert from_mapped, "precondition: the mapped path finds the bond"
        assert set(from_unmapped) == set(from_mapped), (
            "unmapped inference disagreed with exact mapped extraction"
        )

    def test_agrees_with_rbrics_on_paracetamol(self):
        """Must also be namable in the same space R-BRICS uses."""
        from smiles_blocks.retrosynthesis import retrosynthetic_analysis

        rbrics = retrosynthetic_analysis(PARACETAMOL)
        unmapped = unmapped_route_cut_bonds(
            PARACETAMOL, [UnmappedStep(PARACETAMOL, ("CC(=O)Cl", "Nc1ccc(O)cc1"))]
        )

        assert set(unmapped).issubset(set(rbrics))


class TestFragmentationCompatibility:
    def test_drives_process_set_smiles(self):
        """End to end: unmapped route -> bonds -> blocks."""
        from smiles_blocks.smiles_fragmentation import (
            get_semantic_mem_score,
            process_set_smiles,
        )

        bonds = unmapped_route_cut_bonds(
            PARACETAMOL,
            [UnmappedStep(PARACETAMOL, ("CC(=O)Cl", "Nc1ccc(O)cc1"), tag="amide")],
        )
        assert bonds

        randomized = set(
            Chem.MolToRandomSmilesVect(
                Chem.MolFromSmiles(PARACETAMOL),
                200,
                kekuleSmiles=True,
                isomericSmiles=False,
                allBondsExplicit=True,
                randomSeed=11,
            )
        )
        scored = [(smiles, get_semantic_mem_score(smiles)) for smiles in randomized]

        status, blocked, blocks = process_set_smiles(scored, bonds)

        assert status is True
        assert blocked["retro_bond_ratio"] == 1.0
        assert len(blocks) == 2
        assert any("amide" in block["end_tag"] for block in blocks)
