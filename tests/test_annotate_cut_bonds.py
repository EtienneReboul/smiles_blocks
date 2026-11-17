"""Tests for structural annotation of retrosynthesis cut bonds.

The point of this step is that a bond's tags are *useful*, so the tests check the
two things that could silently make them useless again: that a bond coinciding
with an R-BRICS bond really inherits that vocabulary, and that a bond R-BRICS
does not know about still gets a tag rather than nothing.
"""

from rdkit import Chem

from smiles_blocks.retro_backends.annotate import (
    FALLBACK_PREFIX,
    annotate_bonds,
    annotation_stats,
    fallback_atom_tag,
)
from smiles_blocks.retrosynthesis import prep_rbrics_data, retrosynthetic_analysis

PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
STALE_TAG = "0-0-Unrecognized"


def ranks(smiles: str) -> list[int]:
    return list(Chem.CanonicalRankAtoms(Chem.MolFromSmiles(smiles)))


def signature(smiles: str, atom_pair: tuple[int, int]) -> frozenset:
    r = ranks(smiles)
    return frozenset((r[atom_pair[0]], r[atom_pair[1]]))


def stale(sig: frozenset) -> dict:
    """Bonds as stage A stored them: the useless constant tag on both atoms."""
    return {rank: {STALE_TAG} for rank in sig}


class TestFallbackAtomTag:
    def test_aliphatic_is_uppercase(self):
        mol = Chem.MolFromSmiles("CCO")
        assert fallback_atom_tag(mol.GetAtomWithIdx(0)) == "XC"

    def test_aromatic_is_lowercase(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert fallback_atom_tag(mol.GetAtomWithIdx(0)) == "Xc"

    def test_two_letter_element(self):
        mol = Chem.MolFromSmiles("Brc1ccccc1")
        assert fallback_atom_tag(mol.GetAtomWithIdx(0)) == "XBr"

    def test_prefixed_so_it_cannot_pass_as_an_rbrics_class(self):
        mol = Chem.MolFromSmiles("CCO")
        tag = fallback_atom_tag(mol.GetAtomWithIdx(2))
        assert tag.startswith(FALLBACK_PREFIX)
        assert not tag.startswith("L")


class TestAnnotateBonds:
    def test_rbrics_bond_inherits_the_rbrics_vocabulary(self):
        """The amide bond is an R-BRICS bond, so it must come back with L-tags."""
        chem = prep_rbrics_data()
        amide = signature(PARACETAMOL, (1, 3))
        assert amide in retrosynthetic_analysis(PARACETAMOL, chemical_dict=chem), (
            "precondition: R-BRICS finds this bond"
        )

        annotated, from_rbrics, from_fallback = annotate_bonds(
            PARACETAMOL, {amide: stale(amide)}, chemical_dict=chem
        )

        assert (from_rbrics, from_fallback) == (1, 0)
        tags = {t for ann in annotated.values() for ts in ann.values() for t in ts}
        assert tags, "bond must carry tags"
        assert all(t.startswith("L") for t in tags), tags

    def test_stale_tag_is_replaced(self):
        chem = prep_rbrics_data()
        amide = signature(PARACETAMOL, (1, 3))
        annotated, _, _ = annotate_bonds(
            PARACETAMOL, {amide: stale(amide)}, chemical_dict=chem
        )
        tags = {t for ann in annotated.values() for ts in ann.values() for t in ts}
        assert STALE_TAG not in tags
        assert not any("Unrecognized" in t for t in tags)

    def test_non_rbrics_bond_still_gets_a_tag(self):
        """A planner finds disconnections R-BRICS has no rule for; keep them labelled."""
        chem = prep_rbrics_data()
        smiles = "Brc1ccccc1"
        # The aryl-bromide bond: a real disconnection, not an R-BRICS pattern.
        bond = signature(smiles, (0, 1))
        assert bond not in retrosynthetic_analysis(smiles, chemical_dict=chem)

        annotated, from_rbrics, from_fallback = annotate_bonds(
            smiles, {bond: stale(bond)}, chemical_dict=chem
        )

        assert (from_rbrics, from_fallback) == (0, 1)
        tags = {t for ann in annotated.values() for ts in ann.values() for t in ts}
        assert tags == {"XBr", "Xc"}, tags

    def test_bond_signatures_are_preserved(self):
        """Only tags change -- a moved signature would break the stage-B join."""
        chem = prep_rbrics_data()
        amide = signature(PARACETAMOL, (1, 3))
        before = {amide: stale(amide)}
        annotated, _, _ = annotate_bonds(PARACETAMOL, before, chemical_dict=chem)
        assert set(annotated) == set(before)

    def test_annotation_shape_matches_the_pipeline_contract(self):
        chem = prep_rbrics_data()
        amide = signature(PARACETAMOL, (1, 3))
        annotated, _, _ = annotate_bonds(
            PARACETAMOL, {amide: stale(amide)}, chemical_dict=chem
        )
        for sig, ann in annotated.items():
            assert isinstance(sig, frozenset)
            assert set(ann).issubset(set(sig))
            assert all(isinstance(ts, set) for ts in ann.values())

    def test_empty_and_unparseable_inputs(self):
        chem = prep_rbrics_data()
        assert annotate_bonds(PARACETAMOL, {}, chemical_dict=chem) == ({}, 0, 0)

        bogus = frozenset((0, 1))
        bonds, rbrics, fallback = annotate_bonds(
            "not_a_molecule", {bogus: stale(bogus)}, chemical_dict=chem
        )
        assert set(bonds) == {bogus}, "bonds are returned untouched, never dropped"
        assert (rbrics, fallback) == (0, 0)

    def test_builds_its_own_patterns_when_none_passed(self):
        amide = signature(PARACETAMOL, (1, 3))
        _, from_rbrics, _ = annotate_bonds(PARACETAMOL, {amide: stale(amide)})
        assert from_rbrics == 1

    def test_unknown_rank_does_not_drop_the_bond(self):
        """A signature naming an atom the molecule lacks keeps its entry."""
        chem = prep_rbrics_data()
        absurd = frozenset((9998, 9999))
        annotated, _, fallback = annotate_bonds(
            PARACETAMOL, {absurd: stale(absurd)}, chemical_dict=chem
        )
        assert set(annotated) == {absurd}
        assert annotated[absurd] == {}
        assert fallback == 1


class TestAnnotationStats:
    def test_reports_percentages(self):
        line = annotation_stats({"bonds": 100, "rbrics": 95, "fallback": 5})
        assert "95" in line and "5" in line

    def test_handles_nothing_annotated(self):
        assert annotation_stats({"bonds": 0}) == "no bonds annotated"


class TestFragmentationCompatibility:
    def test_annotated_bonds_drive_fragmentation(self):
        """End to end: annotated bonds -> blocks carrying real labels."""
        from smiles_blocks.smiles_fragmentation import (
            get_semantic_mem_score,
            process_set_smiles,
        )

        chem = prep_rbrics_data()
        amide = signature(PARACETAMOL, (1, 3))
        bonds, _, _ = annotate_bonds(PARACETAMOL, {amide: stale(amide)}, chemical_dict=chem)

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
        scored = [(s, get_semantic_mem_score(s)) for s in randomized]

        status, blocked, blocks = process_set_smiles(scored, bonds)

        assert status is True
        assert blocked["retro_bond_ratio"] == 1.0
        assert len(blocks) == 2
        labels = {blk["begin_tag"] for blk in blocks} | {blk["end_tag"] for blk in blocks}
        assert not any("Unrecognized" in label for label in labels), labels
        assert any(label.startswith("L") for label in labels), labels
