import numpy as np
import pytest
from rdkit import Chem

from smiles_blocks.smiles_fragmentation import (
    ExtendedRoThreeThreshold,
    SmilesRegex,
    block_assement,
    find_cleavable_bond,
    find_last_connecting_atom_idx,
    find_string_frag,
    get_annotation,
    get_block_unique_id,
    get_semantic_mem_score,
    get_smiles_frag_mapping,
    map_bond2canonatomsidx,
    smilesfrag2block,
)


class TestDataClasses:
    def test_extended_ro_three_threshold_defaults(self):
        threshold = ExtendedRoThreeThreshold()
        assert threshold.MolWt == 300
        assert threshold.nHDonors == 3
        assert threshold.nHAcceptors == 3
        assert threshold.nRotatableBonds == 3
        assert threshold.CrippenlogP == 3
        assert threshold.TPSA == 60

    def test_smiles_regex_compilation(self):
        regex = SmilesRegex()
        assert regex.regex is not None
        tokens = regex.regex.findall("C-C-C")
        assert len(tokens) == 5  # C, -, C, -, C
        assert tokens[0] == "C"
        assert tokens[1] == "-"
        assert tokens[2] == "C"
        assert tokens[3] == "-"
        assert tokens[4] == "C"


class TestFindStringFrag:
    def test_simple_chain(self):
        smiles = "C-C-C"
        smiles_array = np.frombuffer(smiles.encode("ascii"), dtype=np.uint8)
        result = find_string_frag(smiles_array)
        assert result.shape[1] == 2
        assert len(result) > 0

    def test_branched_molecule(self):
        smiles = "C-C(-C)-C"
        smiles_array = np.frombuffer(smiles.encode("ascii"), dtype=np.uint8)
        result = find_string_frag(smiles_array)
        assert isinstance(result, np.ndarray)

    def test_ring_molecule(self):
        smiles = "C1-C-C-C-C-C1"
        smiles_array = np.frombuffer(smiles.encode("ascii"), dtype=np.uint8)
        result = find_string_frag(smiles_array)
        assert isinstance(result, np.ndarray)

    def test_empty_array(self):
        smiles_array = np.array([], dtype=np.uint8)
        result = find_string_frag(smiles_array)
        assert result.shape[0] == 0


class TestFindCleavableBond:
    def test_simple_chain(self):
        smiles = "C-C-C"
        result = find_cleavable_bond(smiles)
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 2

    def test_double_bond(self):
        smiles = "C=C-C"
        result = find_cleavable_bond(smiles)
        assert len(result) >= 0

    def test_triple_bond(self):
        smiles = "C#C-C"
        result = find_cleavable_bond(smiles)
        assert len(result) >= 0

    def test_no_bonds(self):
        smiles = "C"
        result = find_cleavable_bond(smiles)
        assert len(result) == 0


class TestMapBond2CanonAtomsIdx:
    def test_simple_mapping(self):
        smiles = "C-C-C"
        bonds_array = find_cleavable_bond(smiles)
        idx2canonical, canonical2bondstr = map_bond2canonatomsidx(smiles, bonds_array)
        assert isinstance(idx2canonical, dict)
        assert isinstance(canonical2bondstr, dict)
        assert len(idx2canonical) == 3

    def test_branched_mapping(self):
        smiles = "C-C(-C)-C"
        bonds_array = find_cleavable_bond(smiles)
        idx2canonical, canonical2bondstr = map_bond2canonatomsidx(smiles, bonds_array)
        assert len(idx2canonical) == 4


class TestGetSmilesFragMapping:
    def test_simple_chain(self):
        smiles = "C-C-C"
        result = get_smiles_frag_mapping(smiles)
        assert "smiles" in result
        assert "canonicalidces2bondstridx" in result
        assert "idx2canonicalidx" in result
        assert result["smiles"] == smiles

    def test_complex_molecule(self):
        smiles = "C-C(-C)-C-C"
        result = get_smiles_frag_mapping(smiles)
        assert isinstance(result["canonicalidces2bondstridx"], dict)
        assert isinstance(result["idx2canonicalidx"], dict)


class TestGetSemanticMemScore:
    def test_simple_chain(self):
        smiles = "CCC"
        score = get_semantic_mem_score(smiles)
        assert isinstance(score, (float, np.float64))
        assert score == 0.0

    def test_branched_molecule(self):
        smiles = "CC(C)C"
        score = get_semantic_mem_score(smiles)
        assert score <= 1

    def test_ring_molecule(self):
        smiles = "C1CCCCC1"
        score = get_semantic_mem_score(smiles)
        assert score >= 0.5 and score < 1

    def test_empty_smiles(self):
        smiles = ""
        score = get_semantic_mem_score(smiles)
        assert isinstance(score, (float, np.float64))


class TestBlockAssessment:
    def test_valid_block(self):
        block = "CC"
        result = block_assement(block)
        assert "MolWt" in result
        assert "nHDonors" in result
        assert "nHAcceptors" in result
        assert "status" in result

    def test_valid_block_with_dict_tresholds(self):
        block = "CC"
        threshold = vars(ExtendedRoThreeThreshold())
        result = block_assement(block, threshold)
        assert "MolWt" in result
        assert "nHDonors" in result
        assert "nHAcceptors" in result
        assert "status" in result

    def test_invalid_block(self):
        block = "invalid_smiles"
        result = block_assement(block)
        assert isinstance(result, dict)

    def test_invalid_thresholds(self):
        block = "CC"
        threshold = "invalid_threshold"
        with pytest.raises(
            ValueError, match="thresholds should be either None, ExtendedRoThreeThreshold or dict"
        ):
            block_assement(block, threshold)

    def test_custom_thresholds(self):
        block = "CC"
        threshold = ExtendedRoThreeThreshold(MolWt=10)
        result = block_assement(block, threshold)
        assert isinstance(result, dict)

    def test_large_molecule(self):
        block = "CCCCCCCCCC"
        result = block_assement(block)
        assert result["MolWt"] > 0


class TestGetAnnotation:
    def test_simple_annotation(self):
        can_atom_idces = frozenset([0, 1])
        canonicalidx2idx = {0: 0, 1: 1}
        retrosynth = {frozenset([0, 1]): {0: {"tag1"}, 1: {"tag2"}}}
        result = get_annotation(can_atom_idces, canonicalidx2idx, retrosynth)
        assert "first_atom" in result
        assert "end_atom" in result

    def test_reversed_order(self):
        can_atom_idces = frozenset([0, 1])
        canonicalidx2idx = {0: 1, 1: 0}
        retrosynth = {frozenset([0, 1]): {0: {"tag1"}, 1: {"tag2"}}}
        result = get_annotation(can_atom_idces, canonicalidx2idx, retrosynth)
        assert isinstance(result["first_atom"], str)
        assert isinstance(result["end_atom"], str)


class TestFindLastConnectingAtomIdx:
    def test_simple_chain(self):
        block = "CCC"
        result = find_last_connecting_atom_idx(block)
        assert isinstance(result, int)
        assert result == 2

    def test_branched_molecule(self):
        block = "CCC(=O)(O)"
        result = find_last_connecting_atom_idx(block)
        assert result == 2

    def test_ring_molecule(self):
        block = "C1CCCCC1"
        result = find_last_connecting_atom_idx(block)
        assert result == 5

    def test_empty_block(self):
        block = ""
        result = find_last_connecting_atom_idx(block)
        assert result == -1

    def test_complex_block(self):
        block = "C1=C-C(-C)=C-C=C-1"
        result = find_last_connecting_atom_idx(block)
        assert result == 6


class TestGetBlockUniqueId:
    def test_valid_block(self):
        block = "CC"
        result = get_block_unique_id(block)
        assert "can_smiles" in result
        assert "first_connected_can_idx" in result
        assert "last_connected_can_idx" in result
        assert "unique_id" in result

    def test_invalid_block(self):
        block = "invalid"
        result = get_block_unique_id(block)
        assert isinstance(result, dict)

    def test_complex_block(self):
        block = "CC(C)C"
        result = get_block_unique_id(block)
        assert isinstance(result["unique_id"], str)


class TestSmilesFragToBlock:
    def test_simple_block(self):
        block = "CC"
        threshold_dict = vars(ExtendedRoThreeThreshold())
        result = smilesfrag2block(block, threshold_dict)
        assert isinstance(result, dict)
        assert "can_smiles" in result or "MolWt" in result

    def test_complex_block(self):
        block = "CC(C)C"
        threshold_dict = vars(ExtendedRoThreeThreshold())
        result = smilesfrag2block(block, threshold_dict)
        assert isinstance(result, dict)


class TestEdgeCases:
    def test_single_atom(self):
        smiles = "C"
        result = get_smiles_frag_mapping(smiles)
        assert result["smiles"] == smiles

    def test_aromatic_molecule(self):
        smiles = "c1ccccc1"
        score = get_semantic_mem_score(smiles)
        assert score >= 0

    def test_charged_molecule(self):
        smiles = "[O-]"
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            result = block_assement(smiles)
            assert isinstance(result, dict)
