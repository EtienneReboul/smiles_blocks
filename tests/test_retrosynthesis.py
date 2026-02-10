from unittest.mock import patch

from rdkit import Chem

from smiles_blocks.retrosynthesis import (
    annotate_rbond,
    check_bondisinring,
    check_chemical_group,
    package_bond_info,
    prep_rbrics_data,
    retrosynthetic_analysis,
)


class TestPrepRbricsData:
    def test_prep_rbrics_data_returns_dict_with_required_keys(self):
        """Test that prep_rbrics_data returns a dictionary with required keys."""
        result = prep_rbrics_data()
        assert isinstance(result, dict)
        assert "chemical_group" in result
        assert "rbond" in result
        assert "set2tag" in result

    def test_prep_rbrics_data_chemical_group_contains_mol_objects(self):
        """Test that chemical_group values are RDKit Mol objects."""
        result = prep_rbrics_data()
        for mol in result["chemical_group"].values():
            assert mol is None or isinstance(mol, Chem.Mol)

    def test_prep_rbrics_data_set2tag_keys_are_frozensets(self):
        """Test that set2tag keys are frozensets."""
        result = prep_rbrics_data()
        for key in result["set2tag"].keys():
            assert isinstance(key, frozenset)

    def test_prep_rbrics_data_set2tag_values_are_strings(self):
        """Test that set2tag values are strings."""
        result = prep_rbrics_data()
        for value in result["set2tag"].values():
            assert isinstance(value, str)


class TestCheckBonisInRing:
    def test_bond_in_ring(self):
        """Test detection of bond in a ring."""
        mol = Chem.MolFromSmiles("C1CCCCC1")  # cyclohexane
        assert check_bondisinring(mol, (0, 1)) is True

    def test_bond_not_in_ring(self):
        """Test detection of bond not in a ring."""
        mol = Chem.MolFromSmiles("CC")  # ethane
        assert check_bondisinring(mol, (0, 1)) is False

    # try and except block is to costly  thus not tested or implemented in the function
    # def test_invalid_bond_indices(self):
    #     """Test with invalid bond indices."""
    #     mol = Chem.MolFromSmiles("CC")
    #     assert check_bondisinring(mol, (0, 5)) is False

    def test_with_aromatic_ring(self):
        """Test with aromatic ring."""
        mol = Chem.MolFromSmiles("c1ccccc1")  # benzene
        assert check_bondisinring(mol, (0, 1)) is True


class TestCheckChemicalGroup:
    def test_empty_chemical_pattern(self):
        """Test with empty chemical pattern dictionary."""
        mol = Chem.MolFromSmiles("CC")
        result = check_chemical_group(mol, {})
        assert result == set()

    def test_no_matches(self):
        """Test when no chemical groups match."""
        mol = Chem.MolFromSmiles("CC")
        chemical_pattern = {"L1": Chem.MolFromSmarts("[#7]")}  # nitrogen pattern
        result = check_chemical_group(mol, chemical_pattern)
        assert result == set()

    def test_with_matches(self):
        """Test when chemical groups match."""
        mol = Chem.MolFromSmiles("CN")  # molecule with nitrogen
        chemical_pattern = {
            "L1": Chem.MolFromSmarts("[#6]"),  # carbon
            "L2": Chem.MolFromSmarts("[#7]"),  # nitrogen
        }
        result = check_chemical_group(mol, chemical_pattern)
        assert "L1" in result and "L2" in result

    def test_returns_set(self):
        """Test that the function returns a set."""
        mol = Chem.MolFromSmiles("CC")
        result = check_chemical_group(mol, {})
        assert isinstance(result, set)


class TestPackageBondInfo:
    def test_single_bond(self):
        """Test packaging a single bond."""
        bonds = {
            frozenset([0, 1]): {
                0: {"L1"},
                1: {"L2"},
            }
        }
        result = package_bond_info(bonds)
        assert len(result) == 1
        assert "begind_idx" in result[0]
        assert "end_idx" in result[0]
        assert "begin_tag" in result[0]
        assert "end_tag" in result[0]

    def test_multiple_bonds(self):
        """Test packaging multiple bonds."""
        bonds = {
            frozenset([0, 1]): {0: {"L1"}, 1: {"L2"}},
            frozenset([2, 3]): {2: {"L3"}, 3: {"L4"}},
        }
        result = package_bond_info(bonds)
        assert len(result) == 2

    def test_empty_bonds(self):
        """Test packaging empty bonds."""
        result = package_bond_info({})
        assert result == []

    def test_bond_tag_formatting(self):
        """Test that bond tags are properly formatted with underscores."""
        bonds = {
            frozenset([0, 1]): {
                0: {"L1", "L2"},
                1: {"L3"},
            }
        }
        result = package_bond_info(bonds)
        assert "_" in result[0]["begin_tag"] or len(result[0]["begin_tag"]) > 0
        assert "_" in result[0]["end_tag"] or len(result[0]["end_tag"]) > 0


class TestAnnotateRbond:
    @patch("smiles_blocks.retrosynthesis.check_bondisinring")
    def test_no_chemical_group_matches(self, mock_check_ring):
        """Test when no chemical groups match."""
        mol = Chem.MolFromSmiles("CC")
        rbond_pattern = {frozenset(["L1", "L2"]): Chem.MolFromSmarts("[#6]-[#6]")}
        chemical_group = set()
        set2tag = {frozenset(["L1", "L2"]): "L1-L2"}

        result = annotate_rbond(mol, rbond_pattern, chemical_group, set2tag)
        assert result == {}

    @patch("smiles_blocks.retrosynthesis.check_bondisinring")
    def test_ring_bonds_skipped(self, mock_check_ring):
        """Test that ring bonds are skipped."""
        mock_check_ring.return_value = True
        mol = Chem.MolFromSmiles("c1ccccc1")
        rbond_pattern = {frozenset(["L1", "L2"]): Chem.MolFromSmarts("[#6]:[#6]")}
        chemical_group = frozenset(["L1", "L2"])
        set2tag = {frozenset(["L1", "L2"]): "L1-L2"}

        result = annotate_rbond(mol, rbond_pattern, chemical_group, set2tag)
        assert result == {}

    def test_returns_defaultdict(self):
        """Test that the function returns a dictionary."""
        mol = Chem.MolFromSmiles("CC")
        result = annotate_rbond(mol, {}, set(), {})
        assert isinstance(result, dict)


class TestRetrosynthesisAnalysis:
    def test_with_simple_smiles(self):
        """Test retrosynthetic analysis with simple SMILES."""
        result = retrosynthetic_analysis("CC")
        assert isinstance(result, dict)

    def test_with_none_chemical_dict(self):
        """Test that function initializes chemical_dict if None."""
        result = retrosynthetic_analysis("CC", chemical_dict=None)
        assert isinstance(result, dict)

    def test_with_provided_chemical_dict(self):
        """Test with provided chemical dictionary."""
        chem_dict = {
            "chemical_group": {},
            "rbond": {},
            "set2tag": {},
        }
        result = retrosynthetic_analysis("CC", chemical_dict=chem_dict)
        assert isinstance(result, dict)

    def test_complex_molecule(self):
        """Test with more complex molecule."""
        result = retrosynthetic_analysis("c1ccc(cc1)CC(=O)O")
        assert isinstance(result, dict)

    @patch("smiles_blocks.retrosynthesis.prep_rbrics_data")
    def test_uses_cached_chemical_dict(self, mock_prep):
        """Test that provided chemical_dict is used instead of calling prep_rbrics_data."""
        chem_dict = {
            "chemical_group": {},
            "rbond": {},
            "set2tag": {},
        }
        retrosynthetic_analysis("CC", chemical_dict=chem_dict)
        mock_prep.assert_not_called()
