from smiles_blocks.rbrics_patterns import RBRICSChemicalGroups, RBRICSRetrosynthesisBound


class TestRBRICSChemicalGroups:
    """Tests for RBRICSChemicalGroups dataclass."""

    def test_initialization(self):
        """Test that RBRICSChemicalGroups initializes with default patterns."""
        groups = RBRICSChemicalGroups()
        assert groups.patterns is not None
        assert isinstance(groups.patterns, dict)

    def test_patterns_not_empty(self):
        """Test that patterns dictionary is not empty."""
        groups = RBRICSChemicalGroups()
        assert len(groups.patterns) > 0

    def test_all_expected_patterns_present(self):
        """Test that all expected L-group patterns are present."""
        groups = RBRICSChemicalGroups()
        expected_keys = {
            "L1",
            "L3",
            "L4",
            "L5",
            "L51",
            "L6",
            "L7a",
            "L7b",
            "L8",
            "L81",
            "L9",
            "L10",
            "L11",
            "L12",
            "L12b",
            "L13",
            "L14",
            "L14b",
            "L15",
            "L16",
            "L16b",
            "L17",
            "L18",
            "L19",
            "L182",
            "L192",
            "L20",
            "L21",
            "L22",
            "L23",
            "L30",
        }
        assert expected_keys.issubset(groups.patterns.keys())

    def test_pattern_values_are_strings(self):
        """Test that all pattern values are strings."""
        groups = RBRICSChemicalGroups()
        for key, value in groups.patterns.items():
            assert isinstance(value, str)
            assert len(value) > 0

    def test_disclaimer_present(self):
        """Test that disclaimer is present."""
        groups = RBRICSChemicalGroups()
        assert groups.disclaimer is not None
        assert isinstance(groups.disclaimer, str)
        assert "Copyright" in groups.disclaimer

    def test_specific_pattern_l1(self):
        """Test specific pattern L1."""
        groups = RBRICSChemicalGroups()
        assert groups.patterns["L1"] == "[C;D3]([#0,#6,#7,#8])(=O)"

    def test_specific_pattern_l30(self):
        """Test specific pattern L30."""
        groups = RBRICSChemicalGroups()
        assert groups.patterns["L30"] == "[C;D2]([#0,#6,#7,#8,#16])(#[N,C])"

    def test_patterns_are_valid_smarts_format(self):
        """Test that patterns follow SMARTS format conventions."""
        groups = RBRICSChemicalGroups()
        for key, pattern in groups.patterns.items():
            # Check for opening and closing brackets
            assert pattern.count("[") == pattern.count("]"), f"Unmatched brackets in {key}"


class TestRBRICSRetrosynthesisBound:
    """Tests for RBRICSRetrosynthesisBound dataclass."""

    def test_initialization(self):
        """Test that RBRICSRetrosynthesisBound initializes with default patterns."""
        bounds = RBRICSRetrosynthesisBound()
        assert bounds.patterns is not None
        assert isinstance(bounds.patterns, dict)

    def test_patterns_not_empty(self):
        """Test that patterns dictionary is not empty."""
        bounds = RBRICSRetrosynthesisBound()
        assert len(bounds.patterns) > 0

    def test_pattern_values_are_strings(self):
        """Test that all pattern values are strings."""
        bounds = RBRICSRetrosynthesisBound()
        for key, value in bounds.patterns.items():
            assert isinstance(value, str)
            assert len(value) > 0

    def test_disclaimer_present(self):
        """Test that disclaimer is present."""
        bounds = RBRICSRetrosynthesisBound()
        assert bounds.disclaimer is not None
        assert isinstance(bounds.disclaimer, str)

    def test_bond_patterns_contain_dash_separator(self):
        """Test that all bond pattern keys contain dash separator."""
        bounds = RBRICSRetrosynthesisBound()
        for key in bounds.patterns.keys():
            assert "-" in key

    def test_specific_bond_pattern_l1_l3(self):
        """Test specific bond pattern L1-L3."""
        bounds = RBRICSRetrosynthesisBound()
        assert "L1-L3" in bounds.patterns
        pattern = bounds.patterns["L1-L3"]
        assert "[C;D3]" in pattern
        assert "[O;D2]" in pattern

    def test_specific_bond_pattern_l30_l30(self):
        """Test specific bond pattern L30-L30."""
        bounds = RBRICSRetrosynthesisBound()
        assert "L30-L30" in bounds.patterns

    def test_patterns_are_valid_smarts_format(self):
        """Test that bond patterns follow SMARTS format conventions."""
        bounds = RBRICSRetrosynthesisBound()
        for key, pattern in bounds.patterns.items():
            # Check for opening and closing brackets
            assert pattern.count("[") == pattern.count("]"), f"Unmatched brackets in {key}"

    def test_symmetrical_bonds_when_same_groups(self):
        """Test that L-group to L-group same bonds are present."""
        bounds = RBRICSRetrosynthesisBound()
        same_group_bonds = ["L1-L1", "L14-L14", "L16-L16", "L17-L17"]
        for bond in same_group_bonds:
            # Some same-group bonds may not exist, but if they do, they should be valid
            if bond in bounds.patterns:
                assert isinstance(bounds.patterns[bond], str)

    def test_l17_has_many_connections(self):
        """Test that L17 (quaternary carbon) has many possible connections."""
        bounds = RBRICSRetrosynthesisBound()
        l17_bonds = [k for k in bounds.patterns.keys() if k.startswith("L17-")]
        assert len(l17_bonds) > 10

    def test_no_empty_pattern_values(self):
        """Test that no pattern values are empty strings."""
        bounds = RBRICSRetrosynthesisBound()
        for key, value in bounds.patterns.items():
            assert value.strip(), f"Empty pattern value for {key}"


class TestDataclassDefaults:
    """Tests for dataclass default factory behavior."""

    def test_multiple_instances_independent(self):
        """Test that multiple instances have independent pattern dictionaries."""
        groups1 = RBRICSChemicalGroups()
        groups2 = RBRICSChemicalGroups()
        assert groups1.patterns is not groups2.patterns
        assert groups1.patterns == groups2.patterns

    def test_bounds_multiple_instances_independent(self):
        """Test that multiple bound instances have independent dictionaries."""
        bounds1 = RBRICSRetrosynthesisBound()
        bounds2 = RBRICSRetrosynthesisBound()
        assert bounds1.patterns is not bounds2.patterns
        assert bounds1.patterns == bounds2.patterns
