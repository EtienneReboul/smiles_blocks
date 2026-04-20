import pandas as pd

from smiles_blocks.mcts import FragmentLibrary, MCTSDrugDesign
from smiles_blocks.rbrics_patterns import RBRICSCompatibilityMap


def _minimal_fragment_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["begin_tag", "end_tag", "unique_id", "can_smiles"])


def test_fragment_library_uses_default_rbrics_compatibility_map_when_none_provided():
    library = FragmentLibrary(_minimal_fragment_df(), compatibility_map=None)

    assert "L1" in library.compatibility_map
    assert library.compatibility_map["L1"] == RBRICSCompatibilityMap().patterns["L1"]


def test_mcts_drug_design_uses_default_rbrics_compatibility_map_when_none_provided():
    mcts = MCTSDrugDesign(_minimal_fragment_df(), compatibility_map=None, n_iter=1)

    assert "L17" in mcts.library.compatibility_map
    assert mcts.library.compatibility_map["L17"] == RBRICSCompatibilityMap().patterns["L17"]


def test_mcts_drug_design_preserves_custom_compatibility_map_override():
    custom_map = {"no_tag": {"no_tag"}}
    mcts = MCTSDrugDesign(_minimal_fragment_df(), compatibility_map=custom_map, n_iter=1)

    assert mcts.library.compatibility_map is custom_map
