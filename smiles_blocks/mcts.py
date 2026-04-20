"""
mcts_drug_design.py
===================
Monte Carlo Tree Search module for de novo drug design using R-BRICS fragments.

Fragment table schema expected
-------------------------------
block, can_smiles, first_connected_can_idx, last_connected_can_idx,
unique_id, begin_tag, end_tag, MolWt, nHDonors, nHAcceptors,
nRotatableBonds, CrippenlogP, TPSA, status

Usage
-----
>>> from mcts_drug_design import MCTSDrugDesign
>>> mcts = MCTSDrugDesign(fragment_table, compatibility_map, n_iter=1000)
>>> results = mcts.run(n_molecules=50)
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import logging
import math
import random
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Contrib.SA_Score import sascorer  # pyright: ignore[reportMissingImports]

from smiles_blocks.rbrics_patterns import RBRICSCompatibilityMap

logger = logging.getLogger(__name__)


# ============================================================
# 1.  SCORING FUNCTIONS
# ============================================================


def lipinski_score(mol: Chem.Mol) -> float:
    """
    Soft Lipinski filter → [0, 1].
    Each violated rule subtracts 0.25.  Full compliance = 1.0.
    """
    if mol is None:
        return 0.0
    violations = sum(
        [
            Descriptors.MolWt(mol) > 500,  # pyright: ignore[reportAttributeAccessIssue]
            Descriptors.MolLogP(mol) > 5,  # pyright: ignore[reportAttributeAccessIssue]
            rdMolDescriptors.CalcNumHBD(mol) > 5,
            rdMolDescriptors.CalcNumHBA(mol) > 10,
        ]
    )
    return max(0.0, 1.0 - violations * 0.25)


def sa_score_component(mol: Chem.Mol, threshold: float = 4.4) -> float:
    """
    Returns 1.0 if SA-score <= threshold, else 0.0.
    Silently skips if sascorer is not installed.
    """
    return 1.0 if sascorer.calculateScore(mol) <= threshold else 0.0


def composite_score(
    mol: Chem.Mol, sa_threshold: float = 4.4, lipinski_weight: float = 0.7, sa_weight: float = 0.3
) -> float:
    if mol is None:
        return 0.0
    return lipinski_weight * lipinski_score(mol) + sa_weight * sa_score_component(
        mol, threshold=sa_threshold
    )


def make_composite_scorer(
    lipinski_weight: float = 0.7, sa_weight: float = 0.3, sa_threshold: float = 4.4
):
    """
    Factory for the composite scorer.

    >>> scorer = make_composite_scorer()
    >>> mcts  = MCTSDrugDesign(df, compat_map, score_fn=scorer)
    """
    if abs(lipinski_weight + sa_weight - 1.0) > 1e-6:
        raise ValueError("Weights must sum to 1.0")

    def _score(mol):
        return composite_score(mol, sa_threshold, lipinski_weight, sa_weight)

    return _score


# ============================================================
# 2.  FRAGMENT LIBRARY
# ============================================================


class FragmentLibrary:
    """
    Thin wrapper around the fragment DataFrame with O(1) tag-filtered lookup.

    Parameters
    ----------
    df               : pd.DataFrame  – full fragment table
    compatibility_map: dict[str, set[str]]
        {end_tag: {begin_tags compatible with it}}
        Must include an entry for 'no_tag' if starters should connect.
    """

    def __init__(self, df: pd.DataFrame, compatibility_map: Optional[dict] = None):
        self.df = df.copy()
        self.compatibility_map = compatibility_map or dict(RBRICSCompatibilityMap().patterns)
        self._by_begin: dict[str, pd.DataFrame] = {
            str(tag): grp.reset_index(drop=True) for tag, grp in self.df.groupby("begin_tag")
        }

    def get_start_fragments(self) -> pd.DataFrame:
        """Fragments with begin_tag == 'no_tag'  (chain roots)."""
        return self._by_begin.get("no_tag", pd.DataFrame())

    def get_compatible_extensions(self, end_tag: str) -> pd.DataFrame:
        """All fragments whose begin_tag is compatible with *end_tag*."""
        compat = self.compatibility_map.get(end_tag, set())
        frames = [self._by_begin[t] for t in compat if t in self._by_begin]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_terminal_fragments(self, end_tag: str) -> pd.DataFrame:
        """Compatible extensions that also close the chain (end_tag == 'no_tag')."""
        cands = self.get_compatible_extensions(end_tag)
        if cands.empty:
            return cands
        return cands[cands["end_tag"] == "no_tag"].reset_index(drop=True)


# ============================================================
# 3.  MOLECULE STATE
# ============================================================


@dataclass
class MolState:
    fragment_ids: list = field(default_factory=list)
    smiles_parts: list = field(default_factory=list)
    current_end_tag: str = "no_tag"
    target_n_blocks: int = 1

    @property
    def n_blocks(self) -> int:
        return len(self.fragment_ids)

    @property
    def is_complete(self) -> bool:
        return (
            self.n_blocks >= 1
            and self.n_blocks == self.target_n_blocks
            and self.current_end_tag == "no_tag"
        )

    def assembled_smiles(self) -> str:
        # ---- Replace this with your ligation / attachment-point logic ----
        return ".".join(self.smiles_parts)

    def to_mol(self) -> Optional[Chem.Mol]:
        smi = self.assembled_smiles()
        return Chem.MolFromSmiles(smi) if smi else None

    def clone(self) -> "MolState":
        return deepcopy(self)


# ============================================================
# 4.  MCTS NODE
# ============================================================


class MCTSNode:
    __slots__ = ("state", "parent", "fragment_row", "children", "visits", "value", "_untried")

    def __init__(
        self,
        state: MolState,
        parent: Optional["MCTSNode"] = None,
        fragment_row: Optional[pd.Series] = None,
    ):
        self.state = state
        self.parent = parent
        self.fragment_row = fragment_row
        self.children: list["MCTSNode"] = []
        self.visits: int = 0
        self.value: float = 0.0
        self._untried: Optional[pd.DataFrame] = None  # lazy

    # -- UCB1 ----------------------------------------------------------
    def ucb1(self, c: float) -> float:
        if self.visits == 0:
            return float("inf")
        return self.value / self.visits + c * math.sqrt(math.log(self.parent.visits) / self.visits)  # pyright: ignore[reportOptionalMemberAccess]

    # -- State helpers -------------------------------------------------
    def is_terminal(self) -> bool:
        return self.state.is_complete

    def is_fully_expanded(self, lib: FragmentLibrary) -> bool:
        return self._pool(lib).empty

    def _pool(self, lib: FragmentLibrary) -> pd.DataFrame:
        """Compute (once) the set of untried fragments for this node."""
        if self._untried is not None:
            return self._untried
        s = self.state
        if s.n_blocks == 0:
            pool = lib.get_start_fragments()
        elif s.n_blocks == s.target_n_blocks - 1:
            pool = lib.get_terminal_fragments(s.current_end_tag)
        else:
            pool = lib.get_compatible_extensions(s.current_end_tag)
            pool = pool[pool["end_tag"] != "no_tag"].reset_index(drop=True)
        self._untried = pool
        return pool

    def pop_untried(self, lib: FragmentLibrary) -> Optional[pd.Series]:
        pool = self._pool(lib)
        if pool.empty:
            return None
        idx = random.randrange(len(pool))
        row = pool.iloc[idx]
        self._untried = pool.drop(pool.index[idx]).reset_index(drop=True)
        return row

    def best_child(self, c: float) -> "MCTSNode":
        return max(self.children, key=lambda n: n.ucb1(c))


# ============================================================
# 5.  BLOCK-COUNT SAMPLER
# ============================================================


class BlockCountSampler:
    """Sample chain length from a truncated Gaussian prior."""

    def __init__(self, mu: float, sigma: float, min_blocks: int = 2, max_blocks: int = 10):
        self.mu, self.sigma = mu, sigma
        self.min_blocks, self.max_blocks = min_blocks, max_blocks

    def sample(self) -> int:
        n = int(round(np.random.normal(self.mu, self.sigma)))
        return max(self.min_blocks, min(self.max_blocks, n))


# ============================================================
# 6.  MCTS ENGINE
# ============================================================


class MCTSDrugDesign:
    """
    MCTS-based de novo drug designer.

    Parameters
    ----------
    fragment_df       : pd.DataFrame   Fragment library.
    compatibility_map : dict | None    R-BRICS {end_tag: {begin_tags}}.
                                      Uses RBRICSCompatibilityMap defaults when None.
    block_count_mu    : float          Gaussian mean for chain length.
    block_count_sigma : float          Gaussian std-dev for chain length.
    min_blocks        : int            Hard min chain length (default 2).
    max_blocks        : int            Hard max chain length (default 10).
    n_iter            : int            MCTS iterations per molecule (default 500).
    ucb_c             : float          UCB1 exploration constant.
    score_fn          : callable       mol -> float in [0,1]. Default: lipinski_score.
    rollout_depth     : int            Max steps in simulation rollout (default 20).
    """

    def __init__(
        self,
        fragment_df: pd.DataFrame,
        compatibility_map: Optional[dict] = None,
        block_count_mu: float = 4.0,
        block_count_sigma: float = 1.5,
        min_blocks: int = 2,
        max_blocks: int = 10,
        n_iter: int = 500,
        ucb_c: float = math.sqrt(2),
        score_fn=None,
        rollout_depth: int = 20,
    ):
        self.library = FragmentLibrary(fragment_df, compatibility_map)
        self.sampler = BlockCountSampler(block_count_mu, block_count_sigma, min_blocks, max_blocks)
        self.n_iter = n_iter
        self.ucb_c = ucb_c
        self.score_fn = score_fn or lipinski_score
        self.rollout_depth = rollout_depth

    # ------------------------------------------------------------------ #
    #  PUBLIC API                                                          #
    # ------------------------------------------------------------------ #

    def run(self, n_molecules: int = 10) -> list[dict]:
        """
        Generate *n_molecules* candidates.

        Returns list of dicts with keys:
            smiles, score, n_blocks, fragment_ids
        """
        results = []
        for i in range(n_molecules):
            target = self.sampler.sample()
            root = MCTSNode(state=MolState(target_n_blocks=target))
            best = self._search(root)
            if best is not None:
                mol = best.state.to_mol()
                score = self.score_fn(mol) if mol else 0.0
                results.append(
                    {
                        "smiles": best.state.assembled_smiles(),
                        "score": score,
                        "n_blocks": best.state.n_blocks,
                        "fragment_ids": list(best.state.fragment_ids),
                    }
                )
                logger.debug(
                    "[%d/%d] score=%.3f blocks=%d smiles=%s",
                    i + 1,
                    n_molecules,
                    score,
                    best.state.n_blocks,
                    best.state.assembled_smiles(),
                )
        return results

    # ------------------------------------------------------------------ #
    #  MCTS LOOP                                                           #
    # ------------------------------------------------------------------ #

    def _search(self, root: MCTSNode) -> Optional[MCTSNode]:
        best_node: Optional[MCTSNode] = None
        best_score = -1.0

        for _ in range(self.n_iter):
            node = self._select(root)  # 1. Select
            if not node.is_terminal():
                node = self._expand(node)  # 2. Expand
            terminal_state = self._simulate(node.state.clone())  # 3. Simulate
            mol = terminal_state.to_mol()
            reward = self.score_fn(mol) if mol else 0.0  # 4. Score
            self._backpropagate(node, reward)  # 5. Backprop

            if terminal_state.is_complete and reward > best_score:
                best_score = reward
                best_node = (
                    node if node.is_terminal() else MCTSNode(state=terminal_state, parent=node)
                )

        return best_node

    # ------------------------------------------------------------------ #
    #  FOUR MCTS PHASES                                                    #
    # ------------------------------------------------------------------ #

    def _select(self, node: MCTSNode) -> MCTSNode:
        while not node.is_terminal():
            if not node.is_fully_expanded(self.library):
                return node
            if not node.children:
                return node
            node = node.best_child(self.ucb_c)
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        row = node.pop_untried(self.library)
        if row is None:
            return node
        new_state = node.state.clone()
        new_state.fragment_ids.append(row["unique_id"])
        new_state.smiles_parts.append(row["can_smiles"])
        new_state.current_end_tag = row["end_tag"]
        child = MCTSNode(state=new_state, parent=node, fragment_row=row)
        node.children.append(child)
        return child

    def _simulate(self, state: MolState) -> MolState:
        for _ in range(self.rollout_depth):
            if state.is_complete:
                break
            remaining = state.target_n_blocks - state.n_blocks
            if remaining <= 0:
                break
            if remaining == 1:
                cands = self.library.get_terminal_fragments(state.current_end_tag)
            else:
                cands = self.library.get_compatible_extensions(state.current_end_tag)
                cands = cands[cands["end_tag"] != "no_tag"].reset_index(drop=True)
            if cands.empty:  # fallback: close early
                cands = self.library.get_terminal_fragments(state.current_end_tag)
            if cands.empty:
                break
            row = cands.sample(1).iloc[0]
            state.fragment_ids.append(row["unique_id"])
            state.smiles_parts.append(row["can_smiles"])
            state.current_end_tag = row["end_tag"]
        return state

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        cur: Optional[MCTSNode] = node
        while cur is not None:
            cur.visits += 1
            cur.value += reward
            cur = cur.parent
