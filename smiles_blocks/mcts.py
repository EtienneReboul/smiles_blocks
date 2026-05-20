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
>>> results = mcts.run(n_molecules=50, n_jobs=-1)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
import logging
import math
import os
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
    Soft Lipinski filter -> [0, 1].
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
    """Returns 1.0 if SA-score <= threshold, else 0.0."""
    return 1.0 if sascorer.calculateScore(mol) <= threshold else 0.0


def composite_score(
    mol: Chem.Mol,
    sa_threshold: float = 4.4,
    lipinski_weight: float = 0.7,
    sa_weight: float = 0.3,
) -> float:
    if mol is None:
        return 0.0
    return lipinski_weight * lipinski_score(mol) + sa_weight * sa_score_component(
        mol, threshold=sa_threshold
    )


class _CompositeScorer:
    """
    Named callable so it survives pickle under the spawn start-method
    (required on Windows and macOS with Python >= 3.8).
    """

    def __init__(self, lw: float, sw: float, sa_thresh: float):
        self.lw = lw
        self.sw = sw
        self.sa_thresh = sa_thresh

    def __call__(self, mol: Chem.Mol) -> float:
        return composite_score(mol, self.sa_thresh, self.lw, self.sw)


def make_composite_scorer(
    lipinski_weight: float = 0.7,
    sa_weight: float = 0.3,
    sa_threshold: float = 4.4,
) -> _CompositeScorer:
    """
    Factory for the composite scorer.

    Returns a picklable callable for use with parallel run().

    >>> scorer = make_composite_scorer()
    >>> mcts  = MCTSDrugDesign(df, score_fn=scorer)
    """
    if abs(lipinski_weight + sa_weight - 1.0) > 1e-6:
        raise ValueError("Weights must sum to 1.0")
    return _CompositeScorer(lipinski_weight, sa_weight, sa_threshold)


# ============================================================
# 2.  FRAGMENT LIBRARY
# ============================================================


class FragmentLibrary:
    """
    Thin wrapper around the fragment DataFrame with O(1) tag-filtered lookup.

    Parameters
    ----------
    df               : pd.DataFrame
        Full fragment table (see module docstring for schema).
    compatibility_map: dict[str, set[str]] | None
        {end_tag: {begin_tags compatible with it}}.
        Uses RBRICSCompatibilityMap defaults when None.
    """

    def __init__(self, df: pd.DataFrame, compatibility_map: Optional[dict] = None):
        self.df = df.copy()
        self.compatibility_map = compatibility_map or dict(RBRICSCompatibilityMap().patterns)
        self._empty = self.df.iloc[0:0].copy()
        self._by_begin: dict[str, pd.DataFrame] = {
            str(tag): grp.reset_index(drop=True) for tag, grp in self.df.groupby("begin_tag")
        }

    def get_start_fragments(self) -> pd.DataFrame:
        """Fragments with begin_tag == 'no_tag' (chain roots)."""
        return self._by_begin.get("no_tag", self._empty)

    def get_compatible_extensions(self, end_tag: str) -> pd.DataFrame:
        """All fragments whose begin_tag is compatible with *end_tag*."""
        compat = self.compatibility_map.get(end_tag, set())
        frames = [self._by_begin[t] for t in compat if t in self._by_begin]
        return pd.concat(frames, ignore_index=True) if frames else self._empty

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
        # ---- Replace with your ligation / attachment-point logic ----
        return "".join(self.smiles_parts)

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
        return (
            self.value / self.visits + c * math.sqrt(math.log(self.parent.visits) / self.visits)  # pyright: ignore[reportOptionalMemberAccess]
        )

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
# 6.  MODULE-LEVEL MCTS PHASE FUNCTIONS
#     Stateless free functions so they are picklable and can be
#     called from both the parallel worker and the serial path.
# ============================================================


def _select(node: MCTSNode, lib: FragmentLibrary, c: float) -> MCTSNode:
    while not node.is_terminal():
        if not node.is_fully_expanded(lib):
            return node
        if not node.children:
            return node
        node = node.best_child(c)
    return node


def _expand(node: MCTSNode, lib: FragmentLibrary) -> MCTSNode:
    row = node.pop_untried(lib)
    if row is None:
        return node
    new_state = node.state.clone()
    new_state.fragment_ids.append(row["unique_id"])
    new_state.smiles_parts.append(row["block"])
    new_state.current_end_tag = row["end_tag"]
    child = MCTSNode(state=new_state, parent=node, fragment_row=row)
    node.children.append(child)
    return child


def _simulate(state: MolState, lib: FragmentLibrary, rollout_depth: int) -> MolState:
    for _ in range(rollout_depth):
        if state.is_complete:
            break
        remaining = state.target_n_blocks - state.n_blocks
        if remaining <= 0:
            break
        if remaining == 1:
            cands = lib.get_terminal_fragments(state.current_end_tag)
        else:
            cands = lib.get_compatible_extensions(state.current_end_tag)
            cands = cands[cands["end_tag"] != "no_tag"].reset_index(drop=True)
        if cands.empty:  # fallback: close early
            cands = lib.get_terminal_fragments(state.current_end_tag)
        if cands.empty:
            break
        row = cands.sample(1).iloc[0]
        state.fragment_ids.append(row["unique_id"])
        state.smiles_parts.append(row["block"])
        state.current_end_tag = row["end_tag"]
    return state


def _backpropagate(node: MCTSNode, reward: float) -> None:
    cur: Optional[MCTSNode] = node
    while cur is not None:
        cur.visits += 1
        cur.value += reward
        cur = cur.parent


# ============================================================
# 7.  TOP-LEVEL WORKER  (module-level for pickling)
# ============================================================


def _search_one(args: tuple) -> Optional[dict]:
    """
    Picklable worker executing one full MCTS search in a child process.

    args = (records, compat_map, target_n_blocks,
            n_iter, ucb_c, score_fn, rollout_depth)
    """
    records, compat_map, target_n_blocks, n_iter, ucb_c, score_fn, rollout_depth = args

    df = pd.DataFrame(records)
    library = FragmentLibrary(df, compat_map)
    root = MCTSNode(state=MolState(target_n_blocks=target_n_blocks))

    best_node: Optional[MCTSNode] = None
    best_score = -1.0

    for _ in range(n_iter):
        node = _select(root, library, ucb_c)
        if not node.is_terminal():
            node = _expand(node, library)
        terminal_state = _simulate(node.state.clone(), library, rollout_depth)
        mol = terminal_state.to_mol()
        reward = score_fn(mol) if mol else 0.0
        _backpropagate(node, reward)

        if terminal_state.is_complete and reward > best_score:
            best_score = reward
            best_node = node if node.is_terminal() else MCTSNode(state=terminal_state, parent=node)

    if best_node is None:
        return None

    mol = best_node.state.to_mol()
    score = score_fn(mol) if mol else 0.0
    return {
        "smiles": best_node.state.assembled_smiles(),
        "score": score,
        "n_blocks": best_node.state.n_blocks,
        "fragment_ids": list(best_node.state.fragment_ids),
    }


# ============================================================
# 8.  MCTS ENGINE
# ============================================================


class MCTSDrugDesign:
    """
    MCTS-based de novo drug designer with root parallelisation.

    Each molecule search is independent, so n_molecules searches are
    dispatched across n_jobs worker processes (one search per process
    slot) with zero synchronisation overhead.

    Parameters
    ----------
    fragment_df       : pd.DataFrame
        Fragment library (see module docstring for column schema).
    compatibility_map : dict | None
        R-BRICS {end_tag: {begin_tags}}.
        Uses RBRICSCompatibilityMap defaults when None.
    block_count_mu    : float   Gaussian mean for chain length.
    block_count_sigma : float   Gaussian std-dev for chain length.
    min_blocks        : int     Hard min chain length (default 2).
    max_blocks        : int     Hard max chain length (default 10).
    n_iter            : int     MCTS iterations per molecule (default 500).
    ucb_c             : float   UCB1 exploration constant (default sqrt(2)).
    score_fn          : callable
        mol -> float in [0, 1].  Default: lipinski_score.
        Must be picklable when n_jobs > 1; use make_composite_scorer()
        rather than a plain lambda.
    rollout_depth     : int     Max steps in simulation rollout (default 20).
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

    def run(self, n_molecules: int = 10, n_jobs: int = -1) -> list[dict]:
        """
        Generate *n_molecules* candidate drug-like molecules.

        Parameters
        ----------
        n_molecules : int
            Number of molecules to generate.
        n_jobs : int
            Worker processes to use.
            -1  -> os.cpu_count()   (default)
             1  -> single-process, no fork (recommended for debugging
                   and Jupyter notebooks)
            >1  -> exactly that many workers

        Returns
        -------
        list[dict] with keys: smiles, score, n_blocks, fragment_ids

        Notes
        -----
        Always protect parallel calls with the standard guard when
        running as a script::

            if __name__ == "__main__":
                results = mcts.run(100, n_jobs=8)
        """
        n_workers = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)

        # Serialise DataFrame once as records (compact, fast pickle)
        records = self.library.df.to_dict("records")
        compat = self.library.compatibility_map

        # Sample all target block-counts in the main process so that
        # the Gaussian RNG state is not duplicated across workers.
        targets = [self.sampler.sample() for _ in range(n_molecules)]

        args_list = [
            (
                records,
                compat,
                t,
                self.n_iter,
                self.ucb_c,
                self.score_fn,
                self.rollout_depth,
            )
            for t in targets
        ]

        results: list[dict] = []

        if n_workers == 1:
            # Serial path — no fork overhead, easy to profile / debug
            for args in args_list:
                res = _search_one(args)
                if res is not None:
                    results.append(res)
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_search_one, a): i for i, a in enumerate(args_list)}
                for fut in as_completed(futures):
                    res = fut.result()
                    if res is not None:
                        results.append(res)

        logger.info("Generated %d / %d molecules.", len(results), n_molecules)
        return results

    # ------------------------------------------------------------------ #
    #  SERIAL _search (delegates to module-level phase functions)          #
    # ------------------------------------------------------------------ #

    def _search(self, root: MCTSNode) -> Optional[MCTSNode]:
        """Single-tree search used by the serial path inside run()."""
        best_node: Optional[MCTSNode] = None
        best_score = -1.0

        for _ in range(self.n_iter):
            node = _select(root, self.library, self.ucb_c)
            if not node.is_terminal():
                node = _expand(node, self.library)
            terminal_state = _simulate(node.state.clone(), self.library, self.rollout_depth)
            mol = terminal_state.to_mol()
            reward = self.score_fn(mol) if mol else 0.0
            _backpropagate(node, reward)

            if terminal_state.is_complete and reward > best_score:
                best_score = reward
                best_node = (
                    node if node.is_terminal() else MCTSNode(state=terminal_state, parent=node)
                )

        return best_node
