"""
mcts_drug_design.py
===================
Monte Carlo Tree Search module for de novo drug design using R-BRICS fragments.

Fragment table schema expected
-------------------------------
block, can_smiles, first_connected_can_idx, last_connected_can_idx,
unique_id, begin_tag, end_tag, MolWt, nHDonors, nHAcceptors,
nRotatableBonds, CrippenlogP, TPSA, frequency, status

Usage
-----
>>> from mcts_drug_design import MCTSDrugDesign
>>> mcts = MCTSDrugDesign(fragment_table, compatibility_map, n_iter=1000)
>>> results = mcts.run(n_runs=50, n_jobs=1, temperature=1.0, score_threshold=0.75)
>>> # Timing report is printed automatically at the end of every run().
>>> # Access raw counters via mcts.profiler.report()
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
import logging
import math
from multiprocessing.shared_memory import SharedMemory
import os
import random
import signal
import time
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Contrib.SA_Score import sascorer  # pyright: ignore[reportMissingImports]

from smiles_blocks.rbrics_patterns import RBRICSCompatibilityMap

logger = logging.getLogger(__name__)


# ============================================================
# 0.  PROFILER
# ============================================================


class MCTSProfiler:
    """
    Lightweight wall-clock profiler for the four MCTS phases plus
    sub-operations within each phase.

    All timings are accumulated in seconds.  Call report() to get a
    formatted summary, or access .totals and .counts directly.

    Tracked operations
    ------------------
    select        : tree traversal from root to leaf (UCB/PUCT evaluation)
    expand        : node expansion — pop_untried + child creation + prior update
    simulate      : rollout — fragment sampling + tag filtering
    score         : mol construction + reward function evaluation
    backprop      : value propagation back to root
    library_query : time inside FragmentLibrary get_* methods (subset of simulate)
    prior_compute : time computing/updating PUCT prior vectors (subset of expand)
    """

    OPERATIONS = (
        "select",
        "expand",
        "simulate",
        "score",
        "backprop",
        "library_query",
        "prior_compute",
    )

    def __init__(self):
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self._starts: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Context-manager style  (preferred in hot loops)
    # ------------------------------------------------------------------

    def start(self, op: str) -> None:
        self._starts[op] = time.perf_counter()

    def stop(self, op: str) -> None:
        elapsed = time.perf_counter() - self._starts[op]
        self.totals[op] += elapsed
        self.counts[op] += 1

    # ------------------------------------------------------------------
    # Merge results from worker processes
    # ------------------------------------------------------------------

    def merge(self, other: "MCTSProfiler") -> None:
        """Accumulate *other* into self (used after parallel workers return)."""
        for op in self.OPERATIONS:
            self.totals[op] += other.totals[op]
            self.counts[op] += other.counts[op]

    def merge_dict(self, d: dict) -> None:
        """Merge a plain dict representation (returned from worker processes)."""
        for op in self.OPERATIONS:
            self.totals[op] += d["totals"].get(op, 0.0)
            self.counts[op] += d["counts"].get(op, 0)

    def to_dict(self) -> dict:
        return {
            "totals": dict(self.totals),
            "counts": dict(self.counts),
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> str:
        """
        Return a formatted timing table.

        Example output
        --------------
        ┌─────────────────┬──────────┬─────────┬──────────────┬──────────┐
        │ Operation       │ Total(s) │  Calls  │  Mean(ms)    │  Share % │
        ├─────────────────┼──────────┼─────────┼──────────────┼──────────┤
        │ score           │   12.34  │   5000  │    2.468     │   61.2 % │
        │ simulate        │    4.56  │   5000  │    0.912     │   22.6 % │
        │ select          │    1.23  │   5000  │    0.246     │    6.1 % │
        │ ...             │          │         │              │          │
        └─────────────────┴──────────┴─────────┴──────────────┴──────────┘
        """
        grand_total = sum(self.totals.values()) or 1.0
        rows = []
        for op in self.OPERATIONS:
            t = self.totals.get(op, 0.0)
            n = self.counts.get(op, 0)
            mean_ms = (t / n * 1000) if n > 0 else 0.0
            share = t / grand_total * 100
            rows.append((op, t, n, mean_ms, share))

        rows.sort(key=lambda r: r[1], reverse=True)

        header = (
            f"{'Operation':<17} {'Total(s)':>8}  {'Calls':>7}  {'Mean(ms)':>10}  {'Share%':>8}"
        )
        sep = "-" * len(header)
        lines = [sep, header, sep]
        for op, t, n, mean_ms, share in rows:
            lines.append(f"{op:<17} {t:>8.3f}  {n:>7d}  {mean_ms:>10.3f}  {share:>7.1f}%")
        lines.append(sep)
        lines.append(f"{'TOTAL':<17} {grand_total:>8.3f}")
        lines.append(sep)
        return "\n".join(lines)

    def print_report(self) -> None:
        print("\n=== MCTS Timing Profile ===")
        print(self.report())


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
    >>> mcts  = MCTSDrugDesign(table, score_fn=scorer)
    """
    if abs(lipinski_weight + sa_weight - 1.0) > 1e-6:
        raise ValueError("Weights must sum to 1.0")
    return _CompositeScorer(lipinski_weight, sa_weight, sa_threshold)


# ============================================================
# 2.  FRAGMENT LIBRARY  (PyArrow-backed, shared-memory aware)
# ============================================================


class FragmentLibrary:
    """
    PyArrow-backed fragment library with optional shared-memory transport.

    The full table lives once in memory (or in a shared-memory block).
    The _by_begin index contains zero-copy filtered views — no data is
    duplicated.  All hot-path queries use PyArrow compute (SIMD C++).

    Parameters
    ----------
    table            : pa.Table
        Full fragment table (see module docstring for column schema).
        Must contain a 'frequency' column with corpus occurrence counts
        (Option B: total occurrences across all molecules, capped per
        molecule to avoid outlier inflation).
    compatibility_map: dict[str, set[str]] | None
        {end_tag: {begin_tags compatible with it}}.
        Uses RBRICSCompatibilityMap defaults when None.
    """

    _ROW_COLS = ("unique_id", "block", "end_tag", "frequency")

    def __init__(
        self,
        table: pa.Table,
        compatibility_map: Optional[dict] = None,
        _shm_handle: Optional[SharedMemory] = None,
    ):
        self.table = table
        self.compatibility_map = compatibility_map or dict(RBRICSCompatibilityMap().patterns)
        self._shm_handle = _shm_handle
        self._empty = table.slice(0, 0)

        begin_col = table.column("begin_tag")
        self._by_begin: dict[str, pa.Table] = {}
        for tag in begin_col.unique().to_pylist():
            mask = pc.equal(begin_col, tag)  # pyright: ignore[reportAttributeAccessIssue]
            self._by_begin[str(tag)] = pc.filter(table, mask)  # pyright: ignore[reportAttributeAccessIssue]

    # ------------------------------------------------------------------
    # Tag-filtered queries
    # ------------------------------------------------------------------

    def get_start_fragments(self) -> pa.Table:
        return self._by_begin.get("no_tag", self._empty)

    def get_compatible_extensions(self, end_tag: str) -> pa.Table:
        compat = self.compatibility_map.get(end_tag, set())
        slices = [self._by_begin[t] for t in compat if t in self._by_begin]
        if not slices:
            return self._empty
        return pa.concat_tables(slices)

    def get_terminal_fragments(self, end_tag: str) -> pa.Table:
        cands = self.get_compatible_extensions(end_tag)
        if cands.num_rows == 0:
            return cands
        mask = pc.equal(cands.column("end_tag"), "no_tag")  # pyright: ignore[reportAttributeAccessIssue]
        return pc.filter(cands, mask)  # pyright: ignore[reportAttributeAccessIssue]

    # ------------------------------------------------------------------
    # Row sampling
    # ------------------------------------------------------------------

    def sample_row(self, table: pa.Table) -> Optional[dict]:
        if table.num_rows == 0:
            return None
        idx = random.randrange(table.num_rows)
        return {col: table.column(col)[idx].as_py() for col in self._ROW_COLS}

    def weighted_sample_row(self, table: pa.Table, temperature: float = 1.0) -> Optional[dict]:
        if table.num_rows == 0:
            return None
        if table.num_rows == 1:
            return {col: table.column(col)[0].as_py() for col in self._ROW_COLS}
        freqs = np.array(table.column("frequency").to_pylist(), dtype=np.float32)
        log_f = np.log1p(freqs) / temperature
        log_f -= log_f.max()
        weights = np.exp(log_f)
        weights /= weights.sum()
        idx = int(np.random.choice(len(weights), p=weights))
        return {col: table.column(col)[idx].as_py() for col in self._ROW_COLS}

    # ------------------------------------------------------------------
    # PUCT prior computation
    # ------------------------------------------------------------------

    def compute_prior(self, table: pa.Table, temperature: float = 1.0) -> np.ndarray:
        freqs = np.array(table.column("frequency").to_pylist(), dtype=np.float32)
        log_f = np.log1p(freqs) / temperature
        log_f -= log_f.max()
        w = np.exp(log_f)
        return w / w.sum()

    # ------------------------------------------------------------------
    # Shared-memory transport
    # ------------------------------------------------------------------

    @staticmethod
    def to_shared_memory(table: pa.Table) -> tuple[SharedMemory, int]:
        sink = pa.BufferOutputStream()
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
        buf: pa.Buffer = sink.getvalue()
        n = len(buf)
        shm = SharedMemory(create=True, size=n)
        shm.buf[:n] = buf.to_pybytes()  # pyright: ignore[reportOptionalSubscript]
        return shm, n

    @staticmethod
    def from_shared_memory(
        shm_name: str,
        n_bytes: int,
        compatibility_map: Optional[dict] = None,
    ) -> "FragmentLibrary":
        shm = SharedMemory(name=shm_name, create=False)
        buf = pa.py_buffer(bytes(shm.buf[:n_bytes]))  # pyright: ignore[reportOptionalSubscript]
        reader = ipc.open_file(buf)
        table = reader.read_all()
        lib = FragmentLibrary(table, compatibility_map, _shm_handle=shm)
        return lib


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
# 4.  MCTS NODE  (PUCT-enabled)
# ============================================================


class MCTSNode:
    __slots__ = (
        "state",
        "parent",
        "fragment_row",
        "children",
        "visits",
        "value",
        "_untried",
        "_untried_prior",
        "depth",
        "_prior",
    )

    def __init__(
        self,
        state: MolState,
        parent: Optional["MCTSNode"] = None,
        fragment_row: Optional[dict] = None,
        depth: int = 0,
    ):
        self.state = state
        self.parent = parent
        self.fragment_row = fragment_row
        self.children: list["MCTSNode"] = []
        self.visits: int = 0
        self.value: float = 0.0
        self._untried: Optional[pa.Table] = None
        self._untried_prior: Optional[np.ndarray] = None
        self.depth: int = depth
        self._prior: Optional[np.ndarray] = None

    def puct(self, c: float, child_idx: int) -> float:
        child = self.children[child_idx]
        q = child.value / child.visits if child.visits > 0 else 0.0
        if self._prior is not None and child_idx < len(self._prior):
            prior = float(self._prior[child_idx])
        else:
            prior = 1.0 / max(len(self.children), 1)
        exploration = c * prior * math.sqrt(max(self.visits, 1)) / (1 + child.visits)
        return q + exploration

    def best_child(self, c: float) -> "MCTSNode":
        return self.children[max(range(len(self.children)), key=lambda i: self.puct(c, i))]

    def is_terminal(self) -> bool:
        return self.state.is_complete

    def is_fully_expanded(self, lib: FragmentLibrary) -> bool:
        return self._pool(lib).num_rows == 0

    def _pool(self, lib: FragmentLibrary) -> pa.Table:
        if self._untried is not None:
            return self._untried
        s = self.state
        if s.n_blocks == 0:
            pool = lib.get_start_fragments()
        elif s.n_blocks == s.target_n_blocks - 1:
            pool = lib.get_terminal_fragments(s.current_end_tag)
        else:
            pool = lib.get_compatible_extensions(s.current_end_tag)
            if pool.num_rows > 0:
                mask = pc.not_equal(pool.column("end_tag"), "no_tag")  # pyright: ignore[reportAttributeAccessIssue]
                pool = pc.filter(pool, mask)  # pyright: ignore[reportAttributeAccessIssue]
        self._untried = pool
        return pool

    def pop_untried(self, lib: FragmentLibrary, temperature: float = 1.0) -> Optional[dict]:
        pool = self._pool(lib)
        if pool.num_rows == 0:
            return None

        if self._untried_prior is None:
            self._untried_prior = lib.compute_prior(pool, temperature)

        if pool.num_rows == 1:
            idx = 0
        else:
            w = self._untried_prior.copy()
            w /= w.sum()
            idx = int(np.random.choice(len(w), p=w))

        row = {col: pool.column(col)[idx].as_py() for col in lib._ROW_COLS}

        self._untried = (
            pa.concat_tables([pool.slice(0, idx), pool.slice(idx + 1)])
            if pool.num_rows > 1
            else pool.slice(0, 0)
        )
        keep_mask = np.ones(len(self._untried_prior), dtype=bool)
        keep_mask[idx] = False
        self._untried_prior = self._untried_prior[keep_mask]

        return row

    def add_dirichlet_noise(self, alpha: float = 0.3, epsilon: float = 0.25) -> None:
        if self._prior is None or len(self.children) == 0:
            return
        noise = np.random.dirichlet([alpha] * len(self.children))
        self._prior = (1.0 - epsilon) * self._prior + epsilon * noise


# ============================================================
# 5.  BLOCK-COUNT SAMPLER
# ============================================================


class BlockCountSampler:
    """Sample chain length from a truncated Gaussian prior."""

    def __init__(
        self,
        mu: float,
        sigma: float,
        min_blocks: int = 2,
        max_blocks: int = 10,
    ):
        self.mu, self.sigma = mu, sigma
        self.min_blocks, self.max_blocks = min_blocks, max_blocks

    def sample(self) -> int:
        n = int(round(np.random.normal(self.mu, self.sigma)))
        return max(self.min_blocks, min(self.max_blocks, n))


# ============================================================
# 6.  MODULE-LEVEL MCTS PHASE FUNCTIONS  (profiler-instrumented)
#     Each function accepts an optional MCTSProfiler and records
#     fine-grained timings into it.
# ============================================================


def _select(
    node: MCTSNode,
    lib: FragmentLibrary,
    c: float,
    profiler: Optional[MCTSProfiler] = None,
) -> MCTSNode:
    if profiler:
        profiler.start("select")
    while not node.is_terminal():
        if not node.is_fully_expanded(lib):
            break
        if not node.children:
            break
        node = node.best_child(c)
    if profiler:
        profiler.stop("select")
    return node


def _expand(
    node: MCTSNode,
    lib: FragmentLibrary,
    temperature: float = 1.0,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    use_dirichlet: bool = True,
    profiler: Optional[MCTSProfiler] = None,
) -> MCTSNode:
    if profiler:
        profiler.start("expand")

    row = node.pop_untried(lib, temperature)
    if row is None:
        if profiler:
            profiler.stop("expand")
        return node

    new_state = node.state.clone()
    new_state.fragment_ids.append(row["unique_id"])
    new_state.smiles_parts.append(row["block"])
    new_state.current_end_tag = row["end_tag"]

    child = MCTSNode(
        state=new_state,
        parent=node,
        fragment_row=row,
        depth=node.depth + 1,
    )
    node.children.append(child)

    # Rebuild PUCT prior over expanded children
    if profiler:
        profiler.start("prior_compute")
    freqs = np.array(
        [
            float(c.fragment_row["frequency"]) if c.fragment_row is not None else 1.0
            for c in node.children
        ],
        dtype=np.float32,
    )
    log_f = np.log1p(freqs) / temperature
    log_f -= log_f.max()
    w = np.exp(log_f)
    node._prior = w / w.sum()
    if profiler:
        profiler.stop("prior_compute")

    if use_dirichlet and node.parent is None:
        node.add_dirichlet_noise(alpha=dirichlet_alpha, epsilon=dirichlet_eps)

    if profiler:
        profiler.stop("expand")
    return child


def _simulate(
    state: MolState,
    lib: FragmentLibrary,
    rollout_depth: int,
    temperature: float = 1.0,
    profiler: Optional[MCTSProfiler] = None,
) -> MolState:
    if profiler:
        profiler.start("simulate")

    for _ in range(rollout_depth):
        if state.is_complete:
            break
        remaining = state.target_n_blocks - state.n_blocks
        if remaining <= 0:
            break

        if profiler:
            profiler.start("library_query")
        if remaining == 1:
            cands = lib.get_terminal_fragments(state.current_end_tag)
        else:
            cands = lib.get_compatible_extensions(state.current_end_tag)
            if cands.num_rows > 0:
                mask = pc.not_equal(cands.column("end_tag"), "no_tag")  # pyright: ignore[reportAttributeAccessIssue]
                cands = pc.filter(cands, mask)  # pyright: ignore[reportAttributeAccessIssue]
        if cands.num_rows == 0:
            cands = lib.get_terminal_fragments(state.current_end_tag)
        if profiler:
            profiler.stop("library_query")

        if cands.num_rows == 0:
            break

        row = lib.weighted_sample_row(cands, temperature)
        if row is None:
            break
        state.fragment_ids.append(row["unique_id"])
        state.smiles_parts.append(row["block"])
        state.current_end_tag = row["end_tag"]

    if profiler:
        profiler.stop("simulate")
    return state


def _score_state(
    state: MolState,
    score_fn,
    profiler: Optional[MCTSProfiler] = None,
) -> float:
    """
    Isolated scoring step — mol construction + reward function.
    Kept as a separate function so its timing is unambiguously captured
    and so it can later be parallelised independently (leaf parallelisation).
    """
    if profiler:
        profiler.start("score")
    mol = state.to_mol()
    reward = score_fn(mol) if mol else 0.0
    if profiler:
        profiler.stop("score")
    return reward


def _backpropagate(
    node: MCTSNode,
    reward: float,
    profiler: Optional[MCTSProfiler] = None,
) -> None:
    if profiler:
        profiler.start("backprop")
    cur: Optional[MCTSNode] = node
    while cur is not None:
        cur.visits += 1
        cur.value += reward
        cur = cur.parent
    if profiler:
        profiler.stop("backprop")


# ============================================================
# 7.  TOP-LEVEL WORKER  (module-level for pickling)
# ============================================================


def _search_one(args: tuple) -> tuple[list[dict], dict]:
    """
    Picklable worker executing one full MCTS search in a child process.

    Returns
    -------
    (results, profiler_dict)
        results       : list of molecule dicts above score_threshold
        profiler_dict : serialised MCTSProfiler for merging in main process
    """
    (
        shm_name,
        shm_bytes,
        compat_map,
        target_n_blocks,
        n_iter,
        ucb_c,
        score_fn,
        rollout_depth,
        temperature,
        dirichlet_alpha,
        dirichlet_eps,
        use_dirichlet,
        score_threshold,
    ) = args

    library = FragmentLibrary.from_shared_memory(shm_name, shm_bytes, compat_map)
    root = MCTSNode(state=MolState(target_n_blocks=target_n_blocks))
    profiler = MCTSProfiler()

    collected: list[dict] = []
    seen_smiles: set[str] = set()

    for _ in range(n_iter):
        node = _select(root, library, ucb_c, profiler)
        if not node.is_terminal():
            node = _expand(
                node,
                library,
                temperature,
                dirichlet_alpha,
                dirichlet_eps,
                use_dirichlet,
                profiler,
            )
        terminal_state = _simulate(
            node.state.clone(), library, rollout_depth, temperature, profiler
        )
        reward = _score_state(terminal_state, score_fn, profiler)
        _backpropagate(node, reward, profiler)

        if terminal_state.is_complete and reward >= score_threshold:
            smi = terminal_state.assembled_smiles()
            if smi not in seen_smiles:
                seen_smiles.add(smi)
                collected.append(
                    {
                        "smiles": smi,
                        "score": reward,
                        "n_blocks": terminal_state.n_blocks,
                        "fragment_ids": list(terminal_state.fragment_ids),
                    }
                )

    return collected, profiler.to_dict()


# ============================================================
# 8.  MCTS ENGINE
# ============================================================


class MCTSDrugDesign:
    """
    MCTS-based de novo drug designer with built-in timing profiler.

    After every call to run(), a timing report is printed to stdout and
    the merged profiler is accessible via self.profiler.

    Parameters
    ----------
    fragment_table    : pa.Table
        Fragment library (see module docstring for column schema).
        Must include a 'frequency' column.
    compatibility_map : dict | None
        R-BRICS {end_tag: {begin_tags}}.
        Uses RBRICSCompatibilityMap defaults when None.
    block_count_mu    : float   Gaussian mean for chain length.
    block_count_sigma : float   Gaussian std-dev for chain length.
    min_blocks        : int     Hard min chain length (default 2).
    max_blocks        : int     Hard max chain length (default 10).
    n_iter            : int     MCTS iterations per run (default 500).
    ucb_c             : float   PUCT exploration constant (default sqrt(2)).
    score_fn          : callable
        mol -> float in [0, 1].  Default: lipinski_score.
        Must be picklable; use make_composite_scorer() rather than a lambda.
    rollout_depth     : int     Max steps in simulation rollout (default 20).
    """

    def __init__(
        self,
        fragment_table: pa.Table,
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
        self.library = FragmentLibrary(fragment_table, compatibility_map)
        self.sampler = BlockCountSampler(block_count_mu, block_count_sigma, min_blocks, max_blocks)
        self.n_iter = n_iter
        self.ucb_c = ucb_c
        self.score_fn = score_fn or lipinski_score
        self.rollout_depth = rollout_depth
        self.profiler = MCTSProfiler()  # reset on each run()

    # ------------------------------------------------------------------ #
    #  PUBLIC API                                                          #
    # ------------------------------------------------------------------ #

    def run(
        self,
        n_runs: int = 10,
        n_jobs: int = -1,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        use_dirichlet: bool = True,
        score_threshold: float = 0.5,
        print_profile: bool = True,
    ) -> list[dict]:
        """
        Run *n_runs* independent MCTS searches and collect all molecules
        whose score is at or above *score_threshold*.

        Parameters
        ----------
        n_runs          : int   Number of independent MCTS searches.
        n_jobs          : int   Worker processes (-1 = os.cpu_count(), 1 = serial).
        temperature     : float Log-frequency sharpness (1.0 = neutral).
        dirichlet_alpha : float Dirichlet concentration at root (default 0.3).
        dirichlet_eps   : float Dirichlet mixing weight (default 0.25).
        use_dirichlet   : bool  Toggle root Dirichlet noise (default True).
        score_threshold : float Minimum score to collect a molecule (default 0.5).
        print_profile   : bool  Print timing report after run (default True).

        Returns
        -------
        list[dict] sorted by score descending, each with keys:
            smiles, score, n_blocks, fragment_ids
        """
        self.profiler = MCTSProfiler()  # fresh profiler for this run
        n_workers = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
        targets = [self.sampler.sample() for _ in range(n_runs)]
        results: list[dict] = []

        if n_workers == 1:
            # Serial path — profiler lives in this process
            for target in targets:
                root = MCTSNode(state=MolState(target_n_blocks=target))
                batch = self._search(
                    root,
                    temperature,
                    dirichlet_alpha,
                    dirichlet_eps,
                    use_dirichlet,
                    score_threshold,
                )
                results.extend(batch)
        else:
            # Parallel path
            shm, shm_bytes = FragmentLibrary.to_shared_memory(self.library.table)
            _cleanup_shm = shm

            def _handle_signal(sig, frame):
                _cleanup_shm.close()
                _cleanup_shm.unlink()
                raise SystemExit(1)

            signal.signal(signal.SIGTERM, _handle_signal)

            args_list = [
                (
                    shm.name,
                    shm_bytes,
                    self.library.compatibility_map,
                    t,
                    self.n_iter,
                    self.ucb_c,
                    self.score_fn,
                    self.rollout_depth,
                    temperature,
                    dirichlet_alpha,
                    dirichlet_eps,
                    use_dirichlet,
                    score_threshold,
                )
                for t in targets
            ]

            try:
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futures = {pool.submit(_search_one, a): i for i, a in enumerate(args_list)}
                    for fut in as_completed(futures):
                        batch, prof_dict = fut.result()
                        results.extend(batch)
                        # Merge worker profiler into main profiler
                        self.profiler.merge_dict(prof_dict)
            finally:
                shm.close()
                shm.unlink()

        # Global deduplication + sort
        seen: set[str] = set()
        unique_results: list[dict] = []
        for r in results:
            if r["smiles"] not in seen:
                seen.add(r["smiles"])
                unique_results.append(r)
        unique_results.sort(key=lambda r: r["score"], reverse=True)

        logger.info(
            "%d unique molecules collected above threshold %.2f from %d runs.",
            len(unique_results),
            score_threshold,
            n_runs,
        )

        if print_profile:
            self.profiler.print_report()

        return unique_results

    # ------------------------------------------------------------------ #
    #  SERIAL _search  (delegates to module-level phase functions)         #
    # ------------------------------------------------------------------ #

    def _search(
        self,
        root: MCTSNode,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        use_dirichlet: bool = True,
        score_threshold: float = 0.5,
    ) -> list[dict]:
        """
        Single-tree search collecting all molecules above score_threshold.
        Timing is recorded into self.profiler.
        """
        collected: list[dict] = []
        seen_smiles: set[str] = set()

        for _ in range(self.n_iter):
            node = _select(root, self.library, self.ucb_c, self.profiler)
            if not node.is_terminal():
                node = _expand(
                    node,
                    self.library,
                    temperature,
                    dirichlet_alpha,
                    dirichlet_eps,
                    use_dirichlet,
                    self.profiler,
                )
            terminal_state = _simulate(
                node.state.clone(),
                self.library,
                self.rollout_depth,
                temperature,
                self.profiler,
            )
            reward = _score_state(terminal_state, self.score_fn, self.profiler)
            _backpropagate(node, reward, self.profiler)

            if terminal_state.is_complete and reward >= score_threshold:
                smi = terminal_state.assembled_smiles()
                if smi not in seen_smiles:
                    seen_smiles.add(smi)
                    collected.append(
                        {
                            "smiles": smi,
                            "score": reward,
                            "n_blocks": terminal_state.n_blocks,
                            "fragment_ids": list(terminal_state.fragment_ids),
                        }
                    )

        return collected
