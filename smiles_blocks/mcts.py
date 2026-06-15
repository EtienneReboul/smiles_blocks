"""
mcts_drug_design.py
===================
Monte Carlo Tree Search module for de novo drug design using R-BRICS fragments.

Fragment table schema expected
-------------------------------
block, can_smiles, first_connected_can_idx, last_connected_can_idx,
unique_id, begin_tag, end_tag, MolWt, nHDonors, nHAcceptors,
nRotatableBonds, CrippenlogP, TPSA, frequency, status

Architecture (v14)
------------------
Process parallelisation over independent runs (ProcessPoolExecutor) with
virtual-loss PUCT quality improvement from v13.

Threading was tested in v13 but Activity Monitor confirmed the GIL prevents
real multi-core utilisation for this workload: Python overhead around RDKit
calls (SMILES assembly, dict lookups, argument marshalling) keeps threads
serialised, achieving only ~1.3 cores out of 12 on Apple M2.

Processes bypass the GIL entirely. Each worker:
  - Receives the fragment library via shared memory (zero-copy Arrow IPC)
  - Runs a fully independent serial MCTS tree (no shared state)
  - Returns collected molecules and profiler data to the main process

Diversity across runs comes from independent random seeds and independent
tree structures, not from virtual loss (which is kept for within-run PUCT
quality but does not drive parallelism).

Performance optimisations (cumulative)
---------------------------------------
1.  Pool draining          : zero-weight numpy array instead of Arrow slice+concat
2.  Index selection        : searchsorted on pre-built CDF
3.  Weight building        : pre-cached log-frequency numpy arrays per tag group
4.  Row reads              : pre-materialised Python lists per tag group
5.  Incremental prior      : PUCT prior appends one value per child, no rebuild
6.  Cached group CDFs      : sample_compatible uses pre-built per-end_tag CDFs
7.  Fast clone             : O(1) scalar copy, immutable uid tuple
8.  Lazy CDF rebuild       : pop_untried skips CDF recompute on first draw
9.  Zero-Arrow expand      : pop_untried returns _pool_rows[idx] directly
10. Minimal MolState       : 4 scalars + immutable tuple; SMILES at leaf only
11. O(1) is_fully_expanded : _untried_remaining scalar instead of sum()
12. Virtual-loss PUCT      : (v13) configurable penalty on in-flight nodes
                             improves within-run exploration quality
13. Process parallelism    : (v14) ProcessPoolExecutor over n_runs; each
                             worker is a fully independent serial MCTS tree
                             bypassing the GIL; shared-memory Arrow transport

Usage
-----
>>> from mcts_drug_design import MCTSDrugDesign, make_composite_scorer
>>> scorer = make_composite_scorer(qed_weight=0.4, sa_weight=0.6, lipinski_weight=0.)
>>> mcts = MCTSDrugDesign(fragment_table, compatibility_map, n_iter=40_000)
>>> results = mcts.run(n_runs=100, n_jobs=-1, score_threshold=0.50)
>>> mcts.profiler.print_report()
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import math
from multiprocessing.shared_memory import SharedMemory
import os
import signal
import time
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, rdMolDescriptors
from rdkit.Contrib.SA_Score import sascorer  # pyright: ignore[reportMissingImports]

from smiles_blocks.rbrics_patterns import RBRICSCompatibilityMap

logger = logging.getLogger(__name__)


# ============================================================
# 0.  PROFILER
# ============================================================


class MCTSProfiler:
    """
    Lightweight wall-clock profiler for the MCTS phases.

    Tracked operations
    ------------------
    select              : tree traversal (PUCT evaluation)
    expand              : node expansion — pop_untried + child creation + prior update
    simulate            : rollout — fragment sampling + tag filtering
    score               : mol construction + reward function
    backprop            : value propagation to root
    library_query       : Arrow filter calls inside simulate (subset of simulate)
    prior_compute       : PUCT prior vector update (subset of expand)
    is_fully_expanded   : O(1) scalar check — was O(N) sum() in v11
    """

    OPERATIONS = (
        "select",
        "expand",
        "simulate",
        "score",
        "backprop",
        "library_query",
        "prior_compute",
        "expand_pop",
        "expand_clone",
        "expand_node",
        "expand_pool_init",
        "score_batch",
        "is_fully_expanded",  # v12: newly tracked — was the ~616s hidden gap in v11
        "batch_serial",  # v13: serial phase of each leaf batch (select+expand+simulate ×B)
        "backprop_batch",  # v13: backprop phase of each leaf batch
    )

    def __init__(self):
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self._starts: dict[str, float] = {}

    def start(self, op: str) -> None:
        self._starts[op] = time.perf_counter()

    def stop(self, op: str) -> None:
        self.totals[op] += time.perf_counter() - self._starts[op]
        self.counts[op] += 1

    def merge_dict(self, d: dict) -> None:
        for op in self.OPERATIONS:
            self.totals[op] += d["totals"].get(op, 0.0)
            self.counts[op] += d["counts"].get(op, 0)

    def to_dict(self) -> dict:
        return {"totals": dict(self.totals), "counts": dict(self.counts)}

    def report(self) -> str:
        grand_total = sum(self.totals.values()) or 1.0
        rows = []
        for op in self.OPERATIONS:
            t = self.totals.get(op, 0.0)
            n = self.counts.get(op, 0)
            rows.append((op, t, n, t / n * 1000 if n else 0.0, t / grand_total * 100))
        rows.sort(key=lambda r: r[1], reverse=True)
        hdr = f"{'Operation':<20} {'Total(s)':>8}  {'Calls':>7}  {'Mean(ms)':>10}  {'Share%':>8}"
        sep = "-" * len(hdr)
        lines = [sep, hdr, sep]
        for op, t, n, mean_ms, share in rows:
            lines.append(f"{op:<20} {t:>8.3f}  {n:>7d}  {mean_ms:>10.3f}  {share:>7.1f}%")
        lines += [sep, f"{'TOTAL':<20} {grand_total:>8.3f}", sep]
        return "\n".join(lines)

    def print_report(self) -> None:
        print("\n=== MCTS Timing Profile ===")
        print(self.report())


# ============================================================
# 1.  SCORING FUNCTIONS
# ============================================================


def lipinski_score(mol: Chem.Mol) -> float:
    """Soft Lipinski filter -> [0, 1]. Each violation subtracts 0.25."""
    if mol is None:
        return 0.0
    violations = sum(
        [
            Descriptors.MolWt(mol) > 500,  # pyright: ignore
            Descriptors.MolLogP(mol) > 5,  # pyright: ignore
            rdMolDescriptors.CalcNumHBD(mol) > 5,
            rdMolDescriptors.CalcNumHBA(mol) > 10,
        ]
    )
    return max(0.0, 1.0 - violations * 0.25)


def sa_score_component(mol: Chem.Mol, threshold: float = 4.4) -> float:
    """
    Soft SA component — linear gradient between threshold and 1.0.

    SA score is inverted: lower = easier to synthesise.
      SA > threshold : 0.0  (too hard to synthesise)
      SA = threshold : 0.0  (just at the cutoff)
      SA = 1.0       : 1.0  (trivially easy)
    """
    sa = sascorer.calculateScore(mol)
    if sa > threshold:
        return 0.0
    return (threshold - sa) / (threshold - 1.0)


def qed_component(mol: Chem.Mol, threshold: float = 0.7) -> float:
    """
    Soft QED component — linear gradient between threshold and 1.0.

      QED < threshold : 0.0
      QED = threshold : 0.0
      QED = 1.0       : 1.0
    """
    q = QED.qed(mol)
    if q < threshold:
        return 0.0
    return (q - threshold) / (1.0 - threshold)


def composite_score(
    mol: Chem.Mol,
    sa_threshold: float = 4.4,
    qed_threshold: float = 0.7,
    lipinski_weight: float = 0.5,
    sa_weight: float = 0.25,
    qed_weight: float = 0.25,
) -> float:
    """
    Weighted composite of three soft drug-likeness components.
    Weights sum to 1.0 so the maximum possible score is always 1.0.
    """
    if mol is None:
        return 0.0
    return (
        lipinski_weight * lipinski_score(mol)
        + sa_weight * sa_score_component(mol, threshold=sa_threshold)
        + qed_weight * qed_component(mol, threshold=qed_threshold)
    )


class _CompositeScorer:
    """Picklable composite scorer (spawn-safe on Windows/macOS)."""

    def __init__(
        self,
        lw: float,
        sw: float,
        qw: float,
        sa_thresh: float,
        qed_thresh: float,
    ):
        self.lw = lw
        self.sw = sw
        self.qw = qw
        self.sa_thresh = sa_thresh
        self.qed_thresh = qed_thresh

    def __call__(self, mol: Chem.Mol) -> float:
        return composite_score(
            mol,
            sa_threshold=self.sa_thresh,
            qed_threshold=self.qed_thresh,
            lipinski_weight=self.lw,
            sa_weight=self.sw,
            qed_weight=self.qw,
        )


def make_composite_scorer(
    lipinski_weight: float = 0.5,
    sa_weight: float = 0.25,
    qed_weight: float = 0.25,
    sa_threshold: float = 4.4,
    qed_threshold: float = 0.7,
) -> _CompositeScorer:
    """
    Factory returning a picklable composite scorer.

    Default weights: Lipinski 0.5 | SA 0.25 | QED 0.25  (sum = 1.0)

    >>> scorer = make_composite_scorer()
    >>> mcts   = MCTSDrugDesign(table, score_fn=scorer)
    """
    if abs(lipinski_weight + sa_weight + qed_weight - 1.0) > 1e-6:
        raise ValueError("lipinski_weight + sa_weight + qed_weight must sum to 1.0")
    return _CompositeScorer(
        lipinski_weight,
        sa_weight,
        qed_weight,
        sa_threshold,
        qed_threshold,
    )


# ============================================================
# 2.  FRAGMENT LIBRARY  (PyArrow + pre-materialised caches)
# ============================================================


class FragmentLibrary:
    """
    PyArrow-backed fragment library with shared-memory support and
    fully pre-materialised hot-path caches.

    Parameters
    ----------
    table            : pa.Table  Full fragment table (must have 'frequency').
    compatibility_map: dict | None  {end_tag: {begin_tags}}.
    temperature      : float  Baked into CDF cache at init time.
    conditional_table: pd.DataFrame | pa.Table | None
        Fragment transition counts. Columns: unique_id, next_unique_id,
        frequency, proba.
    """

    _ROW_COLS = ("unique_id", "block", "end_tag", "frequency")

    def __init__(
        self,
        table: pa.Table,
        compatibility_map: Optional[dict] = None,
        _shm_handle: Optional[SharedMemory] = None,
        temperature: float = 1.0,
        conditional_table=None,
    ):
        self.table = table
        self.compatibility_map = compatibility_map or dict(RBRICSCompatibilityMap().patterns)
        self._shm_handle = _shm_handle
        self._empty = table.slice(0, 0)
        self._temperature = temperature

        # Arrow slices
        begin_col = table.column("begin_tag")
        self._by_begin: dict[str, pa.Table] = {}
        for tag in begin_col.unique().to_pylist():
            mask = pc.equal(begin_col, tag)  # pyright: ignore[reportAttributeAccessIssue]
            self._by_begin[str(tag)] = pc.filter(table, mask)  # pyright: ignore[reportAttributeAccessIssue]

        # Pre-materialised row lists
        self._rows: dict[str, dict[str, list]] = {}
        for tag, tbl in self._by_begin.items():
            self._rows[tag] = {col: tbl.column(col).to_pylist() for col in self._ROW_COLS}

        # Log-weight and CDF arrays
        self._logw: dict[str, np.ndarray] = {}
        self._cdf: dict[str, np.ndarray] = {}
        for tag in self._by_begin:
            freqs = np.array(self._rows[tag]["frequency"], dtype=np.float32)
            log_f = np.log1p(freqs) / temperature
            log_f -= log_f.max()
            self._logw[tag] = log_f
            w = np.exp(log_f)
            w /= w.sum()
            self._cdf[tag] = np.cumsum(w)

        # FIX B: pre-cached group CDFs for sample_compatible
        self._compat_tags: dict[str, list[str]] = {}
        self._compat_cdf: dict[str, np.ndarray] = {}
        self._compat_cdf_no_terminal: dict[str, np.ndarray] = {}
        self._compat_cdf_terminal: dict[str, np.ndarray] = {}

        # Terminal index lists and sub-CDFs per begin_tag
        self._terminal_idx: dict[str, list[int]] = {}
        for tag in self._by_begin:
            end_tags = self._rows[tag]["end_tag"]
            self._terminal_idx[tag] = [i for i, et in enumerate(end_tags) if et == "no_tag"]

        self._cdf_terminal_only: dict[str, Optional[np.ndarray]] = {}
        for tag in self._by_begin:
            tidx = self._terminal_idx[tag]
            if not tidx:
                self._cdf_terminal_only[tag] = None
            else:
                sub = self._logw[tag][tidx]
                sub = sub - sub.max()
                w = np.exp(sub)
                w /= w.sum()
                self._cdf_terminal_only[tag] = np.cumsum(w)

        # Per-end_tag compatibility caches
        for end_tag, begin_tags in self.compatibility_map.items():
            avail = [t for t in begin_tags if t in self._by_begin]
            if not avail:
                continue
            self._compat_tags[end_tag] = avail

            sizes_all = np.array(
                [len(self._rows[t]["unique_id"]) for t in avail], dtype=np.float32
            )
            if sizes_all.sum() > 0:
                sizes_all /= sizes_all.sum()
                self._compat_cdf[end_tag] = np.cumsum(sizes_all)

            avail_nt = [t for t in avail if any(et != "no_tag" for et in self._rows[t]["end_tag"])]
            if avail_nt:
                sizes_nt = np.array(
                    [
                        sum(1 for et in self._rows[t]["end_tag"] if et != "no_tag")
                        for t in avail_nt
                    ],
                    dtype=np.float32,
                )
                if sizes_nt.sum() > 0:
                    sizes_nt /= sizes_nt.sum()
                    self._compat_cdf_no_terminal[end_tag] = np.cumsum(sizes_nt)
                    self._compat_tags[end_tag + "__nt"] = avail_nt

            avail_t = [t for t in avail if self._terminal_idx.get(t)]
            if avail_t:
                sizes_t = np.array([len(self._terminal_idx[t]) for t in avail_t], dtype=np.float32)
                if sizes_t.sum() > 0:
                    sizes_t /= sizes_t.sum()
                    self._compat_cdf_terminal[end_tag] = np.cumsum(sizes_t)
                    self._compat_tags[end_tag + "__t"] = avail_t

        # Conditional probability caches
        self._cond_next_ids: dict[str, list] = {}
        self._cond_cdf: dict[str, np.ndarray] = {}
        self._cond_terminal_mask: dict[str, np.ndarray] = {}
        self._cond_terminal_cdf: dict[str, Optional[np.ndarray]] = {}
        self._cond_uid_to_row: dict[str, int] = {
            uid: i for i, uid in enumerate(table.column("unique_id").to_pylist())
        }
        self._rows_by_uid: dict[str, dict] = {
            uid: {col: self._rows[tag][col][i] for col in self._ROW_COLS}
            for tag in self._by_begin
            for i, uid in enumerate(self._rows[tag]["unique_id"])
        }
        if conditional_table is not None:
            self._build_conditional_cache(conditional_table, temperature)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _searchsorted_sample(self, cdf: np.ndarray) -> int:
        return min(int(np.searchsorted(cdf, np.random.random())), len(cdf) - 1)

    def _build_conditional_cache(self, conditional_table, temperature: float = 1.0) -> None:
        """
        Build per-fragment conditional probability caches.
        Expected columns: unique_id, next_unique_id, frequency, proba.
        """
        cond_df = (
            conditional_table.to_pandas()
            if isinstance(conditional_table, pa.Table)
            else conditional_table
        )
        for uid, grp in cond_df.groupby("unique_id"):
            grp = grp.sort_values("next_unique_id").reset_index(drop=True)
            nids = grp["next_unique_id"].tolist()
            probas = grp["proba"].values.astype("float64")
            if temperature != 1.0:
                probas = probas.clip(1e-12) ** (1.0 / temperature)
            probas /= probas.sum()
            uid_str = str(uid)
            self._cond_next_ids[uid_str] = nids
            self._cond_cdf[uid_str] = probas.cumsum()
            t_mask = np.array(
                [self._rows_by_uid.get(nid, {}).get("end_tag") == "no_tag" for nid in nids],
                dtype=bool,
            )
            self._cond_terminal_mask[uid_str] = t_mask
            if t_mask.any():
                sub_p = probas[t_mask]
                sub_p /= sub_p.sum()
                self._cond_terminal_cdf[uid_str] = sub_p.cumsum()
            else:
                self._cond_terminal_cdf[uid_str] = None

    def sample_conditional(self, current_uid: str, terminal_only: bool = False) -> Optional[dict]:
        """
        Sample using P(a | current_uid) from the corpus transition cache.
        Returns None if current_uid is not in the cache.
        """
        if current_uid not in self._cond_next_ids:
            return None
        next_ids = self._cond_next_ids[current_uid]
        cdf = self._cond_cdf[current_uid]
        if terminal_only:
            t_mask = self._cond_terminal_mask.get(current_uid)
            t_cdf = self._cond_terminal_cdf.get(current_uid)
            if t_mask is None or not t_mask.any():
                return None
            t_indices = np.where(t_mask)[0]
            if len(t_indices) == 1:
                chosen_uid = next_ids[int(t_indices[0])]
            elif t_cdf is not None:
                chosen_uid = next_ids[t_indices[self._searchsorted_sample(t_cdf)]]
            else:
                return None
        else:
            chosen_uid = next_ids[self._searchsorted_sample(cdf)]
        return self._rows_by_uid.get(chosen_uid)

    # ------------------------------------------------------------------
    # Tag-filtered table queries
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
        return pc.filter(cands, pc.equal(cands.column("end_tag"), "no_tag"))  # pyright: ignore[reportAttributeAccessIssue]

    # ------------------------------------------------------------------
    # Fast sampling
    # ------------------------------------------------------------------

    def sample_from_tag(
        self,
        tag: str,
        terminal_only: bool = False,
        temperature: Optional[float] = None,
    ) -> Optional[dict]:
        if tag not in self._by_begin:
            return None
        rows = self._rows[tag]
        n = len(rows["unique_id"])
        if n == 0:
            return None
        if terminal_only:
            terminal_idx = self._terminal_idx.get(tag, [])
            if not terminal_idx:
                return None
            if len(terminal_idx) == 1:
                return {col: rows[col][terminal_idx[0]] for col in self._ROW_COLS}
            if temperature and temperature != self._temperature:
                freqs = np.array([rows["frequency"][i] for i in terminal_idx], dtype=np.float32)
                log_f = np.log1p(freqs) / temperature
                log_f -= log_f.max()
                w = np.exp(log_f)
                w /= w.sum()
                cdf = np.cumsum(w)
            else:
                cdf = self._cdf_terminal_only[tag]
                if cdf is None:
                    return None
            pick = terminal_idx[self._searchsorted_sample(cdf)]
            return {col: rows[col][pick] for col in self._ROW_COLS}
        if temperature and temperature != self._temperature:
            freqs = np.array(rows["frequency"], dtype=np.float32)
            log_f = np.log1p(freqs) / temperature
            log_f -= log_f.max()
            w = np.exp(log_f)
            w /= w.sum()
            cdf = np.cumsum(w)
        else:
            cdf = self._cdf[tag]
        idx = min(self._searchsorted_sample(cdf), n - 1)
        return {col: rows[col][idx] for col in self._ROW_COLS}

    def sample_compatible(
        self,
        end_tag: str,
        terminal_only: bool = False,
        exclude_terminal: bool = False,
        temperature: Optional[float] = None,
    ) -> Optional[dict]:
        """FIX B: pre-cached group CDFs — no list comprehension in hot path."""
        if terminal_only:
            cdf = self._compat_cdf_terminal.get(end_tag)
            avail = self._compat_tags.get(end_tag + "__t")
        elif exclude_terminal:
            cdf = self._compat_cdf_no_terminal.get(end_tag)
            avail = self._compat_tags.get(end_tag + "__nt")
        else:
            cdf = self._compat_cdf.get(end_tag)
            avail = self._compat_tags.get(end_tag)
        if cdf is None or not avail:
            return None
        chosen_tag = avail[self._searchsorted_sample(cdf)]
        return self.sample_from_tag(
            chosen_tag, terminal_only=terminal_only, temperature=temperature
        )

    # ------------------------------------------------------------------
    # PUCT prior computation
    # ------------------------------------------------------------------

    def compute_prior(self, table: pa.Table, temperature: Optional[float] = None) -> np.ndarray:
        t = temperature or self._temperature
        freqs = np.array(table.column("frequency").to_pylist(), dtype=np.float32)
        log_f = np.log1p(freqs) / t
        log_f -= log_f.max()
        w = np.exp(log_f)
        return w / w.sum()

    def apply_novelty_penalty(
        self,
        fragment_counts: dict[str, int],
        novelty_beta: float,
    ) -> None:
        """
        Downweight fragments that have already been heavily explored across
        previous waves by subtracting a penalty from their log-weights.

        The adjusted log-weight for fragment a becomes:
            log(f_a + 1) / T  -  novelty_beta * log(seen_a + 1)

        where seen_a is the number of times fragment a appeared in molecules
        collected so far. This reshapes the CDF for every tag group so that
        the next wave's workers sample from a different region of fragment space.

        Called once per wave in the main process before submitting new workers.
        Only rebuilds CDFs for tag groups that contain penalised fragments.

        Parameters
        ----------
        fragment_counts : dict[str, int]
            Maps unique_id -> total occurrence count across all collected molecules.
        novelty_beta : float
            Penalty strength. 0.0 = no effect. 1.0 = halves the effective
            log-weight of a fragment seen 10 times (log(11) ≈ 2.4 nats).
            Typical range: 0.1 – 1.0.
        """
        if not fragment_counts or novelty_beta == 0.0:
            return

        for tag in self._by_begin:
            rows = self._rows[tag]
            uids = rows["unique_id"]
            freqs = rows["frequency"]
            n = len(uids)

            # Check if any fragment in this group has been seen
            if not any(uid in fragment_counts for uid in uids):
                continue

            # Rebuild log-weights with penalty
            log_f = np.array(
                [
                    math.log1p(freqs[i]) / self._temperature
                    - novelty_beta * math.log1p(fragment_counts.get(uids[i], 0))
                    for i in range(n)
                ],
                dtype=np.float64,
            )
            log_f -= log_f.max()
            self._logw[tag] = log_f.astype(np.float32)
            w = np.exp(log_f)
            w /= w.sum()
            self._cdf[tag] = np.cumsum(w).astype(np.float64)

            # Rebuild terminal-only CDF for this tag
            tidx = self._terminal_idx.get(tag, [])
            if tidx:
                sub = self._logw[tag][tidx]
                sub = sub - sub.max()
                wt = np.exp(sub)
                wt /= wt.sum()
                self._cdf_terminal_only[tag] = np.cumsum(wt)

        # Rebuild per-end_tag compatibility CDFs (stage-1 of sample_compatible)
        # These weight tag groups by their total pool size — recompute from
        # updated per-tag CDFs to stay consistent.
        for end_tag, avail in list(self._compat_tags.items()):
            if end_tag.endswith("__nt") or end_tag.endswith("__t"):
                continue  # handled via the base end_tag rebuild below
            # Recompute group-level weights as sum of exp(logw) per tag
            sizes_all = np.array(
                [np.exp(self._logw[t]).sum() if t in self._logw else 0.0 for t in avail],
                dtype=np.float64,
            )
            if sizes_all.sum() > 0:
                sizes_all /= sizes_all.sum()
                self._compat_cdf[end_tag] = np.cumsum(sizes_all)

            avail_nt = self._compat_tags.get(end_tag + "__nt", [])
            if avail_nt:
                sizes_nt = np.array(
                    [
                        np.exp(
                            self._logw[t][
                                [
                                    i
                                    for i, et in enumerate(self._rows[t]["end_tag"])
                                    if et != "no_tag"
                                ]
                            ]
                        ).sum()
                        if t in self._logw
                        else 0.0
                        for t in avail_nt
                    ],
                    dtype=np.float64,
                )
                if sizes_nt.sum() > 0:
                    sizes_nt /= sizes_nt.sum()
                    self._compat_cdf_no_terminal[end_tag] = np.cumsum(sizes_nt)

            avail_t = self._compat_tags.get(end_tag + "__t", [])
            if avail_t:
                sizes_t = np.array(
                    [
                        np.exp(self._logw[t][self._terminal_idx[t]]).sum()
                        if t in self._logw and self._terminal_idx.get(t)
                        else 0.0
                        for t in avail_t
                    ],
                    dtype=np.float64,
                )
                if sizes_t.sum() > 0:
                    sizes_t /= sizes_t.sum()
                    self._compat_cdf_terminal[end_tag] = np.cumsum(sizes_t)

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
        temperature: float = 1.0,
        conditional_records: Optional[list] = None,
        fragment_counts: Optional[dict] = None,
        novelty_beta: float = 0.0,
    ) -> "FragmentLibrary":
        shm = SharedMemory(name=shm_name, create=False)
        buf = pa.py_buffer(bytes(shm.buf[:n_bytes]))  # pyright: ignore[reportOptionalSubscript]
        reader = ipc.open_file(buf)
        table = reader.read_all()
        cond_df = pd.DataFrame(conditional_records) if conditional_records else None
        lib = FragmentLibrary(
            table,
            compatibility_map,
            _shm_handle=shm,
            temperature=temperature,
            conditional_table=cond_df,
        )
        if fragment_counts and novelty_beta > 0.0:
            lib.apply_novelty_penalty(fragment_counts, novelty_beta)
        return lib


# ============================================================
# 3.  MOLECULE STATE
# ============================================================


class MolState:
    """
    Minimal molecule assembly state — optimised for the MCTS hot path.

    Only four scalars + one immutable tuple are carried per node.
    clone() is O(1) with zero copies. SMILES is assembled once at the
    leaf via lib._rows_by_uid — no string accumulation mid-search.
    """

    __slots__ = ("current_end_tag", "last_uid", "n_blocks", "target_n_blocks", "_uid_chain")

    def __init__(self, target_n_blocks: int = 1) -> None:
        self.current_end_tag: str = "no_tag"
        self.last_uid: str = ""
        self.n_blocks: int = 0
        self.target_n_blocks: int = target_n_blocks
        self._uid_chain: tuple = ()

    @property
    def is_complete(self) -> bool:
        return (
            self.n_blocks >= 1
            and self.n_blocks == self.target_n_blocks
            and self.current_end_tag == "no_tag"
        )

    def append(self, row: dict) -> None:
        """Extend the chain with one fragment row — O(1) scalar updates."""
        self._uid_chain = self._uid_chain + (row["unique_id"],)
        self.last_uid = row["unique_id"]
        self.current_end_tag = row["end_tag"]
        self.n_blocks += 1

    def clone(self) -> "MolState":
        """O(1) true clone — tuple shared by reference, zero copies."""
        s = MolState.__new__(MolState)
        s.current_end_tag = self.current_end_tag
        s.last_uid = self.last_uid
        s.n_blocks = self.n_blocks
        s.target_n_blocks = self.target_n_blocks
        s._uid_chain = self._uid_chain
        return s

    def assembled_smiles(self, lib: "FragmentLibrary") -> str:
        """Materialise SMILES at leaf time — zero Arrow .as_py() calls."""
        return "".join(lib._rows_by_uid[uid]["block"] for uid in self._uid_chain)

    def to_mol(self, lib: "FragmentLibrary") -> Optional[Chem.Mol]:
        smi = self.assembled_smiles(lib)
        return Chem.MolFromSmiles(smi) if smi else None

    @property
    def fragment_ids(self) -> list:
        """Compatibility shim — uid chain as a list."""
        return list(self._uid_chain)


# ============================================================
# 4.  MCTS NODE  (PUCT + optimised pool tracking)
# ============================================================


class MCTSNode:
    __slots__ = (
        "state",
        "parent",
        "fragment_row",
        "children",
        "visits",
        "value",
        "depth",
        "_prior",
        "_prior_logf",
        "_pool_table",  # pa.Table — kept for compatibility checks only
        "_pool_rows",  # list[dict] — pre-materialised, zero Arrow in hot path
        "_pool_size",  # int — total pool size (constant after _pool())
        "_untried_remaining",  # int — O(1) exhaustion check
        "_untried_w",  # np.ndarray — zero-weight tracking for weighted sampling
        "_untried_cdf",  # np.ndarray — lazy CDF over untried weights
        "_untried_dirty",  # bool — CDF needs rebuild
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
        self.depth: int = depth
        self._prior: Optional[np.ndarray] = None
        self._prior_logf: Optional[np.ndarray] = None
        self._pool_table: Optional[pa.Table] = None
        self._pool_rows: Optional[list] = None
        self._pool_size: int = 0
        self._untried_remaining: int = 0
        self._untried_w: Optional[np.ndarray] = None
        self._untried_cdf: Optional[np.ndarray] = None
        self._untried_dirty: bool = False

    def puct(self, c: float, child_idx: int) -> float:
        child = self.children[child_idx]
        q = child.value / child.visits if child.visits > 0 else 0.0
        prior = (
            float(self._prior[child_idx])
            if self._prior is not None and child_idx < len(self._prior)
            else 1.0 / max(len(self.children), 1)
        )
        return q + c * prior * math.sqrt(max(self.visits, 1)) / (1 + child.visits)

    def best_child(self, c: float) -> "MCTSNode":
        return self.children[max(range(len(self.children)), key=lambda i: self.puct(c, i))]

    def is_terminal(self) -> bool:
        return self.state.is_complete

    def _pool(
        self,
        lib: FragmentLibrary,
        profiler: Optional["MCTSProfiler"] = None,
    ) -> pa.Table:
        """
        Build (once) the candidate pool and pre-materialise all rows as plain
        Python dicts so pop_untried never calls Arrow .as_py() in the hot path.

        Three paths:
          start (n_blocks==0)    : single tag group "no_tag" — use lib._rows directly
          terminal (last block)  : iterate terminal-capable tags via _compat_tags
          mid-chain              : iterate non-terminal tags via _compat_tags
        All paths populate _pool_rows as list[dict] aligned with _untried_w.
        """
        if self._pool_table is not None:
            return self._pool_table

        if profiler:
            profiler.start("expand_pool_init")

        s = self.state
        ROW_COLS = lib._ROW_COLS

        if s.n_blocks == 0:
            # ── Start fragments: single "no_tag" group, always cached ────────
            pool = lib.get_start_fragments()
            rows = lib._rows.get("no_tag", {})
            n = len(rows.get("unique_id", []))
            pre_rows = [{col: rows[col][i] for col in ROW_COLS} for i in range(n)]
            cdf_cache = lib._cdf.get("no_tag")
            prior = np.exp(lib._logw["no_tag"])
            prior /= prior.sum()

        elif s.n_blocks == s.target_n_blocks - 1:
            # ── Terminal pool: collect rows from terminal-capable tag groups ──
            pool = lib.get_terminal_fragments(s.current_end_tag)
            pre_rows = []
            freq_list = []
            avail_t = lib._compat_tags.get(s.current_end_tag + "__t", [])
            for tag in avail_t:
                tag_rows = lib._rows.get(tag, {})
                t_indices = lib._terminal_idx.get(tag, [])
                for i in t_indices:
                    pre_rows.append({col: tag_rows[col][i] for col in ROW_COLS})
                    freq_list.append(tag_rows["frequency"][i])
            if freq_list:
                log_f = np.log1p(np.array(freq_list, dtype=np.float32))
                log_f -= log_f.max()
                prior = np.exp(log_f)
                prior /= prior.sum()
            else:
                prior = np.ones(len(pre_rows), dtype=np.float32) / max(len(pre_rows), 1)
            cdf_cache = None

        else:
            # ── Mid-chain pool: non-terminal rows across compatible tag groups
            pool = lib.get_compatible_extensions(s.current_end_tag)
            if pool.num_rows > 0:
                pool = pc.filter(pool, pc.not_equal(pool.column("end_tag"), "no_tag"))  # pyright: ignore[reportAttributeAccessIssue]
            pre_rows = []
            freq_list = []
            avail_nt = lib._compat_tags.get(s.current_end_tag + "__nt", [])
            for tag in avail_nt:
                tag_rows = lib._rows.get(tag, {})
                end_tags = tag_rows.get("end_tag", [])
                for i, et in enumerate(end_tags):
                    if et != "no_tag":
                        pre_rows.append({col: tag_rows[col][i] for col in ROW_COLS})
                        freq_list.append(tag_rows["frequency"][i])
            if freq_list:
                log_f = np.log1p(np.array(freq_list, dtype=np.float32))
                log_f -= log_f.max()
                prior = np.exp(log_f)
                prior /= prior.sum()
            else:
                prior = np.ones(len(pre_rows), dtype=np.float32) / max(len(pre_rows), 1)
            cdf_cache = None

        # Store pool table for num_rows checks; store pre-materialised rows
        self._pool_table = pool
        self._pool_rows = pre_rows
        self._pool_size = len(pre_rows)
        # v12: initialise scalar counter — decremented by pop_untried, O(1) check
        self._untried_remaining = self._pool_size

        if pre_rows:
            if cdf_cache is not None:
                self._untried_w = prior.copy()
                self._untried_cdf = cdf_cache.copy()
            else:
                self._untried_w = prior.copy()
                self._untried_cdf = np.cumsum(prior)
            self._untried_dirty = False

        if profiler:
            profiler.stop("expand_pool_init")

        return pool

    def is_fully_expanded(
        self,
        lib: FragmentLibrary,
        profiler: Optional["MCTSProfiler"] = None,
    ) -> bool:
        # v12: O(1) check — replaces _untried_w.sum() == 0.0 which was O(N)
        # and consumed ~616s in the v11 profile (untracked gap inside expand)
        if profiler:
            profiler.start("is_fully_expanded")
        self._pool(lib, profiler)  # ensure pool is initialised
        result = self._pool_size == 0 or self._untried_remaining == 0
        if profiler:
            profiler.stop("is_fully_expanded")
        return result

    def pop_untried(
        self,
        lib: FragmentLibrary,
        temperature: float = 1.0,
        profiler: Optional["MCTSProfiler"] = None,
    ) -> Optional[dict]:
        """
        Sample one fragment from the untried pool.

        v12: returns self._pool_rows[idx] directly — zero Arrow .as_py() calls.
        _untried_remaining is decremented to keep is_fully_expanded O(1).
        """
        self._pool(lib, profiler)
        if self._pool_size == 0 or self._untried_w is None:
            return None
        if self._untried_remaining == 0:
            return None
        if self._untried_dirty:
            # _untried_remaining > 0 guarantees at least one non-zero weight,
            # so sum() > 0 is certain — no zero-sum guard needed here.
            # We still call sum() once for normalisation, but only when the
            # CDF actually needs rebuilding (once per pop, not in is_fully_expanded).
            w_norm = self._untried_w / self._untried_w.sum()
            self._untried_cdf = np.cumsum(w_norm)
            self._untried_dirty = False
        idx = min(
            int(np.searchsorted(self._untried_cdf, np.random.random())),  # type: ignore
            self._pool_size - 1,
        )
        self._untried_w[idx] = 0.0
        self._untried_dirty = True
        # v12: decrement scalar counter — O(1) exhaustion tracking
        self._untried_remaining -= 1
        # v12: direct list lookup — zero Arrow reads (was: column[idx].as_py() + _rows_by_uid)
        return self._pool_rows[idx]  # type: ignore[index]

    def add_dirichlet_noise(self, alpha: float = 0.3, epsilon: float = 0.25) -> None:
        if self._prior is None or len(self.children) == 0:
            return
        noise = np.random.dirichlet([alpha] * len(self.children))
        self._prior = (1.0 - epsilon) * self._prior + epsilon * noise


# ============================================================
# 5.  BLOCK-COUNT SAMPLER
# ============================================================


class BlockCountSampler:
    def __init__(self, mu: float, sigma: float, min_blocks: int = 2, max_blocks: int = 10):
        self.mu, self.sigma = mu, sigma
        self.min_blocks, self.max_blocks = min_blocks, max_blocks

    def sample(self) -> int:
        n = int(round(np.random.normal(self.mu, self.sigma)))
        return max(self.min_blocks, min(self.max_blocks, n))


# ============================================================
# 6.  MODULE-LEVEL MCTS PHASE FUNCTIONS
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
        if not node.is_fully_expanded(lib, profiler):
            break
        if not node.children:
            break
        node = node.best_child(c)
    if profiler:
        profiler.stop("select")
    return node


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
    # is_fully_expanded is now timed inside its own method
    if node.is_fully_expanded(lib, profiler):
        if profiler:
            profiler.stop("expand")
        return node
    if profiler:
        profiler.start("expand_pop")
    row = node.pop_untried(lib, temperature, profiler)
    if profiler:
        profiler.stop("expand_pop")
    if row is None:
        if profiler:
            profiler.stop("expand")
        return node
    if profiler:
        profiler.start("expand_clone")
    new_state = node.state.clone()
    new_state.append(row)
    if profiler:
        profiler.stop("expand_clone")
    if profiler:
        profiler.start("expand_node")
    child = MCTSNode(state=new_state, parent=node, fragment_row=row, depth=node.depth + 1)
    node.children.append(child)
    if profiler:
        profiler.stop("expand_node")

    # Prior update: conditional if available, else incremental log-freq
    if profiler:
        profiler.start("prior_compute")
    parent_uid = node.fragment_row["unique_id"] if node.fragment_row is not None else None
    if parent_uid and parent_uid in lib._cond_next_ids:
        all_uids = lib._cond_next_ids[parent_uid]
        all_prob = np.diff(np.concatenate([[0.0], lib._cond_cdf[parent_uid]]))
        uid_to_p = dict(zip(all_uids, all_prob))
        child_uids = [
            c.fragment_row["unique_id"] for c in node.children if c.fragment_row is not None
        ]
        cond_p = np.array([uid_to_p.get(u, 1e-9) for u in child_uids], dtype=np.float64)
        cond_p /= cond_p.sum()
        node._prior = cond_p.astype(np.float32)
        node._prior_logf = np.log(np.clip(cond_p, 1e-12, None))
    else:
        new_logf = math.log1p(float(row["frequency"])) / temperature
        if node._prior_logf is None:
            node._prior_logf = np.array([new_logf], dtype=np.float64)
        else:
            node._prior_logf = np.append(node._prior_logf, new_logf)
        lf = node._prior_logf - node._prior_logf.max()
        w = np.exp(lf)
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
        last_uid = state.last_uid if state.n_blocks > 0 else None
        if remaining == 1:
            row = (
                lib.sample_conditional(last_uid, terminal_only=True)
                if last_uid and last_uid in lib._cond_next_ids
                else None
            )
            if row is None:
                row = lib.sample_compatible(
                    state.current_end_tag, terminal_only=True, temperature=temperature
                )
        else:
            row = (
                lib.sample_conditional(last_uid, terminal_only=False)
                if last_uid and last_uid in lib._cond_next_ids
                else None
            )
            if row is not None and row.get("end_tag") == "no_tag":
                row = None
            if row is None:
                row = lib.sample_compatible(
                    state.current_end_tag, exclude_terminal=True, temperature=temperature
                )
                if row is None:
                    row = lib.sample_compatible(
                        state.current_end_tag, terminal_only=True, temperature=temperature
                    )
        if profiler:
            profiler.stop("library_query")
        if row is None:
            break
        state.append(row)
    if profiler:
        profiler.stop("simulate")
    return state


def _score_state(
    state: MolState,
    score_fn,
    lib: "FragmentLibrary",
    profiler: Optional[MCTSProfiler] = None,
) -> float:
    if profiler:
        profiler.start("score")
    mol = state.to_mol(lib)
    reward = score_fn(mol) if mol else 0.0
    if profiler:
        profiler.stop("score")
    return reward


# ============================================================
# 7.  TOP-LEVEL WORKER  (module-level for pickling)
# ============================================================


def _search_one(args: tuple) -> tuple[list[dict], dict]:
    """
    Picklable worker for one serial MCTS search in a child process.

    Receives the fragment library via shared memory (zero-copy Arrow IPC).
    The novelty penalty (if any) is applied to the library's CDFs at init
    time so that fragments already heavily explored in previous waves are
    sampled less frequently in this run.
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
        conditional_records,
        fragment_counts,
        novelty_beta,
    ) = args

    library = FragmentLibrary.from_shared_memory(
        shm_name,
        shm_bytes,
        compat_map,
        temperature=temperature,
        conditional_records=conditional_records,
        fragment_counts=fragment_counts,
        novelty_beta=novelty_beta,
    )
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
        reward = _score_state(terminal_state, score_fn, library, profiler)
        _backpropagate(node, reward, profiler)
        if terminal_state.is_complete and reward >= score_threshold:
            smi = terminal_state.assembled_smiles(library)
            if smi not in seen_smiles:
                seen_smiles.add(smi)
                collected.append(
                    {
                        "smiles": smi,
                        "score": reward,
                        "n_blocks": terminal_state.n_blocks,
                        "fragment_ids": list(terminal_state._uid_chain),
                    }
                )

    return collected, profiler.to_dict()


# ============================================================
# 8.  MCTS ENGINE
# ============================================================


class MCTSDrugDesign:
    """
    MCTS-based de novo drug designer — process parallelism with novelty penalty (v15).

    n_runs independent trees are searched in parallel via ProcessPoolExecutor.
    Each worker process receives the fragment library via shared memory and
    runs a fully serial MCTS loop, bypassing the GIL entirely.

    Between waves, a fragment novelty penalty is computed from the fragments
    already seen in collected molecules and applied to the library CDFs for
    the next wave. Fragments that have been heavily explored are downweighted,
    steering subsequent runs toward less-visited regions of fragment space.

    Parameters
    ----------
    fragment_table     : pa.Table   Fragment library (must include 'frequency').
    compatibility_map  : dict|None  R-BRICS {end_tag: {begin_tags}}.
    block_count_mu     : float      Gaussian mean for chain length.
    block_count_sigma  : float      Gaussian std-dev for chain length.
    min_blocks         : int        Hard min chain length (default 2).
    max_blocks         : int        Hard max chain length (default 10).
    n_iter             : int        MCTS iterations per run (default 40_000).
    ucb_c              : float      PUCT exploration constant (default sqrt(2)).
    score_fn           : callable   mol -> [0,1]. Default: lipinski_score.
    rollout_depth      : int        Max rollout steps (default 20).
    conditional_table  : pd.DataFrame | pa.Table | None
        Fragment transition counts. Columns: unique_id, next_unique_id,
        frequency, proba.
    novelty_beta       : float
        Strength of the inter-wave novelty penalty (default 0.3).
        The adjusted log-weight of fragment a is:
            log(f_a + 1) / T  -  novelty_beta * log(seen_a + 1)
        where seen_a is how many collected molecules contain fragment a.
        0.0 disables the penalty. Typical range: 0.1 – 1.0.
    """

    def __init__(
        self,
        fragment_table: pa.Table,
        compatibility_map: Optional[dict] = None,
        block_count_mu: float = 4.0,
        block_count_sigma: float = 1.5,
        min_blocks: int = 2,
        max_blocks: int = 10,
        n_iter: int = 40_000,
        ucb_c: float = math.sqrt(2),
        score_fn=None,
        rollout_depth: int = 20,
        conditional_table=None,
        novelty_beta: float = 0.3,
    ):
        self._conditional_records: Optional[list] = (
            conditional_table.to_pandas().to_dict("records")  # pyright: ignore[reportOptionalMemberAccess]
            if isinstance(conditional_table, pa.Table)
            else conditional_table.to_dict("records")
            if conditional_table is not None
            else None
        )
        self.library = FragmentLibrary(
            fragment_table,
            compatibility_map,
            conditional_table=conditional_table,
        )
        self.sampler = BlockCountSampler(block_count_mu, block_count_sigma, min_blocks, max_blocks)
        self.n_iter = n_iter
        self.ucb_c = ucb_c
        self.score_fn = score_fn or lipinski_score
        self.rollout_depth = rollout_depth
        self.novelty_beta = novelty_beta
        self.profiler = MCTSProfiler()

    # ------------------------------------------------------------------ #
    #  PUBLIC API                                                          #
    # ------------------------------------------------------------------ #

    def run(
        self,
        target_compounds: int = 300_000,
        max_runs: int = 200,
        n_jobs: int = -1,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        use_dirichlet: bool = True,
        score_threshold: float = 0.5,
        print_profile: bool = True,
    ) -> list[dict]:
        """
        Run independent MCTS searches until either target_compounds unique
        molecules above score_threshold are collected, or max_runs searches
        have been completed — whichever comes first.

        Runs are submitted in waves of n_workers at a time. After each wave,
        fragment counts from all collected molecules are tallied and passed to
        the next wave's workers, which apply a novelty penalty to their local
        library CDFs. This steers each successive wave toward fragments not
        yet heavily explored.

        Parameters
        ----------
        target_compounds : int
            Stop early once this many unique molecules are collected.
        max_runs : int
            Hard ceiling on total runs regardless of compound count.
        n_jobs : int
            Worker processes. -1 = os.cpu_count(). Wave size = n_jobs.
        score_threshold : float
            Minimum composite score for a molecule to be collected.
        """
        self.profiler = MCTSProfiler()
        n_workers = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)

        seen: set[str] = set()
        unique: list[dict] = []
        runs_done = 0
        # fragment_counts: unique_id -> total occurrences across all collected molecules
        fragment_counts: dict[str, int] = {}

        def _ingest(batch: list[dict]) -> None:
            """Dedup, accumulate, and update fragment counts from one run."""
            for r in batch:
                if r["smiles"] not in seen:
                    seen.add(r["smiles"])
                    unique.append(r)
                # Count every fragment in every collected molecule (above threshold),
                # including duplicates, so heavily reused fragments accumulate faster.
                for uid in r["fragment_ids"]:
                    fragment_counts[uid] = fragment_counts.get(uid, 0) + 1

        def _make_args(shm_name, shm_bytes, target_n_blocks):
            return (
                shm_name,
                shm_bytes,
                self.library.compatibility_map,
                target_n_blocks,
                self.n_iter,
                self.ucb_c,
                self.score_fn,
                self.rollout_depth,
                temperature,
                dirichlet_alpha,
                dirichlet_eps,
                use_dirichlet,
                score_threshold,
                self._conditional_records,
                # snapshot of counts at wave submission time — immutable in worker
                dict(fragment_counts),
                self.novelty_beta,
            )

        if n_workers == 1:
            # Serial path — useful for debugging.
            while runs_done < max_runs and len(unique) < target_compounds:
                target_n_blocks = self.sampler.sample()
                # Apply novelty penalty to the in-process library directly
                if fragment_counts and self.novelty_beta > 0.0:
                    self.library.apply_novelty_penalty(fragment_counts, self.novelty_beta)
                root = MCTSNode(state=MolState(target_n_blocks=target_n_blocks))
                seen_run: set[str] = set()
                batch: list[dict] = []
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
                    reward = _score_state(
                        terminal_state, self.score_fn, self.library, self.profiler
                    )
                    _backpropagate(node, reward, self.profiler)
                    if terminal_state.is_complete and reward >= score_threshold:
                        smi = terminal_state.assembled_smiles(self.library)
                        if smi not in seen_run:
                            seen_run.add(smi)
                            batch.append(
                                {
                                    "smiles": smi,
                                    "score": reward,
                                    "n_blocks": terminal_state.n_blocks,
                                    "fragment_ids": list(terminal_state._uid_chain),
                                }
                            )
                _ingest(batch)
                runs_done += 1
                logger.info(
                    "Run %d done — %d unique molecules so far (target %d, max_runs %d)",
                    runs_done,
                    len(unique),
                    target_compounds,
                    max_runs,
                )

        else:
            shm, shm_bytes = FragmentLibrary.to_shared_memory(self.library.table)
            _shm = shm

            def _sig(s, f):
                _shm.close()
                _shm.unlink()
                raise SystemExit(1)

            signal.signal(signal.SIGTERM, _sig)

            try:
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    while runs_done < max_runs and len(unique) < target_compounds:
                        wave = min(n_workers, max_runs - runs_done)  # pyright: ignore[reportArgumentType]
                        # fragment_counts snapshot baked into args at submission time
                        wave_args = [
                            _make_args(shm.name, shm_bytes, self.sampler.sample())
                            for _ in range(wave)
                        ]
                        futures = [pool.submit(_search_one, a) for a in wave_args]
                        for fut in as_completed(futures):
                            batch, prof_dict = fut.result()
                            _ingest(batch)
                            self.profiler.merge_dict(prof_dict)
                        runs_done += wave
                        logger.info(
                            "Wave done — %d runs completed, "
                            "%d unique molecules (target %d, budget %d), "
                            "%d distinct fragments penalised",
                            runs_done,
                            len(unique),
                            target_compounds,
                            max_runs,
                            len(fragment_counts),
                        )
            finally:
                shm.close()
                shm.unlink()

        unique.sort(key=lambda r: r["score"], reverse=True)

        stop_reason = (
            "target reached" if len(unique) >= target_compounds else "run budget exhausted"
        )
        logger.info(
            "Finished: %d unique molecules above %.2f from %d runs (%s).",
            len(unique),
            score_threshold,
            runs_done,
            stop_reason,
        )
        if print_profile:
            self.profiler.print_report()
        return unique
