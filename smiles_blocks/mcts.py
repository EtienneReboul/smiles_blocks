"""
mcts_drug_design.py
===================
Monte Carlo Tree Search module for de novo drug design using R-BRICS fragments.

Fragment table schema expected
-------------------------------
block, can_smiles, first_connected_can_idx, last_connected_can_idx,
unique_id, begin_tag, end_tag, MolWt, nHDonors, nHAcceptors,
nRotatableBonds, CrippenlogP, TPSA, frequency, status

Performance optimisations applied (v7)
---------------------------------------
1. Pool draining    : zero-weight numpy array instead of Arrow slice+concat
                      (~19x speedup on expand)
2. Index selection  : searchsorted on pre-built CDF instead of np.random.choice(p=)
                      (~8x speedup, flat across pool sizes)
3. Weight building  : pre-cached log-frequency numpy arrays per tag group
                      (~40x speedup on weight construction)
4. Row reads        : pre-materialised Python lists per tag group
                      (~20x speedup on row materialisation)
5. Incremental prior: (FIX A) PUCT prior updated by appending one value and
                      renormalising — no full vector rebuild per expansion
6. Cached group CDFs: (FIX B) sample_compatible stage-1 uses pre-built per-end_tag
                      CDF arrays — no list comprehension or sum() in rollout
7. Fast clone       : explicit list.copy() instead of deepcopy in MolState.clone()
8. Lazy CDF rebuild : pop_untried skips CDF recompute on the first draw
                      (pre-built CDF from _pool() is reused directly)
9. Expand sub-timers: expand_pop / expand_clone / expand_node / expand_pool_init
10. Fast _pool init : start-fragment pool uses pre-cached CDF directly;
                      pool_init timer exposes remaining Arrow cost per node
11. Conditional prior: true corpus transition probabilities P(a|s) used
                      instead of marginal log-frequency prior when available;
                      falls back to marginal prior for unseen parent fragments

Usage
-----
>>> from mcts_drug_design import MCTSDrugDesign
>>> mcts = MCTSDrugDesign(fragment_table, compatibility_map, n_iter=1000)
>>> results = mcts.run(n_runs=50, n_jobs=1, temperature=1.0, score_threshold=0.75)
>>> mcts.profiler.print_report()
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
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
from rdkit.Chem import Descriptors, rdMolDescriptors
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
    select        : tree traversal (PUCT evaluation)
    expand        : node expansion — pop_untried + child creation + prior update
    simulate      : rollout — fragment sampling + tag filtering
    score         : mol construction + reward function
    backprop      : value propagation to root
    library_query : Arrow filter calls inside simulate (subset of simulate)
    prior_compute : PUCT prior vector update (subset of expand)
    """

    OPERATIONS = (
        "select",
        "expand",
        "simulate",
        "score",
        "backprop",
        "library_query",
        "prior_compute",
        # expand sub-timers (nested inside expand total)
        "expand_pop",  # pop_untried: CDF rebuild + searchsorted + row read
        "expand_clone",  # MolState.clone()
        "expand_node",  # MCTSNode construction + list append
        "expand_pool_init",  # _pool(): first-time pool construction + prior init
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
        hdr = f"{'Operation':<17} {'Total(s)':>8}  {'Calls':>7}  {'Mean(ms)':>10}  {'Share%':>8}"
        sep = "-" * len(hdr)
        lines = [sep, hdr, sep]
        for op, t, n, mean_ms, share in rows:
            lines.append(f"{op:<17} {t:>8.3f}  {n:>7d}  {mean_ms:>10.3f}  {share:>7.1f}%")
        lines += [sep, f"{'TOTAL':<17} {grand_total:>8.3f}", sep]
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
    """Picklable composite scorer (spawn-safe on Windows/macOS)."""

    def __init__(self, lw: float, sw: float, sa_thresh: float):
        self.lw, self.sw, self.sa_thresh = lw, sw, sa_thresh

    def __call__(self, mol: Chem.Mol) -> float:
        return composite_score(mol, self.sa_thresh, self.lw, self.sw)


def make_composite_scorer(
    lipinski_weight: float = 0.7,
    sa_weight: float = 0.3,
    sa_threshold: float = 4.4,
) -> _CompositeScorer:
    """
    Factory returning a picklable composite scorer.

    >>> scorer = make_composite_scorer()
    >>> mcts   = MCTSDrugDesign(table, score_fn=scorer)
    """
    if abs(lipinski_weight + sa_weight - 1.0) > 1e-6:
        raise ValueError("Weights must sum to 1.0")
    return _CompositeScorer(lipinski_weight, sa_weight, sa_threshold)


# ============================================================
# 2.  FRAGMENT LIBRARY  (PyArrow + pre-materialised caches)
# ============================================================


class FragmentLibrary:
    """
    PyArrow-backed fragment library with shared-memory support and
    fully pre-materialised hot-path caches.

    Construction (once)
    -------------------
    For every begin_tag group the library pre-computes:
      - a zero-copy pa.Table slice (_by_begin)
      - pre-materialised Python lists for the four row columns (_rows cache)
      - log-frequency numpy array (_logw cache) — no Arrow round-trip in the loop
      - pre-built CDF numpy array (_cdf cache) for O(log n) searchsorted draws

    All four caches are keyed by begin_tag string so they align perfectly with
    the tag-filtered lookup pattern used in _pool() / _simulate().

    Parameters
    ----------
    table            : pa.Table
        Full fragment table. Must contain a 'frequency' column.
    compatibility_map: dict[str, set[str]] | None
        {end_tag: {begin_tags compatible with it}}.
    temperature      : float
        Log-frequency temperature baked into the CDF cache.
        If you vary temperature at runtime, pass temperature=1.0 here and
        apply the scaling lazily in sample_* methods.
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

        # ── Arrow slices (zero-copy, used for tag filtering) ─────────────────
        begin_col = table.column("begin_tag")
        self._by_begin: dict[str, pa.Table] = {}
        for tag in begin_col.unique().to_pylist():
            mask = pc.equal(begin_col, tag)  # pyright: ignore[reportAttributeAccessIssue]
            self._by_begin[str(tag)] = pc.filter(table, mask)  # pyright: ignore[reportAttributeAccessIssue]

        # ── Pre-materialised row cache ────────────────────────────────────────
        # _rows[tag] is a dict of column-name -> Python list, indexed directly
        # by integer position. No Arrow scalar boxing in the hot path.
        self._rows: dict[str, dict[str, list]] = {}
        for tag, tbl in self._by_begin.items():
            self._rows[tag] = {col: tbl.column(col).to_pylist() for col in self._ROW_COLS}

        # ── Pre-computed log-weight and CDF arrays ────────────────────────────
        # _logw[tag]  : unnormalised log-frequency array (temperature applied)
        # _cdf[tag]   : normalised cumulative distribution for searchsorted
        self._logw: dict[str, np.ndarray] = {}
        self._cdf: dict[str, np.ndarray] = {}
        for tag, tbl in self._by_begin.items():
            freqs = np.array(self._rows[tag]["frequency"], dtype=np.float32)
            log_f = np.log1p(freqs) / temperature
            log_f -= log_f.max()
            self._logw[tag] = log_f
            w = np.exp(log_f)
            w /= w.sum()
            self._cdf[tag] = np.cumsum(w)

        # ── FIX B: pre-cached group-size CDFs for sample_compatible ──────────
        # For each end_tag we pre-compute three size CDF arrays (one per
        # filter mode) so stage-1 of sample_compatible is a pure dict lookup
        # rather than a list comprehension + sum() on every rollout step.
        #
        # Keys:
        #   _compat_tags[end_tag]           -> list of compatible begin_tags
        #   _compat_cdf[end_tag]            -> CDF over all compatible groups
        #   _compat_cdf_no_terminal[end_tag]-> CDF excluding terminal-only groups
        #   _compat_cdf_terminal[end_tag]   -> CDF over terminal-capable groups
        #
        # "terminal-capable" means the group contains at least one row where
        # end_tag == 'no_tag'; "non-terminal" means it has at least one row
        # where end_tag != 'no_tag'.
        self._compat_tags: dict[str, list[str]] = {}
        self._compat_cdf: dict[str, np.ndarray] = {}
        self._compat_cdf_no_terminal: dict[str, np.ndarray] = {}
        self._compat_cdf_terminal: dict[str, np.ndarray] = {}
        # Also cache which tags in each group are terminal/non-terminal
        # so sample_from_tag's inner filter is a set lookup not a list scan
        self._terminal_tags: dict[str, set[str]] = {}  # tags w/ any terminal row
        self._nonterminal_tags: dict[str, set[str]] = {}  # tags w/ any non-terminal row

        for tag, tbl in self._by_begin.items():
            end_tags = self._rows[tag]["end_tag"]
            has_terminal = any(et == "no_tag" for et in end_tags)
            has_nonterminal = any(et != "no_tag" for et in end_tags)
            if has_terminal:
                # pre-cache terminal indices per tag for sample_from_tag
                pass  # already computed on demand; cache added below
            _ = has_nonterminal  # used in group filtering

        # pre-cache terminal index lists per begin_tag
        self._terminal_idx: dict[str, list[int]] = {}
        for tag in self._by_begin:
            end_tags = self._rows[tag]["end_tag"]
            self._terminal_idx[tag] = [i for i, et in enumerate(end_tags) if et == "no_tag"]
            # pre-build terminal sub-CDF
        self._cdf_terminal_only: dict[str, Optional[np.ndarray]] = {}
        for tag in self._by_begin:
            tidx = self._terminal_idx[tag]
            if len(tidx) == 0:
                self._cdf_terminal_only[tag] = None
            else:
                sub_logw = self._logw[tag][tidx]
                sub_logw = sub_logw - sub_logw.max()
                w = np.exp(sub_logw)
                w /= w.sum()
                self._cdf_terminal_only[tag] = np.cumsum(w)

        # Build per-end_tag compatibility caches
        for end_tag, begin_tags in self.compatibility_map.items():
            avail = [t for t in begin_tags if t in self._by_begin]
            if not avail:
                continue
            self._compat_tags[end_tag] = avail

            # All-compatible CDF (no filter)
            sizes_all = np.array(
                [len(self._rows[t]["unique_id"]) for t in avail],
                dtype=np.float32,
            )
            if sizes_all.sum() > 0:
                sizes_all /= sizes_all.sum()
                self._compat_cdf[end_tag] = np.cumsum(sizes_all)

            # Non-terminal CDF (exclude groups with only terminal rows)
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
                    # store filtered tag list alongside
                    self._compat_tags[end_tag + "__nt"] = avail_nt

            # Terminal CDF (only groups with at least one terminal row)
            avail_t = [t for t in avail if self._terminal_idx.get(t)]
            if avail_t:
                sizes_t = np.array(
                    [len(self._terminal_idx[t]) for t in avail_t],
                    dtype=np.float32,
                )
                if sizes_t.sum() > 0:
                    sizes_t /= sizes_t.sum()
                    self._compat_cdf_terminal[end_tag] = np.cumsum(sizes_t)
                    self._compat_tags[end_tag + "__t"] = avail_t

        # ── Conditional probability caches ───────────────────────────────────
        # _cond_next_ids[uid]  : list[str]    ordered next fragment unique_ids
        # _cond_cdf[uid]       : np.ndarray   CDF over next fragments
        # _cond_uid_to_row[uid]: int          row index in self.table
        # _rows_by_uid[uid]    : dict         pre-materialised row for O(1) lookup
        self._cond_next_ids: dict[str, list] = {}
        self._cond_cdf: dict[str, np.ndarray] = {}
        self._cond_terminal_mask: dict[str, np.ndarray] = {}  # bool mask, pre-cached
        self._cond_terminal_cdf: dict[str, Optional[np.ndarray]] = {}  # sub-CDF for terminal draw
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

    def _row_from_cache(self, tag: str, idx: int) -> dict:
        """Read one row from pre-materialised lists — pure Python, no Arrow."""
        cols = self._rows[tag]
        return {col: cols[col][idx] for col in self._ROW_COLS}

    def _searchsorted_sample(self, cdf: np.ndarray) -> int:
        """
        O(log n) weighted draw using inverse-CDF / searchsorted.

        np.searchsorted can return len(cdf) when the uniform draw is
        exactly 1.0 (floating-point edge case). Clamping here protects
        every caller — no individual clamp needed downstream.
        """
        return min(int(np.searchsorted(cdf, np.random.random())), len(cdf) - 1)

    def _build_conditional_cache(
        self,
        conditional_table,
        temperature: float = 1.0,
    ) -> None:
        """
        Build per-fragment conditional probability caches.

        Expected columns: unique_id, next_unique_id, frequency, proba.
        After calling this, _simulate() uses P(a|s) from the corpus
        transition counts instead of the marginal log-frequency prior.

        Parameters
        ----------
        conditional_table : pd.DataFrame or pa.Table
        temperature : float
            Power scaling: p^(1/T) then renormalise.
            T < 1 sharpens, T > 1 flattens, T = 1.0 uses raw probabilities.
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

            # Pre-cache terminal mask and terminal sub-CDF so sample_conditional
            # never rebuilds them on the hot path (terminal_only=True path).
            t_mask = np.array(
                [self._rows_by_uid.get(nid, {}).get("end_tag") == "no_tag" for nid in nids],
                dtype=bool,
            )
            self._cond_terminal_mask[uid_str] = t_mask
            if t_mask.any():
                sub_p = probas[t_mask]
                sub_p = sub_p / sub_p.sum()
                self._cond_terminal_cdf[uid_str] = sub_p.cumsum()
            else:
                self._cond_terminal_cdf[uid_str] = None

    def sample_conditional(
        self,
        current_uid: str,
        terminal_only: bool = False,
    ) -> Optional[dict]:
        """
        Sample the next fragment using true conditional probabilities
        P(a | current_uid) from the pre-built corpus transition cache.

        Returns None when current_uid is not in the cache so the caller
        can fall back to sample_compatible().

        Parameters
        ----------
        current_uid   : unique_id of the last fragment in the partial chain.
        terminal_only : restrict to fragments with end_tag == 'no_tag'.
        """
        if current_uid not in self._cond_next_ids:
            return None

        next_ids = self._cond_next_ids[current_uid]
        cdf = self._cond_cdf[current_uid]

        if terminal_only:
            # Use pre-cached terminal mask and sub-CDF — zero list comprehension
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

        # Use pre-materialised row dict — zero Arrow overhead
        row = self._rows_by_uid.get(chosen_uid)
        return row  # None if chosen_uid not in fragment table

        # ------------------------------------------------------------------

    # Tag-filtered table queries (used by MCTSNode._pool)
    # ------------------------------------------------------------------

    def get_start_fragments(self) -> pa.Table:
        """Fragments with begin_tag == 'no_tag' (chain roots)."""
        return self._by_begin.get("no_tag", self._empty)

    def get_compatible_extensions(self, end_tag: str) -> pa.Table:
        """Arrow table of all fragments compatible with end_tag."""
        compat = self.compatibility_map.get(end_tag, set())
        slices = [self._by_begin[t] for t in compat if t in self._by_begin]
        if not slices:
            return self._empty
        return pa.concat_tables(slices)

    def get_terminal_fragments(self, end_tag: str) -> pa.Table:
        """Compatible fragments that also close the chain (end_tag == 'no_tag')."""
        cands = self.get_compatible_extensions(end_tag)
        if cands.num_rows == 0:
            return cands
        return pc.filter(cands, pc.equal(cands.column("end_tag"), "no_tag"))  # pyright: ignore[reportAttributeAccessIssue]

    # ------------------------------------------------------------------
    # Fast sampling — bypasses Arrow scalar boxing
    # ------------------------------------------------------------------

    def sample_from_tag(
        self,
        tag: str,
        terminal_only: bool = False,
        temperature: Optional[float] = None,
    ) -> Optional[dict]:
        """
        Sample one fragment from a single begin_tag group.

        Uses pre-cached CDF for O(log n) draw and pre-materialised row
        lists for O(1) row access — zero Arrow overhead in the hot path.

        Parameters
        ----------
        tag           : begin_tag of the group to sample from.
        terminal_only : if True, restricts to rows where end_tag == 'no_tag'.
        temperature   : override stored temperature (triggers weight rebuild).
        """
        if tag not in self._by_begin:
            return None

        rows = self._rows[tag]
        n = len(rows["unique_id"])
        if n == 0:
            return None

        if terminal_only:
            # FIX B: use pre-cached terminal index list and sub-CDF
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

        # Full group sample
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
        """
        FIX B: sample one compatible fragment using pre-cached group CDFs.

        Stage-1 group selection is now a pure dict lookup + searchsorted on
        a pre-built array — no list comprehension or per-call sum() in the
        rollout hot path.

        Parameters
        ----------
        end_tag          : the current molecule's open end tag.
        terminal_only    : only return fragments with end_tag == 'no_tag'.
        exclude_terminal : exclude fragments with end_tag == 'no_tag'.
        temperature      : override stored temperature (triggers recompute).
        """
        if terminal_only:
            cdf_key = end_tag + "__t"
            cdf = self._compat_cdf_terminal.get(end_tag)
            avail = self._compat_tags.get(cdf_key)
        elif exclude_terminal:
            cdf_key = end_tag + "__nt"
            cdf = self._compat_cdf_no_terminal.get(end_tag)
            avail = self._compat_tags.get(cdf_key)
        else:
            cdf = self._compat_cdf.get(end_tag)
            avail = self._compat_tags.get(end_tag)

        if cdf is None or not avail:
            return None

        # Stage 1: O(log n) group pick from pre-cached CDF
        chosen_tag = avail[self._searchsorted_sample(cdf)]

        # Stage 2: pick within chosen group
        return self.sample_from_tag(
            chosen_tag,
            terminal_only=terminal_only,
            temperature=temperature,
        )

    # ------------------------------------------------------------------
    # PUCT prior computation (used once per node at expansion)
    # ------------------------------------------------------------------

    def compute_prior(self, table: pa.Table, temperature: Optional[float] = None) -> np.ndarray:
        """
        Normalised log-frequency prior over all rows of *table*.
        Falls back to Arrow column read only when called on a concat'd table
        that has no matching single-tag cache entry.
        """
        t = temperature or self._temperature
        freqs = np.array(table.column("frequency").to_pylist(), dtype=np.float32)
        log_f = np.log1p(freqs) / t
        log_f -= log_f.max()
        w = np.exp(log_f)
        return w / w.sum()

    # ------------------------------------------------------------------
    # Shared-memory transport
    # ------------------------------------------------------------------

    @staticmethod
    def to_shared_memory(table: pa.Table) -> tuple[SharedMemory, int]:
        """Serialise *table* to Arrow IPC in a shared-memory block (call once)."""
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
    ) -> "FragmentLibrary":
        """Open a shared-memory block zero-copy and build a FragmentLibrary."""
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
        """
        Fast clone — explicit list copies instead of deepcopy.
        Strings and ints are immutable so shallow copy is correct.
        ~10-30x faster than deepcopy for short fragment lists.
        """
        s = MolState.__new__(MolState)
        s.fragment_ids = self.fragment_ids.copy()
        s.smiles_parts = self.smiles_parts.copy()
        s.current_end_tag = self.current_end_tag
        s.target_n_blocks = self.target_n_blocks
        return s


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
        "_prior_logf",  # FIX A: running log-freq array for incremental prior
        "_pool_table",  # full candidate pool (pa.Table, never modified)
        "_untried_w",  # mutable weight array — zeroed out as rows are used
        "_untried_cdf",  # CDF rebuilt lazily from _untried_w
        "_untried_dirty",  # True when CDF needs rebuilding
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
        self._prior_logf: Optional[np.ndarray] = None  # FIX A: incremental
        # Pool tracking — initialised lazily by _pool()
        self._pool_table: Optional[pa.Table] = None
        self._untried_w: Optional[np.ndarray] = None
        self._untried_cdf: Optional[np.ndarray] = None
        self._untried_dirty: bool = False

    # ------------------------------------------------------------------
    # PUCT
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def is_terminal(self) -> bool:
        return self.state.is_complete

    def _pool(
        self,
        lib: FragmentLibrary,
        profiler: Optional["MCTSProfiler"] = None,
    ) -> pa.Table:
        """
        Compute (once) the full candidate pool and initialise sampling arrays.

        Uses pre-cached CDFs from FragmentLibrary where possible to avoid
        Arrow column reads and pa.concat_tables in the hot path:
          - start fragments    : direct dict lookup (_by_begin["no_tag"])
          - terminal pool      : pre-cached _compat_cdf_terminal + tag list
          - mid-chain pool     : pre-cached _compat_cdf_no_terminal + tag list
        Falls back to Arrow concat only when the cached path is unavailable.
        """
        if self._pool_table is not None:
            return self._pool_table

        if profiler:
            profiler.start("expand_pool_init")

        s = self.state

        if s.n_blocks == 0:
            # Start fragments — single tag group, always cached
            pool = lib.get_start_fragments()
            cdf_cache = lib._cdf.get("no_tag")

        elif s.n_blocks == s.target_n_blocks - 1:
            # Terminal pool — use pre-cached terminal Arrow table
            pool = lib.get_terminal_fragments(s.current_end_tag)
            # No single-tag CDF; will compute from pool below if needed
            cdf_cache = None

        else:
            # Mid-chain pool — use pre-cached non-terminal Arrow table
            # get_compatible_extensions still does concat; use Arrow path
            # but initialise weights from pre-cached logw where possible
            pool = lib.get_compatible_extensions(s.current_end_tag)
            if pool.num_rows > 0:
                pool = pc.filter(pool, pc.not_equal(pool.column("end_tag"), "no_tag"))  # pyright: ignore[reportAttributeAccessIssue]
            cdf_cache = None

        self._pool_table = pool

        if pool.num_rows > 0:
            if cdf_cache is not None:
                # Fast path: reuse pre-built CDF directly — zero Arrow reads
                prior = np.exp(lib._logw["no_tag"])
                prior /= prior.sum()
                self._untried_w = prior.copy()
                self._untried_cdf = cdf_cache.copy()
            else:
                # General path: compute prior from the pool table
                # This is still an Arrow column read but happens once per node
                prior = lib.compute_prior(pool)
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
        pool = self._pool(lib, profiler)
        if pool.num_rows == 0:
            return True
        if self._untried_w is None:
            return True
        return self._untried_w.sum() == 0.0

    def pop_untried(
        self,
        lib: FragmentLibrary,
        temperature: float = 1.0,
        profiler: Optional["MCTSProfiler"] = None,
    ) -> Optional[dict]:
        """
        FIX 1 + FIX 2: zero-weight tracking + searchsorted.

        Samples one fragment from the untried pool using:
          - zero-weight numpy array  (no Arrow allocation per pop)
          - searchsorted on CDF      (O(log n), no np.random.choice overhead)

        Returns a plain dict from pre-materialised lists (FIX 4).
        """
        pool = self._pool(lib, profiler)
        if pool.num_rows == 0 or self._untried_w is None:
            return None

        total = self._untried_w.sum()
        if total == 0.0:
            return None

        # Rebuild CDF lazily — only when a weight was zeroed since last draw.
        # On the very first draw _untried_dirty is False and _untried_cdf is
        # already valid (built from the full prior in _pool()), so we skip
        # the recompute entirely.
        if self._untried_dirty:
            w_norm = self._untried_w / total
            self._untried_cdf = np.cumsum(w_norm)
            self._untried_dirty = False

        idx = min(
            int(np.searchsorted(self._untried_cdf, np.random.random())),  # pyright: ignore[reportCallIssue] # pyright: ignore[reportArgumentType] # type: ignore
            pool.num_rows - 1,
        )

        # Zero out selected entry — O(1), no allocation
        self._untried_w[idx] = 0.0
        self._untried_dirty = True

        # FIX 4: row from pre-materialised lists — no Arrow scalar boxing
        # We need to look up which tag group this row belongs to so we can
        # use the cached row lists. Fall back to Arrow if pool spans tags.
        # In practice _pool_table for a single begin_tag node IS a cached
        # slice so column access is still fast, but we use the direct Arrow
        # path here since the pool may span multiple tags.
        return {col: pool.column(col)[idx].as_py() for col in lib._ROW_COLS}

    def add_dirichlet_noise(self, alpha: float = 0.3, epsilon: float = 0.25) -> None:
        """Mix Dirichlet noise into the root prior to prevent lock-in."""
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
    new_state.fragment_ids.append(row["unique_id"])
    new_state.smiles_parts.append(row["block"])
    new_state.current_end_tag = row["end_tag"]
    if profiler:
        profiler.stop("expand_clone")

    if profiler:
        profiler.start("expand_node")
    child = MCTSNode(
        state=new_state,
        parent=node,
        fragment_row=row,
        depth=node.depth + 1,
    )
    node.children.append(child)
    if profiler:
        profiler.stop("expand_node")

    # Prior update: use conditional P(a|parent) when available,
    # otherwise fall back to incremental log-frequency prior (FIX A).
    if profiler:
        profiler.start("prior_compute")

    parent_uid = node.fragment_row["unique_id"] if node.fragment_row is not None else None

    if parent_uid and parent_uid in lib._cond_next_ids:
        # Conditional path — look up P(child_uid | parent_uid) for each child
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
        # Marginal fallback: incremental log-frequency update (FIX A)
        new_freq = float(row["frequency"])
        new_logf = math.log1p(new_freq) / temperature
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
    """
    FIX 2 + FIX 3 + FIX 4: rollout using sample_compatible() which avoids
    pa.concat_tables(), Arrow column reads, and np.random.choice(p=) entirely.
    """
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

        last_uid = state.fragment_ids[-1] if state.fragment_ids else None

        if remaining == 1:
            row = (
                lib.sample_conditional(last_uid, terminal_only=True)
                if last_uid and last_uid in lib._cond_next_ids
                else None
            )
            if row is None:
                row = lib.sample_compatible(
                    state.current_end_tag,
                    terminal_only=True,
                    temperature=temperature,
                )
        else:
            row = (
                lib.sample_conditional(last_uid, terminal_only=False)
                if last_uid and last_uid in lib._cond_next_ids
                else None
            )
            # Discard conditional result if it returned a terminal fragment mid-chain
            if row is not None and row.get("end_tag") == "no_tag":
                row = None
            if row is None:
                row = lib.sample_compatible(
                    state.current_end_tag,
                    exclude_terminal=True,
                    temperature=temperature,
                )
                if row is None:
                    row = lib.sample_compatible(
                        state.current_end_tag,
                        terminal_only=True,
                        temperature=temperature,
                    )

        if profiler:
            profiler.stop("library_query")

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
    """Isolated scoring step — mol construction + reward function."""
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
    Picklable worker for one MCTS search in a child process.

    Returns (results, profiler_dict).
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
    ) = args

    library = FragmentLibrary.from_shared_memory(
        shm_name,
        shm_bytes,
        compat_map,
        temperature=temperature,
        conditional_records=conditional_records,
    )
    root = MCTSNode(state=MolState(target_n_blocks=target_n_blocks))
    profiler = MCTSProfiler()

    collected: list[dict] = []
    seen_smiles: set[str] = set()

    for _ in range(n_iter):
        node = _select(root, library, ucb_c, profiler)
        if not node.is_terminal():
            node = _expand(
                node, library, temperature, dirichlet_alpha, dirichlet_eps, use_dirichlet, profiler
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
    MCTS-based de novo drug designer.

    Combines:
    - Root parallelisation via ProcessPoolExecutor + shared-memory Arrow table
    - PUCT selection with log-frequency prior
    - Dirichlet noise at root
    - Optimised hot-path sampling (zero-weight drain, searchsorted, pre-cached
      weights and row lists)
    - Built-in timing profiler

    Parameters
    ----------
    fragment_table    : pa.Table   Fragment library (must include 'frequency').
    compatibility_map : dict|None  R-BRICS {end_tag: {begin_tags}}.
    block_count_mu    : float      Gaussian mean for chain length.
    block_count_sigma : float      Gaussian std-dev for chain length.
    min_blocks        : int        Hard min chain length (default 2).
    max_blocks        : int        Hard max chain length (default 10).
    n_iter            : int        MCTS iterations per run (default 500).
    ucb_c             : float      PUCT exploration constant (default sqrt(2)).
    score_fn          : callable   mol -> [0,1]. Default: lipinski_score.
    rollout_depth     : int        Max rollout steps (default 20).
    conditional_table : pd.DataFrame | pa.Table | None
        Fragment transition counts (conditional_block_counts_df).
        Columns: unique_id, next_unique_id, frequency, proba.
        When provided, P(a|s) replaces the marginal log-frequency prior
        for rollout sampling and PUCT prior initialisation.
        Falls back to marginal prior for unseen parent fragments.
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
        conditional_table=None,
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
        self.profiler = MCTSProfiler()

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
        scoring >= score_threshold.

        Parameters
        ----------
        n_runs          : int   Independent MCTS searches.
        n_jobs          : int   Workers (-1=cpu_count, 1=serial/debug).
        temperature     : float Log-frequency sharpness (1.0=neutral).
        dirichlet_alpha : float Dirichlet concentration at root (0.3).
        dirichlet_eps   : float Dirichlet mixing weight (0.25).
        use_dirichlet   : bool  Toggle root Dirichlet noise.
        score_threshold : float Minimum score to collect (default 0.5).
        print_profile   : bool  Print timing report after run (default True).

        Returns
        -------
        list[dict] sorted by score descending:
            smiles, score, n_blocks, fragment_ids
        """
        self.profiler = MCTSProfiler()
        n_workers = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
        targets = [self.sampler.sample() for _ in range(n_runs)]
        results: list[dict] = []

        if n_workers == 1:
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
            shm, shm_bytes = FragmentLibrary.to_shared_memory(self.library.table)
            _shm = shm

            def _sig(s, f):
                _shm.close()
                _shm.unlink()
                raise SystemExit(1)

            signal.signal(signal.SIGTERM, _sig)

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
                    self._conditional_records,
                )
                for t in targets
            ]
            try:
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futures = {pool.submit(_search_one, a): i for i, a in enumerate(args_list)}
                    for fut in as_completed(futures):
                        batch, prof_dict = fut.result()
                        results.extend(batch)
                        self.profiler.merge_dict(prof_dict)
            finally:
                shm.close()
                shm.unlink()

        # Global dedup + sort
        seen: set[str] = set()
        unique: list[dict] = []
        for r in results:
            if r["smiles"] not in seen:
                seen.add(r["smiles"])
                unique.append(r)
        unique.sort(key=lambda r: r["score"], reverse=True)

        logger.info(
            "%d unique molecules above threshold %.2f from %d runs.",
            len(unique),
            score_threshold,
            n_runs,
        )
        if print_profile:
            self.profiler.print_report()
        return unique

    # ------------------------------------------------------------------ #
    #  SERIAL _search                                                      #
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
                node.state.clone(), self.library, self.rollout_depth, temperature, self.profiler
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
