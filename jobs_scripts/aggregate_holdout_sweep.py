"""Aggregate the per-part hold-out sweep into summary tables.

Each array task of :mod:`smiles_generation_holdout_sweep` writes one parquet with
one row per (molecule, fraction, function). This reads them one model at a time
(~200 M rows each for all of MOSES) and writes:

- sweep_summary.csv: per (function, fraction) distribution of the plateau error,
  full-trajectory R^2 and training coverage.
- convergence_summary.csv: per (function, tolerance) distribution of the
  smallest fraction of the sampling budget from which the predicted plateau stays
  within the tolerance for every larger fraction ("stable convergence").
- convergence_per_molecule.parquet: that fraction for every molecule (git-ignored).

    python jobs_scripts/aggregate_holdout_sweep.py \\
        --input-dir data/processed/holdout_sweep \\
        --output-dir reports/modeling_results/holdout_sweep
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

from smiles_blocks.modelling import FINAL_MODELS  # noqa: E402

FUNCTIONS = list(FINAL_MODELS)
TOLERANCES = [0.20, 0.10, 0.05, 0.01]
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def stable_convergence(err: np.ndarray, fractions: np.ndarray, tol: float) -> np.ndarray:
    """Smallest fraction from which |err| <= tol holds for all larger fractions.

    err: (n_molecules, n_fractions), fractions ascending. NaN (failed fit) counts
    as outside the tolerance. Returns NaN for molecules never stably converged.
    """
    ok = np.abs(err) <= tol  # NaN -> False
    ok_from = np.flip(np.logical_and.accumulate(np.flip(ok, 1), axis=1), 1)
    first = ok_from.argmax(axis=1)
    out = fractions[first].astype(float)
    out[~ok_from.any(axis=1)] = np.nan
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default="data/processed/holdout_sweep")
    p.add_argument("--output-dir", default="reports/modeling_results/holdout_sweep")
    a = p.parse_args()

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset = ds.dataset(a.input_dir, format="parquet")
    print(f"{len(dataset.files)} parts")

    summaries, conv_rows, per_mol = [], [], []
    for f in FUNCTIONS:
        t = dataset.to_table(
            columns=["smiles", "split", "fraction", "fit_ok", "train_coverage",
                     "plateau_rel_err", "r2_full_from_train"],
            filter=pc.field("function") == f)
        df = t.to_pandas()
        fractions = np.sort(df.fraction.unique())
        n_frac = len(fractions)
        n_mol = len(df) // n_frac
        print(f"{f}: {n_mol:,} molecules x {n_frac} fractions")

        # ---- per-fraction distributions
        df["abs_err"] = df.plateau_rel_err.abs()
        g = df.groupby("fraction")
        s = pd.DataFrame({
            "function": f,
            "n": g.size(),
            "fit_fail_rate": 1 - g.fit_ok.mean(),
            "train_coverage_median": g.train_coverage.median(),
            # share of molecules whose observed plateau is already in the training part
            "plateau_reached": g.train_coverage.apply(lambda x: (x >= 1 - 1e-6).mean()),
            "within_5pct": g.abs_err.apply(lambda x: (x <= 0.05).mean()),
            "within_10pct": g.abs_err.apply(lambda x: (x <= 0.10).mean()),
            "abs_err_median": g.abs_err.median(),
        })
        for q in QUANTILES:
            s[f"plateau_rel_err_q{int(q * 100):02d}"] = g.plateau_rel_err.quantile(q)
        for q in (0.25, 0.5, 0.75):
            s[f"r2_full_q{int(q * 100):02d}"] = g.r2_full_from_train.quantile(q)
        summaries.append(s.reset_index())

        # ---- stable convergence per molecule
        # rows are molecule-major with fractions ascending within each molecule
        err = df.plateau_rel_err.to_numpy().reshape(n_mol, n_frac)
        frac_mat = df.fraction.to_numpy().reshape(n_mol, n_frac)
        assert (frac_mat == fractions).all(), "unexpected row order in sweep parts"
        mol = df[["smiles", "split"]].iloc[::n_frac].reset_index(drop=True)
        mol["function"] = f
        for tol in TOLERANCES:
            c = stable_convergence(err, fractions, tol)
            mol[f"conv_frac_{int(tol * 100)}pct"] = c
            cs = pd.Series(c)
            conv_rows.append({
                "function": f, "tolerance": tol, "n": n_mol,
                "never_converged": cs.isna().mean(),
                **{f"conv_frac_q{int(q * 100):02d}": cs.quantile(q) for q in QUANTILES},
                "conv_frac_mean": cs.mean(),
            })
        per_mol.append(mol)
        del df, t

    pd.concat(summaries).to_csv(out / "sweep_summary.csv", index=False)
    conv = pd.DataFrame(conv_rows)
    conv.to_csv(out / "convergence_summary.csv", index=False)
    pd.concat(per_mol).to_parquet(out / "convergence_per_molecule.parquet", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(conv.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
