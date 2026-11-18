"""
Out-of-sample validation of the SMILES-enumeration saturation models (reviewer request).

The original fits (``smiles_generation_modelling.py``) use every point of every
replica and report in-sample R^2. Here each model is instead fitted on part of
the data and scored on data it never saw:

1. Truncated-trajectory extrapolation ("budget_<B>"): fit only on observations with
   ``cumulative_count <= B`` (all replicas pooled, as in the original fits), and
   evaluate on the later observations ``cumulative_count > B``.
2. Cross-replica ("replica"): fit on the full trajectory of replica 0 and evaluate
   on replicas 1 and 2, which are independent randomized-SMILES sampling runs.
3. In-sample reference ("full"): the original protocol, recomputed on the same
   molecules for a like-for-like comparison.

Metrics on the held-out points:
- r2_test: R^2 on held-out points only (undefined when the held-out points are flat).
- r2_full_from_train: R^2 over the full trajectory using parameters fitted on the
  training part only (directly comparable to Figure 2's in-sample R^2).
- nrmse_test: RMSE on held-out points divided by the observed final number of
  unique SMILES (plateau).
- plateau_rel_err: (prediction at the last sampled budget - observed plateau) /
  observed plateau.
- train_coverage: fraction of the final plateau already observed in the training
  part, i.e. how much extrapolation the task requires.
"""

import argparse
import multiprocessing as mp
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import tqdm

from smiles_blocks.files import PROCESSED_DATA_DIR, PROJ_ROOT, REPORTS_DIR
from smiles_blocks.modelling import FINAL_MODELS, ModelsRegistry

MODELS = {name: ModelsRegistry().models[name] for name in FINAL_MODELS}


def _metrics(model, popt, x_tr, y_tr, x_te, y_te, plateau, x_last):
    pred_te = model.predict(x_te, *popt)
    ss_res = np.sum((y_te - pred_te) ** 2)
    ss_tot = np.sum((y_te - y_te.mean()) ** 2)
    x_all = np.concatenate([x_tr, x_te])
    y_all = np.concatenate([y_tr, y_te])
    return {
        "r2_test": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "r2_full_from_train": model.r2_score(x_all, y_all, *popt),
        "nrmse_test": np.sqrt(np.mean((y_te - pred_te) ** 2)) / plateau,
        "plateau_rel_err": (model.predict(np.array([x_last]), *popt)[0] - plateau) / plateau,
    }


def _fit(model, x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, _ = model.fit(x, y, maxfev=10_000)  # per-model initial guess and bounds
    if not np.all(np.isfinite(popt)):
        raise RuntimeError("non-finite parameters")
    return popt


def evaluate_one(args):
    smiles, split, x, y, rep, budgets = args
    plateau = y.max()
    x_last = x.max()
    rows = []

    # (scheme, train mask, test mask)
    schemes = [("full", np.ones_like(x, bool), np.ones_like(x, bool))]
    for b in budgets:
        tr = x <= b
        te = ~tr
        if te.sum() > 0 and tr.sum() >= 4:  # 3-parameter models need >= 4 points
            schemes.append((f"budget_{b}", tr, te))
    if (rep == 0).any() and (rep != 0).any():
        schemes.append(("replica", rep == 0, rep != 0))

    for scheme, tr, te in schemes:
        cov = y[tr].max() / plateau
        for name, model in MODELS.items():
            row = {"smiles": smiles, "split": split, "scheme": scheme, "function": name,
                   "n_train": int(tr.sum()), "n_test": int(te.sum()),
                   "train_coverage": cov, "plateau": plateau, "fit_ok": False}
            try:
                popt = _fit(model, x[tr], y[tr])
                row["fit_ok"] = True
                row["alpha"], row["beta"] = popt[0], popt[1]
                row["gamma"] = popt[2] if len(popt) > 2 else np.nan
                if scheme == "full":
                    row["r2_full_from_train"] = model.r2_score(x, y, *popt)
                else:
                    row.update(_metrics(model, popt, x[tr], y[tr], x[te], y[te], plateau, x_last))
            except Exception:
                pass
            rows.append(row)
    return rows


def _tasks_from_table(df: pd.DataFrame, budgets):
    tasks = []
    for (smi, split), g in df.groupby(["smiles", "split"], sort=False):
        g = g.sort_values("cumulative_count")
        tasks.append((smi, split, g.cumulative_count.to_numpy(float),
                      g.nb_unique_smiles.to_numpy(float), g.replica.to_numpy(), budgets))
    return tasks


def run_per_partition(datafolder: Path, out: Path, n_molecules: int, seed: int,
                      budgets, nb_workers: int, parts=None):
    """Process partitions one at a time (memory-bounded, resumable).

    A uniform random fraction of molecules is drawn in each partition; with
    n_molecules <= 0, every molecule is used.
    """
    files = sorted(ds.dataset(str(datafolder), format="parquet").files)
    total = sum(pq.read_metadata(f).num_rows for f in files)  # rows, for proportional sampling
    if parts is not None:  # e.g. one SLURM array task per part
        wanted = {f"part-{i}.parquet" for i in parts}
        files = [f for f in files if Path(f).name in wanted]
    part_dir = out / "per_part"
    part_dir.mkdir(parents=True, exist_ok=True)
    cols = ["smiles", "split", "cumulative_count", "nb_unique_smiles", "replica"]

    use_all = n_molecules <= 0

    with mp.Pool(nb_workers) as pool:
        for i, f in enumerate(tqdm.tqdm(files, desc="Partitions")):
            target = part_dir / Path(f).name
            if target.exists():
                continue
            df = pq.read_table(f, columns=cols).to_pandas()
            if not use_all:
                smi = df.smiles.unique()
                # proportional to this partition's share of rows
                k = int(round(n_molecules * len(df) / total))
                rng = np.random.default_rng(seed + int(Path(f).stem.split("-")[1]))
                chosen = set(rng.choice(smi, size=min(k, len(smi)), replace=False))
                df = df[df.smiles.isin(chosen)]
            tasks = _tasks_from_table(df, budgets)
            del df
            rows = [r for rs in pool.imap_unordered(evaluate_one, tasks, chunksize=256) for r in rs]
            res = pd.DataFrame(rows)
            for c in ("scheme", "function", "split"):
                res[c] = res[c].astype("category")
            res.to_parquet(target, index=False)


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    def q(s, p):
        return s.quantile(p)

    g = res.groupby(["scheme", "function"], observed=True)
    out = g.agg(
        n=("fit_ok", "size"),
        fit_fail_rate=("fit_ok", lambda s: 1 - s.mean()),
        train_coverage_median=("train_coverage", "median"),
        r2_full_from_train_median=("r2_full_from_train", "median"),
        r2_full_from_train_mean=("r2_full_from_train", "mean"),
        r2_test_median=("r2_test", "median"),
        r2_test_q25=("r2_test", lambda s: q(s, 0.25)),
        nrmse_test_median=("nrmse_test", "median"),
        nrmse_test_q75=("nrmse_test", lambda s: q(s, 0.75)),
        plateau_abs_rel_err_median=("plateau_rel_err", lambda s: s.abs().median()),
        plateau_rel_err_median=("plateau_rel_err", "median"),
        within_10pct=("plateau_rel_err", lambda s: (s.abs() <= 0.10).mean() if s.notna().any() else np.nan),
    )
    return out.reset_index()


def main():
    os.chdir(PROJ_ROOT)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datafolder", default=str(PROCESSED_DATA_DIR / "modelling_data"))
    p.add_argument("--output_dir", default=str(REPORTS_DIR / "modeling_results" / "holdout"))
    p.add_argument("--n_molecules", type=int, default=20_000, help="<= 0 uses all molecules")
    p.add_argument("--budgets", type=int, nargs="+", default=[1_110, 11_110, 111_110])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nb_workers", type=int, default=mp.cpu_count())
    p.add_argument("--parts", type=int, nargs="+", default=None,
                   help="Only fit these part numbers (e.g. one SLURM array task each).")
    p.add_argument("--skip_summary", action="store_true",
                   help="Only write per-part results (summaries are built once all parts exist).")
    p.add_argument("--summary_only", action="store_true",
                   help="Skip fitting; summarise the existing per-part results.")
    a = p.parse_args()

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not a.summary_only:
        run_per_partition(Path(a.datafolder), out, a.n_molecules, a.seed, a.budgets,
                          a.nb_workers, a.parts)
    if a.skip_summary:
        return
    res = pd.read_parquet(out / "per_part", columns=[
        "smiles", "scheme", "function", "train_coverage", "fit_ok", "r2_full_from_train",
        "r2_test", "nrmse_test", "plateau_rel_err"])
    print(f"{res.smiles.nunique():,} molecules evaluated")
    res = res.drop(columns="smiles")

    summary = summarize(res)
    summary.to_csv(out / "holdout_summary.csv", index=False)

    # hard case: budgets where < 90% of the plateau had been observed at training time
    hard = res[res.scheme.astype(str).str.startswith("budget") & (res.train_coverage < 0.9)]
    summarize(hard).to_csv(out / "holdout_summary_unsaturated.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(summary.round(4).to_string())
    print("\nUnsaturated at cutoff (train_coverage < 0.9):")
    print(summarize(hard).round(4).to_string())


if __name__ == "__main__":
    main()
