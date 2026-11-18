"""Hold-out sweep: fit each saturation model on the first X % of the sampling budget.

For every molecule and every X in 1..100, each replica's trajectory is truncated
to the observations with ``cumulative_count <= X/100 * (that replica's final
draw count)``. The truncated replicas are pooled (as in the original fits), the
model is fitted on them, and the fit is scored on the WHOLE trajectory, including
the observations it never saw. This shows how quickly each model converges
towards a good estimate as more sampling is spent.

The training set only changes when a cutoff crosses a grid point of the
log-spaced sampling grid, so many consecutive X share the same training set.
Fits are cached on the training set, which makes the sweep far cheaper than
100 x the number of models.

One input part -> one output part, so it runs as a SLURM array task:

    python jobs_scripts/smiles_generation_holdout_sweep.py \\
        --input data/processed/modelling_data/part-0.parquet \\
        --output data/processed/holdout_sweep/part-0.parquet

Output: one row per (molecule, fraction, function) with
- n_train: number of pooled observations used for fitting
- train_coverage: max unique SMILES in the training part / observed plateau
- plateau_rel_err: (prediction at the last sampled budget - plateau) / plateau
- r2_full_from_train: R^2 over the full trajectory with the truncated-fit params
- nrmse_test: RMSE on held-out observations / plateau (NaN at 100 %)
- fit_ok: whether the fit succeeded (fewer than n_params + 1 points counts as a failure)
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

from smiles_blocks.modelling import FINAL_MODELS, ModelsRegistry  # noqa: E402

FUNCTIONS = list(FINAL_MODELS)
MODELS = {k: ModelsRegistry().models[k] for k in FUNCTIONS}
METRICS = ["train_coverage", "plateau_rel_err", "r2_full_from_train", "nrmse_test"]

SCHEMA = pa.schema([
    ("smiles", pa.dictionary(pa.int32(), pa.string())),
    ("split", pa.dictionary(pa.int8(), pa.string())),
    ("function", pa.dictionary(pa.int8(), pa.string())),
    ("fraction", pa.int8()),
    ("n_train", pa.int16()),
    ("fit_ok", pa.bool_()),
    *[(m, pa.float32()) for m in METRICS],
])


def _fit_and_score(model, x, y, tr, plateau, x_last):
    """Return (fit_ok, plateau_rel_err, r2_full_from_train, nrmse_test)."""
    if tr.sum() < model.n_params + 1:
        return False, np.nan, np.nan, np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = model.fit(x[tr], y[tr], maxfev=10_000)  # per-model initial guess and bounds
            if not np.all(np.isfinite(popt)):
                return False, np.nan, np.nan, np.nan
            pred = model.predict(x, *popt)
            plateau_err = model.predict(np.array([x_last]), *popt)[0] / plateau - 1
    except Exception:
        return False, np.nan, np.nan, np.nan
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - pred) ** 2) / ss_tot if ss_tot > 0 else np.nan
    te = ~tr
    nrmse = np.sqrt(np.mean((y[te] - pred[te]) ** 2)) / plateau if te.any() else np.nan
    return True, plateau_err, r2, nrmse


def sweep_one(args):
    """Sweep all fractions and models for one molecule; returns column arrays."""
    smiles, split, x, y, rep, fractions = args
    plateau = y.max()
    x_last = x.max()
    # per-observation cutoff reference: final draw count of the observation's replica
    xmax_rep = {r: x[rep == r].max() for r in np.unique(rep)}
    ref = np.array([xmax_rep[r] for r in rep])

    n = len(fractions) * len(FUNCTIONS)
    out = {
        "fraction": np.empty(n, np.int8), "function": np.empty(n, object),
        "n_train": np.empty(n, np.int16), "fit_ok": np.empty(n, bool),
        **{m: np.empty(n, np.float32) for m in METRICS},
    }
    cache = {}
    i = 0
    for f in fractions:
        tr = x <= (f / 100.0) * ref
        key = tr.tobytes()
        cov = y[tr].max() / plateau if tr.any() else 0.0
        for name in FUNCTIONS:
            if (key, name) not in cache:
                cache[(key, name)] = _fit_and_score(MODELS[name], x, y, tr, plateau, x_last)
            ok, perr, r2, nrmse = cache[(key, name)]
            out["fraction"][i] = f
            out["function"][i] = name
            out["n_train"][i] = tr.sum()
            out["fit_ok"][i] = ok
            out["train_coverage"][i] = cov
            out["plateau_rel_err"][i] = perr
            out["r2_full_from_train"][i] = r2
            out["nrmse_test"][i] = nrmse
            i += 1
    out["smiles"] = smiles
    out["split"] = split
    return out


def load_tasks(path: Path, fractions):
    t = pq.read_table(path, columns=["smiles", "split", "cumulative_count",
                                     "nb_unique_smiles", "replica"])
    df = t.to_pandas()
    tasks = []
    for (smi, split), g in df.groupby(["smiles", "split"], sort=False):
        g = g.sort_values(["replica", "cumulative_count"])
        tasks.append((smi, split, g.cumulative_count.to_numpy(float),
                      g.nb_unique_smiles.to_numpy(float), g.replica.to_numpy(), fractions))
    return tasks


def to_table(results) -> pa.Table:
    reps = np.array([len(r["fraction"]) for r in results])
    cols = {
        "smiles": np.repeat([r["smiles"] for r in results], reps),
        "split": np.repeat([r["split"] for r in results], reps),
        "function": np.concatenate([r["function"] for r in results]).astype(str),
    }
    for c in ["fraction", "n_train", "fit_ok", *METRICS]:
        cols[c] = np.concatenate([r[c] for r in results])
    arrays = []
    for field in SCHEMA:
        a = pa.array(cols[field.name])
        if pa.types.is_dictionary(field.type):
            a = a.dictionary_encode().cast(field.type)
        else:
            a = a.cast(field.type)
        arrays.append(a)
    return pa.Table.from_arrays(arrays, schema=SCHEMA)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="One modelling_data part (parquet).")
    p.add_argument("--output", required=True, help="Output parquet for this part.")
    p.add_argument("--fraction-min", type=int, default=1)
    p.add_argument("--fraction-max", type=int, default=100)
    p.add_argument("--fraction-step", type=int, default=1)
    p.add_argument("--max-molecules", type=int, default=0, help="Debug: only the first N molecules.")
    p.add_argument("--nb-workers", type=int, default=mp.cpu_count())
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--logfile", default=None)
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler()] +
                        ([logging.FileHandler(a.logfile)] if a.logfile else []))
    log = logging.getLogger(__name__)

    out = Path(a.output)
    if out.exists() and not a.overwrite:
        log.info(f"{out} exists, skipping (use --overwrite to redo)")
        return
    out.parent.mkdir(parents=True, exist_ok=True)

    fractions = list(range(a.fraction_min, a.fraction_max + 1, a.fraction_step))
    t0 = time.time()
    tasks = load_tasks(Path(a.input), fractions)
    if a.max_molecules > 0:
        tasks = tasks[: a.max_molecules]
    log.info(f"{len(tasks):,} molecules, {len(fractions)} fractions, "
             f"{len(FUNCTIONS)} functions, {a.nb_workers} workers (loaded in {time.time() - t0:.0f}s)")

    with mp.Pool(a.nb_workers) as pool:
        results = pool.map(sweep_one, tasks, chunksize=32)
    log.info(f"fitted in {time.time() - t0:.0f}s")

    table = to_table(results)
    # write to a temp file then rename, so a task killed mid-write never leaves
    # a truncated part that a resubmission would skip
    tmp = out.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd", compression_level=6)
    tmp.rename(out)
    log.info(f"wrote {table.num_rows:,} rows to {out} in {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
