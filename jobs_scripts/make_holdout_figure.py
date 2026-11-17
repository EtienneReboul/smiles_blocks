"""
Figure: out-of-sample validation of the SMILES-enumeration saturation models.

A. Per-molecule R^2: in-sample (original protocol) vs. held-out on independent
   sampling replicas (fit on replica 0, scored on replicas 1-2).
B. Relative error of the predicted plateau when fitting only on the first B draws.
C. Worked example: fits on early draws extrapolated over the full trajectory.

Reads the per-partition output of ``smiles_generation_holdout.py``.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from smiles_blocks.files import FIGURES_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from smiles_blocks.modelling import ModelsRegistry

# inverse2 is a reparameterisation of inverse (identical predictions) -> dropped
MODELS = {"exp": "Exponential", "inverse": "Inverse", "log": "Logarithmic", "sqrt": "Square root"}
COLORS = {"exp": "#2a78d6", "inverse": "#eb6834", "log": "#1baf7a", "sqrt": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def box_stats(s: pd.Series, label: str) -> dict:
    s = s.dropna()
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_numpy()
    return {"label": label, "whislo": q[0], "q1": q[1], "med": q[2], "q3": q[3],
            "whishi": q[4], "fliers": []}


def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def draw_boxes(ax, stats, positions, colors, hatch=None, width=0.32):
    b = ax.bxp(stats, positions=positions, widths=width, patch_artist=True, showfliers=False,
               medianprops={"color": INK, "lw": 1.2}, whiskerprops={"color": INK2, "lw": 0.8},
               capprops={"color": INK2, "lw": 0.8}, boxprops={"lw": 0.8})
    for patch, c in zip(b["boxes"], colors):
        patch.set_facecolor(c if hatch is None else "white")
        patch.set_edgecolor(c)
        if hatch:
            patch.set_hatch(hatch)
            patch.set_linewidth(1.2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(REPORTS_DIR / "modeling_results" / "holdout" / "per_part"))
    p.add_argument("--datafolder", default=str(PROCESSED_DATA_DIR / "modelling_data"))
    p.add_argument("--out", default=str(FIGURES_DIR / "modeling_results" / "modeling_holdout_validation.svg"))
    a = p.parse_args()

    res = pd.read_parquet(a.results, columns=["scheme", "function", "r2_full_from_train",
                                              "plateau_rel_err", "train_coverage"])
    res["scheme"] = res.scheme.astype(str)
    res["function"] = res.function.astype(str)
    res = res[res.function.isin(MODELS)]
    n_mol = (res.scheme == "full").sum() // len(MODELS)

    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "svg.fonttype": "none"})
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.45, wspace=0.3)
    axA, axB, axC = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # ---- A: in-sample vs held-out replicas -------------------------------------------
    keys = list(MODELS)
    def one_minus_r2(scheme, f):
        return 1 - res[(res.scheme == scheme) & (res.function == f)].r2_full_from_train

    full = [box_stats(one_minus_r2("full", f), "") for f in keys]
    rep = [box_stats(one_minus_r2("replica", f), "") for f in keys]
    med_r2 = [1 - r["med"] for r in rep]
    x = np.arange(len(keys))
    draw_boxes(axA, full, x - 0.19, [COLORS[f] for f in keys])
    draw_boxes(axA, rep, x + 0.19, [COLORS[f] for f in keys], hatch="////")
    for i, f in enumerate(keys):
        axA.text(x[i] + 0.19, rep[i]["whishi"] * 1.25, f"R² = {med_r2[i]:.4f}", ha="center",
                 va="bottom", fontsize=7, color=INK2)
    axA.set_xticks(x, [MODELS[f] for f in keys], color=INK)
    axA.set_yscale("log")
    axA.set_ylabel("1 − R² per molecule\n(lower is better)", color=INK)
    axA.set_ylim(1e-4, 5)
    axA.set_title("A   Fit on replica 0, scored on independent replicas 1–2",
                  loc="left", fontsize=9, color=INK, fontweight="bold")
    axA.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, fc=INK2, ec=INK2, label="In-sample (all replicas, original protocol)"),
        plt.Rectangle((0, 0), 1, 1, fc="white", ec=INK2, hatch="////", label="Held-out replicas (median R² shown)"),
    ], frameon=False, fontsize=7, loc="upper left")
    style_axis(axA)

    # ---- B: plateau error vs sampling budget ------------------------------------------
    budgets = sorted({int(s.split("_")[1]) for s in res.scheme.unique() if s.startswith("budget")})
    keysB = ["exp", "inverse", "log"]  # sqrt is off-scale (>+200%), reported in text
    offs = np.linspace(-0.26, 0.26, len(keysB))
    for j, f in enumerate(keysB):
        st = [box_stats(100 * res[(res.scheme == f"budget_{b}") & (res.function == f)].plateau_rel_err, "")
              for b in budgets]
        draw_boxes(axB, st, np.arange(len(budgets)) + offs[j], [COLORS[f]] * len(budgets), width=0.22)
    cov = [res[(res.scheme == f"budget_{b}") & (res.function == "exp")].train_coverage.median()
           for b in budgets]
    axB.set_xticks(range(len(budgets)),
                   [f"{b:,}\n{100 * c:.0f}%" for b, c in zip(budgets, cov)], color=INK, fontsize=7.5)
    axB.axhline(0, color=INK2, lw=0.8)
    axB.set_xlabel("Draws used for fitting\n(median share of plateau already observed)", color=INK)
    axB.set_ylabel("Predicted vs observed plateau (%)", color=INK)
    axB.set_ylim(-100, 80)
    axB.set_title("B   Extrapolating from early sampling", loc="left", fontsize=9,
                  color=INK, fontweight="bold")
    axB.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=COLORS[f], ec=COLORS[f], label=MODELS[f])
                        for f in keysB], frameon=False, fontsize=7, loc="upper right")
    style_axis(axB)

    # ---- C: worked example -------------------------------------------------------------
    ex = pq.read_table(Path(a.datafolder) / "part-0.parquet",
                       columns=["smiles", "cumulative_count", "nb_unique_smiles"]).to_pandas()
    plateaus = ex.groupby("smiles").nb_unique_smiles.max()
    smi = (plateaus - plateaus.median()).abs().idxmin()  # molecule with a typical plateau
    g = ex[ex.smiles == smi].sort_values("cumulative_count")
    xs, ys = g.cumulative_count.to_numpy(float), g.nb_unique_smiles.to_numpy(float)
    b = budgets[1] if len(budgets) > 1 else budgets[0]
    tr = xs <= b
    axC.scatter(xs[tr], ys[tr], s=10, color=INK, zorder=3, label=f"Fitted (≤ {b:,} draws)")
    axC.scatter(xs[~tr], ys[~tr], s=10, facecolor="white", edgecolor=INK, zorder=3,
                label="Held out")
    axC.axvline(b, color=INK2, lw=0.8, ls="--")
    grid = np.logspace(0, np.log10(xs.max()), 300)
    reg = ModelsRegistry().models
    for f in keys:
        m = reg[f]
        popt, _ = m.fit(xs[tr], ys[tr], p0=[ys[tr].max(), 1.0], maxfev=10_000)
        axC.plot(grid, m.predict(grid, *popt), color=COLORS[f], lw=1.6, label=MODELS[f])
    axC.set_xscale("log")
    axC.set_ylim(0, ys.max() * 2.3)
    axC.set_xlabel("Randomized SMILES drawn", color=INK)
    axC.set_ylabel("Unique SMILES", color=INK)
    axC.set_title("C   Example molecule (typical plateau)", loc="left", fontsize=9,
                  color=INK, fontweight="bold")
    axC.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=1)
    style_axis(axC)

    fig.text(0.01, -0.04, f"n = {n_mol:,} MOSES molecules. Boxes: IQR; whiskers: 5th–95th percentile.",
             fontsize=6.5, color=INK2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(Path(a.out).with_suffix(".png"), dpi=200, bbox_inches="tight")
    print("saved", a.out, "| example:", smi)


if __name__ == "__main__":
    main()
