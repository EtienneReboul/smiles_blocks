"""
Figure: convergence of the saturation models as more of the sampling budget is used.

A. Predicted vs observed plateau (median and IQR across molecules) as a function
   of the share of the sampling budget used for fitting.
B. Share of molecules whose predicted plateau is within +/-10 %, with the median
   share of the plateau already observed at that budget for reference.
C. Cumulative distribution of the budget needed for STABLE convergence: the
   smallest share from which the prediction stays within +/-10 % for every
   larger share.

Reads the outputs of ``aggregate_holdout_sweep.py``.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from smiles_blocks.files import FIGURES_DIR, REPORTS_DIR

MODELS = {"exp": "Exponential", "inverse": "Inverse", "log": "Logarithmic", "sqrt": "Square root"}
COLORS = {"exp": "#2a78d6", "inverse": "#eb6834", "log": "#1baf7a", "sqrt": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PCT = FuncFormatter(lambda v, _: f"{v:g}%")


def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def budget_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(1, 100)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlabel("Share of sampling budget used for fitting", color=INK)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default=str(REPORTS_DIR / "modeling_results" / "holdout_sweep"))
    p.add_argument("--out", default=str(FIGURES_DIR / "modeling_results" / "modeling_holdout_sweep.svg"))
    p.add_argument("--tolerance", type=int, default=10, help="Convergence tolerance in %% (panels A/C).")
    p.add_argument("--tolerance-b", type=int, default=10, choices=[5, 10],
                   help="Accuracy tolerance in %% for panel B.")
    a = p.parse_args()

    d = Path(a.input_dir)
    s = pd.read_csv(d / "sweep_summary.csv")
    conv = pd.read_parquet(d / "convergence_per_molecule.parquet",
                           columns=["function", f"conv_frac_{a.tolerance}pct"])
    col = f"conv_frac_{a.tolerance}pct"
    n_mol = int(s.n.max())

    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "svg.fonttype": "none"})
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(10.5, 3.4), gridspec_kw={"wspace": 0.32})

    # ---- A: signed plateau error, median + IQR (sqrt off-scale) -------------------------
    for f in ["exp", "inverse", "log"]:
        g = s[s.function == f].sort_values("fraction")
        axA.fill_between(g.fraction, 100 * g.plateau_rel_err_q25, 100 * g.plateau_rel_err_q75,
                         color=COLORS[f], alpha=0.18, lw=0)
        axA.plot(g.fraction, 100 * g.plateau_rel_err_q50, color=COLORS[f], lw=2, label=MODELS[f])
    axA.axhline(0, color=INK2, lw=0.8)
    axA.axhspan(-a.tolerance, a.tolerance, color=INK2, alpha=0.06, lw=0)
    budget_axis(axA)
    axA.set_ylim(-75, 75)
    axA.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+g}%" if v else "0%"))
    axA.set_ylabel("Predicted vs observed plateau", color=INK)
    axA.set_title("A   Plateau estimate (median, IQR)", loc="left", fontsize=9, color=INK,
                  fontweight="bold")
    style_axis(axA)

    # ---- B: share within tolerance -------------------------------------------------------
    wcol = f"within_{a.tolerance_b}pct"
    for f in MODELS:
        g = s[s.function == f].sort_values("fraction")
        axB.plot(g.fraction, 100 * g[wcol], color=COLORS[f], lw=2, label=MODELS[f])
    budget_axis(axB)
    axB.set_ylim(0, 102)
    axB.yaxis.set_major_formatter(PCT)
    axB.set_ylabel(f"Molecules within ±{a.tolerance_b}% of plateau", color=INK)
    axB.set_title(f"B   Accurate to ±{a.tolerance_b}%", loc="left", fontsize=9, color=INK,
                  fontweight="bold")
    style_axis(axB)

    # ---- C: ECDF of budget needed for stable convergence ---------------------------------
    grid = np.arange(1, 101)
    for f in MODELS:
        c = conv.loc[conv.function == f, col].to_numpy()
        ecdf = np.array([(c <= x).mean() for x in grid])  # NaN (never) counts as not converged
        axC.step(grid, 100 * ecdf, where="post", color=COLORS[f], lw=2, label=MODELS[f])
    budget_axis(axC)
    axC.set_ylim(0, 102)
    axC.yaxis.set_major_formatter(PCT)
    axC.set_ylabel(f"Molecules stably within ±{a.tolerance}%", color=INK)
    axC.set_title("C   Budget needed to converge", loc="left", fontsize=9, color=INK,
                  fontweight="bold")
    style_axis(axC)

    handles = [plt.Line2D([], [], color=COLORS[f], lw=2, label=MODELS[f]) for f in MODELS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=4,
               frameon=False, fontsize=8)
    print(f"n = {n_mol:,} molecules")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(Path(a.out).with_suffix(".png"), dpi=200, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
