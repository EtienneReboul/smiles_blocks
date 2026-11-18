"""
Figure: how well each saturation model, fitted on the first X % of the sampling
budget, predicts the number of draws needed to collect 95 % / 99 % of the plateau.

One row per target (95 %, 99 %):
- Predicted / observed draws to reach the target (median and IQR, log scale).
- Share of molecules whose predicted budget is within a factor 1.25 of the observed one.
- Share of the plateau actually collected when sampling stops at the predicted
  budget (median and IQR), against the target.

Reads budget_summary.csv from ``aggregate_holdout_sweep.py``.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

from smiles_blocks.files import FIGURES_DIR, REPORTS_DIR
from smiles_blocks.modelling import FINAL_MODELS, ModelsRegistry

_REG = ModelsRegistry().models
MODELS = {f: _REG[f].name for f in FINAL_MODELS}
COLORS = dict(zip(FINAL_MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PCT = FuncFormatter(lambda v, _: f"{v:g}%")
RATIO_TOL = 1.25


def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def budget_axis(ax, label):
    ax.set_xscale("log")
    ax.set_xlim(1, 100)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.xaxis.set_major_formatter(PCT)
    if label:
        ax.set_xlabel("Share of sampling budget used for fitting", color=INK)


def title(ax, letter, text):
    ax.set_title(f"{letter}   {text}", loc="left", fontsize=9, color=INK, fontweight="bold")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(REPORTS_DIR / "modeling_results" / "holdout_sweep" / "budget_summary.csv"))
    p.add_argument("--out", default=str(FIGURES_DIR / "modeling_results" / "modeling_budget_prediction.svg"))
    a = p.parse_args()

    b = pd.read_csv(a.input)
    qs = sorted(b.q.unique())

    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "svg.fonttype": "none"})
    fig, axes = plt.subplots(len(qs), 3, figsize=(10.5, 3.2 * len(qs)),
                             gridspec_kw={"wspace": 0.32, "hspace": 0.45}, squeeze=False)
    letters = iter("ABCDEFGHI")

    for row, q in enumerate(qs):
        last = row == len(qs) - 1
        axR, axW, axC = axes[row]
        for f in MODELS:
            g = b[(b.function == f) & (b.q == q)].sort_values("fraction")
            axR.fill_between(g.fraction, g.ratio_q25, g.ratio_q75, color=COLORS[f], alpha=0.18, lw=0)
            axR.plot(g.fraction, g.ratio_q50, color=COLORS[f], lw=2)
            axW.plot(g.fraction, 100 * g.within_ratio_tol, color=COLORS[f], lw=2)
            axC.fill_between(g.fraction, 100 * g.cov_q25, 100 * g.cov_q75, color=COLORS[f], alpha=0.18, lw=0)
            axC.plot(g.fraction, 100 * g.cov_q50, color=COLORS[f], lw=2)

        axR.axhline(1, color=INK2, lw=0.8)
        axR.axhspan(1 / RATIO_TOL, RATIO_TOL, color=INK2, alpha=0.06, lw=0)
        axR.set_yscale("log")
        axR.set_ylim(0.3, 20)
        axR.set_yticks([0.5, 1, 2, 5, 10])
        axR.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
        axR.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        axR.set_ylabel(f"Predicted / observed draws to {q}%", color=INK)
        title(axR, next(letters), f"Budget to reach {q}% of plateau")

        axW.set_ylim(0, 102)
        axW.yaxis.set_major_formatter(PCT)
        axW.set_ylabel(f"Molecules within ×{RATIO_TOL:g} of observed", color=INK)
        title(axW, next(letters), f"Budget accurate to ×{RATIO_TOL:g}")

        axC.axhline(q, color=INK2, lw=0.8)
        axC.set_ylim(80, 100.5)
        axC.yaxis.set_major_formatter(PCT)
        axC.set_ylabel("Plateau collected at predicted stop", color=INK)
        title(axC, next(letters), f"Stopping at predicted {q}% budget")

        for ax in (axR, axW, axC):
            budget_axis(ax, last)
            style_axis(ax)

    handles = [plt.Line2D([], [], color=COLORS[f], lw=2, label=MODELS[f]) for f in MODELS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=4,
               frameon=False, fontsize=8)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(Path(a.out).with_suffix(".png"), dpi=200, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
