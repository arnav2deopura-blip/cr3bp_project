"""
Figures for the failure study (Part B). Static PNGs for the repo/paper; colors follow the validated reference palette
(categorical slots 1-5 for the five cost-gap terms: adjacent CVD dE 9.1, normal-vision dE 19.6; three slots are below 3:1
contrast on the light surface, so every bar carries its printed value). One axis per panel, no dual axes, thin marks.

  python make_figures_b.py waterfall --decompose out_b/decompose_<ts> --out part_b/figures/explore
  python make_figures_b.py paradigm  --ledger out_b/policy_<ts>/ledger.csv --out ...
  python make_figures_b.py wheremap  --decompose ... --ledger ... --out ...
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
TERM_COLORS = {"P": "#2a78d6", "S": "#eb6834", "Q": "#1baf7a", "K": "#eda100", "O": "#e87ba4", "R": "#898781"}
TERM_NAMES = {"P": "paradigm", "S": "selection", "Q": "prediction", "K": "safety scale", "O": "overhead", "R": "rounding"}
DIVERGING = LinearSegmentedColormap.from_list("div", ["#184f95", "#3987e5", "#f0efec", "#e66767", "#a62a2a"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": False,
})


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)


def waterfall(dec_dir, ledger_path, out, comparators=("RK45_spec", "RK45_cal")):
    led = pd.read_csv(ledger_path, low_memory=False)
    cols = [("all orbits", lambda d: np.ones(len(d), bool)),
            ("min r2 < 0.05 DU", lambda d: (d["r2_band"] == "<0.05").to_numpy()),
            ("min r2 > 0.8 DU", lambda d: (d["r2_band"] == ">0.8").to_numpy())]
    fig, axes = plt.subplots(len(comparators), len(cols), figsize=(11.2, 3.6 * len(comparators)), sharey=True)
    terms = ["P", "S", "Q", "K", "O", "R"]
    for i, comp in enumerate(comparators):
        oc = pd.read_csv(os.path.join(dec_dir, f"{comp}_orbit_classes.csv"))
        oc = oc.merge(led[["row_id", "r2_band"]], on="row_id", how="left")
        oc = oc[oc["total"].notna()]
        for j, (title, sel) in enumerate(cols):
            ax = axes[i, j]
            g = oc[sel(oc)]
            means = {t: float(g[t].mean()) for t in terms}
            tot = float(g["total"].mean())
            run = 0.0
            for k, t in enumerate(terms):
                v = means[t]
                ax.bar(k, v, bottom=run, width=0.62, color=TERM_COLORS[t], edgecolor=SURFACE, linewidth=1.5)
                ax.text(k, max(run, run + v) + 0.04, f"{v:+.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
                if k < len(terms) - 1:
                    ax.plot([k + 0.31, k + 1 - 0.31], [run + v, run + v], color=AXIS, lw=0.8)
                run += v
            ax.bar(len(terms), tot, width=0.62, color=INK, edgecolor=SURFACE, linewidth=1.5)
            ax.text(len(terms), tot + 0.04, f"{tot:+.2f}", ha="center", va="bottom", fontsize=8.5, color=INK, fontweight="bold")
            ax.axhline(0, color=AXIS, lw=0.8)
            ax.set_xticks(range(len(terms) + 1))
            ax.set_xticklabels([TERM_NAMES[t] for t in terms] + ["total"], rotation=35, ha="right", fontsize=8)
            ax.yaxis.grid(True, color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            _style(ax)
            lab = comp.replace("_", " ")
            ax.set_title(f"vs {lab} - {title} (n={len(g)})", fontsize=9, loc="left", color=INK, pad=14)
            ax.set_ylim(-0.62, 1.85)
            if j == 0:
                ax.set_ylabel("mean contribution to log10(ML cost / adaptive cost)")
    fig.suptitle("Where the extra cost of the ML step-size policy comes from (mean over orbits that met the tolerance, dex; D_explore)",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(out, "waterfall.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def paradigm(ledger_path, out, comp="RK45_spec"):
    led = pd.read_csv(ledger_path, low_memory=False)
    d = led[led[f"{comp}_nfev"].notna() & led["C_fix_star"].notna()].copy()
    d["gap"] = np.log10(d["C_fix_star"] / d[f"{comp}_nfev"])
    x, y = d["log_delta_phys"].to_numpy(), d["gap"].to_numpy()
    b, a = np.polyfit(x, y, 1)
    from scipy.stats import spearmanr
    rho = spearmanr(x, y).statistic
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(x, y, s=5, color="#2a78d6", alpha=0.22, linewidths=0)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, a + b * xs, color=INK, lw=1.4)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.text(0.03, 0.95, f"fit: gap = {a:.2f} + {b:.2f} log10(delta)\nSpearman rho = {rho:.2f}   n = {len(d)}",
            transform=ax.transAxes, va="top", fontsize=8.5, color=INK2)
    ax.text(xs[-1], a + b * xs[-1], " fit", va="center", fontsize=8.5, color=INK)
    ax.text(x.min(), 0.03, "oracle fixed step costlier ↑   cheaper ↓", fontsize=8, color=MUTED, va="bottom")
    ax.set_xlabel("log10 of physical timescale range  delta = T max(1/tau) / ∫(1/tau) dt")
    ax.set_ylabel(f"paradigm gap  log10(C_fixed* / C_{comp.replace('_', ' ')})  (dex)")
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    _style(ax)
    ax.set_title("Even a perfect single step costs more where the trajectory's pace varies (D_explore)", fontsize=9.5, loc="left")
    fig.tight_layout()
    p = os.path.join(out, "paradigm_vs_delta.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def wheremap(dec_dir, ledger_path, out, comp="RK45_spec", min_n=20):
    led = pd.read_csv(ledger_path, low_memory=False)
    regimes = ["I", "II", "III", "IV", "V"]
    bands = ["<0.05", "0.05-0.17", "0.17-0.8", ">0.8"]
    d = led[led[f"{comp}_nfev"].notna() & led["C_fix_star"].notna()].copy()
    d["gap"] = np.log10(d["C_fix_star"] / d[f"{comp}_nfev"])
    panels = [
        ("median paradigm gap (dex)", lambda g: g["gap"].median(), DIVERGING, (-0.8, 0.8), "{:+.2f}", d),
        ("ML violation rate", lambda g: g["violation"].mean(), SEQ, (0, 0.3), "{:.1%}", led),
        ("share where a perfect fixed step is cheaper", lambda g: (g["C_fix_star"] < g[f"{comp}_nfev"]).mean(), SEQ, (0, 1), "{:.0%}", d),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9))
    for ax, (title, fn, cmap, (lo, hi), fmt, frame) in zip(axes, panels):
        M = np.full((len(regimes), len(bands)), np.nan)
        N = np.zeros((len(regimes), len(bands)), int)
        for i, r in enumerate(regimes):
            for j, b in enumerate(bands):
                g = frame[(frame["regime"] == r) & (frame["r2_band"] == b)]
                N[i, j] = len(g)
                if len(g) >= min_n:
                    M[i, j] = fn(g)
        ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
        for i in range(len(regimes)):
            for j in range(len(bands)):
                if np.isnan(M[i, j]):
                    ax.text(j, i, f"n={N[i, j]}" if N[i, j] else "empty", ha="center", va="center", fontsize=7.5, color=MUTED)
                else:
                    v = (M[i, j] - lo) / (hi - lo)
                    col = "#ffffff" if (cmap is SEQ and v > 0.55) or (cmap is DIVERGING and (v < 0.2 or v > 0.82)) else INK
                    ax.text(j, i, f"{fmt.format(M[i, j])}\nn={N[i, j]}", ha="center", va="center", fontsize=8, color=col)
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels(bands, fontsize=8)
        ax.set_yticks(range(len(regimes)))
        ax.set_yticklabels([f"regime {r}" for r in regimes], fontsize=8)
        ax.set_xlabel("closest approach to the secondary, min r2 (DU)")
        ax.set_title(title, fontsize=9.5, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-.5, len(bands), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(regimes), 1), minor=True)
        ax.grid(which="minor", color=SURFACE, lw=2)
        ax.tick_params(which="minor", length=0)
    fig.suptitle(f"Where each ceiling binds: Hill regime (I = closed necks ... V = no forbidden region) x closest lunar approach "
                 f"(vs {comp.replace('_', ' ')}; D_explore, cells with n<{min_n} blank)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(out, "wheremap.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["waterfall", "paradigm", "wheremap", "all"])
    ap.add_argument("--decompose")
    ap.add_argument("--ledger")
    ap.add_argument("--out", default="part_b/figures/explore")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    outs = []
    if a.what in ("waterfall", "all"):
        outs.append(waterfall(a.decompose, a.ledger, a.out))
    if a.what in ("paradigm", "all"):
        outs.append(paradigm(a.ledger, a.out))
    if a.what in ("wheremap", "all"):
        outs.append(wheremap(a.decompose, a.ledger, a.out))
    print("\n".join(outs))


if __name__ == "__main__":
    main()
