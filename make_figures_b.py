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

LABEL = "D_explore"     # dataset named in the titles; set with --label
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
    fig.suptitle(f"Where the extra cost of the ML step-size policy comes from (mean over orbits that met the tolerance, dex; {LABEL})",
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
    ax.set_title(f"Even a perfect single step costs more where the trajectory's pace varies ({LABEL})", fontsize=9.5, loc="left")
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
                 f"(vs {comp.replace('_', ' ')}; {LABEL}, cells with n<{min_n} blank)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(out, "wheremap.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def dial(sweep_ledger_path, out, comp="RK45_spec"):
    d = pd.read_csv(sweep_ledger_path, low_memory=False)
    d = d[d[f"{comp}_nfev"].notna() & d["C_fix_star"].notna()].copy()
    d["paradigm"] = np.log10(d["C_fix_star"] / d[f"{comp}_nfev"])
    m = d[d["ml_ok"].fillna(False).astype(bool) & d["ml_evals"].notna()].copy()
    m["ml_int"] = np.log10(m["ml_evals"] / m[f"{comp}_nfev"])
    Ts = sorted(d["sim_time"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.6))
    for ax, (frame, col, title, color) in zip(axes[:2], [(d, "paradigm", "paradigm gap: best single step vs adaptive", "#2a78d6"),
                                                         (m, "ml_int", "ML integration gap (model cost excluded)", "#e87ba4")]):
        q = frame.groupby("sim_time")[col].quantile([0.25, 0.5, 0.75]).unstack()
        ax.fill_between(Ts, q[0.25].reindex(Ts), q[0.75].reindex(Ts), color=color, alpha=0.18, linewidth=0)
        ax.plot(Ts, q[0.5].reindex(Ts), color=color, lw=1.8, marker="o", ms=4)
        ax.axhline(0, color=AXIS, lw=0.8)
        for T in Ts:
            ax.text(T, q.loc[T, 0.5] + 0.06, f"{q.loc[T, 0.5]:+.2f}", ha="center", fontsize=8, color=INK)
        ax.set_xscale("log")
        ax.set_xticks(Ts)
        ax.set_xticklabels([f"{int(t)}" for t in Ts])
        ax.set_xlabel("horizon T (TU), same initial states at every T")
        ax.set_ylabel("dex (median, band = IQR)")
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        _style(ax)
        ax.set_title(title, fontsize=9.5, loc="left")
    v = d.groupby("sim_time")["violation"].mean().reindex(Ts)
    ax = axes[2]
    ax.plot(Ts, v.to_numpy() * 100, color="#eb6834", lw=1.8, marker="o", ms=4)
    for T, x in zip(Ts, v):
        ax.text(T, x * 100 + 0.4, f"{x:.1%}", ha="center", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xticks(Ts)
    ax.set_xticklabels([f"{int(t)}" for t in Ts])
    ax.set_ylim(0, max(12, v.max() * 100 * 1.25))
    ax.set_xlabel("horizon T (TU)")
    ax.set_ylabel("ML violation rate (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    _style(ax)
    ax.set_title("safety", fontsize=9.5, loc="left")
    fig.suptitle(f"Predictability dial: both gaps and the failure rate rise with the horizon (vs {comp.replace('_', ' ')}; {d['orbit_idx'].nunique()} initial states)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(out, "horizon_dial.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def stepover(e9_dir, out):
    t = pd.read_csv(os.path.join(e9_dir, "e9_by_stepover.csv"))
    g = pd.read_csv(os.path.join(e9_dir, "e9_guard.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    ax = axes[0]
    ax.bar(range(len(t)), t["violation_rate"] * 100, color="#2a78d6", width=0.62, edgecolor=SURFACE, linewidth=1.5)
    for i, v in enumerate(t["violation_rate"]):
        ax.text(i, v * 100 + 1.5, f"{v:.0%}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(t)))
    ax.set_xticklabels([f"{s.replace('(', '').replace(']', '').replace(', ', '-').replace('inf', '+')}\nn={n}"
                        for s, n in zip(t["step_over"], t["n"])], fontsize=7.5)
    ax.set_xlabel("step-over ratio  dt_used / tau_min  (shortest local timescale on the trajectory)")
    ax.set_ylabel("share of runs that violate the tolerance (%)")
    ax.set_ylim(0, 112)
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    _style(ax)
    ax.set_title("failure probability against step-over", fontsize=9.5, loc="left")
    ax = axes[1]
    ax.plot(g["false_alarm_rate"] * 100, g["catch_rate"] * 100, color=INK, lw=1.4, marker="o", ms=4)
    for k, x, y in zip(g["k"], g["false_alarm_rate"], g["catch_rate"]):
        ax.text(x * 100 + 1.2, y * 100 - 1.5, f"{k:g}", fontsize=8, color=INK2)
    ax.plot([0, 100], [0, 100], color=AXIS, lw=0.8, ls=(0, (3, 3)))
    ax.set_xlabel("false alarms: runs that would have met the tolerance but are flagged (%)")
    ax.set_ylabel("violations caught (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    _style(ax)
    ax.set_title("runtime guard: abort if dt / tau exceeds the labelled threshold", fontsize=9.5, loc="left")
    fig.tight_layout()
    p = os.path.join(out, "step_over.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def information(e3a_dir, out):
    s = pd.read_csv(os.path.join(e3a_dir, "e3a_sigma.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), sharey=True)
    for ax, eps in zip(axes, sorted(s["eps"].unique())):
        g = s[np.isclose(s["eps"], eps)].dropna(subset=["lambda_T"]).copy()
        g["bin"] = pd.qcut(g["lambda_T"], 3, labels=["low", "middle", "high"], duplicates="drop")
        res = g.groupby("bin", observed=True)["abs_resid_dex"].median()
        sg = g.groupby("bin", observed=True)["sigma_chosen"].quantile(0.9)
        x = np.arange(len(res))
        ax.bar(x - 0.18, res.to_numpy(), width=0.34, color="#2a78d6", edgecolor=SURFACE, linewidth=1.5)
        ax.bar(x + 0.18, sg.to_numpy(), width=0.34, color="#eb6834", edgecolor=SURFACE, linewidth=1.5)
        for xi, a, b in zip(x, res.to_numpy(), sg.to_numpy()):
            ax.text(xi - 0.18, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
            ax.text(xi + 0.18, b + 0.01, f"{b:.3f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b} lambda T" for b in res.index], fontsize=8)
        ax.set_title(f"perturbation {eps:g} DU  ({eps * 384400:.3g} km)", fontsize=9.5, loc="left")
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        _style(ax)
    axes[0].set_ylabel("dex")
    fig.suptitle("The label is stable under navigation-level perturbations: the model's error is not an information floor", x=0.01, ha="left", fontsize=10)
    fig.text(0.01, 0.01, "blue: the model's median |log10 prediction / label|     orange: 90th percentile of the label's spread under the perturbation "
             "(labels move in steps of 0.0108 dex)", fontsize=8, color=INK2, ha="left", va="bottom")
    for ax in axes:
        ax.set_ylim(0, 0.47)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    p = os.path.join(out, "information_floor.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["waterfall", "paradigm", "wheremap", "all", "dial", "stepover", "information"])
    ap.add_argument("--sweep_ledger")
    ap.add_argument("--e9")
    ap.add_argument("--e3a")
    ap.add_argument("--label", default="D_explore", help="dataset name shown in titles")
    ap.add_argument("--decompose")
    ap.add_argument("--ledger")
    ap.add_argument("--out", default="part_b/figures/explore")
    a = ap.parse_args()
    global LABEL
    LABEL = a.label
    os.makedirs(a.out, exist_ok=True)
    outs = []
    if a.what in ("waterfall", "all"):
        outs.append(waterfall(a.decompose, a.ledger, a.out))
    if a.what in ("paradigm", "all"):
        outs.append(paradigm(a.ledger, a.out))
    if a.what in ("wheremap", "all"):
        outs.append(wheremap(a.decompose, a.ledger, a.out))
    if a.what == "dial":
        outs.append(dial(a.sweep_ledger, a.out))
    if a.what == "stepover":
        outs.append(stepover(a.e9, a.out))
    if a.what == "information":
        outs.append(information(a.e3a, a.out))
    print("\n".join(outs))


if __name__ == "__main__":
    main()
