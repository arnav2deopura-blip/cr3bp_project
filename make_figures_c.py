"""
Figures of Part C. Static PNGs; same palette and style as make_figures_b.py.

  python make_figures_c.py all --sund <apply_policy dir> --phys <apply_policy dir> --global <apply_policy dir> --out part_c/figures --label D_confirm_C
    bars       mean cost-gap decomposition against RK45-spec: Part B (D_confirm), Part C physical-time policy, Part C Sundman-time policy
    paradigm   the paradigm gap against the timescale range, fixed physical step vs Sundman fixed step (the T1 mechanism and its removal)
    wins       share of orbits where each policy (overhead included) is cheaper than each adaptive comparator, with 95 % intervals
    cells      the Sundman policy's share cheaper than RK45-eqr by Hill regime x horizon band
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import make_figures_b as fb  # noqa: E402  (palette and axis style; importing it sets the rcParams)
from orbit_ml_cr3bp import failure_study_3d as fs  # noqa: E402
from orbit_ml_cr3bp.audit_3d import clopper_pearson  # noqa: E402

COMPS = ["RK45_spec", "DOP853_spec", "RK45_cal", "DOP853_cal", "RK45_eqr", "DOP853_eqr"]
COMP_LABEL = {"RK45_spec": "RK45\nper-orbit oracle", "DOP853_spec": "DOP853\nper-orbit oracle", "RK45_cal": "RK45\ncertified rtol", "DOP853_cal": "DOP853\ncertified rtol",
              "RK45_eqr": "RK45\nequal risk", "DOP853_eqr": "DOP853\nequal risk"}
C_PHYS, C_SUND, C_B = "#2a78d6", "#1baf7a", "#898781"


def _led(d):
    p = d if d.endswith(".csv") else os.path.join(sorted(glob.glob(os.path.join(d, "ledger.csv")) or [os.path.join(d, "ledger.csv")])[-1])
    return pd.read_csv(p, low_memory=False)


def _decomp_means(led, kind, comp="RK45_spec"):
    dec = fs.decompose_sundman(led, comp) if kind == "sund" else fs.decompose(led, comp)
    ok = dec["total"].notna()
    names = ["P_s", "Q", "K", "O", "R"] if kind == "sund" else ["P", "S", "Q", "K", "O", "R"]
    return {n: float(dec.loc[ok, n].mean()) for n in names}, float(dec.loc[ok, "total"].mean()), int(ok.sum())


def bars(sund, phys, out, label, partb_decompose=None, comp="RK45_spec"):
    cols = [("Part B\nstate only, 300 trees", partb_decompose, "b"), ("Part C physical time\nperiapsis features", phys, "phys"), ("Part C Sundman time\nperiapsis features", sund, "sund")]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.9), sharey=True)
    colors = {"P": fb.TERM_COLORS["P"], "P_s": fb.TERM_COLORS["P"], "S": fb.TERM_COLORS["S"], "Q": fb.TERM_COLORS["Q"], "K": fb.TERM_COLORS["K"], "O": fb.TERM_COLORS["O"],
              "R": fb.TERM_COLORS["R"]}
    for ax, (title, led, kind) in zip(axes, cols):
        if led is None:
            ax.axis("off")
            continue
        means, tot, n = _decomp_means(led, "sund" if kind == "sund" else "phys", comp)
        run = 0.0
        names = list(means)
        for k, t in enumerate(names):
            v = means[t]
            ax.bar(k, v, bottom=run, width=0.62, color=colors[t], edgecolor=fb.SURFACE, linewidth=1.5)
            ax.text(k, max(run, run + v) + 0.03, f"{v:+.2f}", ha="center", va="bottom", fontsize=8.5, color=fb.INK)
            run += v
        ax.bar(len(names), tot, width=0.62, color=fb.INK, edgecolor=fb.SURFACE, linewidth=1.5)
        ax.text(len(names), max(tot, 0) + 0.03, f"{tot:+.2f}", ha="center", va="bottom", fontsize=8.5, color=fb.INK, fontweight="bold")
        ax.axhline(0, color=fb.AXIS, lw=0.8)
        ax.set_xticks(range(len(names) + 1))
        ax.set_xticklabels(["paradigm" if t in ("P", "P_s") else fb.TERM_NAMES[t] for t in names] + ["total"], fontsize=7.5, rotation=25, ha="right")
        ax.set_title(f"{title}\n(n = {n:,})", fontsize=9, loc="left")
        ax.yaxis.grid(True, color=fb.GRID, lw=0.5)
        ax.set_axisbelow(True)
        fb._style(ax)
    axes[0].set_ylabel(f"mean log10(cost / {comp.replace('_', '-')}), dex")
    fig.suptitle(f"Where the cost gap goes, by policy ({label}; orbits that met the tolerance)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(out, "bars.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def paradigm(sund, out, label, comp="RK45_spec", n_bins=12):
    d = sund[sund[f"{comp}_nfev"].notna() & sund["C_s_star"].notna() & sund["C_fix_star"].notna()]
    x = d["log_delta_phys"].to_numpy(dtype=float)
    gaps = {"fixed physical step": np.log10(d["C_fix_star"] / d[f"{comp}_nfev"]).to_numpy(), "fixed Sundman step": np.log10(d["C_s_star"] / d[f"{comp}_nfev"]).to_numpy()}
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    mid = 0.5 * (edges[1:] + edges[:-1])
    for (name, g), c in zip(gaps.items(), (C_B, C_SUND)):
        b = np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)
        med = np.array([np.median(g[b == i]) for i in range(n_bins)])
        lo = np.array([np.quantile(g[b == i], 0.25) for i in range(n_bins)])
        hi = np.array([np.quantile(g[b == i], 0.75) for i in range(n_bins)])
        ax.fill_between(mid, lo, hi, color=c, alpha=0.18, linewidth=0)
        ax.plot(mid, med, color=c, lw=1.9, marker="o", ms=3.5, label=name)
    ax.axhline(0, color=fb.AXIS, lw=0.8)
    ax.set_xlabel("log10 of the timescale range delta_phys")
    ax.set_ylabel(f"log10(oracle fixed-step cost / {comp.replace('_', '-')})")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.yaxis.grid(True, color=fb.GRID, lw=0.5)
    ax.set_axisbelow(True)
    fb._style(ax)
    ax.set_title(f"The time transformation removes the dependence on the timescale range ({label}; median and quartiles)", fontsize=9, loc="left")
    fig.tight_layout()
    p = os.path.join(out, "paradigm.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def wins(policies, out, label):
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    w = 0.8 / len(policies)
    for j, (name, led, color) in enumerate(policies):
        sh, lo, hi = [], [], []
        for c in COMPS:
            col = f"{c}_nfev"
            if col not in led.columns or led[col].notna().sum() == 0:
                sh.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                continue
            has = led[col].notna()
            k = int((led["meets_tol"].astype(bool) & has & (led["C_ml_tot"] < led[col])).sum())
            n = int(has.sum())
            a, b = clopper_pearson(k, n)
            sh.append(k / n); lo.append(a); hi.append(b)
        x = np.arange(len(COMPS)) + (j - (len(policies) - 1) / 2) * w
        sh, lo, hi = np.array(sh), np.array(lo), np.array(hi)
        ax.bar(x, sh, width=w * 0.92, color=color, label=name)
        ax.errorbar(x, sh, yerr=[sh - lo, hi - sh], fmt="none", ecolor=fb.INK, elinewidth=0.8, capsize=1.5)
        for xi, v in zip(x, sh):
            if np.isfinite(v):
                ax.text(xi, v + 0.025, f"{v:.0%}", ha="center", fontsize=7.2, color=fb.INK)
    ax.axhline(0.5, color=fb.AXIS, lw=0.8, ls="--")
    ax.set_xticks(range(len(COMPS)))
    ax.set_xticklabels([COMP_LABEL[c] for c in COMPS], fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("share of orbits where the policy is cheaper")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
    ax.yaxis.grid(True, color=fb.GRID, lw=0.5)
    ax.set_axisbelow(True)
    fb._style(ax)
    ax.set_title(f"Share of orbits where the policy met the tolerance at lower total cost than each adaptive solver ({label})", fontsize=9, loc="left")
    fig.tight_layout()
    p = os.path.join(out, "wins.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def cells(sund, out, label, comp="RK45_eqr"):
    d = sund.copy()
    has = d[f"{comp}_nfev"].notna()
    d["win"] = d["meets_tol"].astype(bool) & has & (d["C_ml_tot"] < d[f"{comp}_nfev"])
    d = d[has]
    regs = [r for r in fs.REGIMES if r in set(d["regime"])]
    hb = list(fs.H_LABELS)
    M = np.full((len(regs), len(hb)), np.nan)
    N = np.zeros((len(regs), len(hb)), int)
    for i, r in enumerate(regs):
        for j, h in enumerate(hb):
            g = d[(d["regime"] == r) & (d["h_band"] == h)]
            N[i, j] = len(g)
            if len(g) >= fs.MIN_CELL_N:
                M[i, j] = g["win"].mean()
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    im = ax.imshow(M, cmap=fb.SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(len(regs)):
        for j in range(len(hb)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0%}\nn={N[i, j]}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > 0.55 else fb.INK)
    ax.set_xticks(range(len(hb)))
    ax.set_xticklabels([f"{h} TU" for h in hb])
    ax.set_yticks(range(len(regs)))
    ax.set_yticklabels([f"regime {r}" for r in regs])
    ax.set_title(f"Sundman policy: share cheaper than {comp.replace('_', '-')} ({label})", fontsize=9, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    p = os.path.join(out, "cells.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["bars", "paradigm", "wins", "cells", "all"])
    ap.add_argument("--sund", required=True)
    ap.add_argument("--phys", required=True)
    ap.add_argument("--global", dest="glob", default=None)
    ap.add_argument("--partb_ledger", default="part_b/results/confirm/policy/ledger.csv.gz")
    ap.add_argument("--out", default="part_c/figures")
    ap.add_argument("--label", default="D_confirm_C")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sund, phys = _led(a.sund), _led(a.phys)
    glob_ = _led(a.glob) if a.glob else None
    partb = pd.read_csv(a.partb_ledger, low_memory=False) if os.path.exists(a.partb_ledger) else None
    if partb is not None:
        partb = fs.add_derived(partb) if "regime" not in partb.columns else partb
    outs = []
    if a.what in ("bars", "all"):
        outs.append(bars(sund, phys, a.out, a.label, partb_decompose=partb))
    if a.what in ("paradigm", "all"):
        outs.append(paradigm(sund, a.out, a.label))
    if a.what in ("wins", "all"):
        pol = [("Part B (D_confirm)", partb, C_B)] if partb is not None else []
        pol += [("Part C physical time", phys, C_PHYS), ("Part C Sundman time", sund, C_SUND)]
        if glob_ is not None:
            pol.append(("one global Sundman step", glob_, "#eda100"))
        outs.append(wins(pol, a.out, a.label))
    if a.what in ("cells", "all"):
        outs.append(cells(sund, a.out, a.label))
    print("\n".join(outs))


if __name__ == "__main__":
    main()
