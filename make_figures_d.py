"""
Figures of Part D. Static PNGs; same palette and style as make_figures_b.py / make_figures_c.py.

  python make_figures_d.py all --root out_d/confirm --out part_d/figures --label D_confirm_D
    d1    ML policy against the physics rules: paired saving with 95 % intervals, and the median total cost of every arm
    d2    compiled wall-clock: throughput ratio by horizon band against three compiled adaptive solvers, and batch throughput against batch size
    d4    where the risk lands: violation rate with Clopper-Pearson intervals by free stratum and by the closest approach the trajectory actually made
    d5    the frozen policy on the JPL periodic-orbit families: violation rate per family, and its step prediction error against the oracle step
    d7    FTLE field and policy violations on the surface of section (exploratory)
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import make_figures_b as fb  # noqa: E402  (palette and axis style; importing it sets the rcParams)

C_ML, C_RULE, C_RK, C_DOP, C_MUTED = "#1baf7a", "#eb6834", "#2a78d6", "#eda100", "#898781"


def latest(root, step, name=None):
    c = sorted(d for d in glob.glob(os.path.join(root, step, f"{step}_*")) if os.path.isdir(d))
    if not c:
        raise SystemExit(f"no {step} run under {root}")
    return c[-1] if name is None else os.path.join(c[-1], name)


def fig_d1(root, out, label):
    r = json.load(open(latest(root, "d1", "d1_report.json")))
    arms = r["arms"]
    order = [a for a in ("sund", "lin33", "rule_THR", "rule_THP", "rule_THPD", "rule_TP", "rule_TH", "rule_T") if a in arms]
    pretty = {"sund": "ML: 25 trees", "lin33": "linear, 33 features", "rule_THR": "rule T,H,R (best)", "rule_THP": "rule T,H,tau", "rule_THPD": "rule T,H,tau,dC",
              "rule_TP": "rule T,tau", "rule_TH": "rule T,H", "rule_T": "rule T"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9))
    ax = axes[0]
    ys = np.arange(len(order))[::-1]
    for y, a in zip(ys, order):
        if a == "sund":
            ax.plot([0, 0], [y - .3, y + .3], color=C_ML, lw=2)
            continue
        q = r["pairwise_vs_ml"][a]
        lo, hi = q["ci95"]
        ax.errorbar(q["mean_log10_saving"], y, xerr=[[q["mean_log10_saving"] - lo], [hi - q["mean_log10_saving"]]], fmt="o", color=C_RULE if a != "lin33" else C_RK, capsize=3)
        ax.text(hi + 0.004, y, f"{q['mean_log10_saving']:+.3f}", va="center", fontsize=8.5, color=fb.INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([pretty[a] for a in order])
    ax.axvline(0, color=fb.AXIS, lw=1)
    ax.set_xlabel("ML saving (dex): mean log10 cost of the arm minus the ML policy's")
    ax.set_title(f"Paired cost, orbits that met the tolerance under both ({label})")
    fb._style(ax)
    ax = axes[1]
    med = [arms[a]["median_C_tot_met"] for a in order]
    oh = [arms[a]["overhead_rhs"] for a in order]
    ax.barh(ys, [m - o for m, o in zip(med, oh)], color=C_ML, height=0.6, label="integration")
    ax.barh(ys, oh, left=[m - o for m, o in zip(med, oh)], color=C_MUTED, height=0.6, label="model overhead")
    for y, m in zip(ys, med):
        ax.text(m + 4, y, f"{m:.0f}", va="center", fontsize=8.5, color=fb.INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([])
    ax.set_xlabel("median total cost per orbit that met the tolerance (RHS-equivalents)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fb._style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "d1.png"), dpi=170)
    plt.close(fig)


def fig_d2(root, out, label):
    r = json.load(open(latest(root, "d2", "d2_report.json")))
    t = r["D2_single_thread"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9))
    ax = axes[0]
    bands = ["h_band=1-3", "h_band=3-8", "h_band=8-20"]
    comps = [("DP54_PI_eqr", "DP5(4)-PI", C_RK), ("DP54_scipy_eqr", "DP5(4), SciPy controller", C_DOP), ("DOP853_eqr", "DOP853", C_RULE)]
    w = 0.26
    for k, (key, lab, col) in enumerate(comps):
        vals = [t[key]["by_group"][b]["throughput_ratio"] for b in bands] + [t[key]["throughput_ratio"]]
        ax.bar(np.arange(4) + (k - 1) * w, vals, w, color=col, label=lab)
    ax.axhline(1.0, color=fb.INK, lw=1)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["1-3 TU", "3-8 TU", "8-20 TU", "all"])
    ax.set_ylabel("throughput ratio, comparator time / policy time\n(> 1: the policy processes more orbits per second)")
    ax.set_title(f"Compiled single-thread throughput at equal risk ({label})")
    ax.legend(frameon=False, fontsize=8.5)
    fb._style(ax)
    ax = axes[1]
    b = r["batches"]
    sizes = sorted({int(k.split("|")[1]) for k in b})
    nthr = max(int(k.split("|")[2]) for k in b)
    for arm, lab, col in (("policy", "policy", C_ML), ("DP54_PI_eqr", "DP5(4)-PI", C_RK), ("DOP853_eqr", "DOP853", C_RULE)):
        for thr, ls in ((1, "--"), (nthr, "-")):
            ys = [b[f"{arm}|{s}|{thr}"]["orbits_per_s_median"] for s in sizes]
            ax.plot(sizes, ys, ls, marker="o", color=col, label=f"{lab}, {thr} thread{'s' if thr > 1 else ''}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("batch size (orbits drawn from the evaluation set)")
    ax.set_ylabel("orbits per second (median of repeats)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    fb._style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "d2.png"), dpi=170)
    plt.close(fig)


def _forest(ax, tab, col_rate, col_lo, col_hi, y, color, label, marker="o"):
    ax.errorbar(tab[col_rate], y, xerr=[tab[col_rate] - tab[col_lo], tab[col_hi] - tab[col_rate]], fmt=marker, color=color, capsize=2.5, label=label, ms=4.5)


def fig_d4(root, out, label):
    d = latest(root, "d4")
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    t = pd.read_csv(os.path.join(d, "table_strat_by_stratum.csv")).sort_values("rate")
    y = np.arange(len(t))
    _forest(ax, t, "rate", "cp_lo", "cp_hi", y + 0.12, C_ML, "stratified policy")
    t2 = pd.read_csv(os.path.join(d, "table_marginal_by_stratum.csv")).set_index("group").loc[t["group"]].reset_index()
    _forest(ax, t2, "rate", "cp_lo", "cp_hi", y - 0.12, C_MUTED, "one scale for all", marker="s")
    ax.plot(t["RK45_eqr_rate"], y, "x", color=C_RK, label="RK45, equal-risk rtol")
    ax.axvline(0.05, color=fb.AXIS, lw=1, ls="--")
    ax.axvline(0.07, color=C_RULE, lw=1, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g} (n={n})" for g, n in zip(t["group"], t["n"])], fontsize=7.5)
    ax.set_xlabel("violation rate with 95 % Clopper-Pearson interval")
    ax.set_title(f"By free stratum ({label}); dotted: the 7 % level")
    ax.legend(frameon=False, fontsize=8)
    fb._style(ax)
    ax = axes[1]
    order = ["<0.05", "0.05-0.17", "0.17-0.8", ">0.8"]
    for k, (arm, lab, col) in enumerate((("strat", "stratified policy", C_ML), ("marginal", "one scale for all", C_MUTED))):
        tt = pd.read_csv(os.path.join(d, f"table_{arm}_by_r2_band.csv")).set_index("group").loc[order].reset_index()
        _forest(ax, tt, "rate", "cp_lo", "cp_hi", np.arange(4) + (k - 0.5) * 0.2, col, lab, "o" if arm == "strat" else "s")
    tt = pd.read_csv(os.path.join(d, "table_strat_by_r2_band.csv")).set_index("group").loc[order].reset_index()
    ax.plot(tt["RK45_eqr_rate"], np.arange(4) + 0.3, "x", color=C_RK, label="RK45, equal-risk rtol")
    ax.axvline(0.05, color=fb.AXIS, lw=1, ls="--")
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels([f"min r2 {o} DU" for o in order])
    ax.set_xlabel("violation rate")
    ax.set_title("By the closest approach the run actually made\n(not free: needs the trajectory)")
    ax.legend(frameon=False, fontsize=8)
    fb._style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "d4.png"), dpi=170)
    plt.close(fig)


def fig_d5(root, out, label):
    d = latest(root, "d5")
    t = pd.read_csv(os.path.join(d, "table_by_group.csv")).sort_values("rate")
    rep = json.load(open(os.path.join(d, "d5_report.json")))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    ax = axes[0]
    y = np.arange(len(t))
    per = rep["D5"]["per_family"]
    lo = np.array([per[g]["cp95"][0] for g in t["group"]])
    hi = np.array([per[g]["cp95"][1] for g in t["group"]])
    ax.errorbar(t["rate"], y + 0.12, xerr=[t["rate"] - lo, hi - t["rate"]], fmt="o", color=C_ML, capsize=2.5, label="frozen Sundman policy")
    ax.plot(t["RK45_eqr_rate"], y - 0.12, "x", color=C_RK, label="RK45, equal-risk rtol")
    ax.plot(t["RK45_cal_rate"], y - 0.12, "+", color=C_DOP, label="RK45, certified rtol")
    ax.axvline(0.05, color=fb.AXIS, lw=1, ls="--")
    ax.axvline(0.10, color=C_RULE, lw=1, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels(t["group"])
    ax.set_xlabel("violation rate on in-range runs with a trustworthy reference")
    ax.set_title(f"JPL periodic-orbit families ({label}); family-balanced rate {rep['D5']['family_balanced_rate']:.2f}")
    ax.legend(frameon=False, fontsize=8)
    fb._style(ax)
    ax = axes[1]
    f = os.path.join(d, "d5_test_runs_with_labels.csv.gz")
    if os.path.exists(f):
        r = pd.read_csv(f)
        g = r.groupby("group")["log10_raw_over_oracle"].median().reindex(t["group"])
        ax.barh(y, g.to_numpy(), color=C_RULE, height=0.6)
        ax.axvline(0, color=fb.INK, lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.set_xlabel("median log10 (policy's raw ds / oracle ds): > 0 means the step is too long")
        ax.set_title("Prediction error against the oracle step")
        fb._style(ax)
    else:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "d5.png"), dpi=170)
    plt.close(fig)


def fig_d7(root, out, label):
    d = latest(root, "d7") if glob.glob(os.path.join(root, "d7")) else sorted(glob.glob(os.path.join(root, "d7_*")))[-1]
    files = sorted(glob.glob(os.path.join(d, "slice_C*.csv")))
    fig, axes = plt.subplots(1, len(files), figsize=(4.2 * len(files), 4.0), squeeze=False)
    for ax, f in zip(axes[0], files):
        df = pd.read_csv(f)
        ok = df[df["ref_ok"]]
        sc = ax.scatter(ok["x"], ok["vx"], c=ok["ftle"], s=6, cmap=fb.SEQ, rasterized=True)
        v = ok[ok["violation"]]
        ax.scatter(v["x"], v["vx"], s=9, facecolors="none", edgecolors="#c0392b", linewidths=0.6, label=f"policy violation ({len(v)})")
        ax.set_title(os.path.basename(f).replace("slice_", "").replace(".csv", ""))
        ax.set_xlabel("x (DU)")
        ax.set_ylabel("vx")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        fb._style(ax)
        fig.colorbar(sc, ax=ax, label="FTLE")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "d7.png"), dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["all", "d1", "d2", "d4", "d5", "d7"])
    ap.add_argument("--root", default="out_d/confirm")
    ap.add_argument("--explore_root", default="out_d/explore")
    ap.add_argument("--out", default="part_d/figures")
    ap.add_argument("--label", default="D_confirm_D")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fb.LABEL = a.label
    for name, fn, root in (("d1", fig_d1, a.root), ("d2", fig_d2, a.root), ("d4", fig_d4, a.root), ("d5", fig_d5, a.root), ("d7", fig_d7, a.explore_root)):
        if a.what in ("all", name):
            try:
                fn(root, a.out, a.label)
                print("wrote", os.path.join(a.out, name + ".png"))
            except SystemExit as e:
                print("skipped", name, e)


if __name__ == "__main__":
    main()
