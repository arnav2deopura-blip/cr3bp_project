"""Fig 3: mean cost-gap decomposition (Eq 7) against RK45 at the oracle-tuned tolerance, Study B and Study C, with 95 % bootstrap intervals
(4,000 resamples of trajectories). Terms from the project's own failure_study_3d.decompose / decompose_sundman."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import style as S
from orbit_ml_cr3bp.failure_study_3d import decompose, decompose_sundman

rng = np.random.default_rng(42)


def boot_means(df, cols, B=4000):
    X = df[cols].to_numpy(float)
    n = len(X)
    means = np.empty((B, len(cols)))
    for i in range(0, B, 250):
        idx = rng.integers(0, n, size=(min(250, B - i), n))
        means[i:i + idx.shape[0]] = X[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5], axis=0)
    return X.mean(axis=0), lo, hi


LB = pd.read_csv(S.path("part_b/results/confirm/policy/ledger.csv.gz"))
dB = decompose(LB, "RK45_spec").dropna(subset=["total"])
LC = pd.read_csv(S.path("out_c/confirm/sund/apply_policy_20261027T013754Z/ledger.csv"))
dC = decompose_sundman(LC, "RK45_spec").dropna(subset=["total"])

terms_B = [("P", "paradigm"), ("S", "selection"), ("Q", "prediction"), ("K", "safety"), ("O", "overhead"), ("R", "rounding"), ("total", "total")]
terms_C = [("P_s", "paradigm"), (None, "selection"), ("Q", "prediction"), ("K", "safety"), ("O", "overhead"), ("R", "rounding"), ("total", "total")]
labels = {"paradigm": "Paradigm", "selection": "Selection", "prediction": "Prediction", "safety": "Safety scale", "overhead": "Overhead",
          "rounding": "Rounding", "total": "Total"}

fig, axes = plt.subplots(1, 2, figsize=(S.FULL_W, 3.0), sharey=True, gridspec_kw={"wspace": 0.08})
rows = []
for ax, d, terms, letter, name in ((axes[0], dB, terms_B, "A", "Study B, physical time"), (axes[1], dC, terms_C, "B", "Study C, Sundman time")):
    cols = [c for c, _ in terms if c]
    m, lo, hi = boot_means(d, cols)
    est = dict(zip(cols, zip(m, lo, hi)))
    for i, (c, key) in enumerate(terms):
        if c is None:
            ax.text(i, 0.03, "none", ha="center", va="bottom", fontsize=8, color="#555555")
            continue
        v, l, h = est[c]
        ax.bar(i, v, width=0.7, color=S.C[key], hatch=S.HATCH[key], edgecolor="#000000" if key != "total" else "#000000", linewidth=0.6)
        ax.errorbar(i, v, yerr=[[max(v - l, 0)], [max(h - v, 0)]], fmt="none", ecolor="#000000" if key != "total" else "#888888", **S.ERR)
        ax.text(i, max(v, h, 0) + 0.04, (f"{v:+.3f}" if abs(v) < 0.01 else f"{v:+.2f}"), ha="center", va="bottom", fontsize=8)
        rows.append({"study": name, "term": labels[key], "mean_dex": v, "ci95_lo": l, "ci95_hi": h, "n": len(d)})
    ax.axhline(0, color="#000000", lw=0.75)
    ax.set_xticks(range(len(terms)))
    ax.set_xticklabels([labels[k] for _, k in terms], rotation=35, ha="right")
    ax.set_title(f"{name} (n = {len(d):,})", fontsize=9, loc="left")
    S.panel(ax, letter, x=-0.14 if letter == "A" else -0.05, y=1.07)
axes[0].set_ylabel("Mean contribution (dex)")
axes[0].set_ylim(-0.15, 1.6)
print(pd.DataFrame(rows).round(3).to_string(index=False))
S.save(fig, 3)
S.write_data(3, "decomposition_means", pd.DataFrame(rows))
