"""Fig 4: Study B where-map, Hill regime x closest lunar approach. (A) median paradigm gap vs oracle-tuned RK45; (B) violation rate."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import style as S

L = pd.read_csv(S.path("part_b/results/confirm/policy/ledger.csv.gz"), usecols=["regime", "r2_band", "C_fix_star", "RK45_spec_nfev", "violation"])
L["gap"] = np.log10(L["C_fix_star"] / L["RK45_spec_nfev"])
regimes = ["I", "II", "III", "IV", "V"]
bands = ["<0.05", "0.05-0.17", "0.17-0.8", ">0.8"]
band_lab = ["<0.05", "0.05–0.17", "0.17–0.8", ">0.8"]
gap = L.dropna(subset=["gap"]).pivot_table(index="regime", columns="r2_band", values="gap", aggfunc=["median", "count"])
vio = L.pivot_table(index="regime", columns="r2_band", values="violation", aggfunc=["sum", "count"])
print("n gap", int(L["gap"].notna().sum()), "n viol", len(L))

fig, axes = plt.subplots(1, 2, figsize=(S.FULL_W, 3.1), gridspec_kw={"wspace": 0.45})
rows = []
for ax, letter, kind in ((axes[0], "A", "gap"), (axes[1], "B", "viol")):
    M = np.full((5, 4), np.nan)
    N = np.zeros((5, 4), int)
    K = np.zeros((5, 4), int)
    for i, r in enumerate(regimes):
        for j, b in enumerate(bands):
            if kind == "gap":
                n = gap["count"].get(b, {}).get(r, 0) if b in gap["count"] else 0
                n = 0 if pd.isna(n) else int(n)
                if n:
                    M[i, j] = gap["median"][b][r]
                N[i, j] = n
            else:
                n = vio["count"][b][r] if b in vio["count"] and r in vio["count"].index else 0
                n = 0 if pd.isna(n) else int(n)
                k = int(vio["sum"][b][r]) if n else 0
                N[i, j], K[i, j] = n, k
                if n:
                    M[i, j] = 100 * k / n
            rows.append({"panel": letter, "regime": r, "min_r2_band_DU": b, "value": M[i, j], "n": N[i, j], "violations": K[i, j] if kind == "viol" else None})
    norm = Normalize(-0.2, 1.2) if kind == "gap" else Normalize(0, 50)
    im = ax.imshow(M, cmap="cividis", norm=norm, aspect="auto")
    for i in range(5):
        for j in range(4):
            if N[i, j] == 0:
                ax.text(j, i, "–", ha="center", va="center", fontsize=8)
                continue
            v = M[i, j]
            txt = f"{v:+.2f}" if kind == "gap" else f"{v:.1f}%"
            col = "#000000" if norm(v) > 0.55 else "#FFFFFF"
            ax.text(j, i - 0.13, txt, ha="center", va="center", fontsize=8.5, color=col)
            ax.text(j, i + 0.22, f"n = {N[i, j]:,}", ha="center", va="center", fontsize=8, color=col)
    ax.set_xticks(range(4))
    ax.set_xticklabels(band_lab)
    ax.set_yticks(range(5))
    ax.set_yticklabels(regimes)
    ax.set_xlabel("Closest lunar approach (DU)")
    ax.set_ylabel("Hill regime")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Median paradigm gap (dex)" if kind == "gap" else "Violation rate (%)", fontsize=9)
    S.panel(ax, letter, x=-0.18, y=1.02)
S.save(fig, 4)
S.write_data(4, "cells", pd.DataFrame(rows))
