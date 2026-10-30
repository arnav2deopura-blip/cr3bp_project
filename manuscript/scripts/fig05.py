"""Fig 5: (A) share cheaper than six adaptive settings for three Study C policies (+ Study D replication of the Sundman policy);
(B) Sundman policy vs equal-risk RK45 by Hill regime x horizon band (Study C confirmation ledger)."""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import style as S

comps = [("RK45_spec", "RK45\noracle"), ("RK45_cal", "RK45\ncertified"), ("RK45_eqr", "RK45\nequal-risk"),
         ("DOP853_spec", "DOP853\noracle"), ("DOP853_cal", "DOP853\ncertified"), ("DOP853_eqr", "DOP853\nequal-risk")]
pols = [("phys", "physical-time policy", S.C["fixed"], "s", "//"), ("global", "one global Sundman step", S.C["onescale"], "o", ".."),
        ("sund", "Sundman-time policy", S.C["sundman"], "o", "")]


def counts(rep):
    d = json.load(open(rep))["summary"]
    n = d["n_eval"]
    return n, {c: int(round(d[f"share_total_cheaper_{c}"] * n)) for c, _ in comps}


rows = []
fig, (a, b) = plt.subplots(1, 2, figsize=(S.FULL_W, 3.4), gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.3})
w = 0.26
for j, (name, lab, col, mk, hatch) in enumerate(pols):
    n, k = counts(S.path(f"part_c/results/confirm/{name}/apply_policy_report.json"))
    for i, (c, _) in enumerate(comps):
        p = k[c] / n
        lo, hi = S.cp_interval(k[c], n)
        x = i + (j - 1) * w
        a.bar(x, 100 * p, width=w, color=col, hatch=hatch, edgecolor="#000000", linewidth=0.5, label=lab if i == 0 else None)
        a.errorbar(x, 100 * p, yerr=[[100 * (p - lo)], [100 * (hi - p)]], fmt="none", ecolor="#000000", **S.ERR)
        rows.append({"dataset": "Study C confirmation", "policy": lab, "comparator": c, "k": k[c], "n": n, "share": p, "cp_lo": lo, "cp_hi": hi})
n, k = counts(S.path("part_d/results/confirm/d3_sund_apply_policy_report.json"))
for i, (c, _) in enumerate(comps):
    p = k[c] / n
    a.plot(i + w, 100 * p, "D", mfc="#FFFFFF", mec="#000000", mew=0.9, ms=5, zorder=5, label="Sundman-time policy, Study D replication" if i == 0 else None)
    lo, hi = S.cp_interval(k[c], n)
    rows.append({"dataset": "Study D replication", "policy": "Sundman-time policy", "comparator": c, "k": k[c], "n": n, "share": p, "cp_lo": lo, "cp_hi": hi})
S.refline(a, y=50)
a.set_xticks(range(len(comps)))
a.set_xticklabels(["Oracle", "Certified", "Equal-\nrisk"] * 2, fontsize=8.5)
for xc, lab in ((1, "RK45"), (4, "DOP853")):
    a.text(xc, -0.2, lab, transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=9, fontweight="bold")
a.axvline(2.5, color="#BBBBBB", lw=0.6)
a.set_ylabel("Policy cheaper and within tolerance (%)")
a.set_ylim(0, 100)
h, l = a.get_legend_handles_labels()
order = [l.index(x) for x in ("physical-time policy", "one global Sundman step", "Sundman-time policy", "Sundman-time policy, Study D replication")]
a.legend([h[i] for i in order], [l[i] for i in order], loc="upper left", fontsize=8, ncol=2, bbox_to_anchor=(0, 1.2), columnspacing=1.0, handlelength=1.6)
S.panel(a, "A", x=-0.12, y=1.2)

L = pd.read_csv(S.path("out_c/confirm/sund/apply_policy_20261027T013754Z/ledger.csv"), usecols=["regime", "h_band", "meets_tol", "C_ml_tot", "RK45_eqr_nfev"])
L["win"] = L["meets_tol"].astype(bool) & (L["C_ml_tot"] < L["RK45_eqr_nfev"])
regimes, hb = ["I", "II", "III", "IV", "V"], ["1-3", "3-8", "8-20"]
M = np.zeros((5, 3)); K = np.zeros((5, 3), int); N = np.zeros((5, 3), int)
for i, r in enumerate(regimes):
    for j, h in enumerate(hb):
        s = L[(L["regime"] == r) & (L["h_band"] == h)]
        K[i, j], N[i, j] = int(s["win"].sum()), len(s)
        M[i, j] = 100 * K[i, j] / N[i, j]
        rows.append({"dataset": "Study C confirmation (B)", "policy": "Sundman-time policy", "comparator": f"RK45_eqr | regime {r} | {h} TU",
                     "k": K[i, j], "n": N[i, j], "share": K[i, j] / N[i, j], "cp_lo": np.nan, "cp_hi": np.nan})
norm = Normalize(0, 100)
im = b.imshow(M, cmap="cividis", norm=norm, aspect="auto")
for i in range(5):
    for j in range(3):
        col = "#000000" if norm(M[i, j]) > 0.55 else "#FFFFFF"
        b.text(j, i - 0.13, (">99%" if (K[i, j] < N[i, j] and M[i, j] >= 99.5) else f"{M[i, j]:.0f}%"), ha="center", va="center", fontsize=9, color=col)
        b.text(j, i + 0.22, f"{K[i, j]:,}/{N[i, j]:,}", ha="center", va="center", fontsize=8, color=col)
b.set_xticks(range(3)); b.set_xticklabels(["1–3", "3–8", "8–20"])
b.set_yticks(range(5)); b.set_yticklabels(regimes)
b.set_xlabel("Horizon (TU)"); b.set_ylabel("Hill regime")
b.tick_params(length=0)
for sp in b.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=b, fraction=0.06, pad=0.03)
cb.set_label("Cheaper than equal-risk RK45 (%)", fontsize=9)
S.panel(b, "B", x=-0.2, y=1.2)
print("total wins", int(K.sum()), int(N.sum()))
S.save(fig, 5)
S.write_data(5, "shares", pd.DataFrame(rows))
