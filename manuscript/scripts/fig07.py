"""Fig 7: D4 conditional certificate on the Study D replication set. (A) violation rate per free stratum; (B) by the closest lunar approach
actually made. Stratified policy, one-scale policy and RK45 at the equal-risk tolerance of the stratified policy (95 % Clopper-Pearson)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import style as S

R = S.path("part_d/results/confirm")
st = pd.read_csv(f"{R}/d4_table_strat_by_stratum.csv").set_index("group")
mg = pd.read_csv(f"{R}/d4_table_marginal_by_stratum.csv").set_index("group")
st_b = pd.read_csv(f"{R}/d4_table_strat_by_r2_band.csv").set_index("group")
mg_b = pd.read_csv(f"{R}/d4_table_marginal_by_r2_band.csv").set_index("group")

order = ["I|far|1-3", "I|far|3-8", "I|far|8-20", "I|peri<0.05|1-3", "I|peri<0.05|h*",
         "II-III|far|1-3", "II-III|far|3-8", "II-III|far|8-20", "II-III|peri0.05-0.17|h*", "II-III|peri<0.05|1-3", "II-III|peri<0.05|3-8",
         "II-III|peri<0.05|8-20", "IV-V|far|1-3", "IV-V|far|3-8", "IV-V|far|8-20", "IV-V|peri0.05-0.17|h*", "pooled-small"]
assert set(order) == set(st.index)


def label(g):
    if g == "pooled-small":
        return "Pooled small cells"
    r, m, h = g.split("|")
    m = {"far": "far", "peri<0.05": "peri <0.05", "peri0.05-0.17": "peri 0.05–0.17"}[m]
    h = "merged" if h == "h*" else h.replace("-", "–") + " TU"
    return f"{r.replace('-', '–')} · {m} · {h}"


def series(df, rate, lo, hi, idx):
    r = 100 * df.loc[idx, rate].to_numpy()
    return r, r - 100 * df.loc[idx, lo].to_numpy(), 100 * df.loc[idx, hi].to_numpy() - r


fig = plt.figure(figsize=(S.FULL_W, 5.4))
a = fig.add_axes([0.30, 0.10, 0.40, 0.84])
b = fig.add_axes([0.80, 0.42, 0.19, 0.52])
y = np.arange(len(order))
specs = [("stratified policy", st, ("rate", "cp_lo", "cp_hi"), S.C["sundman"], "o", S.C["sundman"], -0.25),
         ("one scale for all", mg, ("rate", "cp_lo", "cp_hi"), S.C["onescale"], "o", "white", 0.0),
         ("RK45, equal-risk", st, ("RK45_eqr_rate", "RK45_eqr_cp_lo", "RK45_eqr_cp_hi"), S.C["rk45"], "^", S.C["rk45"], 0.25)]
rows = []
for lab, df, cols, col, mk, fc, off in specs:
    r, el, eh = series(df, *cols, order)
    a.errorbar(r, y + off, xerr=[el, eh], fmt=mk, color=col, mfc=fc, mec=col, ms=4.5, label=lab, **S.ERR)
    for g, v, l, h in zip(order, r, r - el, r + eh):
        rows.append({"panel": "A", "group": g, "series": lab, "rate_pct": v, "cp_lo_pct": l, "cp_hi_pct": h, "n": int(st.loc[g, "n"])})
a.axvline(7, color="#555555", lw=0.75, ls=(0, (4, 3)))
a.text(7.3, -0.9, "7% test level", fontsize=8, color="#555555", va="center")
a.set_yticks(y)
a.set_yticklabels([f"{label(g)} (n = {int(st.loc[g, 'n']):,})" for g in order], fontsize=8)
a.invert_yaxis()
a.set_xlim(0, 52)
a.set_xlabel("Violation rate (%)")
a.legend(loc="lower right", fontsize=8)
fig.text(0.005, 0.955, "A", fontsize=12, fontweight="bold")

bands = ["<0.05", "0.05-0.17", "0.17-0.8", ">0.8"]
yb = np.arange(len(bands))
specs_b = [("stratified policy", st_b, ("rate", "cp_lo", "cp_hi"), S.C["sundman"], "o", S.C["sundman"], -0.25),
           ("one scale for all", mg_b, ("rate", "cp_lo", "cp_hi"), S.C["onescale"], "o", "white", 0.0),
           ("RK45, equal-risk", st_b, ("RK45_eqr_rate", "RK45_eqr_cp_lo", "RK45_eqr_cp_hi"), S.C["rk45"], "^", S.C["rk45"], 0.25)]
for lab, df, cols, col, mk, fc, off in specs_b:
    r, el, eh = series(df, *cols, bands)
    b.errorbar(r, yb + off, xerr=[el, eh], fmt=mk, color=col, mfc=fc, mec=col, ms=4.5, **S.ERR)
    for g, v, l, h in zip(bands, r, r - el, r + eh):
        rows.append({"panel": "B", "group": g, "series": lab, "rate_pct": v, "cp_lo_pct": l, "cp_hi_pct": h, "n": int(st_b.loc[g, "n"])})
b.set_yticks(yb)
b.set_yticklabels([f"{g.replace('-', '–')}\n(n = {int(st_b.loc[g, 'n']):,})" for g in bands], fontsize=8)
b.invert_yaxis()
b.set_xlim(0, 16)
b.set_xlabel("Violation rate (%)")
b.set_ylabel("Closest lunar approach (DU)")
fig.text(0.715, 0.955, "B", fontsize=12, fontweight="bold")
S.save(fig, 7)
S.write_data(7, "rates", pd.DataFrame(rows))
