"""Fig 1: one trajectory with a close lunar pass (A) and its local timescale with the two step sizes (B).

Trajectory choice (fixed rule, no visual selection): Study C confirmation ledger, libration-point branch, closest lunar approach below 0.05 DU,
exactly one flyby, horizon 3-8 TU, uncensored labels for physical RK4 and Sundman RK4; among those, the trajectory with the median timescale
range. It is re-integrated with the reference integrator (DOP853, 1e-13, max step 0.01, dense output).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import style as S
from orbit_ml_cr3bp.config import MU
from orbit_ml_cr3bp.physics_3d import reference_trajectory_dense_3d, libration_point_location
from orbit_ml_cr3bp.diagnostics_3d import sample_reference, tau_local
from orbit_ml_cr3bp.sundman_3d import _g_only, SUNDMAN_TAU_C

L = pd.read_csv(S.path("out_c/confirm/sund/apply_policy_20261027T013754Z/ledger.csv"))
m = ((L["sampling_branch"] == "lagrange") & (L["min_r2"] < 0.05) & (L["n_flyby"] == 1) & L["sim_time"].between(3, 8)
     & (L["rk4_success"] == 1) & (L["rk4s_success"] == 1) & (~L["rk4_censored"].astype(bool)) & (~L["rk4s_censored"].astype(bool)))
cand = L[m].sort_values("delta_phys").reset_index(drop=True)
row = cand.iloc[len(cand) // 2]
state = row[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(float)
T = float(row["sim_time"])
d_phys = float(row["dstar_rk4"])
ds = float(row["dt_rk4s"])
print(f"{len(cand)} candidates; row_id {int(row['row_id'])}, T = {T:.3f} TU, min r2 = {row['min_r2']:.4f} DU, delta = {row['delta_phys']:.1f}, "
      f"d* = {d_phys:.4g} TU, ds* = {ds:.4g}")

sol, reached = reference_trajectory_dense_3d(state, T, MU)
assert reached
ts, Y = sample_reference(sol)
r1 = np.sqrt((Y[0] + MU) ** 2 + Y[1] ** 2 + Y[2] ** 2)
r2 = np.sqrt((Y[0] + MU - 1) ** 2 + Y[1] ** 2 + Y[2] ** 2)
tau = tau_local(r1, r2, MU)
g = np.array([_g_only(Y[:, i].copy(), MU, SUNDMAN_TAU_C) for i in range(Y.shape[1])])
h_s = g * ds

fig, (a, b) = plt.subplots(1, 2, figsize=(S.FULL_W, 3.3), gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.32})
# (A) rotating frame: zoom on the lunar region, with an overview inset of the whole frame
a.plot(Y[0], Y[1], color=S.C["sundman"], lw=1.0)
a.plot(Y[0, 0], Y[1, 0], "o", color=S.C["sundman"], ms=4)
a.plot([1 - MU], [0], "o", color="#555555", ms=5)
a.text(1 - MU, -0.025, "Moon", ha="center", va="top", fontsize=9)
for p in ("L1", "L2"):
    xl, yl = libration_point_location(MU, p)
    a.plot(xl, yl, "x", color="#000000", ms=5, mew=1.0)
    a.text(xl, yl - 0.025, p[0] + "$_{" + p[1] + "}$", fontsize=9, ha="center", va="top")
a.annotate("start", (Y[0, 0], Y[1, 0]), xytext=(-6, -12), textcoords="offset points", ha="right", fontsize=8, color=S.C["sundman"])
pad = 0.06
xlo, xhi = min(Y[0].min(), 0.83) - pad, max(Y[0].max(), 1.16) + pad
ylo, yhi = Y[1].min() - pad, Y[1].max() + pad
a.set_xlim(xlo, xhi)
a.set_ylim(ylo, yhi)
a.set_aspect("equal", adjustable="datalim")
a.set_xlabel("x (DU)")
a.set_ylabel("y (DU)")
ins = a.inset_axes([0.0, 0.66, 0.42, 0.34])
ins.plot([-MU], [0], "o", color="#555555", ms=4)
ins.plot([1 - MU], [0], "o", color="#555555", ms=2.5)
for p in ("L1", "L2", "L3", "L4", "L5"):
    xl, yl = libration_point_location(MU, p)
    ins.plot(xl, yl, "x", color="#000000", ms=3, mew=0.7)
ins.plot(Y[0], Y[1], color=S.C["sundman"], lw=0.6)
ins.add_patch(plt.Rectangle((xlo, ylo), xhi - xlo, yhi - ylo, fill=False, lw=0.6, ec="#555555"))
ins.text(-MU, -0.25, "Earth", ha="center", va="top", fontsize=8)
ins.set_xlim(-1.25, 1.35); ins.set_ylim(-1.05, 1.05); ins.set_aspect("equal")
ins.set_xticks([]); ins.set_yticks([])
for sp in ins.spines.values():
    sp.set_visible(True); sp.set_linewidth(0.6)
S.panel(a, "A", x=-0.25)
# (B) timescale and steps
b.semilogy(ts, tau, color="#000000", lw=1.25, label=r"local timescale $\tau$")
b.semilogy(ts, h_s, color=S.C["sundman"], lw=1.25, label=r"Sundman step $g\,\Delta s$")
b.axhline(d_phys, color=S.C["fixed"], lw=1.25, ls=(0, (5, 3)))
b.text(T * 0.02, d_phys / 1.2, "largest passing fixed physical step", ha="left", va="top", fontsize=8.5, color=S.C["fixed"])
b.set_xlabel("Time (TU)")
b.set_ylabel("Timescale or step (TU)")
b.set_xlim(0, T)
b.set_ylim(d_phys / 2.5, 3.0)
b.legend(loc="upper left", fontsize=8.5, handlelength=2.2, ncol=2)
S.panel(b, "B", x=-0.17)
S.save(fig, 1)
S.write_data(1, "trajectory", pd.DataFrame({"t_TU": ts, "x_DU": Y[0], "y_DU": Y[1], "z_DU": Y[2], "tau_TU": tau,
                                             "sundman_step_TU": h_s, "fixed_step_TU": d_phys}))
