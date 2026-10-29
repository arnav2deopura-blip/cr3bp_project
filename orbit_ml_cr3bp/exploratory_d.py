"""
Exploratory analyses of Part D (Tier 2 and 3 of the plan; pre-registered as exploratory, outside the Holm family).

  grouped_importance        D6. Which groups of free quantities carry the information in the step label: grouped permutation importance and drop-group refits with paired
                            bootstrap intervals, for the Sundman-time target and the physical-time target, FTLE as an offline explanatory group only
  symplecticity_defect      Tier 3. Finite-difference check that the leapfrog step is a symplectic map (Phi^T J Phi = J) in the canonical variables of the rotating frame
  chaos_slice               D7. Planar surface-of-section grid: FTLE field and the policy's violations on it
  reference_check          D12. The highest-lambda*T orbits recomputed with a tighter / different reference: which orbits are 'unpredictable'
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import MU

GROUPS = {
    "state": ["x", "y", "z", "vx", "vy", "vz"],
    "proximity": ["r1_3d", "r2_3d", "r1_r2_ratio_3d"],
    "jacobi": ["jacobi_3d", "c_minus_c_l1_3d"],
    "periapsis": ["osc_logrp_1", "osc_logtaup_1", "osc_tpn_1", "osc_reach_1", "osc_logrel_1", "osc_ecc_1", "osc_logrp_2", "osc_logtaup_2", "osc_tpn_2",
                  "osc_reach_2", "osc_logrel_2", "osc_ecc_2", "osc_logtau_hat"],
    "horizon": ["sim_time"],
}
FTLE_GROUP = {"ftle (offline, needs an integration)": ["ftle_3d"]}


def _hgb(seed=0, n_iter=200):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss="squared_error", max_iter=n_iter, learning_rate=0.1, max_depth=6, random_state=seed)


def grouped_importance(train, y_train, test, y_test, groups, w_train=None, n_perm=5, n_boot=2000, seed=0, n_iter=200):
    """Per group: (a) the test-MSE increase when the group's columns are jointly permuted (rows shuffled together; mean over n_perm shuffles), (b) the test-MSE increase of a
    model refitted without the group (drop-group), each with a paired bootstrap 95 % interval over test orbits, and the full model's test MSE / median |error|.
    All columns of all groups are the model's inputs. The permutation importance of a correlated group is the less trustworthy of the two (Strobl et al. 2008);
    the drop-group refit is the one to read when groups carry overlapping information."""
    cols = [c for g in groups.values() for c in g]
    rng = np.random.RandomState(seed)
    full = _hgb(seed, n_iter).fit(train[cols].to_numpy(), y_train, sample_weight=w_train)
    base_pred = full.predict(test[cols].to_numpy())
    base_se = (base_pred - y_test) ** 2
    res = {"full": {"mse": float(base_se.mean()), "median_abs_err": float(np.median(np.abs(base_pred - y_test))), "n_test": int(len(test))}, "groups": {}}

    def ci(diff):
        b = np.array([diff[rng.randint(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
        return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]

    for name, g in groups.items():
        perm_diff = np.zeros(len(test))
        for _ in range(n_perm):
            T = test[cols].copy()
            idx = rng.permutation(len(T))
            T[g] = T[g].to_numpy()[idx]
            perm_diff += ((full.predict(T.to_numpy()) - y_test) ** 2 - base_se) / n_perm
        keep = [c for c in cols if c not in g]
        m = _hgb(seed, n_iter).fit(train[keep].to_numpy(), y_train, sample_weight=w_train)
        drop_se = (m.predict(test[keep].to_numpy()) - y_test) ** 2
        res["groups"][name] = {"columns": list(g), "perm_delta_mse": float(perm_diff.mean()), "perm_ci95": ci(perm_diff),
                               "drop_delta_mse": float((drop_se - base_se).mean()), "drop_ci95": ci(drop_se - base_se),
                               "drop_mse": float(drop_se.mean()), "drop_median_abs_err": float(np.median(np.sqrt(drop_se)))}
    return res


def symplecticity_defect(step_fn, states, mu=MU, h=1e-6, dt=0.05):
    """max |Phi^T J Phi - J| of the one-step map `step_fn(state, dt, mu)` written in the canonical variables of the rotating frame, q = (x, y, z) and
    p = (vx - y, vy + x, vz) (so that H = |p|^2 / 2 + y p_x - x p_y - (1 - mu)/r1 - mu/r2), by central differences of step `dt`. Returns one value per state."""
    J = np.block([[np.zeros((3, 3)), np.eye(3)], [-np.eye(3), np.zeros((3, 3))]])

    def to_qp(s):
        return np.array([s[0], s[1], s[2], s[3] - s[1], s[4] + s[0], s[5]])

    def from_qp(z):
        return np.array([z[0], z[1], z[2], z[3] + z[1], z[4] - z[0], z[5]])

    def f(z):
        return to_qp(np.asarray(step_fn(from_qp(z), dt, mu), dtype=float))

    out = []
    for s in states:
        z0 = to_qp(s)
        Phi = np.empty((6, 6))
        for j in range(6):
            e = np.zeros(6)
            e[j] = h
            Phi[:, j] = (f(z0 + e) - f(z0 - e)) / (2 * h)
        out.append(float(np.abs(Phi.T @ J @ Phi - J).max()))
    return np.array(out)


# ---------------------------------------------------------------------------------------------------------------------
# D10: the Sundman label's dense sweep in ds (monotonicity) and the landing steps
# ---------------------------------------------------------------------------------------------------------------------

def _make_sweep():
    from numba import njit
    from .sundman_3d import _integrate_endpoint_rk4s_numba, SUNDMAN_TAU_C
    from .config import EJECTION_RADIUS, collision_radii

    @njit(cache=False)
    def sweep(state, T, grid, mu, tau_c, rc1, rc2, rej, pos, ok, ns, nf):
        for j in range(grid.shape[0]):
            o, _prev, curr, _tp, _t, _r, a, b = _integrate_endpoint_rk4s_numba(state, grid[j], T, mu, 4_000_000, rc1, rc2, rej, tau_c, 0.0)
            ok[j] = o
            ns[j] = a
            nf[j] = b
            for i in range(3):
                pos[j, i] = curr[i]

    def run(state, T, grid, mu=MU, tau_c=SUNDMAN_TAU_C):
        rc1, rc2 = collision_radii(mu)
        n = len(grid)
        pos, ok, ns, nf = np.empty((n, 3)), np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
        sweep(np.asarray(state, dtype=np.float64), float(T), np.asarray(grid, dtype=np.float64), float(mu), float(tau_c), rc1, rc2, EJECTION_RADIUS, pos, ok, ns, nf)
        return pos, ok, ns, nf
    return run


def dense_sweep_landing(states, T, refs, labels, mu=MU, tol_km=100.0, lo=1e-3, hi=4.0, ratio=2.0 ** (1.0 / 16.0)):
    """Per orbit, run the Sundman RK4 on the geometric grid of fictitious steps (ratio 2^(1/16)) and return, per orbit: the grid, ok-and-within-tolerance flags, the share of
    steps that were physical-time landing steps, and whether the success set is non-monotone (a larger ds passes after a smaller one failed)."""
    from .config import du_km_for_mu
    run = _make_sweep()
    n = int(np.floor(np.log(hi / lo) / np.log(ratio))) + 1
    grid = np.sort(hi * ratio ** (-np.arange(n)))
    out = []
    for i in range(len(states)):
        pos, ok, ns, nf = run(states[i], T[i], grid, mu)
        err = np.linalg.norm(pos - refs[i], axis=1) * du_km_for_mu(mu)
        good = ok & (err <= tol_km)
        fails = np.where(~good)[0]
        nonmono = bool(len(fails) > 0 and good[fails[0] + 1:].any())
        land = nf / np.maximum(ns + nf, 1)
        flips = np.where(good[1:] != good[:-1])[0] + 1
        out.append({"nonmonotone": nonmono, "n_fail": int((~good).sum()), "label": float(labels[i]), "dense_max_ok_ds": float(grid[good].max()) if good.any() else float("nan"),
                    "landing_share_at_flips": float(land[flips].mean()) if len(flips) else float("nan"), "landing_share_overall": float(land.mean()),
                    "nf_change_at_flips": float(np.abs(np.diff(nf)[flips - 1]).mean()) if len(flips) else float("nan"), "nf_change_overall": float(np.abs(np.diff(nf)).mean()),
                    "n_flips": int(len(flips))})
    return pd.DataFrame(out)
