"""
Study E, draft section 7 item 9: the E7 policy, a learned 8th-order fixed step in Sundman time. Nothing here is used by a sealed kernel.

  training_labels       RK8 labels (study_e_rk8.rk8s_label, ds in [1e-4, 64], the sealed monotone search) of the Study C fitting sets, cached
  fit_e7                the sund model class (histogram gradient boosting, pinball loss, seed 42, depth 3, learning rate 0.2) with trees and quantile
                        from the grid of draft section 3, on the same 25 features and the Study C importance weights
  GRID                  trees in {25, 50} x quantile in {0.1, 0.2, 0.3}
  select_e7             the draft section 3 rule on E_select: 50 / 50 folds (seed 42, policy_c_3d.split_folds); per config the scale certified on the
                        calibration fold by the sealed fixed-sequence rule, compiled DOP853 in Sundman time set to equal risk with that config's
                        calibration violation rate on the same fold, and the ratio of totals (comparator / policy, overhead included) on the evaluation
                        fold; the largest ratio wins, ties (equal to 3 decimals) go to the smaller model (fewer trees, then lower quantile)
  CompiledE7            features -> forest -> scale -> RK8 Sundman-time run, one compiled pipeline (the E7b timing arm), and its raw step batch
  ceiling_report        the label-ceiling rule (draft section 3, amended 2026-11-13): labels at ds_max are counted, and those whose run at ds_max
                        takes more than one step (more than 12 evaluations) are reported; the rule holds when they are at most 1 % of the labels

Overhead: the RK4 policy's pinned 15.04 RHS-equivalents is measured for 25 trees; the E7 overhead is measured for the chosen model on the idle
machine (compiled_model_3d.measure_single_call_cost on the shimmed forest) and pinned before the seal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from .config import MU, EJECTION_RADIUS, POSITION_TOL_KM, collision_radii
from .sundman_3d import SUNDMAN_TAU_C
from . import compiled_model_3d as cm
from . import defense_runs as dr
from . import study_e_rk8 as e8

__all__ = ["GRID", "MODEL_FIXED", "fit_e7", "raw_ds", "CompiledE7", "e7_policy_one", "loop_e7", "ceiling_report", "training_labels", "select_e7"]

GRID = tuple((n, q) for n in (25, 50) for q in (0.1, 0.2, 0.3))
MODEL_FIXED = {"max_depth": 3, "learning_rate": 0.2, "seed": 42}


def fit_e7(train, weights, features, n_trees, quantile):
    """train has dt_rk8s and rk8s_success. Returns policy_c_3d-style models with candidates ("rk8s",)."""
    from . import policy_c_3d as pc
    return pc.fit_policy_c(train, features, quantile=quantile, seed=MODEL_FIXED["seed"], lf_gate="none", sample_weight=weights, candidates=("rk8s",),
                           reg_kw={"max_iter": int(n_trees), "max_depth": MODEL_FIXED["max_depth"], "learning_rate": MODEL_FIXED["learning_rate"]})


def raw_ds(models, df):
    from . import policy_c_3d as pc
    return 10.0 ** pc.predict_raw_c(models, df)["rk8s"]["logdt"]


def _shim(models):
    """CompiledHGBPolicy knows the Sundman RK4 regressor's key; the forest is the same kind of object."""
    m = dict(models)
    m["candidates"] = ("rk4s",)
    m["reg_rk4s"] = models["reg_rk8s"]
    return m


@njit(cache=True)
def e7_policy_one(st, T, scale, mu, c_l1, fr, tau_c, max_steps, rc1, rc2, rej, A, B, xbuf, flags):
    """One orbit end to end: features and forest (the single-regressor path of compiled_model_3d.policy_hgb_one), ds = scale * raw, the RK8
    Sundman-time run. Returns (ok, y_end, evals, reason, ds)."""
    _c, raw = cm.policy_hgb_one(st, T, mu, c_l1, True, fr, fr, fr, 3, 0.0, False, xbuf, flags)
    ds = raw * scale
    r = dr._integrate_endpoint_rk8s_numba(st.copy(), ds, T, mu, max_steps, rc1, rc2, rej, tau_c, 0.0, A, B)
    return r[0], r[2], A.shape[0] * (r[6] + r[7]), r[5], ds


@njit(cache=True)
def loop_e7(st, T, reps, scale, mu, c_l1, fr, tau_c, max_steps, rc1, rc2, rej, A, B):
    """Timing loop (E6) for the E7 pipeline, as compiled_solvers_d._loop_policy."""
    xbuf = np.zeros(25)
    flags = np.zeros(2)
    s = st.copy()
    acc = 0.0
    for r in range(reps):
        s[0] = st[0] + 1e-15 * (r % 7)
        _o, y, ev, _re, _ds = e7_policy_one(s, T, scale, mu, c_l1, fr, tau_c, max_steps, rc1, rc2, rej, A, B, xbuf, flags)
        acc += y[0] + ev
    return acc


class CompiledE7:
    def __init__(self, models, mu=MU, tau_c=SUNDMAN_TAU_C, max_steps=4_000_000):
        self.cp = cm.CompiledHGBPolicy(_shim(models), mu=mu)
        self.mu, self.tau_c, self.max_steps = float(mu), float(tau_c), int(max_steps)
        self.c_l1 = self.cp.c_l1
        self.rc1, self.rc2 = collision_radii(mu)
        self.rej = EJECTION_RADIUS

    def raw_ds(self, states, T):
        return self.cp.batch(states, T)[1]

    def one(self, state, T, scale):
        xbuf, flags = np.zeros(25), np.zeros(2)
        return e7_policy_one(np.ascontiguousarray(state, dtype=np.float64), float(T), float(scale), self.mu, self.c_l1, self.cp.fr, self.tau_c,
                             self.max_steps, self.rc1, self.rc2, self.rej, dr._A8, dr._B8, xbuf, flags)

    def loop(self, state, T, reps, scale):
        return loop_e7(np.ascontiguousarray(state, dtype=np.float64), float(T), int(reps), float(scale), self.mu, self.c_l1, self.cp.fr, self.tau_c,
                       self.max_steps, self.rc1, self.rc2, self.rej, dr._A8, dr._B8)

    def overhead(self, states, T, repeats=5):
        """RHS-equivalents of features + forest per orbit (compiled_model_3d.measure_single_call_cost on the forest)."""
        return cm.measure_single_call_cost(self.cp, states, T, repeats=repeats)


def ceiling_report(labels, ds_max=e8.RK8_DS_MAX):
    L = pd.DataFrame(labels)
    at = L["rk8s_censored"].astype(bool)
    multi = at & (L["rk8s_evals"] > e8.RK8_STAGES)
    n = len(L)
    return {"ds_max": float(ds_max), "n": int(n), "n_at_ceiling": int(at.sum()), "share_at_ceiling": float(at.mean()),
            "n_at_ceiling_multi_step": int(multi.sum()), "share_at_ceiling_multi_step": float(multi.mean()),
            "rule_holds": bool(multi.mean() <= 0.01)}


def training_labels(Utr, ref, pool):
    """RK8 labels of every row (dict of arrays, study_e_rk8.rk8s_label columns)."""
    S = Utr[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(dtype=float)
    T = Utr["sim_time"].to_numpy(dtype=float)
    res = pool.map(e8.rk8s_label_job, [(S[i], float(T[i]), ref[i]) for i in range(len(Utr))], chunksize=16)
    L = pd.DataFrame(res)
    L.insert(0, "row_id", Utr["row_id"].to_numpy())
    return L


def select_e7(train, weights, features, sel_df, sel_ref, comparator_rows, overhead, grid=GRID, log=print):
    """The draft section 3 selection on E_select (sel_df usable rows, sel_ref their reference positions). comparator_rows(states, T, ref, rtol) ->
    dict(nfev, met, capped) of compiled DOP853 in Sundman time. Returns (chosen (n, q), table of every config)."""
    from . import deploy_d as dd, policy_c_3d as pc, symmetric_cert as sc
    from .posthoc_metrics import cost_stats
    cal, ev = pc.split_folds(len(sel_df), seed=42)
    S = sel_df[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(dtype=float)
    T = sel_df["sim_time"].to_numpy(dtype=float)
    runner = e8.Rk8sRunner(MU)
    rows = []
    for n_trees, q in grid:
        models = fit_e7(train, weights, features, n_trees, q)
        raw = raw_ds(models, sel_df)
        cert = dd.certify_scale(runner, S[cal], T[cal], raw[cal], sel_ref[cal])
        scale = cert["chosen_scale"]
        row = {"n_trees": n_trees, "quantile": q, "scale": scale}
        if scale is None:
            rows.append({**row, "ratio_of_totals": np.nan})
            log(f"  {n_trees} trees q {q}: no scale certified")
            continue
        target = cert["violations_by_scale"][float(scale)] / len(cal)
        cache = {}

        def viol(rt):
            if rt not in cache:
                cache[rt] = int((~comparator_rows(S[cal], T[cal], sel_ref[cal], rt)["met"]).sum())
            return cache[rt], len(cal)
        eqr, _chain = sc.equal_risk_rtol(viol, target)
        r = dd.run_rows(runner, S[ev], T[ev], raw[ev] * scale, sel_ref[ev])
        c = comparator_rows(S[ev], T[ev], sel_ref[ev], eqr)
        st = cost_stats(r["evals"] + overhead, c["nfev"], r["met"], n_boot=200, capped=c["capped"])
        row.update({"cal_violation_rate": target, "comparator_rtol": eqr, "ratio_of_totals": st["aggregate_ratio"], "share": st["share"],
                    "eval_violation_rate": float((~r["met"]).mean()), "comparator_eval_violation_rate": float((~c["met"]).mean())})
        rows.append(row)
        log(f"  {n_trees} trees q {q}: scale {scale:.3f}, ratio of totals {st['aggregate_ratio']:.4f}")
    tab = pd.DataFrame(rows)
    ok = tab[np.isfinite(tab["ratio_of_totals"])].copy()
    ok["r3"] = ok["ratio_of_totals"].round(3)
    best = ok.sort_values(["r3", "n_trees", "quantile"], ascending=[False, True, True]).iloc[0]
    return (int(best["n_trees"]), float(best["quantile"])), tab
