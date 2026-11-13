"""
Confirmatory run of Part E (part_e/PREREGISTRATION_E.md). One command per step; each step on target `confirm` runs ONCE and is logged in
confirm_log_e.jsonl; a second run needs --allow_rerun "<reason>", recorded in the log and as a dated entry under '## Deviations'.

  python run_confirm_e_3d.py steps
  python run_confirm_e_3d.py run <step> --target confirm|dry
  python run_confirm_e_3d.py tests --target confirm|dry          the statistics of section 4, Holm over E1-E5 and E7, the secondary tables

Target `dry` = E_select (seed 60) and the Study D dry JPL sample: the code check and the source of the expectations of section 5. It never
counts, is never logged and runs before the seal. Target `confirm` needs the seal (freeze_e.py check must pass) and uses seeds 61-66 and the
Study E JPL sample.

Steps (every run uses the Part E kernels of events_e.py: endpoint and chord collision checks for every arm):
  e1          RK4 policy (frozen Study C sund) vs DP5(4)-PI in physical and in Sundman time at equal risk (tested), Sundman-time DOP853 (reported);
              per-band equal risk (secondary); E_confirm (1-3 TU)
  e2          the same at 10 km and at 1,000 km on E_tol (1-8 TU), every scale and rtol recalibrated at that tolerance
  e3          the RK4 policy with the path-certified scale (K = 20 checkpoints), comparators at equal path risk; E_confirm
  e4          the full population (usable and doomed rows) of E_confirm at the e1 scale and rtols: success and total cost
  e5          the gated RK4 policy on the JPL runs (abstentions to DP5(4)-PI at its certified rtol); scale and rtol certified on E_confirm_20;
              in-distribution abstention on the E_confirm evaluation rows
  e7          the E7 policy vs Sundman-time DOP853, Sundman-time DP5(4)-PI and physical DP5(4)-PI at equal risk (tested); per band (secondary);
              E_confirm_20 (1-20 TU)
  timing      E1b and E7b: the E6 protocol (refuses to start above load 1.0)
  lock        E_lock (RK4 policy, e1 scale and rtols) and E_lock_20 (E7 policy, e7 scale and rtols): one final evaluation each, descriptive
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from orbit_ml_cr3bp import protocol_3d as proto  # noqa: E402
from orbit_ml_cr3bp.config import MU, POSITION_TOL_KM  # noqa: E402

LOG = os.path.join(ROOT, "confirm_log_e.jsonl")
STEPS = ["e1", "e2", "e3", "e4", "e5", "e7", "timing", "lock"]
RK4_FROZEN = "part_c/frozen/sund/policy_frozen.joblib"
E7_FROZEN = "part_e/frozen/rk8s/policy_frozen.joblib"
GATE_FROZEN = "part_e/frozen/gate/gate.joblib"
OVERHEADS = "part_e/frozen/overheads.json"
TESTED_E1 = ("DP54PI", "DP54PIs")
REPORTED_E1 = ("DOP853s",)
TESTED_E7 = ("DOP853s", "DP54PIs", "DP54PI")
N_CAL = 4000


SMOKE = {"on": False, "n": 0}     # target `smoke`: the Study D selection set stands in for every Part E set, subsampled; code check only


def data_path(name):
    if SMOKE["on"]:
        return os.path.join(ROOT, "data_d", "select", "cr3bp_3d_training_data_accuracy_monotone_seed52.csv")
    seeds = {"select": 60, "confirm": 61, "confirm_8": 62, "tol_10": 63, "tol_1000": 63, "lock": 64, "confirm_20": 65, "lock_20": 66}
    return os.path.join(ROOT, "data_e", name, f"cr3bp_3d_training_data_accuracy_monotone_seed{seeds[name]}.csv")


def frozen(rel):
    """Frozen inputs: part_e/frozen/... (smoke: out_e/smoke/frozen/..., written by run_part_e_3d.py --smoke)."""
    if SMOKE["on"]:
        return os.path.join(ROOT, "out_e", "smoke", "frozen", rel.split("part_e/frozen/", 1)[1])
    return os.path.join(ROOT, rel)


def out_dir(target, step):
    target = "smoke" if SMOKE["on"] else target
    d = os.path.join(ROOT, "out_e", target, step)
    os.makedirs(d, exist_ok=True)
    return d


def save_json(obj, path):
    json.dump(proto.json_safe(obj), open(path, "w"), indent=1)


# ---------------------------------------------------------------------------------------------------------------------
# data, policies, certification
# ---------------------------------------------------------------------------------------------------------------------

def load_usable(name, t_max=None):
    import run_part_c_3d as rc
    U, _w = rc.load_c_frame([data_path(name)], MU)
    if t_max is not None:
        U = U[U["sim_time"] <= t_max + 1e-12].reset_index(drop=True)
    if SMOKE["on"]:
        U = U.iloc[: SMOKE["n"]].reset_index(drop=True)
    return U


def load_full(name, t_max=None):
    """Every candidate row whose reference reached T or ended in a collision / ejection (E4's population); other outcomes are counted, not run."""
    d = pd.read_csv(data_path(name), low_memory=False)
    if t_max is not None:
        d = d[d["sim_time"] <= t_max + 1e-12]
    keep = d["reference_outcome"].astype(str).isin(["ok", "primary_collision", "secondary_collision", "ejection"])
    d = d[keep].reset_index(drop=True)
    if SMOKE["on"]:
        d = d.iloc[: SMOKE["n"]].reset_index(drop=True)
    return d, d["reference_outcome"].astype(str).value_counts().to_dict()


def split(U, n_cal=N_CAL):
    from run_part_d_3d import standard_split
    n_cal = min(n_cal, len(U) // 2)
    return standard_split(U, n_cal=n_cal)


def states_T(df):
    return df[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(dtype=float), df["sim_time"].to_numpy(dtype=float)


def refs(df, tag):
    from orbit_ml_cr3bp import deploy_d as dd
    return dd.reference_positions(df, MU, cache=os.path.join(ROOT, "out_e", "refs", f"{'smoke_' if SMOKE['on'] else ''}{tag}.npz"), n_jobs=8)


def overheads():
    return json.load(open(frozen(OVERHEADS)))


def policy_features(df, mu=MU):
    """The 25 policy inputs rebuilt from x..vz and sim_time (as run_posthoc._policy_features, at the given mass ratio)."""
    from orbit_ml_cr3bp.physics_3d import add_orbit_features_3d
    from orbit_ml_cr3bp.osculating_3d import add_osc_features
    return add_osc_features(add_orbit_features_3d(df.reset_index(drop=True), mu), mu)


def rk4_raw(df):
    import joblib
    from run_part_d_3d import raw_log_ds
    return 10.0 ** raw_log_ds(joblib.load(os.path.join(ROOT, RK4_FROZEN))["models"], df)          # the Study C policy itself, never a smoke copy


def e7_raw(df):
    import joblib
    from orbit_ml_cr3bp import e7_policy as e7
    return e7.raw_ds(joblib.load(frozen(E7_FROZEN))["models"], df)


class Runner:
    """rows_with_event over a pool, with violation counts cached per (method, parameter, rows, tolerance, criterion)."""

    def __init__(self, pool):
        self.pool = pool

    def rows(self, method, S, T, param, ref, rck=None, tol_km=POSITION_TOL_KM):
        from orbit_ml_cr3bp import events_e as ee
        return ee.rows_with_event(method, S, T, param, ref, rck, seg=True, tol_km=tol_km, pool=self.pool)

    def certify_scale(self, method, S, T, raw, ref, rck=None, tol_km=POSITION_TOL_KM, log=print):
        from orbit_ml_cr3bp import policy_c_3d as pc
        key = "met_path" if rck is not None else "met"
        cache = {}

        def viol(s):
            if s not in cache:
                cache[s] = int((~self.rows(method, S, T, raw * s, ref, rck, tol_km)[key]).sum())
                log(f"    scale {s:.4f}: {cache[s]} / {len(S)} violations")
            return cache[s], len(S)
        from orbit_ml_cr3bp.confirm_e_3d import SCALE_GRID_E
        rep = pc.certify_sequence(viol, SCALE_GRID_E, pc.ALPHA, pc.DELTA, "fixed_sequence")
        s = rep["chosen_scale"]
        return {"scale": s, "chain": rep["chain"], "cal_violation_rate": None if s is None else cache[float(s)] / len(S)}

    def equal_risk(self, method, S, T, ref, target, rck=None, tol_km=POSITION_TOL_KM, log=print):
        from orbit_ml_cr3bp import symmetric_cert as sc
        key = "met_path" if rck is not None else "met"
        cache = {}

        def viol(rt):
            if rt not in cache:
                cache[rt] = int((~self.rows(method, S, T, rt, ref, rck, tol_km)[key]).sum())
                log(f"    {method} rtol {rt:.3e}: {cache[rt]} / {len(S)} violations")
            return cache[rt], len(S)
        eqr, chain = sc.equal_risk_rtol(viol, target)
        return {"rtol": eqr, "chain": chain}

    def certified(self, method, S, T, ref, log=print):
        from orbit_ml_cr3bp import symmetric_cert as sc
        cache = {}

        def viol(rt):
            if rt not in cache:
                cache[rt] = int((~self.rows(method, S, T, rt, ref)["met"]).sum())
            return cache[rt], len(S)
        return sc.certify_rtol_fixed_sequence(viol, sc.RTOL_GRID_FINE)


def ledger_cols(prefix, r):
    return {f"{prefix}_{k}": v for k, v in r.items()}


# ---------------------------------------------------------------------------------------------------------------------
# the steps
# ---------------------------------------------------------------------------------------------------------------------

def _policy_vs(R, policy, raw, oh, U, ref, comps, rck=None, tol_km=POSITION_TOL_KM, per_band=True, log=print, cal_ev=None):
    """Certify the policy scale on the calibration rows, set each comparator to equal risk, run everything on the evaluation rows.
    Returns (report, ledger DataFrame)."""
    cal, ev = cal_ev if cal_ev is not None else split(U)
    S, T = states_T(U)
    rc_cal = None if rck is None else rck[cal]
    rc_ev = None if rck is None else rck[ev]
    key = "met_path" if rck is not None else "met"
    log(f"  certifying the {policy} scale on {len(cal)} calibration rows")
    cert = R.certify_scale(policy, S[cal], T[cal], raw[cal], ref[cal], rc_cal, tol_km, log)
    rep = {"policy": policy, "n_cal": int(len(cal)), "n_eval": int(len(ev)), "overhead": oh, "tol_km": tol_km, "certificate": cert,
           "criterion": "path" if rck is not None else "endpoint", "comparators": {}}
    if cert["scale"] is None:
        return rep, None
    led = pd.DataFrame({"row_id": U.iloc[ev]["row_id"].to_numpy(), "sim_time": T[ev]})
    led["h_band"] = pd.cut(led["sim_time"], [0, 3, 8, 1e9], labels=["1-3", "3-8", "8-20"]).astype(str)
    pr = R.rows(policy, S[ev], T[ev], raw[ev] * cert["scale"], ref[ev], rc_ev, tol_km)
    led = pd.concat([led, pd.DataFrame(ledger_cols("pol", pr))], axis=1)
    led["pol_cost"] = led["pol_evals"] + oh
    for c in comps:
        log(f"  {c}: equal risk {cert['cal_violation_rate']:.4f}")
        e = R.equal_risk(c, S[cal], T[cal], ref[cal], cert["cal_violation_rate"], rc_cal, tol_km, log)
        rep["comparators"][c] = {"equal_risk": e}
        if e["rtol"] is None:
            continue
        cr = R.rows(c, S[ev], T[ev], e["rtol"], ref[ev], rc_ev, tol_km)
        led = pd.concat([led, pd.DataFrame(ledger_cols(c, cr))], axis=1)
        if per_band:
            bands_cal = pd.cut(T[cal], [0, 3, 8, 1e9], labels=["1-3", "3-8", "8-20"]).astype(str)
            pb = {}
            pol_viol_cal = ~R.rows(policy, S[cal], T[cal], raw[cal] * cert["scale"], ref[cal], rc_cal, tol_km)[key]
            band_cost = np.full(len(ev), np.nan)
            led[f"{c}_band_met"] = np.nan
            for b in sorted(set(bands_cal)):
                sb = bands_cal == b
                if sb.sum() < 100:
                    continue
                tb = float(pol_viol_cal[sb].mean())
                eb = R.equal_risk(c, S[cal][sb], T[cal][sb], ref[cal][sb], tb, None if rc_cal is None else rc_cal[sb], tol_km, lambda *_: None)
                pb[b] = {"target": tb, "rtol": eb["rtol"]}
                se = (led["h_band"] == b).to_numpy()
                if eb["rtol"] is not None and se.any():
                    q = R.rows(c, S[ev][se], T[ev][se], eb["rtol"], ref[ev][se], None if rc_ev is None else rc_ev[se], tol_km)
                    band_cost[se] = q["evals"]
                    led.loc[se, f"{c}_band_met"] = q[key].astype(float)
            led[f"{c}_band_evals"] = band_cost
            rep["comparators"][c]["per_band_equal_risk"] = pb
    return rep, led


def step_e1(a, R, log):
    U = load_usable("select", 3.0) if a.target == "dry" else load_usable("confirm")
    ref = refs(U, f"{a.target}_e1")
    rep, led = _policy_vs(R, "rk4s", rk4_raw(U), overheads()["rk4s"], U, ref, TESTED_E1 + REPORTED_E1, log=log)
    return rep, led


def step_e2(a, R, log):
    out = {}
    leds = []
    for tol in (10.0, 1000.0):
        U = load_usable("select", 8.0) if a.target == "dry" else load_usable("tol_10" if tol == 10.0 else "tol_1000")
        ref = refs(U, f"{a.target}_e2_{int(tol)}")
        rep, led = _policy_vs(R, "rk4s", rk4_raw(U), overheads()["rk4s"], U, ref, TESTED_E1 + REPORTED_E1, tol_km=tol, per_band=False, log=log)
        out[str(int(tol))] = rep
        if led is not None:
            led["tol_km"] = tol
            leds.append(led)
    return out, (pd.concat(leds, ignore_index=True) if leds else None)


def step_e3(a, R, log):
    from orbit_ml_cr3bp import events_e as ee
    U = load_usable("select", 3.0) if a.target == "dry" else load_usable("confirm")
    ref = refs(U, f"{a.target}_e1")
    S, T = states_T(U)
    rck = ee.reference_checkpoints_rows(S, T, 20, MU, pool=R.pool)
    return _policy_vs(R, "rk4s", rk4_raw(U), overheads()["rk4s"], U, ref, TESTED_E1, rck=rck, per_band=False, log=log)


def step_e4(a, R, log):
    """Usable rows: success = final error <= tol. Doomed rows: the strict event rule. Scale and rtols from step e1 (same target)."""
    from orbit_ml_cr3bp import events_e as ee
    e1 = json.load(open(os.path.join(out_dir(a.target, "e1"), "report.json")))
    D, counts = load_full("select", 3.0) if a.target == "dry" else load_full("confirm")
    S, T = states_T(D)
    usable = (D["reference_outcome"].astype(str) == "ok").to_numpy()
    jobs = [(S[i], float(T[i])) for i in range(len(D))]
    rev = R.pool.starmap(ee.reference_event, jobs, chunksize=16)
    ref_code = np.array([r[0] for r in rev])
    ref_t = np.array([r[1] for r in rev])
    ref_v = np.array([r[2] for r in rev])
    ref_pos = np.full((len(D), 3), np.nan)
    U = D[usable].reset_index(drop=True)
    ref_pos[usable] = refs(U, f"{a.target}_e4_usable")
    raw = rk4_raw(D)
    arms = {"rk4s": raw * e1["certificate"]["scale"]}
    for c in TESTED_E1:
        arms[c] = e1["comparators"][c]["equal_risk"]["rtol"]
    led = pd.DataFrame({"row_id": D["orbit_idx"].to_numpy(), "usable": usable, "ref_event": ref_code, "ref_t": ref_t, "ref_speed": ref_v})
    for name, prm in arms.items():
        r = R.rows(name, S, T, prm, np.where(np.isfinite(ref_pos), ref_pos, 0.0))
        succ = np.where(usable, r["met"], [ee.strict_success(r["event"][i], r["t_event"][i], ref_code[i], ref_t[i], ref_v[i]) for i in range(len(D))])
        led[f"{name}_success"] = succ.astype(bool)
        led[f"{name}_evals"] = r["evals"]
        led[f"{name}_event"] = r["event"]
        led[f"{name}_t_event"] = r["t_event"]
    led["rk4s_cost"] = led["rk4s_evals"] + overheads()["rk4s"]
    rep = {"population_counts": counts, "n_run": int(len(D)), "n_usable": int(usable.sum()), "n_doomed": int((~usable).sum()),
           "scale": e1["certificate"]["scale"], "rtols": {c: arms[c] for c in TESTED_E1}}
    return rep, led


def step_e5(a, R, log):
    import joblib
    from orbit_ml_cr3bp import ood_e as oe, periodic_d as pdd
    gate = joblib.load(frozen(GATE_FROZEN))
    # scale of the RK4 policy and the certified comparator rtol over 1-20 TU (E_confirm_20 calibration rows; dry: E_select)
    U20 = load_usable("select") if a.target == "dry" else load_usable("confirm_20")
    cal, _ev = split(U20)
    S, T = states_T(U20)
    ref20 = refs(U20, f"{a.target}_e7")
    raw20 = rk4_raw(U20)
    cert = R.certify_scale("rk4s", S[cal], T[cal], raw20[cal], ref20[cal], log=log)
    comp = R.certified("DP54PI", S[cal], T[cal], ref20[cal], log=log)
    # the JPL runs
    jpl = os.path.join(ROOT, "data_d", "jpl")
    if a.target == "dry":
        M, runs, mu = _build_runs_with(pdd.select_indices, jpl, "dry", 3 if SMOKE["on"] else 50)
    else:
        M, runs, mu = oe.build_e5_runs(jpl)
    test = runs[runs["in_range"] & runs["ref_ok"]].reset_index(drop=True)
    test = policy_features(test, mu)          # the 25 inputs at JPL's mass ratio (the orbits are periodic only for it), for the policy and the gate
    Sj, Tj = states_T(test)
    refj = test[["ref_x", "ref_y", "ref_z"]].to_numpy(dtype=float)
    from orbit_ml_cr3bp import events_e as ee
    pol = ee.rows_with_event("rk4s", Sj, Tj, rk4_raw(test) * cert["scale"], refj, mu=mu, pool=R.pool)
    cmp_ = ee.rows_with_event("DP54PI", Sj, Tj, comp["chosen_rtol"], refj, mu=mu, pool=R.pool)
    ab = gate.abstain(test)
    test["abstain"], test["pol_violation"], test["comp_violation"] = ab, ~pol["met"], ~cmp_["met"]
    test["violation"] = oe.gated_violations(ab, ~pol["met"], ~cmp_["met"])
    test["violation_ungated"] = ~pol["met"]
    # in-distribution abstention on the E_confirm evaluation rows
    Ui = load_usable("select", 3.0) if a.target == "dry" else load_usable("confirm")
    _c, evi = split(Ui)
    ab_in = gate.abstain(Ui.iloc[evi])
    rep = {"gate": gate.spec(), "scale": cert["scale"], "comparator_certified_rtol": comp["chosen_rtol"], "n_members": int(len(M)),
           "n_runs": int(len(runs)), "n_test_runs": int(len(test)), "mu_jpl": mu, "in_distribution_abstentions": int(ab_in.sum()),
           "in_distribution_n": int(len(ab_in))}
    return rep, test


def _build_runs_with(select_fn, jpl, which, n_per_stream=50):
    """The dry-run JPL runs: the Study D dry sample, built like ood_e.build_e5_runs."""
    from orbit_ml_cr3bp import ood_e as oe
    orig = oe.select_indices_e
    try:
        oe.select_indices_e = lambda n_total, n: select_fn(n_total, n, which)
        return oe.build_e5_runs(jpl, n_per_stream=n_per_stream)
    finally:
        oe.select_indices_e = orig


def step_e7(a, R, log):
    U = load_usable("select") if a.target == "dry" else load_usable("confirm_20")
    ref = refs(U, f"{a.target}_e7")
    return _policy_vs(R, "rk8s", e7_raw(U), overheads()["rk8s"], U, ref, TESTED_E7, log=log)


def step_timing(a, R, log):
    """E1b on the e1 evaluation rows, E7b on the e7 evaluation rows (E6 protocol)."""
    import joblib
    from orbit_ml_cr3bp import timing_e as te, compiled_solvers_d as csd, compiled_sundman_e as cs, e7_policy as e7
    from orbit_ml_cr3bp.config import collision_radii, EJECTION_RADIUS
    if not SMOKE["on"]:
        te.require_idle()
    rc1, rc2 = collision_radii(MU)
    rej = EJECTION_RADIUS
    res = {"machine": te.machine_info()}
    fz = joblib.load(os.path.join(ROOT, RK4_FROZEN))
    sund = csd.SundmanPolicy(fz["models"], mu=MU)
    e7m = e7.CompiledE7(joblib.load(frozen(E7_FROZEN))["models"])
    sa = {c: cs.SundmanAdaptive(c) for c in ("DP54PIs", "DOP853s")}
    log_path = os.path.join(out_dir(a.target, "timing"), "sessions.jsonl")
    for which, step, policy_arm in (("E1b", "e1", "rk4s"), ("E7b", "e7", "rk8s")):
        rep = json.load(open(os.path.join(out_dir(a.target, step), "report.json")))
        led = pd.read_csv(os.path.join(out_dir(a.target, step), "ledger.csv.gz"))
        U = (load_usable("select", 3.0) if a.target == "dry" else load_usable("confirm")) if step == "e1" else \
            (load_usable("select") if a.target == "dry" else load_usable("confirm_20"))
        U = U.set_index("row_id").loc[led["row_id"]].reset_index()
        S, T = states_T(U)
        scale = rep["certificate"]["scale"]
        arms = {}
        if policy_arm == "rk4s":
            arms["policy"] = lambda i, reps: csd._loop_policy(S[i], T[i], reps, scale, MU, sund.c_l1, sund.cp.fr, sund.tau_c, sund.max_steps, rc1, rc2, rej)
            comps = TESTED_E1
        else:
            arms["policy"] = lambda i, reps: e7m.loop(S[i], T[i], reps, scale)
            comps = TESTED_E7
        for c in comps:
            rt = rep["comparators"][c]["equal_risk"]["rtol"]
            if c == "DP54PI":
                arms[c] = lambda i, reps, rt=rt: csd.KR.loop_dp54(S[i], T[i], reps, rt, csd.ATOL_RATIO, MU, csd.CTRL_PI, csd.MAX_NFEV, rc1, rc2, rej)
            else:
                k = sa[c]
                arms[c] = lambda i, reps, rt=rt, k=k: cs.loop_rk_s(S[i], T[i], reps, rt, k.atol_ratio, MU, k.tau_c, k.method, k.ctrl, k.max_nfev,
                                                                   k.rc1, k.rc2, k.rej, *k.tab)
        n = len(S) if a.n_timing <= 0 else min(len(S), a.n_timing)
        sess = te.time_sessions(arms, n, sessions=2, log_path=log_path, max_load=1e9 if SMOKE["on"] else te.MAX_LOAD)
        res[which] = {c: te.throughput(sess, "policy", c) for c in comps}
        res[which]["sessions"] = sess["sessions"]
        np.savez(os.path.join(out_dir(a.target, "timing"), f"{which}_times.npz"),
                 **{f"s{s}_{arm}": v for s, d in sess["times"].items() for arm, v in d.items()})
    return res, None


def step_lock(a, R, log):
    if a.target == "dry":
        raise SystemExit("lock has no dry target")
    out = {}
    leds = []
    for name, step, policy, rawf, oh_key, comps in (("lock", "e1", "rk4s", rk4_raw, "rk4s", TESTED_E1), ("lock_20", "e7", "rk8s", e7_raw, "rk8s", TESTED_E7)):
        rep = json.load(open(os.path.join(out_dir("confirm", step), "report.json")))
        U = load_usable(name)
        S, T = states_T(U)
        ref = refs(U, f"confirm_{name}")
        pr = R.rows(policy, S, T, rawf(U) * rep["certificate"]["scale"], ref)
        led = pd.DataFrame({"set": name, "row_id": U["row_id"].to_numpy(), "pol_cost": pr["evals"] + overheads()[oh_key], "pol_met": pr["met"]})
        from orbit_ml_cr3bp import confirm_e_3d as ce
        out[name] = {"policy_violation_rate": float((~pr["met"]).mean()), "n": int(len(U))}
        for c in comps:
            cr = R.rows(c, S, T, rep["comparators"][c]["equal_risk"]["rtol"], ref)
            led[f"{c}_evals"], led[f"{c}_met"] = cr["evals"], cr["met"]
            out[name][c] = {**ce.ratio_test(led["pol_cost"], cr["evals"]), "comparator_violation_rate": float((~cr["met"]).mean())}
        leds.append(led)
    return out, pd.concat(leds, ignore_index=True)


STEP_FN = {"e1": step_e1, "e2": step_e2, "e3": step_e3, "e4": step_e4, "e5": step_e5, "e7": step_e7, "timing": step_timing, "lock": step_lock}


# ---------------------------------------------------------------------------------------------------------------------
# the tests (section 4)
# ---------------------------------------------------------------------------------------------------------------------

def _read(target, step):
    d = out_dir(target, step)
    rep = json.load(open(os.path.join(d, "report.json"))) if os.path.exists(os.path.join(d, "report.json")) else None
    led = pd.read_csv(os.path.join(d, "ledger.csv.gz"), low_memory=False) if os.path.exists(os.path.join(d, "ledger.csv.gz")) else None
    return rep, led


def _ratio_parts(led, comps, cost_col="pol_cost"):
    from orbit_ml_cr3bp import confirm_e_3d as ce
    return {c: ce.ratio_test(led[cost_col], led[f"{c}_evals"]) for c in comps if f"{c}_evals" in led}


def cmd_tests(a):
    from orbit_ml_cr3bp import confirm_e_3d as ce
    from orbit_ml_cr3bp.confirm_d_3d import d5_ood
    T_ = a.target
    res, pvals = {}, {}
    rep, led = _read(T_, "e1")
    tim, _ = _read(T_, "timing")
    if led is not None:
        e1a = ce.iut(_ratio_parts(led, TESTED_E1))
        res["E1a"] = {"iut": e1a, "parts": _ratio_parts(led, TESTED_E1), "reported": _ratio_parts(led, REPORTED_E1),
                      "per_band_secondary": {c: ce.band_table(led["h_band"], led["pol_cost"], led[f"{c}_band_evals"], ~led["pol_met"].astype(bool),
                                                              ~led[f"{c}_band_met"].fillna(False).astype(bool)) for c in TESTED_E1 + REPORTED_E1
                                             if f"{c}_band_evals" in led}}
        parts = {"E1a": e1a}
        if tim is not None and "E1b" in tim:
            parts["E1b"] = ce.iut({c: tim["E1b"][c] for c in TESTED_E1})
        res["E1"] = ce.iut(parts)
        pvals["E1"] = res["E1"]["p"]
    rep2, led2 = _read(T_, "e2")
    if led2 is not None:
        parts = {f"{int(t)}km:{c}": ce.ratio_test(x["pol_cost"], x[f"{c}_evals"]) for t, x in led2.groupby("tol_km") for c in TESTED_E1}
        res["E2"] = ce.iut(parts)
        pvals["E2"] = res["E2"]["p"]
    rep3, led3 = _read(T_, "e3")
    if led3 is not None:
        k, n = int((~led3["pol_met_path"].astype(bool)).sum()), len(led3)
        parts = {"rate": ce.binom_le(k, n, 0.05), **{f"ratio:{c}": ce.ratio_test(led3["pol_cost"], led3[f"{c}_evals"]) for c in TESTED_E1}}
        res["E3"] = ce.iut(parts)
        res["E3"]["rate"] = parts["rate"]
        pvals["E3"] = res["E3"]["p"]
    rep4, led4 = _read(T_, "e4")
    if led4 is not None:
        parts = {}
        for c in TESTED_E1:
            parts[f"success:{c}"] = ce.noninferiority(led4["rk4s_success"], led4[f"{c}_success"])
            parts[f"ratio:{c}"] = ce.ratio_test(led4["rk4s_cost"], led4[f"{c}_evals"])
        res["E4"] = ce.iut(parts)
        res["E4"]["details"] = parts
        pvals["E4"] = res["E4"]["p"]
    rep5, led5 = _read(T_, "e5")
    if led5 is not None:
        d5 = d5_ood(led5, threshold=0.10)
        ab = ce.binom_le(rep5["in_distribution_abstentions"], rep5["in_distribution_n"], 0.05)
        res["E5"] = ce.iut({"family_balanced": d5, "in_distribution_abstention": ab})
        res["E5"]["family_balanced"], res["E5"]["abstention"] = d5, ab
        res["E5"]["ungated_family_balanced"] = d5_ood(led5.assign(violation=led5["violation_ungated"]), threshold=0.10)
        pvals["E5"] = res["E5"]["p"]
    rep7, led7 = _read(T_, "e7")
    if led7 is not None:
        e7a = ce.iut(_ratio_parts(led7, TESTED_E7))
        res["E7a"] = {"iut": e7a, "parts": _ratio_parts(led7, TESTED_E7),
                      "per_band_secondary": {c: ce.band_table(led7["h_band"], led7["pol_cost"], led7[f"{c}_band_evals"], ~led7["pol_met"].astype(bool),
                                                              ~led7[f"{c}_band_met"].fillna(False).astype(bool)) for c in TESTED_E7
                                             if f"{c}_band_evals" in led7}}
        parts = {"E7a": e7a}
        if tim is not None and "E7b" in tim:
            parts["E7b"] = ce.iut({c: tim["E7b"][c] for c in TESTED_E7})
        res["E7"] = ce.iut(parts)
        pvals["E7"] = res["E7"]["p"]
    res["holm"] = ce.holm(pvals) if pvals else {}
    res["note"] = "dry run on E_select: code check and expectations only, not a test" if T_ == "dry" else "confirmatory"
    save_json(res, os.path.join(out_dir(T_, "tests"), "tests.json"))
    for k, v in res["holm"].items():
        print(f"{k}: p = {v['p']:.4g}, Holm-adjusted {v['p_adj']:.4g}, {'supported' if v['reject'] else 'not supported'}")


# ---------------------------------------------------------------------------------------------------------------------
# once-only bookkeeping
# ---------------------------------------------------------------------------------------------------------------------

def _logged():
    return [json.loads(l) for l in open(LOG) if l.strip()] if os.path.exists(LOG) else []


def _guard(target):
    if target != "confirm":
        return None
    import freeze_e
    st = freeze_e.status()
    if st is None:
        raise SystemExit("Part E is not sealed (FREEZE_E.json missing): run `python freeze_e.py seal` first.")
    if st["problems"]:
        raise SystemExit("the code or frozen files changed since the seal:\n  " + "\n  ".join(st["problems"]))
    return st["record"]


def cmd_steps(a):
    done = {(r["step"], r["target"]) for r in _logged() if r.get("returncode") == 0}
    for s in STEPS:
        print(f"{s:8s} {'done' if (s, 'confirm') in done else '-'}")


def cmd_run(a):
    from multiprocessing import Pool
    seal = _guard(a.target)
    if a.target == "confirm":
        prior = [r for r in _logged() if r["step"] == a.step and r["target"] == "confirm" and r.get("returncode") == 0]
        if prior and not a.allow_rerun:
            raise SystemExit(f"step {a.step} already ran at {prior[-1]['utc']}. A rerun needs --allow_rerun '<reason>'.")
    t0 = time.time()
    d = out_dir(a.target, a.step)
    log_f = open(os.path.join(d, "run.log"), "a")

    def log(msg):
        line = f"[{time.time() - t0:7.0f}s] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()
    rc = 1
    try:
        with Pool(a.n_jobs) as pool:
            rep, led = STEP_FN[a.step](a, Runner(pool), log)
        rep = {"step": a.step, "target": a.target, "utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": time.time() - t0,
               "seal_utc": None if seal is None else seal["sealed_utc"], **(rep if isinstance(rep, dict) else {"result": rep})}
        if led is not None:
            led.to_csv(os.path.join(d, "ledger.csv.gz"), index=False)
        save_json(rep, os.path.join(d, "report.json"))
        rc = 0
    finally:
        if a.target == "confirm":
            rec = {"utc": datetime.now(timezone.utc).isoformat(), "step": a.step, "target": a.target, "returncode": rc, "rerun_reason": a.allow_rerun,
                   "code_sha256": proto.code_fingerprint(ROOT)["combined_sha256"]}
            with open(LOG, "a") as f:
                f.write(json.dumps(rec) + "\n")
    log(f"done ({a.step}, {a.target})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("steps").set_defaults(fn=cmd_steps)
    sp = sub.add_parser("run")
    sp.add_argument("step", choices=STEPS)
    sp.add_argument("--target", choices=["confirm", "dry"], required=True)
    sp.add_argument("--n_jobs", type=int, default=8)
    sp.add_argument("--n_timing", type=int, default=0, help="timing: orbits per arm (0 = every evaluation row)")
    sp.add_argument("--allow_rerun", default=None)
    sp.add_argument("--smoke", type=int, default=0, help="code check: Study D selection rows stand in for every set (with --target dry)")
    sp.set_defaults(fn=cmd_run)
    sp = sub.add_parser("tests")
    sp.add_argument("--target", choices=["confirm", "dry"], required=True)
    sp.add_argument("--smoke", type=int, default=0)
    sp.set_defaults(fn=cmd_tests)
    a = ap.parse_args()
    if getattr(a, "smoke", 0):
        if a.target != "dry":
            raise SystemExit("--smoke goes with --target dry")
        SMOKE.update(on=True, n=int(a.smoke))
    a.fn(a)


if __name__ == "__main__":
    main()
