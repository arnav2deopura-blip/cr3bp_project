from __future__ import annotations

import random
from functools import partial
from itertools import product
from multiprocessing import Pool

import math
import numpy as np
import pandas as pd

from .config import (
    MU,
    REFERENCE_RTOL,
    REFERENCE_ATOL,
    REFERENCE_MAX_STEP,
    DT_CEILING_PHYSICAL_ABS,
    CHUNK_SIZE,
    N_WORKERS,
    FTLE_ENABLED,
    FTLE_METHOD,
    FTLE_INTEGRATION_TIME,
    SIM_TIME_LOG_MIN,
    SIM_TIME_LOG_MAX,
    ACCURACY_SIM_TIME_LOG_MIN,
    ACCURACY_SIM_TIME_LOG_MAX,
    SAMPLING_WEIGHTS_3D,
    POSITION_TOL_KM,
    TRAINING_DATASET_3D_NAME,
    EJECTION_RADIUS,
    INTEGRATOR_TOLERANCES_3D,
    collision_radii,
    system_params,
    du_km_for_mu,
)
from .physics_3d import (
    generate_3d_orbit,
    compute_ftle_3d_variational,
    compute_ftle_3d_divergence,
    step_rk4_3d,
    add_orbit_features_3d,
    jacobi_constant_3d,
    reference_trajectory_dense_3d,
    initial_inside_body,
    REFERENCE_EVENT_NAMES,
    REASON_REFERENCE_DIVERGENT,
    REASON_OK,
    EVALS_PER_STEP_3D,
)
from .search_3d import (
    find_dt_for_integrator_cr3bp_3d,
    find_dt_for_integrator_accuracy_3d,
    find_dt_for_integrator_accuracy_monotone_3d,
    _dt_upper_bound_3d,
    label_dt_max,
)
from .diagnostics_3d import (
    DIAGNOSTIC_COLUMNS,
    DIAGNOSTICS_VERSION,
    trajectory_diagnostics_3d,
)

LABEL_PIPELINE_VERSION = 3
LABEL_PIPELINE_VERSION_V4 = 4
LABEL_PIPELINE_VERSION_V5 = 5          # Part C: v4 + Sundman-time RK4 labels + periapsis feature columns + optional sampling weights
LABEL_PIPELINE_VERSION_V6 = 6          # Study E: v5 without the FTLE (ftle_3d = NaN, ftle_status = FTLE_STATUS_NOT_COMPUTED); labels identical to v5
FTLE_STATUS_NOT_COMPUTED = -99
ACCURACY_LABEL_MODES = ("accuracy", "accuracy_monotone")
PIPELINES = ("legacy", "v4", "v5", "v6")


def extract_features_3d(state, mu=MU, sim_time: float | None = None, compute_ftle: bool = True):
    """compute_ftle=False (pipeline v6): no FTLE integration; ftle_3d is NaN and ftle_status FTLE_STATUS_NOT_COMPUTED."""
    x, y, z, vx, vy, vz = state
    features = {"x": x, "y": y, "z": z, "vx": vx, "vy": vy, "vz": vz}
    if sim_time is not None:
        features["sim_time"] = sim_time
    else:
        features["sim_time"] = np.nan

    df = pd.DataFrame([features])

    # Use variational FTLE if enabled and method set to "variational"
    use_variational = FTLE_ENABLED and FTLE_METHOD == "variational" and compute_ftle
    # If sim_time is provided, use it as FTLE integration time for relevance
    ftle_T = sim_time if (sim_time is not None and sim_time > 0) else FTLE_INTEGRATION_TIME
    df = add_orbit_features_3d(df, mu, use_variational_ftle=use_variational, ftle_T=ftle_T)

    for col in ["jacobi_3d", "r1_3d", "r2_3d", "r1_r2_ratio_3d", "ftle_3d", "c_minus_c_l1_3d",
                "ftle_status", "ftle_t_frac"]:
        features[col] = df.iloc[0][col]

    if not compute_ftle:
        features.update({"ftle_3d": np.nan, "ftle_status": FTLE_STATUS_NOT_COMPUTED, "ftle_t_frac": np.nan})
    elif not FTLE_ENABLED:
        features["ftle_3d"] = 0.0
    elif FTLE_METHOD == "divergence":
        ftle = compute_ftle_3d_divergence(
            state, mu, T=ftle_T,
            step_func=step_rk4_3d, dt=0.01
        )
        features["ftle_3d"] = ftle

    return features


def label_orbit_3d(state, sim_time, mu=MU):
    """
    Original Jacobi-conservation-based labeling: "stable" means bounded
    relative energy (Jacobi) drift over the horizon. See
    label_orbit_accuracy_3d below for the accuracy-based alternative and why
    it was added -- this function is unchanged; label_mode="jacobi" (the
    default everywhere below) was checked to reproduce the original code's
    rows for orbit indices 0-2 (a narrow check, not a full regression test).
    """
    r_e = find_dt_for_integrator_cr3bp_3d("euler", state, sim_time, mu, return_details=True)
    r_r = find_dt_for_integrator_cr3bp_3d("rk4", state, sim_time, mu, return_details=True)
    r_l = find_dt_for_integrator_cr3bp_3d("leapfrog", state, sim_time, mu, return_details=True)
    return {
        "dt_euler": r_e["dt"], "dt_rk4": r_r["dt"], "dt_leapfrog": r_l["dt"],
        "euler_censored": r_e["censored"], "rk4_censored": r_r["censored"], "leapfrog_censored": r_l["censored"],
        "euler_reason": r_e["reason"], "rk4_reason": r_r["reason"], "leapfrog_reason": r_l["reason"],
        "euler_drift": r_e["drift"], "rk4_drift": r_r["drift"], "leapfrog_drift": r_l["drift"],
        "dt_ceiling": r_r["dt_ceiling"], "C0": r_r["C0"],
        "reference_divergent": False,
    }


def _divergent_label(state, mu, outcome, sim_time=None, ceiling_mode="legacy"):
    return {
        "dt_euler": np.nan, "dt_rk4": np.nan, "dt_leapfrog": np.nan,
        "euler_censored": False, "rk4_censored": False, "leapfrog_censored": False,
        "euler_reason": REASON_REFERENCE_DIVERGENT, "rk4_reason": REASON_REFERENCE_DIVERGENT,
        "leapfrog_reason": REASON_REFERENCE_DIVERGENT,
        "euler_drift": float("nan"), "rk4_drift": float("nan"), "leapfrog_drift": float("nan"),
        "dt_ceiling": label_dt_max(state, sim_time, mu, ceiling_mode) if ceiling_mode == "physical"
        else _dt_upper_bound_3d(state, mu), "C0": jacobi_constant_3d(state, mu),
        "reference_divergent": True, "reference_outcome": outcome,
    }


def _sundman_labels(state, sim_time, mu, tol_km, reference, tau_c):
    """Part C: the largest fictitious step ds of the Sundman-time RK4 that meets the tolerance (same monotone search, ds in
    [1e-4, 4]), and the counted cost of the run at that label."""
    from .physics_3d import Rk4sStep
    from .sundman_3d import DS_MIN, DS_MAX, integrate_endpoint_rk4s, rk4s_evals, SUNDMAN_TAU_C
    tc = SUNDMAN_TAU_C if tau_c is None else float(tau_c)
    r = find_dt_for_integrator_accuracy_monotone_3d("rk4s", state, sim_time, mu, dt_min=DS_MIN, dt_max=DS_MAX, tol_km=tol_km,
                                                   reference=reference, return_details=True, step_func=Rk4sStep(tc))
    out = {"dt_rk4s": r["dt"], "rk4s_censored": r["censored"], "rk4s_reason": r["reason"], "rk4s_drift": r["pos_err_km"],
           "rk4s_ns": np.nan, "rk4s_nf": np.nan, "rk4s_evals": np.nan, "sundman_tau_c": tc}
    if r["reason"] == REASON_OK:
        ok, _p, _c, _tp, _t, _rs, ns, nf = integrate_endpoint_rk4s(state, r["dt"], sim_time, mu, 4_000_000, tau_c=tc)
        if ok:
            out.update({"rk4s_ns": int(ns), "rk4s_nf": int(nf), "rk4s_evals": rk4s_evals(ns, nf)})
    return out


def label_orbit_accuracy_3d(state, sim_time, mu=MU, tol_km=None, monotone=False, ceiling_mode="legacy",
                            diagnostics=False, extra_integrators=(), tau_c=None):
    """
    Accuracy-based labeling: "stable" means the integrator's own final
    position at sim_time lands within tol_km (config.POSITION_TOL_KM if not
    given) of a DOP853 reference trajectory. See search_3d.py's
    find_dt_for_integrator_accuracy_3d module-level comment for the full
    rationale for replacing the Jacobi-drift criterion with this.

    Column names in the returned dict deliberately match label_orbit_3d's
    (dt_euler, euler_censored, euler_reason, euler_drift, dt_ceiling, C0,
    ...) so the saved CSV loads and trains through modeling_3d.py and
    conformal_uncertainty.py without changes -- only the *meaning* of
    "success"/"censored"/"dt_ceiling" changes (position-error-vs-reference
    instead of Jacobi-drift). CAUTION: the downstream *audit* functions
    (modeling_3d.held_out_violation_audit_3d / fit_and_validate_safety_factor_3d,
    validation_3d.tolerance_sensitivity_analysis / dense_sweep_validation /
    validate_on_richardson_halo) still judge "stable" by Jacobi drift, so
    their outputs on accuracy-mode data are NOT consistent with these labels
    until they are updated to use position error. The one new column, "reference_divergent",
    flags orbits where the reference trajectory itself collided or was
    ejected before sim_time, so "accurate vs reference" was undefined for
    all three integrators at once -- filter on it the same way ftle_nan is
    already filtered on elsewhere in this pipeline (run_3d_analysis_from_data.py,
    generalization_tests.py both do `data[~data["ftle_nan"]]`; do the same
    with "reference_divergent" for accuracy-mode data before modeling).
    euler_drift/rk4_drift/leapfrog_drift hold pos_err_km (not a jacobi drift
    value) under this label_mode -- these columns were already just
    diagnostic/logging output, not consumed as ML features anywhere in this
    codebase, so repurposing them is safe.

    UPDATED: ``monotone=True`` uses search_3d.find_dt_for_integrator_accuracy_monotone_3d
    (label_mode="accuracy_monotone"): the largest dt such that it and every tested
    finer dt down to dt/4 meet the tolerance, instead of the first passing dt found
    scanning down. Orbits whose reference collides/escapes (or that start inside a
    body) are returned with reference_divergent=True and a reference_outcome
    string; they are KEPT in the CSV so a crash/escape classifier can be trained
    and evaluated (outcome_3d.py) rather than silently dropped.
    """
    ceiling_mode = ceiling_mode or "legacy"
    if initial_inside_body(state, mu):
        return _divergent_label(state, mu, "initial_inside_body", sim_time, ceiling_mode)

    sol, reached = reference_trajectory_dense_3d(state, sim_time, mu)
    diag = trajectory_diagnostics_3d(sol, state, mu, du_km_for_mu(mu), with_emp=reached, with_check=reached) \
        if diagnostics else {}

    if not reached:
        event = int(getattr(sol, "reference_event", 9))
        lab = _divergent_label(state, mu, REFERENCE_EVENT_NAMES.get(event, "solver_failure"), sim_time, ceiling_mode)
        lab.update(diag)
        return lab

    dt_max = label_dt_max(state, sim_time, mu, ceiling_mode)
    reference = (sol, reached)
    search = find_dt_for_integrator_accuracy_monotone_3d if monotone else find_dt_for_integrator_accuracy_3d

    r_e = search("euler", state, sim_time, mu, dt_max=dt_max,
                 tol_km=tol_km, reference=reference, return_details=True)
    r_r = search("rk4", state, sim_time, mu, dt_max=dt_max,
                 tol_km=tol_km, reference=reference, return_details=True)
    r_l = search("leapfrog", state, sim_time, mu, dt_max=dt_max,
                 tol_km=tol_km, reference=reference, return_details=True)
    extra = _sundman_labels(state, sim_time, mu, tol_km, reference, tau_c) if "rk4s" in extra_integrators else {}
    return {**diag, **extra,
        "dt_euler": r_e["dt"], "dt_rk4": r_r["dt"], "dt_leapfrog": r_l["dt"],
        "euler_censored": r_e["censored"], "rk4_censored": r_r["censored"], "leapfrog_censored": r_l["censored"],
        "euler_reason": r_e["reason"], "rk4_reason": r_r["reason"], "leapfrog_reason": r_l["reason"],
        "euler_drift": r_e["pos_err_km"], "rk4_drift": r_r["pos_err_km"], "leapfrog_drift": r_l["pos_err_km"],
        "dt_ceiling": dt_max, "C0": jacobi_constant_3d(state, mu),
        "reference_divergent": False, "reference_outcome": "ok",
    }


def _sample_orbit_for_generation(mu, label_mode, weights=None):
    """
    Sampling-mode dispatch for generate_one_3d_training_orbit. label_mode
    "jacobi" is UNCHANGED from before (always "mixed": 80% clustered around
    a random L1-L5 point, 20% "wide"). label_mode "accuracy" instead draws
    from config.SAMPLING_WEIGHTS_3D across lagrange/wide/near_moon, so
    near-Moon states (see physics_3d.generate_3d_orbit's "near_moon" mode)
    are actually represented in training -- see that constant's comment in
    config.py for why.
    """
    if label_mode not in ACCURACY_LABEL_MODES:
        return generate_3d_orbit(mu=mu, sampling_mode="mixed", return_label=True)

    roll = np.random.rand()
    w = SAMPLING_WEIGHTS_3D if weights is None else weights
    w_lagrange = w["lagrange"]
    w_wide = w["wide"]
    if roll < w_lagrange:
        points = ["L1", "L2", "L3", "L4", "L5"]
        chosen = np.random.choice(points)
        return generate_3d_orbit(mu=mu, point=str(chosen), sampling_mode="lagrange",
                                  offset=0.1, velocity_perturb=0.15, return_label=True)
    elif roll < w_lagrange + w_wide:
        return generate_3d_orbit(mu=mu, sampling_mode="wide", return_label=True)
    else:
        return generate_3d_orbit(mu=mu, sampling_mode="near_moon", return_label=True)


def generate_one_3d_training_orbit(idx, mu=MU, base_seed=12345, label_mode="jacobi", tol_km=None, pipeline="legacy", tau_c=None,
                                   sampling_weights=None, sim_time_range=None):
    """pipeline="legacy": label_pipeline_version 3 behaviour exactly. pipeline="v4" (Part B): physical step ceiling, per-orbit
    physics diagnostics, success = the search's own reason code (Issue 15) instead of dt > dt_min.
    sim_time_range = (lo, hi) in TU (Study E): the horizon is log-uniform on [lo, hi] instead of the default range; the same single
    random draw is used, so the sampled states are unchanged and None reproduces every earlier dataset exactly."""
    v4 = pipeline in ("v4", "v5", "v6")
    v5 = pipeline in ("v5", "v6")
    v6 = pipeline == "v6"
    seed32 = int(np.random.SeedSequence([int(base_seed), int(idx)]).generate_state(1)[0])
    np.random.seed(seed32)
    random.seed(seed32)

    state, orbit_label = _sample_orbit_for_generation(mu, label_mode, sampling_weights)

    if sim_time_range is not None:
        log_time = random.uniform(math.log10(sim_time_range[0]), math.log10(sim_time_range[1]))
    elif label_mode in ACCURACY_LABEL_MODES:
        log_time = random.uniform(ACCURACY_SIM_TIME_LOG_MIN, ACCURACY_SIM_TIME_LOG_MAX)
    else:
        log_time = random.uniform(SIM_TIME_LOG_MIN, SIM_TIME_LOG_MAX)
    sim_time = 10.0 ** log_time

    features = extract_features_3d(state, mu, sim_time=sim_time, compute_ftle=not v6)
    ftle_nan = (not v6) and not np.isfinite(features.get("ftle_3d", 0.0))
    row = {**features, "sim_time": sim_time, "orbit_label": orbit_label, "ftle_nan": ftle_nan,
           "orbit_idx": idx}
    if label_mode in ACCURACY_LABEL_MODES:
        row["mu"] = mu
        row["du_km"] = du_km_for_mu(mu)
        row["tol_km"] = POSITION_TOL_KM if tol_km is None else tol_km

    if ftle_nan:
        row.update({
            "dt_euler": np.nan, "dt_rk4": np.nan, "dt_leapfrog": np.nan,
            "euler_success": 0, "rk4_success": 0, "leapfrog_success": 0,
            "reference_divergent": False, "reference_outcome": "unlabeled_ftle_nan",
        })
        return {"row": row, "usable": 0, "combo": (0, 0, 0), "ftle_nan": True}

    if label_mode in ACCURACY_LABEL_MODES:
        labels = label_orbit_accuracy_3d(state, sim_time, mu, tol_km=tol_km,
                                         monotone=(label_mode == "accuracy_monotone"),
                                         ceiling_mode="physical" if v4 else "legacy", diagnostics=v4,
                                         extra_integrators=("rk4s",) if v5 else (), tau_c=tau_c)
    else:
        labels = label_orbit_3d(state, sim_time, mu)
    row.update(labels)

    if v4:
        success_e = int(labels["euler_reason"] == REASON_OK)
        success_r = int(labels["rk4_reason"] == REASON_OK)
        success_l = int(labels["leapfrog_reason"] == REASON_OK)
    else:
        success_e = int(labels["dt_euler"] > 0.0001)
        success_r = int(labels["dt_rk4"] > 0.0001)
        success_l = int(labels["dt_leapfrog"] > 0.0001)
    row["euler_success"] = success_e
    row["rk4_success"] = success_r
    row["leapfrog_success"] = success_l
    if v5:
        from .osculating_3d import osc_features, OSC_FEATURE_COLS, OSC_FLAG_COLS
        row["rk4s_success"] = int(labels.get("rk4s_reason", -1) == REASON_OK)
        row["sampling_branch"] = "near_moon" if orbit_label == "near_moon" else ("wide" if orbit_label == "wide" else "lagrange")
        f_, fl_ = osc_features(np.asarray(state, dtype=float)[None, :], np.array([sim_time]), mu)
        row.update({c: float(v) for c, v in zip(OSC_FEATURE_COLS + OSC_FLAG_COLS, np.concatenate([f_[0], fl_[0]]))})

    usable = int(success_e or success_r or success_l)
    return {"row": row, "usable": usable, "combo": (success_e, success_r, success_l)}


def generate_3d_training_dataset(
    num_orbits: int = 5000,
    n_workers: int = N_WORKERS,
    chunk_size: int = CHUNK_SIZE,
    mu: float = MU,
    label_mode: str = "jacobi",
    tol_km: float | None = None,
    base_seed: int = 12345,
    tol_du: float | None = None,
    pipeline: str = "legacy",
    tau_c: float | None = None,
    sampling_weights: dict | None = None,
    sim_time_range: tuple | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    label_mode="jacobi" (default): unchanged original behavior --
    "mixed" sampling, sim_time in config.SIM_TIME_LOG_MIN/MAX (1-100 TU),
    Jacobi-drift-based stability (label_orbit_3d). Existing callers that
    don't pass label_mode get the same generation logic as before this
    change.

    label_mode="accuracy": lagrange/wide/near_moon sampling per
    config.SAMPLING_WEIGHTS_3D, sim_time in config.ACCURACY_SIM_TIME_LOG_MIN/MAX
    (1-20 TU, mission scale), DOP853-reference position-error-based
    stability (label_orbit_accuracy_3d) at tol_km (config.POSITION_TOL_KM if
    None). This is substantially more expensive per orbit (a DOP853
    reference plus three position-tracked fixed-step searches, vs three
    Jacobi-drift searches) -- expect a full 20,000-orbit regeneration to
    take considerably longer than the original run; consider starting with
    a smaller num_orbits to sanity-check timing before committing to a full
    run.

    label_mode="accuracy_monotone": same sampling/horizon/reference as
    "accuracy", but the dt label is the locally-monotone envelope from
    search_3d.find_dt_for_integrator_accuracy_monotone_3d.

    base_seed: orbit i is seeded from SeedSequence([base_seed, i]), so different
    base seeds give statistically independent datasets (output file name gets a
    _seed<N> suffix when base_seed != 12345). Datasets with label_pipeline_version
    2 seeded orbit i with base_seed + i, so two of them with nearby base seeds
    are the SAME orbits shifted by a few indices; protocol_3d.check_independent_datasets
    refuses to compare such datasets. tol_du: position tolerance in this system's own
    length unit (DU); if given and tol_km is None, tol_km = tol_du * DU(mu),
    so Earth-Moon and Sun-Earth datasets can be generated at the same
    NORMALIZED tolerance (100 km Earth-Moon = POSITION_TOL_DU_EARTH_MOON DU).
    """
    if label_mode not in ("jacobi",) + ACCURACY_LABEL_MODES:
        raise ValueError("label_mode must be 'jacobi', 'accuracy' or 'accuracy_monotone'")
    if pipeline not in PIPELINES:
        raise ValueError(f"pipeline must be one of {PIPELINES}")
    if pipeline in ("v4", "v5", "v6") and label_mode not in ACCURACY_LABEL_MODES:
        raise ValueError(f"pipeline '{pipeline}' needs an accuracy label mode")
    if tol_du is not None and tol_km is None:
        tol_km = float(tol_du) * du_km_for_mu(mu)

    print(f"Generating {num_orbits} 3D CR3BP training orbits (μ={mu:.6f}, label_mode={label_mode})...")
    if FTLE_ENABLED:
        print(f"  FTLE computation ENABLED (method: {FTLE_METHOD})")
    else:
        print("  FTLE computation DISABLED (ftle_3d will be 0).")
    if label_mode in ACCURACY_LABEL_MODES:
        effective_tol_km = POSITION_TOL_KM if tol_km is None else tol_km
        print(f"  Accuracy search: tol_km={effective_tol_km}, DU_KM={du_km_for_mu(mu)}, "
              f"sim_time range=[{10.0**ACCURACY_SIM_TIME_LOG_MIN:.1f}, {10.0**ACCURACY_SIM_TIME_LOG_MAX:.1f}] TU, "
              f"sampling weights={SAMPLING_WEIGHTS_3D}")

    rows = []
    usable_count = 0
    combo_counts = {combo: 0 for combo in product([0, 1], repeat=3)}
    worker_func = partial(generate_one_3d_training_orbit, mu=mu, label_mode=label_mode, tol_km=tol_km,
                          base_seed=base_seed, pipeline=pipeline, tau_c=tau_c, sampling_weights=sampling_weights,
                          sim_time_range=sim_time_range)

    with Pool(processes=n_workers) as pool:
        results = pool.imap_unordered(worker_func, range(num_orbits), chunksize=chunk_size)
        ftle_nan_count = 0
        for result in results:
            rows.append(result["row"])
            if result.get("ftle_nan"):
                ftle_nan_count += 1
                continue
            combo_counts[result["combo"]] += 1
            if result["usable"]:
                usable_count += 1

    df = pd.DataFrame(rows)

    basic_cols = ["x", "y", "z", "vx", "vy", "vz", "sim_time"]
    phys_cols = ["jacobi_3d", "r1_3d", "r2_3d", "r1_r2_ratio_3d", "ftle_3d", "c_minus_c_l1_3d"]
    target_cols = ["dt_euler", "dt_rk4", "dt_leapfrog"]
    success_cols = ["euler_success", "rk4_success", "leapfrog_success"]
    status_cols = ["euler_censored", "rk4_censored", "leapfrog_censored",
                   "euler_reason", "rk4_reason", "leapfrog_reason",
                   "euler_drift", "rk4_drift", "leapfrog_drift",
                   "dt_ceiling", "C0", "ftle_nan", "reference_divergent",
                   "reference_outcome", "ftle_status", "ftle_t_frac"]
    extra_cols = ["orbit_label", "orbit_idx", "mu", "du_km", "tol_km"] + list(DIAGNOSTIC_COLUMNS)
    if pipeline in ("v5", "v6"):
        from .osculating_3d import OSC_FEATURE_COLS, OSC_FLAG_COLS
        extra_cols += ["dt_rk4s", "rk4s_success", "rk4s_censored", "rk4s_reason", "rk4s_drift", "rk4s_ns", "rk4s_nf", "rk4s_evals",
                       "sundman_tau_c", "sampling_branch"] + OSC_FEATURE_COLS + OSC_FLAG_COLS

    all_cols = basic_cols + phys_cols + target_cols + success_cols + status_cols + extra_cols
    existing_cols = [c for c in all_cols if c in df.columns]
    df = df[existing_cols]

    fname = TRAINING_DATASET_3D_NAME
    if abs(mu - MU) > 1e-12:
        fname = f"cr3bp_3d_training_mu_{mu:.6f}.csv"
    if label_mode in ACCURACY_LABEL_MODES:
        fname = fname.replace(".csv", f"_{label_mode}.csv")
    if base_seed != 12345:
        fname = fname.replace(".csv", f"_seed{base_seed}.csv")
    df.to_csv(fname, index=False)
    print(f"Saved {len(df)} rows ({usable_count} usable, {ftle_nan_count} FTLE-NaN) to {fname}")

    labeled_mask = ~df["ftle_nan"]
    if "reference_divergent" in df.columns:
        labeled_mask = labeled_mask & ~df["reference_divergent"].fillna(False)
    denom = int(labeled_mask.sum())
    failure_rates = {}
    for name, col in [("Euler", "euler_success"), ("RK4", "rk4_success"), ("Leapfrog", "leapfrog_success")]:
        fails = denom - int(df.loc[labeled_mask, col].sum())
        failure_rates[name] = fails / denom if denom else float("nan")

    combo_report = {
        "num_candidates": num_orbits,
        "usable_count": usable_count,
        "all_failed_count": combo_counts[(0, 0, 0)],
        "combo_counts": combo_counts,
        "ftle_nan_count": ftle_nan_count,
        "reference_divergent_count": int(df["reference_divergent"].sum()) if "reference_divergent" in df.columns else 0,
    }

    import json, sys, sklearn
    r1_crash, r2_crash = collision_radii(mu)
    metadata = {
        "mu": mu,
        "system": system_params(mu)["name"],
        "label_pipeline_version": {"v4": LABEL_PIPELINE_VERSION_V4, "v5": LABEL_PIPELINE_VERSION_V5, "v6": LABEL_PIPELINE_VERSION_V6}.get(pipeline, LABEL_PIPELINE_VERSION),
        "pipeline": pipeline,
        "seeding": "seedsequence",
        "collision_radius_primary_du": r1_crash,
        "collision_radius_secondary_du": r2_crash,
        "ejection_radius": EJECTION_RADIUS,
        "label_mode": label_mode,
        "num_orbits": num_orbits,
        "base_seed": base_seed,
        "sim_time_range_tu": None if sim_time_range is None else [float(sim_time_range[0]), float(sim_time_range[1])],
        "ftle_enabled": FTLE_ENABLED and pipeline != "v6",
        "ftle_method": FTLE_METHOD,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
    }
    if label_mode in ACCURACY_LABEL_MODES:
        metadata.update({
            "integrator_tolerances": None,
            "position_tol_km": POSITION_TOL_KM if tol_km is None else tol_km,
            "du_km": du_km_for_mu(mu),
            "position_tol_du": (POSITION_TOL_KM if tol_km is None else tol_km) / du_km_for_mu(mu),
            "sim_time_log_range": [ACCURACY_SIM_TIME_LOG_MIN, ACCURACY_SIM_TIME_LOG_MAX],
            "sampling_weights": SAMPLING_WEIGHTS_3D if sampling_weights is None else sampling_weights,
        })
    else:
        metadata.update({
            "integrator_tolerances": INTEGRATOR_TOLERANCES_3D,
            "sim_time_log_range": [SIM_TIME_LOG_MIN, SIM_TIME_LOG_MAX],
        })
    if pipeline in ("v5", "v6"):
        from .sundman_3d import SUNDMAN_TAU_C, DS_MIN, DS_MAX
        metadata.update({"sundman_tau_c": SUNDMAN_TAU_C if tau_c is None else tau_c, "sundman_ds_range": [DS_MIN, DS_MAX]})
    if pipeline in ("v4", "v5", "v6"):
        metadata.update({
            "ceiling_mode": "physical", "dt_ceiling_abs": DT_CEILING_PHYSICAL_ABS,
            "reference_rtol": REFERENCE_RTOL, "reference_atol": REFERENCE_ATOL, "reference_max_step": REFERENCE_MAX_STEP,
            "diagnostics_version": DIAGNOSTICS_VERSION, "evals_per_step": dict(EVALS_PER_STEP_3D),
            "success_definition": "search reason == REASON_OK",
        })
    meta_fname = fname.replace(".csv", "_metadata.json")
    with open(meta_fname, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {meta_fname}")

    return df, failure_rates, combo_report