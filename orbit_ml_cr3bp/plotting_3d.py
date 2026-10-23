from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from .config import MU, EJECTION_RADIUS, collision_radii
from .physics_3d import (
    EVALS_PER_STEP_3D,
    generate_3d_orbit,
    jacobi_constant_3d,
    step_euler_3d,
    step_leapfrog_symplectic_3d,
    step_rk4_3d,
)
from .modeling_3d import predict_hurdle

INTEGRATORS_NAMES_3D = ["euler", "rk4", "leapfrog"]
COLORS_3D = {"euler": "red", "rk4": "blue", "leapfrog": "green"}


def create_all_figures_3d(
    classifier_acc: dict,
    regressor_r2: dict,
    regressor_mae: dict,
    sel_acc: float,
    always_rk4_acc: float,
    imp_df: pd.DataFrame,
    sens_df: pd.DataFrame,
    dense_df: pd.DataFrame,
    acc_df: pd.DataFrame,
    val_df: pd.DataFrame,
    conformal_df: pd.DataFrame,
) -> None:
    print("Generating 3D figures...")
    _fig_classifier_accuracy(classifier_acc)
    _fig_regressor_performance(regressor_r2, regressor_mae)
    _fig_selection_accuracy(sel_acc, always_rk4_acc)
    _fig_feature_importance(imp_df)
    _fig_tolerance_sensitivity(sens_df)
    _fig_dense_sweep(dense_df)
    _fig_accuracy_comparison(acc_df)
    _fig_richardson_validation(val_df)
    _fig_conformal(conformal_df)
    print("3D figures saved.")


def _fig_classifier_accuracy(classifier_acc):
    if not classifier_acc:
        return
    names = [i for i in INTEGRATORS_NAMES_3D if i in classifier_acc]
    vals = [classifier_acc[i] for i in names]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar([n.capitalize() for n in names], vals, color=[COLORS_3D[n] for n in names])
    ax.set_ylabel("Classifier Accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_title("3D: Feasibility Classifier Accuracy (held-out)")
    ax.grid(True, alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig("fig3d_01_classifier_accuracy.png", dpi=300)
    plt.close(fig)


def _fig_regressor_performance(regressor_r2, regressor_mae):
    if not regressor_r2:
        return
    names = [i for i in INTEGRATORS_NAMES_3D if i in regressor_r2]
    r2_vals = [regressor_r2[n] if regressor_r2[n] is not None else np.nan for n in names]
    mae_vals = [regressor_mae[n] if regressor_mae[n] is not None else np.nan for n in names]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar([n.capitalize() for n in names], r2_vals, color=[COLORS_3D[n] for n in names])
    ax1.set_ylabel("R\u00b2 (successes only)")
    ax1.set_title("3D: Regressor R\u00b2")
    ax1.grid(True, alpha=0.3)
    ax2.bar([n.capitalize() for n in names], mae_vals, color=[COLORS_3D[n] for n in names])
    ax2.set_ylabel("MAE (successes only)")
    ax2.set_title("3D: Regressor MAE")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig3d_02_regressor_performance.png", dpi=300)
    plt.close(fig)


def _fig_selection_accuracy(sel_acc, always_rk4_acc):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Model", "Always RK4"], [sel_acc, always_rk4_acc], color=["blue", "gray"])
    ax.set_ylabel("Selection Accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_title("3D: Integrator Selection Accuracy")
    ax.grid(True, alpha=0.3)
    for i, v in enumerate([sel_acc, always_rk4_acc]):
        ax.text(i, v + 0.015, f"{100*v:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig("fig3d_03_selection_accuracy.png", dpi=300)
    plt.close(fig)


def _fig_feature_importance(imp_df):
    if imp_df is None or len(imp_df) == 0:
        print("  Skipping feature importance: no data.")
        return
    df = imp_df.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df["feature"], df["importance"], xerr=df["std"])
    ax.set_xlabel("Permutation Importance (RK4 regressor)")
    ax.set_title("3D: Feature Importance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig3d_04_feature_importance.png", dpi=300)
    plt.close(fig)


def _fig_tolerance_sensitivity(sens_df):
    if sens_df is None or len(sens_df) == 0:
        print("  Skipping tolerance sensitivity: no data.")
        return
    tolerances = sorted(sens_df["tolerance"].unique())
    data = [sens_df[sens_df["tolerance"] == t]["dt_rk4"].values for t in tolerances]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(data)
    ax.set_xticks(range(1, len(tolerances) + 1))
    ax.set_xticklabels([str(t) for t in tolerances])
    ax.set_xlabel("Jacobi Drift Tolerance")
    ax.set_ylabel("Stable dt (RK4), per-orbit distribution")
    ax.set_title("3D: Tolerance Sensitivity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig3d_05_tolerance_sensitivity.png", dpi=300)
    plt.close(fig)


def _fig_dense_sweep(dense_df):
    if dense_df is None or len(dense_df) == 0:
        print("  Skipping dense sweep: no data.")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    errs = dense_df["relative_error"].clip(lower=1e-6)
    ax.hist(errs, bins=30)
    ax.set_yscale("log")
    ax.set_xlabel("Relative Error (search dt vs dense grid dt)")
    ax.set_ylabel("Count")
    ax.set_title("3D: Dense Sweep Validation\n(non-monotonic stability inflates this metric)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig3d_06_dense_sweep_validation.png", dpi=300)
    plt.close(fig)


def _fig_accuracy_comparison(acc_df):
    if acc_df is None or len(acc_df) == 0:
        print("  Skipping accuracy comparison: no data.")
        return
    names = [i for i in INTEGRATORS_NAMES_3D if i in acc_df["integrator"].unique()]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    pos_data = [acc_df[acc_df["integrator"] == n]["pos_err"].values for n in names]
    ax1.boxplot(pos_data)
    ax1.set_xticks(range(1, len(names) + 1))
    ax1.set_xticklabels([n.capitalize() for n in names])
    ax1.set_yscale("log")
    ax1.set_ylabel("Final Position Error (vs DOP853)")
    ax1.set_title("3D: Position Error")
    ax1.grid(True, alpha=0.3)
    phase_data = [acc_df[acc_df["integrator"] == n]["phase_err"].values for n in names]
    ax2.boxplot(phase_data)
    ax2.set_xticks(range(1, len(names) + 1))
    ax2.set_xticklabels([n.capitalize() for n in names])
    ax2.set_ylabel("Final Phase Error (rad)")
    ax2.set_title("3D: Phase Error")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig3d_07_accuracy_comparison.png", dpi=300)
    plt.close(fig)


def _fig_richardson_validation(val_df):
    if val_df is None or len(val_df) == 0:
        print("  Skipping Richardson validation: no data.")
        return
    grouped = val_df.groupby(["point", "Az", "integrator"])["stable"].mean().reset_index()

    combos = grouped[["point", "Az"]].drop_duplicates().sort_values(["point", "Az"])
    combos["label"] = combos["point"] + " Az=" + combos["Az"].astype(str)
    label_order = combos["label"].tolist()

    fig, ax = plt.subplots(figsize=(9, 6))
    for integrator in INTEGRATORS_NAMES_3D:
        subset = grouped[grouped["integrator"] == integrator].copy()
        if len(subset) == 0:
            continue
        subset["label"] = subset["point"] + " Az=" + subset["Az"].astype(str)
        subset = subset.set_index("label").reindex(label_order)
        ax.plot(label_order, subset["stable"].values, marker="o",
                label=integrator.capitalize(), color=COLORS_3D[integrator])
    ax.set_ylabel("Stable Fraction")
    ax.set_title("3D: Halo-Orbit Validation\n(differentially-corrected ICs, periodic to ~1e-10)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig("fig3d_08_richardson_validation.png", dpi=300)
    plt.close(fig)


def _fig_conformal(conformal_df):
    if conformal_df is None or len(conformal_df) == 0:
        print("  Skipping conformal prediction: no data.")
        return

    # "covered_uncensored" (added by conformal_uncertainty.py) excludes
    # rk4_censored rows, whose recorded dt_rk4 is a lower bound rather
    # than a true value -- "coverage" against a lower bound is not a
    # well-posed statistic, so it must not be mixed into the headline
    # number. Fall back to plain "covered" only for older conformal_df
    # objects that predate this column.
    has_uncensored_col = (
        "covered_uncensored" in conformal_df.columns
        and conformal_df["covered_uncensored"].notna().any()
    )
    if has_uncensored_col:
        coverage = conformal_df["covered_uncensored"].mean(skipna=True)
        censored_frac = conformal_df["covered_uncensored"].isna().mean()
        title = f"3D: Conformal Prediction (coverage={100*coverage:.1f}% on non-censored orbits; {100*censored_frac:.0f}% censored, excluded)"
    else:
        coverage = conformal_df["covered"].mean()
        title = f"3D: Conformal Prediction (coverage={100*coverage:.1f}%)"

    values = conformal_df["conformal_width"].values
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    rng = vmax - vmin
    width = float(np.nanmean(values))

    fig, ax = plt.subplots(figsize=(7, 5))
    if rng < 1e-8 * max(abs(width), 1e-12):
        ax.bar(["RK4 interval"], [width], color="steelblue")
        ax.set_ylabel("Interval Width (dt_rk4)")
        ax.set_title(title.replace("3D: Conformal Prediction", f"3D: Conformal Prediction (fixed width={width:.5f}"))
    else:
        ax.hist(values, bins=30)
        ax.set_xlabel("Conformal Interval Width (RK4 dt)")
        ax.set_ylabel("Count")
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig3d_09_conformal_prediction.png", dpi=300)
    plt.close(fig)


# =============================================================================
# SHARED 3D TRAJECTORY INTEGRATOR
# =============================================================================
# Single source of truth for collision/ejection break logic, used by both
# the trajectory+drift figure and the Poincare section figure below.

def _run_trajectory_3d(state, dt, sim_time, step_func, mu=MU, max_steps=200_000):
    states = [list(state)]
    times = [0.0]
    curr = list(state)
    t = 0.0
    steps = 0
    cr1, cr2 = collision_radii(mu)
    while t < sim_time and steps < max_steps:
        curr = step_func(curr, dt, mu)
        r1 = np.sqrt((curr[0] + mu) ** 2 + curr[1] ** 2 + curr[2] ** 2)
        r2 = np.sqrt((curr[0] + mu - 1.0) ** 2 + curr[1] ** 2 + curr[2] ** 2)
        r_origin = np.linalg.norm(curr[:3])
        if r1 < cr1 or r2 < cr2 or r_origin > EJECTION_RADIUS:
            break
        if not np.all(np.isfinite(curr)):
            break
        states.append(list(curr))
        t += dt
        steps += 1
        times.append(t)
    return states, times


def _integrate_3d_trajectory(state, dt, sim_time, step_func, mu=MU, max_steps=200_000):
    states, times = _run_trajectory_3d(state, dt, sim_time, step_func, mu, max_steps)
    path = np.array([s[:3] for s in states])
    C0 = jacobi_constant_3d(states[0], mu)
    drift = []
    for s in states:
        C = jacobi_constant_3d(s, mu)
        drift.append(abs((C - C0) / C0) if abs(C0) > 1e-12 else abs(C - C0))
    return path, np.array(times), np.array(drift)


# =============================================================================
# TRAJECTORY + JACOBI DRIFT (3D)
# =============================================================================

def _fig_trajectory_drift_3d(mu=MU, dt=0.01, sim_time=50.0):
    state0 = generate_3d_orbit(mu=mu, point="L1", offset=0.05, velocity_perturb=0.02, sampling_mode="lagrange")

    fig = plt.figure(figsize=(13, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)

    for name, step_func in [("Euler", step_euler_3d), ("RK4", step_rk4_3d), ("Leapfrog", step_leapfrog_symplectic_3d)]:
        key = name.lower()
        path, times, drift = _integrate_3d_trajectory(state0, dt, sim_time, step_func, mu)
        if len(path) < 2:
            continue
        ax1.plot(path[:, 0], path[:, 1], path[:, 2], color=COLORS_3D[key], label=name, alpha=0.8)
        ax2.semilogy(times, np.clip(drift, 1e-16, None), color=COLORS_3D[key], label=name)

    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
    ax1.set_title(f"3D Trajectories (dt={dt})")
    ax1.legend()

    ax2.set_xlabel("Time")
    ax2.set_ylabel("Relative Jacobi Drift")
    ax2.set_title(f"3D Jacobi Drift (dt={dt})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("fig3d_10_trajectory_drift.png", dpi=300)
    plt.close(fig)


# =============================================================================
# CHAOTIC HOLDOUT (3D)
# =============================================================================

def _fig_chaotic_holdout_3d(context):
    test_data = context["data_test"]
    pred_basic = context["pred_basic"]
    pred_phys = context["pred_phys"]
    succ_mask = test_data["rk4_success"].values == 1
    if succ_mask.sum() == 0:
        print("  Skipping chaotic holdout 3D: no RK4 successes in holdout.")
        return
    y_true = test_data["dt_rk4"].values[succ_mask]
    err_basic = np.abs(pred_basic["rk4"]["dt"][succ_mask] - y_true)
    err_phys = np.abs(pred_phys["rk4"]["dt"][succ_mask] - y_true)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([err_basic, err_phys])
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Baseline", "Physics-Aware"])
    ax.set_ylabel("Absolute Prediction Error (RK4)")
    ax.set_yscale("log")
    ax.set_title(f"3D Chaotic Holdout ({context['split_col']})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig3d_11_chaotic_holdout.png", dpi=300)
    plt.close(fig)


# =============================================================================
# FEATURE ABLATION (3D)
# =============================================================================

def _fig_ablation_3d(report):
    integrators = ["Euler", "RK4", "Leapfrog"]
    keys3 = ["euler", "rk4", "leapfrog"]
    x_pos = np.arange(3)
    width = 0.35
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def _bars(ax, base_key, phys_key, ylabel, title):
        b_mean = [report[base_key][i][0] for i in keys3]
        b_err = [report[base_key][i][1] for i in keys3]
        p_mean = [report[phys_key][i][0] for i in keys3]
        p_err = [report[phys_key][i][1] for i in keys3]
        ax.bar(x_pos - width / 2, b_mean, width, yerr=b_err, label="Basic (3D)", capsize=3)
        ax.bar(x_pos + width / 2, p_mean, width, yerr=p_err, label="Physics-aware (3D)", capsize=3)
        ax.set_xticks(x_pos); ax.set_xticklabels(integrators)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.3)

    _bars(axes[0, 0], "baseline_clf_acc", "phys_clf_acc", "Classifier Accuracy", "3D Stage 1: Feasibility")
    _bars(axes[0, 1], "baseline_clf_f1", "phys_clf_f1", "Classifier F1", "3D Stage 1: F1")
    _bars(axes[1, 0], "baseline_reg_r2", "phys_reg_r2", "R\u00b2 (successes only)", "3D Stage 2: Regression R\u00b2")
    _bars(axes[1, 1], "baseline_reg_mae", "phys_reg_mae", "MAE (successes only)", "3D Stage 2: Regression MAE")

    fig.tight_layout()
    fig.savefig("fig3d_13_ablation_study.png", dpi=300)
    plt.close(fig)


# =============================================================================
# LEARNING CURVE (3D)
# =============================================================================

def _fig_dataset_size_3d(study_df):
    if study_df is None or len(study_df) == 0:
        print("  Skipping learning curve 3D: no data.")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(study_df["size"], study_df["mae"], "o-", color="blue")
    ax1.set_xlabel("Training Orbits"); ax1.set_ylabel("MAE (RK4, successes)")
    ax1.set_title("3D: Error vs Dataset Size"); ax1.grid(True, alpha=0.3)
    ax2.plot(study_df["size"], study_df["variance"], "o-", color="purple")
    ax2.set_xlabel("Training Orbits"); ax2.set_ylabel("Prediction Variance (RK4)")
    ax2.set_title("3D: Uncertainty vs Dataset Size"); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig3d_14_learning_curves.png", dpi=300)
    plt.close(fig)


# =============================================================================
# SPEED COMPARISON (3D)
# =============================================================================

def _fig_speed_comparison_3d(speed_report):
    fig, ax = plt.subplots(figsize=(6, 5))
    names = ["ML Prediction", "Brute Force"]
    times_ = [speed_report["ml_time_ms"], speed_report["brute_time_ms"]]
    ax.bar(names, times_, color=["blue", "red"])
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"3D Speed Comparison ({speed_report['speedup']:.0f}x)")
    ax.grid(True, alpha=0.3)
    for i, v in enumerate(times_):
        ax.text(i, v + 0.05, f"{v:.3f} ms", ha="center")
    fig.tight_layout()
    fig.savefig("fig3d_15_speed_comparison.png", dpi=300)
    plt.close(fig)


# =============================================================================
# FTLE CORRELATION (3D)
# =============================================================================

def _fig_ftle_correlation_3d(data, final_model, ftle_column="ftle_3d"):
    if ftle_column not in data.columns or data[ftle_column].sum() == 0:
        print("  Skipping FTLE correlation 3D: column unavailable or all zero.")
        return
    succ_mask = data["rk4_success"].values == 1
    if succ_mask.sum() == 0:
        return
    pred = predict_hurdle(final_model, data[succ_mask])
    y_pred = pred["rk4"]["dt"]
    y_true = data["dt_rk4"].values[succ_mask]
    finite = np.isfinite(y_pred)
    errors = np.abs(y_pred[finite] - y_true[finite])
    ftle = data[ftle_column].values[succ_mask][finite]
    if len(ftle) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ftle, errors, s=5, alpha=0.5)
    ax.set_xlabel("FTLE (3D)")
    ax.set_ylabel("|Predicted dt - True dt| (RK4)")
    ax.set_title("3D: Prediction Error vs Local Chaos (FTLE)")
    ax.grid(True, alpha=0.3)
    if np.std(ftle) > 0:
        z = np.polyfit(ftle, errors, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(ftle.min(), ftle.max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", label=f"Trend (slope={z[0]:.3f})")
        ax.legend()
    fig.tight_layout()
    fig.savefig("fig3d_12_error_vs_ftle.png", dpi=300)
    plt.close(fig)


# =============================================================================
# RKF45 COMPARISON (3D)
# =============================================================================

def _fig_rkf45_comparison_3d(df_results):
    if df_results is None or len(df_results) == 0:
        print("  Skipping RKF45 comparison 3D: no data.")
        return
    valid = df_results[(df_results["rk4_steps"] > 0) & (df_results["rkf45_steps"] > 0)]
    if len(valid) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["RK4 (ML dt)", "RKF45 (adaptive)"],
           [valid["rk4_evals"].mean(), valid["rkf45_evals"].mean()],
           color=["blue", "orange"])
    ax.set_ylabel("Mean RHS Evaluations (approximate, see caveat in compare_rkf45_3d)")
    ax.set_title("3D: ML-Guided RK4 vs Adaptive RKF45")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig3d_16_rkf45_comparison.png", dpi=300)
    plt.close(fig)


# =============================================================================
# VIOLATION AUDIT + SAFETY FACTOR (3D)
# =============================================================================

def _fig_violation_audit_3d(summary):
    names = [i for i in ["euler", "rk4", "leapfrog"] if summary.get(i) is not None]
    if not names:
        print("  Skipping violation audit 3D: no data.")
        return
    rates = [100.0 * summary[i]["violation_rate"] for i in names]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar([n.capitalize() for n in names], rates, color=[COLORS_3D[n] for n in names])
    ax.set_ylabel("Violation Rate (%)")
    ax.set_title("3D: Held-Out Predictions Violating Stability Criterion")
    ax.grid(True, alpha=0.3)
    for i, v in enumerate(rates):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig("fig3d_17_violation_audit.png", dpi=300)
    plt.close(fig)


def _fig_safety_factor_3d(safety_report):
    per_integrator = safety_report.get("per_integrator")
    if not per_integrator:
        return
    names = [i for i in INTEGRATORS_NAMES_3D if i in per_integrator]
    if not names:
        return
    coverages = [per_integrator[i]["eval_coverage"] * 100.0 for i in names]
    factors = [per_integrator[i]["chosen_factor"] for i in names]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([n.capitalize() for n in names], coverages, color=[COLORS_3D[n] for n in names])
    ax.set_ylabel("Coverage on eval set (%)")
    ax.set_ylim(0, 105)
    ax.set_title("3D: Per-Integrator Safety Factor (eval coverage)")
    ax.grid(True, alpha=0.3)
    for i, (cov, factor) in enumerate(zip(coverages, factors)):
        ax.text(i, cov + 1.5, f"{cov:.1f}%\nfactor={factor}", ha="center")
    fig.tight_layout()
    fig.savefig("fig3d_18_safety_factor.png", dpi=300)
    plt.close(fig)


# =============================================================================
# PHYSICS BASELINES (3D)
# =============================================================================

def _fig_physics_baselines_3d(baseline_results, ml_mae=None):
    if not baseline_results or "error" in baseline_results:
        print("  Skipping physics baselines 3D: no data.")
        return
    names = list(baseline_results.keys())
    maes = [baseline_results[n]["mae"] for n in names]
    labels = [n.capitalize() for n in names]
    colors = ["gray"] * len(names)
    if ml_mae is not None:
        labels.append("ML Model")
        maes.append(ml_mae)
        colors.append("blue")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(labels, maes, color=colors)
    ax.set_ylabel("MAE (RK4 dt)")
    ax.set_title("3D: Physics Baseline Comparison")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig3d_19_physics_baselines.png", dpi=300)
    plt.close(fig)


# =============================================================================
# POINCARE SECTION (3D)
# =============================================================================

def _poincare_section_3d(states, z_threshold=0.0, vz_min=0.0):
    points_x, points_y = [], []
    for i in range(len(states) - 1):
        z0, z1 = states[i][2], states[i + 1][2]
        if (z0 - z_threshold) * (z1 - z_threshold) < 0:
            vz = states[i][5]
            if vz > vz_min:
                frac = (z_threshold - z0) / (z1 - z0 + 1e-12)
                x_interp = states[i][0] + frac * (states[i + 1][0] - states[i][0])
                y_interp = states[i][1] + frac * (states[i + 1][1] - states[i][1])
                points_x.append(x_interp)
                points_y.append(y_interp)
    return np.array(points_x), np.array(points_y)


def _fig_poincare_section_3d(mu=MU, dt=0.005, sim_time=200.0):
    labels_and_modes = [
        ("L1", "lagrange"), ("L2", "lagrange"), ("L3", "lagrange"),
        ("L4", "lagrange"), ("L5", "lagrange"), ("wide", "wide"),
    ]
    colors = {"L1": "red", "L2": "orange", "L3": "purple", "L4": "blue", "L5": "cyan", "wide": "gray"}

    fig, ax = plt.subplots(figsize=(8, 6))
    any_points = False
    for label, mode in labels_and_modes:
        kwargs = dict(mu=mu, sampling_mode=mode, offset=0.05, velocity_perturb=0.02)
        if mode == "lagrange":
            kwargs["point"] = label
        state0 = generate_3d_orbit(**kwargs)
        states, _ = _run_trajectory_3d(state0, dt, sim_time, step_rk4_3d, mu)
        x_pts, y_pts = _poincare_section_3d(states)
        if len(x_pts) > 0:
            ax.scatter(x_pts, y_pts, s=3, c=colors[label], alpha=0.5, label=label)
            any_points = True

    if not any_points:
        print("  Skipping Poincare section 3D: no crossings found for any sampled orbit.")
        plt.close(fig)
        return

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("3D Poincar\u00e9 Section: z = 0, v_z > 0")
    ax.legend(markerscale=5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig3d_20_poincare_section.png", dpi=300)
    plt.close(fig)


# =============================================================================
# INTEGRATOR-SELECTION HEATMAP (3D)
# =============================================================================

def _fig_integrator_heatmap_3d(training_data, n_bins=10):
    if "ftle_3d" not in training_data.columns or "r2_3d" not in training_data.columns:
        print("  Skipping integrator heatmap 3D: missing columns.")
        return
    ftle = training_data["ftle_3d"]
    r2 = training_data["r2_3d"]
    if ftle.std() < 1e-12:
        print("  Skipping integrator heatmap 3D: ftle_3d has no variance (FTLE disabled for this run?).")
        return

    ftle_bins = np.unique(np.percentile(ftle, np.linspace(0, 100, n_bins + 1)))
    r2_bins = np.unique(np.percentile(r2, np.linspace(0, 100, n_bins + 1)))
    if len(ftle_bins) < 2 or len(r2_bins) < 2:
        print("  Skipping integrator heatmap 3D: not enough distinct bin edges.")
        return

    dt_cols = {"euler": "dt_euler", "rk4": "dt_rk4", "leapfrog": "dt_leapfrog"}
    grid = np.full((len(ftle_bins) - 1, len(r2_bins) - 1), np.nan)
    for i in range(len(ftle_bins) - 1):
        for j in range(len(r2_bins) - 1):
            mask = (
                (ftle >= ftle_bins[i]) & (ftle < ftle_bins[i + 1]) &
                (r2 >= r2_bins[j]) & (r2 < r2_bins[j + 1])
            )
            subset = training_data[mask]
            if len(subset) == 0:
                continue
            normalized = {}
            for integ, col in dt_cols.items():
                ok = subset[f"{integ}_success"] == 1
                if ok.sum() == 0:
                    continue
                normalized[integ] = float((subset.loc[ok, col] / EVALS_PER_STEP_3D[integ]).median())
            if not normalized:
                continue
            best = max(normalized, key=normalized.get)
            grid[i, j] = {"euler": 0, "rk4": 1, "leapfrog": 2}[best]

    cmap = ListedColormap(["red", "blue", "green"])
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=-0.5, vmax=2.5, origin="lower")
    ax.set_xticks(range(len(r2_bins) - 1))
    ax.set_xticklabels([f"{r2_bins[i]:.2f}-{r2_bins[i+1]:.2f}" for i in range(len(r2_bins) - 1)], rotation=45, fontsize=8)
    ax.set_yticks(range(len(ftle_bins) - 1))
    ax.set_yticklabels([f"{ftle_bins[i]:.3f}-{ftle_bins[i+1]:.3f}" for i in range(len(ftle_bins) - 1)], fontsize=8)
    ax.set_xlabel("r2_3d (distance to secondary)")
    ax.set_ylabel("ftle_3d")
    ax.set_title("3D: Cost-Normalized Best Integrator by (r2_3d, ftle_3d)\n(marginalized over sim_time)")
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.set_ticklabels(["Euler", "RK4", "Leapfrog"])
    fig.tight_layout()
    fig.savefig("fig3d_21_integrator_heatmap.png", dpi=300)
    plt.close(fig)


# =============================================================================
# BEST INTEGRATOR BY ORBIT TYPE (3D)
# =============================================================================

def _fig_integrator_by_type_3d(training_data):
    if "orbit_label" not in training_data.columns:
        print("  Skipping integrator-by-type 3D: no orbit_label column.")
        return
    orbit_types = sorted(training_data["orbit_label"].unique())
    if len(orbit_types) <= 1:
        print("  Skipping integrator-by-type 3D: orbit_label has no variation (old-style 'mixed' data).")
        return

    dt_cols = {"euler": "dt_euler", "rk4": "dt_rk4", "leapfrog": "dt_leapfrog"}
    normalized = pd.DataFrame({
        integ: (training_data[col] / EVALS_PER_STEP_3D[integ]).where(
            training_data[f"{integ}_success"] == 1)
        for integ, col in dt_cols.items()
    }, index=training_data.index)

    has_any_success = normalized.notna().any(axis=1)
    if not has_any_success.any():
        print("  Skipping integrator-by-type 3D: no orbit has any successful integrator.")
        return
    normalized = normalized[has_any_success]
    labels = training_data.loc[has_any_success, "orbit_label"]
    best = normalized.idxmax(axis=1)

    fractions = []
    for otype in orbit_types:
        mask = (labels == otype).to_numpy()
        subset_best = best[mask]
        if len(subset_best) == 0:
            fractions.append([np.nan, np.nan, np.nan])
            continue
        fractions.append([
            np.mean(subset_best == "euler"),
            np.mean(subset_best == "rk4"),
            np.mean(subset_best == "leapfrog"),
        ])
    fractions = np.array(fractions)

    x_pos = np.arange(len(orbit_types))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x_pos - width, fractions[:, 0], width, label="Euler", color="red")
    ax.bar(x_pos, fractions[:, 1], width, label="RK4", color="blue")
    ax.bar(x_pos + width, fractions[:, 2], width, label="Leapfrog", color="green")
    ax.set_xticks(x_pos); ax.set_xticklabels(orbit_types)
    ax.set_ylabel("Fraction (cost-normalized best)")
    ax.set_title("3D: Best Integrator by Orbit Type")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig3d_22_integrator_by_type.png", dpi=300)
    plt.close(fig)


# =============================================================================
# ORCHESTRATOR (EXTRA FIGURES)
# =============================================================================

def create_extra_figures_3d(
    chaotic_context=None,
    ablation_report=None,
    dataset_size_df=None,
    speed_report=None,
    ftle_data=None,
    final_model=None,
    rkf45_df=None,
    violation_summary=None,
    safety_report=None,
    baseline_results=None,
    ml_mae=None,
    training_data=None,
):
    print("Generating extra 3D figures...")
    if chaotic_context is not None:
        _fig_chaotic_holdout_3d(chaotic_context)
    if ablation_report is not None:
        _fig_ablation_3d(ablation_report)
    if dataset_size_df is not None:
        _fig_dataset_size_3d(dataset_size_df)
    if speed_report is not None:
        _fig_speed_comparison_3d(speed_report)
    if ftle_data is not None and final_model is not None:
        _fig_ftle_correlation_3d(ftle_data, final_model)
    if rkf45_df is not None:
        _fig_rkf45_comparison_3d(rkf45_df)
    if violation_summary is not None:
        _fig_violation_audit_3d(violation_summary)
    if safety_report is not None:
        _fig_safety_factor_3d(safety_report)
    if baseline_results is not None:
        _fig_physics_baselines_3d(baseline_results, ml_mae)
    if training_data is not None:
        _fig_integrator_heatmap_3d(training_data)
        _fig_integrator_by_type_3d(training_data)
    _fig_trajectory_drift_3d()
    _fig_poincare_section_3d()
    print("Extra 3D figures saved.")