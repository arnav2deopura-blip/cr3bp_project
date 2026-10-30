import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from orbit_ml_cr3bp.config import (
    MU, PHYSICS_FEATURE_COLUMNS_3D, TRAINING_DATASET_3D_NAME, MODEL_3D_NAME,
    SAFETY_FACTOR_CANDIDATES_3D,
)
from orbit_ml_cr3bp.modeling_3d import (
    fit_hurdle_models,
    predict_hurdle,
    train_final_model_3d,
    save_model,
    cost_normalized_choice,
    true_best_integrator_row,
    create_grouped_split_3d,
    integrator_selection_stats,
    evaluate_hurdle_predictions,
    run_failure_classifier,
    summarize_violation_audit,
    print_feature_ablation,
    chaotic_holdout_3d,
    run_feature_ablation_grouped_3d,
    run_dataset_size_study_3d,
    speed_comparison_3d,
    run_ftle_correlation_analysis_3d,
    compare_rkf45_3d,
    held_out_violation_audit_3d,
    fit_and_validate_safety_factor_3d,
)
from orbit_ml_cr3bp.validation_3d import (
    tolerance_sensitivity_analysis,
    dense_sweep_validation,
    accuracy_comparison,
    validate_on_richardson_halo,
)
from orbit_ml_cr3bp.conformal_uncertainty import conformal_prediction_rk4
from orbit_ml_cr3bp.plotting_3d import create_all_figures_3d, create_extra_figures_3d
from orbit_ml_cr3bp.baselines import evaluate_physics_baselines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=TRAINING_DATASET_3D_NAME)
    parser.add_argument("--retrain_final_model", action="store_true")
    parser.add_argument("--skip_extra", action="store_true",
                         help="Skip the new diagnostics (chaotic holdout, ablation, RKF45, "
                              "safety factor, learning curve) -- these are the slow ones.")
    parser.add_argument("--ablation_seeds", type=int, default=10)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    print(f"Loading {args.data}...")
    data = pd.read_csv(args.data)

    if "ftle_nan" in data.columns:
        n_raw = len(data)
        data = data[~data["ftle_nan"]].reset_index(drop=True)
        n_dropped = n_raw - len(data)
        if n_dropped:
            print(f"  Dropped {n_dropped} FTLE-NaN row(s) (excluded from modeling/analysis).")
    if "reference_divergent" in data.columns:
        n_raw = len(data)
        data = data[~data["reference_divergent"].fillna(False).astype(bool)].reset_index(drop=True)
        n_dropped = n_raw - len(data)
        if n_dropped:
            print(f"  Dropped {n_dropped} reference-divergent row(s) (their dt labels are undefined).")
    print(f"  Loaded {len(data)} usable orbits.")

    train_idx, test_idx = create_grouped_split_3d(data, test_size=0.3, random_state=42, n_bins=5)
    data_train = data.iloc[train_idx].reset_index(drop=True)
    data_test = data.iloc[test_idx].reset_index(drop=True)
    print(f"  Train size: {len(data_train)}, Test size: {len(data_test)}")

    phys_cols_3d = [c for c in PHYSICS_FEATURE_COLUMNS_3D if c in data.columns]

    print("Refitting held-out hurdle models (train split only)...")
    models = fit_hurdle_models(data, phys_cols_3d, train_idx, random_state=42, n_estimators=150)
    pred_test = predict_hurdle(models, data_test)

    classifier_acc, regressor_r2, regressor_mae = evaluate_hurdle_predictions(data_test, pred_test)
    for integrator in ["euler", "rk4", "leapfrog"]:
        print(f"  {integrator.capitalize()} classifier acc: {classifier_acc[integrator]:.4f}")
        if regressor_r2[integrator] is not None:
            print(f"    Regressor R\u00b2: {regressor_r2[integrator]:.4f}, "
                  f"MAE: {regressor_mae[integrator]:.5f}")

    selection_report_all = integrator_selection_stats(data_test, pred_test)
    selection_report = integrator_selection_stats(data_test, pred_test, exclude=["euler"])
    sel_acc = selection_report["accuracy"]
    always_rk4_acc = selection_report["rk4_accuracy"]
    print(f"  Selection accuracy (model, euler excluded): {sel_acc:.4f}")
    print(f"  Selection accuracy (model, all 3 candidates): {selection_report_all['accuracy']:.4f}")
    print(f"  Always RK4 accuracy: {always_rk4_acc:.4f}")
    print(f"  True-best fractions (all 3 candidates): euler={selection_report_all['true_euler_fraction']:.3f} "
          f"rk4={selection_report_all['true_rk4_fraction']:.3f} "
          f"leapfrog={selection_report_all['true_leapfrog_fraction']:.3f} "
          f"none={selection_report_all['true_none_feasible_fraction']:.3f}")
    print(f"  True-best fractions (euler excluded): rk4={selection_report['true_rk4_fraction']:.3f} "
          f"leapfrog={selection_report['true_leapfrog_fraction']:.3f} "
          f"none={selection_report['true_none_feasible_fraction']:.3f}")

    print("Permutation importance (RK4 regressor)...")
    imp_df = pd.DataFrame()
    X_test = data_test[phys_cols_3d].values
    y_test_rk4 = data_test["dt_rk4"].values
    mask = data_test["rk4_success"].values == 1
    if mask.sum() > 5:
        reg = models["rk4"]["regressor"]
        imp = permutation_importance(reg, X_test[mask], y_test_rk4[mask],
                                      n_repeats=10, random_state=42, n_jobs=-1)
        imp_df = pd.DataFrame({
            "feature": phys_cols_3d,
            "importance": imp.importances_mean,
            "std": imp.importances_std,
        }).sort_values("importance", ascending=False)
        imp_df.to_csv("3d_feature_importance.csv", index=False)

    print("Tolerance sensitivity analysis...")
    sens_df = tolerance_sensitivity_analysis(data_test, mu=MU, sim_time=20.0, num_samples=100,
                                              tolerances=(1e-5, 1e-4, 0.001))
    if not sens_df.empty:
        sens_df.to_csv("3d_tolerance_sensitivity.csv", index=False)

    print("Dense sweep validation...")
    dense_df = dense_sweep_validation(data_test, mu=MU, sim_time=20.0, num_samples=50,
                                       dt_min=1e-4, dt_max=0.5, n_steps=500, tolerance=1e-5)
    if not dense_df.empty:
        dense_df.to_csv("3d_dense_sweep_validation.csv", index=False)

    print("Accuracy comparison against DOP853 reference...")
    acc_df = accuracy_comparison(data_test, models, mu=MU, sim_time=20.0, num_samples=100)
    if not acc_df.empty:
        acc_df.to_csv("3d_accuracy_comparison.csv", index=False)

    print("Validating on Richardson halo orbits...")
    val_df = validate_on_richardson_halo(models, mu=MU)
    if not val_df.empty:
        val_df.to_csv("3d_richardson_validation.csv", index=False)

    print("Conformal prediction intervals for RK4...")
    conformal_df = pd.DataFrame()
    train_succ = data_train[data_train["rk4_success"] == 1].copy()
    test_succ = data_test[data_test["rk4_success"] == 1].copy()
    if len(train_succ) >= 50 and len(test_succ) >= 10:
        conformal_df = conformal_prediction_rk4(
            train_succ, test_succ, feature_cols=phys_cols_3d, alpha=0.1, random_state=42,
            method="normalized",
        )
        conformal_df.to_csv("3d_conformal_rk4.csv", index=False)
        cov_all = conformal_df["covered"].mean()
        cov_uncensored = (
            conformal_df["covered_uncensored"].mean(skipna=True)
            if "covered_uncensored" in conformal_df.columns else float("nan")
        )
        censored_frac = (
            conformal_df["rk4_censored"].astype(bool).mean()
            if "rk4_censored" in conformal_df.columns else float("nan")
        )
        print(f"  Normalized: coverage (non-censored, the number that matters)={cov_uncensored:.3f}  "
              f"coverage (all rows, includes {100*censored_frac:.1f}% censored)={cov_all:.3f}  "
              f"mean_width={conformal_df['conformal_width'].mean():.5f}")

        conformal_fixed_df = conformal_prediction_rk4(
            train_succ, test_succ, feature_cols=phys_cols_3d, alpha=0.1, random_state=42,
            method="fixed",
        )
        conformal_fixed_df.to_csv("3d_conformal_rk4_fixed_baseline.csv", index=False)
        cov_all_fixed = conformal_fixed_df["covered"].mean()
        cov_uncensored_fixed = (
            conformal_fixed_df["covered_uncensored"].mean(skipna=True)
            if "covered_uncensored" in conformal_fixed_df.columns else float("nan")
        )
        print(f"  Fixed-width baseline: coverage (non-censored)={cov_uncensored_fixed:.3f}  "
              f"coverage (all rows)={cov_all_fixed:.3f}  "
              f"mean_width={conformal_fixed_df['conformal_width'].mean():.5f}")

    if args.retrain_final_model:
        print("Retraining final model on full dataset...")
        final_model = train_final_model_3d(data)
        save_model(final_model, MODEL_3D_NAME)

    create_all_figures_3d(
        classifier_acc=classifier_acc,
        regressor_r2=regressor_r2,
        regressor_mae=regressor_mae,
        sel_acc=sel_acc,
        always_rk4_acc=always_rk4_acc,
        imp_df=imp_df,
        sens_df=sens_df,
        dense_df=dense_df,
        acc_df=acc_df,
        val_df=val_df,
        conformal_df=conformal_df,
    )

    # =========================================================================
    # EXTRA 3D DIAGNOSTICS (audit items A.1-A.11, minus what's blocked/deferred)
    # =========================================================================

    print("\nFailure classifier (rk4_success diagnostic)...")
    fail_result = run_failure_classifier(data, phys_cols_3d, target_col="rk4_success")
    fail_result["feature_importance"].to_csv("3d_failure_classifier_importance.csv", index=False)

    print("\nPhysics baseline comparison (RK4)...")
    baseline_results = evaluate_physics_baselines(data_train, data_test, integrator="rk4")
    print(baseline_results)

    if not args.skip_extra:
        print("\nChaotic vs regular holdout (3D)...")
        chaotic_context = chaotic_holdout_3d(data, random_state=42, n_bins=5)
        print(f"  split_col={chaotic_context['split_col']} train={chaotic_context['train_size']} "
              f"test={chaotic_context['test_size']}")

        print(f"\nFeature ablation study (3D, {args.ablation_seeds} seeds -- this is the slow one)...")
        ablation_report = run_feature_ablation_grouped_3d(data, seed_count=args.ablation_seeds, n_bins=5)
        print_feature_ablation(ablation_report)

        print("\nLearning curve (3D)...")
        dataset_size_df = run_dataset_size_study_3d(data)
        dataset_size_df.to_csv("3d_learning_curve.csv", index=False)

        print("\nSpeed comparison (ML vs brute force, 3D)...")
        speed_report = speed_comparison_3d(models)
        print(f"  ML: {speed_report['ml_time_ms']:.3f} ms  Brute: {speed_report['brute_time_ms']:.1f} ms  "
              f"Speedup: ~{speed_report['speedup']:.0f}x")

        print("\nFTLE correlation with prediction error (3D)...")
        ftle_report = run_ftle_correlation_analysis_3d(data_test, models)
        print(f"  {ftle_report}")

        print("\nRKF45 comparison (3D, approximate step counts -- see caveat in source)...")
        rkf45_summary, rkf45_df = compare_rkf45_3d(models, data_test, phys_cols_3d, num_samples=100, sim_time=20.0)
        print(f"  {rkf45_summary}")
        if isinstance(rkf45_df, pd.DataFrame) and not rkf45_df.empty:
            rkf45_df.to_csv("3d_rkf45_comparison.csv", index=False)

        print("\nHeld-out violation audit + safety factor (3D)...")
        audit_df = held_out_violation_audit_3d(data_test, pred_test, safety_factor=1.0, sim_time=20.0)
        violation_summary = summarize_violation_audit(audit_df)
        print(f"  {violation_summary}")
        safety_report = fit_and_validate_safety_factor_3d(data_test, pred_test, sim_time=20.0)
        for integrator in ["euler", "rk4", "leapfrog"]:
            info = safety_report["per_integrator"].get(integrator)
            if info is None:
                continue
            print(f"  {integrator}: chosen_factor={info['chosen_factor']}  "
                  f"eval_coverage={info['eval_coverage']:.3f}  target_met={info['target_met']}")
            if not info["target_met"]:
                print(f"    WARNING: {integrator} did not reach the <=5% tune-set violation "
                      f"target down to factor={min(SAFETY_FACTOR_CANDIDATES_3D)} "
                      f"(best tune violation: {info['tune_violation_at_chosen_factor']:.1%}).")

        create_extra_figures_3d(
            chaotic_context=chaotic_context,
            ablation_report=ablation_report,
            dataset_size_df=dataset_size_df,
            speed_report=speed_report,
            ftle_data=data_test,
            final_model=models,
            rkf45_df=rkf45_df,
            violation_summary=violation_summary,
            safety_report=safety_report,
            baseline_results=baseline_results,
            ml_mae=regressor_mae.get("rk4"),
            training_data=data,
        )
    else:
        create_extra_figures_3d(baseline_results=baseline_results, ml_mae=regressor_mae.get("rk4"),
                                 training_data=data)


if __name__ == "__main__":
    main()