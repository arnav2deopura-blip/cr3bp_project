# Changes: fixes 1-8 (and three extra bugs found while doing them)

All datasets and results generated BEFORE these changes are invalid for new claims (see "Extra bugs"). Regenerate.
New datasets carry `label_pipeline_version: 2` in their `_metadata.json`; `run_rigor_suite_3d.py` warns when it is missing.

## Where each fix lives
1. Collision radii: `config.py` (`SYSTEMS`, `system_params`, `collision_radii`; Earth 6371 km = 0.01657 DU, Moon 1737.4 km = 0.00452 DU;
   Sun-Earth uses Sun 695,700 km / Earth 6371 km in its own DU = 1 au). `physics_3d.py` kernels take separate primary/secondary radii.
   Near-Moon sampling now log-uniform from 1.2 x Moon radius (~0.0054 DU) to 0.15 DU. Orbits starting inside a body are flagged.
2. Full cost: `fair_baseline_3d.py` (FTLE + prediction cost, `ml_total_evals_equiv`, calibrated adaptive rtol `_cal`),
   `run_rigor_suite_3d.py` `gate` / `ablate`, `rigor_3d.py` (`feature_ablation_evaluation`).
3. Labels: `search_3d.py` (`find_dt_for_integrator_accuracy_monotone_3d`; `--label_mode accuracy_monotone`), `error_model_3d.py`
   + `errmodel` / `convergence` commands, `audit_3d.dense_sweep_accuracy_3d` (default 200 orbits, CI on non-monotone fraction).
4. Hard cases kept: crash/escape orbits stay in the CSV (`reference_outcome`); FTLE is truncated at a crash instead of NaN;
   `outcome_3d.py` + `crash` command.
5. Safety margin: finer, lower scale grid; `audit --ltt_repeats` (`calibration_stability_3d`); `--regressor hgb_quantile --quantile q`
   (`modeling_3d.fit_hurdle_models`); gate defaults `--n_cal 2000`.
6. Statistics: `generate_3d_data.py --base_seed`; `replicate` command; `lock` / `final` (`protocol_3d.py`); `PREREGISTRATION.md` (DRAFT, you must edit);
   gate default `--n_baseline 1000`.
7. Cross-mu: `generate_3d_data.py --tol_du` (same normalized tolerance); `mu` command uses the same test fold for within and transfer,
   adds a pooled model with mu as a feature, warns if normalized tolerances differ; per-system DU everywhere.
8. Housekeeping: per-run folders + `manifest.json`; Euler is not a candidate unless `--include_euler`; `--tol_km` defaults to the dataset's
   own labeling tolerance and errors on a mismatch; fidelity default time unit is the nominal 4.348 d (window value printed as a diagnostic;
   files carry the TU in their names).

## Extra bugs found and fixed
- End-of-run interpolation: fixed-step runs overshot the horizon and positions were linearly interpolated, adding an O(dt^2) error that
  depends on whether dt divides T. Test orbit (near L4, T=2): RK4 error 0.028 km at dt=0.1 but 2.49 km at dt=0.0999 (old); 0.0279 km (new).
  The last step now lands exactly on T. After the fix the fitted convergence slopes are Euler 0.995, leapfrog 1.996, RK4 4.07 (24 small test orbits).
  Non-monotone dense sweeps on 14 short, loose-tolerance test orbits: 2/14 (old) vs 0/14 (new) for both RK4 and leapfrog.
  Re-run `audit --dense 200` at your real settings; I have not.
- Censoring flag: `censored` compared a 6-decimal-rounded dt with the unrounded ceiling, so ceiling-limited labels were often unflagged.
  In your 20k file: RK4 18.3% flagged vs 31.7% at the ceiling (2,234 unflagged); leapfrog 4.1% vs 7.6% (577 unflagged).
- FTLE step: fixed dt=0.01 is too coarse near the bodies. The step now shrinks near them. On three test states FTLE changed by 0.1%, 25%, 0.1%
  (the 25% case sits 0.14 DU from the Moon); states that used to give NaN now give a finite, truncated value with `ftle_status != 0`.

## Not tested here (read this)
- numba is not installed in my sandbox, so the modified numba kernels (`_integrate_endpoint_*`, `_integrate_variational_rk4_numba`, `_max_jacobi_drift_*`)
  ran only as pure Python. Run `python3 smoke_test_3d.py` on your machine first; it compiles and exercises them.
- No network: `validate_on_capstone.py` (Horizons query, `--save_raw`) and the live `fidelity` query were not run; `cmd_fidelity` was run against a mock.
- Everything was smoke-tested on 40-120 orbit datasets with short horizons and loose tolerances. Statistics from those runs are meaningless;
  only "does it run and are the numbers plausible" was checked. Nothing was run at 20,000 orbits / 100 km / 20 TU; run time there is unknown
  (more orbits now crash near the Moon, FTLE steps are smaller, the monotone label evaluates more steps).
- Quantile regressors and the error-model recommender are new and unvalidated on real-scale data.

# 2026-10-23: H1' (no-FTLE variant, prediction cost charged, global fixed-step baseline)
- `orbit_ml_cr3bp/audit_3d.py`: `GLOBAL_DT_GRID`, `calibrate_global_dt_ltt` (one certified fixed step per integrator, no model).
- `orbit_ml_cr3bp/fair_baseline_3d.py`: `measure_prediction_cost`; `matched_accuracy_baseline_3d` gains `charge_prediction`, `global_dt`,
  `adaptive_spec_cache` (all default off; the FTLE integration is skipped when `ftle_3d` is not a model feature and replaced by
  `CHEAP_FEATURE_EVALS_EQUIV` = 1).
- `run_rigor_suite_3d.py`: `gate` and `final` now also run H1' (`report["h1_prime"]`, `*_baseline_h1_prime.csv`) unless `--no_h1_prime`.
  H1'a (vs RK45/DOP853) and H1'b (vs global fixed step) get separate verdicts; an unevaluable comparison gives NOT_FULLY_EVALUATED.
  The adaptive rtol calibration and per-orbit "_spec" adaptive solves are shared with H1. The original H1 output keys are unchanged.
- `PREREGISTRATION.md`: H1'a/H1'b and a dated Deviations entry.
- Not changed: items 3-8 of the readiness review (regressor default, lock discipline, scale-chain fragility, exchangeability,
  replicate split mismatch, subset verdict in the ORIGINAL H1).

# 2026-10-23: pinned regressor
- `run_rigor_suite_3d.py`: `PINNED_REGRESSOR = "hgb_quantile"`, `PINNED_QUANTILE = 0.1` are now the defaults of `--regressor/--quantile` for
  gate, final, audit, and (newly accepting the flags) ablate, replicate, mu. A different value prints a sensitivity-run warning.
  `final` records regressor and quantile in `final_eval_log.jsonl`. `crash`/`proxy`/`errmodel`/`convergence` fit no hurdle regressor and are unchanged.
- `orbit_ml_cr3bp/rigor_3d.py`: `model_kw` on `multiseed_evaluation`, `cross_mu_evaluation`, `replication_evaluation`,
  `feature_ablation_evaluation` (None keeps the old RF behaviour for library callers).
- Legacy scripts (`run_3d_analysis_from_data.py`, `generalization_tests.py`, helpers inside `modeling_3d.py`) still use RF.
- Metrics from the multi-seed commands (R^2 vs the oracle step, selection accuracy) are lower than the earlier RF numbers by design.

# 2026-10-23: pre-registration fill-in and freeze tooling
- `PREREGISTRATION.md`: tolerance (100 km), quantile (0.1), datasets (seeds 12/13/15 Earth-Moon, 14 Sun-Earth, SHA-256), locked set
  (seed 2024, fraction 0.25), "Which data evaluates what", exact "Run plan"; H1 wording corrected (prediction cost is charged only in H1').
  Three placeholders remain on purpose: `<<DATE>>`, `<<NAME>>`, `<<CODE_SHA256>>` (filled at the seal step). Not sealed.
- `freeze_snapshot.py` (new): `snapshot` / `seal` / `check`. No git here, so the code identity is a SHA-256 over every `*.py`
  (`CODE_SNAPSHOT.json`, archive in `snapshots/`); `seal` writes `FREEZE.json` (code hash, hash of the pre-registration text above
  "## Deviations", data CSV hashes).
- `orbit_ml_cr3bp/protocol_3d.py`: `code_fingerprint`, `compare_to_frozen`, `prereg_body_sha256`, `freeze_status`; every `manifest.json`
  now has a `code` entry.
- `run_rigor_suite_3d.py`: reports store `prereg_body_sha256`; after a seal `final` refuses to run if the code or the pre-registration
  body changed (override `--allow_post_freeze_changes`, recorded in `final_eval_log.jsonl`); other commands warn.
