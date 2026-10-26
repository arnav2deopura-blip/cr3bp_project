# Part B log: when and why ML step-size prediction fails in the 3D CR3BP

Working log for the failure study defined in `ML_failure_study_plan.md` (the plan; kept outside the repo in Downloads at the
time of writing). The sealed Part A (`PREREGISTRATION.md`, `FREEZE.json`, `out/final_20261024T005235Z`) is not touched.
Code for Part A is commit `68a554b`; the changes below keep Part A reproducible (see "Backward compatibility").

Dates are the machine date of the run (2026-10-25).

## 1. Decisions asked for in Section 11 of the plan, with the evidence for each

| # | Question | Decision | Why (what I checked) |
|---|---|---|---|
| 1 | Reframe as a failure-mechanism study? | Yes. | Instruction to execute the plan. I first reproduced the plan's Section 2 numbers from the sealed outputs (13.8/17.2/56.3/46.5 % oracle-fixed cheaper; 2.43x ML over oracle fixed; 24.2 % raw over-prediction on uncensored rows; 37/37 violations over-predicted; 0 violations when the scaled step is at or below the label). All matched, so the framing rests on numbers I could regenerate. |
| 2 | Ceiling: physical or model censoring? | Physical: `dt_max = min(T, 2.0 TU)`, flat and mu-independent (`DT_CEILING_PHYSICAL_ABS`). | Part A: RK4 censored on 46.4 % of 16,247 usable seed-12 rows (matches the plan). Pilot of 400 old orbits and a fresh 397-orbit v4 pilot: 0 % censored for RK4 and leapfrog with either a 1.0 or a 2.0 ceiling. The ceiling only bounds the search from above, and 2 TU is above the RK4 absolute-stability limit set by the Coriolis frequency alone (about 1.4 TU), so a larger ceiling cannot change any non-censored label. Cost: none measurable (large steps are cheap to test). |
| 3 | Truth: 1e-13 everywhere, or 1e-11 + E1 flag? | 1e-13 everywhere (`REFERENCE_RTOL = REFERENCE_ATOL = 1e-13`). | The plan's own acceptance rule (1e-11 vs 1e-13 within 1 km on at least 95 %) is met: max 0.040 km, p99 0.010 km over 400 orbits. The plan's premise for keeping 1e-11 was that 1e-13 is slower; it is not (mean 0.0637 s vs 0.0645 s, because `max_step = 0.01`, not rtol, sets the cost). Equal cost plus "one truth everywhere" (Issue 3) settles it. E1 still compares against an independent method. Note scipy raises any rtol below 100*eps = 2.2e-14, so 1e-13 is not clipped. |
| 4 | Include E8 (families) and E12 (2D control)? | Defer both to the end; do them only if the core (Phases 0-6) is finished. | They are marked optional in the plan, E8 needs new continuation code that must be verified against published initial conditions (plan Section 10), and none of HB1-HB6 depends on them. |
| 5 | epsilon for E3a | The plan's three levels (1e-9, 1e-7, 1e-5 DU) with physical equivalents stated: 0.38 m, 38 m, 3.8 km in position (1 DU = 384,400 km); the same numbers in DU/TU for velocity (1 DU/TU = 1.023 km/s, so 1e-9 is 1.0 um/s, 1e-7 is 0.10 mm/s, 1e-5 is 1.0 cm/s). | A navigation-level uncertainty of about 1 km and 1 cm/s (2.6e-6 DU, 9.8e-6 DU/TU) sits between the middle and the largest level, so the three levels bracket it. Conversion computed from `DU_KM = 384400` and the nominal TU = 4.348 d (375,667 s). |
| 6 | Is the old locked set "spent"? | Yes. | It was evaluated once by Part A and read post hoc for the plan's Section 2. It is used only for exploration. Every confirmatory claim in Part B is made on new seeds. |

Extra decisions I made while executing (not in Section 11):

- **Primary deployable model for the decomposition = state-only (no FTLE).** Same pinned learner (HGB quantile q = 0.1 on log10 dt, RF
  classifiers). Reason: the FTLE feature costs a full variational integration and leaks the crash/escape outcome (Issue 11). The FTLE model
  is kept as a comparison arm; FTLE x T is still used as a physics diagnostic (lambda T) but never as a deployable input.
- **Primary adaptive comparators:** spec (oracle rtol per orbit), cal (certified single rtol), and equal-risk. All three are reported (plan 5.1).
- **Part A defaults are kept.** New behaviour is opt-in (`--pipeline v4`, `ceiling_mode="physical"`, `count_ml_divergence=True`) except two
  deliberate global changes: the single 1e-13 truth, and leapfrog cost = 1 evaluation per step.

## 2. Phase 0 changes (Issues 1, 2, 3, 6, 7, 8, 15 of the plan)

| Issue | Verified against the code/data first? | Change |
|---|---|---|
| 1 ceiling | Yes: 46.4 % RK4 censored, median ceiling 0.186 TU, `censored = (best_k == 0)`. | `search_3d._dt_upper_bound_3d(..., mode)` / `label_dt_max`; `physical` mode; `data_3d.label_orbit_accuracy_3d(ceiling_mode=)`. |
| 2 dense sweep | Yes: 7/200 rows had `dense_dt = 1e-4` with larger labels (rows 3173, 179, 2322, 1068, 3200, 51, 325). | `audit_3d.dense_sweep_log_3d`, `summarize_dense_sweep_log` (geometric grid 2^(1/16), all candidate integrators, by decade of the label). The linear sweep is kept for Part A. |
| 3 two truths | Yes: labels/audit 1e-11, cost comparison 1e-13. | `config.REFERENCE_RTOL/ATOL = 1e-13`; `fair_baseline_3d.TRUTH_* = REFERENCE_*`. Per-orbit `ref_check_km` (second DOP853, rtol 1e-12, half max_step) saved at generation; E1 adds an independent method. |
| 6 ML divergence skipped | Yes: `if not ok: n_skipped += 1`. | `matched_accuracy_baseline_3d(count_ml_divergence=True)`; Part B's own cost code counts divergence as a violation from the start. |
| 7 unequal calibration | Yes: `max_rows = 300`, 21-value grid vs ML `n_cal = 2000`, 18 values. | `calibrate_adaptive_rtol(n_jobs=)` (parallel, so 2,000 rows is affordable), `RTOL_GRID_B` (18 values), `equal_risk_rtol`. |
| 8 leapfrog cost | Yes: the kernel recomputes the acceleration at the start of every step although it equals the end of the previous one. | `_step_leapfrog_fsal_3d_numba` used by the endpoint kernel; `EVALS_PER_STEP_3D["leapfrog"] = 1` (`EVALS_PER_STEP_3D_V3` keeps 2). Bit-identical to the old kernel (test). One extra evaluation at the very start is ignored. |
| 15 success flag | Yes: `dt > 0.0001`. | v4 uses `reason == REASON_OK`. |

New modules/tests: `orbit_ml_cr3bp/diagnostics_3d.py` (plan 5.2 columns, Hill regimes), `tests/test_phase0.py`.

### Backward compatibility (Part A reproducibility)
`label_orbit_accuracy_3d(ceiling_mode="legacy")` reproduced 24/24 sealed RK4 and leapfrog labels exactly even with the new 1e-13
reference (`tests/test_phase0.py::test_legacy_labels_reproduce`). `freeze_snapshot.py check` will now report the code as changed; that is
expected, and Part A's `final` must not be re-run on this tree (use commit `68a554b` or `snapshots/code_79a6455ddef8.tar.gz`).

## 3. Phase-0 acceptance checks (all on data never used before)

| Check (plan 6.1) | Result | Pass |
|---|---|---|
| RK4 censored fraction < 2 % on a 500-orbit pilot | 0.0 % RK4, 0.0 % leapfrog (397 usable of 500, seed 9001) | yes |
| Log dense sweep on 200 pilot orbits, all candidate integrators, non-monotone by dt band | 2 of 400 (0.5 %, 95 % CI 0.06-1.8 %) non-monotone, both RK4 in the 0.1-1 TU band; 0 of 292 in the bands below 0.1 | yes (reported) |
| Reference agreement within 1 km on at least 95 % | 100 %; max 0.003 km (v4 pilot, 1e-13 vs 1e-12 half step); max 0.040 km (1e-11 vs 1e-13 on 400 old orbits) | yes |
| Numba kernels compile and run on this machine | `smoke_test_3d.py`: ALL PASSED; `tests/test_phase0.py`: 9/9 | yes |

Timing: 500 orbits in 13.4 s on 8 workers, so 20,000 orbits is about 9 minutes. The proposed dataset sizes are affordable.

Effect of the fixes on labels (old orbits, 400 sampled): median RK4 label 0.034 TU (Part A, 41 % censored) vs 0.124 TU (physical ceiling).
So the median oracle fixed step is about 3.7x larger (fixed-step cost scales as 1/dt), and the paradigm-ceiling numbers of the plan's Section 2.1 will move,
as the plan warned (Section 10). They must not be quoted; the Part B values come from D_explore/D_confirm.

## 4. Phases 1-2 on D_explore (exploratory; seed 21, 20,000 orbits, 16,170 usable; generation 12 min on 8 workers)

Everything here was computed on the set used for development and is NOT confirmatory. Raw outputs: `part_b/results/explore/`.

Reproduction check first: the plan's Section 2 numbers (computed on Part A's censored labels) reproduce exactly from the sealed outputs
(oracle fixed cheaper on 13.8 / 17.2 / 56.3 / 46.5 %; ML integration 2.43x the oracle step; all 37 violations over-predictions). On v4 labels
they move, as the plan warned: oracle fixed cheaper than RK45-spec on 25.5 % (DOP853: 25.8 %), ML integration 3.8x the oracle step.

| Experiment | Finding |
|---|---|
| Policy (state-only, certified scales 0.5/0.5, n_cal 2,000) | violation rate 3.97 % on 3,000 held-out orbits (cal rows 4.05 %); cheaper than adaptive on 0/3000 orbits against all six comparators once the model's cost (1,288 evaluations here, 1,128 pinned later) is charged |
| Decomposition vs RK45-spec (means, dex) | P +0.21, S +0.02, Q +0.38, K +0.30, O +0.54, R -0.01, total +1.44; paradigm is the dominant ceiling on 20 % of orbits, overhead on 40 %, prediction on 32 % |
| Paradigm mechanism | rho(log10 delta_phys, P) = 0.91 [0.90, 0.92], OLS slope 0.90 [0.88, 0.91]; delta_emp (empirical step ratio) is a much weaker predictor (rho 0.28-0.36) |
| Where | closest approach is the organizer: min r2 < 0.05 DU holds 90 of 119 violations (15.6 % rate vs 0.4 % beyond 0.8 DU) and the largest paradigm gap (+0.65 dex vs about 0); regime I has the largest gap, regimes IV-V are near parity |
| Violation anatomy | 17 of 119 violations pick leapfrog where no step >= 1e-4 meets the tolerance (classifier error, own taxonomy class); on the rest the scaled step exceeds the label (median 2.0x, tail 110x); violations with a scaled step at or below the label: 0 of 2,854 |
| E1 reference validity | Radau 1e-12 vs the DOP853 label reference: max 0.056 km over 599 orbits, 0 flagged (threshold 10 km); 0 outcome disagreements, event times equal to 1e-15 on 300 doomed orbits |
| E3a label sensitivity | spread of the label under perturbations up to 1e-5 DU / DU/TU (3.8 km, 1 cm/s) is at most 0.023 dex (label resolution 0.0108 dex; 8 % of orbits show any change) against a 0.335 dex median model residual: the information floor is not what limits the model. Sign as hypothesised: rho(lambda T, sigma) = 0.43 at 1e-5 |
| E3b oracle trajectory features | median |log10 raw/label| 0.294 -> 0.057 dex, certified scale 0.5 -> 0.8, violations 3.97 -> 0.83 %, ML integration / oracle step 3.8x -> 1.4x. The information is in the trajectory (closest approach, timescales), not in model capacity; those features cost a full integration |
| E4 learning curves | q = 0.1 pinball loss still falls with training size in 23 of 30 merged cells (4 flat, 3 rising; slope -0.024 per decade overall); the near-pass cells of regimes III and V are the most data-starved (-0.10 to -0.13); over-prediction rate 20.8 % (1k rows) -> 12.8 % (11k) against a nominal 10 %; by regime 10.8-17.7 % |
| E5 calibration alternatives (certified at alpha 0.05) | Mondrian by regime and dispersion bins lower the realized risk (2.4 %) but cost MORE (mean +37 % / +82 %), opposite to the plan; abstaining on the 20 % most dispersed orbits to adaptive cuts the mean cost 27 % at 3.2 % risk. Label-conformal scaling is not a valid substitute (6.3 % realized risk) |
| E9 step-over | violation rate rises with dt / tau_min: 2 % below 0.02, 40 % at 0.2-0.5, 86 % at 0.5-1, 100 % above 1 (AUC 0.84); a guard at 0.3 catches 58 % of violations at 8 % false alarms |
| E10 | a one-timescale physics rule (certified, same rule) is 11.8x costlier in integration than the ML policy, so "ML adds nothing beyond a Kepler timescale" is FALSE here. Break-even: ML integration alone beats RK45-cal on 28.7 % of orbits but with a headroom of only a median 48 evaluations; the pinned scikit-learn overhead is 1,128 |
| Horizon dial pilot | see PREREGISTRATION_B.md; both gaps rise with T (per-state rho 0.91 / 0.75) |

Not done in this round (see the pre-registration's deviations): E3c lookahead (HB3), the mu dial (E7), E8 families, E12 2D control.
Arithmetic that bears on HB3 without being a test of it: any lookahead adds cost to a policy whose integration cost alone already exceeds the adaptive cost on 71-99.7 % of orbits.

Process notes: (i) a background job driven from a stdin script left a stray process that respawned workers and slowed other jobs for about
12 minutes; found via `ps`, stopped; no result was affected (all affected runs were reruns of deterministic steps). (ii) The first policy run
used the half-decade rtol grid, which made the equal-risk rtol equal to the certified one; refined and rerun (deviation 6). (iii) One
combined shell command with `rm` and `sh -c` was blocked by a safety check and was not re-issued; the stale scratch folder it targeted was left alone.

## 5. Exploratory follow-ups (2026-10-26): E3c lookahead and the mu dial
Pre-registered first (git 5d24baf), run, results in `part_b/RESULTS_E3C_MU_DIAL.md`. E3c: median error 0.32 -> 0.07 dex as the fraction of the horizon integrated first goes 0 -> 1; total cost
1,539 -> 635 evaluations against 134 (RK45-spec) and 380 (RK45-cal); 8 of 9 groups need f = 1 to match the full-trajectory error. Mu dial (seeds 31-35, 8,000 orbits each): paradigm gap
+0.06 / +0.10 / +0.11 / +0.17 / +0.21 dex and integration-only ML gap +0.50 / +0.55 / +0.70 / +1.20 / +1.35 dex for mu = 1e-6 / 1e-4 / 1e-3 / 0.01215 / 0.05; violations 2.7-3.7 % at every mu.
Caveat to keep: Hill-scaled sampling leaves most small-mu orbits primary-dominated (median closest approach 54 r_H at 1e-6, 1.5 r_H at 0.05).

