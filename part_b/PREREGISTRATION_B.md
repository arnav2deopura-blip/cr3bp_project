# Pre-registration, Part B: when and why ML step-size prediction fails in the 3D CR3BP

Status: DRAFT until `freeze_b.py seal` writes `FREEZE_B.json`; after that, everything above "## Deviations" is frozen and later changes
are dated entries under "Deviations". Part A (`PREREGISTRATION.md`, `FREEZE.json`) is a separate, sealed study and is not modified.
The plan this implements is `ML_failure_study_plan.md`; where this document differs from it, "Deviations from the plan" says so.

Seal mechanics: `freeze_b.py seal` records (a) the SHA-256 fingerprint of every `*.py` file (the document cannot contain a hash of the
code it describes), (b) the SHA-256 of this document above "## Deviations", (c) the SHA-256 of the frozen model file and of the
D_explore CSV, and (d) the git commit it was sealed at. The confirmatory runner (`run_confirm_b_3d.py`) refuses to start a step if the
code fingerprint differs from the seal, and runs each step once per target (`confirm_log_b.jsonl`).

## 1. Question

For which CR3BP trajectories, and for what physical reasons, can a step size chosen from the initial state be both safe and cheaper
than adaptive integration? Where it cannot, which ceiling binds: paradigm (even the best single step loses), information (the initial
state does not carry what the step depends on), calibration (the safety margin), or overhead (features and model cost)?

## 2. Data (all independent seeds; the Part A sets and the Phase-0 pilot rows are never reused)

Generation: `python generate_3d_data.py --num_orbits N --label_mode accuracy_monotone --pipeline v4 --base_seed S --n_workers 8
--output_dir data_b --tol_km 100 --chunk_size 8`. Version 4 labels: step ceiling min(T, 2 TU), single 1e-13 DOP853 truth, per-orbit
physics diagnostics, success by the search's reason code. Tolerance 100 km (as Part A). Earth-Moon mu = 0.01215058535.

| Set | Seed | Size | Use |
|---|---|---|---|
| D_explore | 21 | 20,000 (16,170 usable) | all development; model fitted on 11,170 of its usable rows (`train_row_ids.csv`) |
| D_confirm | 22 | 20,000 | the confirmatory tests T1-T5 and every descriptive table below; touched only by `run_confirm_b_3d.py` after the seal |
| D_lock | 23 | 5,000 | one evaluation of the gated policy (E11) |
| Horizon sweep | 1 | 2,600 initial states drawn from D_confirm's usable rows (the frozen model never trained on any of them), relabelled at T = 1, 2, 5, 10, 20 TU; kept if the reference reaches 20 TU | T5 |

Usable = finite FTLE and a reference that reaches the horizon. Reference-divergent (crash/escape) orbits are 19 % of the draw
(D_explore) and are excluded from the step-size study (survivorship: they are concentrated in Hill regime I, see the plan, 2.5).

## 3. The policy under test (frozen before D_confirm exists)

- **Model**: hurdle models with random-forest classifiers (150 trees, seed 42) and a HistGradientBoosting QUANTILE regressor,
  q = 0.1 on log10(dt), per integrator; **state-only features** x, y, z, vx, vy, vz, T, Jacobi, r1, r2, r1/r2, C - C_L1.
  Fitted once on D_explore and never refitted: `out_b/policy_20261025T213540Z/policy.joblib`
  (SHA-256 06179fe8c7726704d8aeaeb56690c01c5c72af1e0ae3e642c87f1df692317a54).
- **Candidates**: RK4 (4 evaluations/step) and leapfrog (1 evaluation/step, first-same-as-last). Choice by the largest scaled step per
  evaluation (`cost_normalized_choice`).
- **Safety scales**: re-certified on 2,000 randomly drawn usable D_confirm rows (seed 42), per integrator, with the Part A rule: grid
  `LTT_SCALE_GRID` (18 values), one-sided binomial test of "violation rate > 0.05", Bonferroni at delta/m = 0.1/18, the LARGEST
  certified scale. A violation is an error above 100 km, a collision, an escape, a non-finite result, or a run that hits the step cap.
- **Adaptive comparators** (scipy `solve_ivp`, atol = 1e-3 rtol; cost = `nfev`): RK45 and DOP853, each in three forms
  - `spec`: per-orbit oracle rtol, the loosest (to 0.02 decade) with final error <= 100 km, found by bracket and bisection;
  - `cal`: one rtol per method, certified on the same 2,000 rows with the same rule over an 18-value grid 1e-12..1e-3;
  - `eqr`: the loosest rtol whose realized violation rate on those rows is <= the ML policy's realized rate on them (4 bisection steps).
- **Costs** are right-hand-side evaluations. Features: 1 (one acceleration-sized computation). Model inference: **1,128 RHS-equivalents
  per orbit, pinned** (median of five isolated measurements, 1,101-1,142; single wall-time measurements inside a run ranged 1,100-1,430
  and Part A reported 591), so the ledger does not inherit wall-time noise. The overhead is a property of the scikit-learn model, not of
  "ML": the break-even analysis (E10) gives the cost a model would need.
- **Evaluation rows**: every usable D_confirm row that is not one of the 2,000 calibration rows (about 14,000).

## 4. Quantities (definitions fixed; code: `failure_study_3d.py`)

- **Oracle fixed step** C_fix* = min over RK4/leapfrog (with a valid label) of e_k ceil(T / d*_k), d* the v4 label.
- **Cost-gap identity** (exact in log10, residual R = ceil rounding): log10(C_ML,tot / C_ad) = P + S + Q + K + O + R with
  P = log10(C_fix*/C_ad) paradigm, S selection, Q = log10(d*_k/d_hat_k) prediction, K = log10(1/s_k) safety scale,
  O = log10(1 + (C_feat + C_pred)/C_ML,int) overhead. Medians of the terms do not add to the median total; the waterfall uses MEANS.
- **Taxonomy** (mutually exclusive, in order): F0-ref (second reference differs by > 10 km), F0-label (chosen label censored), abstain,
  violation:{infeasible-choice | information | epistemic | calibration | unattributed}, cost-loss:{P,S,Q,K,O}, ml-win.
- **Cells**: Hill regime I-V (initial Jacobi constant against C_L1..C_L5) x min r2 band (<0.05, 0.05-0.17, 0.17-0.8, >0.8 DU, from the
  reference) x horizon band (1-3, 3-8, 8-20 TU). Merge rule fixed in advance: a base cell with fewer than 50 orbits joins the other small
  cells of its (regime, r2 band) along the horizon axis; if that union is still < 50 it joins the small remainder of its regime along the
  r2 axis; anything left is "pooled-small" (`assign_cells`, tested).
- **Physical timescale range** delta_phys = T max(1/tau) / integral(1/tau) dt, tau = min(sqrt(r1^3/(1-mu)), sqrt(r2^3/mu)) along the
  reference. This, not the empirical step-size ratio delta_emp (rho = 0.28-0.36 in D_explore), is the pre-registered predictor.
- **lambda T** = FTLE x T (FTLE truncated at a crash; diagnostic only, never a deployable input).

## 5. Confirmatory tests (Holm-Bonferroni across T1-T5, familywise 0.05; each is an intersection-union test)

All p-values are in the direction stated; permutation and bootstrap counts are in `confirm_b_3d.py`.

- **T1 paradigm (HB1).** (a) For RK45-spec and for DOP853-spec: H0 "share of orbits where C_fix* < C_ad is >= 0.5", exact one-sided
  binomial. (b) Spearman rho between log10 delta_phys and the paradigm gap log10(C_fix*/C_RK45-spec), H0 rho <= 0 (permutation).
  p(T1) = the largest of the three. Estimates reported: shares with Clopper-Pearson intervals, rho, OLS slope with bootstrap CI.
- **T2 information (HB2).** On the base evaluation orbits (10 per merged cell, about 300), 16 clones at each of eps = 1e-9, 1e-7, 1e-5 (position
  part of norm eps DU, velocity part of norm eps DU/TU, uniform directions; 0.38 m / 38 m / 3.8 km and 1.0 um/s / 0.10 mm/s / 1.0 cm/s):
  sigma = standard deviation of log10 d* of the chosen integrator over clones that still reach the horizon. **Primary eps = 1e-5.**
  H0: Spearman(lambda T, sigma) <= 0 (permutation). Reported with it, NOT part of the p-value: median |residual|/sigma and the share of
  orbits whose residual is within sigma, overall and in the top lambda-T tercile. The label is quantized (0.0108 dex steps), so most
  sigma are exactly 0; this limits resolution, not the validity of the comparison with a 0.3 dex residual.
- **T3 calibration shape (HB4).** (a) Chi-square homogeneity of the violation indicator across the merged cells. (b) Paired per-orbit
  comparison of log10 cost of ML with Mondrian-by-Hill-regime scales (groups merged to >= 300 calibration rows per integrator,
  Bonferroni over grid x groups) against the global scale, over orbits that met the tolerance under both; H0 "Mondrian mean log cost
  >= global" (one-sided paired bootstrap). p(T3) = max of (a) and (b).
- **T4 where (HB5).** Three one-sided contrasts, Holm within the three, p(T4) = the largest adjusted p: paradigm gap vs RK45-spec is
  larger for (C1) min r2 < 0.17 than >= 0.17 DU and (C2) Hill regimes II+III than IV+V (Mann-Whitney); (C3) violation rate is higher for
  min r2 < 0.05 than >= 0.05 DU (pooled two-proportion z).
- **T5 horizon dial (HB6, horizon only).** Same initial states at five horizons, the frozen policy with its frozen calibration. Both
  the paradigm gap and the integration-only ML gap log10(C_ML,int / C_RK45-spec) must RISE with T. Test: per initial state, Spearman rho
  between T and the metric; one-sided bootstrap over initial states for H0 mean rho <= 0; p(T5) = max over the two metrics. The overhead
  is excluded from the ML gap because a fixed model cost is a larger share of a shorter, cheaper solve, which would oppose the
  hypothesis for a reason unrelated to the dynamics. The plan's Jonckheere-Terpstra test assumes independent groups; the horizons share
  their initial states, so the paired trend is used instead.

Decision rule: a test is "supported" if its Holm-adjusted p <= 0.05. A test that is not supported is reported as not supported.

### What D_explore already showed (exploratory; these shaped the tests and are stated so nobody mistakes them for surprises)
Paradigm: oracle fixed step cheaper than RK45-spec on 25.5 % (DOP853: 25.8 %), median paradigm gap +0.15 dex, rho(log delta_phys, gap) = 0.91,
slope 0.90 (T1 expected to be supported). Information: sigma <= 0.023 dex at eps = 1e-5 against a 0.335 dex median residual; rho(lambda T,
sigma) = 0.43 at 1e-5 (T2 expected to be supported in sign, with the information floor far below the model error). Calibration: the
global certificate holds marginally (4.0 %) but conditionally ranges from 0.4 % to 15.6 % across min-r2 bands; Mondrian-by-regime scales were
COSTLIER (mean +37 %) at their certified risk, opposite to the plan's expectation (T3 expected NOT to be supported through part (b)).
Where: violations concentrate at min r2 < 0.05 DU (90 of 119), the paradigm gap is largest there (+0.65 dex) and ~0 beyond 0.8 DU
(T4 expected to be supported). Horizon: a pilot on 2,354 D_explore initial states (rows the frozen model did not train on) relabelled at T = 1, 2, 5, 10, 20 gave a median paradigm gap of -0.22, -0.03, +0.15, +0.22, +0.32 dex and a median integration-only ML gap of +0.30, +0.49, +0.71, +1.20, +1.24 dex, with mean per-state Spearman rho 0.91 and 0.75, and violation rates 1.2, 1.6, 4.2, 5.4, 9.6 % (T5 expected to be supported). The total ML gap, overhead included, FALLS with T (1.71 to 1.32 dex mean) because a fixed 1,128-evaluation model cost dominates short solves; that is why T5 uses the integration-only gap.
A dry run of the whole confirmatory pipeline on D_explore (small sizes, `--target dry`) was used to debug the code before the seal; its p-values are not results.

## 6. Descriptive outputs on D_confirm (estimates with intervals, no hypothesis test)

Decomposition (means and medians in dex) against all six comparators; taxonomy shares per Hill regime and per cell; risk ledger (rate,
CVaR90 of error/tol, catastrophic share > 10 tol); where-maps (regime x min r2); E1 reference validity on 600 usable + 300 doomed orbits
(Radau at 1e-12 as the independent method; F0-ref threshold 10 km); E5 calibration alternatives (global, Mondrian by regime, dispersion
bins, label-conformal, abstention to adaptive); E9 step-over (dt/tau_min) and a runtime guard; E10 physics-rule baseline (one-parameter
timescale rule, certified with the same rule) and break-even inference cost; E11 deployable gate on D_lock (cells = regime x horizon band,
chosen on D_explore; ML allowed only where its win share >= 0.5 and its violation rate <= 0.05 under an assumed overhead; elsewhere adaptive
at the equal-risk rtol); E11b ML as a predictor of adaptive cost for lockstep batching, trained on D_confirm and tested on D_lock.
Pre-registered expectation for E11: the gated policy does not beat adaptive-spec overall.

## 7. Deviations from the plan (decided before D_confirm exists)

1. Primary deployable model is state-only (no FTLE); the FTLE variable is a diagnostic and E3b feeds trajectory features as an upper bound.
2. HB3 (lookahead) is not confirmatory: it needs a partial-horizon feature pipeline that is exploratory only (E3c, reported from D_explore if run).
3. HB6 is tested on the horizon dial only. The mu dial (E7) needs mu-scaled sampling and dimensionless features and is not run in this round.
4. Family size is 5 (T1-T5), not 6, because of (2); JT replaced by the paired trend (T5).
5. Learning curves (E4) reach 11,170 training rows, not 20,000, because D_explore has 16,170 usable orbits.
6. Equal-risk adaptive rtol uses a 4-step log bisection beyond the 18-value grid, since a half-decade grid made it coincide with `cal`.
7. Model-inference cost is pinned (1,128 RHS-equivalents), not re-measured inside each run.
8. The gate in E11 uses deployable cells (regime x horizon band); min r2 is a trajectory quantity unknown before integrating.
9. E8 (periodic families) and E12 (2D control) are deferred.
10. Reference: 1e-13 everywhere, decided on a measurement that contradicted the plan's premise (no wall-time cost), see `part_b/PART_B_LOG.md`.

## 8. Stopping rule

Each confirmatory step runs once per target (`run_confirm_b_3d.py`; log `confirm_log_b.jsonl`). A rerun needs `--allow_rerun "<reason>"` and a
dated entry below.

## Deviations
(none yet)
