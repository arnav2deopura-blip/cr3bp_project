# Pre-registration, Part C: a better step-size policy, and what limits it

Status: DRAFT until `freeze_c.py seal` writes `FREEZE_C.json`; after that, everything above "## Deviations" is frozen and later changes are dated entries
under "Deviations". Parts A (`PREREGISTRATION.md`, `FREEZE.json`) and B (`part_b/PREREGISTRATION_B.md`, `FREEZE_B.json`) are separate, sealed studies and are not modified or re-run on this tree
(Part A: commit `68a554b`; Part B: commit `68a43ff`). The plan this implements is `PART_C_IMPROVEMENT_PLAN.md` (kept outside the repository); `part_c/PART_C_LOG.md` is the dated record of every
decision and every exploratory result that led here. Where this document differs from the plan, section 8 says so.

Seal mechanics: `freeze_c.py seal` records (a) the SHA-256 fingerprint of every `*.py` file, (b) the SHA-256 of this document above "## Deviations", (c) the SHA-256 of every file in
`part_c/FROZEN_FILES.json` (the four frozen policies and the five datasets), and (d) the git commit. `run_confirm_c_3d.py` refuses to start a step if the code, this text or a frozen file differs from the seal,
and runs each step once per target (`confirm_log_c.jsonl`).

## 1. Questions

Part B measured where the cost of an ML step-size policy goes (mean decomposition against RK45-spec: paradigm +0.21, selection +0.03, prediction +0.38, safety scale +0.30, overhead +0.50 dex) and found it
cheaper than any adaptive solver on 0 of 14,161 orbits. Part C asks, for each term, whether a specific change removes it, and states in advance which changes are tested:
C1 the prediction term (free two-body periapsis features); C2 whether that is enough to beat an equal-risk adaptive solver with a physical-time step; C3 the paradigm term (a Sundman-time step); C4 whether an ML policy
in Sundman time beats an equal-risk adaptive solver; C5 whether the ML adds value there or one global step does the work.

## 2. Data (new seeds; nothing from Parts A / B is reused)

Generation (pipeline v5 = v4 + the Sundman-time label, the periapsis columns and optional sampling weights): `python generate_3d_data.py --num_orbits N --label_mode accuracy_monotone --pipeline v5 --tau_c 2.0 --base_seed S --n_workers 8 --output_dir data_c/<set> --tol_km 100 --chunk_size 8`
(enriched: `--sampling_weights 0.2933 0.1067 0.6`). Tolerance 100 km, Earth-Moon mu = 0.01215058535, horizons log-uniform 1-20 TU, reference DOP853 at 1e-13.

| Set | Seed | Orbits (usable) | Use |
|---|---|---|---|
| D_train | 41 | 100,000 (81,111) | fitting only, weight 1 |
| D_train_enriched | 141 | 20,000 | fitting only, weight p_natural / p_enriched per sampling branch (1.875 / 1.875 / 0.4167 for lagrange / wide / near_moon) |
| D_select | 42 | 20,000 (16,190) | every choice (section 3); split 50 / 50 (seed 42) into a calibration and an evaluation fold |
| D_confirm_C | 43 | 20,000 | the tests C1-C5 and every descriptive table; 4,000 randomly drawn usable rows (seed 42) calibrate, every other usable row is evaluated |
| D_lock_C | 44 | 5,000 | one run of the final policies with the scales and tolerances calibrated on D_confirm_C (as in Part B: a final number, not a test) |

Usable = a reference that reaches the horizon (finite FTLE; none failed). The 19 % reference-divergent (doomed) orbits are not in the tests; they have their own ledger (section 6).

## 3. The policies (frozen; `part_c/frozen/<name>/policy_frozen.joblib`, specs in `policy_frozen_spec.json`)

All four are fitted once on D_train + D_train_enriched (weighted) and never refitted. Features: the twelve state features of Part B plus 13 analytic two-body periapsis features (`osculating_3d.py`; r_p and time to the next
periapsis about each body from the osculating orbit, the rotating-frame velocity transformed to inertial; 3.5-3.9 RHS-equivalents). Histogram gradient boosting, quantile loss, seed 42.

- **`sund`** (C4, C5): the Sundman-time RK4 (`sundman_3d.py`, tau_c = 2, fictitious step ds, 4 evaluations per step, landing on T in physical time); one regressor on log10 ds*, **25 trees of depth 3, lr 0.2, q = 0.2**; overhead pinned at **15.04 RHS-equivalents**.
- **`phys`** (C1, C2): physical-time RK4 only (leapfrog dropped, `PART_C_LOG.md` section 2); one regressor on log10 d* - log10 tau_hat ("structured": tau_hat is the periapsis-based timescale feature), **10 trees of depth 3, lr 0.3, q = 0.1**; abstention to adaptive RK45 (the certified rtol)
  when the osculating orbit about the Moon returns within the horizon and log10 tau_hat < -1.0; no step-over guard; overhead pinned at **11.74**.
- **`global`** (C5 control): the Sundman-time RK4 with no model; one fictitious step for every orbit; overhead 0.
- **`partb_refit`** (C1 control): Part B's model class (state features, 300 trees, lr 0.05, q = 0.1, RK4 label) refitted on the same weighted D_train.

**Safety scale (end to end).** One common scale s applied to the step of whatever the policy runs (abstained orbits run the adaptive solve at the certified rtol). Certified on the 4,000 calibration rows of D_confirm_C by the exact fixed-sequence rule: the 28 values
0.1 * 2^(k/8), k = 0 ... 26, and 1.0, tested safest first at level delta = 0.1 against H0 "violation rate > alpha = 0.05" (one-sided exact binomial); the certified scale is the last rejected value before the first non-rejection. For `global` the grid is 0.01 * 2^(j/4), j = 0 ... 32.
A violation is a run that ends more than 100 km from the reference, collides, escapes, is non-finite or hits the step cap (4,000,000 steps), including any orbit that is abstained or falls back to the adaptive solver. The realized risk per selected integrator is reported with Clopper-Pearson bounds.

**Adaptive comparators** (scipy `solve_ivp`, atol = 1e-3 rtol, cost = `nfev`): RK45 and DOP853, each as `spec` (per-orbit oracle rtol), `cal` (one certified rtol per method on the same 4,000 rows, grid of 18 values, Bonferroni delta/18) and `eqr` (the loosest rtol
whose realized violation rate on the 4,000 rows is <= that of the policy being compared; recomputed per policy). **Costs** are right-hand-side evaluations; the model's cost is the pinned overhead above, charged once per orbit; the features are inside it.

**Selection (all on D_select; the protocol was written before any D_select experiment: `PART_C_LOG.md` section 4).** Model class and size S1 (linear, trees 10 x 3 ... 300), quantile S2, guard and abstention S3; criterion the median total cost per orbit that met the tolerance, ties within 2 % to the simpler model;
a candidate with an evaluation-fold violation rate above 6 % is excluded. Overhead gate 20 RHS-equivalents (**Deviation 1 of the log: the plan's 10 is not attainable, floor 10.7**). The results are in `part_c/selection/`.
Known limits of the selection: the fast certificate used there assumes monotone risk; the S3 abstention optimum sits at the edge of its grid (-1.0); guard and abstention were not combined; the Sundman label is not monotone in ds on 4.5 % of orbits (`PART_C_LOG.md` section 3).

## 4. Confirmatory tests (Holm-Bonferroni over C1-C5, familywise 0.05; every test one-sided in the direction stated)

All on the evaluation rows of D_confirm_C (every usable row that is not one of the 4,000 calibration rows), all five policies on the same rows.

| ID | Hypothesis | Test | Expectation (D_select evaluation, `part_c/results/dry`) |
|---|---|---|---|
| C1 prediction | the `phys` model's median abs(log10 prediction / label) of the RK4 step is lower than `partb_refit`'s | paired bootstrap over orbits (4,000), p = share of resamples with difference <= 0 | median abs error 0.285 (`phys`) vs 0.315 (`partb_refit`): difference 0.030 dex [0.020, 0.042], n = 12,189; supported |
| C2 physical-time policy | share of orbits where `phys` (overhead included) met the tolerance AND cost less than RK45-eqr is above 0.5 | exact one-sided binomial, H0 share <= 0.5 | share 26.1 % [25.4, 26.9] (n = 12,190; RK45-cal: 32.6 %); **not supported** (the policy's median total cost is 536 against about 300 for RK45-eqr) |
| C3 Sundman paradigm | (a) the oracle fixed-ds Sundman run (counted cost) is cheaper than RK45-spec on more than 50 % of orbits AND (b) Spearman(log10 delta_phys, log10(C_s* / C_RK45-spec)) is smaller than Spearman(log10 delta_phys, log10(C_fix* / C_RK45-spec)) | (a) exact binomial; (b) paired bootstrap (2,000); p = max | (a) 51.1 % [50.2, 52.0] (n = 12,189; the pilot gave 49.5 %: **a coin flip, narrowly expected supported**); (b) Spearman 0.226 vs 0.908; supported if (a) is |
| C4 Sundman ML policy | share of orbits where `sund` (overhead included) met the tolerance AND cost less than RK45-eqr is above 0.5 | exact one-sided binomial | share 69.9 % [69.1, 70.7] (n = 12,190; against RK45-cal 81.5 %, DOP853-eqr 50.4 %); strongly supported |
| C5 ML adds value | mean log10 total cost of `sund` is below that of `global`, orbits that met the tolerance under both, paired | one-sided paired bootstrap (4,000) | mean saving 0.306 dex [0.302, 0.310] (n = 11,363 orbits met under both; `sund` cheaper on 89 %; `global` ds certified at 0.08); strongly supported |

RK45-eqr is the equal-risk comparator of the policy in the test (its realized violation rate on the 4,000 calibration rows is the target). The share definition counts a violating orbit as a loss, as Part B's 'cheaper than adaptive' did.
If a test cannot be evaluated (a policy fails to certify any scale on the grid) it is reported as not supported with its p set to 1.

## 5. Reported, not tested

For `sund` and `phys`: the realized violation rate with Clopper-Pearson bounds, overall and per selected integrator; the full decomposition against all six comparators (`failure_study_3d.decompose`, and `decompose_sundman` with P_s, Q, K, O, R); the where-map by Hill regime x horizon band and by closest approach; the lever surface recomputed from the ledgers;
the D_lock_C run; the costs against RK45-spec, DOP853-spec, -cal and -eqr; the Sundman label's monotonicity audit; the endpoint-only error audit of Track 1.4 (rule: if more than 2 % of orbits in any merged cell have max checkpoint error above tol while the final error is within, both labels are reported for that cell; the label definition is not changed).

## 6. Doomed orbits (Track 1.3; descriptive, a separate ledger, never pooled silently)

The frozen `sund` and `phys` models run on the reference-divergent orbits of D_confirm_C exactly as on usable ones (no calibration beyond the scale already certified). Outcome agreement: the fixed-step run ends in the same event type (primary collision, secondary collision, escape) AND
|t_run - t_ref| <= tol / |v| (strict, the plan's rule) or the reference event time lies inside the step that detected it (relaxed). A disagreement is a violation of the doomed-risk ledger. The physical crash flag (osculating periapsis inside the body's collision radius and reached before T) is reported with recall and precision per outcome type.

## 7. Safety valves

Stopping rule: each confirmatory step runs once per target (`confirm_log_c.jsonl`); a rerun needs `--allow_rerun "<reason>"` and a dated entry below. If the Sundman pilot's go rule had failed, C3-C5 would have been dropped before the seal; it passed (`PART_C_LOG.md` section 3).

## 8. Deviations from the plan (all decided before D_confirm_C exists)

1. Overhead gate 20 RHS-equivalents instead of 10 (`PART_C_LOG.md` section 5).
2. Leapfrog dropped from the physical-time policy (arms F, G).
3. Hybrid {Sundman, RK4, leapfrog} not used (oracle bound 0.07 dex).
4. The Sundman label misses the plan's "< 2 % non-monotone" acceptance (4.5 % [2.1, 8.4]; label-window failures 2.0 %).
5. tau_c = 2 (best in the plan's grid {0.5, 1, 2}); C3(a)'s 50 % threshold is a coin flip at that choice (pilot 49.5 %).
6. Track 1.2's leapfrog gate and the by-integrator certificate are implemented and were exercised on D_explore (arms A-G) but are not part of any frozen policy.
7. The fast (bisection) certificate is used for selection only; the sealed scales use the exact ascending sequence.
8. Pipeline v5 omits the plan's checkpoint-error columns; Track 1.4 is run as a separate audit.
9. Not in the confirmatory family and not done in this round: E8 periodic families, E12 2D control, the guard + abstention combination, Track 4.1.2-4.1.4 (matched resampling, model-size control, dimensionless transfer), E11b measured.

## Deviations
- 2026-10-27, after every confirmatory step had run once (`confirm_log_c.jsonl`: apply_sund, apply_phys, apply_global, c1_resid, doomed_sund, doomed_phys, audit_sund, audit_phys on D_confirm_C; apply_sund, apply_phys, apply_global on D_lock_C; the `tests` step is a pure
  computation on those outputs and is logged only in `part_c/results/confirm/confirm_tests_confirm.json`): two EXPLORATORY additions written before any of their code exists. They add nothing to the Holm family and change nothing above "## Deviations".
  **Track 4.2, the free-lookahead arm of E3c**: `run_part_c_3d.py e3c_paired --osc_arm` adds, on D_explore with the same splits as E3c, an arm that uses the 13 periapsis features at no lookahead cost (charged 4 evaluations for the features), at the same scale grid and the same risk-matched comparison as the Track 0 errata.
  Expectation stated now: the arm's median error is about 0.14 dex (`PART_C_LOG.md` section 2, 300 trees), i.e. as good as lookahead at f = 0.5, and at equal realized risk it is cheaper than every f > 0 arm against RK45-cal. **Report writing**: `RESULTS_PART_C.md` and the figures of `make_figures_c.py`
  (sealed code, run after the tests). `freeze_c.py check` will report `run_part_c_3d.py` as changed after the first addition; no confirmatory step remains to be guarded.
