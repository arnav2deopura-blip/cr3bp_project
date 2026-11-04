# Post hoc analyses of the sealed Studies A–D (rigor plan of 2026-10-30)

Everything in this folder is **post hoc**. Studies A–D are sealed and their preregistered verdicts stand unchanged; nothing here reruns a
confirmatory test or replaces a sealed number. Every new number is shown next to the sealed number it qualifies and is labelled
"post hoc" wherever it is quoted (manuscript, results files, figures).

This file was written and committed **before** any of the analyses below was computed, so the definitions, the alternative sampling
mixtures and the comparator grids cannot have been chosen after seeing their results. Changes after that commit are dated in the
"Deviations" section at the end.

Code: `orbit_ml_cr3bp/posthoc_metrics.py` (pure functions, tested in `tests/test_posthoc.py`) and `run_posthoc.py` (one command per
analysis, each writing a run folder with a `manifest.json` under `posthoc/results/`).

## P1. Aggregate cost next to every share (plan F1)

For each sealed policy ledger and each adaptive comparator column `<method>_<tolerance>_nfev`:

| Statistic | Definition |
|---|---|
| share cheaper | the sealed definition (`confirm_c_3d.win_share`): orbits where the policy met the tolerance AND its total cost, overhead included, is strictly below the comparator's, over orbits with a comparator cost |
| aggregate ratio | Σ comparator cost / Σ policy total cost over the same orbits, violating policy runs included. Above 1 favours the policy. |
| mean log ratio | mean of log10(comparator / policy total cost) over the same orbits (dex; above 0 favours the policy) |
| medians | median policy total cost over **all** orbits and over orbits that met the tolerance; median comparator cost |
| denominators | number of evaluation orbits, number without a comparator cost (failed or capped comparator runs), number capped |

Intervals: 95 % percentile bootstrap over orbits (4,000 resamples, seed 0); for the share also the Clopper–Pearson interval.

Breakdowns (each a separate table): horizon band (1–3, 3–8, 8–20 TU, the `h_band` column), Hill regime (`regime`), sampling family
(`orbit_label`: L1–L5, wide, near_moon).

Ledgers: Study C confirmation (Sundman policy), Study D replication (D3), Study C lock, Study D lock, Study D confirmation (D1 ML arm
and D1 best rule), D4 stratified and one-scale policies on the replication set.

## P2. Sampling mixture (plan F5)

Declared alternative mixtures, applied by reweighting the orbits of each ledger by family (weight = target share of the family / its
observed share among the evaluation orbits). Bootstrap resamples within family so the family counts stay fixed.

| Mixture | Family weights |
|---|---|
| `as_sampled` | the ledger as it is (the 0.55 / 0.20 / 0.25 generation mixture after excluding reference-divergent orbits) |
| `uniform_families` | 1/7 for each of L1, L2, L3, L4, L5, wide, near_moon |
| `cislunar` | 1/3 each for near_moon, L1, L2; 0 for the others |

Reported for every ledger and comparator of P1: weighted share, weighted aggregate ratio, weighted mean log ratio.

## P3. Overhead charged per column of Table 3 (plan F3 item 5)

The Table 3 columns were charged 15.04 (Study C sets, D replication) or 16.09 (Study D lock) RHS-equivalents. P1 is recomputed with
the policy total cost `C_ml_int + h` for h = 15.04 and h = 16.09 on every column, so the effect of the difference is visible.

## P4. "Three quarters" on one dataset (plan F3 item 4)

The sealed sentence divides a rule gap measured on the Study D confirmation set (0.0715 dex) by a global-step gap measured on the
replication set (0.274 dex). P4 runs the frozen Study C global fictitious step on the Study D confirmation set with the D1 protocol
(same 4,000 calibration rows, fixed-sequence certificate, overhead 0) and computes its gap to the D1 ML ledger with the D1 statistic
(`confirm_d_3d.d1_ml_vs_rule`: mean paired log10 cost difference over orbits that met the tolerance under both). The share recovered is
1 − gap(rule) / gap(global), both on the confirmation set; the bootstrap interval resamples orbits jointly for the two gaps.

## P5. Fig 4 empty cell (plan F6)

Histogram of the closest lunar approach `min_r2` of usable Hill-regime-I orbits of the Study B confirmation set, and the count of
regime-I orbits by reference outcome and closest-approach band (including reference-divergent ones), to check whether the empty cell
regime I × 0.05–0.17 DU is physical.

## P6. Fit-seed variability (plan Phase 1, item 5)

The frozen Study C Sundman configuration (`part_c/frozen/sund`: 25 trees, depth 3, learning rate 0.2, quantile 0.2, osculating
features, same training rows and weights) refitted with model seeds 1–5 in addition to the frozen fit. Each fit (and the frozen one) is
certified on the standard 4,000 calibration rows of the Study D confirmation set and evaluated on its 12,239 evaluation rows with the
D1 protocol (overhead 16.09, comparators recalibrated per fit). Reported: scale, violation rate, share and aggregate ratio against
equal-risk RK45 and DOP853, and their spread over the six fits.

## P7. Symmetric comparator certification (plan F2)

The sealed certified tolerance of the adaptive solvers used 18 values from 1e-12 to 1e-3 (0.53 dex apart) at δ/18 (Bonferroni), while
the policy scale used 28 values 0.038 dex apart by fixed sequence at the full δ. P7 re-certifies RK45 and DOP853 with the policy's rule:

- grid: rtol = 10^(−10 + k/8), k = 0 … 56 (1e-10 to 1e-3, 0.125 dex apart; at an RK45 cost exponent of about 1/5 this is a finer cost
  resolution than the policy grid's 0.038 dex);
- rule: fixed sequence, safest (tightest) first, full δ = 0.1, α = 0.05, the exact binomial p-value of Eq (cert); the certified rtol is the
  last value of the uninterrupted run of rejections;
- same calibration rows, same violation definition (final error > 100 km, failure or the 2,000,000-evaluation cap).

Applied on the Study C confirmation set and the Study D replication set (SciPy RK45 and DOP853), recomputing the "certified" rows of
Table 3; and to D4b, with the same grid per stratum at δ/G. The compiled path (D2 comparators) is re-certified the same way on the Study
D confirmation set. Old (sealed) and new numbers are reported side by side.

## P8. Comparator absolute-tolerance and δ sensitivity (plan F2 actions 3–4)

Equal-risk RK45 recomputed exactly as sealed (`policy_b_3d.certify_rtol_lazy`: scan of the 18-value grid, then four log bisection steps) with
atol = r · rtol for r ∈ {1e-1, 1e-3 (sealed), 1e-6}, and with a position-only absolute tolerance: atol = 0.26 · rtol on the three position
components (2.6e-4 DU = 100 km at the loosest grid value rtol = 1e-3) and 1e3 · rtol on the velocity components. The setting with the
lowest mean RK45 cost on the **calibration** rows at equal risk is the "best atol" comparator; all four are reported on the evaluation
rows. Study D replication set.

δ = 0.05: the policy scale re-certified from the sealed calibration chain at δ = 0.05, the policy rerun on the evaluation rows at that
scale if it differs, and the comparators recalibrated to its realized calibration risk (equal-risk) and certified by P7's rule at δ = 0.05.

## P9. Collision crossing between step endpoints (plan F6)

A segment–sphere closest-approach test between consecutive integrator nodes (`orbit_ml_cr3bp/collision_segment.py`), with unit tests.
It is not wired into any sealed kernel (that would change sealed results); it is provided for Study E, where it becomes part of the
violation definition of physical-time arms.

## Defense-plan analyses (2026-11-03; P10–P23)

Added on 2026-11-03 from the review in `DEFENSE_PLAN.md`. The definitions below were written into this file **before** any of them was computed
through `run_posthoc.py`. The review itself had already computed a few of the numbers informally (its "[new]" items: the zero-overhead shares
of Study B, the realized comparator violation rates and the McNemar test on the Study C confirmation set, violation severities, the physical
paradigm term on three subsets, the second-reference agreement, the cost concentration); those definitions were therefore written knowing those
values, and they are fixed here so that the formal runs can only reproduce or contradict them. Nothing was committed separately for this
section (the user had not asked for commits); the order is recorded by the run folders' timestamps, which are all later than this edit.
Labels: P10–P19 are post hoc summaries of sealed ledgers; P20–P23 are new **exploratory** computations (new comparator runs), not tests.

Common conventions (as in P1): a comparator run violates when it failed, hit the 2,000,000-evaluation cap, or ended more than 100 km from
the reference, i.e. NOT (`<comp>_ok` and `<comp>_err_km` ≤ 100); the policy violates when `meets_tol` is false. Intervals are 95 % percentile
bootstrap over orbits (4,000 resamples, seed 0) unless stated.

### P10. Realized risk and severity of every comparator (plan P1-1, P1-2)

Ledgers: Study C confirmation (Sundman and physical-time policies), Study D replication (D3), Study C lock, Study D lock, D4 stratified policy
on the replication set; plus the symmetric-certified columns of P7 (`posthoc/results/symmetric_*/eval_columns.csv.gz`) on the Study C
confirmation and Study D replication sets. For each comparator column: realized violations k / n with Clopper–Pearson interval, and the paired
**exact McNemar test** of policy against comparator over the same orbits: b = orbits where only the policy violates, c = only the comparator,
two-sided p = min(1, 2 · P[Bin(b + c, ½) ≤ min(b, c)]).
Severity, for the policy and each comparator: among violating runs with a finite final error, the median and 90th percentile of error / 100 km
and the share above 10 × tol; violating runs without a finite error (collision, escape, failure, cap) are counted separately and are not in the
error quantiles.

### P11. Same-subset decomposition waterfall (plan P1-3, P0-2, P0-3)

Comparator RK45 at the oracle-tuned tolerance (`RK45_spec`, as Fig 3), repeated against certified RK45 (`RK45_cal`, the same column in both
Study C ledgers). Terms as in Eq (decomp) (`failure_study_3d.decompose`, `decompose_sundman`).
- **(a) common set (within Study C):** Study C confirmation orbits that met the tolerance under both the physical-time and the Sundman-time
  policy, were not abstained by the physical-time policy, and have every term of both decompositions. Mean of every term for both policies
  on this one set, and the paired difference of the totals (paired orbit bootstrap).
- **(b) all usable:** the physical paradigm term P and the Sundman paradigm term P_s over every evaluation orbit with a label, and the total
  log10(C_tot / C_ad) over every evaluation orbit, violating runs included (abstained physical-time runs at their recorded cost), for the
  Study B policy (Study B confirmation set) and both Study C policies.
- The physical paradigm term of Study C also on the sealed subsets (orbits the physical policy met; met and not abstained) and on the
  abstained orbits, to show where the sealed +0.12 came from.
- Study B and Study C are different draws (seeds 22 and 43) of one generator, so the Study B → Study C comparison is between populations,
  not orbit by orbit; Study C physical → Study C Sundman is orbit by orbit.
- The share chain against equal-risk and certified RK45: Study B policy, Study C physical-time policy, Study C Sundman policy (sealed counts).

### P12. Overhead (plan P1-4, P1-7)

- Study B confirmation ledger: share cheaper and ratio of totals with policy cost `C_ml_int + h`, h ∈ {0, 15.04, the sealed C_feat + C_pred},
  against all six comparator columns. The comparator tolerances are unchanged (overhead does not change risk).
- Study C confirmation and Study D replication Sundman ledgers: h = k × the sealed overhead (15.04), k ∈ {0, 1, 2, 5, 10}, against RK45 and
  DOP853 at equal-risk and certified tolerance, overall and by horizon band.

### P13. Concentration of cost (plan P1-5)

Study C confirmation and Study D replication Sundman ledgers: share of the policy's total cost (`C_ml_tot`) carried by its costliest 1 % and
10 % of orbits (ranked by its own cost), the same for equal-risk RK45 (ranked by RK45's cost), and by horizon band the share of orbits and
the share of each method's total.

### P14. Timescale range and horizon (plan P1-6)

Gap = log10(C_fix* / RK45_spec), as T1. On the Study B confirmation set (physical oracle step) and the Study C confirmation set (physical and
Sundman oracle steps; the Sundman gap uses C_s*): Spearman ρ(log10 Δ_τ, gap); the partial Spearman correlation controlling log10 T (Pearson
correlation of the residuals of the ranks of gap and of log10 Δ_τ after a linear regression of each on the rank of log10 T); the OLS slope of gap
on log10 Δ_τ alone and with log10 T as a second regressor; and the OLS slope within each horizon band. Bootstrap intervals for the slopes and the
partial correlation.

### P15. Sampling mixture as a continuous dial (plan P1-9)

Study C confirmation and Study D replication Sundman ledgers against equal-risk and certified RK45: the near-Moon family weight w = 0, 0.05,
…, 1, the other six families keeping their observed proportions among the evaluation orbits and sharing 1 − w; weighted share and ratio of
totals with family-stratified bootstrap (P2's method). The observed near-Moon share is included as a check (it must reproduce the unweighted
numbers). Provenance of the 0.55 / 0.20 / 0.25 mixture: from `git log` (reported, not computed).

### P16. Multiplicity, the D4 level and resampling floors (plan P1-10, P1-12, P0-4)

- Holm over all 15 confirmatory tests of Studies B–D at once (raw p-values from the three sealed `confirm_tests_confirm.json`); adjusted p and
  rejection at 0.05, next to the within-study Holm.
- D4 strata at level 0.05 instead of 0.07: from the sealed D4 stratified ledger, per stratum with at least 300 evaluation orbits the exact
  one-sided binomial p-value of H0 "rate ≥ level", intersection–union p = the largest; first reproduced at 0.07 (must equal the sealed
  `p_strata`).
- Floors: every raw p-value equal to the floor of its resampling test (1/10,001 for 10,000 permutations, 1/4,001 for 4,000 bootstrap
  resamples) is flagged.

### P17. BCa intervals for the ratio of totals (plan P1-11)

Study C confirmation and Study D replication Sundman ledgers, four comparator columns: the percentile interval of P1 and the BCa interval
(bias correction from the share of resamples below the estimate, acceleration from the leave-one-out jackknife over orbits), same 4,000
resamples.

### P18. Agreement of the two references (plan P0-10)

On every usable row of the Study B, C and D datasets (`reference_outcome == "ok"`): `ref_check_km`, the distance between the final positions
of the reference DOP853 (1e-13) and the second DOP853 (1e-12, step ≤ 0.005 TU). Reported: maximum, median, 99.9th percentile, and counts above
1 km and 10 km.

### P19. Distance to the training set (plan P1-8)

Feature space: the 25 inputs of the frozen Sundman policy, each standardized by its mean and standard deviation over the 94,611 training rows
(Study C training and enriched sets). Distance of a point: mean Euclidean distance to its 10 nearest training rows. Computed for the Study D
confirmation evaluation rows (in-distribution reference), the 1,647 in-range D5 runs with a trusted reference, and the trusted grid points of
the three Fig 10 surfaces of section (horizon 8 TU). Reported per group: median distance, and the share above the 95th and 99th percentile of the
in-distribution distances; for D5 per family. Descriptive; it pilots Study E's E5 gate but sets no threshold.

### P20. Adaptive integration in Sundman time (plan P2-1; exploratory)

SciPy RK45 and DOP853 (`solve_ivp`) on the seven-dimensional system (X, t) of Eq (sundman) in the fictitious time s (same g, τ_c = 2), rtol r,
atol = 1e-3 · r on all seven components, from s = 0 with a terminal event t(s) − T = 0 (increasing); the final position is the state at the
event. A run without the event before s = 1e7, a failure, a non-finite state or more than 2,000,000 RHS calls is a violation, as is a final
error above 100 km. Cost = RHS calls of the seven-dimensional system (one evaluation each, as for the Sundman policy). No collision checks (as
for the physical-time comparators). Set: Study D replication, sealed calibration (4,000) / evaluation (12,253) split. Tolerances: certified by
P7's rule (fixed sequence, full δ, `RTOL_GRID_FINE`) and equal-risk by the sealed rule (scan of `RTOL_GRID_B` and four bisections) to the
policy's sealed calibration violation rate on that set. Reported for each: realized violation rate, the policy's share cheaper, ratio of totals
and mean log ratio, next to the same statistics for the physical-time solver at the same rule (sealed equal-risk column; P7 symmetric column).

### P21. An adaptive solver given the same information (plan P2-2; exploratory)

A per-orbit RK45 tolerance predicted from the same 25 features with the same model class and size as the policy (histogram gradient boosting,
25 trees, depth 3, learning rate 0.2, pinball loss at quantile 0.2, seed 42; `policy_c_3d._hgb_regressor`), target log10 of the oracle-tuned
RK45 rtol (`RK45_spec_rtol`). Training rows: the 12,192 evaluation rows of the Study C confirmation ledger (the only rows that carry the oracle
rtol; seed 43, disjoint from the evaluation set). Applied rtol = clip(10^prediction · m, 1e-13, 1e-1) with m on the grid 10^(−3 + j/8),
j = 0 … 24. m certified on the Study D replication calibration rows by the policy's rule (fixed sequence, safest first, full δ) and, separately,
set at equal risk by the sealed rule (scan from the loosest m and four bisections) to the policy's calibration violation rate. The comparator is
charged the policy's overhead (15.04 RHS-equivalents). Evaluation: the Study D replication evaluation rows, policy share cheaper, ratio of
totals, comparator violation rate.

### P22. A multistep comparator (plan P2-3; exploratory)

SciPy LSODA (`solve_ivp`, Adams with automatic switching to BDF if stiffness is detected), atol = 1e-3 · rtol, cost = all RHS calls (finite-
difference Jacobian calls included), the same cap and violation rule; certified symmetric (P7's rule) and equal-risk (sealed rule) on the Study D
replication calibration rows; policy against it on the evaluation rows.

### P23. A higher-order fixed step in Sundman time (plan P2-5; exploratory, oracle only)

300 orbits drawn (seed 0) from the Study D replication evaluation rows. Fixed-step 8th-order Runge–Kutta (the 12-stage propagation formula of
DOP853, no error control) in Sundman time with the RK4 landing rule (landing steps with the same formula in physical time), and, for symmetry,
the sealed RK4 kernel. Oracle step of each: on the grid ds = 4 · 2^(−k/4) down to 1e-4, the largest value that passes together with its eight
finer neighbours (the two-octave rule without the refinement points). Cost at the oracle step = stages × steps (12 or 4). Reported: share of orbits
on which each oracle fixed step is cheaper than oracle-tuned RK45 and DOP853, and the median log ratio of the RK8 to the RK4 oracle cost.

## Deviations

- 2026-10-30, after P1 had run: P2-P8 commands were added to `run_posthoc.py` in the order of this file; the definitions above were not changed.
  Two clarifications that were implicit above: (i) in P7 the delta = 0.05 policy scale is read from the sealed calibration chain (a prefix of
  the delta = 0.1 chain), and the policy is re-run only if it differs (it did not, on either set); (ii) P7 for D4b also re-ran the sealed
  coarse grid first, and P7 for D2 re-ran the sealed policy counts first, as reproduction checks (both reproduced exactly).
- P9 was implemented as code and tests only, as stated; no count of undetected crossings on existing runs was made.
- 2026-11-03/04, defense-plan analyses (P10–P23), differences from the definitions above, all decided before or without looking at the affected
  numbers unless stated:
  - P12: 15 Study B runs (collisions or escapes, all violations) have no recorded cost. They stay in the share's denominator as non-wins (the
    sealed definition, so the shares reproduce `confirm_c_3d.win_share`) and are left out of the ratio of totals and the mean log ratio. The first
    `overhead` run stopped on these rows with an error before writing any number; its empty folder was deleted.
  - P11: the report's entry `C_sund_physical_oracle` repeats the Sundman policy's total next to the physical paradigm term on the same rows; only
    its paradigm term is meaningful.
  - P19: the features of the D5 runs and the Fig 10 grid points are rebuilt from the states (`physics_3d.add_orbit_features_3d` +
    `osculating_3d.add_osc_features`); the rebuild was checked to reproduce the dataset's features on the Study D confirmation rows to 1e-8.
  - P21: a diagnostic was added (median of log10 oracle rtol − predicted rtol on the evaluation rows); it enters no comparison.
  - P23: the oracle step of the 8th-order method is censored at the top of the grid (ds = 4) on some orbits, which makes its cost an upper bound
    (conservative against the 8th-order method); the censored counts are reported.
  - P20–P23 were first run on a handful of rows (`--smoke`, written to a scratch folder outside the repository) to test the code path; no smoke
    number is reported.
  - P20, P21: after the overall results were known, the same statistics were broken down by horizon band (share and ratio of totals against
    Sundman-time RK45 and DOP853 at equal risk and the learned-tolerance RK45 at equal risk), because the manuscript's practical claim is about
    1–3 TU. Not in the definitions above; computed from the run folders' `eval_columns.csv.gz` and the sealed D3 ledger (overhead 15.04 added to
    the learned-tolerance RK45). 1–3 / 3–8 / 8–20 TU: Sundman RK45 share 97.3 / 72.1 / 23.9 %, ratio 1.31 / 0.85 / 0.62; Sundman DOP853 79.8 /
    44.9 / 2.5 %, ratio 0.85 / 0.50 / 0.35; learned-tolerance RK45 66.0 / 23.1 / 5.2 %, ratio 1.16 / 0.97 / 0.83.
