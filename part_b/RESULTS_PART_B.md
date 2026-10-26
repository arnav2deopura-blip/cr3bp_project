# Part B results: when and why ML step-size prediction fails in the 3D Earth-Moon CR3BP

Pre-registration: `part_b/PREREGISTRATION_B.md` (sealed 2026-10-25T23:01:57Z, `FREEZE_B.json`; first seal `FREEZE_B.v1.json`).
Run log (each confirmatory step ran once, no reruns): `part_b/results/confirm/confirm_log_b.jsonl`. Working log with every decision and
the exploratory work: `part_b/PART_B_LOG.md`. Figures: `part_b/figures/confirm/` (D_confirm) and `part_b/figures/explore/`.

All numbers below are on **D_confirm** (seed 22, 20,000 orbits, 16,161 usable; the policy is the frozen D_explore model with scales and
adaptive tolerances re-certified on 2,000 D_confirm rows; **14,161 evaluation orbits**) unless stated. Tolerance 100 km, Earth-Moon, horizons
1-20 TU, cost = right-hand-side evaluations. Part A (sealed 2026-10-23) is unchanged; Part B explains its negative result, it does not revise it.

## 1. Answer in one paragraph

An ML step-size policy that picks one fixed step from the initial state loses to adaptive integration for four separate reasons that
stack, and which one binds depends on the orbit. (1) **Paradigm**: against an adaptive solver tuned per orbit to just meet the tolerance, even
a perfect single step is cheaper on only 25.8 % of orbits, and the penalty tracks how much the trajectory's pace varies (rho = 0.91 with the
physical timescale range). (2) **Information**: the step depends on what happens later on the trajectory (closest lunar approach, timescales);
the initial state does not carry it, but this is a limit of the *features*, not noise: the label is stable under 3.8 km / 1 cm/s
perturbations, and trajectory-derived features cut the error 5x. (3) **Calibration**: one global safety scale halves the step (median 0.30 dex)
and still leaves violations concentrated at close lunar approaches (15.4 % within 0.05 DU of the Moon vs 0.2 % beyond 0.8 DU). (4) **Overhead**: the scikit-learn model costs
about 1,100-1,300 evaluations per orbit, as much as the median fixed-step integration. The policy was cheaper than adaptive on 0 of 14,161
orbits against all six comparators. It is not useless: the same kind of model predicts the adaptive solver's cost well enough to sort
orbits into lockstep batches (modelled lane efficiency 0.08 to 0.54).

## 2. Confirmatory tests (Holm-Bonferroni over five tests, familywise 0.05)

| Test | Hypothesis | Result | p (Holm) | Verdict |
|---|---|---|---|---|
| T1 paradigm | oracle fixed step beats adaptive-spec on < 50 % of orbits, and the gap rises with timescale range | cheaper on 25.8 % vs RK45 and 26.4 % vs DOP853 (n = 14,160); Spearman rho(log10 delta_phys, gap) = 0.909; OLS slope 0.904 [0.898, 0.910] | 4e-4 | **supported** |
| T2 information | label spread under tiny perturbations grows with lambda T (eps = 1e-5 DU) | rho = 0.41 (n = 443); sigma > 0 for 8.8 % of orbits, max 0.136 dex | 4e-4 | **supported in sign**; the floor is far below the model error (below) |
| T3 calibration shape | cells differ in conditional risk AND per-regime (Mondrian) scales cut cost at equal certified risk | chi-square across 45 cells = 3,011 (dof 44), p < 1e-300; Mondrian costs MORE: +0.073 dex [0.072, 0.075] (about +18 %), realized risk 2.6 % vs 3.9 % | 1.0 | **not supported** |
| T4 where | paradigm gap larger at min r2 < 0.17 DU and in regimes II-III than IV-V; violations more frequent at min r2 < 0.05 DU | gap median +0.48 vs +0.07 dex; +0.24 vs +0.06 dex; violations 15.4 % vs 1.3 % | 1e-220 | **supported** (all three contrasts) |
| T5 horizon dial | paradigm gap and integration-only ML gap rise with T, same initial states | per-state mean rho 0.908 and 0.727 (2,326 states); medians -0.20, 0.00, +0.17, +0.24, +0.34 dex and +0.44, +0.55, +0.74, +1.20, +1.24 dex for T = 1, 2, 5, 10, 20; violations 1.1, 1.3, 3.9, 5.6, 9.2 % | 5e-4 | **supported** |

Four of five supported, one not, which is what the exploratory results on D_explore predicted (the pre-registration states those
expectations). The permutation p-values sit at their floor (1/10,001), so the Holm-adjusted values for T1, T2, T5 are floors, not exact.
**T2 needs a caveat**: only the sign hypothesis is tested. The comparison that matters for the information ceiling is size: the model's median
|log10 prediction/label| is 0.32 dex, the label's spread at 3.8 km is at most 0.14 dex and almost always 0 (labels move in 0.0108 dex steps);
in the top lambda T tercile the median residual is 25 times the spread and no orbit's residual is within it.

*Addendum (2026-10-26, Part C Track 0.7):* T2 is a **sign-only** result. It does not say there is an information floor on the model's accuracy: the label is stable under 3.8 km perturbations,
so the model's error is not due to an information floor in the initial state's precision.

## 3. Where each ceiling binds (descriptive, with intervals)

Mean cost-gap decomposition against RK45-spec over the 13,540 orbits that met the tolerance (dex, additive; 95 % CI): paradigm +0.21
[0.21, 0.22], selection +0.03, prediction +0.38 [0.37, 0.38], safety scale +0.30, overhead +0.50 [0.49, 0.51], rounding -0.01, total
+1.41 [1.40, 1.41] (the ML policy costs about 25x the adaptive-spec solve on a geometric-mean basis). Dominant ceiling per orbit
(share of all orbits): overhead 39.7 %, prediction 32.1 %, paradigm 20.6 %, safety scale 2.8 %, selection 0.5 %; violations 3.9 %.
`part_b/figures/confirm/waterfall.png`.

- **The paradigm ceiling depends on how the adaptive solver is tuned.** Against the per-orbit oracle tolerance (the hardest baseline) it
  binds: a perfect fixed step is cheaper on 25.8 % (RK45) / 26.4 % (DOP853) and 70 % of the cost losses are paradigm-bound. Against one
  certified tolerance for all orbits it does not: a perfect fixed step is cheaper on 67.9 % (RK45-cal) / 55.8 % (DOP853-cal), and 63 % / 53 %
  against the equal-risk tolerance. "Single-step prediction cannot win" is true against an oracle-tuned adaptive solver and false against a
  realistically tuned one; the study's negative result then comes from prediction, safety scale and overhead.
- **Closest lunar approach organizes everything** (`wheremap.png`). Orbits that pass within 0.05 DU of the Moon: paradigm gap +0.32 to
  +1.10 dex by regime, violation rates 1-48 %; beyond 0.8 DU the paradigm gap is about 0 (-0.06 to -0.01 in regimes III-V) and violations are 0.0-0.4 % in
  regimes III-V. Hill regime I (closed necks) has the largest paradigm gap overall (+0.36 dex median), regimes IV-V the smallest (+0.02, +0.08).
  A perfect fixed step is cheaper than RK45-spec on 59 % of regime-V orbits that never come within 0.8 DU.
- **Violations are step-overs.** The violation rate rises with dt_used / tau_min from 3 % (below 0.02) to 42 % (0.2-0.5), 85 % (0.5-1) and 99-100 % (above 1);
  AUC 0.85 for violation vs not (`step_over.png`). A cheap guard that aborts when dt / tau_min exceeds 0.3 catches 57 % of violations with 7.9 % false alarms;
  at 0.5 it catches 36 % with 1.5 %. (These use runs at the certified scale, 1.0 and 2.0 times the raw prediction, so violations are over-represented.)
  Of the 546 violations, 99 pick leapfrog for an orbit where no step >= 1e-4 meets the tolerance (a classifier error, its own class); of the other
  447, 446 have a scaled step above the label (median 2.0x, 90th percentile 11.6x, maximum 53x) and one sits 1 % below it; only 1 of 13,524 runs whose scaled
  step was at or below its label violated.
- **Information vs model** (`information_floor.png`). E3b on D_explore: giving the same model trajectory-derived features (closest approach,
  timescales, transit counts) cut the median error from 0.294 to 0.057 dex, raised the certified scale from 0.5 to 0.8, and cut violations from
  3.97 % to 0.83 %. Those features need the integration the policy is trying to avoid, so this is an upper bound, not a design.
- **Calibration alternatives** (E5, certified at alpha 0.05): the global scale gives realized risk 3.9 %, mean cost 15,034; Mondrian by regime 2.6 %, 20,085;
  dispersion bins 2.5 %, 24,130; abstaining on the 20 % most dispersed orbits to adaptive 2.9 %, 11,059 (26 % lower mean cost); label-conformal scaling is cheapest
  but its realized risk is 6.4 %, above the 5 % target. None makes the policy cheaper than adaptive.
- **A one-parameter physics rule does not match the model.** The best certified rule dt = c tau(IC) T^-b is 12.0x costlier in integration than the ML policy at a similar
  realized risk (3.3 % vs 3.9 %, both certified at alpha 0.05), so the model learned more than a Kepler timescale.
- **Reference validity** (E1): Radau at 1e-12 vs the DOP853 label reference, 600 usable orbits: maximum difference 0.003 km, none flagged (threshold 10 km),
  no outcome disagreement; for 300 crash/escape orbits the event type agreed on all and event times to 1e-15 TU.

## 4. What the policy and the model are good for

- **Cost of the model is the closable gap.** ML integration alone is cheaper than RK45-cal on 28.3 % of orbits, but its median headroom is 48 evaluations
  (18 vs DOP853-cal, 4 vs RK45-spec); a model whose features and inference cost more than that cannot win at the median. The scikit-learn model costs
  about 1,128 (pinned; isolated measurements 1,100-1,340 depending on machine load). Whether a compiled model fits inside ~50 evaluations (about 5 microseconds) was not tested.
- **Gated policy (E11, D_lock, 4,034 orbits).** Using the fixed step only in deployable cells (Hill regime x horizon band) where it wins on at least half
  of D_explore orbits and violates at most 5 %: with overhead 0-10 evaluations two cells qualify, 20 % of orbits are routed to ML, and the total cost is
  1.065x pure adaptive at the equal-risk tolerance (risk 4.2 % vs 4.0 %); with overhead 50 or more no cell qualifies and the gate reduces to adaptive.
  As pre-registered, it does not beat adaptive.
- **ML as a cost predictor for batching (E11b).** A regressor on the initial state, trained on D_confirm and tested on D_lock, ranks the adaptive solver's cost with
  Spearman rho 0.964. In a lockstep-batch cost MODEL (not a GPU measurement) sorting by predicted cost raises lane efficiency from 0.080 to 0.536 at batch size 32
  (oracle ordering 0.826) and from 0.042 to 0.397 at batch size 128 (0.602), cutting modelled batch cost to 15 % and 11 % of unsorted. Not part of any hypothesis test.

## 5. Limits you should keep in mind

- One tolerance (100 km), one system (Earth-Moon), one model family (random-forest classifiers plus a q = 0.1 gradient-boosted regressor, trained on 11,170 orbits;
  E4 shows the error still falls with more data, fastest for close-approach cells). Integrator error against the CR3BP, not against real trajectories.
- Costs are evaluation counts. The adaptive solvers are scipy's; the fixed-step kernels are compiled. The per-orbit "spec" tolerance is an oracle no user has;
  "cal" and "equal-risk" are the realistic baselines, and the paradigm result differs between them (section 3).
- The T3 null is specific to 2,000 calibration rows and a Bonferroni split over groups; with far more calibration data Mondrian scales could pay.
- Model-inference cost is a wall-time ratio and noisy; it was pinned in advance. The conclusion does not depend on it: with zero overhead the policy still
  beats RK45-spec on 0.3 % and DOP853-spec on 4 % of orbits.
- D_confirm's calibration rows and evaluation rows come from the same sample, so the certified risks are marginal, and in the horizon sweep and D_lock the calibration
  is frozen from D_confirm.
- The label is quantized (0.0108 dex), so E3a cannot resolve spreads below that; this is far below the model residual but it limits what "no information floor" means.
- Crash/escape orbits (19 % of the draw, concentrated in regime I) are outside the study; the evaluated population under-represents bound near-Moon motion.

## 6. Not done (and why it matters)

HB3 (E3c lookahead) and the mu dial (E7) were run afterwards as pre-registered EXPLORATORY studies: see `part_b/RESULTS_E3C_MU_DIAL.md` (seeing the trajectory buys accuracy but not net cost; both gaps
fall as mu falls, the violation rate does not). Neither is in the confirmatory family (corrections of 2026-10-26 are in the Errata section at the end of `RESULTS_E3C_MU_DIAL.md`: the short-lookahead gain does not survive risk matching, and the mu trends are confounded). Cross-mu transfer with dimensionless features (Issue 10) is still not tested, so Part A's transfer failure is still
confounded. E8 (periodic families, halo/NRHO out-of-distribution behaviour, Issue 12) and E12 (2D control) were deferred.
Plan issues 5 (the deployed policy is still certified per integrator, not end to end), 11-14 and 16-18 were not addressed in code. Issue 4 is covered by the pinball/coverage
reporting in E4/E5, issue 9 by the physics baseline in E10, issues 3 and 14 only partly (E1 validates the reference; the endpoint-only error definition is unchanged), and issue 10 not at all (no mu dial). The one-time `RESULTS_REPORT.md.bak` duplicate was left alone.

## 7. Deviations and process notes (all also in the pre-registration or `PART_B_LOG.md`)

Pre-registered deviations from the plan are in `PREREGISTRATION_B.md` section 7. After the seal: the first seal was replaced before any confirmatory step ran (a runner
folder bug that would have mixed the D_lock and D_confirm policies); figure code was edited for legibility and a dataset-label argument after the confirmatory steps ran
(the first confirm render of the where-map and waterfall wrongly said D_explore in the title and was regenerated). `freeze_b.py check` therefore reports
`make_figures_b.py` as changed, and nothing else. A stray background process from a failed stdin-driven job slowed two exploratory runs and was stopped; no result changed.

## 8. Reproduce

```bash
PY=/Users/arnavdeopura/miniforge3/envs/orbit-ml/bin/python
$PY generate_3d_data.py --num_orbits 20000 --label_mode accuracy_monotone --pipeline v4 --base_seed 21 --n_workers 8 --output_dir data_b --tol_km 100 --chunk_size 8   # D_explore (22: D_confirm; 23 with 5000: D_lock)
$PY run_failure_study_3d.py policy --data data_b/cr3bp_3d_training_data_accuracy_monotone_seed21.csv --out_dir out_b                                                   # fits the frozen model
$PY run_confirm_b_3d.py steps        # then: run <step> --target confirm|lock, tests (needs FREEZE_B.json to match)
$PY tests/test_phase0.py && $PY tests/test_failure_study.py
```
Ledgers (one row per orbit, 126 columns) are `part_b/results/*/policy/ledger.csv*`; the datasets and the 25 MB frozen model are not in git (hashes in `FREEZE_B.json`, `part_b/DATA_HASHES_B.json`).
