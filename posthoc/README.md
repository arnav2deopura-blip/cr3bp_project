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

## Deviations

(none yet)
