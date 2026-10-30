# Study A: summary

Extracted unchanged from section 1 of the full Study A report, which moved to `archive/study_a/RESULTS_REPORT.md` (9,106 lines; every
table there is generated from the run folders). Studies B-D are reported in `part_b/RESULTS_PART_B.md`, `part_c/RESULTS_PART_C.md`,
`part_d/RESULTS_PART_D.md`, and the manuscript in `manuscript/main.tex`.

## Executive summary

### 1.1 Hypothesis scorecard

All numbers come from the locked-set final run unless the row says otherwise.

| ID | Question | Result | Key numbers |
|---|---|---|---|
| H1 | Does ML (RK4/leapfrog + certified safety scale, FTLE feature cost counted) stay within tolerance on ≥90% of orbits AND need fewer evaluations than RK45 and DOP853 on ≥50% of orbits? | NOT supported | Within tolerance: 96.3% (pass). Cheaper than adaptive: 0.0% / 0.0% / 0.0% / 0.0% of orbits (RK45 spec / RK45 cal / DOP853 spec / DOP853 cal; needs ≥50%). Median ML cost (feature cost included) is 55× to 193× the adaptive cost. |
| H1'a | Same, but model trained without FTLE, prediction cost charged (added after seeing development data) | NOT supported | Within tolerance: 96.1%. Cheaper than adaptive in all four comparisons: 0.0%, 0.0%, 0.0%, 0.0%. (Before prediction cost is charged, 29.3% / 20.3% of orbits are cheaper against the calibrated RK45 / DOP853 only.) |
| H1'b | Same, but compared with the best single certified fixed step per integrator (added after seeing development data) | Supported (post-hoc comparator) | ML cheaper than global fixed RK4 on 89.2% and than global fixed leapfrog on 98.1% of orbits, at 96.1% in-tolerance for ML (global RK4: 96.1%; global leapfrog: 96.8%). |
| H2 | Certified scales give ≤5% violation rate on the locked audit rows | Point estimates meet 5%; 95% intervals reach above 5% | RK4: 17/500 = 3.40% (Clopper-Pearson 95% CI 1.99% to 5.39%). Leapfrog: 19/498 = 3.82% (CI 2.31% to 5.89%). Without the scale: RK4 11.2%, leapfrog 14.5%, Euler 27.1%. (CIs computed by this report's script, not stored in the result files.) |
| H3 | Is FTLE needed? (removing it lowers RK4 R² or selection accuracy with the whole paired 95% interval below zero) | Not met under the rule | Pinned HGB-quantile model: RK4 R² change -0.2526 (95% CI -0.543 to 0.03776); selection accuracy change 0.007913 (CI -0.008581 to 0.02441). RF sensitivity run: RK4 R² change -0.07033 (CI -0.1428 to 0.002113); selection accuracy change -7.516e-04 (CI -0.01053 to 0.009027). All intervals include zero. |
| H4 | Labels: fraction of orbits with non-monotone pass/fail over a dense step sweep | Reported (no threshold) | 1.5% of 200 orbits (95% CI 0.3% to 4.3%). Median relative disagreement of the halving search vs dense sweep: 0.82%; monotone label vs dense sweep: 0.80%. |
| H5 | Conformal error-model recommender: violation rate ≤10% among covered test orbits | NOT met | Violation rate 17.4% (15 of 86 covered orbits; 95% CI 10.1% to 27.1%); coverage 95.6%; 90 test orbits only. Cheaper than RK45/DOP853 on 0.0% / 0.0% of orbits. |
| H6 | Crash / escape prediction with and without FTLE | Reported (no threshold) | With FTLE: accuracy 98.3% (majority baseline 83.1%), AUC 0.9929, recall 86.1%. Without FTLE: accuracy 80.2%, AUC 0.7801, recall 19.9%. Caveat: FTLE is computed over the same horizon as the trajectory (section 9). |
| H7 | Convergence slopes vs textbook order | Reported (no threshold) | Median fitted slope: Euler 1.000 (theory 1), leapfrog 2.000 (theory 2), RK4 4.017 (theory 4). RK4 deviates most near the Moon / L1. |

### 1.2 What this means in plain terms


**The headline.** The ML model does **not** save computation compared with standard adaptive solvers (RK45, DOP853). The preregistered rule for that outcome is: *"ML does not beat adaptive solvers on cost."* The final run's own label is `ML_DOES_NOT_BEAT_ADAPTIVE_ON_COST: do not claim a speed advantage`.

**Why it loses.**
- The ML recommendation is a *fixed* step size, shrunk by a safety factor so that it stays under 100 km error. It averages 15,468.4 force evaluations per orbit; the adaptive solvers need about 1,077.2 (RK45) and 898.1 (DOP853) at the per-orbit oracle tolerance.
- The FTLE feature costs more than a whole adaptive solve (variational FTLE steps cost 7.3× an RK4 step). Dropping it (H1′) removes that cost but model inference still costs about 590.9 force-evaluation equivalents per orbit; the median adaptive solve on the same orbits needs 146 (RK45, oracle rtol) or 122 (DOP853, oracle rtol) evaluations.

**What does work.**
- The calibrated safety layer cuts the violation rate from 11.2% / 14.5% (RK4 / leapfrog, raw) to 3.4% / 3.8% (scaled).
- Against the best *single* fixed step, ML is cheaper on 89% (RK4) and 98% (leapfrog) of orbits. Mean evaluations: ML 15,786.6 vs global RK4 41,874.1.
- Integrator selection beats "always RK4" by a small margin: 89.4% vs 86.8% on the locked set.
- Crash/escape detection with FTLE is strong (AUC 0.9929 vs 0.7801 without). FTLE is integrated over the same horizon as the crash it is asked to predict (sections 9 and 15).
- Within the Sun-Earth system (5,000 orbits), integrator selection reaches 91.8% vs 61.1% for always-RK4 (95% CIs 89.2%-94.4% vs 52.1%-70.0%). This is a within-system result and does not carry over to the other system.
- The numerical methods behave exactly as theory says (slopes 1, 2, 4).

**What does not work.**
- Euler is essentially unpredictable (R² ≈ 0.01754) and is excluded as a candidate (used as a negative control).
- Transfer between systems does not beat the trivial always-RK4 baseline in either direction (section 8): Earth-Moon → Sun-Earth selection accuracy 61.8% vs 61.1%; Sun-Earth → Earth-Moon 81.9% vs 86.1%.
- The conformal error-model recommender violates tolerance more often than its target (section 11).
- The CR3BP model itself differs from a real ephemeris by thousands of km, so a 100 km tolerance only describes integrator error *inside the model* (section 13).


### 1.3 Wall-clock note (read before quoting any speed number)


The ML integration step on its own ran in a mean of 0.001496 s per orbit versus 0.004375 s (oracle rtol) / 0.007062 s (calibrated rtol) for RK45 and 0.003448 s / 0.004654 s for DOP853 (H1 run). That is **not** a like-for-like comparison: the fixed-step code is compiled (Numba) while the adaptive solvers are SciPy, and the project's own code (`fair_baseline_3d.py`) notes that wall-clock comparisons are biased against the adaptive solvers. Once feature extraction and prediction are included, the mean total ML wall time is 0.08888 s (H1 run, FTLE model) and 0.1203 s (H1′ run), i.e. slower than the adaptive solvers. The preregistered cost metric is right-hand-side evaluations, not wall time.


