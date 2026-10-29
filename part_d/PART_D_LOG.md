# Part D log: is the ML needed, does it matter in wall-clock, is it safe where it counts, does it generalize

Working log for `CR3BP_Improvement_Plan_Part_D_1.md` (the plan, prepared 2026-10-28; kept outside the repository, in Downloads). Parts A-C are sealed and are not
re-run or modified in their results: Part C is commit `4586803` (`FREEZE_C.json`). Part D has its own pre-registration (`part_d/PREREGISTRATION_D.md`), its own seal (`FREEZE_D.json`)
and fresh seeds. "dex" = log10 units; costs are right-hand-side evaluations unless a wall time is stated. Dates are the machine date of the run (UTC where a log line says so).
Entries are appended as each step finishes; nothing above an entry is edited afterwards except to add a pointer to a correction.

## 0. Decisions asked for in Section 8 of the plan

The plan recommends or defaults on each; I follow it and record what I did. The user can overrule any of them (the work is organized so that dropping one does not touch the others).

| # | Question | What this log does | Why |
|---|---|---|---|
| 1 | Fair date | Unknown, so all five Tier 1 tests are implemented and the Tier 2 items that need no new labelling; the GPU / lock-step SIMD kernel is NOT implemented (lane efficiency is modelled from measured step counts, see D2) | the plan's fallback list (D1, D3, D4, D5, no GPU part of D2) is a subset of what is done |
| 2 | Accept 'the physics did the work' as a possible headline | Yes. D1 is pre-registered one-sided with the rule chosen as the best of six pre-specified rules on D_select_D (a strong baseline), and the result is reported whichever way it falls | the plan recommends yes |
| 3 | Wall-clock: Dormand-Prince 5(4) alone? | DP5(4) with a PI controller is the primary compiled comparator; DP5(4) with SciPy's controller and a compiled DOP853 (SciPy's own tableau) are also run | the plan allows the compiled DOP853 as optional; porting it from SciPy's tableau was cheap and the compiled DP5(4) reproduces `solve_ivp`'s `nfev` exactly |
| 4 | Schedule E12 (2D control) and the leapfrog symplecticity check for the NHSJS 2D paper | the symplecticity check is done (Tier 3, small); E12 is not (see section 9 of the results) | the check is a day of work, E12 needs a planar pipeline |

## 1. Environment and data (2026-10-28)

* Project environment: `/Users/arnavdeopura/miniforge3/envs/orbit-ml` (Python 3.11, numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.0, numba 0.67.0). Machine: the user's Apple M5, 10 cores. Note the desktop is in use
  (WindowServer, an aerial-wallpaper extension and a video decoder service take roughly 30 % of a core between them), so the timing steps are run at a 1-minute load average below 1.5 and the load is recorded.
* Part D datasets (pipeline v5, the Part C command with new seeds, `gen_part_d.sh`): seed 52 select (10,000 candidate orbits, 8,101 usable), 53 confirm (20,000, 16,239 usable), 54 lock (5,000, 4,074 usable), 55 replication
  (20,000, 16,253 usable). Seed 51 (the plan skeleton's 'train extension') is not used: the ML policy and the physics rules are fitted on the SAME Part C D_train (41, 141), which is what makes D1 a fair comparison.
  Generation took 4.7 / 12.5 / 3.2 / 13.4 minutes (the Part C log recorded 25 minutes for 20,000 orbits on a busier machine). Reference final positions (DOP853, rtol 1e-13) are cached in `out_d/refs/<set>.npz`.
* JPL Earth-Moon periodic-orbit families fetched by `fetch_jpl_d.py` (21 streams, 40 MB, raw responses and SHA-256 in `data_d/jpl/jpl_hashes.json`; the hashes are copied into `part_d/DATA_HASHES_D.json`).
  Their mass ratio is 1.215058560962404e-2 against config.MU = 1.215058535e-2 (relative difference 2.1e-8) and their length unit is 389,703.26 km against the project's DU = 384,400 km: D5 integrates at the JPL mass
  ratio and keeps the tolerance at the project's 100 km / 384,400 DU (101.4 km in JPL units).

## 2. Code written before any result (all new modules; the only edit to an existing module is a 4-line `rule` branch in `policy_c_3d.predict_raw_c`, taken only by Part D rule models)

`compiled_solvers_d.py` (Numba DP5(4) with SciPy's controller or Hairer's PI controller, DOP853 from SciPy's tableau, the compiled Sundman policy pipeline, batch kernels, rtol certification), `rules_d.py`, `deploy_d.py` (lean
deployment: the Part C arithmetic without the Python process pool per scale), `strata_d.py`, `periodic_d.py`, `confirm_d_3d.py`, `run_part_d_3d.py`, `run_confirm_d_3d.py`, `freeze_d.py`, `fetch_jpl_d.py`,
`tests/test_part_d.py` (16 checks).

Checks that the new code is the old code where it must be:
* the compiled DP5(4) with SciPy's controller reproduces `solve_ivp(method='RK45')`'s `nfev` on every one of 40 test orbits at rtol 1e-4, 1e-7 and 1e-10 (and the final state to 1e-9); the compiled DOP853 matches `nfev` on 99.3 % of 300 orbits at
  rtol 1e-5 and on all of them at 1e-8 (one orbit differs at 1e-5 because floating-point summation order flips a borderline step acceptance);
* `deploy_d.run_rows` reproduces the Part C `sund` ledger of the D_select dry run orbit for orbit (600 orbits: `ml_evals` identical, `ml_err_km` within 6e-14), and the compiled policy pipeline (features, 25 trees, integrator) gives the same `ds`
  as scikit-learn's `predict` to the last bit.

## 3. D1 rule selection on D_select_D (2026-10-28, `run_part_d_3d.py select_rules`; every choice made here, nothing on the confirmation data)

Overheads measured first, one session (`part_d/selection/overheads_idle.json`; the RHS took 18.3 ns on this machine against Part C's 25-28 ns): `sund` 16.09 RHS-equivalents (Part C pinned 15.04 on the older timing), rules with the periapsis kernel 7.3-7.4, `rule_T` 2.05,
`rule_TH` 2.07, `lin33` 11.39. (The first measurement, taken while the previous job's load was still draining, gave 16.29 and 7.0-7.2; the second, at 94 % idle CPU, is the pinned one.)

Fitting: weighted linear quantile regression on the whole D_train + D_train_enriched (94,611 usable rows), HiGHS interior point (the simplex solver needed 16 s at 20,000 rows and was abandoned after 25 minutes at 60,000). The label-fit quality of the rules on D_select_D at q = 0.2 (median
|log10 prediction / label|, raw model output): rule_T 0.227, rule_TH 0.186, rule_TP 0.167, rule_THP 0.163, rule_THR 0.163, rule_THPD 0.163 dex, against 0.082 for the 25-tree model at the same quantile.

Selection (calibration fold = half of D_select_D, evaluation fold = the other half, seed 42; 4,050 calibration rows, 4,051 evaluation rows): median total cost per orbit that met the tolerance, overhead included:
ML (`sund`) 252.1 at scale 0.80, violation rate 2.6 %; rule_THR 307.3 (q = 0.1 and 0.2 tie), rule_THP 321.4, rule_THPD 323.3, rule_TH 330.1, rule_TP 331.3, rule_T 342.1. **Winner: `rule_THR`, q = 0.1** (the lowest median; the 2 % tie with q = 0.2 goes to the lower quantile
by the sort order of the selection code; the other quantiles are 4-9 % worse). All six rules and `lin33` are frozen in `part_d/frozen/` (each rule at its own best quantile; `lin33` at q = 0.1 as in Part C's S1).

## 4. Dry runs on D_select_D (never counted; the expectations of the pre-registration come from here)

* **D1** (`out_d/dry/d1`): ML saves **0.069 dex [0.065, 0.072]** against rule_THR, orbits that met the tolerance under both (n = 3,863), ML cheaper on 68.4 % of them. Against the other arms (paired, same orbits): rule_T 0.131, rule_TH 0.089, rule_TP 0.085, rule_THP 0.075,
  rule_THPD 0.076 dex; **the linear model on the 33 engineered features is only 0.014 dex [0.011, 0.017] behind the trees** (ML cheaper on 48 % of orbits: a toss-up per orbit, a 3 % gap in the mean). So the periapsis features carry the information; most of the tree model's edge
  over the plain rules is that it uses those features without a hand-chosen functional form, and a linear model captures nearly all of that. Against equal-risk RK45 the ML policy is cheaper on 71.8 % of the evaluation orbits (Part C: 71.4 %), the best rule on 54.5 %; against equal-risk
  DOP853 52.1 % and 33.9 %.
* **D4** (`out_d/dry/d4`; two halves of D_select_D: 4,050 calibration rows, 4,051 evaluation rows, so only 7 strata survive the 400-row merge rule and none abstains): the stratified policy has a realized violation rate of 2.47 % (marginal-certificate policy: 2.62 %); the worst reported stratum
  (regime I, pooled over the Moon classes and horizons) 4.0 % with a 95 % upper bound of 5.8 %; share cheaper than the equal-risk RK45 68.0 % (marginal policy: higher). The IUT p is 7.4e-4. **But the risk is still concentrated where the trajectory actually passes the Moon**: by the a-posteriori closest-approach
  band the stratified policy has 7.5 % [CP upper 9.6 %] violations at min r2 < 0.05 DU (marginal policy 8.7 %), and the equal-risk RK45 has 8.2 % (marginal-risk-matched) there against 0.1 % beyond 0.8 DU: the equal-risk adaptive solver concentrates its violations in the same place, which the plan asked to be shown.
  The free strata (osculating periapsis) see only part of the close passes. The hypothesis is about the free strata, as the plan specifies; the a-posteriori band is tabulated and not tested.
* **D5** (`out_d/dry/d5`, the dry sample: 1,050 sampled members, 919 kept, 106 dropped because the reference stopped before one period (mostly halo L1 members that impact the Moon: 38 of 50 sampled halo-L1 members of each branch) and 25 by closure; 1,599 in-range runs with a trustworthy reference):
  **the family-balanced violation rate is 70 % [68, 72]**; by family (rate): axial L1/L2 100 %, vertical L2 100 %, vertical L1 95 %, Lyapunov L2 81 %, L1 80 %, butterfly 51 %, DRO 51 %, halo L1 41 %, resonant 41 %, halo L2 35 %. The equal-risk RK45 violates in 26-64 % of the runs of the same families
  (it was calibrated on random orbits too, so unstable families fail for it as well), but far less than the policy on the vertical and axial families. This is NOT a bug: on a vertical L2 member (stability index 1.0, one period, T = 6.3) the Sundman RK4 converges at fourth order to 0.0 km as ds shrinks (673 km at ds = 0.1,
  3.9 km at 0.03, 0.04 km at 0.01), the policy chose ds = 0.168 (6,605 km) where about 0.03 is needed. The Part C policy mis-predicts the step on these structured orbits by a factor of about 5 and the certificate (a marginal statement about random orbits) says nothing about them.
  Recalibrating a scale on the other sample brings the violation rate to 4 %, at a certified scale of 0.18 (against 0.87) and a median total cost of 3,287 RHS evaluations, and the policy is then never cheaper than RK45.
* **D2** (`out_d/dry/d2*`; 4,101 evaluation rows of D_select_D, compiled solvers recalibrated on the 4,050 calibration rows in 3-5 seconds each: the certified rtol of every compiled solver is 1.97e-7, the grid value Part C's SciPy RK45 certified; equal-risk rtols DP5(4)-PI 3.1e-7, DP5(4)-SciPy-controller 4.2e-7, DOP853 3.3e-7).
  The chronology matters because the first two attempts were unfair to one arm and are not what is reported:
  (1) first dry run: the policy with Part C's integrator kernel took 26.5 ns per RHS-equivalent against 18.6 ns for the compiled DP5(4); the Sundman right-hand side needs two more square roots (the time scaling `g`) and Part C's landing steps allocate arrays. Throughput ratio 0.58 [0.55, 0.62].
  (2) second attempt: a micro-optimized policy kernel (one reciprocal per body, no allocation; same `ok`, same `evals` on 200 test orbits, final state within 5e-11) and a runtime `fast` flag in the comparators' right-hand side, which made the COMPARATORS 50 % slower (a dispatch layer that blocks inlining); throughput 1.04 was an artefact of the dispatch layer and is discarded.
  (3) what is reported: both arms compiled with the right-hand side as a compile-time constant (`make_adaptive_kernels`, no dispatch): policy 22.6 ns, DP5(4)-PI 23.1 ns per RHS-equivalent (median over orbits). **Throughput ratio 0.87 [0.82, 0.91]**: the policy is faster on **79.6 %** of the orbits (median per-orbit ratio 1.49) but slower in total,
  its mean time being 32.8 us against 28.4 us, because the long chaotic horizons dominate the sum. By horizon band the ratio is **1.62 (1-3 TU, policy faster on every orbit), 1.09 (3-8 TU), 0.72 (8-20 TU)**; by Hill regime 0.79 (I) ... 1.21 (IV). Against DP5(4) with SciPy's controller 0.81, against the certified-tolerance DP5(4)-PI 0.94, against the compiled DOP853 at equal risk 0.63.
  All-core batches (10^3 ... 10^5 orbits, chunk size 16, 7 repeats): the comparator-over-policy ratio is 0.89-0.95 for DP5(4)-PI, 0.82 for DOP853; one thread 0.85-0.95. No batch size passes the pre-set criterion. With Part C's kernel arithmetic (variant R) the ratio is 0.58, which is the number a naive implementation would report.
  The RHS-count and wall-clock rankings agree orbit by orbit (Spearman 0.99 between the two log ratios) with an offset: the RHS count says the policy is cheaper on 77 % of the orbits, the stopwatch on 80 %, and the totals are dominated by the same few long orbits. Modelled lane efficiency of a lock-step batch from these step counts:
  random order 0.54 / 0.31 / 0.18 / 0.06 for 2 / 4 / 8 / 32 lanes (the adaptive solver 0.57 / 0.34 / 0.21 / 0.08): no better, so lock-step execution does not favour the fixed-step policy without sorting; with orbits sorted by predicted work 0.99 / 0.96 / 0.94 / 0.79 (adaptive 1.00 / 0.99 / 0.97 / 0.85). No GPU kernel was written or timed.

## 5. Seal (2026-10-28T21:37:15Z, `FREEZE_D.json`)

Everything above was done before any confirmatory step. After the dry runs: the D2 kernels were corrected twice (section 4, D2), `d5` gained the cluster bootstrap, the oracle-label and recalibration explanations, `d10` and the IAS15 commands were written, `fetch_jpl_d.py` hashes and the dry reports were copied to `part_d/results/dry/`
and added to `part_d/FROZEN_FILES.json`; the pre-registration's expectations were filled from the dry runs (dry D3 on the select set reproduces Part C's command path: scale 0.8, 72.0 % cheaper than RK45-eqr, C1-C5 as in Part C; the dry `tests` step runs end to end: D1, D3, D4 supported, D2 and D5 not). Parts A-C tests: `tests/test_part_c.py` 30 passed, `tests/test_phase0.py` and
`tests/test_failure_study.py` pass with the one-line hook; `tests/test_part_d.py` 16 passed. The seal records git HEAD `4586803` plus 23 uncommitted paths (nothing was committed), 83 source files and 60 frozen files.
Smoke tests of the exploratory commands (D6 at 6,000 rows, D12 on 82 orbits, D7 on a 14 x 14 grid) preceded the seal; the D7 expectation in the pre-registration was written after the 14 x 14 smoke test and says so.

## 6. Confirmatory run (2026-10-28 21:44Z - 2026-10-29 00:02Z, `confirm_log_d.jsonl`; every step once; the seal check passed before each)

Order: d2 (alone, idle machine; started with `--max_load 1.6`, load 1.499 at the start), then the chain d1, d5, d4, d3_apply_sund, d3_apply_phys, d3_apply_global, d3_c1_resid, lock, tests. Wall time of the chain about 2 h 14 min (d4 54 min: the 16,239-row calibrations of two policies' comparators dominate).
Holm over D1-D5: **D3 (p < 1e-300), D4 (p = 1.3e-4, Holm 5.0e-4), D1 (p = 2.5e-4, Holm 7.5e-4) supported; D2 (p = 1) and D5 (p = 1) not**, as the dry runs predicted; every number is in `part_d/RESULTS_PART_D.md`. Differences from the written expectations worth recording: D1's linear arm flipped sign
(dry: 0.014 dex behind the trees; confirmation: 0.017 dex ahead, at a higher violation rate; a tie within the grid's resolution); D3's companion C3(a) came out at 50.6 % (p = 0.11; the dry run had 50.9 %): not above 50 %; D4 had 17 strata (7 on the dry halves) and none abstained.
Not as expected: D5's expectation said 'not supported', which held; its expectation for the a posteriori mechanism (a factor of about 3 in the step) was 0.16 dex in the median and 0.3-0.5 dex on the worst families, i.e. a factor 1.4 overall and 2-3 on the vertical, axial and Lyapunov families.

## 7. Exploratory runs (after the confirmatory chain; the code of D6, D7, D10, D12, symplectic, IAS15 is sealed, D4b is not)

* D6 (grouped importance), D7 (chaos map), D12 (reference check), the leapfrog symplecticity check, D10 (tau_c sweep and the non-monotone labels), IAS15 (REBOUND), and D4b (the per-stratum certified adaptive comparator; written after the confirmatory D4 result): results in section 9 of the report; expectations met except that D7's AUC is higher than the pre-registered range
  (0.92-0.94 against 0.7-0.95: inside it) and D10's landing hypothesis is only weakly supported (+13 % in the change of the landing-step count at a flip).
* Two implementation notes: (a) the D12 independent method was changed from Radau (which did not finish in 10 minutes at 1e-13) to LSODA before any confirmatory use; (b) the REBOUND 5 API sets the IAS15 tolerance with `sim.integrator.epsilon` (not `sim.ri_ias15.epsilon`) and its default adaptive mode is the Pham-Rein-Spiegel criterion.
* Not done: D8, D9, D11, KS-regularized RK4, E12, a GPU kernel.
