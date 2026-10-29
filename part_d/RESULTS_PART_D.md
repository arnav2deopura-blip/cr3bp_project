# Part D results: is the ML needed, does it matter in wall-clock, is it safe where it counts, does it generalize

Pre-registration: `part_d/PREREGISTRATION_D.md` (sealed 2026-10-28T21:37:15Z, `FREEZE_D.json`; the seal check passed before every confirmatory step and before the tests). Log of every decision, the dry runs that the expectations come from, and the two kernel corrections that preceded the seal: `part_d/PART_D_LOG.md`.
Plan: `CR3BP_Improvement_Plan_Part_D_1.md`. Data: new seeds 52 (select, 8,101 usable orbits), **53 (D_confirm_D, 16,239)**, 54 (D_lock_D, 4,074), **55 (D_replication, 16,253)**, the JPL periodic-orbit families, and Part C's D_train for every fit. Earth-Moon, 100 km, horizons 1-20 TU.
Costs are right-hand-side (RHS) evaluations unless a wall time is stated; "dex" is log10; a policy's cost includes its measured model overhead. Every number below is read from `part_d/results/` (copies of the run outputs; the full folders are in the untracked `out_d/`).

## 1. The result in one paragraph

Part C's Sundman-time policy **replicates**: on 16,253 orbits nobody had looked at, it is cheaper than an equal-risk RK45 on **71.6 % [70.8, 72.4]** at a realized violation rate of 2.7 % (Part C: 71.4 %, 4.4 %). The ML is **needed in the sense that a rule with 3-4 coefficients on the same free quantities is 0.072 dex (about 18 %) more expensive** (D1), but **not in the sense of trees**: a linear model on the 33 engineered periapsis features ties it (-0.017 dex, at a higher violation rate, below the 0.038 dex resolution of the scale grid). The certificate can be made **conditional**: certifying a scale per free stratum (17 strata, none abstains) puts every stratum's violation rate at or below 3.1 % (largest 95 % upper bound 4.9 %) while the policy stays cheaper than the equal-risk RK45 on 66.4 % of the orbits (D4); Part C's marginal certificate hid 8.7 % violations at 8-20 TU and 13.1 % where the trajectory passes within 0.05 DU of the Moon.
Two things do **not** hold up, and they are the findings of this round. **(D2) In compiled wall-clock the policy does not out-run a compiled adaptive solver in total:** throughput ratio 0.885 [0.862, 0.908] against DP5(4) with a PI controller at equal risk (0.64 against DOP853), although the policy is faster on 78.2 % of the orbits (median per-orbit ratio 1.48) and 1.6 times faster at 1-3 TU: the long chaotic horizons dominate the total time (the policy needs 1,539 evaluations on average against 1,305) and the adaptive solver wins there. The RHS-count ranking and the stopwatch agree orbit by orbit (Spearman 0.99); in a careful implementation the cost per evaluation is at parity (22.4 against 23.1 ns), in Part C's kernel it was 42 % higher for the policy (26.6 against 18.7 ns). **(D5) The frozen policy does not transfer to structured orbits:** on 1,647 in-range runs of the JPL Earth-Moon periodic-orbit families its family-balanced violation rate is **70 % [68, 72]** (vertical and axial families about 100 %), because its predicted step is 0.3-0.5 dex too long there; a scale recalibrated on a second sample brings the violation rate to 3.5 % at a cost that is never below RK45's.

## 2. Confirmatory tests (Holm-Bonferroni over five tests, familywise 0.05; all one-sided; numbers from `part_d/results/confirm/confirm_tests_confirm.json`)

| Test | Hypothesis | Result | Expectation written on D_select_D | p | p (Holm) | Verdict |
|---|---|---|---|---|---|---|
| D1 ML vs best rule | ML mean log10 total cost below `rule_THR`'s (orbits met by both) | saving **0.0715 dex [0.0695, 0.0736]**, ML cheaper on 69.5 % of 11,565 paired orbits | 0.069 dex [0.065, 0.072], 68.4 % | 2.5e-04 (floor of 4,000 resamples) | 7.5e-04 | **supported** |
| D2 compiled throughput | compiled policy pipeline processes more orbits per second than compiled DP5(4)-PI at equal risk (single thread) | ratio **0.885 [0.862, 0.908]**; policy faster on 78.2 % of 12,239 orbits | 0.87 [0.82, 0.91] | 1.0 | 1.0 | **not supported** |
| D3 replication | share cheaper than RK45-eqr above 0.5 on D_replication | **71.56 % [70.8, 72.4]** (k = 8,768 of 12,253); violations 2.69 % [2.41, 3.00] | 72.0 % [70.6, 73.4] | < 1e-300 | < 1e-300 | **supported** |
| D4 conditional safety | every reported stratum below 7 % AND cheaper than RK45-eqr on > 50 % | 17 of 17 strata reported, worst stratum 3.1 % (upper bound 3.8 %), largest upper bound 4.9 %; cheaper on 66.4 % [65.7, 67.1] | all strata below 7 %, 68.0 % | 1.3e-04 | 5.0e-04 | **supported** |
| D5 out-of-distribution | family-balanced violation rate on in-range JPL runs below 10 % | **70.0 % [67.8, 72.3]** over 11 families, 1,647 runs | 70 % [68, 72] | 1.0 | 1.0 | **not supported** |

Three of five supported, two not, exactly as the dry runs on D_select_D predicted; the two negative results are the ones that bear on the practical claim. D1, D2 and D5 are bootstrap p-values (4,000 resamples; the floor is 2.5e-4), D3 and D4 are exact binomial (D3's underflows to 0). The decisions are not close: the supported tests have Holm-adjusted p below 1e-3, the other two have p = 1.

## 3. D1: is the ML needed? (D_confirm_D, 4,000 calibration rows, 12,239 evaluation rows; every arm certified by the same end-to-end rule and charged its own measured overhead)

| arm | coefficients | overhead (RHS-eq.) | certified scale | violations on evaluation [95 % CP] | median total cost | ML saving vs this arm, dex [95 % CI] (orbits met by both) | ML cheaper |
|---|---|---|---|---|---|---|---|
| ML: 25 trees (Part C `sund`) | 25 trees x 25 features | 16.09 | 0.800 | 2.59 % [2.32, 2.89] | 256 | - | - |
| linear model, 33 engineered features | 34 | 11.39 | 0.951 | 4.73 % [4.36, 5.12] | 263 | -0.017 [-0.019, -0.016] (n = 11,569) | 31.7 % |
| rule T,H,R (the D1 test arm) | 4 | 7.32 | 0.872 | 3.59 % [3.26, 3.93] | 323 | +0.072 [+0.069, +0.074] (n = 11,565) | 69.5 % |
| rule T,H,tau | 4 | 7.43 | 0.734 | 4.06 % [3.72, 4.43] | 325 | +0.077 [+0.075, +0.080] (n = 11,518) | 70.2 % |
| rule T,H,tau,dC | 5 | 7.28 | 0.872 | 3.52 % [3.20, 3.86] | 327 | +0.078 [+0.076, +0.081] (n = 11,579) | 70.3 % |
| rule T,tau | 3 | 7.26 | 0.872 | 4.00 % [3.66, 4.37] | 339 | +0.087 [+0.085, +0.090] (n = 11,509) | 73.2 % |
| rule T,H | 3 | 2.07 | 0.872 | 3.83 % [3.50, 4.19] | 334 | +0.090 [+0.088, +0.093] (n = 11,557) | 72.3 % |
| rule T (horizon only) | 2 | 2.05 | 0.673 | 2.45 % [2.18, 2.74] | 394 | +0.163 [+0.160, +0.166] (n = 11,679) | 83.3 % |

The overheads are single-call costs measured in one session (16.09 for the 25 trees against 7.3 for the rules with the periapsis kernel and 2.0 for the rules without; this machine's RHS takes 18.3 ns against Part C's 25-28 ns, so the same model costs more RHS-equivalents than Part C's pinned 15.04). Reading:
* The ML policy beats the best 4-coefficient rule by 0.072 dex; the horizon-only rule costs 0.163 dex more; the plan's formula (`rule_THP`, with the osculating timescale) is 0.077 dex behind. A rule gets **most of the way from a single global step to the ML policy** (the global Sundman step is 0.274 dex above the ML policy, section 5; the best rule is 0.072 above it), so 'the physics did the work' is the right headline for about three quarters of the saving.
* **The tree structure is not what buys the last quarter.** The linear model on the 33 engineered features comes out 0.017 dex CHEAPER than the trees on the orbits both meet, at a higher realized risk (4.7 % against 2.6 %; scale 0.951 against 0.800 on a certification grid whose steps are 2^(1/8), i.e. 0.038 dex in ds). The dry run had it 0.014 dex behind; the sign changed between datasets, so the honest reading is a tie within the resolution of the safety-scale grid. What both use is the 13 analytic periapsis features.
* Against the adaptive solvers (same ledgers): the ML policy is cheaper than equal-risk RK45 on 70.2 %, equal-risk DOP853 49.3 %, certified-tolerance RK45 78.6 %; `rule_THR` 53.0 %, 34.5 % and 64.7 %. **A rule is cheaper than equal-risk RK45 on a bare majority (53 %) and loses to DOP853 on two thirds of the orbits; the ML policy is what takes the claim from 'a bare majority' to 'about 70 %'.**

## 4. D2: does the RHS-count conclusion survive in compiled wall-clock time? (single thread, 12,239 evaluation orbits, Apple M5, numba 0.67; CPU 94.5 % idle at the start)

Both arms are Numba kernels with the same right-hand-side arithmetic, the same collision / ejection / non-finite checks and a compile-time-constant right-hand side (no dispatch). The adaptive solvers' tolerances are recalibrated for the compiled solvers on the same 4,000 rows (certified rtol 1.97e-7 for all three, the grid value Part C's SciPy RK45 certified; equal-risk rtols 3.4e-7 DP5(4)-PI, 4.3e-7 DP5(4)-SciPy-controller, 4.4e-7 DOP853). The compiled DP5(4) with SciPy's controller reproduces `solve_ivp`'s `nfev` exactly (tests).

| comparator (compiled, equal risk unless stated) | throughput ratio [95 % CI] | policy faster on | median per-orbit ratio | policy cheaper by RHS count | ns per RHS: policy / comparator |
|---|---|---|---|---|---|
| **DP5(4), PI controller (primary)** | 0.885 [0.862, 0.908] | 78.2 % | 1.48 | 75.1 % | 22.4 / 23.1 |
| DP5(4), SciPy's controller | 0.831 [0.810, 0.852] | 72.7 % | 1.34 | 71.5 % | 22.4 / 22.5 |
| DOP853 | 0.642 [0.625, 0.658] | 43.9 % | 0.93 | 50.0 % | 22.4 / 21.1 |
| DP5(4)-PI at the certified rtol | 0.973 [0.948, 0.998] | 83.8 % | 1.62 | 81.2 % | 22.4 / 23.0 |
| DP5(4)-PI, kernels with Part C / SciPy arithmetic (variant R) | 0.595 [0.580, 0.611] | 50.2 % | 1.00 | 75.1 % | 26.6 / 18.7 |

Per-orbit time is the minimum over three repeats of an inner loop of about 0.5 ms; throughput ratio = (total comparator time) / (total policy time) over the same orbits, bootstrapped over orbits. By subset, against DP5(4)-PI:

| subset | n | throughput ratio | policy faster on |
|---|---|---|---|
| horizon 1-3 | 4,660 | 1.61 | 99 % |
| horizon 3-8 | 4,025 | 1.02 | 89 % |
| horizon 8-20 | 3,554 | 0.75 | 38 % |
| Hill regime I | 2,062 | 0.82 | 66 % |
| Hill regime II | 406 | 0.95 | 70 % |
| Hill regime III | 2,929 | 0.96 | 69 % |
| Hill regime IV | 1,559 | 1.21 | 89 % |
| Hill regime V | 5,283 | 1.13 | 85 % |

Batches of orbits drawn from the evaluation set (all threads = 10; chunked scheduling, chunk 16; 7 repeats; comparator-over-policy ratio, median and worst repeat; below 1 means the policy is slower):

| batch size | threads | DP5(4)-PI / policy (median, worst repeat) | DP5(4)-SciPy-controller | DOP853 |
|---|---|---|---|---|
| 1,000 | 1 | 0.91 (0.90) | 0.85 | 0.66 |
| 1,000 | 10 | 0.82 (0.79) | 0.76 | 0.68 |
| 10,000 | 1 | 0.89 (0.78) | 0.87 | 0.65 |
| 10,000 | 10 | 0.82 (0.57) | 0.71 | 0.70 |
| 100,000 | 1 | 0.88 (0.87) | 0.83 | 0.64 |
| 100,000 | 10 | 0.91 (0.73) | 0.81 | 0.79 |

No batch size meets the pre-set criterion. Modelled lane efficiency of a lock-step batch from the measured step counts (no GPU or SIMD kernel was written), at 2 / 4 / 8 / 32 lanes: random order 0.55 / 0.31 / 0.18 / 0.06 for the policy and 0.57 / 0.33 / 0.20 / 0.08 for the adaptive solver; sorted by work 1.00 / 0.99 / 0.98 / 0.92 and 1.00 / 0.99 / 0.98 / 0.95. Reading:
* **The policy wins where Part C said it would and loses where it said it would, now in seconds**: 1.6 times the throughput of the compiled DP5(4) at 1-3 TU (faster on 99 % of the orbits), parity at 3-8 TU, 0.75 at 8-20 TU (faster on 38 %); by Hill regime from 0.82 (I) to 1.21 (IV). A Monte Carlo dispersion study at horizons up to about 8 TU would gain; one over the whole 1-20 TU range would not.
* **Why the RHS count flattered the policy in Part C's kernel**: a Sundman step evaluates the time scaling `g` (two more square roots) in every stage, and Part C's landing steps allocate arrays: 26.6 ns per RHS-equivalent against 18.7 for the compiled DP5(4), throughput ratio 0.595. A careful implementation (one reciprocal per body, no allocation, both arms) brings the per-evaluation costs to parity (22.4 against 23.1 ns); the remaining deficit is the long-horizon tail, where the policy needs more evaluations (1,539 against 1,305 on average) and the totals live. Both arms were optimized; the first two attempts at this comparison were not symmetric and are in the log.
* Lock-step execution does not rescue a fixed step: with random batching both keep about 0.55 / 0.3 / 0.2 of their lane-steps busy at 2 / 4 / 8 lanes, and both need sorting by work to approach 1.

## 5. D3: replication of the Part C headline (D_replication, Part C's own `apply_policy`, frozen policies, 4,000 calibration rows, 12,253 evaluation rows)

The scale certified on the new calibration rows is 0.800 (Part C: 0.872); the policy violates in 2.69 % of the evaluation orbits and is cheaper than: RK45-eqr 71.6 %, DOP853-eqr 51.3 %, RK45 at the certified rtol 79.1 %, DOP853 certified 54.1 %, the per-orbit-oracle RK45 5.5 % and DOP853 3.6 % (Part C: 71.4, 50.6, 82.4, 58.6, 8.7, 6.1 %). Same direction, same size; the headline did not come from tuning on one data stream.
The rest of Part C's family on the new data (reported, not in the Holm family): **C1** prediction 0.031 dex [0.021, 0.04], supported (p = 2.5e-04); **C2** physical-time policy 26.9 % cheaper than RK45-eqr, not supported (as in Part C); **C3** (a) the oracle fixed-ds Sundman run is cheaper than RK45-spec on 50.56 % [49.7, 51.4] (p = 0.11: **not above 50 %** on this dataset, the coin flip Part C flagged at 51.1 %) and (b) Spearman 0.221 against 0.905 (p = 5.0e-04), so C3 as a conjunction would NOT have been supported here while its mechanism (b) is; **C5** ML against one global step 0.274 dex [0.27, 0.278], ML cheaper on 85.8 %, supported.

## 6. D4: conditional safety (scales certified per free stratum on D_confirm_D, all 16,239 usable rows; evaluated on D_replication, 16,253 rows; none of them calibrated)

Strata are `regime group | Moon class | horizon band` from quantities available before any integration; cells with fewer than 400 calibration rows are merged by Part B's fixed rule, which gives 17 strata, all with at least 411 calibration rows; every stratum had a certified scale, so none abstained (abstention to RK45 at the certified rtol is implemented and was not needed). The equal-risk RK45 of each policy is matched to that policy's own calibration violation rate (2.47 % stratified, 4.43 % one scale).

| stratum (regime group, Moon class, horizon band) | scale | n | stratified policy [95 % CP] | one scale for all | RK45, equal-risk rtol | policy cheaper than RK45-eqr |
|---|---|---|---|---|---|---|
| II-III|far|1-3 | 0.872 | 812 | 1.7 % [0.9, 2.9] | 1.7 % | 0.0 % | 98 % |
| II-III|far|3-8 | 0.872 | 601 | 1.5 % [0.7, 2.8] | 1.5 % | 0.0 % | 94 % |
| II-III|far|8-20 | 0.673 | 485 | 2.3 % [1.1, 4.0] | 6.6 % | 1.9 % | 8 % |
| II-III|peri0.05-0.17|h* | 0.734 | 983 | 2.3 % [1.5, 3.5] | 5.4 % | 1.1 % | 48 % |
| II-III|peri<0.05|1-3 | 0.800 | 565 | 2.1 % [1.1, 3.7] | 3.9 % | 0.0 % | 88 % |
| II-III|peri<0.05|3-8 | 0.617 | 507 | 0.8 % [0.2, 2.0] | 8.1 % | 0.6 % | 23 % |
| II-III|peri<0.05|8-20 | 0.617 | 447 | 2.9 % [1.6, 4.9] | 8.9 % | 7.2 % | 0 % |
| IV-V|far|1-3 | 0.951 | 3,142 | 3.1 % [2.5, 3.8] | 0.8 % | 0.0 % | 97 % |
| IV-V|far|3-8 | 0.872 | 2,753 | 2.4 % [1.9, 3.0] | 2.4 % | 0.0 % | 96 % |
| IV-V|far|8-20 | 0.617 | 2,516 | 3.1 % [2.4, 3.8] | 10.4 % | 0.8 % | 20 % |
| IV-V|peri0.05-0.17|h* | 0.734 | 456 | 2.6 % [1.4, 4.6] | 7.9 % | 0.7 % | 58 % |
| I|far|1-3 | 0.872 | 641 | 1.9 % [1.0, 3.2] | 1.9 % | 0.2 % | 98 % |
| I|far|3-8 | 0.872 | 504 | 2.0 % [1.0, 3.6] | 2.0 % | 2.8 % | 93 % |
| I|far|8-20 | 0.800 | 402 | 1.5 % [0.5, 3.2] | 3.5 % | 24.1 % | 3 % |
| I|peri<0.05|1-3 | 0.800 | 378 | 0.0 % [0.0, 1.0] | 1.9 % | 0.0 % | 97 % |
| I|peri<0.05|h* | 0.800 | 628 | 1.9 % [1.0, 3.3] | 6.7 % | 30.4 % | 24 % |
| pooled-small | 0.734 | 433 | 2.1 % [1.0, 3.9] | 8.3 % | 2.1 % | 52 % |

The stratified policy's overall violation rate is 2.39 % against 4.43 % for one scale for all, at a price: the equal-risk RK45 of the safer policy is tighter, and the policy is cheaper than it on 66.4 % of the orbits (the one-scale policy: 70.1 %). **Where the risk lands** (not free: these bands need the trajectory):

| closest approach actually made | n | stratified policy [95 % CP] | one scale for all | RK45, equal-risk rtol | DOP853, equal-risk rtol |
|---|---|---|---|---|---|
| min r2 <0.05 DU | 2,914 | 5.1 % [4.3, 5.9] | 13.1 % [11.9, 14.4] | 8.6 % | 9.6 % |
| min r2 0.05-0.17 DU | 2,013 | 4.5 % [3.7, 5.5] | 8.7 % [7.5, 10.1] | 0.5 % | 0.5 % |
| min r2 0.17-0.8 DU | 6,710 | 1.6 % [1.3, 1.9] | 2.2 % [1.8, 2.6] | 1.9 % | 2.2 % |
| min r2 >0.8 DU | 4,616 | 0.9 % [0.7, 1.2] | 0.4 % [0.2, 0.6] | 0.1 % | 0.1 % |

| subset | stratified policy | one scale for all | RK45, equal-risk rtol |
|---|---|---|---|
| horizon 1-3 TU | 2.3 % | 1.9 % | 0.0 % |
| horizon 3-8 TU | 2.1 % | 3.5 % | 1.0 % |
| horizon 8-20 TU | 2.9 % | 8.7 % | 7.0 % |
| Hill regime I | 1.5 % | 3.2 % | 11.4 % |
| Hill regime II | 0.5 % | 1.8 % | 0.5 % |
| Hill regime III | 2.1 % | 5.4 % | 1.4 % |
| Hill regime IV | 2.7 % | 5.2 % | 0.2 % |
| Hill regime V | 2.9 % | 4.3 % | 0.3 % |

* The conditional certificate works on the strata it is built on and **improves the risk where the trajectory passes the Moon (13.1 % to 5.1 %) and at long horizons (8.7 % to 2.9 %)**, but the a-posteriori close-approach bands stay at 4.5-5.1 %: the osculating periapsis sees only part of the close passes. The one-scale policy's worst free stratum is 10.4 % (IV-V|far|8-20, upper bound 11.7 %): the marginal 4.4 % of Part C was true on average and false where it mattered.
* **The equal-risk comparator is equal only on average.** RK45 at the policy's marginal risk violates in 11.4 % of regime-I orbits and in 24 % and 30 % of two regime-I strata (long horizons, and close approaches): the plan's point that 'equal risk is not equal where the risk lands' holds for the comparator too.
* **Exploratory D4b: the fair conditional comparator.** The adaptive solver certified in the same strata by the same rule (fixed-sequence, delta / 17; the compiled DP5(4) reproduces SciPy's RK45 `nfev` exactly): its overall violation rate is 1.46 % (the policy's 2.39 %) and the stratified policy is cheaper than it on **56.2 % [55.5, 57.0]** of the orbits (75 % at 1-3 TU, 71 % at 3-8 TU, 15 % at 8-20 TU; regime I 83 %). Against the compiled DOP853 certified per stratum (violation rate 2.45 %) the policy is cheaper on only **23.5 %** (median cost 194 against the policy's 271). So the claim that survives a conditional comparison is: cheaper than conditionally certified RK45 on a modest majority, not cheaper than conditionally certified DOP853.

## 7. D5: the frozen policy on the JPL periodic-orbit families (confirm sample of 1,071 members; 942 kept; 1,647 in-range runs with a trustworthy reference; nothing calibrated on them)

JPL's mass ratio is 1.215058560962404e-2 (config.MU 1.215058535e-2, relative difference 2.1e-8) and its length unit 389,703 km (the project's 384,400 km): the states are integrated at JPL's mass ratio (the one-period closure error is below 1e-8 DU on every kept member, which validates the units) and the tolerance is the project's 100 km (101.4 km in JPL units). 129 of 1,071 sampled members were dropped before any policy run: 103 whose reference stopped inside a body before one period (halo L1 76 of 102 samples, the large members that impact the Moon; Lyapunov L2 14, resonant 7, halo L2 6) and 26 that did not close to 1e-8 (resonant 20, DRO 4, Lyapunov L2 2). 102 of 1,749 in-range runs have an untrusted reference (two references disagreeing by more than 5 km: unstable members at several periods) and are excluded.

| family | in-range runs (orbits) | policy violations [95 % CP] | RK45 certified rtol | RK45 equal-risk rtol | DOP853 equal-risk rtol | median log10 (policy ds / oracle ds) |
|---|---|---|---|---|---|---|
| axial_L1 | 102 (51) | 100 % [96, 100] | 33 % | 50 % | 1 % | +0.37 |
| axial_L2 | 102 (51) | 100 % [96, 100] | 50 % | 50 % | 2 % | +0.50 |
| vertical_L2 | 102 (51) | 100 % [96, 100] | 34 % | 45 % | 32 % | +0.49 |
| vertical_L1 | 102 (51) | 96 % [90, 99] | 30 % | 41 % | 17 % | +0.53 |
| lyapunov_L2 | 70 (35) | 84 % [74, 92] | 44 % | 66 % | 36 % | +0.31 |
| lyapunov_L1 | 102 (51) | 78 % [69, 86] | 45 % | 56 % | 39 % | +0.34 |
| dro | 97 (45) | 47 % [37, 58] | 13 % | 36 % | 5 % | +0.06 |
| halo_L1 | 74 (13) | 46 % [34, 58] | 27 % | 41 % | 30 % | +0.01 |
| butterfly | 218 (51) | 44 % [37, 51] | 18 % | 28 % | 29 % | +0.05 |
| resonant | 412 (330) | 41 % [36, 46] | 8 % | 26 % | 15 % | +0.01 |
| halo_L2 | 266 (48) | 33 % [27, 39] | 26 % | 31 % | 23 % | -0.07 |

By stability index: 1-10: 41 % (RK45-eqr 28 %, n = 494); 10-100: 64 % (RK45-eqr 44 %, n = 411); >100: 83 % (RK45-eqr 46 %, n = 500); stable (<=1.0001): 39 % (RK45-eqr 25 %, n = 242). By number of periods: 1 period: 47 % (n = 882); 2 periods: 77 % (n = 613); 5 periods: 55 % (n = 109); 10 periods: 72 % (n = 36); 20 periods: 14 % (n = 7). The extrapolated horizons (above 20 TU) are in `part_d/results/confirm/d5_table_extrapolated_by_group.csv`: the policy and RK45 both violate in 32-100 % of those runs.
* The failure is a **prediction** failure, not an integration artefact: against the oracle Sundman step of the same run (the largest ds, on the Part C label grid, that meets the tolerance) the policy's raw step is too long by 0.15 dex in the median and by 0.3-0.5 dex on the vertical, axial and Lyapunov families, and about right (within 0.07 dex) on halos, butterflies, resonant orbits and DROs, whose violations come from the unstable members (the policy has no feature for instability). On a vertical L2 member (stability index 1.0, one period) the Sundman RK4 converges at fourth order to 0.0 km as ds shrinks: the integrator is right and the step is wrong.
* **Recalibrating does not rescue it economically**: a scale certified on the other (dry) sample, 0.183 instead of 0.872, gives a violation rate of 3.5 % (family-balanced 4.2 %) on this sample at a median total cost of 3,299 RHS evaluations, and the policy is then cheaper than RK45 on none of the runs.
* The comparators are not safe here either (the orbits are unstable: a stability index of 100 amplifies a local error a hundredfold per period): RK45 at Part C's certified rtol violates in 8-50 % of the family runs and at the equal-risk rtol in 26-66 %, DOP853 at its equal-risk rtol in 1-39 %. The policy violates more often than RK45 at the equal-risk rtol in every one of the 11 families: by 40-55 points on the vertical and axial families, by 5 points or less on the halos.

## 8. Locked evaluation (D_lock_D, 4,074 usable orbits; scales and tolerances from D_confirm_D's D1; one run)

| policy | violations [95 % CP] | median total cost | cheaper than RK45-cal / DOP853-cal | RK45-eqr / DOP853-eqr |
|---|---|---|---|---|
| ML (`sund`) | 2.45 % [2.00, 2.98] | 260 | 78.3 % / 54.0 % | 70.3 % / 48.9 % |
| best rule (`rule_THR`) | 4.32 % [3.72, 4.99] | 323 | 64.3 % / 40.2 % | 52.3 % / 34.2 % |

## 9. Exploratory results (no Holm family; the expectation of each was written before it was run)

**D6, which quantities carry the information** (60,000 training rows, 200 trees of depth 6, test on D_confirm_D; label standard deviation 0.33 dex for the Sundman target against 0.91 for the physical-time target; full-model median error 0.036 and 0.056 dex):

Sundman-time target (log10 ds*):

| group | drop-group refit: increase of the test MSE [95 % CI] | permutation: increase of the test MSE |
|---|---|---|
| state (6 col.) | +0.0008 [+0.0007, +0.0009] | 0.0021 |
| proximity (3 col.) | -0.0001 [-0.0001, -0.0000] | 0.0016 |
| jacobi (2 col.) | +0.0001 [+0.0000, +0.0002] | 0.0014 |
| periapsis (13 col.) | +0.0048 [+0.0045, +0.0050] | 0.0989 |
| horizon (1 col.) | +0.0086 [+0.0082, +0.0091] | 0.0852 |
| FTLE (offline; refit WITH the other five) | +0.0023 [+0.0020, +0.0025] | 0.0844 |

Physical-time target (log10 dt*):

| group | drop-group refit: increase of the test MSE [95 % CI] | permutation: increase of the test MSE |
|---|---|---|
| state (6 col.) | +0.0173 [+0.0155, +0.0192] | 0.0455 |
| proximity (3 col.) | +0.0005 [-0.0002, +0.0012] | 0.0434 |
| jacobi (2 col.) | +0.0007 [-0.0001, +0.0015] | 0.0072 |
| periapsis (13 col.) | +0.0443 [+0.0413, +0.0473] | 1.2002 |
| horizon (1 col.) | +0.0090 [+0.0079, +0.0102] | 0.0665 |
| FTLE (offline; refit WITH the other five) | +0.0078 [+0.0059, +0.0099] | 0.3802 |

The time transformation removes most of the dependence on the geometry the project started from: for the Sundman target the periapsis group is 9 times less important by refit and 12 times by permutation than for the physical-time target, the proximity group {r1, r2, r1/r2} and the Jacobi constant contribute nothing a refit can detect (they are redundant given the horizon and the periapsis features), and the horizon is the most informative group by refit (by permutation the periapsis is marginally ahead). FTLE carries real but small information beyond the deployable groups (a refit loses 0.002 of 0.007 mean squared error), costs an integration to compute and cannot be a pre-simulation feature. The hypothesis the project started from, that FTLE, proximity and the Jacobi constant govern the difficulty, is **not supported as a statement about what a pre-simulation predictor needs**: the horizon and the analytic two-body periapsis information carry it.
**D7, FTLE and the failures** (surface of section y = 0, z = vz = 0, vy > 0, a 110 x 110 grid at C = 3.19 / 3.10 / 2.95, T = 8 TU; frozen Part C policy and scale; the violation rate on these planar sections is 26-30 %, itself an out-of-distribution signal): the FTLE of a point separates the violations with an AUC of 0.93 / 0.92 / 0.93; the lowest four FTLE deciles are essentially violation-free and the top decile violates in 84-97 % (`part_d/figures/d7.png`). The violations sit in the near-Earth block and along the ridges, where the orbit is both fast and sensitive: FTLE explains where the policy fails after the fact.
**D12, is the reference the problem** (the 1 % highest lambda*T orbits of D_confirm_D, lambda*T >= 10.8, 163 orbits): a DOP853 at 1e-14 with step <= 0.002 and an LSODA at 1e-13 agree with the standard reference to 0.007 and 0.23 km at most; **no orbit is 'unpredictable'**, no label flips, and the policy violates on **25.2 %** of these orbits against 2.6 % overall: the policy's risk concentrates in the high-lambda*T tail and it is the policy's error, not the reference's. (SciPy raised the requested 1e-14 tolerance to machine epsilon.)
**D10, the Sundman design** (1,500 orbits of D_replication; the oracle fixed-ds Sundman run against RK45 at the per-orbit oracle tolerance of Part C's replication ledger, n = 1,136): cheaper on 47.5 % / 51.6 % / 53.2 % / 54.0 % for tau_c = 1 / 2 / 4 / 8 (95 % intervals about +-3 points; the Part C choice tau_c = 2 sits at the knee). Non-monotone labels: 20 of 500 orbits (4.0 % [2.5, 6.1]; Part C's pilot 4.5 %), **concentrated at short horizons (8 % at 1-3 TU, none at 8-20 TU), far from the Moon (7 % beyond 0.8 DU against 1 % below 0.05 DU) and at low lambda*T (median 2.5 against 3.9)**, not near collisions. The landing-step suspect is weak: the number of physical-time landing steps changes by 0.55 at a pass/fail flip against 0.49 between any two neighbouring steps (+13 %), the landing share of the steps 11.2 % against 8.5 %: a flip is somewhat more likely where the landing logic is busy, which does not explain most of them. A smooth alternative to `g` was not tried.
**Leapfrog symplecticity** (finite-difference Phi^T J Phi - J in the canonical variables of the rotating frame, 100 states): the leapfrog step's median defect is 1.1e-10 at every step size (the finite-difference floor, epsilon / h), RK4's is 5e-05 at dt = 0.2 and 4e-08 at 0.05 (at 0.01 it too reaches the floor): the leapfrog is symplectic, not merely symmetric.
**IAS15** (REBOUND 5.2.2 in the inertial frame, Pham-Rein-Spiegel adaptive mode, 300 evaluation orbits, the loosest epsilon that meets 100 km per orbit, searched up to epsilon = 0.1): the oracle epsilon is the edge of the search (median -1) and a run takes a median of 217 us (11 steps) against 6.4 us for the compiled policy and 9.2 us for the compiled DP5(4) on the same orbits: 29 times slower in total, never faster on an orbit. The IAS15 time includes REBOUND's per-call overhead from Python, so the factor is an upper bound on the algorithmic gap, and the comparison is wall time only (REBOUND exposes no force-evaluation counter). Expected to lose at 100 km and it does; it is a baseline for the objection, not a competitor.

## 10. What the contribution is now, and what it is not

Supported: a fixed-step RK4 in Sundman time, with one fictitious step predicted per orbit by a small model that costs about 16 RHS-equivalents, and certified end to end, is **cheaper in RHS evaluations than RK45 at equal marginal risk on 72 % of the orbits of an independent dataset** (D3) and **than RK45 certified in the same strata on 56 %** (D4b), and the certificate can be made conditional on free quantities at a modest cost (D4). What carries the saving is the horizon and the analytic two-body periapsis information (D6), most of which a four-coefficient rule recovers (D1).
Not supported, and to be stated with the claim: (i) **not cheaper than DOP853** certified per stratum (24 %) or at equal marginal risk (49 %); (ii) **not faster in compiled wall-clock over a 1-20 TU population** (0.89 against DP5(4)-PI; it is faster at short horizons, 1.6 at 1-3 TU); (iii) **not safe on structured orbits it was not trained on** (70 % violations on periodic-orbit families); (iv) its risk concentrates in the high-lambda*T tail (25 % there, D12) and the conditional certificate still leaves about 5 % at close lunar passes. Wording discipline: 'cheaper than RK45 at equal marginal risk / certified tolerance', never 'faster than adaptive solvers'; 'integrator error inside the CR3BP', never 'accurate trajectories'; the CR3BP against the CAPSTONE ephemeris differs by 2,000-50,000 km against a 100 km tolerance.

## 11. Limits, stated plainly

1. One system, one tolerance, one horizon range: D8 (10 km, 1,000 km, 50 TU) was not run, so how the D2 and D3 conclusions move with the tolerance is unknown.
2. D1 compares against rules fitted by linear quantile regression on the same data, with a quantile and a feature subset chosen on D_select_D; a nonlinear hand-built rule was not tried, and the linear model on 33 features is a stronger 'rule' than the six pre-specified ones and ties the trees.
3. D2 is one machine (Apple M5, 10 cores, a desktop in use) and one compiler; the multi-core ratios (0.82-0.95) are noisier than the single-thread number, and no GPU was measured. The 1-minute load average was 1.499 at the start of the step (the guard had been relaxed to 1.6 beforehand because the figure hovered at 1.50 with the CPU 94.5 % idle; the dated deviation records this) and 4.2 at the end, after the 10-thread batch phase.
4. The D4 stratification uses osculating-periapsis quantities that miss part of the close passes (about 5 % where it matters); abstention was implemented and never triggered, so its cost is unmeasured.
5. D5 is a sample of 50 members per stream from a catalogue; the halo-L1 stream lost 76 of 102 samples to impact; the unstable families at several periods are partly excluded by the reference check, so the reported violation rates are for the part of the catalogue where a 100 km answer exists.
6. The seal does not require a clean git tree (nothing was committed); the sealed tree is identified by the hashes of every file.

## 12. Plan items: done / not done

Done: T0.1 citation audit (`part_d/references.bib`: 19 entries, each tagged with what was opened; the review's wrong entries are replaced, items after the knowledge cutoff flagged unverifiable); T0.2 (`part_d/NARRATIVE.md`); T0.3 (fresh pre-registration, seeds, seal); D1-D5; D6, D7, D10 (without the smooth-g alternative), D12; leapfrog symplecticity; IAS15 baseline; the additional D4b.
Not done: D8 (tolerance and horizon dial), D9 (path-error label), D11 (cross-mu with dimensionless features), KS-regularized RK4, E12 (2D control), a GPU / lock-step SIMD kernel, the smooth alternative to g.

## 13. Reproduce

```bash
PY=/Users/arnavdeopura/miniforge3/envs/orbit-ml/bin/python
./gen_part_d.sh                                           # seeds 52, 53, 54, 55 (about 35 minutes)
$PY fetch_jpl_d.py                                        # the JPL families, raw and hashed
$PY run_part_d_3d.py refs                                 # cached reference final positions
$PY run_part_d_3d.py overheads                            # idle machine
$PY run_part_d_3d.py select_rules                         # then freeze_rule --rule <name> --q <q> --out_name <name> for each rule, and freeze_lin33
python3 part_d/make_frozen_list.py && $PY freeze_d.py seal
$PY run_confirm_d_3d.py run d2 --target confirm           # alone, on an idle machine
for s in d1 d5 d4 d3_apply_sund d3_apply_phys d3_apply_global d3_c1_resid lock; do $PY run_confirm_d_3d.py run $s --target confirm; done
$PY run_confirm_d_3d.py tests --target confirm
$PY run_explore_d_3d.py d6 ; ... d7 ; ... d10 ; ... d12 ; ... d4b ; ... symplectic ; ... ias15_prep   # then python3 ias15_baseline_d.py --inp <csv> --out <csv>, then ias15_combine
$PY make_figures_d.py all
$PY tests/test_part_d.py && $PY tests/test_part_c.py && $PY tests/test_phase0.py && $PY tests/test_failure_study.py
```
The datasets (about 330 MB with the JPL responses) and the run folders are not in git (hashes in `part_d/DATA_HASHES_D.json` and `FREEZE_D.json`); the frozen rules (a few KB) are in `part_d/frozen/`.
