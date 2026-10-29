# Pre-registration, Part D: is the ML needed, does it matter in wall-clock, is it safe where it counts, does it generalize

Status: DRAFT until `freeze_d.py seal` writes `FREEZE_D.json`; after that, everything above "## Deviations" is frozen and later changes are dated entries under "Deviations". Parts A (`PREREGISTRATION.md`, `FREEZE.json`), B
(`part_b/PREREGISTRATION_B.md`, `FREEZE_B.json`) and C (`part_c/PREREGISTRATION_C.md`, `FREEZE_C.json`) are separate, sealed studies. Part C is the base commit (`4586803`); its policies are used here FROZEN and unchanged (SHA-256 in
`part_d/FROZEN_FILES.json`). The plan this implements is `CR3BP_Improvement_Plan_Part_D_1.md` (kept outside the repository); `part_d/PART_D_LOG.md` is the dated record of every decision and every dry-run result that led here.
Where this document differs from the plan, section 8 says so.

Seal mechanics (`freeze_d.py`): (a) SHA-256 of every `*.py` file, (b) SHA-256 of this document above "## Deviations", (c) SHA-256 of every file in `part_d/FROZEN_FILES.json` (the frozen policies, the datasets, the JPL responses), (d) the git HEAD and the
list of paths that differ from it (nothing is committed on the user's behalf, so the commit alone does not identify the sealed tree; the file hashes do). `run_confirm_d_3d.py` refuses to start a step if the code, this text or a frozen file differs from the
seal, and runs each step once per target (`confirm_log_d.jsonl`). The timing step additionally refuses to start on a busy machine.

## 1. Questions

Part C found a cheaper-than-RK45 policy. Part D asks what that result is worth and where it breaks, in five questions that are each answered by one pre-specified test:
D1 is the ML needed, or does a rule with a handful of coefficients on the same free quantities do the same work (the Sundman time transformation is a century old; the ML claim is about the step prediction); D2 does the RHS-count conclusion survive in compiled
wall-clock time against equally compiled adaptive solvers; D3 does the headline (C4) replicate on data nobody has looked at; D4 can the safety certificate be made conditional where the risk concentrates (close lunar approaches) at a cost that still beats the
adaptive solver; D5 does the frozen policy work on structured orbits (halos, NRHOs, DROs, Lyapunov, vertical, axial, butterfly, resonant families from the JPL catalog) it was never trained on.
Everything else (Tier 2, Tier 3 of the plan) is exploratory and listed in section 6.
The dry runs on D_select_D (`part_d/PART_D_LOG.md` section 4) say in advance which way each test is expected to go: D1, D3 and D4 supported, D2 and D5 not. The document is written to make the two negative results as testable as the positive ones.

## 2. Data (new seeds; nothing from Parts A-C is reused as a test set)

Generation is Part C's (`gen_part_d.sh`: pipeline v5, `accuracy_monotone`, tau_c = 2, 100 km, Earth-Moon mu = 0.01215058535, horizons log-uniform 1-20 TU, reference DOP853 at 1e-13), only the seeds are new.

| Set | Seed | Candidate orbits (usable) | Use |
|---|---|---|---|
| D_train, D_train_enriched | 41, 141 | Part C's (81,111 + enriched, weights as in Part C) | the ONLY fitting data: the Part C policies, the rules, the linear arm, D6 |
| D_select_D | 52 | 10,000 (8,101) | every choice of Part D (rule selection, strata design check, the dry runs and the written expectations) |
| D_confirm_D | 53 | 20,000 (16,239) | D1, D2 (standard split: 4,000 randomly drawn usable rows with seed 42 calibrate, every other usable row is evaluated), D4 (all usable rows calibrate the strata scales), D6 test rows, D12 |
| D_lock_D | 54 | 5,000 (4,074) | one run of the final policies with the scales and tolerances calibrated in D1 (a number, not a test) |
| D_replication | 55 | 20,000 (16,253) | D3 (standard split as above), D4 (all usable rows are the evaluation set; no scale was calibrated on them) |
| JPL periodic orbits | - | 21 streams (halo L1/L2 N/S, Lyapunov, vertical, axial L1/L2, DRO, butterfly N/S, resonant 12 21 13 31 23 32 34 43), 40 MB | D5; raw responses with SHA-256 in `part_d/DATA_HASHES_D.json` |

Seed 51 (the plan skeleton's 'train extension') is not generated: the ML policy and the rules are fitted on the same Part C D_train, which is what makes D1 a comparison of model classes at equal data. Usable = a reference that reaches the horizon; the reference-divergent
('doomed') orbits are not in any test (Part C keeps their ledger; Part D does not repeat it).

## 3. Frozen inputs

* **Part C policies** (`part_c/frozen/<name>/`, unchanged): `sund` (25 trees of depth 3, lr 0.2, q = 0.2, 25 features, Sundman-time RK4 with tau_c = 2), `phys`, `global`, `partb_refit`. Their pinned overheads (15.04, 11.74, 0, -) are used in D3, D4 and D5.
* **Physics rules** (`part_d/frozen/<rule>/`): a rule is `log10 ds = b + w . f` with a subset of the features `lT = log10 T`, `reg` (Hill regime I..V as 1..5), `tau = osc_logtau_hat`, `lrp` (log10 of the smallest osculating periapsis radius over the
  two bodies, the current distance where no periapsis is reached before T), `dC = C - C_L1`, all free before any integration (`rules_d.py`). Pre-specified rules: `rule_T` {lT}, `rule_TH` {lT, reg}, `rule_TP` {lT, tau}, `rule_THP` {lT, reg, tau}
  (the plan's formula), `rule_THR` {lT, reg, lrp}, `rule_THPD` {lT, reg, tau, dC}; fitted by weighted linear quantile regression (HiGHS interior point) on the whole of D_train + D_train_enriched, the quantile of each in {0.1, 0.2, 0.3, 0.5} chosen on D_select_D.
  **The D1 test arm is the rule chosen by `run_part_d_3d.py select_rules` on D_select_D** (lowest median total cost per orbit that met the tolerance on the evaluation fold, overhead included; candidates with an evaluation-fold violation rate above 6 % excluded;
  ties within 2 % go to the rule with fewer coefficients): **`rule_THR`, q = 0.1** (`part_d/selection/rule_selection.json`). The other five rules and `lin33` (Part C's linear M0 class on the 33 engineered features, q = 0.1) are frozen as descriptive arms.
* **Overheads** (`part_d/selection/overheads_idle.json`, one session, Part C's single-call protocol, medians of 7 repeats): the model and every rule are timed in the same session against the same RHS, so D1 charges each arm the value measured there
  (sund 16.09, rule_THR 7.32, lin33 11.39, rule_THP 7.43, rule_TH 2.07, rule_T 2.05 RHS-equivalents). This machine's RHS takes 18.3 ns against Part C's 25-28 ns, so the same model costs more RHS-equivalents than Part C's pinned 15.04.
* **Certification** (unchanged from Part C): the exact fixed-sequence rule over the 28 values `0.1 * 2^(k/8)`, k = 0..26, and 1.0, tested safest first at delta = 0.1 against H0 'violation rate > alpha = 0.05' (one-sided exact binomial); a violation is a run that ends more than 100 km from
  the reference, collides, escapes, is non-finite or hits the step cap (4,000,000 steps). Adaptive comparators: SciPy RK45 and DOP853, atol = 1e-3 rtol, cost = `nfev`, `cal` = one certified rtol per method on the calibration rows (Bonferroni delta/18 over the 18-value grid),
  `eqr` = the loosest rtol whose realized violation rate on the calibration rows is at most that of the policy it is compared with (recomputed per policy).

## 4. Confirmatory tests (Holm-Bonferroni over D1-D5, familywise 0.05; all one-sided in the direction stated)

| ID | Hypothesis | Data and test | Expectation (D_select_D dry run, `part_d/results/dry`) |
|---|---|---|---|
| D1 ML versus the best physics rule | the `sund` ML policy's mean log10 total cost (overhead included) is below `rule_THR`'s, orbits that met the tolerance under both | D_confirm_D standard split; both certified on the 4,000 calibration rows; paired bootstrap over orbits (4,000), p = share of resamples with a saving <= 0 (as C5) | ML saves 0.069 dex [0.065, 0.072] (4,051 evaluation rows of D_select_D; 3,863 paired orbits; ML cheaper on 68.4 %); against the other rules 0.075-0.131 dex; `lin33` is only 0.014 dex behind the trees. **Supported** |
| D2 compiled throughput | the compiled policy pipeline (features, 25 trees, certified scale, Sundman RK4; micro-optimized kernel F) processes more orbits per second on one thread than compiled Dormand-Prince 5(4) with a PI controller (kernel F) at its equal-risk tolerance: (sum of the comparator's wall times) / (sum of the policy's) > 1 over the evaluation orbits | D_confirm_D standard split; the comparator's rtol is recalibrated for the compiled solver on the same 4,000 rows (target = the policy's calibration violation rate); per-orbit time = the minimum over 3 repeats of an inner loop of about 0.5 ms (no Python in the timed region); orbit-level bootstrap of the ratio (4,000); machine idle | throughput ratio 0.87 [0.82, 0.91] on D_select_D (the policy is faster on 80 % of the orbits, median ratio 1.49, but slower in total: its time is dominated by long chaotic horizons, by horizon band 1.62 / 1.09 / 0.72). **Not supported** |
| D3 replication | the share of orbits where `sund` (overhead included) met the tolerance AND cost less than RK45-eqr is above 0.5 | D_replication, Part C's own `apply_policy` command with the frozen policy (its scale and the comparators' tolerances are calibrated on the data's own 4,000 rows); exact one-sided binomial | 72.0 % [70.6, 73.4] of 4,101 evaluation rows of D_select_D by Part C's own command with its pinned overhead (the lean re-implementation with this session's overhead gives 71.8 %, scale and violations identical; Part C measured 71.4 % on its confirmation data, its dry run had said 69.9 %). **Supported** |
| D4 conditional safety | every reported stratum has a realized violation rate below 7 % AND the policy (abstentions and overhead included) is cheaper than the equal-risk RK45 on more than half of the orbits | strata from free quantities (section 4.1), scales certified per stratum on all usable D_confirm_D rows, evaluated on all usable D_replication rows; intersection-union test: per stratum an exact one-sided binomial against H0 'rate >= 7 %', and the C4-type binomial on the cost share; p = the largest of these | on the halves of D_select_D (7 strata, none abstains): every reported stratum below 7 % (worst 4.0 %, CP upper bound 5.8 %), 68.0 % cheaper than the equal-risk RK45, p = 7e-4. **Supported**; the a-posteriori closest-approach band min r2 < 0.05 DU stays at 7.5 % [CP upper 9.6 %] because the free strata see only part of the close passes (tabulated, not tested) |
| D5 out-of-distribution | the family-balanced violation rate of the frozen Part C policy on in-range JPL periodic-orbit runs with a trustworthy reference is below 10 % (twice the nominal alpha) | section 4.2; one-sided bootstrap (orbits resampled within family, 4,000) of the mean over families of the family violation rate, H0: the mean is at least 10 % | family-balanced violation rate 70 % [68, 72] on the dry sample (every family above 30 %, vertical and axial about 100 %). **Not supported**, p ~ 1: against the oracle Sundman step the policy's raw prediction is too large by 0.16 dex in the median over runs and by about 0.5 dex on the vertical and axial families (a recalibrated scale of 0.18 instead of 0.87 brings the violation rate to 4 % at a median total cost of about 3,300 RHS evaluations, never cheaper than RK45) |

### 4.1 D4 strata (all free before any integration)

A base cell is `regime group | moon class | horizon band`: regime group I, II-III, IV-V (Hill regime from the Jacobi constant); moon class `crash` (the physical crash flag of `doomed_3d.crash_flag_physical`: the osculating periapsis about a body is inside its collision radius and
reached before T), `peri<0.05` (osculating periapsis about the Moon below 0.05 DU, reached before T), `peri0.05-0.17`, `far`; horizon band 1-3, 3-8, 8-20 TU: 36 base cells. Cells with fewer than 400 calibration rows are merged by Part B's fixed rule
(`failure_study_3d.assign_cells`: along the horizon within (regime group, moon class); then along the moon class within the regime group; the rest 'pooled-small'), using the calibration counts only. A stratum certifies its own scale by the fixed-sequence rule at level
delta / G (G = the number of strata, Bonferroni); a stratum with fewer than 150 calibration rows or with no certified scale ABSTAINS: its orbits are run with SciPy RK45 at the rtol certified on the calibration rows (the Part C fallback), and those runs count in the
risk and the cost. The model overhead is charged on every orbit, abstained or not. A stratum is REPORTED in the hypothesis if it has at least 300 evaluation rows; smaller strata are tabulated but cannot be tested. The unstratified policy (one scale for all) is run on the same
rows as the contrast ('where does the marginal certificate leave the risk'). The equal-risk RK45 / DOP853 of each policy is matched to that policy's own violation rate on the calibration rows.

### 4.2 D5 protocol

From each of the 21 JPL streams an evenly spaced sample of 50 members along the family (`periodic_d.select_indices`; the `dry` sample used for the written expectations and the `confirm` sample are disjoint). A member is dropped BEFORE any policy run if one period at 1e-13 does not close
(max-norm closure error above 1e-8), reaches a body's collision radius, or the reference stops. Horizons: 1, 2, 5, 10, 20 periods, those between 1 and 60 TU; the TEST uses the in-range ones (1 <= T <= 20 TU, the training horizon range); the others are tabulated as
extrapolation. Explanatory, reported and not tested: the oracle Sundman step of every in-range run (the policy's prediction error in dex per family) and a scale certified on the OTHER (dry) sample, evaluated on this one (the cost of fixing the transfer by recalibration). A reference is trustworthy at a horizon if a second, differently stepped DOP853 reference (rtol 1e-12, step <= 0.004) agrees within 5 % of the tolerance; untrusted runs are listed and excluded (they are the 'unpredictable' of the plan's D12, here for periodic orbits).
The states are integrated at the JPL mass ratio (relative difference from `config.MU` 2.1e-8) and the tolerance is the project's 100 km = 100 / 384,400 DU (101.4 km in JPL's own length unit). The policy is the frozen Part C `sund` at its frozen Part C scale; the adaptive comparators use the frozen
Part C certified and equal-risk rtols. Nothing is calibrated on these orbits. Family = the 11 groups of `periodic_d.GROUPS` (halo L1, halo L2, Lyapunov L1, L2, vertical L1, L2, axial L1, L2, DRO, butterfly, resonant); the statistic gives each family the same weight; the bootstrap resamples whole orbits (all horizons of a member, and the north / south mirror image of a halo or butterfly member, are one cluster).

## 5. Reported, not tested

For every policy and every comparator: the violation rate per stratum, per Hill regime, per horizon band and per closest-approach band with Clopper-Pearson intervals (so that the risk of the adaptive comparators is shown where it lands, not only the policy's); the costs against RK45 / DOP853 at the
certified and the equal-risk tolerance; the lock run on D_lock_D; the other physics rules and the linear arm against the ML policy, paired as in D1; the compiled DP5(4) with SciPy's controller, the compiled DOP853 and the certified-tolerance comparators in D2; batch throughput of 10^3, 10^4, 10^5 orbits on one thread and on all
cores (pass criterion stated here, not part of the Holm family: the minimum over 7 repeats of the comparator's batch time exceeds the maximum of the policy's; expected NOT to pass: ratios 0.85-0.95 on D_select_D), the throughput ratio restricted to horizons up to 8 TU (D2b, descriptive; expected about 1.3: the Part C win/loss map by horizon band, in wall-clock), kernel variant R (the arithmetic of Part C and SciPy: expected ratio 0.58); the modelled lane efficiency of a lock-step batch from the measured step counts; the RHS-count and wall-clock conclusions side by side orbit by orbit; the other Part C tests (C1, C2, C3, C5) on D_replication;
D5 by stability band, by number of periods and for the extrapolated horizons; the doomed-orbit ledger is NOT repeated.

## 6. Exploratory (pre-registered as exploratory; no Holm family; every number is labelled exploratory)

* **D6** grouped permutation importance and drop-group refits (groups: state, proximity {r1, r2, r1/r2}, Jacobi {C, C - C_L1}, periapsis, horizon; FTLE as an offline explanatory group), both targets, paired bootstrap intervals (`run_explore_d_3d.py d6`).
  Expectation: for the Sundman target the proximity and periapsis groups matter much less than for the physical-time target; the horizon matters in both.
* **D7** surface-of-section chaos map (FTLE and policy violations at three Jacobi constants). Expectation (written after a 14 x 14 smoke test of the code, 80 points, which gave an AUC of 0.98, so it is weak evidence): violations are enriched on high-FTLE points, AUC of FTLE for the violation label between 0.7 and 0.95, the top FTLE decile violating several times more often than the rest; whether they sit on the ridges themselves or only in the same regions is what the map shows.
* **D10** tau_c sweep {1, 2, 4, 8} of the oracle Sundman cost on a random 1,500-orbit subsample of D_replication (against RK45 at the per-orbit oracle rtol of Part C's replication ledger) and the diagnosis of the non-monotone labels on 500 orbits (dense sweep in ds, ratio 2^(1/16)). Expectation: the oracle is cheaper than RK45-spec on about 46 / 51 / 52 / 53 % of orbits for tau_c = 1 / 2 / 4 / 8 (Part C's pilot: 45.9 % at 1, 49.5 % at 2, 51.6-52.8 % at 4-8); about 4-5 % of orbits are non-monotone in ds, they sit where the number of physical-time landing steps changes between neighbouring ds (the landing logic switches to physical time near T): the mean change in that number at a pass / fail flip is larger than the mean change between any two neighbouring ds.
* **D12** the 1 % highest lambda*T orbits of D_confirm_D with a tighter DOP853 (1e-14, step <= 0.002) and an LSODA (1e-13) reference. Expectation (D_select_D, 82 orbits with lambda*T >= 10.7): the references agree within 0.2 km on every orbit (none 'unpredictable'), so the policy's violations in this tail are real; its violation rate there is about 18 % against 2.7 % overall.
* **Tier 3** the leapfrog symplecticity check (done); IAS15 through REBOUND on a subsample (wall time); not done: KS-regularized RK4, E12 (2D control), D8 (tolerance and horizon dial), D9 (path-error label), D11 (cross-mu with dimensionless features).
* The expectation for each is stated before it is run and the result is reported whichever way it falls.

## 7. Safety valves

Stopping rule: each confirmatory step runs once per target (`confirm_log_d.jsonl`); a rerun needs `--allow_rerun "<reason>"` and a dated entry below. If the D2 timing cannot be done on an idle machine (1-minute load average above 1.5 after waiting) the step refuses; a run with `--allow_busy` is logged with its reason and D2 is then reported as 'run on a busy machine'.
If a D4 stratum cannot be certified it abstains (that is a result, not a failure of the test). If fewer than 100 D5 runs are in range with a trustworthy reference the test is reported as not evaluable (p = 1).

## 8. Deviations from the plan (all decided before D_confirm_D, D_replication, D_lock_D or the confirm sample of the JPL orbits is used)

1. No seed 51 (section 2); D_select_D has 10,000 candidate orbits instead of 20,000 (the dry runs only need the order of magnitude and the strata design check).
2. D1's rule set is 6 rules plus the linear arm, the winner chosen on D_select_D; the plan listed '2-4-parameter rules'; `rule_THPD` has 5 coefficients and is the plan's 'optional' C - C_L1 variant.
3. D2: the test statistic is the single-thread throughput ratio over the evaluation orbits with an orbit-level bootstrap; the plan's hypothesis was about batches. The batch throughput is measured and reported with a pass criterion (section 5). The primary comparator is DP5(4) with a PI controller; there is NO GPU or lock-step SIMD kernel
   (the M5's NEON registers are two doubles wide): lane efficiency is modelled from measured step counts.
4. D4: calibration on D_confirm_D and evaluation on D_replication (independent datasets) rather than a split of one dataset, which raises the evaluation count per stratum from about 8,000 to 16,000 rows.
5. D5 is a descriptive hypothesis in the plan; the Holm family needs a p-value, so it is the one-sided test of section 4 with a pre-set 10 % level, and the per-family rates are the descriptive result.
6. The seal does not require a clean git tree (nothing is committed on the user's behalf); it records the HEAD and the dirty paths and hashes every source file.
7. One 4-line branch was added to `policy_c_3d.predict_raw_c` (`models["rule"]`); Part C models never take it, and `tests/test_part_c.py` is re-run. `freeze_c.py check` therefore reports Part C's tree as changed; Part C's numbers are not recomputed.
8. Overheads are re-measured on this machine (section 3) instead of reusing Part C's 15.04 in D1.
9. Not done in this round (section 6): KS-regularized RK4, E12, D8, D9, D11.

## Deviations
- 2026-10-28, before the D2 confirmation step ran (the seal check passed): the 1-minute load average on this desktop machine did not fall below 1.5 (it sat at 1.49-1.50 with the CPU 94.5 % idle by `top`; macOS's load average lags and the desktop's own services contribute), so `run_confirm_d_3d.py run d2` was started with `--max_load 1.6` instead of the default 1.5. The step's log records the load average at its start and end (`d2_report.json`); no other job ran during it.
  Correction, same day, after the step: the load average recorded at the start of the step was 1.499 (below the default guard of 1.5), so the relaxation to 1.6 turned out not to be needed; it is recorded here because the step was launched with it.
- 2026-10-29, after every confirmatory step had run once (`confirm_log_d.jsonl`: d2, d1, d5, d4, d3_apply_sund, d3_apply_phys, d3_apply_global, d3_c1_resid, lock on the confirmation targets, then `tests`): two EXPLORATORY additions and one cosmetic edit, none of which adds to the Holm family or changes anything above "## Deviations". **D4b** (`run_explore_d_3d.py d4b`, written after the confirmatory D4 result was known and because of it): the adaptive solver certified in the same strata by the same fixed-sequence rule as the policy, compared with the stratified policy's D_replication ledger; the expectation was not stated in advance and the result is reported as post-hoc. **Figure edit**: the legend of `part_d/figures/d1.png` moved (it overlapped a label). `freeze_d.py check` therefore reports exactly `make_figures_d.py` and `run_explore_d_3d.py` as changed; no confirmatory step remains to be guarded. The other exploratory commands (D6, D7, D10, D12, symplectic, IAS15) were run after the seal on sealed code.
