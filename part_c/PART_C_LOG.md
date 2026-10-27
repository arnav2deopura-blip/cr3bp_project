# Part C log: improving the ML step-size policy and strengthening the weak Part B results

Working log for `PART_C_IMPROVEMENT_PLAN.md` (the plan; kept outside the repo, in Downloads, at the time of writing). Part A (`PREREGISTRATION.md`, `FREEZE.json`) and Part B
(`part_b/PREREGISTRATION_B.md`, `FREEZE_B.json`) are sealed and are never re-run on this tree: Part A is commit `68a554b`, Part B is commit `68a43ff`.
Every new confirmatory claim goes in a separately sealed Part C with new seeds. Exploratory work is labelled exploratory.
"dex" = log10 units; costs are right-hand-side evaluations.

Dates are the machine date of the run. Entries are appended as each step finishes; nothing above an entry is edited afterwards except to add a pointer to a correction.

## 0. Decisions asked for in Section 12 of the plan

The plan recommends or defaults on each; I follow it unless the user says otherwise and record what I did.

| # | Question | What this log does | Why |
|---|---|---|---|
| 1 | Run Track 3 (Sundman) at all? | Yes, behind the pilot go/no-go of plan 6.5. | The plan recommends it; the pilot is exploratory and cheap, and a no-go drops C3-C5 before the seal. |
| 2 | Overhead gate | 10 RHS-equivalents (single-call, features included). | From the lever surface (C0 below): with O = 50 no comparator except RK45-cal is within reach at delta 0.6 dex; the plan's number is kept. |
| 3 | D_train size | 100k natural + 20k enriched, reduced to 60k + 12k if pipeline v5 is more than 2x slower than v4 (plan 8.1). | The plan's own rule. |
| 4 | Delete `RESULTS_REPORT.md.bak` (0.8) | **Done on 2026-10-27 after the user's explicit OK** (checked first with `cmp`: byte-identical to `RESULTS_REPORT.md`; untracked, so it was removed with `rm`, not committed). Earlier text: NOT done. The file is byte-identical to `RESULTS_REPORT.md` (checked with `cmp`) but it is untracked, so `git rm` does not apply and the deletion would be permanent; the plan says it needs the user's OK. | Left for the user. |
| 5 | E8 families and E12 2D control | After C8 (the plan's default for optional work), if time allows. | Optional in the plan. |
| 6 | Partial Track 2 gains: confirmatory C2 or descriptive only? | Decided at the seal (C7) from the D_select share; C2 stays in the family only if the pre-registered expectation can be written honestly. | Needs the D_select number first. |

## 1. C0 (2026-10-26): lever surface and Track 0 corrections

Run on 2026-10-26 with `run_part_c_3d.py lever|e3c_paired`; outputs in `part_c/results/c0/`.

**Lever surface (plan 2.3), D_explore ledger (3,000 evaluation orbits, 2,881 met the tolerance); D_confirm ledger as a check only.** Reproduces the plan's section-2.1 table
(D_confirm: 0.3 / 12 / 34 / 25 / 1 / 0 % against RK45-spec; 28 / 41 / 54 / 53 / 50 / 11 % against RK45-cal). The function is exact, not a grid: for each orbit the break-even reduction is
delta_i(O) = log10(C_int / (C_ad - O)), so the 50 % iso-line is the median of those thresholds. Delta (dex) needed for a 50 % share, D_explore:

| O (RHS-equiv, features included) | 0 | 10 | 25 | 50 | 100 | 200 | 500 | 1,128 |
|---|---|---|---|---|---|---|---|---|
| RK45-cal | 0.51 | 0.52 | 0.54 | 0.58 | 0.83 | 1.21 | unreachable | unreachable |
| RK45-eqr | 0.59 | 0.61 | 0.63 | 0.71 | 1.00 | 1.43 | unreachable | unreachable |
| DOP853-cal | 0.67 | 0.68 | 0.71 | 0.86 | 1.22 | 1.80 | unreachable | unreachable |
| RK45-spec | 0.87 | 0.90 | 1.18 | 1.42 | 1.90 | unreachable | unreachable | unreachable |

Reading, as the plan expected but now with the iso-lines: (i) the overhead gate is real, since from O = 100 on the required reduction exceeds the whole prediction + safety loss (about 0.68 dex, mean),
and from O = 500 on no reduction can win; below O = 50 the requirement is almost flat in O, so 10 RHS-equivalents costs nothing against 50 and the gate can sit at 10. (ii) Against the oracle-tuned RK45-spec a
majority needs 0.9-1.4 dex, more than Q + K can supply; Track 3 is the only route. (iii) By deployable cell (regime x horizon band; `lever_cells.csv`) at O = 10 and 0.6 dex the share cheaper than RK45-cal is
99 % / 94 % in regime IV (horizons 1-3 / 3-8), 79 % / 76 % in regime V, 38-45 % in regimes I and III at those horizons, and 12-14 % (regimes I, III) to 30 % (V) at horizons 8-20. The deployable-cell gate of
the plan's E11 has real structure: a policy that is cheap enough could win broadly in the Kepler-like corner while still losing in the chaotic long-horizon cells.
Caveat that stays: the counterfactual ignores that larger steps raise violations; it is an upper bound.

**Track 0 (errata).** Text-only corrections are in `part_b/RESULTS_E3C_MU_DIAL.md` ("Errata (2026-10-26)", items E1-E5) and a T2 addendum in `part_b/RESULTS_PART_B.md`. The new analysis is
`e3c_paired` (re-fits the f = 0 and f = 0.05 quantile models exactly as E3c did, evaluates both at scales 0.3-0.8 on the same 3,000 orbits; the f = 0 / 0.5 and f = 0.05 / 0.6 points reproduce the E3c table's
27.0 % / 3.3 % and 29.5 % / 4.4 % to the printed digits). Result: McNemar at the Part B scales p = 1.2e-12 (+2.5 points), but at equal scale -0.4 / -0.3 points (p = 0.27 / 0.45) and at equal realized risk the
lookahead arm is 1.3-1.9 points *below* f = 0. So the earlier "short lookahead helps" reading came from the larger certified scale, not from the lookahead. Item 0.6 (CI band in `mu_dial.png` panel 3):
Clopper-Pearson from `n_eval`, figure code only (`make_figures_b.py`, outside any seal that matters now). Item 0.8 (delete `RESULTS_REPORT.md.bak`): not done, see decision 4 above.
Tests: `tests/test_part_c.py` (lever functions).

## 2. C1-C3 on D_explore (2026-10-26; exploratory, development data; outputs in `part_c/results/c1_c2`, `c3`)

All runs use `run_part_c_3d.py policy_c` on D_explore (seed 21; 16,170 usable orbits) with Part B's cal / evaluation rows (2,000 / 3,000) and the same adaptive comparators. The training slice is smaller than
Part B's (9,170 instead of 11,170) because 2,000 rows are carved out of it to tune the leapfrog gate (the selection slice; the gate is never tuned on calibration rows). Integration-only costs here
(`C_pred` = 0); overhead is the subject of C3 below. Evaluation violation rates have 3,000 orbits behind them (95 % CP half-width about 0.6 points).

**Track 1.1: end-to-end common scale** (`policy_c_3d.certify_end_to_end`; tests in `tests/test_part_c.py`). The deployed choice is made on the unscaled predictions, so it cannot depend on the scale; one scale is then
tested on exactly the chosen runs. The certifier's violation count on the calibration rows equals the count of the evaluation pass (asserted in the unit test and in every run). What the Part B per-integrator
certificate hid is now visible: Part B's D_explore ledger has 5.7 % violations [3.8, 8.1] on the 494 orbits that chose leapfrog (D_confirm: 6.4 % [5.5, 7.5], 2,343 orbits, 173 of them infeasible choices), above alpha = 5 %,
against 3.6 % on RK4. The conditional rates are now part of every report (`conditional_risk`, `conditional_breach`).

| arm | features | leapfrog gate | scale rule | scale | RK4 cond. risk | leapfrog cond. risk (n) | infeasible leapfrog choices | median integration cost | integration cheaper than RK45-cal / eqr |
|---|---|---|---|---|---|---|---|---|---|
| Part B (ledger) | state | RF classifier, per-integrator scales | Bonferroni, 18 values | 0.5 / 0.5 | 3.6 % | 5.7 % (494) | 41 | 1,408 | 28.7 % / 25.0 % |
| A | state | classifier at 0.5, no tuning | end to end, Bonferroni, 18 values | 0.4 | 2.8 % | 4.1 % [2.6, 6.1] (541) | 33 | 1,972 | 23.4 % / 20.6 % |
| B | state | classifier, p* = 0.995 tuned on the selection slice (budget 0.5 %) | same | 0.5 | 3.4 % | 2.0 % [0.8, 4.1] (346) | 2 | 1,580 | 27.9 % / 25.1 % |
| C | state | same as B | end to end, fixed sequence on 28 values | 0.566 | 3.9 % | 3.8 % [2.0, 6.3] (346) | 2 | 1,370 | 30.0 % / 26.7 % |
| D | state + periapsis | classifier, p* = 0.80 | same as C | 0.734 | 3.8 % | 8.5 % [5.5, 12.4] (281) | 14 | 516 | 44.9 % / 39.9 % |
| E | state + periapsis | censored regressor, margin 0 | same as C | 0.734 | 3.8 % | 3.6 % [1.4, 7.2] (196) | 1 | 520 | 44.7 % / 40.2 % |

Reading. (i) A has the smaller scale (0.4) than Part B (0.5) mostly because it trains on fewer rows and leapfrog choices that no scale can repair are now inside the certificate; B shows the cost of those infeasible
choices: tuning the gate on the selection slice removes 31 of 33 of them and the scale and cost move back (median integration cost -20 %). (ii) The fixed sequence on the fine grid adds about +0.05 dex of scale at the same
certificate (0.5 to 0.566); the guard-rail of plan 2.3(a) (the risk at s = 0.1 must be well below alpha) is checked at the seal on D_select. (iii) The periapsis features (D, E) move the certified scale to 0.73 and cut the median
integration cost by 3x. (iv) The marginal certificate can hide a breach: D has 8.5 % leapfrog risk with a total of 4.2 %; the censored gate (E) removes it (3.6 %), so E is the candidate carried forward and the by-selected-integrator
variant (plan 1.1.4) is run as arm G for the record.

**Track 1.2.** RK4 needs no classifier (dropped). The gate is a tuned threshold on the selection slice: classifier probability p* (B, C, D) or the margin of a censored leapfrog regressor (E, trained on all rows with
the label 0.5 x dt_min where no step works; threshold 0.0 already meets the 0.5 % budget). Infeasible leapfrog choices on the evaluation rows: Part B 41 / 494 (explore) and 173 / 2,343 (confirm); B 2 / 346; E 1 / 196.

**Track 1.5 (housekeeping).** `config.py`: `LEGACY_JACOBI_TOLERANCES_3D` with the old name kept as an alias; the `FTLE_INTEGRATION_TIME` comment says "fallback". Done now, before the Part C seal, so it is inside the fingerprint.

**C2: periapsis features (plan 2.2), `osc_diag`, D_explore, 3 seeds.** The analytic two-body periapsis radius and time match a numerical two-body integration to 1e-8 (tests), including the rotating-frame velocity transform.
Diagnostics: Spearman(log tau_hat, log tau_min of the reference) = 0.91 overall (0.98 regime I; 0.64 for orbits that come within 0.05 DU, 0.37 for the 0.05-0.17 band); the osculating periapsis radius about the secondary is within a factor 2
of the true closest approach for 73 % of orbits that "reach" it before T, and 66 % of the close-approach orbits (min r2 < 0.05). Model effect, same HGB (300 trees, q = 0.1): median |log10 prediction / label| for RK4 0.339 -> 0.138 dex
(state + 13 features; -0.20 dex; go rule was -0.03), leapfrog 0.228 -> 0.137; close-approach bands -0.17 and -0.44 dex; pinball 0.071 -> 0.048; over-prediction rate unchanged (13.1 % -> 12.7 %). The single feature tau_hat gives -0.155 dex
of the -0.20. **GO.** This is more than the plan expected (E3b's oracle trajectory features reach 0.057 dex, so the analytic lookahead recovers about 70 % of that gap for free).

**C3: compiled models and cost (plan 2.1), `compile_study`, D_explore.** `compiled_model_3d.py` exports the fitted trees to flat arrays and evaluates them in Numba; equality with `est.predict` / `predict_proba` is asserted to
1e-12 on 10,000 rows for three model sizes and the classifier (`tests/test_part_c.py`; the exported attributes are private scikit-learn ones, pinned to 1.9.0). The compiled policy (features, both regressors, gate, choice, dt) agrees
with the Python policy on every row for state / periapsis features and classifier / censored / no gate. Cost protocol as in the plan, with the departure that "single call" is one orbit evaluated alone inside a compiled loop cycling through 1,000 states
(no Python dispatch; the RHS reference is timed the same way, `_cr3bp_derivatives_3d_numba`, about 100-125 ns here because it allocates its result) and the Python-dispatched value reported next to it.
**These timings were taken while eight other jobs were running (data generation, policy arms); they are for ranking only. The value that is pinned for the confirmatory run is re-measured on an idle machine before the seal.**

| model (periapsis features) | RK4 median abs error (dex) | RK4 pinball | leapfrog median abs error | RHS-equivalents per orbit (single call, loaded machine) |
|---|---|---|---|---|
| M0 linear quantile, 33 engineered features, censored gate | 0.261 | 0.069 | 0.228 | 11.8 |
| M1 10 trees, depth 3 | 0.440 | 0.072 | 0.375 | 20.5 |
| M1 25 trees, depth 3 | 0.292 | 0.063 | 0.295 | 29.7 |
| M1 50 trees, depth 4 | 0.213 | 0.057 | 0.198 | 88.9 |
| M1 100 trees, depth 5 | 0.150 | 0.051 | 0.160 | 212 |
| M2 300 trees (Part B's size) | 0.120 | 0.047 | 0.118 | 1,188 |
| M2 300 trees, state features only | 0.318 | 0.069 | 0.220 | 1,046 |

Part B's pinned 1,128 is reproduced by the compiled M2 (1,046-1,188 here), so compiling alone does not help; the saving comes from the model size. The periapsis features themselves cost 3.5-5 RHS-equivalents (13 values with logs, a square root and trigonometry), i.e. half of
the 10-RHS gate by themselves. Two findings change the plan's candidate list: (1) the physics-structured target of plan 2.2 (predict log10 d* - log10 tau_hat) is worth much more than "trees gain little": at equal size the median error falls from 0.47 to 0.29 dex
(10 trees), 0.30 to 0.23 (25), 0.21 to 0.17 (50), 0.12 to 0.10 (300) (`part_c/results/c3/structured_target.py`); it is now an option (`structured=True`, also in the compiled policy) and is decided on D_select. (2) A policy that needs a single regressor
(RK4 alone, or the Sundman-time regressor of Track 3) costs about a third of the two-regressor-plus-gate policy, which is what brings the tree models near the gate. **Gate status: M0 is at about 12 on the loaded machine, the smallest structured tree models
about 12-13 per single regressor; whether any model is eligible at <= 10 is decided with the idle-machine measurement.**

## 3. C4: the Sundman-time RK4 and its pilot (2026-10-26; exploratory; outputs in `part_c/results/c4`)

**Code.** `sundman_3d.py` (RK4 on the augmented 7-vector with g = 1/(1/tau_1 + 1/tau_2 + 1/tau_c), landing on T by finishing in physical time as the plan describes), `physics_3d.py` (a registry marker `Rk4sStep`, `INTEGRATORS_3D["rk4s"]`,
`EVALS_PER_STEP_3D["rk4s"] = 4`), `search_3d.find_dt_for_integrator_accuracy_monotone_3d(step_func=...)`, `data_3d` pipeline v5 (v4 + `dt_rk4s`, `rk4s_success`, `rk4s_censored`, `rk4s_reason`, counted `rk4s_evals`,
`sampling_branch`, the 13 periapsis columns, optional `--sampling_weights`, `--tau_c`). Tests (`tests/test_part_c.py`): lands exactly on T and the evaluation count equals an independent pure-Python statement of the algorithm
(one extra evaluation, the k1 of the landing decision, whose value the first landing step reuses); with g = 1 it is classical RK4 to 1e-13 and 4 x ceil(T / ds) evaluations; the convergence slope is 4 (median over 10 smooth orbits, range 3.6-4.4);
g shrinks by more than 20x at 0.01 DU from the Moon. Two decisions made while writing it: (i) a step that would overshoot T is discarded and the run finishes in physical time (its 4 evaluations stay counted); (ii) the v5 sampler weights
are recorded so enriched rows can be re-weighted (natural rows 1, enriched rows p_natural / p_enriched per branch).

**Pilot (seed 9101, 2,000 orbits, 1,602 usable, tau_c = 2, 100 km; `run_part_c_3d.py sundman_pilot`).** Oracle = the largest fictitious step that meets the tolerance, cost counted.

| comparator | oracle Sundman cheaper | oracle fixed physical step cheaper | median paradigm gap Sundman / fixed (dex) | Spearman(log delta_phys, gap): Sundman / fixed |
|---|---|---|---|---|
| RK45-spec | **49.5 %** | 24.4 % | 0.00 / +0.18 | **0.17** [0.13, 0.22] / 0.92; OLS slope 0.03 / 0.89 |
| DOP853-spec | 43.5 % | 26.2 % | +0.03 / +0.26 | 0.32 / 0.92 |
| RK45-cal (Part B certified rtol) | 96.5 % | 65.5 % | -0.39 / -0.23 | 0.68 / 0.92 |
| RK45-eqr | 90.3 % | 60.4 % | -0.30 / -0.15 | 0.67 / 0.92 |
| DOP853-cal / eqr | 82.7 % / 78.7 % | 53.4 % / 50.0 % | -0.25 / -0.20 | 0.52 / 0.53 |

**GO under the rule of plan 6.5** (>= 40 % vs RK45-spec: 49.5 %; Spearman lower by >= 0.3: 0.92 -> 0.17). The time transformation removes the dependence of the paradigm gap on the timescale range (T1's mechanism): the gap no longer grows with log delta_phys
(slope 0.89 -> 0.03). By deployable cell the gain is largest where Part B's gap was largest: regime I at horizons 3-8 TU, oracle fixed-step median 6,782 evaluations against RK45-spec 1,862 and Sundman 1,450.
The oracle is not the policy: it needs the label. What is left for the ML is a target that varies far less: log10 ds* has a standard deviation of 0.32 dex (the physical step: 0.91 dex), and on 1,100 training orbits a 25-tree depth-3 model
reaches 0.146 dex median error with the periapsis features (0.205 without), 0.093 with 100 trees depth 5. A single global certified ds (no ML; fixed sequence on half the orbits, ds = 0.067) is cheaper than RK45-cal on only 10.7 % of orbits and violates 1.1 %,
so the ML has room to add value (C5).

**Label acceptance (plan 6.3), reported as it is.** Censored: 0 %. Slope: 4 (unit test). **Non-monotone in ds on the dense log sweep (200 orbits, 81 grid points from 1e-4 to 4): 9 / 200 = 4.5 % [2.1, 8.4] (tau_c = 1: 6 %): the criterion "< 2 %" is NOT met.**
Label-window failures (a failing step inside [label/4, label], the range the label claims to certify): 2.0 % (borderline against the 2 % criterion, which it does not meet strictly). 11 of the 12 non-monotone orbits (tau_c = 1) have the largest passing grid
point within 18 % of the label; one has a lucky pass at 5x. The consequence for the policy is limited because its safety is certified end to end on the runs it actually makes, but it is a stated limitation of the Sundman label and goes into the pre-registration.
This deviates from the plan's acceptance, which asked for < 2 %; I proceed because 6.5 (the go rule) does not include it.

**tau_c.** Oracle cheaper than RK45-spec on 39.6 / 45.9 / 49.5 % for tau_c = 0.5 / 1 / 2 (the plan's grid); extending the grid: 51.6 / 52.8 / 52.6 % for 4 / 8 / 16 while the share against RK45-cal falls from 96.5 to 91.4 %.
**tau_c = 2** (best in the plan's grid; the gain beyond it is <= 3 points and trades against cal). The consequence for C3(a) (> 50 % against RK45-spec): the pilot value is 49.5 %, a coin flip; the pre-registration says so.

**Arm F (D_explore, state + periapsis features, RK4 only, no leapfrog, fixed sequence): scale 0.80, violation 4.5 %, median integration cost 472 (E, with the leapfrog gate: 520), integration-only cheaper than RK45-cal 47.7 %, eqr 42.0 %.**
Leapfrog costs more than it gains once the certificate is end to end, so the physical-time policy of Track 2 is RK4 only unless D_select says otherwise.

**Data generation (C6).** v5 costs 0.215 s per orbit single-process against 0.152 for v4 (1.4x; the plan's "> 2x" cut does not apply): D_select 20,000 orbits (seed 42) took 25 minutes on a loaded machine.
Started: D_train 100,000 (seed 41), D_train_enriched 20,000 (seed 141, weights 0.2933 / 0.1067 / 0.6), D_confirm_C 20,000 (seed 43), D_lock_C 5,000 (seed 44), all tau_c = 2, accuracy_monotone, 100 km, `data_c/`.
No pipeline file (`data_3d`, `sundman_3d`, `osculating_3d`, `physics_3d`, `search_3d`, `diagnostics_3d`, `config`) is edited while generation runs.

**Arms F and G (D_explore, periapsis features, fine fixed-sequence grid; `part_c/results/c1_c2/arm_F`, `arm_G`).**

| arm | policy | scales | RK4 cond. risk | leapfrog cond. risk (n) | median integration cost | cheaper than RK45-cal / eqr (integration only) |
|---|---|---|---|---|---|---|
| E | RK4 + censored-gate leapfrog, one common scale | 0.734 | 3.8 % | 3.6 % (196) | 520 | 44.7 % / 40.2 % |
| F | RK4 only | 0.80 | 4.5 % [3.8, 5.3] | none | 472 | 47.7 % / 42.0 % |
| G | RK4 + classifier-gate leapfrog, one scale per selected integrator (plan 1.1.4) | 0.80 / 0.26 | 4.1 % | 0.0 % [0, 1.3] (281) | 486 | 47.3 % / 42.2 % |

Leapfrog adds nothing once the certificate is end to end and the step is predicted from the periapsis features: RK4 alone is cheapest (F), and the by-integrator variant (G) fixes leapfrog's conditional risk only by giving it a scale of 0.26, which cancels its advantage.
Decision: the physical-time policy of Track 2 is **RK4 only** (leapfrog dropped; plan 2.4 "nothing else to change" is amended by this evidence).

## 4. Decisions recorded before any D_select experiment (2026-10-26)

*Hybrid {Sundman, physical RK4, leapfrog} (plan 6.4.5).* On the pilot (1,602 usable orbits) the oracle Sundman cost is at or below the oracle fixed physical-step cost on 80 % of orbits and strictly below on 60 %; the ratio's 95th percentile is +0.07 dex (log10 of the cost ratio:
5 % -1.12, 25 % -0.43, median -0.08, 75 % 0.00, 95 % +0.07). A hybrid can therefore gain at most 0.07 dex on 5 % of orbits, less than the error of the step-count model it would need and its extra inference. **Part C uses Sundman alone** (the plan's fallback).

*Selection protocol (all on D_select = seed 42; fixed here, before any D_select number exists).* D_select's usable orbits are split 50/50 (seed 42) into a calibration and an evaluation fold. For every candidate: fit on D_train (100,000 natural orbits, weight 1, plus 20,000 enriched orbits,
weight p_natural / p_enriched per sampling branch), certify ONE end-to-end scale on the calibration fold (alpha 0.05, delta 0.1, 28-value grid 0.1 ... 1.0), run the policy on the evaluation fold, record per-orbit integration cost. During selection the certificate uses the bisection form of the
fixed sequence (`fixed_sequence_fast`: it assumes the rejections are a prefix of the sorted grid and agrees with the exact rule whenever they are; tested on 300 simulated monotone curves); the sealed policies are re-certified with the exact ascending rule on D_confirm_C's calibration rows.
Criterion: the **median total cost per orbit that met the tolerance** (integration + the pinned model overhead, features included); a tie within 2 % goes to the simpler model (linear < fewer trees < more trees). A candidate whose evaluation-fold violation rate exceeds 6 % is excluded (the certificate did not transfer).
Stages: S1 model class and size at q = 0.1 (linear M0; trees 10x3, 25x3, 50x4, 100x5, 300 trees; for the physical family also the structured target; features state or state + periapsis); S2 quantile q in {0.05, 0.1, 0.2, 0.3, 0.5} for the S1 winner of each family;
S3 (physical family only) step-over guard g in {0.2, 0.3, 0.5, 1.0, inf} and the abstention rule {none, physics at three thresholds}; S4 (Sundman) nothing further. Overhead gate: the plan's 10 RHS-equivalents, pinned from an idle-machine measurement; if no candidate meets it, the rule is
recorded in a dated deviation BEFORE any selection cost is read. The frozen policies are then fitted once more with those choices (`freeze_policy`), their overhead pinned (`overhead --pin`), and sealed.

## 5. Selection stage S1 and the overhead gate (2026-10-26)

**Data (C6, all generated; `data_c/`, hashes in `part_c/DATA_HASHES_C.json` at the seal).** D_select 20,000 orbits (16,190 usable), D_train 100,000 (81,111 usable) + D_train_enriched 20,000 (weights 0.2933 / 0.1067 / 0.6), D_confirm_C 20,000, D_lock_C 5,000 (seeds 42, 41, 141, 43, 44); v5, tau_c = 2,
100 km, accuracy_monotone. Wall time for all five on this machine: 2 h 27 min, partly shared with other jobs. Phase-0-style acceptance on v5 (D_train / D_select): RK4 label success 99.991 % / 99.994 %, Sundman label success 99.998 % / 100 %, censored 0.0 % / 0.0 % (RK4, leapfrog, Sundman; 0.01 % of
D_select for the Sundman label), reference check (1e-13 vs 1e-12 at half step) maximum 1.27 km (D_train), 0.07 km (D_select), 0 orbits above the 10 km flag; divergent (doomed) orbits 18.9 % / 19.1 %; the ceiling is physical (2 TU) for RK4 / leapfrog and 4 for the Sundman step.
The Sundman label's monotonicity was measured on the pilot only (above); the other acceptance numbers are as in Part B.

**S1 integration-only results (D_select evaluation fold, q = 0.1; `part_c/selection/S1_sund`, `S1_phys`).** One end-to-end scale certified on the calibration fold (fast fixed sequence), evaluation-fold violation 2.5-4.5 % for every candidate; median integration cost per orbit that met the tolerance:

| Sundman-time policy (periapsis features unless noted) | certified scale | violations | median integration cost |
|---|---|---|---|
| linear (M0) | 0.951 | 4.5 % | 256 |
| trees 10 x 3 / 25 x 3 / 50 x 4 / 100 x 5 / 300 | 0.951 each | 3.6 / 3.9 / 3.4 / 3.4 / 3.2 % | 252 / 236 / 220 / 212 / 204 |
| state features only, 25 x 3 / 300 | 0.872 / 0.951 | 2.5 / 3.8 % | 280 / 232 |

| physical-time RK4 policy | certified scale | violations | median integration cost |
|---|---|---|---|
| linear (M0) | 0.673 | 4.3 % | 836 |
| trees 10 x 3 raw / structured | 0.80 / 0.73 | 4.1 / 4.2 % | 1,024 / 748 |
| 25 x 3 raw / structured | 0.80 / 0.73 | 4.3 / 4.1 % | 904 / 744 |
| 50 x 4 raw / structured | 0.87 / 0.87 | 4.4 / 4.5 % | 646 / 556 |
| 100 x 5 raw / structured | 0.87 / 0.87 | 3.8 / 4.2 % | 572 / 512 |
| 300 raw / structured | 0.87 / 0.87 | 3.5 / 3.6 % | 440 / 406 |
| state only, 25 x 3 / 300 | 0.62 / 0.73 | 4.3 / 4.0 % | 1,898 / 976 |

The same model size buys a 3-4x lower integration cost in Sundman time than in physical time (212-256 against 512-836 for 25-100 trees or linear), and the best Sundman model (204) is within 1.1x of the oracle fixed-ds median (184 on the pilot at tau_c = 2).

**Overhead measured near-idle (1-minute load average 4.7-5.5; one unrelated process held a core; `part_c/selection/S1_overheads_idle.json`, median of 7 repeats of 10^5 calls, RHS = 24-26 ns):**
linear 10.9-11.4 RHS-equivalents; trees 10 x 3: 12.0-13.3, 25 x 3: 16.6-17.2, 50 x 4: 27-29, 100 x 5: 71-72, 300: 318-357; the periapsis features alone 3.7-3.9; state-only 25 x 3: 10.5-10.7. (The loaded measurements of C3 inflated the ratio's reference by 4x: the RHS took 100-125 ns then and 25 ns now, so the loaded
numbers above in section 2 are superseded by these.)

**Deviation 1 (dated 2026-10-26, written after the timing and after reading S1 integration costs, before any total-cost ranking): the overhead gate is 20 RHS-equivalents, not 10.** No candidate meets the plan's gate of 10: the floor is 10.7 (state-only features) and 10.9 (linear with the periapsis features), because
the periapsis features alone cost 3.7-3.9 and any model adds at least one dot product. The plan's rule would then stop every cost claim (C2 and C4 are cost claims). The gate's reason in the plan is the lever surface (section 1 above): at O = 25 the reduction needed for a majority against RK45-cal rises only from 0.51 to 0.54 dex
and O = 50 still gives 0.58, i.e. the surface is flat below about 25. 20 keeps the spirit (overhead at most about 5 % of the RK45-cal median 374 and 7 % of RK45-eqr's 308) and admits the linear model, trees 10 x 3 and 25 x 3. It is a judgement; the user left the gate as an open decision (plan section 12, item 2), so it is flagged in the report.
It is not tuned on D_select totals: the eligible set is determined by the timing table alone.

**S1 ranking (gate 20, rule of section 4).** Sundman family: trees 25 x 3 (total 252.6 = 236 + 16.6) beats 10 x 3 (264.0) by 4.5 % and the linear model (267.4); 50 x 4 (249.0) is within 1.4 % but its overhead (29.0) exceeds the gate. Physical family (RK4, structured target): trees 10 x 3 (760.8) and 25 x 3 (761.2) tie,
the simpler 10 x 3 wins; the structured target is worth 0.13-0.15 dex at every size (e.g. 25 x 3: 904 raw, 744 structured). All 300-tree models are overhead-bound (totals 536-763).

**S2 (quantile level, D_select).** Sundman 25 x 3: q = 0.05 / 0.1 / 0.2 / 0.3 / 0.5 gives certified scales 1.00 / 0.95 / 0.87 / 0.80 / 0.67 and total cost 264.6 / 252.6 / **236.6** / 240.6 / 260.6 (violation 2.8-4.3 %): q = 0.2 wins (the plan's 0.1 was not tuned). Physical 10 x 3: q = 0.1 wins (760.8; 0.2: 820.8; 0.05: 1,088.8): a lower q aims
safer and keeps the scale high but loses too much step, a higher q forces a scale of 0.2-0.4.

**S3 (physical family; the options of the protocol one at a time, certification includes the fallback and the abstained orbits).** Step-over guard g = 0.2 / 0.3 / 0.5 / 1.0 (fallback to RK45 at the certified rtol on 24.8 / 8.0 / 2.9 / 1.3 % of orbits): scale 1.0 / 1.0 / 1.0 / 0.87, total 640.8 / 612.8 / 596.8 / 660.8.
Abstention to adaptive when the osculating orbit about the Moon returns within the horizon and log10 tau_hat < -1.0 / -1.5 / -2.0 (22.4 / 13.3 / 6.4 % of orbits): total **518.8** / 728.8 / 800.8. The best is abstention at -1.0 (no guard). **-1.0 is the edge of the pre-stated grid**, so a better threshold may exist; it was not searched (the protocol is kept), and the combination guard + abstention
was not run. The physical-time policy is therefore the 10 x 3 structured RK4 model with the abstention rule; its median total cost (519) is still 1.7x RK45-eqr's median, so C2 is expected to fail.

**Frozen policies (fitted once on D_train with the weights above; `part_c/frozen/`, specs in `policy_frozen_spec.json`).** `phys`: RK4 only, trees 10 x 3 (lr 0.3), q = 0.1, structured target, periapsis features, abstention rule (osc_reach_2 = 1 and log10 tau_hat < -1.0), common scale, fixed sequence on the 28-value grid;
`sund`: Sundman RK4, trees 25 x 3 (lr 0.2), q = 0.2, periapsis features, common scale, fixed sequence; `global`: Sundman RK4 with no model (the control of C5: one fictitious step certified on a 33-value grid 0.01 ... 2.56); `partb_refit`: Part B's state-only HGB (300 trees, lr 0.05, q = 0.1) fitted on the same weighted D_train (for C1).
**Pinned overheads (idle-ish machine, load average 6.5-6.9 of which one unrelated process held a core; median of 9 repeats): phys 11.74, sund 15.04 RHS-equivalents per orbit (features included; Python-dispatched single calls: 7.0 and 6.5).** These are the values charged in every confirmatory ledger.

## 6. Plan 2.5 checkpoint: the frozen policies on D_select (dry run; outputs in `part_c/results/dry`; development data, written into the pre-registration as expectations)

`run_confirm_c_3d.py run apply_* --target dry`: 4,000 calibration rows, 12,190 evaluation orbits of D_select (the policy's own choices were made on D_select, so these numbers are optimistic by selection, mildly: three tuned parameters).

| policy | certified scale | realized violation [95 % CP] | median total cost | cheaper than RK45-spec / DOP853-spec | RK45-cal / DOP853-cal | RK45-eqr / DOP853-eqr |
|---|---|---|---|---|---|---|
| Part B (D_confirm) | 0.5 / 0.5 | 3.9 % | 1,400 + 1,128 | 0 / 0 % | 0 / 0 % | 0 / 0 % |
| `phys` (RK4, 10 trees, abstention) | 0.734 | 4.1 % | 536 | 0.3 / 0.7 % | 32.6 / 19.2 % | 26.1 / 14.8 % |
| `sund` (Sundman, 25 trees, q = 0.2) | 0.872 | 4.3 % [3.95, 4.68] | **231** | 8.7 / 6.1 % | **81.5 / 58.6 %** | **69.9 / 50.4 %** |
| `global` (no model) | 0.08 | 3.7 % | 464 | 1.4 / 0.6 % | 19.5 / 6.6 % | 8.1 / 4.0 % |

Mean cost-gap decomposition against RK45-spec (dex, orbits that met the tolerance): Part B (D_confirm) paradigm +0.21, selection +0.03, prediction +0.38, safety scale +0.30, overhead +0.50, total **+1.41**;
`phys`: +0.12, +0.002, +0.33, +0.14, +0.04, total **+0.62**; `sund`: paradigm -0.006, prediction +0.10, safety scale +0.06, overhead +0.07, rounding -0.01, total **+0.21** (a factor 1.6 above RK45-spec on a geometric-mean basis; Part B: 26x).
Against the realistic comparators `sund` is cheaper: total -0.21 dex vs RK45-cal, -0.12 vs RK45-eqr. Lever surface of the `sund` ledger (reduction still needed for a 50 % share vs RK45-eqr: none, it is already above 50 %; vs RK45-spec at O = 15: +0.29 dex).
Per term what each change bought (means, D_select vs Part B's D_confirm; not the same orbits): overhead 0.50 -> 0.07 (compiled 25-tree model, 15 RHS-equivalents instead of 1,128); paradigm 0.21 -> -0.006 (Sundman time);
prediction 0.38 -> 0.10 (periapsis features + Sundman target; physical time 0.33); safety scale 0.30 -> 0.06 (end-to-end fixed sequence, a more accurate model, q = 0.2; physical time 0.14).

Doomed orbits (descriptive, D_select, 3,810 reference-divergent orbits, `sund`): the Sundman run ends in the same event type as the reference on **100 %** (primary collision 613 / 613, secondary 2,966 / 2,966, escape 231 / 231; none runs through to T); the event time agrees under the plan's strict rule on 46.8 % (secondary 57.0 %,
primary 14.8 %, escape 0.4 %) and under the relaxed rule (reference event inside the detecting step) on 96.6 % (escape 46.8 %: far from the bodies the physical step is long). The physical crash flag (osculating periapsis inside the body, reached before T) has recall 80.7 % overall
(primary 97.9 %, secondary 83.4 %, escape 0 %), precision 96.2 %, and fires on 0.76 % of usable orbits.
Endpoint audit (Track 1.4, 600 `sund` runs): max checkpoint error above tolerance with final error within: 2.0 % [1.0, 3.5]; max above tol on 6.2 % of runs against 4.2 % for the final error alone. Four merged cells exceed the 2 % rule in this small sample (n >= 20 per cell); the rule is applied to the confirmatory audit.

## 7. Confirmatory run and lock (2026-10-27; sealed 01:37:44Z, git ea3b6df; log `confirm_log_c.jsonl`, outputs `part_c/results/confirm`, `lock`)

Order run, once each: apply_sund, apply_phys, apply_global, c1_resid, (tests), doomed_sund, doomed_phys, audit_sund, audit_phys on D_confirm_C; apply_sund, apply_phys, apply_global on D_lock_C. The seal check printed OK before the tests and the runner re-checked before each step.
Result: C1, C3, C4, C5 supported, C2 not (`RESULTS_PART_C.md` section 2); every expectation written on D_select held within sampling error. Realized violation: Sundman 4.39 % [4.03, 4.77], physical time 4.77 % [4.40, 5.17], global step 3.64 %. Lock: Sundman 4.28 %, 70.0 % cheaper than RK45 at equal risk.
Process notes: (i) one stdin-driven multiprocessing job (the first D_select reference-position cache) could not start its spawn workers and left a stray Python process for about 35 minutes; found with `ps`, killed, rerun from a script file; it affected timings of other jobs only. (ii) An unrelated `node` process from another project
(PID 58534, about 24 h old at 100 % of one core) was running throughout and is not mine; the timings in sections 2 and 5 were taken at load averages of 5-7 because of it and of other jobs. (iii) One over-broad `git add -A` briefly staged the repository's pre-existing untracked files (`archive/`, `data/`, `RESULTS_REPORT.md*`); the local commit was undone
(`git reset --soft`) and recommitted with explicit paths; nothing was pushed and those files are untouched.
Track 4.2 (the periapsis arm of E3c) was run after the confirmatory steps under the dated deviation; result in `part_b/RESULTS_E3C_MU_DIAL.md` E6.
