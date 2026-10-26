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
| 4 | Delete `RESULTS_REPORT.md.bak` (0.8) | NOT done. The file is byte-identical to `RESULTS_REPORT.md` (checked with `cmp`) but it is untracked, so `git rm` does not apply and the deletion would be permanent; the plan says it needs the user's OK. | Left for the user. |
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
