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
