# Part C results: a better step-size policy, and what still limits it

Pre-registration: `part_c/PREREGISTRATION_C.md` (sealed 2026-10-27T01:37:44Z, `FREEZE_C.json`, git `ea3b6df`; the seal check passed before every confirmatory step and before the tests). Log of every decision: `part_c/PART_C_LOG.md`.
Plan: `PART_C_IMPROVEMENT_PLAN.md`. Data: D_train (seed 41, 100,000) + D_train_enriched (141, 20,000), D_select (42), **D_confirm_C (43, 20,000 orbits, 16,192 usable; 4,000 calibrate, 12,192 evaluated)**, D_lock_C (44, 5,000). Everything is Earth-Moon, 100 km, horizons 1-20 TU.
Costs are right-hand-side evaluations; a policy's cost includes its pinned model overhead (Sundman policy 15.04, physical-time policy 11.74 RHS-equivalents per orbit). "Cheaper" counts an orbit as a win only if the policy also met the tolerance.

## 1. The result in one paragraph

Part B's ML policy was never cheaper than an adaptive solver (0 of 14,161 orbits). Part C removes most of the gap. A policy that predicts ONE number per orbit, the fictitious step of a Sundman-time RK4, from the initial state and 13 free two-body periapsis features, with an end-to-end certified safety scale and a 25-tree model that costs 15 RHS-equivalents,
**is cheaper than an equal-risk RK45 on 71.4 % of orbits [70.6, 72.2], than a certified-tolerance RK45 on 82.4 %, than an equal-risk DOP853 on 50.6 %, and than a certified DOP853 on 58.6 %, at a realized violation rate of 4.4 % [4.0, 4.8].** It is **not** cheaper than the per-orbit oracle-tuned RK45 (8.7 %) or DOP853 (6.1 %), which choose their tolerance
knowing the answer; and the same ideas in physical time (the plan's Track 2 alone) do not get there (31.6 % against RK45 equal risk). The mean cost gap against the oracle-tuned RK45 fell from +1.41 dex (Part B, a factor 26) to +0.21 dex (a factor 1.6).

## 2. Confirmatory tests (Holm-Bonferroni over five tests, familywise 0.05; all one-sided; every number from `part_c/results/confirm/confirm_tests_confirm.json`)

| Test | Hypothesis | Result on D_confirm_C | Expectation written on D_select | p (Holm) | Verdict |
|---|---|---|---|---|---|
| C1 prediction | the periapsis-feature model's median abs(log10 pred / label) of the RK4 step is below the state-only 300-tree model refitted on the same data | 0.282 vs 0.319 dex, difference **0.037 [0.027, 0.048]** (n = 12,191) | 0.030 [0.020, 0.042] | 1.0e-3 (floor of 4,000 resamples) | **supported** |
| C2 physical-time policy | share cheaper than RK45 equal risk > 0.5 | **31.6 % [30.8, 32.5]** (n = 12,192) | 26.1 % | 1.0 | **not supported** |
| C3 Sundman paradigm | (a) the oracle fixed-ds Sundman run beats RK45-spec on > 50 % of orbits and (b) its gap rises less with the timescale range | (a) **51.1 % [50.2, 52.0]** (n = 12,190); (b) Spearman 0.225 vs 0.911, difference 0.686 [0.667, 0.705] | 51.1 %; 0.226 vs 0.908 | 0.012 | **supported**; (a) is a narrow pass (the pilot gave 49.5 %) |
| C4 Sundman ML policy | share cheaper than RK45 equal risk > 0.5 | **71.4 % [70.6, 72.2]** (n = 12,192) | 69.9 % | < 1e-300 | **supported** |
| C5 ML adds value | mean log10 total cost below one global certified step | saving **0.306 dex [0.302, 0.310]**, ML cheaper on 89.9 % (n = 11,359 orbits met by both) | 0.306 dex | 1.0e-3 | **supported** |

Four of five supported, one not: as expected on D_select (the pre-registered expectations were met within sampling error, C2 included). C1's margin is small (0.04 dex of a 0.3 dex error): the model that fits under the overhead budget is a 10-tree model, and the large prediction gain
of the periapsis features (0.20 dex with a 300-tree model, `PART_C_LOG.md` section 2) is mostly spent on being cheap enough to run.

## 3. What each change bought (means, dex, orbits that met the tolerance, against RK45-spec; the Part B column is D_confirm)

| term | Part B | Part C physical time | Part C Sundman time | what changed |
|---|---|---|---|---|
| paradigm | +0.21 | +0.12 | **-0.005** | time transformation (the gap no longer grows with the timescale range) |
| selection | +0.03 | +0.001 | - | leapfrog dropped (arms F, G) |
| prediction | +0.38 | +0.34 | **+0.10** | periapsis features; Sundman target far less variable (0.32 dex std against 0.91) |
| safety scale | +0.30 | +0.06 | **+0.06** | end-to-end fixed sequence on a fine grid, quantile level tuned (q = 0.2) |
| overhead | +0.50 | +0.05 | **+0.07** | compiled 10- / 25-tree model, 11.7 / 15.0 RHS-equivalents instead of 1,128 |
| rounding | -0.01 | -0.01 | -0.01 | |
| **total** | **+1.41** | **+0.55** | **+0.21** | |

Against the comparators a user would actually run, the Sundman policy's total is **-0.21 dex against RK45 with a certified tolerance and -0.13 against RK45 at equal risk** (negative = cheaper). Median costs on orbits that met the tolerance: Sundman policy 235 (integration 220 + 15), RK45-spec 146, RK45-cal 398, RK45-eqr 332, DOP853-eqr 230;
the oracle fixed Sundman step 168, the oracle fixed physical step 224. Against RK45-spec the policy would need a further 0.29 dex of integration saving (at its overhead) to win a majority; against DOP853 at equal risk it is at the edge (0.02 dex short).

## 4. Safety: what the certificate does and does not say

| policy | scale | realized violations on the evaluation orbits [95 % CP] | calibration rows |
|---|---|---|---|
| Sundman (`sund`) | 0.872 | **4.39 % [4.03, 4.77]** (535 / 12,192) | 4.3 % |
| physical time with abstention (`phys`) | 0.872 | 4.77 % [4.40, 5.17] (582 / 12,192); 22.8 % of orbits go to RK45 | 4.3 % |
| one global Sundman step (`global`) | 0.08 | 3.64 % [3.32, 3.99] | 3.6 % |

The Sundman policy's marginal risk is below alpha = 5 % with room; the physical-time policy's interval reaches 5.17 %, so its certificate did not transfer cleanly (a marginal guarantee from 4,000 calibration rows; the scale found on D_confirm_C, 0.872, is two grid steps above the one found on D_select's calibration fold, 0.734, at 0.872 the exact binomial p-value was 0.021 (172 violations in 4,000 rows, abstained orbits counted) against the level 0.1, at the next value 0.951 it was 0.49).
**The risk is concentrated where it was in Part B**: for the Sundman policy 13.5 % of orbits with min r2 < 0.05 DU violate, 8.4 % at 0.05-0.17 DU, 1.9 % at 0.17-0.8 DU and 0.4 % beyond (12,192 orbits); the certificate is marginal and says nothing about the conditional risk near the Moon.
Catastrophic violations (error above 10 x tol) are 6.6 % of the Sundman policy's violations (median error / tol 1.7, 90th percentile 6.0, maximum 427).
**Endpoint-only audit (Track 1.4, 1,000 re-run orbits, 50 checkpoints each).** Sundman: 2.6 % [1.7, 3.8] of runs have a maximum checkpoint error above tolerance while the final error is within it; the maximum exceeds the tolerance on 7.7 % of the audited runs against 5.1 % for the final error alone. Nine merged cells exceed the pre-registered 2 % rule, so for those cells
(`part_c/results/confirm/audit_sund/endpoint_audit_report.json`) both labels apply; the label definition is unchanged. Physical time: 0.5 % [0.2, 1.2], three cells. The practical reading: a run judged by its final position can be 2-3 points more often wrong in the middle than the certificate says.

## 5. Where it wins and where it does not (Sundman policy, share cheaper than RK45 at equal risk, by Hill regime and horizon band)

| regime | 1-3 TU | 3-8 TU | 8-20 TU |
|---|---|---|---|
| I (Hill regime I: C > C_L1, necks closed) | 95 % | 67 % | **3 %** |
| II (C_L2 < C < C_L1) | 97 % | 72 % | 6 % |
| III (C_L3 < C < C_L2) | 94 % | 68 % | 10 % |
| IV (C_L4 < C < C_L3) | 100 % | 94 % | 52 % |
| V (C < C_L4: no forbidden region) | 98 % | 89 % | 40 % |

By closest approach: 92 % beyond 0.8 DU, 70 % at 0.17-0.8, 55 % below 0.05, 54 % at 0.05-0.17. The policy loses on long horizons in the chaotic regimes, where the integration cost is large and an adaptive solver's own step control is well matched. Figure: `part_c/figures/cells.png`; the same data per comparator: `wins.png`.

## 6. Doomed orbits (Track 1.3; 3,808 reference-divergent orbits of D_confirm_C, 19 % of the draw; a separate ledger)

Part A / B excluded them. Run exactly like usable orbits, **the Sundman policy ends in the same event type as the reference on 99.9 % (3,805 / 3,808; 100 % of 237 escapes, 99.8 % of 600 primary collisions, 100 % of 2,971 secondary collisions)** and never runs through a collision: the time transformation shrinks the step where the bodies are. The event time agrees to the plan's strict rule
(|dt| <= tol / |v|) on only 46.7 % (secondary collisions 56.7 %, primary 15.2 %, escapes 0.4 %: a fixed step cannot place an event closer than its own length) and to the relaxed rule (reference event inside the detecting step) on 96.8 %. A fixed physical step is much weaker: its abstention rule sends 85.4 % of doomed orbits to RK45; of the 555 it runs, the event type matches on 93.3 %, and only 25.5 % of the 47 secondary collisions
(51 % never detected). The free physical crash flag (osculating periapsis inside the body's collision radius, reached before T): recall 81.1 % (primary collision 98.3 %, secondary 84.1 %, escape 0 %), precision 96.3 %, fires on 0.73 % of usable orbits.

## 7. Locked evaluation (D_lock_C, 4,020 usable orbits; the final policies with the scales and tolerances calibrated on D_confirm_C; one run)

| policy | violations [95 % CP] | median total cost | cheaper than RK45-spec / DOP853-spec | RK45-cal / DOP853-cal | RK45-eqr / DOP853-eqr |
|---|---|---|---|---|---|
| Sundman | 4.28 % [3.67, 4.95] | 251 | 9.0 / 6.1 % | 81.2 / 58.0 % | 70.0 / 49.4 % |
| physical time | 4.05 % [3.47, 4.71] | 520 | 0.5 / 1.3 % | 36.3 / 22.7 % | 30.0 / 19.1 % |
| one global step | 3.98 % [3.40, 4.63] | 484 | 1.5 / 0.7 % | 19.6 / 6.9 % | 9.3 / 4.6 % |

## 8. Limits, stated plainly

1. **The oracle-tuned adaptive solver is not beaten** (8.7 % / 6.1 %), and cannot be by a single-step policy that has to guess; "cheaper than RK45" holds against a certified or equal-risk tolerance, which is what a practitioner without the answer could set.
2. **The equal-risk comparator is calibrated to the policy's own risk on 4,000 rows** (4.3 %); a tolerance that is both safer and cheaper might exist; the certified comparator (RK45-cal, 82.4 %) and the equal-risk one bracket the claim.
3. **Costs are right-hand-side evaluations, not wall-clock.** The policy's 15 RHS-equivalents are measured against this project's own (allocating) RHS function at 25-28 ns; a leaner RHS would raise the overhead in these units. No wall-clock comparison with scipy was made.
4. **The Sundman label is not monotone in ds on 4.5 % of orbits** (pilot; the plan's acceptance was < 2 %) and 2 % have a failure inside the label window; safety is certified on the runs actually made, not on the label.
5. **C3(a) is a narrow pass** (51.1 %): it depends on tau_c = 2 (51.6 - 52.8 % at 4 - 8 in the pilot; 45.9 % at 1).
6. **Selection limits**: the overhead gate had to be moved from 10 to 20 RHS-equivalents (floor 10.7); the abstention optimum for the physical-time policy is at the edge of its grid; guard and abstention were not combined; the fast (bisection) certificate was used for selection only.
7. **One system, one tolerance, one horizon range.** Earth-Moon, 100 km, 1-20 TU; the mu dial was reanalysed (`part_c/results/c9/mu_strata`) but not rerun with matched resampling; periodic families (E8) and the 2D control (E12) were not done.
8. The physical-time policy is reported in full but is the weaker of the two; its confirmatory test failed as predicted.

## 9. Plan items: done / not done

Done: Track 0 errata (E1-E5, T2 addendum, mu-dial figure band); lever surface; end-to-end certification, leapfrog gate (exercised, then dropped), housekeeping; periapsis features; compiled models and the cost protocol; the Sundman integrator, labels, pilot (GO) and policy; guard kernels with fallback, abstention, fine-grid fixed sequence;
v5 data; selection; pre-registration, seal, confirmatory run, lock; doomed ledger and crash flag; endpoint audit; mu strata (4.1.1); the free-lookahead arm (4.2). **Not done**: 0.8 (deleting `RESULTS_REPORT.md.bak`: it is byte-identical to `RESULTS_REPORT.md` but untracked, so the deletion is permanent and needs your OK); 4.1.2-4.1.4 (matched resampling, model-size control, dimensionless transfer); 4.3 (measured batching); 4.4 (E8); 4.5 (E12);
the guard + abstention combination; the Sundman policy with a leapfrog-like symplectic transform.

## 10. Reproduce

```bash
PY=/Users/arnavdeopura/miniforge3/envs/orbit-ml/bin/python
$PY generate_3d_data.py --num_orbits 100000 --label_mode accuracy_monotone --pipeline v5 --tau_c 2.0 --base_seed 41 --n_workers 8 --output_dir data_c/train --tol_km 100 --chunk_size 8   # 141 enriched (--sampling_weights 0.2933 0.1067 0.6), 42, 43, 44
$PY run_part_c_3d.py freeze_policy --config part_c/frozen_configs/sund.json --train <D_train> <D_train_enriched> --train_enriched <D_train_enriched> --select <D_select> --out_dir out_c/freeze
$PY freeze_c.py check
$PY run_confirm_c_3d.py run apply_sund --target confirm     # then apply_phys, apply_global, c1_resid, doomed_*, audit_*; `tests`; the same apply_* with --target lock
$PY tests/test_part_c.py && $PY tests/test_phase0.py && $PY tests/test_failure_study.py
```
The datasets (about 160 MB) and run folders are not in git (hashes in `part_c/DATA_HASHES_C.json` and `FREEZE_C.json`); the frozen policies (about 1.3 MB) are in `part_c/frozen/`.
