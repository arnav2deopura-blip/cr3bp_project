# Pre-registration, Part E: a deployable short-horizon claim, tested on total cost

Status: **DRAFT, not sealed.** Nothing in this document has been run. It implements section 3 of the rigor plan of 2026-10-30
(`CR3BP_Rigor_and_Bug_Plan.md`, kept outside the repository). Before the seal: build the missing pieces listed in section 7, run the dry
run on the selection set, write the expectations into section 5, and commit the tree. **Register this document on OSF (or an equivalent public
registry with a server timestamp) before the dry run**, and push the sealed commit to a public remote: the seals of Studies A–D are local and
their timestamps cannot be verified by a third party (`DEFENSE_PLAN.md`, P0-8). The seal (`freeze_e.py`, to be written on the model of
`freeze_d.py`) must refuse to run on a dirty git tree (Study D's seal recorded the dirty-file list instead). After the seal everything above
"## Deviations" is frozen and later changes are dated entries there.

Studies A–D are sealed and their verdicts stand. Part E does not reinterpret them. It tests on new seeds the claim that survived the post hoc
analyses of `posthoc/README.md`: the policy is cheaper in **total** only at short horizons (equal-risk RK45 needed 1.48 times the policy's
evaluations at 1–3 TU on the Study D replication set, 0.97 at 3–8 TU, 0.69 at 8–20 TU). That claim was found in Study D data, so it is a
hypothesis here, not a result.

**Added 2026-11-13, before any seal or any Study E data: arm E7.** The exploratory P23 run found that an 8th-order fixed step in Sundman time,
with the step chosen by an oracle, was cheaper than oracle-tuned RK45 on 297 of 300 orbits and than oracle-tuned DOP853 on 294 of 300, at every
horizon band, while the oracle RK4 step won 127. P20 found that Sundman-time DOP853 needs 0.41 of the RK4 policy's total (0.85 at 1–3 TU), so
the RK4 policy is unlikely to pass E1 against that comparator. E7 asks whether a **learned** 8th-order step keeps enough of the oracle margin to
beat the strongest adaptive comparators in total cost. P23 is an oracle on 300 orbits and did not include Sundman-time adaptive solvers, so
E7's expectations come from the pilots P24–P25 (`posthoc/README.md`, run on Study D data only) and the E_select dry run.

## 1. Questions

- **E1 (primary).** At horizons of 1–3 TU, is the frozen Sundman-time policy cheaper in aggregate than compiled DP5(4)-PI at equal risk, in RHS
  counts and in compiled wall-clock throughput?
- **E2.** Does E1 hold at 10 km and at 1,000 km tolerance (Study D's planned D8)?
- **E3.** Can a path-error certificate keep the mid-run violation rate at or below 5 % while the E1 property holds?
- **E4.** On the full population, doomed orbits included, is the policy no worse than the comparator in success rate and aggregate cost?
- **E5.** Can a pre-run out-of-distribution gate keep the policy safe on held-out periodic-orbit families while abstaining rarely in distribution?
- **E7.** Over 1–20 TU, is a learned 8th-order fixed step in Sundman time (the E7 policy, section 3) cheaper in aggregate than the strongest
  adaptive comparators at equal risk, in RHS counts and in compiled wall-clock throughput?
- **E6** is the timing protocol for the wall-clock part of E1 (idle machine, two sessions, a second CPU architecture); it is not a separate test.

## 2. Data (new seeds; nothing from Studies A–D is reused as a test set)

Generation: pipeline **v6** (`generate_3d_data.py --pipeline v6`), which is pipeline v5 without the FTLE integration
(`tests/test_hardening.py::test_pipeline_v6_labels_equal_v5` checks that every label column equals v5's). `accuracy_monotone`, tau_c = 2,
Earth–Moon mu = 0.01215058535, reference DOP853 at 1e-13. The usable rows are `reference_outcome == "ok"` (identical to the sealed rule on
every existing dataset, `test_usable_rows_equivalence_on_every_dataset`).

| Set | Seed | Candidates | Horizons | Use |
|---|---|---|---|---|
| E_select | 60 | 10,000 | 1–20 TU | dry run, expectations, OOD gate threshold selection |
| E_confirm | 61 | 20,000 | 1–3 TU (log-uniform) | E1, E3, E4 (4,000 calibration, rest evaluation) |
| E_confirm_8 | 62 | 20,000 | 1–8 TU | optional second arm of E1 (Holm), E2 |
| E_tol | 63 | 10,000 | 1–8 TU | E2: relabelled at 10 km and at 1,000 km |
| E_lock | 64 | 5,000 | 1–3 TU | one final evaluation |
| E_confirm_20 | 65 | 20,000 | 1–20 TU (log-uniform) | E7 (4,000 calibration, rest evaluation) |
| E_lock_20 | 66 | 5,000 | 1–20 TU | one final evaluation of the E7 policy |
| JPL families | — | leave-one-family-out over the 11 D5 families | 1–20 TU | E5 |

Training data of the E7 policy: the Study C fitting sets D_train (seed 41) and D_train_enriched (seed 141, same weights as Study C), relabelled
with the 8th-order Sundman-time step (section 7, item 9). They were never test data in any study; they are used for fitting only. E_confirm_20
spans 1–20 TU because the oracle advantage of the 8th-order step did not depend on the horizon in P23 (98–100 % in every band); the 1–3 TU rows of
E_confirm are evaluated for E7 as a secondary population.

The horizon range of E_confirm is restricted **by design** (the claim is about 1–3 TU). The sampling mixture is the Study C one (0.55 / 0.20
/ 0.25); the post hoc cislunar mixture (near-Moon, L1, L2 in equal parts) is reported for every test as a declared secondary population.

## 3. Frozen inputs

- The frozen Study C Sundman policy (`part_c/frozen/sund/policy_frozen.joblib`), unchanged; its overhead re-measured once on the idle
  machine of E6 and pinned before the seal.
- Comparators: compiled DP5(4) with the PI controller (primary), compiled DOP853, SciPy RK45 (for continuity with Table 3), and **the same
  adaptive solvers integrated in Sundman time** (the seven-dimensional system of the policy, same g and tau_c = 2, ending at the event t(s) = T;
  `orbit_ml_cr3bp/defense_runs.py::run_sundman_adaptive` for SciPy; compiled versions to be built, section 7). Added 2026-11-03, before any seal,
  because the post hoc P20 run (`posthoc/README.md`) found that a time transformation also helps the adaptive solvers: on the Study D replication
  set, at equal risk, the policy was cheaper than Sundman-time RK45 on 67.6 % of orbits (ratio of totals 0.72) and than Sundman-time DOP853 on
  45.7 % (ratio of totals 0.41). Proposed (to be decided before the seal): E1a and E1b tested against compiled DP5(4)-PI in physical time **and** in Sundman
  time (intersection–union: p = the larger), with the Sundman-time DOP853 reported for every test.
- **Comparator certification is symmetric** with the policy's (`orbit_ml_cr3bp/symmetric_cert.py`): fixed sequence, safest first, full δ,
  grid `RTOL_GRID_FINE` (1e-10 … 1e-3, 0.125 dex). `tests/test_posthoc.py::test_comparator_symmetry` must pass at the seal. The equal-risk
  comparator keeps the sealed rule (scan + four bisections).
- Absolute tolerance of the comparators: the setting chosen by the post hoc P8 analysis on calibration rows (`posthoc/results/atol_*`), fixed
  before the seal and recorded here: **atol = 0.1 · rtol** (cheapest at equal risk on the Study D replication calibration rows among
  1e-1, 1e-3, 1e-6 and the position-only variant; `posthoc/results/atol_*`). Confirm it on E_select before the seal.
- **The E7 policy** (`part_e/frozen/rk8s/policy_frozen.joblib`, frozen before the seal). Integrator: fixed-step 8th-order Runge–Kutta (the
  12-stage propagation formula of DOP853, no error control) in Sundman time, same g and τ_c = 2, with the RK4 kernel's landing rule
  (`defense_runs.integrate_endpoint_rk8s`); cost 12 RHS evaluations per step. Label: log10 of the largest fictitious step that meets 100 km, by
  the sealed monotone search (`find_dt_for_integrator_accuracy_monotone_3d`) with an RK8 step function, ds in [1e-4, ds_max]; ds_max is set on
  E_select so that at most 1 % of labels are censored (P23 found 40 of 300 censored at 4). Features: the same 25 as the Study C `sund` policy.
  Model: histogram gradient boosting, pinball loss, seed 42, depth 3, learning rate 0.2; trees ∈ {25, 50} and quantile ∈ {0.1, 0.2, 0.3},
  the one chosen on E_select (calibration fold for the scale, evaluation fold for the ratio of totals against the compiled Sundman-time DOP853 at
  equal risk; ties go to the smaller model). Safety scale: the sealed fixed-sequence rule (α = 0.05, δ = 0.1) on the 4,000 calibration rows of
  E_confirm_20. Overhead: re-measured on the idle machine of E6 and pinned before the seal. Nothing else (features, integrator, landing rule) is
  chosen on data.
- α = 0.05, δ = 0.1 (δ = 0.05 reported).

## 4. Confirmatory tests (Holm–Bonferroni over E1–E5 and E7, familywise 0.05; one-sided in the stated direction)

| Test | Statistic | H1 | Unit / resampling |
|---|---|---|---|
| **E1a** | ratio of totals, RHS evaluations: Σ comparator / Σ policy (policy overhead included, violating runs included), compiled DP5(4)-PI at equal risk, E_confirm evaluation rows | > 1 | orbit bootstrap, 4,000 resamples, p = share of resamples ≤ 1 |
| **E1b** | throughput ratio, compiled wall-clock, same rows and comparator, single thread, E6 protocol | > 1 | same |
| E1 | intersection–union: p = max(p_E1a, p_E1b) | | |
| **E2** | E1a at 10 km and at 1,000 km on E_tol (1–8 TU), each comparator recalibrated at that tolerance | ratio > 1 at both (intersection–union) | orbit bootstrap |
| **E3** | violation rate of the policy under a path-error certificate (maximum position error at K = 20 checkpoints on the dense reference) | rate ≤ 5 % (H0: rate > 5 %) and E1a > 1 with the path-certified scale | exact binomial; bootstrap; IUT |
| **E4** | full population (usable and doomed orbits), success defined below; policy minus comparator success rate, and ratio of totals | success difference > −1 point (non-inferiority) and ratio of totals > 1 | orbit bootstrap; IUT |
| **E7a** | ratio of totals, RHS evaluations: Σ comparator / Σ E7 policy (overhead and violating runs included), each comparator at equal risk to the E7 policy's **overall** calibration violation rate (the sealed rule; decided 2026-11-13, see below), E_confirm_20 evaluation rows; comparators: compiled DOP853 in Sundman time, compiled DP5(4)-PI in Sundman time and in physical time | > 1 against every comparator (intersection–union: p = the largest) | orbit bootstrap, 4,000 resamples |
| **E7b** | throughput ratio, compiled wall-clock, same rows and comparators, single thread, E6 protocol | > 1 against every comparator (IUT) | same |
| E7 | intersection–union: p = max(p_E7a, p_E7b) | | |
| **E5** | leave-one-family-out family-balanced violation rate with the OOD gate (abstentions run on the certified comparator) | < 10 %, with in-distribution abstention ≤ 5 % on E_confirm | cluster bootstrap within family (D5 statistic); IUT |

Success in E4, fixed before any run: a usable orbit succeeds when its final position is within the tolerance; a doomed orbit succeeds when
the run ends in the same event type as the reference **and** the event time is within tol / |v| of the reference event time (the strict rule
of the Study C doomed-orbit ledger), where v is the reference speed at the event. The comparator needs compiled event detection (section 7).

Collisions of every run are detected between nodes by the segment–sphere test (`orbit_ml_cr3bp/collision_segment.py`), not only at step
endpoints.

## 5. Expectations (to be written from the E_select dry run BEFORE the seal)

- E1: [dry run]. Priors at 1–3 TU: RHS ratio of totals 1.48 (1.44–1.52) against SciPy RK45 at equal risk (Study D replication set, post hoc);
  compiled throughput 1.61 against DP5(4)-PI (D2, Study D confirmation set, by horizon band).
- E2: expected to shrink at 10 km (the policy's certified scale shrinks and the landing steps weigh more): [dry run].
- E3: [dry run]; the endpoint audit of Study C found 2.6 % of runs wrong mid-run.
- E4: [dry run]; the policy matched the event type on 99.9 % of doomed orbits but the strict event time on 46.7 %, so E4 may fail on success.
- E7: [dry run]. **Pilots, run 2026-11-13 on Study D data only** (`posthoc/results/rk8s_vs_sundman_20261113T133953Z`,
  `posthoc/results/rk8s_learned_20261113T134043Z`). P24: the oracle RK8 step was cheaper than the oracle-tuned Sundman-time DOP853 on 97.3 %
  of the 300 orbits, ratio of totals 1.28 (1.27 with atol = 0.1 · rtol); raising the grid ceiling from 4 to 64 changed no cost (the 33 orbits
  still at the ceiling are run almost entirely by landing steps). P25: the learned RK8 policy (sund model class and size, certified scale 0.80,
  calibration violation rate 3.85 %, evaluation 3.50 %) needed 1/2.57 of the RK4 policy's total. Ratio of totals at equal risk on the 12,253
  evaluation rows: physical RK45 1.95, physical DOP853 1.63, Sundman RK45 1.74, **Sundman DOP853 1.016 (95 % CI 0.994–1.040)**; share cheaper
  against Sundman DOP853 88.9 %. So E7a against Sundman-time DOP853 is expected to be close to a tie and could fail. **The risk is not equal
  within horizon bands**: at 1–3 TU the policy violated 3.8 % and the equal-risk comparators about 0.1 %; at 8–20 TU 4.9 % against about 10.6 %.
  By-band ratios (Sundman DOP853: 1.51 at 1–3 TU, 1.11 at 3–8, 0.91 at 8–20) are therefore not equal-risk comparisons. Whether to
  calibrate per horizon band was decided on 2026-11-13, before any Study E data: equal risk stays **overall** for E7 (and E1, E2), as in
  Studies B–D; the per-band version is a declared secondary analysis (section 6). Earlier priors, kept for the record: the oracle 8th-order step needed a median 0.27 dex (≈ 1.9×) fewer evaluations than the oracle
  RK4 step (P23); the RK4 policy's total was 2.4× that of Sundman-time DOP853 (P20, ratio 0.41). The gain grows with the horizon (median
  0.12 dex at 1–3 TU, 0.30 at 3–8 TU, 0.42 at 8–20 TU, i.e. 1.3×, 2.0× and 2.6×), and the long horizons carry most of the total, so against
  Sundman-time DOP853 the outcome is close and E7 may fail. A learned step also loses some of the oracle margin to prediction error and the safety
  scale. P25 has since measured it (above).
- E5: risk that periodic orbits lie inside the feature distribution (they are a measure-zero subset of it), so a feature-distance gate may not
  fire on them; expected outcome [dry run].

## 6. Exploratory (not in the Holm family; every number labelled exploratory)

- A smooth landing rule (solve for the last fictitious step from t(s), one Newton iteration on the time component).
- A smooth alternative to g (listed as not done in D10).
- Dimensionless transfer across mass ratios (D11).
- Doomed-orbit cost comparison under the relaxed event rule.

- The E7 policy on E_tol, under the path-error certificate and on the full population (doomed orbits included): the E2–E4 questions asked of
  the 8th-order step, reported as exploratory.
- The E7 policy against the frozen RK4 policy, orbit by orbit (descriptive).
- **Per-band equal risk (secondary, declared 2026-11-13).** For E7a (and E1a), each comparator is also set to equal risk with the policy
  separately within each horizon band (1–3, 3–8, 8–20 TU; the sealed scan-and-four-bisections rule on the calibration rows of that band, target
  = the policy's calibration violation rate in that band), and the ratio of totals is reported per band and pooled over bands. Reported next to
  the primary result, never in its place; every band's realized violation rates of policy and comparator are shown beside it. Reason: in the
  P25 pilot the overall-equal-risk comparators violated ~0.1 % at 1–3 TU and ~10.6 % at 8–20 TU while the policy violated 3.8 % and 4.9 %.

Not included: hybrid fixed/adaptive switching (excluded by the plan); new feature families unless the dry run justifies them.

## 7. To build before the dry run (none of this exists yet)

0. OSF registration of this document and a public remote for the repository (see the header).
1. `freeze_e.py` (clean-tree seal) and `run_confirm_e_3d.py` (once-only runner with a log), modelled on Study D's.
2. E2 relabelling at 10 km and 1,000 km (label search at the other tolerance; `generate_3d_data.py --tol_km`).
3. E3 path-error labels and certificate: dense-reference checkpoints and a `run_rows` variant that returns the maximum checkpoint error.
4. E4 compiled event detection for DP5(4)-PI and DOP853 (collision / escape time by dense interpolation), and the strict event-time rule.
5. E5 OOD gate: kNN distance in the 25-feature space with a conformal threshold chosen on E_select; leave-one-family-out driver over the
   JPL families.
6. E6: a timing script that refuses to start above load 1.0 (no relaxation), logs two sessions, and runs on an x86-64 Linux machine.
7. Wire `collision_segment.crossed_bodies` into the physical-time runners used in Part E (not into any sealed kernel).
8. Compiled DP5(4)-PI and DOP853 in Sundman time (event at t = T by dense interpolation), with unit tests against the SciPy version of P20.
9. E7: an RK8 Sundman-time step function for the monotone label search; relabelling of D_train and D_train_enriched (and of E_select and
   E_confirm_20 for the oracle columns); the training and selection script; a compiled E7 policy kernel (features, forest, RK8 steps) with a unit
   test against the Python kernel.
10. The pilots P24–P25 (`posthoc/README.md`), on Study D data only, with their definitions committed before they run.

## Deviations

(none; the document is not sealed)
