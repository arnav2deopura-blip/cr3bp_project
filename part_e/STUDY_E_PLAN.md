# Study E: status and plan to finish it (written 2026-11-13)

## Status: not done

`part_e/` holds one file, `PREREGISTRATION_E.md` (last changed 2026-11-03). It is a **draft**: no seal (`FREEZE_E.json` does not exist),
no `freeze_e.py`, no runner, no `data_e/`, no confirm log, section 5 (expectations) is still `[dry run]` placeholders, and it is not on OSF.
`CR3BP_Project_Explained.md` Section 15A.7 already says "DRAFT: not sealed, nothing run", which is still correct.

What already exists and can be reused (checked 2026-11-13):

| Draft §7 item | What exists | What is missing |
|---|---|---|
| 0. OSF + public remote | nothing (`git remote -v` is empty) | everything; this one is yours (account, registration, choosing to make the repo public) |
| 1. `freeze_e.py`, `run_confirm_e_3d.py` | `freeze_d.py`, `run_confirm_d_3d.py` as models | both scripts; the clean-tree check (`git status --porcelain` must be empty) |
| 2. Relabel at 10 km / 1,000 km | `generate_3d_data.py --tol_km` already exists | a check that the same seed gives the same initial states at every tolerance; nothing else |
| 3. Path-error certificate | `endpoint_audit_3d.checkpoint_errors` (Study C, 50 checkpoints) | a K = 20 checkpoint variant inside the runner, path-certified scale calibration |
| 4. Compiled event detection | compiled DP5(4)-PI / DOP853 in `compiled_solvers_d.py` (no events) | collision / escape time by dense interpolation; the strict event-time rule |
| 5. OOD gate | `conformal_uncertainty.py`, the JPL family responses from D5 | kNN distance on the 25 features, conformal threshold, leave-one-family-out driver |
| 6. Timing script | Study D timing code (`time_per_orbit`) | load-average gate (`os.getloadavg`), two-session log, **an x86-64 Linux machine** |
| 7. Segment collision in runners | `collision_segment.crossed_bodies` + its test | wiring into the physical-time runners used by Part E only |
| 8. Compiled adaptive solvers in Sundman time | SciPy version `defense_runs.run_sundman_adaptive` | numba version with event at t(s) = T; unit test against the SciPy one |

Data generation is not the bottleneck: on this machine Study D's 20,000-orbit set took about 13 minutes (`data_d/gen_part_d.log`), so all
five Study E sets plus the two extra tolerances should take roughly 1–2 hours.

## Step 0: decisions only you can make (before any code)

These change what gets built, so they come first. Each must be settled and written into the draft **before** the dry run.

**D1. Is Study E about the RK4 policy only, or does it add a learned 8th-order step?** **Decided 2026-11-13: Option B (add E7).**
Written into the draft as question E7, sets E_confirm_20 (seed 65) and E_lock_20 (seed 66), the frozen E7 policy in §3, tests E7a/E7b in §4
(Holm over E1–E5 and E7), a prior in §5 and build item 9 in §7. The pilots are defined as P24–P25 in `posthoc/README.md` (not yet run).
The current draft tests the frozen RK4 Sundman policy. The data already in hand point towards E1 failing against the strongest comparator
(Sundman-time DOP853, ratio of totals 0.85 at 1–3 TU, P20). The one lead from Nov 3–4 (P23) is an 8th-order fixed step in Sundman time,
but that result is (a) an oracle, not a learned step, (b) compared with oracle-tuned *physical-time* RK45/DOP853, **not** with Sundman-time
DOP853, and (c) censored on 40 of 300 orbits.
- Option A: keep the draft's scope (RK4 policy). Quicker; probably an honest negative.
- Option B (my recommendation if you have the time): add an **E7** arm, a learned RK8 step in Sundman time, trained on a new training set
  (for example seed 59), with its own selection on E_select and its own frozen model. This is the only arm with a real chance of a positive
  result against the strongest comparator. It adds a training and selection phase (Step 1b and Step 3) before the seal.
- Option C: a minimal Study E (E1 + E4 only) if time is short.

**D6 (new, from the P25 pilot). Equal risk overall or per horizon band?** **Decided 2026-11-13: overall (primary), per band reported as a declared secondary analysis** (draft §4 and §6). In P25 the learned RK8 policy and its equal-risk comparators had
the same overall violation rate (3.5 % vs 3.6–3.9 %), but the policy put its violations at short horizons (3.8 % at 1–3 TU vs ~0.1 %) and the
comparators at long ones (~10.6 % at 8–20 TU vs 4.9 %). Overall equal risk is the sealed rule and keeps E7 comparable with Studies C–D;
per-band calibration (as D4's per-stratum certificate) is fairer per band but costs power. Decide before the seal; report the other as secondary.

**D2. Which comparators enter E1?** The draft lists Sundman-time solvers as "proposed, decide before the seal". The McGreivy–Hakim standard
that the earlier parts followed says include the strongest baseline. My recommendation: include compiled DP5(4)-PI in physical **and** Sundman
time in the intersection–union test, and report Sundman-time DOP853 for every test, with the 0.85 prior written into the expectations.

**D3. Deadline.** If there is a competition or submission date, Study E must fit before it, or it becomes "future work, preregistered" in the
manuscript. That is a legitimate outcome: a sealed, OSF-registered Study E that has not yet been run is still worth stating.

**D4. Second CPU architecture (E6).** Do you have access to an x86-64 Linux machine (school cluster, a cloud VM, a friend's PC)? If not, drop
the second architecture from E6 before the seal rather than as a deviation afterwards.

**D5. Committing and going public.** A clean-tree seal means the whole tree has to be committed first, including or ignoring the untracked
poster files (`CR3BP_Poster.af`, `poster/`, `affinity_bridge/`, `cr3bp_poster_48x36.js`). Pushing to a public remote is your call.

## Step 1: exploratory pilots on Study D data (no Study E data touched)

Study D data are not Study E test data, so looking at them is allowed. Everything here is labelled exploratory and only informs the design.
- **1a.** Compare the oracle RK8-Sundman step with **Sundman-time DOP853 at equal risk** on the same 300 orbits (the missing comparison of
  P23), and widen the step grid so none of the 40 orbits is censored.
- **1b (only if D1 = B).** Train a learned RK8 step on the Study C training set, evaluate on the Study D replication set with the safety scale
  and overhead included. If it does not beat Sundman-time DOP853 in total there, it will not in Study E; drop E7.
- Both are defined as P24/P25 in `posthoc/README.md` (committed `dcfa44a`) and were **run on 2026-11-13** (code `6144fe7`; results in
  `posthoc/results/rk8s_vs_sundman_*` and `rk8s_learned_*`). Headline: the learned RK8 policy is 2.57× cheaper in total than the RK4 policy
  and beats every adaptive comparator at equal risk except Sundman-time DOP853, which it ties (ratio of totals 1.016, 95 % CI 0.994–1.040).
  E7 stays in; its expectation is "close to a tie" (draft §5). See D6.
- To build for the pilots: an RK8 step function for the monotone label search (`search_3d`), `run_posthoc.py` commands `rk8s_vs_sundman` (P24)
  and `rk8s_learned` (P25). RK8 labelling of the ~100,000 Study C training orbits should take minutes (P23 labelled 300 orbits, both kernels,
  in 3 seconds).

## Step 2: build the missing pieces (draft §7), each with a unit test

Suggested order (dependencies first):
1. **Item 8**: compiled DP5(4)-PI and DOP853 in Sundman time; test RHS counts and endpoints against `run_sundman_adaptive` on ~50 orbits.
2. **Item 4**: compiled event detection (collision / escape) for every comparator; strict event-time rule; test on the Study C doomed ledger.
3. **Item 7**: segment–sphere collision in the Part E physical-time runners only (sealed kernels untouched); test that it changes nothing
   on Study D runs except where a crossing is real.
4. **Item 3**: K = 20 path-error checkpoints and a path-certified scale calibration; test against `endpoint_audit_3d`.
5. **Item 2**: tolerance relabelling; test that seeds give identical initial states at 10, 100 and 1,000 km.
6. **Item 5**: OOD gate (kNN, conformal threshold) and the leave-one-family-out driver over the 11 JPL families.
7. **Item 6**: timing script with the load gate and two-session log.
8. **E7 policy** (item 9 of the draft): training script, selection on E_select, compiled kernel, frozen model with pinned overhead.
9. **Item 1**: `run_confirm_e_3d.py` (once-only, log file, `freeze_e.py check` before every step) and `freeze_e.py` (clean tree required,
   records HEAD, remote URL and OSF URL). Write these last so they wrap the finished code.

Run the full test suite after each item; the existing 7 suites must still pass (sealed numbers unchanged).

## Step 3: E_select dry run (seed 60 only)

1. Generate E_select (10,000 orbits, v6). Do **not** generate seeds 61–64 yet.
2. Re-measure the frozen policy's overhead on the idle machine and pin it.
3. Confirm atol = 0.1·rtol on E_select calibration rows.
4. Choose the OOD threshold (E5) on E_select.
5. Choose the E7 label ceiling ds_max (≤ 1 % censored) and the E7 model (trees × quantile grid of draft §3) on E_select; freeze it.
6. Run every test end to end in dry-run mode; write the numbers into §5 (Expectations) of the draft.
7. If the dry run shows an unfair comparison, fix it now and log the chronology (as in Part D).

## Step 4: freeze and register

1. Resolve every "proposed" item in the draft (D1, D2, D4); remove the "DRAFT" header line; keep "## Deviations" empty.
2. Commit the tree (clean). **Register `PREREGISTRATION_E.md` on OSF** and put the OSF link in the document (you do this; I cannot create
   accounts). Push the commit to a public remote if you chose to.
3. Generate seeds 61–66 (and the 10 km / 1,000 km relabels of 63), hash them into `part_e/FROZEN_FILES.json`, and do not load them.
4. Run `freeze_e.py seal`; commit `FREEZE_E.json`.

## Step 5: the confirmatory run (once)

1. `run_confirm_e_3d.py` on E_confirm, E_confirm_8, E_tol, E_confirm_20 and the JPL families; Holm over E1–E5 and E7.
2. E6 timing on this Mac (two sessions) and on the x86-64 machine if D4 = yes.
3. Last: E_lock (seed 64) and E_lock_20 (seed 66), one evaluation each.
4. Write `part_e/RESULTS_PART_E.md`; anything that goes wrong is a dated entry under "## Deviations".

## Step 6: write-up

- Manuscript: a Study E section and its rows in the test table. Under the ISEF Oct-2025 AI rules the paper, abstract, poster and conclusions
  text must be your own first draft; I can give correction lists, not prose.
- Update `CR3BP_Project_Explained.md` (Sections 1, 15A.7, 19, 21) once results exist.
- Record in the AI-use log that the preregistration and code were AI-assisted.

## Order of work at a glance

Step 0 decisions → 1a pilot (→ 1b if E7) → build items 8, 4, 7, 3, 2, 5, 6, (E7), 1 → E_select dry run → expectations → commit, OSF, seal
→ generate confirmation data → one confirmatory run → results → write-up.
