# Defense plan: what was implemented (2026-11-03/04)

Item-by-item status of `DEFENSE_PLAN.md`. Every number below comes from a run folder under `posthoc/results/` (definitions written first in
`posthoc/README.md`, P10–P23) or from the sealed reports. Nothing is committed (the user has not asked for commits).

## Read this first: fair rules on AI use

The Society for Science table "Use of generative AI to support a research project" (Oct 2025,
`https://sspcdn.blob.core.windows.net/files/Documents/SEP/ISEF/2026/Rules/Generative-AI-Use-Table.pdf`) says that using generative AI to
**initially write the research plan, abstract, paper or poster is never acceptable**. It says the same of using AI to produce the conclusions or
future steps, or to add points to the paper. **AI-written code is acceptable only with explicit citation of which parts were AI-generated and a
log of the prompts.** The poster text in `poster/cr3bp_poster_v4.js`, the manuscript text and the corrections made today were all written by an AI
agent. For a fair, the student must write the abstract, poster and research paper in their own words. Treat the corrected poster strings as a
list of what must change, not as text to print. Keep a prompt log (the claude.ai and Claude Code transcripts) for the code. For PLOS ONE,
AI-assisted drafting with disclosure is a separate question; the manuscript already discloses it.

## P0: errors and overclaims

| Item | Status | Where |
|---|---|---|
| P0-1 "never in physical time" | Fixed in the abstract, introduction, Results heading, Study B and C paragraphs, conclusions and poster. Now says: 0/14,161 with an uncompiled model and 31.6 % with a compiled one, with violations at 12.1× tol in the median. Zero-overhead Study B shares are added: 27.3 / 22.7 % against certified / equal-risk RK45, and 17.0 / 14.8 % against DOP853. Even at zero overhead, RK45 needs only 7 % of the policy's total. | `main.tex`; S4 `overhead_multiples` |
| P0-2 title | Retitled to the plan's wording. The waterfall (P11) shows, over all usable orbits, +1.39 (B) → +0.47 (C physical) → +0.20 dex (C Sundman). The time transformation accounts for 0.27 dex (paired) of the 1.19. The text attributes the cost drop to the other changes and the share jump (31.6 → 71.4 %) to Sundman time. | title, Results, Discussion, Conclusions |
| P0-3 "+0.12 on the same trajectories" | Fixed. Over all usable orbits the physical-time paradigm term is +0.25 dex (B: +0.25) and the Sundman one 0.00. The sealed +0.12 is over met-and-not-abstained orbits; the abstained orbits average +0.57. | Results |
| P0-4 p-value floors | Table 2 prints the Holm-adjusted floor values with a footnote: T1, T2 0.00040; T5 0.00050; C1, C5 0.0010; D1 0.00075. D4 (exact, 5.0e-4) is not at a floor. | Table 2, Methods |
| P0-5 FTLE | Rewritten. FTLE does carry information (AUC 0.92–0.93; adding it cuts a third of the error), but computing it costs an integration. The Study A test was underpowered. | Results A, D6 sentence, Discussion |
| P0-6 per-stratum cost | Added to the abstract, conclusions and limitation (h): with symmetric per-stratum certification, adaptive solvers win on most trajectories (policy 42.5 % / 20.7 %). Also added to the poster. | |
| P0-7 S1 Appendix | Written: `manuscript/si/S1_Appendix.tex` → `.pdf`, 8 pages. It covers provenance, the dated deviations of every study, references, labels, Sundman, periapsis edge cases, certification and strata, leapfrog, terminating trajectories, endpoint audit, Study B exploratory work, Study A secondary results, IAS15, D2 kernel chronology, the literature search and an index. | |
| P0-8 verifiability | Done: a Methods sentence and S1 section 1 state that the seals are local, list the 23 uncommitted paths of Study D, and say the timestamps are not verifiable. The Study E draft now requires OSF registration and a public remote before the dry run. **The Zenodo/OSF deposit itself is yours to do** (it needs your account and publishes the repository). | |
| P0-9 data/code deposit | Not done (needs your Zenodo account). The `[repository DOI to be assigned]` placeholder remains. PLOS asks for a Data Availability Statement that names the repository and DOI (archived PLOS data policy; check the current page). | |
| P0-10 "reference-divergent" | Renamed "terminating" throughout the manuscript. Added the reference agreement (P18): ≤ 0.050 km on the Study C confirmation set and ≤ 0.25 km on every Study C/D set. The worst case is 2.0 km on one Study B trajectory. | Methods |
| P0-11 conditioning on survival | Stated next to Eq (cert), in the abstract and in limitation (e), with the 46.7 % strict event-time match. | |
| P0-12 contributions and mentor | A placeholder is in Acknowledgments, plus a comment saying CRediT roles go into the submission system. **You must write the mentor's name and role and the contributions; I could not know them.** ISEF: see the box above. Form 2A (Student Support Disclosure) is part of the 2026–27 forms. | |

## P1: post hoc analyses (all through `run_posthoc.py`, manifests in each run folder)

| ID | Command | Result |
|---|---|---|
| P1-1 | `risk` | Realized risk on test rows. Study C confirmation: policy 4.39 %, RK45-eqr 4.11 %, McNemar p = 0.24. Replication: 2.69 vs 2.89 %, p = 0.33. Certified RK45 / DOP853 are much safer (1.1–1.9 %), so the 82.4 % row is optimistic; it is now footnoted and placed below the symmetric rows. |
| P1-2 | `risk` | Severity. Policy violations have a median of 1.69× tol and 6.6 % exceed 10×; RK45-eqr 2.48× and 14.2 %; DOP853-eqr 2.50× and 12.8 %. Milder on 3 of the 4 sets, but **not on the Study D lock set** (2.50× vs 2.13×). The physical-time policy: 12.1× median, 52 % of finite-error violations > 10×, plus 36 collisions or failures. |
| P1-3 | `waterfall` | See P0-2. On the common Study C set (8,796 orbits) the physical-time policy costs 0.31 dex more than the Sundman policy. |
| P1-4 | `overhead` | Study B at overhead 0 / 15.04 / 1,128 against all six comparators. At 15.04 the policy wins 26.0 % against RK45-cal and 19.6 % against RK45-eqr. |
| P1-5 | `concentration` | The top 10 % of orbits carry 75.7 % (C) and 76.9 % (D) of the policy's evaluations (RK45-eqr: 67.5 %). The 8–20 TU band carries 68.2 / 70.3 %. |
| P1-6 | `timescale` | Partial Spearman of gap on log Δτ given log T: 0.900 (B), 0.902 (C physical), 0.02 (Sundman). The slope with log T in the model is 0.873. |
| P1-7 | `overhead` | Overhead is 1.0 % of the policy's total. At 1–3 TU the ratio of totals is 1.48 (×1), 1.40 (×2), 1.21 (×5), 0.99 (×10) on replication. |
| P1-8 | `ood` | Fig 10 grid points: 84–93 % lie beyond the in-distribution 95th percentile of kNN distance. D5: vertical, Lyapunov, DRO and resonant families 59–97 %; **axial and halo only 4–12 %, yet axial runs fail 96–100 %**, so a distance gate would not catch them (important for E5). |
| P1-9 | `mixture_curve` | The share falls from 74.8 % (no near-Moon orbits) to 54.7 % (all near-Moon) on replication; the ratio of totals goes from 0.98 to 0.70. The best case is parity (0.99, 0.98–1.01). Provenance: the mixture is in the first commit `4ec0009` (2026-10-23 16:46 −0400), 2.5 h before the Study A seal; it is not in `PREREGISTRATION.md`. |
| P1-10 | `multiplicity` | Holm over all 15 tests changes no verdict. The largest adjusted p for a supported test is C3 = 0.030. |
| P1-11 | `bca` | BCa intervals move by ≤ 0.004 from the percentile intervals. |
| P1-12 | `multiplicity` | D4 at a 5 % level: IUT p = 0.021, so it would still pass. The 0.07 result reproduces the sealed p exactly. |

## P2: new comparators (exploratory, Study D replication set, symmetric rules)

| ID | Command | Result |
|---|---|---|
| P2-1 adaptive in Sundman time | `sundman_adaptive` | **The ML contribution shrinks.** Sundman-time RK45 at equal risk: the policy wins 67.6 % (vs 71.6 % against physical RK45), ratio of totals 0.72 (vs 0.80). Sundman-time DOP853 at equal risk: the policy wins 45.7 %, ratio of totals **0.41**, i.e. DOP853 in Sundman time needs 41 % of the policy's evaluations. With symmetric certification: 62.2 % / 0.66 (RK45) and 43.5 % / 0.39 (DOP853). |
| by band (added) | eval columns | At 1–3 TU the policy keeps a total-cost edge over Sundman RK45 (1.31) and learned-tolerance RK45 (1.16) but **not over Sundman DOP853 (0.85)**. Recorded as a deviation in `posthoc/README.md`. |
| P2-2 learned tolerance | `learned_rtol` | **Giving RK45 the same information flips the per-trajectory result.** A 25-tree q = 0.2 model predicts each orbit's RK45 rtol (trained on the Study C oracle rtols) and is charged the same overhead. At equal risk (2.71 % vs 2.69 %, McNemar p = 0.96) the policy wins only **34.0 %** and the ratio of totals is 0.89. Certified: 32.6 % and 0.875. |
| P2-3 multistep | `multistep` | LSODA behaves like RK45: at equal risk (2.9 %) the policy wins 70.3 %, ratio of totals 0.85; with symmetric certification 67.4 % and 0.80. |
| P2-4 IAS15 | text | Removed from the Discussion and moved to S1 (section 12) with its caveats (ε at the edge of the search, wall time only, no force counter). |
| P2-5 RK8 in Sundman time | `rk8s` | **The negative result is specific to RK4 at the oracle level.** The oracle 8th-order fixed step in Sundman time beats oracle-tuned RK45 on 297/300 and DOP853 on 294/300 trajectories (median −0.23 / −0.21 dex). Oracle RK4: 127/300. Spot-checked against a fresh reference. A learned RK8 would pay roughly as much again in prediction, safety and overhead, so this is the most promising lead for Study E, not a result. |
| P2-6 retrain on one family | not done | This is Study E's E5. Note P1-8: the axial and halo families are inside the feature distribution. |

Part E: the Sundman-time adaptive comparator is added to E1's comparator list. Making E1 an intersection–union test against it is marked
"proposed, decide before the seal". The OSF step is added. The plan's advice to cut Part E down to E1 + E6 + E2 if the fair is close is your
decision and is not applied.

## What changed on disk

- `manuscript/main.tex`: title, abstract, introduction, Methods, Table 2, Table 3, Results, Discussion, limitations, conclusions, SI captions,
  and the acknowledgment placeholder. `main.pdf` was rebuilt.
- `manuscript/si/S1_Appendix.tex` and `.pdf` (new); `manuscript/si/S4_Table_posthoc_*.csv` (new tables from `si_tables.py --only s4`).
- `posthoc/README.md`: P10–P23 definitions (written before the runs) and deviations.
- `run_posthoc.py`: 14 new commands. `orbit_ml_cr3bp/defense_metrics.py` and `orbit_ml_cr3bp/defense_runs.py` are new, with tests in
  `tests/test_defense.py` (12 tests pass).
- `manuscript/scripts/si_tables.py`: the S4 export of the new runs.
- `poster/cr3bp_poster_v4.js`: claim strings (see the AI box above). `affinity_bridge/poster.js`, `poster/cr3bp_poster_v3.js` and
  `cr3bp_poster_48x36.js` still contain the old claims. The Affinity `.af` files were not touched; re-run the script in Affinity.
- `part_e/PREREGISTRATION_E.md`: comparator, OSF and build list (still an unsealed draft).

## Not done, and why

- Zenodo/OSF deposit, OSF registration of Study E: these need your accounts and publish material.
- Mentor acknowledgment and contributions: only you know them.
- The hand-written check script (plan §4): it is only worth something if you write it yourself.
- Q&A rehearsal: the bank below is updated with today's numbers.

## Q&A bank, updated

- **Q-4 "Is equal risk equal?"** On test data: 4.39 % vs 4.11 % (p = 0.24); on replication 2.69 % vs 2.89 % (p = 0.33). The policy's failures
  are milder on 3 of 4 sets, but not on the Study D lock set.
- **Q-6 "Is Δτ vs gap circular?"** Controlling for the horizon, the partial rank correlation is still 0.90 and the slope 0.87. In Sundman time it
  drops to 0.02.
- **Q-8 "Couldn't an adaptive solver use Sundman time too?"** Yes, and it gains. Sundman-time DOP853 needs 41 % of the policy's evaluations, and
  the policy wins on only 45.7 % of trajectories against it. Against Sundman-time RK45 the policy still wins 67.6 % per trajectory but loses in total (0.72).
- **New Q "Did the ML add anything over the adaptive solver?"** Not when the solver gets the same information. An RK45 whose tolerance is
  predicted by the same small model leaves the policy cheaper on only 34 % of trajectories at matched risk. The honest summary: the learned component
  helps; a fixed step is the wrong place to put it.
- **Q-9 periodic orbits:** add P1-8. The axial and halo members sit inside the training distribution in feature space and still fail.
