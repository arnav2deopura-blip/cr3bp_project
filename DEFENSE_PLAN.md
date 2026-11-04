# Defense plan: every claim in the CR3BP step-size project

Written 2026-11-03 from a review of `manuscript/main.tex`, the sealed results (`STUDY_A_SUMMARY.md`, `part_b|c|d/RESULTS_PART_*.md`,
`part_d/NARRATIVE.md`), `posthoc/README.md`, `part_e/PREREGISTRATION_E.md`, the seals (`FREEZE_*.json`), the poster text
(`poster/cr3bp_poster_v4.js`) and spot checks of the sealed ledgers.

**Numbers marked [new]** are ones I computed today from the sealed ledgers (`out_b/confirm/policy_20261025T231605Z`,
`out_c/confirm/{sund,phys}/apply_policy_*`). They are post hoc, are not in the manuscript, and should be re-run through `run_posthoc.py`
(with a manifest) before anyone quotes them. Each reproduced a sealed number first (8,707 and 3,855 wins against equal-risk RK45 both matched
exactly), so I trust the code path, but nothing was written to the repository.

Not read in full: `CR3BP_Project_Explained.md` (150 KB), `MANUSCRIPT_PLAN.md`, the part logs, `PREREGISTRATION_B/C/D.md`. I couldn't see the
2026-10-30 rigor plan (`CR3BP_Rigor_and_Bug_Plan.md`) either, because it's outside this folder. Some items below may repeat it.

---

## 0. Bottom line

Your rigor is already unusual for this level: four sealed studies, independent seeds, end-to-end certification, overhead charged,
replication, locked sets, and a post hoc plan committed before it was computed. The weak points are not missing experiments as much as
**sentences that claim more than the sealed data supports**, plus **one missing comparator**:

1. "Physical time never wins" is false as written (Study C's physical-time policy won on 31.6%).
2. The title credits Sundman with "most of the cost gap", but most of the gap came from other changes.
3. Adaptive integration in Sundman time is the obvious comparator, and it was never run.
4. The preregistrations can't be verified by anyone outside the project (no remote, and Study D sealed with uncommitted code).
5. S1 Appendix, which the manuscript cites about 15 times, does not exist yet.

With a negative result, a reviewer or judge who finds **one** overclaim will discount everything else. Fix the P0 items before submission or
printing the poster.

Priority key: **P0** = error or overclaim, fix before submission or fair. **P1** = cheap analysis on sealed ledgers that closes a likely attack.
**P2** = new computation or experiment. **Q** = no new work, just a prepared answer.

---

## 1. P0: errors and overclaims to fix now

### P0-1. "In physical time the answer was no on every trajectory tested" is false
- **Where:** Introduction (main.tex l.155: "In physical time the answer was no on every trajectory tested"), Conclusions (l.509: "in physical
  time, was never cheaper than adaptive integration … 0 of 14,161"), poster ("A learned fixed step in physical time **never** beat adaptive
  integration").
- **Contradicting sealed evidence:**
  - Study C's physical-time policy (C2) was cheaper than equal-risk RK45 on 3,855/12,192 (31.6%), and 30.0% on the Study C lock set.
  - [new] Its 2,776 abstained trajectories contributed **0** of those wins, so all 3,855 are genuine fixed-step RK4 runs.
  - [new] Study B's policy with its overhead set to zero is cheaper than certified RK45 on 27.3%, equal-risk RK45 on 22.7%, certified DOP853
    on 17.0% and equal-risk DOP853 on 14.8%. Its 0/14,161 against realistic comparators is caused entirely by the scikit-learn overhead.
    That overhead was 1,128 RHS-equivalents per trajectory, while the policy's median integration cost was 1,510 and the median adaptive solve
    was about 150.
  - `RESULTS_PART_B.md` l.97 ("does not depend on it: with zero overhead the policy still beats RK45-spec on 0.3%") quotes only the
    oracle-tuned comparator. Against the comparators a practitioner would actually use, the conclusion *does* depend on overhead.
- **What is robust:** (a) against oracle-tuned solvers a physical fixed step essentially never wins, even with perfect knowledge (25.8%) and zero
  overhead (0.3%); (b) against realistic solvers a physical-time policy wins only a minority (≤32%) and loses in median cost (physical-time policy
  median total about 520 on the Study C lock set, 519 on selection, against 332 for RK45-eqr on the confirmation set; compare on one set
  before quoting). Physical-time violations are also severe: [new] the C2 policy's violations have a median error of 12.1× tol, and
  49% exceed 10× tol (Sundman policy: 1.7× and 6.5%).
- **Action:** Reword the abstract, intro, conclusion and poster to: "with an uncompiled model, 0/14,161; with a compiled model and periapsis
  features, 31.6%, a minority, with violations at 12× tolerance in the median." Add the zero-overhead shares to S4 Table.

### P0-2. The title over-attributes the result to Sundman
- **Title:** "…how a Sundman time transformation recovers most of the cost gap".
- **Problem:** The gap against oracle-tuned RK45 fell from +1.41 to +0.21 dex (`RESULTS_PART_C.md` §3), but the physical-time Study C policy,
  with the same compiled model, periapsis features and fine-grid certificate and **no** Sundman, already reached +0.55. On that table, the
  non-Sundman changes bought 0.86 dex, and overhead alone went from 0.50 to 0.05 dex (compiling the model). Sundman's increment was 0.34 dex.
  The table also mixes subsets (see P0-3), so the true split is unknown until it is recomputed.
- **Second problem:** "Recovers most of the cost gap" reads as a cost win, but the policy loses in total evaluations (RK45 total = 0.80–0.84
  of the policy's).
- **Action:** (1) Run P1-3 (same-subset waterfall) and attribute from it. (2) Retitle so the title states the share result and its limit,
  for example: *"A single pre-run time step loses to adaptive integration in the Earth–Moon three-body problem; in Sundman time it wins on
  most trajectories but not in total cost"*. In share terms, Sundman *is* the decisive change (31.6% → 71.4% with everything else held
  equal), so state it that way.

### P0-3. "+0.12 dex for the physical step on the same trajectories" uses a different subset
- **Where:** main.tex l.392, and the "Part C physical time" column of `RESULTS_PART_C.md` §3.
- [new] The physical-time oracle paradigm term on the Study C confirmation set is **+0.25 dex over all usable trajectories** (Study B: +0.25).
  It is +0.22 over those the physical policy met, and **+0.12 only over trajectories that the physical policy met and did not abstain on**.
  The abstained 22.8% are the hard near-Moon cases, with a mean paradigm term of +0.57.
- So the Sundman figure (−0.005 over 11,657 trajectories met by the Sundman policy) is being compared with a figure from an easier subset.
  The current wording **understates** Sundman's effect on the paradigm term (0.25 → −0.005, not 0.12 → −0.005), but "on the same trajectories"
  is wrong. A reviewer who recomputes it will find the mismatch.
- **Action:** Report both terms on one subset (all usable trajectories, or trajectories met by both policies). Fix "on the same trajectories".
  The same issue explains the unexplained "0.12 vs 0.21 in Study B".

### P0-4. Table 2 p-values for C1 and C5
- Table 2 prints "<0.001" (Holm). The sealed results give 1.0e-3 for both: the bootstrap floor (2.5e-4) times the Holm multiplier.
- **Action:** Print "0.001 (floor)" and footnote that bootstrap p-values at their floor are bounds. Check D1 (7.5e-4), D4 (5.0e-4) and the B
  tests the same way.

### P0-5. FTLE: "changed neither prediction quality" is absence of evidence
- **Where:** Discussion, first paragraph; Results l.357 ("did not change predictive quality by an amount the preregistered test could detect" is
  fine; the Discussion sentence is not).
- The point estimate was ΔR² = −0.25 (CI −0.54 to 0.04), which is a large, underpowered effect, not "no change". In Study D, adding FTLE cut the
  Sundman-target MSE by 0.0023 of 0.007, about **a third**, which is not "small".
- **Defensible version:** FTLE carries real information about difficulty (it explains failures at AUC 0.92–0.93 and cuts error by about a
  third), but it costs a full variational integration, more than an adaptive solve, so it cannot be a pre-run feature. That partly *supports*
  the original hypothesis, and the student should say so: the hypothesis was right about what governs difficulty and wrong that it could be
  exploited before the run.

### P0-6. The per-stratum safety result is reported without its cost result
- The abstract and conclusions report the D4 safety success ("every stratum at or below 3.1%"). They leave out the post hoc result that, with
  the adaptive solvers certified per stratum by the same rule on the fine grid, the policy was cheaper than RK45 on only 42.5% and than DOP853
  on 20.7%. In other words, **with symmetric conditional certification, both adaptive solvers win on most trajectories**.
- That is selective reporting by omission. Add one sentence to the abstract or conclusions and to limitation (h).

### P0-7. S1 Appendix does not exist
- `manuscript/si/` holds only the CSV tables. The main text sends the reader to S1 Appendix for the deviations log, non-monotone labels,
  periapsis edge cases (near-circular and near-parabolic), the strata merging rule, the IAS15 method, the reference ledger, the endpoint audit,
  the μ dial, Study A H2 and H4–H7, and Sun–Earth. Reviewers will look for exactly these.
- **Action:** Write `si/S1_Appendix.tex`. It must include the **full deviations list of every study with dates**. Known items, all disclosed
  in the part documents, need to be in it:
  - Study A: H1′a was added after development data had been seen.
  - Study C: the overhead gate moved from 10 to 20 RHS-equivalents; the abstention threshold landed at the edge of its grid; the Sundman
    label missed its <2% non-monotone acceptance criterion (4.5%).
  - Study D: the load guard was relaxed from 1.5 to 1.6; 23 paths were uncommitted at the seal; the D2 kernel was corrected twice in dry runs.

### P0-8. Can outsiders verify the preregistrations?
- `git remote -v` is empty, so every seal timestamp is a local commit date, which its author can rewrite.
- `FREEZE_D.json` records 23 uncommitted paths at the Study D seal, including all of Study D's own code (`confirm_d_3d.py`, `rules_d.py`,
  `strata_d.py`, `run_confirm_d_3d.py`, and `part_d/` itself). Only their SHA-256 hashes were sealed, and those hashes sit in a local file
  that was committed later.
- This doesn't mean anything was altered. It means nobody outside the project can confirm the order of events, and the manuscript calls these
  "preregistered".
- **Action:**
  1. Archive the repository on Zenodo or OSF **now**. That timestamps the current state independently; it can't backdate the seals.
  2. Add one honest sentence to the Methods: "the seals are local hash manifests in a git repository that was not mirrored publicly; their
     timestamps are not independently verifiable." The current text, "internal preregistrations … not registered reports", is close but
     doesn't say this.
  3. Supporting evidence, if wanted: claude.ai conversation logs have server-side timestamps for when each study was designed and run. They
     are informal, but useful if a judge asks.
  4. Study E: register on OSF **before** any run, and keep the clean-tree seal its draft already requires.

### P0-9. Data and code deposit
- The manuscript says "[repository DOI to be assigned]", and PLOS ONE requires a data availability statement. The datasets (about 160 MB per
  study) are not in git. Deposit the code, seals, frozen models, ledgers and figure data with a DOI before submission. The same Zenodo
  archive from P0-8 can do both jobs.

### P0-10. Terminology: "reference-divergent"
- The manuscript uses "reference-divergent" for trajectories whose reference *collides or escapes* (3,808 = 19%). A reader will take it to mean
  "the two references disagree", which is a different check.
- Rename it ("terminating" or "doomed") and report the real agreement check separately. [new] On the usable Study C confirmation rows, the
  second DOP853 reference agrees with the first to **at most 0.050 km** (median 1.4e-8 km), about 2,000× below the tolerance. That is a strong
  number and should be in the text.

### P0-11. The guarantee is conditional on the future, so say so
- Calibration and evaluation use only trajectories whose reference reaches T. A user can't know that before the run, so the certificate is
  P(violation | trajectory survives to T) ≤ 5%. It is not a guarantee on the population a user would actually draw.
- The doomed-orbit ledger shows the event *type* matches 99.9%, but the strict event *time* rule matches only 46.7% (`RESULTS_PART_C.md` §6).
  The manuscript reports the 99.9% and not the 46.7%.
- **Action:** State the conditioning next to Eq (cert) and in the abstract's "certified rate", and add the 46.7%. Part E's E4 tests the full
  population.

### P0-12. Contribution and AI statement (journal and fair)
- The manuscript has one author. It says "The author designed the studies, set the hypotheses," and that the code was written by an AI agent.
  An adult mentor is also involved and is not acknowledged.
- Judges routinely ask "what did you do yourself?". An inaccurate contribution statement is the single fastest way to lose an otherwise strong
  project.
- **Action:**
  - Write an accurate CRediT-style contributions list (PLOS ONE asks for one, as I understand it; verify) and acknowledge the mentor.
  - Keep the AI disclosure. The model names were flagged in `CITATION_AUDIT.md` for confirmation.
  - Check the **current** ISEF and NCSEF rules on AI tools, adult help and continuation projects (the 2D NHSJS study came first). I haven't
    verified those rules, and they change from year to year.

---

## 2. P1: cheap analyses on the sealed ledgers (label all as post hoc)

Run each through `run_posthoc.py` with a manifest, add a dated entry to the deviations section of `posthoc/README.md` (the definitions below
are new), and place the result next to the sealed number it qualifies.

| ID | Attack it answers | Analysis | Status / [new] result |
|---|---|---|---|
| P1-1 | "Equal risk" is only matched on calibration rows | Realized **test** violation rate of each comparator column, plus a paired exact McNemar test against the policy | [new] Study C confirmation: policy 535 (4.39%), RK45-eqr 501 (4.11%), DOP853-eqr 532 (4.36%); discordant pairs 410 policy-only vs 376 RK45-only, two-sided exact p = 0.24 → risk equal within noise. Certified RK45 1.27% and certified DOP853 1.90% are **much safer than the policy**, which is why the preregistered "certified" rows (82.4%) overstate. Repeat on D3, the locks and D4. |
| P1-2 | Violations aren't all equal | Severity of violations (median err/tol, share >10× tol) for the policy and every comparator | [new] Policy median 1.69×, 6.5% >10×; RK45-eqr 2.48×, 14.2%; DOP853-eqr 2.50×, 12.8%. **The policy fails more gently than equal-risk adaptive solvers**, a real point in its favor that isn't reported anywhere. |
| P1-3 | The attribution behind P0-2 | Same-subset decomposition waterfall: Study B policy → C physical (compiled, periapsis) → C Sundman, each term over one common set of trajectories, plus an all-usable version with violating runs included | Not done. Needed before retitling. |
| P1-4 | "0/14,161 is an overhead artifact" | Study B shares at zero, 15 (compiled-equivalent) and 1,128 overhead, against all six comparators | [new] At zero overhead: 27.3 / 22.7 / 0.26% (RK45 cal / eqr / spec), 17.0 / 14.8 / 3.6% (DOP853). Add the 15-RHS case. |
| P1-5 | "Share vs total" | Concentration of cost: share of total evaluations in the costliest 10% of trajectories, and the per-band share of evaluations | [new] Study C confirmation: the top 10% of trajectories use 75.7% of the policy's evaluations; 8–20 TU uses 68.2%. One sentence makes the share/total split intuitive. |
| P1-6 | "Δτ vs paradigm gap is circular" | Partial Spearman of the paradigm gap on log Δτ controlling for log T, and the residual scatter. Show the slope ≈ 1 is a falsifiable prediction (it would fail if accuracy rather than timescale limited the step). | Not done. See Q-6. |
| P1-7 | "Overhead is machine-specific" | Share and ratio of totals with overhead ×2, ×5, ×10, by horizon band | [new] Overhead is 1.05% of the policy's total, so totals barely move. The 1–3 TU share is where it matters, so compute it there. |
| P1-8 | "Periodic orbits and the section grid aren't really out of distribution" | Distance of D5 members and Fig 10 grid points to the training set in the 25-feature space (kNN), compared with in-distribution test rows | Not done. It turns the bare claim "outside the sampled population" into evidence and pilots E5's gate. |
| P1-9 | "The mixture is arbitrary" | Share and ratio of totals as a continuous function of the near-Moon weight (0 → 1), not just three mixtures | Partly done (P2). Also find where the 0.55/0.20/0.25 mixture was first fixed (commit before the Study A seal?) and cite it; `PREREGISTRATION.md` doesn't mention it. |
| P1-10 | "Holm within each study isn't enough" | Holm over all 15 B–D tests at once | Not done. Report which verdicts survive. C3 (Holm p = 0.012 within C) is the one at risk. |
| P1-11 | Heavy-tailed totals | BCa or studentized bootstrap for the ratio of totals, compared with percentile | Not done; low risk. |
| P1-12 | "The 7% D4 test level is lenient" | Rerun each D4 stratum's test against 5% | The worst stratum's Clopper–Pearson upper bound is 4.9%, so it probably passes at 5% too; confirm formally. |

---

## 3. P2: missing comparators and experiments

Ordered by how likely a reviewer is to demand each, and how much it would change the story.

1. **Adaptive solver in Sundman time (most important).** RK45 and DOP853 on the same 7-D Sundman system (same g, τ_c = 2), certified and
   equal-risk, on the D3 replication set.
   - *Attack:* "Your gain is the time transformation, which any adaptive solver can use too. Did the learned fixed step add anything over
     regularize-and-adapt?"
   - *Expected (my guess, unverified):* adaptive controllers already track τ, so Sundman should help them less. If adaptive-in-Sundman beats
     the policy, the ML contribution shrinks to zero, and the paper must say so.
   - *Cost:* about the same as the existing comparator runs.
2. **Give the adaptive solver the same information.** Predict a per-trajectory RK45 rtol from the same 25 features with the same 25-tree model
   class (label: the oracle rtol, already computed as `RK45_spec_rtol`), certify a scale on it by the same LTT rule, and charge the same
   overhead.
   - *Attack:* "You compared a per-trajectory learned knob with a global knob."
   - D4b already hints at the answer: per-stratum RK45 beats the policy. This tests it properly.
   - Check first whether the training sets carry oracle rtols; if not, label a subset.
3. **Multistep comparator.** Adams–Bashforth–Moulton (LSODA in non-stiff mode, or a compiled ABM), evaluation counts only, at equal risk.
   Production astrodynamics tools use multistep and high-order RK. DOP853 covers high-order RK; nothing covers multistep.
4. **IAS15: fix it or cut it.** The current comparison (217 µs against 6.4 µs, dominated by Python call overhead, at the edge of the searched
   range) invites attack and adds little. Report force-evaluation counts at a converged accuracy parameter, or drop it from the Discussion.
5. **A higher-order fixed step in Sundman time** (oracle only, a few hundred trajectories). This answers "is the negative result specific to
   RK4?".
6. **Retrain on one target family** (dispersions around one halo or NRHO member). The "practical implication" is a dispersion study, and D5
   shows halo families violate 33–47% with the frozen policy. Without this, the practical claim rests on a population unlike the use case.
   This could become part of Part E (E5).

### Part E (preregistered): what to prioritize
`PREREGISTRATION_E.md` is well designed but needs seven components built (its §7). If the fair date is close, cut it down to:
- **E1** (total cost at 1–3 TU, RHS counts and compiled wall-clock, symmetric comparator certification, tuned atol), with **E6** as its timing
  protocol on a second machine.
- **E2** (10 km and 1,000 km).

E3–E5 can follow for the journal. Register on OSF before the dry run (P0-8). Add the **adaptive-in-Sundman comparator** to E1's comparator list
**before** the seal; otherwise E1 can be attacked the same way as C4.

---

## 4. Claim register: attack, current defense, what to add

Short form, by study. "OK" means the current evidence holds as long as the student can explain it.

### Framing and design
| Claim | Likely attack | Defense now | Add |
|---|---|---|---|
| A negative result is worth publishing | "It failed, so what?" | Preregistered; diagnosed mechanism (B); intervention tested on new data (C); replicated (D); a scope where it works | Q-1 script |
| The four studies are sequential and each was designed after seeing the last | Forking paths | Each study used new seeds; designs were frozen on selection data and confirmed once | OK; one sentence in the Methods |
| Novelty ("we are not aware of…") | "This has been done" | `LITERATURE_REVIEW.md` | Record the search (databases, queries, date) in S1 Appendix |
| 100 km tolerance | "Arbitrary" | Labelled illustrative; ≥20× tighter than the CR3BP-to-ephemeris gap | E2 (10 and 1,000 km) |
| 1–20 TU horizons | "Why these?" | 4–87 days covers transfers and dispersion studies | Q answer only |
| Synthetic mixture | "Every share depends on it" | Stated; P2 mixtures | P1-9 curve and provenance |

### Reference, labels, integrators
| Claim | Attack | Defense now | Add |
|---|---|---|---|
| Reference is accurate | "DOP853 at 1e-13 in chaos?" | Radau (≤0.003 km on 600), tightest DOP853 and LSODA on the top 1% λT (≤0.23 km) | [new] second reference ≤0.050 km on all usable rows (P0-10) |
| Label definition (two-octave monotone) | "Ad hoc; non-monotone 4%" | Certification uses actual end-to-end runs, so the label shapes training only and **cannot invalidate the guarantee** | Make that sentence explicit; record the missed <2% acceptance as a deviation |
| Leapfrog is symplectic and was dropped | "Why drop the structure-preserving method?" | Finite-difference symplecticity check; it cost more than it saved under the whole-policy certificate | OK |
| Sundman isn't symplectic or regularizing | "Use KS/Levi-Civita or a log-Hamiltonian leapfrog" | Stated as future work | Q answer: certification is empirical, so structure isn't needed for the claim |
| τ_c = 2 | "Tuned" | Chosen on pilot data; Study D sweep flat for 2–8 | C3(a) depends on τ_c (45.9% at τ_c = 1); say so |

### Cost accounting
| Claim | Attack | Defense now | Add |
|---|---|---|---|
| RHS count as the cost | "Sundman RHS is dearer; the controller has overhead" | D2: compiled per-evaluation cost at parity (22.4 vs 23.1 ns); adaptive controller overhead is **not** charged in RHS counts, which favors the comparator (conservative) | Say "conservative" explicitly |
| Overhead charged | "Pinned value is noisy" | Pinned before each seal, re-measured in D | P1-7 |
| Share as the primary statistic | "You picked the metric that flatters ML" | Preregistered with a stated rationale; total cost is now reported next to it; Part E tests total | Q-3 |
| Decomposition is exact | "Order-dependent?" | Each term is a ratio of defined quantities; means add exactly | P1-3 same-subset version |

### Certification and statistics
| Claim | Attack | Defense now | Add |
|---|---|---|---|
| LTT fixed-sequence guarantee | "Needs monotone risk?" | Fixed-sequence testing controls error for **any** pre-set order; monotonicity only affects power | Q-5 |
| Exchangeability | "Calibration ≠ test" | Same generator, disjoint rows; D3/D4 calibrate and evaluate on different seeds | State the conditioning on survival (P0-11) |
| "Equal risk" | "Equal only on calibration" | — | P1-1 (p = 0.24), P1-2 |
| Certified comparators | "Weaker rule on a coarser grid" | Post hoc P7 symmetric certification | Make the symmetric rows primary in the text and on the poster; demote 82.4% |
| Bootstrap and Holm | Floors, families | Reported | P0-4, P1-10 |
| D4 at 7%, D5 at 10% | "Lenient thresholds" | Preregistered | P1-12; D5 failed anyway |

### Study-specific
- **A:** H3 was underpowered (P0-5). H2's CIs reach 5.4% and 5.9%, which is consistent with the guarantee (the probability that the rate
  exceeds α is at most δ), so say that rather than "point estimates met".
- **B:** T2 "supported in sign only": explain what the preregistered magnitude was. Reword the "information about the future" phrasing: the
  initial state *determines* the future, so what is missing is a *cheap summary* of it. A dynamics reviewer will catch this.
- **C:** C3(a) was a narrow pass and **failed to replicate** (50.6%, p = 0.11). Lead with that whenever C3 is mentioned. C1's small effect
  came from the overhead budget (a 300-tree model gains 0.20 dex). For C2, see P0-1.
- **D:** D1 "is ML needed" should be read as: physics gets 73%, and the model matters for the adaptive comparison (70.2% vs 53.0%). For D2, the
  kernel fixes were made in **dry runs before the seal** (`PART_D_LOG.md` §4); cite this, because a reviewer will suspect otherwise. D5 is fine
  as it stands. Fig 10: 26–30% violations on simple planar grids is a second piece of OOD-fragility evidence, so don't bury it as
  "exploratory"; use P1-8 to show *why* it's out of distribution.

### Reproducibility and provenance
- `reproduce.sh` with `reproduce_out/` and `environment.lock.txt` is good; record that a reproduction matched.
- Code written by an AI agent: the defense is the tests (step-for-step SciPy agreement, 4th-order convergence, decomposition identity). The
  strongest extra step is an **independent check script written by the student by hand**: Sundman RK4, the evaluation counter and the
  violation check, re-scoring 100 ledger rows and matching `ml_evals` and `ml_err_km`. It defends the code and the student's understanding
  at once.

---

## 5. Manuscript edits (in addition to the P0 items)
1. Abstract: add that per-stratum certified adaptive solvers win (P0-6) and that the guarantee is conditional on survival (P0-11). The abstract
   is at the PLOS ONE limit (about 300 words), so cut elsewhere.
2. Table 3: put the symmetric-certification rows ahead of the Bonferroni rows, or footnote the 82.4% as optimistic. Add the realized
   comparator violation rates (P1-1).
3. Results l.380: "the prediction error came from the features, not from an information floor": keep it, but tie it to P1-5 and P1-6.
4. Discussion, IAS15 sentence: fix it or remove it (P2-4).
5. Limitations: add "no adaptive-in-Sundman comparator" until P2-1 is done.
6. Contributions and acknowledgments (P0-12).

---

## 6. Q&A bank (fair judges and reviewers)

The student should be able to give each answer without notes, including **deriving** Q-5 to Q-7 on a whiteboard.

- **Q-1 "Your method lost. Why is this a good project?"** It answers a question nobody had tested fairly: overhead charged, a certified failure
  rate, and comparators at equal risk. It explains *why* a fixed step loses: the penalty tracks the timescale range, ρ = 0.91 with slope 0.90.
  It tests a fix on new data (71%, replicated) and finds where the fix stops: total cost, DOP853, periodic orbits.
- **Q-2 "Why not just use an adaptive solver?"** For most users you should, and that is a finding. A pre-run step only pays at 1–3 TU, inside
  a known population (RK45 needs 1.48× the evaluations there).
- **Q-3 "Did you pick the metric that made ML look good?"** It was fixed before the data. When we added total cost, it showed the policy loses
  over 1–20 TU, and we report that in the abstract.
- **Q-4 "Is 'equal risk' really equal?"** On test data the policy violated 4.39% against RK45's 4.11% (p = 0.24), and the policy's failures
  were milder (1.7× vs 2.5× tolerance).
- **Q-5 "How does the certificate work?"** Test scales from safest to loosest. At each one, an exact binomial test asks whether the violation
  rate could exceed 5%, and you stop at the first failure. The chance that the chosen scale's true rate exceeds 5% is at most 10%. It is
  marginal, and conditional on the trajectory surviving to T.
- **Q-6 "Isn't Δτ vs the gap circular?"** Δτ predicts the cost ratio only if the step is limited by the timescale. If accuracy (chaos) set the
  step instead, the slope would not be about 1. The slope of 0.90 is a test the argument could have failed. Mention the partial correlation
  from P1-6.
- **Q-7 "What does Sundman time do?"** dt = g ds with g ≈ τ near a body, so a constant ds gives a physical step proportional to the local
  orbital timescale. The step adapts without an error estimator. It does not regularize the collision.
- **Q-8 "Couldn't an adaptive solver use Sundman time too?"** Yes; give the P2-1 result. Before it exists, say: not tested yet, and it is the
  first item of future work.
- **Q-9 "Why does it fail on periodic orbits?"** The predicted step is 0.3–0.5 dex too long on vertical, axial and Lyapunov families. The
  integrator converges at 4th order, so the step was wrong, not the method. The frozen RK45 tolerances failed there too.
- **Q-10 "What did you do, and what did your mentor and the AI do?"** This needs an honest, specific, rehearsed answer that matches the
  written contributions statement exactly.
- **Q-11 "How does this relate to your 2D two-body paper?"** Prepare a reconciliation. The 2D study measured selection accuracy, not cost
  against adaptive solvers, and Kepler orbits have a narrower timescale range than CR3BP flybys. Verify both statements against the 2D paper
  before using them.
- **Q-12 "Why trust AI-written code?"** Unit tests against SciPy, the convergence order, the exact decomposition, sealed hashes, and the
  hand-written check script.

---

## 7. Suggested order of work
1. P0-1 to P0-6 text fixes and the poster correction (hours).
2. P1-1 to P1-5 post hoc analyses (they feed P0-2 and P0-6; under a day).
3. P0-8 and P0-9 archive with a DOI (an hour).
4. P0-7 S1 Appendix (days).
5. P2-1 adaptive-in-Sundman comparator, then P2-2 (days of compute; decisive for the "is ML needed" story).
6. Part E: OSF registration → build E1/E2/E6 → dry run → seal → run.
7. Q&A rehearsal and the hand-written check script (ongoing).

## 8. Unverified or outside what I could check
- Current ISEF and NCSEF rules on AI disclosure, adult help, continuation projects and research timelines; also the fair dates.
- PLOS ONE's exact contributions and data-availability requirements (I believe CRediT and a public deposit are required; verify).
- Where the sampling mixture was first fixed.
- Whether S1 Appendix content is drafted elsewhere (e.g. in `CR3BP_Project_Explained.md`).
- The contents of `CR3BP_Rigor_and_Bug_Plan.md`.
- Whether the training sets carry oracle rtols (needed for P2-2).
- My expectation for P2-1's outcome is a guess.
