# Manuscript plan v4: PLOS ONE (LaTeX template, PDF submission)

Plan only: structure, rules and specifications. No manuscript prose.

**v4 changes from v3:** the target is now **PLOS ONE**. Every PLOS rule below was read on 2026-10-29 from journals.plos.org, from plos.org/fees, and from the PLOS LaTeX template v3.6 (August 2022, the official Overleaf copy). SciPost-only items are removed: the SciPost class, `SciPost_bibstyle`, the bold abstract, numbered sections, figures embedded in the PDF, and the funding paragraph in the text. What carries over: the four-part story (A-D), the wording rules, the number provenance, and the figure color code.

**Project sources behind the content** (unchanged from v1):
- Read: `part_d/NARRATIVE.md`, `part_d/RESULTS_PART_D.md`, `part_d/PREREGISTRATION_D.md`, `part_d/references.bib`, `part_c/RESULTS_PART_C.md`, `part_b/RESULTS_PART_B.md`, `part_b/RESULTS_E3C_MU_DIAL.md`, `RESULTS_REPORT.md` (scorecard, §5.7, §13, §15), `LITERATURE_REVIEW.md`, all figures, and the method code.
- Not read in full: the `PART_*_LOG.md` files, `CHANGES.md`, and the Part A-C preregistrations. Read the relevant one before writing any Methods paragraph that depends on it.

---

## 1. PLOS ONE requirements (verified)

### 1.1 Publication criteria (all seven; a reviewer checks each)
1. Original research.
2. Not published elsewhere. Preprints (arXiv), theses and conference presentations are allowed. The paper must not be under consideration elsewhere.
3. Experiments, statistics and analyses done to a high technical standard and described in sufficient detail.
4. Conclusions presented appropriately and supported by the data. Do not overstate. Implications must be identified as hypotheses, not conclusions.
5. Intelligible, in standard English. PLOS ONE does not copyedit; poor language can mean rejection.
6. Meets the standards for research ethics and integrity.
7. Follows reporting guidelines and community standards for data availability.

**Why PLOS ONE suits this paper:**
- "We consider negative and null results."
- "Editorial decisions do not rely on perceived significance or impact."
- Methods and software studies are accepted.

**Related rules:**
- Work "divided into parts" will not be considered, which supports one paper covering Parts A-D.
- A study that replicates or closely resembles earlier work must justify itself and discuss that literature.

### 1.2 Files and format
**What you submit:**
- LaTeX manuscripts are **submitted as a PDF** built from the PLOS template.
- If the paper is accepted, you upload the LaTeX source as **one single .tex file**. The template says not to use `\input` or `\externaldocument`.
- Figures are **not** included in the PDF. They are uploaded separately (section 8).

**Layout (template v3.6):**
- `\documentclass[10pt,letterpaper]{article}`.
- Template geometry: `top=0.85in, left=2.75in`, `\textwidth 5.25in`, `\textheight 8.75in`, `\raggedright`, paragraph indent 0.5 cm. The wide left margin is intentional.
- Do not delete any package listed in the template; other packages are allowed.
- No colors or graphics in the text.

**General rules (submission guidelines):**
- Standard font. Double-spaced text: the template has `setspace`/`\doublespacing` commented out with the note "Remove comment for double spacing", so uncomment it.
- No multiple columns.
- **Page numbers and continuous line numbers.** The template puts page "x/y" in the footer and calls `\linenumbers` after the abstract.
- **No footnotes.** Move their content into the text or the reference list.
- At most **3 heading levels**, written as `\section*`, `\subsection*`, `\subsubsection*` (unnumbered), in **sentence case**.
- English only.
- Define each abbreviation at first appearance. Use non-standard abbreviations only if they appear at least 3 times. Keep abbreviations to a minimum.
- SI units. For non-SI units, give the SI value in parentheses.
- No length limits: no word, figure or supporting-information caps. Be concise.
- Figures: "PLOS One waives all formatting requirements until … provisional Editorial Accept", but the closer the files are to spec, the faster publication goes. Plan to meet the spec from the start.

### 1.3 Manuscript organization (required order)
1. **Title page:** full title, authors, affiliations, corresponding author email(s).
2. **Abstract.**
3. **Introduction.**
4. Middle sections, which may be renamed or reordered: **Materials and methods**, **Results**, **Discussion**, **Conclusions** (optional). Results and Discussion may be combined.
5. **Acknowledgments.**
6. **References.**
7. **Supporting information captions** (if any).

**Placement rules:**
- Figure captions go in the text immediately after the paragraph that first cites the figure.
- Tables go in the text immediately after the paragraph that first cites them.
- Figure files and SI files are uploaded separately.

**Template conflict:** template v3.6 puts "Supporting information" *before* Acknowledgments. The submission guidelines put SI captions *after* References. Follow the submission guidelines (the current text) and note it in the cover letter if asked.

**Items that do NOT go in the manuscript file** (they go in the submission form):
- Funding statement (Financial Disclosure);
- competing interests;
- author contributions (CRediT roles);
- Data Availability Statement (also published with the article).

### 1.4 Title page and abstract
- **Full title:** ≤ 250 characters. Specific, descriptive, concise, understandable outside the field. **Sentence case**, with the first word of a subtitle capitalized after a colon. Avoid specialist abbreviations (no "CR3BP" in the title).
- **Short title:** ≤ 100 characters, stating the topic. It is entered in the submission system.
- **Authors:** first name, middle name or initials, last name. No titles or degrees.
  - Superscript affiliation numbers. Each affiliation is "Dept/Program/Center, Institution Name, City, State, Country". Every author needs one.
  - Corresponding author marked `*` with an email.
  - Equal-contribution and current-address notes use the template's symbols.
  - The corresponding author needs an **ORCID iD** in the submission system.
- **Abstract:** ≤ 300 words. It states the main objective, how the study was done (no methodological detail), and the most important results and their significance. **No citations, and abbreviations avoided if possible.** It is entered in the submission form too.

### 1.5 Introduction / Methods / Results / Discussion (content rules)
- **Introduction:**
  - Background that lets readers outside the field understand the purpose and significance.
  - Define the problem and why it matters.
  - Brief review of the key literature; note relevant controversies.
  - **End with the overall aim and whether it was achieved.**
- **Materials and methods:**
  - Enough detail for suitably skilled investigators to replicate the study.
  - New methods in full.
  - Established methods may be cited, but the paper must stand alone.
  - A **statistical analysis section** that:
    - names each software package and version with references;
    - gives the procedures needed to reproduce the analysis;
    - gives the repository identifier of the code;
    - defines α;
    - describes corrections for multiple comparisons (or justifies none);
    - says whether each test is one- or two-tailed.
- **Results / Discussion / Conclusions:**
  - Relate the results to the hypothesis.
  - Explain the implications relative to previous studies and future directions.
  - Avoid overstatement.

### 1.6 Statistical reporting (PLOS rules that change how numbers are written)
- **Percentages must come with numerator and denominator.** Write "71.6% (8,768/12,253)", not "71.6%".
- **p-values:** exact when ≥ 0.001; below that, "p < 0.001".
- **Spread and centre:** state which measure of spread (SD, SEM, CI) and which centre (mean, median) is shown, in the text and in every figure.
- **Test details:** give test statistics with degrees of freedom where they exist, effect sizes and CIs.
- **Regression analyses:** full results go in a supplementary file: all coefficients, SEs, p-values, CIs, goodness of fit. This applies to the physics rules (linear quantile regression) and the timescale-gap OLS slope.
- **Plots:** depict the sample distribution accurately; no 3D effects.
- **Open data:** individual data points, the data behind every graph, and the summary statistics must be available at publication.

### 1.7 References (Vancouver / ICMJE; template uses `plos2015.bst`)
**In the text:**
- Numbered in **order of first appearance**, in **square brackets**: "[19]".
- The `cite` package (required by the template) sorts and compresses: [3-5].
- Do not cite in the abstract.
- Put the manuscript parts in their final order before numbering.

**Entry format:**
- First **six** authors, then "et al."
- Authors as "Surname Initials" with no periods: "Conant GC, Wolfe KH."
- Title in sentence case.
- Journal abbreviated as in the NCBI (NLM) databases.
- `Year Mon;Vol(Issue):pages.`
- A DOI is allowed alongside or instead of volume/pages, written `doi: 10.xxxx/...`: label plus full DOI, **not** a URL or shortened DOI.

**Formats for specific source types:**
- **Preprints:** `Author(s). Title. arXiv:XXXX.XXXXX [Preprint]. Year [cited Year Mon Day]. Available from: URL`. Preprints are citable if they have a DOI or arXiv URL.
- **Books:** `Author. Title. Edition. City: Publisher; Year.`
- **Book chapters:** `Author. Chapter. In: Editors, editors. Book. City: Publisher; Year. pp. x-y.`
- **Web sites and new media:** `Author/Org. Title. Year Mon Day [cited ...]. In: Site [Internet]. ... Available from: URL`.
- **Databases and repositories** (Zenodo, figshare): `Author. Title; Year [cited ...]. Database: name [Internet]. Available from: URL`. Data citations need a persistent identifier (DOI).

**Not allowed:**
- Unpublished or submitted-but-not-accepted work, "data not shown", personal communications.
- Retracted work, unless discussed as retracted.

**LaTeX procedure (template):** compile the BibTeX database with `plos2015.bst`, then **paste the resulting `.bbl` contents into the .tex** as `\begin{thebibliography}...\end{thebibliography}`. Final submission is one file.

### 1.8 Figures (technical requirements)
| Item | Requirement |
|---|---|
| Format | **TIFF** (preferred; LZW compression **required**, flattened, no layers, **no alpha channel**) or EPS. PLOS does **not** accept vector EPS generated in LaTeX. |
| Width | 789-2,250 px at 300 dpi = **2.63-7.5 in** (6.68-19.05 cm). To align with the text column: **≤ 5.2 in** (13.2 cm). |
| Height | ≤ 2,625 px at 300 dpi = **8.75 in** (22.23 cm), caption excluded |
| Resolution | 300-600 dpi (600 recommended for figures with many charts or small text) |
| Color | RGB 8 bit/channel or grayscale |
| File size | < 10 MB each |
| Text in figures | **Arial, Times or Symbol only, 8-12 pt** |
| Content | no author names, article title, figure number, figure title or caption inside the file |
| White space | 2-pt white border recommended; excess white space cropped |
| Multi-panel | all panels in one file, on one page |
| Orientation | as it should be published |
| Naming | `Fig1.tif`, `Fig2.tif`, ... matching "Fig 1" in the caption and the text |
| Check | run every main figure through PLOS's **NAAS** tool before submission (not SI figures) |
| Copyright | figures must be original; no previously published or copyrighted material without permission; CC BY on publication |

**Captions:**
- The label is "**Fig 1.**" ("Figure" abbreviated to "Fig"; the template sets bold label and period).
- A **title of ≤ 15 words** in bold (`{\bf ...}`).
- An optional legend after it:
  - describe each panel with its letter, (A) or (a);
  - define every non-standard symbol and abbreviation;
  - be succinct, but make the figure understandable without the text;
  - put any credit in the last sentence.
- Cite figures as "Fig 1", "Fig 1A", in ascending order of first appearance.
- In LaTeX, use `\begin{figure}[!h] \caption{{\bf Title.} Legend...} \label{figN} \end{figure}` with **no `\includegraphics`** in the submitted PDF. Use the caption text for panel descriptions, not subfigure commands.

### 1.9 Tables
- Editable, cell-based. No graphics, text boxes, nested tables, or line breaks/spaces used for alignment. Text black only (cell shading allowed, solid and light, contrast ≥ 4.5:1).
- Label "Table 1" plus a brief descriptive title **above** the table. Legend and footnotes **below**.
- Placed right after the paragraph that first cites them; cited in numerical order ("Table 1", "Tables 1-3").
- No strict size limit. Use the template's `adjustwidth` environment for tables wider than the text column; do not split tables.
- A table too complex to be cell-based becomes a figure.

### 1.10 Supporting information
- Any file type, **< 20 MB each**. Named with "S" + number + type: "S1 Appendix", "S1 Fig", "S1 Table", "S1 File", "S1 Data".
- Published exactly as provided: not copyedited, and also deposited on figshare by PLOS.
- Captions are listed at the **end of the manuscript** (not in a separate file). Each caption has the label, a recommended one-line bold title, and an optional legend.
- In LaTeX, cite SI items with `\nameref{S1_Appendix}`, using the template's `\paragraph*{S1 Appendix.}\label{S1_Appendix}` blocks.

### 1.11 Data, code and AI policies
- **Data:** all data needed to replicate the findings (the "minimal data set") must be public at publication. That includes the values behind every mean, interval, graph and statistic.
  - Repository deposit is strongly recommended, with a DOI and a license no more restrictive than CC BY.
  - The Data Availability Statement is given at submission. DOIs may be placeholders until acceptance.
  - "Data not shown" is not allowed.
- **Code:** author-generated code that underpins the findings must be shared without restriction at publication: a repository that issues DOIs, an open-source license, documentation of dependencies, and a mention in the Data Availability Statement.
  - Language-only editing does not need disclosure.
  - The disclosure must name the tools, describe how they were used and **which parts of the study, article, data or files were affected**, and describe **how the authors checked the validity of the outputs**.
  - Location: a **dedicated subsection of Methods**, the Acknowledgments, or the relevant figure legends.
  - Undisclosed or unacceptable use can lead to rejection or retraction.
- **Authorship** (McNutt et al. 2018 criteria, adopted by PLOS in May 2023). Each author must meet all three:
  1. a substantial contribution to conception/design, data, software, or drafting/revising;
  2. approval of the submitted version;
  3. accountability.

  Contributions are given as CRediT roles in the form. Everyone else goes in Acknowledgments, with their consent. Do not name PLOS editors or staff there.
- **Related manuscripts:** if related work is under consideration or published elsewhere, it must be declared, a copy submitted, and the relation described. This applies to the NHSJS 2D paper (section 2.3).

### 1.12 Other submission items
- **Cover letter, ≤ 1 page, uploaded separately**, covering:
  - the contribution;
  - the relation to prior work;
  - the article type (Research Article);
  - any prior interactions with PLOS;
  - suggested Academic Editors;
  - opposed reviewers.
  - No fee-waiver requests in the letter.
- **Funding:** in the form only. If unfunded, use this exact sentence: "The author(s) received no specific funding for this work."
- **Competing interests:** in the form only.
- **Preprints** are welcome (arXiv is fine).
- **Optional "striking image":** derived from a figure.
- **Publication fee:** PLOS ONE "All other articles" (Research Articles) **USD 2,477**, payable on acceptance (plos.org/fees, read 2026-10-29). Fee assistance and institutional partnerships exist and are requested in the submission system, not in the letter.

---

## 2. Decisions and risks before writing

### 2.1 Cost and fee assistance
The USD 2,477 fee is due on acceptance. Check:
- whether the student's school or any affiliated institution is a PLOS institutional partner;
- whether to apply for PLOS fee assistance (requested in the submission system).

Decide this before submitting, since the financial disclosure cannot be changed after submission without approval.


### 2.3 Related and prior work
- If the NHSJS 2D paper is under review or published: declare it as related, upload a copy, describe the relation, and cite it if published. Keep the text non-overlapping.
- Do not submit to any other journal while under review at PLOS ONE.
- A science fair entry or an arXiv preprint is fine.

### 2.4 Academic Editor fit
PLOS ONE's board covers physical sciences, but astrodynamics and celestial mechanics editors may be few. Before submitting:
- search the Academic Editor list for numerical analysis, computational physics, celestial mechanics, dynamical systems and machine learning;
- name 2-4 in the cover letter, with no recent collaborators and no one from the same institution.

### 2.5 Carried over from v1-v3
- **Paper scope:** one paper covering Parts A-D. Secondary results go in S1 Appendix.
- **The 2D study:** leave it out, except as related work (2.3).
- **Headline numbers:**
  - Part C confirmation is the primary test.
  - D3 is the replication.
  - The lock runs go in a table.
  - Overheads are stated per machine (15.04 vs 16.09 RHS-equivalents).
- **Must be reported, not buried:**
  - C3(a) did not replicate (50.56%, p = 0.11).
  - D2 throughput 0.885 (not supported).
  - D5: 70% violations on the JPL families.
  - The linear model ties the trees.
  - The label is endpoint-only (2.6% of runs exceed the tolerance mid-run).
  - About 19% of draws (reference-divergent) are excluded from the cost tests.
  - The CR3BP differs from the real ephemeris by 2,000-50,000 km.
- **Not used:** the Richardson halo validation (flagged earlier).
- **Read before citing:** the Dellnitz et al. description conflicts between `LITERATURE_REVIEW.md` and `references.bib`.
- **Preregistration wording:**
  - The studies were preregistered internally (sealed with SHA-256), not as PLOS Registered Reports. Say "preregistered (sealed before the confirmatory data were analyzed)" and deposit the preregistration documents with the code.
  - Do not call the paper a Registered Report.

---

## 3. Claims and wording rules

| # | Claim | Evidence | Status |
|---|---|---|---|
| C-1 | A fixed physical step predicted from the initial state never cost less than an adaptive solver once feature and model costs were counted; FTLE cost ~70 times a DOP853 solve | Part A H1/H1'a/H3 | preregistered |
| C-2 | The gap (+1.41 dex) splits into paradigm +0.21, selection +0.03, prediction +0.38, safety scale +0.30, overhead +0.50; the paradigm term tracks the timescale range (ρ = 0.909) and depends on how the adaptive comparator is tuned | Part B T1, T4, T5 | preregistered |
| C-3 | Sundman time + periapsis features + ~15 RHS-equivalent model: cheaper than equal-risk RK45 on 71.4% (8,705/12,192; verify the numerator) at 4.39% (535/12,192) violations; timescale dependence removed (ρ 0.225 vs 0.911) | Part C C1, C3, C4, C5 | preregistered |
| C-4 | Replicates: 71.56% (8,768/12,253), 2.69%; C3(a) does not | D3 | preregistered |
| C-5 | Most of the saving comes from the time transformation plus analytic periapsis information; a 4-coefficient rule is 0.072 dex behind; the linear model ties | D1, C5, D6 | D1 preregistered; D6 exploratory |
| C-6 | A per-stratum certificate keeps all 17 strata ≤ 3.1%, while still cheaper than equal-risk RK45 on 66.4%; ~5% remains at close passes | D4 (D4b exploratory) | preregistered |
| C-7 | Not faster in compiled wall-clock over 1-20 TU (0.885); faster at 1-3 TU (1.61); 70% violations on JPL periodic families | D2, D5 | preregistered, not supported |
| C-8 | FTLE explains failures post hoc (AUC 0.92-0.93) and cannot be used before the run | D7, D12 | exploratory |

**Wording rules:**
- Say "cheaper than RK45 at equal (marginal) risk" or "at a certified tolerance". Never "faster than adaptive solvers".
- Say "integrator error inside the circular restricted three-body problem". Never "accurate trajectories".
- State that the policy is not cheaper than DOP853 or the oracle-tuned RK45.
- Every percentage has its numerator and denominator (1.6).
- Implications are labelled as hypotheses (criterion 4).

**Numerators not yet extracted:** every percentage must get its counts from the result JSON/ledgers via the number script (section 10). Do not compute them by hand from the rounded percentages.

---

## 4. Title, short title, abstract

### 4.1 Title (≤ 250 characters, sentence case, no specialist abbreviations)
1. "Why a single predicted time step fails in the Earth–Moon three-body problem, and how a Sundman time transformation recovers most of the cost" (140). **Recommended.**
2. "Pre-simulation step-size prediction for the spatial Earth–Moon three-body problem: Failure in physical time, recovery in Sundman time, and limits" (145).

### 4.2 Short title (≤ 100 characters)
- "Predicting integration step sizes in the three-body problem" (59), or
- "Step-size prediction in the Earth–Moon three-body problem" (57).

### 4.3 Abstract (≤ 300 words; no citations; avoid abbreviations; aim for 250-280)
| Move | Content | Words |
|---|---|---|
| Background and problem | many trajectory integrations for missions near the Moon; adaptive solvers as the default; whether one step chosen before the run can be both cheap and reliably safe | ~50 |
| Objective | the aim, stated as the question | ~25 |
| How it was done | about 100,000 simulated trajectories; 100 km final-position tolerance against a high-accuracy reference; cost as force evaluations including model cost; four preregistered studies on independent data; statistically certified failure rate | ~50 |
| Main results | with counts where space allows: fixed physical step never cheaper; why (cost-gap decomposition; penalty tracks the range of timescales); Sundman time plus analytic periapsis information cheaper than an equal-risk adaptive Runge–Kutta solver on about 72% of new trajectories; a four-coefficient rule gets most of the way; not faster in compiled wall-clock overall; 70% failures on periodic-orbit families | ~110 |
| Significance | what drives the cost; where pre-run step selection is usable (short horizons, inside the training population) and where it is not | ~40 |

- Spell out "finite-time Lyapunov exponent" and "circular restricted three-body problem" if used. Better: avoid them in the abstract.
- The abstract is pasted into the submission form too.

---

## 5. Section plan

```
Title page
Abstract
Introduction
Materials and methods
  Dynamical model and timescales
  Trajectory sampling and datasets
  Reference solutions and step-size labels
  Integrators
  Features and predictive models
  Failure-rate certification
  Cost accounting and decomposition of the cost gap
  Wall-clock timing protocol
  Periodic orbits and chaos maps
  Preregistration and statistical analysis
  Use of AI tools                                   (required disclosure, 2.2)
Results
  A fixed physical step never cost less than adaptive integration
  The cost gap and the timescale range
  Sundman time removes the timescale penalty
  Replication, and what carries the information
  Failure rates near the Moon and at long horizons
  Compiled wall-clock time
  Periodic orbits and chaos
Discussion
Conclusions
Acknowledgments
References
Supporting information (captions only)
```
- These headings are placeholders describing content. The student writes the final headings in sentence case, with at most 3 levels.
- Figures and tables are placed by their captions/tables after the paragraph that first cites them (1.3).

### 5.1 Introduction (≈ 900-1,100 words)
| ¶ | Content | References needed |
|---|---|---|
| 1 | Why many trajectory integrations are needed (cislunar mission design, Monte Carlo dispersion); the circular restricted three-body problem as the standard model, explained for readers outside the field | CR3BP texts (Szebehely; Koon, Lo, Marsden & Ross); a cislunar mission source |
| 2 | What makes integration hard: close passes compress the local timescale; chaos; fixed vs adaptive step control; conserved quantities vs trajectory accuracy | Dormand & Prince; Hairer et al.; Rein & Spiegel; Pham, Rein & Spiegel; Boekholt & Portegies Zwart |
| 3 | Time transformations and regularization near bodies | Sundman; Mikkola & Tanikawa; Preto & Tremaine; Berry & Healy; Wang & Nitadori |
| 4 | Learned step control and its baselines; the controversy over weak baselines in ML for numerical methods (PLOS asks for controversies to be noted) | Saz Ulibarrena & Portegies Zwart; Dellnitz et al. (after reading); McGreivy & Hakim |
| 5 | The gap: pre-simulation, per-trajectory step selection with certified risk and full cost accounting, in 3D; why preregistration | Nosek et al.; Angelopoulos et al. |
| 6 | Hypothesis: the original one (FTLE, proximity and the Jacobi constant govern difficulty) and the timescale-range hypothesis that followed from Study A | — |
| 7 | **Overall aim and whether it was achieved** (PLOS requirement): aim partly achieved: cheaper than equal-risk RK45 within distribution; not faster overall in wall-clock; unsafe out of distribution | — |

### 5.2 Materials and methods (≈ 2,500-3,000 words; replication-grade)
**Display equations:**
- Use `equation` or `eqnarray`/`align`. Cite them as "Eq 1".
- PLOS template rules for math:
  - whole expressions go inside math mode;
  - no punctuation inside inline math;
  - `\textsubscript` for non-math text;
  - `\mathcal`, not `\cal`;
  - group super/subscripts outside brackets in braces;
  - break long display equations to fit the 5.25 in column.

**Equations planned:**
1. Equations of motion (rotating frame).
2. Jacobi constant; Hill regimes I-V from C vs C_L1..C_L4.
3. Local timescale τ and timescale range δ (`diagnostics_3d.py`).
4. Sundman transformation dt = g(x) ds, g = (1/τ₁ + 1/τ₂ + 1/τ_c)⁻¹, augmented system (`sundman_3d.py`).
5. Osculating periapsis radius and time (`osculating_3d.py`).
6. Certification rule (fixed-sequence binomial test).
7. Cost-gap decomposition (`failure_study_3d.decompose`).

**Content by subsection:**
- **Model:**
  - μ = 0.01215058535.
  - Units: DU = 384,400 km; TU ≈ 4.348 d (give the value in seconds; SI rule).
  - Collision radii 6,371 km and 1,737.4 km; escape radius 10 DU.
- **Sampling and datasets:**
  - Mixture 55/20/25 (Lagrange-point neighborhoods ±0.1 DU / ±0.15; wide box; near-Moon log-uniform 1.2 lunar radii to 0.15 DU); enrichment weights.
  - Horizons log-uniform over 1-20 TU.
  - Datasets in **Table 1**.
- **Reference and labels:**
  - DOP853 at 10⁻¹³ with maximum step 0.01.
  - Failure definition.
  - Monotone label (2^(1/4) grid, 8-point window); non-monotone fractions.
  - Reference checks.
- **Integrators:**
  - RK4 and Sundman-time RK4 (τ_c = 2, landing in physical time).
  - Leapfrog in Studies A and B only.
  - SciPy RK45/DOP853 (atol = 10⁻³ rtol; cost = function evaluations).
  - Compiled Dormand-Prince 5(4) with PI controller, and compiled DOP853 for timing.
- **Features and models:**
  - State, horizon and 13 periapsis features (full list in S1 Appendix).
  - FTLE explanatory only.
  - Final model: histogram gradient-boosted quantile regressor (25 iterations, depth 3, learning rate 0.2, quantile 0.2).
  - Earlier models briefly.
  - Physics rules (linear quantile regression).
  - Overhead measurement.
- **Certification:**
  - Fixed-sequence test (grid 0.1·2^(k/8), α = 0.05, δ = 0.1), 4,000 calibration trajectories.
  - Stratified version (17 strata, Bonferroni δ/G, abstention).
  - Comparator tolerances: per-trajectory oracle, certified, equal-risk.
- **Cost accounting:** right-hand-side evaluations; the decomposition; "dex" defined once (or replaced by cost ratios in text).
- **Timing:**
  - Apple M5, single thread, identical right-hand-side arithmetic in both arms, minimum of 3 repeats.
  - Idle guard and its dated deviation.
  - Numba version.
- **Periodic orbits:** JPL Three-Body Periodic Orbits API; 11 families; 50 members per stream; closure < 10⁻⁸; horizons; trusted-reference rule; JPL mass ratio.
- **Chaos maps:** surface of section, 110 × 110 grid, three Jacobi constants, T = 8 TU, variational FTLE.
- **Statistical analysis (PLOS-required content):**
  - Software and versions: Python 3.11.16, NumPy, SciPy, scikit-learn, Numba 0.67, REBOUND 5.2.2, Matplotlib. Read the rest from the conda environment, and cite each.
  - α = 0.05.
  - Tests are one-sided, with the direction fixed in the sealed preregistrations (give the dates).
  - Holm-Bonferroni within each study's family of five tests.
  - Exact binomial tests and Clopper-Pearson intervals.
  - Orbit-level bootstrap (4,000 resamples; state the p-value floor).
  - Permutation tests (Study B); Spearman correlations.
  - Units of analysis (independent trajectories; repeated horizons of one periodic orbit treated as a cluster).
  - Handling of excluded data: reference-divergent trajectories, untrusted references.
  - Confirmatory vs exploratory labelling.
  - Code repository DOI.
- **Use of AI tools:** section 2.2.

### 5.3 Results (≈ 3,000-3,500 words)
Each subsection follows this pattern:
1. why the analysis was done;
2. what was done (one sentence, pointing to Methods);
3. the result with estimate, 95% CI, n/N, test and p (Holm-adjusted where relevant);
4. "Fig N" / "Table N";
5. a short interpretation, related back to the hypothesis.

Label exploratory results in the sentence.

| Subsection | Numbers to report | Fig / Table | Source |
|---|---|---|---|
| Fixed step | 0.0% cheaper vs all four adaptive settings (0/N each); 96.3% within tolerance; FTLE ~70 times DOP853 (n = 80, development); ablation CIs include 0; H1′ marked as post hoc | Table 2 | `RESULTS_REPORT.md` §1.1, §3.4, §3.7, §6 |
| Cost gap | decomposition (+1.41 dex; terms with CIs); T1 25.8% (n = 14,160), ρ = 0.909, slope 0.904 (95% CI 0.898-0.910); T4 contrasts; T5 dial; 67.9% vs certified RK45; T3 not supported; step-over AUC 0.85 | Fig 2A, Fig 3, Fig 4 | Part B §2-3 |
| Sundman | C3(b) 0.225 vs 0.911; paradigm term -0.005; total +0.21; C4 71.4% (n/N), violations 4.39% (535/12,192); C1; C5; C2 not supported; regime × horizon | Fig 2B, Fig 3, Fig 5 | Part C §2-5 |
| Replication | D3 71.56% (8,768/12,253), 2.69%; C3(a) 50.56%, p = 0.11; locks; D1 0.0715 dex (0.0695-0.0736), 69.5% of 11,565 paired; rules; linear tie; D6 (exploratory) | Fig 6, Table 3 | Part D §3, §5, §8, §9 |
| Failure rates | one-scale: 13.1% at min r₂ < 0.05 DU, worst stratum 10.4%; stratified ≤ 3.1% (largest upper bound 4.9%), 66.4%; close passes 5.1%; comparator unevenness; D4b and D12 (exploratory) | Fig 7 | Part D §6, §9 |
| Wall-clock | 0.885 (0.862-0.908); 78.2% of 12,239 faster; 1.61 / 1.02 / 0.75 by horizon band; ns per RHS; variant R 0.595; batches | Fig 8 | Part D §4 |
| Periodic orbits and chaos | 70.0% (67.8-72.3), 1,647 runs; step error 0.3-0.5 dex on vertical/axial/Lyapunov; recalibration 3.5% but never cheaper; comparators unsafe on unstable members; FTLE AUC 0.93/0.92/0.93; decile profile (exploratory; out-of-distribution sections) | Fig 9, Fig 10 | Part D §7, §9 |

### 5.4 Discussion (≈ 1,000-1,200 words)
1. **Relation to the hypothesis.** The original hypothesis held only post hoc; the timescale-range hypothesis was supported, and the intervention removed the dependence.
2. **Mechanism.** Target spread 0.33 vs 0.91 dex; periapsis timing as the missing information. Label it as interpretation where it was not tested.
3. **Relation to previous studies:**
   - in-run reinforcement-learning step control;
   - IAS15 (S1 Appendix result);
   - regularization literature;
   - the weak-baseline critique.
4. **Limitations, ranked by likely impact:**
   - (a) distribution shift (D5);
   - (b) one system, one tolerance, 1-20 TU;
   - (c) synthetic sampling mixture, so percentages are mixture-specific;
   - (d) endpoint-only label;
   - (e) excluded reference-divergent trajectories;
   - (f) one machine and compiler for timing, no GPU;
   - (g) CR3BP vs ephemeris (2,000-50,000 km);
   - (h) oracle-tuned RK45 and DOP853 not beaten.
5. **Implications, labelled as hypotheses** (criterion 4): short-horizon dispersion studies inside a known population, with abstention to an adaptive solver.

### 5.5 Conclusions (optional in PLOS; recommended, ≈ 300-400 words)
- Summary.
- Reach and limits.
- Future work:
  - tolerance and horizon dial;
  - training on periodic families;
  - abstention;
  - Kustaanheimo-Stiefel regularization;
  - ephemeris model;
  - other mass ratios;
  - path-error label;
  - GPU batches.

### 5.6 Acknowledgments
- People who helped but are not authors, with their consent.
- Data sources: JPL Solar System Dynamics API; JPL Horizons.
- **No funding information here** (it goes in the form).
- No PLOS editors or staff.
- The AI disclosure can sit here instead of in Methods; choose one location and use it consistently.

---

## 6. Supporting information plan (each file < 20 MB; captions at the end of the manuscript)

| Item | Content |
|---|---|
| S1 Appendix | Secondary methods and results, as a PDF made from LaTeX:<br>• reference validation (Radau, DOP853 at 10⁻¹⁴, LSODA);<br>• label construction and non-monotone labels (D10);<br>• Sundman landing rule and τ_c sweep;<br>• periapsis feature formulas;<br>• certification algorithm and strata merging rule;<br>• IAS15 baseline;<br>• leapfrog symplecticity check;<br>• doomed-trajectory ledger;<br>• endpoint audit;<br>• μ dial and lookahead (Study B exploratory);<br>• Study A secondary hypotheses (H2, H4-H7), Sun–Earth transfer;<br>• preregistration deviations log |
| S1 Table | All 17 strata (D4): scale, n, violation rates with Clopper-Pearson CIs, share cheaper |
| S2 Table | JPL families by stability index and number of periods, plus extrapolated horizons (D5) |
| S3 Table | Full regression results for the physics rules and the linear model (coefficients, SE or bootstrap CI, goodness of fit), as PLOS requires |
| S4 Table | All preregistered tests across Studies A-D with raw statistics (if Table 2 in the main text is condensed) |
| S1 Data | Processed per-figure data: the values behind every plot, as CSV (PLOS: "values used to build graphs"). One file per figure inside a .zip under 20 MB, or deposited with the code |
| S1 Fig, S2 Fig, ... | Secondary figures: Study B where-map third panel, information-floor, step-over, horizon dial, the other two FTLE sections, batch timings, μ dial |

- Captions are `\paragraph*{S1 Table.}\label{S1_Table}` + `{\bf Title.}` + an optional legend, in the "Supporting information" section after the References.
- Cite them in the text with `\nameref{S1_Table}`. SI citations need not be in numerical order.

**Data Availability Statement (form) — content plan:**
- Code repository: GitHub archived to Zenodo with a DOI, under an open-source license (MIT or BSD), with documentation.
- Sealed preregistrations and freeze manifests.
- Datasets: either the ~500 MB of generated data on Zenodo (check size limits), or the seeds and generator plus SHA-256 hashes, with the per-figure processed data in S1 Data.
- The JPL API responses, saved raw with hashes.
- Placeholder DOIs are allowed at submission.

---

## 7. Citations: exact procedure

### 7.1 In the text
- Use `\cite{key}`, which gives [n]; multiple keys in one `\cite{a,b,c}` give [3-5] (`cite` package).
- Put a non-breaking space before the citation and place it before punctuation, consistently: `... the Jacobi constant~\cite{szebehely1967}.`
  - The template mixes both placements; choose one and use it everywhere.
- Figures: `Fig~\ref{fig2}` and "Fig 2A" (panel letter typed after the reference). Equations: `Eq~(\ref{eq:sundman})`. Tables: `Table~\ref{table1}`. SI: `\nameref{S1_Appendix}`.
  - Never write "Figure" or "Equation" in running text.

### 7.2 The .bib file and the .bbl
1. Keep `references.bib` (built by hand from opened sources) as the working database.
2. Compile with `\bibliographystyle{plos2015}` and `\bibliography{references}` while drafting.
3. **Before submission, paste the generated `.bbl` contents into the .tex** in place of `\bibliography{}`, so the file is self-contained.

**Field rules:**
- `author` in BibTeX "Last, First" form. The bst outputs "Surname Initials" and applies the six-authors-then-et-al rule. **Check this on a compiled entry with more than six authors.**
- `title` with braces on capitals that must stay: `{S}undman`, `{E}arth`, `{M}oon`, `{H}amiltonian`, `{IAS15}`, `{REBOUND}`, `{R}unge`, `{K}utta`, `{L}yapunov`.
- `journal`: the NLM Catalog abbreviation, without periods (e.g., the NLM form of each astronomy and physics journal). Where a journal is not in the NLM Catalog (many physics journals are not), use the ISO 4 abbreviation without periods. That fallback is my recommendation; PLOS names only NCBI.
- `year`, `month`, `volume`, `number`, `pages`, and `doi`.

**Verify on compiled output:**
- that `plos2015.bst` prints the DOI in the PLOS form `doi: 10.xxxx/...`. If it does not, add it via the `note` field in that exact form. I could not open the .bst, so this is unverified;
- that journal titles are not italicized and titles are in sentence case;
- that every DOI resolves.

**Entry types:**
- **Preprints:** `@misc` with `note = {arXiv:XXXX.XXXXX [Preprint]. Year [cited Year Mon Day]. Available from: https://arxiv.org/abs/XXXX.XXXXX}`, or hand-edit the .bbl entry into the PLOS preprint form.
- **Web resources** (JPL APIs): hand-format to the PLOS "new media" pattern in the .bbl: `Title. Year [cited Year Mon Day]. In: Site [Internet]. Available from: URL`.
- **Data and code deposits** (Zenodo): the PLOS data-citation form with DOI, e.g. `Authors. Title. Year. Zenodo. https://doi.org/...`.
- **Software:** cite the reference article each project asks for (NumPy, SciPy, scikit-learn, Numba, Matplotlib, REBOUND), with the DOI. Versions go in Methods.

**Illustration** (fields from `part_d/references.bib`, record opened; the issue is not recorded there, so look it up). Target output shape:
`Preto M, Tremaine S. A class of symplectic integrators with adaptive timestep for separable Hamiltonian systems. Astron J. 1999;118(5):2532–2541. doi: 10.1086/301102` (the "(5)" is a placeholder to verify, and "Astron J" must be checked against the NLM Catalog).

### 7.3 Reference inventory (target 40-60; PLOS requires every cited source to be relevant, and "no irrelevant sources")
- **Verified record in `part_d/references.bib`:** Preto & Tremaine 1999; Wang & Nitadori 2020; Mikkola & Tanikawa 1999; Berry & Healy 2002; Rein & Spiegel 2015 (do not quote "~100 steps per orbit"); Rein & Liu 2012; Pham, Rein & Spiegel 2024; Boekholt & Portegies Zwart 2015; McGreivy & Hakim 2024; Angelopoulos et al. 2021; Dellnitz et al. 2023; Saz Ulibarrena & Portegies Zwart 2025; Nosek et al. 2018; Dormand & Prince 1980; JPL Periodic Orbits API.
- **In the bib but not verified:** Strobl et al. 2008; Hairer, Nørsett & Wanner 1993; Short, Howell & Tricoche 2011 (author order uncertain).
- **Needed** (find, open, record the DOI):
  - Szebehely; Koon, Lo, Marsden & Ross;
  - Sundman's original paper;
  - a cislunar mission source;
  - FTLE/LCS in the CR3BP (verified);
  - Holm 1979; Clopper & Pearson 1934; Efron 1979; Koenker & Bassett 1978; Friedman 2001;
  - software articles;
  - McNutt et al. 2018 is **not** needed in the text (it is PLOS policy, not science);
  - JPL Horizons;
  - Kustaanheimo-Stiefel regularization;
  - any prior work on step-size or integrator selection for orbital problems found during the search.
- **Do not cite** the unverifiable 2026 items or the author-less entries in `LITERATURE_REVIEW.md`.

---

## 8. Figures: specification and build instructions

### 8.1 Global style (every main figure)
| Property | Setting (PLOS limits in brackets) |
|---|---|
| Size | Two widths: **column 5.2 in** (align with text) and **full 7.5 in** [2.63-7.5 in]; height ≤ 8.75 in. Design at the final size; never rescale afterwards. |
| Resolution | **600 dpi** [300-600]: 5.2 in → 3,120 px; 7.5 in → 4,500 px (above PLOS's 2,250 px "maximum width at 300 dpi" table value but within the dpi limit). If NAAS objects, export at 300 dpi instead (1,560 / 2,250 px). |
| Format | **TIFF, LZW compression, RGB 8-bit, flattened, NO alpha channel**. Matplotlib writes RGBA TIFFs by default, so convert to RGB (e.g., Pillow `convert("RGB")`) before saving with `compression="tiff_lzw"` and `dpi=(600, 600)`. Check each file is < 10 MB. |
| Font | **Arial** for all text (macOS ships it; set it as the Matplotlib font family; math text via `mathtext.fontset = "custom"` with Arial roman/italic, or plain-text labels) [Arial/Times/Symbol only] |
| Font sizes at final size | axis labels 10 pt; tick labels 9 pt; legend 9 pt; in-panel annotations ≥ 8 pt; panel letters 12 pt bold [8-12 pt] |
| Lines and markers | lines 1.25 pt; reference lines 0.75 pt dashed; markers 4-5 pt; error bars 0.75 pt with 2 pt caps; left and bottom spines only |
| No text other than data labels | no figure number, title, caption or author names in the file (all go in the caption) |
| Border | 2-pt white border; crop other white space (`bbox_inches="tight", pad_inches=0.03`) |
| Panels | all in one file; letters **A, B, C** (uppercase, matching "Fig 2A" citations) at the upper left of each panel |
| Axes | "name (unit)": "Horizon (TU)", "Closest lunar approach (DU)", "Cost (right-hand-side evaluations)". Define DU and TU in the caption. Log axes: log scale with decade ticks. Panels comparing one quantity share the y-range. |
| Intervals | 95% CIs: Clopper-Pearson for proportions, orbit bootstrap for differences; IQR bands for binned medians. The caption states which, plus n. |
| Plots | show distributions honestly (PLOS): hexbin or scatter plus binned medians instead of bare means where the spread matters; no 3D effects |
| Dataset and comparator names | plain names defined in Methods and Table 1 (e.g., "confirmation set", "replication set", "RK45 at equal risk", "oracle-tuned RK45"); no internal codes (`D_confirm_D`, `RK45-eqr`) |
| Check | run every main figure through **NAAS**; then open at 100% and confirm 8-pt text is legible |

**Color and symbol code** (Okabe-Ito, colorblind-safe; the same everywhere; every distinction is also carried by marker, line style or hatching):
| Quantity | Color | Marker / line |
|---|---|---|
| Sundman-time policy | #0072B2 blue | ● solid |
| Fixed physical step (Study A/B policy; oracle fixed step) | #555555 grey | ■ dashed |
| RK45 | #E69F00 orange | ▲ |
| DOP853 | #CC79A7 reddish purple | ◆ |
| One scale for all (marginal certificate) | #56B4E9 sky blue | ○ |
| Physics rules | #000000 black | □ |
| Linear model | #009E73 bluish green | ▽ |
| Decomposition terms | paradigm #0072B2, selection #999999, prediction #E69F00, safety scale #56B4E9, overhead #CC79A7, rounding #DDDDDD, total #000000; hatching on each | |
| Heatmaps | `cividis`, with the value and n printed in cells | |
| FTLE maps | `cividis`; failures as open #D55E00 vermillion circles | |

**Captions (in the .tex, after the first-citing paragraph):**
- `\caption{{\bf <Title, ≤ 15 words, stating what the figure shows>.} (A) ... (B) ... <data set and n> <what the intervals are> <test and p if any> <abbreviations defined>}`.
- Aim for 60-150 words. The figure must be understandable without the text. The student writes every caption.
- No `\includegraphics` in the submitted PDF.
- **Drafting tip:** keep a `\newif\ifdraftfigs` switch to show the figures while writing, and set it false for the submission PDF.

### 8.2 Figure list (main text; SI figures in section 6)
The student writes one plotting script per figure plus a shared style module that encodes 8.1. All scripts are deposited with the code (data policy). The existing `make_figures_*.py` show column names only; rebuild everything to this spec.

| Fig | Width | Panels | Data | Changes from the current figure |
|---|---|---|---|---|
| 1 | 7.5 in | (A) rotating-frame x-y plane: Earth, Moon, L1-L5, one sampled trajectory with a close lunar pass; (B) local timescale τ(t) along it (log axis), with the fixed step it would need and the Sundman step | one trajectory with min r₂ < 0.05 DU from the Study C ledger `out_c/confirm/sund/apply_policy_20261027T013754Z/ledger.csv`, re-integrated with the reference integrator | new; a computed technical drawing, not an illustration |
| 2 | 7.5 in | (A) Study B: hexbin of log₁₀ timescale range vs log₁₀(oracle fixed-step cost / oracle RK45 cost), binned medians + IQR; (B) Study C: fixed physical vs fixed Sundman step on the same trajectories, same bins | `part_b/results/confirm/policy/ledger.csv.gz`; Study C ledger | (B) from `paradigm.png` without its title; (A) new |
| 3 | 7.5 in | (A) Study B and (B) Study C cost-gap decomposition, same y-axis, 95% bootstrap CIs | Study B ledger + `decompose`; Study C ledger + `decompose_sundman` | from `bars.png`; add CIs; plain term names |
| 4 | 7.5 in | (A) median paradigm gap, (B) failure rate by Hill regime × closest-approach band, values and n in cells | Study B ledger | from `wheremap.png`; third panel to SI |
| 5 | 7.5 in | (A) share cheaper than each of 6 comparator settings for 3 policies, Clopper-Pearson CIs, replication marker; (B) regime × horizon heatmap vs RK45 at equal risk | Study C `apply_policy_report.json` and ledgers; `part_d/results/confirm/d3_sund_apply_policy_report.json` | from `wins.png` and `cells.png`; label the 0% bars |
| 6 | 7.5 in | (A) extra cost of each rule, the linear model and one global step relative to the ML policy (95% CI); (B) drop-group refit importance, Sundman vs physical-time target (exploratory, labelled) | `part_d/results/confirm/d1_report.json`; `part_d/results/explore/d6_report.json` | from `d1.png`; (B) new |
| 7 | 7.5 in | (A) failure rate per free stratum (17 rows; stratified vs one scale vs RK45 equal-risk); (B) by closest approach actually made | `part_d/results/confirm/d4_table_{strat,marginal}_by_{stratum,r2_band}.csv` | from `d4.png`; readable stratum labels with a key in the caption; at 9 pt, 17 rows need ~5 in of height: check legibility |
| 8 | 7.5 in | (A) compiled throughput ratio by horizon band and overall, 3 comparators; (B) batch throughput vs batch size, 1 and 10 threads | `part_d/results/confirm/d2_report.json` | from `d2.png`; two legend blocks (color = method, line style = threads) |
| 9 | 7.5 in | (A) failure rate per JPL family: policy (CI), RK45 certified, RK45 equal-risk; (B) median log₁₀(policy step / oracle step) per family | `part_d/results/confirm/d5_table_by_group.csv`, `d5_report.json` | from `d5.png`; mark the 10% test level |
| 10 | 7.5 in | (A) one FTLE surface of section (C = 3.10) with failures; (B) failure rate by FTLE decile for the three sections | `out_d/explore/d7_20261028T211057Z/slice_C{3.19,3.1,2.95}.csv` | from `d7.png`; legend off the data; the other two sections as S Fig |

If the paper is long, Fig 4 and Fig 8 can move to SI. They are the least central.

### 8.3 Figure checklist (per figure)
- [ ] TIFF, LZW, RGB without alpha, flattened, 300-600 dpi, < 10 MB; width 5.2 or 7.5 in; height ≤ 8.75 in
- [ ] Arial only, 8-12 pt at final size; no title, caption or figure number inside the file
- [ ] Panel letters A, B, ... match the caption and the in-text citations
- [ ] Axis names with units; DU and TU defined in the caption
- [ ] Intervals and n stated in the caption; colors and markers per the code; legible in grayscale
- [ ] File named `FigN.tif`; caption "Fig N." with a bold title of ≤ 15 words; placed after the first-citing paragraph
- [ ] NAAS check passed; data behind the plot included in S1 Data
- [ ] If AI helped prepare the figure, the legend says so (PLOS rule)

---

## 9. Tables
**Template style:**
- `tabular` with `\hline`/`\thickhline` from the template (or `booktabs`, which is allowed since the template only forbids deleting its packages).
- No nested tabulars, no `\\` or spacing inside cells to force layout, black text only (light cell shading allowed).
- `adjustwidth{-2.25in}{0in}` for tables wider than the 5.25 in column, as in the template example.

**Caption and placement:**
- `\caption{{\bf Table title.}}` **above** the tabular.
- Notes in a `flushleft` block **below** it, defining every abbreviation and symbol.
- Placed right after the first-citing paragraph.

**Numbers:** every percentage with its numerator/denominator, either in a separate n column or "k/N" in parentheses. CIs as "71.6 (70.8–72.4)".

| Table | Content | Source |
|---|---|---|
| 1 | Datasets: name used in the paper, study, seed, candidate trajectories, usable trajectories, role | preregistrations B, C, D; `RESULTS_REPORT.md` §2.2 |
| 2 | Preregistered confirmatory tests, Studies A-D: test, hypothesis (short), statistic with 95% CI and k/N, one-sided, Holm-adjusted p, verdict | `confirm_tests_*.json`; `RESULTS_REPORT.md` §1.1 |
| 3 | Headline across datasets (Study C confirmation, D3 replication, Study C lock, Study D lock): failures k/N (CI); share cheaper vs RK45 equal-risk / certified / oracle and DOP853 equal-risk / certified (k/N); median total cost; model overhead | Part C §2, §7; Part D §5, §8 |

---

## 10. LaTeX workflow (PLOS: one .tex, PDF submission, figures separate)

```
manuscript/
  plos_latex_template.tex     % official v3.6 (download the zip from journals.plos.org/plosone/s/latex)
  plos2015.bst                % from the same zip
  main.tex                    % ONE file: copy of the template, filled in; all macros in the template's "MACROS" block
  references.bib              % working database (not submitted; its .bbl is pasted into main.tex)
  figures/Fig1.tif ... Fig10.tif
  si/S1_Appendix.tex -> S1_Appendix.pdf; S1_Table.xlsx/csv ...; S1_Data.zip
  scripts/
    style.py                  % section 8.1
    fig01.py ... fig10.py
    make_numbers.py           % writes the \newcommand number block between markers in main.tex's macro section
    make_tables.py            % writes the table bodies between markers in main.tex
    check_submission.py       % title ≤ 250, short title ≤ 100, abstract ≤ 300 words and no \cite; every \cite key in the .bbl; no \includegraphics when final; Fig/Table cited in order; every percentage followed by (k/N); TIFF checks (mode RGB, no alpha, LZW, dpi, size, < 10 MB); no footnotes; heading depth ≤ 3
```

**Template rules to keep:**
- Do not delete any template package.
- No `\input`. If sections are drafted in separate files, merge them into `main.tex` before building the submission PDF.
- `\linenumbers` after the abstract; page "x/y" footer.
- Uncomment `\doublespacing`.
- Unnumbered sentence-case headings.
- No colors in the text.

**Builds:**
- Draft: `latexmk -pdf main.tex`, with BibTeX and the figures shown.
- Submission:
  1. Paste the `.bbl` into the file.
  2. Set the figure switch to off.
  3. Rebuild, and check that the PDF contains text, captions, tables and references only.
  4. Upload the PDF, the separate TIFFs, the SI files and the cover letter.

**On acceptance:** upload the single `.tex` (no tracked changes; `latexdiff` for revision PDFs, as the template recommends).

---

## 11. Style and language rules
- Standard English, consistent spelling. PLOS does not copyedit, so get a careful human proofread. Language-only AI editing needs no disclosure, but code and content help does.
- "We"; past tense for what was done; present tense for general facts and for what the figures show.
- **Abbreviations:**
  - Define at first use in the text (the abstract is separate: avoid abbreviations there).
  - Only use a non-standard abbreviation if it appears at least 3 times.
  - Planned set: CR3BP, FTLE, RHS, RK4, RK45, DOP853, DU, TU, JPL, CI.
  - Avoid ad-hoc codes (eqr, spec, cal); write "equal-risk", "oracle-tuned", "certified".
- **Units:** SI, with DU and TU defined in km and in days/seconds. The tolerance is 100 km.
- **Numbers:** percentages with k/N; p exact to 3 decimals when ≥ 0.001, else "p < 0.001"; CIs "95% CI 70.8–72.4%".
- No footnotes; no colored text.
- Every "cheaper" names the comparator and its tuning. Exploratory results are labelled. Implications are labelled as hypotheses.

---

## 12. Order of work
1. Decide fee assistance, authorship and affiliations. Draft the AI disclosure and the related-manuscript statement (section 2).
2. Download the PLOS template zip; set up `manuscript/` (section 10).
3. Verify and complete the references (7.3): every entry is opened by a person and has its DOI.
4. Run `make_numbers.py` (including all k/N counts) and Tables 1-3.
5. Build Figures 1-10 to spec; run NAAS.
6. Write, in this order: Methods → Results → S1 Appendix → Discussion → Conclusions → Introduction → Abstract → titles.
7. Deposit the code, preregistrations, processed data and figure scripts (Zenodo DOI). Write the Data Availability Statement.
8. Write the cover letter (≤ 1 page) and pick Academic Editors.
9. Final checks (section 13). Optionally post to arXiv. Submit the PDF and the separate files.

---

## 13. Final checklist (PLOS criteria and format)
- [ ] **Criterion 1-2:** original; not under consideration elsewhere; related manuscripts declared with copies
- [ ] **Criterion 3:** Methods replicable; statistical analysis section complete (software and versions, α, tails, corrections, code DOI)
- [ ] **Criterion 4:** conclusions supported; no overstatement; implications labelled as hypotheses; negative results reported (D2, D5, C2, C3(a) replication)
- [ ] **Criterion 5:** clear standard English; proofread
- [ ] **Criterion 6:** AI use fully disclosed (tools, uses, affected parts, validation); authorship meets PLOS criteria; CRediT roles entered
- [ ] **Criterion 7:** Data Availability Statement; minimal data set (values behind every figure and statistic); code public with DOI and license
- [ ] **Title page:** title ≤ 250 characters, sentence case, no specialist abbreviations; short title ≤ 100 (form); names, affiliations (dept, institution, city, state, country), corresponding email; ORCID in the system
- [ ] **Abstract:** ≤ 300 words, objective / approach / results / significance, no citations, minimal abbreviations
- [ ] **Introduction** ends with the aim and whether it was achieved
- [ ] **Order:** Title page → Abstract → Introduction → Methods/Results/Discussion/Conclusions → Acknowledgments → References → SI captions
- [ ] **Layout:** double spacing; page numbers; continuous line numbers; ≤ 3 heading levels, sentence case; no footnotes; one .tex file
- [ ] **Figures:** TIFF to spec; Arial 8-12 pt; captions "Fig N." with bold title ≤ 15 words after the first-citing paragraph; not in the PDF; cited in order
- [ ] **Tables:** cell-based, title above, notes below, after the first-citing paragraph, cited in order
- [ ] **References:** Vancouver via `plos2015.bst`, .bbl pasted in, [n] in order of appearance, six authors then et al., NLM abbreviations, `doi:` form, no unpublished or "data not shown" citations
- [ ] **SI:** "S1 Appendix" style names, < 20 MB each, captions at the end, cited with `\nameref`
- [ ] **Not in the manuscript:** funding, competing interests, author contributions (all in the form)
- [ ] **Cover letter:** ≤ 1 page, with the required items; no fee requests
- [ ] **Fee:** USD 2,477 on acceptance, or a fee-assistance request in the system

---

## 14. PLOS sources read for this plan (2026-10-29)
- Submission guidelines: https://journals.plos.org/plosone/s/submission-guidelines
- LaTeX: https://journals.plos.org/plosone/s/latex (template v3.6, August 2022, read via the official PLOS Overleaf copy: https://www.overleaf.com/latex/templates/latex-template-for-plos-public-library-of-science-articles/tmfpqfxwdxbs)
- Figures: https://journals.plos.org/plosone/s/figures
- Tables: https://journals.plos.org/plosone/s/tables
- Criteria for publication: https://journals.plos.org/plosone/s/criteria-for-publication
- Data availability: https://journals.plos.org/plosone/s/data-availability
- Materials, software and code sharing: https://journals.plos.org/plosone/s/materials-software-and-code-sharing
- Ethical publishing practice (AI policy, related manuscripts): https://journals.plos.org/plosone/s/ethical-publishing-practice
- Authorship: https://journals.plos.org/plosone/s/authorship
- Publication fees: https://plos.org/fees/

**Not read:**
- the dedicated supporting-information page (its rules here come from the submission guidelines and the template);
- the sample manuscript PDFs;
- the `plos2015.bst` file itself (the zip download is redirected off-site and could not be opened);
- the Academic Editor list.

Check the .bst behaviour on a compiled bibliography (section 7.2).
