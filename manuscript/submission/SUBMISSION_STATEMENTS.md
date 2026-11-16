# Statements for the PLOS ONE submission form

PLOS ONE collects these in the online submission system, not in the manuscript file
(`journals.plos.org/plosone/s/submission-guidelines`, read 2026-11-15). They are kept here so that the manuscript and the form agree.
Items marked **[AUTHOR]** can only be supplied by the author. `scripts/check_manuscript.py` reads the short title from this file.

## Title

**Full title (250 characters at most; 161 used):** A single pre-run time step loses to adaptive integration in the Earth–Moon three-body problem; in Sundman time it wins on most trajectories but not in total cost

**Short title (100 characters at most):** Pre-run step-size selection in the Earth–Moon restricted three-body problem

## Authors

- Arnav Deopura. Affiliation: **[AUTHOR]** (department or programme, institution, city, state, country). Corresponding author email: **[AUTHOR]**.
- ORCID: **[AUTHOR]** (PLOS requires one for the corresponding author).

## Author contributions (CRediT taxonomy, entered per author in the form)

**[AUTHOR]** to confirm. The roles must agree with the "Use of AI tools" subsection of the manuscript, which states that the analysis code
and the first draft of the text were produced with AI tools under the author's direction. Roles that the project records support for the
author: Conceptualization, Methodology, Investigation, Data curation, Project administration, Validation, Writing (review and editing).
Software, Formal analysis, Visualization and Writing (original draft) were carried out with the AI tools; claim them only to the extent
that reflects what you did yourself. If a mentor contributed, give the mentor's role in the Acknowledgments (placeholder in `main.tex`).

## Data Availability Statement

Draft, to be completed with the DOI once the deposit exists **[AUTHOR]**:

> All code, the preregistration documents and seal manifests, the frozen models, the generated datasets, the per-trajectory result
> ledgers and the processed data behind every figure and statistic are available from [repository name] at [DOI]. The values plotted in
> every figure are also provided as S1 Data. Periodic-orbit initial conditions were obtained from the JPL Three-Body Periodic Orbits API
> (https://ssd-api.jpl.nasa.gov/doc/periodic_orbits.html) and the CAPSTONE ephemeris from the JPL Horizons system
> (https://ssd.jpl.nasa.gov/horizons/); both are public.

The deposit has not been made (it needs the author's repository account). Until it exists, the manuscript carries the placeholder
`[repository DOI to be assigned]` in "Preregistration and statistical analysis".

## Financial Disclosure

**[AUTHOR]** to confirm. If there was no funding: "The author received no specific funding for this work."

## Competing interests

**[AUTHOR]** to confirm. If none: "The author has declared that no competing interests exist."

## Ethics statement

Not applicable: the study involves no human participants, animals, cell lines or field sampling.

## Use of AI tools

Disclosed in the manuscript, in the Methods subsection "Use of AI tools", as the PLOS policy asks (name of each tool, how it was used, how
the validity of its output was evaluated; `journals.plos.org/plosone/s/ethical-publishing-practice`). **[AUTHOR]**: confirm the model names
and your role from your own records, and keep the prompt logs. A science fair's rules on AI-written text are a separate and stricter
matter (`DEFENSE_STATUS.md` at the repository root).

## Previous publication and related work

**[AUTHOR]**: state in the cover letter whether any part of this work has been presented or posted elsewhere (for example a fair poster or
a preprint).

## Files to upload

| Item | File |
|---|---|
| Manuscript (LaTeX submissions are uploaded as PDF, with the source) | `main.pdf`; source `build/submission/main.tex` (references pasted in) |
| Figures, one file each, named as cited | `figures/Fig1.tif` … `figures/Fig10.tif` |
| Supporting information, one file per item | `si/S1_Appendix.pdf`, `si/S1_Table.zip` … `si/S4_Table.zip`, `si/S1_File.zip`, `si/S1_Data.zip` |

`./build.sh submission` collects all of them in `build/submission/`. Each of S1–S4 Table is a set of CSV files (`si/SN_Table_*.csv`);
`scripts/si_archives.py` packs each set into one archive, because PLOS takes one file per supporting item.
