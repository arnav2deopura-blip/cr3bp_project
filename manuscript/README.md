# Manuscript (PLOS ONE)

First complete draft of the article, covering everything done in the project up to 2026-11-15: Studies A–D (sealed and run), the post hoc
and exploratory analyses (P1–P25), and Study E (protocol, code, pilots and a rehearsal on selection data; **not registered, not sealed, not
run**). The structure follows the PLOS ONE submission guidelines and the PLOS LaTeX template v3.8.

## Rebuild

```bash
./build.sh all
```

Individual targets: `figures`, `si`, `si-full`, `pdf`, `draft`, `submission`, `check`, `clean` (`./build.sh` alone lists them).
Python comes from the project environment (`environment.yml` at the repository root; override with `PY=/path/to/python ./build.sh ...`);
the PDFs need a TeX distribution with `latexmk`. From the repository root, `./reproduce.sh manuscript` does the same as `./build.sh all`.

`figures` and `si` read the result files of the studies, including the git-ignored run folders `out_c/`, `out_d/` and `out_e/`.
`pdf`, `draft`, `submission` and `check` need only this folder.

## What is a source and what is generated

Edit only the sources. Everything else is rewritten by `build.sh`.

| Path | Kind | Written by |
|---|---|---|
| `main.tex` | **source** | the article (one file, no `\input`, as PLOS asks) |
| `references.bib`, `plos2025.bst` | **source** | bibliography and the PLOS BibTeX style |
| `si/S1_Appendix.tex` | **source** | S1 Appendix |
| `submission/SUBMISSION_STATEMENTS.md` | **source** | short title and the statements PLOS collects in its form |
| `scripts/*.py`, `build.sh` | **source** | the build |
| `template/plos_latex_template.tex` | reference | the unmodified PLOS template (downloaded 2026-10-29), for comparison only |
| `CITATION_AUDIT.md` | record | which references were checked against which record, and when |
| `main.pdf` | generated | `build.sh pdf`: the submission PDF, without graphics |
| `main_draft.pdf` | generated | `build.sh draft`: same text with the figures under their captions (for reading) |
| `figures/FigN.tif`, `figures/draft/FigN.png`, `figures/data/*.csv` | generated | `build.sh figures` (`scripts/fig01.py` … `fig11.py`, `style.py`) |
| `si/S1_Table_*.csv` … `si/S4_Table_*.csv` | generated | `build.sh si` (`scripts/si_tables.py`; S3 only with `si-full`) |
| `si/S5_Table_study_e_*.csv` | generated | `build.sh si` (`scripts/study_e_tables.py`) |
| `si/S1_Appendix.pdf` | generated | `build.sh si` |
| `si/*.zip` (S1–S5 Table, S1 File, S1 Data) | generated, git-ignored | `build.sh si` (`scripts/si_archives.py`; deterministic archives) |
| `build/` | generated, git-ignored | LaTeX intermediates, `check_report.txt`, and `submission/` |

`build/submission/` is the upload set: `main.tex` with the references pasted in (compiled on its own to show it is self-contained),
`Fig1.tif` … `Fig11.tif`, and one file per supporting item.

What "reproducible" means here: the figure TIFFs, CSVs and zip archives come out byte-identical from the same inputs. The PDFs come
out with the same content but not the same bytes, because the PLOS template prints the build date in the footer.

## Where each number comes from

- Studies A–D: the sealed result files (`archive/study_a/`, `part_b/`, `part_c/`, `part_d/results`); post hoc numbers: `posthoc/results/`
  (definitions in `posthoc/README.md`). The comment block at the top of `main.tex` says the same.
- Study E pilots and rehearsal: every number in the two Study E subsections of the Results, in Table 4 and in Fig 11 is a row of
  `si/S5_Table_study_e_*.csv` or `figures/data/Fig11_rk8_pilot.csv`. Those files are copied from `posthoc/results/rk8s_*` and
  `part_e/results/dry/` by `scripts/study_e_tables.py` and `scripts/fig11.py`; nothing is typed by hand.

## PLOS ONE requirements and how they are met

`scripts/check_manuscript.py` (`./build.sh check`) tests the points below and writes `build/check_report.txt`; it exits with an error if a
requirement fails. The guidelines were read on 2026-11-15 (`journals.plos.org/plosone/s/submission-guidelines`,
`…/s/ethical-publishing-practice`, `…/s/latex`).

| Requirement | Where |
|---|---|
| Order: title page, Abstract, Introduction, Materials and methods, Results, Discussion, Conclusions, Acknowledgments, References, Supporting information captions | `main.tex` section order |
| Title ≤ 250 characters; short title ≤ 100 | title page; short title in `submission/SUBMISSION_STATEMENTS.md` |
| Abstract ≤ 300 words, no citations | Abstract |
| Double spacing, line numbers, page numbers, no footnotes, at most three heading levels | preamble (template), checked |
| Figures not in the manuscript file; captions after the first-citing paragraph; cited in order; "Fig N" | `\ifdraftfigs` switch (off by default), checked |
| Figure files: TIFF, RGB, 300–600 dpi, ≤ 7.5 × 8.75 in, < 10 MB | `scripts/style.py`, checked |
| Tables after the first-citing paragraph, title above, legend below | Tables 1–4, checked |
| References numbered in order of appearance (Vancouver) | `plos2025.bst`; pasted into the `.tex` for submission |
| Supporting information captions at the end; one file per item, < 20 MB | last section; `si/`, checked |
| AI tools disclosed: name, how used, how validity was evaluated | Methods, "Use of AI tools"; tools in the reference list |
| Data availability, funding, competing interests, author contributions | entered in the PLOS form; drafts in `submission/SUBMISSION_STATEMENTS.md` |

## Open before submission

Only the author can do these (the checker lists the placeholders that remain in `main.tex`):

1. Affiliation and corresponding-author email on the title page; ORCID.
2. Mentor acknowledgment (placeholder in Acknowledgments) and the CRediT contributions in the form.
3. The repository deposit and its DOI (placeholder `[repository DOI to be assigned]`), then the Data Availability Statement.
4. Confirm the "Use of AI tools" subsection from your own records: model names, the tools used, and the description of your role.
5. Run each `figures/FigN.tif` through the PLOS figure checker (NAAS) once.
6. Resolve every DOI in `references.bib` once (`CITATION_AUDIT.md` says which were checked individually).
7. Study E: when its confirmatory run exists, replace the rehearsal subsection and Table 4 with the confirmatory results; until then the
   draft is honest as it stands (pilot = exploratory, rehearsal = not confirmatory).

A science fair's rules on AI-written text are stricter than PLOS's; see `DEFENSE_STATUS.md` at the repository root before using this text
for a fair.
