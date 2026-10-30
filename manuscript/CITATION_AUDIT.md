# Citation audit (2026-10-30)

Checked: every key cited in `main.tex` (43) against `references.bib` (43 entries, no unused entry) and against the flags in
`LITERATURE_REVIEW.md` (items marked *unverified*, "not verified", or "unverified leads").

## Flagged items in the literature review: none reached the manuscript

| Flagged in LITERATURE_REVIEW.md | In references.bib? |
|---|---|
| Richardson (1980), DOI not verified | no |
| "Highly Efficient Numerical Integrator for the CR3BP" (authors/venue not verified) | no |
| Bowman group, exactly conservative n-body integrator (authors not verified) | no |
| Portegies Zwart & Boekholt (2014), journal not verified | no |
| Dellnitz, Hüllermeier et al., RL time stepping (author list and journal not verified) | **yes, `dellnitz2023`**: verified below |
| Chen et al. (2026) PINN periodic orbits (figures not verified) | no |
| arXiv:1809.01257, arXiv:2206.07022 (unverified leads) | no |

## Verified against publisher or index records in this audit

| Key | Record checked | Result |
|---|---|---|
| `dellnitz2023` | SIAM J Sci Comput 45(2):A579–A595 (2023), doi 10.1137/21M1412682; seven authors as in the .bib (OpenAlex, Paderborn and ZIB repository records) | matches |
| `angelopoulos2025` | Ann Appl Stat 19(2):1641–1662 (2025), doi 10.1214/24-AOAS1998; five authors (NSF PAR record) | matches |
| `sazulibarrena2025` | Commun Nonlinear Sci Numer Simul 145:108723 (June 2025) (Leiden publication list; arXiv 2502.12809). The Elsevier page itself was not opened. | matches |
| `schork2020` (new) | Math Program Comput 12(4):603–635 (2020), doi 10.1007/s12532-020-00181-8 (Springer, Edinburgh report page) | added; cited for HiGHS's interior point solver IPX |

## Not individually re-verified in this audit

The other 39 entries are standard references (textbooks, the software papers for NumPy/SciPy/scikit-learn/pandas/Numba/Matplotlib/REBOUND,
classical statistics papers, and the astrodynamics and integration literature) that `LITERATURE_REVIEW.md` does not flag. Their DOIs were
not re-resolved one by one here. Before submission, resolve every DOI once (for example with `doi.org` in a browser) and check the
journal abbreviation style.

## Other open items for submission (not citations)

- `\cite{huangfu2018}` now refers to HiGHS itself; the interior point code is cited as `schork2020`. SciPy's `highs-ipm` calls HiGHS's IPX
  solver in the SciPy 1.17 build used here (see the SciPy `linprog(method='highs-ipm')` documentation for the installed version).
- The AI-use paragraph names "Claude Sonnet 5.5" for the analysis code and "Claude Opus 5.5" for the text. The author should confirm the
  model names from their own records, and check the current ISEF/NCSEF disclosure rules, which were not checked here.
