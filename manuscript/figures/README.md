# Figures for the PLOS ONE manuscript

- `Fig1.tif` ... `Fig10.tif`: the files to upload. RGB, 8 bit, LZW, 600 dpi, no alpha, 6.2-7.4 in wide, Arial at 8-12 pt (MANUSCRIPT_PLAN.md section 8). Run each main figure through PLOS NAAS before submission.
- `data/`: the values behind every panel, one or more CSV files per figure (S1 Data).
- `draft/`: low-resolution PNG copies, used only by the draft build of `main.tex`.

Rebuild (each script reads the sealed result files and ledgers; `out_c/` and `out_d/` must be present):

```bash
cd manuscript/scripts && for f in fig01 fig02 fig03 fig04 fig05 fig06 fig07 fig08 fig09 fig10; do /Users/arnavdeopura/miniforge3/envs/orbit-ml/bin/python $f.py; done
```

Build the submission PDF (no graphics) and a draft PDF with the figures shown:

```bash
cd manuscript && latexmk -pdf main.tex
latexmk -pdf -jobname=main_draft -pdflatex='pdflatex %O "\def\drafton{}\input{%S}"' main.tex
```
