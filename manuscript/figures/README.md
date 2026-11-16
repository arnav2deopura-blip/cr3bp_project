# Figures for the PLOS ONE manuscript

- `Fig1.tif` … `Fig10.tif`: the files to upload. RGB, 8 bit, LZW, 600 dpi, no alpha, 6.2–7.4 in wide, Arial at 8–12 pt
  (`scripts/style.py`). Run each through the PLOS figure checker (NAAS) before submission.
- `data/`: the values behind every panel, one or more CSV files per figure (packed into `si/S1_Data.zip` as S1 Data).
- `draft/`: low-resolution PNG copies, used only by the draft build of `main.tex` (`main_draft.pdf`).

Everything in this folder is generated. Rebuild from the manuscript folder:

```bash
./build.sh figures
```

Each script `scripts/figNN.py` reads the sealed result files and ledgers (`out_c/` and `out_d/` must be present). The scripts are deterministic: a rebuild from the same inputs
rewrites identical files. See `../README.md` for the PDFs.
