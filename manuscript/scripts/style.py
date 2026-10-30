"""Shared figure style for the PLOS ONE manuscript (MANUSCRIPT_PLAN.md section 8.1).

Every figure is designed at its final size (full width 7.5 in), drawn in Arial at 8-12 pt, and written as an RGB TIFF with LZW compression at
600 dpi and no alpha channel (PLOS figure requirements). The data behind each panel are written next to it as CSV (S1 Data).
Run any figN.py with the project's environment (/Users/arnavdeopura/miniforge3/envs/orbit-ml/bin/python) from any folder.
"""
from __future__ import annotations

import io
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))          # the project folder
OUT = os.path.join(HERE, "..", "figures")
DATA_OUT = os.path.join(OUT, "data")
PREVIEW = os.environ.get("FIG_PREVIEW_DIR")                     # optional low-resolution PNG copies for checking
sys.path.insert(0, ROOT)

FULL_W = 7.4          # in (7.5 in maximum after the 2-pt border)
COL_W = 5.2           # in
DPI = 600

# Okabe-Ito palette, one meaning per colour everywhere (plan 8.1)
C = {
    "sundman": "#0072B2", "fixed": "#555555", "rk45": "#E69F00", "dop853": "#CC79A7", "onescale": "#56B4E9",
    "rules": "#000000", "linear": "#009E73", "vermillion": "#D55E00",
    "paradigm": "#0072B2", "selection": "#999999", "prediction": "#E69F00", "safety": "#56B4E9", "overhead": "#CC79A7",
    "rounding": "#DDDDDD", "total": "#000000",
}
HATCH = {"paradigm": "", "selection": "..", "prediction": "//", "safety": "\\\\", "overhead": "xx", "rounding": "--", "total": ""}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.sf": "Arial",
    "axes.linewidth": 0.75,
    "lines.linewidth": 1.25,
    "lines.markersize": 4.5,
    "errorbar.capsize": 2,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "hatch.linewidth": 0.6,
    "pdf.fonttype": 42,
})
ERR = dict(elinewidth=0.75, capsize=2, capthick=0.75)


def panel(ax, letter, x=-0.12, y=1.04):
    """Upper-case panel letter, 12 pt bold, at the upper left of the axes."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")


def refline(ax, y=None, x=None, **kw):
    kw = {"color": "#555555", "lw": 0.75, "ls": (0, (4, 3)), "zorder": 0, **kw}
    if y is not None:
        ax.axhline(y, **kw)
    if x is not None:
        ax.axvline(x, **kw)


def save(fig, n):
    """FigN.tif: RGB, 8 bit, LZW, 600 dpi, no alpha, 2-pt border; returns (path, width_in, height_in, MB)."""
    os.makedirs(OUT, exist_ok=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    path = os.path.join(OUT, f"Fig{n}.tif")
    im.save(path, format="TIFF", compression="tiff_lzw", dpi=(DPI, DPI))
    draft = os.path.join(OUT, "draft")                          # PNG copies for the draft build of main.tex (\draftfigstrue)
    os.makedirs(draft, exist_ok=True)
    im.resize((im.width // 3, im.height // 3), Image.LANCZOS).save(os.path.join(draft, f"Fig{n}.png"))
    if PREVIEW:
        os.makedirs(PREVIEW, exist_ok=True)
        im.resize((im.width // 6, im.height // 6), Image.LANCZOS).save(os.path.join(PREVIEW, f"Fig{n}.png"))
    w, h = im.width / DPI, im.height / DPI
    mb = os.path.getsize(path) / 1e6
    print(f"Fig{n}.tif  {im.width} x {im.height} px  {w:.2f} x {h:.2f} in  {mb:.2f} MB")
    return path, w, h, mb


def write_data(n, name, df):
    os.makedirs(DATA_OUT, exist_ok=True)
    p = os.path.join(DATA_OUT, f"Fig{n}_{name}.csv")
    df.to_csv(p, index=False)
    return p


def cp_interval(k, n, a=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def path(*p):
    return os.path.join(ROOT, *p)
