"""Shared matplotlib/seaborn style — one consistent look across every figure (PRD §7.5).

Consistency is the tell of a professional repo: two accent colors, a muted whitegrid theme,
uniform fonts and figure sizes. Call `apply_style()` at the top of any notebook/script that
produces a figure destined for the README or the Tier-1 dashboard.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# --- Palette: two accent colors, used everywhere -----------------------------
PRIMARY = "#2C6E9B"    # calm clinical blue — the model / internal / "good" series
ACCENT = "#D1495B"     # muted red — external / alerts / the "watch out" series
MUTED = "#6C757D"      # neutral grey — reference lines, secondary text
GOOD = "#2A9D8F"       # teal — positive deltas / targets met
TARGET_LINE = "#E9C46A"  # amber — target/threshold reference lines

# Convenient sequences for grouped charts (internal vs external, before vs after).
PAIR = [PRIMARY, ACCENT]
CLASS_COLORS = {"NORMAL": PRIMARY, "PNEUMONIA": ACCENT}

FIGSIZE = (8, 5)       # default single panel
DPI = 130


def apply_style() -> None:
    """Set the global matplotlib/seaborn theme. Idempotent — safe to call repeatedly."""
    sns.set_theme(style="whitegrid", context="notebook")
    mpl.rcParams.update(
        {
            "figure.figsize": FIGSIZE,
            "figure.dpi": 110,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#CED4DA",
            "axes.grid": True,
            "grid.color": "#E9ECEF",
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def takeaway(ax, text: str) -> None:
    """Put a one-line takeaway caption ABOVE a panel (the dashboard design rule, §7.5)."""
    ax.set_title(text, loc="left", color=MUTED, fontsize=10, fontweight="normal", pad=10)
