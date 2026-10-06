"""
Academic publication-quality figure styling for Nature/Science/IEEE standards.

Typography: Arial/Helvetica family, consistent hierarchy
Colors: Perceptually uniform, colorblind-safe, print-ready
Layout: Single/double column layouts (89mm / 183mm)
Resolution: 300-600 DPI for publication
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from typing import Literal

# Nature/Science single column: 89mm = 3.5in
# Nature/Science double column: 183mm = 7.2in
# IEEE single column: 88.9mm = 3.5in
# IEEE double column: 181mm = 7.13in

SINGLE_COL_WIDTH = 3.5  # inches
DOUBLE_COL_WIDTH = 7.2  # inches

# Academic color palette - colorblind safe, print ready
COLORS = {
    # Primary scientific palette (Wong 2011, Nature Methods)
    "blue": "#0173B2",      # Trustworthy, data
    "orange": "#DE8F05",    # Attention, highlights
    "green": "#029E73",     # Growth, positive
    "red": "#CC3311",       # Alert, negative
    "purple": "#925E9F",    # Alternative category
    "brown": "#846A4B",     # Neutral, context
    "pink": "#EC7FA8",      # Secondary highlight
    "gray": "#5D5D5D",      # Text, structure

    # Extended palette
    "light_blue": "#56B4E9",
    "light_orange": "#F0E442",
    "light_green": "#6DB399",
    "light_gray": "#999999",

    # Semantic colors
    "baseline": "#5D5D5D",
    "proposed": "#0173B2",
    "improvement": "#029E73",
    "degradation": "#CC3311",
    "context": "#999999",

    # Map visualization
    "ego": "#2C2C2C",
    "target": "#CC3311",
    "obstacle": "#DE8F05",
    "lane": "#0173B2",
    "crosswalk": "#029E73",
    "background": "#F8F8F8",
}

# Categorical palettes for multi-series plots
CATEGORICAL_SAFE = [
    COLORS["blue"],
    COLORS["orange"],
    COLORS["green"],
    COLORS["red"],
    COLORS["purple"],
    COLORS["brown"],
]

# Sequential palette for heatmaps/gradients
SEQUENTIAL_BLUES = ["#F7FBFF", "#DEEBF7", "#C6DBEF", "#9ECAE1", "#6BAED6", "#4292C6", "#2171B5", "#084594"]


def set_academic_style(
    context: Literal["paper", "poster", "notebook"] = "paper",
    font_scale: float = 1.0,
) -> None:
    """
    Configure matplotlib for academic publication.

    Args:
        context: Target output context (paper=normal, poster=larger, notebook=screen)
        font_scale: Multiplicative scaling for all fonts
    """

    base_sizes = {
        "paper": {"title": 10, "label": 9, "tick": 8, "legend": 8},
        "poster": {"title": 18, "label": 16, "tick": 14, "legend": 14},
        "notebook": {"title": 12, "label": 11, "tick": 10, "legend": 10},
    }

    sizes = base_sizes[context]

    # Font configuration
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    rcParams["font.size"] = sizes["label"] * font_scale
    rcParams["axes.titlesize"] = sizes["title"] * font_scale
    rcParams["axes.labelsize"] = sizes["label"] * font_scale
    rcParams["xtick.labelsize"] = sizes["tick"] * font_scale
    rcParams["ytick.labelsize"] = sizes["tick"] * font_scale
    rcParams["legend.fontsize"] = sizes["legend"] * font_scale
    rcParams["figure.titlesize"] = sizes["title"] * font_scale

    # Line and marker configuration
    rcParams["lines.linewidth"] = 1.5
    rcParams["lines.markersize"] = 6
    rcParams["patch.linewidth"] = 0.5

    # Axes configuration
    rcParams["axes.linewidth"] = 0.8
    rcParams["axes.edgecolor"] = "#2C2C2C"
    rcParams["axes.labelcolor"] = "#2C2C2C"
    rcParams["axes.spines.top"] = False
    rcParams["axes.spines.right"] = False
    rcParams["axes.grid"] = False  # Enable per-plot when needed
    rcParams["axes.axisbelow"] = True

    # Grid configuration (when enabled)
    rcParams["grid.color"] = "#E0E0E0"
    rcParams["grid.linewidth"] = 0.5
    rcParams["grid.linestyle"] = "-"
    rcParams["grid.alpha"] = 0.7

    # Tick configuration
    rcParams["xtick.direction"] = "out"
    rcParams["ytick.direction"] = "out"
    rcParams["xtick.major.width"] = 0.8
    rcParams["ytick.major.width"] = 0.8
    rcParams["xtick.minor.visible"] = False
    rcParams["ytick.minor.visible"] = False
    rcParams["xtick.color"] = "#2C2C2C"
    rcParams["ytick.color"] = "#2C2C2C"

    # Legend configuration
    rcParams["legend.frameon"] = False
    rcParams["legend.numpoints"] = 1
    rcParams["legend.scatterpoints"] = 1

    # Figure configuration
    rcParams["figure.facecolor"] = "white"
    rcParams["figure.dpi"] = 100  # Screen preview
    rcParams["savefig.dpi"] = 600  # Publication quality
    rcParams["savefig.bbox"] = "tight"
    rcParams["savefig.pad_inches"] = 0.05
    rcParams["savefig.transparent"] = False

    # PDF backend for vector graphics
    rcParams["pdf.fonttype"] = 42  # TrueType fonts in PDF
    rcParams["ps.fonttype"] = 42


def create_figure(
    width: Literal["single", "double", "full"] | float = "double",
    aspect: float = 0.618,  # Golden ratio by default
    dpi: int = 100,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a figure with academic dimensions.

    Args:
        width: Column width preset or custom inches
        aspect: Height/width ratio
        dpi: Display DPI (save DPI is 600)

    Returns:
        Figure and single Axes object
    """

    if width == "single":
        w = SINGLE_COL_WIDTH
    elif width == "double":
        w = DOUBLE_COL_WIDTH
    elif width == "full":
        w = DOUBLE_COL_WIDTH
    else:
        w = float(width)

    h = w * aspect

    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
    return fig, ax


def create_multipanel(
    rows: int,
    cols: int,
    width: Literal["single", "double"] | float = "double",
    aspect: float = 0.618,
    hspace: float = 0.3,
    wspace: float = 0.3,
    dpi: int = 100,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Create multi-panel figure with consistent spacing.

    Returns:
        Figure and array of Axes objects
    """

    if width == "single":
        w = SINGLE_COL_WIDTH
    elif width == "double":
        w = DOUBLE_COL_WIDTH
    else:
        w = float(width)

    # Total height accounts for spacing
    panel_w = w / cols
    panel_h = panel_w * aspect
    total_h = panel_h * rows + (rows - 1) * hspace * panel_h

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(w, total_h),
        dpi=dpi,
        constrained_layout=False,
    )

    plt.subplots_adjust(hspace=hspace, wspace=wspace)

    return fig, axes


def add_panel_label(ax: plt.Axes, label: str, loc: Literal["top-left", "top-right"] = "top-left") -> None:
    """
    Add bold panel label (A, B, C, ...) to subplot.

    Standard practice in Nature/Science multi-panel figures.
    """

    if loc == "top-left":
        x, y, ha = 0.02, 0.98, "left"
    else:
        x, y, ha = 0.98, 0.98, "right"

    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha=ha,
    )


def save_figure(fig: plt.Figure, path: str, dpi: int = 600, transparent: bool = False) -> None:
    """
    Save figure with publication settings.

    Args:
        fig: Matplotlib figure
        path: Output path (.png, .pdf, .svg)
        dpi: Resolution for raster formats
        transparent: Transparent background
    """

    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=transparent,
        facecolor="white" if not transparent else "none",
    )
    plt.close(fig)
