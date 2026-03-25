"""publication-quality plotting style for CUP template (newtx fonts)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# muted academic palette
C_POS = "#c0392b"
C_NEG = "#7f8c8d"
C_K2 = "#2c3e50"
C_K3 = "#c0392b"
C_ALL = "#2c3e50"
C_LOW_F = "#2980b9"
C_MID_F = "#d4a017"
C_HIGH_F = "#c0392b"

# journal column widths (inches)
W_FULL = 7.0
W_HALF = 3.4

DPI = 300


def setup_style():
    """configure matplotlib rcParams for CUP journal style."""
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}",
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.framealpha": 0.8,
        "legend.edgecolor": "0.8",
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.5,
    })


def panel_label(ax, label, x=-0.12, y=1.06):
    """add (a), (b), (c) panel label to an axis."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
