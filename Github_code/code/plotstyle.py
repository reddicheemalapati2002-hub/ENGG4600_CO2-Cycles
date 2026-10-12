"""Shared matplotlib style for all thesis figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.titlesize": 10, "axes.labelsize": 10, "legend.fontsize": 8.5,
    "axes.grid": True, "grid.alpha": 0.28, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "lines.markersize": 5,
})
from co2whr.sites import COLORS, MARKERS, LABELS, SITE_ORDER, SITES, CYCLES  # noqa

SITE_SHORT = {k: SITES[k]["name"].replace(": ", ":\n") for k in SITES}
GREY = "#777777"


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print("  saved", path)
