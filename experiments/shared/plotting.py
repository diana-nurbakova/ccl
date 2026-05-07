"""Shared plotting utilities for CCL validation experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from .ccl_mappings import CCL_COLORS

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "figures"


def setup_style() -> None:
    """Apply consistent plot styling across all experiments."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
    })


def ccl_palette() -> dict[str, str]:
    """Return the CCL category colour palette."""
    return dict(CCL_COLORS)


def save_figure(
    fig: plt.Figure,
    name: str,
    formats: tuple[str, ...] = ("png", "pdf"),
    output_dir: Path | None = None,
) -> list[Path]:
    """Save a figure in multiple formats.

    Returns list of saved file paths.
    """
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    paths = []
    for fmt in formats:
        p = out / f"{name}.{fmt}"
        fig.savefig(p)
        paths.append(p)
    return paths


def grouped_bar_chart(
    data: dict[str, float],
    title: str,
    ylabel: str,
    colors: dict[str, str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Simple grouped bar chart with CCL colours."""
    if ax is None:
        _, ax = plt.subplots()
    palette = colors or ccl_palette()

    labels = list(data.keys())
    values = list(data.values())
    bar_colors = [palette.get(l, "#666666") for l in labels]

    ax.bar(labels, values, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return ax


def trajectory_plot(
    session_data: dict[str, list[float]],
    title: str,
    ylabel: str,
    xlabel: str = "Session",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Line plot for session-level trajectories."""
    if ax is None:
        _, ax = plt.subplots()

    for label, values in session_data.items():
        ax.plot(range(1, len(values) + 1), values, marker="o", label=label)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.legend()
    return ax
