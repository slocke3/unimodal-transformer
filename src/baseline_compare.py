"""Canonical API for transformer versus empirical k-gram comparisons.

The implementation remains in :mod:`src.floor_compare` temporarily so existing
imports continue to work. New code should import from this module.
"""

from .floor_compare import (
    compute_panel,
    panel_family,
    panel_in_distribution,
    plot_figure1,
    run_figure1,
)

__all__ = [
    "compute_panel",
    "panel_family",
    "panel_in_distribution",
    "plot_figure1",
    "run_figure1",
]
