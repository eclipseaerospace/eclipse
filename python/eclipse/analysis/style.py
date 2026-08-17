# SPDX-License-Identifier: Apache-2.0
#
# eclipse.analysis.style — the figure palette and the rc settings every runner
# shares.
#
# Six runners had grown their own copy of this block, and one of them had
# already found the split below on its own: a base of everything invariant, with
# a per-figure layer over it. The base here is exactly the intersection of the
# six, no wider. Type sizes look invariant and are not -- three runners set tick
# labels at 8.5 and three at 9.0 -- so they stay in the per-figure layer where
# they were, rather than being averaged into a default that silently redraws
# half the figures.
#
# No matplotlib import. The style is a plain mapping handed to rc_context by the
# caller, so the library stays a pure numpy dependency and only the runners,
# which are not packaged, need a plotting stack.
#
# Colors are ink on paper rather than a screen palette: a warm off-white ground
# with three weights of near-black for structure, and two accents chosen to stay
# distinguishable in grayscale print and to the most common color vision
# deficiencies.

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

__all__ = [
    "ACCENT_PRIMARY",
    "ACCENT_SECONDARY",
    "BASE_STYLE",
    "GRID",
    "INK_MUTED",
    "INK_PRIMARY",
    "INK_SECONDARY",
    "SURFACE",
    "figure_style",
]

INK_PRIMARY: Final = "#0b0b0b"
INK_SECONDARY: Final = "#52514e"
INK_MUTED: Final = "#8a8880"
SURFACE: Final = "#fcfcfb"
GRID: Final = "#e6e5e0"

ACCENT_PRIMARY: Final = "#1f4e9c"
ACCENT_SECONDARY: Final = "#d4570a"

BASE_STYLE: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "figure.dpi": 200,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK_SECONDARY,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "legend.frameon": False,
        "savefig.facecolor": SURFACE,
    }
)


def figure_style(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {**BASE_STYLE, **(overrides or {})}
