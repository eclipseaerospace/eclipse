# SPDX-License-Identifier: Apache-2.0
#
# eclipse.analysis.boundary — where published parameters stop supporting an
# answer.
#
# The distinction this records is not confident against uncertain. It is
# between a number interpolated inside a range somebody measured and a model
# form carried into a regime nobody has. Those fail differently and only the
# first has an error bar, so a study that reports one uncertainty band over both
# is reporting a number it does not have.
#
# Extracted at the second study that needed it. The rows themselves stay with
# the study that knows what it asked of each quantity; only the shape and the
# rendering are shared.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

__all__ = [
    "INSIDE",
    "OUTSIDE",
    "UNMEASURED",
    "BoundaryRow",
    "tally",
    "text_table",
    "toml_lines",
]

INSIDE: Final = "inside_published_range"
OUTSIDE: Final = "outside_published_range"
UNMEASURED: Final = "no_published_range"


@dataclass(frozen=True, slots=True)
class BoundaryRow:
    quantity: str
    published_range: str
    used: str
    status: str
    basis: str


def tally(rows: Sequence[BoundaryRow]) -> str:
    counts = {
        status: sum(1 for row in rows if row.status == status)
        for status in (INSIDE, OUTSIDE, UNMEASURED)
    }
    return (
        f"Of {len(rows)} quantities, {counts[INSIDE]} sit inside a published "
        f"range, {counts[OUTSIDE]} outside one, and {counts[UNMEASURED]} have "
        "no published range at all."
    )


def toml_lines(rows: Sequence[BoundaryRow]) -> list[str]:
    return [
        line
        for row in rows
        for line in (
            "[[boundary_row]]",
            f'quantity = "{row.quantity}"',
            f'published_range = "{row.published_range}"',
            f'used = "{row.used}"',
            f'status = "{row.status}"',
            f'basis = "{row.basis}"',
            "",
        )
    ]


def text_table(rows: Sequence[BoundaryRow]) -> str:
    headings = ("quantity", "published", "used", "status")
    columns = [
        (row.quantity, row.published_range, row.used, row.status) for row in rows
    ]
    widths = [
        max(len(heading), *(len(cells[index]) for cells in columns))
        for index, heading in enumerate(headings)
    ]
    return "\n".join(
        "  "
        + "  ".join(
            cell.ljust(width) for cell, width in zip(cells, widths)
        ).rstrip()
        for cells in (headings, *columns)
    )
