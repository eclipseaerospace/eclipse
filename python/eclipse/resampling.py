# SPDX-License-Identifier: Apache-2.0
#
# eclipse.resampling — put repeat tests on a common grid and average them.
#
# Repeat bevameter tests stop at whatever pressure each run reached and sample
# it at whatever points the operator loaded, so the runs share no abscissa and
# cannot be averaged as they stand. Interpolating each onto one grid first is
# what makes an ensemble mean meaningful, and it is the step published
# bevameter analyses take before fitting.
#
# The interpolant is a not-a-knot cubic spline, which is what the reference
# analyses use and therefore what reproducing them requires. Not-a-knot drops
# the artificial end condition a natural spline imposes by requiring the third
# derivative to stay continuous across the second and second-to-last knots,
# which makes the spline reproduce any cubic exactly.
#
# Evaluation outside the sampled range continues the end piece rather than
# refusing, because the published grids deliberately run past where the larger
# plates stopped. That is extrapolation and it is the caller's judgement to
# make, so the sampled bounds are carried on the result for the caller to check
# against rather than being silently clamped here.
#
# Deviation is the sample standard deviation, over repeats at each grid point,
# normalized by one less than the count. A population deviation would
# understate the scatter of a handful of runs.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from eclipse._validation import first_violation

__all__ = [
    "EnsembleCurve",
    "MINIMUM_SPLINE_SAMPLES",
    "ensemble",
    "resample_cubic",
]

MINIMUM_SPLINE_SAMPLES: Final = 4
MINIMUM_ENSEMBLE_CURVES: Final = 2


def _as_strictly_increasing(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 1:
        raise ValueError(
            f"sample positions must be one-dimensional, got shape {positions.shape}"
        )
    if positions.size < MINIMUM_SPLINE_SAMPLES:
        raise ValueError(
            f"a not-a-knot cubic spline needs at least {MINIMUM_SPLINE_SAMPLES} "
            f"samples, got {positions.size}. Below that the end condition has no "
            "interior knot to attach to"
        )
    violations = np.asarray(~np.isfinite(positions))
    if violations.any():
        count, total, first = first_violation(violations, positions)
        raise ValueError(
            f"sample positions must be finite; {count} of {total} are not, the "
            f"first being {first}"
        )
    gaps = np.diff(positions)
    violations = np.asarray(gaps <= 0.0)
    if violations.any():
        count, total, first = first_violation(violations, gaps)
        raise ValueError(
            f"sample positions must be strictly increasing; {count} of {total} "
            f"steps are not, the first being {first}. Duplicated or unsorted "
            "abscissae make the interpolant undefined rather than inaccurate"
        )
    return positions


def _second_derivatives(
    positions: NDArray[np.float64], values: NDArray[np.float64]
) -> NDArray[np.float64]:
    count = positions.size
    gaps = np.diff(positions)
    slopes = np.diff(values) / gaps

    system = np.zeros((count, count), dtype=np.float64)
    target = np.zeros(count, dtype=np.float64)

    rows = np.arange(1, count - 1)
    system[rows, rows - 1] = gaps[:-1]
    system[rows, rows] = 2.0 * (gaps[:-1] + gaps[1:])
    system[rows, rows + 1] = gaps[1:]
    target[1:-1] = 6.0 * (slopes[1:] - slopes[:-1])

    system[0, 0] = gaps[1]
    system[0, 1] = -(gaps[0] + gaps[1])
    system[0, 2] = gaps[0]
    system[-1, -3] = gaps[-1]
    system[-1, -2] = -(gaps[-2] + gaps[-1])
    system[-1, -1] = gaps[-2]

    return np.asarray(np.linalg.solve(system, target))


def resample_cubic(
    *,
    sample_positions: NDArray[np.float64],
    sample_values: NDArray[np.float64],
    positions: NDArray[np.float64],
) -> NDArray[np.float64]:
    abscissa = _as_strictly_increasing(sample_positions)
    ordinate = np.asarray(sample_values, dtype=np.float64)
    if ordinate.shape != abscissa.shape:
        raise ValueError(
            f"sample values have shape {ordinate.shape} against positions "
            f"{abscissa.shape}; the two are recorded together and must match"
        )
    violations = np.asarray(~np.isfinite(ordinate))
    if violations.any():
        count, total, first = first_violation(violations, ordinate)
        raise ValueError(
            f"sample values must be finite; {count} of {total} are not, the "
            f"first being {first}"
        )

    curvature = _second_derivatives(abscissa, ordinate)
    gaps = np.diff(abscissa)
    slopes = np.diff(ordinate) / gaps
    linear = slopes - gaps * (2.0 * curvature[:-1] + curvature[1:]) / 6.0
    cubic = (curvature[1:] - curvature[:-1]) / (6.0 * gaps)

    wanted = np.asarray(positions, dtype=np.float64)
    piece = np.clip(
        np.searchsorted(abscissa, wanted, side="right") - 1, 0, abscissa.size - 2
    )
    offset = wanted - abscissa[piece]
    return np.asarray(
        ordinate[piece]
        + linear[piece] * offset
        + curvature[piece] / 2.0 * offset**2
        + cubic[piece] * offset**3
    )


@dataclass(frozen=True, slots=True, eq=False)
class EnsembleCurve:
    positions: NDArray[np.float64]
    mean_values: NDArray[np.float64]
    deviation: NDArray[np.float64]
    curve_count: int
    sampled_maximum: float

    @property
    def extrapolated(self) -> NDArray[np.bool_]:
        return np.asarray(self.positions > self.sampled_maximum)

    @property
    def maximum_deviation(self) -> float:
        return float(np.max(self.deviation))


def ensemble(
    *,
    sample_positions: Sequence[NDArray[np.float64]],
    sample_values: Sequence[NDArray[np.float64]],
    positions: NDArray[np.float64],
) -> EnsembleCurve:
    if len(sample_positions) != len(sample_values):
        raise ValueError(
            f"{len(sample_positions)} position arrays against "
            f"{len(sample_values)} value arrays"
        )
    if len(sample_positions) < MINIMUM_ENSEMBLE_CURVES:
        raise ValueError(
            f"an ensemble needs at least {MINIMUM_ENSEMBLE_CURVES} curves to "
            f"carry a deviation, got {len(sample_positions)}"
        )
    grid = np.asarray(positions, dtype=np.float64)
    resampled = np.vstack(
        [
            resample_cubic(
                sample_positions=abscissa, sample_values=ordinate, positions=grid
            )
            for abscissa, ordinate in zip(sample_positions, sample_values)
        ]
    )
    return EnsembleCurve(
        positions=grid,
        mean_values=np.asarray(resampled.mean(axis=0)),
        deviation=np.asarray(resampled.std(axis=0, ddof=1)),
        curve_count=len(sample_positions),
        sampled_maximum=float(min(np.max(abscissa) for abscissa in sample_positions)),
    )
