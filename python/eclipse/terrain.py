# SPDX-License-Identifier: Apache-2.0
#
# eclipse.terrain — slope from a grid, and how it depends on the grid.
#
# Slope is not a property of terrain. It is a property of terrain at a baseline,
# and the baseline is whatever the grid happens to post at. Two things follow
# that this module exists to make measurable rather than assumable.
#
# The algorithm matters. Central difference, Horn and Zevenbergen-Thorne give
# measurably different distributions on the same grid, so the method is named in
# every output rather than left to a default. Which one a published product used
# can be recovered by reproducing its own slope raster, and for the LOLA polar
# DEMs that identification comes out as central difference.
#
# The baseline matters more. For a self-affine surface the mean slope over a
# baseline L goes as L to the power H-1, with H the Hurst exponent, so slope
# statistics steepen as the baseline shortens and the rate is a property of the
# terrain. Measuring that exponent is how a grid is asked whether it contains
# roughness at all, and a value near zero says it does not -- either because the
# surface really is smooth over that range, or because whatever produced the
# grid smoothed it.
#
# Those two explanations are separable when the smoothing has a direction. An
# altimeter in a near-polar orbit samples densely along track and sparsely
# across it, so gap-filling at a polar site should leave anisotropy aligned with
# the orbit. Terrain on a crater wall is anisotropic too, but aligned with the
# fall line. Comparing the measured principal axis against both is a test with
# no free parameters, and it discriminates the mechanism that would matter most.
#
# It does not discriminate every mechanism. An interpolator that smooths
# isotropically leaves no directional signature and this test cannot see it, so
# a terrain-aligned result rejects the artifact it can test for rather than
# establishing that the surface is real.
#
# References
#   Horn BKP (1981) Hill shading and the reflectance map. Proceedings of the
#     IEEE 69(1), 14-47.
#   Shepard MK et al. (2001) The roughness of natural terrain. Journal of
#     Geophysical Research 106(E12), 32777-32795.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DirectionalRoughness",
    "SLOPE_METHODS",
    "ScaleTrend",
    "aggregate",
    "anisotropy",
    "directional_rms_slope_degrees",
    "scale_trend",
    "slope_degrees",
]

DEGREES_IN_HALF_TURN: Final = 180.0

# Published Hurst exponents for natural terrain sit well below one, so the mean
# slope exponent H-1 sits in this band. A measured exponent far above it is a
# statement that the grid holds no roughness over the range measured.
NATURAL_TERRAIN_SLOPE_EXPONENT: Final = (-0.30, -0.10)


def _central(values: NDArray[np.float64], cell_size_m: float) -> NDArray[np.float64]:
    padded = np.pad(values, 1, mode="edge")
    east_west = (padded[1:-1, 2:] - padded[1:-1, :-2]) / (2.0 * cell_size_m)
    north_south = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2.0 * cell_size_m)
    return np.asarray(np.hypot(east_west, north_south))


def _horn(values: NDArray[np.float64], cell_size_m: float) -> NDArray[np.float64]:
    padded = np.pad(values, 1, mode="edge")
    east_west = (
        (padded[:-2, 2:] + 2.0 * padded[1:-1, 2:] + padded[2:, 2:])
        - (padded[:-2, :-2] + 2.0 * padded[1:-1, :-2] + padded[2:, :-2])
    ) / (8.0 * cell_size_m)
    north_south = (
        (padded[2:, :-2] + 2.0 * padded[2:, 1:-1] + padded[2:, 2:])
        - (padded[:-2, :-2] + 2.0 * padded[:-2, 1:-1] + padded[:-2, 2:])
    ) / (8.0 * cell_size_m)
    return np.asarray(np.hypot(east_west, north_south))


SLOPE_METHODS: Final = {"central_difference": _central, "horn": _horn}


def slope_degrees(
    values: NDArray[np.float64], *, cell_size_m: float, method: str = "central_difference"
) -> NDArray[np.float64]:
    if method not in SLOPE_METHODS:
        raise ValueError(
            f"no slope method {method!r}; this module has {sorted(SLOPE_METHODS)}. "
            "The choice changes the distribution measurably, so it is named "
            "rather than defaulted silently"
        )
    if not (math.isfinite(cell_size_m) and cell_size_m > 0.0):
        raise ValueError(f"cell_size_m must be finite and positive, got {cell_size_m}")
    return np.degrees(np.arctan(SLOPE_METHODS[method](values, cell_size_m)))


def aggregate(values: NDArray[np.float64], factor: int) -> NDArray[np.float64]:
    """Coarsen by block mean, which is what changing the baseline means here."""
    if factor < 1:
        raise ValueError(f"factor must be at least one, got {factor}")
    if factor == 1:
        return values
    rows = values.shape[0] // factor * factor
    columns = values.shape[1] // factor * factor
    block = values[:rows, :columns].reshape(
        rows // factor, factor, columns // factor, factor
    )
    return np.asarray(block.mean(axis=(1, 3)))


@dataclass(frozen=True, slots=True)
class ScaleTrend:
    baseline_m: NDArray[np.float64]
    mean_slope_degrees: NDArray[np.float64]

    @property
    def exponent(self) -> float:
        """The fitted power of baseline, which is the Hurst exponent minus one."""
        fit = np.polyfit(
            np.log(self.baseline_m), np.log(self.mean_slope_degrees), 1
        )
        return float(fit[0])

    @property
    def hurst_exponent(self) -> float:
        return 1.0 + self.exponent

    @property
    def holds_roughness(self) -> bool:
        """Whether the measured exponent is as steep as natural terrain gives."""
        return self.exponent <= NATURAL_TERRAIN_SLOPE_EXPONENT[1]


def scale_trend(
    values: NDArray[np.float64],
    *,
    cell_size_m: float,
    factors: tuple[int, ...],
    method: str = "central_difference",
) -> ScaleTrend:
    baselines, means = [], []
    for factor in factors:
        coarse = aggregate(values, factor)
        slope = slope_degrees(
            coarse, cell_size_m=cell_size_m * factor, method=method
        )[1:-1, 1:-1]
        usable = slope[np.isfinite(slope)]
        if usable.size == 0:
            raise ValueError(
                f"aggregating by {factor} left no interior cells to measure; the "
                "grid is too small for that baseline"
            )
        baselines.append(cell_size_m * factor)
        means.append(float(usable.mean()))
    return ScaleTrend(
        baseline_m=np.asarray(baselines, dtype=np.float64),
        mean_slope_degrees=np.asarray(means, dtype=np.float64),
    )


def directional_rms_slope_degrees(
    values: NDArray[np.float64],
    *,
    cell_size_m: float,
    lag_cells: int,
    axis_degrees: float,
) -> float:
    """Root-mean-square slope along one direction at one lag.

    The axis is measured from the raster's row direction toward its column
    direction, and is an axis rather than a heading: a lag and its negation give
    the same answer, so directions are reported modulo half a turn.
    """
    if lag_cells < 1:
        raise ValueError(f"lag_cells must be at least one, got {lag_cells}")
    # No guard against a zero offset: it cannot happen. Both components round
    # to zero only if the lag is under one over root two, and the lag is at
    # least one cell.
    angle = math.radians(axis_degrees)
    down = int(round(lag_cells * math.cos(angle)))
    across = int(round(lag_cells * math.sin(angle)))

    rows, columns = values.shape
    row_from = slice(max(0, -down), rows - max(0, down))
    row_to = slice(max(0, down), rows - max(0, -down))
    column_from = slice(max(0, -across), columns - max(0, across))
    column_to = slice(max(0, across), columns - max(0, -across))

    difference = values[row_to, column_to] - values[row_from, column_from]
    separation = cell_size_m * math.hypot(down, across)
    finite = difference[np.isfinite(difference)]
    return math.degrees(
        math.atan(float(np.sqrt(np.mean(finite**2))) / separation)
    )


@dataclass(frozen=True, slots=True)
class DirectionalRoughness:
    axis_degrees: NDArray[np.float64]
    rms_slope_degrees: NDArray[np.float64]
    lag_m: float

    @property
    def roughest_axis_degrees(self) -> float:
        return float(self.axis_degrees[int(np.argmax(self.rms_slope_degrees))])

    @property
    def smoothest_axis_degrees(self) -> float:
        return float(self.axis_degrees[int(np.argmin(self.rms_slope_degrees))])

    @property
    def ratio(self) -> float:
        return float(
            self.rms_slope_degrees.max() / self.rms_slope_degrees.min()
        )

    def separation_from(self, axis_degrees: float) -> float:
        gap = abs(self.roughest_axis_degrees - axis_degrees) % DEGREES_IN_HALF_TURN
        return min(gap, DEGREES_IN_HALF_TURN - gap)


def anisotropy(
    values: NDArray[np.float64],
    *,
    cell_size_m: float,
    lag_cells: int,
    step_degrees: float = 10.0,
) -> DirectionalRoughness:
    axes = np.arange(0.0, DEGREES_IN_HALF_TURN, step_degrees)
    rms = np.asarray(
        [
            directional_rms_slope_degrees(
                values,
                cell_size_m=cell_size_m,
                lag_cells=lag_cells,
                axis_degrees=float(axis),
            )
            for axis in axes
        ]
    )
    return DirectionalRoughness(
        axis_degrees=axes, rms_slope_degrees=rms, lag_m=cell_size_m * lag_cells
    )
