# SPDX-License-Identifier: Apache-2.0
#
# eclipse.sortie — energy along a route over real ground, out and back.
#
# The first layer that composes everything below it: terrain gives slope along a
# line, the traction balance gives the slip that slope demands, the mobility
# model gives a cost per metre, and integrating gives a number with a journey in
# it. Nothing here is new physics. What is new is that the inputs are a measured
# surface rather than a swept parameter.
#
# This is not a sortie envelope and the naming should not suggest otherwise. A
# sortie needs six axes -- slope, illumination, thermal, power, comms, cold-trap
# range -- and one is populated. What this computes is the locomotion energy of
# a route, which is one term of one axis, and its value is partly in showing how
# small a term it is.
#
# A transect, not a plan. The route is a straight line sampled from the grid,
# because a planner would optimise against the same cost model being evaluated
# and confound the measurement with the search. Route optimisation is a later
# rung and it can only reduce the number computed here.
#
# The asymmetry is the point. Going down, gravity does negative work while shear,
# compaction and swing stay positive and indifferent to direction. Going up,
# every term is positive. So a return leg is not an outbound leg reversed and a
# round trip costs more than twice the cheaper direction.
#
# Descending does not refund energy. Where the gravitational term exceeds the
# dissipative ones the segment is free, not negative: a leg without regeneration
# dissipates that work rather than storing it. Clamping at zero is the
# conservative reading and it is why a steep descent saves less than gravity
# suggests.
#
# References
#   Carrier WD III, Olhoeft GR, Mendell W (1991) Physical Properties of the
#     Lunar Surface. In: Lunar Sourcebook, ch. 9. Cambridge University Press.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from eclipse.io.terrain import GeoRaster
from eclipse.mobility import cost_of_transport
from eclipse.platform import Platform, equilibrium_slip_ratio, swing_work_per_meter
from eclipse.terramechanics import (
    ContactModel,
    JanosiHanamotoModel,
    MohrCoulombModel,
)

__all__ = [
    "Leg",
    "RoundTrip",
    "Transect",
    "sample_transect",
    "walk_leg",
    "walk_round_trip",
]

JOULES_PER_WATT_HOUR: Final = 3600.0


@dataclass(frozen=True, slots=True)
class Transect:
    """Elevation sampled along a straight line, with distance measured on the ground.

    Ground distance rather than map distance: the along-line spacing is the
    horizontal step, and the surface a platform walks is longer than its
    projection wherever the ground is not level.
    """

    distance_m: NDArray[np.float64]
    elevation_m: NDArray[np.float64]

    @property
    def slope_degrees(self) -> NDArray[np.float64]:
        """Signed slope of each segment, positive where the route climbs."""
        rise = np.diff(self.elevation_m)
        run = np.diff(self.distance_m)
        return np.asarray(np.degrees(np.arctan2(rise, run)))

    @property
    def segment_length_m(self) -> NDArray[np.float64]:
        rise = np.diff(self.elevation_m)
        run = np.diff(self.distance_m)
        return np.asarray(np.hypot(rise, run))

    @property
    def descent_m(self) -> float:
        return float(self.elevation_m[0] - self.elevation_m[-1])

    def reversed(self) -> Transect:
        return Transect(
            distance_m=float(self.distance_m[-1]) - self.distance_m[::-1],
            elevation_m=self.elevation_m[::-1],
        )


def sample_transect(
    raster: GeoRaster,
    *,
    start_row_column: tuple[int, int],
    end_row_column: tuple[int, int],
    samples: int,
) -> Transect:
    """Elevation along a straight raster line, by nearest cell.

    Nearest rather than interpolated on purpose. The grid is already about nine
    tenths interpolated by its producers, and adding a second interpolation
    would smooth the profile further while looking like extra resolution.
    """
    if samples < 2:
        raise ValueError(f"a transect needs at least two samples, got {samples}")
    rows = np.linspace(start_row_column[0], end_row_column[0], samples)
    columns = np.linspace(start_row_column[1], end_row_column[1], samples)

    height, width = raster.values.shape
    if not (
        0 <= rows.min() and rows.max() < height and 0 <= columns.min() and columns.max() < width
    ):
        raise ValueError(
            "the transect leaves the raster; endpoints must both lie inside "
            f"a {height} by {width} grid"
        )

    elevation = raster.values[
        np.rint(rows).astype(int), np.rint(columns).astype(int)
    ]
    step = raster.cell_size_m * math.hypot(
        (end_row_column[0] - start_row_column[0]) / (samples - 1),
        (end_row_column[1] - start_row_column[1]) / (samples - 1),
    )
    # Sampling finer than the grid is not extra resolution, it is a fabrication
    # of one. Consecutive samples land on the same cell and report no rise,
    # then a whole cell-to-cell rise is charged against a sub-cell run and the
    # segment comes out far steeper than the ground is. Day 11 asked for four
    # hundred samples across half a kilometre on a five metre grid and got two
    # metre steps, which turned a thirty degree slope into a sixty-six degree
    # cliff and closed a candidate region that was never closed.
    if step < raster.cell_size_m:
        raise ValueError(
            f"{samples} samples across this line gives a {step:.3g} m step on a "
            f"{raster.cell_size_m:.3g} m grid, which is finer than the grid "
            "resolves. Consecutive samples would share cells and the slopes "
            "would come out steeper than the terrain is; ask for at most "
            f"{int(math.floor(max(abs(end_row_column[0] - start_row_column[0]), abs(end_row_column[1] - start_row_column[1])))) + 1} "
            "samples"
        )
    return Transect(
        distance_m=np.arange(samples, dtype=np.float64) * step,
        elevation_m=np.asarray(elevation, dtype=np.float64),
    )


@dataclass(frozen=True, slots=True)
class Leg:
    slope_degrees: NDArray[np.float64]
    segment_length_m: NDArray[np.float64]
    slip_ratio: NDArray[np.float64]
    gravitational_J: NDArray[np.float64]
    dissipative_J: NDArray[np.float64]

    @property
    def segment_J(self) -> NDArray[np.float64]:
        """Clamped at zero: a descent steep enough to be free is not a source."""
        return np.asarray(np.maximum(self.gravitational_J + self.dissipative_J, 0.0))

    @property
    def total_J(self) -> float:
        return float(self.segment_J.sum())

    @property
    def cumulative_J(self) -> NDArray[np.float64]:
        return np.asarray(np.cumsum(self.segment_J))

    @property
    def free_fraction(self) -> float:
        """Share of the route where gravity more than pays for the ground."""
        return float(
            (self.gravitational_J + self.dissipative_J <= 0.0).mean()
        )

    @property
    def distance_m(self) -> float:
        return float(self.segment_length_m.sum())


def walk_leg(
    *,
    transect: Transect,
    platform: Platform,
    contact_model: ContactModel,
    strength: MohrCoulombModel,
    mobilization: JanosiHanamotoModel,
    gravity_m_per_s2: float,
    feet_in_stance: int,
    level_ground_slip_ratio: float,
) -> Leg:
    """Energy for each segment of one direction of travel.

    Slip is the larger of what the slope demands and what the gait demands on
    the level, since a platform cannot slip less than its own legs make it.
    """
    slope = transect.slope_degrees
    lengths = transect.segment_length_m

    slope_driven = equilibrium_slip_ratio(
        platform=platform,
        strength=strength,
        mobilization=mobilization,
        gravity_m_per_s2=gravity_m_per_s2,
        slope_degrees=np.abs(slope),
    )
    slip = np.maximum(slope_driven, level_ground_slip_ratio)

    gravitational = np.empty_like(slope)
    dissipative = np.empty_like(slope)
    for index, (angle, ratio) in enumerate(zip(slope, slip)):
        if not math.isfinite(ratio) or ratio >= 1.0:
            gravitational[index] = math.nan
            dissipative[index] = math.nan
            continue
        swing = float(
            swing_work_per_meter(
                platform=platform,
                gravity_m_per_s2=gravity_m_per_s2,
                slip_ratio=float(ratio),
            ).total_J
        )
        cost = cost_of_transport(
            mass_kg=platform.total_mass_kg,
            gravity_m_per_s2=gravity_m_per_s2,
            slope_degrees=float(angle),
            slip_ratio=float(ratio),
            patch=platform.contact_patch,
            feet_in_stance=feet_in_stance,
            stride_length_m=platform.stride_length_m,
            stance_length_m=platform.stride_length_m,
            contact_model=contact_model,
            strength=strength,
            mobilization=mobilization,
            swing_work_per_meter_J=swing,
        )
        gravitational[index] = float(cost.gravitational_J_per_m) * lengths[index]
        dissipative[index] = (
            float(cost.shear_J_per_m)
            + float(cost.compaction_J_per_m)
            + float(cost.swing_J_per_m)
        ) * lengths[index]

    return Leg(
        slope_degrees=slope,
        segment_length_m=lengths,
        slip_ratio=np.asarray(slip),
        gravitational_J=gravitational,
        dissipative_J=dissipative,
    )


@dataclass(frozen=True, slots=True)
class RoundTrip:
    outbound: Leg
    inbound: Leg

    @property
    def total_J(self) -> float:
        return self.outbound.total_J + self.inbound.total_J

    @property
    def asymmetry(self) -> float:
        """Return leg over outbound leg. One would mean direction did not matter."""
        return self.inbound.total_J / self.outbound.total_J

    @property
    def over_twice_outbound(self) -> float:
        return self.total_J / (2.0 * self.outbound.total_J)

    def battery_J_required(self, *, derating: float) -> float:
        return self.total_J * derating

    def reachable_index(self, *, battery_J: float, derating: float) -> int:
        """How far along the outbound leg a budget reaches, allowing for the return.

        The cost of turning back at a node is everything spent reaching it plus
        everything the return costs from there. The second term is the tail of
        the return leg, not its head: the return is walked from the far end, so
        the energy to come home from node k is what remains of it after the
        first n-k segments. Getting that backwards makes the far end look
        cheapest, which is how this was caught.
        """
        out = np.concatenate([[0.0], self.outbound.cumulative_J])
        remaining = self.inbound.total_J - self.inbound.cumulative_J[::-1]
        back = np.concatenate([remaining, [self.inbound.total_J]])
        required = (out + back) * derating
        affordable = np.flatnonzero(required <= battery_J)
        return int(affordable[-1]) if affordable.size else 0


def walk_round_trip(
    *,
    transect: Transect,
    platform: Platform,
    contact_model: ContactModel,
    strength: MohrCoulombModel,
    mobilization: JanosiHanamotoModel,
    gravity_m_per_s2: float,
    feet_in_stance: int,
    level_ground_slip_ratio: float,
) -> RoundTrip:
    common = {
        "platform": platform,
        "contact_model": contact_model,
        "strength": strength,
        "mobilization": mobilization,
        "gravity_m_per_s2": gravity_m_per_s2,
        "feet_in_stance": feet_in_stance,
        "level_ground_slip_ratio": level_ground_slip_ratio,
    }
    return RoundTrip(
        outbound=walk_leg(transect=transect, **common),  # type: ignore[arg-type]
        inbound=walk_leg(transect=transect.reversed(), **common),  # type: ignore[arg-type]
    )
