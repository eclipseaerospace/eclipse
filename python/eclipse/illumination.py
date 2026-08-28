# SPDX-License-Identifier: Apache-2.0
#
# eclipse.illumination — where the Sun reaches, computed from the ground.
#
# The axis that has quietly been doing the work in every earlier result. Day 7
# walked to a destination whose darkness was never established. Day 8 priced a
# survival power for a shadow whose duration was assumed. Both rest on this, and
# this is measurable from terrain already in hand rather than modelled from
# invented inputs.
#
# Illumination at a point is a geometric question: is the Sun above the local
# horizon, where the horizon is the greatest elevation angle over all azimuths.
# Near a pole that question is delicate in three ways, and each is handled
# explicitly because each would otherwise bias the answer the same way -- toward
# more light than there is.
#
# The Sun is a disc, not a point. It subtends about half a degree, and at a pole
# where solar elevations are a couple of degrees at most, that is the difference
# between lit, penumbral and dark over large areas. A point Sun would report a
# hard shadow boundary that does not exist.
#
# The Moon is round. Over a twenty-kilometre window the curvature drop is
# nearly thirty metres, comparable to the relief that decides a horizon at these
# grazing angles. Ignoring it would make distant terrain look higher than it is
# and overstate shadowing -- the one bias here that runs the other way.
#
# And the horizon does not stop at the edge of the data. A ray that leaves the
# window is treated as clear sky, which is false: distant massifs shadow polar
# sites from tens of kilometres away. Every illumination fraction from a
# truncated DEM is therefore an upper bound, and this module reports the
# truncation rather than hiding it.
#
# Solar position uses a low-precision analytic model rather than SPICE. At a
# lunar pole the Sun's elevation is set by the sub-solar latitude, which
# oscillates within the Moon's small obliquity, and its azimuth sweeps once per
# lunation. Sweeping both over their ranges gives an illumination fraction
# without a kernel, and the answer is a fraction rather than an event time, so
# the ephemeris precision that SPICE would add is not what limits it.
#
# References
#   Mazarico E et al. (2011) Illumination conditions of the lunar polar regions
#     using LOLA topography. Icarus 211, 1066-1081.
#   Paige DA et al. (2010) Science 330, 479-482. doi:10.1126/science.1187726

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from eclipse.io.terrain import GeoRaster

__all__ = [
    "LUNAR_OBLIQUITY_DEG",
    "LUNATION_HOURS",
    "SOLAR_ANGULAR_RADIUS_DEG",
    "SUBSOLAR_LATITUDE_PERIOD_HOURS",
    "HorizonMap",
    "Illumination",
    "IlluminationSeries",
    "ShadowTarget",
    "best_charge_point",
    "contiguous_interval_hours",
    "horizon_elevation_deg",
    "hour_angle_deg",
    "illumination_fraction",
    "illumination_series",
    "shadow_targets",
    "solar_elevation_deg",
    "subsolar_latitude_deg",
]

# The Sun's angular radius at one astronomical unit. Half a degree of disc is
# the whole difference between a hard shadow and a penumbra at a pole.
SOLAR_ANGULAR_RADIUS_DEG: Final = 0.265

# The Moon's obliquity to the ecliptic. The sub-solar latitude stays inside
# this, which is why polar illumination is a question about terrain rather than
# about season.
LUNAR_OBLIQUITY_DEG: Final = 1.54

# The synodic month, which is the period of the solar hour angle at any point
# on the Moon, and the natural clock for a sortie: a seven-hour round trip is a
# hundredth of one.
LUNATION_HOURS: Final = 29.530589 * 24.0

# The sub-solar latitude oscillates within the obliquity once per year. The
# 18.6-year nodal precession that modulates it is deliberately absent; it moves
# the amplitude, not the period, and the amplitude is already the quantity this
# module is least sure of.
SUBSOLAR_LATITUDE_PERIOD_HOURS: Final = 365.25 * 24.0


@dataclass(frozen=True, slots=True)
class HorizonMap:
    """Greatest sky-blocking elevation angle at each azimuth, for one point set.

    Azimuths are measured in the raster frame, from the +row direction toward
    the +column direction, matching eclipse.terrain. Angles are degrees above
    the local horizontal plane; negative means the terrain falls away and the
    sky is open below the horizontal.
    """

    azimuth_deg: NDArray[np.float64]
    elevation_deg: NDArray[np.float64]
    searched_to_m: float
    truncated_fraction: float

    def at(self, azimuth_deg: NDArray[np.float64]) -> NDArray[np.float64]:
        """Horizon at arbitrary azimuths, by interpolation around the circle."""
        wrapped = np.mod(azimuth_deg, 360.0)
        closed_azimuth = np.concatenate([self.azimuth_deg, [360.0]])
        closed_elevation = np.concatenate(
            [self.elevation_deg, self.elevation_deg[..., :1]], axis=-1
        )
        return np.asarray(
            np.interp(wrapped, closed_azimuth, closed_elevation)
        )


def horizon_elevation_deg(
    raster: GeoRaster,
    *,
    rows: NDArray[np.int_],
    columns: NDArray[np.int_],
    azimuths: int = 72,
    samples_along_ray: int = 120,
    minimum_range_m: float = 50.0,
    maximum_range_m: float | None = None,
) -> HorizonMap:
    """Horizon at a set of query points, searched outward along rays.

    Distances are spaced geometrically because the horizon angle a ray can
    contribute falls as one over distance, so near ground deserves far more
    samples than far ground. Rays that leave the grid stop there and the
    fraction of samples lost that way is reported, because a truncated search
    can only overstate how much sky is open.

    The search starts at a stand-off rather than at the first cell, and the
    default matters. At a five-metre lag a two-metre rise subtends nearly
    thirty degrees, so a horizon searched from the adjacent cell is not a
    horizon at all -- it is the local surface tilt, and on a grid whose
    producers report about nine tenths of its pixels as interpolated it is the
    interpolator's tilt. Starting further out reads terrain that the data
    actually resolves. This was found by a point coming out permanently
    shadowed because the cell beside it was two metres higher.
    """
    values = raster.values
    height, width = values.shape
    cell = raster.cell_size_m
    limit = maximum_range_m if maximum_range_m is not None else cell * max(height, width)
    start = max(minimum_range_m, cell)
    if start >= limit:
        raise ValueError(
            f"the stand-off of {start} m leaves no room below the search limit "
            f"of {limit} m; there is nothing to look at"
        )

    base = values[rows, columns]
    azimuth_deg = np.linspace(0.0, 360.0, azimuths, endpoint=False)
    distances = np.geomspace(start, limit, samples_along_ray)

    elevation = np.full((azimuths, rows.size), -90.0, dtype=np.float64)
    lost = 0
    total = 0
    for index, azimuth in enumerate(azimuth_deg):
        angle = math.radians(float(azimuth))
        best = np.full(rows.size, -90.0, dtype=np.float64)
        for distance in distances:
            step_rows = rows + np.rint(distance * math.cos(angle) / cell).astype(int)
            step_columns = columns + np.rint(
                distance * math.sin(angle) / cell
            ).astype(int)
            inside = (
                (step_rows >= 0)
                & (step_rows < height)
                & (step_columns >= 0)
                & (step_columns < width)
            )
            total += inside.size
            lost += int((~inside).sum())
            if not inside.any():
                continue
            # Curvature drops the far point below the local tangent plane by
            # d^2/2R, which at twenty kilometres is tens of metres.
            drop = distance**2 / (2.0 * raster.reference_radius_m)
            rise = (
                values[step_rows[inside], step_columns[inside]]
                - base[inside]
                - drop
            )
            angles = np.degrees(np.arctan2(rise, distance))
            best[inside] = np.maximum(best[inside], angles)
        elevation[index] = best

    return HorizonMap(
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation.T,
        searched_to_m=limit,
        truncated_fraction=lost / max(total, 1),
    )


def solar_elevation_deg(
    *,
    latitude_deg: NDArray[np.float64] | float,
    subsolar_latitude_deg: NDArray[np.float64],
    hour_angle_deg: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Elevation of the Sun's centre above the local horizontal plane."""
    latitude = np.radians(latitude_deg)
    declination = np.radians(subsolar_latitude_deg)
    hour = np.radians(hour_angle_deg)
    sine = np.sin(latitude) * np.sin(declination) + np.cos(latitude) * np.cos(
        declination
    ) * np.cos(hour)
    return np.asarray(np.degrees(np.arcsin(np.clip(sine, -1.0, 1.0))))


def _solar_azimuth_deg(
    *,
    latitude_deg: NDArray[np.float64] | float,
    subsolar_latitude_deg: NDArray[np.float64],
    hour_angle_deg: NDArray[np.float64],
    elevation_deg: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Azimuth east of north, from the standard spherical triangle."""
    latitude = np.radians(latitude_deg)
    declination = np.radians(subsolar_latitude_deg)
    elevation = np.radians(elevation_deg)
    cosine = (np.sin(declination) - np.sin(latitude) * np.sin(elevation)) / (
        np.cos(latitude) * np.cos(elevation)
    )
    azimuth = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return np.asarray(np.where(np.sin(np.radians(hour_angle_deg)) > 0.0, 360.0 - azimuth, azimuth))


@dataclass(frozen=True, slots=True)
class Illumination:
    lit_fraction: NDArray[np.float64]
    penumbral_fraction: NDArray[np.float64]
    horizon: HorizonMap

    @property
    def dark_fraction(self) -> NDArray[np.float64]:
        return np.asarray(1.0 - self.lit_fraction - self.penumbral_fraction)

    @property
    def any_sunlight_fraction(self) -> NDArray[np.float64]:
        return np.asarray(self.lit_fraction + self.penumbral_fraction)


def illumination_fraction(
    *,
    horizon: HorizonMap,
    latitude_deg: NDArray[np.float64] | float,
    north_azimuth_deg: NDArray[np.float64],
    lunation_samples: int = 180,
    season_samples: int = 9,
) -> Illumination:
    """Fraction of a year of lunations for which the Sun's disc clears the horizon.

    Swept over both cycles that matter at a pole: the hour angle once per
    lunation, and the sub-solar latitude once per year within the Moon's
    obliquity. Fully lit means the whole disc is clear; penumbral means part of
    it is, which near a pole covers a great deal of ground.

    north_azimuth_deg converts between the solar azimuth, which is measured
    from north, and the raster frame the horizon is stored in. It varies from
    point to point in a polar projection because every meridian points a
    different way on the grid.

    latitude_deg is per point, or one value broadcast over all of them. A
    twenty-kilometre window near a pole spans two thirds of a degree of
    latitude, which is a large fraction of the obliquity that drives the
    seasonal sweep, so a single mean latitude for a batch makes each point's
    illumination depend on which other points were computed alongside it.
    """
    hour = np.linspace(0.0, 360.0, lunation_samples, endpoint=False)
    declination = np.linspace(
        -LUNAR_OBLIQUITY_DEG, LUNAR_OBLIQUITY_DEG, season_samples
    )
    hour_grid, declination_grid = np.meshgrid(hour, declination, indexing="ij")

    points = horizon.elevation_deg.shape[0]
    latitude = np.broadcast_to(
        np.asarray(latitude_deg, dtype=np.float64), (points,)
    )
    lit = np.zeros(points, dtype=np.float64)
    penumbral = np.zeros(points, dtype=np.float64)
    wrapped_azimuth = np.concatenate([horizon.azimuth_deg, [360.0]])

    for index in range(points):
        elevation = solar_elevation_deg(
            latitude_deg=float(latitude[index]),
            subsolar_latitude_deg=declination_grid,
            hour_angle_deg=hour_grid,
        )
        flat_elevation = elevation.ravel()
        flat_azimuth = _solar_azimuth_deg(
            latitude_deg=float(latitude[index]),
            subsolar_latitude_deg=declination_grid,
            hour_angle_deg=hour_grid,
            elevation_deg=elevation,
        ).ravel()
        raster_azimuth = np.mod(flat_azimuth + north_azimuth_deg[index], 360.0)
        blocked = np.interp(
            raster_azimuth,
            wrapped_azimuth,
            np.concatenate(
                [horizon.elevation_deg[index], horizon.elevation_deg[index, :1]]
            ),
        )
        clearance = flat_elevation - blocked
        lit[index] = float(np.mean(clearance >= SOLAR_ANGULAR_RADIUS_DEG))
        penumbral[index] = float(
            np.mean(
                (clearance > -SOLAR_ANGULAR_RADIUS_DEG)
                & (clearance < SOLAR_ANGULAR_RADIUS_DEG)
            )
        )

    return Illumination(
        lit_fraction=lit, penumbral_fraction=penumbral, horizon=horizon
    )


@dataclass(frozen=True, slots=True)
class IlluminationSeries:
    """Sun clearance above the horizon at each point, through time.

    clearance_deg is the Sun's centre elevation minus the horizon it is up
    against, in degrees, indexed point-major. Positive by more than the solar
    angular radius is fully lit; negative by more than it is fully dark; between
    the two the disc is partly clear, which near a pole covers a great deal of
    ground and a great deal of time.
    """

    hours: NDArray[np.float64]
    clearance_deg: NDArray[np.float64]
    horizon: HorizonMap

    @property
    def lit(self) -> NDArray[np.bool_]:
        return np.asarray(self.clearance_deg >= SOLAR_ANGULAR_RADIUS_DEG)

    @property
    def dark(self) -> NDArray[np.bool_]:
        return np.asarray(self.clearance_deg <= -SOLAR_ANGULAR_RADIUS_DEG)

    @property
    def any_sunlight(self) -> NDArray[np.bool_]:
        return np.asarray(self.clearance_deg > -SOLAR_ANGULAR_RADIUS_DEG)


def subsolar_latitude_deg(hours: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sub-solar latitude, oscillating within the obliquity once per year."""
    return np.asarray(
        LUNAR_OBLIQUITY_DEG
        * np.sin(2.0 * np.pi * hours / SUBSOLAR_LATITUDE_PERIOD_HOURS)
    )


def hour_angle_deg(hours: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solar hour angle, sweeping once per synodic month."""
    return np.asarray(360.0 * hours / LUNATION_HOURS)


def illumination_series(
    *,
    horizon: HorizonMap,
    latitude_deg: NDArray[np.float64] | float,
    north_azimuth_deg: NDArray[np.float64],
    hours: NDArray[np.float64],
) -> IlluminationSeries:
    """Sun clearance at each point over a given run of hours.

    illumination_fraction answers what share of a year a point sees sunlight.
    This answers when, which is a different question and the one a schedule
    turns on: a sortie shorter than a lunation is cold or not depending on
    where in the cycle it departs.

    The two do not have to agree exactly and should not be expected to.
    illumination_fraction samples the sub-solar latitude uniformly in angle;
    time spends longer near the extremes of a sinusoid than near the middle, so
    a long enough series weights the seasons differently. Which is the more
    physical weighting is not in question -- this one is -- but the fraction is
    a summary and this is a history, and they are reported separately for that
    reason.
    """
    points = horizon.elevation_deg.shape[0]
    latitude = np.broadcast_to(
        np.asarray(latitude_deg, dtype=np.float64), (points,)
    )
    declination = subsolar_latitude_deg(hours)
    angle = hour_angle_deg(hours)
    wrapped_azimuth = np.concatenate([horizon.azimuth_deg, [360.0]])

    clearance = np.zeros((points, hours.size), dtype=np.float64)
    for index in range(points):
        elevation = solar_elevation_deg(
            latitude_deg=float(latitude[index]),
            subsolar_latitude_deg=declination,
            hour_angle_deg=angle,
        )
        azimuth = _solar_azimuth_deg(
            latitude_deg=float(latitude[index]),
            subsolar_latitude_deg=declination,
            hour_angle_deg=angle,
            elevation_deg=elevation,
        )
        blocked = np.interp(
            np.mod(azimuth + north_azimuth_deg[index], 360.0),
            wrapped_azimuth,
            np.concatenate(
                [horizon.elevation_deg[index], horizon.elevation_deg[index, :1]]
            ),
        )
        clearance[index] = elevation - blocked

    return IlluminationSeries(
        hours=hours, clearance_deg=clearance, horizon=horizon
    )


def contiguous_interval_hours(
    *, mask: NDArray[np.bool_], hours: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Durations of the runs of True in mask, dropping any that touch an end.

    A run reaching either end of the window is censored -- the window stopped
    it, not the terrain -- so reporting its length would understate it and
    reporting it as complete would be wrong. Sweep more lunations if the
    dropped runs matter.
    """
    if mask.ndim != 1:
        raise ValueError(
            "contiguous_interval_hours takes one point's mask at a time; "
            f"got an array of {mask.ndim} dimensions with shape {mask.shape}"
        )
    if mask.size != hours.size:
        raise ValueError(
            "mask and hours must describe the same samples; got "
            f"{mask.size} mask entries against {hours.size} hours"
        )
    if mask.size == 0 or not bool(mask.any()):
        return np.zeros(0, dtype=np.float64)

    padded = np.concatenate([[False], mask, [False]])
    change = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(change == 1)
    stops = np.flatnonzero(change == -1) - 1
    keep = (starts > 0) & (stops < mask.size - 1)
    step = float(np.mean(np.diff(hours))) if hours.size > 1 else 0.0
    return np.asarray(
        hours[stops[keep]] - hours[starts[keep]] + step, dtype=np.float64
    )


@dataclass(frozen=True, slots=True)
class ShadowTarget:
    """One candidate destination in permanent shadow, and why it was chosen.

    Which shadow a mission visits is a design question rather than a modelling
    detail: the nearest is cheapest, the largest gives the most ground to work
    over, and the deepest is a different undertaking. Naming them separately is
    what lets the three be priced against each other instead of one being
    assumed.
    """

    id: str
    row: int
    column: int
    distance_km: float
    drop_m: float
    region_area_km2: float


def shadow_targets(
    raster: GeoRaster,
    *,
    start: tuple[int, int],
    rows: NDArray[np.int_],
    columns: NDArray[np.int_],
    any_sunlight_fraction: NDArray[np.float64],
) -> dict[str, ShadowTarget]:
    """Nearest, largest and deepest permanent shadow, from an illumination grid.

    rows and columns index the raster at the sampled points and carry the grid
    shape; any_sunlight_fraction is the illumination at those same points.
    Regions are four-connected components of the fully dark cells. The largest
    is entered at its nearest member rather than its centroid, because a
    platform walks to the edge of a shadow and not into the middle of it.
    """
    if rows.shape != columns.shape or rows.shape != any_sunlight_fraction.shape:
        raise ValueError(
            "rows, columns and any_sunlight_fraction must share a shape; got "
            f"{rows.shape}, {columns.shape} and {any_sunlight_fraction.shape}"
        )
    if rows.ndim != 2:
        raise ValueError(
            "shadow_targets takes a two-dimensional sampling of the raster so "
            f"that regions are connected; got {rows.ndim} dimensions"
        )
    dark = any_sunlight_fraction <= 0.0
    if not bool(dark.any()):
        raise ValueError(
            "no fully shadowed cell on the illumination grid; there is nowhere "
            "for a cold-trap sortie to go"
        )

    cell = raster.cell_size_m
    distance_m = np.hypot((rows - start[0]) * cell, (columns - start[1]) * cell)
    drop = raster.values[start[0], start[1]] - raster.values[rows, columns]

    label = np.full(dark.shape, -1, dtype=int)
    count = 0
    for i in range(dark.shape[0]):
        for j in range(dark.shape[1]):
            if dark[i, j] and label[i, j] < 0:
                stack = [(i, j)]
                label[i, j] = count
                while stack:
                    a, b = stack.pop()
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        p, q = a + da, b + db
                        if (
                            0 <= p < dark.shape[0]
                            and 0 <= q < dark.shape[1]
                            and dark[p, q]
                            and label[p, q] < 0
                        ):
                            label[p, q] = count
                            stack.append((p, q))
                count += 1
    sizes = np.array([int((label == k).sum()) for k in range(count)])
    sample_area_km2 = (rows[1, 0] - rows[0, 0]) ** 2 * cell**2 / 1e6

    def make(identifier: str, index: tuple[int, int]) -> ShadowTarget:
        return ShadowTarget(
            id=identifier,
            row=int(rows[index]),
            column=int(columns[index]),
            distance_km=float(distance_m[index]) / 1000.0,
            drop_m=float(drop[index]),
            region_area_km2=float(sizes[int(label[index])]) * sample_area_km2,
        )

    nearest = np.unravel_index(
        int(np.argmin(np.where(dark, distance_m, np.inf))), dark.shape
    )
    deepest = np.unravel_index(
        int(np.argmax(np.where(dark, drop, -np.inf))), dark.shape
    )
    members = np.argwhere(label == int(np.argmax(sizes)))
    entry = members[int(np.argmin([distance_m[a, b] for a, b in members]))]
    return {
        "nearest": make("nearest", (int(nearest[0]), int(nearest[1]))),
        "largest": make("largest", (int(entry[0]), int(entry[1]))),
        "deepest": make("deepest", (int(deepest[0]), int(deepest[1]))),
    }


def best_charge_point(
    *,
    rows: NDArray[np.int_],
    columns: NDArray[np.int_],
    any_sunlight_fraction: NDArray[np.float64],
    elevation_m: NDArray[np.float64],
) -> tuple[int, int]:
    """The sampled cell that sees the most Sun, ties broken by elevation.

    Not the highest cell. Height is a proxy for sunlight that holds on a rim
    and fails on a crater floor, and a survey that picks the highest ground on
    a crater floor picks somewhere arbitrary. Elevation only breaks ties, which
    is where it is a genuine tiebreak: among equally lit ground, higher is
    further from whatever fills the basin.

    This is not a landing or trafficability assessment. It is a cell that sees
    the most Sun, not a place anyone has said a lander could sit.
    """
    if not (rows.shape == columns.shape == any_sunlight_fraction.shape == elevation_m.shape):
        raise ValueError(
            "rows, columns, illumination and elevation must share a shape; got "
            f"{rows.shape}, {columns.shape}, {any_sunlight_fraction.shape} and "
            f"{elevation_m.shape}"
        )
    spread = max(float(np.ptp(elevation_m)), 1.0)
    ranked = any_sunlight_fraction + 1e-9 * (elevation_m - elevation_m.min()) / spread
    best = np.unravel_index(int(np.argmax(ranked)), ranked.shape)
    return int(rows[best]), int(columns[best])
