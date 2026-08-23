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
    "SOLAR_ANGULAR_RADIUS_DEG",
    "HorizonMap",
    "Illumination",
    "horizon_elevation_deg",
    "illumination_fraction",
    "solar_elevation_deg",
]

# The Sun's angular radius at one astronomical unit. Half a degree of disc is
# the whole difference between a hard shadow and a penumbra at a pole.
SOLAR_ANGULAR_RADIUS_DEG: Final = 0.265

# The Moon's obliquity to the ecliptic. The sub-solar latitude stays inside
# this, which is why polar illumination is a question about terrain rather than
# about season.
LUNAR_OBLIQUITY_DEG: Final = 1.54


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
