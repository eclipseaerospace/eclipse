# SPDX-License-Identifier: Apache-2.0
#
# eclipse.comms — whether the platform can be seen, from Earth or from a relay.
#
# The last of the six site axes declared on Day 6, and the one that decides
# whether a mission is teleoperated or autonomous. That distinction has been
# claimed since this project's first day and never tested against terrain.
#
# The physics is the same problem as illumination with a different target and a
# much slower clock. Earth sits near the lunar horizon at these latitudes --
# half a degree up at de Gerlache, nearly four at Malapert, and below the
# horizon at Shackleton -- and libration swings it by seven degrees either way
# over a month. So a point sees Earth if Earth's disc clears the local skyline,
# which is the horizon machinery already built for the Sun.
#
# Three things about the geometry that a naive version gets wrong, each biasing
# toward more contact than there is.
#
# Earth is not a point. It subtends about 1.9 degrees from the Moon, four times
# the Sun's disc, so at these grazing elevations the difference between the
# whole disc clearing a ridge and none of it clearing is a large band of ground.
# Partial visibility is a real state and it is reported separately.
#
# Earth is not at infinity. Seen from the surface rather than from the Moon's
# centre, a body at 384,400 km sits lower by up to a quarter of a degree at
# these angular distances -- comparable to the elevation itself. Treating the
# sub-Earth direction as the local zenith direction would overstate contact at
# exactly the sites where the answer is marginal.
#
# And libration is the whole story rather than a correction. At a site where the
# mean elevation is half a degree, a seven degree swing is the difference
# between permanent contact and none, so a fraction over a libration cycle is
# the only honest answer and a single-epoch calculation is meaningless.
#
# The relay question is a different geometry and gets its own function. Earth
# visibility asks whether a ray to the sky clears the skyline; relay visibility
# asks whether two points on the surface can see each other, which is a viewshed
# and is computed as one.
#
# References
#   Meeus J (1998) Astronomical Algorithms, 2nd ed. Chapter 53, libration.
#   Mazarico E et al. (2011) Illumination conditions of the lunar polar regions
#     using LOLA topography. Icarus 211, 1066-1081.

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import maximum_filter1d

from eclipse.illumination import HorizonMap
from eclipse.io.terrain import GeoRaster

__all__ = [
    "EARTH_ANGULAR_RADIUS_DEG",
    "EARTH_DISTANCE_M",
    "LIBRATION_LATITUDE_DEG",
    "LIBRATION_LATITUDE_PERIOD_HOURS",
    "LIBRATION_LONGITUDE_DEG",
    "LIBRATION_LONGITUDE_PERIOD_HOURS",
    "EarthVisibility",
    "earth_elevation_deg",
    "earth_visibility",
    "sub_earth_latitude_deg",
    "sub_earth_longitude_deg",
    "viewshed",
]

# Earth's radius over its distance, as an angle. Four times the Sun's disc, and
# at grazing elevations that is a wide band of partly-visible ground.
EARTH_ANGULAR_RADIUS_DEG: Final = 0.95
EARTH_DISTANCE_M: Final = 384400.0e3

# Optical libration, which is what moves Earth in the lunar sky. Longitude
# libration comes from the Moon's orbital eccentricity against its uniform
# rotation and runs on the anomalistic month; latitude libration comes from the
# tilt of the spin axis to the orbit and runs on the draconic month. The two
# periods differ, so the sub-Earth point traces a slowly precessing figure
# rather than closing on itself, and a fraction taken over one month is not
# quite the fraction taken over a year.
LIBRATION_LONGITUDE_DEG: Final = 7.9
LIBRATION_LATITUDE_DEG: Final = 6.7
LIBRATION_LONGITUDE_PERIOD_HOURS: Final = 27.554550 * 24.0
LIBRATION_LATITUDE_PERIOD_HOURS: Final = 27.212221 * 24.0


def sub_earth_longitude_deg(hours: NDArray[np.float64]) -> NDArray[np.float64]:
    """Selenographic longitude of the sub-Earth point, from optical libration."""
    return np.asarray(
        LIBRATION_LONGITUDE_DEG
        * np.sin(2.0 * np.pi * hours / LIBRATION_LONGITUDE_PERIOD_HOURS)
    )


def sub_earth_latitude_deg(hours: NDArray[np.float64]) -> NDArray[np.float64]:
    """Selenographic latitude of the sub-Earth point, from optical libration."""
    return np.asarray(
        LIBRATION_LATITUDE_DEG
        * np.sin(2.0 * np.pi * hours / LIBRATION_LATITUDE_PERIOD_HOURS)
    )


def earth_elevation_deg(
    *,
    latitude_deg: NDArray[np.float64] | float,
    longitude_deg: NDArray[np.float64] | float,
    sub_earth_latitude_deg: NDArray[np.float64] | float,
    sub_earth_longitude_deg: NDArray[np.float64] | float,
    reference_radius_m: float,
) -> NDArray[np.float64]:
    """Elevation of Earth's centre above the local horizontal plane.

    Corrected for the observer standing on the surface rather than at the
    Moon's centre, which lowers Earth by up to a quarter of a degree at these
    angular distances. That correction is the same size as the elevation it is
    correcting at the marginal sites, so it is not optional.
    """
    site_latitude = np.radians(latitude_deg)
    site_longitude = np.radians(longitude_deg)
    earth_latitude = np.radians(sub_earth_latitude_deg)
    earth_longitude = np.radians(sub_earth_longitude_deg)
    cosine = np.sin(site_latitude) * np.sin(earth_latitude) + np.cos(
        site_latitude
    ) * np.cos(earth_latitude) * np.cos(site_longitude - earth_longitude)
    separation = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.asarray(
        np.degrees(
            np.arctan2(
                EARTH_DISTANCE_M * np.cos(separation) - reference_radius_m,
                EARTH_DISTANCE_M * np.sin(separation),
            )
        )
    )


def _earth_azimuth_deg(
    *,
    latitude_deg: float,
    longitude_deg: float,
    sub_earth_latitude_deg: NDArray[np.float64],
    sub_earth_longitude_deg: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Bearing east of north toward the sub-Earth point."""
    site_latitude = np.radians(latitude_deg)
    earth_latitude = np.radians(sub_earth_latitude_deg)
    difference = np.radians(sub_earth_longitude_deg - longitude_deg)
    return np.asarray(
        np.degrees(
            np.arctan2(
                np.sin(difference) * np.cos(earth_latitude),
                np.cos(site_latitude) * np.sin(earth_latitude)
                - np.sin(site_latitude) * np.cos(earth_latitude) * np.cos(difference),
            )
        )
        % 360.0
    )


@dataclass(frozen=True, slots=True)
class EarthVisibility:
    """How much of a libration cycle each point can see Earth for."""

    full_fraction: NDArray[np.float64]
    partial_fraction: NDArray[np.float64]
    horizon: HorizonMap

    @property
    def any_contact_fraction(self) -> NDArray[np.float64]:
        return np.asarray(self.full_fraction + self.partial_fraction)

    @property
    def blind_fraction(self) -> NDArray[np.float64]:
        return np.asarray(1.0 - self.any_contact_fraction)


def earth_visibility(
    *,
    horizon: HorizonMap,
    latitude_deg: NDArray[np.float64],
    longitude_deg: NDArray[np.float64],
    north_azimuth_deg: NDArray[np.float64],
    reference_radius_m: float,
    samples: int = 240,
) -> EarthVisibility:
    """Fraction of a libration cycle for which Earth clears the local skyline.

    Swept over both libration periods rather than evaluated at an epoch,
    because at a site whose mean Earth elevation is half a degree a seven
    degree swing decides everything. The sweep runs over the longer of the two
    periods; the two do not close together, so this is a cycle rather than the
    cycle, and a longer sweep would move the marginal sites slightly.
    """
    points = horizon.elevation_deg.shape[0]
    if not (
        latitude_deg.size == longitude_deg.size == north_azimuth_deg.size == points
    ):
        raise ValueError(
            "latitude, longitude and north azimuth must give one value per "
            f"horizon point; got {latitude_deg.size}, {longitude_deg.size} and "
            f"{north_azimuth_deg.size} against {points} points"
        )
    hours = np.linspace(
        0.0, LIBRATION_LONGITUDE_PERIOD_HOURS, samples, endpoint=False
    )
    earth_latitude = sub_earth_latitude_deg(hours)
    earth_longitude = sub_earth_longitude_deg(hours)
    wrapped_azimuth = np.concatenate([horizon.azimuth_deg, [360.0]])

    full = np.zeros(points, dtype=np.float64)
    partial = np.zeros(points, dtype=np.float64)
    for index in range(points):
        elevation = earth_elevation_deg(
            latitude_deg=float(latitude_deg[index]),
            longitude_deg=float(longitude_deg[index]),
            sub_earth_latitude_deg=earth_latitude,
            sub_earth_longitude_deg=earth_longitude,
            reference_radius_m=reference_radius_m,
        )
        azimuth = _earth_azimuth_deg(
            latitude_deg=float(latitude_deg[index]),
            longitude_deg=float(longitude_deg[index]),
            sub_earth_latitude_deg=earth_latitude,
            sub_earth_longitude_deg=earth_longitude,
        )
        blocked = np.interp(
            np.mod(azimuth + north_azimuth_deg[index], 360.0),
            wrapped_azimuth,
            np.concatenate(
                [horizon.elevation_deg[index], horizon.elevation_deg[index, :1]]
            ),
        )
        clearance = elevation - blocked
        full[index] = float(np.mean(clearance >= EARTH_ANGULAR_RADIUS_DEG))
        partial[index] = float(
            np.mean(
                (clearance > -EARTH_ANGULAR_RADIUS_DEG)
                & (clearance < EARTH_ANGULAR_RADIUS_DEG)
            )
        )
    return EarthVisibility(
        full_fraction=full, partial_fraction=partial, horizon=horizon
    )


def viewshed(
    raster: GeoRaster,
    *,
    origin: tuple[int, int],
    mast_height_m: float = 0.0,
    minimum_range_m: float = 50.0,
    azimuth_bins: int | None = None,
) -> NDArray[np.bool_]:
    """Which cells a mast at the origin can see, over the whole grid.

    Marched outward one range shell at a time, keeping a running skyline per
    azimuth bin. Every cell is visited exactly once, unlike ray sampling, which
    at a quarter-degree spacing misses most cells beyond a few hundred metres
    and reports the misses as shadow.

    A cell occludes the whole angular sector it subtends, not the single bin its
    centre falls in, and that distinction is the difference between a working
    viewshed and a decorative one. A cell one grid step wide subtends 57 degrees
    over its range in cells, so within a couple of hundred cells of the origin
    consecutive cells along a ridge land in bins several apart. Charging only
    the centre bin leaves the gaps between them open, and a solid ridge close in
    then fails to block anything -- silently, and in the direction that
    overstates coverage. The sector is therefore dilated to the cell's own
    angular width before it is folded into the skyline.

    Lunar curvature drops distant ground by d squared over twice the radius,
    which at ten kilometres is thirty metres and is the difference between a
    relay seeing a crater floor and not.

    The stand-off is the same correction Day 9 needed for the solar horizon and
    for the same reason. These products are about nine tenths interpolated at
    5 m, so the nearest cells carry the interpolator's own roughness rather than
    the ground's, and letting them occlude produces a viewshed that is mostly
    noise -- a relay on a rim reporting that it can see almost nothing. Cells
    inside the stand-off are visible and do not occlude.
    """
    height, width = raster.values.shape
    if not (0 <= origin[0] < height and 0 <= origin[1] < width):
        raise ValueError(f"origin {origin} lies outside a {height} by {width} grid")
    if azimuth_bins is None:
        corner_radius_cells = max(
            float(np.hypot(row - origin[0], column - origin[1]))
            for row in (0, height - 1)
            for column in (0, width - 1)
        )
        azimuth_bins = max(8, int(np.ceil(2.0 * np.pi * corner_radius_cells)))
    if azimuth_bins < 8:
        raise ValueError(
            f"azimuth_bins must be at least eight to resolve a skyline; got "
            f"{azimuth_bins}"
        )

    rows, columns = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    north = (rows - origin[0]) * raster.cell_size_m
    east = (columns - origin[1]) * raster.cell_size_m
    distance = np.hypot(north, east)
    drop = distance**2 / (2.0 * raster.reference_radius_m)
    rise = raster.values - (raster.values[origin] + mast_height_m) - drop
    with np.errstate(divide="ignore", invalid="ignore"):
        angle = np.degrees(np.arctan2(rise, distance))
    angle[origin] = -np.inf

    near = distance < minimum_range_m
    bin_width_deg = 360.0 / azimuth_bins
    bins = np.mod(
        np.floor(np.degrees(np.arctan2(east, north)) / bin_width_deg).astype(np.int64),
        azimuth_bins,
    )
    shells = np.floor(distance / raster.cell_size_m).astype(np.int64)

    visible = np.zeros(raster.values.shape, dtype=bool)
    visible[origin] = True
    visible[near] = True

    flat_bins = bins.ravel()
    flat_angle = angle.ravel()
    flat_shells = shells.ravel()
    flat_visible = visible.ravel()
    flat_near = near.ravel()

    order = np.lexsort((flat_bins, flat_shells))
    sorted_shells = flat_shells[order]
    shell_edges = np.concatenate(
        [[0], np.flatnonzero(np.diff(sorted_shells)) + 1, [sorted_shells.size]]
    )
    skyline = np.full(azimuth_bins, -np.inf)
    cell_half_diagonal_m = 0.5 * np.sqrt(2.0) * raster.cell_size_m
    for start, stop in zip(shell_edges[:-1], shell_edges[1:]):
        segment = order[start:stop]
        segment_bins = flat_bins[segment]
        segment_angle = flat_angle[segment]
        flat_visible[segment] |= segment_angle >= skyline[segment_bins]

        occluding = np.where(flat_near[segment], -np.inf, segment_angle)
        group_starts = np.concatenate([[0], np.flatnonzero(np.diff(segment_bins)) + 1])
        shell_skyline = np.full(azimuth_bins, -np.inf)
        shell_skyline[segment_bins[group_starts]] = np.maximum.reduceat(
            occluding, group_starts
        )
        shell_radius_m = max(float(sorted_shells[start]), 1.0) * raster.cell_size_m
        spread = int(
            np.rint(
                np.degrees(np.arctan2(cell_half_diagonal_m, shell_radius_m))
                / bin_width_deg
            )
        )
        if spread > 0:
            shell_skyline = maximum_filter1d(
                shell_skyline, size=min(2 * spread + 1, azimuth_bins), mode="wrap"
            )
        np.maximum(skyline, shell_skyline, out=skyline)
    return visible
