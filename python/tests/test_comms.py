# SPDX-License-Identifier: Apache-2.0
#
# Tests for eclipse.comms.
#
# The Earth geometry is checked against closed forms that can be written down --
# the sub-Earth point sees Earth overhead, a point ninety degrees away sees it on
# the horizon, and the finite-distance correction lowers it rather than raising
# it. The viewshed is checked on terrain whose answer is obvious: a wall hides
# what is behind it and nothing else.

from __future__ import annotations

import numpy as np
import pytest

from eclipse.comms import (
    EARTH_ANGULAR_RADIUS_DEG,
    EARTH_DISTANCE_M,
    LIBRATION_LATITUDE_DEG,
    LIBRATION_LONGITUDE_DEG,
    LIBRATION_LONGITUDE_PERIOD_HOURS,
    earth_elevation_deg,
    earth_visibility,
    sub_earth_latitude_deg,
    sub_earth_longitude_deg,
    viewshed,
)
from eclipse.illumination import horizon_elevation_deg
from eclipse.io.terrain import GeoRaster

LUNAR_RADIUS_M = 1737400.0


def raster_from(values: np.ndarray, *, cell_size_m: float = 5.0) -> GeoRaster:
    return GeoRaster(
        values=values,
        origin_x_m=0.0,
        origin_y_m=0.0,
        cell_size_m=cell_size_m,
        reference_radius_m=LUNAR_RADIUS_M,
    )


# --- where Earth is


def test_earth_is_overhead_at_the_sub_earth_point() -> None:
    assert float(
        earth_elevation_deg(
            latitude_deg=0.0,
            longitude_deg=0.0,
            sub_earth_latitude_deg=0.0,
            sub_earth_longitude_deg=0.0,
            reference_radius_m=LUNAR_RADIUS_M,
        )
    ) == pytest.approx(90.0)


def test_earth_sits_below_the_horizontal_at_ninety_degrees_away() -> None:
    # Exactly on the limb of the near side, an observer standing on the surface
    # sees Earth below the local horizontal, not on it: the surface is a radius
    # further out than the centre the geometry is measured from. Getting this
    # backwards would overstate contact at every marginal site in the study.
    elevation = float(
        earth_elevation_deg(
            latitude_deg=0.0,
            longitude_deg=90.0,
            sub_earth_latitude_deg=0.0,
            sub_earth_longitude_deg=0.0,
            reference_radius_m=LUNAR_RADIUS_M,
        )
    )
    assert elevation < 0.0
    assert elevation == pytest.approx(
        -np.degrees(np.arctan2(LUNAR_RADIUS_M, EARTH_DISTANCE_M)), abs=1e-6
    )


def test_the_finite_distance_correction_only_ever_lowers_earth() -> None:
    separation = np.linspace(1.0, 179.0, 179)
    surface = earth_elevation_deg(
        latitude_deg=0.0,
        longitude_deg=separation,
        sub_earth_latitude_deg=0.0,
        sub_earth_longitude_deg=0.0,
        reference_radius_m=LUNAR_RADIUS_M,
    )
    centre = 90.0 - separation
    assert bool((surface <= centre + 1e-9).all())


def test_the_correction_is_the_size_of_the_answer_at_a_polar_site() -> None:
    # de Gerlache: Earth half a degree up, and the correction is a fifth of a
    # degree. A model that skipped it would be wrong by nearly half.
    common = dict(
        latitude_deg=-88.230,
        longitude_deg=-64.628,
        sub_earth_latitude_deg=0.0,
        sub_earth_longitude_deg=0.0,
    )
    surface = float(
        earth_elevation_deg(reference_radius_m=LUNAR_RADIUS_M, **common)
    )
    centre = float(earth_elevation_deg(reference_radius_m=0.0, **common))
    assert 0.0 < surface < 1.0
    assert centre - surface > 0.2


def test_libration_stays_inside_its_stated_amplitude() -> None:
    hours = np.linspace(0.0, 10.0 * LIBRATION_LONGITUDE_PERIOD_HOURS, 5000)
    assert float(np.abs(sub_earth_longitude_deg(hours)).max()) == pytest.approx(
        LIBRATION_LONGITUDE_DEG, rel=1e-3
    )
    assert float(np.abs(sub_earth_latitude_deg(hours)).max()) == pytest.approx(
        LIBRATION_LATITUDE_DEG, rel=1e-3
    )


def test_libration_decides_a_marginal_site() -> None:
    # Shackleton sits below the horizontal at mean libration and above it at
    # the extreme, which is why a single-epoch answer would be meaningless.
    hours = np.linspace(0.0, LIBRATION_LONGITUDE_PERIOD_HOURS, 400)
    elevation = earth_elevation_deg(
        latitude_deg=-89.767,
        longitude_deg=-171.870,
        sub_earth_latitude_deg=sub_earth_latitude_deg(hours),
        sub_earth_longitude_deg=sub_earth_longitude_deg(hours),
        reference_radius_m=LUNAR_RADIUS_M,
    )
    assert float(elevation.min()) < 0.0
    assert float(elevation.max()) > 0.0


# --- visibility against a horizon


def test_flat_ground_at_a_polar_site_sees_earth_for_part_of_a_cycle() -> None:
    raster = raster_from(np.zeros((200, 200)), cell_size_m=20.0)
    horizon = horizon_elevation_deg(
        raster,
        rows=np.array([100]),
        columns=np.array([100]),
        azimuths=36,
        samples_along_ray=60,
    )
    seen = earth_visibility(
        horizon=horizon,
        latitude_deg=np.array([-88.0]),
        longitude_deg=np.array([-20.0]),
        north_azimuth_deg=np.array([0.0]),
        reference_radius_m=LUNAR_RADIUS_M,
        samples=120,
    )
    assert 0.0 < float(seen.any_contact_fraction[0]) <= 1.0
    assert float(seen.blind_fraction[0]) == pytest.approx(
        1.0 - float(seen.any_contact_fraction[0])
    )


def test_a_deep_pit_never_sees_earth() -> None:
    values = np.zeros((240, 240))
    values[110:130, 110:130] = -3000.0
    raster = raster_from(values, cell_size_m=20.0)
    horizon = horizon_elevation_deg(
        raster,
        rows=np.array([120]),
        columns=np.array([120]),
        azimuths=36,
        samples_along_ray=80,
    )
    seen = earth_visibility(
        horizon=horizon,
        latitude_deg=np.array([-88.5]),
        longitude_deg=np.array([-10.0]),
        north_azimuth_deg=np.array([0.0]),
        reference_radius_m=LUNAR_RADIUS_M,
        samples=120,
    )
    assert float(seen.any_contact_fraction[0]) == 0.0


def test_visibility_refuses_a_mismatched_point_set() -> None:
    raster = raster_from(np.zeros((60, 60)), cell_size_m=20.0)
    horizon = horizon_elevation_deg(
        raster,
        rows=np.array([30, 20]),
        columns=np.array([30, 20]),
        azimuths=24,
        samples_along_ray=40,
    )
    with pytest.raises(ValueError, match="one value per horizon point"):
        earth_visibility(
            horizon=horizon,
            latitude_deg=np.array([-88.0]),
            longitude_deg=np.array([-20.0]),
            north_azimuth_deg=np.array([0.0]),
            reference_radius_m=LUNAR_RADIUS_M,
        )


def test_the_disc_makes_partial_contact_a_real_state() -> None:
    assert EARTH_ANGULAR_RADIUS_DEG > 4.0 * 0.2  # wider than the Sun's disc


# --- the viewshed


def test_a_wall_hides_what_is_behind_it_and_nothing_else() -> None:
    values = np.zeros((81, 81))
    values[:, 38:43] = 400.0
    raster = raster_from(values, cell_size_m=20.0)
    seen = viewshed(raster, origin=(40, 5), mast_height_m=2.0, minimum_range_m=0.0)
    assert bool(seen[40, 20])
    assert bool(seen[40, 38])
    assert not bool(seen[40, 60])
    assert not bool(seen[10, 70])


def test_a_one_cell_wall_blocks_as_completely_as_a_thick_one() -> None:
    # The reason a cell occludes its whole angular sector rather than the bin
    # its centre falls in. Charging only the centre bin, a one-cell wall thirty
    # cells out leaks: consecutive cells along it differ in azimuth by more than
    # a degree while a bin is a quarter of one, so the gaps between them stay
    # open sky and a solid ridge blocks nothing. It fails silently and in the
    # direction that overstates coverage.
    behind = np.zeros((81, 81), dtype=bool)
    behind[:, 43:] = True
    for thickness in (slice(40, 41), slice(38, 43)):
        values = np.zeros((81, 81))
        values[:, thickness] = 400.0
        seen = viewshed(
            raster_from(values, cell_size_m=20.0),
            origin=(40, 5),
            mast_height_m=2.0,
            minimum_range_m=0.0,
        )
        assert int((seen & behind).sum()) == 0


def test_the_stand_off_puts_the_near_field_back_in_sight() -> None:
    # A ridge right beside the mast blocks everything past it when nothing is
    # excused, and the near field is exactly where an interpolated grid is
    # least trustworthy. The stand-off is the Day 9 correction applied here.
    values = np.zeros((81, 81))
    values[:, 6] = 30.0
    raster = raster_from(values, cell_size_m=5.0)
    blocked = viewshed(raster, origin=(40, 5), mast_height_m=2.0, minimum_range_m=0.0)
    excused = viewshed(raster, origin=(40, 5), mast_height_m=2.0, minimum_range_m=50.0)
    assert excused.sum() > blocked.sum()


def test_an_origin_outside_the_grid_is_refused() -> None:
    with pytest.raises(ValueError, match="lies outside a 5 by 5 grid"):
        viewshed(raster_from(np.zeros((5, 5))), origin=(9, 0))


def test_too_few_azimuth_bins_are_refused() -> None:
    with pytest.raises(ValueError, match="at least eight"):
        viewshed(raster_from(np.zeros((20, 20))), origin=(10, 10), azimuth_bins=4)


def test_a_taller_mast_never_sees_less() -> None:
    rng = np.random.default_rng(11)
    rough = rng.normal(0.0, 8.0, (161, 161)).cumsum(axis=0).cumsum(axis=1) * 0.002
    raster = raster_from(rough, cell_size_m=20.0)
    low = viewshed(raster, origin=(80, 80), mast_height_m=0.0)
    high = viewshed(raster, origin=(80, 80), mast_height_m=25.0)
    assert bool((low <= high).all())
    assert int(high.sum()) > int(low.sum())


def test_a_dome_hides_its_own_base() -> None:
    # The intuition this replaces was that a summit sees further than the ground
    # beside it, and on a smooth convex hill that is false: the near flank sits
    # at a shallower depression angle than the distant floor, so it occludes it.
    # Worth keeping as a test because the porous version of this algorithm
    # leaked rays through the flank and returned the intuitive answer instead.
    rows, columns = np.meshgrid(np.arange(121), np.arange(121), indexing="ij")
    radius = np.hypot(rows - 60.0, columns - 60.0)
    values = 300.0 * np.exp(-(radius**2) / 400.0)
    raster = raster_from(values, cell_size_m=20.0)
    summit = viewshed(raster, origin=(60, 60), mast_height_m=2.0)
    beside = viewshed(raster, origin=(60, 100), mast_height_m=2.0)
    assert int(summit.sum()) < int(beside.sum())


def test_finer_binning_never_sees_less() -> None:
    # Coarse bins merge cells that are angularly apart and let the highest of
    # them occlude the rest, so under-resolved binning removes ground nothing
    # stands in front of and the shortfall reads as terrain. The bias has a
    # direction, and this pins it: refining the binning can only add ground.
    rng = np.random.default_rng(7)
    rough = rng.normal(0.0, 6.0, (401, 401)).cumsum(axis=0).cumsum(axis=1) * 0.0006
    raster = raster_from(rough, cell_size_m=5.0)
    seen = [
        int(
            viewshed(
                raster, origin=(200, 200), mast_height_m=2.0, azimuth_bins=bins
            ).sum()
        )
        for bins in (900, 1800, 3600, 7200)
    ]
    assert seen == sorted(seen)
    assert seen[-1] > seen[0]


def test_the_bin_count_is_derived_from_the_grid() -> None:
    # One bin subtending about a cell at the furthest corner. Left fixed it is a
    # free parameter that sets the answer: at 1440 bins on the Day 15 window the
    # relay area came out a fifth of its converged value.
    raster = raster_from(np.zeros((401, 401)), cell_size_m=5.0)
    corner = float(np.hypot(200.0, 200.0))
    derived = viewshed(raster, origin=(200, 200))
    explicit = viewshed(
        raster, origin=(200, 200), azimuth_bins=int(np.ceil(2.0 * np.pi * corner))
    )
    assert bool((derived == explicit).all())
