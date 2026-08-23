# SPDX-License-Identifier: Apache-2.0
#
# Tests for eclipse.illumination.
#
# Mostly on synthetic terrain with known horizons, because the geometry is what
# needs pinning and a real DEM has no analytic answer to check against. Flat
# ground, a wall at a known distance and a pit of known depth each have a
# horizon that can be written down.
#
# The three polar subtleties get a test each, because each of them biases the
# answer the same way if dropped -- toward reporting more light than there is.

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from eclipse.illumination import (
    LUNAR_OBLIQUITY_DEG,
    SOLAR_ANGULAR_RADIUS_DEG,
    horizon_elevation_deg,
    illumination_fraction,
    solar_elevation_deg,
)
from eclipse.io.terrain import GeoRaster

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LUNAR_RADIUS_M = 1737400.0


def raster_from(values: np.ndarray, *, cell_size_m: float = 5.0) -> GeoRaster:
    return GeoRaster(
        values=values,
        origin_x_m=0.0,
        origin_y_m=0.0,
        cell_size_m=cell_size_m,
        reference_radius_m=LUNAR_RADIUS_M,
    )


# --- the horizon, against terrain whose answer is known


def test_flat_ground_has_a_horizon_that_curvature_pushes_below_level() -> None:
    # On a plane on a sphere the horizon is not zero: distant ground falls away
    # by d^2/2R, so every ray looks slightly downhill. Dropping curvature would
    # report a horizon of exactly zero and overstate shadowing.
    raster = raster_from(np.zeros((200, 200)), cell_size_m=20.0)
    # Searched from one cell, so the shallowest ray is the nearest one and its
    # angle is exactly the curvature drop over a single cell.
    horizon = horizon_elevation_deg(
        raster,
        rows=np.array([100]),
        columns=np.array([100]),
        azimuths=8,
        samples_along_ray=60,
        minimum_range_m=20.0,
    )
    assert np.all(horizon.elevation_deg < 0.0)
    nearest = math.degrees(
        math.atan2(-(20.0**2) / (2.0 * LUNAR_RADIUS_M), 20.0)
    )
    assert float(horizon.elevation_deg.max()) == pytest.approx(nearest, abs=1e-6)


def test_the_stand_off_keeps_the_adjacent_cell_out_of_the_horizon() -> None:
    # The default exists because a horizon searched from the next cell is not a
    # horizon. A two-metre step five metres away subtends more than twenty
    # degrees, and on a grid that is mostly interpolated that step is the
    # interpolator. Found by a real point coming out permanently shadowed for
    # exactly this reason.
    values = np.zeros((200, 200))
    values[100, 101] = 2.0
    raster = raster_from(values, cell_size_m=5.0)
    close = horizon_elevation_deg(
        raster, rows=np.array([100]), columns=np.array([100]),
        azimuths=4, samples_along_ray=60, minimum_range_m=5.0,
    )
    standing_off = horizon_elevation_deg(
        raster, rows=np.array([100]), columns=np.array([100]),
        azimuths=4, samples_along_ray=60, minimum_range_m=50.0,
    )
    assert float(close.elevation_deg.max()) > 20.0
    assert float(standing_off.elevation_deg.max()) < 0.0


def test_a_stand_off_beyond_the_search_limit_is_refused() -> None:
    raster = raster_from(np.zeros((60, 60)), cell_size_m=5.0)
    with pytest.raises(ValueError, match="nothing to look at"):
        horizon_elevation_deg(
            raster, rows=np.array([30]), columns=np.array([30]),
            azimuths=4, samples_along_ray=10,
            minimum_range_m=500.0, maximum_range_m=200.0,
        )


def test_a_wall_shows_up_at_the_angle_it_subtends() -> None:
    values = np.zeros((200, 200))
    values[:, 150:] = 100.0
    raster = raster_from(values, cell_size_m=20.0)
    horizon = horizon_elevation_deg(
        raster,
        rows=np.array([100]),
        columns=np.array([100]),
        azimuths=4,
        samples_along_ray=200,
    )
    # Azimuth 90 degrees is the +column direction, straight at the wall, whose
    # near face is 50 cells away.
    distance = 50 * 20.0
    expected = math.degrees(
        math.atan2(100.0 - distance**2 / (2.0 * LUNAR_RADIUS_M), distance)
    )
    toward_wall = int(np.argmin(np.abs(horizon.azimuth_deg - 90.0)))
    assert float(horizon.elevation_deg[0, toward_wall]) == pytest.approx(
        expected, rel=0.02
    )


def test_a_deeper_pit_has_a_higher_horizon() -> None:
    def horizon_of(depth: float) -> float:
        values = np.zeros((200, 200))
        values[95:105, 95:105] = -depth
        raster = raster_from(values, cell_size_m=20.0)
        return float(
            horizon_elevation_deg(
                raster,
                rows=np.array([100]),
                columns=np.array([100]),
                azimuths=8,
                samples_along_ray=60,
            ).elevation_deg.max()
        )

    assert horizon_of(200.0) > horizon_of(50.0) > horizon_of(0.0)


def test_a_truncated_search_is_reported_rather_than_hidden() -> None:
    raster = raster_from(np.zeros((60, 60)), cell_size_m=20.0)
    horizon = horizon_elevation_deg(
        raster,
        rows=np.array([30]),
        columns=np.array([30]),
        azimuths=8,
        samples_along_ray=40,
    )
    assert horizon.truncated_fraction > 0.0, (
        "rays leaving a small grid are treated as clear sky, which can only "
        "overstate illumination; the fraction lost has to be visible"
    )


# --- solar geometry


def test_the_sun_at_a_pole_sits_at_the_subsolar_latitude() -> None:
    # Exactly at the pole the hour angle does nothing: elevation is the
    # declination whatever the time of day.
    for declination in (-LUNAR_OBLIQUITY_DEG, 0.0, LUNAR_OBLIQUITY_DEG):
        elevation = solar_elevation_deg(
            latitude_deg=-90.0,
            subsolar_latitude_deg=np.array([declination]),
            hour_angle_deg=np.array([0.0, 90.0, 180.0, 270.0]),
        )
        assert np.allclose(elevation, -declination, atol=1e-9)


def test_solar_elevation_stays_within_a_few_degrees_near_the_pole() -> None:
    declination, hour = np.meshgrid(
        np.linspace(-LUNAR_OBLIQUITY_DEG, LUNAR_OBLIQUITY_DEG, 9),
        np.linspace(0.0, 360.0, 180),
        indexing="ij",
    )
    elevation = solar_elevation_deg(
        latitude_deg=-88.23,
        subsolar_latitude_deg=declination,
        hour_angle_deg=hour,
    )
    assert float(np.abs(elevation).max()) < LUNAR_OBLIQUITY_DEG + 2.0, (
        "the whole reason polar illumination is a terrain question is that the "
        "Sun never gets far from the horizon"
    )


# --- illumination, and the finite disc


def test_flat_ground_at_a_pole_is_lit_about_half_the_time() -> None:
    raster = raster_from(np.zeros((200, 200)), cell_size_m=20.0)
    horizon = horizon_elevation_deg(
        raster,
        rows=np.array([100]),
        columns=np.array([100]),
        azimuths=36,
        samples_along_ray=60,
    )
    illumination = illumination_fraction(
        horizon=horizon,
        latitude_deg=-89.0,
        north_azimuth_deg=np.array([0.0]),
    )
    assert 0.3 < float(illumination.lit_fraction[0]) < 0.7


def test_the_finite_disc_makes_a_penumbra(
) -> None:
    # A point Sun would report a hard boundary. Half a degree of disc against
    # elevations of a degree or two puts a real band between lit and dark.
    raster = raster_from(np.zeros((200, 200)), cell_size_m=20.0)
    horizon = horizon_elevation_deg(
        raster,
        rows=np.array([100]),
        columns=np.array([100]),
        azimuths=36,
        samples_along_ray=60,
    )
    illumination = illumination_fraction(
        horizon=horizon,
        latitude_deg=-89.5,
        north_azimuth_deg=np.array([0.0]),
    )
    assert float(illumination.penumbral_fraction[0]) > 0.0
    assert SOLAR_ANGULAR_RADIUS_DEG > 0.2


def test_the_three_fractions_partition_the_year() -> None:
    raster = raster_from(np.zeros((120, 120)), cell_size_m=20.0)
    horizon = horizon_elevation_deg(
        raster,
        rows=np.array([60, 30]),
        columns=np.array([60, 30]),
        azimuths=24,
        samples_along_ray=40,
    )
    illumination = illumination_fraction(
        horizon=horizon,
        latitude_deg=-88.5,
        north_azimuth_deg=np.array([0.0, 137.0]),
    )
    total = (
        illumination.lit_fraction
        + illumination.penumbral_fraction
        + illumination.dark_fraction
    )
    assert np.allclose(total, 1.0)


def test_a_deep_hole_is_dark_whatever_the_season() -> None:
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
    illumination = illumination_fraction(
        horizon=horizon,
        latitude_deg=-88.5,
        north_azimuth_deg=np.array([0.0]),
    )
    assert float(illumination.lit_fraction[0]) == 0.0
    assert float(illumination.dark_fraction[0]) == pytest.approx(1.0)


def test_darkness_survives_a_wider_horizon_search_but_light_may_not() -> None:
    # The asymmetry that makes the truncation caveat one-sided. Extra terrain
    # can only raise a horizon, so a dark point stays dark and a lit point can
    # lose light. Conclusions about shadow are robust; conclusions about
    # sunlight are upper bounds.
    values = np.zeros((240, 240))
    values[110:130, 110:130] = -3000.0
    raster = raster_from(values, cell_size_m=20.0)
    # Sample count scales with the log-range so that widening the search adds
    # far-field rays without thinning the near ones. A fixed count would spread
    # the same samples further apart and could miss a peak it previously found,
    # which is a property of the discretisation rather than of the horizon.
    def searched(limit: float) -> float:
        octaves = math.log2(limit / 100.0)
        return float(
            horizon_elevation_deg(
                raster,
                rows=np.array([120]),
                columns=np.array([120]),
                azimuths=24,
                samples_along_ray=int(round(30 * octaves)),
                minimum_range_m=100.0,
                maximum_range_m=limit,
            ).elevation_deg.max()
        )

    assert searched(2000.0) >= searched(400.0) - 1e-9


def test_a_point_is_lit_the_same_whoever_it_is_computed_beside() -> None:
    # A single latitude for a whole batch made each point's illumination depend
    # on its companions: over a twenty-kilometre polar window the batch mean
    # moves by two thirds of a degree, against an obliquity of 1.54. The runner
    # that found this read one crest as 87.7% lit beside a far corner and 90.4%
    # beside a near one, for no reason on the ground.
    raster = raster_from(np.zeros((300, 300)), cell_size_m=40.0)
    rows = np.array([150, 40])
    columns = np.array([150, 40])
    latitudes = np.array([-87.5, -89.5])
    north = np.array([0.0, 137.0])
    horizon = horizon_elevation_deg(
        raster, rows=rows, columns=columns, azimuths=24, samples_along_ray=40
    )
    together = illumination_fraction(
        horizon=horizon, latitude_deg=latitudes, north_azimuth_deg=north
    )
    averaged = illumination_fraction(
        horizon=horizon,
        latitude_deg=float(latitudes.mean()),
        north_azimuth_deg=north,
    )
    for index in range(2):
        alone = illumination_fraction(
            horizon=horizon_elevation_deg(
                raster,
                rows=rows[index : index + 1],
                columns=columns[index : index + 1],
                azimuths=24,
                samples_along_ray=40,
            ),
            latitude_deg=float(latitudes[index]),
            north_azimuth_deg=north[index : index + 1],
        )
        assert float(together.lit_fraction[index]) == pytest.approx(
            float(alone.lit_fraction[0])
        )
        # And the old behaviour would have failed that, so the check is not
        # passing on a quantity that happens to be insensitive to latitude.
        assert float(averaged.lit_fraction[index]) != pytest.approx(
            float(alone.lit_fraction[0])
        )


def test_a_batch_latitude_is_broadcast_when_it_is_one_value() -> None:
    raster = raster_from(np.zeros((200, 200)), cell_size_m=20.0)
    horizon = horizon_elevation_deg(
        raster,
        rows=np.array([100, 60]),
        columns=np.array([100, 60]),
        azimuths=24,
        samples_along_ray=40,
    )
    north = np.array([0.0, 137.0])
    scalar = illumination_fraction(
        horizon=horizon, latitude_deg=-89.0, north_azimuth_deg=north
    )
    array = illumination_fraction(
        horizon=horizon,
        latitude_deg=np.array([-89.0, -89.0]),
        north_azimuth_deg=north,
    )
    assert np.allclose(scalar.lit_fraction, array.lit_fraction)
    assert np.allclose(scalar.penumbral_fraction, array.penumbral_fraction)
