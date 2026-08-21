# SPDX-License-Identifier: Apache-2.0
#
# Tests for eclipse.terrain and eclipse.io.terrain.
#
# The terrain products are fetched rather than committed, so these tests skip
# when they are absent rather than failing. That is a deliberate asymmetry: a
# missing archive product is a setup state, while a product whose bytes do not
# match the manifest is a real failure and tools/fetch_terrain.py raises it.
#
# The strongest test here is a known-answer one. The producers publish their own
# slope raster alongside the elevation, so the slope algorithm can be identified
# rather than assumed -- and reproducing it to a ten-thousandth of a degree
# validates the TIFF reader, the georeferencing and the slope expression at
# once.

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from eclipse.io.terrain import (
    GeoRaster,
    TerrainFileError,
    latitude_of_radius,
    model_to_latitude_longitude,
    point_scale_factor,
    read_float_geotiff,
)
from eclipse.terrain import (
    NATURAL_TERRAIN_SLOPE_EXPONENT,
    aggregate,
    anisotropy,
    scale_trend,
    slope_degrees,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TERRAIN = REPOSITORY_ROOT / "data" / "terrain"
ELEVATION = TERRAIN / "SL2_final_adj_5mpp_surf.tif"
PUBLISHED_SLOPE = TERRAIN / "SL2_final_adj_5mpp_slp.tif"

LUNAR_RADIUS_M = 1737400.0

needs_product = pytest.mark.skipif(
    not ELEVATION.exists(),
    reason="terrain products are fetched, not committed; run tools/fetch_terrain.py",
)


@pytest.fixture(scope="module")
def elevation() -> GeoRaster:
    return read_float_geotiff(ELEVATION)


# --- the projection, which needs no data


def test_the_scale_factor_is_the_derivative_of_the_projection() -> None:
    # Checked against the projection rather than against a hand-rounded number.
    # The point scale factor is by definition the rate at which map radius grows
    # with true arc distance, so differentiating one has to reproduce the other.
    assert point_scale_factor(-90.0) == pytest.approx(1.0)
    for latitude in (-89.5, -88.5, -87.5):
        step = 1e-6
        radius = [
            2.0 * LUNAR_RADIUS_M * math.tan(math.radians(45.0 + (latitude + d) / 2.0))
            for d in (-step, step)
        ]
        arc = 2.0 * step * math.radians(1.0) * LUNAR_RADIUS_M
        assert (radius[1] - radius[0]) / arc == pytest.approx(
            point_scale_factor(latitude), rel=1e-6
        )
    assert point_scale_factor(-87.5) > point_scale_factor(-89.0) > 1.0


def test_the_projection_inverts_itself() -> None:
    for latitude, longitude in ((-88.5, -87.1), (-89.9, 12.0), (-87.5, 179.0)):
        radius = 2.0 * LUNAR_RADIUS_M * math.tan(math.radians(45.0 + latitude / 2.0))
        x = radius * math.sin(math.radians(longitude))
        y = radius * math.cos(math.radians(longitude))
        back_lat, back_lon = model_to_latitude_longitude(
            x, y, reference_radius_m=LUNAR_RADIUS_M
        )
        assert back_lat == pytest.approx(latitude, abs=1e-9)
        assert back_lon == pytest.approx(longitude, abs=1e-9)


def test_the_map_edge_of_the_pds_polar_product_is_where_its_label_says() -> None:
    # Known answer from a different product's own label: ldem_875s_5m_float
    # states MAXIMUM_LATITUDE = -87.5 with 30336 pixels at 5 m about an offset
    # of 15167.5. The projection has to put the edge there.
    half_width_m = (30336 - 15167.5) * 5.0
    assert latitude_of_radius(
        half_width_m, reference_radius_m=LUNAR_RADIUS_M
    ) == pytest.approx(-87.5, abs=1e-3)


# --- the reader


@needs_product
def test_the_grid_matches_what_the_manifest_records(elevation: GeoRaster) -> None:
    assert elevation.shape == (4000, 4000)
    assert elevation.cell_size_m == 5.0
    assert elevation.origin_x_m == -58500.0
    assert elevation.origin_y_m == 33000.0
    assert elevation.extent_m == (-58500.0, -38500.0, 13000.0, 33000.0)


@needs_product
def test_the_elevation_range_matches_the_products_own_metadata(
    elevation: GeoRaster,
) -> None:
    # The GeoTIFF carries actual_range in its GDAL metadata. Reproducing it
    # exactly means the strip offset, byte order and sample format are all right.
    assert float(np.nanmin(elevation.values)) == pytest.approx(-1860.06201171875)
    assert float(np.nanmax(elevation.values)) == pytest.approx(1765.37353515625)


@needs_product
def test_the_site_sits_where_the_site_config_says(elevation: GeoRaster) -> None:
    latitude, longitude = elevation.center_latitude_longitude()
    assert latitude == pytest.approx(-88.230, abs=0.001)
    assert longitude == pytest.approx(-64.63, abs=0.01)
    assert elevation.arc_distance_from_pole_m() / 1000.0 == pytest.approx(
        53.7, abs=0.1
    )


def test_a_file_that_is_not_a_tiff_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "not-a-tiff.tif"
    path.write_bytes(b"MM\x00\x2a" + b"\x00" * 64)
    with pytest.raises(TerrainFileError, match="little-endian"):
        read_float_geotiff(path)


# --- slope, identified rather than assumed


@needs_product
def test_central_difference_reproduces_the_producers_own_slope_raster(
    elevation: GeoRaster,
) -> None:
    # The identification. Central difference lands on their published raster;
    # Horn does not, by three orders of magnitude more. That pins the algorithm
    # and validates the reader in the same step.
    published = read_float_geotiff(PUBLISHED_SLOPE).values
    window = (slice(1000, 3000), slice(1000, 3000))

    for method, tolerance in (("central_difference", 1e-3), ("horn", None)):
        mine = slope_degrees(
            elevation.values, cell_size_m=elevation.cell_size_m, method=method
        )
        difference = np.abs(mine[window] - published[window])
        median = float(np.median(difference[np.isfinite(difference)]))
        if tolerance is not None:
            assert median < tolerance, (
                f"{method} should reproduce the published raster, got {median}"
            )
        else:
            assert median > 1e-2, (
                "Horn should NOT reproduce it; if it does, the identification "
                "of the producers' algorithm is not discriminating"
            )


def test_an_unknown_slope_method_is_refused() -> None:
    with pytest.raises(ValueError, match="no slope method"):
        slope_degrees(np.zeros((4, 4)), cell_size_m=5.0, method="zevenbergen_thorne")


def test_a_plane_has_the_slope_of_its_gradient() -> None:
    rows, columns = np.mgrid[0:50, 0:50].astype(np.float64)
    for method in ("central_difference", "horn"):
        # A plane rising one metre per five-metre cell eastward is 45 degrees.
        slope = slope_degrees(columns * 5.0, cell_size_m=5.0, method=method)
        assert float(slope[10:-10, 10:-10].mean()) == pytest.approx(45.0)


# --- baseline


def test_aggregation_preserves_the_mean_and_coarsens_the_grid() -> None:
    values = np.arange(64, dtype=np.float64).reshape(8, 8)
    coarse = aggregate(values, 2)
    assert coarse.shape == (4, 4)
    assert float(coarse.mean()) == pytest.approx(float(values.mean()))
    assert aggregate(values, 1) is values


@needs_product
def test_this_grid_holds_no_roughness_across_the_baselines_it_posts(
    elevation: GeoRaster,
) -> None:
    # The day's result. For a self-affine surface mean slope goes as baseline to
    # the power H-1, and natural terrain gives roughly -0.3 to -0.1. This grid
    # gives an exponent near zero over six octaves, so there is no scale
    # dependence in it to extrapolate toward a stride.
    trend = scale_trend(
        elevation.values,
        cell_size_m=elevation.cell_size_m,
        factors=(1, 2, 4, 8, 16, 32, 64),
    )
    assert trend.baseline_m[0] == 5.0 and trend.baseline_m[-1] == 320.0
    assert trend.exponent > NATURAL_TERRAIN_SLOPE_EXPONENT[1], (
        f"exponent {trend.exponent:.3f} is as steep as natural terrain, which "
        "would mean the grid does carry roughness and the day's finding is wrong"
    )
    assert not trend.holds_roughness
    assert trend.hurst_exponent > 0.95


# --- direction, which is what discriminates terrain from artifact


@needs_product
def test_the_anisotropy_is_aligned_with_the_crater_and_not_with_the_orbit(
    elevation: GeoRaster,
) -> None:
    # If the smoothness were gap-filling, the directional signature would follow
    # LRO's ground tracks, which run near-meridionally at this latitude and so
    # radially from the pole in this projection. It follows the crater's fall
    # line instead.
    centre_x, centre_y = elevation.center_model_m

    def raster_axis(dx: float, dy: float) -> float:
        return math.degrees(math.atan2(dx, -dy)) % 180.0

    # de Gerlache centre, IAU Gazetteer, projected the same way.
    crater_radius = 2.0 * LUNAR_RADIUS_M * math.tan(math.radians(45.0 - 88.5 / 2.0))
    crater_x = crater_radius * math.sin(math.radians(-87.1))
    crater_y = crater_radius * math.cos(math.radians(-87.1))

    fall_line = raster_axis(centre_x - crater_x, centre_y - crater_y)
    ground_track = raster_axis(centre_x, centre_y)

    measured = anisotropy(
        elevation.values, cell_size_m=elevation.cell_size_m, lag_cells=20
    )
    assert measured.ratio > 1.2, "there should be a direction to find at all"
    assert measured.separation_from(fall_line) < 5.0
    assert measured.separation_from(ground_track) > 30.0


@needs_product
def test_the_direction_is_the_same_at_every_lag(elevation: GeoRaster) -> None:
    axes = [
        anisotropy(
            elevation.values, cell_size_m=elevation.cell_size_m, lag_cells=lag
        ).roughest_axis_degrees
        for lag in (4, 8, 20, 40)
    ]
    assert len(set(axes)) == 1, (
        f"a direction that moved with lag would be a sampling artifact; got {axes}"
    )


def test_every_axis_at_the_shortest_lag_gives_a_real_offset() -> None:
    from eclipse.terrain import directional_rms_slope_degrees

    # A one-cell lag is the shortest there is, and no axis degenerates at it.
    rows, columns = np.mgrid[0:40, 0:40].astype(np.float64)
    for axis in range(0, 180, 10):
        value = directional_rms_slope_degrees(
            columns * 5.0, cell_size_m=5.0, lag_cells=1, axis_degrees=float(axis)
        )
        assert math.isfinite(value)
