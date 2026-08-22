# SPDX-License-Identifier: Apache-2.0
#
# Tests for eclipse.sortie.
#
# Mostly on synthetic ground, because the properties worth pinning are about
# direction and bookkeeping rather than about any particular hill. A constant
# gradient is enough to check that gravity cancels over a round trip, that the
# dissipative terms do not, and that a descent steep enough to be free is not
# treated as a source.
#
# The reachable-index test is a regression. The first version of that function
# indexed the return leg from its head rather than its tail, which made the far
# end of a route look cheaper to reach than the near end -- a step function in
# what should be a monotone budget, and exactly backwards.

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eclipse.io.platform import load_platform
from eclipse.io.soil import janosi_hanamoto_model, load_soil, mohr_coulomb_model
from eclipse.io.terrain import GeoRaster
from eclipse.platform import Platform
from eclipse.sortie import (
    JOULES_PER_WATT_HOUR,
    Transect,
    sample_transect,
    walk_leg,
    walk_round_trip,
)
from eclipse.terramechanics import ContactModel, JanosiHanamotoModel, MohrCoulombModel

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LUNAR_GRAVITY = 1.62
FEET_IN_STANCE = 3
FLAT_SLIP = 0.0468


@pytest.fixture(scope="module")
def platform() -> Platform:
    return load_platform(
        REPOSITORY_ROOT / "configs" / "platforms" / "nominal-quadruped.toml"
    ).platform


@pytest.fixture(scope="module")
def ground() -> tuple[ContactModel, MohrCoulombModel, JanosiHanamotoModel]:
    dataset = load_soil(
        REPOSITORY_ROOT / "data" / "soils" / "lunar-intercrater.toml"
    ).datasets["carrier1991"]
    return (
        dataset.models["bekker"].extrapolating,
        mohr_coulomb_model(dataset, depth_range_cm="0-15"),
        janosi_hanamoto_model(dataset),
    )


def constant_grade(*, drop_per_metre: float, length_m: float = 1000.0) -> Transect:
    distance = np.linspace(0.0, length_m, 201)
    return Transect(distance_m=distance, elevation_m=-drop_per_metre * distance)


def _walk(transect: Transect, platform: Platform, ground: tuple[Any, ...]):  # type: ignore[no-untyped-def]
    contact, strength, mobilization = ground
    return walk_leg(
        transect=transect,
        platform=platform,
        contact_model=contact,
        strength=strength,
        mobilization=mobilization,
        gravity_m_per_s2=LUNAR_GRAVITY,
        feet_in_stance=FEET_IN_STANCE,
        level_ground_slip_ratio=FLAT_SLIP,
    )


# --- the transect itself


def test_slope_is_signed_and_climbing_is_positive() -> None:
    descending = constant_grade(drop_per_metre=0.2)
    assert np.all(descending.slope_degrees < 0.0)
    assert np.all(descending.reversed().slope_degrees > 0.0)
    assert float(descending.slope_degrees[0]) == pytest.approx(
        -math.degrees(math.atan(0.2))
    )


def test_segment_length_is_along_the_ground_not_the_map() -> None:
    transect = constant_grade(drop_per_metre=1.0)
    run = float(np.diff(transect.distance_m)[0])
    assert float(transect.segment_length_m[0]) == pytest.approx(run * math.sqrt(2.0))


def test_reversing_preserves_the_profile_and_flips_the_descent() -> None:
    transect = constant_grade(drop_per_metre=0.1)
    back = transect.reversed()
    assert np.allclose(back.elevation_m, transect.elevation_m[::-1])
    assert float(back.distance_m[-1]) == pytest.approx(float(transect.distance_m[-1]))
    assert back.descent_m == pytest.approx(-transect.descent_m)


def test_a_transect_needs_two_points() -> None:
    raster = GeoRaster(
        values=np.zeros((10, 10)),
        origin_x_m=0.0,
        origin_y_m=0.0,
        cell_size_m=5.0,
        reference_radius_m=1737400.0,
    )
    with pytest.raises(ValueError, match="at least two samples"):
        sample_transect(
            raster, start_row_column=(0, 0), end_row_column=(9, 9), samples=1
        )


def test_a_transect_may_not_leave_the_raster() -> None:
    raster = GeoRaster(
        values=np.zeros((10, 10)),
        origin_x_m=0.0,
        origin_y_m=0.0,
        cell_size_m=5.0,
        reference_radius_m=1737400.0,
    )
    with pytest.raises(ValueError, match="leaves the raster"):
        sample_transect(
            raster, start_row_column=(0, 0), end_row_column=(10, 10), samples=20
        )


# --- the asymmetry, which is the point


def test_gravity_cancels_over_a_round_trip_and_the_rest_does_not(
    platform: Platform, ground: tuple[ContactModel, MohrCoulombModel, JanosiHanamotoModel]
) -> None:
    transect = constant_grade(drop_per_metre=0.05)
    trip = walk_round_trip(
        transect=transect,
        platform=platform,
        contact_model=ground[0],
        strength=ground[1],
        mobilization=ground[2],
        gravity_m_per_s2=LUNAR_GRAVITY,
        feet_in_stance=FEET_IN_STANCE,
        level_ground_slip_ratio=FLAT_SLIP,
    )
    gravitational = float(
        trip.outbound.gravitational_J.sum() + trip.inbound.gravitational_J.sum()
    )
    assert gravitational == pytest.approx(0.0, abs=1e-6)

    dissipative = float(
        trip.outbound.dissipative_J.sum() + trip.inbound.dissipative_J.sum()
    )
    assert dissipative > 0.0
    assert float(trip.outbound.dissipative_J.sum()) == pytest.approx(
        float(trip.inbound.dissipative_J.sum()), rel=1e-9
    ), "the dissipative terms are indifferent to direction, which is why they double"


def test_climbing_costs_more_than_descending(
    platform: Platform, ground: tuple[ContactModel, MohrCoulombModel, JanosiHanamotoModel]
) -> None:
    transect = constant_grade(drop_per_metre=0.1)
    trip = walk_round_trip(
        transect=transect,
        platform=platform,
        contact_model=ground[0],
        strength=ground[1],
        mobilization=ground[2],
        gravity_m_per_s2=LUNAR_GRAVITY,
        feet_in_stance=FEET_IN_STANCE,
        level_ground_slip_ratio=FLAT_SLIP,
    )
    assert trip.asymmetry > 1.0
    assert trip.over_twice_outbound > 1.0, (
        "a round trip costs more than twice the cheaper direction whenever any "
        "of the descent is free"
    )


def test_a_free_descent_is_not_a_source(
    platform: Platform, ground: tuple[ContactModel, MohrCoulombModel, JanosiHanamotoModel]
) -> None:
    # Steep enough that gravity more than pays for the ground. Without the clamp
    # the segments would come out negative and the route would generate energy.
    leg = _walk(constant_grade(drop_per_metre=0.5), platform, ground)
    assert float(leg.gravitational_J.sum()) < 0.0
    assert np.all(leg.segment_J >= 0.0)
    assert leg.free_fraction > 0.0
    assert leg.total_J >= 0.0


def test_level_ground_has_no_free_segments_and_no_gravity_term(
    platform: Platform, ground: tuple[ContactModel, MohrCoulombModel, JanosiHanamotoModel]
) -> None:
    leg = _walk(constant_grade(drop_per_metre=0.0), platform, ground)
    assert float(np.abs(leg.gravitational_J).max()) == pytest.approx(0.0, abs=1e-9)
    assert leg.free_fraction == 0.0
    assert leg.total_J > 0.0, "level ground still costs, because the legs still swing"


# --- the budget


def test_the_reachable_point_is_monotone_in_battery(
    platform: Platform, ground: tuple[ContactModel, MohrCoulombModel, JanosiHanamotoModel]
) -> None:
    transect = constant_grade(drop_per_metre=0.1, length_m=4000.0)
    trip = walk_round_trip(
        transect=transect,
        platform=platform,
        contact_model=ground[0],
        strength=ground[1],
        mobilization=ground[2],
        gravity_m_per_s2=LUNAR_GRAVITY,
        feet_in_stance=FEET_IN_STANCE,
        level_ground_slip_ratio=FLAT_SLIP,
    )
    reached = [
        trip.reachable_index(battery_J=wh * JOULES_PER_WATT_HOUR, derating=4.0)
        for wh in (5.0, 10.0, 20.0, 40.0, 80.0, 400.0)
    ]
    assert reached == sorted(reached)
    assert reached[0] < reached[-1], "the sweep should actually bind somewhere"


def test_a_budget_that_cannot_leave_reaches_nothing(
    platform: Platform, ground: tuple[ContactModel, MohrCoulombModel, JanosiHanamotoModel]
) -> None:
    transect = constant_grade(drop_per_metre=0.1, length_m=4000.0)
    trip = walk_round_trip(
        transect=transect,
        platform=platform,
        contact_model=ground[0],
        strength=ground[1],
        mobilization=ground[2],
        gravity_m_per_s2=LUNAR_GRAVITY,
        feet_in_stance=FEET_IN_STANCE,
        level_ground_slip_ratio=FLAT_SLIP,
    )
    assert trip.reachable_index(battery_J=0.0, derating=4.0) == 0


def test_the_far_end_is_never_cheaper_than_a_nearer_one(
    platform: Platform, ground: tuple[ContactModel, MohrCoulombModel, JanosiHanamotoModel]
) -> None:
    # The regression. Indexing the return leg from its head rather than its tail
    # made turning back at the far end look cheap, because the tail of the
    # return was read as its head. Required energy has to rise along the route.
    transect = constant_grade(drop_per_metre=0.1, length_m=4000.0)
    trip = walk_round_trip(
        transect=transect,
        platform=platform,
        contact_model=ground[0],
        strength=ground[1],
        mobilization=ground[2],
        gravity_m_per_s2=LUNAR_GRAVITY,
        feet_in_stance=FEET_IN_STANCE,
        level_ground_slip_ratio=FLAT_SLIP,
    )
    out = np.concatenate([[0.0], trip.outbound.cumulative_J])
    remaining = trip.inbound.total_J - trip.inbound.cumulative_J[::-1]
    back = np.concatenate([remaining, [trip.inbound.total_J]])
    required = out + back

    assert float(required[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(required[-1]) == pytest.approx(trip.total_J, rel=1e-9)
    assert np.all(np.diff(required) >= -1e-9), (
        "the cost of turning back must rise with distance from home"
    )
