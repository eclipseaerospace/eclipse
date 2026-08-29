# SPDX-License-Identifier: Apache-2.0
#
# Tests for eclipse.planning.
#
# On synthetic terrain with paths that can be written down: a flat plane where
# the answer is the straight line, a wall with a gap where the answer has to go
# through the gap, and a ramp where climbing and descending differ so the
# outbound and return paths are not each other reversed.
#
# The cost curve is synthetic throughout. The planner takes a table rather than
# a model precisely so that the search can be checked without any physics in
# the room, and a test that had to build a soil to check a shortest path would
# be testing two things at once.

from __future__ import annotations

import numpy as np
import pytest

from eclipse.planning import (
    NEIGHBOURS,
    Route,
    TraversalCost,
    minimum_slope_capability_deg,
    plan_route,
    round_trip_energy_J,
)

SLOPES = np.linspace(-89.0, 89.0, 1781)


def flat_cost(*, limit_deg: float = 45.0) -> TraversalCost:
    """One joule per metre whatever the slope: distance is the only cost."""
    return TraversalCost(
        slope_deg=SLOPES,
        joules_per_metre=np.ones_like(SLOPES),
        limit_deg=limit_deg,
    )


def climbing_cost(*, limit_deg: float = 45.0) -> TraversalCost:
    """Cheap downhill, and uphill that gets dearer the steeper it is.

    Cost that depends only on the sign of the slope is not enough to separate
    the directions: with cost proportional to length alone, the cheapest climb
    and the cheapest descent are both the shortest monotone path and come out
    identical. Making the climb steepness-dependent is what makes a gentle
    detour worth taking in one direction and not the other, which is the
    asymmetry the directed graph exists for.
    """
    return TraversalCost(
        slope_deg=SLOPES,
        joules_per_metre=np.where(SLOPES > 0.0, 1.0 + SLOPES**2 / 20.0, 1.0),
        limit_deg=limit_deg,
    )


# --- the cost table refuses what would make the search invalid


def test_a_negative_cost_is_refused() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        TraversalCost(
            slope_deg=SLOPES,
            joules_per_metre=np.where(SLOPES < -20.0, -1.0, 1.0),
            limit_deg=40.0,
        )


def test_an_unsorted_slope_axis_is_refused() -> None:
    with pytest.raises(ValueError, match="must increase strictly"):
        TraversalCost(
            slope_deg=np.array([0.0, -1.0, 2.0]),
            joules_per_metre=np.ones(3),
            limit_deg=40.0,
        )


def test_an_impossible_limit_is_refused() -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 90"):
        TraversalCost(
            slope_deg=SLOPES, joules_per_metre=np.ones_like(SLOPES), limit_deg=0.0
        )


def test_the_table_reports_infinity_beyond_the_limit() -> None:
    cost = flat_cost(limit_deg=30.0)
    values = cost.at(np.array([0.0, 29.0, 31.0, -31.0]))
    assert np.isfinite(values[:2]).all()
    assert not np.isfinite(values[2:]).any()


# --- the search, against answers that can be written down


def test_on_a_plane_the_cheapest_path_is_the_straight_one() -> None:
    plan = plan_route(
        elevation_m=np.zeros((21, 21)),
        cell_size_m=1.0,
        start=(10, 0),
        goal=(10, 20),
        cost=flat_cost(),
    )
    route = plan.route
    assert route is not None
    assert route.steps == 20
    assert route.distance_m == pytest.approx(20.0)
    assert set(route.rows.tolist()) == {10}


def test_a_diagonal_costs_its_true_length_rather_than_one_step() -> None:
    # Twenty diagonal steps of root two, not twenty steps of one. Charging a
    # diagonal as a single step is what makes a planner produce staircases.
    plan = plan_route(
        elevation_m=np.zeros((21, 21)),
        cell_size_m=1.0,
        start=(0, 0),
        goal=(20, 20),
        cost=flat_cost(),
    )
    route = plan.route
    assert route is not None
    assert route.steps == 20
    assert route.distance_m == pytest.approx(20.0 * np.sqrt(2.0))


def test_a_wall_is_gone_around_and_a_gap_is_gone_through() -> None:
    values = np.zeros((41, 41))
    values[:, 20] = 500.0
    values[30:33, 20] = 0.0
    plan = plan_route(
        elevation_m=values,
        cell_size_m=10.0,
        start=(5, 5),
        goal=(5, 35),
        cost=flat_cost(limit_deg=40.0),
    )
    route = plan.route
    assert route is not None
    crossing = route.rows[route.columns == 20]
    assert crossing.size > 0
    assert set(crossing.tolist()) <= {30, 31, 32}


def test_a_sealed_goal_is_reported_rather_than_raised() -> None:
    values = np.zeros((41, 41))
    values[:, 20] = 5000.0
    plan = plan_route(
        elevation_m=values,
        cell_size_m=10.0,
        start=(5, 5),
        goal=(5, 35),
        cost=flat_cost(limit_deg=40.0),
    )
    assert plan.route is None
    assert not plan.found
    assert "no path exists" in plan.reason
    assert plan.reached_cell_fraction < 1.0


def test_the_return_is_not_the_outbound_reversed_when_direction_costs() -> None:
    # A ridge with two ways over it: a short steep notch on the straight line
    # and a long gentle saddle away from it. Climbing prefers the saddle
    # because steepness is dear; descending prefers the notch because going
    # down is cheap and the notch is shorter. Same terrain, different route,
    # which is why the graph is directed.
    rows, columns = np.meshgrid(np.arange(31), np.arange(31), indexing="ij")
    values = np.zeros_like(rows, dtype=float)
    values += 120.0 * np.exp(-((columns - 15.0) ** 2) / 8.0)
    values -= 105.0 * np.exp(
        -((columns - 15.0) ** 2) / 8.0 - ((rows - 28.0) ** 2) / 40.0
    )
    values += 2.0 * columns
    up = plan_route(
        elevation_m=values,
        cell_size_m=10.0,
        start=(15, 0),
        goal=(15, 30),
        cost=climbing_cost(limit_deg=80.0),
    )
    down = plan_route(
        elevation_m=values,
        cell_size_m=10.0,
        start=(15, 30),
        goal=(15, 0),
        cost=climbing_cost(limit_deg=80.0),
    )
    assert up.route is not None and down.route is not None
    assert up.route.total_energy_J > down.route.total_energy_J
    assert not np.array_equal(up.route.rows, down.route.rows[::-1])


def test_a_route_never_costs_more_than_any_other_route_it_could_have_taken() -> None:
    # The straight line is a path the search was free to choose, so the search
    # cannot come back with something dearer. This is the invariant that caught
    # the study comparing two routes at different samplings.
    values = np.cumsum(
        np.cumsum(np.sin(np.arange(41.0))[:, None] * np.cos(np.arange(41.0)), axis=0),
        axis=1,
    )
    cost = flat_cost(limit_deg=89.0)
    plan = plan_route(
        elevation_m=values, cell_size_m=10.0, start=(0, 0), goal=(40, 40), cost=cost
    )
    route = plan.route
    assert route is not None
    steps = np.arange(41)
    rise = np.diff(values[steps, steps])
    run = 10.0 * np.hypot(1.0, 1.0)
    straight = float(
        (cost.at(np.degrees(np.arctan2(rise, run))) * np.hypot(rise, run)).sum()
    )
    assert route.total_energy_J <= straight + 1e-9


def test_a_goal_outside_the_grid_is_refused() -> None:
    with pytest.raises(ValueError, match="lies outside a 5 by 5 grid"):
        plan_route(
            elevation_m=np.zeros((5, 5)),
            cell_size_m=1.0,
            start=(0, 0),
            goal=(0, 9),
            cost=flat_cost(),
        )


def test_undecidable_steps_count_only_those_near_the_limit() -> None:
    route = Route(
        rows=np.arange(4),
        columns=np.arange(4),
        elevation_m=np.zeros(4),
        step_length_m=np.ones(3),
        step_slope_deg=np.array([5.0, 38.0, -39.0]),
        step_energy_J=np.ones(3),
    )
    assert route.undecidable_steps(limit_deg=40.0, error_deg=2.5) == 2


# --- the capability a journey needs, which is a connectivity question


def test_the_capability_needed_is_the_gentlest_way_through() -> None:
    values = np.zeros((41, 41))
    values[:, 20] = 500.0
    values[30, 20] = 100.0
    needed = minimum_slope_capability_deg(
        elevation_m=values,
        cell_size_m=10.0,
        start=(5, 5),
        goal=(5, 35),
        tolerance_deg=0.01,
    )
    assert needed == pytest.approx(np.degrees(np.arctan2(100.0, 10.0)), abs=0.05)


def test_a_sealed_goal_needs_a_near_vertical_capability() -> None:
    # Not infinite. Every step on a grid of finite elevations has a slope short
    # of vertical, so the answer is always finite and a sealed goal reports how
    # absurd the climb would be rather than that there is none.
    values = np.zeros((21, 21))
    values[:, 10] = 1.0e6
    assert minimum_slope_capability_deg(
        elevation_m=values, cell_size_m=1.0, start=(5, 2), goal=(5, 18)
    ) > 89.0


def test_a_plane_needs_nothing() -> None:
    assert minimum_slope_capability_deg(
        elevation_m=np.zeros((21, 21)),
        cell_size_m=10.0,
        start=(0, 0),
        goal=(20, 20),
        tolerance_deg=0.01,
    ) == pytest.approx(0.0, abs=0.02)


def test_the_neighbourhood_is_eight_connected_and_symmetric() -> None:
    assert len(NEIGHBOURS) == 8
    assert len(set(NEIGHBOURS)) == 8
    for row_step, column_step in NEIGHBOURS:
        assert (-row_step, -column_step) in NEIGHBOURS


# --- round trips, where the asymmetry is the whole point


def test_a_round_trip_on_a_plane_is_twice_the_one_way() -> None:
    field = round_trip_energy_J(
        elevation_m=np.zeros((11, 11)),
        cell_size_m=1.0,
        home=(5, 5),
        cost=flat_cost(),
    )
    assert field[5, 5] == pytest.approx(0.0)
    assert field[0, 0] == pytest.approx(2.0 * 5.0 * np.sqrt(2.0))


def test_a_round_trip_is_dearer_than_twice_the_cheap_direction() -> None:
    # Down is cheap and up is dear, so a round trip to a low place costs more
    # than twice the descent and less than twice the climb. A model that
    # doubled either one would be wrong in a known direction.
    rows, columns = np.meshgrid(np.arange(21), np.arange(21), indexing="ij")
    values = -30.0 * columns.astype(float)
    cost = climbing_cost(limit_deg=80.0)
    field = round_trip_energy_J(
        elevation_m=values, cell_size_m=10.0, home=(10, 0), cost=cost
    )
    down = plan_route(
        elevation_m=values, cell_size_m=10.0, start=(10, 0), goal=(10, 20), cost=cost
    )
    up = plan_route(
        elevation_m=values, cell_size_m=10.0, start=(10, 20), goal=(10, 0), cost=cost
    )
    assert down.route is not None and up.route is not None
    assert field[10, 20] == pytest.approx(
        down.route.total_energy_J + up.route.total_energy_J
    )
    assert field[10, 20] > 2.0 * down.route.total_energy_J
    assert field[10, 20] < 2.0 * up.route.total_energy_J


def test_an_unreachable_cell_has_no_round_trip() -> None:
    values = np.zeros((21, 21))
    values[:, 10] = 5000.0
    field = round_trip_energy_J(
        elevation_m=values, cell_size_m=10.0, home=(5, 2), cost=flat_cost(limit_deg=40.0)
    )
    assert np.isfinite(field[5, 5])
    assert not np.isfinite(field[5, 18])


def test_a_home_outside_the_grid_is_refused() -> None:
    with pytest.raises(ValueError, match="lies outside a 5 by 5 grid"):
        round_trip_energy_J(
            elevation_m=np.zeros((5, 5)),
            cell_size_m=1.0,
            home=(9, 0),
            cost=flat_cost(),
        )


def test_a_unit_cost_makes_the_round_trip_exactly_twice_the_one_way() -> None:
    # A metre costs a metre in either direction, so a field over a unit cost is
    # symmetric and the crew's traverse range can be read off it by halving.
    # The energy field is not symmetric and halving it would be wrong; this is
    # the property that makes one of the two safe to halve and not the other.
    rows, columns = np.meshgrid(np.arange(25), np.arange(25), indexing="ij")
    values = 20.0 * columns + 8.0 * np.sin(rows / 2.0)
    unit = flat_cost(limit_deg=60.0)
    field = round_trip_energy_J(
        elevation_m=values, cell_size_m=10.0, home=(12, 0), cost=unit
    )
    out = plan_route(
        elevation_m=values, cell_size_m=10.0, start=(12, 0), goal=(12, 24), cost=unit
    )
    back = plan_route(
        elevation_m=values, cell_size_m=10.0, start=(12, 24), goal=(12, 0), cost=unit
    )
    assert out.route is not None and back.route is not None
    assert out.route.total_energy_J == pytest.approx(back.route.total_energy_J)
    assert field[12, 24] == pytest.approx(2.0 * out.route.total_energy_J)

    dear = climbing_cost(limit_deg=60.0)
    asymmetric = round_trip_energy_J(
        elevation_m=values, cell_size_m=10.0, home=(12, 0), cost=dear
    )
    up = plan_route(
        elevation_m=values, cell_size_m=10.0, start=(12, 0), goal=(12, 24), cost=dear
    )
    assert up.route is not None
    assert asymmetric[12, 24] != pytest.approx(2.0 * up.route.total_energy_J)
