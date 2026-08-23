# SPDX-License-Identifier: Apache-2.0
#
# Tests for eclipse.schedule.
#
# On synthetic darkness rather than real terrain, because what needs pinning is
# the bookkeeping: that the hours attributed to a route sum to the sortie, that
# a place is read at the time it is stood on rather than at departure, and that
# the dwell is charged at the destination. A real illumination series has no
# analytic answer to check any of that against.

from __future__ import annotations

import numpy as np
import pytest

from eclipse.schedule import (
    OperatingCycle,
    arrival_weights_h,
    run_lunation,
    shadowed_hours,
)


def elapsed(walk_hours: float, samples: int) -> np.ndarray:
    return np.linspace(0.0, walk_hours, samples)


def constant(dark: bool, *, points: int, times: int) -> np.ndarray:
    return np.full((points, times), dark, dtype=bool)


# --- attributing the traverse to the samples that describe it


def test_the_weights_sum_to_the_walk() -> None:
    for samples in (2, 3, 17, 68):
        weights = arrival_weights_h(elapsed(1.56, samples))
        assert float(weights.sum()) == pytest.approx(1.56)


def test_the_ends_own_one_half_gap_and_the_middle_owns_two() -> None:
    weights = arrival_weights_h(np.array([0.0, 1.0, 2.0, 3.0]))
    assert weights.tolist() == pytest.approx([0.5, 1.0, 1.0, 0.5])


def test_uneven_samples_are_weighted_by_their_own_gaps() -> None:
    weights = arrival_weights_h(np.array([0.0, 1.0, 5.0]))
    assert weights.tolist() == pytest.approx([0.5, 2.5, 2.0])


def test_weights_refuse_a_route_that_goes_backwards() -> None:
    with pytest.raises(ValueError, match="must not decrease"):
        arrival_weights_h(np.array([0.0, 2.0, 1.0]))


def test_weights_refuse_a_single_sample() -> None:
    with pytest.raises(ValueError, match="at least two route samples"):
        arrival_weights_h(np.array([0.0]))


# --- the shadowed integral


def test_a_wholly_dark_world_charges_the_entire_sortie() -> None:
    hours = np.arange(0.0, 200.0, 0.25)
    walk, dwell = 1.5, 4.0
    steps = elapsed(walk, 25)
    charged = shadowed_hours(
        dark=constant(True, points=25, times=hours.size),
        hours=hours,
        elapsed_hours=steps,
        departure_hours=np.arange(0.0, 100.0, 5.0),
        dwell_hours=dwell,
    )
    assert charged.tolist() == pytest.approx([2.0 * walk + dwell] * charged.size)


def test_a_wholly_lit_world_charges_nothing() -> None:
    hours = np.arange(0.0, 200.0, 0.25)
    charged = shadowed_hours(
        dark=constant(False, points=25, times=hours.size),
        hours=hours,
        elapsed_hours=elapsed(1.5, 25),
        departure_hours=np.arange(0.0, 100.0, 5.0),
        dwell_hours=4.0,
    )
    assert charged.tolist() == pytest.approx([0.0] * charged.size)


def test_a_permanently_dark_destination_costs_its_dwell_and_its_approach() -> None:
    # The last sample is dark at every time and nothing else ever is, which is
    # the shape of the real route: the dwell plus one half-gap on each pass.
    hours = np.arange(0.0, 200.0, 0.25)
    walk, dwell, samples = 2.0, 6.0, 21
    dark = constant(False, points=samples, times=hours.size)
    dark[-1, :] = True
    steps = elapsed(walk, samples)
    charged = shadowed_hours(
        dark=dark,
        hours=hours,
        elapsed_hours=steps,
        departure_hours=np.array([0.0, 17.0, 41.0]),
        dwell_hours=dwell,
    )
    half_gap = 0.5 * float(steps[-1] - steps[-2])
    assert charged.tolist() == pytest.approx([dwell + 2.0 * half_gap] * 3)


def test_a_place_is_read_when_it_is_stood_on_and_not_at_departure() -> None:
    # One sample, dark for a single hour in the middle of the window. A
    # departure timed so the platform is there during that hour is charged; the
    # same departure would be charged nothing if the state were read at the
    # start of the sortie instead.
    hours = np.arange(0.0, 60.0, 0.25)
    walk, samples = 4.0, 9
    steps = elapsed(walk, samples)
    dark = constant(False, points=samples, times=hours.size)
    middle = samples // 2
    window = (hours >= 20.0) & (hours < 21.0)
    dark[middle, window] = True

    arrives_in_the_dark = 20.5 - float(steps[middle])
    charged = shadowed_hours(
        dark=dark,
        hours=hours,
        elapsed_hours=steps,
        departure_hours=np.array([arrives_in_the_dark, 0.0]),
        dwell_hours=0.0,
    )
    assert charged[0] > 0.0
    assert charged[1] == pytest.approx(0.0)


def test_a_sortie_running_off_the_end_of_the_series_is_refused() -> None:
    hours = np.arange(0.0, 10.0, 0.25)
    with pytest.raises(ValueError, match="runs past the illumination series"):
        shadowed_hours(
            dark=constant(True, points=5, times=hours.size),
            hours=hours,
            elapsed_hours=elapsed(2.0, 5),
            departure_hours=np.array([9.0]),
            dwell_hours=4.0,
        )


def test_the_shadowed_integral_refuses_a_mismatched_grid() -> None:
    hours = np.arange(0.0, 40.0, 0.25)
    with pytest.raises(ValueError, match="one row per route sample"):
        shadowed_hours(
            dark=constant(True, points=4, times=hours.size),
            hours=hours,
            elapsed_hours=elapsed(2.0, 5),
            departure_hours=np.array([0.0]),
            dwell_hours=1.0,
        )


# --- the operating cycle


def test_a_free_sortie_on_a_full_battery_packs_the_lunation() -> None:
    hours = np.arange(0.0, 200.0, 0.25)
    cycle = run_lunation(
        hours=hours,
        energy_Wh=np.zeros(hours.size),
        sortie_hours=10.0,
        charge_W=np.zeros(hours.size),
        battery_Wh=100.0,
        lunation_hours=100.0,
    )
    assert cycle.sorties == 10
    assert cycle.charging_hours == pytest.approx(0.0)
    assert cycle.waiting_hours == pytest.approx(0.0)
    assert cycle.limited_by == "sortie duration"


def test_an_unaffordable_sortie_never_departs_however_long_it_charges() -> None:
    hours = np.arange(0.0, 200.0, 0.25)
    cycle = run_lunation(
        hours=hours,
        energy_Wh=np.full(hours.size, 500.0),
        sortie_hours=5.0,
        charge_W=np.full(hours.size, 200.0),
        battery_Wh=100.0,
        lunation_hours=100.0,
    )
    assert cycle.sorties == 0
    assert cycle.limited_by == "battery"


def test_a_dark_charge_point_stops_the_cycle_after_the_stored_sorties() -> None:
    # The array makes nothing, so the battery buys exactly as many sorties as
    # it holds and the rest of the lunation is spent charging on zero.
    hours = np.arange(0.0, 200.0, 0.25)
    cycle = run_lunation(
        hours=hours,
        energy_Wh=np.full(hours.size, 40.0),
        sortie_hours=5.0,
        charge_W=np.zeros(hours.size),
        battery_Wh=100.0,
        lunation_hours=100.0,
    )
    assert cycle.sorties == 2
    assert cycle.charging_hours > cycle.sortie_hours
    assert cycle.limited_by == "charge rate"


def test_waiting_is_distinguished_from_charging_by_a_full_battery() -> None:
    # Affordable only in the second half of the window, and the battery starts
    # full, so every hour before that is a wait rather than a charge.
    hours = np.arange(0.0, 200.0, 0.25)
    energy = np.where(hours < 50.0, 500.0, 10.0)
    cycle = run_lunation(
        hours=hours,
        energy_Wh=energy,
        sortie_hours=5.0,
        charge_W=np.full(hours.size, 400.0),
        battery_Wh=100.0,
        lunation_hours=100.0,
    )
    assert cycle.sorties > 0
    assert cycle.waiting_hours == pytest.approx(50.0, abs=0.5)
    assert cycle.limited_by == "departure windows"


def test_the_cycle_refuses_an_impossible_battery() -> None:
    hours = np.arange(0.0, 20.0, 0.25)
    with pytest.raises(ValueError, match="battery_Wh must be positive"):
        run_lunation(
            hours=hours,
            energy_Wh=np.zeros(hours.size),
            sortie_hours=1.0,
            charge_W=np.zeros(hours.size),
            battery_Wh=0.0,
            lunation_hours=10.0,
        )


def test_a_cycle_cannot_report_negative_hours() -> None:
    with pytest.raises(ValueError, match="waiting_hours must not be negative"):
        OperatingCycle(
            sorties=1, sortie_hours=1.0, charging_hours=1.0, waiting_hours=-1.0
        )
