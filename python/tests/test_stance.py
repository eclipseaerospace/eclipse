# SPDX-License-Identifier: Apache-2.0
#
# Tests for eclipse.stance.
#
# Equilibrium is checked directly rather than through the solver that produced
# it: the returned loads are substituted back into the force and moment sums.
# A solver that agreed with itself would prove nothing.
#
# The resolution rule is tested as a property rather than against stored
# numbers. Minimum sum of squares means every other solution of the same
# equilibrium is larger, and the null-space direction of a rectangular footprint
# is known in closed form, so that can be asserted rather than trusted.
#
# The stances that have no solution get as much attention as the ones that do.
# A diagonal pair balances on level ground and on no slope at all, which is the
# stance rungs two and three of this project assumed throughout.

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from eclipse.io.platform import load_platform
from eclipse.platform import FootPosition, Platform
from eclipse.stance import (
    Gait,
    UnbalanceableStanceError,
    distribute_normal_load,
    executable_duty_ceiling,
    maximum_walking_speed,
    statically_stable_duty_factor,
    swing_reaction,
    wave_gait,
    within_stride_slip_ratio,
)
from eclipse.terramechanics import JanosiHanamotoModel, MohrCoulombModel

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
NOMINAL_QUADRUPED = (
    REPOSITORY_ROOT / "configs" / "platforms" / "nominal-quadruped.toml"
)

LUNAR_GRAVITY = 1.62
EARTH_GRAVITY = 9.81


@pytest.fixture(scope="module")
def quadruped() -> Platform:
    return load_platform(NOMINAL_QUADRUPED).platform


def _feet(platform: Platform, *identifiers: str) -> tuple[FootPosition, ...]:
    by_id = {foot.id: foot for foot in platform.footprint}
    return tuple(by_id[identifier] for identifier in identifiers)


def _loads(platform: Platform, slope: float, *identifiers: str) -> np.ndarray:
    stance = _feet(platform, *identifiers) if identifiers else platform.footprint
    return np.ravel(
        distribute_normal_load(
            platform=platform,
            stance=stance,
            gravity_m_per_s2=LUNAR_GRAVITY,
            slope_degrees=slope,
        ).normal_load_N
    )


# --- equilibrium, checked against the equations rather than the solver


@pytest.mark.parametrize("slope", [0.0, 10.0, 25.0, 35.0])
def test_the_loads_satisfy_force_and_moment_balance(
    quadruped: Platform, slope: float
) -> None:
    loads = _loads(quadruped, slope)
    weight_N = quadruped.total_mass_kg * LUNAR_GRAVITY
    radians = math.radians(slope)

    assert loads.sum() == pytest.approx(weight_N * math.cos(radians))

    positions = np.array([[f.x_m, f.y_m] for f in quadruped.footprint])
    pitching = -weight_N * math.sin(radians) * quadruped.center_of_mass_height_m
    assert float(positions[:, 0] @ loads) == pytest.approx(pitching, abs=1e-9)
    assert float(positions[:, 1] @ loads) == pytest.approx(0.0, abs=1e-9)


def test_level_ground_splits_the_load_evenly(quadruped: Platform) -> None:
    loads = _loads(quadruped, 0.0)
    assert np.allclose(loads, loads[0])
    assert float(loads[0]) == pytest.approx(
        quadruped.total_mass_kg * LUNAR_GRAVITY / quadruped.legs
    )


def test_load_moves_downhill_and_the_spread_grows_with_slope(
    quadruped: Platform,
) -> None:
    downhill = {"rear-left", "rear-right"}
    spreads = []
    for slope in (0.0, 10.0, 20.0, 30.0):
        distribution = distribute_normal_load(
            platform=quadruped,
            gravity_m_per_s2=LUNAR_GRAVITY,
            slope_degrees=slope,
        )
        loads = dict(
            zip(
                (foot.id for foot in distribution.feet),
                np.ravel(distribution.normal_load_N),
            )
        )
        if slope > 0.0:
            for uphill in ("front-left", "front-right"):
                for behind in downhill:
                    assert loads[behind] > loads[uphill], (
                        "climbing shifts load onto the downhill feet; if it did "
                        "not, the centre of mass height is not entering the "
                        "moment balance"
                    )
        spreads.append(float(np.ravel(distribution.spread)[0]))
    assert all(later > earlier for earlier, later in zip(spreads, spreads[1:]))


def test_the_spread_is_what_the_single_patch_model_averaged_away(
    quadruped: Platform,
) -> None:
    distribution = distribute_normal_load(
        platform=quadruped, gravity_m_per_s2=LUNAR_GRAVITY, slope_degrees=35.0
    )
    assert float(np.ravel(distribution.spread)[0]) == pytest.approx(1.84, abs=0.02)
    assert float(np.ravel(distribution.mean_N)[0]) == pytest.approx(
        quadruped.total_mass_kg * LUNAR_GRAVITY * math.cos(math.radians(35.0)) / 4.0
    ), "the mean is exactly what dividing by the foot count gives, which is why it hides this"


# --- the resolution rule


def test_the_solution_minimises_the_sum_of_squared_loads(
    quadruped: Platform,
) -> None:
    # For a rectangular footprint the null space of the equilibrium map is
    # spanned by (1, -1, -1, 1): it carries no net force and no moment about
    # either axis, so adding any multiple of it is another valid distribution.
    loads = _loads(quadruped, 20.0)
    null_direction = np.array([1.0, -1.0, -1.0, 1.0])

    positions = np.array([[f.x_m, f.y_m] for f in quadruped.footprint])
    assert null_direction.sum() == pytest.approx(0.0)
    assert float(positions[:, 0] @ null_direction) == pytest.approx(0.0)
    assert float(positions[:, 1] @ null_direction) == pytest.approx(0.0)

    chosen = float(loads @ loads)
    for step in (-2.0, -0.5, 0.5, 2.0):
        alternative = loads + step * null_direction
        assert float(alternative @ alternative) > chosen


def test_a_tripod_is_determinate_so_the_rule_does_not_apply(
    quadruped: Platform,
) -> None:
    # Three equations, three unknowns. There is one distribution and no choice
    # to make, which is why a three-legged platform is the honest test of
    # whether the rule leaked into anything downstream.
    stance = _feet(quadruped, "front-left", "rear-left", "rear-right")
    positions = np.array([[f.x_m, f.y_m] for f in stance])
    balance = np.vstack([np.ones(3), positions[:, 0], positions[:, 1]])
    assert np.linalg.matrix_rank(balance) == 3

    loads = _loads(quadruped, 20.0, "front-left", "rear-left", "rear-right")
    weight_N = quadruped.total_mass_kg * LUNAR_GRAVITY
    radians = math.radians(20.0)
    unique = np.linalg.solve(
        balance,
        np.array(
            [
                weight_N * math.cos(radians),
                -weight_N
                * math.sin(radians)
                * quadruped.center_of_mass_height_m,
                0.0,
            ]
        ),
    )
    assert np.allclose(loads, unique)


# --- stances with no solution


def test_a_diagonal_pair_balances_on_the_flat_and_on_no_slope_at_all(
    quadruped: Platform,
) -> None:
    # The stance rungs two and three assumed. Its two feet lie on a line through
    # the body centre, so the lateral moment equation forces the loads equal
    # while the pitching moment from a slope requires them unequal. On level
    # ground there is no pitching moment and the stance is fine; on any gradient
    # it has no quasi-static solution, and a trotting quadruped on a slope is
    # balanced dynamically rather than statically.
    flat = _loads(quadruped, 0.0, "front-left", "rear-right")
    assert flat[0] == pytest.approx(flat[1])
    assert flat.sum() == pytest.approx(quadruped.total_mass_kg * LUNAR_GRAVITY)

    for slope in (1.0, 10.0, 30.0):
        with pytest.raises(UnbalanceableStanceError, match="cannot balance"):
            _loads(quadruped, slope, "front-left", "rear-right")


def test_a_lateral_pair_cannot_balance_even_on_the_flat(
    quadruped: Platform,
) -> None:
    with pytest.raises(UnbalanceableStanceError, match="cannot balance"):
        _loads(quadruped, 0.0, "front-left", "front-right")


def test_an_empty_stance_is_refused(quadruped: Platform) -> None:
    with pytest.raises(UnbalanceableStanceError, match="no feet"):
        distribute_normal_load(
            platform=quadruped,
            stance=(),
            gravity_m_per_s2=LUNAR_GRAVITY,
            slope_degrees=0.0,
        )


# --- tipping


def test_a_front_biased_tripod_is_already_tipping_on_level_ground(
    quadruped: Platform,
) -> None:
    # Two feet uphill and one downhill puts the centre of mass on the edge of
    # the support triangle before the slope moves it anywhere.
    distribution = distribute_normal_load(
        platform=quadruped,
        stance=_feet(quadruped, "front-left", "front-right", "rear-left"),
        gravity_m_per_s2=LUNAR_GRAVITY,
        slope_degrees=np.array([0.0, 20.0]),
    )
    assert bool(np.all(distribution.any_foot_unloaded))


def test_the_four_foot_stance_tips_where_the_geometry_says_it_should(
    quadruped: Platform,
) -> None:
    # With feet at plus and minus L and the centre of mass at height h, the
    # uphill pair carries m*g*cos/4 - m*g*sin*h/(4L), which reaches zero at
    # tan(slope) = L/h. Mass and gravity cancel: tipping is pure geometry.
    half_length = max(foot.x_m for foot in quadruped.footprint)
    expected = math.degrees(
        math.atan(half_length / quadruped.center_of_mass_height_m)
    )

    for gravity in (LUNAR_GRAVITY, EARTH_GRAVITY):
        below = distribute_normal_load(
            platform=quadruped,
            gravity_m_per_s2=gravity,
            slope_degrees=expected - 0.5,
        )
        above = distribute_normal_load(
            platform=quadruped,
            gravity_m_per_s2=gravity,
            slope_degrees=expected + 0.5,
        )
        assert not bool(np.ravel(below.any_foot_unloaded)[0])
        assert bool(np.ravel(above.any_foot_unloaded)[0])


def test_tipping_arrives_before_the_traction_limit(quadruped: Platform) -> None:
    from eclipse.platform import maximum_traversable_slope_degrees
    from eclipse.terramechanics import MohrCoulombModel

    half_length = max(foot.x_m for foot in quadruped.footprint)
    tipping = math.degrees(
        math.atan(half_length / quadruped.center_of_mass_height_m)
    )
    slipping = maximum_traversable_slope_degrees(
        platform=quadruped,
        strength=MohrCoulombModel(cohesion=0.52, friction_angle_degrees=42.0),
        gravity_m_per_s2=LUNAR_GRAVITY,
    )
    assert tipping < slipping, (
        f"tipping at {tipping:.1f} degrees and slipping at {slipping:.1f}: the "
        "platform rotates about its downhill feet before they slide, so the "
        "traction limit is a bound on something that happens second"
    )


# --- what the mobility layer consumes


def test_a_stance_produces_one_loaded_patch_per_foot(quadruped: Platform) -> None:
    distribution = distribute_normal_load(
        platform=quadruped, gravity_m_per_s2=LUNAR_GRAVITY, slope_degrees=25.0
    )
    patches = distribution.loaded_patches(index=0)
    assert len(patches) == quadruped.legs
    assert [p.id for p in patches] == [f.id for f in quadruped.footprint]
    for patch in patches:
        assert patch.normal_stress_kPa() == pytest.approx(
            patch.normal_load_N / quadruped.foot_contact_area_m2 / 1000.0
        )


def test_the_loaded_patch_carries_no_notion_of_what_stands_on_it(
    quadruped: Platform,
) -> None:
    patch = distribute_normal_load(
        platform=quadruped, gravity_m_per_s2=LUNAR_GRAVITY, slope_degrees=0.0
    ).loaded_patches(index=0)[0]
    assert set(type(patch.patch).__dataclass_fields__) == {"half_width_m", "area_m2"}


# --- the consequence the mean hides


def test_a_small_foot_leaves_the_bearing_range_while_the_mean_stays_inside(
    quadruped: Platform,
) -> None:
    # Sinkage is non-linear in pressure, so averaging load across feet is not
    # conservative. At a 20 mm half-width the most-loaded foot passes the
    # published 20 mm sinkage ceiling on a gentle slope while the mean, which is
    # what the single-patch model used, never does at all.
    half_width = 0.020
    small_footed = Platform(
        **{
            **{
                name: getattr(quadruped, name)
                for name in Platform.__dataclass_fields__
            },
            "foot_half_width_m": half_width,
            "foot_contact_area_m2": math.pi * half_width**2,
        }
    )
    deformation_modulus_kPa_per_m = 1.4 / half_width + 820.0
    ceiling_m = 0.020

    def sinkage(load_N: float) -> float:
        stress_kPa = load_N / (math.pi * half_width**2) / 1000.0
        return stress_kPa / deformation_modulus_kPa_per_m

    distribution = distribute_normal_load(
        platform=small_footed,
        gravity_m_per_s2=LUNAR_GRAVITY,
        slope_degrees=np.linspace(0.0, 30.0, 61),
    )
    worst = sinkage(float(np.max(distribution.maximum_N)))
    mean_worst = sinkage(float(np.max(distribution.mean_N)))

    assert worst > ceiling_m
    assert mean_worst <= ceiling_m, (
        "the mean staying inside the range while the most-loaded foot leaves it "
        "is the whole reason four contacts are not one contact scaled"
    )


# --- guards


def test_feet_must_have_distinct_identifiers() -> None:
    with pytest.raises(ValueError, match="distinct ids"):
        Platform(
            body_mass_kg=40.0,
            leg_mass_kg=1.0,
            leg_length_m=0.3,
            footprint=(
                FootPosition(id="a", x_m=0.2, y_m=0.1),
                FootPosition(id="a", x_m=-0.2, y_m=-0.1),
            ),
            center_of_mass_height_m=0.3,
            feet_in_stance=2,
            foot_half_width_m=0.03,
            foot_contact_area_m2=math.pi * 0.03**2,
            stride_length_m=0.3,
            foot_clearance_m=0.05,
            nominal_speed_m_per_s=0.5,
        )


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_a_foot_at_no_finite_position_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        FootPosition(id="nowhere", x_m=bad, y_m=0.0)


# --- gait as a schedule


@pytest.fixture(scope="module")
def crawl() -> Gait:
    # Lift order rear-left, front-left, rear-right, front-right against a
    # footprint ordered front-left, front-right, rear-left, rear-right.
    return wave_gait(lift_order=(2, 0, 3, 1), duty_factor=0.75)


def test_a_wave_gait_lifts_one_foot_at_a_time_at_three_quarter_duty(
    crawl: Gait,
) -> None:
    phase = np.linspace(0.0, 1.0, 401, endpoint=False)
    assert np.all(crawl.feet_down(phase) == 3)


def test_every_leg_is_in_stance_for_exactly_the_duty_factor(crawl: Gait) -> None:
    phase = np.linspace(0.0, 1.0, 20001, endpoint=False)
    fraction = crawl.in_stance(phase).mean(axis=1)
    assert np.allclose(fraction, crawl.duty_factor, atol=1e-3)


def test_a_lift_order_that_is_not_a_permutation_is_refused() -> None:
    with pytest.raises(ValueError, match="permutation"):
        wave_gait(lift_order=(0, 0, 1, 2), duty_factor=0.75)


@pytest.mark.parametrize("duty", [0.0, -0.1, 1.5])
def test_an_unusable_duty_factor_is_refused(duty: float) -> None:
    with pytest.raises(ValueError, match=r"duty_factor must lie in \(0, 1\]"):
        Gait(duty_factor=duty, phase_offsets=(0.0, 0.5))


def test_a_gait_that_does_not_match_the_footprint_is_refused(
    quadruped: Platform,
) -> None:
    with pytest.raises(ValueError, match="does not match the footprint"):
        swing_reaction(
            platform=quadruped,
            gait=Gait(duty_factor=0.75, phase_offsets=(0.0, 0.5)),
        )


# --- within-stride slip, which replaces rung two's exact zero


def test_level_ground_slip_is_no_longer_zero(
    quadruped: Platform, crawl: Gait
) -> None:
    slip, reaction = within_stride_slip_ratio(
        platform=quadruped,
        gait=crawl,
        strength=MohrCoulombModel(cohesion=0.52, friction_angle_degrees=42.0),
        mobilization=JanosiHanamotoModel(shear_deformation_modulus=0.018),
        gravity_m_per_s2=LUNAR_GRAVITY,
    )
    assert reaction.peak_N > 0.0
    assert slip > 0.0, (
        "rung two returned exactly zero here because level ground demands no "
        "net traction; swinging a leg demands traction within the stride, and "
        "that is the mechanism the earlier model lacked"
    )


def test_swing_reaction_cancels_when_swings_are_in_antiphase(
    quadruped: Platform,
) -> None:
    # Four legs evenly spaced at half duty puts two legs in swing at once, one
    # accelerating while the other decelerates, and the reactions cancel
    # exactly. A real trot phases its diagonal pairs together instead, so they
    # add. The schedule decides, which is why it is a schedule.
    antiphase = wave_gait(lift_order=(2, 0, 3, 1), duty_factor=0.5)
    trot = Gait(duty_factor=0.5, phase_offsets=(0.0, 0.5, 0.5, 0.0))

    assert swing_reaction(platform=quadruped, gait=antiphase).peak_N == pytest.approx(
        0.0, abs=1e-9
    )
    assert swing_reaction(platform=quadruped, gait=trot).peak_N > 0.0


def test_the_reaction_goes_as_the_square_of_speed(
    quadruped: Platform, crawl: Gait
) -> None:
    def peak(speed: float) -> float:
        faster = Platform(
            **{
                **{
                    name: getattr(quadruped, name)
                    for name in Platform.__dataclass_fields__
                },
                "nominal_speed_m_per_s": speed,
            }
        )
        return swing_reaction(platform=faster, gait=crawl).peak_N

    assert peak(1.0) / peak(0.5) == pytest.approx(4.0, rel=1e-9)


def test_raising_the_duty_factor_raises_the_demand(quadruped: Platform) -> None:
    # Static stability is bought with tangential demand: more feet down means a
    # shorter swing, and swing acceleration goes as the inverse square of swing
    # duration. Past some duty factor the demand exceeds capacity and the gait
    # cannot be executed at this speed at all.
    peaks = [
        swing_reaction(
            platform=quadruped,
            gait=wave_gait(lift_order=(2, 0, 3, 1), duty_factor=duty),
        ).peak_N
        for duty in (0.75, 0.80, 0.85)
    ]
    assert all(later > earlier for earlier, later in zip(peaks, peaks[1:]))

    infeasible, _ = within_stride_slip_ratio(
        platform=quadruped,
        gait=wave_gait(lift_order=(2, 0, 3, 1), duty_factor=0.85),
        strength=MohrCoulombModel(cohesion=0.52, friction_angle_degrees=42.0),
        mobilization=JanosiHanamotoModel(shear_deformation_modulus=0.018),
        gravity_m_per_s2=LUNAR_GRAVITY,
    )
    assert not math.isfinite(infeasible)


def test_a_gait_with_a_flight_phase_is_refused(quadruped: Platform) -> None:
    with pytest.raises(ValueError, match="flight phase"):
        within_stride_slip_ratio(
            platform=quadruped,
            gait=Gait(duty_factor=0.2, phase_offsets=(0.0, 0.25, 0.5, 0.75)),
            strength=MohrCoulombModel(cohesion=0.52, friction_angle_degrees=42.0),
            mobilization=JanosiHanamotoModel(shear_deformation_modulus=0.018),
            gravity_m_per_s2=LUNAR_GRAVITY,
        )


def test_the_reaction_is_reported_against_what_it_would_otherwise_be(
    quadruped: Platform, crawl: Gait
) -> None:
    # The ground force is an upper bound. A body free to slow down takes the
    # same momentum change as a speed fluctuation and demands nothing of the
    # soil, so the honest answer is the pair rather than either alone.
    reaction = swing_reaction(platform=quadruped, gait=crawl)
    assert reaction.body_speed_fluctuation_m_per_s > 0.0
    assert reaction.body_speed_fluctuation_m_per_s < (
        quadruped.nominal_speed_m_per_s
    ), "a fluctuation larger than the mean speed would mean the body reverses"


def test_only_the_slope_feasible_gait_pays_for_it_on_the_flat(
    quadruped: Platform,
) -> None:
    strength = MohrCoulombModel(cohesion=0.52, friction_angle_degrees=42.0)
    mobilization = JanosiHanamotoModel(shear_deformation_modulus=0.018)

    def flat_slip(gait: Gait) -> float:
        return within_stride_slip_ratio(
            platform=quadruped,
            gait=gait,
            strength=strength,
            mobilization=mobilization,
            gravity_m_per_s2=LUNAR_GRAVITY,
        )[0]

    trot = Gait(duty_factor=0.5, phase_offsets=(0.0, 0.5, 0.5, 0.0))
    crawl = wave_gait(lift_order=(2, 0, 3, 1), duty_factor=0.75)

    assert flat_slip(crawl) > flat_slip(trot), (
        "the gait that keeps three feet down, and so has a quasi-static "
        "solution on a slope, is the one with more within-stride slip on the "
        "flat; static feasibility and traction demand pull against each other"
    )


# --- the interface, tested by a different morphology


NOMINAL_TRIPOD = REPOSITORY_ROOT / "configs" / "platforms" / "nominal-tripod.toml"


@pytest.fixture(scope="module")
def tripod() -> Platform:
    return load_platform(NOMINAL_TRIPOD).platform


def test_a_different_morphology_is_a_different_file_and_no_code_change(
    tripod: Platform,
) -> None:
    # The falsifiable test the architecture states. Everything below is the same
    # call the quadruped makes, against a platform with a different number of
    # legs, and nothing in the library was told which it is.
    distribution = distribute_normal_load(
        platform=tripod,
        gravity_m_per_s2=LUNAR_GRAVITY,
        slope_degrees=np.array([0.0, 10.0, 20.0]),
    )
    assert distribution.normal_load_N.shape == (3, 3)
    assert not bool(np.any(distribution.any_foot_unloaded))

    patches = distribution.loaded_patches(index=0)
    assert len(patches) == 3
    assert [p.id for p in patches] == ["front", "rear-left", "rear-right"]


def test_the_tripod_needs_no_resolution_rule(tripod: Platform) -> None:
    positions = np.array([[f.x_m, f.y_m] for f in tripod.footprint])
    balance = np.vstack([np.ones(3), positions[:, 0], positions[:, 1]])
    assert np.linalg.matrix_rank(balance) == 3, (
        "three non-collinear feet make the normal-load problem determinate, so "
        "the quadruped's minimum-norm rule has nothing to choose here; if a "
        "result changes when that rule changes, it leaked"
    )


def test_a_tripod_has_no_statically_stable_walking_gait(tripod: Platform) -> None:
    # Lifting any foot leaves two, and two feet balance a body only if its
    # centre of mass lies on the line between them. A tripod can stand and it
    # can fall over. This is a fact about tripods, surfaced by the same solve
    # the quadruped uses.
    for lifted in range(3):
        remaining = tuple(
            foot for index, foot in enumerate(tripod.footprint) if index != lifted
        )
        with pytest.raises(UnbalanceableStanceError):
            distribute_normal_load(
                platform=tripod,
                stance=remaining,
                gravity_m_per_s2=LUNAR_GRAVITY,
                slope_degrees=5.0,
            )


def test_the_tripod_tips_where_its_own_geometry_says(tripod: Platform) -> None:
    # Same formula, different footprint: the rear pair is at -0.25 and the front
    # foot at +0.25, so the front foot unloads at tan(slope) = L/h just as the
    # quadruped's uphill pair does. The rule is geometry, not leg count.
    half_length = max(foot.x_m for foot in tripod.footprint)
    expected = math.degrees(
        math.atan(half_length / tripod.center_of_mass_height_m)
    )
    below = distribute_normal_load(
        platform=tripod, gravity_m_per_s2=LUNAR_GRAVITY, slope_degrees=expected - 0.5
    )
    above = distribute_normal_load(
        platform=tripod, gravity_m_per_s2=LUNAR_GRAVITY, slope_degrees=expected + 0.5
    )
    assert not bool(np.ravel(below.any_foot_unloaded)[0])
    assert bool(np.ravel(above.any_foot_unloaded)[0])


# --- the duty window, which is a property of speed rather than of the platform


def test_the_stability_floor_is_arithmetic_about_leg_count(
    quadruped: Platform, tripod: Platform
) -> None:
    # Three of four feet down needs three quarters. No soil, no speed, no
    # gravity in it. Worth having as a function so a study reports it as the
    # textbook condition rather than as something it discovered.
    assert statically_stable_duty_factor(platform=quadruped, feet_down=3) == 0.75
    assert statically_stable_duty_factor(platform=quadruped, feet_down=2) == 0.5
    assert statically_stable_duty_factor(platform=tripod, feet_down=3) == 1.0


def test_the_solver_rediscovers_the_stability_floor_from_equilibrium(
    quadruped: Platform, crawl: Gait
) -> None:
    # The validation claim: nothing told the schedule that three quarters keeps
    # three feet down, and it does.
    phase = np.linspace(0.0, 1.0, 4001, endpoint=False)
    floor = statically_stable_duty_factor(platform=quadruped, feet_down=3)

    at_floor = wave_gait(lift_order=(2, 0, 3, 1), duty_factor=floor)
    just_below = wave_gait(lift_order=(2, 0, 3, 1), duty_factor=floor - 0.01)

    assert int(at_floor.feet_down(phase).min()) == 3
    assert int(just_below.feet_down(phase).min()) == 2


def test_the_closed_form_ceiling_matches_the_bisected_boundary(
    quadruped: Platform,
) -> None:
    # The ceiling is derived by setting swing demand equal to capacity. This
    # checks it against the boundary the slip solve actually refuses at, which
    # is a different computation reaching the same place.
    strength = MohrCoulombModel(cohesion=0.52, friction_angle_degrees=42.0)
    mobilization = JanosiHanamotoModel(shear_deformation_modulus=0.018)

    def executable(platform: Platform, duty: float) -> bool:
        slip, _ = within_stride_slip_ratio(
            platform=platform,
            gait=wave_gait(lift_order=(2, 0, 3, 1), duty_factor=duty),
            strength=strength,
            mobilization=mobilization,
            gravity_m_per_s2=LUNAR_GRAVITY,
        )
        return math.isfinite(slip)

    for speed in (0.20, 0.35, 0.50, 0.60):
        moving = Platform(
            **{
                **{
                    name: getattr(quadruped, name)
                    for name in Platform.__dataclass_fields__
                },
                "nominal_speed_m_per_s": speed,
            }
        )
        predicted = float(
            executable_duty_ceiling(
                platform=moving,
                strength=strength,
                gravity_m_per_s2=LUNAR_GRAVITY,
                speed_m_per_s=speed,
            )
        )
        low, high = 0.75, 0.999
        for _ in range(50):
            middle = 0.5 * (low + high)
            if executable(moving, middle):
                low = middle
            else:
                high = middle
        assert predicted == pytest.approx(0.5 * (low + high), abs=1e-6)


def test_the_ceiling_is_linear_in_speed(quadruped: Platform) -> None:
    strength = MohrCoulombModel(cohesion=0.52, friction_angle_degrees=42.0)
    speeds = np.array([0.1, 0.2, 0.4, 0.8])
    ceilings = executable_duty_ceiling(
        platform=quadruped,
        strength=strength,
        gravity_m_per_s2=LUNAR_GRAVITY,
        speed_m_per_s=speeds,
    )
    slopes = np.diff(ceilings) / np.diff(speeds)
    assert np.allclose(slopes, slopes[0])
    assert float(slopes[0]) < 0.0


def test_the_window_closes_at_a_finite_walking_speed(quadruped: Platform) -> None:
    # The number worth carrying out of rung three. Above it no duty factor both
    # keeps three feet down and leaves a swing the feet can react.
    strength = MohrCoulombModel(cohesion=0.52, friction_angle_degrees=42.0)
    limit = maximum_walking_speed(
        platform=quadruped, strength=strength, gravity_m_per_s2=LUNAR_GRAVITY
    )
    floor = statically_stable_duty_factor(platform=quadruped, feet_down=3)

    assert 0.5 < limit < 1.0
    for speed, expected_open in ((limit * 0.9, True), (limit * 1.1, False)):
        ceiling = float(
            executable_duty_ceiling(
                platform=quadruped,
                strength=strength,
                gravity_m_per_s2=LUNAR_GRAVITY,
                speed_m_per_s=speed,
            )
        )
        assert (ceiling > floor) is expected_open


def test_walking_slower_widens_the_window(quadruped: Platform) -> None:
    strength = MohrCoulombModel(cohesion=0.52, friction_angle_degrees=42.0)
    widths = [
        float(
            executable_duty_ceiling(
                platform=quadruped,
                strength=strength,
                gravity_m_per_s2=LUNAR_GRAVITY,
                speed_m_per_s=speed,
            )
        )
        - statically_stable_duty_factor(platform=quadruped, feet_down=3)
        for speed in (0.6, 0.5, 0.35, 0.25)
    ]
    assert all(later > earlier for earlier, later in zip(widths, widths[1:])), (
        "the duty window is a property of speed, not of the platform; reporting "
        "it as a fixed band overstates the constraint"
    )
