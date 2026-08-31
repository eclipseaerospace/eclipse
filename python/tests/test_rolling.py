# SPDX-License-Identifier: Apache-2.0
#
# Tests for eclipse.rolling.
#
# The compaction resistance is the one form here that could be wrong silently,
# so it is checked against Wong's closed form typed straight from the equation
# and sharing no code with the module. The two routes are algebraically the same
# result reached differently: the module integrates pressure through depth over
# the rut, Wong substitutes the sinkage equation into that integral and collects
# terms. Agreement to machine precision across exponents, loads and geometries
# is what makes the form verified rather than transcribed.
#
# The rest of the file pins the three things that distinguish a wheel from a
# foot: it pays no swing, it pays compaction as a resistance rather than
# downward, and it cannot mobilise its full shear capacity at any slip.

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from eclipse.io.platform import (
    PlatformFileError,
    load_platform,
    load_wheeled_platform,
)
from eclipse.rolling import (
    SLIP_CEILING,
    WheeledPlatform,
    compaction_resistance_N,
    contact_length_m,
    mobilized_tractive_fraction,
    rigid_wheel_sinkage_m,
    rolling_cost_of_transport,
    wheel_equilibrium_slip_ratio,
    wheel_maximum_traversable_slope_degrees,
)
from eclipse.terramechanics import (
    BekkerModel,
    JanosiHanamotoModel,
    MohrCoulombModel,
)

LUNAR_GRAVITY = 1.62
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
NOMINAL_ROVER = REPOSITORY_ROOT / "configs" / "platforms" / "nominal-rover.toml"
NOMINAL_QUADRUPED = (
    REPOSITORY_ROOT / "configs" / "platforms" / "nominal-quadruped.toml"
)


def rover(**overrides: float | int) -> WheeledPlatform:
    values: dict[str, float | int] = {
        "body_mass_kg": 44.0,
        "wheel_mass_kg": 1.5,
        "wheels": 4,
        "wheel_diameter_m": 0.25,
        "wheel_width_m": 0.10,
        "wheelbase_m": 0.50,
        "track_width_m": 0.40,
        "center_of_mass_height_m": 0.30,
        "nominal_speed_m_per_s": 0.5,
    }
    values.update(overrides)
    return WheeledPlatform(**values)  # type: ignore[arg-type]


def soil(sinkage_exponent: float = 1.0) -> BekkerModel:
    return BekkerModel(
        cohesive_modulus=1.4,
        frictional_modulus=820.0,
        sinkage_exponent=sinkage_exponent,
    )


STRENGTH = MohrCoulombModel(cohesion=0.52, friction_angle_degrees=42.0)
MOBILIZATION = JanosiHanamotoModel(shear_deformation_modulus=0.018)


def wong_compaction_resistance_N(
    *, load_N: float, width_m: float, diameter_m: float, modulus_kPa: float, n: float
) -> float:
    """Wong (2008) eq. for rigid-wheel compaction resistance, typed from source.

    R_c = 1/((n+1) * (b*k)^(1/(2n+1))) * [3W/((3-n)*sqrt(D))]^((2n+2)/(2n+1))

    Shares no code with eclipse.rolling on purpose: a check that imports the
    thing it is checking proves only that the code agrees with itself.
    """
    modulus = modulus_kPa * 1.0e3
    return float(
        (1.0 / ((n + 1.0) * (width_m * modulus) ** (1.0 / (2.0 * n + 1.0))))
        * (3.0 * load_N / ((3.0 - n) * math.sqrt(diameter_m)))
        ** ((2.0 * n + 2.0) / (2.0 * n + 1.0))
    )


# --- the compaction resistance, against source


@pytest.mark.parametrize("sinkage_exponent", [0.6, 0.8, 1.0, 1.2])
@pytest.mark.parametrize("load_N", [20.25, 100.0, 400.0])
@pytest.mark.parametrize(
    ("width_m", "diameter_m"), [(0.10, 0.25), (0.16, 0.50), (0.06, 0.20)]
)
def test_compaction_resistance_matches_wong(
    sinkage_exponent: float, load_N: float, width_m: float, diameter_m: float
) -> None:
    model = soil(sinkage_exponent)
    platform = rover(wheel_width_m=width_m, wheel_diameter_m=diameter_m)
    sinkage = rigid_wheel_sinkage_m(
        platform=platform, contact_model=model, normal_load_N=load_N
    )
    computed = float(
        compaction_resistance_N(
            platform=platform, contact_model=model, sinkage_m=sinkage
        )
    )
    published = wong_compaction_resistance_N(
        load_N=load_N,
        width_m=width_m,
        diameter_m=diameter_m,
        modulus_kPa=float(model.deformation_modulus(platform.wheel_half_width_m)),
        n=sinkage_exponent,
    )
    assert computed == pytest.approx(published, rel=1e-12)


def test_the_modulus_takes_a_half_width_and_the_rut_takes_the_full_one() -> None:
    # The units trap. Wong writes one symbol b for two roles and this project's
    # contact models take a half-width, so using one number for both understates
    # sinkage and every quantity downstream of it. Silent, and it flatters the
    # wheel, which is why it is pinned rather than commented.
    platform = rover(wheel_width_m=0.10)
    assert platform.rut_width_m == 0.10
    assert platform.wheel_half_width_m == 0.05
    model = soil()
    assert model.deformation_modulus(0.05) > model.deformation_modulus(0.10)


def test_sinkage_inverts_the_load_it_was_solved_from() -> None:
    # Round trip through Bekker's contact integral: pressing the wheel to the
    # depth the solver returns must carry the load that produced it.
    model = soil(0.8)
    platform = rover()
    load_N = np.array([5.0, 20.25, 80.0, 300.0])
    depth = rigid_wheel_sinkage_m(
        platform=platform, contact_model=model, normal_load_N=load_N
    )
    exponent = model.sinkage_exponent
    modulus = model.deformation_modulus(platform.wheel_half_width_m) * 1.0e3
    carried = (
        platform.rut_width_m
        * modulus
        * (3.0 - exponent)
        / 3.0
        * np.sqrt(platform.wheel_diameter_m)
        * depth ** (exponent + 0.5)
    )
    assert carried == pytest.approx(load_N, rel=1e-12)


def test_a_wider_wheel_sinks_less() -> None:
    model = soil()
    narrow = rigid_wheel_sinkage_m(
        platform=rover(wheel_width_m=0.06), contact_model=model, normal_load_N=20.25
    )
    wide = rigid_wheel_sinkage_m(
        platform=rover(wheel_width_m=0.20), contact_model=model, normal_load_N=20.25
    )
    assert float(wide) < float(narrow)


def test_a_wheel_with_no_load_is_refused() -> None:
    with pytest.raises(ValueError, match="no load"):
        rigid_wheel_sinkage_m(
            platform=rover(), contact_model=soil(), normal_load_N=np.array([20.0, 0.0])
        )


# --- mobilization along the contact


def test_mobilization_is_lower_than_a_foot_at_the_same_slip() -> None:
    # A foot slides at one displacement; a wheel's patch runs from zero shear at
    # the entry to its maximum at the bottom, so the mean is always below the
    # single-point value. This is why a wheel needs more slip for the same grip.
    length = 0.045
    slip = 0.3
    patch_mean = float(
        mobilized_tractive_fraction(
            mobilization=MOBILIZATION, contact_length_m=length, slip_ratio=slip
        )
    )
    single_point = MOBILIZATION.mobilized_fraction(shear_displacement=slip * length)
    assert patch_mean < float(single_point)


def test_mobilization_vanishes_at_zero_slip_without_dividing_by_zero() -> None:
    values = mobilized_tractive_fraction(
        mobilization=MOBILIZATION,
        contact_length_m=0.045,
        slip_ratio=np.array([0.0, 1e-12, 1e-6]),
    )
    assert bool(np.isfinite(values).all())
    assert float(values[0]) == 0.0
    assert bool((np.diff(values) >= 0.0).all())


def test_a_wheel_never_reaches_full_capacity() -> None:
    # The result that decides the rover's slope limit: at the slip ceiling this
    # contact patch develops about two thirds of the shear the soil could give,
    # because the patch is only a few times the shear deformation modulus long.
    reached = float(
        mobilized_tractive_fraction(
            mobilization=MOBILIZATION, contact_length_m=0.045, slip_ratio=SLIP_CEILING
        )
    )
    assert 0.5 < reached < 0.8


def test_a_longer_contact_mobilizes_more_at_the_same_slip() -> None:
    short = mobilized_tractive_fraction(
        mobilization=MOBILIZATION, contact_length_m=0.03, slip_ratio=0.3
    )
    long = mobilized_tractive_fraction(
        mobilization=MOBILIZATION, contact_length_m=0.09, slip_ratio=0.3
    )
    assert float(long) > float(short)


# --- slip and the slope limit


def test_a_wheel_slips_on_level_ground_and_a_foot_does_not() -> None:
    # The structural difference. Level ground demands no tangential force of a
    # foot, so its equilibrium slip is exactly zero. A wheel must still drag
    # itself out of its own rut, so it never rolls without slipping.
    level = float(
        wheel_equilibrium_slip_ratio(
            platform=rover(),
            contact_model=soil(),
            strength=STRENGTH,
            mobilization=MOBILIZATION,
            gravity_m_per_s2=LUNAR_GRAVITY,
            slope_degrees=0.0,
        )
    )
    assert 0.0 < level < 0.5


def test_slip_increases_with_slope_and_then_has_no_solution() -> None:
    slopes = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 35.0])
    slip = wheel_equilibrium_slip_ratio(
        platform=rover(),
        contact_model=soil(),
        strength=STRENGTH,
        mobilization=MOBILIZATION,
        gravity_m_per_s2=LUNAR_GRAVITY,
        slope_degrees=slopes,
    )
    finite = np.isfinite(slip)
    assert bool((np.diff(slip[finite]) > 0.0).all())
    assert not bool(finite[-1])


def test_the_solved_slip_actually_balances_the_demand() -> None:
    # The bisection is the only iteration in the physics layer, so its answer is
    # checked against the balance it claims to solve rather than trusted.
    platform, model = rover(), soil()
    slope = 12.0
    slip = float(
        wheel_equilibrium_slip_ratio(
            platform=platform,
            contact_model=model,
            strength=STRENGTH,
            mobilization=MOBILIZATION,
            gravity_m_per_s2=LUNAR_GRAVITY,
            slope_degrees=slope,
        )
    )
    weight_N = platform.total_mass_kg * LUNAR_GRAVITY
    load_N = weight_N * np.cos(np.radians(slope)) / platform.wheels
    depth = rigid_wheel_sinkage_m(
        platform=platform, contact_model=model, normal_load_N=load_N
    )
    length = contact_length_m(platform=platform, sinkage_m=depth)
    demand = float(
        compaction_resistance_N(
            platform=platform, contact_model=model, sinkage_m=depth
        )
    ) + weight_N * np.sin(np.radians(slope)) / platform.wheels
    capacity = (
        STRENGTH.cohesion * 1.0e3 * platform.rut_width_m * float(length)
        + load_N * STRENGTH.friction_coefficient
    )
    developed = float(
        mobilized_tractive_fraction(
            mobilization=MOBILIZATION, contact_length_m=length, slip_ratio=slip
        )
    )
    assert developed * capacity == pytest.approx(demand, rel=1e-6)


def test_the_slope_limit_sits_where_slip_diverges() -> None:
    platform, model = rover(), soil()
    limit = wheel_maximum_traversable_slope_degrees(
        platform=platform,
        contact_model=model,
        strength=STRENGTH,
        mobilization=MOBILIZATION,
        gravity_m_per_s2=LUNAR_GRAVITY,
    )
    below, above = wheel_equilibrium_slip_ratio(
        platform=platform,
        contact_model=model,
        strength=STRENGTH,
        mobilization=MOBILIZATION,
        gravity_m_per_s2=LUNAR_GRAVITY,
        slope_degrees=np.array([limit - 0.05, limit + 0.05]),
    )
    assert np.isfinite(below)
    assert not np.isfinite(above)


def test_a_bigger_wheel_climbs_further() -> None:
    # Diameter sets contact length, contact length sets how much of the soil's
    # strength any slip can mobilise. The single most consequential number on a
    # rover, and the one this platform is least generous about.
    limits = [
        wheel_maximum_traversable_slope_degrees(
            platform=rover(wheel_diameter_m=diameter),
            contact_model=soil(),
            strength=STRENGTH,
            mobilization=MOBILIZATION,
            gravity_m_per_s2=LUNAR_GRAVITY,
        )
        for diameter in (0.20, 0.25, 0.35, 0.50)
    ]
    assert limits == sorted(limits)
    assert limits[-1] - limits[0] > 5.0


# --- cost of transport


def test_a_wheel_pays_no_swing() -> None:
    cost = rolling_cost_of_transport(
        platform=rover(),
        contact_model=soil(),
        strength=STRENGTH,
        mobilization=MOBILIZATION,
        gravity_m_per_s2=LUNAR_GRAVITY,
        slope_degrees=np.array([0.0, 10.0]),
        slip_ratio=np.array([0.1, 0.3]),
    )
    assert bool((cost.swing_J_per_m == 0.0).all())


def test_the_terms_sum_to_the_standard_result() -> None:
    # Total energy per metre advanced is (resistance + climb)/(1 - slip). The
    # decomposition exists so the mechanism is visible, but it must add up.
    platform, model = rover(), soil()
    slope, slip = 8.0, 0.24
    cost = rolling_cost_of_transport(
        platform=platform,
        contact_model=model,
        strength=STRENGTH,
        mobilization=MOBILIZATION,
        gravity_m_per_s2=LUNAR_GRAVITY,
        slope_degrees=slope,
        slip_ratio=slip,
    )
    driven = float(cost.compaction_J_per_m) + float(cost.gravitational_J_per_m)
    assert float(cost.total_J_per_m) == pytest.approx(driven / (1.0 - slip))


def test_compaction_is_a_resistance_and_does_not_vanish_on_the_level() -> None:
    # For a foot, compaction work is paid downward and never enters the traction
    # balance. For a wheel it is the whole of the level-ground cost.
    cost = rolling_cost_of_transport(
        platform=rover(),
        contact_model=soil(),
        strength=STRENGTH,
        mobilization=MOBILIZATION,
        gravity_m_per_s2=LUNAR_GRAVITY,
        slope_degrees=0.0,
        slip_ratio=0.0,
    )
    assert float(cost.gravitational_J_per_m) == pytest.approx(0.0)
    assert float(cost.compaction_J_per_m) > 0.0
    assert float(cost.total_J_per_m) == pytest.approx(
        float(cost.compaction_J_per_m)
    )


def test_a_spinning_wheel_is_refused() -> None:
    with pytest.raises(ValueError, match=r"slip_ratio must lie in \[0, 1\)"):
        rolling_cost_of_transport(
            platform=rover(),
            contact_model=soil(),
            strength=STRENGTH,
            mobilization=MOBILIZATION,
            gravity_m_per_s2=LUNAR_GRAVITY,
            slope_degrees=0.0,
            slip_ratio=1.0,
        )


# --- the body


def test_the_rover_tips_where_the_quadruped_tips() -> None:
    # Chosen, not discovered: the wheelbase is matched to the quadruped's
    # support polygon so that the comparison isolates locomotion. Pinned here so
    # that changing one platform's geometry without the other is caught.
    assert rover().tipping_slope_degrees == pytest.approx(39.8055710922652)


def test_an_odd_wheel_count_is_refused() -> None:
    with pytest.raises(ValueError, match="even"):
        rover(wheels=5)


def test_the_masses_match_the_legged_platform() -> None:
    assert rover().total_mass_kg == pytest.approx(50.0)


# --- the config, and the loader that had to be added for it


def test_the_rover_config_loads_and_names_itself_an_assumption() -> None:
    definition = load_wheeled_platform(NOMINAL_ROVER)
    assert definition.id == "nominal-rover"
    assert definition.morphology == "wheeled"
    assert definition.basis == "assumed"


def test_the_config_matches_the_quadruped_where_it_claims_to() -> None:
    # The comparison is only about locomotion if the bodies agree everywhere
    # else. If either file drifts, this catches it before a study reports a
    # difference that is really a mass or a support polygon.
    rover_platform = load_wheeled_platform(NOMINAL_ROVER).platform
    quadruped = load_platform(NOMINAL_QUADRUPED).platform
    assert rover_platform.total_mass_kg == pytest.approx(quadruped.total_mass_kg)
    assert rover_platform.tipping_slope_degrees == pytest.approx(
        math.degrees(
            math.atan2(
                max(foot.x_m for foot in quadruped.footprint),
                quadruped.center_of_mass_height_m,
            )
        )
    )
    assert rover_platform.nominal_speed_m_per_s == pytest.approx(
        quadruped.nominal_speed_m_per_s
    )


def test_a_legged_file_is_refused_by_the_wheeled_loader() -> None:
    with pytest.raises(PlatformFileError, match="declares a footprint"):
        load_wheeled_platform(NOMINAL_QUADRUPED)


def test_a_renamed_rover_parameter_fails_at_construction(tmp_path: Path) -> None:
    text = NOMINAL_ROVER.read_text(encoding="utf-8").replace(
        "wheel_width_m    = 0.10", "wheel_width = 0.10"
    )
    path = tmp_path / "renamed.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(PlatformFileError, match="does not match the WheeledPlatform"):
        load_wheeled_platform(path)
