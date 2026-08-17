# SPDX-License-Identifier: Apache-2.0
#
# Tests for eclipse.platform and its loader.
#
# The platform is read from configs/platforms/nominal-quadruped.toml rather
# than built here, so a changed config fails these tests instead of silently
# changing a locomotion result.
#
# Two properties get more attention than the arithmetic. The first is the seam:
# the contact layer must remain unable to tell what is standing on it, and that
# is asserted rather than assumed. The second is the traction margin on a slope
# at its angle of repose, where the frictional term cancels exactly and the
# whole margin is cohesive -- a closed-form identity, so it is tested as one.

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from eclipse.io.platform import PlatformFileError, load_platform
from eclipse.platform import (
    Platform,
    equilibrium_slip_ratio,
    maximum_traversable_slope_degrees,
    swing_work_per_meter,
    swing_work_per_stride,
    traction_balance,
)
from eclipse.terramechanics import JanosiHanamotoModel, MohrCoulombModel

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
NOMINAL_QUADRUPED = (
    REPOSITORY_ROOT / "configs" / "platforms" / "nominal-quadruped.toml"
)

EARTH_GRAVITY = 9.81
LUNAR_GRAVITY = 1.62


@pytest.fixture(scope="module")
def quadruped() -> Platform:
    return load_platform(NOMINAL_QUADRUPED).platform


@pytest.fixture(scope="module")
def lunar_strength() -> MohrCoulombModel:
    return MohrCoulombModel(cohesion=0.52, friction_angle_degrees=42.0)


@pytest.fixture(scope="module")
def lunar_mobilization() -> JanosiHanamotoModel:
    return JanosiHanamotoModel(shear_deformation_modulus=0.018)


# --- loading


def test_the_config_loads_and_names_itself_an_assumption() -> None:
    definition = load_platform(NOMINAL_QUADRUPED)
    assert definition.id == "nominal-quadruped"
    assert definition.basis == "assumed", (
        "a platform file records assumptions; if it ever claims another basis "
        "that claim needs evidence behind it"
    )


def test_a_renamed_parameter_fails_at_construction(tmp_path: Path) -> None:
    text = NOMINAL_QUADRUPED.read_text(encoding="utf-8").replace(
        "leg_mass_kg  = 1.5", "leg_mass = 1.5"
    )
    path = tmp_path / "renamed.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(PlatformFileError, match="does not match the Platform"):
        load_platform(path)


def test_an_unsupported_schema_version_is_refused(tmp_path: Path) -> None:
    text = NOMINAL_QUADRUPED.read_text(encoding="utf-8").replace(
        "schema_version = 1", "schema_version = 2"
    )
    path = tmp_path / "future.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(PlatformFileError, match="not supported"):
        load_platform(path)


# --- the seam


def test_the_platform_hands_down_a_patch_that_knows_nothing_about_it(
    quadruped: Platform,
) -> None:
    patch = quadruped.contact_patch
    assert set(type(patch).__dataclass_fields__) == {"half_width_m", "area_m2"}, (
        "the contact patch is the seam between morphology and soil; if adding a "
        "platform widened it, the seam was in the wrong place"
    )


def test_the_contact_layer_does_not_import_the_platform_layer() -> None:
    # The dependency must run one way. A platform builds a patch and hands it
    # down; the physics never reaches back up. Checked against the import graph
    # rather than the text, since both modules legitimately use the word.
    for module in ("terramechanics", "mobility"):
        tree = ast.parse(
            (REPOSITORY_ROOT / "python" / "eclipse" / f"{module}.py").read_text(
                encoding="utf-8"
            )
        )
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
        assert "eclipse.platform" not in imported, (
            f"eclipse.{module} imports the platform layer, which inverts the "
            "dependency the architecture rests on"
        )


def test_the_declared_patch_area_matches_a_circular_pad(quadruped: Platform) -> None:
    assert quadruped.foot_contact_area_m2 == pytest.approx(
        math.pi * quadruped.foot_half_width_m**2, rel=1e-12
    )


# --- swing work


def test_the_inertial_term_does_not_depend_on_gravity(quadruped: Platform) -> None:
    on_earth = swing_work_per_stride(
        platform=quadruped, gravity_m_per_s2=EARTH_GRAVITY
    )
    on_moon = swing_work_per_stride(
        platform=quadruped, gravity_m_per_s2=LUNAR_GRAVITY
    )
    assert float(on_earth.inertial_J) == pytest.approx(float(on_moon.inertial_J))


def test_the_clearance_term_scales_exactly_with_gravity(
    quadruped: Platform,
) -> None:
    on_earth = swing_work_per_stride(
        platform=quadruped, gravity_m_per_s2=EARTH_GRAVITY
    )
    on_moon = swing_work_per_stride(
        platform=quadruped, gravity_m_per_s2=LUNAR_GRAVITY
    )
    assert float(on_moon.clearance_J) / float(on_earth.clearance_J) == pytest.approx(
        LUNAR_GRAVITY / EARTH_GRAVITY
    )


def test_swing_work_matches_the_closed_form(quadruped: Platform) -> None:
    work = swing_work_per_stride(platform=quadruped, gravity_m_per_s2=LUNAR_GRAVITY)

    sweep = 2.0 * math.asin(
        quadruped.stride_length_m / (2.0 * quadruped.leg_length_m)
    )
    swing_time = (1.0 - quadruped.feet_in_stance / quadruped.legs) * (
        quadruped.stride_length_m / quadruped.nominal_speed_m_per_s
    )
    inertia = quadruped.leg_mass_kg * quadruped.leg_length_m**2 / 3.0
    peak = 2.0 * sweep / swing_time

    assert float(work.inertial_J) == pytest.approx(0.5 * inertia * peak**2)
    assert float(work.clearance_J) == pytest.approx(
        quadruped.leg_mass_kg * LUNAR_GRAVITY * quadruped.foot_clearance_m / 2.0
    )


def test_swing_cost_is_dominated_by_the_term_gravity_cannot_touch(
    quadruped: Platform,
) -> None:
    on_moon = swing_work_per_stride(
        platform=quadruped, gravity_m_per_s2=LUNAR_GRAVITY
    )
    assert float(on_moon.inertial_fraction) > 0.9, (
        "at one sixth gravity the clearance term nearly vanishes, so swing cost "
        "is almost entirely inertial and almost entirely gravity-independent"
    )


def test_swing_work_per_meter_rises_with_slip(quadruped: Platform) -> None:
    ratios = np.array([0.0, 0.1, 0.3, 0.5])
    totals = swing_work_per_meter(
        platform=quadruped, gravity_m_per_s2=LUNAR_GRAVITY, slip_ratio=ratios
    ).total_J
    assert np.all(np.diff(totals) > 0.0)


def test_swing_work_per_meter_goes_as_the_square_of_speed(
    quadruped: Platform,
) -> None:
    # The inertial term goes as the square of angular velocity and the number of
    # swings per meter does not depend on speed, so this model always prefers
    # walking slower. Real platforms do not, and that limit is recorded.
    def at_speed(speed: float) -> float:
        faster = Platform(
            **{
                **{
                    field: getattr(quadruped, field)
                    for field in Platform.__dataclass_fields__
                },
                "nominal_speed_m_per_s": speed,
            }
        )
        return float(
            swing_work_per_meter(
                platform=faster, gravity_m_per_s2=LUNAR_GRAVITY, slip_ratio=0.0
            ).inertial_J
        )

    assert at_speed(1.0) / at_speed(0.5) == pytest.approx(4.0, rel=1e-9)


@pytest.mark.parametrize("ratio", [1.0, 1.5, -0.1, math.nan])
def test_a_platform_making_no_progress_is_refused(
    quadruped: Platform, ratio: float
) -> None:
    with pytest.raises(ValueError, match=r"lie in \[0, 1\)"):
        swing_work_per_meter(
            platform=quadruped, gravity_m_per_s2=LUNAR_GRAVITY, slip_ratio=ratio
        )


def test_a_platform_with_every_leg_down_cannot_swing(quadruped: Platform) -> None:
    standing = Platform(
        **{
            **{
                field: getattr(quadruped, field)
                for field in Platform.__dataclass_fields__
            },
            "feet_in_stance": quadruped.legs,
        }
    )
    with pytest.raises(ValueError, match="no leg swings"):
        swing_work_per_stride(platform=standing, gravity_m_per_s2=LUNAR_GRAVITY)


# --- traction and slip


def test_level_ground_demands_no_traction_and_produces_no_slip(
    quadruped: Platform,
    lunar_strength: MohrCoulombModel,
    lunar_mobilization: JanosiHanamotoModel,
) -> None:
    # Exactly zero, and a lower bound rather than a prediction: real walking
    # slips on the flat through within-stride acceleration, control error and
    # foot placement, none of which is modelled here.
    slip = float(
        equilibrium_slip_ratio(
            platform=quadruped,
            strength=lunar_strength,
            mobilization=lunar_mobilization,
            gravity_m_per_s2=LUNAR_GRAVITY,
            slope_degrees=0.0,
        )
    )
    assert slip == 0.0
    assert math.copysign(1.0, slip) > 0.0, (
        "a negative zero here is harmless arithmetically and prints as -0.0000 "
        "in every report downstream"
    )


def test_slip_keeps_its_leading_digits_on_gentle_slopes(
    quadruped: Platform,
    lunar_strength: MohrCoulombModel,
    lunar_mobilization: JanosiHanamotoModel,
) -> None:
    # At small demand the slide is K times the mobilized fraction to first
    # order. Computing log(1 - f) instead of log1p(-f) would lose the leading
    # digits exactly here, where the answer is otherwise most trustworthy.
    slope = 1.0e-6
    balance = traction_balance(
        platform=quadruped,
        strength=lunar_strength,
        gravity_m_per_s2=LUNAR_GRAVITY,
        slope_degrees=slope,
    )
    slip = float(
        equilibrium_slip_ratio(
            platform=quadruped,
            strength=lunar_strength,
            mobilization=lunar_mobilization,
            gravity_m_per_s2=LUNAR_GRAVITY,
            slope_degrees=slope,
        )
    )
    linear = (
        lunar_mobilization.shear_deformation_modulus
        * float(balance.mobilized_fraction)
        / quadruped.stride_length_m
    )
    assert slip == pytest.approx(linear, rel=1e-9)


def test_slip_rises_with_slope_and_diverges_at_the_traction_limit(
    quadruped: Platform,
    lunar_strength: MohrCoulombModel,
    lunar_mobilization: JanosiHanamotoModel,
) -> None:
    limit = maximum_traversable_slope_degrees(
        platform=quadruped, strength=lunar_strength, gravity_m_per_s2=LUNAR_GRAVITY
    )
    slopes = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    slip = equilibrium_slip_ratio(
        platform=quadruped,
        strength=lunar_strength,
        mobilization=lunar_mobilization,
        gravity_m_per_s2=LUNAR_GRAVITY,
        slope_degrees=slopes,
    )
    assert np.all(np.diff(slip) > 0.0)
    assert np.all(np.isfinite(slip))

    beyond = equilibrium_slip_ratio(
        platform=quadruped,
        strength=lunar_strength,
        mobilization=lunar_mobilization,
        gravity_m_per_s2=LUNAR_GRAVITY,
        slope_degrees=limit + 0.5,
    )
    assert not np.isfinite(beyond), (
        "past the traction limit there is no equilibrium; the platform "
        "accelerates downhill rather than walking with a large slip"
    )


def test_slip_inverts_the_mobilization_curve(
    quadruped: Platform,
    lunar_strength: MohrCoulombModel,
    lunar_mobilization: JanosiHanamotoModel,
) -> None:
    slope = 25.0
    slip = float(
        equilibrium_slip_ratio(
            platform=quadruped,
            strength=lunar_strength,
            mobilization=lunar_mobilization,
            gravity_m_per_s2=LUNAR_GRAVITY,
            slope_degrees=slope,
        )
    )
    balance = traction_balance(
        platform=quadruped,
        strength=lunar_strength,
        gravity_m_per_s2=LUNAR_GRAVITY,
        slope_degrees=slope,
    )
    developed = float(
        lunar_mobilization.mobilized_fraction(
            shear_displacement=slip * quadruped.stride_length_m
        )
    )
    assert developed == pytest.approx(float(balance.mobilized_fraction), rel=1e-12), (
        "the slip that is reported must be the slip that develops exactly the "
        "demanded traction, or the equilibrium is not one"
    )


def test_descending_demands_the_same_traction_as_climbing(
    quadruped: Platform,
    lunar_strength: MohrCoulombModel,
    lunar_mobilization: JanosiHanamotoModel,
) -> None:
    uphill, downhill = (
        equilibrium_slip_ratio(
            platform=quadruped,
            strength=lunar_strength,
            mobilization=lunar_mobilization,
            gravity_m_per_s2=LUNAR_GRAVITY,
            slope_degrees=slope,
        )
        for slope in (20.0, -20.0)
    )
    assert float(uphill) == pytest.approx(float(downhill))


# --- the repose identity


@pytest.mark.parametrize("repose_degrees", [30.0, 32.5, 35.0])
@pytest.mark.parametrize("gravity", [EARTH_GRAVITY, LUNAR_GRAVITY])
def test_on_a_repose_slope_the_whole_traction_margin_is_cohesive(
    quadruped: Platform, repose_degrees: float, gravity: float
) -> None:
    # A slope standing at its angle of repose has mobilized its friction angle
    # exactly, so cos(theta)*tan(phi) - sin(theta) vanishes identically and the
    # frictional term contributes nothing to the margin. Cohesion, which Day 2
    # showed is a few percent of shear strength and badly determined, is then
    # the entire reserve. Strength and margin are different questions.
    at_repose = MohrCoulombModel(
        cohesion=0.52, friction_angle_degrees=repose_degrees
    )
    balance = traction_balance(
        platform=quadruped,
        strength=at_repose,
        gravity_m_per_s2=gravity,
        slope_degrees=repose_degrees,
    )
    assert float(balance.margin_N) == pytest.approx(
        float(balance.cohesive_capacity_N), rel=1e-12
    )
    assert float(balance.cohesive_share_of_margin) == pytest.approx(1.0, rel=1e-12)


def test_a_cohesionless_soil_cannot_stand_on_its_own_repose_slope(
    quadruped: Platform, lunar_mobilization: JanosiHanamotoModel
) -> None:
    cohesionless = MohrCoulombModel(cohesion=0.0, friction_angle_degrees=35.0)
    limit = maximum_traversable_slope_degrees(
        platform=quadruped, strength=cohesionless, gravity_m_per_s2=LUNAR_GRAVITY
    )
    assert limit == pytest.approx(35.0), (
        "with no cohesion the traction limit is exactly the friction angle, so "
        "a slope at repose is walkable only in the limit and never in practice"
    )


# --- the traction limit


@pytest.mark.parametrize("gravity", [EARTH_GRAVITY, 3.71, LUNAR_GRAVITY])
def test_the_traction_limit_exceeds_the_friction_angle_by_the_cohesive_reserve(
    quadruped: Platform, lunar_strength: MohrCoulombModel, gravity: float
) -> None:
    limit = maximum_traversable_slope_degrees(
        platform=quadruped, strength=lunar_strength, gravity_m_per_s2=gravity
    )
    assert limit > lunar_strength.friction_angle_degrees
    assert limit < 90.0


def test_cohesion_buys_more_slope_at_lower_gravity(
    quadruped: Platform, lunar_strength: MohrCoulombModel
) -> None:
    gained = {
        gravity: maximum_traversable_slope_degrees(
            platform=quadruped, strength=lunar_strength, gravity_m_per_s2=gravity
        )
        - lunar_strength.friction_angle_degrees
        for gravity in (EARTH_GRAVITY, LUNAR_GRAVITY)
    }
    assert gained[LUNAR_GRAVITY] > gained[EARTH_GRAVITY]
    assert gained[LUNAR_GRAVITY] / gained[EARTH_GRAVITY] == pytest.approx(
        EARTH_GRAVITY / LUNAR_GRAVITY, rel=0.02
    ), "the cohesive reserve goes as one over gravity, so the gain should too"


def test_a_bigger_foot_and_more_feet_both_buy_slope(
    quadruped: Platform, lunar_strength: MohrCoulombModel
) -> None:
    def limit(half_width: float, feet: int) -> float:
        variant = Platform(
            **{
                **{
                    field: getattr(quadruped, field)
                    for field in Platform.__dataclass_fields__
                },
                "foot_half_width_m": half_width,
                "foot_contact_area_m2": math.pi * half_width**2,
                "feet_in_stance": feet,
            }
        )
        return maximum_traversable_slope_degrees(
            platform=variant, strength=lunar_strength, gravity_m_per_s2=LUNAR_GRAVITY
        )

    baseline = limit(quadruped.foot_half_width_m, quadruped.feet_in_stance)
    assert limit(0.050, quadruped.feet_in_stance) > baseline
    assert limit(quadruped.foot_half_width_m, 4) > baseline


# --- guards


@pytest.mark.parametrize(
    "field,bad",
    [
        ("body_mass_kg", 0.0),
        ("leg_mass_kg", -1.0),
        ("leg_length_m", math.nan),
        ("nominal_speed_m_per_s", math.inf),
    ],
)
def test_unphysical_platform_quantities_are_refused(
    quadruped: Platform, field: str, bad: float
) -> None:
    with pytest.raises(ValueError, match="must be finite and positive"):
        Platform(
            **{
                **{
                    name: getattr(quadruped, name)
                    for name in Platform.__dataclass_fields__
                },
                field: bad,
            }
        )


def test_more_feet_in_stance_than_legs_is_refused(quadruped: Platform) -> None:
    with pytest.raises(ValueError, match="between one and legs"):
        Platform(
            **{
                **{
                    name: getattr(quadruped, name)
                    for name in Platform.__dataclass_fields__
                },
                "feet_in_stance": quadruped.legs + 1,
            }
        )


def test_a_stride_the_hip_cannot_sweep_is_refused(quadruped: Platform) -> None:
    with pytest.raises(ValueError, match="cannot sweep"):
        Platform(
            **{
                **{
                    name: getattr(quadruped, name)
                    for name in Platform.__dataclass_fields__
                },
                "stride_length_m": 2.5 * quadruped.leg_length_m,
            }
        )
