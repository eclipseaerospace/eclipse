# SPDX-License-Identifier: Apache-2.0
#
# Tests for eclipse.mobility.
#
# Soil parameters are read from data/soils/lunar-intercrater.toml rather than
# written here, so a changed transcription fails these tests rather than
# silently changing a locomotion result.
#
# Both work integrals have closed forms, and both are checked against numerical
# quadrature of the underlying stress law rather than against a restatement of
# the same algebra. The compaction check deliberately uses a non-unit sinkage
# exponent, since at an exponent of one the trapezoid rule is exact and would
# confirm nothing.
#
# The gravity-scaling tests are the substantive ones. Each term is a power of
# gravity once normalized, the exponents are derivable in closed form, and they
# are what decides whether lunar cost of transport is dominated by the soil or
# by the legs.

from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eclipse.mobility import (
    ContactPatch,
    compaction_work_per_footfall,
    cost_of_transport,
    shear_work_per_footfall,
    slip_displacement,
)
from eclipse.terramechanics import (
    BekkerModel,
    JanosiHanamotoModel,
    MohrCoulombModel,
    shear_stress,
)

LUNAR_SOIL = (
    Path(__file__).resolve().parents[2] / "data" / "soils" / "lunar-intercrater.toml"
)

CENTIMETERS_PER_METER = 100.0

EARTH_GRAVITY = 9.81
MARS_GRAVITY = 3.71
LUNAR_GRAVITY = 1.62

WALKING_MASS_KG = 50.0
STRIDE_M = 0.30


def _lunar() -> dict[str, Any]:
    table = tomllib.loads(LUNAR_SOIL.read_text(encoding="utf-8"))
    dataset = table["dataset"][0]
    return {
        "bekker": next(m for m in dataset["model"] if m["id"] == "bekker"),
        "mohr_coulomb": next(
            m for m in dataset["shear_model"] if m["id"] == "mohr_coulomb"
        ),
        "janosi_hanamoto": next(
            m for m in dataset["shear_model"] if m["id"] == "janosi_hanamoto"
        ),
    }


@pytest.fixture(scope="module")
def lunar_contact() -> BekkerModel:
    parameters = _lunar()["bekker"]["parameters"]
    return BekkerModel(
        cohesive_modulus=parameters["cohesive_modulus"]["value"],
        frictional_modulus=parameters["frictional_modulus"]["value"],
        sinkage_exponent=parameters["sinkage_exponent"]["value"],
    )


@pytest.fixture(scope="module")
def lunar_strength() -> MohrCoulombModel:
    row = next(
        entry
        for entry in _lunar()["mohr_coulomb"]["by_depth"]["rows"]
        if entry["depth_range_cm"] == "0-15"
    )
    return MohrCoulombModel(
        cohesion=row["cohesion_kPa"], friction_angle_degrees=row["friction_angle_deg"]
    )


@pytest.fixture(scope="module")
def lunar_mobilization() -> JanosiHanamotoModel:
    modulus_cm = _lunar()["janosi_hanamoto"]["parameters"][
        "shear_deformation_modulus"
    ]["value"]
    return JanosiHanamotoModel(
        shear_deformation_modulus=modulus_cm / CENTIMETERS_PER_METER
    )


@pytest.fixture(scope="module")
def foot() -> ContactPatch:
    half_width = 0.030
    return ContactPatch(half_width_m=half_width, area_m2=math.pi * half_width**2)


def _walk(
    *,
    gravity: float,
    foot: ContactPatch,
    contact: BekkerModel,
    strength: MohrCoulombModel,
    mobilization: JanosiHanamotoModel,
    slope_degrees: float = 0.0,
    slip_ratio: float = 0.10,
    feet_in_stance: int = 2,
    swing_work_per_meter_J: float = 25.0,
) -> Any:
    return cost_of_transport(
        mass_kg=WALKING_MASS_KG,
        gravity_m_per_s2=gravity,
        slope_degrees=slope_degrees,
        slip_ratio=slip_ratio,
        patch=foot,
        feet_in_stance=feet_in_stance,
        stride_length_m=STRIDE_M,
        stance_length_m=STRIDE_M,
        contact_model=contact,
        strength=strength,
        mobilization=mobilization,
        swing_work_per_meter_J=swing_work_per_meter_J,
    )


# --- the contact patch


def test_normal_stress_is_load_over_area_in_kilopascals(foot: ContactPatch) -> None:
    stress = float(foot.normal_stress_kPa(normal_load_N=100.0))
    assert stress == pytest.approx(100.0 / foot.area_m2 / 1000.0)


def test_normal_stress_broadcasts(foot: ContactPatch) -> None:
    loads = np.array([10.0, 20.0, 40.0])
    stresses = foot.normal_stress_kPa(normal_load_N=loads)
    assert stresses.shape == loads.shape
    assert np.all(np.diff(stresses) > 0.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_a_patch_with_no_extent_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="must be finite and positive"):
        ContactPatch(half_width_m=bad, area_m2=1e-3)
    with pytest.raises(ValueError, match="must be finite and positive"):
        ContactPatch(half_width_m=0.02, area_m2=bad)


def test_the_patch_carries_no_notion_of_what_is_standing_on_it() -> None:
    fields = set(ContactPatch.__dataclass_fields__)
    assert fields == {"half_width_m", "area_m2"}, (
        "the contact patch is the seam between morphology and soil; a field "
        "naming a leg, a foot or a robot here means the seam has leaked"
    )


# --- slip


def test_slip_displacement_is_the_slipped_fraction_of_the_stance() -> None:
    assert float(
        slip_displacement(stance_length_m=0.30, slip_ratio=0.10)
    ) == pytest.approx(0.030)


def test_no_slip_slides_nothing() -> None:
    assert float(slip_displacement(stance_length_m=0.30, slip_ratio=0.0)) == 0.0


@pytest.mark.parametrize("ratio", [1.0, 1.5, -0.1, math.nan, math.inf])
def test_a_slip_ratio_outside_the_half_open_unit_interval_is_refused(
    ratio: float,
) -> None:
    with pytest.raises(ValueError, match=r"lie in \[0, 1\)"):
        slip_displacement(stance_length_m=0.30, slip_ratio=ratio)


def test_the_slip_error_names_the_first_offending_value() -> None:
    with pytest.raises(ValueError, match="1.4"):
        slip_displacement(
            stance_length_m=0.30, slip_ratio=np.array([0.1, 0.2, 1.4, 2.0])
        )


# --- shear work


def test_shear_work_matches_quadrature_of_the_mobilization_curve(
    foot: ContactPatch,
    lunar_strength: MohrCoulombModel,
    lunar_mobilization: JanosiHanamotoModel,
) -> None:
    load_N = 40.0
    slid_m = 0.030
    closed_form = float(
        shear_work_per_footfall(
            patch=foot,
            strength=lunar_strength,
            mobilization=lunar_mobilization,
            normal_load_N=load_N,
            slip_displacement_m=slid_m,
        )
    )

    distance = np.linspace(0.0, slid_m, 200_001)
    carried_kPa = shear_stress(
        strength=lunar_strength,
        mobilization=lunar_mobilization,
        normal_stress=foot.normal_stress_kPa(normal_load_N=load_N),
        shear_displacement=distance,
    )
    quadrature = float(np.trapezoid(carried_kPa * 1000.0 * foot.area_m2, distance))

    assert closed_form == pytest.approx(quadrature, rel=1e-8)


def test_a_foot_that_does_not_slip_does_no_shear_work(
    foot: ContactPatch,
    lunar_strength: MohrCoulombModel,
    lunar_mobilization: JanosiHanamotoModel,
) -> None:
    assert (
        float(
            shear_work_per_footfall(
                patch=foot,
                strength=lunar_strength,
                mobilization=lunar_mobilization,
                normal_load_N=40.0,
                slip_displacement_m=0.0,
            )
        )
        == 0.0
    )


def test_shear_work_is_quadratic_in_slip_well_inside_the_mobilization_length(
    foot: ContactPatch,
    lunar_strength: MohrCoulombModel,
    lunar_mobilization: JanosiHanamotoModel,
) -> None:
    # Below K the soil has barely gripped, so the work goes as the square of the
    # slid distance rather than its first power. Doubling a small slip therefore
    # roughly quadruples the cost, which is why slip is worth suppressing early.
    modulus = lunar_mobilization.shear_deformation_modulus
    small = modulus / 50.0

    def work(slid: float) -> float:
        return float(
            shear_work_per_footfall(
                patch=foot,
                strength=lunar_strength,
                mobilization=lunar_mobilization,
                normal_load_N=40.0,
                slip_displacement_m=slid,
            )
        )

    assert work(2.0 * small) / work(small) == pytest.approx(4.0, rel=0.05)


def test_shear_work_rises_with_slip_and_with_load(
    foot: ContactPatch,
    lunar_strength: MohrCoulombModel,
    lunar_mobilization: JanosiHanamotoModel,
) -> None:
    slid = np.array([0.0, 0.005, 0.018, 0.054, 0.100])
    by_slip = shear_work_per_footfall(
        patch=foot,
        strength=lunar_strength,
        mobilization=lunar_mobilization,
        normal_load_N=40.0,
        slip_displacement_m=slid,
    )
    assert np.all(np.diff(by_slip) > 0.0)

    by_load = shear_work_per_footfall(
        patch=foot,
        strength=lunar_strength,
        mobilization=lunar_mobilization,
        normal_load_N=np.array([10.0, 40.0, 200.0]),
        slip_displacement_m=0.030,
    )
    assert np.all(np.diff(by_load) > 0.0)


# --- compaction work


@pytest.mark.parametrize("exponent", [0.7, 1.0, 1.3])
def test_compaction_work_matches_quadrature_of_the_pressure_law(
    foot: ContactPatch, exponent: float
) -> None:
    contact = BekkerModel(
        cohesive_modulus=1.4, frictional_modulus=820.0, sinkage_exponent=exponent
    )
    depth_m = 0.012
    closed_form = float(
        compaction_work_per_footfall(
            patch=foot, contact_model=contact, sinkage_m=depth_m
        )
    )

    depths = np.linspace(0.0, depth_m, 400_001)
    pressure_kPa = contact.pressure(
        sinkage=depths, contact_half_width=foot.half_width_m
    )
    quadrature = float(np.trapezoid(pressure_kPa * 1000.0 * foot.area_m2, depths))

    assert closed_form == pytest.approx(quadrature, rel=1e-6)


def test_compaction_work_reads_the_exponent_off_the_model(
    foot: ContactPatch, lunar_contact: BekkerModel
) -> None:
    steeper = BekkerModel(
        cohesive_modulus=lunar_contact.cohesive_modulus,
        frictional_modulus=lunar_contact.frictional_modulus,
        sinkage_exponent=lunar_contact.sinkage_exponent * 1.5,
    )
    at_published = float(
        compaction_work_per_footfall(
            patch=foot, contact_model=lunar_contact, sinkage_m=0.010
        )
    )
    at_steeper = float(
        compaction_work_per_footfall(
            patch=foot, contact_model=steeper, sinkage_m=0.010
        )
    )
    assert at_published != at_steeper, (
        "the exponent must come from the model; a caller cannot be trusted to "
        "pass one that matches the curve the sinkage came from"
    )


def test_pressing_no_depth_costs_nothing(
    foot: ContactPatch, lunar_contact: BekkerModel
) -> None:
    assert (
        float(
            compaction_work_per_footfall(
                patch=foot, contact_model=lunar_contact, sinkage_m=0.0
            )
        )
        == 0.0
    )


# --- the decomposition


def test_the_terms_sum_to_the_total(
    foot: ContactPatch,
    lunar_contact: BekkerModel,
    lunar_strength: MohrCoulombModel,
    lunar_mobilization: JanosiHanamotoModel,
) -> None:
    walk = _walk(
        gravity=LUNAR_GRAVITY,
        foot=foot,
        contact=lunar_contact,
        strength=lunar_strength,
        mobilization=lunar_mobilization,
        slope_degrees=12.0,
    )
    parts = sum(
        float(getattr(walk, f"{term}_J_per_m"))
        for term in ("gravitational", "shear", "compaction", "swing")
    )
    assert parts == pytest.approx(float(walk.total_J_per_m))
    shares = sum(
        float(walk.fraction(term))
        for term in ("gravitational", "shear", "compaction", "swing")
    )
    assert shares == pytest.approx(1.0)


def test_the_dimensionless_cost_is_the_total_over_weight(
    foot: ContactPatch,
    lunar_contact: BekkerModel,
    lunar_strength: MohrCoulombModel,
    lunar_mobilization: JanosiHanamotoModel,
) -> None:
    walk = _walk(
        gravity=LUNAR_GRAVITY,
        foot=foot,
        contact=lunar_contact,
        strength=lunar_strength,
        mobilization=lunar_mobilization,
    )
    assert float(walk.dimensionless) == pytest.approx(
        float(walk.total_J_per_m) / (WALKING_MASS_KG * LUNAR_GRAVITY)
    )


def test_flat_ground_costs_nothing_gravitationally(
    foot: ContactPatch,
    lunar_contact: BekkerModel,
    lunar_strength: MohrCoulombModel,
    lunar_mobilization: JanosiHanamotoModel,
) -> None:
    walk = _walk(
        gravity=LUNAR_GRAVITY,
        foot=foot,
        contact=lunar_contact,
        strength=lunar_strength,
        mobilization=lunar_mobilization,
        slope_degrees=0.0,
    )
    assert float(walk.gravitational_J_per_m) == pytest.approx(0.0, abs=1e-12)


def test_the_gravitational_term_is_the_sine_of_the_slope_and_nothing_else(
    foot: ContactPatch,
    lunar_contact: BekkerModel,
    lunar_strength: MohrCoulombModel,
    lunar_mobilization: JanosiHanamotoModel,
) -> None:
    slope = 25.0
    walk = _walk(
        gravity=LUNAR_GRAVITY,
        foot=foot,
        contact=lunar_contact,
        strength=lunar_strength,
        mobilization=lunar_mobilization,
        slope_degrees=slope,
    )
    assert float(walk.gravitational_J_per_m) == pytest.approx(
        WALKING_MASS_KG * LUNAR_GRAVITY * math.sin(math.radians(slope))
    )


def test_descending_refunds_gravity_and_nothing_else(
    foot: ContactPatch,
    lunar_contact: BekkerModel,
    lunar_strength: MohrCoulombModel,
    lunar_mobilization: JanosiHanamotoModel,
) -> None:
    # Compaction, shear and swing are strictly positive and indifferent to
    # direction: a crater descent does not hand back the work of pressing soil
    # down or of cycling a leg.
    downhill = _walk(
        gravity=LUNAR_GRAVITY,
        foot=foot,
        contact=lunar_contact,
        strength=lunar_strength,
        mobilization=lunar_mobilization,
        slope_degrees=-20.0,
    )
    uphill = _walk(
        gravity=LUNAR_GRAVITY,
        foot=foot,
        contact=lunar_contact,
        strength=lunar_strength,
        mobilization=lunar_mobilization,
        slope_degrees=20.0,
    )

    assert float(downhill.gravitational_J_per_m) < 0.0
    for term in ("shear", "compaction", "swing"):
        assert float(getattr(downhill, f"{term}_J_per_m")) > 0.0
        assert float(getattr(downhill, f"{term}_J_per_m")) == pytest.approx(
            float(getattr(uphill, f"{term}_J_per_m"))
        )


def test_slope_enters_the_soil_terms_only_through_the_normal_component(
    foot: ContactPatch,
    lunar_contact: BekkerModel,
    lunar_strength: MohrCoulombModel,
    lunar_mobilization: JanosiHanamotoModel,
) -> None:
    flat = _walk(
        gravity=LUNAR_GRAVITY,
        foot=foot,
        contact=lunar_contact,
        strength=lunar_strength,
        mobilization=lunar_mobilization,
        slope_degrees=0.0,
    )
    tilted = _walk(
        gravity=LUNAR_GRAVITY,
        foot=foot,
        contact=lunar_contact,
        strength=lunar_strength,
        mobilization=lunar_mobilization,
        slope_degrees=30.0,
    )
    assert float(tilted.compaction_J_per_m) < float(flat.compaction_J_per_m), (
        "a tilted platform presses less hard into the slope, so it sinks less"
    )


def test_asking_for_a_term_that_does_not_exist_is_refused(
    foot: ContactPatch,
    lunar_contact: BekkerModel,
    lunar_strength: MohrCoulombModel,
    lunar_mobilization: JanosiHanamotoModel,
) -> None:
    walk = _walk(
        gravity=LUNAR_GRAVITY,
        foot=foot,
        contact=lunar_contact,
        strength=lunar_strength,
        mobilization=lunar_mobilization,
    )
    with pytest.raises(ValueError, match="no term 'thermal'"):
        walk.fraction("thermal")


def test_the_sweep_broadcasts_over_slope_and_slip(
    foot: ContactPatch,
    lunar_contact: BekkerModel,
    lunar_strength: MohrCoulombModel,
    lunar_mobilization: JanosiHanamotoModel,
) -> None:
    slopes = np.linspace(0.0, 30.0, 7)
    walk = cost_of_transport(
        mass_kg=WALKING_MASS_KG,
        gravity_m_per_s2=LUNAR_GRAVITY,
        slope_degrees=slopes,
        slip_ratio=0.10,
        patch=foot,
        feet_in_stance=2,
        stride_length_m=STRIDE_M,
        stance_length_m=STRIDE_M,
        contact_model=lunar_contact,
        strength=lunar_strength,
        mobilization=lunar_mobilization,
        swing_work_per_meter_J=25.0,
    )
    for term in ("gravitational", "shear", "compaction", "swing"):
        assert getattr(walk, f"{term}_J_per_m").shape == slopes.shape
    assert np.all(np.diff(walk.total_J_per_m) > 0.0)


# --- how each term scales with gravity


def _gravity_exponent(quantity: Any, *, gravity: float = LUNAR_GRAVITY) -> float:
    step = 1e-4
    low, high = gravity * (1.0 - step), gravity * (1.0 + step)
    return (math.log(quantity(high)) - math.log(quantity(low))) / (
        math.log(high) - math.log(low)
    )


def test_compaction_cost_collapses_under_reduced_gravity(
    foot: ContactPatch,
    lunar_contact: BekkerModel,
    lunar_strength: MohrCoulombModel,
    lunar_mobilization: JanosiHanamotoModel,
) -> None:
    # A lighter foot sinks less, and the work of pressing goes as depth to the
    # exponent plus one. Normalized by weight the compaction cost therefore goes
    # as gravity to the one over the sinkage exponent, which is positive for
    # every physical exponent: the soil always gets cheaper faster than the
    # weight falls. This is the opposite of the usual telling and it is what the
    # decomposition is for.
    def normalized(gravity: float) -> float:
        walk = _walk(
            gravity=gravity,
            foot=foot,
            contact=lunar_contact,
            strength=lunar_strength,
            mobilization=lunar_mobilization,
        )
        return float(walk.compaction_J_per_m) / (WALKING_MASS_KG * gravity)

    expected = 1.0 / lunar_contact.sinkage_exponent
    assert _gravity_exponent(normalized) == pytest.approx(expected, rel=1e-6)
    assert normalized(LUNAR_GRAVITY) < normalized(EARTH_GRAVITY)


def test_swing_dominates_the_terms_that_rise_under_reduced_gravity(
    foot: ContactPatch,
    lunar_contact: BekkerModel,
    lunar_strength: MohrCoulombModel,
    lunar_mobilization: JanosiHanamotoModel,
) -> None:
    def normalized(term: str):  # type: ignore[no-untyped-def]
        def evaluate(gravity: float) -> float:
            walk = _walk(
                gravity=gravity,
                foot=foot,
                contact=lunar_contact,
                strength=lunar_strength,
                mobilization=lunar_mobilization,
                slope_degrees=10.0,
            )
            return abs(float(getattr(walk, f"{term}_J_per_m"))) / (
                WALKING_MASS_KG * gravity
            )

        return evaluate

    exponents = {
        term: _gravity_exponent(normalized(term))
        for term in ("gravitational", "shear", "compaction", "swing")
    }
    assert exponents["swing"] == pytest.approx(-1.0, rel=1e-6)
    assert exponents["gravitational"] == pytest.approx(0.0, abs=1e-9)
    assert exponents["compaction"] > 0.0

    # Two terms rise as gravity falls, for the same reason: neither depends on
    # weight. But swing does not scale at all while only the cohesive part of
    # shear fails to, so the two are not comparable in size and reporting them
    # as a pair would suggest they are.
    rising = sorted(
        (term for term, value in exponents.items() if value < -1e-6),
        key=lambda term: exponents[term],
    )
    assert rising == ["swing", "shear"], f"got {exponents}"
    assert abs(exponents["swing"]) > 20.0 * abs(exponents["shear"]), (
        "swing is the reason lunar walking is expensive; shear rises with it "
        f"but by the cohesive fraction alone, which is {exponents['shear']:.1%}"
    )


def test_the_cohesive_fraction_is_exactly_the_shear_terms_gravity_deficit(
    foot: ContactPatch,
    lunar_contact: BekkerModel,
    lunar_strength: MohrCoulombModel,
    lunar_mobilization: JanosiHanamotoModel,
) -> None:
    # Shear work goes as the maximum shear stress, whose frictional part scales
    # with weight and whose cohesive part does not. So the elasticity of shear
    # work with respect to gravity is one minus the cohesive fraction, exactly.
    # The Day 2 measurement that cohesion is a few percent of shear strength at
    # foot stress is therefore the same statement as shear being very nearly
    # gravity-neutral, and this test ties the two together.
    feet_in_stance = 2

    def shear_work(gravity: float) -> float:
        walk = _walk(
            gravity=gravity,
            foot=foot,
            contact=lunar_contact,
            strength=lunar_strength,
            mobilization=lunar_mobilization,
            feet_in_stance=feet_in_stance,
        )
        return float(walk.shear_J_per_m)

    stress_kPa = foot.normal_stress_kPa(
        normal_load_N=WALKING_MASS_KG * LUNAR_GRAVITY / feet_in_stance
    )
    cohesive_share = float(lunar_strength.cohesive_fraction(normal_stress=stress_kPa))

    assert _gravity_exponent(shear_work) == pytest.approx(
        1.0 - cohesive_share, rel=1e-5
    )
    assert cohesive_share < 0.05


# --- guards


def test_a_platform_with_nothing_on_the_ground_is_refused(
    foot: ContactPatch,
    lunar_contact: BekkerModel,
    lunar_strength: MohrCoulombModel,
    lunar_mobilization: JanosiHanamotoModel,
) -> None:
    with pytest.raises(ValueError, match="at least one"):
        _walk(
            gravity=LUNAR_GRAVITY,
            foot=foot,
            contact=lunar_contact,
            strength=lunar_strength,
            mobilization=lunar_mobilization,
            feet_in_stance=0,
        )


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_unphysical_platform_quantities_are_refused(
    bad: float,
    foot: ContactPatch,
    lunar_contact: BekkerModel,
    lunar_strength: MohrCoulombModel,
    lunar_mobilization: JanosiHanamotoModel,
) -> None:
    with pytest.raises(ValueError, match="must be finite and positive"):
        cost_of_transport(
            mass_kg=bad,
            gravity_m_per_s2=LUNAR_GRAVITY,
            slope_degrees=0.0,
            slip_ratio=0.1,
            patch=foot,
            feet_in_stance=2,
            stride_length_m=STRIDE_M,
            stance_length_m=STRIDE_M,
            contact_model=lunar_contact,
            strength=lunar_strength,
            mobilization=lunar_mobilization,
            swing_work_per_meter_J=25.0,
        )


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_an_unusable_swing_cost_is_refused(
    bad: float,
    foot: ContactPatch,
    lunar_contact: BekkerModel,
    lunar_strength: MohrCoulombModel,
    lunar_mobilization: JanosiHanamotoModel,
) -> None:
    with pytest.raises(ValueError, match="swing_work_per_meter_J"):
        _walk(
            gravity=LUNAR_GRAVITY,
            foot=foot,
            contact=lunar_contact,
            strength=lunar_strength,
            mobilization=lunar_mobilization,
            swing_work_per_meter_J=bad,
        )


def test_zero_swing_work_is_allowed_because_it_is_the_no_platform_baseline(
    foot: ContactPatch,
    lunar_contact: BekkerModel,
    lunar_strength: MohrCoulombModel,
    lunar_mobilization: JanosiHanamotoModel,
) -> None:
    walk = _walk(
        gravity=LUNAR_GRAVITY,
        foot=foot,
        contact=lunar_contact,
        strength=lunar_strength,
        mobilization=lunar_mobilization,
        swing_work_per_meter_J=0.0,
    )
    assert float(walk.swing_J_per_m) == 0.0
    assert float(walk.total_J_per_m) > 0.0


# --- against the published lunar parameters


def test_a_stride_mobilizes_the_distances_the_soil_file_records(
    lunar_mobilization: JanosiHanamotoModel,
) -> None:
    recorded = _lunar()["janosi_hanamoto"]["mobilization"]
    for distance_key, fraction_key in (
        ("slide_at_63_percent_mm", "fraction_at_one_K"),
        ("slide_at_95_percent_mm", "fraction_at_three_K"),
    ):
        slipped = float(
            slip_displacement(
                stance_length_m=STRIDE_M,
                slip_ratio=recorded[distance_key] / 1000.0 / STRIDE_M,
            )
        )
        reached = float(
            lunar_mobilization.mobilized_fraction(shear_displacement=slipped)
        )
        assert reached == pytest.approx(recorded[fraction_key], abs=5e-3)


def test_a_two_point_stance_on_a_small_foot_leaves_the_published_sinkage_range(
    lunar_contact: BekkerModel,
) -> None:
    # The boundary between measurement and extrapolation, as a test. Table 9.14
    # is valid to twenty millimeters of sinkage. A 20 mm half-width foot in a
    # two-point stance passes that; the same foot in a four-point stance does
    # not. Which gait is being simulated therefore decides whether the soil
    # model is being used or extrapolated, and that has to fail loudly if the
    # transcribed validity range ever changes.
    ceiling_m = _lunar()["bekker"]["validity"]["sinkage"]["max"]
    half_width = 0.020
    small_foot = ContactPatch(
        half_width_m=half_width, area_m2=math.pi * half_width**2
    )

    def sinkage(feet_in_stance: int) -> float:
        load = WALKING_MASS_KG * LUNAR_GRAVITY / feet_in_stance
        return float(
            lunar_contact.sinkage(
                pressure=small_foot.normal_stress_kPa(normal_load_N=load),
                contact_half_width=half_width,
            )
        )

    assert sinkage(4) <= ceiling_m
    assert sinkage(2) > ceiling_m, (
        "a trot on a 20 mm foot sinks past the published validity range, so a "
        "cost of transport computed there is extrapolated rather than measured"
    )


def test_a_larger_foot_stays_inside_the_published_range_in_either_gait(
    lunar_contact: BekkerModel,
) -> None:
    ceiling_m = _lunar()["bekker"]["validity"]["sinkage"]["max"]
    half_width = 0.030
    for feet_in_stance in (2, 4):
        patch = ContactPatch(
            half_width_m=half_width, area_m2=math.pi * half_width**2
        )
        load = WALKING_MASS_KG * LUNAR_GRAVITY / feet_in_stance
        depth = float(
            lunar_contact.sinkage(
                pressure=patch.normal_stress_kPa(normal_load_N=load),
                contact_half_width=half_width,
            )
        )
        assert depth <= ceiling_m


def test_the_lunar_parameters_have_no_invertibility_bound(
    lunar_contact: BekkerModel,
) -> None:
    # Every simulant fit in this repository dies at one end of the half-width
    # range because its two moduli have opposite signs. Both lunar moduli are
    # positive, so this one does not, and a foot-scale patch is usable.
    span = lunar_contact.invertible_half_width_range()
    assert not span.is_empty
    assert span.contains(0.020)
    assert span.contains(0.250)
