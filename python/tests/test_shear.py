# SPDX-License-Identifier: Apache-2.0
#
# Tests for the shear models in eclipse.terramechanics.
#
# Expected values are read from data/soils/lunar-intercrater.toml rather than
# written here, so adding a soil adds its own cases. The mobilization distances
# recorded in that file are the ones a gait analysis will consume, and they are
# checked against the model rather than against a comment.

from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eclipse.terramechanics import (
    JanosiHanamotoModel,
    MohrCoulombModel,
    shear_stress,
)

LUNAR_SOIL = (
    Path(__file__).resolve().parents[2] / "data" / "soils" / "lunar-intercrater.toml"
)


def _models() -> dict[str, Any]:
    table = tomllib.loads(LUNAR_SOIL.read_text(encoding="utf-8"))
    return {model["id"]: model for model in table["dataset"][0]["shear_model"]}


def _strength_at(depth_range: str) -> MohrCoulombModel:
    rows = _models()["mohr_coulomb"]["by_depth"]["rows"]
    row = next(entry for entry in rows if entry["depth_range_cm"] == depth_range)
    return MohrCoulombModel(
        cohesion=row["cohesion_kPa"], friction_angle_degrees=row["friction_angle_deg"]
    )


def test_strength_is_cohesion_at_zero_normal_stress() -> None:
    strength = _strength_at("0-15")
    assert float(strength.maximum_shear_stress(normal_stress=0.0)) == pytest.approx(
        strength.cohesion
    )


def test_the_offset_at_zero_load_is_what_distinguishes_this_from_a_friction_cone() -> None:
    strength = _strength_at("0-15")
    carried = float(strength.maximum_shear_stress(normal_stress=0.0))
    assert carried > 0.0, (
        "a patch carries tangential load at zero normal force, which a pure "
        "friction cone forbids; the offset is small but it changes the "
        "constraint structure rather than only its magnitude"
    )


@pytest.mark.parametrize("depth_range", ["0-15", "0-30", "30-60", "0-60"])
def test_every_published_depth_row_builds_a_usable_model(depth_range: str) -> None:
    strength = _strength_at(depth_range)
    stresses = np.array([0.0, 5.0, 16.3, 30.0])
    carried = strength.maximum_shear_stress(normal_stress=stresses)
    assert np.all(np.isfinite(carried))
    assert np.all(np.diff(carried) > 0.0), "shear strength must rise with normal load"


def test_cohesion_is_a_minor_term_at_lunar_foot_stress() -> None:
    # 50 kg quadruped at one sixth gravity, four-point stance, 20 mm half-width.
    foot_stress = 50.0 * 9.81 / 6.0 / 4.0 / (math.pi * 0.020**2) / 1000.0
    share = float(_strength_at("0-15").cohesive_fraction(normal_stress=foot_stress))
    assert share < 0.10, (
        f"cohesion contributes {share:.1%} of shear strength at {foot_stress:.1f} "
        "kPa; it does not dominate lunar foot traction, and a claim that it does "
        "would have to survive this arithmetic"
    )


def test_friction_angle_carries_more_uncertainty_than_cohesion() -> None:
    rows = _models()["mohr_coulomb"]["by_depth"]["rows"]
    row = next(entry for entry in rows if entry["depth_range_cm"] == "0-15")
    foot_stress = 16.3
    from_cohesion = row["cohesion_max_kPa"] - row["cohesion_min_kPa"]
    from_friction = foot_stress * (
        math.tan(math.radians(row["friction_angle_max_deg"]))
        - math.tan(math.radians(row["friction_angle_min_deg"]))
    )
    assert from_friction > from_cohesion, (
        "the published ranges put more absolute uncertainty in the friction "
        "angle than in cohesion, so cohesion is not the binding parameter"
    )


def test_mobilization_reaches_the_recorded_fractions_at_the_recorded_distances() -> None:
    model = _models()["janosi_hanamoto"]
    modulus_cm = model["parameters"]["shear_deformation_modulus"]["value"]
    recorded = model["mobilization"]
    janosi = JanosiHanamotoModel(shear_deformation_modulus=modulus_cm * 10.0)

    for distance_key, fraction_key in (
        ("slide_at_63_percent_mm", "fraction_at_one_K"),
        ("slide_at_95_percent_mm", "fraction_at_three_K"),
    ):
        reached = float(
            janosi.mobilized_fraction(shear_displacement=recorded[distance_key])
        )
        assert reached == pytest.approx(recorded[fraction_key], abs=5e-3), (
            f"{distance_key} should mobilize {recorded[fraction_key]}, got {reached}"
        )


def test_one_and_three_moduli_give_the_canonical_fractions() -> None:
    janosi = JanosiHanamotoModel(shear_deformation_modulus=18.0)
    assert float(
        janosi.mobilized_fraction(shear_displacement=18.0)
    ) == pytest.approx(1.0 - math.exp(-1.0))
    assert float(
        janosi.mobilized_fraction(shear_displacement=54.0)
    ) == pytest.approx(1.0 - math.exp(-3.0))


def test_displacement_for_fraction_inverts_mobilized_fraction() -> None:
    janosi = JanosiHanamotoModel(shear_deformation_modulus=18.0)
    for fraction in (0.0, 0.25, 0.632, 0.95, 0.999):
        distance = janosi.displacement_for_fraction(fraction)
        assert float(
            janosi.mobilized_fraction(shear_displacement=distance)
        ) == pytest.approx(fraction, abs=1e-12)


def test_mobilization_is_zero_at_no_slide_and_rises_without_exceeding_one() -> None:
    modulus = 18.0
    janosi = JanosiHanamotoModel(shear_deformation_modulus=modulus)
    assert float(janosi.mobilized_fraction(shear_displacement=0.0)) == 0.0

    # Strictly below one over the range a gait actually visits. Further out the
    # exponential underflows and the fraction saturates at exactly 1.0, which is
    # correct double-precision behaviour rather than a defect: the asymptote is
    # mathematical and is not representable.
    within_reach = np.array([0.0, 0.5, 1.0, 3.0, 6.0]) * modulus
    fractions = janosi.mobilized_fraction(shear_displacement=within_reach)
    assert np.all(np.diff(fractions) > 0.0)
    assert np.all(fractions < 1.0)

    saturated = janosi.mobilized_fraction(shear_displacement=100.0 * modulus)
    assert float(saturated) == 1.0
    assert np.all(
        janosi.mobilized_fraction(shear_displacement=np.array([1e3, 1e9]) * modulus)
        <= 1.0
    ), "the fraction is a probability-like quantity and must never exceed one"


def test_shear_stress_composes_strength_with_mobilization() -> None:
    strength = _strength_at("0-15")
    janosi = JanosiHanamotoModel(shear_deformation_modulus=18.0)
    carried = shear_stress(
        strength=strength, mobilization=janosi,
        normal_stress=16.3, shear_displacement=18.0,
    )
    expected = float(
        strength.maximum_shear_stress(normal_stress=16.3)
    ) * (1.0 - math.exp(-1.0))
    assert float(carried) == pytest.approx(expected)


@pytest.mark.parametrize("cohesion", [-1e-9, math.nan, math.inf])
def test_an_unphysical_cohesion_is_refused(cohesion: float) -> None:
    with pytest.raises(ValueError, match="cohesion must be finite and non-negative"):
        MohrCoulombModel(cohesion=cohesion, friction_angle_degrees=35.0)


@pytest.mark.parametrize("angle", [0.0, 90.0, -5.0, 120.0, math.nan])
def test_a_friction_angle_outside_the_open_interval_is_refused(angle: float) -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 90"):
        MohrCoulombModel(cohesion=0.5, friction_angle_degrees=angle)


@pytest.mark.parametrize("modulus", [0.0, -1.0, math.nan, math.inf])
def test_an_unusable_deformation_modulus_is_refused(modulus: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        JanosiHanamotoModel(shear_deformation_modulus=modulus)


@pytest.mark.parametrize("fraction", [-0.1, 1.0, 1.5])
def test_asking_for_full_mobilization_is_refused(fraction: float) -> None:
    with pytest.raises(ValueError, match=r"lie in \[0, 1\)"):
        JanosiHanamotoModel(shear_deformation_modulus=18.0).displacement_for_fraction(
            fraction
        )


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_negative_or_non_finite_inputs_are_refused(bad: float) -> None:
    strength = MohrCoulombModel(cohesion=0.5, friction_angle_degrees=35.0)
    janosi = JanosiHanamotoModel(shear_deformation_modulus=18.0)
    with pytest.raises(ValueError, match="normal_stress must be finite"):
        strength.maximum_shear_stress(normal_stress=[1.0, bad])
    with pytest.raises(ValueError, match="shear_displacement must be finite"):
        janosi.mobilized_fraction(shear_displacement=[1.0, bad])


def test_the_models_are_value_objects() -> None:
    assert MohrCoulombModel(0.5, 35.0) == MohrCoulombModel(0.5, 35.0)
    with pytest.raises(AttributeError):
        MohrCoulombModel(0.5, 35.0).cohesion = 1.0  # type: ignore[misc]
