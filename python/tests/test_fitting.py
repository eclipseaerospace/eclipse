# SPDX-License-Identifier: Apache-2.0
#
# Tests for biome.fitting.
#
# The load-bearing test is parameter recovery: given exact output of a published
# model, the fit must return exactly the parameters that produced it. That is
# what licenses trusting a fit of digitized points, where the true answer is
# unknown and only the published parameters are available for comparison.

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pytest
from conftest import observations_from

from biome.fitting import (
    PressureSinkageObservations,
    WeightingScheme,
    coefficient_of_determination,
    coefficient_of_determination_by_plate,
    coefficient_of_determination_ceiling,
    fit_contact_model,
    fit_shared_power_law,
    mean_relative_residual,
    relative_deviation,
)
from biome.io.soil import CalibratedContactModel

EXACT_RECOVERY_TOLERANCE = 1e-9
NOISY_RECOVERY_TOLERANCE = 0.05
WEIGHTINGS: list[WeightingScheme] = ["uniform", "pressure_squared"]


@pytest.fixture
def bekker_observations(
    published_models: Mapping[str, CalibratedContactModel],
    tested_half_widths: np.ndarray,
) -> PressureSinkageObservations:
    return observations_from(published_models["bekker"], tested_half_widths)


@pytest.mark.parametrize("model_id", ["bekker", "reece"])
@pytest.mark.parametrize("weighting", WEIGHTINGS)
def test_a_fit_recovers_the_parameters_that_generated_the_data(
    model_id: str,
    weighting: WeightingScheme,
    published_models: Mapping[str, CalibratedContactModel],
    tested_half_widths: np.ndarray,
) -> None:
    published = published_models[model_id]
    fitted = fit_contact_model(
        model_id,
        observations_from(published, tested_half_widths),
        weighting=weighting,
    )
    for name, published_value in published.parameters.items():
        assert fitted.parameters[name] == pytest.approx(
            published_value, rel=EXACT_RECOVERY_TOLERANCE
        ), f"{model_id}/{weighting} did not recover {name}"


@pytest.mark.parametrize("model_id", ["bekker", "reece"])
def test_a_fit_of_exact_data_explains_all_of_its_variance(
    model_id: str,
    published_models: Mapping[str, CalibratedContactModel],
    tested_half_widths: np.ndarray,
) -> None:
    observations = observations_from(published_models[model_id], tested_half_widths)
    fitted = fit_contact_model(model_id, observations)
    assert coefficient_of_determination(fitted.model, observations) == pytest.approx(
        1.0, abs=1e-12
    )
    by_plate = coefficient_of_determination_by_plate(fitted.model, observations)
    assert len(by_plate) == tested_half_widths.size
    for value in by_plate.values():
        assert value == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("model_id", ["bekker", "reece"])
def test_recovery_survives_proportional_noise(
    model_id: str,
    published_models: Mapping[str, CalibratedContactModel],
    tested_half_widths: np.ndarray,
) -> None:
    published = published_models[model_id]
    exact = observations_from(published, tested_half_widths)
    generator = np.random.default_rng(20260803)
    noisy = PressureSinkageObservations(
        contact_half_width_m=exact.contact_half_width_m,
        sinkage_m=exact.sinkage_m,
        pressure_kPa=exact.pressure_kPa
        * (1.0 + generator.normal(0.0, 0.02, exact.count)),
    )
    fitted = fit_contact_model(model_id, noisy)
    for name, published_value in published.parameters.items():
        assert fitted.parameters[name] == pytest.approx(
            published_value, rel=NOISY_RECOVERY_TOLERANCE
        ), f"{model_id} lost {name} under 2 percent proportional noise"


def test_the_weighting_scheme_changes_the_answer(
    bekker_observations: PressureSinkageObservations,
) -> None:
    generator = np.random.default_rng(20260803)
    noisy = PressureSinkageObservations(
        contact_half_width_m=bekker_observations.contact_half_width_m,
        sinkage_m=bekker_observations.sinkage_m,
        pressure_kPa=bekker_observations.pressure_kPa
        * (1.0 + generator.normal(0.0, 0.05, bekker_observations.count)),
    )
    uniform = fit_contact_model("bekker", noisy, weighting="uniform")
    weighted = fit_contact_model("bekker", noisy, weighting="pressure_squared")
    assert uniform.parameters != weighted.parameters, (
        "the weighting argument is inert; on noisy data an unweighted log-space "
        "fit must differ from a pressure-weighted one"
    )
    assert uniform.weighting == "uniform"
    assert weighted.weighting == "pressure_squared"


def test_a_shared_exponent_is_fitted_once_across_plates(
    bekker_observations: PressureSinkageObservations,
    published_models: Mapping[str, CalibratedContactModel],
    tested_half_widths: np.ndarray,
) -> None:
    power_law = fit_shared_power_law(bekker_observations)
    assert power_law.sinkage_exponent == pytest.approx(
        published_models["bekker"].parameters["sinkage_exponent"],
        rel=EXACT_RECOVERY_TOLERANCE,
    )
    assert power_law.plate_moduli.size == tested_half_widths.size
    np.testing.assert_allclose(power_law.contact_half_widths_m, tested_half_widths)
    assert np.all(power_law.plate_moduli > 0.0)


def test_fitted_parameter_keys_match_the_model_constructor(
    bekker_observations: PressureSinkageObservations,
    published_models: Mapping[str, CalibratedContactModel],
) -> None:
    fitted = fit_contact_model("bekker", bekker_observations)
    assert set(fitted.parameters) == set(published_models["bekker"].parameters), (
        "a fit must be comparable to a transcribed parameters block key for key"
    )


def test_an_unregistered_model_is_refused(
    bekker_observations: PressureSinkageObservations,
) -> None:
    with pytest.raises(ValueError, match="no plate scaling is implemented"):
        fit_contact_model("dimensional_analysis_lim2021", bekker_observations)


@pytest.mark.parametrize("bad_value", [0.0, -1.0, math.nan, math.inf])
@pytest.mark.parametrize(
    "column", ["contact_half_width_m", "sinkage_m", "pressure_kPa"]
)
def test_non_positive_observations_are_refused(
    bekker_observations: PressureSinkageObservations, column: str, bad_value: float
) -> None:
    columns = {
        "contact_half_width_m": bekker_observations.contact_half_width_m.copy(),
        "sinkage_m": bekker_observations.sinkage_m.copy(),
        "pressure_kPa": bekker_observations.pressure_kPa.copy(),
    }
    columns[column][0] = bad_value
    with pytest.raises(ValueError, match=f"{column} must be finite and strictly"):
        PressureSinkageObservations(**columns)


def test_mismatched_column_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="must share one shape"):
        PressureSinkageObservations(
            contact_half_width_m=np.array([0.03, 0.03]),
            sinkage_m=np.array([0.01, 0.02, 0.03]),
            pressure_kPa=np.array([1.0, 2.0]),
        )


def test_a_single_plate_cannot_separate_the_moduli() -> None:
    depth = np.linspace(0.01, 0.09, 8)
    observations = PressureSinkageObservations(
        contact_half_width_m=np.full_like(depth, 0.03),
        sinkage_m=depth,
        pressure_kPa=2113.29 * np.power(depth, 1.2594),
    )
    with pytest.raises(ValueError, match="at least 2 distinct plate sizes"):
        fit_contact_model("bekker", observations)


def test_one_observation_per_plate_is_not_identifiable() -> None:
    observations = PressureSinkageObservations(
        contact_half_width_m=np.array([0.030, 0.035, 0.0375]),
        sinkage_m=np.array([0.05, 0.05, 0.05]),
        pressure_kPa=np.array([48.6, 53.4, 55.3]),
    )
    with pytest.raises(ValueError, match="not jointly identifiable"):
        fit_contact_model("bekker", observations)


def test_observations_can_be_selected_by_plate(
    bekker_observations: PressureSinkageObservations, tested_half_widths: np.ndarray
) -> None:
    total = 0
    for half_width in tested_half_widths:
        selected = bekker_observations.for_plate(float(half_width))
        assert np.all(selected.contact_half_width_m == half_width)
        total += selected.count
    assert total == bekker_observations.count


def test_a_model_does_not_deviate_from_itself(
    published_models: Mapping[str, CalibratedContactModel],
    tested_half_widths: np.ndarray,
) -> None:
    bekker = published_models["bekker"]
    deviation = relative_deviation(
        bekker.extrapolating,
        bekker.extrapolating,
        sinkage=np.linspace(0.01, 0.09, 32),
        contact_half_width=tested_half_widths[0],
    )
    np.testing.assert_allclose(deviation, 0.0, atol=1e-15)


def test_deviation_is_independent_of_sinkage_when_the_exponent_is_shared(
    published_models: Mapping[str, CalibratedContactModel],
    tested_half_widths: np.ndarray,
) -> None:
    bekker, reece = published_models["bekker"], published_models["reece"]
    assert (
        bekker.parameters["sinkage_exponent"] == reece.parameters["sinkage_exponent"]
    ), "this property only holds while both published fits share one exponent"
    for half_width in tested_half_widths:
        deviation = relative_deviation(
            bekker.extrapolating,
            reece.extrapolating,
            sinkage=np.linspace(0.001, 0.095, 64),
            contact_half_width=float(half_width),
        )
        assert deviation.max() - deviation.min() == pytest.approx(0.0, abs=1e-12)
        assert abs(float(deviation[0])) < 0.05


def test_deviation_carries_the_sign_of_the_second_model(
    published_models: Mapping[str, CalibratedContactModel],
) -> None:
    bekker, reece = published_models["bekker"], published_models["reece"]
    for half_width, expected_sign in ((0.030, -1.0), (0.0375, 1.0)):
        deviation = float(
            relative_deviation(
                bekker.extrapolating,
                reece.extrapolating,
                sinkage=0.05,
                contact_half_width=half_width,
            )
        )
        assert np.sign(deviation) == expected_sign


def test_deviation_at_zero_sinkage_is_refused(
    published_models: Mapping[str, CalibratedContactModel],
) -> None:
    bekker, reece = published_models["bekker"], published_models["reece"]
    with pytest.raises(ValueError, match="undefined where the reference pressure"):
        relative_deviation(
            bekker.extrapolating,
            reece.extrapolating,
            sinkage=0.0,
            contact_half_width=0.03,
        )


def test_mean_relative_residual_is_zero_for_the_generating_model(
    published_models: Mapping[str, CalibratedContactModel],
    tested_half_widths: np.ndarray,
) -> None:
    bekker = published_models["bekker"]
    observations = observations_from(bekker, tested_half_widths)
    assert mean_relative_residual(bekker, observations) == pytest.approx(0.0, abs=1e-12)


def test_mean_relative_residual_carries_the_sign_of_the_offset(
    published_models: Mapping[str, CalibratedContactModel],
    tested_half_widths: np.ndarray,
) -> None:
    bekker = published_models["bekker"]
    exact = observations_from(bekker, tested_half_widths)
    for factor, expected in ((1.10, 0.10), (0.90, -0.10)):
        lifted = PressureSinkageObservations(
            contact_half_width_m=exact.contact_half_width_m,
            sinkage_m=exact.sinkage_m,
            pressure_kPa=exact.pressure_kPa * factor,
        )
        assert mean_relative_residual(bekker, lifted) == pytest.approx(expected, abs=1e-9)


def test_the_determination_ceiling_is_one_without_replicates(
    published_models: Mapping[str, CalibratedContactModel],
    tested_half_widths: np.ndarray,
) -> None:
    exact = observations_from(published_models["bekker"], tested_half_widths)
    assert coefficient_of_determination_ceiling(exact) == pytest.approx(1.0, abs=1e-12)


def test_the_determination_ceiling_falls_with_replicate_scatter(
    published_models: Mapping[str, CalibratedContactModel],
    tested_half_widths: np.ndarray,
) -> None:
    exact = observations_from(published_models["bekker"], tested_half_widths)
    generator = np.random.default_rng(20260805)
    previous = 1.0
    for spread in (0.02, 0.05, 0.10):
        duplicated = PressureSinkageObservations(
            contact_half_width_m=np.tile(exact.contact_half_width_m, 4),
            sinkage_m=np.tile(exact.sinkage_m, 4),
            pressure_kPa=np.tile(exact.pressure_kPa, 4)
            * (1.0 + generator.normal(0.0, spread, exact.count * 4)),
        )
        ceiling = coefficient_of_determination_ceiling(duplicated)
        assert ceiling < previous, "wider replicate scatter must lower the ceiling"
        previous = ceiling


def test_no_fit_can_beat_the_determination_ceiling(
    published_models: Mapping[str, CalibratedContactModel],
    tested_half_widths: np.ndarray,
) -> None:
    exact = observations_from(published_models["bekker"], tested_half_widths)
    generator = np.random.default_rng(20260805)
    scattered = PressureSinkageObservations(
        contact_half_width_m=np.tile(exact.contact_half_width_m, 5),
        sinkage_m=np.tile(exact.sinkage_m, 5),
        pressure_kPa=np.tile(exact.pressure_kPa, 5)
        * (1.0 + generator.normal(0.0, 0.06, exact.count * 5)),
    )
    fitted = fit_contact_model("bekker", scattered)
    assert coefficient_of_determination(fitted.model, scattered) <= (
        coefficient_of_determination_ceiling(scattered) + 1e-9
    )
