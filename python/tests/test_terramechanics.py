# SPDX-License-Identifier: Apache-2.0
#
# Tests for biome.terramechanics.
#
# Known-answer cases are read from the soil files in data/soils, never written
# here, so adding a soil adds its own tests and a transcription change fails
# loudly. Anything asserted about a specific number lives in the data.
#
# Model-behaviour tests parametrise over CONTACT_MODELS rather than naming
# classes, so registering a model covers it automatically. They rely on every
# registered model sharing one constructor signature, which is itself asserted
# by test_registered_models_share_one_parameter_signature.

from __future__ import annotations

import inspect
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pytest
from numpy.typing import NDArray

from biome.terramechanics import (
    CONTACT_MODELS,
    BekkerModel,
    ContactModel,
    DegenerateContactModelError,
)

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
SOIL_DIRECTORY: Final = REPOSITORY_ROOT / "data" / "soils"
SHARED_PARAMETER_NAMES: Final = (
    "cohesive_modulus",
    "frictional_modulus",
    "sinkage_exponent",
)
SINKAGE_SAMPLE_COUNT: Final = 512


@dataclass(frozen=True, slots=True)
class DatasetUnderTest:
    soil_id: str
    dataset_id: str
    apparatus: dict[str, Any]

    def __str__(self) -> str:
        return f"{self.soil_id}/{self.dataset_id}"

    @property
    def plates(self) -> list[dict[str, float]]:
        return list(self.apparatus["plates"])


@dataclass(frozen=True, slots=True)
class ModelUnderTest:
    soil_id: str
    dataset_id: str
    model_id: str
    status: str
    specification: dict[str, Any]
    plates: tuple[dict[str, float], ...]

    def __str__(self) -> str:
        return f"{self.soil_id}/{self.dataset_id}/{self.model_id}"

    @property
    def parameters(self) -> dict[str, float]:
        return {
            name: entry["value"]
            for name, entry in self.specification["parameters"].items()
        }

    @property
    def verification_cases(self) -> list[dict[str, float]]:
        return list(self.specification.get("verification", {}).get("cases", []))

    @property
    def tested_half_widths(self) -> NDArray[np.float64]:
        return np.array(
            [plate["contact_half_width_m"] for plate in self.plates], dtype=np.float64
        )

    @property
    def sinkage_bounds(self) -> tuple[float, float]:
        bounds = self.specification["validity"]["sinkage"]
        return float(bounds["min"]), float(bounds["max"])

    def build(self) -> ContactModel:
        return CONTACT_MODELS[self.model_id](**self.parameters)

    def sinkage_sweep(self) -> NDArray[np.float64]:
        minimum, maximum = self.sinkage_bounds
        return np.linspace(minimum, maximum, SINKAGE_SAMPLE_COUNT)


def _discover() -> tuple[list[DatasetUnderTest], list[ModelUnderTest]]:
    datasets: list[DatasetUnderTest] = []
    models: list[ModelUnderTest] = []
    for path in sorted(SOIL_DIRECTORY.glob("*.toml")):
        soil = tomllib.loads(path.read_text(encoding="utf-8"))
        for dataset in soil["dataset"]:
            plates = tuple(dataset["apparatus"]["plates"])
            datasets.append(
                DatasetUnderTest(
                    soil_id=soil["id"],
                    dataset_id=dataset["id"],
                    apparatus=dataset["apparatus"],
                )
            )
            for specification in dataset["model"]:
                models.append(
                    ModelUnderTest(
                        soil_id=soil["id"],
                        dataset_id=dataset["id"],
                        model_id=specification["id"],
                        status=specification["status"],
                        specification=specification,
                        plates=plates,
                    )
                )
    return datasets, models


ALL_DATASETS, ALL_MODELS = _discover()
VERIFIED_MODELS: Final = [model for model in ALL_MODELS if model.status == "verified"]
REGISTERED_MODEL_IDS: Final = sorted(CONTACT_MODELS)


def _reference_model(model_id: str) -> ContactModel:
    return CONTACT_MODELS[model_id](
        cohesive_modulus=-1.0, frictional_modulus=1.0, sinkage_exponent=1.25
    )


def test_soil_corpus_is_discovered() -> None:
    assert SOIL_DIRECTORY.is_dir(), f"no soil directory at {SOIL_DIRECTORY}"
    assert ALL_MODELS, f"no models discovered under {SOIL_DIRECTORY}"
    assert VERIFIED_MODELS, f"no verified models discovered under {SOIL_DIRECTORY}"
    assert ALL_DATASETS, f"no datasets discovered under {SOIL_DIRECTORY}"


def test_registered_models_share_one_parameter_signature() -> None:
    signatures = {
        model_id: tuple(inspect.signature(factory).parameters)
        for model_id, factory in CONTACT_MODELS.items()
    }
    assert set(signatures.values()) == {SHARED_PARAMETER_NAMES}, (
        "model-behaviour tests construct every registered model with "
        f"{SHARED_PARAMETER_NAMES}; signatures found were {signatures}"
    )


@pytest.mark.parametrize("model_id", REGISTERED_MODEL_IDS)
def test_registered_models_satisfy_the_contact_model_protocol(model_id: str) -> None:
    assert isinstance(_reference_model(model_id), ContactModel)


@pytest.mark.parametrize("model", ALL_MODELS, ids=str)
def test_status_matches_registry_membership(model: ModelUnderTest) -> None:
    registered = model.model_id in CONTACT_MODELS
    assert (model.status == "verified") == registered, (
        f"{model} has status {model.status!r} but is "
        f"{'present in' if registered else 'absent from'} CONTACT_MODELS"
    )


@pytest.mark.parametrize("model", VERIFIED_MODELS, ids=str)
def test_parameter_keys_match_the_constructor_signature(model: ModelUnderTest) -> None:
    expected = set(inspect.signature(CONTACT_MODELS[model.model_id]).parameters)
    assert set(model.parameters) == expected


@pytest.mark.parametrize("model", ALL_MODELS, ids=str)
def test_validity_bounds_are_ordered(model: ModelUnderTest) -> None:
    for quantity, bounds in model.specification["validity"].items():
        assert bounds["min"] <= bounds["max"], f"{model}: {quantity} bounds inverted"


@pytest.mark.parametrize("model", VERIFIED_MODELS, ids=str)
def test_verification_cases_are_well_formed(model: ModelUnderTest) -> None:
    assert model.verification_cases, f"{model} is verified but has no cases"
    for case in model.verification_cases:
        assert {
            "contact_half_width_m",
            "sinkage_m",
            "pressure_kPa",
            "rel_tol",
        } <= set(case), f"{model}: case missing keys, got {sorted(case)}"
        assert case["rel_tol"] > 0.0, f"{model}: non-positive rel_tol"
        assert math.isfinite(case["pressure_kPa"]), f"{model}: non-finite pressure"


@pytest.mark.parametrize("model", VERIFIED_MODELS, ids=str)
def test_verification_cases_lie_inside_the_validity_range(
    model: ModelUnderTest,
) -> None:
    validity = model.specification["validity"]
    for case in model.verification_cases:
        for case_key, quantity in (
            ("contact_half_width_m", "contact_half_width"),
            ("sinkage_m", "sinkage"),
        ):
            bounds = validity[quantity]
            assert bounds["min"] <= case[case_key] <= bounds["max"], (
                f"{model}: verification case {case_key}={case[case_key]} lies "
                f"outside the fitted range [{bounds['min']}, {bounds['max']}], "
                "so it would be verifying extrapolation"
            )


@pytest.mark.parametrize("model", ALL_MODELS, ids=str)
def test_quality_and_verification_reference_tested_plates(
    model: ModelUnderTest,
) -> None:
    tested = {plate["contact_half_width_m"] for plate in model.plates}
    for row in model.specification["quality"]["by_plate"]:
        assert row["contact_half_width_m"] in tested, f"{model}: unknown quality plate"
    for case in model.verification_cases:
        assert case["contact_half_width_m"] in tested, f"{model}: unknown case plate"


@pytest.mark.parametrize("model", VERIFIED_MODELS, ids=str)
def test_verified_models_reproduce_their_verification_cases(
    model: ModelUnderTest,
) -> None:
    instance = model.build()
    for case in model.verification_cases:
        computed = instance.pressure(
            sinkage=case["sinkage_m"],
            contact_half_width=case["contact_half_width_m"],
        )
        assert float(computed) == pytest.approx(
            case["pressure_kPa"], rel=case["rel_tol"], abs=0.0
        )


@pytest.mark.parametrize("model", VERIFIED_MODELS, ids=str)
def test_verification_cases_agree_between_scalar_and_array_evaluation(
    model: ModelUnderTest,
) -> None:
    instance = model.build()
    cases = model.verification_cases
    batched = instance.pressure(
        sinkage=np.array([case["sinkage_m"] for case in cases]),
        contact_half_width=np.array(
            [case["contact_half_width_m"] for case in cases]
        ),
    )
    one_at_a_time = np.array(
        [
            float(
                instance.pressure(
                    sinkage=case["sinkage_m"],
                    contact_half_width=case["contact_half_width_m"],
                )
            )
            for case in cases
        ]
    )
    np.testing.assert_allclose(batched, one_at_a_time, rtol=1e-15, atol=0.0)


@pytest.mark.parametrize("dataset", ALL_DATASETS, ids=str)
def test_plate_half_width_is_exactly_half_the_diameter(
    dataset: DatasetUnderTest,
) -> None:
    assert dataset.apparatus["plate_length_scale"] == "radius", (
        f"{dataset}: b == diameter/2 only holds when the reported plate length "
        "scale is the radius"
    )
    for plate in dataset.plates:
        assert plate["contact_half_width_m"] == plate["diameter_m"] / 2.0


@pytest.mark.parametrize("dataset", ALL_DATASETS, ids=str)
def test_plate_area_matches_circular_geometry(dataset: DatasetUnderTest) -> None:
    assert dataset.apparatus["plate_geometry"] == "circular_rigid_flat", (
        f"{dataset}: pi*b^2 only holds for circular plates"
    )
    for plate in dataset.plates:
        assert plate["area_m2"] == pytest.approx(
            math.pi * plate["contact_half_width_m"] ** 2, rel=1e-6
        )


@pytest.mark.parametrize("model", VERIFIED_MODELS, ids=str)
def test_pressure_vanishes_at_zero_sinkage(model: ModelUnderTest) -> None:
    pressure = model.build().pressure(
        sinkage=0.0, contact_half_width=model.tested_half_widths
    )
    np.testing.assert_array_equal(pressure, np.zeros_like(model.tested_half_widths))


@pytest.mark.parametrize("model", VERIFIED_MODELS, ids=str)
def test_pressure_is_strictly_increasing_in_sinkage(model: ModelUnderTest) -> None:
    instance = model.build()
    depth = model.sinkage_sweep()
    for half_width in model.tested_half_widths:
        pressure = instance.pressure(sinkage=depth, contact_half_width=half_width)
        assert np.all(np.diff(pressure) > 0.0), (
            f"{model}: pressure not strictly increasing at "
            f"contact_half_width={half_width}"
        )


@pytest.mark.parametrize("model", VERIFIED_MODELS, ids=str)
def test_sinkage_inverts_pressure(model: ModelUnderTest) -> None:
    instance = model.build()
    depth = model.sinkage_sweep()
    for half_width in model.tested_half_widths:
        recovered = instance.sinkage(
            pressure=instance.pressure(sinkage=depth, contact_half_width=half_width),
            contact_half_width=half_width,
        )
        np.testing.assert_allclose(recovered, depth, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("model", VERIFIED_MODELS, ids=str)
def test_pressure_broadcasts_over_sinkage_and_half_width(
    model: ModelUnderTest,
) -> None:
    instance = model.build()
    depth = np.linspace(*model.sinkage_bounds, 7)
    half_widths = model.tested_half_widths
    grid = instance.pressure(
        sinkage=depth[:, np.newaxis], contact_half_width=half_widths[np.newaxis, :]
    )
    assert grid.shape == (depth.size, half_widths.size)
    for row, single_depth in enumerate(depth):
        np.testing.assert_allclose(
            grid[row],
            instance.pressure(sinkage=single_depth, contact_half_width=half_widths),
            rtol=1e-15,
            atol=0.0,
        )


@pytest.mark.parametrize("model", VERIFIED_MODELS, ids=str)
def test_empty_sinkage_yields_empty_pressure(model: ModelUnderTest) -> None:
    pressure = model.build().pressure(
        sinkage=np.array([]), contact_half_width=model.tested_half_widths[0]
    )
    assert pressure.shape == (0,)


@pytest.mark.parametrize("model", VERIFIED_MODELS, ids=str)
@pytest.mark.parametrize("bad_sinkage", [-1e-12, -0.05, math.nan, math.inf])
def test_pressure_rejects_invalid_sinkage(
    model: ModelUnderTest, bad_sinkage: float
) -> None:
    with pytest.raises(ValueError, match="sinkage must be finite and non-negative"):
        model.build().pressure(
            sinkage=bad_sinkage, contact_half_width=model.tested_half_widths[0]
        )


@pytest.mark.parametrize("model", VERIFIED_MODELS, ids=str)
@pytest.mark.parametrize("bad_pressure", [-1e-12, -100.0, math.nan, math.inf])
def test_sinkage_rejects_invalid_pressure(
    model: ModelUnderTest, bad_pressure: float
) -> None:
    with pytest.raises(ValueError, match="pressure must be finite and non-negative"):
        model.build().sinkage(
            pressure=bad_pressure, contact_half_width=model.tested_half_widths[0]
        )


@pytest.mark.parametrize("model", VERIFIED_MODELS, ids=str)
@pytest.mark.parametrize("bad_half_width", [0.0, -0.03, math.nan, math.inf])
def test_evaluation_rejects_invalid_half_width(
    model: ModelUnderTest, bad_half_width: float
) -> None:
    instance = model.build()
    expected = "contact_half_width must be finite and positive"
    with pytest.raises(ValueError, match=expected):
        instance.pressure(sinkage=0.05, contact_half_width=bad_half_width)
    with pytest.raises(ValueError, match=expected):
        instance.sinkage(pressure=1.0, contact_half_width=bad_half_width)
    with pytest.raises(ValueError, match=expected):
        instance.deformation_modulus(bad_half_width)


@pytest.mark.parametrize("model_id", REGISTERED_MODEL_IDS)
@pytest.mark.parametrize("bad_exponent", [0.0, -1.0, math.nan, math.inf, 1e-320])
def test_construction_rejects_invalid_sinkage_exponent(
    model_id: str, bad_exponent: float
) -> None:
    with pytest.raises(ValueError, match="sinkage_exponent"):
        CONTACT_MODELS[model_id](
            cohesive_modulus=1.0,
            frictional_modulus=1.0,
            sinkage_exponent=bad_exponent,
        )


@pytest.mark.parametrize("model_id", REGISTERED_MODEL_IDS)
@pytest.mark.parametrize(
    "parameter", ["cohesive_modulus", "frictional_modulus"]
)
@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_construction_rejects_non_finite_parameters(
    model_id: str, parameter: str, bad_value: float
) -> None:
    arguments: dict[str, float] = {
        "cohesive_modulus": 1.0,
        "frictional_modulus": 1.0,
        "sinkage_exponent": 1.25,
    }
    arguments[parameter] = bad_value
    with pytest.raises(ValueError, match=f"{parameter} must be finite"):
        CONTACT_MODELS[model_id](**arguments)


@pytest.mark.parametrize("model_id", REGISTERED_MODEL_IDS)
def test_pressure_and_sinkage_reject_positional_arguments(model_id: str) -> None:
    instance = _reference_model(model_id)
    with pytest.raises(TypeError):
        instance.pressure(0.05, 0.03)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        instance.sinkage(1.0, 0.03)  # type: ignore[call-arg]


@pytest.mark.parametrize("model", VERIFIED_MODELS, ids=str)
def test_evaluation_refuses_below_the_invertibility_threshold(
    model: ModelUnderTest,
) -> None:
    instance = model.build()
    threshold = instance.minimum_invertible_half_width()
    if not 0.0 < threshold < math.inf:
        pytest.skip(f"{model} is invertible across all half-widths")
    with pytest.raises(DegenerateContactModelError, match="deformation modulus"):
        instance.pressure(sinkage=0.05, contact_half_width=threshold * (1.0 - 1e-6))
    with pytest.raises(DegenerateContactModelError, match="deformation modulus"):
        instance.sinkage(pressure=1.0, contact_half_width=threshold * (1.0 - 1e-6))


@pytest.mark.parametrize("model", VERIFIED_MODELS, ids=str)
def test_invertibility_threshold_is_the_modulus_sign_change(
    model: ModelUnderTest,
) -> None:
    instance = model.build()
    threshold = instance.minimum_invertible_half_width()
    if not 0.0 < threshold < math.inf:
        pytest.skip(f"{model} is invertible across all half-widths")
    assert float(instance.deformation_modulus(threshold * (1.0 - 1e-6))) < 0.0
    assert float(instance.deformation_modulus(threshold * (1.0 + 1e-6))) > 0.0


@pytest.mark.parametrize("model", VERIFIED_MODELS, ids=str)
def test_tested_plates_lie_above_the_invertibility_threshold(
    model: ModelUnderTest,
) -> None:
    threshold = model.build().minimum_invertible_half_width()
    assert np.all(model.tested_half_widths > threshold), (
        f"{model}: a plate the parameters were fitted to lies below the "
        f"invertibility threshold {threshold}, which would make the fit itself "
        "degenerate"
    )


@pytest.mark.parametrize("model_id", REGISTERED_MODEL_IDS)
def test_threshold_is_infinite_when_no_half_width_is_invertible(
    model_id: str,
) -> None:
    nowhere = CONTACT_MODELS[model_id](
        cohesive_modulus=-1.0, frictional_modulus=-1.0, sinkage_exponent=1.25
    )
    assert nowhere.minimum_invertible_half_width() == math.inf
    with pytest.raises(DegenerateContactModelError):
        nowhere.pressure(sinkage=0.05, contact_half_width=1.0)


@pytest.mark.parametrize("model_id", REGISTERED_MODEL_IDS)
def test_threshold_is_zero_when_every_half_width_is_invertible(
    model_id: str,
) -> None:
    everywhere = CONTACT_MODELS[model_id](
        cohesive_modulus=1.0, frictional_modulus=1.0, sinkage_exponent=1.25
    )
    assert everywhere.minimum_invertible_half_width() == 0.0


def test_bekker_refuses_a_half_width_that_overflows_the_modulus() -> None:
    instance = BekkerModel(
        cohesive_modulus=1.0, frictional_modulus=1.0, sinkage_exponent=1.0
    )
    with (
        np.errstate(over="ignore"),
        pytest.raises(DegenerateContactModelError, match="finite and positive"),
    ):
        instance.pressure(sinkage=0.05, contact_half_width=1e-320)
