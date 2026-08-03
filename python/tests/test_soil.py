# SPDX-License-Identifier: Apache-2.0
#
# Tests for biome.io.soil.
#
# Structural tests corrupt a copy of a real soil file rather than inventing a
# synthetic one, so they stay honest about the schema actually in use. Every
# corruption asserts that its target was present before mutating, because a
# mutation that silently fails to apply produces a test that cannot fail.

from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Final

import numpy as np
import pytest

from biome.io.soil import (
    CalibratedContactModel,
    OutsideValidityRangeError,
    Soil,
    SoilFileError,
    load_soil,
)
from biome.terramechanics import ContactModel, DegenerateContactModelError

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
SOIL_DIRECTORY: Final = REPOSITORY_ROOT / "data" / "soils"
SOIL_PATHS: Final = sorted(SOIL_DIRECTORY.glob("*.toml"))
REFERENCE_SOIL_PATH: Final = SOIL_DIRECTORY / "kls1.toml"


def _soil_id(path: Path) -> str:
    return path.stem


def _variant(tmp_path: Path, replacements: list[tuple[str, str]]) -> Path:
    text = REFERENCE_SOIL_PATH.read_text(encoding="utf-8")
    for old, new in replacements:
        assert old in text, f"mutation target absent from the reference soil: {old!r}"
        text = text.replace(old, new, 1)
    path = tmp_path / "variant.toml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def reference_soil() -> Soil:
    return load_soil(REFERENCE_SOIL_PATH)


@pytest.fixture(scope="module")
def reference_model(reference_soil: Soil) -> CalibratedContactModel:
    return reference_soil.datasets["lim2021"].models["bekker"]


def test_soil_corpus_is_discovered() -> None:
    assert SOIL_PATHS, f"no soil files under {SOIL_DIRECTORY}"
    assert REFERENCE_SOIL_PATH.is_file(), f"missing {REFERENCE_SOIL_PATH}"


@pytest.mark.parametrize("path", SOIL_PATHS, ids=_soil_id)
def test_every_soil_file_loads(path: Path) -> None:
    soil = load_soil(path)
    assert soil.source_path == path
    assert soil.datasets, f"{path}: no datasets loaded"


@pytest.mark.parametrize("path", SOIL_PATHS, ids=_soil_id)
def test_verified_models_load_and_others_are_excluded_without_raising(
    path: Path,
) -> None:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    soil = load_soil(path)
    for raw_dataset in raw["dataset"]:
        dataset = soil.datasets[raw_dataset["id"]]
        for specification in raw_dataset["model"]:
            model_id = specification["id"]
            if specification["status"] == "verified":
                assert model_id in dataset.models
                assert model_id not in dataset.excluded_models
            else:
                assert model_id not in dataset.models
                assert dataset.excluded_models[model_id] == specification["status"]


@pytest.mark.parametrize("path", SOIL_PATHS, ids=_soil_id)
def test_loaded_models_reproduce_their_verification_cases(path: Path) -> None:
    soil = load_soil(path)
    checked = 0
    for dataset in soil.datasets.values():
        for model in dataset.models.values():
            assert model.verification_cases, f"{model.id}: verified but no cases"
            for case in model.verification_cases:
                computed = model.pressure(
                    sinkage=case["sinkage_m"],
                    contact_half_width=case["contact_half_width_m"],
                )
                assert float(computed) == pytest.approx(
                    case["pressure_kPa"], rel=case["rel_tol"], abs=0.0
                )
                checked += 1
    assert checked, f"{path}: no verification cases exercised"


@pytest.mark.parametrize("path", SOIL_PATHS, ids=_soil_id)
def test_loaded_models_satisfy_the_contact_model_protocol(path: Path) -> None:
    for dataset in load_soil(path).datasets.values():
        for model in dataset.models.values():
            assert isinstance(model, ContactModel)


@pytest.mark.parametrize("path", SOIL_PATHS, ids=_soil_id)
def test_apparatus_exposes_the_tested_plate_half_widths(path: Path) -> None:
    for dataset in load_soil(path).datasets.values():
        half_widths = dataset.apparatus.contact_half_widths
        assert half_widths.size == len(dataset.apparatus.plates)
        assert np.all(half_widths > 0.0)


def test_pressure_refuses_half_width_below_the_fitted_range(
    reference_model: CalibratedContactModel,
) -> None:
    below = reference_model.contact_half_width_validity.min / 2.0
    with pytest.raises(OutsideValidityRangeError, match="contact_half_width"):
        reference_model.pressure(sinkage=0.05, contact_half_width=below)


def test_pressure_refuses_half_width_above_the_fitted_range(
    reference_model: CalibratedContactModel,
) -> None:
    above = reference_model.contact_half_width_validity.max * 2.0
    with pytest.raises(OutsideValidityRangeError, match="contact_half_width"):
        reference_model.pressure(sinkage=0.05, contact_half_width=above)


def test_pressure_refuses_sinkage_above_the_fitted_range(
    reference_model: CalibratedContactModel,
) -> None:
    beyond = reference_model.sinkage_validity.max * 2.0
    with pytest.raises(OutsideValidityRangeError, match="sinkage"):
        reference_model.pressure(
            sinkage=beyond,
            contact_half_width=reference_model.contact_half_width_validity.min,
        )


def test_sinkage_refuses_when_the_inverted_result_leaves_the_fitted_range(
    reference_model: CalibratedContactModel,
) -> None:
    half_width = reference_model.contact_half_width_validity.min
    just_inside = reference_model.pressure(
        sinkage=reference_model.sinkage_validity.max, contact_half_width=half_width
    )
    with pytest.raises(OutsideValidityRangeError, match="sinkage"):
        reference_model.sinkage(
            pressure=float(just_inside) * 10.0, contact_half_width=half_width
        )


def test_deformation_modulus_refuses_outside_the_fitted_range(
    reference_model: CalibratedContactModel,
) -> None:
    with pytest.raises(OutsideValidityRangeError, match="contact_half_width"):
        reference_model.deformation_modulus(
            reference_model.contact_half_width_validity.min / 2.0
        )


def test_a_single_out_of_range_array_entry_is_refused(
    reference_model: CalibratedContactModel,
) -> None:
    bounds = reference_model.sinkage_validity
    depth = np.array([bounds.min, bounds.max, bounds.max * 2.0])
    with pytest.raises(OutsideValidityRangeError, match="1 of 3 values"):
        reference_model.pressure(
            sinkage=depth,
            contact_half_width=reference_model.contact_half_width_validity.min,
        )


def test_evaluation_inside_the_fitted_range_is_permitted(
    reference_model: CalibratedContactModel,
) -> None:
    bounds = reference_model.sinkage_validity
    depth = np.linspace(bounds.min, bounds.max, 64)
    pressure = reference_model.pressure(
        sinkage=depth,
        contact_half_width=reference_model.contact_half_width_validity.max,
    )
    assert pressure.shape == depth.shape
    assert np.all(np.diff(pressure) > 0.0)


def test_extrapolating_bypasses_the_validity_range(
    reference_model: CalibratedContactModel,
) -> None:
    below = reference_model.contact_half_width_validity.min / 2.0
    assert below > reference_model.minimum_invertible_half_width()
    with pytest.raises(OutsideValidityRangeError):
        reference_model.pressure(sinkage=0.05, contact_half_width=below)
    extrapolated = reference_model.extrapolating.pressure(
        sinkage=0.05, contact_half_width=below
    )
    assert math.isfinite(float(extrapolated))


def test_extrapolating_still_honours_the_physics_guards(
    reference_model: CalibratedContactModel,
) -> None:
    floor = reference_model.minimum_invertible_half_width()
    assert 0.0 < floor < math.inf
    with pytest.raises(DegenerateContactModelError):
        reference_model.extrapolating.pressure(
            sinkage=0.05, contact_half_width=floor / 2.0
        )


def test_loaded_mappings_are_read_only(reference_soil: Soil) -> None:
    dataset = reference_soil.datasets["lim2021"]
    for mapping in (
        reference_soil.datasets,
        dataset.models,
        dataset.excluded_models,
        dataset.conditions,
        dataset.models["bekker"].parameters,
        dataset.models["bekker"].validity,
    ):
        with pytest.raises(TypeError):
            mapping["injected"] = None  # type: ignore[index]


@pytest.mark.parametrize("version", ["2", '"1"', "0"])
def test_unsupported_schema_version_is_refused(tmp_path: Path, version: str) -> None:
    path = _variant(tmp_path, [("schema_version = 1", f"schema_version = {version}")])
    with pytest.raises(SoilFileError, match="schema_version"):
        load_soil(path)


def test_missing_schema_version_is_refused(tmp_path: Path) -> None:
    path = _variant(tmp_path, [("schema_version = 1", "")])
    with pytest.raises(SoilFileError, match="schema_version"):
        load_soil(path)


def test_unexpected_key_is_refused(tmp_path: Path) -> None:
    path = _variant(
        tmp_path,
        [('kind         = "simulant"', 'kind         = "simulant"\nunexpected   = 1')],
    )
    with pytest.raises(SoilFileError, match="Material"):
        load_soil(path)


def test_renamed_model_parameter_is_refused(tmp_path: Path) -> None:
    path = _variant(
        tmp_path, [("cohesive_modulus   = { value =  -44.0554", "cohesion   = { value =  -44.0554")]
    )
    with pytest.raises(SoilFileError, match="do not match the constructor"):
        load_soil(path)


def test_half_width_that_is_not_half_the_diameter_is_refused(tmp_path: Path) -> None:
    path = _variant(
        tmp_path,
        [
            (
                "{ diameter_m = 0.070, contact_half_width_m = 0.0350",
                "{ diameter_m = 0.072, contact_half_width_m = 0.0350",
            )
        ],
    )
    with pytest.raises(SoilFileError, match="contact_half_width_m must equal"):
        load_soil(path)


def test_plate_area_inconsistent_with_geometry_is_refused(tmp_path: Path) -> None:
    path = _variant(tmp_path, [("area_m2 = 2.827433e-3", "area_m2 = 2.927433e-3")])
    with pytest.raises(SoilFileError, match="area_m2 must"):
        load_soil(path)


def test_inverted_validity_bounds_are_refused(tmp_path: Path) -> None:
    path = _variant(
        tmp_path,
        [
            (
                "contact_half_width = { min = 0.0300, max = 0.0375, units = \"m\" }",
                "contact_half_width = { min = 0.0375, max = 0.0300, units = \"m\" }",
            )
        ],
    )
    with pytest.raises(SoilFileError, match="inverted"):
        load_soil(path)


def test_missing_required_validity_quantity_is_refused(tmp_path: Path) -> None:
    path = _variant(
        tmp_path,
        [("sinkage            = { min = 0.0,    max = 0.095,  units = \"m\" }", "")],
    )
    with pytest.raises(SoilFileError, match="validity block is missing"):
        load_soil(path)


def test_verified_model_absent_from_the_registry_is_refused(tmp_path: Path) -> None:
    path = _variant(tmp_path, [('status     = "not_reproducible"', 'status     = "verified"')])
    with pytest.raises(SoilFileError, match="not registered in CONTACT_MODELS"):
        load_soil(path)


def test_invalid_toml_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "broken.toml"
    path.write_text("schema_version = = 1", encoding="utf-8")
    with pytest.raises(SoilFileError, match="not valid TOML"):
        load_soil(path)


def test_file_without_datasets_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "empty.toml"
    path.write_text(
        "schema_version = 1\n"
        'id = "empty"\n'
        "[material]\n"
        'name = "n"\n'
        'short_name = "n"\n'
        'kind = "simulant"\n'
        'analogue_for = "none"\n'
        'fidelity = "none"\n'
        'producer = "none"\n',
        encoding="utf-8",
    )
    with pytest.raises(SoilFileError, match="declares no datasets"):
        load_soil(path)
