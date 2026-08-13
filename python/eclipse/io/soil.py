# SPDX-License-Identifier: Apache-2.0
#
# eclipse.io.soil — load curated soil parameter files into typed objects.
#
# Every table that becomes a dataclass is constructed with **, so a renamed,
# missing or unexpected key fails at load rather than being silently dropped.
# Model parameters reach the contact model the same way, so a parameter rename
# fails at construction with no mapping layer in between.
#
# Models whose status is not "verified" are excluded rather than raising, and
# recorded in Dataset.excluded_models so the omission stays visible.
#
# A loaded model refuses to evaluate outside the range its parameters were
# fitted over, including the sinkage it returns when inverting a pressure.
# Extrapolation is reachable through .extrapolating, named so that every
# deliberate departure from the fitted range is greppable.

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from eclipse._validation import first_violation
from eclipse.terramechanics import (
    CONTACT_MODELS,
    ContactModel,
    InvertibleHalfWidthRange,
)

__all__ = [
    "Apparatus",
    "CalibratedContactModel",
    "Dataset",
    "Material",
    "Measurement",
    "OutsideValidityRangeError",
    "Plate",
    "Soil",
    "SoilFileError",
    "ValidityRange",
    "load_soil",
]

SUPPORTED_SCHEMA_VERSIONS: Final = frozenset({1})
VERIFIED_STATUS: Final = "verified"
CIRCULAR_PLATE_GEOMETRY: Final = "circular_rigid_flat"
RADIUS_LENGTH_SCALE: Final = "radius"
PLATE_AREA_RELATIVE_TOLERANCE: Final = 1e-6
REQUIRED_VALIDITY_QUANTITIES: Final = ("contact_half_width", "sinkage")
APPARATUS_MEASUREMENTS: Final = (
    "penetration_rate",
    "max_normal_load",
    "max_sinkage",
    "bin_diameter",
    "bin_depth",
)

_Constructed = TypeVar("_Constructed")


class SoilFileError(ValueError):
    pass


class OutsideValidityRangeError(ValueError):
    pass


def _construct(
    target: type[_Constructed], values: Mapping[str, Any], context: str
) -> _Constructed:
    try:
        return target(**values)
    except TypeError as error:
        raise SoilFileError(
            f"{context}: cannot build {target.__name__} from keys "
            f"{sorted(values)}: {error}"
        ) from error


@dataclass(frozen=True, slots=True)
class Measurement:
    value: float
    units: str


@dataclass(frozen=True, slots=True)
class ValidityRange:
    min: float
    max: float
    units: str

    def __post_init__(self) -> None:
        if not (math.isfinite(self.min) and math.isfinite(self.max)):
            raise SoilFileError(
                f"validity bounds must be finite, got [{self.min}, {self.max}]"
            )
        if self.min > self.max:
            raise SoilFileError(
                f"validity bounds are inverted, min {self.min} exceeds max {self.max}"
            )

    def violations(self, values: NDArray[np.float64]) -> NDArray[np.bool_]:
        return np.asarray(~((values >= self.min) & (values <= self.max)))


@dataclass(frozen=True, slots=True)
class Plate:
    diameter_m: float
    contact_half_width_m: float
    area_m2: float


@dataclass(frozen=True, slots=True)
class Apparatus:
    method: str
    control_mode: str
    penetration_rate: Measurement
    max_normal_load: Measurement
    max_sinkage: Measurement
    n_tests: int
    bin_diameter: Measurement
    bin_depth: Measurement
    plate_geometry: str
    plate_length_scale: str
    plates: tuple[Plate, ...]

    def __post_init__(self) -> None:
        if not self.plates:
            raise SoilFileError("apparatus reports no plates")
        for plate in self.plates:
            self._require_geometry_invariants(plate)

    def _require_geometry_invariants(self, plate: Plate) -> None:
        if self.plate_length_scale == RADIUS_LENGTH_SCALE:
            if plate.contact_half_width_m != plate.diameter_m / 2.0:
                raise SoilFileError(
                    f"plate_length_scale is {self.plate_length_scale!r}, so "
                    f"contact_half_width_m must equal diameter_m/2, but "
                    f"{plate.contact_half_width_m} != {plate.diameter_m / 2.0}"
                )
        if self.plate_geometry == CIRCULAR_PLATE_GEOMETRY:
            expected_area = math.pi * plate.contact_half_width_m**2
            if not math.isclose(
                plate.area_m2, expected_area, rel_tol=PLATE_AREA_RELATIVE_TOLERANCE
            ):
                raise SoilFileError(
                    f"plate_geometry is {self.plate_geometry!r}, so area_m2 must "
                    f"equal pi*contact_half_width_m^2 = {expected_area!r}, but "
                    f"{plate.area_m2!r} differs by more than "
                    f"{PLATE_AREA_RELATIVE_TOLERANCE} relative"
                )

    @property
    def contact_half_widths(self) -> NDArray[np.float64]:
        return np.array(
            [plate.contact_half_width_m for plate in self.plates], dtype=np.float64
        )


@dataclass(frozen=True, slots=True)
class CalibratedContactModel:
    id: str
    status: str
    fit_method: str
    parameters: Mapping[str, float]
    validity: Mapping[str, ValidityRange]
    quality: Mapping[str, Any]
    verification_cases: tuple[Mapping[str, float], ...]
    extrapolating: ContactModel

    @property
    def contact_half_width_validity(self) -> ValidityRange:
        return self.validity["contact_half_width"]

    @property
    def sinkage_validity(self) -> ValidityRange:
        return self.validity["sinkage"]

    def _require_within_validity(
        self, values: ArrayLike, quantity: str
    ) -> NDArray[np.float64]:
        array = np.asarray(values, dtype=np.float64)
        bounds = self.validity[quantity]
        violations = bounds.violations(array)
        if violations.any():
            count, total, first = first_violation(violations, array)
            raise OutsideValidityRangeError(
                f"{self.id}: {quantity} must lie within [{bounds.min}, "
                f"{bounds.max}] {bounds.units}, the range these parameters were "
                f"fitted over; {count} of {total} values violate this, the first "
                f"being {first}. Evaluate through .extrapolating to leave the "
                "fitted range deliberately"
            )
        return array

    def invertible_half_width_range(self) -> InvertibleHalfWidthRange:
        return self.extrapolating.invertible_half_width_range()

    def minimum_invertible_half_width(self) -> float:
        return self.extrapolating.minimum_invertible_half_width()

    def deformation_modulus(
        self, contact_half_width: ArrayLike
    ) -> NDArray[np.float64]:
        return self.extrapolating.deformation_modulus(
            self._require_within_validity(contact_half_width, "contact_half_width")
        )

    def pressure(
        self, *, sinkage: ArrayLike, contact_half_width: ArrayLike
    ) -> NDArray[np.float64]:
        return self.extrapolating.pressure(
            sinkage=self._require_within_validity(sinkage, "sinkage"),
            contact_half_width=self._require_within_validity(
                contact_half_width, "contact_half_width"
            ),
        )

    def sinkage(
        self, *, pressure: ArrayLike, contact_half_width: ArrayLike
    ) -> NDArray[np.float64]:
        depth = self.extrapolating.sinkage(
            pressure=pressure,
            contact_half_width=self._require_within_validity(
                contact_half_width, "contact_half_width"
            ),
        )
        return self._require_within_validity(depth, "sinkage")


@dataclass(frozen=True, slots=True)
class Material:
    name: str
    short_name: str
    kind: str
    analogue_for: str
    fidelity: str
    producer: str


@dataclass(frozen=True, slots=True)
class Dataset:
    id: str
    title: str
    authors: tuple[str, ...]
    year: int
    journal: str
    volume: int
    issue: int
    pages: str
    doi: str
    license: str
    tables: tuple[str, ...]
    accessed: date
    conditions: Mapping[str, Any]
    apparatus: Apparatus
    models: Mapping[str, CalibratedContactModel]
    excluded_models: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Soil:
    schema_version: int
    id: str
    material: Material
    datasets: Mapping[str, Dataset]
    source_path: Path


def _build_apparatus(table: Mapping[str, Any], context: str) -> Apparatus:
    values = dict(table)
    values["plates"] = tuple(
        _construct(Plate, plate, f"{context} plate") for plate in values.get("plates", ())
    )
    for name in APPARATUS_MEASUREMENTS:
        if name in values:
            values[name] = _construct(Measurement, values[name], f"{context} {name}")
    return _construct(Apparatus, values, context)


def _build_validity(
    table: Mapping[str, Any], context: str
) -> Mapping[str, ValidityRange]:
    ranges = {
        quantity: _construct(ValidityRange, bounds, f"{context} validity {quantity}")
        for quantity, bounds in table.items()
    }
    missing = [name for name in REQUIRED_VALIDITY_QUANTITIES if name not in ranges]
    if missing:
        raise SoilFileError(
            f"{context}: validity block is missing {missing}, which the loader "
            "needs to refuse extrapolation"
        )
    return MappingProxyType(ranges)


def _build_model(table: Mapping[str, Any], context: str) -> CalibratedContactModel:
    model_id = table["id"]
    if model_id not in CONTACT_MODELS:
        raise SoilFileError(
            f"{context}: model {model_id!r} is marked {VERIFIED_STATUS!r} but is "
            f"not registered in CONTACT_MODELS, which holds {sorted(CONTACT_MODELS)}"
        )
    parameters = {
        name: entry["value"] for name, entry in table["parameters"].items()
    }
    try:
        model = CONTACT_MODELS[model_id](**parameters)
    except TypeError as error:
        raise SoilFileError(
            f"{context}: parameter keys {sorted(parameters)} do not match the "
            f"constructor of {model_id!r}: {error}"
        ) from error
    return CalibratedContactModel(
        id=model_id,
        status=table["status"],
        fit_method=table["fit_method"],
        parameters=MappingProxyType(parameters),
        validity=_build_validity(table["validity"], context),
        quality=MappingProxyType(dict(table.get("quality", {}))),
        verification_cases=tuple(
            MappingProxyType(dict(case))
            for case in table.get("verification", {}).get("cases", ())
        ),
        extrapolating=model,
    )


def _build_dataset(table: Mapping[str, Any], context: str) -> Dataset:
    values = dict(table)
    dataset_id = values.get("id", "<unnamed>")
    context = f"{context} dataset {dataset_id}"
    specifications = values.pop("model", ())
    models: dict[str, CalibratedContactModel] = {}
    excluded: dict[str, str] = {}
    for specification in specifications:
        if specification["status"] != VERIFIED_STATUS:
            excluded[specification["id"]] = specification["status"]
            continue
        model = _build_model(specification, f"{context} model {specification['id']}")
        models[model.id] = model
    values["authors"] = tuple(values.get("authors", ()))
    values["tables"] = tuple(values.get("tables", ()))
    values["conditions"] = MappingProxyType(dict(values.get("conditions", {})))
    values["apparatus"] = _build_apparatus(values["apparatus"], context)
    values["models"] = MappingProxyType(models)
    values["excluded_models"] = MappingProxyType(excluded)
    return _construct(Dataset, values, context)


def load_soil(path: Path | str) -> Soil:
    source_path = Path(path)
    try:
        table = tomllib.loads(source_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise SoilFileError(f"{source_path}: not valid TOML: {error}") from error

    schema_version = table.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise SoilFileError(
            f"{source_path}: schema_version {schema_version!r} is not supported, "
            f"this loader reads {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )

    values = dict(table)
    soil_id = values.get("id", "<unnamed>")
    context = f"{source_path} soil {soil_id}"
    values["material"] = _construct(
        Material, values.get("material", {}), f"{context} material"
    )
    datasets = [
        _build_dataset(dataset, context) for dataset in values.pop("dataset", ())
    ]
    if not datasets:
        raise SoilFileError(f"{context}: file declares no datasets")
    values["datasets"] = MappingProxyType({dataset.id: dataset for dataset in datasets})
    values["source_path"] = source_path
    return _construct(Soil, values, context)
