# SPDX-License-Identifier: Apache-2.0
#
# eclipse.io.series — load digitized measurement series from a manifest and CSV.
#
# Provenance lives in TOML and the numbers live in CSV, following the split the
# project uses everywhere else. The manifest names the source figure, the tool
# and method used to digitize it, the axis calibration the digitizer was given,
# the per-axis uncertainty implied by that calibration, and the SHA-256 of the
# CSV. The loader verifies that digest, so the pair is tamper-evident and an
# edit to either file without the other fails loudly.
#
# Points read from a figure are estimates, not measurements. The uncertainty is
# recorded rather than inferred, and it is deliberately not fed back into the
# fit weighting: a fit is compared against published parameters, so it has to
# use the published method, not a better one.

from __future__ import annotations

import csv
import hashlib
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, TypeVar

import numpy as np

from eclipse.fitting import PressureSinkageObservations

__all__ = [
    "AxisCalibration",
    "Digitization",
    "PUBLISHED_CURVE_KIND",
    "PressureSinkageSeries",
    "SeriesFileError",
    "Source",
    "load_pressure_sinkage_series",
]

SUPPORTED_SCHEMA_VERSIONS: Final = frozenset({1})
PRESSURE_SINKAGE_KIND: Final = "pressure_sinkage"
PUBLISHED_CURVE_KIND: Final = "published_curve"
SUPPORTED_KINDS: Final = frozenset({PRESSURE_SINKAGE_KIND, PUBLISHED_CURVE_KIND})
REQUIRED_COLUMNS: Final = (
    "contact_half_width_m",
    "test_id",
    "sinkage_m",
    "pressure_kPa",
)
NUMERIC_COLUMNS: Final = ("contact_half_width_m", "sinkage_m", "pressure_kPa")

_Constructed = TypeVar("_Constructed")


class SeriesFileError(ValueError):
    pass


def _construct(
    target: type[_Constructed], values: Mapping[str, Any], context: str
) -> _Constructed:
    try:
        return target(**values)
    except TypeError as error:
        raise SeriesFileError(
            f"{context}: cannot build {target.__name__} from keys "
            f"{sorted(values)}: {error}"
        ) from error


@dataclass(frozen=True, slots=True)
class Source:
    doi: str
    figure: str
    license: str


@dataclass(frozen=True, slots=True)
class AxisCalibration:
    minimum: float
    maximum: float
    units: str
    scale: str

    def __post_init__(self) -> None:
        if self.minimum >= self.maximum:
            raise SeriesFileError(
                f"axis calibration is not increasing, {self.minimum} >= {self.maximum}"
            )
        if self.scale != "linear":
            raise SeriesFileError(
                f"axis scale {self.scale!r} is not supported; a per-axis scalar "
                "uncertainty only describes a linear axis, so a logarithmic one "
                "needs a per-point uncertainty column that this schema does not "
                "yet carry"
            )


@dataclass(frozen=True, slots=True)
class Digitization:
    tool: str
    method: str
    operator: str
    performed_on: date
    sinkage_uncertainty_m: float
    pressure_uncertainty_kPa: float
    axis_calibration: Mapping[str, AxisCalibration]

    def __post_init__(self) -> None:
        for name, uncertainty in (
            ("sinkage_uncertainty_m", self.sinkage_uncertainty_m),
            ("pressure_uncertainty_kPa", self.pressure_uncertainty_kPa),
        ):
            if not uncertainty > 0.0:
                raise SeriesFileError(
                    f"{name} must be positive; points read off a figure always "
                    f"carry uncertainty, got {uncertainty}"
                )


@dataclass(frozen=True, slots=True, eq=False)
class PressureSinkageSeries:
    schema_version: int
    id: str
    kind: str
    source: Source
    digitization: Digitization
    observations: PressureSinkageObservations
    test_ids: tuple[str, ...]
    manifest_path: Path
    series_path: Path

    def __post_init__(self) -> None:
        if len(self.test_ids) != self.observations.count:
            raise SeriesFileError(
                f"{len(self.test_ids)} test identifiers for "
                f"{self.observations.count} observations"
            )

    @property
    def distinct_tests(self) -> int:
        return len(set(self.test_ids))

    @property
    def contact_half_widths(self) -> Sequence[float]:
        return [
            float(half_width)
            for half_width in self.observations.contact_half_widths
        ]


def _read_observations(
    path: Path, declared_columns: Sequence[str]
) -> tuple[PressureSinkageObservations, tuple[str, ...]]:
    if tuple(declared_columns) != REQUIRED_COLUMNS:
        raise SeriesFileError(
            f"{path}: manifest declares columns {list(declared_columns)}, but a "
            f"pressure-sinkage series must declare {list(REQUIRED_COLUMNS)}"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != REQUIRED_COLUMNS:
            raise SeriesFileError(
                f"{path}: header is {reader.fieldnames}, expected "
                f"{list(REQUIRED_COLUMNS)}"
            )
        columns: dict[str, list[float]] = {name: [] for name in NUMERIC_COLUMNS}
        test_ids: list[str] = []
        for number, row in enumerate(reader, start=2):
            test_ids.append(row["test_id"])
            for name in NUMERIC_COLUMNS:
                cell = row[name]
                try:
                    columns[name].append(float(cell))
                except (TypeError, ValueError) as error:
                    raise SeriesFileError(
                        f"{path} line {number}: column {name} holds {cell!r}, "
                        "which is not a number"
                    ) from error
    if not columns[NUMERIC_COLUMNS[0]]:
        raise SeriesFileError(f"{path}: holds a header but no observations")
    try:
        observations = PressureSinkageObservations(
            contact_half_width_m=np.array(columns["contact_half_width_m"]),
            sinkage_m=np.array(columns["sinkage_m"]),
            pressure_kPa=np.array(columns["pressure_kPa"]),
        )
    except ValueError as error:
        raise SeriesFileError(f"{path}: {error}") from error
    return observations, tuple(test_ids)


def load_pressure_sinkage_series(manifest_path: Path | str) -> PressureSinkageSeries:
    manifest_path = Path(manifest_path)
    try:
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise SeriesFileError(f"{manifest_path}: not valid TOML: {error}") from error

    schema_version = manifest.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise SeriesFileError(
            f"{manifest_path}: schema_version {schema_version!r} is not supported, "
            f"this loader reads {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    kind = manifest.get("kind")
    if kind not in SUPPORTED_KINDS:
        raise SeriesFileError(
            f"{manifest_path}: kind is {kind!r}, this loader reads "
            f"{sorted(SUPPORTED_KINDS)}. A published_curve series traces a fitted "
            "model and must never be fitted to, only compared against"
        )

    context = f"{manifest_path} series {manifest.get('id', '<unnamed>')}"
    digitization_table = dict(manifest.get("digitization", {}))
    digitization_table["axis_calibration"] = MappingProxyType(
        {
            axis: _construct(
                AxisCalibration, bounds, f"{context} axis_calibration {axis}"
            )
            for axis, bounds in digitization_table.pop("axis_calibration", {}).items()
        }
    )

    series_table = manifest.get("series", {})
    series_path = manifest_path.parent / series_table.get("path", "")
    if not series_path.is_file():
        raise SeriesFileError(
            f"{context}: series.path points at {series_path}, which does not exist"
        )
    digest = hashlib.sha256(series_path.read_bytes()).hexdigest()
    if digest != series_table.get("sha256"):
        raise SeriesFileError(
            f"{context}: {series_path.name} has sha256 {digest}, but the manifest "
            f"records {series_table.get('sha256')!r}. Either the numbers changed "
            "without their provenance, or the provenance changed without the "
            "numbers; update the manifest deliberately"
        )

    observations, test_ids = _read_observations(
        series_path, series_table.get("columns", ())
    )
    return _construct(
        PressureSinkageSeries,
        {
            "schema_version": schema_version,
            "id": manifest["id"],
            "kind": kind,
            "source": _construct(
                Source, manifest.get("source", {}), f"{context} source"
            ),
            "digitization": _construct(
                Digitization, digitization_table, f"{context} digitization"
            ),
            "observations": observations,
            "test_ids": test_ids,
            "manifest_path": manifest_path,
            "series_path": series_path,
        },
        context,
    )
