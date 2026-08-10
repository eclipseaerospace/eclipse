# SPDX-License-Identifier: Apache-2.0
#
# biome.io.channels — load raw bevameter sensor channels and convert them.
#
# The file this reads holds instrument readings, not measurements: displacement
# in millivolts and load in millivolts per volt, exactly as the source printed
# them. Turning those into pressure and sinkage needs two calibration constants,
# the excitation voltage recorded for that test, and the plate area, and every
# one of them is a place a silent factor can enter. The conversion therefore
# lives here in tested code rather than in the data, which is the whole reason
# the data stores raw counts.
#
# Excitation is measured per test and varies by half a percent across the
# campaign. It is read from the file and never assumed, because substituting a
# nominal value produces a plausible curve that is quietly wrong.
#
# Both channels are zeroed on their own first sample, which is the convention
# the source uses: the first row of every test is the unloaded reading, so it
# becomes the origin rather than a data point.

from __future__ import annotations

import csv
import hashlib
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TypeVar

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "BevameterChannels",
    "BevameterTest",
    "Calibration",
    "ChannelsFileError",
    "Plate",
    "Quantity",
    "load_bevameter_channels",
    "pressure_from_load",
    "sinkage_from_displacement",
]

SUPPORTED_SCHEMA_VERSIONS: Final = frozenset({1})
RAW_CHANNELS_KIND: Final = "raw_bevameter_channels"
REQUIRED_COLUMNS: Final = (
    "plate_diameter_m",
    "test_id",
    "excitation_V",
    "lvdt_mV",
    "load_cell_mV_per_V",
)
NUMERIC_COLUMNS: Final = (
    "plate_diameter_m",
    "excitation_V",
    "lvdt_mV",
    "load_cell_mV_per_V",
)
MINIMUM_SAMPLES_PER_TEST: Final = 2
AREA_RELATIVE_TOLERANCE: Final = 1e-6
MILLI: Final = 1e-3

_Constructed = TypeVar("_Constructed")


class ChannelsFileError(ValueError):
    pass


def _construct(
    target: type[_Constructed], values: Mapping[str, Any], context: str
) -> _Constructed:
    try:
        return target(**values)
    except TypeError as error:
        raise ChannelsFileError(
            f"{context}: cannot build {target.__name__} from keys "
            f"{sorted(values)}: {error}"
        ) from error


@dataclass(frozen=True, slots=True)
class Quantity:
    value: float
    units: str


@dataclass(frozen=True, slots=True)
class Calibration:
    source: str
    load_cell: Quantity
    displacement: Quantity

    def __post_init__(self) -> None:
        for name, quantity in (
            ("load_cell", self.load_cell),
            ("displacement", self.displacement),
        ):
            if not np.isfinite(quantity.value) or quantity.value == 0.0:
                raise ChannelsFileError(
                    f"{name} calibration must be finite and non-zero, got "
                    f"{quantity.value}; a zero constant silently flattens every "
                    "converted channel to zero"
                )


@dataclass(frozen=True, slots=True)
class Plate:
    diameter_m: float
    contact_half_width_m: float
    area_m2: float

    def __post_init__(self) -> None:
        if not self.diameter_m > 0.0:
            raise ChannelsFileError(
                f"plate diameter must be positive, got {self.diameter_m}"
            )
        expected_half_width = self.diameter_m / 2.0
        if self.contact_half_width_m != expected_half_width:
            raise ChannelsFileError(
                f"plate of diameter {self.diameter_m} m records "
                f"contact_half_width_m {self.contact_half_width_m}, but Bekker's "
                f"length scale is the radius, so it must be {expected_half_width}. "
                "Passing a diameter halves the cohesive term and still draws a "
                "plausible curve, so this is checked rather than trusted"
            )
        expected_area = float(np.pi * expected_half_width**2)
        if abs(self.area_m2 - expected_area) > AREA_RELATIVE_TOLERANCE * expected_area:
            raise ChannelsFileError(
                f"plate of diameter {self.diameter_m} m records area_m2 "
                f"{self.area_m2}, but pi * radius^2 is {expected_area}"
            )


def sinkage_from_displacement(
    *,
    displacement_mV: NDArray[np.float64],
    excitation_V: float,
    calibration_mm: float,
) -> NDArray[np.float64]:
    zeroed = displacement_mV - displacement_mV[0]
    return np.asarray(zeroed * MILLI / excitation_V * calibration_mm * MILLI)


def pressure_from_load(
    *,
    load_mV_per_V: NDArray[np.float64],
    calibration_N: float,
    area_m2: float,
) -> NDArray[np.float64]:
    zeroed = load_mV_per_V - load_mV_per_V[0]
    return np.asarray(zeroed * calibration_N * MILLI / area_m2)


@dataclass(frozen=True, slots=True, eq=False)
class BevameterTest:
    test_id: str
    plate: Plate
    excitation_V: float
    displacement_mV: NDArray[np.float64]
    load_mV_per_V: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.displacement_mV.shape != self.load_mV_per_V.shape:
            raise ChannelsFileError(
                f"test {self.test_id!r} has {self.displacement_mV.size} "
                f"displacement samples against {self.load_mV_per_V.size} load "
                "samples; the two channels are recorded together and must match"
            )
        if self.displacement_mV.size < MINIMUM_SAMPLES_PER_TEST:
            raise ChannelsFileError(
                f"test {self.test_id!r} has {self.displacement_mV.size} samples; "
                f"at least {MINIMUM_SAMPLES_PER_TEST} are needed, because the "
                "first is consumed as the unloaded origin"
            )
        if not (np.isfinite(self.excitation_V) and self.excitation_V > 0.0):
            raise ChannelsFileError(
                f"test {self.test_id!r} records excitation {self.excitation_V} V; "
                "the displacement channel is divided by it, so it must be finite "
                "and positive"
            )

    @property
    def sample_count(self) -> int:
        return int(self.displacement_mV.size)


@dataclass(frozen=True, slots=True, eq=False)
class BevameterChannels:
    schema_version: int
    id: str
    kind: str
    calibration: Calibration
    plates: tuple[Plate, ...]
    tests: tuple[BevameterTest, ...]
    manifest_path: Path
    series_path: Path

    def test(self, test_id: str) -> BevameterTest:
        for candidate in self.tests:
            if candidate.test_id == test_id:
                return candidate
        raise ChannelsFileError(
            f"no test {test_id!r} in {self.id}; it holds "
            f"{[test.test_id for test in self.tests]}"
        )

    def sinkage_m(self, test_id: str) -> NDArray[np.float64]:
        selected = self.test(test_id)
        return sinkage_from_displacement(
            displacement_mV=selected.displacement_mV,
            excitation_V=selected.excitation_V,
            calibration_mm=self.calibration.displacement.value,
        )

    def pressure_kPa(self, test_id: str) -> NDArray[np.float64]:
        selected = self.test(test_id)
        return pressure_from_load(
            load_mV_per_V=selected.load_mV_per_V,
            calibration_N=self.calibration.load_cell.value,
            area_m2=selected.plate.area_m2,
        )

    @property
    def test_ids(self) -> tuple[str, ...]:
        return tuple(test.test_id for test in self.tests)


def _read_tests(
    path: Path, declared_columns: Sequence[str], plates: Mapping[float, Plate]
) -> tuple[BevameterTest, ...]:
    if tuple(declared_columns) != REQUIRED_COLUMNS:
        raise ChannelsFileError(
            f"{path}: manifest declares columns {list(declared_columns)}, but raw "
            f"bevameter channels must declare {list(REQUIRED_COLUMNS)}"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != REQUIRED_COLUMNS:
            raise ChannelsFileError(
                f"{path}: header is {reader.fieldnames}, expected "
                f"{list(REQUIRED_COLUMNS)}"
            )
        order: list[str] = []
        grouped: dict[str, dict[str, list[float]]] = {}
        for number, row in enumerate(reader, start=2):
            test_id = row["test_id"]
            if test_id not in grouped:
                grouped[test_id] = {name: [] for name in NUMERIC_COLUMNS}
                order.append(test_id)
            for name in NUMERIC_COLUMNS:
                cell = row[name]
                try:
                    grouped[test_id][name].append(float(cell))
                except (TypeError, ValueError) as error:
                    raise ChannelsFileError(
                        f"{path} line {number}: column {name} holds {cell!r}, "
                        "which is not a number"
                    ) from error
    if not order:
        raise ChannelsFileError(f"{path}: holds a header but no samples")

    tests: list[BevameterTest] = []
    for test_id in order:
        columns = grouped[test_id]
        for name in ("plate_diameter_m", "excitation_V"):
            distinct = set(columns[name])
            if len(distinct) != 1:
                raise ChannelsFileError(
                    f"{path}: test {test_id!r} records {len(distinct)} different "
                    f"values of {name}, {sorted(distinct)}. It is one scalar per "
                    "test, repeated per row so that this can be checked"
                )
        diameter = columns["plate_diameter_m"][0]
        if diameter not in plates:
            raise ChannelsFileError(
                f"{path}: test {test_id!r} uses plate diameter {diameter} m, "
                f"which the manifest does not describe; it lists {sorted(plates)}"
            )
        tests.append(
            BevameterTest(
                test_id=test_id,
                plate=plates[diameter],
                excitation_V=columns["excitation_V"][0],
                displacement_mV=np.array(columns["lvdt_mV"]),
                load_mV_per_V=np.array(columns["load_cell_mV_per_V"]),
            )
        )
    return tuple(tests)


def load_bevameter_channels(manifest_path: Path | str) -> BevameterChannels:
    manifest_path = Path(manifest_path)
    try:
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ChannelsFileError(f"{manifest_path}: not valid TOML: {error}") from error

    schema_version = manifest.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ChannelsFileError(
            f"{manifest_path}: schema_version {schema_version!r} is not supported, "
            f"this loader reads {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    kind = manifest.get("kind")
    if kind != RAW_CHANNELS_KIND:
        raise ChannelsFileError(
            f"{manifest_path}: kind is {kind!r}, this loader reads "
            f"{RAW_CHANNELS_KIND!r}. Converted pressure-sinkage series are read by "
            "biome.io.series instead"
        )

    context = f"{manifest_path} channels {manifest.get('id', '<unnamed>')}"
    calibration_table = dict(manifest.get("calibration", {}))
    for name in ("load_cell", "displacement"):
        calibration_table[name] = _construct(
            Quantity, calibration_table.get(name, {}), f"{context} calibration {name}"
        )
    calibration = _construct(Calibration, calibration_table, f"{context} calibration")

    plates = tuple(
        _construct(Plate, entry, f"{context} plate")
        for entry in manifest.get("apparatus", {}).get("plates", ())
    )
    if not plates:
        raise ChannelsFileError(f"{context}: apparatus.plates is empty")
    by_diameter = {plate.diameter_m: plate for plate in plates}

    series_table = manifest.get("series", {})
    series_path = manifest_path.parent / series_table.get("path", "")
    if not series_path.is_file():
        raise ChannelsFileError(
            f"{context}: series.path points at {series_path}, which does not exist"
        )
    digest = hashlib.sha256(series_path.read_bytes()).hexdigest()
    if digest != series_table.get("sha256"):
        raise ChannelsFileError(
            f"{context}: {series_path.name} has sha256 {digest}, but the manifest "
            f"records {series_table.get('sha256')!r}. Either the numbers changed "
            "without their provenance, or the provenance changed without the "
            "numbers; update the manifest deliberately"
        )

    return _construct(
        BevameterChannels,
        {
            "schema_version": schema_version,
            "id": manifest["id"],
            "kind": kind,
            "calibration": calibration,
            "plates": plates,
            "tests": _read_tests(
                series_path, series_table.get("columns", ()), by_diameter
            ),
            "manifest_path": manifest_path,
            "series_path": series_path,
        },
        context,
    )
