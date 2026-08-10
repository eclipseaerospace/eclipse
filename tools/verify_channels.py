#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# tools/verify_channels.py — check and regenerate conversion blocks in raw
# bevameter channel files.
#
# Independent by construction: standard library only, nothing from biome and no
# numpy. A verifier that imports the code it checks proves only that the code
# agrees with itself, so the arithmetic below is typed from the conversion the
# manifest documents and evaluated one scalar at a time.
#
# The conversion has five places a silent factor can enter — two calibration
# constants, the per-test excitation voltage, the plate area, and the choice of
# origin — and none of them announces itself in the result. A wrong constant
# produces a smooth, plausible pressure-sinkage curve. That is what this checks.
#
# --check, the default, recomputes each test's converted extremes and compares
# against the stored block. --write regenerates that block, editing text in
# place so comments and formatting survive, then re-parses the result and
# refuses to save if the edit did not produce exactly the intended values.

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

CONVERSION_HEADER: Final = "[[conversion_check]]"
MILLI: Final = 1e-3
RELATIVE_TOLERANCE: Final = 1e-12


@dataclass(frozen=True, slots=True)
class ConvertedTest:
    test_id: str
    sample_count: int
    maximum_sinkage_m: float
    maximum_pressure_kPa: float

    def matches(self, other: ConvertedTest) -> bool:
        if self.test_id != other.test_id or self.sample_count != other.sample_count:
            return False
        return all(
            math.isclose(mine, theirs, rel_tol=RELATIVE_TOLERANCE, abs_tol=0.0)
            for mine, theirs in (
                (self.maximum_sinkage_m, other.maximum_sinkage_m),
                (self.maximum_pressure_kPa, other.maximum_pressure_kPa),
            )
        )


def sinkage_m(
    displacement_mV: Sequence[float], excitation_V: float, calibration_mm: float
) -> list[float]:
    origin = displacement_mV[0]
    return [
        (sample - origin) * MILLI / excitation_V * calibration_mm * MILLI
        for sample in displacement_mV
    ]


def pressure_kPa(
    load_mV_per_V: Sequence[float], calibration_N: float, area_m2: float
) -> list[float]:
    origin = load_mV_per_V[0]
    return [(sample - origin) * calibration_N * MILLI / area_m2 for sample in load_mV_per_V]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _convert(table: Mapping[str, Any], manifest_path: Path) -> list[ConvertedTest]:
    calibration = table["calibration"]
    load_constant = float(calibration["load_cell"]["value"])
    displacement_constant = float(calibration["displacement"]["value"])
    area_by_diameter = {
        float(plate["diameter_m"]): float(plate["area_m2"])
        for plate in table["apparatus"]["plates"]
    }

    series = table["series"]
    series_path = manifest_path.parent / series["path"]
    digest = hashlib.sha256(series_path.read_bytes()).hexdigest()
    if digest != series["sha256"]:
        raise SystemExit(
            f"{series_path}: sha256 {digest} does not match the manifest's "
            f"{series['sha256']}"
        )

    grouped: dict[str, list[dict[str, str]]] = {}
    order: list[str] = []
    for row in _read_rows(series_path):
        if row["test_id"] not in grouped:
            grouped[row["test_id"]] = []
            order.append(row["test_id"])
        grouped[row["test_id"]].append(row)

    converted: list[ConvertedTest] = []
    for test_id in order:
        rows = grouped[test_id]
        diameter = float(rows[0]["plate_diameter_m"])
        excitation = float(rows[0]["excitation_V"])
        depth = sinkage_m(
            [float(row["lvdt_mV"]) for row in rows], excitation, displacement_constant
        )
        load = pressure_kPa(
            [float(row["load_cell_mV_per_V"]) for row in rows],
            load_constant,
            area_by_diameter[diameter],
        )
        converted.append(
            ConvertedTest(test_id, len(rows), max(depth), max(load))
        )
    return converted


def _stored(table: Mapping[str, Any]) -> list[ConvertedTest]:
    return [
        ConvertedTest(
            entry["test_id"],
            int(entry["sample_count"]),
            float(entry["maximum_sinkage_m"]),
            float(entry["maximum_pressure_kPa"]),
        )
        for entry in table.get("conversion_check", ())
    ]


def _render(converted: Sequence[ConvertedTest]) -> str:
    blocks = []
    for entry in converted:
        blocks.append(
            f"{CONVERSION_HEADER}\n"
            f'test_id              = "{entry.test_id}"\n'
            f"sample_count         = {entry.sample_count}\n"
            f"maximum_sinkage_m    = {entry.maximum_sinkage_m!r}\n"
            f"maximum_pressure_kPa = {entry.maximum_pressure_kPa!r}\n"
        )
    return "\n".join(blocks)


def _replace_blocks(text: str, converted: Sequence[ConvertedTest]) -> str:
    rendered = _render(converted)
    marker = text.find(CONVERSION_HEADER)
    if marker == -1:
        separator = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
        return text + separator + rendered
    tail = text.find("\n[", marker + len(CONVERSION_HEADER))
    while tail != -1 and text.startswith(f"\n{CONVERSION_HEADER}", tail):
        tail = text.find("\n[", tail + 2)
    return text[:marker] + rendered + ("" if tail == -1 else text[tail + 1 :])


def _process(path: Path, write: bool) -> int:
    text = path.read_text(encoding="utf-8")
    table = tomllib.loads(text)
    converted = _convert(table, path)

    if write:
        updated = _replace_blocks(text, converted)
        round_tripped = _stored(tomllib.loads(updated))
        if len(round_tripped) != len(converted) or not all(
            mine.matches(theirs) for mine, theirs in zip(converted, round_tripped)
        ):
            raise SystemExit(f"{path}: rewriting did not round trip, refusing to save")
        path.write_text(updated, encoding="utf-8")
        print(f"{path}: wrote {len(converted)} conversion checks")
        return 0

    stored = _stored(table)
    if not stored:
        print(f"{path}: no conversion_check blocks; run with --write", file=sys.stderr)
        return 1
    if len(stored) != len(converted):
        print(
            f"{path}: {len(stored)} stored checks against {len(converted)} tests",
            file=sys.stderr,
        )
        return 1
    failures = [
        theirs.test_id
        for mine, theirs in zip(converted, stored)
        if not mine.matches(theirs)
    ]
    for mine, theirs in zip(converted, stored):
        if not mine.matches(theirs):
            print(
                f"  {theirs.test_id}: stored sinkage {theirs.maximum_sinkage_m} "
                f"pressure {theirs.maximum_pressure_kPa}, recomputed "
                f"{mine.maximum_sinkage_m} {mine.maximum_pressure_kPa}",
                file=sys.stderr,
            )
    if failures:
        print(f"{path}: {len(failures)} of {len(stored)} checks disagree", file=sys.stderr)
        return 1
    print(f"{path}: {len(stored)} conversion checks agree")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check raw bevameter channel files by converting them independently "
            "of biome."
        )
    )
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument(
        "--write",
        action="store_true",
        help="regenerate the conversion_check blocks instead of checking them",
    )
    arguments = parser.parse_args(argv)
    paths = arguments.paths or sorted(
        (Path(__file__).resolve().parents[1] / "data" / "literature").glob(
            "*-raw-channels.toml"
        )
    )
    if not paths:
        print("no raw channel files found", file=sys.stderr)
        return 1
    return max(_process(path, arguments.write) for path in paths)


if __name__ == "__main__":
    raise SystemExit(main())
