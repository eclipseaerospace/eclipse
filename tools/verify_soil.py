#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# tools/verify_soil.py — check and regenerate verification blocks in soil files.
#
# Independent by construction: standard library only, nothing from eclipse. A
# verifier that imports the code it checks proves only that the code agrees
# with itself. The equations below are typed from the cited papers and
# evaluated with scalar math.pow, so even the numerical path shares nothing
# with eclipse.terramechanics.
#
# --check, the default, recomputes the pressure at each stored case's own
# operating point and compares. --write regenerates the case set across the
# corners and interior of each model's fitted range, editing text in place so
# that comments and formatting survive, then re-parses the result and refuses
# to save if the edit did not produce exactly the intended cases.
#
# References
#   Bekker MG (1956) Theory of Land Locomotion. University of Michigan Press.
#   Reece AR (1965) Principles of soil-vehicle mechanics. Proceedings of the
#     Institution of Mechanical Engineers: Automobile Division 180(1), 45-66.

from __future__ import annotations

import argparse
import math
import sys
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_SOIL_DIRECTORY: Final = REPOSITORY_ROOT / "data" / "soils"
VERIFIED_STATUS: Final = "verified"
RELATIVE_TOLERANCE: Final = 1e-9
SINKAGE_FRACTIONS: Final = (0.25, 0.5, 0.75, 1.0)
VERIFICATION_HEADER: Final = "[dataset.model.verification]"
CASES_OPENING: Final = "cases = ["


def bekker_pressure_kPa(
    parameters: Mapping[str, float], contact_half_width_m: float, sinkage_m: float
) -> float:
    cohesive_modulus = parameters["cohesive_modulus"]
    frictional_modulus = parameters["frictional_modulus"]
    sinkage_exponent = parameters["sinkage_exponent"]
    return (cohesive_modulus / contact_half_width_m + frictional_modulus) * math.pow(
        sinkage_m, sinkage_exponent
    )


def reece_pressure_kPa(
    parameters: Mapping[str, float], contact_half_width_m: float, sinkage_m: float
) -> float:
    cohesive_modulus = parameters["cohesive_modulus"]
    frictional_modulus = parameters["frictional_modulus"]
    sinkage_exponent = parameters["sinkage_exponent"]
    return (cohesive_modulus + contact_half_width_m * frictional_modulus) * math.pow(
        sinkage_m / contact_half_width_m, sinkage_exponent
    )


PressureEquation = Callable[[Mapping[str, float], float, float], float]

PRESSURE_EQUATIONS: Final[Mapping[str, PressureEquation]] = {
    "bekker": bekker_pressure_kPa,
    "reece": reece_pressure_kPa,
}


@dataclass(frozen=True, slots=True)
class VerificationCase:
    contact_half_width_m: float
    sinkage_m: float
    pressure_kPa: float

    def as_toml(self) -> str:
        return (
            f"  {{ contact_half_width_m = {self.contact_half_width_m!r}, "
            f"sinkage_m = {self.sinkage_m!r}, "
            f"pressure_kPa = {self.pressure_kPa!r}, "
            f"rel_tol = {RELATIVE_TOLERANCE!r} }},"
        )


@dataclass(frozen=True, slots=True)
class ModelReport:
    model_id: str
    outcome: str
    detail: str


def _parameters(specification: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: float(entry["value"])
        for name, entry in specification["parameters"].items()
    }


def _plates_or_validity_span(
    dataset: Mapping[str, Any]
) -> Sequence[Mapping[str, float]]:
    """Half-widths to generate cases at.

    A plate campaign supplies them directly. A soil measured in situ has no
    plates, so the ends and middle of its own declared half-width validity range
    are used instead: the corners are where a wrong parameter shows first.
    """
    apparatus = dataset.get("apparatus")
    if apparatus is not None and "plates" in apparatus:
        return list(apparatus["plates"])
    spans = {
        model["id"]: model["validity"]["contact_half_width"]
        for model in dataset["model"]
        if "validity" in model
    }
    if not spans:
        return []
    span = next(iter(spans.values()))
    low, high = float(span["min"]), float(span["max"])
    low = high / 8.0 if low <= 0.0 else low
    return [
        {"contact_half_width_m": low},
        {"contact_half_width_m": 0.5 * (low + high)},
        {"contact_half_width_m": high},
    ]


def _generate_cases(
    specification: Mapping[str, Any], plates: Sequence[Mapping[str, float]]
) -> list[VerificationCase]:
    equation = PRESSURE_EQUATIONS[specification["id"]]
    parameters = _parameters(specification)
    maximum_sinkage = float(specification["validity"]["sinkage"]["max"])
    return [
        VerificationCase(
            contact_half_width_m=plate["contact_half_width_m"],
            sinkage_m=maximum_sinkage * fraction,
            pressure_kPa=equation(
                parameters, plate["contact_half_width_m"], maximum_sinkage * fraction
            ),
        )
        for plate in plates
        for fraction in SINKAGE_FRACTIONS
    ]


def _check_model(specification: Mapping[str, Any]) -> ModelReport:
    model_id = specification["id"]
    if specification["status"] != VERIFIED_STATUS:
        return ModelReport(model_id, "skip", f"status = {specification['status']}")
    if model_id not in PRESSURE_EQUATIONS:
        return ModelReport(
            model_id,
            "fail",
            f"marked {VERIFIED_STATUS} but no equation is typed here; this tool "
            f"knows {sorted(PRESSURE_EQUATIONS)}",
        )
    cases = specification.get("verification", {}).get("cases", [])
    if not cases:
        return ModelReport(model_id, "fail", f"marked {VERIFIED_STATUS} but has no cases")

    parameters = _parameters(specification)
    equation = PRESSURE_EQUATIONS[model_id]
    disagreements: list[str] = []
    for case in cases:
        computed = equation(parameters, case["contact_half_width_m"], case["sinkage_m"])
        stored = case["pressure_kPa"]
        if not math.isclose(computed, stored, rel_tol=case["rel_tol"], abs_tol=0.0):
            relative = abs(computed - stored) / abs(stored) if stored else math.inf
            disagreements.append(
                f"b={case['contact_half_width_m']!r} z={case['sinkage_m']!r}: "
                f"stored {stored!r}, computed {computed!r}, relative {relative:.3e} "
                f"exceeds rel_tol {case['rel_tol']!r}"
            )
    if disagreements:
        return ModelReport(model_id, "fail", "; ".join(disagreements))
    return ModelReport(model_id, "ok", f"{len(cases)} cases")


def _replace_cases(text: str, generated: Mapping[str, list[VerificationCase]]) -> str:
    lines = text.splitlines()
    rewritten: list[str] = []
    current_model_id: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.startswith("id ") and stripped.split("=", 1)[0].strip() == "id":
            current_model_id = stripped.split("=", 1)[1].strip().strip('"')
        if stripped == VERIFICATION_HEADER and current_model_id in generated:
            model_id = str(current_model_id)
            rewritten.append(line)
            index += 1
            while index < len(lines) and lines[index].strip() != CASES_OPENING:
                rewritten.append(lines[index])
                index += 1
            if index >= len(lines):
                raise SystemExit(
                    f"{VERIFICATION_HEADER} for {model_id} has no {CASES_OPENING}"
                )
            rewritten.append(lines[index])
            index += 1
            while index < len(lines) and lines[index].strip() != "]":
                index += 1
            if index >= len(lines):
                raise SystemExit(f"unterminated {CASES_OPENING} for {model_id}")
            rewritten.extend(case.as_toml() for case in generated[model_id])
            rewritten.append(lines[index])
            index += 1
            continue
        rewritten.append(line)
        index += 1
    return "\n".join(rewritten) + "\n"


def _write_soil(path: Path, table: Mapping[str, Any], text: str) -> list[ModelReport]:
    reports: list[ModelReport] = []
    generated: dict[str, list[VerificationCase]] = {}
    missing_block: list[str] = []
    for dataset in table["dataset"]:
        plates = _plates_or_validity_span(dataset)
        for specification in dataset["model"]:
            model_id = specification["id"]
            if specification["status"] != VERIFIED_STATUS:
                reports.append(
                    ModelReport(model_id, "skip", f"status = {specification['status']}")
                )
                continue
            if model_id not in PRESSURE_EQUATIONS:
                reports.append(
                    ModelReport(model_id, "fail", "no equation is typed here")
                )
                continue
            generated[model_id] = _generate_cases(specification, plates)
            if "verification" not in specification:
                missing_block.append(model_id)

    updated = _replace_cases(text, generated)
    verified_table = tomllib.loads(updated)
    for dataset in verified_table["dataset"]:
        for specification in dataset["model"]:
            model_id = specification["id"]
            if model_id not in generated:
                continue
            if model_id in missing_block:
                continue
            round_tripped = [
                VerificationCase(
                    case["contact_half_width_m"], case["sinkage_m"], case["pressure_kPa"]
                )
                for case in specification["verification"]["cases"]
            ]
            if round_tripped != generated[model_id]:
                raise SystemExit(
                    f"{path}: rewriting {model_id} did not round trip, refusing to save"
                )
            reports.append(
                ModelReport(model_id, "written", f"{len(round_tripped)} cases")
            )

    for model_id in missing_block:
        block = "\n".join(
            [f"\n{VERIFICATION_HEADER}", CASES_OPENING]
            + [case.as_toml() for case in generated[model_id]]
            + ["]"]
        )
        reports.append(
            ModelReport(
                model_id,
                "fail",
                f"has no {VERIFICATION_HEADER}; add this block under it:\n{block}",
            )
        )

    if not missing_block:
        path.write_text(updated, encoding="utf-8")
    return reports


def _process(path: Path, write: bool) -> list[ModelReport]:
    text = path.read_text(encoding="utf-8")
    table = tomllib.loads(text)
    if write:
        return _write_soil(path, table, text)
    return [
        _check_model(specification)
        for dataset in table["dataset"]
        for specification in dataset["model"]
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check or regenerate the verification blocks in soil files, using "
            "equations typed independently of eclipse."
        )
    )
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument(
        "--write",
        action="store_true",
        help="regenerate cases across each model's fitted range instead of checking",
    )
    arguments = parser.parse_args(argv)
    paths = arguments.paths or sorted(DEFAULT_SOIL_DIRECTORY.glob("*.toml"))
    if not paths:
        print(f"no soil files found under {DEFAULT_SOIL_DIRECTORY}", file=sys.stderr)
        return 1

    failed = False
    for path in paths:
        print(path)
        for report in _process(path, arguments.write):
            marker = {"ok": "ok", "written": "written", "skip": "skipped"}.get(
                report.outcome, "FAIL"
            )
            print(f"  {report.model_id:30s} {marker:8s} {report.detail}")
            failed |= report.outcome == "fail"
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
