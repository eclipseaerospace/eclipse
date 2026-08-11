# SPDX-License-Identifier: Apache-2.0
#
# Tests for calibration/contact/pressure_sinkage.py, KLS-1 campaign.
#
# Run as a subprocess rather than imported, because it is a script and its entry
# point is what the repository invokes to regenerate the committed outputs. A
# figure that no longer regenerates is a broken result even when every library
# test passes.
#
# The report is asserted to be deterministic. A generated artifact that changes
# on every run cannot be committed usefully, because a diff would stop meaning
# that a result moved.

from __future__ import annotations

import hashlib
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Final

import pytest

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
SCRIPT_PATH: Final = (
    REPOSITORY_ROOT / "calibration" / "contact" / "pressure_sinkage.py"
)
CAMPAIGN: Final = "kls1"
COMMITTED_FIGURE: Final = (
    REPOSITORY_ROOT / "calibration" / "contact" / "figures" / "kls1-pressure-sinkage.png"
)
COMMITTED_REPORT: Final = (
    REPOSITORY_ROOT / "calibration" / "contact" / "results" / "kls1-pressure-sinkage.toml"
)
SOIL_PATH: Final = REPOSITORY_ROOT / "data" / "soils" / "kls1.toml"
EXPECTED_MODELS: Final = ("bekker", "reece")


def _run(*arguments: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable, str(SCRIPT_PATH),
            "--campaign", CAMPAIGN, *map(str, arguments),
        ],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
    )


def _generate(tmp_path: Path, *extra: str | Path) -> dict[str, Any]:
    figure = tmp_path / "figure.png"
    report = tmp_path / "report.toml"
    completed = _run(
        f"--{CAMPAIGN}-figure", figure, f"--{CAMPAIGN}-report", report, *extra
    )
    assert completed.returncode == 0, completed.stderr
    assert figure.is_file() and figure.stat().st_size > 0
    return {
        "figure": figure,
        "report": report,
        "table": tomllib.loads(report.read_text(encoding="utf-8")),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def test_the_committed_outputs_exist() -> None:
    assert SCRIPT_PATH.is_file(), f"missing {SCRIPT_PATH}"
    assert COMMITTED_FIGURE.is_file(), (
        f"missing {COMMITTED_FIGURE}; Day 0 delivers a committed figure"
    )
    assert COMMITTED_REPORT.is_file(), (
        f"missing {COMMITTED_REPORT}; a figure without its numbers is not a result"
    )


def test_the_committed_report_parses_and_names_both_models() -> None:
    table = tomllib.loads(COMMITTED_REPORT.read_text(encoding="utf-8"))
    assert table["schema_version"] == 1
    assert {entry["model"] for entry in table["published"]} == set(EXPECTED_MODELS)
    assert table["model_comparison"]["reference"] == "bekker"
    assert table["model_comparison"]["model"] == "reece"


def test_outputs_regenerate_without_a_series(tmp_path: Path) -> None:
    generated = _generate(tmp_path, "--series", tmp_path / "absent.toml")
    assert "no digitized series" in generated["stderr"]
    assert "fit" not in generated["table"], (
        "nothing was fitted, so the report must not claim a fit"
    )
    assert "series" not in generated["table"]["inputs"]


def test_the_report_is_deterministic(tmp_path: Path) -> None:
    first = _generate(tmp_path / "a", "--series", tmp_path / "absent.toml")
    second = _generate(tmp_path / "b", "--series", tmp_path / "absent.toml")
    assert first["report"].read_bytes() == second["report"].read_bytes(), (
        "the report changes between runs on identical inputs, so a diff would "
        "stop meaning that a result moved"
    )


def test_the_report_pins_its_inputs_by_digest(tmp_path: Path) -> None:
    inputs = _generate(tmp_path, "--series", tmp_path / "absent.toml")["table"]["inputs"]
    assert inputs["soil_sha256"] == hashlib.sha256(SOIL_PATH.read_bytes()).hexdigest()
    assert inputs["soil_id"] == "kls1"
    assert inputs["doi"] == "10.5140/JASS.2021.38.4.237"


def test_the_report_records_the_model_comparison(tmp_path: Path) -> None:
    table = _generate(tmp_path, "--series", tmp_path / "absent.toml")["table"]
    by_plate = table["model_comparison"]["by_plate"]
    assert len(by_plate) == 3
    for row in by_plate:
        assert row["relative_deviation_min"] == pytest.approx(
            row["relative_deviation_max"], abs=1e-12
        ), (
            "both published models share one sinkage exponent, so their ratio is "
            "independent of sinkage; a spread here means that changed"
        )
        assert abs(row["relative_deviation_min"]) < 0.05


def test_the_report_records_the_invertibility_floor(tmp_path: Path) -> None:
    table = _generate(tmp_path, "--series", tmp_path / "absent.toml")["table"]
    floors = {
        entry["model"]: entry["minimum_invertible_half_width_m"]
        for entry in table["published"]
    }
    assert floors["bekker"] == pytest.approx(0.0122997, rel=1e-4)
    for model_id, floor in floors.items():
        assert 0.0 < floor < 0.03, f"{model_id}: floor should sit below the plates"


def test_outputs_regenerate_with_a_series(
    tmp_path: Path, digitized_series: Path
) -> None:
    generated = _generate(tmp_path, "--series", digitized_series)
    table = generated["table"]
    assert {entry["model"] for entry in table["fit"]} == set(EXPECTED_MODELS)
    assert table["inputs"]["observation_count"] == 60
    assert table["inputs"]["series_sha256"] == hashlib.sha256(
        (digitized_series.parent / "series.csv").read_bytes()
    ).hexdigest()


def test_the_fit_recovers_the_published_parameters(
    tmp_path: Path, digitized_series: Path
) -> None:
    table = _generate(tmp_path, "--series", digitized_series)["table"]
    bekker = next(entry for entry in table["fit"] if entry["model"] == "bekker")
    assert bekker["coefficient_of_determination"] == pytest.approx(1.0, abs=1e-9)
    for name, deviation in bekker["relative_deviation_from_published"].items():
        assert deviation == pytest.approx(0.0, abs=1e-9), (
            f"{name} was not recovered from points generated by the published model"
        )


def test_a_corrupt_series_is_reported_rather_than_plotted(
    tmp_path: Path, digitized_series: Path
) -> None:
    csv_path = digitized_series.parent / "series.csv"
    csv_path.write_text(
        csv_path.read_text(encoding="utf-8").replace("0.03", "0.031", 1),
        encoding="utf-8",
    )
    figure = tmp_path / "figure.png"
    report = tmp_path / "report.toml"
    completed = _run("--series", digitized_series, f"--{CAMPAIGN}-figure", figure, f"--{CAMPAIGN}-report", report)
    assert completed.returncode == 1
    assert "cannot read the digitized series" in completed.stderr
    assert "Traceback" not in completed.stderr, (
        "a bad series must be refused with an explanation, not a stack trace:\n"
        + completed.stderr
    )
    assert not figure.exists(), "a figure was written from a series that failed to load"
    assert not report.exists(), "a report was written from a series that failed to load"


def test_a_soil_without_both_models_is_refused(tmp_path: Path) -> None:
    text = SOIL_PATH.read_text(encoding="utf-8")
    marker = 'id         = "reece"\nstatus     = "verified"'
    assert marker in text, "the reference soil no longer has a verified reece model"
    reduced = tmp_path / "reduced.toml"
    reduced.write_text(
        text.replace(marker, 'id         = "reece"\nstatus     = "superseded"'),
        encoding="utf-8",
    )
    completed = _run(
        "--soil", reduced, f"--{CAMPAIGN}-figure", tmp_path / "f.png",
        f"--{CAMPAIGN}-report", tmp_path / "r.toml",
    )
    assert completed.returncode != 0
    assert "no verified model" in completed.stderr


def test_a_series_too_small_to_fit_is_reported_not_raised(
    tmp_path: Path, digitized_series: Path
) -> None:
    csv_path = digitized_series.parent / "series.csv"
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    single = [lines[0]] + [row for row in lines[1:] if row.startswith("0.03,")]
    csv_path.write_text("\n".join(single) + "\n", encoding="utf-8")
    manifest = digitized_series.read_text(encoding="utf-8")
    stale = next(
        line.split('"')[1] for line in manifest.splitlines() if line.startswith("sha256")
    )
    digitized_series.write_text(
        manifest.replace(
            stale, hashlib.sha256(csv_path.read_bytes()).hexdigest()
        ),
        encoding="utf-8",
    )

    figure = tmp_path / "figure.png"
    report = tmp_path / "report.toml"
    completed = _run("--series", digitized_series, f"--{CAMPAIGN}-figure", figure, f"--{CAMPAIGN}-report", report)
    assert completed.returncode == 0, completed.stderr
    assert "cannot support a fit" in completed.stderr
    assert "Traceback" not in completed.stderr, (
        "one plate is a data limitation, not a crash:\n" + completed.stderr
    )
    assert figure.is_file(), "the published models are still worth plotting"
    table = tomllib.loads(report.read_text(encoding="utf-8"))
    assert "fit" not in table, "nothing was fitted, so the report must not claim a fit"
    assert "band_residual" in table, "the band is still measurable against one plate"
