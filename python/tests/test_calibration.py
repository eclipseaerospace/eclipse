# SPDX-License-Identifier: Apache-2.0
#
# Tests for calibration/contact/sinkage.py.
#
# Run as a subprocess rather than imported, because it is a script and its entry
# point is what the repository actually invokes to regenerate the committed
# figure. A figure that no longer regenerates is a broken result even when every
# library test passes.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
SCRIPT_PATH: Final = REPOSITORY_ROOT / "calibration" / "contact" / "sinkage.py"
COMMITTED_FIGURE: Final = (
    REPOSITORY_ROOT / "calibration" / "contact" / "figures" / "kls1-pressure-sinkage.png"
)


def _run(*arguments: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *map(str, arguments)],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
    )


def test_the_script_exists_and_its_figure_is_committed() -> None:
    assert SCRIPT_PATH.is_file(), f"missing {SCRIPT_PATH}"
    assert COMMITTED_FIGURE.is_file(), (
        f"missing {COMMITTED_FIGURE}; the Day 0 deliverable is a committed figure"
    )


@pytest.mark.parametrize("model_id", ["bekker", "reece"])
def test_the_figure_regenerates_without_a_series(
    tmp_path: Path, model_id: str
) -> None:
    output = tmp_path / "figure.png"
    completed = _run(
        "--model", model_id,
        "--series", tmp_path / "absent.toml",
        "--output", output,
    )
    assert completed.returncode == 0, completed.stderr
    assert output.is_file()
    assert output.stat().st_size > 0
    assert "no digitized series" in completed.stderr


def test_the_figure_regenerates_with_a_series(
    tmp_path: Path, digitized_series: Path
) -> None:
    output = tmp_path / "figure.png"
    completed = _run("--series", digitized_series, "--output", output)
    assert completed.returncode == 0, completed.stderr
    assert output.is_file()
    assert "re-fitted" in completed.stdout or "published" in completed.stdout


def test_the_reported_comparison_recovers_the_published_parameters(
    tmp_path: Path, digitized_series: Path
) -> None:
    completed = _run("--series", digitized_series, "--output", tmp_path / "figure.png")
    assert completed.returncode == 0, completed.stderr
    reported = [
        line.split()
        for line in completed.stdout.splitlines()
        if line.strip().startswith(("sinkage_exponent", "cohesive_", "frictional_"))
    ]
    assert len(reported) == 3, completed.stdout
    for name, published, refitted, relative in reported:
        assert float(relative.rstrip("%")) == pytest.approx(0.0, abs=0.01), (
            f"{name} was not recovered: published {published}, re-fitted {refitted}"
        )


def test_an_unknown_model_is_refused(tmp_path: Path) -> None:
    completed = _run("--model", "dimensional_analysis_lim2021", "--output", tmp_path / "f.png")
    assert completed.returncode != 0
    assert "no verified model" in completed.stderr


def test_a_corrupt_series_is_reported_rather_than_plotted(
    tmp_path: Path, digitized_series: Path
) -> None:
    csv_path = digitized_series.parent / "series.csv"
    csv_path.write_text(
        csv_path.read_text(encoding="utf-8").replace("0.03", "0.031", 1),
        encoding="utf-8",
    )
    output = tmp_path / "figure.png"
    completed = _run("--series", digitized_series, "--output", output)
    assert completed.returncode == 1
    assert "cannot read the digitized series" in completed.stderr
    assert not output.exists(), "a figure was written from a series that failed to load"
