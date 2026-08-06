# SPDX-License-Identifier: Apache-2.0
#
# Tests for biome.io.series.
#
# Corruptions are applied to a valid pair written by the fixture, and each one
# asserts its target was present before mutating, so a mutation that silently
# fails to apply cannot masquerade as a passing test.

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from biome.io.series import SeriesFileError, load_pressure_sinkage_series
from biome.io.soil import CalibratedContactModel


def _corrupt(manifest_path: Path, old: str, new: str) -> Path:
    text = manifest_path.read_text(encoding="utf-8")
    assert old in text, f"mutation target absent from the manifest: {old!r}"
    manifest_path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return manifest_path


def test_a_valid_pair_loads(digitized_series: Path) -> None:
    series = load_pressure_sinkage_series(digitized_series)
    assert series.id == "test-series"
    assert series.kind == "pressure_sinkage"
    assert series.source.doi == "10.5140/JASS.2021.38.4.237"
    assert series.digitization.method == "exact_model_evaluation"
    assert series.observations.count == 60
    assert series.distinct_tests == 2
    assert len(series.test_ids) == 60
    assert len(series.contact_half_widths) == 3
    assert series.series_path.is_file()
    assert set(series.digitization.axis_calibration) == {"x", "y"}


def test_the_csv_digest_is_verified(digitized_series: Path) -> None:
    csv_path = digitized_series.parent / "series.csv"
    csv_path.write_text(
        csv_path.read_text(encoding="utf-8").replace("0.03", "0.031", 1),
        encoding="utf-8",
    )
    with pytest.raises(SeriesFileError, match="but the manifest records"):
        load_pressure_sinkage_series(digitized_series)


def test_a_manifest_digest_that_no_longer_matches_is_refused(
    digitized_series: Path,
) -> None:
    csv_path = digitized_series.parent / "series.csv"
    actual = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    _corrupt(digitized_series, actual, "0" * 64)
    with pytest.raises(SeriesFileError, match=actual):
        load_pressure_sinkage_series(digitized_series)


def test_a_missing_csv_is_refused(digitized_series: Path) -> None:
    (digitized_series.parent / "series.csv").unlink()
    with pytest.raises(SeriesFileError, match="does not exist"):
        load_pressure_sinkage_series(digitized_series)


@pytest.mark.parametrize("version", ["2", "0", '"1"'])
def test_an_unsupported_schema_version_is_refused(
    digitized_series: Path, version: str
) -> None:
    _corrupt(digitized_series, "schema_version = 1", f"schema_version = {version}")
    with pytest.raises(SeriesFileError, match="schema_version"):
        load_pressure_sinkage_series(digitized_series)


def test_a_mismatched_kind_is_refused(digitized_series: Path) -> None:
    _corrupt(digitized_series, 'kind = "pressure_sinkage"', 'kind = "shear"')
    with pytest.raises(SeriesFileError, match="this loader reads"):
        load_pressure_sinkage_series(digitized_series)


def test_a_logarithmic_axis_is_refused(digitized_series: Path) -> None:
    _corrupt(digitized_series, 'scale = "linear" }\ny =', 'scale = "log" }\ny =')
    with pytest.raises(SeriesFileError, match="is not supported"):
        load_pressure_sinkage_series(digitized_series)


def test_a_decreasing_axis_calibration_is_refused(digitized_series: Path) -> None:
    _corrupt(
        digitized_series,
        "x = { minimum = 0.0, maximum = 0.095",
        "x = { minimum = 0.095, maximum = 0.0",
    )
    with pytest.raises(SeriesFileError, match="not increasing"):
        load_pressure_sinkage_series(digitized_series)


@pytest.mark.parametrize(
    "field", ["sinkage_uncertainty_m", "pressure_uncertainty_kPa"]
)
def test_a_non_positive_digitization_uncertainty_is_refused(
    digitized_series: Path, field: str
) -> None:
    text = digitized_series.read_text(encoding="utf-8")
    line = next(line for line in text.splitlines() if line.startswith(field))
    _corrupt(digitized_series, line, f"{field} = 0.0")
    with pytest.raises(SeriesFileError, match="must be positive"):
        load_pressure_sinkage_series(digitized_series)


def test_an_unexpected_manifest_key_is_refused(digitized_series: Path) -> None:
    _corrupt(
        digitized_series, 'operator = "test suite"', 'operator = "x"\nunexpected = 1'
    )
    with pytest.raises(SeriesFileError, match="Digitization"):
        load_pressure_sinkage_series(digitized_series)


def test_declared_columns_must_match_the_schema(digitized_series: Path) -> None:
    _corrupt(
        digitized_series,
        'columns = ["contact_half_width_m", "test_id", "sinkage_m", "pressure_kPa"]',
        'columns = ["sinkage_m", "pressure_kPa"]',
    )
    with pytest.raises(SeriesFileError, match="must declare"):
        load_pressure_sinkage_series(digitized_series)


def _rewrite_csv(digitized_series: Path, text: str) -> Path:
    csv_path = digitized_series.parent / "series.csv"
    csv_path.write_text(text, encoding="utf-8")
    stale_digest = next(
        line.split('"')[1]
        for line in digitized_series.read_text(encoding="utf-8").splitlines()
        if line.startswith("sha256")
    )
    return _corrupt(
        digitized_series,
        stale_digest,
        hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    )


def test_a_csv_header_that_does_not_match_is_refused(digitized_series: Path) -> None:
    _rewrite_csv(digitized_series, "a,b,c,d\n0.03,r1,0.01,10.0\n")
    with pytest.raises(SeriesFileError, match="header is"):
        load_pressure_sinkage_series(digitized_series)


def test_a_non_numeric_cell_is_refused(digitized_series: Path) -> None:
    _rewrite_csv(
        digitized_series,
        "contact_half_width_m,test_id,sinkage_m,pressure_kPa\n0.03,r1,0.01,x\n",
    )
    with pytest.raises(SeriesFileError, match="which is not a number"):
        load_pressure_sinkage_series(digitized_series)


def test_a_csv_with_no_observations_is_refused(digitized_series: Path) -> None:
    _rewrite_csv(digitized_series, "contact_half_width_m,test_id,sinkage_m,pressure_kPa\n")
    with pytest.raises(SeriesFileError, match="no observations"):
        load_pressure_sinkage_series(digitized_series)


def test_a_zero_sinkage_row_is_refused_with_the_fitting_reason(
    digitized_series: Path,
) -> None:
    _rewrite_csv(
        digitized_series,
        "contact_half_width_m,test_id,sinkage_m,pressure_kPa\n"
        "0.03,r1,0.0,0.0\n0.03,r1,0.02,10.0\n0.035,r1,0.02,11.0\n",
    )
    with pytest.raises(SeriesFileError, match="strictly positive for a log-space"):
        load_pressure_sinkage_series(digitized_series)


def test_invalid_toml_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "broken.toml"
    path.write_text("schema_version = = 1", encoding="utf-8")
    with pytest.raises(SeriesFileError, match="not valid TOML"):
        load_pressure_sinkage_series(path)


def test_observations_round_trip_through_the_csv(
    digitized_series: Path, published_models: Mapping[str, CalibratedContactModel]
) -> None:
    series = load_pressure_sinkage_series(digitized_series)
    bekker = published_models["bekker"]
    recomputed = bekker.pressure(
        sinkage=series.observations.sinkage_m,
        contact_half_width=series.observations.contact_half_width_m,
    )
    np.testing.assert_allclose(
        series.observations.pressure_kPa, recomputed, rtol=1e-15, atol=0.0
    )


def test_a_published_curve_series_loads_under_its_own_kind(
    digitized_series: Path,
) -> None:
    _corrupt(digitized_series, 'kind = "pressure_sinkage"', 'kind = "published_curve"')
    series = load_pressure_sinkage_series(digitized_series)
    assert series.kind == "published_curve", (
        "the kind must survive loading so a traced model curve can never be "
        "mistaken for measured data and fitted to"
    )


def test_an_unknown_kind_names_the_circularity_it_prevents(
    digitized_series: Path,
) -> None:
    _corrupt(digitized_series, 'kind = "pressure_sinkage"', 'kind = "whatever"')
    with pytest.raises(SeriesFileError, match="must never be fitted to"):
        load_pressure_sinkage_series(digitized_series)
