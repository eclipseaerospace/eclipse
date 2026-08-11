# SPDX-License-Identifier: Apache-2.0
#
# Tests for eclipse.io.channels.
#
# The committed GRC-1 file is the fixture. Corruptions are applied to a copy of
# it, and each asserts its target was present before mutating, so a mutation
# that silently fails to apply cannot masquerade as a passing test.
#
# Expected converted values are read from the file's own conversion_check
# blocks, which tools/verify_channels.py generates without importing eclipse.
# Nothing numeric is written down here.

from __future__ import annotations

import hashlib
import shutil
import tomllib
from pathlib import Path

import numpy as np
import pytest

from eclipse.io.channels import (
    ChannelsFileError,
    Plate,
    load_bevameter_channels,
)

LITERATURE = Path(__file__).resolve().parents[2] / "data" / "literature"
MANIFEST = LITERATURE / "oravec2009-grc1-raw-channels.toml"


@pytest.fixture
def channels_manifest(tmp_path: Path) -> Path:
    manifest = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    copied = tmp_path / MANIFEST.name
    shutil.copy(MANIFEST, copied)
    shutil.copy(LITERATURE / manifest["series"]["path"], tmp_path)
    return copied


def _corrupt(manifest_path: Path, old: str, new: str) -> Path:
    text = manifest_path.read_text(encoding="utf-8")
    assert old in text, f"mutation target absent from the manifest: {old!r}"
    manifest_path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return manifest_path


def _rewrite_series(manifest_path: Path, text: str) -> Path:
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    series_path = manifest_path.parent / manifest["series"]["path"]
    series_path.write_text(text, encoding="utf-8")
    return _corrupt(
        manifest_path,
        manifest["series"]["sha256"],
        hashlib.sha256(series_path.read_bytes()).hexdigest(),
    )


def test_the_committed_file_loads(channels_manifest: Path) -> None:
    channels = load_bevameter_channels(channels_manifest)
    assert channels.id == "oravec2009-grc1-raw-channels"
    assert channels.kind == "raw_bevameter_channels"
    assert len(channels.plates) == 3
    assert len(channels.tests) == 15
    assert len(set(channels.test_ids)) == 15


def test_conversion_matches_the_independent_verifier() -> None:
    manifest = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    stored = manifest["conversion_check"]
    assert stored, "the manifest carries no conversion checks to test against"

    channels = load_bevameter_channels(MANIFEST)
    assert [entry["test_id"] for entry in stored] == list(channels.test_ids)

    for entry in stored:
        test_id = entry["test_id"]
        sinkage = channels.sinkage_m(test_id)
        pressure = channels.pressure_kPa(test_id)
        assert sinkage.size == entry["sample_count"]
        assert float(sinkage.max()) == pytest.approx(
            entry["maximum_sinkage_m"], rel=1e-12, abs=0.0
        )
        assert float(pressure.max()) == pytest.approx(
            entry["maximum_pressure_kPa"], rel=1e-12, abs=0.0
        )


def test_both_channels_are_zeroed_on_their_first_sample(
    channels_manifest: Path,
) -> None:
    channels = load_bevameter_channels(channels_manifest)
    for test_id in channels.test_ids:
        assert channels.sinkage_m(test_id)[0] == 0.0
        assert channels.pressure_kPa(test_id)[0] == 0.0


def test_every_test_uses_a_plate_the_manifest_describes(
    channels_manifest: Path,
) -> None:
    channels = load_bevameter_channels(channels_manifest)
    described = {plate.diameter_m for plate in channels.plates}
    assert {test.plate.diameter_m for test in channels.tests} <= described


def test_an_unknown_test_is_named_with_the_alternatives(
    channels_manifest: Path,
) -> None:
    channels = load_bevameter_channels(channels_manifest)
    with pytest.raises(ChannelsFileError, match="no test 'absent'"):
        channels.sinkage_m("absent")


def test_the_series_digest_is_verified(channels_manifest: Path) -> None:
    manifest = tomllib.loads(channels_manifest.read_text(encoding="utf-8"))
    series_path = channels_manifest.parent / manifest["series"]["path"]
    series_path.write_text(
        series_path.read_text(encoding="utf-8").replace("0.076", "0.078", 1),
        encoding="utf-8",
    )
    with pytest.raises(ChannelsFileError, match="but the manifest records"):
        load_bevameter_channels(channels_manifest)


def test_a_missing_series_is_refused(channels_manifest: Path) -> None:
    manifest = tomllib.loads(channels_manifest.read_text(encoding="utf-8"))
    (channels_manifest.parent / manifest["series"]["path"]).unlink()
    with pytest.raises(ChannelsFileError, match="does not exist"):
        load_bevameter_channels(channels_manifest)


@pytest.mark.parametrize("version", ["2", "0", '"1"'])
def test_an_unsupported_schema_version_is_refused(
    channels_manifest: Path, version: str
) -> None:
    _corrupt(channels_manifest, "schema_version = 1", f"schema_version = {version}")
    with pytest.raises(ChannelsFileError, match="schema_version"):
        load_bevameter_channels(channels_manifest)


def test_a_converted_series_is_refused_by_this_loader(
    channels_manifest: Path,
) -> None:
    _corrupt(
        channels_manifest,
        'kind           = "raw_bevameter_channels"',
        'kind           = "pressure_sinkage"',
    )
    with pytest.raises(ChannelsFileError, match="eclipse.io.series"):
        load_bevameter_channels(channels_manifest)


def test_declared_columns_must_match_the_schema(channels_manifest: Path) -> None:
    _corrupt(
        channels_manifest,
        'columns = ["plate_diameter_m", "test_id", "excitation_V", "lvdt_mV", '
        '"load_cell_mV_per_V"]',
        'columns = ["test_id", "lvdt_mV"]',
    )
    with pytest.raises(ChannelsFileError, match="must declare"):
        load_bevameter_channels(channels_manifest)


def test_a_header_that_does_not_match_is_refused(channels_manifest: Path) -> None:
    _rewrite_series(channels_manifest, "a,b,c,d,e\n0.076,small-1,10.0,0.0,0.0\n")
    with pytest.raises(ChannelsFileError, match="header is"):
        load_bevameter_channels(channels_manifest)


def test_a_non_numeric_cell_is_refused(channels_manifest: Path) -> None:
    _rewrite_series(
        channels_manifest,
        "plate_diameter_m,test_id,excitation_V,lvdt_mV,load_cell_mV_per_V\n"
        "0.076,small-1,10.0,x,0.0\n",
    )
    with pytest.raises(ChannelsFileError, match="which is not a number"):
        load_bevameter_channels(channels_manifest)


def test_a_series_with_no_samples_is_refused(channels_manifest: Path) -> None:
    _rewrite_series(
        channels_manifest,
        "plate_diameter_m,test_id,excitation_V,lvdt_mV,load_cell_mV_per_V\n",
    )
    with pytest.raises(ChannelsFileError, match="no samples"):
        load_bevameter_channels(channels_manifest)


def test_excitation_varying_within_a_test_is_refused(
    channels_manifest: Path,
) -> None:
    _rewrite_series(
        channels_manifest,
        "plate_diameter_m,test_id,excitation_V,lvdt_mV,load_cell_mV_per_V\n"
        "0.076,small-1,10.0,0.0,0.0\n"
        "0.076,small-1,10.5,1.0,0.1\n",
    )
    with pytest.raises(ChannelsFileError, match="different values of excitation_V"):
        load_bevameter_channels(channels_manifest)


def test_a_plate_the_manifest_does_not_describe_is_refused(
    channels_manifest: Path,
) -> None:
    _rewrite_series(
        channels_manifest,
        "plate_diameter_m,test_id,excitation_V,lvdt_mV,load_cell_mV_per_V\n"
        "0.5,small-1,10.0,0.0,0.0\n"
        "0.5,small-1,10.0,1.0,0.1\n",
    )
    with pytest.raises(ChannelsFileError, match="which the manifest does not describe"):
        load_bevameter_channels(channels_manifest)


def test_a_single_sample_test_is_refused(channels_manifest: Path) -> None:
    _rewrite_series(
        channels_manifest,
        "plate_diameter_m,test_id,excitation_V,lvdt_mV,load_cell_mV_per_V\n"
        "0.076,small-1,10.0,0.0,0.0\n",
    )
    with pytest.raises(ChannelsFileError, match="consumed as the unloaded origin"):
        load_bevameter_channels(channels_manifest)


def test_a_zero_calibration_constant_is_refused(channels_manifest: Path) -> None:
    _corrupt(
        channels_manifest,
        "load_cell    = { value = 1155.1",
        "load_cell    = { value = 0.0",
    )
    with pytest.raises(ChannelsFileError, match="finite and non-zero"):
        load_bevameter_channels(channels_manifest)


def test_invalid_toml_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "broken.toml"
    path.write_text("schema_version = = 1", encoding="utf-8")
    with pytest.raises(ChannelsFileError, match="not valid TOML"):
        load_bevameter_channels(path)


def test_a_half_width_that_is_not_the_radius_is_refused() -> None:
    with pytest.raises(ChannelsFileError, match="length scale is the radius"):
        Plate(diameter_m=0.076, contact_half_width_m=0.076, area_m2=4.5364598e-3)


def test_an_area_inconsistent_with_the_radius_is_refused() -> None:
    with pytest.raises(ChannelsFileError, match="but pi \\* radius\\^2 is"):
        Plate(diameter_m=0.076, contact_half_width_m=0.038, area_m2=1.0e-3)


def test_the_recorded_plate_areas_are_consistent(channels_manifest: Path) -> None:
    channels = load_bevameter_channels(channels_manifest)
    for plate in channels.plates:
        assert plate.area_m2 == pytest.approx(
            float(np.pi * plate.contact_half_width_m**2), rel=1e-6
        )
