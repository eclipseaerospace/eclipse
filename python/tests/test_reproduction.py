# SPDX-License-Identifier: Apache-2.0
#
# Reproduction of published fits from committed raw channels.
#
# This is the test the GRC-1 transcription exists to support: raw sensor
# readings, converted, resampled onto the recorded grid, fitted by the
# published estimator, must return the published parameters. It exercises
# eclipse.io.channels, eclipse.resampling and eclipse.fitting together, which is
# where a factor that each module considers reasonable would otherwise survive.
#
# Every expected value and every preprocessing choice is read from the manifest.
# Nothing is written down here, so adding a campaign adds its own cases.
#
# Both fitting windows are checked. They come from one set of channels and
# differ only in the pressure range resampled onto, so reproducing the pair
# constrains the reconstruction far more tightly than either alone: a wrong
# calibration constant cannot satisfy both.

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eclipse.fitting import PressureSinkageObservations, fit_contact_model
from eclipse.io.channels import BevameterChannels, load_bevameter_channels
from eclipse.resampling import ensemble

LITERATURE = Path(__file__).resolve().parents[2] / "data" / "literature"
MANIFEST = LITERATURE / "oravec2009-grc1-raw-channels.toml"
PLATE_NAMES = ("small", "medium", "large")
REPEATS = "12345"
PARAMETERS = ("sinkage_exponent", "cohesive_modulus", "frictional_modulus")


def _manifest() -> dict[str, Any]:
    return tomllib.loads(MANIFEST.read_text(encoding="utf-8"))


def _plate_curve(
    channels: BevameterChannels, plate: str, top_kPa: float, step_kPa: float
) -> tuple[np.ndarray, np.ndarray, float]:
    grid = np.arange(0.0, top_kPa + step_kPa / 2.0, step_kPa)
    curve = ensemble(
        sample_positions=[
            channels.pressure_kPa(f"{plate}-{repeat}") for repeat in REPEATS
        ],
        sample_values=[channels.sinkage_m(f"{plate}-{repeat}") for repeat in REPEATS],
        positions=grid,
    )
    return grid, curve.mean_values, curve.maximum_deviation


def _observations(
    channels: BevameterChannels, block: dict[str, Any]
) -> PressureSinkageObservations:
    half_width, sinkage, pressure = [], [], []
    for plate, top, dropped in zip(
        PLATE_NAMES,
        block["resampling_endpoints_kPa"],
        block["leading_samples_dropped"],
    ):
        grid, mean, _ = _plate_curve(
            channels, plate, top, block["resampling_step_kPa"]
        )
        usable = (grid[dropped:] > 0.0) & (mean[dropped:] > 0.0)
        half_width.append(
            np.full(
                int(usable.sum()),
                channels.test(f"{plate}-1").plate.contact_half_width_m,
            )
        )
        sinkage.append(mean[dropped:][usable])
        pressure.append(grid[dropped:][usable])
    return PressureSinkageObservations(
        contact_half_width_m=np.concatenate(half_width),
        sinkage_m=np.concatenate(sinkage),
        pressure_kPa=np.concatenate(pressure),
    )


def _blocks() -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = _manifest()["verification"]
    return blocks


def _identifiers() -> list[str]:
    return [block["id"] for block in _blocks()]


@pytest.fixture(scope="module")
def channels() -> BevameterChannels:
    return load_bevameter_channels(MANIFEST)


def test_the_manifest_records_more_than_one_window() -> None:
    assert len(_blocks()) >= 2, (
        "reproducing a single window would not separate a wrong calibration "
        "constant from a right one, so the pair is the point"
    )


@pytest.mark.parametrize("block", _blocks(), ids=_identifiers())
def test_the_published_parameters_are_reproduced(
    channels: BevameterChannels, block: dict[str, Any]
) -> None:
    fit = fit_contact_model(
        "bekker",
        _observations(channels, block),
        weighting="pressure_squared",
        estimator="averaged_exponent",
    )
    for name in PARAMETERS:
        assert fit.parameters[name] == pytest.approx(block[name], rel=2e-4), (
            f"{block['id']}: {name} does not reproduce, so the chain from raw "
            "channel to fitted parameter has moved"
        )


@pytest.mark.parametrize("block", _blocks(), ids=_identifiers())
def test_the_ensemble_deviation_is_reproduced(
    channels: BevameterChannels, block: dict[str, Any]
) -> None:
    recorded = block["maximum_deviation_mm"]
    if not np.isfinite(recorded).all():
        pytest.skip(f"{block['id']} records no deviations to check against")
    for plate, top, expected in zip(
        PLATE_NAMES, block["resampling_endpoints_kPa"], recorded
    ):
        _, _, deviation = _plate_curve(
            channels, plate, top, block["resampling_step_kPa"]
        )
        assert round(deviation * 1e3, 1) == pytest.approx(expected), (
            f"{block['id']}, {plate} plate: the resampling and ensemble steps "
            "are checked here independently of the fit that follows them"
        )


def test_a_wrong_calibration_constant_breaks_both_windows(
    channels: BevameterChannels,
) -> None:
    blocks = _blocks()
    reproduced = []
    for block in blocks:
        observations = _observations(channels, block)
        disturbed = PressureSinkageObservations(
            contact_half_width_m=observations.contact_half_width_m,
            sinkage_m=observations.sinkage_m,
            pressure_kPa=observations.pressure_kPa * 1.005,
        )
        fit = fit_contact_model(
            "bekker", disturbed, weighting="pressure_squared",
            estimator="averaged_exponent",
        )
        reproduced.append(
            fit.parameters["cohesive_modulus"]
            == pytest.approx(block["cohesive_modulus"], rel=2e-4)
        )
    assert not any(reproduced), (
        "a half-percent error in the load calibration must move the fit; if it "
        "does not, the reproduction tests above would pass on wrong data"
    )
