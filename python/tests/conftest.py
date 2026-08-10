# SPDX-License-Identifier: Apache-2.0
#
# Shared fixtures.
#
# Observations are generated from the published models rather than read from a
# committed file. A fabricated measurement file in a repository whose value is
# its provenance is a liability, however clearly it is labeled; generating
# points inside a fixture keeps them off disk and out of every commit.
#
# Generating from the published model is also the stronger test: a fit given
# exact model output must return exactly the parameters that produced it, which
# is what makes the fitter trustworthy before it ever sees digitized points.

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import numpy as np
import pytest

from biome.fitting import PressureSinkageObservations
from biome.io.soil import CalibratedContactModel, load_soil

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
REFERENCE_SOIL_PATH: Final = REPOSITORY_ROOT / "data" / "soils" / "kls1.toml"
SINKAGE_SAMPLES: Final = 20
REPLICATES: Final = 2

MANIFEST_TEMPLATE: Final = """schema_version = 1
id = "test-series"
kind = "pressure_sinkage"

[source]
doi = "10.5140/JASS.2021.38.4.237"
figure = "Figure 12"
license = "CC-BY-NC-3.0"

[digitization]
tool = "generated from the published model inside a test fixture"
method = "exact_model_evaluation"
operator = "test suite"
performed_on = 2026-08-03
sinkage_uncertainty_m = 0.0005
pressure_uncertainty_kPa = 0.8

[digitization.axis_calibration]
x = {{ minimum = 0.0, maximum = 0.095, units = "m", scale = "linear" }}
y = {{ minimum = 0.0, maximum = 130.0, units = "kPa", scale = "linear" }}

[series]
path = "{csv_name}"
sha256 = "{sha256}"
columns = ["contact_half_width_m", "test_id", "sinkage_m", "pressure_kPa"]
"""


@pytest.fixture(scope="session")
def published_models() -> Mapping[str, CalibratedContactModel]:
    return load_soil(REFERENCE_SOIL_PATH).datasets["lim2021"].models


@pytest.fixture(scope="session")
def tested_half_widths() -> np.ndarray:
    dataset = load_soil(REFERENCE_SOIL_PATH).datasets["lim2021"]
    return dataset.apparatus.contact_half_widths


def observations_from(
    model: CalibratedContactModel, half_widths: np.ndarray
) -> PressureSinkageObservations:
    bounds = model.sinkage_validity
    depth = np.linspace(bounds.max / SINKAGE_SAMPLES, bounds.max, SINKAGE_SAMPLES)
    grid_half_width, grid_sinkage = np.meshgrid(half_widths, depth, indexing="ij")
    pressure = np.asarray(
        model.pressure(sinkage=grid_sinkage, contact_half_width=grid_half_width)
    )
    return PressureSinkageObservations(
        contact_half_width_m=grid_half_width.ravel(),
        sinkage_m=grid_sinkage.ravel(),
        pressure_kPa=pressure.ravel(),
    )


def write_series(directory: Path, observations: PressureSinkageObservations) -> Path:
    test_identifiers = [
        f"run-{index % REPLICATES + 1}" for index in range(observations.count)
    ]
    rows = ["contact_half_width_m,test_id,sinkage_m,pressure_kPa"]
    rows.extend(
        f"{float(half_width)!r},{test_id},{float(sinkage)!r},{float(pressure)!r}"
        for test_id, (half_width, sinkage, pressure) in zip(
            test_identifiers,
            zip(
                observations.contact_half_width_m,
                observations.sinkage_m,
                observations.pressure_kPa,
                strict=True,
            ),
            strict=True,
        )
    )
    csv_path = directory / "series.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    manifest_path = directory / "series.toml"
    manifest_path.write_text(
        MANIFEST_TEMPLATE.format(
            csv_name=csv_path.name,
            sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        ),
        encoding="utf-8",
    )
    return manifest_path


@pytest.fixture
def digitized_series(
    tmp_path: Path,
    published_models: Mapping[str, CalibratedContactModel],
    tested_half_widths: np.ndarray,
) -> Path:
    return write_series(
        tmp_path, observations_from(published_models["bekker"], tested_half_widths)
    )
