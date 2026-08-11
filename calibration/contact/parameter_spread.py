# SPDX-License-Identifier: Apache-2.0
#
# calibration/contact/parameter_spread.py — what moves the cohesive modulus.
#
# A thin runner. Every number here is computed from committed data by the same
# library the rest of the project uses; nothing is carried over from a notebook.
#
# The question this answers is not whether k_c is uncertain but what makes it
# so. Each row varies exactly one thing and holds the rest fixed, so the rows
# are comparable and the ordering means something. Rows are computed on the
# campaign named in each label, because the two campaigns differ in plate
# geometry and in whether the points are scattered or smoothed, and those are
# themselves candidate explanations rather than nuisances to average over.
#
# The window row leads, and it is the one row that is not an error of any kind.
# Changing the fitted pressure range changes the sinkage exponent, and k_c
# carries units of kN/m^(n+1), so the two ends of that row are not two estimates
# of one quantity — they are quantities with different dimensions. That is also
# why cross-campaign tables of k_c are not comparable, which is the practical
# consequence worth stating.
#
# Two measures are reported per row and both are needed, because they disagree
# about how far ahead the window row is: the full range over the mean magnitude
# puts it at four times the next row, while the ratio of largest to smallest
# magnitude puts it at thirty. A range understates a row spanning orders of
# magnitude and a ratio overstates a row that does not, so quoting one measure
# for one row and the other measure for another manufactures a gap. Spread is a
# range rather than a standard deviation because the values it summarises are
# choices an analyst makes, not draws from a distribution.

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from biome.fitting import (
    Estimator,
    PressureSinkageObservations,
    WeightingScheme,
    fit_contact_model,
)
from biome.io.channels import BevameterChannels, load_bevameter_channels
from biome.io.series import load_pressure_sinkage_series
from biome.resampling import ensemble

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
LITERATURE: Final = REPOSITORY_ROOT / "data" / "literature"
DEFAULT_CHANNELS_PATH: Final = LITERATURE / "oravec2009-grc1-raw-channels.toml"
DEFAULT_SERIES_PATH: Final = LITERATURE / "lim2021-pressure-sinkage.toml"
DEFAULT_FIGURE_PATH: Final = (
    Path(__file__).resolve().parent / "figures" / "cohesive-modulus-spread.png"
)
DEFAULT_REPORT_PATH: Final = (
    Path(__file__).resolve().parent / "results" / "cohesive-modulus-spread.toml"
)

MODEL_ID: Final = "bekker"
PARAMETER: Final = "cohesive_modulus"
PLATE_NAMES: Final = ("small", "medium", "large")
REPEATS: Final = "12345"
REPORT_SCHEMA_VERSION: Final = 1
ESTIMATORS: Final[tuple[Estimator, ...]] = ("two_stage", "averaged_exponent", "direct")
WEIGHTINGS: Final[tuple[WeightingScheme, ...]] = ("uniform", "pressure_squared")

INK_PRIMARY: Final = "#0b0b0b"
INK_SECONDARY: Final = "#52514e"
INK_MUTED: Final = "#8a8880"
SURFACE: Final = "#fcfcfb"
BAR_COLOR: Final = "#1f4e9c"
WINDOW_COLOR: Final = "#d4570a"

FIGURE_STYLE: Final[dict[str, Any]] = {
    "figure.figsize": (9.6, 5.2),
    "figure.dpi": 200,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK_SECONDARY,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9.0,
    "font.size": 9.5,
    "legend.frameon": False,
    "savefig.facecolor": SURFACE,
    "figure.subplot.top": 0.815,
    "figure.subplot.bottom": 0.250,
    "figure.subplot.left": 0.345,
    "figure.subplot.right": 0.975,
}


@dataclass(frozen=True, slots=True)
class Source:
    label: str
    detail: str
    values: tuple[float, ...]
    dimensionally_comparable: bool

    @property
    def spread(self) -> float:
        magnitude = abs(float(np.mean(self.values)))
        return (max(self.values) - min(self.values)) / magnitude * 100.0

    @property
    def ratio(self) -> float:
        magnitudes = [abs(value) for value in self.values]
        smallest = min(magnitudes)
        return max(magnitudes) / smallest if smallest > 0.0 else float("inf")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _cohesive(
    observations: PressureSinkageObservations,
    estimator: Estimator,
    weighting: WeightingScheme,
) -> float:
    fit = fit_contact_model(
        MODEL_ID, observations, weighting=weighting, estimator=estimator
    )
    return fit.parameters[PARAMETER]


def _ensemble_observations(
    channels: BevameterChannels, block: dict[str, Any]
) -> PressureSinkageObservations:
    half_width, sinkage, pressure = [], [], []
    for plate, top, dropped in zip(
        PLATE_NAMES,
        block["resampling_endpoints_kPa"],
        block["leading_samples_dropped"],
    ):
        step = block["resampling_step_kPa"]
        curve = ensemble(
            sample_positions=[channels.pressure_kPa(f"{plate}-{r}") for r in REPEATS],
            sample_values=[channels.sinkage_m(f"{plate}-{r}") for r in REPEATS],
            positions=np.arange(0.0, top + step / 2.0, step),
        )
        grid, mean = curve.positions[dropped:], curve.mean_values[dropped:]
        usable = (grid > 0.0) & (mean > 0.0)
        half_width.append(
            np.full(
                int(usable.sum()),
                channels.test(f"{plate}-1").plate.contact_half_width_m,
            )
        )
        sinkage.append(mean[usable])
        pressure.append(grid[usable])
    return PressureSinkageObservations(
        contact_half_width_m=np.concatenate(half_width),
        sinkage_m=np.concatenate(sinkage),
        pressure_kPa=np.concatenate(pressure),
    )


def _raw_observations(
    channels: BevameterChannels, block: dict[str, Any]
) -> PressureSinkageObservations:
    half_width, sinkage, pressure = [], [], []
    for plate, top in zip(PLATE_NAMES, block["resampling_endpoints_kPa"]):
        for repeat in REPEATS:
            identifier = f"{plate}-{repeat}"
            observed_pressure = channels.pressure_kPa(identifier)
            observed_sinkage = channels.sinkage_m(identifier)
            usable = (
                (observed_pressure > 0.0)
                & (observed_sinkage > 0.0)
                & (observed_pressure <= top)
            )
            half_width.append(
                np.full(
                    int(usable.sum()),
                    channels.test(identifier).plate.contact_half_width_m,
                )
            )
            sinkage.append(observed_sinkage[usable])
            pressure.append(observed_pressure[usable])
    return PressureSinkageObservations(
        contact_half_width_m=np.concatenate(half_width),
        sinkage_m=np.concatenate(sinkage),
        pressure_kPa=np.concatenate(pressure),
    )


def _combinations(observations: PressureSinkageObservations) -> tuple[float, ...]:
    return tuple(
        _cohesive(observations, estimator, weighting)
        for estimator in ESTIMATORS
        for weighting in WEIGHTINGS
    )


def collect_sources(
    channels: BevameterChannels, blocks: list[dict[str, Any]], kls1_path: Path
) -> list[Source]:
    by_id = {block["id"]: block for block in blocks}
    lunar, entire = by_id["figure-5.52"], by_id["figure-5.53"]

    lunar_ensemble = _ensemble_observations(channels, lunar)
    entire_ensemble = _ensemble_observations(channels, entire)
    lunar_raw = _raw_observations(channels, lunar)
    kls1 = load_pressure_sinkage_series(kls1_path).observations

    sources = [
        Source(
            "sinkage exponent\nshared or averaged",
            "GRC-1 lunar window, pressure-squared weighting, one estimator choice",
            (
                _cohesive(lunar_ensemble, "two_stage", "pressure_squared"),
                _cohesive(lunar_ensemble, "averaged_exponent", "pressure_squared"),
            ),
            True,
        ),
        Source(
            "log-space weighting\nuniform or squared",
            "GRC-1 lunar window, averaged exponent, one weighting choice",
            (
                _cohesive(lunar_ensemble, "averaged_exponent", "uniform"),
                _cohesive(lunar_ensemble, "averaged_exponent", "pressure_squared"),
            ),
            True,
        ),
        Source(
            "estimator x weighting\nKLS-1, digitized",
            "Lim et al. (2021) digitized points, six combinations",
            _combinations(kls1),
            True,
        ),
        Source(
            "estimator x weighting\nGRC-1, raw samples",
            "GRC-1 lunar window, per-test samples, six combinations",
            _combinations(lunar_raw),
            True,
        ),
        Source(
            "data reduction\nsamples or ensemble",
            "GRC-1 lunar window, averaged exponent, pressure-squared weighting",
            (
                _cohesive(lunar_raw, "averaged_exponent", "pressure_squared"),
                _cohesive(lunar_ensemble, "averaged_exponent", "pressure_squared"),
            ),
            True,
        ),
        Source(
            "fitting window\nlunar or entire",
            "GRC-1, averaged exponent, pressure-squared weighting, published windows",
            (
                _cohesive(lunar_ensemble, "averaged_exponent", "pressure_squared"),
                _cohesive(entire_ensemble, "averaged_exponent", "pressure_squared"),
            ),
            False,
        ),
    ]
    return sorted(sources, key=lambda source: source.spread)


def build_figure(sources: list[Source], report_path: Path) -> Figure:
    with plt.rc_context(cast(Any, FIGURE_STYLE)):
        figure, axes = plt.subplots()
        positions = np.arange(len(sources), dtype=float)
        spreads = [source.spread for source in sources]
        colors = [
            BAR_COLOR if source.dimensionally_comparable else WINDOW_COLOR
            for source in sources
        ]
        axes.barh(positions, spreads, height=0.62, color=colors, zorder=3)
        for position, source in zip(positions, sources):
            axes.text(
                source.spread * 1.12,
                position,
                f"{source.spread:,.0f}%",
                va="center",
                ha="left",
                fontsize=8.5,
                color=INK_PRIMARY,
            )
        axes.set_xscale("log")
        axes.set_xlim(10.0, 420.0)
        axes.set_yticks(positions)
        axes.set_yticklabels([source.label for source in sources], fontsize=8.5)
        axes.set_xlabel("spread in the cohesive modulus  (percent of its own mean)")
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        axes.set_axisbelow(True)

        figure.text(
            0.030, 0.960,
            "What moves the cohesive modulus",
            fontsize=12.5, color=INK_PRIMARY, weight="bold", ha="left", va="top",
        )
        figure.text(
            0.030, 0.905,
            "Each row varies one choice and holds the rest fixed, so the "
            "ordering means something. Plate geometry is not among them:\n"
            "KLS-1 and GRC-1 differ fivefold in conditioning yet give 35 and 43 "
            "percent on the same six combinations.",
            fontsize=8.5, color=INK_SECONDARY, ha="left", va="top",
        )
        figure.text(
            0.030, 0.150,
            "Spread is the full range over the mean magnitude; the "
            "largest-over-smallest ratio is recorded alongside it, because the "
            "two measures disagree about how far ahead the\nwindow row is — four "
            f"times the next row by range, "
            f"{next(s.ratio for s in sources if not s.dimensionally_comparable):.0f} "
            "times by ratio. Orange is not an error: a different fitted pressure "
            "range gives a different\nsinkage exponent, and k_c carries units of "
            "kN/m^(n+1), so its two ends are quantities of different dimension. "
            "That is also why published tables of k_c are not comparable.\n"
            f"Numbers behind this figure: {_display_path(report_path)}",
            fontsize=7.5, color=INK_MUTED, ha="left", va="top",
        )
        return figure


def build_report(sources: list[Source], inputs: dict[str, Path]) -> str:
    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# Generated by calibration/contact/parameter_spread.py. Do not edit.",
        "#",
        "# No timestamp: re-running against unchanged inputs must leave this",
        "# byte-identical, so any diff means a result actually moved.",
        "",
        f"schema_version = {REPORT_SCHEMA_VERSION}",
        f'model = "{MODEL_ID}"',
        f'parameter = "{PARAMETER}"',
        'spread = "full range over the mean magnitude, in percent"',
        "",
        "[inputs]",
    ]
    for name, path in inputs.items():
        lines += [
            f'{name} = "{_display_path(path)}"',
            f'{name}_sha256 = "{_digest(path)}"',
        ]
    lines += [
        "",
        "[environment]",
        f'python = "{platform.python_version()}"',
        f'numpy = "{np.__version__}"',
        "",
    ]
    for source in sources:
        lines += [
            "[[source]]",
            f'label = "{source.label.replace(chr(10), ", ")}"',
            f'detail = "{source.detail}"',
            f"dimensionally_comparable = {str(source.dimensionally_comparable).lower()}",
            "values = [" + ", ".join(repr(float(v)) for v in source.values) + "]",
            f"spread_percent = {float(source.spread)!r}",
            f"ratio_largest_over_smallest = {float(source.ratio)!r}",
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Decompose the spread in the fitted cohesive modulus by source."
    )
    parser.add_argument("--channels", type=Path, default=DEFAULT_CHANNELS_PATH)
    parser.add_argument("--series", type=Path, default=DEFAULT_SERIES_PATH)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    arguments = parser.parse_args(argv)

    channels = load_bevameter_channels(arguments.channels)
    blocks = tomllib.loads(arguments.channels.read_text(encoding="utf-8")).get(
        "verification", []
    )
    if len(blocks) < 2:
        print(
            f"{arguments.channels}: two fitting windows are needed to separate "
            "the window from everything else",
            file=sys.stderr,
        )
        return 1

    sources = collect_sources(channels, blocks, arguments.series)
    print(f"  {'source':>40s} {'spread':>9s} {'ratio':>9s}")
    for source in sources:
        print(
            f"  {source.label.replace(chr(10), ' / '):>40s} "
            f"{source.spread:8.0f}% {source.ratio:8.1f}x"
        )

    arguments.figure.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure(sources, arguments.report)
    figure.savefig(arguments.figure, dpi=FIGURE_STYLE["figure.dpi"])
    plt.close(figure)
    arguments.report.write_text(
        build_report(
            sources, {"channels": arguments.channels, "series": arguments.series}
        ),
        encoding="utf-8",
    )
    print(f"figure   {arguments.figure}")
    print(f"report   {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
