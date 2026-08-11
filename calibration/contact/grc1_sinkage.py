# SPDX-License-Identifier: Apache-2.0
#
# calibration/contact/grc1_sinkage.py — GRC-1 reproduction from raw channels.
#
# A thin runner. Loading lives in biome.io.channels, resampling in
# biome.resampling and fitting in biome.fitting, so all three are unit tested
# and type checked; this file wires paths, draws, and writes the result.
#
# The figure is the visual proof that the reconstruction is right, and it shows
# both fitting windows side by side because that is the finding. One set of
# fifteen tests, two choices of pressure range, and a cohesive modulus that
# moves by a factor of forty-two between them. Neither panel is wrong; the two
# are fits of different models, because a different exponent gives k_c
# different units, and quantities with different units are not comparable.
#
# Axes follow the bevameter convention: pressure across, sinkage increasing
# down the page. Plate identity carries on line style and marker shape, so the
# panels stay readable in one ink.
#
# The report carries no timestamp: re-running against unchanged inputs must
# leave it byte-identical, so any diff means a result actually moved.
#
# References
#   Oravec HA (2009) Understanding Mechanical Behavior of Lunar Soils for the
#     Study of Vehicle Mobility. PhD dissertation, Case Western Reserve
#     University. Appendix D Code D5, Figures 5.52 and 5.53.

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
import tomllib
from pathlib import Path
from typing import Any, Final, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from biome.fitting import (
    FittedContactModel,
    PressureSinkageObservations,
    fit_contact_model,
)
from biome.io.channels import (
    BevameterChannels,
    ChannelsFileError,
    load_bevameter_channels,
)
from biome.resampling import EnsembleCurve, ensemble

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_CHANNELS_PATH: Final = (
    REPOSITORY_ROOT / "data" / "literature" / "oravec2009-grc1-raw-channels.toml"
)
DEFAULT_FIGURE_PATH: Final = (
    Path(__file__).resolve().parent / "figures" / "grc1-pressure-sinkage.png"
)
DEFAULT_REPORT_PATH: Final = (
    Path(__file__).resolve().parent / "results" / "grc1-pressure-sinkage.toml"
)

MODEL_ID: Final = "bekker"
WEIGHTING: Final = "pressure_squared"
ESTIMATOR: Final = "averaged_exponent"
PLATE_NAMES: Final = ("small", "medium", "large")
REPEATS: Final = "12345"
REPORT_SCHEMA_VERSION: Final = 1
CURVE_SAMPLES: Final = 300
PARAMETERS: Final = ("sinkage_exponent", "cohesive_modulus", "frictional_modulus")

PLATE_LINESTYLES: Final[tuple[Any, ...]] = ("solid", (0, (6, 2)), (0, (7, 2, 1.5, 2)))
PLATE_MARKERS: Final = ("o", "^", "s")
FIT_COLOR: Final = "#1f4e9c"
INK_PRIMARY: Final = "#0b0b0b"
INK_SECONDARY: Final = "#52514e"
INK_MUTED: Final = "#8a8880"
SURFACE: Final = "#fcfcfb"
MARKER_SIZE: Final = 3.0

FIGURE_STYLE: Final[dict[str, Any]] = {
    "figure.figsize": (10.4, 5.6),
    "figure.dpi": 200,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK_SECONDARY,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.titlesize": 9.5,
    "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "font.size": 9.5,
    "legend.frameon": False,
    "legend.fontsize": 8.0,
    "savefig.facecolor": SURFACE,
    "figure.subplot.top": 0.752,
    "figure.subplot.bottom": 0.212,
    "figure.subplot.left": 0.062,
    "figure.subplot.right": 0.985,
    "figure.subplot.wspace": 0.165,
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _format_float(value: float) -> str:
    return "nan" if not np.isfinite(value) else repr(float(value))


def _plate_half_width(channels: BevameterChannels, plate: str) -> float:
    return channels.test(f"{plate}-{REPEATS[0]}").plate.contact_half_width_m


def _ensemble_for(
    channels: BevameterChannels, plate: str, top_kPa: float, step_kPa: float
) -> EnsembleCurve:
    return ensemble(
        sample_positions=[channels.pressure_kPa(f"{plate}-{r}") for r in REPEATS],
        sample_values=[channels.sinkage_m(f"{plate}-{r}") for r in REPEATS],
        positions=np.arange(0.0, top_kPa + step_kPa / 2.0, step_kPa),
    )


def _window(
    channels: BevameterChannels, block: dict[str, Any]
) -> tuple[dict[str, EnsembleCurve], PressureSinkageObservations]:
    curves: dict[str, EnsembleCurve] = {}
    half_width, sinkage, pressure = [], [], []
    for plate, top, dropped in zip(
        PLATE_NAMES,
        block["resampling_endpoints_kPa"],
        block["leading_samples_dropped"],
    ):
        curve = _ensemble_for(channels, plate, top, block["resampling_step_kPa"])
        curves[plate] = curve
        grid = curve.positions[dropped:]
        mean = curve.mean_values[dropped:]
        usable = (grid > 0.0) & (mean > 0.0)
        half_width.append(
            np.full(int(usable.sum()), _plate_half_width(channels, plate))
        )
        sinkage.append(mean[usable])
        pressure.append(grid[usable])
    return curves, PressureSinkageObservations(
        contact_half_width_m=np.concatenate(half_width),
        sinkage_m=np.concatenate(sinkage),
        pressure_kPa=np.concatenate(pressure),
    )


def _draw_panel(
    axes: Axes,
    channels: BevameterChannels,
    block: dict[str, Any],
    curves: dict[str, EnsembleCurve],
    fit: FittedContactModel,
) -> None:
    top = max(block["resampling_endpoints_kPa"])
    deepest = 0.0
    for index, plate in enumerate(PLATE_NAMES):
        linestyle, marker = PLATE_LINESTYLES[index], PLATE_MARKERS[index]
        half_width = _plate_half_width(channels, plate)
        for repeat in REPEATS:
            identifier = f"{plate}-{repeat}"
            observed_pressure = channels.pressure_kPa(identifier)
            observed_sinkage = channels.sinkage_m(identifier)
            inside = observed_pressure <= top
            axes.plot(
                observed_pressure[inside],
                observed_sinkage[inside] * 1e3,
                marker=marker,
                markersize=MARKER_SIZE,
                markerfacecolor="none",
                markeredgecolor=INK_SECONDARY,
                markeredgewidth=0.7,
                linestyle="none",
                zorder=3,
            )
            deepest = max(deepest, float(observed_sinkage[inside].max() * 1e3))
        depth = np.linspace(
            0.0, float(curves[plate].mean_values.max()), CURVE_SAMPLES
        )
        axes.plot(
            fit.model.pressure(sinkage=depth, contact_half_width=half_width),
            depth * 1e3,
            color=FIT_COLOR,
            linewidth=1.6,
            linestyle=linestyle,
            zorder=5,
        )

    axes.set_xlim(0.0, top)
    axes.set_ylim(deepest * 1.06, 0.0)
    axes.set_xlabel("pressure  (kPa)")
    axes.set_ylabel("sinkage  (mm)")
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.set_axisbelow(True)
    # The fitted parameters live in the title rather than inside the axes.
    # The empty corner moves between the panels, because the entire-range
    # panel carries the punch-through scatter, and a label placed by eye for
    # one dataset lands on the data of the next.
    axes.set_title(
        f"{block['window'].replace('_', ' ')}, to "
        f"{'/'.join(f'{end:.0f}' for end in block['resampling_endpoints_kPa'])} kPa\n"
        f"n = {fit.parameters['sinkage_exponent']:.4f}    "
        f"$k_c$ = {fit.parameters['cohesive_modulus']:.2f}    "
        f"$k_\\varphi$ = {fit.parameters['frictional_modulus']:.2f}",
        color=INK_PRIMARY,
        loc="left",
        linespacing=1.6,
    )


def _legend_entries(channels: BevameterChannels) -> tuple[list[Any], list[str]]:
    handles: list[Any] = []
    labels: list[str] = []
    for index, plate in enumerate(PLATE_NAMES):
        handles.append(
            Line2D(
                [], [],
                color=INK_SECONDARY,
                linewidth=1.4,
                linestyle=PLATE_LINESTYLES[index],
                marker=PLATE_MARKERS[index],
                markersize=MARKER_SIZE,
                markerfacecolor="none",
                markeredgewidth=0.7,
            )
        )
        labels.append(f"b = {_plate_half_width(channels, plate) * 1e3:.1f} mm")
    return handles, labels


def build_figure(
    channels: BevameterChannels,
    blocks: list[dict[str, Any]],
    fits: dict[str, FittedContactModel],
    windows: dict[str, dict[str, EnsembleCurve]],
    report_path: Path,
) -> Figure:
    with plt.rc_context(cast(Any, FIGURE_STYLE)):
        figure, axes = plt.subplots(1, len(blocks), squeeze=False)
        for column, block in enumerate(blocks):
            _draw_panel(
                axes[0][column],
                channels,
                block,
                windows[block["id"]],
                fits[block["id"]],
            )
        handles, labels = _legend_entries(channels)
        axes[0][0].legend(handles=handles, labels=labels, loc="lower left")

        figure.text(
            0.062, 0.965,
            "GRC-1 pressure-sinkage: one set of tests, two fitting windows",
            fontsize=12, color=INK_PRIMARY, weight="bold", ha="left", va="top",
        )
        cohesive = [fits[block["id"]].parameters["cohesive_modulus"] for block in blocks]
        ratio = max(cohesive) / min(cohesive) if min(cohesive) != 0.0 else float("nan")
        figure.text(
            0.062, 0.935,
            f"Reconstructed from {len(channels.tests)} raw bevameter tests "
            f"({sum(test.sample_count for test in channels.tests)} samples) "
            "transcribed from Appendix D Code D5.\n"
            "Markers are the individual repeat tests; curves are the fit each "
            f"window produces.\nThe cohesive modulus differs by {ratio:.0f}x "
            "between the panels, from the analysis window alone.",
            fontsize=8.5, color=INK_SECONDARY, ha="left", va="top",
        )
        figure.text(
            0.062, 0.108,
            "The two fits carry different sinkage exponents, so their cohesive "
            "moduli carry different units and are not\n"
            "the same quantity: the ratio measures a choice of analysis window, "
            "not a disagreement about the soil.\n"
            "Plate-scale parameters do not transfer to foot-scale contact "
            "patches.\n"
            f"Numbers behind this figure: {_display_path(report_path)}",
            fontsize=7.5, color=INK_MUTED, ha="left", va="top",
        )
        return figure


def build_report(
    channels: BevameterChannels,
    blocks: list[dict[str, Any]],
    fits: dict[str, FittedContactModel],
    windows: dict[str, dict[str, EnsembleCurve]],
) -> str:
    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# Generated by calibration/contact/grc1_sinkage.py. Do not edit.",
        "#",
        "# Reproduction of published fits from raw sensor channels. No timestamp:",
        "# re-running against unchanged inputs must leave this byte-identical.",
        "",
        f"schema_version = {REPORT_SCHEMA_VERSION}",
        f'model = "{MODEL_ID}"',
        f'weighting = "{WEIGHTING}"',
        f'estimator = "{ESTIMATOR}"',
        "",
        "[inputs]",
        f'channels_manifest = "{_display_path(channels.manifest_path)}"',
        f'channels_manifest_sha256 = "{_digest(channels.manifest_path)}"',
        f'channels_series = "{_display_path(channels.series_path)}"',
        f'channels_series_sha256 = "{_digest(channels.series_path)}"',
        f"test_count = {len(channels.tests)}",
        f"sample_count = {sum(test.sample_count for test in channels.tests)}",
        "",
        "[environment]",
        f'python = "{platform.python_version()}"',
        f'numpy = "{np.__version__}"',
        "",
    ]
    for block in blocks:
        fit = fits[block["id"]]
        lines += [
            "[[window]]",
            f'id = "{block["id"]}"',
            f'window = "{block["window"]}"',
            f'resampling_endpoints = "{block["resampling_endpoints"]}"',
            "resampling_endpoints_kPa = ["
            + ", ".join(_format_float(v) for v in block["resampling_endpoints_kPa"])
            + "]",
            f"resampling_step_kPa = {_format_float(block['resampling_step_kPa'])}",
            "leading_samples_dropped = ["
            + ", ".join(str(int(v)) for v in block["leading_samples_dropped"])
            + "]",
            f"observation_count = {fit.observation_count}",
            f"plate_count = {fit.plate_count}",
            "",
            "[window.fitted]",
        ]
        lines += [f"{name} = {_format_float(fit.parameters[name])}" for name in PARAMETERS]
        lines += ["", "[window.published]"]
        lines += [f"{name} = {_format_float(float(block[name]))}" for name in PARAMETERS]
        lines += ["", "[window.relative_deviation]"]
        lines += [
            f"{name} = "
            + _format_float(
                abs(fit.parameters[name] - float(block[name]))
                / abs(float(block[name]))
            )
            for name in PARAMETERS
        ]
        lines += ["", "[window.maximum_deviation_mm]"]
        for plate in PLATE_NAMES:
            lines.append(
                f"{plate} = "
                + _format_float(windows[block["id"]][plate].maximum_deviation * 1e3)
            )
        lines.append("")

    cohesive = [fits[block["id"]].parameters["cohesive_modulus"] for block in blocks]
    exponents = [fits[block["id"]].parameters["sinkage_exponent"] for block in blocks]
    lines += [
        "# The ratio measures the analysis window, not the soil: the two fits",
        "# carry different exponents, so their cohesive moduli carry different",
        "# units and are not the same quantity.",
        "[across_windows]",
        f"cohesive_modulus_ratio = {_format_float(max(cohesive) / min(cohesive))}",
        f"sinkage_exponent_span = {_format_float(max(exponents) - min(exponents))}",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the published GRC-1 fits from raw bevameter channels."
    )
    parser.add_argument("--channels", type=Path, default=DEFAULT_CHANNELS_PATH)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    arguments = parser.parse_args(argv)

    try:
        channels = load_bevameter_channels(arguments.channels)
    except ChannelsFileError as error:
        print(error, file=sys.stderr)
        return 1

    manifest = tomllib.loads(arguments.channels.read_text(encoding="utf-8"))
    blocks = manifest.get("verification", [])
    if not blocks:
        print(
            f"{arguments.channels}: no verification blocks to reproduce",
            file=sys.stderr,
        )
        return 1

    fits: dict[str, FittedContactModel] = {}
    windows: dict[str, dict[str, EnsembleCurve]] = {}
    for block in blocks:
        curves, observations = _window(channels, block)
        windows[block["id"]] = curves
        fits[block["id"]] = fit_contact_model(
            MODEL_ID, observations, weighting=WEIGHTING, estimator=ESTIMATOR
        )

    print(f"  {'window':>24s} {'parameter':>20s} {'fitted':>14s} {'published':>13s} {'rel':>9s}")
    worst = 0.0
    for block in blocks:
        fit = fits[block["id"]]
        for name in PARAMETERS:
            published = float(block[name])
            relative = abs(fit.parameters[name] - published) / abs(published)
            worst = max(worst, relative)
            print(
                f"  {block['id']:>24s} {name:>20s} {fit.parameters[name]:14.4f} "
                f"{published:13.4f} {relative * 100:8.4f}%"
            )
    print(f"\n  worst relative deviation across both windows: {worst * 100:.4f}%")

    arguments.figure.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure(channels, blocks, fits, windows, arguments.report)
    figure.savefig(arguments.figure, dpi=FIGURE_STYLE["figure.dpi"])
    plt.close(figure)
    arguments.report.write_text(
        build_report(channels, blocks, fits, windows), encoding="utf-8"
    )
    print(f"figure   {arguments.figure}")
    print(f"report   {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
