# SPDX-License-Identifier: Apache-2.0
#
# calibration/contact/pressure_sinkage.py — pressure-sinkage calibration.
#
# A thin runner for both campaigns. Loading lives in eclipse.io, resampling in
# eclipse.resampling and fitting in eclipse.fitting, so all three are unit tested
# and type checked; this file wires paths, draws, and writes the results.
#
# Two campaigns, one drawing core. KLS-1 compares two published models on one
# panel against digitized points; GRC-1 compares two fitting windows across two
# panels against raw channels it reconstructs. What they share is every mark on
# the page — plate identity on line style and marker shape, pressure across,
# sinkage increasing down the page — so the marks are drawn in one place and the
# campaigns supply only what differs.
#
# Colour is spent differently by each, and deliberately. KLS-1 has two model
# curves that very nearly coincide, so colour separates them where a dash
# pattern cannot; GRC-1 has one curve per panel and needs no second channel.
#
# Both reports carry no timestamp: re-running against unchanged inputs must
# leave them byte-identical, so any diff means a result actually moved.
#
# References
#   Lim Y, Le VD, Bahati PA (2021) Development of a New Pressure-Sinkage Model
#     for Rover Wheel-Lunar Soil Interaction. J. Astron. Space Sci. 38(4).
#     doi:10.5140/JASS.2021.38.4.237
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
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from eclipse.fitting import (
    DEFAULT_WEIGHTING,
    FittedContactModel,
    PressureSinkageObservations,
    WeightingScheme,
    coefficient_of_determination,
    coefficient_of_determination_ceiling,
    fit_contact_model,
    mean_relative_residual,
    parameter_bound_under_bias_permutation,
    relative_deviation,
)
from eclipse.io.channels import (
    BevameterChannels,
    ChannelsFileError,
    load_bevameter_channels,
)
from eclipse.io.series import (
    PressureSinkageSeries,
    SeriesFileError,
    load_pressure_sinkage_series,
)
from eclipse.io.soil import CalibratedContactModel, Dataset, Soil, load_soil
from eclipse.resampling import EnsembleCurve, ensemble

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
LITERATURE: Final = REPOSITORY_ROOT / "data" / "literature"
FIGURES: Final = Path(__file__).resolve().parent / "figures"
RESULTS: Final = Path(__file__).resolve().parent / "results"
GENERATED_BY: Final = "calibration/contact/pressure_sinkage.py"
REPORT_SCHEMA_VERSION: Final = 1

DEFAULT_SOIL_PATH: Final = REPOSITORY_ROOT / "data" / "soils" / "kls1.toml"
DEFAULT_SERIES_PATH: Final = LITERATURE / "lim2021-pressure-sinkage.toml"
DEFAULT_TRACE_PATH: Final = LITERATURE / "lim2021-published-bekker-curve.toml"
DEFAULT_CHANNELS_PATH: Final = LITERATURE / "oravec2009-grc1-raw-channels.toml"
KLS1_FIGURE_PATH: Final = FIGURES / "kls1-pressure-sinkage.png"
KLS1_REPORT_PATH: Final = RESULTS / "kls1-pressure-sinkage.toml"
GRC1_FIGURE_PATH: Final = FIGURES / "grc1-pressure-sinkage.png"
GRC1_REPORT_PATH: Final = RESULTS / "grc1-pressure-sinkage.toml"

REFERENCE_MODEL: Final = "bekker"
COMPARED_MODEL: Final = "reece"
MODEL_COLORS: Final[dict[str, str]] = {
    REFERENCE_MODEL: "#1f4e9c",
    COMPARED_MODEL: "#d4570a",
}
# The two published fits differ by a few percent, so their curves very nearly
# coincide. Drawing the compared model thinner and on top leaves the reference
# visible as a wider stroke underneath rather than hidden by whichever was
# drawn last.
MODEL_LINEWIDTHS: Final[dict[str, float]] = {
    REFERENCE_MODEL: 2.6,
    COMPARED_MODEL: 1.0,
}
WEIGHTINGS: Final[tuple[WeightingScheme, ...]] = ("uniform", "pressure_squared")
CALIBRATION_BIAS_MINIMUM_SINKAGE_M: Final = 0.010
CURVE_SAMPLES: Final = 400

GRC1_MODEL_ID: Final = "bekker"
GRC1_WEIGHTING: Final = "pressure_squared"
GRC1_ESTIMATOR: Final = "averaged_exponent"
GRC1_PLATE_NAMES: Final = ("small", "medium", "large")
GRC1_REPEATS: Final = "12345"
GRC1_PARAMETERS: Final = (
    "sinkage_exponent",
    "cohesive_modulus",
    "frictional_modulus",
)
GRC1_FIT_COLOR: Final = "#1f4e9c"
GRC1_CURVE_SAMPLES: Final = 300

PLATE_LINESTYLES: Final[tuple[Any, ...]] = (
    "solid",
    (0, (6, 2)),
    (0, (7, 2, 1.5, 2)),
)
PLATE_MARKERS: Final = ("o", "^", "s")
INK_PRIMARY: Final = "#0b0b0b"
INK_SECONDARY: Final = "#52514e"
INK_MUTED: Final = "#8a8880"
SURFACE: Final = "#fcfcfb"

BASE_STYLE: Final[dict[str, Any]] = {
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
    "legend.frameon": False,
    "savefig.facecolor": SURFACE,
}
KLS1_STYLE: Final[dict[str, Any]] = {
    **BASE_STYLE,
    "figure.figsize": (7.8, 5.4),
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.size": 10,
    "legend.fontsize": 8.5,
    "figure.subplot.top": 0.845,
    "figure.subplot.bottom": 0.215,
    "figure.subplot.left": 0.085,
    "figure.subplot.right": 0.965,
}
GRC1_STYLE: Final[dict[str, Any]] = {
    **BASE_STYLE,
    "figure.figsize": (10.4, 5.6),
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "font.size": 9.5,
    "legend.fontsize": 8.0,
    "figure.subplot.top": 0.752,
    "figure.subplot.bottom": 0.212,
    "figure.subplot.left": 0.062,
    "figure.subplot.right": 0.985,
    "figure.subplot.wspace": 0.165,
}
KLS1_MARKER: Final = (3.4, 0.8)
GRC1_MARKER: Final = (3.0, 0.7)
PLATE_LEGEND_ANCHOR: Final = 0.755


def _plate_style(index: int) -> tuple[Any, str]:
    if index >= len(PLATE_LINESTYLES):
        raise SystemExit(
            f"the figure carries {len(PLATE_LINESTYLES)} distinguishable plate "
            f"styles but this dataset has {index + 1} plates; fold the extra "
            "plates into small multiples rather than inventing a dash pattern "
            "no reader can name"
        )
    return PLATE_LINESTYLES[index], PLATE_MARKERS[index]


def _draw_plate_markers(
    axes: Axes,
    index: int,
    pressure_kPa: NDArray[np.float64],
    sinkage_m: NDArray[np.float64],
    marker: tuple[float, float],
    zorder: int = 5,
) -> None:
    size, edge_width = marker
    axes.plot(
        pressure_kPa,
        sinkage_m * 1e3,
        marker=_plate_style(index)[1],
        markersize=size,
        markerfacecolor="none",
        markeredgecolor=INK_SECONDARY,
        markeredgewidth=edge_width,
        linestyle="none",
        zorder=zorder,
    )


def _draw_plate_curve(
    axes: Axes,
    index: int,
    pressure_kPa: NDArray[np.float64],
    sinkage_m: NDArray[np.float64],
    color: str,
    linewidth: float,
    zorder: int = 4,
) -> None:
    axes.plot(
        pressure_kPa,
        sinkage_m * 1e3,
        color=color,
        linewidth=linewidth,
        linestyle=_plate_style(index)[0],
        zorder=zorder,
    )


def _finish_axes(axes: Axes, sinkage_limit_mm: float) -> None:
    axes.set_xlabel("pressure  (kPa)")
    axes.set_ylabel("sinkage  (mm)")
    axes.set_xlim(left=0.0)
    axes.set_ylim(sinkage_limit_mm, 0.0)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.set_axisbelow(True)


def _plate_legend_handle(index: int, marker: tuple[float, float]) -> Line2D:
    linestyle, shape = _plate_style(index)
    size, edge_width = marker
    return Line2D(
        [], [], color=INK_SECONDARY, linewidth=1.4, linestyle=linestyle,
        marker=shape, markersize=size, markerfacecolor="none",
        markeredgewidth=edge_width,
    )


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


def _inline_table(values: dict[str, float]) -> str:
    return (
        "{ "
        + ", ".join(f"{name} = {_format_float(value)}" for name, value in values.items())
        + " }"
    )


def _environment_lines() -> list[str]:
    return [
        "[environment]",
        f'python = "{platform.python_version()}"',
        f'numpy = "{np.__version__}"',
    ]


# --- KLS-1: two published models, one panel, digitized points


def _sinkage_sweep(published: CalibratedContactModel) -> np.ndarray:
    bounds = published.sinkage_validity
    return np.linspace(max(bounds.min, bounds.max * 1e-6), bounds.max, CURVE_SAMPLES)


def _plate_observation_count(
    series: PressureSinkageSeries, half_width: float
) -> int:
    return int(
        np.count_nonzero(series.observations.contact_half_width_m == half_width)
    )


def _kls1_legend_entries(has_series: bool) -> tuple[list[Any], list[str]]:
    models = (REFERENCE_MODEL, COMPARED_MODEL)
    handles: list[Any] = [
        Line2D(
            [], [], color=MODEL_COLORS[model_id],
            linewidth=MODEL_LINEWIDTHS[model_id],
        )
        for model_id in models
    ]
    labels = [f"{model_id.title()} (published)" for model_id in models]
    if has_series:
        handles.append(
            tuple(
                Line2D(
                    [], [], color=INK_SECONDARY, marker=marker,
                    markersize=KLS1_MARKER[0], markerfacecolor="none",
                    markeredgewidth=KLS1_MARKER[1], linestyle="none",
                )
                for marker in PLATE_MARKERS
            )
        )
        labels.append("digitized points")
    return handles, labels


def _kls1_plate_legend_handles(
    half_widths: list[float],
    published: CalibratedContactModel,
    series: PressureSinkageSeries | None,
) -> list[Any]:
    handles: list[Any] = []
    for index, half_width in enumerate(half_widths):
        label = f"b = {half_width * 1e3:.1f} mm"
        if series is not None and _plate_observation_count(series, half_width) >= 2:
            residual = mean_relative_residual(
                published.extrapolating, series.observations.for_plate(half_width)
            )
            label += f"    residual {residual * 100:+.1f}%"
        handle = _plate_legend_handle(index, KLS1_MARKER)
        handle.set_label(label)
        handles.append(handle)
    return handles


def _kls1_figure(
    models: dict[str, CalibratedContactModel],
    half_widths: list[float],
    series: PressureSinkageSeries | None,
    report_path: Path,
) -> Figure:
    published = models[REFERENCE_MODEL]
    depth = _sinkage_sweep(published)
    with plt.rc_context(cast(Any, KLS1_STYLE)):
        figure, axes = plt.subplots()
        if series is not None:
            for index, half_width in enumerate(half_widths):
                if _plate_observation_count(series, half_width) < 2:
                    continue
                selected = series.observations.for_plate(half_width)
                _draw_plate_markers(
                    axes, index, selected.pressure_kPa, selected.sinkage_m,
                    KLS1_MARKER,
                )
        for index, half_width in enumerate(half_widths):
            for order, model_id in enumerate((REFERENCE_MODEL, COMPARED_MODEL)):
                model = models.get(model_id)
                if model is None:
                    continue
                _draw_plate_curve(
                    axes, index,
                    np.asarray(
                        model.pressure(sinkage=depth, contact_half_width=half_width)
                    ),
                    depth,
                    MODEL_COLORS[model_id],
                    MODEL_LINEWIDTHS[model_id],
                    zorder=4 + order,
                )

        _finish_axes(axes, published.sinkage_validity.max * 1e3 * 1.02)
        handles, labels = _kls1_legend_entries(series is not None)
        style_legend = axes.legend(
            handles=handles, labels=labels, loc="upper right",
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.7)},
        )
        axes.add_artist(style_legend)
        axes.legend(
            handles=_kls1_plate_legend_handles(half_widths, published, series),
            loc="upper right",
            bbox_to_anchor=(1.0, PLATE_LEGEND_ANCHOR),
        )

        subtitle = (
            "Published parameters only. No digitized measurements yet,\n"
            "so this is the transcribed fit, not a comparison against data."
            if series is None
            else (
                f"{series.observations.count} points digitized from "
                f"{series.source.figure} of {series.source.doi}.\nMarker shape and "
                "dash pattern identify the plate; color identifies the model."
            )
        )
        figure.text(
            0.085, 0.965,
            f"KLS-1 pressure-sinkage: {REFERENCE_MODEL.title()} against "
            f"{COMPARED_MODEL.title()}",
            fontsize=12, color=INK_PRIMARY, weight="bold", ha="left", va="top",
        )
        figure.text(
            0.085, 0.922, subtitle,
            fontsize=8.5, color=INK_SECONDARY, ha="left", va="top",
        )
        figure.text(
            0.085, 0.098,
            "Plotted only within the fitted range: contact half-width "
            f"{published.contact_half_width_validity.min * 1e3:.1f}"
            f"–{published.contact_half_width_validity.max * 1e3:.1f} mm, "
            f"sinkage 0–{published.sinkage_validity.max * 1e3:.0f} mm.\n"
            "These plate-scale parameters do not transfer to foot-scale contact "
            "patches.\nNumbers behind this figure: "
            f"{_display_path(report_path)}",
            fontsize=7.5, color=INK_MUTED, ha="left", va="top",
        )
        return figure


def _kls1_report(
    soil: Soil,
    dataset: Dataset,
    models: dict[str, CalibratedContactModel],
    half_widths: list[float],
    soil_path: Path,
    series: PressureSinkageSeries | None,
    fits: dict[tuple[str, str], FittedContactModel],
    bias: dict[float, float],
) -> str:
    depth = _sinkage_sweep(models[REFERENCE_MODEL])
    lines = [
        "# Generated by calibration/contact/pressure_sinkage.py. Do not edit by hand.",
        "#",
        "# Deterministic by construction: no timestamp, so re-running against",
        "# unchanged inputs leaves this file byte-identical and any diff means a",
        "# result moved. Input digests pin the exact bytes behind every number.",
        "",
        f"schema_version = {REPORT_SCHEMA_VERSION}",
        'generated_by = "calibration/contact/pressure_sinkage.py"',
        "",
        "[environment]",
        f'python = "{platform.python_version()}"',
        f'numpy = "{np.__version__}"',
        "",
        "# Two fits per model. Wong (1980) prescribes weighting each log residual",
        "# by the squared pressure and this dataset cites that method, yet uniform",
        "# weighting reproduces the published parameters far better. Sampling",
        "# density does not explain it: the bevameter ran displacement-controlled,",
        "# so its samples and these digitized columns are both uniform in sinkage.",
        "# The discrepancy is recorded, not resolved.",
        "",
        "[inputs]",
        f'soil = "{_display_path(soil_path)}"',
        f'soil_sha256 = "{_digest(soil_path)}"',
        f'soil_id = "{soil.id}"',
        f'dataset_id = "{dataset.id}"',
        f'doi = "{dataset.doi}"',
    ]
    if series is not None:
        lines.extend(
            [
                f'series = "{_display_path(series.manifest_path)}"',
                f'series_manifest_sha256 = "{_digest(series.manifest_path)}"',
                f'series_sha256 = "{_digest(series.series_path)}"',
                f"observation_count = {series.observations.count}",
                "coefficient_of_determination_ceiling = "
                + _format_float(
                    coefficient_of_determination_ceiling(series.observations)
                ),
            ]
        )

    lines.extend(
        [
            "",
            "[model_comparison]",
            f'reference = "{REFERENCE_MODEL}"',
            f'model = "{COMPARED_MODEL}"',
            "by_plate = [",
        ]
    )
    for half_width in half_widths:
        deviation = relative_deviation(
            models[REFERENCE_MODEL],
            models[COMPARED_MODEL],
            sinkage=depth,
            contact_half_width=half_width,
        )
        lines.append(
            f"  {{ contact_half_width_m = {_format_float(half_width)}, "
            f"relative_deviation_min = {_format_float(float(deviation.min()))}, "
            f"relative_deviation_max = {_format_float(float(deviation.max()))} }},"
        )
    lines.append("]")

    for model_id, model in models.items():
        lines.extend(
            [
                "",
                "[[published]]",
                f'model = "{model_id}"',
                f'fit_method = "{model.fit_method}"',
                "minimum_invertible_half_width_m = "
                + _format_float(model.minimum_invertible_half_width()),
                "parameters = " + _inline_table(dict(model.parameters)),
            ]
        )

    if series is not None:
        lines.extend(["", "[band_residual]", "# mean relative residual of the",
                      "# digitized band against each published model"])
        for model_id, model in models.items():
            rows = ", ".join(
                f"{{ contact_half_width_m = {_format_float(half_width)}, "
                f"mean_relative_residual = "
                f"{_format_float(mean_relative_residual(model.extrapolating, series.observations.for_plate(half_width)))} }}"
                for half_width in half_widths
                if _plate_observation_count(series, half_width) >= 2
            )
            lines.append(f"{model_id} = [{rows}]")

    if bias and series is not None:
        lines.extend([
            "",
            "# Per-figure axis-calibration bias, measured by tracing the published",
            "# Bekker curve on each source figure and comparing against the",
            "# transcribed parameters. Points below the minimum sinkage are",
            "# excluded: there a fixed click error is a large relative one.",
            "[calibration_bias]",
            f"minimum_sinkage_m = {_format_float(CALIBRATION_BIAS_MINIMUM_SINKAGE_M)}",
            "by_plate = ["
        ])
        for half_width in half_widths:
            if half_width in bias:
                lines.append(
                    f"  {{ contact_half_width_m = {_format_float(half_width)}, "
                    f"relative_bias = {_format_float(bias[half_width])} }},"
                )
        lines.append("]")

        observed = [float(b) for b in series.observations.contact_half_widths]
        if sorted(bias) == sorted(observed):
            lines.extend([
                "",
                "# Envelope of each parameter when the measured biases above are",
                "# reassigned across plates, over all permutations. This is a bound",
                "# on the systematic reachable with these biases in their worst",
                "# arrangement, not a standard deviation. It is wide because the",
                "# plates carry very unequal leverage in the 1/b regression, so it",
                "# matters which plate a given bias lands on.",
            ])
            for model_id in models:
                for scheme in WEIGHTINGS:
                    bound = parameter_bound_under_bias_permutation(
                        model_id, series.observations,
                        [bias[plate] for plate in observed],
                        weighting=scheme,
                    )
                    lines.extend([
                        "",
                        "[[calibration_bias_bound]]",
                        f'model = "{model_id}"',
                        f'weighting = "{scheme}"',
                        *(
                            f"{name} = {{ minimum = {_format_float(low)}, "
                            f"maximum = {_format_float(high)} }}"
                            for name, (low, high) in bound.items()
                        ),
                    ])

    if series is None:
        return "\n".join(lines) + "\n"

    for (model_id, weighting), fit in fits.items():
        published = models[model_id]
        lines.extend(
            [
                "",
                "[[fit]]",
                f'model = "{model_id}"',
                f'weighting = "{weighting}"',
                f"observation_count = {fit.observation_count}",
                f"plate_count = {fit.plate_count}",
                "coefficient_of_determination = "
                + _format_float(
                    coefficient_of_determination(fit.model, series.observations)
                ),
                "parameters = " + _inline_table(dict(fit.parameters)),
                "relative_deviation_from_published = "
                + _inline_table(
                    {
                        name: (value - published.parameters[name])
                        / published.parameters[name]
                        for name, value in fit.parameters.items()
                    }
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def _comparison_rows(
    published: CalibratedContactModel, fit: FittedContactModel
) -> list[str]:
    rows = [
        f"    {'parameter':22s} {'published':>16s} {'re-fitted':>16s} {'relative':>10s}"
    ]
    for name, published_value in published.parameters.items():
        recovered = fit.parameters[name]
        relative = (
            abs(recovered - published_value) / abs(published_value)
            if published_value
            else float("nan")
        )
        rows.append(
            f"    {name:22s} {published_value:16.4f} {recovered:16.4f} {relative:9.2%}"
        )
    return rows


def run_kls1(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    soil = load_soil(arguments.soil)
    dataset = next(iter(soil.datasets.values()))
    missing = [
        model_id
        for model_id in (REFERENCE_MODEL, COMPARED_MODEL)
        if model_id not in dataset.models
    ]
    if missing:
        parser.error(
            f"{arguments.soil} has no verified model(s) {missing}; it holds "
            f"{sorted(dataset.models)}"
        )
    models = {
        model_id: dataset.models[model_id]
        for model_id in (REFERENCE_MODEL, COMPARED_MODEL)
    }
    half_widths = [plate.contact_half_width_m for plate in dataset.apparatus.plates]

    series: PressureSinkageSeries | None = None
    fits: dict[tuple[str, str], FittedContactModel] = {}
    if arguments.series.is_file():
        try:
            series = load_pressure_sinkage_series(arguments.series)
        except SeriesFileError as error:
            print(f"cannot read the digitized series: {error}", file=sys.stderr)
            return 1
        try:
            fits = {
                (model_id, weighting): fit_contact_model(
                    model_id, series.observations, weighting=weighting
                )
                for model_id in models
                for weighting in WEIGHTINGS
            }
        except ValueError as error:
            print(
                f"the series loaded but cannot support a fit: {error}. "
                "Plotting the published models against it instead.",
                file=sys.stderr,
            )
    else:
        print(
            f"no digitized series at {arguments.series}, so the figure shows "
            "published parameters only. To add measurements, write that manifest "
            "beside a CSV whose columns are contact_half_width_m, sinkage_m, "
            "pressure_kPa.",
            file=sys.stderr,
        )

    bias: dict[float, float] = {}
    if arguments.trace.is_file():
        try:
            trace = load_pressure_sinkage_series(arguments.trace)
        except SeriesFileError as error:
            print(f"cannot read the published-curve trace: {error}", file=sys.stderr)
            return 1
        usable = trace.observations.above_sinkage(CALIBRATION_BIAS_MINIMUM_SINKAGE_M)
        bias = {
            float(half_width): mean_relative_residual(
                models[REFERENCE_MODEL].extrapolating,
                usable.for_plate(float(half_width)),
            )
            for half_width in usable.contact_half_widths
        }
    else:
        print(
            f"no published-curve trace at {arguments.trace}; the calibration bias "
            "of the digitization cannot be measured",
            file=sys.stderr,
        )

    figure = _kls1_figure(models, half_widths, series, arguments.kls1_report)
    arguments.kls1_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.kls1_figure)
    plt.close(figure)

    arguments.kls1_report.parent.mkdir(parents=True, exist_ok=True)
    arguments.kls1_report.write_text(
        _kls1_report(
            soil, dataset, models, half_widths, arguments.soil, series, fits, bias
        ),
        encoding="utf-8",
    )

    print(f"soil     {soil.id} ({dataset.doi})")
    print(f"models   {', '.join(models)}")
    if series is not None:
        print(f"series   {series.id}, {series.observations.count} points")
        for (model_id, weighting), fit in fits.items():
            marker = " (default)" if weighting == DEFAULT_WEIGHTING else ""
            print(f"  {model_id}, weighting {weighting}{marker}")
            print("\n".join(_comparison_rows(models[model_id], fit)))
    print(f"figure   {arguments.kls1_figure}")
    print(f"report   {arguments.kls1_report}")
    return 0


# --- GRC-1: one model, two fitting windows, raw channels


def _grc1_half_width(channels: BevameterChannels, plate: str) -> float:
    return channels.test(f"{plate}-{GRC1_REPEATS[0]}").plate.contact_half_width_m


def _grc1_ensemble(
    channels: BevameterChannels, plate: str, top_kPa: float, step_kPa: float
) -> EnsembleCurve:
    return ensemble(
        sample_positions=[
            channels.pressure_kPa(f"{plate}-{r}") for r in GRC1_REPEATS
        ],
        sample_values=[channels.sinkage_m(f"{plate}-{r}") for r in GRC1_REPEATS],
        positions=np.arange(0.0, top_kPa + step_kPa / 2.0, step_kPa),
    )


def _grc1_window(
    channels: BevameterChannels, block: dict[str, Any]
) -> tuple[dict[str, EnsembleCurve], PressureSinkageObservations]:
    curves: dict[str, EnsembleCurve] = {}
    half_width, sinkage, pressure = [], [], []
    for plate, top, dropped in zip(
        GRC1_PLATE_NAMES,
        block["resampling_endpoints_kPa"],
        block["leading_samples_dropped"],
    ):
        curve = _grc1_ensemble(channels, plate, top, block["resampling_step_kPa"])
        curves[plate] = curve
        grid = curve.positions[dropped:]
        mean = curve.mean_values[dropped:]
        usable = (grid > 0.0) & (mean > 0.0)
        half_width.append(
            np.full(int(usable.sum()), _grc1_half_width(channels, plate))
        )
        sinkage.append(mean[usable])
        pressure.append(grid[usable])
    return curves, PressureSinkageObservations(
        contact_half_width_m=np.concatenate(half_width),
        sinkage_m=np.concatenate(sinkage),
        pressure_kPa=np.concatenate(pressure),
    )


def _grc1_panel(
    axes: Axes,
    channels: BevameterChannels,
    block: dict[str, Any],
    curves: dict[str, EnsembleCurve],
    fit: FittedContactModel,
) -> None:
    top = max(block["resampling_endpoints_kPa"])
    deepest = 0.0
    for index, plate in enumerate(GRC1_PLATE_NAMES):
        half_width = _grc1_half_width(channels, plate)
        for repeat in GRC1_REPEATS:
            identifier = f"{plate}-{repeat}"
            observed_pressure = channels.pressure_kPa(identifier)
            observed_sinkage = channels.sinkage_m(identifier)
            inside = observed_pressure <= top
            _draw_plate_markers(
                axes, index, observed_pressure[inside], observed_sinkage[inside],
                GRC1_MARKER, zorder=3,
            )
            deepest = max(deepest, float(observed_sinkage[inside].max() * 1e3))
        depth = np.linspace(
            0.0, float(curves[plate].mean_values.max()), GRC1_CURVE_SAMPLES
        )
        _draw_plate_curve(
            axes, index,
            np.asarray(
                fit.model.pressure(sinkage=depth, contact_half_width=half_width)
            ),
            depth, GRC1_FIT_COLOR, 1.6, zorder=5,
        )

    _finish_axes(axes, deepest * 1.06)
    axes.set_xlim(0.0, top)
    # The fitted parameters live in the title rather than inside the axes. The
    # empty corner moves between the panels, because the entire-range panel
    # carries the punch-through scatter, and a label placed by eye for one
    # dataset lands on the data of the next.
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


def _grc1_figure(
    channels: BevameterChannels,
    blocks: list[dict[str, Any]],
    fits: dict[str, FittedContactModel],
    windows: dict[str, dict[str, EnsembleCurve]],
    report_path: Path,
) -> Figure:
    with plt.rc_context(cast(Any, GRC1_STYLE)):
        figure, axes = plt.subplots(1, len(blocks), squeeze=False)
        for column, block in enumerate(blocks):
            _grc1_panel(
                axes[0][column], channels, block,
                windows[block["id"]], fits[block["id"]],
            )
        axes[0][0].legend(
            handles=[
                _plate_legend_handle(index, GRC1_MARKER)
                for index in range(len(GRC1_PLATE_NAMES))
            ],
            labels=[
                f"b = {_grc1_half_width(channels, plate) * 1e3:.1f} mm"
                for plate in GRC1_PLATE_NAMES
            ],
            loc="lower left",
        )

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


def _grc1_report(
    channels: BevameterChannels,
    blocks: list[dict[str, Any]],
    fits: dict[str, FittedContactModel],
    windows: dict[str, dict[str, EnsembleCurve]],
) -> str:
    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# Generated by calibration/contact/pressure_sinkage.py. Do not edit.",
        "#",
        "# Reproduction of published fits from raw sensor channels. No timestamp:",
        "# re-running against unchanged inputs must leave this byte-identical.",
        "",
        f"schema_version = {REPORT_SCHEMA_VERSION}",
        f'model = "{GRC1_MODEL_ID}"',
        f'weighting = "{GRC1_WEIGHTING}"',
        f'estimator = "{GRC1_ESTIMATOR}"',
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
        lines += [f"{name} = {_format_float(fit.parameters[name])}" for name in GRC1_PARAMETERS]
        lines += ["", "[window.published]"]
        lines += [f"{name} = {_format_float(float(block[name]))}" for name in GRC1_PARAMETERS]
        lines += ["", "[window.relative_deviation]"]
        lines += [
            f"{name} = "
            + _format_float(
                abs(fit.parameters[name] - float(block[name]))
                / abs(float(block[name]))
            )
            for name in GRC1_PARAMETERS
        ]
        lines += ["", "[window.maximum_deviation_mm]"]
        for plate in GRC1_PLATE_NAMES:
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


def run_grc1(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
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
        curves, observations = _grc1_window(channels, block)
        windows[block["id"]] = curves
        fits[block["id"]] = fit_contact_model(
            GRC1_MODEL_ID, observations,
            weighting=GRC1_WEIGHTING, estimator=GRC1_ESTIMATOR,
        )

    print(f"  {'window':>24s} {'parameter':>20s} {'fitted':>14s} {'published':>13s} {'rel':>9s}")
    worst = 0.0
    for block in blocks:
        fit = fits[block["id"]]
        for name in GRC1_PARAMETERS:
            published = float(block[name])
            relative = abs(fit.parameters[name] - published) / abs(published)
            worst = max(worst, relative)
            print(
                f"  {block['id']:>24s} {name:>20s} {fit.parameters[name]:14.4f} "
                f"{published:13.4f} {relative * 100:8.4f}%"
            )
    print(f"\n  worst relative deviation across both windows: {worst * 100:.4f}%")

    arguments.grc1_figure.parent.mkdir(parents=True, exist_ok=True)
    arguments.grc1_report.parent.mkdir(parents=True, exist_ok=True)
    figure = _grc1_figure(channels, blocks, fits, windows, arguments.grc1_report)
    figure.savefig(arguments.grc1_figure, dpi=GRC1_STYLE["figure.dpi"])
    plt.close(figure)
    arguments.grc1_report.write_text(
        _grc1_report(channels, blocks, fits, windows), encoding="utf-8"
    )
    print(f"figure   {arguments.grc1_figure}")
    print(f"report   {arguments.grc1_report}")
    return 0


CAMPAIGNS: Final = {"kls1": run_kls1, "grc1": run_grc1}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the pressure-sinkage calibration outputs."
    )
    parser.add_argument(
        "--campaign", choices=[*CAMPAIGNS, "all"], default="all",
        help="which campaign to regenerate; both by default",
    )
    parser.add_argument("--soil", type=Path, default=DEFAULT_SOIL_PATH)
    parser.add_argument("--series", type=Path, default=DEFAULT_SERIES_PATH)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("--channels", type=Path, default=DEFAULT_CHANNELS_PATH)
    parser.add_argument("--kls1-figure", type=Path, default=KLS1_FIGURE_PATH)
    parser.add_argument("--kls1-report", type=Path, default=KLS1_REPORT_PATH)
    parser.add_argument("--grc1-figure", type=Path, default=GRC1_FIGURE_PATH)
    parser.add_argument("--grc1-report", type=Path, default=GRC1_REPORT_PATH)
    arguments = parser.parse_args(argv)

    selected = list(CAMPAIGNS) if arguments.campaign == "all" else [arguments.campaign]
    status = 0
    for index, name in enumerate(selected):
        if index:
            print()
        print(f"=== {name} ===")
        status = max(status, CAMPAIGNS[name](arguments, parser))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
