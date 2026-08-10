# SPDX-License-Identifier: Apache-2.0
#
# calibration/contact/sinkage.py — Day 0 pressure-sinkage calibration.
#
# A thin runner. The fitting lives in biome.fitting and the loading in biome.io
# so that both are unit tested and type checked; this file wires paths, draws,
# and writes the result.
#
# Two outputs, because a figure is not a result. The figure shows Bekker against
# Reece over the range their parameters were fitted over. The report records the
# numbers behind it together with the SHA-256 of every input, so a plot can be
# traced to the exact bytes that produced it. The report carries no timestamp:
# re-running against unchanged inputs must leave it byte-identical, so any diff
# means a result actually moved.
#
# Axes follow the bevameter convention used throughout the terramechanics
# literature: pressure across, sinkage increasing down the page, so a curve falls
# the way the plate it describes sinks.
#
# Plate identity carries on line style and marker shape, model identity on
# color. That assignment is the way round it is because the two model curves sit
# nearly on top of each other while the three plate curves are well separated: a
# dash pattern cannot separate coincident strokes, and a hue can.
#
# No shaded bands. The replicate scatter is the marker cloud itself, and drawing
# an envelope over it asserts a distribution the digitization does not support.

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
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

from biome.fitting import (
    DEFAULT_WEIGHTING,
    parameter_bound_under_bias_permutation,
    coefficient_of_determination_ceiling,
    FittedContactModel,
    WeightingScheme,
    coefficient_of_determination,
    fit_contact_model,
    mean_relative_residual,
    relative_deviation,
)
from biome.io.series import (
    PressureSinkageSeries,
    SeriesFileError,
    load_pressure_sinkage_series,
)
from biome.io.soil import CalibratedContactModel, Dataset, Soil, load_soil

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_SOIL_PATH: Final = REPOSITORY_ROOT / "data" / "soils" / "kls1.toml"
DEFAULT_SERIES_PATH: Final = (
    REPOSITORY_ROOT / "data" / "literature" / "lim2021-pressure-sinkage.toml"
)
DEFAULT_TRACE_PATH: Final = (
    REPOSITORY_ROOT / "data" / "literature" / "lim2021-published-bekker-curve.toml"
)
CALIBRATION_BIAS_MINIMUM_SINKAGE_M: Final = 0.010
DEFAULT_FIGURE_PATH: Final = (
    Path(__file__).resolve().parent / "figures" / "kls1-pressure-sinkage.png"
)
DEFAULT_REPORT_PATH: Final = (
    Path(__file__).resolve().parent / "results" / "kls1-pressure-sinkage.toml"
)

REFERENCE_MODEL: Final = "bekker"
COMPARED_MODEL: Final = "reece"
MODEL_COLORS: Final[dict[str, str]] = {
    REFERENCE_MODEL: "#1f4e9c",
    COMPARED_MODEL: "#d4570a",
}
# The two fits differ by a few percent, so their curves very nearly coincide.
# Drawing the compared model thinner and on top leaves the reference visible as a
# wider stroke underneath rather than hidden by whichever was drawn last.
MODEL_LINEWIDTHS: Final[dict[str, float]] = {
    REFERENCE_MODEL: 2.6,
    COMPARED_MODEL: 1.0,
}

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
CURVE_SAMPLES: Final = 400
MARKER_SIZE: Final = 3.4
PLATE_LEGEND_ANCHOR: Final = 0.755
WEIGHTINGS: Final[tuple[WeightingScheme, ...]] = ("uniform", "pressure_squared")
REPORT_SCHEMA_VERSION: Final = 1

FIGURE_STYLE: Final[dict[str, Any]] = {
    "figure.figsize": (7.8, 5.4),
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
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.size": 10,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "savefig.facecolor": SURFACE,
    "figure.subplot.top": 0.845,
    "figure.subplot.bottom": 0.215,
    "figure.subplot.left": 0.085,
    "figure.subplot.right": 0.965,
}


def _plate_style(index: int) -> tuple[Any, str]:
    if index >= len(PLATE_LINESTYLES):
        raise SystemExit(
            f"the figure carries {len(PLATE_LINESTYLES)} distinguishable plate "
            f"styles but this dataset has {index + 1} plates; fold the extra "
            "plates into small multiples rather than inventing a dash pattern "
            "no reader can name"
        )
    return PLATE_LINESTYLES[index], PLATE_MARKERS[index]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _sinkage_sweep(published: CalibratedContactModel) -> np.ndarray:
    bounds = published.sinkage_validity
    return np.linspace(max(bounds.min, bounds.max * 1e-6), bounds.max, CURVE_SAMPLES)


def _plate_observation_count(
    series: PressureSinkageSeries, half_width: float
) -> int:
    return int(
        np.count_nonzero(series.observations.contact_half_width_m == half_width)
    )


def _draw_observations(
    axes: Axes, series: PressureSinkageSeries, half_widths: list[float]
) -> None:
    for index, half_width in enumerate(half_widths):
        if _plate_observation_count(series, half_width) < 2:
            continue
        selected = series.observations.for_plate(half_width)
        axes.plot(
            selected.pressure_kPa,
            selected.sinkage_m * 1e3,
            marker=_plate_style(index)[1],
            markersize=MARKER_SIZE,
            markerfacecolor="none",
            markeredgecolor=INK_SECONDARY,
            markeredgewidth=0.8,
            linestyle="none",
            zorder=5,
        )


def _draw_curves(
    axes: Axes,
    models: dict[str, CalibratedContactModel],
    half_widths: list[float],
    depth: np.ndarray,
) -> None:
    for index, half_width in enumerate(half_widths):
        linestyle, _ = _plate_style(index)
        for order, model_id in enumerate((REFERENCE_MODEL, COMPARED_MODEL)):
            model = models.get(model_id)
            if model is None:
                continue
            axes.plot(
                model.pressure(sinkage=depth, contact_half_width=half_width),
                depth * 1e3,
                color=MODEL_COLORS[model_id],
                linewidth=MODEL_LINEWIDTHS[model_id],
                linestyle=linestyle, zorder=4 + order,
            )

    axes.set_xlim(left=0.0)


def _legend_entries(has_series: bool) -> tuple[list[Any], list[str]]:
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
                    markersize=MARKER_SIZE, markerfacecolor="none",
                    markeredgewidth=0.8, linestyle="none",
                )
                for marker in PLATE_MARKERS
            )
        )
        labels.append("digitized points")
    return handles, labels


def _plate_legend_handles(
    half_widths: list[float],
    published: CalibratedContactModel,
    series: PressureSinkageSeries | None,
) -> list[Any]:
    handles: list[Any] = []
    for index, half_width in enumerate(half_widths):
        linestyle, marker = _plate_style(index)
        label = f"b = {half_width * 1e3:.1f} mm"
        if series is not None and _plate_observation_count(series, half_width) >= 2:
            residual = mean_relative_residual(
                published.extrapolating, series.observations.for_plate(half_width)
            )
            label += f"    residual {residual * 100:+.1f}%"
        handles.append(
            Line2D(
                [], [], color=INK_SECONDARY, linewidth=1.4, linestyle=linestyle,
                marker=marker, markersize=MARKER_SIZE, markerfacecolor="none",
                markeredgewidth=0.8, label=label,
            )
        )
    return handles


def build_figure(
    models: dict[str, CalibratedContactModel],
    half_widths: list[float],
    series: PressureSinkageSeries | None,
    report_path: Path,
) -> Figure:
    published = models[REFERENCE_MODEL]
    depth = _sinkage_sweep(published)
    with plt.rc_context(cast(Any, FIGURE_STYLE)):
        figure, axes = plt.subplots()
        if series is not None:
            _draw_observations(axes, series, half_widths)
        _draw_curves(axes, models, half_widths, depth)

        axes.set_xlabel("pressure  (kPa)")
        axes.set_ylabel("sinkage  (mm)")
        axes.set_ylim(published.sinkage_validity.max * 1e3 * 1.02, 0.0)
        handles, labels = _legend_entries(series is not None)
        style_legend = axes.legend(
            handles=handles, labels=labels, loc="upper right",
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.7)},
        )
        axes.add_artist(style_legend)
        axes.legend(
            handles=_plate_legend_handles(half_widths, published, series),
            loc="upper right",
            bbox_to_anchor=(1.0, PLATE_LEGEND_ANCHOR),
        )
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        axes.set_axisbelow(True)

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


def _format_float(value: float) -> str:
    return repr(float(value))


def _inline_table(values: dict[str, float]) -> str:
    return (
        "{ "
        + ", ".join(f"{name} = {_format_float(value)}" for name, value in values.items())
        + " }"
    )


def build_report(
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
        "# Generated by calibration/contact/sinkage.py. Do not edit by hand.",
        "#",
        "# Deterministic by construction: no timestamp, so re-running against",
        "# unchanged inputs leaves this file byte-identical and any diff means a",
        "# result moved. Input digests pin the exact bytes behind every number.",
        "",
        f"schema_version = {REPORT_SCHEMA_VERSION}",
        'generated_by = "calibration/contact/sinkage.py"',
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the Day 0 pressure-sinkage calibration outputs."
    )
    parser.add_argument("--soil", type=Path, default=DEFAULT_SOIL_PATH)
    parser.add_argument("--series", type=Path, default=DEFAULT_SERIES_PATH)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    arguments = parser.parse_args(argv)

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

    figure = build_figure(models, half_widths, series, arguments.report)
    arguments.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.figure)
    plt.close(figure)

    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        build_report(
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
    print(f"figure   {arguments.figure}")
    print(f"report   {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
