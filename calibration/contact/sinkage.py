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
# The legend carries linestyle only. Colour identifies the plate and is read off
# the direct labels at the curve ends, because every plate draws both a solid and
# a dashed curve and a single coloured swatch would misdescribe them.
#
# Colours are the first three slots of a categorical palette validated for
# colour-vision deficiency at all pairs (worst CVD dE 9.2, normal-vision 24.0).
# Curves carry direct labels because one slot sits below 3:1 contrast on the
# figure surface, so identity may not rest on colour alone.

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from biome.fitting import (
    FittedContactModel,
    coefficient_of_determination,
    fit_contact_model,
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
    REPOSITORY_ROOT / "data" / "literature" / "lim2021-figure12.toml"
)
DEFAULT_FIGURE_PATH: Final = (
    Path(__file__).resolve().parent / "figures" / "kls1-pressure-sinkage.png"
)
DEFAULT_REPORT_PATH: Final = (
    Path(__file__).resolve().parent / "results" / "kls1-pressure-sinkage.toml"
)

REFERENCE_MODEL: Final = "bekker"
COMPARED_MODEL: Final = "reece"
MODEL_LINESTYLES: Final[dict[str, Any]] = {
    REFERENCE_MODEL: "solid",
    COMPARED_MODEL: (0, (5, 2)),
}

PLATE_COLOURS: Final = ("#2a78d6", "#eb6834", "#1baf7a")
INK_PRIMARY: Final = "#0b0b0b"
INK_SECONDARY: Final = "#52514e"
INK_MUTED: Final = "#8a8880"
SURFACE: Final = "#fcfcfb"
CURVE_SAMPLES: Final = 400
LABEL_MINIMUM_GAP_FRACTION: Final = 0.05
REPORT_SCHEMA_VERSION: Final = 1

FIGURE_STYLE: Final[dict[str, Any]] = {
    "figure.figsize": (7.2, 5.2),
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
    "figure.subplot.top": 0.840,
    "figure.subplot.bottom": 0.205,
    "figure.subplot.left": 0.100,
    "figure.subplot.right": 0.965,
}


def _plate_colour(index: int) -> str:
    if index >= len(PLATE_COLOURS):
        raise SystemExit(
            f"the validated palette carries {len(PLATE_COLOURS)} categorical "
            f"slots but this dataset has {index + 1} plates; fold the extra "
            "plates into small multiples rather than inventing a hue"
        )
    return PLATE_COLOURS[index]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _separated(values: list[float], minimum_gap: float) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    separated = list(values)
    for previous, current in zip(order, order[1:], strict=False):
        if separated[current] - separated[previous] < minimum_gap:
            separated[current] = separated[previous] + minimum_gap
    return separated


def _sinkage_sweep(published: CalibratedContactModel) -> np.ndarray:
    bounds = published.sinkage_validity
    return np.linspace(max(bounds.min, bounds.max * 1e-6), bounds.max, CURVE_SAMPLES)


def _draw_curves(
    axes: Axes,
    models: dict[str, CalibratedContactModel],
    half_widths: list[float],
    depth: np.ndarray,
) -> None:
    endpoints: list[float] = []
    for index, half_width in enumerate(half_widths):
        for model_id, model in models.items():
            pressure = model.pressure(sinkage=depth, contact_half_width=half_width)
            axes.plot(
                depth * 1e3,
                pressure,
                color=_plate_colour(index),
                linewidth=1.5,
                linestyle=MODEL_LINESTYLES[model_id],
                zorder=3,
            )
            if model_id == REFERENCE_MODEL:
                endpoints.append(float(pressure[-1]))

    axes.set_ylim(bottom=0.0, top=max(endpoints) * 1.08)
    gap = axes.get_ylim()[1] * LABEL_MINIMUM_GAP_FRACTION
    for index, height in enumerate(_separated(endpoints, gap)):
        axes.annotate(
            f"b = {half_widths[index] * 1e3:.1f} mm",
            xy=(depth[-1] * 1e3, height),
            xytext=(5, 0),
            textcoords="offset points",
            color=INK_SECONDARY,
            fontsize=8.5,
            va="center",
            annotation_clip=False,
        )


def _legend_handles(has_series: bool) -> list[Line2D]:
    handles = [
        Line2D(
            [], [], color=INK_SECONDARY, linewidth=1.5,
            linestyle=MODEL_LINESTYLES[model_id],
            label=f"{model_id.title()} (published)",
        )
        for model_id in (REFERENCE_MODEL, COMPARED_MODEL)
    ]
    if has_series:
        handles.append(
            Line2D(
                [], [], color=INK_SECONDARY, marker="o", markersize=3.5,
                markerfacecolor=SURFACE, linestyle="none", label="digitized points",
            )
        )
    return handles


def _draw_observations(
    axes: Axes,
    series: PressureSinkageSeries,
    half_widths: list[float],
) -> None:
    for index, half_width in enumerate(half_widths):
        selected = series.observations.for_plate(half_width)
        axes.errorbar(
            selected.sinkage_m * 1e3,
            selected.pressure_kPa,
            xerr=series.digitization.sinkage_uncertainty_m * 1e3,
            yerr=series.digitization.pressure_uncertainty_kPa,
            fmt="o",
            markersize=3.5,
            markerfacecolor=SURFACE,
            markeredgecolor=_plate_colour(index),
            markeredgewidth=1.1,
            ecolor=INK_MUTED,
            elinewidth=0.7,
            capsize=0,
            linestyle="none",
            zorder=5,
        )


def build_figure(
    models: dict[str, CalibratedContactModel],
    half_widths: list[float],
    series: PressureSinkageSeries | None,
    report_path: Path,
) -> Figure:
    published = models[REFERENCE_MODEL]
    depth = _sinkage_sweep(published)
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots()
        _draw_curves(axes, models, half_widths, depth)
        if series is not None:
            _draw_observations(axes, series, half_widths)

        axes.set_xlabel("sinkage  (mm)")
        axes.set_ylabel("pressure  (kPa)")
        axes.set_xlim(0.0, published.sinkage_validity.max * 1e3 * 1.17)
        axes.legend(handles=_legend_handles(series is not None), loc="upper left")
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        axes.set_axisbelow(True)

        subtitle = (
            "Published parameters only. No digitized measurements yet,\n"
            "so this is the transcribed fit, not a comparison against data."
            if series is None
            else (
                f"{series.observations.count} points digitized from "
                f"{series.source.figure} of {series.source.doi},\n"
                "shown with their digitization uncertainty."
            )
        )
        figure.text(
            0.100, 0.960,
            f"KLS-1 pressure-sinkage: {REFERENCE_MODEL.title()} against "
            f"{COMPARED_MODEL.title()}",
            fontsize=12, color=INK_PRIMARY, weight="bold", ha="left", va="top",
        )
        figure.text(
            0.100, 0.908, subtitle,
            fontsize=8.5, color=INK_SECONDARY, ha="left", va="top",
        )
        figure.text(
            0.100, 0.098,
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
    fits: dict[str, FittedContactModel],
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

    for model_id, fit in fits.items() if series is not None else ():
        published = models[model_id]
        lines.extend(
            [
                "",
                "[[fit]]",
                f'model = "{model_id}"',
                f'weighting = "{fit.weighting}"',
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
    fits: dict[str, FittedContactModel] = {}
    if arguments.series.is_file():
        try:
            series = load_pressure_sinkage_series(arguments.series)
        except SeriesFileError as error:
            print(f"cannot read the digitized series: {error}", file=sys.stderr)
            return 1
        fits = {
            model_id: fit_contact_model(model_id, series.observations)
            for model_id in models
        }
    else:
        print(
            f"no digitized series at {arguments.series}, so the figure shows "
            "published parameters only. To add measurements, write that manifest "
            "beside a CSV whose columns are contact_half_width_m, sinkage_m, "
            "pressure_kPa.",
            file=sys.stderr,
        )

    figure = build_figure(models, half_widths, series, arguments.report)
    arguments.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.figure)
    plt.close(figure)

    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        build_report(soil, dataset, models, half_widths, arguments.soil, series, fits),
        encoding="utf-8",
    )

    print(f"soil     {soil.id} ({dataset.doi})")
    print(f"models   {', '.join(models)}")
    if series is not None:
        print(f"series   {series.id}, {series.observations.count} points")
        for model_id, fit in fits.items():
            print(f"  {model_id}, weighting {fit.weighting}")
            print("\n".join(_comparison_rows(models[model_id], fit)))
    print(f"figure   {arguments.figure}")
    print(f"report   {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
