# SPDX-License-Identifier: Apache-2.0
#
# calibration/contact/sinkage.py — Day 0 pressure-sinkage calibration figure.
#
# A thin runner. The fitting lives in biome.fitting and the loading in biome.io
# so that both are unit tested and type checked; this file only wires paths,
# draws, and reports.
#
# With no digitized series present it plots the published fitted models over
# their fitted range and says so on the figure. With a series present it adds
# the measured points and a curve re-fitted from them, and prints the recovered
# parameters beside the published ones. Recovering the published parameters is
# what makes a digitization trustworthy, so that comparison is the result, not
# the picture.
#
# Colors are the first three slots of a categorical palette validated for
# color-vision deficiency at all pairs (worst CVD dE 9.2, normal-vision 24.0).
# Curves carry direct labels because one slot sits below 3:1 contrast on the
# figure surface, so identity may not rest on color alone.

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from biome.fitting import (
    FittedContactModel,
    PressureSinkageObservations,
    coefficient_of_determination,
    fit_contact_model,
)
from biome.io.series import (
    PressureSinkageSeries,
    SeriesFileError,
    load_pressure_sinkage_series,
)
from biome.io.soil import CalibratedContactModel, load_soil

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_SOIL_PATH: Final = REPOSITORY_ROOT / "data" / "soils" / "kls1.toml"
DEFAULT_SERIES_PATH: Final = (
    REPOSITORY_ROOT / "data" / "literature" / "lim2021-figure12.toml"
)
DEFAULT_OUTPUT_PATH: Final = (
    Path(__file__).resolve().parent / "figures" / "kls1-pressure-sinkage.png"
)

PLATE_COLOURS: Final = ("#2a78d6", "#eb6834", "#1baf7a")
INK_PRIMARY: Final = "#0b0b0b"
INK_SECONDARY: Final = "#52514e"
INK_MUTED: Final = "#8a8880"
SURFACE: Final = "#fcfcfb"
CURVE_SAMPLES: Final = 400

FIGURE_STYLE: Final = {
    "figure.figsize": (7.2, 5.0),
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
    "figure.subplot.top": 0.82,
    "figure.subplot.bottom": 0.17,
    "figure.subplot.left": 0.10,
    "figure.subplot.right": 0.97,
}


def _plate_colour(index: int) -> str:
    if index >= len(PLATE_COLOURS):
        raise SystemExit(
            f"the validated palette carries {len(PLATE_COLOURS)} categorical "
            f"slots but this dataset has {index + 1} plates; fold the extra "
            "plates into small multiples rather than inventing a hue"
        )
    return PLATE_COLOURS[index]


def _draw_published(
    axes: Axes, model: CalibratedContactModel, half_widths: list[float]
) -> None:
    bounds = model.sinkage_validity
    depth = np.linspace(max(bounds.min, 1e-6), bounds.max, CURVE_SAMPLES)
    for index, half_width in enumerate(half_widths):
        pressure = model.pressure(sinkage=depth, contact_half_width=half_width)
        axes.plot(
            depth * 1e3,
            pressure,
            color=_plate_colour(index),
            linewidth=1.6,
            solid_capstyle="round",
            zorder=3,
        )
        axes.annotate(
            f"b = {half_width * 1e3:.1f} mm",
            xy=(depth[-1] * 1e3, float(pressure[-1])),
            xytext=(6, 0),
            textcoords="offset points",
            color=INK_SECONDARY,
            fontsize=8.5,
            va="center",
        )


def _draw_refit(
    axes: Axes, fitted: FittedContactModel, half_widths: list[float], bounds: tuple[float, float]
) -> None:
    depth = np.linspace(max(bounds[0], 1e-6), bounds[1], CURVE_SAMPLES)
    for index, half_width in enumerate(half_widths):
        axes.plot(
            depth * 1e3,
            fitted.model.pressure(sinkage=depth, contact_half_width=half_width),
            color=_plate_colour(index),
            linewidth=1.4,
            linestyle=(0, (5, 2)),
            zorder=4,
        )


def _draw_observations(
    axes: Axes, series: PressureSinkageSeries, half_widths: list[float]
) -> None:
    observations = series.observations
    for index, half_width in enumerate(half_widths):
        selected = observations.for_plate(half_width)
        axes.errorbar(
            selected.sinkage_m * 1e3,
            selected.pressure_kPa,
            xerr=series.digitization.sinkage_uncertainty_m * 1e3,
            yerr=series.digitization.pressure_uncertainty_kPa,
            fmt="o",
            markersize=4,
            markerfacecolor=SURFACE,
            markeredgecolor=_plate_colour(index),
            markeredgewidth=1.2,
            ecolor=INK_MUTED,
            elinewidth=0.7,
            capsize=0,
            linestyle="none",
            zorder=5,
        )


def _legend_handles(half_widths: list[float], has_series: bool) -> list[Line2D]:
    handles = [
        Line2D(
            [], [], color=_plate_colour(index), linewidth=1.6,
            label=f"b = {half_width * 1e3:.1f} mm",
        )
        for index, half_width in enumerate(half_widths)
    ]
    handles.append(
        Line2D([], [], color=INK_SECONDARY, linewidth=1.6, label="published fit")
    )
    if has_series:
        handles.append(
            Line2D(
                [], [], color=INK_SECONDARY, linewidth=1.4, linestyle=(0, (5, 2)),
                label="re-fitted from digitized points",
            )
        )
        handles.append(
            Line2D(
                [], [], color=INK_SECONDARY, marker="o", markersize=4,
                markerfacecolor=SURFACE, linestyle="none",
                label="digitized points, with digitization uncertainty",
            )
        )
    return handles


def _compare(
    published: CalibratedContactModel, fitted: FittedContactModel
) -> list[str]:
    lines = [
        f"  {'parameter':22s} {'published':>16s} {'re-fitted':>16s} {'relative':>10s}"
    ]
    for name, published_value in published.parameters.items():
        recovered = fitted.parameters[name]
        relative = (
            abs(recovered - published_value) / abs(published_value)
            if published_value
            else float("nan")
        )
        lines.append(
            f"  {name:22s} {published_value:16.4f} {recovered:16.4f} {relative:9.2%}"
        )
    return lines


def build_figure(
    model_id: str,
    published: CalibratedContactModel,
    half_widths: list[float],
    series: PressureSinkageSeries | None,
    fitted: FittedContactModel | None,
) -> plt.Figure:
    with plt.rc_context(FIGURE_STYLE):
        figure, axes = plt.subplots()
        _draw_published(axes, published, half_widths)
        if series is not None and fitted is not None:
            bounds = (published.sinkage_validity.min, published.sinkage_validity.max)
            _draw_refit(axes, fitted, half_widths, bounds)
            _draw_observations(axes, series, half_widths)

        axes.set_xlabel("sinkage  (mm)")
        axes.set_ylabel("pressure  (kPa)")
        axes.set_xlim(0.0, published.sinkage_validity.max * 1e3 * 1.16)
        axes.set_ylim(bottom=0.0)
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        axes.set_axisbelow(True)
        axes.legend(handles=_legend_handles(half_widths, series is not None), loc="upper left")

        if series is None or fitted is None:
            subtitle = (
                "Published parameters only. No digitized measurements yet,\n"
                "so this is the transcribed fit, not a comparison against data."
            )
        else:
            subtitle = (
                f"{series.observations.count} points digitized from "
                f"{series.source.figure} of {series.source.doi}.\nDashed curves "
                f"are re-fitted from those points, {fitted.weighting} weighting."
            )
        figure.text(
            0.10, 0.955,
            f"KLS-1 pressure-sinkage, {model_id.title()} model",
            fontsize=12, color=INK_PRIMARY, weight="bold", ha="left", va="top",
        )
        figure.text(
            0.10, 0.900, subtitle,
            fontsize=8.5, color=INK_SECONDARY, ha="left", va="top",
        )
        figure.text(
            0.10, 0.065,
            "Plotted only within the fitted range: contact half-width "
            f"{published.contact_half_width_validity.min * 1e3:.1f}"
            f"–{published.contact_half_width_validity.max * 1e3:.1f} mm, "
            f"sinkage 0–{published.sinkage_validity.max * 1e3:.0f} mm.\n"
            "These plate-scale parameters do not transfer to foot-scale contact "
            "patches.",
            fontsize=7.5, color=INK_MUTED, ha="left", va="top",
        )
        return figure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soil", type=Path, default=DEFAULT_SOIL_PATH)
    parser.add_argument("--series", type=Path, default=DEFAULT_SERIES_PATH)
    parser.add_argument("--model", default="bekker")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args(argv)

    soil = load_soil(arguments.soil)
    dataset = next(iter(soil.datasets.values()))
    if arguments.model not in dataset.models:
        parser.error(
            f"{arguments.soil} has no verified model {arguments.model!r}; it holds "
            f"{sorted(dataset.models)}"
        )
    published = dataset.models[arguments.model]
    half_widths = [plate.contact_half_width_m for plate in dataset.apparatus.plates]

    series: PressureSinkageSeries | None = None
    fitted: FittedContactModel | None = None
    if arguments.series.is_file():
        try:
            series = load_pressure_sinkage_series(arguments.series)
        except SeriesFileError as error:
            print(f"cannot read the digitized series: {error}", file=sys.stderr)
            return 1
        fitted = fit_contact_model(arguments.model, series.observations)
    else:
        print(
            f"no digitized series at {arguments.series}; plotting published "
            "parameters only",
            file=sys.stderr,
        )

    figure = build_figure(arguments.model, published, half_widths, series, fitted)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output)
    plt.close(figure)

    print(f"soil     {soil.id} ({dataset.doi})")
    print(f"model    {arguments.model}, published fit_method {published.fit_method}")
    if series is not None and fitted is not None:
        print(
            f"series   {series.id}, {series.observations.count} points across "
            f"{fitted.plate_count} plates, weighting {fitted.weighting}"
        )
        print("\n".join(_compare(published, fitted)))
        print(
            "  coefficient of determination against the digitized points: "
            f"{coefficient_of_determination(fitted.model, series.observations):.4f}"
        )
    print(f"figure   {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
