# SPDX-License-Identifier: Apache-2.0
#
# validation/contact/plate_transfer.py — does a Bekker fit predict a plate it
# was not fitted at?
#
# This is validation, not calibration, and the distinction is the point. Every
# result before it fitted parameters to data and reported how well the fit
# matched that same data. This holds one plate out, fits the other two, and
# predicts the plate it never saw. That is the question the project actually
# needs answered, because a foot is a contact patch at a size no bevameter
# plate reaches.
#
# The answer is negative, and the negative result is the deliverable.
#
# Mechanism first: Bekker assumes one sinkage exponent for the soil, and the
# three plates disagree about it by 35 to 45 percent. That holds when all three
# are fitted over one common pressure window, so it is a property of the plates
# rather than of the analysis range. Each plate's own modulus therefore carries
# units of kN/m^(n+1) at its own n — three different dimensions — and the plate
# scaling regresses them against 1/b as if they were commensurable. Averaging
# the exponent is what makes that step possible at all.
#
# Read alongside the window effect in calibration/: both are the same category
# error. k_c and k_phi are meaningful only at fixed n, and n is not fixed.
#
# Holding one plate out of three leaves two, so the plate scaling is exactly
# determined — zero residual, no redundancy. Each test is a genuine prediction
# but confounds model form, extrapolation distance and fit noise, and there are
# only three of them. Resolving that needs generated data, not this campaign.
#
# References
#   Oravec HA (2009) Understanding Mechanical Behavior of Lunar Soils for the
#     Study of Vehicle Mobility. PhD dissertation, Case Western Reserve
#     University. Appendix D Code D5.

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
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from eclipse.fitting import (
    FittedContactModel,
    PressureSinkageObservations,
    fit_averaged_power_law,
    fit_contact_model,
)
from eclipse.io.channels import (
    BevameterChannels,
    ChannelsFileError,
    load_bevameter_channels,
)
from eclipse.resampling import ensemble

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_CHANNELS_PATH: Final = (
    REPOSITORY_ROOT / "data" / "literature" / "oravec2009-grc1-raw-channels.toml"
)
DEFAULT_FIGURE_PATH: Final = (
    Path(__file__).resolve().parent / "figures" / "grc1-plate-transfer.png"
)
DEFAULT_REPORT_PATH: Final = (
    Path(__file__).resolve().parent / "results" / "grc1-plate-transfer.toml"
)

MODEL_ID: Final = "bekker"
WEIGHTING: Final = "pressure_squared"
ESTIMATOR: Final = "averaged_exponent"
PLATE_NAMES: Final = ("small", "medium", "large")
REPEATS: Final = "12345"
REPORT_SCHEMA_VERSION: Final = 1
FIGURE_WINDOW: Final = "figure-5.52"
CURVE_SAMPLES: Final = 300

PLATE_MARKERS: Final = ("o", "^", "s")
OBSERVED_COLOR: Final = "#52514e"
PREDICTED_COLOR: Final = "#d4570a"
INK_PRIMARY: Final = "#0b0b0b"
INK_SECONDARY: Final = "#52514e"
INK_MUTED: Final = "#8a8880"
SURFACE: Final = "#fcfcfb"
MARKER: Final = (3.4, 0.8)

FIGURE_STYLE: Final[dict[str, Any]] = {
    "figure.figsize": (10.6, 5.0),
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
    "figure.subplot.top": 0.718,
    "figure.subplot.bottom": 0.250,
    "figure.subplot.left": 0.058,
    "figure.subplot.right": 0.986,
    "figure.subplot.wspace": 0.200,
}


@dataclass(frozen=True, slots=True)
class PlateCurve:
    name: str
    contact_half_width_m: float
    pressure_kPa: NDArray[np.float64]
    sinkage_m: NDArray[np.float64]

    @property
    def reciprocal_half_width(self) -> float:
        return 1.0 / self.contact_half_width_m


@dataclass(frozen=True, slots=True)
class HeldOut:
    plate: str
    kind: str
    distance: float
    fit: FittedContactModel
    predicted_kPa: NDArray[np.float64]
    observed_kPa: NDArray[np.float64]

    @property
    def relative_error(self) -> NDArray[np.float64]:
        return np.asarray((self.predicted_kPa - self.observed_kPa) / self.observed_kPa)

    @property
    def mean_absolute_relative_error(self) -> float:
        return float(np.mean(np.abs(self.relative_error)))

    @property
    def maximum_absolute_relative_error(self) -> float:
        return float(np.max(np.abs(self.relative_error)))

    @property
    def minimum_invertible_half_width_m(self) -> float:
        return float(self.fit.model.invertible_half_width_range().minimum)

    @property
    def maximum_invertible_half_width_m(self) -> float:
        return float(self.fit.model.invertible_half_width_range().maximum)


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


def plate_curves(
    channels: BevameterChannels, block: dict[str, Any]
) -> dict[str, PlateCurve]:
    curves: dict[str, PlateCurve] = {}
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
        grid = curve.positions[dropped:]
        mean = curve.mean_values[dropped:]
        usable = (grid > 0.0) & (mean > 0.0)
        curves[plate] = PlateCurve(
            name=plate,
            contact_half_width_m=channels.test(f"{plate}-1").plate.contact_half_width_m,
            pressure_kPa=grid[usable],
            sinkage_m=mean[usable],
        )
    return curves


def _observations(curves: dict[str, PlateCurve], plates: list[str]) -> PressureSinkageObservations:
    return PressureSinkageObservations(
        contact_half_width_m=np.concatenate(
            [np.full(curves[p].pressure_kPa.size, curves[p].contact_half_width_m) for p in plates]
        ),
        sinkage_m=np.concatenate([curves[p].sinkage_m for p in plates]),
        pressure_kPa=np.concatenate([curves[p].pressure_kPa for p in plates]),
    )


def hold_out(curves: dict[str, PlateCurve], plate: str) -> HeldOut:
    kept = [name for name in PLATE_NAMES if name != plate]
    fit = fit_contact_model(
        MODEL_ID, _observations(curves, kept), weighting=WEIGHTING, estimator=ESTIMATOR
    )
    target = curves[plate]
    low, high = sorted(curves[name].reciprocal_half_width for name in kept)
    position = target.reciprocal_half_width
    inside = low < position < high
    span = high - low
    distance = (
        0.0
        if inside
        else (position - high if position > high else low - position) / span
    )
    predicted = np.asarray(
        fit.model.pressure(
            sinkage=target.sinkage_m, contact_half_width=target.contact_half_width_m
        )
    )
    return HeldOut(
        plate=plate,
        kind="interpolation" if inside else "extrapolation",
        distance=distance,
        fit=fit,
        predicted_kPa=predicted,
        observed_kPa=target.pressure_kPa,
    )


def per_plate_exponents(
    curves: dict[str, PlateCurve], top_kPa: float
) -> dict[str, float]:
    exponents: dict[str, float] = {}
    for plate, curve in curves.items():
        inside = curve.pressure_kPa <= top_kPa
        one = PressureSinkageObservations(
            contact_half_width_m=np.full(
                int(inside.sum()), curve.contact_half_width_m
            ),
            sinkage_m=curve.sinkage_m[inside],
            pressure_kPa=curve.pressure_kPa[inside],
        )
        exponents[plate] = fit_averaged_power_law(
            one, weighting=WEIGHTING
        ).sinkage_exponent
    return exponents


def build_figure(
    curves: dict[str, PlateCurve],
    results: list[HeldOut],
    exponents: dict[str, float],
    common_top_kPa: float,
    report_path: Path,
) -> Figure:
    with plt.rc_context(cast(Any, FIGURE_STYLE)):
        figure, axes = plt.subplots(1, len(results), squeeze=False)
        for column, result in enumerate(results):
            panel = axes[0][column]
            target = curves[result.plate]
            index = PLATE_NAMES.index(result.plate)
            panel.plot(
                target.pressure_kPa,
                target.sinkage_m * 1e3,
                marker=PLATE_MARKERS[index],
                markersize=MARKER[0],
                markerfacecolor="none",
                markeredgecolor=OBSERVED_COLOR,
                markeredgewidth=MARKER[1],
                linestyle="none",
                zorder=4,
            )
            order = np.argsort(result.predicted_kPa)
            panel.plot(
                result.predicted_kPa[order],
                target.sinkage_m[order] * 1e3,
                color=PREDICTED_COLOR,
                linewidth=1.8,
                zorder=5,
            )
            deepest = float(target.sinkage_m.max() * 1e3)
            panel.set_xlim(0.0, float(max(target.pressure_kPa.max(), result.predicted_kPa.max())) * 1.04)
            panel.set_ylim(deepest * 1.08, 0.0)
            panel.set_xlabel("pressure  (kPa)")
            if column == 0:
                panel.set_ylabel("sinkage  (mm)")
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)
            panel.set_axisbelow(True)
            qualifier = (
                "interpolated"
                if result.kind == "interpolation"
                else f"extrapolated {result.distance:.2f}x span"
            )
            panel.set_title(
                f"{result.plate} plate held out, b = "
                f"{target.contact_half_width_m * 1e3:.0f} mm\n{qualifier}  ·  "
                f"mean error {result.mean_absolute_relative_error * 100:.0f}%",
                color=INK_PRIMARY,
                loc="left",
                linespacing=1.6,
            )

        axes[0][0].legend(
            handles=[
                Line2D([], [], color=OBSERVED_COLOR, marker="o", markersize=MARKER[0],
                       markerfacecolor="none", markeredgewidth=MARKER[1],
                       linestyle="none"),
                Line2D([], [], color=PREDICTED_COLOR, linewidth=1.8),
            ],
            labels=["measured", "predicted from the other two plates"],
            loc="lower left",
        )

        spread = (max(exponents.values()) - min(exponents.values())) / float(
            np.mean(list(exponents.values()))
        )
        figure.text(
            0.058, 0.968,
            "GRC-1 plate transfer: can a Bekker fit predict a plate it was not fitted at?",
            fontsize=12, color=INK_PRIMARY, weight="bold", ha="left", va="top",
        )
        figure.text(
            0.058, 0.924,
            "Each panel holds one plate out, fits the other two, and predicts the "
            "one it never saw. Exact sensor readings, no digitization error.\n"
            "The mechanism: over one common window to "
            f"{common_top_kPa:.0f} kPa the three plates disagree about the sinkage "
            f"exponent by {spread * 100:.0f}% "
            f"({' / '.join(f'{exponents[p]:.2f}' for p in PLATE_NAMES)}),\n"
            "so each plate's modulus carries a different unit and the three are "
            "not commensurable.",
            fontsize=8.5, color=INK_SECONDARY, ha="left", va="top",
        )
        figure.text(
            0.058, 0.135,
            "Bekker assumes one exponent for the soil. Where the plates disagree "
            "about it, their moduli are not commensurable and the plate scaling "
            "regresses quantities of different dimension.\n"
            "Holding one plate out of three leaves two, so the scaling is exactly "
            "determined: each test is a genuine prediction but confounds model "
            "form, extrapolation distance and fit noise.\n"
            f"Numbers behind this figure: {_display_path(report_path)}",
            fontsize=7.5, color=INK_MUTED, ha="left", va="top",
        )
        return figure


def build_report(
    channels: BevameterChannels,
    blocks: list[dict[str, Any]],
    windows: dict[str, list[HeldOut]],
    exponents: dict[str, dict[str, float]],
    common_tops: dict[str, float],
    references: dict[str, FittedContactModel],
) -> str:
    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# Generated by validation/contact/plate_transfer.py. Do not edit.",
        "#",
        "# Validation, not calibration: every fit here is scored against a plate it",
        "# was not fitted at. No timestamp, so re-running against unchanged inputs",
        "# leaves this byte-identical and any diff means a result moved.",
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
        "",
        "[environment]",
        f'python = "{platform.python_version()}"',
        f'numpy = "{np.__version__}"',
        "",
        "# Bekker assumes one sinkage exponent for the soil. Fitted over one common",
        "# pressure window, so that the analysis range cannot explain it, the three",
        "# plates disagree. Each plate's modulus therefore carries units of",
        "# kN/m^(n+1) at its own n, and the plate scaling regresses quantities of",
        "# different dimension against 1/b.",
        "",
    ]
    for block in blocks:
        identifier = block["id"]
        own = exponents[identifier]
        values = list(own.values())
        lines += [
            "[[exponent_disagreement]]",
            f'window = "{identifier}"',
            f"common_window_max_kPa = {_format_float(common_tops[identifier])}",
            *(f"{plate} = {_format_float(own[plate])}" for plate in PLATE_NAMES),
            "relative_spread = "
            + _format_float((max(values) - min(values)) / float(np.mean(values))),
            "",
        ]

    for block in blocks:
        identifier = block["id"]
        reference = references[identifier]
        lines += [
            "[[window]]",
            f'id = "{identifier}"',
            f'window = "{block["window"]}"',
            "# the fit using all three plates, for reference only",
            "all_plates = { "
            + ", ".join(
                f"{name} = {_format_float(value)}"
                for name, value in reference.parameters.items()
            )
            + " }",
            "",
        ]
        for result in windows[identifier]:
            lines += [
                "[[held_out]]",
                f'window = "{identifier}"',
                f'plate = "{result.plate}"',
                f'kind = "{result.kind}"',
                f"extrapolation_distance_in_spans = {_format_float(result.distance)}",
                "fitted = { "
                + ", ".join(
                    f"{name} = {_format_float(value)}"
                    for name, value in result.fit.parameters.items()
                )
                + " }",
                "minimum_invertible_half_width_m = "
                + _format_float(result.minimum_invertible_half_width_m),
                "maximum_invertible_half_width_m = "
                + _format_float(result.maximum_invertible_half_width_m),
                "mean_absolute_relative_error = "
                + _format_float(result.mean_absolute_relative_error),
                "maximum_absolute_relative_error = "
                + _format_float(result.maximum_absolute_relative_error),
                "",
            ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Predict a held-out plate from a Bekker fit to the others."
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

    blocks = tomllib.loads(arguments.channels.read_text(encoding="utf-8")).get(
        "verification", []
    )
    if not blocks:
        print(f"{arguments.channels}: no windows to validate over", file=sys.stderr)
        return 1

    windows: dict[str, list[HeldOut]] = {}
    exponents: dict[str, dict[str, float]] = {}
    common_tops: dict[str, float] = {}
    references: dict[str, FittedContactModel] = {}
    curves_by_window: dict[str, dict[str, PlateCurve]] = {}

    for block in blocks:
        curves = plate_curves(channels, block)
        curves_by_window[block["id"]] = curves
        common_tops[block["id"]] = float(min(block["resampling_endpoints_kPa"]))
        exponents[block["id"]] = per_plate_exponents(curves, common_tops[block["id"]])
        references[block["id"]] = fit_contact_model(
            MODEL_ID,
            _observations(curves, list(PLATE_NAMES)),
            weighting=WEIGHTING,
            estimator=ESTIMATOR,
        )
        windows[block["id"]] = [hold_out(curves, plate) for plate in PLATE_NAMES]

    for block in blocks:
        identifier = block["id"]
        own = exponents[identifier]
        values = list(own.values())
        print(f"\n=== {identifier} ===")
        print(
            f"  per-plate exponent over a common window to "
            f"{common_tops[identifier]:.0f} kPa: "
            + ", ".join(f"{p} {own[p]:.4f}" for p in PLATE_NAMES)
            + f"  ({(max(values) - min(values)) / float(np.mean(values)) * 100:.0f}% spread)"
        )
        print(
            f"  {'held out':>9s} {'kind':>14s} {'distance':>9s} {'k_c':>10s} "
            f"{'k_phi':>11s} {'valid b mm':>16s} {'mean err':>9s} {'max err':>9s}"
        )
        for result in windows[identifier]:
            low = result.minimum_invertible_half_width_m * 1e3
            high = result.maximum_invertible_half_width_m * 1e3
            valid = (
                f"{low:.0f} to {high:.0f}" if np.isfinite(high) else f"above {low:.0f}"
            )
            print(
                f"  {result.plate:>9s} {result.kind:>14s} {result.distance:8.2f}x "
                f"{result.fit.parameters['cohesive_modulus']:10.1f} "
                f"{result.fit.parameters['frictional_modulus']:11.1f} "
                f"{valid:>16s} "
                f"{result.mean_absolute_relative_error * 100:8.1f}% "
                f"{result.maximum_absolute_relative_error * 100:8.1f}%"
            )

    arguments.figure.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure(
        curves_by_window[FIGURE_WINDOW],
        windows[FIGURE_WINDOW],
        exponents[FIGURE_WINDOW],
        common_tops[FIGURE_WINDOW],
        arguments.report,
    )
    figure.savefig(arguments.figure, dpi=FIGURE_STYLE["figure.dpi"])
    plt.close(figure)
    arguments.report.write_text(
        build_report(channels, blocks, windows, exponents, common_tops, references),
        encoding="utf-8",
    )
    print(f"\nfigure   {arguments.figure}")
    print(f"report   {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
