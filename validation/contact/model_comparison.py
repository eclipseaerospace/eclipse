# SPDX-License-Identifier: Apache-2.0
#
# validation/contact/model_comparison.py — does a richer contact model earn its
# extra parameter?
#
# Three candidates. Bekker and Reece are the incumbents at three parameters
# each. The third replaces Bekker's single exponent with a regressed one,
# n(b) = n0 + beta*ln(b/b0), which is the smallest step that lets the exponent
# depend on plate size and so the most favourable version of "fit the scale
# dependence the data appears to show".
#
# It is not adopted, on three grounds, and the second is the one that generalises.
#
# First, it costs accuracy. On data generated from a process that genuinely has
# one exponent, the scale-aware model predicts a held-out plate worse than both
# incumbents at every configuration tested — 2.6 times worse at GRC-1's own, and
# still 1.5 times worse at six plates and twenty repeats. It never catches up.
#
# Second, it amplifies the dimensional error it was meant to address. Each
# plate's modulus is now evaluated at that plate's own exponent, so the moduli
# carry units of kN/m^(n+1) at different n, and the plate scaling regresses
# quantities of different dimension against 1/b. Fitted on GRC-1 this returns
# k_c = 12300 against Bekker's 4096. The same objection applies to any
# dimensional-analysis form that reintroduces a size-dependent exponent, which
# is worth stating because Lim et al. (2021) is exactly such a model and its
# published parameters are marked not_reproducible in this repository's own soil
# file: its plate scaling is negative across the entire tested range.
#
# Third, the evidence for scale dependence is marginal once the null is
# specified correctly, and how it is specified decides the verdict. Fitting beta
# to data with no scale dependence at all, under a null using a uniform noise
# level and one common pressure window, puts GRC-1's beta at p = 0.007. Under a
# null matched to GRC-1's own per-plate spread and per-plate windows, the same
# beta sits at p = 0.040 — a sixfold difference from the null's specification
# alone, on identical observed data. Both are reported because that gap is the
# lesson: the null, not the data, decided that verdict.
#
# What p = 0.040 means is worth stating plainly, because a reader will want it:
# real scale dependence is not excluded. One campaign at this noise level cannot
# establish it. The campaign specified in campaign_design.py could — twenty
# repeats shrinks the null's spread enough to make the test decisive — which is
# what ties that specification to this question rather than leaving them as two
# separate results.
#
# References
#   Lim Y, Le VD, Bahati PA (2021) Development of a New Pressure-Sinkage Model
#     for Rover Wheel-Lunar Soil Interaction. J. Astron. Space Sci. 38(4).
#   Oravec HA (2009) Understanding Mechanical Behavior of Lunar Soils for the
#     Study of Vehicle Mobility. PhD dissertation, Case Western Reserve
#     University. Appendix D Code D5.

from __future__ import annotations

import argparse
import platform
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
    PressureSinkageObservations,
    fit_averaged_power_law,
    fit_contact_model,
)
from eclipse.io.channels import load_bevameter_channels
from eclipse.resampling import ensemble
from eclipse.terramechanics import BekkerModel, DegenerateContactModelError

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_CHANNELS_PATH: Final = (
    REPOSITORY_ROOT / "data" / "literature" / "oravec2009-grc1-raw-channels.toml"
)
DEFAULT_FIGURE_PATH: Final = (
    Path(__file__).resolve().parent / "figures" / "model-comparison.png"
)
DEFAULT_REPORT_PATH: Final = (
    Path(__file__).resolve().parent / "results" / "model-comparison.toml"
)

TRUE_MODEL: Final = BekkerModel(
    cohesive_modulus=4096.3537,
    frictional_modulus=-22284.5786,
    sinkage_exponent=1.232,
)
PLATE_ORDER: Final = ("small", "medium", "large")
REPEATS_LABEL: Final = "12345"
WINDOW_ID: Final = "figure-5.52"

# GRC-1 as measured: per-plate spread, per-plate windows.
MATCHED_SIGMA: Final[dict[str, float]] = {
    "small": 0.249, "medium": 0.230, "large": 0.181
}
MATCHED_LAG_ONE: Final[dict[str, float]] = {
    "small": 0.995, "medium": 0.960, "large": 0.904
}
# A designed campaign instead: one noise level, one window for every plate.
DESIGNED_SIGMA: Final = 0.22
DESIGNED_LAG_ONE: Final = 0.95
DESIGNED_TOP_KPA: Final = 50.0
STEP_KPA: Final = 0.5
SPAN_M: Final = (0.038, 0.095)

NULL_REALISATIONS: Final = 600
HOLDOUT_REALISATIONS: Final = 250
NULL_REPEAT_COUNTS: Final = (5, 20)
HOLDOUT_CASES: Final = ((5, 3), (20, 3), (20, 6))
SEED: Final = 20260813
REPORT_SCHEMA_VERSION: Final = 1
MODELS: Final = ("bekker", "reece", "scale_aware")
MODEL_LABELS: Final[dict[str, str]] = {
    "bekker": "Bekker",
    "reece": "Reece",
    "scale_aware": "scale-aware n(b)",
}

INK_PRIMARY: Final = "#0b0b0b"
INK_SECONDARY: Final = "#52514e"
INK_MUTED: Final = "#8a8880"
SURFACE: Final = "#fcfcfb"
MATCHED_COLOR: Final = "#1f4e9c"
DESIGNED_COLOR: Final = "#8a8880"
OBSERVED_COLOR: Final = "#d4570a"
MODEL_COLORS: Final[dict[str, str]] = {
    "bekker": "#1f4e9c", "reece": "#4d8fd6", "scale_aware": "#d4570a"
}

FIGURE_STYLE: Final[dict[str, Any]] = {
    "figure.figsize": (10.4, 5.2),
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
    "figure.subplot.top": 0.726,
    "figure.subplot.bottom": 0.232,
    "figure.subplot.left": 0.062,
    "figure.subplot.right": 0.986,
    "figure.subplot.wspace": 0.215,
}


@dataclass(frozen=True, slots=True)
class ScaleAwareFit:
    reference_half_width_m: float
    exponent_at_reference: float
    exponent_slope: float
    cohesive_modulus: float
    frictional_modulus: float

    def pressure(
        self, sinkage: NDArray[np.float64], contact_half_width: float
    ) -> NDArray[np.float64]:
        modulus = self.cohesive_modulus / contact_half_width + self.frictional_modulus
        exponent = self.exponent_at_reference + self.exponent_slope * np.log(
            contact_half_width / self.reference_half_width_m
        )
        return np.asarray(modulus * sinkage**exponent)


def _format_float(value: float) -> str:
    return "nan" if not np.isfinite(value) else repr(float(value))


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def per_plate_exponents_and_moduli(
    observations: PressureSinkageObservations,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    widths = observations.contact_half_widths
    exponents = np.array(
        [
            fit_averaged_power_law(
                observations.for_plate(float(width)), weighting="pressure_squared"
            ).sinkage_exponent
            for width in widths
        ]
    )
    return widths, exponents


def fit_scale_aware(
    observations: PressureSinkageObservations,
    reference_half_width_m: float | None = None,
) -> ScaleAwareFit:
    widths, exponents = per_plate_exponents_and_moduli(observations)
    reference = (
        float(np.exp(np.mean(np.log(widths))))
        if reference_half_width_m is None
        else reference_half_width_m
    )
    scaled = np.log(widths / reference)
    slope, intercept = np.linalg.lstsq(
        np.column_stack([scaled, np.ones_like(scaled)]), exponents, rcond=None
    )[0]
    # Each plate's modulus is evaluated at that plate's own exponent, which is
    # where the dimensional inconsistency enters: the moduli below carry units
    # of kN/m^(n+1) at different n, and are then regressed against 1/b.
    moduli = []
    for width in widths:
        plate = observations.for_plate(float(width))
        exponent = intercept + slope * np.log(width / reference)
        weights = plate.pressure_kPa**2
        residual = np.log(plate.pressure_kPa) - exponent * np.log(plate.sinkage_m)
        moduli.append(float(np.exp(np.sum(weights * residual) / np.sum(weights))))
    inverse = 1.0 / widths
    cohesive, frictional = np.linalg.lstsq(
        np.column_stack([inverse, np.ones_like(inverse)]),
        np.array(moduli),
        rcond=None,
    )[0]
    return ScaleAwareFit(
        reference, float(intercept), float(slope), float(cohesive), float(frictional)
    )


def matched_pressure_grid(plate: str, block: dict[str, Any]) -> NDArray[np.float64]:
    index = PLATE_ORDER.index(plate)
    top = block["resampling_endpoints_kPa"][index]
    dropped = block["leading_samples_dropped"][index]
    step = block["resampling_step_kPa"]
    return np.asarray(np.arange(0.0, top + step / 2.0, step)[dropped:])


def cholesky_for(grid: NDArray[np.float64], lag_one: float) -> NDArray[np.float64]:
    gap = grid[:, None] - grid[None, :]
    length = STEP_KPA / float(np.sqrt(-2.0 * np.log(lag_one)))
    return np.asarray(
        np.linalg.cholesky(
            np.exp(-0.5 * (gap / length) ** 2) + 1e-10 * np.eye(grid.size)
        )
    )


def ensemble_from_truth(
    grid: NDArray[np.float64],
    half_width: float,
    repeats: int,
    sigma: float,
    factor: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64] | None:
    base = np.asarray(
        TRUE_MODEL.sinkage(pressure=grid, contact_half_width=half_width)
    )
    deviation = (factor @ rng.standard_normal((grid.size, repeats))).T * sigma
    mean = np.mean(base[None, :] * (1.0 + deviation), axis=0)
    return None if (mean <= 0.0).any() else np.asarray(mean)


def observations_from(
    grids: list[NDArray[np.float64]],
    widths: list[float],
    curves: list[NDArray[np.float64]],
) -> PressureSinkageObservations:
    return PressureSinkageObservations(
        contact_half_width_m=np.concatenate(
            [np.full(grid.size, width) for grid, width in zip(grids, widths)]
        ),
        sinkage_m=np.concatenate(curves),
        pressure_kPa=np.concatenate(grids),
    )


def grc1_observations(
    channels: Any, block: dict[str, Any]
) -> PressureSinkageObservations:
    grids, widths, curves = [], [], []
    for plate in PLATE_ORDER:
        index = PLATE_ORDER.index(plate)
        top = block["resampling_endpoints_kPa"][index]
        dropped = block["leading_samples_dropped"][index]
        step = block["resampling_step_kPa"]
        curve = ensemble(
            sample_positions=[
                channels.pressure_kPa(f"{plate}-{r}") for r in REPEATS_LABEL
            ],
            sample_values=[
                channels.sinkage_m(f"{plate}-{r}") for r in REPEATS_LABEL
            ],
            positions=np.arange(0.0, top + step / 2.0, step),
        )
        grid = curve.positions[dropped:]
        mean = curve.mean_values[dropped:]
        usable = (grid > 0.0) & (mean > 0.0)
        grids.append(np.asarray(grid[usable]))
        widths.append(channels.test(f"{plate}-1").plate.contact_half_width_m)
        curves.append(np.asarray(mean[usable]))
    return observations_from(grids, widths, curves)


def matched_null(
    block: dict[str, Any], repeats: int, reference: float, rng: np.random.Generator
) -> list[float]:
    grids = [matched_pressure_grid(plate, block) for plate in PLATE_ORDER]
    widths = [
        {"small": 0.038, "medium": 0.051, "large": 0.095}[plate]
        for plate in PLATE_ORDER
    ]
    factors = [
        cholesky_for(grid, MATCHED_LAG_ONE[plate])
        for grid, plate in zip(grids, PLATE_ORDER)
    ]
    slopes: list[float] = []
    for _ in range(NULL_REALISATIONS):
        curves = [
            ensemble_from_truth(
                grid, width, repeats, MATCHED_SIGMA[plate], factor, rng
            )
            for grid, width, plate, factor in zip(grids, widths, PLATE_ORDER, factors)
        ]
        if any(curve is None for curve in curves):
            continue
        try:
            slopes.append(
                fit_scale_aware(
                    observations_from(grids, widths, cast(Any, curves)), reference
                ).exponent_slope
            )
        except (ValueError, DegenerateContactModelError):
            continue
    return slopes


def designed_grid() -> NDArray[np.float64]:
    return np.asarray(np.arange(STEP_KPA, DESIGNED_TOP_KPA + STEP_KPA / 2.0, STEP_KPA))


def designed_widths(count: int) -> list[float]:
    low, high = SPAN_M
    return [float(v) for v in 1.0 / np.linspace(1.0 / low, 1.0 / high, count)]


def designed_null(
    repeats: int, count: int, reference: float, rng: np.random.Generator
) -> list[float]:
    grid = designed_grid()
    factor = cholesky_for(grid, DESIGNED_LAG_ONE)
    widths = designed_widths(count)
    slopes: list[float] = []
    for _ in range(NULL_REALISATIONS):
        curves = [
            ensemble_from_truth(grid, width, repeats, DESIGNED_SIGMA, factor, rng)
            for width in widths
        ]
        if any(curve is None for curve in curves):
            continue
        try:
            slopes.append(
                fit_scale_aware(
                    observations_from([grid] * count, widths, cast(Any, curves)),
                    reference,
                ).exponent_slope
            )
        except (ValueError, DegenerateContactModelError):
            continue
    return slopes


def holdout_errors(
    repeats: int, count: int, reference: float, rng: np.random.Generator
) -> dict[str, list[float]]:
    grid = designed_grid()
    factor = cholesky_for(grid, DESIGNED_LAG_ONE)
    widths = designed_widths(count)
    collected: dict[str, list[float]] = {name: [] for name in MODELS}
    for _ in range(HOLDOUT_REALISATIONS):
        curves = [
            ensemble_from_truth(grid, width, repeats, DESIGNED_SIGMA, factor, rng)
            for width in widths
        ]
        if any(curve is None for curve in curves):
            continue
        for held in range(1, count - 1):
            keep = [index for index in range(count) if index != held]
            subset = observations_from(
                [grid] * len(keep),
                [widths[index] for index in keep],
                [cast(Any, curves)[index] for index in keep],
            )
            target = np.asarray(
                TRUE_MODEL.sinkage(pressure=grid, contact_half_width=widths[held])
            )
            try:
                for name in ("bekker", "reece"):
                    fit = fit_contact_model(
                        name, subset, weighting="pressure_squared",
                        estimator="averaged_exponent",
                    )
                    predicted = fit.model.pressure(
                        sinkage=target, contact_half_width=widths[held]
                    )
                    collected[name].append(
                        float(np.mean(np.abs((predicted - grid) / grid)))
                    )
                predicted = fit_scale_aware(subset, reference).pressure(
                    target, widths[held]
                )
                collected["scale_aware"].append(
                    float(np.mean(np.abs((predicted - grid) / grid)))
                )
            except (ValueError, DegenerateContactModelError):
                continue
    return collected


def build_figure(
    observed_slope: float,
    nulls: dict[tuple[str, int], list[float]],
    holdout: dict[tuple[int, int], dict[str, list[float]]],
    report_path: Path,
) -> Figure:
    with plt.rc_context(cast(Any, FIGURE_STYLE)):
        figure, axes = plt.subplots(1, 2, squeeze=False)

        left = axes[0][0]
        edges = np.linspace(-0.75, 0.75, 46)
        for (kind, repeats), values in nulls.items():
            if repeats != NULL_REPEAT_COUNTS[0]:
                continue
            left.hist(
                values, bins=edges, histtype="step", density=True, linewidth=1.6,
                color=MATCHED_COLOR if kind == "matched" else DESIGNED_COLOR,
            )
        left.axvline(
            observed_slope, color=OBSERVED_COLOR, linewidth=1.8, linestyle=(0, (5, 2))
        )
        left.set_xlabel("exponent slope $\\beta$ fitted to data with none")
        left.set_ylabel("density")
        left.set_title(
            f"two null specifications, {NULL_REPEAT_COUNTS[0]} repeats",
            color=INK_SECONDARY, loc="left",
        )
        left.legend(
            handles=[
                Line2D([], [], color=MATCHED_COLOR, linewidth=1.6),
                Line2D([], [], color=DESIGNED_COLOR, linewidth=1.6),
                Line2D([], [], color=OBSERVED_COLOR, linewidth=1.8,
                       linestyle=(0, (5, 2))),
            ],
            labels=[
                "null matched to GRC-1",
                "null with uniform noise and window",
                f"GRC-1 observed, $\\beta$ = {observed_slope:.3f}",
            ],
            loc="upper left",
        )

        right = axes[0][1]
        positions = np.arange(len(HOLDOUT_CASES), dtype=float)
        width = 0.26
        for order, name in enumerate(MODELS):
            values = [
                float(np.median(holdout[case][name])) * 100.0 for case in HOLDOUT_CASES
            ]
            right.bar(
                positions + (order - 1) * width, values, width * 0.92,
                color=MODEL_COLORS[name], label=MODEL_LABELS[name], zorder=3,
            )
        right.set_xticks(positions)
        right.set_xticklabels(
            [f"{repeats} repeats\n{plates} plates" for repeats, plates in HOLDOUT_CASES]
        )
        right.set_ylabel("error at a held-out plate")
        right.set_yticks([0, 3, 6, 9, 12])
        right.set_yticklabels(["0%", "3%", "6%", "9%", "12%"])
        right.set_title(
            "held-out error, single-exponent truth",
            color=INK_SECONDARY, loc="left",
        )
        right.legend(loc="upper right")

        for panel in (left, right):
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)
            panel.set_axisbelow(True)

        figure.text(
            0.062, 0.968,
            "A fourth parameter does not earn its place",
            fontsize=12, color=INK_PRIMARY, weight="bold", ha="left", va="top",
        )
        figure.text(
            0.062, 0.918,
            "Letting the sinkage exponent vary with plate size, n(b) = n0 + "
            "beta*ln(b/b0), is the smallest step that fits the scale dependence\n"
            "GRC-1 appears to show. It predicts worse than both incumbents on "
            "data whose truth genuinely has one exponent, and the evidence that\n"
            "GRC-1 has any scale dependence is marginal once the null is matched "
            "to GRC-1's own per-plate noise and windows.",
            fontsize=8.5, color=INK_SECONDARY, ha="left", va="top",
        )
        figure.text(
            0.062, 0.148,
            "Marginal does not mean absent: real scale dependence is not "
            "excluded. It means one campaign at this noise cannot establish it,\n"
            "and the campaign specified in campaign_design.py could.\n"
            "Each plate's modulus is evaluated at that plate's own exponent, so "
            "the moduli carry different units and the plate scaling sums\n"
            "quantities of different dimension — fitted on GRC-1 that returns "
            "k_c = 12300 against Bekker's 4096.\n"
            f"Numbers behind this figure: {_display_path(report_path)}",
            fontsize=7.5, color=INK_MUTED, ha="left", va="top",
        )
        return figure


def build_report(
    observed: ScaleAwareFit,
    exponents: NDArray[np.float64],
    condition: float,
    bekker_reference: dict[str, float],
    nulls: dict[tuple[str, int], list[float]],
    holdout: dict[tuple[int, int], dict[str, list[float]]],
) -> str:
    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# Generated by validation/contact/model_comparison.py. Do not edit.",
        "#",
        "# No timestamp: re-running against an unchanged seed leaves this",
        "# byte-identical.",
        "",
        f"schema_version = {REPORT_SCHEMA_VERSION}",
        f"seed = {SEED}",
        'verdict = "do not adopt the scale-aware exponent"',
        "",
        "[environment]",
        f'python = "{platform.python_version()}"',
        f'numpy = "{np.__version__}"',
        "",
        "# The full fit is over-determined at three plates: three per-plate",
        "# exponents constrain two parameters and three moduli constrain two",
        "# more. It is leave-one-out that degenerates, where two plates leave the",
        "# exponent slope fitted to exactly two points with zero residual.",
        "[identifiability]",
        f"plates = {exponents.size}",
        "per_plate_exponents = ["
        + ", ".join(_format_float(v) for v in exponents)
        + "]",
        f"exponent_regression_condition_number = {_format_float(condition)}",
        "residual_degrees_of_freedom_full_fit = "
        f"{exponents.size - 2}",
        "residual_degrees_of_freedom_leave_one_out = 0",
        "",
        "# Each plate's modulus is evaluated at that plate's own exponent, so the",
        "# moduli carry units of kN/m^(n+1) at different n and the plate scaling",
        "# sums quantities of different dimension. The comparison below is the",
        "# measured consequence.",
        "[dimensional_consequence]",
        f"scale_aware_cohesive_modulus = {_format_float(observed.cohesive_modulus)}",
        f"scale_aware_frictional_modulus = {_format_float(observed.frictional_modulus)}",
        f"bekker_cohesive_modulus = {_format_float(bekker_reference['cohesive_modulus'])}",
        f"bekker_frictional_modulus = {_format_float(bekker_reference['frictional_modulus'])}",
        "",
        "[grc1_scale_aware_fit]",
        f"reference_half_width_m = {_format_float(observed.reference_half_width_m)}",
        f"exponent_at_reference = {_format_float(observed.exponent_at_reference)}",
        f"exponent_slope = {_format_float(observed.exponent_slope)}",
        "",
        "# Two nulls, both generating from a process with NO scale dependence.",
        "# The matched one uses GRC-1's per-plate spread and per-plate windows;",
        "# the designed one uses a single noise level and one common window. They",
        "# disagree about whether GRC-1's slope is remarkable, and the gap is the",
        "# point: a null's specification, not the data, decided that verdict.",
        "",
    ]
    for (kind, repeats), values in sorted(nulls.items()):
        array = np.array(values)
        exceed = float((np.abs(array) >= abs(observed.exponent_slope)).mean())
        lines += [
            "[[null]]",
            f'specification = "{kind}"',
            f"repeats = {repeats}",
            f"realisations = {array.size}",
            "percentiles = { "
            + ", ".join(
                f"p{int(q)} = {_format_float(float(np.percentile(array, q)))}"
                for q in (5, 50, 95, 99)
            )
            + " }",
            f"probability_at_least_as_extreme = {_format_float(exceed)}",
            "",
        ]

    lines += [
        "# Held-out prediction error on data generated from a single-exponent",
        "# process. Lower is better, and the extra parameter never pays for",
        "# itself.",
        "",
    ]
    for (repeats, plates), models in sorted(holdout.items()):
        lines += ["[[holdout]]", f"repeats = {repeats}", f"plates = {plates}"]
        for name in MODELS:
            lines.append(
                f"{name} = {_format_float(float(np.median(models[name])))}"
            )
        lines.append("")

    lines += [
        "# A marginal probability is not an absence. Real scale dependence is not",
        "# excluded; one campaign at this noise cannot establish it, and the",
        "# configuration in campaign_design.py could.",
        "",
        "# Lim et al. (2021) is a Buckingham-Pi pressure-sinkage model with a",
        "# size-dependent scaling, and its published parameters are recorded as",
        "# not_reproducible in data/soils/kls1.toml: the plate scaling is",
        "# negative across the entire tested range of half-widths. That is direct",
        "# evidence about this class of approach and belongs with this result.",
        "[related_evidence]",
        'lim2021_status = "not_reproducible"',
        'lim2021_soil_file = "data/soils/kls1.toml"',
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare Bekker, Reece and a scale-aware exponent."
    )
    parser.add_argument("--channels", type=Path, default=DEFAULT_CHANNELS_PATH)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    arguments = parser.parse_args(argv)

    channels = load_bevameter_channels(arguments.channels)
    block = next(
        entry
        for entry in tomllib.loads(arguments.channels.read_text(encoding="utf-8"))[
            "verification"
        ]
        if entry["id"] == WINDOW_ID
    )
    observations = grc1_observations(channels, block)
    widths, exponents = per_plate_exponents_and_moduli(observations)
    observed = fit_scale_aware(observations)
    scaled = np.log(widths / observed.reference_half_width_m)
    condition = float(
        np.linalg.cond(np.column_stack([scaled, np.ones_like(scaled)]))
    )
    incumbent = fit_contact_model(
        "bekker", observations, weighting="pressure_squared",
        estimator="averaged_exponent",
    )

    print("  (a) GRC-1 under a scale-aware exponent")
    print(f"      per-plate exponents {np.round(exponents, 4)}")
    print(
        f"      n0 = {observed.exponent_at_reference:.4f}   "
        f"beta = {observed.exponent_slope:.4f}   "
        f"condition {condition:.1f}"
    )
    print(
        f"      k_c = {observed.cohesive_modulus:.1f} against Bekker's "
        f"{incumbent.parameters['cohesive_modulus']:.1f}"
    )

    nulls: dict[tuple[str, int], list[float]] = {}
    for index, repeats in enumerate(NULL_REPEAT_COUNTS):
        nulls[("matched", repeats)] = matched_null(
            block, repeats, observed.reference_half_width_m,
            np.random.default_rng(SEED + index),
        )
        nulls[("designed", repeats)] = designed_null(
            repeats, len(PLATE_ORDER), observed.reference_half_width_m,
            np.random.default_rng(SEED + 100 + index),
        )

    print("\n  (b) exponent slope fitted to data with no scale dependence")
    print(
        f"  {'null':>10s} {'repeats':>8s} {'p5':>8s} {'p50':>8s} {'p95':>8s} "
        f"{'P(|null| >= observed)':>22s}"
    )
    for (kind, repeats), values in sorted(nulls.items()):
        array = np.array(values)
        exceed = float((np.abs(array) >= abs(observed.exponent_slope)).mean())
        print(
            f"  {kind:>10s} {repeats:8d} {np.percentile(array, 5):8.3f} "
            f"{np.median(array):8.3f} {np.percentile(array, 95):8.3f} "
            f"{exceed:21.3f}"
        )

    holdout: dict[tuple[int, int], dict[str, list[float]]] = {}
    for index, (repeats, plates) in enumerate(HOLDOUT_CASES):
        holdout[(repeats, plates)] = holdout_errors(
            repeats, plates, observed.reference_half_width_m,
            np.random.default_rng(SEED + 200 + index),
        )

    print("\n  (c) held-out error on single-exponent data")
    print(f"  {'repeats':>8s} {'plates':>7s} " + " ".join(f"{MODEL_LABELS[m]:>17s}" for m in MODELS))
    for (repeats, plates), models in sorted(holdout.items()):
        print(
            f"  {repeats:8d} {plates:7d} "
            + " ".join(f"{np.median(models[m])*100:16.1f}%" for m in MODELS)
        )

    arguments.figure.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure(
        observed.exponent_slope, nulls, holdout, arguments.report
    )
    figure.savefig(arguments.figure, dpi=FIGURE_STYLE["figure.dpi"])
    plt.close(figure)
    arguments.report.write_text(
        build_report(
            observed, exponents, condition,
            dict(incumbent.parameters), nulls, holdout,
        ),
        encoding="utf-8",
    )
    print("\n  verdict: do not adopt the scale-aware exponent")
    print(f"figure   {arguments.figure}")
    print(f"report   {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
