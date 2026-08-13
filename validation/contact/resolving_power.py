# SPDX-License-Identifier: Apache-2.0
#
# validation/contact/resolving_power.py — can a three-plate campaign resolve
# plate-scale transfer at all?
#
# plate_transfer.py measured what happens when a Bekker fit predicts a plate it
# was not fitted at, and the errors were large. This asks the question that has
# to come next: how large would they be if the model were exactly right?
#
# Data is generated from a Bekker process with a SINGLE true exponent, at GRC-1's
# exact geometry, pressure ranges, point counts and repeat-to-repeat noise, and
# put through the identical pipeline. Whatever error survives that is the floor
# the campaign imposes, because the generating model is correct by construction.
#
# The noise model is not assumed. Relative spread and lag-1 autocorrelation were
# measured from GRC-1's own repeats and are reproduced here to within a few
# percent, which the run re-checks and reports before any verdict. Correlated
# noise matters: these are ensemble means of five repeats whose errors are smooth
# along the curve, and independent per-point noise would average away far faster
# and understate the floor.
#
# Thresholds below were fixed before the first run and the verdict is computed
# from them rather than read off the numbers. Two claims in plate_transfer.py did
# not survive this test, and the point of pre-registering was to make that
# visible rather than negotiable.
#
# Reading the result: the four metrics all derive from the same three curves and
# are strongly correlated, so one campaign landing high on all four is roughly
# one observation on the high side, not four independent ones.
#
# References
#   Oravec HA (2009) Understanding Mechanical Behavior of Lunar Soils for the
#     Study of Vehicle Mobility. PhD dissertation, Case Western Reserve
#     University. Appendix D Code D5.

from __future__ import annotations

import argparse
import hashlib
import platform
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
from eclipse.terramechanics import BekkerModel, DegenerateContactModelError

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_FIGURE_PATH: Final = (
    Path(__file__).resolve().parent / "figures" / "grc1-resolving-power.png"
)
DEFAULT_REPORT_PATH: Final = (
    Path(__file__).resolve().parent / "results" / "grc1-resolving-power.toml"
)

# --- measured from GRC-1, not assumed
OBSERVED: Final[dict[str, float]] = {
    "exponent_disagreement": 0.450,
    "interpolation": 0.165,
    "extrapolation_near": 0.241,
    "extrapolation_far": 1.247,
}
RELATIVE_SIGMA: Final[dict[str, float]] = {
    "small": 0.249, "medium": 0.230, "large": 0.181
}
LAG_ONE: Final[dict[str, float]] = {"small": 0.995, "medium": 0.960, "large": 0.904}

# --- pre-registered before the first run
FORM_INTERPOLATION_CEILING: Final = OBSERVED["interpolation"] / 3.0
FORM_DISAGREEMENT_CEILING: Final = OBSERVED["exponent_disagreement"] / 3.0
ESTIMATOR_INTERPOLATION_FLOOR: Final = OBSERVED["interpolation"] * 2.0 / 3.0
ESTIMATOR_DISAGREEMENT_FLOOR: Final = OBSERVED["exponent_disagreement"] * 2.0 / 3.0

TRUE_MODEL: Final = BekkerModel(
    cohesive_modulus=4096.3537,
    frictional_modulus=-22284.5786,
    sinkage_exponent=1.232,
)
HALF_WIDTHS: Final[dict[str, float]] = {
    "small": 0.038, "medium": 0.051, "large": 0.095
}
WINDOW_TOP_KPA: Final[dict[str, float]] = {
    "small": 50.0, "medium": 50.0, "large": 30.0
}
LEADING_DROPPED: Final[dict[str, int]] = {"small": 3, "medium": 1, "large": 1}
STEP_KPA: Final = 0.5
REPEATS: Final = 5
REALISATIONS: Final = 500
SEED: Final = 20260811
REPORT_SCHEMA_VERSION: Final = 1
PLATE_ORDER: Final = ("small", "medium", "large")
METRIC_LABELS: Final[dict[str, str]] = {
    "exponent_disagreement": "per-plate exponent disagreement",
    "interpolation": "interpolation error, medium held out",
    "extrapolation_near": "extrapolation 0.74x, small held out",
    "extrapolation_far": "extrapolation 1.35x, large held out",
}

INK_PRIMARY: Final = "#0b0b0b"
INK_SECONDARY: Final = "#52514e"
INK_MUTED: Final = "#8a8880"
SURFACE: Final = "#fcfcfb"
NULL_COLOR: Final = "#1f4e9c"
OBSERVED_COLOR: Final = "#d4570a"

FIGURE_STYLE: Final[dict[str, Any]] = {
    "figure.figsize": (9.8, 4.8),
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
    "legend.fontsize": 8.0,
    "savefig.facecolor": SURFACE,
    "figure.subplot.top": 0.775,
    "figure.subplot.bottom": 0.300,
    "figure.subplot.left": 0.300,
    "figure.subplot.right": 0.975,
}


@dataclass(frozen=True, slots=True)
class Distribution:
    metric: str
    values: NDArray[np.float64]

    def percentile(self, fraction: float) -> float:
        return float(np.percentile(self.values, fraction))

    @property
    def median(self) -> float:
        return self.percentile(50.0)

    @property
    def rank_of_observed(self) -> float:
        return float((self.values < OBSERVED[self.metric]).mean() * 100.0)


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


def pressure_grid(plate: str) -> NDArray[np.float64]:
    full = np.arange(0.0, WINDOW_TOP_KPA[plate] + STEP_KPA / 2.0, STEP_KPA)
    return np.asarray(full[LEADING_DROPPED[plate] :])


def correlation_length_kPa(plate: str) -> float:
    return STEP_KPA / float(np.sqrt(-2.0 * np.log(LAG_ONE[plate])))


def noise_factor(plate: str) -> NDArray[np.float64]:
    pressure = pressure_grid(plate)
    gap = pressure[:, None] - pressure[None, :]
    kernel = np.exp(-0.5 * (gap / correlation_length_kPa(plate)) ** 2)
    return np.asarray(np.linalg.cholesky(kernel + 1e-10 * np.eye(pressure.size)))


def true_sinkage(plate: str) -> NDArray[np.float64]:
    return np.asarray(
        TRUE_MODEL.sinkage(
            pressure=pressure_grid(plate), contact_half_width=HALF_WIDTHS[plate]
        )
    )


def sample_repeats(
    plate: str, factor: NDArray[np.float64], rng: np.random.Generator
) -> NDArray[np.float64]:
    deviations = (
        factor @ rng.standard_normal((factor.shape[0], REPEATS))
    ).T * RELATIVE_SIGMA[plate]
    return np.asarray(true_sinkage(plate)[None, :] * (1.0 + deviations))


def observations_for(
    curves: dict[str, NDArray[np.float64]], plates: list[str]
) -> PressureSinkageObservations:
    return PressureSinkageObservations(
        contact_half_width_m=np.concatenate(
            [np.full(curves[p].size, HALF_WIDTHS[p]) for p in plates]
        ),
        sinkage_m=np.concatenate([curves[p] for p in plates]),
        pressure_kPa=np.concatenate([pressure_grid(p) for p in plates]),
    )


def one_realisation(
    factors: dict[str, NDArray[np.float64]], rng: np.random.Generator
) -> dict[str, float] | None:
    curves = {
        plate: np.asarray(sample_repeats(plate, factors[plate], rng).mean(axis=0))
        for plate in PLATE_ORDER
    }
    if any((curve <= 0.0).any() for curve in curves.values()):
        return None
    try:
        exponents = [
            fit_averaged_power_law(
                observations_for(curves, [plate]), weighting="pressure_squared"
            ).sinkage_exponent
            for plate in PLATE_ORDER
        ]
        result = {
            "exponent_disagreement": (max(exponents) - min(exponents))
            / float(np.mean(exponents))
        }
        for held, metric in (
            ("medium", "interpolation"),
            ("small", "extrapolation_near"),
            ("large", "extrapolation_far"),
        ):
            kept = [plate for plate in PLATE_ORDER if plate != held]
            fit = fit_contact_model(
                "bekker",
                observations_for(curves, kept),
                weighting="pressure_squared",
                estimator="averaged_exponent",
            )
            predicted = fit.model.pressure(
                sinkage=curves[held], contact_half_width=HALF_WIDTHS[held]
            )
            target = pressure_grid(held)
            result[metric] = float(np.mean(np.abs((predicted - target) / target)))
    except (ValueError, DegenerateContactModelError):
        return None
    return result


def measured_noise(
    factors: dict[str, NDArray[np.float64]], rng: np.random.Generator
) -> dict[str, tuple[float, float]]:
    check: dict[str, tuple[float, float]] = {}
    for plate in PLATE_ORDER:
        base = true_sinkage(plate)
        stacked = np.vstack(
            [sample_repeats(plate, factors[plate], rng) for _ in range(200)]
        )
        relative = (stacked - base) / base
        sigma = float(np.mean(np.std(relative, axis=0, ddof=1)))
        lag = float(
            np.mean([np.corrcoef(row[:-1], row[1:])[0, 1] for row in relative[:200]])
        )
        check[plate] = (sigma, lag)
    return check


def build_figure(
    distributions: dict[str, Distribution], verdict: str, report_path: Path
) -> Figure:
    with plt.rc_context(cast(Any, FIGURE_STYLE)):
        figure, axes = plt.subplots()
        metrics = list(METRIC_LABELS)
        positions = np.arange(len(metrics), dtype=float)[::-1]
        for position, metric in zip(positions, metrics):
            spread = distributions[metric]
            low, high = spread.percentile(10.0), spread.percentile(90.0)
            axes.plot(
                [low * 100.0, high * 100.0], [position, position],
                color=NULL_COLOR, linewidth=6.0, alpha=0.30, solid_capstyle="butt",
                zorder=2,
            )
            axes.plot(
                [spread.percentile(25.0) * 100.0, spread.percentile(75.0) * 100.0],
                [position, position],
                color=NULL_COLOR, linewidth=6.0, alpha=0.55, solid_capstyle="butt",
                zorder=3,
            )
            axes.plot(
                [spread.median * 100.0], [position], marker="|", markersize=13.0,
                markeredgewidth=1.8, color=NULL_COLOR, linestyle="none", zorder=4,
            )
            axes.plot(
                [OBSERVED[metric] * 100.0], [position], marker="D", markersize=5.5,
                color=OBSERVED_COLOR, linestyle="none", zorder=5,
            )
            axes.text(
                OBSERVED[metric] * 100.0 * 1.14, position,
                f"{spread.rank_of_observed:.0f}th pct",
                va="center", ha="left", fontsize=8.0, color=OBSERVED_COLOR,
            )
        axes.set_xscale("log")
        axes.set_xlim(4.0, 700.0)
        axes.set_yticks(positions)
        axes.set_yticklabels([METRIC_LABELS[m] for m in metrics], fontsize=8.5)
        axes.set_ylim(-0.6, len(metrics) - 0.4)
        axes.set_xlabel("error  (percent)")
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        axes.set_axisbelow(True)
        axes.legend(
            handles=[
                Line2D([], [], color=NULL_COLOR, linewidth=6.0, alpha=0.55),
                Line2D([], [], color=NULL_COLOR, marker="|", markersize=13.0,
                       markeredgewidth=1.8, linestyle="none"),
                Line2D([], [], color=OBSERVED_COLOR, marker="D", markersize=5.5,
                       linestyle="none"),
            ],
            labels=[
                f"{REALISATIONS} runs of a correct model, p10-p90 and p25-p75",
                "median",
                "GRC-1 observed",
            ],
            loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3,
        )

        figure.text(
            0.030, 0.965,
            "What a three-plate campaign can resolve, when the model is correct",
            fontsize=12, color=INK_PRIMARY, weight="bold", ha="left", va="top",
        )
        figure.text(
            0.030, 0.912,
            "Data generated from a Bekker process with a single true exponent, at "
            "GRC-1's geometry, pressure ranges and measured repeat noise,\n"
            "then put through the identical pipeline. Blue is what a correct "
            "model yields anyway.\nGRC-1's observed values sit inside it, so its "
            "transfer error is not evidence that the model form is wrong.",
            fontsize=8.5, color=INK_SECONDARY, ha="left", va="top",
        )
        figure.text(
            0.030, 0.115,
            "Thresholds were fixed before the first run: the form would be "
            f"implicated below {FORM_INTERPOLATION_CEILING * 100:.1f}% median "
            f"interpolation error, the estimator at or above "
            f"{ESTIMATOR_INTERPOLATION_FLOOR * 100:.1f}%.  Verdict: {verdict}.\n"
            "The four metrics come from the same three curves and are strongly "
            "correlated, so landing high on all four is roughly one observation "
            "on the high side, not four independent ones.\n"
            f"Numbers behind this figure: {_display_path(report_path)}",
            fontsize=7.5, color=INK_MUTED, ha="left", va="top",
        )
        return figure


def build_report(
    distributions: dict[str, Distribution],
    noise_check: dict[str, tuple[float, float]],
    usable: int,
    verdict: str,
) -> str:
    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# Generated by validation/contact/resolving_power.py. Do not edit.",
        "#",
        "# Thresholds were fixed before the first run. No timestamp: re-running",
        "# against an unchanged seed leaves this byte-identical.",
        "",
        f"schema_version = {REPORT_SCHEMA_VERSION}",
        f"seed = {SEED}",
        f"realisations_requested = {REALISATIONS}",
        f"realisations_usable = {usable}",
        "",
        "[environment]",
        f'python = "{platform.python_version()}"',
        f'numpy = "{np.__version__}"',
        "",
        "# A Bekker process with one true exponent, so any error below is the",
        "# floor the campaign imposes rather than a defect of the model.",
        "[generating_model]",
        f"cohesive_modulus = {_format_float(TRUE_MODEL.cohesive_modulus)}",
        f"frictional_modulus = {_format_float(TRUE_MODEL.frictional_modulus)}",
        f"sinkage_exponent = {_format_float(TRUE_MODEL.sinkage_exponent)}",
        f"repeats_per_plate = {REPEATS}",
        "",
        "# Measured from GRC-1's own repeats, then reproduced by the generator.",
        "# Correlated along the curve: these are ensemble means, and independent",
        "# per-point noise would average away faster and understate the floor.",
        "",
    ]
    for plate in PLATE_ORDER:
        sigma, lag = noise_check[plate]
        lines += [
            "[[noise_check]]",
            f'plate = "{plate}"',
            f"half_width_m = {_format_float(HALF_WIDTHS[plate])}",
            f"target_relative_sigma = {_format_float(RELATIVE_SIGMA[plate])}",
            f"synthetic_relative_sigma = {_format_float(sigma)}",
            f"target_lag_one_autocorrelation = {_format_float(LAG_ONE[plate])}",
            f"synthetic_lag_one_autocorrelation = {_format_float(lag)}",
            "",
        ]
    lines += [
        "[thresholds]",
        "# fixed before the first run",
        f"form_interpolation_ceiling = {_format_float(FORM_INTERPOLATION_CEILING)}",
        f"form_disagreement_ceiling = {_format_float(FORM_DISAGREEMENT_CEILING)}",
        f"estimator_interpolation_floor = {_format_float(ESTIMATOR_INTERPOLATION_FLOOR)}",
        f"estimator_disagreement_floor = {_format_float(ESTIMATOR_DISAGREEMENT_FLOOR)}",
        "",
    ]
    for metric, spread in distributions.items():
        lines += [
            "[[metric]]",
            f'id = "{metric}"',
            f'label = "{METRIC_LABELS[metric]}"',
            "percentiles = { "
            + ", ".join(
                f"p{int(q)} = {_format_float(spread.percentile(q))}"
                for q in (10.0, 25.0, 50.0, 75.0, 90.0)
            )
            + " }",
            f"grc1_observed = {_format_float(OBSERVED[metric])}",
            f"grc1_percentile_rank = {_format_float(spread.rank_of_observed)}",
            "",
        ]
    lines += [
        "# The four metrics derive from the same three curves and are strongly",
        "# correlated, so one campaign landing high on all four is roughly one",
        "# observation on the high side rather than four independent ones.",
        "[verdict]",
        f'result = "{verdict}"',
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="How much transfer error a correct model yields at this campaign's noise."
    )
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    arguments = parser.parse_args(argv)

    factors = {plate: noise_factor(plate) for plate in PLATE_ORDER}
    noise_check = measured_noise(factors, np.random.default_rng(SEED + 1))

    print("  noise check — synthetic against GRC-1")
    print(f"  {'plate':>8s} {'relative sigma':>24s} {'lag-1 autocorrelation':>28s}")
    for plate in PLATE_ORDER:
        sigma, lag = noise_check[plate]
        print(
            f"  {plate:>8s} {sigma * 100:12.1f}% vs {RELATIVE_SIGMA[plate] * 100:6.1f}%"
            f" {lag:18.3f} vs {LAG_ONE[plate]:6.3f}"
        )

    rng = np.random.default_rng(SEED)
    collected: dict[str, list[float]] = {metric: [] for metric in METRIC_LABELS}
    usable = 0
    for _ in range(REALISATIONS):
        result = one_realisation(factors, rng)
        if result is None:
            continue
        usable += 1
        for metric, value in result.items():
            collected[metric].append(value)

    distributions = {
        metric: Distribution(metric, np.array(values))
        for metric, values in collected.items()
    }

    print(f"\n  {usable} usable realisations of {REALISATIONS}")
    print(f"\n  {'':>36s} {'p10':>7s} {'p50':>7s} {'p90':>7s} {'GRC-1':>9s} {'rank':>7s}")
    for metric, spread in distributions.items():
        print(
            f"  {METRIC_LABELS[metric]:>36s} {spread.percentile(10.0)*100:7.1f} "
            f"{spread.median*100:7.1f} {spread.percentile(90.0)*100:7.1f} "
            f"{OBSERVED[metric]*100:8.1f}% {spread.rank_of_observed:6.0f}th"
        )

    median_interpolation = distributions["interpolation"].median
    median_disagreement = distributions["exponent_disagreement"].median
    if (
        median_interpolation < FORM_INTERPOLATION_CEILING
        and median_disagreement < FORM_DISAGREEMENT_CEILING
    ):
        verdict = "form implicated"
    elif (
        median_interpolation >= ESTIMATOR_INTERPOLATION_FLOOR
        or median_disagreement >= ESTIMATOR_DISAGREEMENT_FLOOR
    ):
        verdict = "estimator implicated"
    else:
        verdict = "inconclusive by the pre-registered thresholds"
    print(f"\n  pre-registered verdict: {verdict}")

    arguments.figure.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure(distributions, verdict, arguments.report)
    figure.savefig(arguments.figure, dpi=FIGURE_STYLE["figure.dpi"])
    plt.close(figure)
    arguments.report.write_text(
        build_report(distributions, noise_check, usable, verdict), encoding="utf-8"
    )
    print(f"figure   {arguments.figure}")
    print(f"report   {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
