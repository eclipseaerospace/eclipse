# SPDX-License-Identifier: Apache-2.0
#
# validation/contact/campaign_design.py — what campaign would resolve
# plate-scale transfer?
#
# resolving_power.py showed that GRC-1's configuration cannot: its transfer
# errors are ordinary draws from a correct single-exponent process. This asks
# what configuration could, by sweeping the four things a campaign designer
# controls — how many plate sizes, how they are placed, how many repeats per
# plate, and how much of the repeat-to-repeat error is shared rather than
# independent.
#
# The result inverts the intuition it was built to test. Plate count is nearly
# flat and placement is nearly irrelevant; repeats carry the whole effect. The
# bottleneck is the quality of each pressure-sinkage curve, not the number of
# sizes. Published practice is three to five repeats, and the requirement here
# is twenty, so the gap is not marginal and no published campaign closes it.
#
# Two error metrics are reported and the difference between them is the reason
# the plate-count result is visible at all. Scoring a prediction against the
# held-out plate's own ensemble mean charges it for that curve's measurement
# noise, which at five repeats is around ten percent and swamps everything
# else. Scoring against the known true curve isolates what the fit actually got
# wrong. Only the second answers a design question; the first is what an
# experimenter would observe, so both are kept.
#
# The shared fraction is swept rather than assumed. GRC-1's own repeats give a
# point estimate near 0.09 on one plate of three, which is neither excludable
# nor established: the multiplicity is unfavourable and the null band came from
# an assumed correlation length. So the recommendation is required to hold
# across the whole range, and where it does not, that is itself the finding —
# a campaign that cannot measure its own shared fraction cannot know which
# regime it is in.
#
# References
#   Oravec HA (2009) Understanding Mechanical Behavior of Lunar Soils for the
#     Study of Vehicle Mobility. PhD dissertation, Case Western Reserve
#     University. Appendix D Code D5.

from __future__ import annotations

import argparse
import platform
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Final, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from eclipse.analysis.style import (
    ACCENT_PRIMARY,
    ACCENT_SECONDARY,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    figure_style,
)
from eclipse.fitting import PressureSinkageObservations, fit_contact_model
from eclipse.terramechanics import BekkerModel, DegenerateContactModelError

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_FIGURE_PATH: Final = (
    Path(__file__).resolve().parent / "figures" / "campaign-design.png"
)
DEFAULT_REPORT_PATH: Final = (
    Path(__file__).resolve().parent / "results" / "campaign-design.toml"
)

TRUE_MODEL: Final = BekkerModel(
    cohesive_modulus=4096.3537,
    frictional_modulus=-22284.5786,
    sinkage_exponent=1.232,
)
SPAN_M: Final = (0.038, 0.095)
RELATIVE_SIGMA: Final = 0.22
LAG_ONE: Final = 0.95
STEP_KPA: Final = 0.5
TOP_KPA: Final = 50.0
CLUSTER_POWER: Final = 2.5

PLATE_COUNTS: Final = (3, 4, 5, 6, 8)
REPEAT_COUNTS: Final = (3, 5, 10, 20, 40)
PLACEMENTS: Final = (
    "even_in_b",
    "even_in_reciprocal_b",
    "clustered_small",
    "clustered_large",
)
SHARED_FRACTIONS: Final = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)
TARGETS: Final = (0.10, 0.05, 0.03)
REALISATIONS: Final = 300
SEED: Final = 20260812
REPORT_SCHEMA_VERSION: Final = 1
PUBLISHED_PRACTICE_REPEATS: Final = 5

PRESSURE: Final[NDArray[np.float64]] = np.arange(
    STEP_KPA, TOP_KPA + STEP_KPA / 2.0, STEP_KPA
)

PLATE_COLOR: Final = ACCENT_PRIMARY
SHARED_COLOR: Final = ACCENT_SECONDARY

FIGURE_STYLE: Final[dict[str, Any]] = figure_style(
    {
        "figure.figsize": (10.2, 5.4),
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "font.size": 9.5,
        "legend.fontsize": 8.0,
        "figure.subplot.top": 0.726,
        "figure.subplot.bottom": 0.248,
        "figure.subplot.left": 0.062,
        "figure.subplot.right": 0.986,
        "figure.subplot.wspace": 0.210,
    }
)


@dataclass(frozen=True, slots=True)
class Configuration:
    placement: str
    plates: int
    repeats: int
    shared_fraction: float

    @property
    def total_tests(self) -> int:
        return self.plates * self.repeats


@dataclass(frozen=True, slots=True)
class Outcome:
    against_truth: float
    against_measurement: float
    usable: int


def half_widths(placement: str, count: int) -> NDArray[np.float64]:
    low, high = SPAN_M
    fraction = np.linspace(0.0, 1.0, count)
    if placement == "even_in_b":
        return np.asarray(np.linspace(low, high, count))
    if placement == "even_in_reciprocal_b":
        return np.asarray(1.0 / np.linspace(1.0 / low, 1.0 / high, count))
    if placement == "clustered_small":
        return np.asarray(low + (high - low) * fraction**CLUSTER_POWER)
    if placement == "clustered_large":
        return np.asarray(
            low + (high - low) * (1.0 - (1.0 - fraction) ** CLUSTER_POWER)
        )
    raise ValueError(f"unknown placement {placement!r}")


def noise_factor() -> NDArray[np.float64]:
    gap = PRESSURE[:, None] - PRESSURE[None, :]
    length = STEP_KPA / float(np.sqrt(-2.0 * np.log(LAG_ONE)))
    return np.asarray(
        np.linalg.cholesky(
            np.exp(-0.5 * (gap / length) ** 2) + 1e-10 * np.eye(PRESSURE.size)
        )
    )


def true_sinkage(half_width: float) -> NDArray[np.float64]:
    return np.asarray(
        TRUE_MODEL.sinkage(pressure=PRESSURE, contact_half_width=half_width)
    )


def ensemble_mean(
    half_width: float,
    repeats: int,
    shared_fraction: float,
    factor: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    # A shared component survives averaging; an independent one falls as
    # 1/sqrt(repeats). That difference is the whole reason the fraction matters.
    common = factor @ rng.standard_normal(PRESSURE.size)
    independent = (factor @ rng.standard_normal((PRESSURE.size, repeats))).T
    deviation = (
        np.sqrt(shared_fraction) * common[None, :]
        + np.sqrt(1.0 - shared_fraction) * independent
    ) * RELATIVE_SIGMA
    return np.asarray(
        np.mean(true_sinkage(half_width)[None, :] * (1.0 + deviation), axis=0)
    )


def observations_for(
    widths: NDArray[np.float64],
    curves: list[NDArray[np.float64]],
    keep: list[int],
) -> PressureSinkageObservations:
    return PressureSinkageObservations(
        contact_half_width_m=np.concatenate(
            [np.full(PRESSURE.size, widths[index]) for index in keep]
        ),
        sinkage_m=np.concatenate([curves[index] for index in keep]),
        pressure_kPa=np.tile(PRESSURE, len(keep)),
    )


def one_realisation(
    widths: NDArray[np.float64],
    repeats: int,
    shared_fraction: float,
    factor: NDArray[np.float64],
    rng: np.random.Generator,
) -> tuple[float, float] | None:
    curves = [
        ensemble_mean(width, repeats, shared_fraction, factor, rng) for width in widths
    ]
    if any((curve <= 0.0).any() for curve in curves):
        return None
    truth_errors, measured_errors = [], []
    for held in range(1, widths.size - 1):
        keep = [index for index in range(widths.size) if index != held]
        try:
            fit = fit_contact_model(
                "bekker",
                observations_for(widths, curves, keep),
                weighting="pressure_squared",
                estimator="averaged_exponent",
            )
            on_truth = fit.model.pressure(
                sinkage=true_sinkage(widths[held]), contact_half_width=widths[held]
            )
            on_measurement = fit.model.pressure(
                sinkage=curves[held], contact_half_width=widths[held]
            )
        except (ValueError, DegenerateContactModelError):
            return None
        truth_errors.append(np.mean(np.abs((on_truth - PRESSURE) / PRESSURE)))
        measured_errors.append(np.mean(np.abs((on_measurement - PRESSURE) / PRESSURE)))
    if not truth_errors:
        return None
    return float(np.mean(truth_errors)), float(np.mean(measured_errors))


def sweep() -> dict[Configuration, Outcome]:
    factor = noise_factor()
    grid: dict[Configuration, Outcome] = {}
    # Seeds are derived from the enumeration index rather than hash(), which is
    # salted per process and would make the report non-reproducible.
    for index, (placement, plates, repeats, shared) in enumerate(
        product(PLACEMENTS, PLATE_COUNTS, REPEAT_COUNTS, SHARED_FRACTIONS)
    ):
        widths = half_widths(placement, plates)
        rng = np.random.default_rng(SEED + index)
        truths, measured = [], []
        for _ in range(REALISATIONS):
            outcome = one_realisation(widths, repeats, shared, factor, rng)
            if outcome is None:
                continue
            truths.append(outcome[0])
            measured.append(outcome[1])
        if truths:
            grid[Configuration(placement, plates, repeats, shared)] = Outcome(
                float(np.median(truths)), float(np.median(measured)), len(truths)
            )
    return grid


def best_placement(grid: dict[Configuration, Outcome]) -> str:
    scores = {
        placement: float(
            np.mean(
                [
                    outcome.against_truth
                    for config, outcome in grid.items()
                    if config.placement == placement
                ]
            )
        )
        for placement in PLACEMENTS
    }
    return min(scores, key=lambda name: scores[name])


def cheapest(
    grid: dict[Configuration, Outcome], placement: str, shared: float, target: float
) -> Configuration | None:
    feasible = [
        config
        for config, outcome in grid.items()
        if config.placement == placement
        and config.shared_fraction == shared
        and outcome.against_truth <= target
    ]
    return min(feasible, key=lambda c: (c.total_tests, c.plates)) if feasible else None


def _format_float(value: float) -> str:
    return "nan" if not np.isfinite(value) else repr(float(value))


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def build_figure(
    grid: dict[Configuration, Outcome], placement: str, report_path: Path
) -> Figure:
    with plt.rc_context(cast(Any, FIGURE_STYLE)):
        figure, axes = plt.subplots(1, 2, squeeze=False)

        left = axes[0][0]
        for order, plates in enumerate(PLATE_COUNTS):
            values = [
                grid[Configuration(placement, plates, repeats, 0.0)].against_truth * 100
                for repeats in REPEAT_COUNTS
                if Configuration(placement, plates, repeats, 0.0) in grid
            ]
            left.plot(
                REPEAT_COUNTS[: len(values)], values,
                color=PLATE_COLOR, linewidth=1.5,
                alpha=0.35 + 0.65 * order / max(1, len(PLATE_COUNTS) - 1),
                marker="o", markersize=3.4, markerfacecolor="none",
                label=f"{plates} plates",
            )
        left.set_title(
            "by plate count, independent repeats (f = 0)",
            color=INK_SECONDARY, loc="left",
        )
        left.legend(loc="upper right", ncol=2)

        right = axes[0][1]
        plates = 4
        for order, shared in enumerate(SHARED_FRACTIONS):
            values = [
                grid[Configuration(placement, plates, repeats, shared)].against_truth
                * 100
                for repeats in REPEAT_COUNTS
                if Configuration(placement, plates, repeats, shared) in grid
            ]
            right.plot(
                REPEAT_COUNTS[: len(values)], values,
                color=SHARED_COLOR, linewidth=1.5,
                alpha=0.30 + 0.70 * order / max(1, len(SHARED_FRACTIONS) - 1),
                marker="o", markersize=3.4, markerfacecolor="none",
                label=f"f = {shared:.2f}",
            )
        right.set_title(
            "by shared fraction, four plates",
            color=INK_SECONDARY, loc="left",
        )
        right.legend(loc="upper right", ncol=2)

        for panel in (left, right):
            panel.set_xscale("log")
            panel.set_yscale("log")
            panel.set_xlabel("repeats per plate")
            panel.minorticks_off()
            panel.set_xticks(REPEAT_COUNTS)
            panel.set_xticklabels([str(v) for v in REPEAT_COUNTS])
            panel.set_ylim(0.7, 40.0)
            panel.set_yticks([1, 2, 5, 10, 20])
            panel.set_yticklabels(["1%", "2%", "5%", "10%", "20%"])
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)
            panel.set_axisbelow(True)
        left.set_ylabel("model error at a held-out plate")

        spread = [
            grid[Configuration(name, 4, 10, 0.0)].against_truth * 100
            for name in PLACEMENTS
            if Configuration(name, 4, 10, 0.0) in grid
        ]
        figure.text(
            0.062, 0.968,
            "What a campaign needs: repeats, not plate sizes",
            fontsize=12, color=INK_PRIMARY, weight="bold", ha="left", va="top",
        )
        figure.text(
            0.062, 0.912,
            "Error of a fit predicting a plate it was not fitted at, scored "
            "against the known true curve, so the held-out plate's own\n"
            "measurement noise is not charged to the model. Placement is flat "
            f"too: the four schemes span {min(spread):.1f} to "
            f"{max(spread):.1f}% at four plates\nand ten repeats. Published "
            f"bevameter practice is {PUBLISHED_PRACTICE_REPEATS} repeats.",
            fontsize=8.5, color=INK_SECONDARY, ha="left", va="top",
        )
        figure.text(
            0.062, 0.150,
            "The shared fraction f is the part of each repeat's error common to "
            "every repeat of that plate. It does not average away, so it caps\n"
            "what any number of repeats can reach, and GRC-1 cannot determine "
            "its own f.\n"
            f"Numbers behind this figure: {_display_path(report_path)}",
            fontsize=7.5, color=INK_MUTED, ha="left", va="top",
        )
        return figure


def build_report(grid: dict[Configuration, Outcome], placement: str) -> str:
    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# Generated by validation/contact/campaign_design.py. Do not edit.",
        "#",
        "# No timestamp: re-running against an unchanged seed leaves this",
        "# byte-identical. Seeds derive from the enumeration index, not hash().",
        "",
        f"schema_version = {REPORT_SCHEMA_VERSION}",
        f"seed = {SEED}",
        f"realisations_per_configuration = {REALISATIONS}",
        f'best_placement = "{placement}"',
        "",
        "[environment]",
        f'python = "{platform.python_version()}"',
        f'numpy = "{np.__version__}"',
        "",
        "# against_truth scores the prediction on the known true curve, isolating",
        "# what the fit got wrong. against_measurement scores it on the held-out",
        "# plate's own noisy ensemble mean, which additionally charges the model",
        "# for that curve's measurement error. Only the first answers a design",
        "# question; the second is what an experimenter would observe.",
        "[metric]",
        'against_truth = "prediction versus the generating model"',
        'against_measurement = "prediction versus the held-out noisy ensemble"',
        "",
        "[generating_model]",
        f"cohesive_modulus = {_format_float(TRUE_MODEL.cohesive_modulus)}",
        f"frictional_modulus = {_format_float(TRUE_MODEL.frictional_modulus)}",
        f"sinkage_exponent = {_format_float(TRUE_MODEL.sinkage_exponent)}",
        f"relative_sigma = {_format_float(RELATIVE_SIGMA)}",
        f"lag_one_autocorrelation = {_format_float(LAG_ONE)}",
        f"half_width_span_m = [{_format_float(SPAN_M[0])}, {_format_float(SPAN_M[1])}]",
        "",
        "# Placement is nearly irrelevant, which the sweep tested rather than",
        "# assumed: the regression is linear in reciprocal half-width, so even",
        "# spacing there was predicted to win. It does, by a margin too small to",
        "# act on.",
        "",
    ]
    for name in PLACEMENTS:
        config = Configuration(name, 4, 10, 0.0)
        if config in grid:
            lines += [
                "[[placement]]",
                f'id = "{name}"',
                "half_widths_m = ["
                + ", ".join(_format_float(v) for v in half_widths(name, 4))
                + "]",
                f"error_at_four_plates_ten_repeats = "
                f"{_format_float(grid[config].against_truth)}",
                "",
            ]

    lines += [
        "# The specification, computed from the grid rather than written here.",
        "# Cost is plates times repeats. Where a target is unreachable at a given",
        "# shared fraction, no configuration in the swept grid attains it.",
        "",
    ]
    for target, shared in product(TARGETS, SHARED_FRACTIONS):
        reached = cheapest(grid, placement, shared, target)
        lines += [
            "[[specification]]",
            f"target_error = {_format_float(target)}",
            f"shared_fraction = {_format_float(shared)}",
        ]
        if reached is None:
            lines += ["reachable = false", ""]
        else:
            lines += [
                "reachable = true",
                f"plates = {reached.plates}",
                f"repeats = {reached.repeats}",
                f"total_tests = {reached.total_tests}",
                f"achieved_error = {_format_float(grid[reached].against_truth)}",
                "",
            ]

    lines += ["# Every swept configuration.", ""]
    for config in sorted(
        grid, key=lambda c: (c.placement, c.plates, c.repeats, c.shared_fraction)
    ):
        outcome = grid[config]
        lines += [
            "[[configuration]]",
            f'placement = "{config.placement}"',
            f"plates = {config.plates}",
            f"repeats = {config.repeats}",
            f"shared_fraction = {_format_float(config.shared_fraction)}",
            f"against_truth = {_format_float(outcome.against_truth)}",
            f"against_measurement = {_format_float(outcome.against_measurement)}",
            f"usable_realisations = {outcome.usable}",
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sweep campaign designs for plate-scale transfer resolution."
    )
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    arguments = parser.parse_args(argv)

    grid = sweep()
    placement = best_placement(grid)
    print(f"  swept {len(grid)} configurations; best placement {placement}\n")

    print("  model error against truth, f = 0")
    print(f"  {'plates':>7s} " + " ".join(f"{r:>7d}" for r in REPEAT_COUNTS))
    for plates in PLATE_COUNTS:
        row = [
            grid[Configuration(placement, plates, repeats, 0.0)].against_truth * 100
            for repeats in REPEAT_COUNTS
        ]
        print(f"  {plates:7d} " + " ".join(f"{v:6.1f}%" for v in row))

    print("\n  model error against truth, 4 plates, by shared fraction")
    print(f"  {'f':>7s} " + " ".join(f"{r:>7d}" for r in REPEAT_COUNTS))
    for shared in SHARED_FRACTIONS:
        row = [
            grid[Configuration(placement, 4, repeats, shared)].against_truth * 100
            for repeats in REPEAT_COUNTS
        ]
        print(f"  {shared:7.2f} " + " ".join(f"{v:6.1f}%" for v in row))

    print("\n  specification: cheapest configuration reaching each target")
    print(f"  {'target':>8s} {'f':>6s} {'plates':>7s} {'repeats':>8s} {'achieved':>9s}")
    for target in TARGETS:
        for shared in SHARED_FRACTIONS:
            reached = cheapest(grid, placement, shared, target)
            if reached is None:
                print(
                    f"  {target*100:7.0f}% {shared:6.2f} {'--':>7s} {'--':>8s}"
                    f"  unreachable"
                )
            else:
                print(
                    f"  {target*100:7.0f}% {shared:6.2f} {reached.plates:7d} "
                    f"{reached.repeats:8d} {grid[reached].against_truth*100:8.1f}%"
                )

    arguments.figure.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure(grid, placement, arguments.report)
    figure.savefig(arguments.figure, dpi=FIGURE_STYLE["figure.dpi"])
    plt.close(figure)
    arguments.report.write_text(build_report(grid, placement), encoding="utf-8")
    print(f"\nfigure   {arguments.figure}")
    print(f"report   {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
