# SPDX-License-Identifier: Apache-2.0
#
# studies.mobility.stance_and_gait — what four contacts show that one averaged
# away.
#
# Rungs two and three of this project modelled a single patch and divided the
# weight by the number of feet. That is exact for the total and wrong for every
# foot, and this study is about the difference.
#
# Three results, in the order they were found.
#
# Averaging is not conservative. Sinkage is non-linear in pressure, so the
# most-loaded foot leaves the bearing model's published range while the mean --
# which is what the single-patch model used -- never does at all, because the
# mean falls as cos(slope) while the spread grows. There is a foot size at which
# the lumped model reports valid precisely where the distributed one reports
# invalid, and reports it with more confidence as the slope steepens. That is a
# general warning about lumped contact models rather than a fact about this
# platform.
#
# The stance those earlier rungs named does not exist. A diagonal pair lies on a
# line through the body centre, so the lateral moment equation forces its two
# loads equal while a slope's pitching moment requires them unequal. It balances
# on level ground and on no gradient at all. What that invalidates is narrow and
# worth stating precisely: the traction limit, the repose identity and the
# cohesion logarithm all survive, because they are properties of one patch under
# a given normal load and do not care how the load arrived. What needed the
# stance to exist was per-foot pressure, which is what this study computes.
#
# And the gait trade, which is the finding. Swing acceleration goes as the
# inverse square of swing duration, so keeping more feet on the ground shortens
# the swing and raises the tangential demand the stance feet must supply. The
# only gait here with a quasi-static solution on a slope is therefore the one
# that costs the most slip on level ground. That constraint comes from statics
# rather than from control, and it is the first thing in this project that
# constrains how the robot walks rather than what the soil does.
#
# On the flat-ground number being a bound rather than a value. Holding the body
# at constant speed puts the whole swing reaction through the feet; leaving it
# free puts none there and takes it as a velocity ripple instead. The truth
# depends on how stiff a controller is about speed, and there is no controller.
# The lower bound is not free either -- a velocity ripple costs energy elsewhere,
# in acceleration work this model does not carry -- so neither end is the
# comfortable one.
#
# Not a sortie envelope. No power, no thermal, no illumination, no terrain.
#
# References
#   Carrier WD III, Olhoeft GR, Mendell W (1991) Physical Properties of the
#     Lunar Surface. In: Lunar Sourcebook, ch. 9. Cambridge University Press.

from __future__ import annotations

import argparse
import math
import platform as host_platform
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from eclipse.analysis.boundary import (
    INSIDE,
    OUTSIDE,
    UNMEASURED,
    BoundaryRow,
    tally,
    text_table,
    toml_lines,
)
from eclipse.analysis.style import (
    ACCENT_PRIMARY,
    ACCENT_SECONDARY,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    figure_style,
)
from eclipse.io.platform import PlatformDefinition, load_platform
from eclipse.io.soil import janosi_hanamoto_model, load_soil, mohr_coulomb_model
from eclipse.platform import Platform, maximum_traversable_slope_degrees
from eclipse.stance import (
    Gait,
    StanceDistribution,
    UnbalanceableStanceError,
    distribute_normal_load,
    swing_reaction,
    wave_gait,
    within_stride_slip_ratio,
)
from eclipse.terramechanics import ContactModel, JanosiHanamotoModel, MohrCoulombModel

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
LUNAR_SOIL_PATH: Final = REPOSITORY_ROOT / "data" / "soils" / "lunar-intercrater.toml"
QUADRUPED_PATH: Final = (
    REPOSITORY_ROOT / "configs" / "platforms" / "nominal-quadruped.toml"
)
TRIPOD_PATH: Final = REPOSITORY_ROOT / "configs" / "platforms" / "nominal-tripod.toml"

FIGURE_DIRECTORY: Final = Path(__file__).resolve().parent / "figures"
DEFAULT_REPORT_PATH: Final = (
    Path(__file__).resolve().parent / "results" / "stance-and-gait.toml"
)

REPORT_SCHEMA_VERSION: Final = 1
SHALLOWEST_DEPTH_RANGE: Final = "0-15"
MILLIMETERS_PER_METER: Final = 1000.0
LUNAR_GRAVITY: Final = 1.62

SLOPE_DEGREES: Final[NDArray[np.float64]] = np.linspace(0.0, 40.0, 401)
DUTY_FACTORS: Final[NDArray[np.float64]] = np.linspace(0.30, 0.90, 121)
SPEEDS_M_PER_S: Final[NDArray[np.float64]] = np.linspace(0.10, 0.80, 141)

# The footprint is ordered front-left, front-right, rear-left, rear-right, so
# this lift order is rear-left, front-left, rear-right, front-right -- the
# sequence that keeps the centre of mass inside the remaining triangle.
CRAWL_LIFT_ORDER: Final = (2, 0, 3, 1)
CRAWL_DUTY: Final = 0.75
TROT_OFFSETS: Final = (0.0, 0.5, 0.5, 0.0)

SMALL_FOOT_HALF_WIDTH_M: Final = 0.020
FEASIBILITY_TEST_SLOPE_DEGREES: Final = 20.0
PHASE_SAMPLES: Final = 97

DAY_FOUR_SLIP_AT_THIRTY_DEGREES: Final = 0.0569


def caption(text: str, width: int = 140) -> str:
    return "\n".join(
        textwrap.fill(" ".join(paragraph.split()), width=width)
        for paragraph in text.split("\n")
    )


@dataclass(frozen=True, slots=True)
class Ground:
    contact: ContactModel
    strength: MohrCoulombModel
    mobilization: JanosiHanamotoModel
    sinkage_ceiling_m: float


def load_ground() -> Ground:
    dataset = load_soil(LUNAR_SOIL_PATH).datasets["carrier1991"]
    bekker = dataset.models["bekker"]
    return Ground(
        contact=bekker.extrapolating,
        strength=mohr_coulomb_model(dataset, depth_range_cm=SHALLOWEST_DEPTH_RANGE),
        mobilization=janosi_hanamoto_model(dataset),
        sinkage_ceiling_m=bekker.sinkage_validity.max,
    )


def variant(base: Platform, **changes: Any) -> Platform:
    fields = {name: getattr(base, name) for name in Platform.__dataclass_fields__}
    return Platform(**{**fields, **changes})


def gaits(platform: Platform) -> dict[str, Gait]:
    return {
        "trot": Gait(duty_factor=0.5, phase_offsets=TROT_OFFSETS),
        "wave-half-duty": wave_gait(
            lift_order=CRAWL_LIFT_ORDER, duty_factor=0.5
        ),
        "crawl": wave_gait(lift_order=CRAWL_LIFT_ORDER, duty_factor=CRAWL_DUTY),
        "crawl-slow-swing": wave_gait(
            lift_order=CRAWL_LIFT_ORDER, duty_factor=0.80
        ),
    }


def sinkage_m(ground: Ground, platform: Platform, load_N: ArrayLike) -> Any:
    stress_kPa = platform.contact_patch.normal_stress_kPa(normal_load_N=load_N)
    return ground.contact.sinkage(
        pressure=stress_kPa, contact_half_width=platform.foot_half_width_m
    )


def distribution_over_slope(
    platform: Platform, *, until_tipping: bool = True
) -> tuple[NDArray[np.float64], StanceDistribution]:
    """Per-foot load across the slope sweep, stopped where a foot lifts.

    Past tipping the platform is rotating about an edge of its support polygon
    and the loads the solve returns are negative, which is not a distribution.
    """
    distribution = distribute_normal_load(
        platform=platform,
        gravity_m_per_s2=LUNAR_GRAVITY,
        slope_degrees=SLOPE_DEGREES,
    )
    if not until_tipping:
        return SLOPE_DEGREES, distribution
    standing = np.asarray(~distribution.any_foot_unloaded)
    return SLOPE_DEGREES[standing], StanceDistribution(
        feet=distribution.feet,
        patch=distribution.patch,
        normal_load_N=distribution.normal_load_N[:, standing],
    )


def tipping_slope_degrees(platform: Platform) -> float:
    half_length = max(foot.x_m for foot in platform.footprint)
    return math.degrees(
        math.atan(half_length / platform.center_of_mass_height_m)
    )


def gait_is_feasible_on_slope(
    platform: Platform, gait: Gait, slope_degrees: float
) -> bool:
    """Whether every stance the schedule visits can balance the body.

    A gait is not a sequence of feasible instants by construction; a diagonal
    pair is perfectly good on the flat and has no solution on any gradient, so
    the whole cycle has to be walked.
    """
    for phase in np.linspace(0.0, 1.0, PHASE_SAMPLES, endpoint=False):
        down = np.flatnonzero(np.ravel(gait.in_stance(np.array([phase]))))
        try:
            distribute_normal_load(
                platform=platform,
                stance=tuple(platform.footprint[index] for index in down),
                gravity_m_per_s2=LUNAR_GRAVITY,
                slope_degrees=slope_degrees,
            )
        except UnbalanceableStanceError:
            return False
    return True


def flat_slip(platform: Platform, gait: Gait, ground: Ground) -> float:
    slip, _ = within_stride_slip_ratio(
        platform=platform,
        gait=gait,
        strength=ground.strength,
        mobilization=ground.mobilization,
        gravity_m_per_s2=LUNAR_GRAVITY,
    )
    return slip


def build_load_figure(platform: Platform, ground: Ground) -> Figure:
    slopes, distribution = distribution_over_slope(platform)
    ceiling_mm = ground.sinkage_ceiling_m * MILLIMETERS_PER_METER
    small = variant(
        platform,
        foot_half_width_m=SMALL_FOOT_HALF_WIDTH_M,
        foot_contact_area_m2=math.pi * SMALL_FOOT_HALF_WIDTH_M**2,
    )
    _, small_distribution = distribution_over_slope(small)

    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (10.4, 5.2),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.680,
                    "figure.subplot.bottom": 0.200,
                    "figure.subplot.left": 0.062,
                    "figure.subplot.right": 0.986,
                    "figure.subplot.wspace": 0.225,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 2, squeeze=False)

        left = axes[0][0]
        styles = ((0, ()), (0, (5, 2)), (0, (1.5, 1.8)), (0, (6, 2, 1.5, 2)))
        for order, (foot, loads) in enumerate(
            zip(distribution.feet, distribution.normal_load_N)
        ):
            left.plot(
                slopes,
                loads,
                color=ACCENT_PRIMARY if foot.x_m > 0 else ACCENT_SECONDARY,
                linewidth=1.5,
                linestyle=styles[order % len(styles)],
                label=foot.id,
            )
        left.plot(
            slopes,
            distribution.mean_N,
            color=INK_PRIMARY,
            linewidth=1.2,
            linestyle=(0, (2, 2)),
            label="mean, what one patch used",
        )
        left.set_title(
            "per-foot normal load against slope", color=INK_SECONDARY, loc="left"
        )
        left.set_ylabel("normal load (N)")
        left.legend(loc="upper left", ncol=2)

        right = axes[0][1]
        for label, dist, half_width, color in (
            (
                f"{platform.foot_half_width_m * MILLIMETERS_PER_METER:.0f} mm foot",
                distribution,
                platform.foot_half_width_m,
                ACCENT_PRIMARY,
            ),
            (
                f"{SMALL_FOOT_HALF_WIDTH_M * MILLIMETERS_PER_METER:.0f} mm foot",
                small_distribution,
                SMALL_FOOT_HALF_WIDTH_M,
                ACCENT_SECONDARY,
            ),
        ):
            patch_platform = variant(
                platform,
                foot_half_width_m=half_width,
                foot_contact_area_m2=math.pi * half_width**2,
            )
            for series, style, suffix in (
                (dist.maximum_N, (0, ()), "most-loaded foot"),
                (dist.mean_N, (0, (2, 2)), "mean"),
            ):
                right.plot(
                    slopes,
                    np.asarray(sinkage_m(ground, patch_platform, series))
                    * MILLIMETERS_PER_METER,
                    color=color,
                    linewidth=1.5,
                    linestyle=style,
                    label=f"{label}, {suffix}",
                )
        right.axhline(ceiling_mm, color=INK_PRIMARY, linewidth=1.1)
        right.annotate(
            f"published sinkage ceiling, {ceiling_mm:.0f} mm",
            xy=(slopes[-1], ceiling_mm),
            xytext=(-4, 4),
            textcoords="offset points",
            ha="right",
            color=INK_PRIMARY,
            fontsize=7.8,
        )
        right.set_title(
            "sinkage against slope, two foot sizes", color=INK_SECONDARY, loc="left"
        )
        right.set_ylabel("sinkage (mm)")
        right.legend(loc="upper left", ncol=2)

        tipping = tipping_slope_degrees(platform)
        for panel in (left, right):
            panel.axvline(
                tipping, color=INK_MUTED, linewidth=0.9, linestyle=(0, (2, 2))
            )
            panel.annotate(
                f"uphill feet lift, {tipping:.1f}°",
                xy=(tipping, 0.02),
                xycoords=("data", "axes fraction"),
                xytext=(-4, 0),
                textcoords="offset points",
                ha="right",
                va="bottom",
                rotation=90.0,
                color=INK_MUTED,
                fontsize=7.4,
            )
            panel.set_xlabel("slope (degrees)")
            panel.set_xlim(0.0, SLOPE_DEGREES[-1])
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)

        figure.suptitle(
            "The mean never leaves the bearing model's range; the most-loaded "
            "foot can",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.062,
            ha="left",
            y=0.960,
        )
        figure.text(
            0.062,
            0.905,
            caption(
                "Climbing shifts load onto the downhill feet, so the mean falls "
                "as cos(slope) while the spread grows. The lateral pairs "
                "coincide exactly, the slope having no lateral component, so "
                "four curves draw as two. Sinkage is non-linear in pressure, so "
                "averaging across feet is not conservative.\n"
                "At the nominal 30 mm foot both stay inside the published range. "
                "At 20 mm the most-loaded foot passes the ceiling on a gentle "
                "slope while the mean never reaches it at all -- the lumped "
                "model reports valid exactly where the distributed one does not.",
                width=148,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def build_gait_figure(platform: Platform, ground: Ground) -> Figure:
    duty_slip: list[float] = []
    duty_feasible: list[bool] = []
    for duty in DUTY_FACTORS:
        gait = wave_gait(lift_order=CRAWL_LIFT_ORDER, duty_factor=float(duty))
        try:
            duty_slip.append(flat_slip(platform, gait, ground))
        except ValueError:
            duty_slip.append(math.nan)
        duty_feasible.append(
            gait_is_feasible_on_slope(
                platform, gait, FEASIBILITY_TEST_SLOPE_DEGREES
            )
        )

    crawl = wave_gait(lift_order=CRAWL_LIFT_ORDER, duty_factor=CRAWL_DUTY)
    speed_slip = [
        flat_slip(
            variant(platform, nominal_speed_m_per_s=float(speed)), crawl, ground
        )
        for speed in SPEEDS_M_PER_S
    ]

    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (10.4, 5.2),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.680,
                    "figure.subplot.bottom": 0.200,
                    "figure.subplot.left": 0.066,
                    "figure.subplot.right": 0.986,
                    "figure.subplot.wspace": 0.225,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 2, squeeze=False)

        left = axes[0][0]
        slip = np.array(duty_slip)
        feasible = np.array(duty_feasible)
        finite = np.asarray(np.isfinite(slip))
        left.plot(
            DUTY_FACTORS[finite],
            slip[finite],
            color=ACCENT_PRIMARY,
            linewidth=1.6,
        )
        ceiling = (
            float(DUTY_FACTORS[np.argmax(~finite)])
            if bool(np.any(~finite))
            else float(DUTY_FACTORS[-1])
        )
        if bool(np.any(feasible)):
            # The usable window is the intersection: statically balanced on a
            # slope and executable at this speed. Shading past the ceiling would
            # advertise a region where the gait cannot be run at all.
            floor = float(DUTY_FACTORS[np.argmax(feasible)])
            left.axvspan(
                floor,
                ceiling,
                color=ACCENT_SECONDARY,
                alpha=0.16,
                linewidth=0.0,
                label=(
                    f"balances on a {FEASIBILITY_TEST_SLOPE_DEGREES:.0f}° slope "
                    "and is executable"
                ),
            )
        if bool(np.any(~finite)):
            left.axvline(
                ceiling, color=INK_PRIMARY, linewidth=1.1, linestyle=(0, (3, 2))
            )
            left.annotate(
                f"demand exceeds capacity\nabove duty {ceiling:.2f}",
                xy=(ceiling, 0.72),
                xycoords=("data", "axes fraction"),
                xytext=(-8, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                color=INK_PRIMARY,
                fontsize=7.8,
            )
        antiphase = float(np.argmin(np.abs(DUTY_FACTORS - 0.5)))
        left.annotate(
            "two legs swing in anti-phase\nand cancel exactly",
            xy=(0.5, float(slip[int(antiphase)])),
            xytext=(6, 26),
            textcoords="offset points",
            ha="left",
            color=INK_SECONDARY,
            fontsize=7.6,
            arrowprops={
                "arrowstyle": "-",
                "color": INK_MUTED,
                "linewidth": 0.7,
            },
        )
        left.set_title(
            "level-ground slip against duty factor, wave gait",
            color=INK_SECONDARY,
            loc="left",
        )
        left.set_xlabel("duty factor")
        left.set_ylabel("peak within-stride slip ratio")
        left.set_xlim(DUTY_FACTORS[0], DUTY_FACTORS[-1])
        left.set_ylim(0.0, 0.30)
        left.legend(loc="upper left")

        right = axes[0][1]
        right.plot(
            SPEEDS_M_PER_S, speed_slip, color=ACCENT_PRIMARY, linewidth=1.6
        )
        right.axhline(
            DAY_FOUR_SLIP_AT_THIRTY_DEGREES,
            color=INK_MUTED,
            linewidth=0.9,
            linestyle=(0, (4, 3)),
        )
        right.annotate(
            "slip from walking a 30° slope, rung two",
            xy=(SPEEDS_M_PER_S[0], DAY_FOUR_SLIP_AT_THIRTY_DEGREES),
            xytext=(4, 4),
            textcoords="offset points",
            color=INK_SECONDARY,
            fontsize=7.6,
        )
        right.axvline(
            platform.nominal_speed_m_per_s,
            color=INK_MUTED,
            linewidth=0.7,
            linestyle=(0, (2, 3)),
        )
        right.set_title(
            f"level-ground slip against speed, crawl at duty {CRAWL_DUTY:.2f}",
            color=INK_SECONDARY,
            loc="left",
        )
        right.set_xlabel("walking speed (m/s)")
        right.set_ylabel("peak within-stride slip ratio")
        right.set_xlim(SPEEDS_M_PER_S[0], SPEEDS_M_PER_S[-1])
        right.set_ylim(0.0, 0.30)

        for panel in (left, right):
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)

        figure.suptitle(
            "The only gait that balances on a slope is the one that slips most "
            "on the flat",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.066,
            ha="left",
            y=0.960,
        )
        figure.text(
            0.066,
            0.905,
            caption(
                "Swing acceleration goes as the inverse square of swing "
                "duration, so keeping more feet down to stay statically "
                "balanced shortens the swing and raises the tangential demand "
                "the stance feet must supply. Past a duty factor the demand "
                "exceeds capacity and the gait cannot be executed at this speed "
                "at all, which is a different failure from a large slip.\n"
                "Speed is the strongest lever in the model, quadratically. It is "
                "also the second independent reason this model prefers walking "
                "slowly, and it still has no standing-power term to push back — "
                "a known bias, recorded in the boundary table.",
                width=148,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def boundary_rows(platform: Platform, ground: Ground) -> tuple[BoundaryRow, ...]:
    return (
        BoundaryRow(
            quantity="contact force indeterminacy",
            published_range="not applicable",
            used="minimum sum of squared normal loads",
            status=UNMEASURED,
            basis=(
                "four feet give three equilibrium equations and four unknowns; "
                "a real controller optimises friction margin or actuator torque "
                "and would distribute differently. A tripod is determinate, so "
                "it is where a leak of this rule would show"
            ),
        ),
        BoundaryRow(
            quantity="stride kinematics",
            published_range="none",
            used="triangular angular-velocity profile over the hip sweep",
            status=UNMEASURED,
            basis=(
                "assumed; sets swing acceleration and so the whole tangential "
                "demand. A smoother profile lowers the peak and a stiffer one "
                "raises it"
            ),
        ),
        BoundaryRow(
            quantity="body speed regulation",
            published_range="none",
            used="constant speed, so the feet carry the whole swing reaction",
            status=UNMEASURED,
            basis=(
                "an upper bound. A body left free demands nothing of the soil "
                "and takes a 13 percent velocity ripple instead, which is not "
                "free either; closing this needs a controller"
            ),
        ),
        BoundaryRow(
            quantity="standing power and actuator efficiency",
            published_range="none",
            used="absent",
            status=UNMEASURED,
            basis=(
                "a known bias with a known direction: swing cost and now slip "
                "both fall with speed and nothing in the model pushes back, so "
                "it prefers walking arbitrarily slowly. A real platform has a "
                "cost-of-transport minimum at non-zero speed"
            ),
        ),
        BoundaryRow(
            quantity="stance regime",
            published_range="none",
            used="quasi-static throughout the stride",
            status=UNMEASURED,
            basis=(
                "gaits with a flight phase are refused rather than approximated; "
                "a trotting quadruped on a slope is balanced dynamically and is "
                "outside this model entirely"
            ),
        ),
        BoundaryRow(
            quantity="per-foot bearing, 30 mm foot",
            published_range=(
                f"0 to {ground.sinkage_ceiling_m * MILLIMETERS_PER_METER:.0f} mm"
            ),
            used=(
                f"{float(np.max(sinkage_m(ground, platform, distribution_over_slope(platform)[1].maximum_N))) * MILLIMETERS_PER_METER:.1f}"
                " mm at the most-loaded foot"
            ),
            status=INSIDE,
            basis="Heiken et al. (1991) Table 9.14, quasi-static normal bearing",
        ),
        BoundaryRow(
            quantity="per-foot bearing, 20 mm foot",
            published_range=(
                f"0 to {ground.sinkage_ceiling_m * MILLIMETERS_PER_METER:.0f} mm"
            ),
            used="most-loaded foot passes the ceiling; the mean never does",
            status=OUTSIDE,
            basis=(
                "the regime where a lumped contact model reports valid and a "
                "distributed one does not"
            ),
        ),
        BoundaryRow(
            quantity="angle of repose",
            published_range="none transcribed in this repository",
            used="30 to 35 degrees, carried as a band",
            status=UNMEASURED,
            basis="stated range, not a transcribed parameter; annotation only",
        ),
        BoundaryRow(
            quantity="shear mobilization under gait",
            published_range="none",
            used=(
                "K = "
                f"{ground.mobilization.shear_deformation_modulus * MILLIMETERS_PER_METER:.1f}"
                " mm, now inverted once per stride rather than once per slope"
            ),
            status=UNMEASURED,
            basis=(
                "steady in-situ slip observations carried into repeated loading "
                "at gait rates; it sets the within-stride slip as directly as it "
                "set the slope-driven one"
            ),
        ),
    )


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    return repr(float(value))


def build_report(
    quadruped: PlatformDefinition, tripod: PlatformDefinition, ground: Ground
) -> str:
    platform = quadruped.platform
    rows = boundary_rows(platform, ground)
    slopes, distribution = distribution_over_slope(platform)
    tipping = tipping_slope_degrees(platform)
    slipping = maximum_traversable_slope_degrees(
        platform=platform, strength=ground.strength, gravity_m_per_s2=LUNAR_GRAVITY
    )

    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# Stance distribution and gait, and what a single averaged patch hid.",
        "#",
        "# Generated by studies/mobility/stance_and_gait.py. Do not edit.",
        "#",
        "# No timestamp and no sampling: closed-form throughout, so re-running",
        "# leaves this byte-identical.",
        "#",
        "# Not a sortie envelope. No power, no thermal, no illumination.",
        "",
        f"schema_version = {REPORT_SCHEMA_VERSION}",
        "",
        "[environment]",
        f'python = "{host_platform.python_version()}"',
        f'numpy = "{np.__version__}"',
        "",
        "[platform]",
        f'id = "{quadruped.id}"',
        f'source = "{quadruped.source_path.relative_to(REPOSITORY_ROOT)}"',
        f"legs = {platform.legs}",
        f"total_mass_kg = {_format_float(platform.total_mass_kg)}",
        "center_of_mass_height_m = "
        f"{_format_float(platform.center_of_mass_height_m)}",
        "",
        "# Tipping is pure geometry: the uphill feet unload to zero at",
        "# tan(slope) = half_length / height, with mass and gravity cancelling.",
        "# It arrives before the foot-slip limit, so the platform rotates about",
        "# its downhill feet before they slide -- the reverse of what rung two",
        "# implied. Both still sit above the repose band, so neither binds on a",
        "# slope that stands, but the ordering is a design constraint the moment",
        "# anyone considers a taller or shorter body.",
        "[failure_ordering]",
        f"tipping_slope_degrees = {_format_float(tipping)}",
        f"foot_slip_slope_degrees = {_format_float(slipping)}",
        "tipping_first = " + str(tipping < slipping).lower(),
        "",
        "# Per-foot load across the slope sweep, stopped where a foot lifts.",
        "",
    ]

    for index in range(0, slopes.size, 40):
        lines += [
            "[[stance_distribution]]",
            f"slope_degrees = {_format_float(float(slopes[index]))}",
            "load_N = ["
            + ", ".join(
                _format_float(float(row[index]))
                for row in distribution.normal_load_N
            )
            + "]",
            f"mean_N = {_format_float(float(distribution.mean_N[index]))}",
            f"maximum_N = {_format_float(float(distribution.maximum_N[index]))}",
            f"spread = {_format_float(float(distribution.spread[index]))}",
            "",
        ]

    lines += [
        "# The gait trade. Static feasibility on a slope and tangential demand",
        "# on the flat pull against each other through swing duration, so the",
        "# only schedule here with a quasi-static solution on a gradient is the",
        "# one that slips most on level ground.",
        "#",
        "# slope_feasible walks the whole cycle rather than sampling one instant:",
        "# a diagonal pair is a perfectly good stance on the flat and has no",
        "# solution on any gradient.",
        "",
    ]
    for name, gait in gaits(platform).items():
        try:
            slip = flat_slip(platform, gait, ground)
        except ValueError:
            slip = math.nan
        reaction = swing_reaction(platform=platform, gait=gait)
        lines += [
            "[[gait]]",
            f'id = "{name}"',
            f"duty_factor = {_format_float(gait.duty_factor)}",
            "phase_offsets = ["
            + ", ".join(_format_float(v) for v in gait.phase_offsets)
            + "]",
            f"minimum_feet_down = {int(gait.feet_down(reaction.phase).min())}",
            f"peak_swing_reaction_N = {_format_float(reaction.peak_N)}",
            f"level_ground_slip_ratio = {_format_float(slip)}",
            "executable = " + str(math.isfinite(slip)).lower(),
            "slope_feasible = "
            + str(
                gait_is_feasible_on_slope(
                    platform, gait, FEASIBILITY_TEST_SLOPE_DEGREES
                )
            ).lower(),
            "body_speed_fluctuation_m_per_s = "
            f"{_format_float(reaction.body_speed_fluctuation_m_per_s)}",
            "",
        ]

    crawl = wave_gait(lift_order=CRAWL_LIFT_ORDER, duty_factor=CRAWL_DUTY)
    lines += [
        "# Speed is the strongest lever available, quadratically: the swing",
        "# reaction goes as its square. The model has no standing-power term to",
        "# push back, so it prefers walking arbitrarily slowly, and that bias is",
        "# recorded in the boundary table rather than left implicit.",
        "",
    ]
    for speed in (0.20, 0.35, 0.50, 0.65, 0.80):
        moving = variant(platform, nominal_speed_m_per_s=speed)
        reaction = swing_reaction(platform=moving, gait=crawl)
        lines += [
            "[[speed]]",
            f"speed_m_per_s = {_format_float(speed)}",
            f"peak_swing_reaction_N = {_format_float(reaction.peak_N)}",
            "level_ground_slip_ratio = "
            f"{_format_float(flat_slip(moving, crawl, ground))}",
            "",
        ]

    lines += [
        "# The interface test. Same evaluator, different platform file, no code",
        "# change -- and a different physical conclusion out of the unchanged",
        "# machinery, which is the stronger form of the test passing.",
        "#",
        "# Three non-collinear feet make the normal-load problem determinate, so",
        "# the quadruped's resolution rule has nothing to choose here. And",
        "# lifting any one of them leaves two, which balance a body only if its",
        "# centre of mass lies on the line between them: a tripod can stand and",
        "# it can fall over, and has no statically stable walking gait. No figure",
        "# for it, because the result is categorical rather than a curve.",
        "[morphology_test]",
        f'platform = "{tripod.id}"',
        f'source = "{tripod.source_path.relative_to(REPOSITORY_ROOT)}"',
        f"legs = {tripod.platform.legs}",
        "normal_load_problem_determinate = true",
        "has_statically_stable_walking_gait = false",
        f"tipping_slope_degrees = {_format_float(tipping_slope_degrees(tripod.platform))}",
        "load_per_foot_on_the_flat_N = "
        + _format_float(
            tripod.platform.total_mass_kg * LUNAR_GRAVITY / tripod.platform.legs
        ),
        "quadruped_load_per_foot_on_the_flat_N = "
        + _format_float(platform.total_mass_kg * LUNAR_GRAVITY / platform.legs),
        "",
        "# What the unbalanceable trot stance does and does not invalidate in the",
        "# earlier rungs. The traction limit, the repose identity and the",
        "# cohesion logarithm are properties of one patch under a given normal",
        "# load and do not depend on how that load arrived, so they stand. What",
        "# needed the stance to exist was per-foot pressure, which is computed",
        "# here rather than assumed.",
        "[supersedes]",
        'rung_two_stance = "two feet, divided evenly"',
        'status = "no quasi-static solution on any slope"',
        "survives = [",
        '  "maximum traversable slope",',
        '  "the repose-slope margin identity",',
        '  "slip depending on cohesion logarithmically",',
        "]",
        "superseded = [",
        '  "per-foot normal load on a slope",',
        '  "per-foot sinkage on a slope",',
        '  "slip on level ground, which was exactly zero",',
        "]",
        "",
        "# The measured-versus-extrapolated boundary for this study.",
        f"# {tally(rows)}",
        "",
        *toml_lines(rows),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Distribute load across four contacts, schedule them, and find the "
            "slip level ground still demands."
        )
    )
    parser.add_argument("--figure-directory", type=Path, default=FIGURE_DIRECTORY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    arguments = parser.parse_args(argv)

    quadruped = load_platform(QUADRUPED_PATH)
    tripod = load_platform(TRIPOD_PATH)
    ground = load_ground()

    arguments.figure_directory.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)

    for name, figure in (
        ("per-foot-load-against-slope", build_load_figure(quadruped.platform, ground)),
        ("level-ground-slip", build_gait_figure(quadruped.platform, ground)),
    ):
        path = arguments.figure_directory / f"{name}.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        print(f"wrote {path.relative_to(REPOSITORY_ROOT)}")

    arguments.report.write_text(
        build_report(quadruped, tripod, ground), encoding="utf-8"
    )
    print(f"wrote {arguments.report.relative_to(REPOSITORY_ROOT)}")

    print("\n  measured against extrapolated\n")
    print(text_table(boundary_rows(quadruped.platform, ground)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
