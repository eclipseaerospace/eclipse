# SPDX-License-Identifier: Apache-2.0
#
# studies.mobility.traversability — what a slope costs, and what decides whether
# it can be walked at all.
#
# Day 3 assumed slip and swing. Day 4 computes them from a stated body, and two
# things fall out that the assumed version could not have shown.
#
# The first is that traction is not the constraint. The foot-slip limit lands
# near 43 degrees for lunar parameters, well above the 30 to 35 degree repose
# band of regolith that stands. So a legged platform of this size is never
# traction-limited on a slope that exists, and the vertical asymptote in the
# first figure is the least interesting thing in it.
#
# The second is what replaces it. A slope standing at its angle of repose has
# mobilized its friction angle exactly, so cos(theta)*tan(phi) - sin(theta)
# vanishes identically and the frictional term contributes nothing whatever to
# the traction margin. The entire reserve is cohesion. That is an identity, not
# an approximation, and it holds at any gravity.
#
# Which inverts a Day 2 result without contradicting it. Cohesion is a few
# percent of shear strength at foot stress, so it barely affects how much
# traction exists -- that was measured and it stands. On a repose slope it is
# one hundred percent of how much traction is spare. Strength and margin are
# different questions and the mission-relevant one is margin.
#
# So the cohesion sweep is the centre of this study rather than a sensitivity
# appendix. It has a third part, and the third part is the one that decides what
# to do about the second.
#
# Slip depends on cohesion only logarithmically. Janosi-Hanamoto is exponential
# in slide, so inverting it puts a logarithm around the demand ratio, and losing
# a decade of cohesion costs a factor of 1.8 in slip rather than a decade. The
# published in-situ range of plus or minus seventeen percent moves slip by six
# percent. Slip reaches one -- the foot sliding a whole stance without advancing
# -- only near 1e-7 kPa, six decades below the measured value and not a physical
# quantity of soil.
#
# The three results are a chain, not a contradiction. Cohesion barely affects
# how much traction exists. It is the entirety of how much traction is spare.
# And it still does not decide the outcome, because the mobilization law is
# exponential and its inverse is forgiving. The middle result is the structural
# one and the third is the one a mission would act on.
#
# Modelling note on the repose slope, because it looks like an assumption and is
# not. The friction angle used there is the slope angle itself, on the grounds
# that a slope which stands at theta has mobilized exactly theta whatever its
# density. The published 42 degrees belongs to soil at 65 percent relative
# density averaged over the top 15 cm, and the Sourcebook is explicit that the
# top few centimetres are looser than that average without saying by how much --
# recorded in the soil file as the surface_density_unresolved anomaly. Using the
# slope angle sidesteps that gap rather than guessing at it.
#
# Not a sortie envelope. No power, no thermal, no illumination, no real terrain.
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
from numpy.typing import NDArray

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
from eclipse.io.soil import (
    cohesion_range_kPa,
    janosi_hanamoto_model,
    load_soil,
    mohr_coulomb_model,
)
from eclipse.mobility import CostOfTransport, cost_of_transport
from eclipse.platform import (
    Platform,
    equilibrium_slip_ratio,
    maximum_traversable_slope_degrees,
    swing_work_per_meter,
    swing_work_per_stride,
    traction_balance,
)
from eclipse.terramechanics import ContactModel, JanosiHanamotoModel, MohrCoulombModel

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
LUNAR_SOIL_PATH: Final = REPOSITORY_ROOT / "data" / "soils" / "lunar-intercrater.toml"
PLATFORM_PATH: Final = (
    REPOSITORY_ROOT / "configs" / "platforms" / "nominal-quadruped.toml"
)
FIGURE_DIRECTORY: Final = Path(__file__).resolve().parent / "figures"
DEFAULT_REPORT_PATH: Final = (
    Path(__file__).resolve().parent / "results" / "traversability.toml"
)

REPORT_SCHEMA_VERSION: Final = 1
SHALLOWEST_DEPTH_RANGE: Final = "0-15"
MILLIMETERS_PER_METER: Final = 1000.0

EARTH_GRAVITY: Final = 9.81
MARS_GRAVITY: Final = 3.71
LUNAR_GRAVITY: Final = 1.62
GRAVITIES: Final = (
    ("earth", EARTH_GRAVITY),
    ("mars", MARS_GRAVITY),
    ("moon", LUNAR_GRAVITY),
)

# Observed lunar slopes stand up to roughly this band and no steeper. Not a
# transcribed parameter -- no repose angle is recorded in the soil file -- so it
# is carried as a stated range rather than a number, and the report says so.
REPOSE_BAND_DEGREES: Final = (30.0, 35.0)
REPRESENTATIVE_REPOSE_DEGREES: Final = 32.0

# Day 3's assumption, kept only so the computed value can be shown against it.
DAY_THREE_SWING_J_PER_METER: Final = 25.0
DAY_THREE_SLIP_RATIO: Final = 0.10

# GRC-1's five-level bevameter regression puts sigma_c/c near 300 percent. The
# lunar range comes from the soil file and is about seventeen percent.
SIMULANT_GRADE_RELATIVE_SIGMA: Final = 3.0

SLOPE_DEGREES: Final[NDArray[np.float64]] = np.linspace(0.0, 45.0, 451)
GRAVITY_SWEEP: Final[NDArray[np.float64]] = np.linspace(1.0, 10.0, 91)
FOOT_HALF_WIDTHS_M: Final = (0.020, 0.030, 0.040, 0.050, 0.060)
NO_PROGRESS_SLIP: Final = 1.0


def caption(text: str, width: int = 132) -> str:
    return "\n".join(
        textwrap.fill(" ".join(paragraph.split()), width=width)
        for paragraph in text.split("\n")
    )


@dataclass(frozen=True, slots=True)
class Ground:
    contact: ContactModel
    strength: MohrCoulombModel
    mobilization: JanosiHanamotoModel
    cohesion_range_kPa: tuple[float, float]


def load_ground() -> Ground:
    soil = load_soil(LUNAR_SOIL_PATH)
    dataset = soil.datasets["carrier1991"]
    return Ground(
        contact=dataset.models["bekker"].extrapolating,
        strength=mohr_coulomb_model(dataset, depth_range_cm=SHALLOWEST_DEPTH_RANGE),
        mobilization=janosi_hanamoto_model(dataset),
        cohesion_range_kPa=cohesion_range_kPa(
            dataset, depth_range_cm=SHALLOWEST_DEPTH_RANGE
        ),
    )


def variant(base: Platform, **changes: Any) -> Platform:
    fields = {name: getattr(base, name) for name in Platform.__dataclass_fields__}
    return Platform(**{**fields, **changes})


def slip_against_slope(
    *, platform: Platform, ground: Ground, gravity_m_per_s2: float
) -> NDArray[np.float64]:
    return equilibrium_slip_ratio(
        platform=platform,
        strength=ground.strength,
        mobilization=ground.mobilization,
        gravity_m_per_s2=gravity_m_per_s2,
        slope_degrees=SLOPE_DEGREES,
    )


def cost_against_slope(
    *, platform: Platform, ground: Ground, gravity_m_per_s2: float
) -> tuple[NDArray[np.float64], CostOfTransport]:
    """Cost of transport with slip and swing computed rather than assumed.

    Slopes where slip reaches one are dropped: the foot slides a whole stance
    without advancing, so cost per meter is not large there, it is undefined.
    """
    slip = slip_against_slope(
        platform=platform, ground=ground, gravity_m_per_s2=gravity_m_per_s2
    )
    walkable = np.asarray(slip < NO_PROGRESS_SLIP)
    slopes = SLOPE_DEGREES[walkable]
    ratios = slip[walkable]

    swing = swing_work_per_meter(
        platform=platform, gravity_m_per_s2=gravity_m_per_s2, slip_ratio=ratios
    )
    # cost_of_transport takes one swing figure, so the sweep is evaluated
    # pointwise where swing itself varies with slip.
    costs = [
        cost_of_transport(
            mass_kg=platform.total_mass_kg,
            gravity_m_per_s2=gravity_m_per_s2,
            slope_degrees=float(slope),
            slip_ratio=float(ratio),
            patch=platform.contact_patch,
            feet_in_stance=platform.feet_in_stance,
            stride_length_m=platform.stride_length_m,
            stance_length_m=platform.stride_length_m,
            contact_model=ground.contact,
            strength=ground.strength,
            mobilization=ground.mobilization,
            swing_work_per_meter_J=float(value),
        )
        for slope, ratio, value in zip(slopes, ratios, np.atleast_1d(swing.total_J))
    ]
    combined = CostOfTransport(
        gravitational_J_per_m=np.array(
            [float(c.gravitational_J_per_m) for c in costs]
        ),
        shear_J_per_m=np.array([float(c.shear_J_per_m) for c in costs]),
        compaction_J_per_m=np.array([float(c.compaction_J_per_m) for c in costs]),
        swing_J_per_m=np.array([float(c.swing_J_per_m) for c in costs]),
        mass_kg=platform.total_mass_kg,
        gravity_m_per_s2=gravity_m_per_s2,
    )
    return slopes, combined


def cohesion_sweep_kPa(ground: Ground) -> NDArray[np.float64]:
    """Geometric, because the dependence is logarithmic and a linear sweep hides it.

    Swept far below any physical cohesion on purpose. The point of the low end
    is not that regolith might be there; it is to show how many decades of
    cohesion have to be given up before the platform stops, which is the honest
    measure of how much the parameter's uncertainty matters.
    """
    top = ground.strength.cohesion * (1.0 + SIMULANT_GRADE_RELATIVE_SIGMA)
    return np.geomspace(1.0e-4, top, 601)


def slip_on_a_repose_slope(
    *, platform: Platform, ground: Ground, gravity_m_per_s2: float, cohesion_kPa: Any
) -> NDArray[np.float64]:
    """Slip at a slope standing at its own angle of repose, against cohesion.

    The friction angle is the slope angle, so the frictional margin is
    identically zero and this curve is cohesion doing all of the work.
    """
    values = np.atleast_1d(np.asarray(cohesion_kPa, dtype=np.float64))
    return np.array(
        [
            float(
                equilibrium_slip_ratio(
                    platform=platform,
                    strength=MohrCoulombModel(
                        cohesion=float(value),
                        friction_angle_degrees=REPRESENTATIVE_REPOSE_DEGREES,
                    ),
                    mobilization=ground.mobilization,
                    gravity_m_per_s2=gravity_m_per_s2,
                    slope_degrees=REPRESENTATIVE_REPOSE_DEGREES,
                )
            )
            for value in values
        ]
    )


def _slope_reaching_slip(
    platform: Platform, ground: Ground, target: float
) -> float:
    """The slope at which slip first reaches a value, at lunar gravity.

    Read off the swept curve rather than solved, because the curve is what the
    figure draws and a caption that disagreed with its own plot would be worse
    than an imprecise one.
    """
    slip = slip_against_slope(
        platform=platform, ground=ground, gravity_m_per_s2=LUNAR_GRAVITY
    )
    reached = np.flatnonzero(slip >= target)
    if reached.size == 0:
        return float(SLOPE_DEGREES[-1])
    return float(SLOPE_DEGREES[reached[0]])


def build_slope_figure(platform: Platform, ground: Ground) -> Figure:
    limit = maximum_traversable_slope_degrees(
        platform=platform, strength=ground.strength, gravity_m_per_s2=LUNAR_GRAVITY
    )
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
                    "figure.subplot.top": 0.695,
                    "figure.subplot.bottom": 0.205,
                    "figure.subplot.left": 0.062,
                    "figure.subplot.right": 0.986,
                    "figure.subplot.wspace": 0.215,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 2, squeeze=False)

        left = axes[0][0]
        for order, (name, gravity) in enumerate(GRAVITIES):
            slip = slip_against_slope(
                platform=platform, ground=ground, gravity_m_per_s2=gravity
            )
            usable = np.asarray(slip < NO_PROGRESS_SLIP)
            left.plot(
                SLOPE_DEGREES[usable],
                slip[usable],
                color=ACCENT_PRIMARY,
                linewidth=1.5,
                alpha=0.35 + 0.65 * order / (len(GRAVITIES) - 1),
                label=f"{name}, {gravity:.2f} m/s²",
            )
        left.axhline(
            DAY_THREE_SLIP_RATIO,
            color=INK_MUTED,
            linewidth=0.9,
            linestyle=(0, (4, 3)),
        )
        left.annotate(
            f"Day 3 assumed {DAY_THREE_SLIP_RATIO:.2f} at every slope",
            xy=(1.0, DAY_THREE_SLIP_RATIO),
            xytext=(4, 4),
            textcoords="offset points",
            color=INK_SECONDARY,
            fontsize=7.6,
        )
        left.set_title(
            "slip ratio against slope, three gravities", color=INK_SECONDARY, loc="left"
        )
        left.set_ylabel("slip ratio")
        left.set_ylim(0.0, 0.45)
        left.legend(loc="upper left")

        right = axes[0][1]
        slopes, costs = cost_against_slope(
            platform=platform, ground=ground, gravity_m_per_s2=LUNAR_GRAVITY
        )
        right.plot(
            slopes, costs.dimensionless, color=ACCENT_SECONDARY, linewidth=1.6
        )
        right.set_title(
            "cost of transport against slope, lunar gravity",
            color=INK_SECONDARY,
            loc="left",
        )
        right.set_ylabel("cost of transport (dimensionless)")
        right.set_ylim(0.0, None)

        for panel in (left, right):
            panel.axvspan(
                *REPOSE_BAND_DEGREES, color=ACCENT_SECONDARY, alpha=0.10, linewidth=0.0
            )
            panel.axvline(
                limit, color=INK_MUTED, linewidth=0.9, linestyle=(0, (2, 2))
            )
            panel.set_xlabel("slope (degrees)")
            panel.set_xlim(0.0, SLOPE_DEGREES[-1])
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)
            panel.annotate(
                "repose band",
                xy=(sum(REPOSE_BAND_DEGREES) / 2.0, 0.97),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                color=INK_SECONDARY,
                fontsize=7.6,
            )
            panel.annotate(
                f"traction limit {limit:.1f}°",
                xy=(limit, 0.22),
                xycoords=("data", "axes fraction"),
                xytext=(-4, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=90.0,
                color=INK_MUTED,
                fontsize=7.4,
            )

        figure.suptitle(
            "The slope fails before the foot does, so the repose band is where "
            "walking actually happens",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.062,
            ha="left",
            y=0.955,
        )
        figure.text(
            0.062,
            0.895,
            caption(
                "Slip is an output here, not an input: it is the slide that "
                "develops exactly the traction the slope demands. It stays "
                f"below Day 3's assumed {DAY_THREE_SLIP_RATIO:.2f} until "
                f"{_slope_reaching_slip(platform, ground, DAY_THREE_SLIP_RATIO):.0f} "
                "degrees, which is already past the repose band.\n"
                "Lunar regolith, intercrater, at the published 0-15 cm friction "
                "angle. Slip on level ground is exactly zero because level "
                "ground demands no traction; real walking slips there through "
                "mechanisms not modelled, so the left end is a lower bound.",
                width=148,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def build_cohesion_figure(platform: Platform, ground: Ground) -> Figure:
    cohesion = cohesion_sweep_kPa(ground)
    published_low, published_high = ground.cohesion_range_kPa
    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (9.6, 5.4),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.665,
                    "figure.subplot.bottom": 0.185,
                    "figure.subplot.left": 0.090,
                    "figure.subplot.right": 0.975,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 1, squeeze=False)
        panel = axes[0][0]

        for order, (name, gravity) in enumerate(GRAVITIES):
            slip = slip_on_a_repose_slope(
                platform=platform,
                ground=ground,
                gravity_m_per_s2=gravity,
                cohesion_kPa=cohesion,
            )
            drawable = np.asarray(np.isfinite(slip) & (slip < NO_PROGRESS_SLIP))
            panel.plot(
                cohesion[drawable],
                slip[drawable],
                color=ACCENT_PRIMARY,
                linewidth=1.5,
                alpha=0.35 + 0.65 * order / (len(GRAVITIES) - 1),
                label=f"{name}, {gravity:.2f} m/s²",
            )

        simulant_high = float(cohesion[-1])
        panel.axvspan(
            ground.strength.cohesion * 0.25,
            simulant_high,
            color=INK_MUTED,
            alpha=0.12,
            linewidth=0.0,
            label="simulant-grade uncertainty, ±300%",
        )
        panel.axvspan(
            published_low,
            published_high,
            color=ACCENT_SECONDARY,
            alpha=0.30,
            linewidth=0.0,
            label="lunar in situ, Table 9.12, ±17%",
        )

        panel.set_xscale("log")
        panel.set_xlabel("cohesion (kPa)")
        panel.set_ylabel(
            f"slip ratio on a {REPRESENTATIVE_REPOSE_DEGREES:.0f}° repose slope"
        )
        panel.set_xlim(float(cohesion[0]), simulant_high)
        panel.set_ylim(0.0, 0.85)
        panel.legend(loc="lower left")
        panel.spines["top"].set_visible(False)
        panel.spines["right"].set_visible(False)

        published_slip = float(
            slip_on_a_repose_slope(
                platform=platform,
                ground=ground,
                gravity_m_per_s2=LUNAR_GRAVITY,
                cohesion_kPa=ground.strength.cohesion,
            )[0]
        )
        panel.annotate(
            f"slip reaches one near 1e-7 kPa,\nsix decades below the measured "
            f"{ground.strength.cohesion} kPa",
            xy=(float(cohesion[0]) * 1.6, 0.80),
            ha="left",
            va="top",
            color=INK_SECONDARY,
            fontsize=7.6,
        )
        panel.annotate(
            f"published: slip {published_slip:.3f}",
            xy=(ground.strength.cohesion, published_slip),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            color=INK_PRIMARY,
            fontsize=7.8,
        )

        figure.suptitle(
            "The whole traction margin is cohesive, yet slip moves only 1.8× "
            "per decade of cohesion",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.090,
            ha="left",
            y=0.960,
        )
        figure.text(
            0.090,
            0.905,
            caption(
                "A slope standing at theta has mobilized its friction angle "
                "exactly, so cos(theta)tan(phi) − sin(theta) vanishes and the "
                "frictional term contributes nothing to the margin. The whole "
                "reserve is cohesive, at any gravity.\n"
                "But Janosi-Hanamoto is exponential in slide, so inverting it "
                "wraps the demand in a logarithm. Cohesion being the entire "
                "margin and cohesion being poorly known are therefore both true "
                "and jointly tolerable: the ±17% in-situ range moves slip by "
                "±6%, and even simulant-grade uncertainty spans less than a "
                "factor of three.",
                width=140,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def build_swing_figure(platform: Platform) -> Figure:
    inertial = np.array(
        [
            float(
                swing_work_per_meter(
                    platform=platform, gravity_m_per_s2=float(g), slip_ratio=0.0
                ).inertial_J
            )
            for g in GRAVITY_SWEEP
        ]
    )
    clearance = np.array(
        [
            float(
                swing_work_per_meter(
                    platform=platform, gravity_m_per_s2=float(g), slip_ratio=0.0
                ).clearance_J
            )
            for g in GRAVITY_SWEEP
        ]
    )
    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (9.2, 5.0),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.700,
                    "figure.subplot.bottom": 0.190,
                    "figure.subplot.left": 0.086,
                    "figure.subplot.right": 0.975,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 1, squeeze=False)
        panel = axes[0][0]

        panel.stackplot(
            GRAVITY_SWEEP,
            inertial,
            clearance,
            colors=[INK_PRIMARY, ACCENT_SECONDARY],
            labels=["inertial, gravity-independent", "clearance, scales with gravity"],
            edgecolor="none",
            alpha=0.9,
        )
        panel.axhline(
            DAY_THREE_SWING_J_PER_METER,
            color=ACCENT_PRIMARY,
            linewidth=1.2,
            linestyle=(0, (4, 3)),
        )
        panel.annotate(
            f"Day 3 assumed {DAY_THREE_SWING_J_PER_METER:.0f} J/m",
            xy=(GRAVITY_SWEEP[-1], DAY_THREE_SWING_J_PER_METER),
            xytext=(-4, 5),
            textcoords="offset points",
            ha="right",
            color=ACCENT_PRIMARY,
            fontsize=7.8,
        )
        for name, gravity in GRAVITIES:
            panel.axvline(
                gravity, color=INK_MUTED, linewidth=0.7, linestyle=(0, (2, 3))
            )

        panel.set_xlabel("gravity (m/s²)")
        panel.set_ylabel("swing work (J per meter)")
        panel.set_xlim(GRAVITY_SWEEP[0], GRAVITY_SWEEP[-1])
        panel.set_ylim(0.0, 28.0)
        panel.set_xticks([g for _, g in GRAVITIES])
        panel.set_xticklabels([f"{g:.2f}" for _, g in GRAVITIES])
        panel.minorticks_off()
        panel.legend(loc="upper left")
        panel.spines["top"].set_visible(False)
        panel.spines["right"].set_visible(False)

        figure.suptitle(
            "Swing cost barely falls under reduced gravity, because almost all "
            "of it is inertia",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.086,
            ha="left",
            y=0.955,
        )
        figure.text(
            0.086,
            0.895,
            caption(
                "Computed from the nominal quadruped rather than assumed. The "
                "inertial term is identical at every gravity; only foot "
                "clearance scales, and at one sixth g it nearly vanishes.\n"
                "The total lands below Day 3's assumed value at every gravity, "
                "so that crossover claim was conservative. The inertial term "
                "counts the positive mechanical work of one acceleration phase; "
                "a drive with no regeneration would roughly double it.",
                width=136,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def boundary_rows(platform: Platform, ground: Ground) -> tuple[BoundaryRow, ...]:
    low, high = ground.cohesion_range_kPa
    return (
        BoundaryRow(
            quantity="cohesion, lunar in situ",
            published_range=f"{low} to {high} kPa",
            used=f"{ground.strength.cohesion} kPa, swept over four decades",
            status=INSIDE,
            basis=(
                "Heiken et al. (1991) Table 9.12, 327 in-situ tests; it is the "
                "entire traction margin on a repose slope but slip depends on "
                "it only logarithmically, so the published range moves slip 6%"
            ),
        ),
        BoundaryRow(
            quantity="friction angle on a repose slope",
            published_range="none at the surface; 42 degrees over 0-15 cm",
            used=f"{REPRESENTATIVE_REPOSE_DEGREES} degrees, the slope angle itself",
            status=UNMEASURED,
            basis=(
                "a slope standing at theta has mobilized theta whatever its "
                "density; the surface density gap is the soil file's "
                "surface_density_unresolved anomaly"
            ),
        ),
        BoundaryRow(
            quantity="angle of repose",
            published_range="none transcribed in this repository",
            used=(
                f"{REPOSE_BAND_DEGREES[0]:.0f} to {REPOSE_BAND_DEGREES[1]:.0f} "
                "degrees, carried as a band"
            ),
            status=UNMEASURED,
            basis="stated range, not a transcribed parameter; annotation only",
        ),
        BoundaryRow(
            quantity="leg inertia",
            published_range="none",
            used=(
                f"uniform rod, {platform.leg_mass_kg} kg over "
                f"{platform.leg_length_m} m"
            ),
            status=UNMEASURED,
            basis=(
                "assumed; a real leg puts more mass proximally, so this "
                "overestimates the inertial term"
            ),
        ),
        BoundaryRow(
            quantity="walking speed",
            published_range="none",
            used=f"{platform.nominal_speed_m_per_s} m/s",
            status=UNMEASURED,
            basis=(
                "an operating condition, not a platform property; swing cost per "
                "meter goes as its square while the soil terms do not depend on "
                "it, so this model always prefers walking slower and a real one "
                "does not"
            ),
        ),
        BoundaryRow(
            quantity="stance regime",
            published_range="none",
            used="quasi-static, feet planted",
            status=UNMEASURED,
            basis=(
                "gaits with a flight phase violate it and the traction balance "
                "does not apply to them; not detected, so do not apply this to "
                "a bounding or galloping gait"
            ),
        ),
        BoundaryRow(
            quantity="slip on level ground",
            published_range="none",
            used="zero, as the balance gives it",
            status=UNMEASURED,
            basis=(
                "level ground demands no tangential force; real slip there comes "
                "from within-stride acceleration, control error and foot "
                "placement, none modelled, so this is a lower bound"
            ),
        ),
        BoundaryRow(
            quantity="shear mobilization under gait",
            published_range="none",
            used=(
                "K = "
                f"{ground.mobilization.shear_deformation_modulus * MILLIMETERS_PER_METER:.1f}"
                " mm per footfall"
            ),
            status=UNMEASURED,
            basis=(
                "K now propagates into slip and from there into every term; its "
                "sensitivity is swept in the report rather than stated"
            ),
        ),
        BoundaryRow(
            quantity="compaction as tangential resistance",
            published_range="not applicable",
            used="excluded from the traction balance",
            status=OUTSIDE,
            basis=(
                "a wheel makes new rut and pays in forward motion; a placed foot "
                "pays that work downward, so a vehicle-derived resistance term "
                "would inflate demand and understate every slope"
            ),
        ),
    )


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    return repr(float(value))


def build_report(definition: PlatformDefinition, ground: Ground) -> str:
    platform = definition.platform
    rows = boundary_rows(platform, ground)
    limit = maximum_traversable_slope_degrees(
        platform=platform, strength=ground.strength, gravity_m_per_s2=LUNAR_GRAVITY
    )
    low, high = ground.cohesion_range_kPa

    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# What a slope costs a stated platform, and what decides whether it can",
        "# be walked at all.",
        "#",
        "# Generated by studies/mobility/traversability.py. Do not edit.",
        "#",
        "# No timestamp and no sampling: closed-form throughout, so re-running",
        "# leaves this byte-identical.",
        "#",
        "# Not a sortie envelope. No power, no thermal, no illumination, no",
        "# terrain.",
        "",
        f"schema_version = {REPORT_SCHEMA_VERSION}",
        "",
        "[environment]",
        f'python = "{host_platform.python_version()}"',
        f'numpy = "{np.__version__}"',
        "",
        "[platform]",
        f'id = "{definition.id}"',
        f'source = "{definition.source_path.relative_to(REPOSITORY_ROOT)}"',
        f'basis = "{definition.basis}"',
        f"total_mass_kg = {_format_float(platform.total_mass_kg)}",
        f"duty_factor = {_format_float(platform.duty_factor)}",
        f"hip_sweep_degrees = {_format_float(math.degrees(platform.hip_sweep_radians))}",
        f"swing_duration_s = {_format_float(platform.swing_duration_s)}",
        "leg_inertia_about_hip_kg_m2 = "
        f"{_format_float(platform.leg_inertia_about_hip_kg_m2)}",
        "",
        "# Day 3 assumed both of these. They are outputs now, and the report",
        "# states the assumed value beside the computed one so the size and the",
        "# direction of the earlier error are both visible.",
        "[superseded_assumption]",
        f"day_three_slip_ratio = {_format_float(DAY_THREE_SLIP_RATIO)}",
        "day_three_swing_work_J_per_m = "
        f"{_format_float(DAY_THREE_SWING_J_PER_METER)}",
        "",
    ]

    for name, gravity in GRAVITIES:
        per_stride = swing_work_per_stride(
            platform=platform, gravity_m_per_s2=gravity
        )
        per_meter = swing_work_per_meter(
            platform=platform, gravity_m_per_s2=gravity, slip_ratio=0.0
        )
        lines += [
            "[[swing]]",
            f'gravity = "{name}"',
            f"gravity_m_per_s2 = {_format_float(gravity)}",
            f"inertial_J_per_stride = {_format_float(float(per_stride.inertial_J))}",
            f"clearance_J_per_stride = {_format_float(float(per_stride.clearance_J))}",
            f"inertial_J_per_m = {_format_float(float(per_meter.inertial_J))}",
            f"clearance_J_per_m = {_format_float(float(per_meter.clearance_J))}",
            f"total_J_per_m = {_format_float(float(per_meter.total_J))}",
            "inertial_fraction = "
            f"{_format_float(float(per_meter.inertial_fraction))}",
            "ratio_to_day_three_assumption = "
            f"{_format_float(float(per_meter.total_J) / DAY_THREE_SWING_J_PER_METER)}",
            "",
        ]

    lines += [
        "# The foot-slip limit, and why it is not the binding constraint. It",
        "# lands above the repose band at every gravity and every foot size",
        "# swept, so a slope that stands is always walkable and the limit is a",
        "# bound on something that does not occur.",
        "[repose_band]",
        f"minimum_degrees = {_format_float(REPOSE_BAND_DEGREES[0])}",
        f"maximum_degrees = {_format_float(REPOSE_BAND_DEGREES[1])}",
        'basis = "stated range, not a transcribed parameter"',
        "",
    ]

    for name, gravity in GRAVITIES:
        for half_width in FOOT_HALF_WIDTHS_M:
            for feet in (2, 4):
                candidate = variant(
                    platform,
                    foot_half_width_m=half_width,
                    foot_contact_area_m2=math.pi * half_width**2,
                    feet_in_stance=feet,
                )
                lines += [
                    "[[traction_limit]]",
                    f'gravity = "{name}"',
                    f"foot_half_width_m = {_format_float(half_width)}",
                    f"feet_in_stance = {feet}",
                    "maximum_traversable_slope_degrees = "
                    + _format_float(
                        maximum_traversable_slope_degrees(
                            platform=candidate,
                            strength=ground.strength,
                            gravity_m_per_s2=gravity,
                        )
                    ),
                    "",
                ]

    lines += [
        "# Slip as an output, at lunar gravity, against the flat 0.10 Day 3",
        "# assumed at every slope.",
        "",
    ]
    for slope in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 32.0, 35.0, 40.0, 43.0):
        slip = float(
            equilibrium_slip_ratio(
                platform=platform,
                strength=ground.strength,
                mobilization=ground.mobilization,
                gravity_m_per_s2=LUNAR_GRAVITY,
                slope_degrees=slope,
            )
        )
        lines += [
            "[[slip_against_slope]]",
            f"slope_degrees = {_format_float(slope)}",
            f"slip_ratio = {_format_float(slip)}",
            f"walkable = {str(slip < NO_PROGRESS_SLIP).lower()}",
            "",
        ]

    lines += [
        "# The centre of this study. On a slope standing at its angle of repose",
        "# the frictional term contributes nothing to the margin, so cohesion is",
        "# the entire reserve -- cohesive_share_of_margin is one to fourteen",
        "# digits in every case below with non-zero cohesion.",
        "#",
        "# And yet slip depends on it only logarithmically, because inverting an",
        "# exponential mobilization law wraps the demand in a logarithm. Losing a",
        "# decade of cohesion costs a factor of 1.8 in slip. The zero case is the",
        "# mathematical limit and not a soil: slip reaches one near 1e-7 kPa.",
        "[repose_margin]",
        f"slope_degrees = {_format_float(REPRESENTATIVE_REPOSE_DEGREES)}",
        'friction_angle_basis = "the slope angle itself"',
        f"published_cohesion_kPa = {_format_float(ground.strength.cohesion)}",
        f"published_cohesion_range_kPa = [{_format_float(low)}, "
        f"{_format_float(high)}]",
        "",
    ]

    representative = (
        ("lunar_low", low),
        ("lunar_published", ground.strength.cohesion),
        ("lunar_high", high),
        ("mathematical_zero", 0.0),
        ("one_decade_below_published", ground.strength.cohesion * 0.1),
        ("simulant_low", ground.strength.cohesion * 0.25),
        ("simulant_high", ground.strength.cohesion * (1.0 + SIMULANT_GRADE_RELATIVE_SIGMA)),
    )
    for label, value in representative:
        strength = MohrCoulombModel(
            cohesion=value, friction_angle_degrees=REPRESENTATIVE_REPOSE_DEGREES
        )
        balance = traction_balance(
            platform=platform,
            strength=strength,
            gravity_m_per_s2=LUNAR_GRAVITY,
            slope_degrees=REPRESENTATIVE_REPOSE_DEGREES,
        )
        slip = float(
            slip_on_a_repose_slope(
                platform=platform,
                ground=ground,
                gravity_m_per_s2=LUNAR_GRAVITY,
                cohesion_kPa=value,
            )[0]
        )
        lines += [
            "[[repose_margin.case]]",
            f'id = "{label}"',
            f"cohesion_kPa = {_format_float(value)}",
            f"margin_N = {_format_float(float(balance.margin_N))}",
            "cohesive_share_of_margin = "
            f"{_format_float(float(balance.cohesive_share_of_margin))}",
            f"slip_ratio = {_format_float(slip)}",
            f"walkable = {str(slip < NO_PROGRESS_SLIP).lower()}",
            "",
        ]

    lines += [
        "# K under gait is unmeasured and now propagates into slip, so its",
        "# sensitivity is swept rather than stated. Slip is very nearly linear",
        "# in K at fixed demand, since the mobilized fraction does not depend on",
        "# K at all -- only the slide needed to reach it does.",
        "",
    ]
    for factor in (0.5, 1.0, 2.0, 4.0):
        modulus = ground.mobilization.shear_deformation_modulus * factor
        slip = float(
            equilibrium_slip_ratio(
                platform=platform,
                strength=ground.strength,
                mobilization=JanosiHanamotoModel(shear_deformation_modulus=modulus),
                gravity_m_per_s2=LUNAR_GRAVITY,
                slope_degrees=REPRESENTATIVE_REPOSE_DEGREES,
            )
        )
        lines += [
            "[[mobilization_sensitivity]]",
            f"factor_on_K = {_format_float(factor)}",
            f"shear_deformation_modulus_m = {_format_float(modulus)}",
            f"slope_degrees = {_format_float(REPRESENTATIVE_REPOSE_DEGREES)}",
            f"slip_ratio = {_format_float(slip)}",
            f"walkable = {str(slip < NO_PROGRESS_SLIP).lower()}",
            "",
        ]

    lines += [
        "# The measured-versus-extrapolated boundary for this study. Counts are",
        "# generated, not written.",
        f"# {tally(rows)}",
        "",
        *toml_lines(rows),
        "[traction_limit_summary]",
        f"lunar_degrees = {_format_float(limit)}",
        "exceeds_repose_band = "
        + str(limit > REPOSE_BAND_DEGREES[1]).lower(),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute slip and swing from a platform, and state what decides "
            "whether a slope can be walked."
        )
    )
    parser.add_argument("--figure-directory", type=Path, default=FIGURE_DIRECTORY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--platform", type=Path, default=PLATFORM_PATH)
    arguments = parser.parse_args(argv)

    definition = load_platform(arguments.platform)
    ground = load_ground()

    arguments.figure_directory.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)

    for name, figure in (
        ("slip-and-cost-against-slope", build_slope_figure(definition.platform, ground)),
        (
            "cohesion-margin-on-a-repose-slope",
            build_cohesion_figure(definition.platform, ground),
        ),
        ("swing-decomposition", build_swing_figure(definition.platform)),
    ):
        path = arguments.figure_directory / f"{name}.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        print(f"wrote {path.relative_to(REPOSITORY_ROOT)}")

    arguments.report.write_text(build_report(definition, ground), encoding="utf-8")
    print(f"wrote {arguments.report.relative_to(REPOSITORY_ROOT)}")

    print("\n  measured against extrapolated\n")
    print(text_table(boundary_rows(definition.platform, ground)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
