# SPDX-License-Identifier: Apache-2.0
#
# studies.sortie.energy_envelope — the locomotion energy of a real route, and
# whether it is the term that matters.
#
# Every rung has deferred this and the deferral has run out. The inputs now
# exist: contact physics calibrated at foot scale, cost of transport as a
# function of slope and soil and gait, achievable slope from two independent
# failure modes, and a measured surface to walk over. What comes out is the
# first number in this project with a journey in it.
#
# It is not a sortie envelope, and the temptation to call it one is strongest
# today because the shape finally looks like a mission. A sortie needs six axes.
# One is populated. Illumination, thermal, power, comms and cold-trap range are
# declared and empty, every one of them would reduce reachable depth or bound
# the schedule, and so every number here is an upper bound.
#
# The route is a transect, not a plan: a straight line sampled from the grid,
# rim crest to the lowest point in the window. A planner would optimise against
# the same cost model being measured and confound the two, and it could only
# make the number smaller.
#
# Three results.
#
# The asymmetry is large and it is not the two that intuition offers. Descending,
# gravity does negative work while shear, compaction and swing stay positive and
# indifferent to direction; climbing, everything is positive. The return leg
# costs nearly four times the outbound and the round trip costs about two and a
# half times the outbound rather than twice. Nineteen percent of the descent is
# free, in the sense that gravity more than pays for the ground -- and free is
# where it stops, because a leg without regeneration dissipates the surplus
# rather than banking it.
#
# The envelope is shallow in the useful sense: about a third of a watt-hour per
# metre of depth, roughly flat across the range, so reachable depth scales with
# battery almost linearly. That is the platform-agnostic form and it is the same
# move as the swing crossover, reporting a rate rather than a single depth that
# depends on a capacity nobody has specified.
#
# And the third result reorders the roadmap. Spread over the time the route
# takes, locomotion averages tens of watts -- comparable to the housekeeping
# loads of a small robot, and small against what a forty-kelvin environment
# plausibly demands. Locomotion is not the binding term. This project has spent
# seven days on the term that is not binding, which was worth doing because it
# is the term that had to be shown adequate, but rung five is thermal.
#
# On the crossover rather than a citation: no survival-heating figure is
# transcribed in this repository, so none is quoted. What is reported is the
# continuous power at which a subsystem equals locomotion, which is a property
# of this route and this platform and needs no source.
#
# References
#   Carrier WD III, Olhoeft GR, Mendell W (1991) Physical Properties of the
#     Lunar Surface. In: Lunar Sourcebook, ch. 9. Cambridge University Press.

from __future__ import annotations

import argparse
import math
import platform as host_platform
import textwrap
import tomllib
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
from eclipse.io.platform import load_platform
from eclipse.io.soil import janosi_hanamoto_model, load_soil, mohr_coulomb_model
from eclipse.io.terrain import GeoRaster, read_float_geotiff
from eclipse.platform import Platform
from eclipse.sortie import (
    JOULES_PER_WATT_HOUR,
    RoundTrip,
    Transect,
    sample_transect,
    walk_round_trip,
)
from eclipse.stance import wave_gait, within_stride_slip_ratio
from eclipse.terramechanics import ContactModel, JanosiHanamotoModel, MohrCoulombModel

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
ELEVATION_PATH: Final = (
    REPOSITORY_ROOT / "data" / "terrain" / "SL2_final_adj_5mpp_surf.tif"
)
SITE_PATH: Final = REPOSITORY_ROOT / "configs" / "sites" / "de-gerlache-rim-2.toml"
PLATFORM_PATH: Final = (
    REPOSITORY_ROOT / "configs" / "platforms" / "nominal-quadruped.toml"
)
SOIL_PATH: Final = REPOSITORY_ROOT / "data" / "soils" / "lunar-intercrater.toml"
FIGURE_DIRECTORY: Final = Path(__file__).resolve().parent / "figures"
DEFAULT_REPORT_PATH: Final = (
    Path(__file__).resolve().parent / "results" / "energy-envelope.toml"
)

REPORT_SCHEMA_VERSION: Final = 1
LUNAR_GRAVITY: Final = 1.62
SHALLOWEST_DEPTH_RANGE: Final = "0-15"
TRANSECT_SAMPLES: Final = 1200
FEET_IN_STANCE: Final = 3
CRAWL_LIFT_ORDER: Final = (2, 0, 3, 1)
CRAWL_DUTY: Final = 0.75

# Mechanical work is not battery draw. Named rather than folded in, and swept
# because the multiplier is a design property nobody here has specified.
DERATING_RANGE: Final = (3.0, 5.0)
NOMINAL_DERATING: Final = 4.0

BATTERY_SWEEP_WH: Final[NDArray[np.float64]] = np.linspace(20.0, 1400.0, 70)
SPEED_SWEEP: Final[NDArray[np.float64]] = np.linspace(0.10, 0.68, 60)
HOUSEKEEPING_SWEEP_W: Final[NDArray[np.float64]] = np.linspace(0.0, 200.0, 201)
NOMINAL_SPEED: Final = 0.25


def caption(text: str, width: int = 148) -> str:
    return "\n".join(
        textwrap.fill(" ".join(paragraph.split()), width=width)
        for paragraph in text.split("\n")
    )


@dataclass(frozen=True, slots=True)
class Ground:
    contact: ContactModel
    strength: MohrCoulombModel
    mobilization: JanosiHanamotoModel


def load_ground() -> Ground:
    dataset = load_soil(SOIL_PATH).datasets["carrier1991"]
    return Ground(
        contact=dataset.models["bekker"].extrapolating,
        strength=mohr_coulomb_model(dataset, depth_range_cm=SHALLOWEST_DEPTH_RANGE),
        mobilization=janosi_hanamoto_model(dataset),
    )


def build_transect(elevation: GeoRaster) -> Transect:
    """Rim crest to the lowest point in the window, straight.

    Chosen from the data rather than from a coordinate, because the Artemis
    region's own centre has no citable source. The destination is the deepest
    ground the product covers; whether it is in permanent shadow is the
    illumination axis, which is declared and empty.
    """
    highest = np.unravel_index(int(np.argmax(elevation.values)), elevation.values.shape)
    lowest = np.unravel_index(int(np.argmin(elevation.values)), elevation.values.shape)
    return sample_transect(
        elevation,
        start_row_column=(int(highest[0]), int(highest[1])),
        end_row_column=(int(lowest[0]), int(lowest[1])),
        samples=TRANSECT_SAMPLES,
    )


def level_ground_slip(platform: Platform, ground: Ground) -> float:
    slip, _ = within_stride_slip_ratio(
        platform=platform,
        gait=wave_gait(lift_order=CRAWL_LIFT_ORDER, duty_factor=CRAWL_DUTY),
        strength=ground.strength,
        mobilization=ground.mobilization,
        gravity_m_per_s2=LUNAR_GRAVITY,
    )
    return slip


def reachable_depth_m(
    trip: RoundTrip, transect: Transect, *, battery_wh: float, derating: float
) -> float:
    index = trip.reachable_index(
        battery_J=battery_wh * JOULES_PER_WATT_HOUR, derating=derating
    )
    return float(transect.elevation_m[0] - transect.elevation_m[index])


def sortie_hours(trip: RoundTrip, speed_m_per_s: float) -> float:
    distance = trip.outbound.distance_m + trip.inbound.distance_m
    return distance / speed_m_per_s / 3600.0


def average_locomotion_power_w(
    trip: RoundTrip, *, speed_m_per_s: float, derating: float
) -> float:
    return (
        trip.total_J
        * derating
        / JOULES_PER_WATT_HOUR
        / sortie_hours(trip, speed_m_per_s)
    )


def build_route_figure(transect: Transect, trip: RoundTrip) -> Figure:
    distance_km = transect.distance_m / 1000.0
    midpoints = 0.5 * (distance_km[:-1] + distance_km[1:])
    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (10.2, 6.6),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.700,
                    "figure.subplot.bottom": 0.160,
                    "figure.subplot.left": 0.086,
                    "figure.subplot.right": 0.908,
                    "figure.subplot.hspace": 0.400,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(2, 1, squeeze=False, sharex=True)

        upper = axes[0][0]
        upper.plot(
            distance_km, transect.elevation_m, color=INK_PRIMARY, linewidth=1.6
        )
        upper.fill_between(
            distance_km,
            transect.elevation_m.min(),
            transect.elevation_m,
            color=INK_MUTED,
            alpha=0.16,
            linewidth=0.0,
        )
        upper.set_ylabel("elevation (m)")
        upper.set_title(
            "the route: rim crest to the deepest ground in the window",
            color=INK_SECONDARY,
            loc="left",
        )
        slope_axis = upper.twinx()
        slope_axis.plot(
            midpoints,
            transect.slope_degrees,
            color=ACCENT_SECONDARY,
            linewidth=0.55,
            alpha=0.45,
        )
        slope_axis.axhline(0.0, color=INK_MUTED, linewidth=0.7)
        slope_axis.set_ylabel("slope (°)", color=ACCENT_SECONDARY)
        slope_axis.tick_params(axis="y", colors=ACCENT_SECONDARY)
        slope_axis.grid(False)

        lower = axes[1][0]
        for leg, label, colour, style in (
            (trip.outbound, "outbound, descending", ACCENT_PRIMARY, "solid"),
            (trip.inbound, "return, climbing", ACCENT_SECONDARY, "solid"),
        ):
            walked = np.concatenate([[0.0], np.cumsum(leg.segment_length_m)]) / 1000.0
            energy = np.concatenate([[0.0], leg.cumulative_J]) / JOULES_PER_WATT_HOUR
            lower.plot(walked, energy, color=colour, linewidth=1.7, linestyle=style,
                       label=f"{label} — {leg.total_J / JOULES_PER_WATT_HOUR:.1f} Wh")
        lower.set_xlabel("distance walked (km)")
        lower.set_ylabel("cumulative energy (Wh)")
        lower.set_title(
            "energy accumulated, each direction from its own start",
            color=INK_SECONDARY,
            loc="left",
        )
        lower.legend(loc="upper left")

        for panel in (upper, lower):
            panel.spines["top"].set_visible(False)
        lower.spines["right"].set_visible(False)

        figure.suptitle(
            f"The return leg costs {trip.asymmetry:.1f} times the outbound, and "
            f"the round trip {trip.over_twice_outbound:.2f} times the outbound",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.086,
            ha="left",
            y=0.962,
        )
        figure.text(
            0.086,
            0.912,
            caption(
                f"{transect.distance_m[-1] / 1000:.1f} km of horizontal distance and "
                f"{transect.descent_m:.0f} m of descent, sampled from LOLA at 5 m. "
                "Slip is the larger of what each slope demands and what the "
                "crawl demands on the level.\n"
                "Gravity does negative work going down while shear, compaction "
                "and swing stay positive and indifferent to direction, so the "
                f"legs are not mirror images. {trip.outbound.free_fraction:.0%} of the "
                "descent is free — gravity more than paying for the ground — and "
                "free is where it stops, since a leg without regeneration "
                "dissipates the surplus rather than banking it.",
                width=150,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def build_envelope_figure(transect: Transect, trip: RoundTrip) -> Figure:
    low, high = DERATING_RANGE
    depths = {
        derating: np.asarray(
            [
                reachable_depth_m(
                    trip, transect, battery_wh=float(wh), derating=derating
                )
                for wh in BATTERY_SWEEP_WH
            ]
        )
        for derating in (low, NOMINAL_DERATING, high)
    }
    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (9.6, 5.6),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.635,
                    "figure.subplot.bottom": 0.185,
                    "figure.subplot.left": 0.096,
                    "figure.subplot.right": 0.975,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 1, squeeze=False)
        panel = axes[0][0]

        panel.fill_between(
            BATTERY_SWEEP_WH,
            depths[high],
            depths[low],
            color=ACCENT_PRIMARY,
            alpha=0.16,
            linewidth=0.0,
            label=f"actuator and avionics derating {low:.0f}× to {high:.0f}×",
        )
        panel.plot(
            BATTERY_SWEEP_WH,
            depths[NOMINAL_DERATING],
            color=ACCENT_PRIMARY,
            linewidth=1.8,
            label=f"nominal {NOMINAL_DERATING:.0f}× derating",
        )
        panel.axhline(
            transect.descent_m, color=INK_MUTED, linewidth=0.9, linestyle=(0, (3, 2))
        )
        panel.annotate(
            f"the route ends here, {transect.descent_m:.0f} m",
            xy=(BATTERY_SWEEP_WH[0], transect.descent_m),
            xytext=(6, -12),
            textcoords="offset points",
            color=INK_SECONDARY,
            fontsize=7.8,
        )

        for offset, label in (
            (0.78, "survival heating: absent, would reduce depth"),
            (0.66, "active illumination: absent, would reduce depth"),
            (0.54, "comms blackout below the rim: an autonomy requirement, not an energy one"),
            (0.42, "charge duty cycle: absent, sets sorties per week rather than depth"),
        ):
            panel.annotate(
                label,
                xy=(0.055, offset),
                xycoords="axes fraction",
                ha="left",
                va="center",
                color=INK_SECONDARY,
                fontsize=7.8,
            )
        panel.annotate(
            "",
            xy=(0.035, 0.36),
            xytext=(0.035, 0.84),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "-|>", "color": INK_MUTED, "linewidth": 1.1},
        )

        panel.set_xlabel("battery capacity available to the sortie (Wh)")
        panel.set_ylabel("reachable depth below the rim crest (m)")
        panel.set_xlim(BATTERY_SWEEP_WH[0], BATTERY_SWEEP_WH[-1])
        panel.set_ylim(0.0, transect.descent_m * 1.12)
        panel.legend(loc="lower right")
        panel.spines["top"].set_visible(False)
        panel.spines["right"].set_visible(False)

        per_metre = (
            BATTERY_SWEEP_WH[-1] / max(depths[NOMINAL_DERATING][-1], 1.0)
        )
        figure.suptitle(
            "Reachable depth is about a third of a watt-hour per metre, and "
            "every missing term reduces it",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.096,
            ha="left",
            y=0.958,
        )
        figure.text(
            0.096,
            0.900,
            caption(
                "Depth a platform can reach and still return, against the energy "
                "it starts with. Mechanical work times a derating for actuator "
                "efficiency and avionics, which is a named parameter here rather "
                "than folded in, and swept because no design has fixed it.\n"
                "Every axis this study does not carry pushes the curve down or "
                "bounds the schedule, so this is an upper bound and the arrow "
                "says so. It is also why the useful output is the rate — roughly "
                f"{per_metre:.2f} Wh per metre of depth — rather than a single "
                "number that would depend on a capacity nobody has specified.",
                width=140,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def build_binding_figure(trip: RoundTrip) -> Figure:
    powers = np.asarray(
        [
            average_locomotion_power_w(
                trip, speed_m_per_s=float(v), derating=NOMINAL_DERATING
            )
            for v in SPEED_SWEEP
        ]
    )
    nominal_power = average_locomotion_power_w(
        trip, speed_m_per_s=NOMINAL_SPEED, derating=NOMINAL_DERATING
    )
    hours = sortie_hours(trip, NOMINAL_SPEED)
    locomotion_wh = trip.total_J * NOMINAL_DERATING / JOULES_PER_WATT_HOUR
    share = locomotion_wh / (locomotion_wh + HOUSEKEEPING_SWEEP_W * hours)

    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (10.2, 5.2),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.660,
                    "figure.subplot.bottom": 0.195,
                    "figure.subplot.left": 0.070,
                    "figure.subplot.right": 0.986,
                    "figure.subplot.wspace": 0.235,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 2, squeeze=False)

        left = axes[0][0]
        left.plot(SPEED_SWEEP, powers, color=ACCENT_PRIMARY, linewidth=1.8)
        left.axvline(
            NOMINAL_SPEED, color=INK_MUTED, linewidth=0.8, linestyle=(0, (2, 3))
        )
        left.plot([NOMINAL_SPEED], [nominal_power], marker="o", markersize=5.0,
                  markerfacecolor="none", color=INK_PRIMARY)
        left.annotate(
            f"{nominal_power:.0f} W at {NOMINAL_SPEED:.2f} m/s\n"
            f"over a {hours:.0f} hour sortie",
            xy=(NOMINAL_SPEED, nominal_power),
            xytext=(10, 6),
            textcoords="offset points",
            color=INK_PRIMARY,
            fontsize=8.0,
        )
        left.set_xlabel("walking speed (m/s)")
        left.set_ylabel("average locomotion power over the sortie (W)")
        left.set_title(
            "locomotion, spread over the time the route takes",
            color=INK_SECONDARY,
            loc="left",
        )
        left.set_xlim(SPEED_SWEEP[0], SPEED_SWEEP[-1])
        left.set_ylim(0.0, None)

        right = axes[0][1]
        right.plot(
            HOUSEKEEPING_SWEEP_W, share * 100.0, color=ACCENT_PRIMARY, linewidth=1.8
        )
        right.axhline(50.0, color=INK_MUTED, linewidth=0.9, linestyle=(0, (3, 2)))
        right.plot([nominal_power], [50.0], marker="o", markersize=5.0,
                   markerfacecolor="none", color=INK_PRIMARY)
        right.annotate(
            f"a continuous {nominal_power:.0f} W halves it",
            xy=(nominal_power, 50.0),
            xytext=(10, 10),
            textcoords="offset points",
            color=INK_PRIMARY,
            fontsize=8.0,
        )
        for power in (25.0, 50.0, 100.0):
            index = int(np.searchsorted(HOUSEKEEPING_SWEEP_W, power))
            right.annotate(
                f"{share[index] * 100:.0f}%",
                xy=(power, share[index] * 100.0),
                xytext=(0, -14),
                textcoords="offset points",
                ha="center",
                color=INK_SECONDARY,
                fontsize=7.8,
            )
        right.set_xlabel("continuous non-locomotion power (W)")
        right.set_ylabel("locomotion share of sortie energy (%)")
        right.set_title(
            f"how that share falls, at {NOMINAL_SPEED:.2f} m/s",
            color=INK_SECONDARY,
            loc="left",
        )
        right.set_xlim(0.0, HOUSEKEEPING_SWEEP_W[-1])
        right.set_ylim(0.0, 102.0)

        for panel in (left, right):
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)

        figure.suptitle(
            "Locomotion is not the binding term, and this is the number that "
            "says so",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.070,
            ha="left",
            y=0.958,
        )
        figure.text(
            0.070,
            0.900,
            caption(
                "A route is a distance and a duration, and dividing energy by "
                f"duration gives a power. This one averages {nominal_power:.0f} W, "
                "which is the bar any other subsystem has to clear to matter as "
                "much as walking does.\n"
                "No survival-heating figure is quoted, because none is "
                "transcribed in this repository. What is reported is the "
                "crossover: a continuous load equal to the average locomotion "
                "power halves locomotion's share, and a forty-kelvin environment "
                "plausibly demands more than that. Seven days have gone into the "
                "term that is not binding — which was worth doing, because it is "
                "the term that had to be shown adequate.",
                width=150,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def boundary_rows(trip: RoundTrip, transect: Transect) -> tuple[BoundaryRow, ...]:
    nominal_power = average_locomotion_power_w(
        trip, speed_m_per_s=NOMINAL_SPEED, derating=NOMINAL_DERATING
    )
    return (
        BoundaryRow(
            quantity="route",
            published_range="not applicable",
            used=f"straight transect, {transect.distance_m[-1] / 1000:.1f} km",
            status=UNMEASURED,
            basis=(
                "a transect rather than a plan; a planner would optimise against "
                "the cost model being measured, and could only reduce the energy"
            ),
        ),
        BoundaryRow(
            quantity="terrain",
            published_range="5 m LOLA grid, 20 km window",
            used="sampled by nearest cell along the line",
            status=INSIDE,
            basis=(
                "no second interpolation: the grid is already about nine tenths "
                "interpolated by its producers"
            ),
        ),
        BoundaryRow(
            quantity="actuator and avionics derating",
            published_range="none",
            used=f"{DERATING_RANGE[0]:.0f}× to {DERATING_RANGE[1]:.0f}×, nominal "
            f"{NOMINAL_DERATING:.0f}×",
            status=UNMEASURED,
            basis=(
                "mechanical work is not battery draw; a named parameter and "
                "swept, because no design has fixed it"
            ),
        ),
        BoundaryRow(
            quantity="battery capacity",
            published_range="none",
            used="swept; depth reported per watt-hour",
            status=UNMEASURED,
            basis=(
                "assumed like leg inertia was, so the rate is reported rather "
                "than a single depth"
            ),
        ),
        BoundaryRow(
            quantity="survival heating",
            published_range="none transcribed here",
            used="absent",
            status=UNMEASURED,
            basis=(
                f"would reduce reachable depth. The bar it must clear to matter "
                f"more than walking is {nominal_power:.0f} W continuous, and a "
                "forty-kelvin environment plausibly exceeds it"
            ),
        ),
        BoundaryRow(
            quantity="active illumination",
            published_range="none",
            used="absent",
            status=UNMEASURED,
            basis="would reduce reachable depth; a shadowed route carries its light",
        ),
        BoundaryRow(
            quantity="charge duty cycle",
            published_range="none",
            used="absent",
            status=UNMEASURED,
            basis=(
                "sets sorties per week rather than depth per sortie, so it bounds "
                "the schedule rather than this curve"
            ),
        ),
        BoundaryRow(
            quantity="comms below the rim",
            published_range="none",
            used="absent",
            status=UNMEASURED,
            basis=(
                "an autonomy requirement rather than an energy one, and the "
                "reason the perception and estimation seam is kept modular"
            ),
        ),
        BoundaryRow(
            quantity="destination shadowing",
            published_range="none",
            used="not established",
            status=UNMEASURED,
            basis=(
                "the route ends at the deepest ground the product covers; "
                "whether that is a cold trap is the illumination axis, and "
                "de Gerlache's own interior lies outside this window"
            ),
        ),
        BoundaryRow(
            quantity="regeneration",
            published_range="not applicable",
            used="none; descent clamped at zero cost",
            status=INSIDE,
            basis=(
                "a leg without regeneration dissipates surplus gravitational "
                "work rather than banking it, which is the conservative reading"
            ),
        ),
    )


def _format_float(value: float) -> str:
    return "nan" if not math.isfinite(value) else repr(float(value))


def build_report(transect: Transect, trip: RoundTrip, platform: Platform) -> str:
    rows = boundary_rows(trip, transect)
    site = tomllib.loads(SITE_PATH.read_text(encoding="utf-8"))
    nominal_power = average_locomotion_power_w(
        trip, speed_m_per_s=NOMINAL_SPEED, derating=NOMINAL_DERATING
    )
    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# Locomotion energy along a route over measured ground, out and back.",
        "#",
        "# Generated by studies/sortie/energy_envelope.py. Do not edit.",
        "#",
        "# THIS IS NOT A SORTIE ENVELOPE. A sortie needs six axes and one is",
        "# populated. Illumination, thermal, power, comms and cold-trap range are",
        "# declared and empty; every one of them reduces reachable depth or",
        "# bounds the schedule, so every number here is an upper bound.",
        "",
        f"schema_version = {REPORT_SCHEMA_VERSION}",
        "",
        "[environment]",
        f'python = "{host_platform.python_version()}"',
        f'numpy = "{np.__version__}"',
        "",
        "[inputs]",
        f'site = "{site["id"]}"',
        'terrain = "SL2_final_adj_5mpp_surf"',
        'platform = "nominal-quadruped"',
        'soil = "lunar-intercrater"',
        f"gravity_m_per_s2 = {_format_float(LUNAR_GRAVITY)}",
        f"feet_in_stance = {FEET_IN_STANCE}",
        f"crawl_duty_factor = {_format_float(CRAWL_DUTY)}",
        "",
        "[route]",
        'construction = "straight transect, rim crest to the deepest cell"',
        f"samples = {TRANSECT_SAMPLES}",
        f"horizontal_km = {_format_float(float(transect.distance_m[-1]) / 1000.0)}",
        f"descent_m = {_format_float(transect.descent_m)}",
        "walked_km_one_way = "
        f"{_format_float(trip.outbound.distance_m / 1000.0)}",
        "mean_absolute_slope_deg = "
        f"{_format_float(float(np.abs(transect.slope_degrees).mean()))}",
        "max_absolute_slope_deg = "
        f"{_format_float(float(np.abs(transect.slope_degrees).max()))}",
        "",
        "# The asymmetry. Gravity is signed and the dissipative terms are not, so",
        "# a return leg is not an outbound leg reversed.",
        "[round_trip]",
        "outbound_Wh = "
        f"{_format_float(trip.outbound.total_J / JOULES_PER_WATT_HOUR)}",
        "return_Wh = "
        f"{_format_float(trip.inbound.total_J / JOULES_PER_WATT_HOUR)}",
        f"total_Wh = {_format_float(trip.total_J / JOULES_PER_WATT_HOUR)}",
        f"return_over_outbound = {_format_float(trip.asymmetry)}",
        f"total_over_twice_outbound = {_format_float(trip.over_twice_outbound)}",
        "outbound_free_segment_fraction = "
        f"{_format_float(trip.outbound.free_fraction)}",
        "return_free_segment_fraction = "
        f"{_format_float(trip.inbound.free_fraction)}",
        'free_meaning = "gravity more than pays for the ground; clamped at zero '
        'because a leg without regeneration cannot bank the surplus"',
        "",
        "# Reachable depth per watt-hour, which is the platform-agnostic form.",
        "[envelope]",
        f"nominal_derating = {_format_float(NOMINAL_DERATING)}",
        f"derating_range = [{_format_float(DERATING_RANGE[0])}, "
        f"{_format_float(DERATING_RANGE[1])}]",
        "full_route_battery_Wh = "
        + _format_float(
            trip.battery_J_required(derating=NOMINAL_DERATING) / JOULES_PER_WATT_HOUR
        ),
        "",
    ]
    for battery in (100.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0):
        depth = reachable_depth_m(
            trip, transect, battery_wh=battery, derating=NOMINAL_DERATING
        )
        lines += [
            "[[envelope.at_capacity]]",
            f"battery_Wh = {_format_float(battery)}",
            f"reachable_depth_m = {_format_float(depth)}",
            "wh_per_metre_of_depth = "
            f"{_format_float(battery / depth if depth > 0 else math.nan)}",
            "",
        ]

    lines += [
        "# Whether locomotion binds. A route is a distance and a duration, so",
        "# energy over duration is a power, and that power is the bar another",
        "# subsystem has to clear to matter as much as walking.",
        "#",
        "# No survival-heating figure is quoted because none is transcribed in",
        "# this repository. The crossover is a property of this route and this",
        "# platform and needs no source.",
        "[binding]",
        "",
    ]
    for speed in (0.15, 0.25, 0.50):
        lines += [
            "[[binding.at_speed]]",
            f"speed_m_per_s = {_format_float(speed)}",
            f"sortie_hours = {_format_float(sortie_hours(trip, speed))}",
            "average_locomotion_power_W = "
            + _format_float(
                average_locomotion_power_w(
                    trip, speed_m_per_s=speed, derating=NOMINAL_DERATING
                )
            ),
            "",
        ]
    lines += [
        "[binding.conclusion]",
        f"average_locomotion_power_W = {_format_float(nominal_power)}",
        f'at_speed_m_per_s = {_format_float(NOMINAL_SPEED)}',
        "statement = \"\"\"",
        "Locomotion is not the binding term. Spread over the time this route",
        "takes, it averages tens of watts, which is comparable to the",
        "housekeeping of a small robot and small against what a forty-kelvin",
        "environment plausibly demands. A continuous load equal to that average",
        "halves locomotion's share of the sortie.",
        "",
        "Seven days have gone into the term that does not bind. That was worth",
        "doing, because it is the term that had to be shown adequate before any",
        "of the others could be trusted -- but rung five is thermal, and this is",
        "the measurement that says so.",
        '"""',
        "",
        f"# {tally(rows)}",
        "",
        *toml_lines(rows),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Integrate locomotion energy along a route over measured terrain, "
            "and test whether locomotion is the binding term."
        )
    )
    parser.add_argument("--figure-directory", type=Path, default=FIGURE_DIRECTORY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    arguments = parser.parse_args(argv)

    if not ELEVATION_PATH.exists():
        print(
            f"{ELEVATION_PATH.relative_to(REPOSITORY_ROOT)} is absent. Terrain "
            "products are fetched, not committed; run tools/fetch_terrain.py"
        )
        return 1

    elevation = read_float_geotiff(ELEVATION_PATH)
    platform = load_platform(PLATFORM_PATH).platform
    ground = load_ground()
    transect = build_transect(elevation)
    trip = walk_round_trip(
        transect=transect,
        platform=platform,
        contact_model=ground.contact,
        strength=ground.strength,
        mobilization=ground.mobilization,
        gravity_m_per_s2=LUNAR_GRAVITY,
        feet_in_stance=FEET_IN_STANCE,
        level_ground_slip_ratio=level_ground_slip(platform, ground),
    )

    arguments.figure_directory.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)

    for name, figure in (
        ("route-and-asymmetry", build_route_figure(transect, trip)),
        ("reachable-depth", build_envelope_figure(transect, trip)),
        ("what-binds", build_binding_figure(trip)),
    ):
        path = arguments.figure_directory / f"{name}.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        print(f"wrote {path.relative_to(REPOSITORY_ROOT)}")

    arguments.report.write_text(
        build_report(transect, trip, platform), encoding="utf-8"
    )
    print(f"wrote {arguments.report.relative_to(REPOSITORY_ROOT)}")

    print("\n  measured against extrapolated\n")
    print(text_table(boundary_rows(trip, transect)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
