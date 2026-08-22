# SPDX-License-Identifier: Apache-2.0
#
# studies.sortie.thermal_budget — what it costs to stay warm, and whether that
# is the term that binds.
#
# Day 7 established that locomotion is not the binding term: spread over the
# time its route takes it averages about nineteen watts. The honest response is
# to go and find the term that is, and the physics points at thermal for a
# reason with no terrestrial analogue. Vacuum removes convection entirely, so a
# body in permanent shadow loses heat two ways only, and both are one-way.
#
# Three results, and the first was expected to be the interesting one and is not.
#
# Conduction through the feet is negligible, by three orders of magnitude. The
# expectation going in -- stated in the brief and shared here -- was that foot
# geometry would turn out to be a thermal parameter as well as a bearing one,
# with a larger foot trading sinkage against heat loss and a design window
# somewhere between. There is no such trade. Lunar regolith at the surface
# conducts about 0.0015 W/(m K), roughly a tenth of silica aerogel, so the
# ground is a better insulator than anything the platform could be wrapped in.
# The contact radius at which conduction would match bare radiation is tens of
# metres. Foot size is a bearing parameter and nothing else, and the Day 3
# constraint on it stands alone.
#
# Radiation is the whole problem, and what it costs is a design choice rather
# than a fact about the Moon. A bare high-emissivity surface loses about two
# hundred watts holding a battery above its lower limit, which is ten times
# locomotion and makes the sortie decisively thermally bounded. Multi-layer
# insulation at an effective emissivity of a few percent brings it under
# locomotion. The physics does not decide; it sets a requirement.
#
# So the deliverable is that requirement rather than a survival time. Below an
# effective emissivity of about eight percent, thermal costs less than walking;
# above it, walking is a rounding error. That number is a property of this
# platform's radiating area and the temperature it must hold, and it is what a
# thermal design would have to be specified against.
#
# One node, quasi-steady, and both are limits rather than simplifications to
# apologise for. A real platform has a gradient between its battery and its
# feet, and the engineering lives in that gradient -- but a single node answers
# whether the mission shape exists, and a finer model would answer it no
# differently while hiding which term dominates.
#
# Still not a sortie envelope. Adding thermal makes it two axes of six.
# Illumination, charge duty cycle, comms and cold-trap range remain empty.
#
# References
#   Vaniman D et al. (1991) The Lunar Environment. In: Lunar Sourcebook, ch. 3.
#   Paige DA et al. (2010) Science 330, 479-482. doi:10.1126/science.1187726

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
from eclipse.io.soil import (
    janosi_hanamoto_model,
    load_soil,
    mohr_coulomb_model,
    regolith_conductivity_W_per_m_K,
)
from eclipse.io.terrain import read_float_geotiff
from eclipse.sortie import JOULES_PER_WATT_HOUR, RoundTrip, Transect, walk_round_trip
from eclipse.stance import wave_gait, within_stride_slip_ratio
from eclipse.thermal import (
    ThermalEnvelope,
    equilibrium_temperature_K,
    conductive_loss_W,
    cooling_time_s,
    radiative_loss_W,
    survival_power_W,
)

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
SOIL_PATH: Final = REPOSITORY_ROOT / "data" / "soils" / "lunar-intercrater.toml"
PLATFORM_PATH: Final = (
    REPOSITORY_ROOT / "configs" / "platforms" / "nominal-quadruped.toml"
)
ELEVATION_PATH: Final = (
    REPOSITORY_ROOT / "data" / "terrain" / "SL2_final_adj_5mpp_surf.tif"
)
FIGURE_DIRECTORY: Final = Path(__file__).resolve().parent / "figures"
DEFAULT_REPORT_PATH: Final = (
    Path(__file__).resolve().parent / "results" / "thermal-budget.toml"
)

REPORT_SCHEMA_VERSION: Final = 1
LUNAR_GRAVITY: Final = 1.62
FEET_IN_STANCE: Final = 3
NOMINAL_SPEED: Final = 0.25
NOMINAL_DERATING: Final = 4.0
DAY_SEVEN_LOCOMOTION_POWER_W: Final = 19.2

SKY_K: Final = 3.0
SOIL_K: Final = 38.0
SURFACE_DEPTH_RANGE: Final = "0-2"

# Every one of these is assumed. A 50 kg platform of mixed aluminium and
# electronics, a body that is mostly its own radiator, and a lithium cell that
# stops working around minus twenty. Named here so the sensitivities below are
# read as sensitivities.
RADIATING_AREA_M2: Final = 1.0
HEAT_CAPACITY_J_PER_K: Final = 45000.0
BATTERY_LIMIT_K: Final = 273.15 - 20.0
OPERATING_K: Final = 293.15
NOMINAL_EMISSIVITY: Final = 0.85

EMISSIVITY_SWEEP: Final[NDArray[np.float64]] = np.geomspace(0.01, 1.0, 100)
INTERNAL_POWER_SWEEP: Final = (0.0, 5.0, 20.0, 50.0)
FOOT_RADIUS_SWEEP: Final[NDArray[np.float64]] = np.linspace(0.010, 0.100, 91)
BATTERY_SWEEP_WH: Final[NDArray[np.float64]] = np.linspace(20.0, 1400.0, 70)
SINKAGE_CEILING_M: Final = 0.020


def caption(text: str, width: int = 148) -> str:
    return "\n".join(
        textwrap.fill(" ".join(paragraph.split()), width=width)
        for paragraph in text.split("\n")
    )


def envelope_with(*, emissivity: float, contact_radius_m: float) -> ThermalEnvelope:
    return ThermalEnvelope(
        radiating_area_m2=RADIATING_AREA_M2,
        emissivity=emissivity,
        heat_capacity_J_per_K=HEAT_CAPACITY_J_PER_K,
        contact_radius_m=contact_radius_m,
        contacts=FEET_IN_STANCE + 1,
    )


def holding_power_W(
    *, emissivity: float, contact_radius_m: float, conductivity: float
) -> float:
    return float(
        survival_power_W(
            envelope=envelope_with(
                emissivity=emissivity, contact_radius_m=contact_radius_m
            ),
            temperature_K=BATTERY_LIMIT_K,
            sky_K=SKY_K,
            soil_K=SOIL_K,
            soil_conductivity_W_per_m_K=conductivity,
        )
    )


def emissivity_matching(power_W: float) -> float:
    """Effective emissivity at which radiation alone costs a given power.

    Conduction is left out because it is three orders of magnitude smaller;
    including it would move this in the fourth decimal place.
    """
    from eclipse.thermal import STEFAN_BOLTZMANN

    return power_W / (
        STEFAN_BOLTZMANN * RADIATING_AREA_M2 * (BATTERY_LIMIT_K**4 - SKY_K**4)
    )


@dataclass(frozen=True, slots=True)
class Setting:
    trip: RoundTrip
    transect: Transect
    conductivity: float


def load_setting() -> Setting:
    elevation = read_float_geotiff(ELEVATION_PATH)
    platform = load_platform(PLATFORM_PATH).platform
    soil = load_soil(SOIL_PATH)
    mechanical = soil.datasets["carrier1991"]
    thermal = soil.datasets["vaniman1991"]
    strength = mohr_coulomb_model(mechanical, depth_range_cm="0-15")
    mobilization = janosi_hanamoto_model(mechanical)

    from eclipse.sortie import sample_transect

    highest = np.unravel_index(int(np.argmax(elevation.values)), elevation.values.shape)
    lowest = np.unravel_index(int(np.argmin(elevation.values)), elevation.values.shape)
    transect = sample_transect(
        elevation,
        start_row_column=(int(highest[0]), int(highest[1])),
        end_row_column=(int(lowest[0]), int(lowest[1])),
        samples=1200,
    )
    flat_slip, _ = within_stride_slip_ratio(
        platform=platform,
        gait=wave_gait(lift_order=(2, 0, 3, 1), duty_factor=0.75),
        strength=strength,
        mobilization=mobilization,
        gravity_m_per_s2=LUNAR_GRAVITY,
    )
    trip = walk_round_trip(
        transect=transect,
        platform=platform,
        contact_model=mechanical.models["bekker"].extrapolating,
        strength=strength,
        mobilization=mobilization,
        gravity_m_per_s2=LUNAR_GRAVITY,
        feet_in_stance=FEET_IN_STANCE,
        level_ground_slip_ratio=flat_slip,
    )
    return Setting(
        trip=trip,
        transect=transect,
        conductivity=regolith_conductivity_W_per_m_K(
            thermal, depth_range_cm=SURFACE_DEPTH_RANGE
        ),
    )


def reachable_index_with_thermal(
    setting: Setting, *, battery_wh: float, survival_W: float
) -> int:
    """Turn-back point when the budget must also keep the platform warm.

    Coupled, because going further takes longer and time is what survival costs.
    Walked outward until the sum of derated locomotion and survival over the
    whole round trip exceeds the battery.
    """
    trip = setting.trip
    out_J = np.concatenate([[0.0], trip.outbound.cumulative_J])
    remaining_J = trip.inbound.total_J - trip.inbound.cumulative_J[::-1]
    back_J = np.concatenate([remaining_J, [trip.inbound.total_J]])

    out_m = np.concatenate([[0.0], np.cumsum(trip.outbound.segment_length_m)])
    back_m = trip.inbound.distance_m - np.concatenate(
        [np.cumsum(trip.inbound.segment_length_m)[::-1], [0.0]]
    )
    seconds = (out_m + back_m) / NOMINAL_SPEED

    required_J = (out_J + back_J) * NOMINAL_DERATING + survival_W * seconds
    affordable = np.flatnonzero(required_J <= battery_wh * JOULES_PER_WATT_HOUR)
    return int(affordable[-1]) if affordable.size else 0


def reachable_depth_with_thermal(
    setting: Setting, *, battery_wh: float, survival_W: float
) -> float:
    index = reachable_index_with_thermal(
        setting, battery_wh=battery_wh, survival_W=survival_W
    )
    return float(setting.transect.elevation_m[0] - setting.transect.elevation_m[index])


def build_cooling_figure(setting: Setting) -> Figure:
    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (10.2, 5.4),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.650,
                    "figure.subplot.bottom": 0.190,
                    "figure.subplot.left": 0.070,
                    "figure.subplot.right": 0.986,
                    "figure.subplot.wspace": 0.245,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 2, squeeze=False)

        left = axes[0][0]
        # Two emissivities, because the contrast is the finding. Bare, the loss
        # swamps any plausible avionics and the platform dies in hours whatever
        # it is doing. Insulated, its own waste heat can hold it indefinitely,
        # and the curve stops at the temperature it settles to rather than
        # running off the bottom.
        cases: tuple[tuple[float, float, Any], ...] = (
            (NOMINAL_EMISSIVITY, 0.0, "solid"),
            (NOMINAL_EMISSIVITY, 50.0, "solid"),
            (0.05, 0.0, (0, (5, 2))),
            (0.05, 5.0, (0, (5, 2))),
            (0.05, 20.0, (0, (5, 2))),
        )
        for emissivity, power, style in cases:
            envelope = envelope_with(emissivity=emissivity, contact_radius_m=0.030)
            settled = equilibrium_temperature_K(
                envelope=envelope,
                internal_power_W=power,
                sky_K=SKY_K,
                soil_K=SOIL_K,
                soil_conductivity_W_per_m_K=setting.conductivity,
            )
            floor = max(settled + 0.5, BATTERY_LIMIT_K - 25.0)
            if floor >= OPERATING_K:
                continue
            temperatures = np.linspace(OPERATING_K, floor, 300)
            hours = [
                cooling_time_s(
                    envelope=envelope,
                    start_K=OPERATING_K,
                    limit_K=float(target),
                    internal_power_W=power,
                    sky_K=SKY_K,
                    soil_K=SOIL_K,
                    soil_conductivity_W_per_m_K=setting.conductivity,
                )
                / 3600.0
                for target in temperatures[1:]
            ]
            holds = settled >= BATTERY_LIMIT_K
            left.plot(
                np.concatenate([[0.0], np.asarray(hours)]),
                temperatures - 273.15,
                color=ACCENT_PRIMARY if emissivity < 0.5 else INK_PRIMARY,
                linewidth=1.6,
                linestyle=style,
                label=(
                    f"ε {emissivity:.2f}, {power:.0f} W internal"
                    + ("  — holds" if holds else "")
                ),
            )
        left.axhline(
            BATTERY_LIMIT_K - 273.15,
            color=ACCENT_SECONDARY,
            linewidth=1.2,
            linestyle=(0, (3, 2)),
        )
        left.annotate(
            "lithium cells stop working here",
            xy=(0.97, BATTERY_LIMIT_K - 273.15),
            xycoords=("axes fraction", "data"),
            xytext=(0, 5),
            textcoords="offset points",
            ha="right",
            color=ACCENT_SECONDARY,
            fontsize=7.8,
        )
        left.set_xlabel("time in shadow (hours)")
        left.set_ylabel("body temperature (°C)")
        left.set_title(
            "cooling, bare against insulated", color=INK_SECONDARY, loc="left"
        )
        left.set_xscale("symlog", linthresh=1.0)
        left.set_xlim(0.0, 400.0)
        left.set_ylim(-45.0, 22.0)
        left.legend(loc="lower left", fontsize=7.4)

        right = axes[0][1]
        powers = np.asarray(
            [
                holding_power_W(
                    emissivity=float(e),
                    contact_radius_m=0.030,
                    conductivity=setting.conductivity,
                )
                for e in EMISSIVITY_SWEEP
            ]
        )
        right.plot(EMISSIVITY_SWEEP, powers, color=ACCENT_PRIMARY, linewidth=1.8)
        right.axhline(
            DAY_SEVEN_LOCOMOTION_POWER_W,
            color=ACCENT_SECONDARY,
            linewidth=1.2,
            linestyle=(0, (4, 3)),
        )
        crossover = emissivity_matching(DAY_SEVEN_LOCOMOTION_POWER_W)
        right.plot(
            [crossover],
            [DAY_SEVEN_LOCOMOTION_POWER_W],
            marker="o",
            markersize=6.0,
            markerfacecolor="none",
            color=INK_PRIMARY,
        )
        right.annotate(
            f"equals locomotion at ε = {crossover:.3f}",
            xy=(crossover, DAY_SEVEN_LOCOMOTION_POWER_W),
            xytext=(10, -14),
            textcoords="offset points",
            color=INK_PRIMARY,
            fontsize=8.0,
        )
        right.annotate(
            f"locomotion, {DAY_SEVEN_LOCOMOTION_POWER_W:.0f} W",
            xy=(EMISSIVITY_SWEEP[0], DAY_SEVEN_LOCOMOTION_POWER_W),
            xytext=(4, 5),
            textcoords="offset points",
            color=ACCENT_SECONDARY,
            fontsize=7.8,
        )
        right.set_xscale("log")
        right.set_yscale("log")
        right.set_xlabel("effective emissivity of the radiating surface")
        right.set_ylabel(f"power to hold {BATTERY_LIMIT_K - 273.15:.0f} °C (W)")
        right.set_title(
            "what staying warm costs, against how well it is wrapped",
            color=INK_SECONDARY,
            loc="left",
        )

        for panel in (left, right):
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)

        figure.suptitle(
            "Whether the sortie is thermally bounded is a design choice, not a "
            "fact about the Moon",
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
                f"A {RADIATING_AREA_M2:.0f} m² body of "
                f"{HEAT_CAPACITY_J_PER_K / 1000:.0f} kJ/K holding a lithium cell "
                "above its lower limit, in a 38 K trap under a 3 K sky. Every "
                "platform property here is assumed, which is why the output is a "
                "requirement rather than a survival time.\n"
                f"Bare, it costs about {holding_power_W(emissivity=NOMINAL_EMISSIVITY, contact_radius_m=0.030, conductivity=setting.conductivity):.0f} W — "
                "ten times locomotion, and the sortie is thermally bounded "
                f"outright. Below an effective emissivity of {crossover:.3f}, "
                "thermal costs less than walking. Multi-layer insulation reaches "
                "that routinely, so the physics sets a requirement rather than a "
                "verdict.",
                width=150,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def build_coupling_figure(setting: Setting) -> Figure:
    from eclipse.io.terrain import read_float_geotiff  # noqa: F401

    soil = load_soil(SOIL_PATH)
    bekker = soil.datasets["carrier1991"].models["bekker"]
    platform = load_platform(PLATFORM_PATH).platform
    weight_N = platform.total_mass_kg * LUNAR_GRAVITY

    sinkage_mm = []
    conduction_W = []
    for radius in FOOT_RADIUS_SWEEP:
        stress_kPa = weight_N / FEET_IN_STANCE / (math.pi * radius**2) / 1000.0
        sinkage_mm.append(
            float(
                bekker.extrapolating.sinkage(
                    pressure=stress_kPa, contact_half_width=float(radius)
                )
            )
            * 1000.0
        )
        conduction_W.append(
            float(
                conductive_loss_W(
                    envelope=envelope_with(
                        emissivity=NOMINAL_EMISSIVITY, contact_radius_m=float(radius)
                    ),
                    temperature_K=BATTERY_LIMIT_K,
                    soil_K=SOIL_K,
                    soil_conductivity_W_per_m_K=setting.conductivity,
                )
            )
        )
    radiation_W = float(
        radiative_loss_W(
            envelope=envelope_with(
                emissivity=NOMINAL_EMISSIVITY, contact_radius_m=0.030
            ),
            temperature_K=BATTERY_LIMIT_K,
            sky_K=SKY_K,
        )
    )

    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (10.2, 5.4),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.635,
                    "figure.subplot.bottom": 0.190,
                    "figure.subplot.left": 0.078,
                    "figure.subplot.right": 0.900,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 1, squeeze=False)
        panel = axes[0][0]

        panel.plot(
            FOOT_RADIUS_SWEEP * 1000.0,
            sinkage_mm,
            color=ACCENT_PRIMARY,
            linewidth=1.8,
            label="sinkage, the bearing constraint",
        )
        panel.axhline(
            SINKAGE_CEILING_M * 1000.0,
            color=ACCENT_PRIMARY,
            linewidth=1.0,
            linestyle=(0, (3, 2)),
        )
        panel.annotate(
            "published bearing ceiling, 20 mm",
            xy=(FOOT_RADIUS_SWEEP[-1] * 1000.0, SINKAGE_CEILING_M * 1000.0),
            xytext=(-4, 5),
            textcoords="offset points",
            ha="right",
            color=ACCENT_PRIMARY,
            fontsize=7.8,
        )
        panel.set_xlabel("foot contact radius (mm)")
        panel.set_ylabel("sinkage (mm)", color=ACCENT_PRIMARY)
        panel.tick_params(axis="y", colors=ACCENT_PRIMARY)
        panel.set_ylim(0.0, 60.0)

        thermal_axis = panel.twinx()
        thermal_axis.plot(
            FOOT_RADIUS_SWEEP * 1000.0,
            conduction_W,
            color=ACCENT_SECONDARY,
            linewidth=1.8,
            label="conduction through the feet",
        )
        thermal_axis.axhline(
            radiation_W, color=INK_PRIMARY, linewidth=1.1, linestyle=(0, (4, 3))
        )
        thermal_axis.annotate(
            f"radiation from the body, {radiation_W:.0f} W — off the top of this axis "
            f"by {radiation_W / max(conduction_W):.0f}×",
            xy=(0.03, 0.94),
            xycoords="axes fraction",
            ha="left",
            va="top",
            color=INK_PRIMARY,
            fontsize=8.0,
        )
        thermal_axis.set_ylabel(
            "conductive loss (W)", color=ACCENT_SECONDARY
        )
        thermal_axis.tick_params(axis="y", colors=ACCENT_SECONDARY)
        thermal_axis.set_ylim(0.0, 1.0)
        thermal_axis.grid(False)

        panel.spines["top"].set_visible(False)

        figure.suptitle(
            "Foot size is a bearing parameter and not a thermal one: the two "
            "constraints never meet",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.078,
            ha="left",
            y=0.958,
        )
        figure.text(
            0.078,
            0.898,
            caption(
                "The expectation was a trade — a larger foot sinking less and "
                "losing more heat, with a design window between. There is no "
                "trade. Conduction through four feet spans a fraction of a watt "
                "across every plausible size, against a radiative loss of "
                f"{radiation_W:.0f} W from the body.\n"
                "The reason is measured, not assumed: regolith at the surface "
                f"conducts {setting.conductivity:.4f} W/(m K), about a tenth of "
                "silica aerogel. The ground is a better insulator than anything "
                "a platform could be wrapped in, so the bearing constraint on "
                "foot size stands alone and Day 3's answer needs no revision.",
                width=150,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def build_reordered_figure(setting: Setting) -> Figure:
    crossover = emissivity_matching(DAY_SEVEN_LOCOMOTION_POWER_W)
    cases = (
        (0.0, "locomotion only, Day 7", ACCENT_PRIMARY, "solid"),
        (
            holding_power_W(
                emissivity=0.05, contact_radius_m=0.030, conductivity=setting.conductivity
            ),
            "with insulated survival, ε = 0.05",
            ACCENT_SECONDARY,
            (0, (5, 2)),
        ),
        (
            holding_power_W(
                emissivity=NOMINAL_EMISSIVITY,
                contact_radius_m=0.030,
                conductivity=setting.conductivity,
            ),
            f"with bare survival, ε = {NOMINAL_EMISSIVITY:.2f}",
            INK_PRIMARY,
            (0, (2, 2)),
        ),
    )
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
                    "figure.subplot.top": 0.640,
                    "figure.subplot.bottom": 0.185,
                    "figure.subplot.left": 0.096,
                    "figure.subplot.right": 0.975,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 1, squeeze=False)
        panel = axes[0][0]

        for power, label, colour, style in cases:
            depths = [
                reachable_depth_with_thermal(
                    setting, battery_wh=float(wh), survival_W=power
                )
                for wh in BATTERY_SWEEP_WH
            ]
            panel.plot(
                BATTERY_SWEEP_WH,
                depths,
                color=colour,
                linewidth=1.8,
                linestyle=style,
                label=f"{label} — {power:.0f} W" if power else label,
            )

        panel.axhline(
            setting.transect.descent_m,
            color=INK_MUTED,
            linewidth=0.9,
            linestyle=(0, (3, 2)),
        )
        panel.annotate(
            f"the route ends here, {setting.transect.descent_m:.0f} m",
            xy=(BATTERY_SWEEP_WH[0], setting.transect.descent_m),
            xytext=(6, -12),
            textcoords="offset points",
            color=INK_SECONDARY,
            fontsize=7.8,
        )
        panel.set_xlabel("battery capacity available to the sortie (Wh)")
        panel.set_ylabel("reachable depth below the rim crest (m)")
        panel.set_xlim(BATTERY_SWEEP_WH[0], BATTERY_SWEEP_WH[-1])
        panel.set_ylim(0.0, setting.transect.descent_m * 1.12)
        panel.legend(loc="lower right")
        panel.spines["top"].set_visible(False)
        panel.spines["right"].set_visible(False)

        figure.suptitle(
            "Adding survival power to Day 7's envelope, at two levels of "
            "insulation",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.096,
            ha="left",
            y=0.958,
        )
        figure.text(
            0.096,
            0.898,
            caption(
                "Survival is a power and a sortie is a duration, so the two "
                f"couple: at {NOMINAL_SPEED:.2f} m/s, going further costs more "
                "time and therefore more heating, and the curve bends rather "
                "than shifting.\n"
                "Insulated, the envelope is close to Day 7's and locomotion "
                "still matters. Bare, depth collapses and the mission is "
                "thermally bounded outright. That gap is a design branch — "
                "insulation, radioisotope heating, shorter sorties, or a warm-up "
                f"cycle on the rim — and the requirement it sets is ε below "
                f"{crossover:.3f}.",
                width=140,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def boundary_rows(setting: Setting) -> tuple[BoundaryRow, ...]:
    crossover = emissivity_matching(DAY_SEVEN_LOCOMOTION_POWER_W)
    return (
        BoundaryRow(
            quantity="regolith thermal conductivity",
            published_range="1.5e-5 W/(cm K) in the upper 1-2 cm",
            used=f"{setting.conductivity:.4f} W/(m K)",
            status=INSIDE,
            basis=(
                "Vaniman et al. (1991) ch. 3, Apollo heat-flow probes; the "
                "printed unit is wrong and is recorded as an anomaly in the "
                "soil file rather than silently repaired"
            ),
        ),
        BoundaryRow(
            quantity="cold trap temperature",
            published_range="25 K measured minimum, 38 K at the LCROSS site",
            used=f"{SOIL_K:.0f} K",
            status=INSIDE,
            basis="Paige et al. (2010), Diviner; the Sourcebook predates it",
        ),
        BoundaryRow(
            quantity="thermal model",
            published_range="not applicable",
            used="one node, lumped capacitance, quasi-steady",
            status=UNMEASURED,
            basis=(
                "a real platform has a gradient between its battery and its "
                "feet and the engineering lives there; one node answers whether "
                "the mission shape exists and hides nothing about which term "
                "dominates"
            ),
        ),
        BoundaryRow(
            quantity="conduction geometry",
            published_range="not applicable",
            used="spreading resistance of a disc on a half-space",
            status=INSIDE,
            basis=(
                "exact for the geometry, and it needs no path length -- a "
                "quantity nobody could have supplied honestly for semi-infinite "
                "ground"
            ),
        ),
        BoundaryRow(
            quantity="radiating area",
            published_range="none",
            used=f"{RADIATING_AREA_M2:.1f} m2",
            status=UNMEASURED,
            basis="assumed; the survival power scales with it directly",
        ),
        BoundaryRow(
            quantity="effective emissivity",
            published_range="none",
            used=f"swept 0.01 to 1.0, nominal {NOMINAL_EMISSIVITY:.2f}",
            status=UNMEASURED,
            basis=(
                "the single most consequential platform property here, and a "
                f"design choice: below {crossover:.3f} thermal costs less than "
                "walking and above it walking is a rounding error"
            ),
        ),
        BoundaryRow(
            quantity="thermal mass",
            published_range="none",
            used=f"{HEAT_CAPACITY_J_PER_K / 1000:.0f} kJ/K",
            status=UNMEASURED,
            basis=(
                "assumed; it sets how long cooling takes and not what holding "
                "temperature costs, so it moves one figure and not the other"
            ),
        ),
        BoundaryRow(
            quantity="battery lower limit",
            published_range="none transcribed here",
            used=f"{BATTERY_LIMIT_K - 273.15:.0f} °C",
            status=UNMEASURED,
            basis=(
                "conventional for lithium cells; the first thing to stop "
                "working, and the limit the cooling curve is measured against"
            ),
        ),
        BoundaryRow(
            quantity="internal dissipation",
            published_range="none",
            used="swept 0 to 50 W",
            status=UNMEASURED,
            basis=(
                "avionics waste heat is free warmth; a platform that computes "
                "harder stays warmer, which is a real design coupling this "
                "study does not resolve"
            ),
        ),
        BoundaryRow(
            quantity="illumination and charge duty cycle",
            published_range="none",
            used="absent",
            status=UNMEASURED,
            basis=(
                "still empty after two axes; they bound sorties per week rather "
                "than depth per sortie"
            ),
        ),
    )


def _format_float(value: float) -> str:
    return "nan" if not math.isfinite(value) else repr(float(value))


def build_report(setting: Setting) -> str:
    rows = boundary_rows(setting)
    crossover = emissivity_matching(DAY_SEVEN_LOCOMOTION_POWER_W)
    bare = holding_power_W(
        emissivity=NOMINAL_EMISSIVITY,
        contact_radius_m=0.030,
        conductivity=setting.conductivity,
    )
    conduction = float(
        conductive_loss_W(
            envelope=envelope_with(emissivity=NOMINAL_EMISSIVITY, contact_radius_m=0.030),
            temperature_K=BATTERY_LIMIT_K,
            soil_K=SOIL_K,
            soil_conductivity_W_per_m_K=setting.conductivity,
        )
    )
    radiation = float(
        radiative_loss_W(
            envelope=envelope_with(emissivity=NOMINAL_EMISSIVITY, contact_radius_m=0.030),
            temperature_K=BATTERY_LIMIT_K,
            sky_K=SKY_K,
        )
    )

    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# What it costs to stay warm in permanent shadow, and whether that is",
        "# the term that binds.",
        "#",
        "# Generated by studies/sortie/thermal_budget.py. Do not edit.",
        "#",
        "# STILL NOT A SORTIE ENVELOPE. Adding thermal makes it two axes of six.",
        "# Illumination, charge duty cycle, comms and cold-trap range are empty.",
        "",
        f"schema_version = {REPORT_SCHEMA_VERSION}",
        "",
        "[environment]",
        f'python = "{host_platform.python_version()}"',
        f'numpy = "{np.__version__}"',
        "",
        "[setting]",
        f"sky_K = {_format_float(SKY_K)}",
        f"soil_K = {_format_float(SOIL_K)}",
        "soil_conductivity_W_per_m_K = " f"{_format_float(setting.conductivity)}",
        'soil_conductivity_source = "Vaniman et al. (1991) ch. 3, upper 1-2 cm"',
        f"battery_limit_C = {_format_float(BATTERY_LIMIT_K - 273.15)}",
        f"radiating_area_m2 = {_format_float(RADIATING_AREA_M2)}",
        f"heat_capacity_J_per_K = {_format_float(HEAT_CAPACITY_J_PER_K)}",
        "",
        "# The result that refuted the expectation. Conduction through the feet",
        "# is negligible because regolith is a better insulator than aerogel, so",
        "# foot size is a bearing parameter and nothing else.",
        "[channels]",
        f"radiative_W = {_format_float(radiation)}",
        f"conductive_W = {_format_float(conduction)}",
        f"ratio = {_format_float(radiation / conduction)}",
        "contact_radius_for_parity_m = "
        + _format_float(
            radiation
            / ((FEET_IN_STANCE + 1) * 4.0 * setting.conductivity * (BATTERY_LIMIT_K - SOIL_K))
        ),
        'conclusion = "the two channels never compete at any plausible foot size"',
        "",
        "# What holding temperature costs, against how well the body is wrapped.",
        "",
    ]
    for emissivity in (1.0, 0.85, 0.30, 0.10, 0.05, 0.02, 0.01):
        power = holding_power_W(
            emissivity=emissivity,
            contact_radius_m=0.030,
            conductivity=setting.conductivity,
        )
        lines += [
            "[[survival]]",
            f"emissivity = {_format_float(emissivity)}",
            f"holding_power_W = {_format_float(power)}",
            "against_locomotion = "
            f"{_format_float(power / DAY_SEVEN_LOCOMOTION_POWER_W)}",
            "",
        ]

    lines += [
        "# How long it lasts unheated, and how much heat postpones that.",
        "",
    ]
    envelope = envelope_with(emissivity=NOMINAL_EMISSIVITY, contact_radius_m=0.030)
    for power in INTERNAL_POWER_SWEEP:
        seconds = cooling_time_s(
            envelope=envelope,
            start_K=OPERATING_K,
            limit_K=BATTERY_LIMIT_K,
            internal_power_W=power,
            sky_K=SKY_K,
            soil_K=SOIL_K,
            soil_conductivity_W_per_m_K=setting.conductivity,
        )
        lines += [
            "[[cooling]]",
            f"internal_power_W = {_format_float(power)}",
            f"emissivity = {_format_float(NOMINAL_EMISSIVITY)}",
            "hours_to_battery_limit = "
            f"{_format_float(seconds / 3600.0 if math.isfinite(seconds) else math.inf)}",
            "",
        ]

    lines += [
        "# Day 7's envelope, recomputed with survival power folded in. Survival",
        "# is a power and a sortie is a duration, so they couple: going further",
        "# costs more time and therefore more heating.",
        "",
    ]
    for label, power in (
        ("locomotion_only", 0.0),
        (
            "insulated",
            holding_power_W(
                emissivity=0.05, contact_radius_m=0.030, conductivity=setting.conductivity
            ),
        ),
        ("bare", bare),
    ):
        for battery in (400.0, 800.0, 1200.0):
            lines += [
                "[[reordered_envelope]]",
                f'case = "{label}"',
                f"survival_W = {_format_float(power)}",
                f"battery_Wh = {_format_float(battery)}",
                "reachable_depth_m = "
                + _format_float(
                    reachable_depth_with_thermal(
                        setting, battery_wh=battery, survival_W=power
                    )
                ),
                "",
            ]

    lines += [
        "# The answer the week has been converging on.",
        "[verdict]",
        f"locomotion_power_W = {_format_float(DAY_SEVEN_LOCOMOTION_POWER_W)}",
        f"bare_survival_power_W = {_format_float(bare)}",
        f"emissivity_for_parity = {_format_float(crossover)}",
        "statement = \"\"\"",
        "The sortie is thermally bounded unless the platform is insulated, and",
        "insulation is a design choice rather than a physical limit. Bare, "
        "holding a lithium cell above its lower limit costs about ten times what",
        "walking costs, and reachable depth collapses. Below an effective",
        "emissivity of about eight percent it costs less than walking, and",
        "multi-layer insulation reaches that routinely.",
        "",
        "So the physics does not deliver a verdict; it delivers a requirement,",
        "and the requirement is on the radiating surface. Nothing about the feet",
        "enters: conduction into regolith is three orders of magnitude smaller",
        "than radiation to the sky, because the ground is a better insulator",
        "than anything the platform could be wrapped in.",
        '"""',
        "",
        f"# {tally(rows)}",
        "",
        *toml_lines(rows),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="What staying warm costs in permanent shadow, and whether it binds."
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

    setting = load_setting()
    arguments.figure_directory.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)

    for name, figure in (
        ("cooling-and-survival-power", build_cooling_figure(setting)),
        ("foot-size-coupling", build_coupling_figure(setting)),
        ("reordered-envelope", build_reordered_figure(setting)),
    ):
        path = arguments.figure_directory / f"{name}.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        print(f"wrote {path.relative_to(REPOSITORY_ROOT)}")

    arguments.report.write_text(build_report(setting), encoding="utf-8")
    print(f"wrote {arguments.report.relative_to(REPOSITORY_ROOT)}")

    print("\n  measured against extrapolated\n")
    print(text_table(boundary_rows(setting)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
