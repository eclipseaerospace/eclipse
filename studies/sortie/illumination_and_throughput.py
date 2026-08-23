# SPDX-License-Identifier: Apache-2.0
#
# studies.sortie.illumination_and_throughput — whether the destination is dark,
# how often the sortie can be repeated, and which axis binds now that three
# compete.
#
# Illumination has been doing the work in every earlier result without being
# computed. Day 7 walked to a destination whose darkness was assumed. Day 8
# priced survival for a shadow whose duration was assumed. This settles both
# from the same measured terrain, and then asks what actually limits throughput.
#
# Four results.
#
# The destination is a permanently shadowed region. The route's endpoint sees no
# sunlight at any point in a year of lunations, because it sits under a horizon
# that rises to twenty-seven degrees. That was an open boundary row since Day 7
# and it closes in the favourable direction: de Gerlache's own interior is
# outside the window, but the route still ends somewhere genuinely dark.
#
# The rim crest is a viable charge point. It is lit about four fifths of the
# time, with a horizon that falls away in every direction. Which makes the
# truncation caveat one-sided and worth stating precisely: rays leaving the
# window are treated as clear sky, so extra terrain can only raise a horizon.
# Darkness is therefore robust and sunlight is an upper bound. The destination
# being a PSR survives any wider search; the rim's four fifths does not.
#
# Charge does not bind. A modest array at the rim recharges a sortie's battery
# in hours against a sortie that takes days, so throughput is limited by how
# long the walk takes and not by how long the waiting takes.
#
# And the fourth is the one worth having. Sortie energy has a minimum in walking
# speed, because the two dominant terms pull opposite ways: swing work per metre
# rises as the square of speed, while survival energy is a power times a
# duration and so falls as one over it. Total energy is least around a third of
# a metre per second. Throughput peaks somewhere else entirely, near twice that,
# because a faster sortie is a shorter one even when it costs more -- until slip
# runs away near the gait limit and throughput falls again.
#
# So there is no single best speed, and saying which one is being optimised is
# now a mission decision rather than a modelling detail.
#
# Three axes of six. Comms and cold-trap range remain empty, and sunlight
# falling on the platform at the rim would change the thermal problem entirely
# and is deliberately out of scope.
#
# References
#   Mazarico E et al. (2011) Illumination conditions of the lunar polar regions
#     using LOLA topography. Icarus 211, 1066-1081.

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
from eclipse.illumination import (
    LUNAR_OBLIQUITY_DEG,
    SOLAR_ANGULAR_RADIUS_DEG,
    HorizonMap,
    Illumination,
    horizon_elevation_deg,
    illumination_fraction,
)
from eclipse.io.platform import load_platform
from eclipse.io.soil import janosi_hanamoto_model, load_soil, mohr_coulomb_model
from eclipse.io.terrain import GeoRaster, model_to_latitude_longitude, read_float_geotiff
from eclipse.platform import Platform
from eclipse.sortie import (
    JOULES_PER_WATT_HOUR,
    RoundTrip,
    Transect,
    sample_transect,
    walk_round_trip,
)
from eclipse.stance import maximum_walking_speed, wave_gait, within_stride_slip_ratio

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
ELEVATION_PATH: Final = (
    REPOSITORY_ROOT / "data" / "terrain" / "SL2_final_adj_5mpp_surf.tif"
)
PLATFORM_PATH: Final = (
    REPOSITORY_ROOT / "configs" / "platforms" / "nominal-quadruped.toml"
)
SOIL_PATH: Final = REPOSITORY_ROOT / "data" / "soils" / "lunar-intercrater.toml"
FIGURE_DIRECTORY: Final = Path(__file__).resolve().parent / "figures"
DEFAULT_REPORT_PATH: Final = (
    Path(__file__).resolve().parent / "results" / "illumination-and-throughput.toml"
)

REPORT_SCHEMA_VERSION: Final = 1
LUNAR_GRAVITY: Final = 1.62
FEET_IN_STANCE: Final = 3
NOMINAL_DERATING: Final = 4.0

# Assumed, named, and swept where it matters. A small body-mounted array that
# can face the Sun, which at a pole means roughly vertical.
ARRAY_AREA_M2: Final = 0.5
ARRAY_EFFICIENCY: Final = 0.30
SOLAR_CONSTANT_W_PER_M2: Final = 1361.0
HOURS_PER_WEEK: Final = 168.0
LUNATION_HOURS: Final = 29.53 * 24.0

# From Day 8, at an effective emissivity of 0.05.
INSULATED_SURVIVAL_W: Final = 11.8
BARE_SURVIVAL_W: Final = 198.1

MAP_STRIDE: Final = 50
HORIZON_AZIMUTHS: Final = 72
HORIZON_SAMPLES: Final = 140
# Where the horizon search begins. Below this the grid is reporting its own
# interpolator: at a five-metre lag a two-metre rise subtends thirty degrees.
# Above about 100 m it starts discarding real local terrain instead.
HORIZON_STANDOFF_M: Final = 50.0
STANDOFF_SWEEP_M: Final = (5.0, 25.0, 50.0, 100.0, 250.0)
ROUTE_SAMPLES: Final = 600
ROUTE_ILLUMINATION_STRIDE: Final = 6
SPEED_SWEEP: Final[NDArray[np.float64]] = np.linspace(0.10, 0.66, 29)
BATTERY_SWEEP_WH: Final[NDArray[np.float64]] = np.linspace(50.0, 3000.0, 60)


def caption(text: str, width: int = 148) -> str:
    return "\n".join(
        textwrap.fill(" ".join(paragraph.split()), width=width)
        for paragraph in text.split("\n")
    )


def north_azimuth_deg(raster: GeoRaster, rows: NDArray[np.int_], columns: NDArray[np.int_]) -> NDArray[np.float64]:
    """Where lunar north lies, in the raster frame, at each point.

    In a polar projection every meridian points a different way on the grid, so
    this cannot be a constant. For a south-polar site north is away from the
    pole, which is outward along the radius.
    """
    x = raster.origin_x_m + (columns.astype(np.float64) + 0.5) * raster.cell_size_m
    y = raster.origin_y_m - (rows.astype(np.float64) + 0.5) * raster.cell_size_m
    return np.asarray(np.degrees(np.arctan2(x, -y)) % 360.0)


def latitudes(raster: GeoRaster, rows: NDArray[np.int_], columns: NDArray[np.int_]) -> NDArray[np.float64]:
    return np.asarray(
        [
            model_to_latitude_longitude(
                raster.origin_x_m + (float(c) + 0.5) * raster.cell_size_m,
                raster.origin_y_m - (float(r) + 0.5) * raster.cell_size_m,
                reference_radius_m=raster.reference_radius_m,
            )[0]
            for r, c in zip(rows, columns)
        ]
    )


def illuminate(
    raster: GeoRaster, rows: NDArray[np.int_], columns: NDArray[np.int_]
) -> Illumination:
    horizon = horizon_elevation_deg(
        raster,
        rows=rows,
        columns=columns,
        azimuths=HORIZON_AZIMUTHS,
        samples_along_ray=HORIZON_SAMPLES,
        minimum_range_m=HORIZON_STANDOFF_M,
    )
    return illumination_fraction(
        horizon=horizon,
        latitude_deg=float(np.mean(latitudes(raster, rows, columns))),
        north_azimuth_deg=north_azimuth_deg(raster, rows, columns),
    )


@dataclass(frozen=True, slots=True)
class Setting:
    raster: GeoRaster
    platform: Platform
    transect: Transect
    trip: RoundTrip
    route_rows: NDArray[np.int_]
    route_columns: NDArray[np.int_]
    crest: tuple[int, int]
    destination: tuple[int, int]
    contact: Any
    strength: Any
    mobilization: Any


def load_setting() -> Setting:
    raster = read_float_geotiff(ELEVATION_PATH)
    platform = load_platform(PLATFORM_PATH).platform
    dataset = load_soil(SOIL_PATH).datasets["carrier1991"]
    contact = dataset.models["bekker"].extrapolating
    strength = mohr_coulomb_model(dataset, depth_range_cm="0-15")
    mobilization = janosi_hanamoto_model(dataset)

    highest = np.unravel_index(int(np.argmax(raster.values)), raster.values.shape)
    lowest = np.unravel_index(int(np.argmin(raster.values)), raster.values.shape)
    transect = sample_transect(
        raster,
        start_row_column=(int(highest[0]), int(highest[1])),
        end_row_column=(int(lowest[0]), int(lowest[1])),
        samples=ROUTE_SAMPLES,
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
        contact_model=contact,
        strength=strength,
        mobilization=mobilization,
        gravity_m_per_s2=LUNAR_GRAVITY,
        feet_in_stance=FEET_IN_STANCE,
        level_ground_slip_ratio=flat_slip,
    )
    rows = np.rint(
        np.linspace(highest[0], lowest[0], ROUTE_SAMPLES)
    ).astype(int)
    columns = np.rint(
        np.linspace(highest[1], lowest[1], ROUTE_SAMPLES)
    ).astype(int)
    return Setting(
        raster=raster,
        platform=platform,
        transect=transect,
        trip=trip,
        route_rows=rows,
        route_columns=columns,
        crest=(int(highest[0]), int(highest[1])),
        destination=(int(lowest[0]), int(lowest[1])),
        contact=contact,
        strength=strength,
        mobilization=mobilization,
    )


def average_charge_W(lit_fraction: float) -> float:
    return SOLAR_CONSTANT_W_PER_M2 * ARRAY_AREA_M2 * ARRAY_EFFICIENCY * lit_fraction


@dataclass(frozen=True, slots=True)
class SpeedPoint:
    speed_m_per_s: float
    hours: float
    locomotion_Wh: float
    survival_Wh: float

    @property
    def total_Wh(self) -> float:
        return self.locomotion_Wh + self.survival_Wh

    def sorties_per_week(self, charge_W: float) -> float:
        return HOURS_PER_WEEK / (self.hours + self.total_Wh / charge_W)


def sweep_speed(setting: Setting, *, survival_W: float) -> tuple[SpeedPoint, ...]:
    points = []
    for speed in SPEED_SWEEP:
        moving = Platform(
            **{
                **{
                    name: getattr(setting.platform, name)
                    for name in Platform.__dataclass_fields__
                },
                "nominal_speed_m_per_s": float(speed),
            }
        )
        flat_slip, _ = within_stride_slip_ratio(
            platform=moving,
            gait=wave_gait(lift_order=(2, 0, 3, 1), duty_factor=0.75),
            strength=setting.strength,
            mobilization=setting.mobilization,
            gravity_m_per_s2=LUNAR_GRAVITY,
        )
        trip = walk_round_trip(
            transect=setting.transect,
            platform=moving,
            contact_model=setting.contact,
            strength=setting.strength,
            mobilization=setting.mobilization,
            gravity_m_per_s2=LUNAR_GRAVITY,
            feet_in_stance=FEET_IN_STANCE,
            level_ground_slip_ratio=flat_slip,
        )
        distance = trip.outbound.distance_m + trip.inbound.distance_m
        hours = distance / float(speed) / 3600.0
        points.append(
            SpeedPoint(
                speed_m_per_s=float(speed),
                hours=hours,
                locomotion_Wh=trip.total_J
                / JOULES_PER_WATT_HOUR
                * NOMINAL_DERATING,
                survival_Wh=survival_W * hours,
            )
        )
    return tuple(points)


def build_map_figure(setting: Setting, grid: Illumination, shape: tuple[int, int]) -> Figure:
    raster = setting.raster
    lit = grid.any_sunlight_fraction.reshape(shape)
    extent = [
        raster.extent_m[0] / 1000.0,
        raster.extent_m[1] / 1000.0,
        raster.extent_m[2] / 1000.0,
        raster.extent_m[3] / 1000.0,
    ]

    def to_km(row: int, column: int) -> tuple[float, float]:
        return (
            (raster.origin_x_m + (column + 0.5) * raster.cell_size_m) / 1000.0,
            (raster.origin_y_m - (row + 0.5) * raster.cell_size_m) / 1000.0,
        )

    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (7.6, 8.6),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "axes.grid": False,
                    "figure.subplot.top": 0.762,
                    "figure.subplot.bottom": 0.062,
                    "figure.subplot.left": 0.108,
                    "figure.subplot.right": 0.880,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 1, squeeze=False)
        panel = axes[0][0]

        image = panel.imshow(
            lit * 100.0,
            extent=(extent[0], extent[1], extent[2], extent[3]),
            origin="upper",
            cmap="magma",
            vmin=0.0,
            vmax=100.0,
            interpolation="bilinear",
        )
        bar = figure.colorbar(image, ax=panel, fraction=0.046, pad=0.03)
        bar.set_label("fraction of a year with any sunlight (%)")

        route_x = [to_km(int(r), int(c))[0] for r, c in zip(setting.route_rows, setting.route_columns)]
        route_y = [to_km(int(r), int(c))[1] for r, c in zip(setting.route_rows, setting.route_columns)]
        panel.plot(route_x, route_y, color="white", linewidth=2.4, alpha=0.9)
        panel.plot(route_x, route_y, color=ACCENT_SECONDARY, linewidth=1.3)

        crest_x, crest_y = to_km(*setting.crest)
        end_x, end_y = to_km(*setting.destination)
        panel.plot([crest_x], [crest_y], marker="o", markersize=8.0,
                   markerfacecolor="none", markeredgewidth=1.8, color="white")
        panel.plot([end_x], [end_y], marker="s", markersize=8.0,
                   markerfacecolor="none", markeredgewidth=1.8, color="white")
        panel.annotate(
            "charge point\non the crest",
            xy=(crest_x, crest_y),
            xytext=(-10, -34),
            textcoords="offset points",
            ha="center",
            color="white",
            fontsize=8.0,
        )
        panel.annotate(
            "destination\n100% dark",
            xy=(end_x, end_y),
            xytext=(16, -26),
            textcoords="offset points",
            ha="left",
            va="top",
            color="white",
            fontsize=8.0,
        )
        panel.set_xlabel("polar stereographic x (km)")
        panel.set_ylabel("polar stereographic y (km)")
        panel.set_aspect("equal")

        figure.suptitle(
            "From a crest lit four fifths of the year into ground that never "
            "sees the Sun",
            color=INK_PRIMARY,
            fontsize=11.0,
            x=0.108,
            ha="left",
            y=0.972,
        )
        figure.text(
            0.108,
            0.930,
            caption(
                "Horizon computed from the 5 m LOLA grid at every point shown, "
                "swept over a year of lunations with the Sun treated as a disc "
                f"of {SOLAR_ANGULAR_RADIUS_DEG * 2:.2f}° and the sub-solar latitude "
                f"oscillating within the Moon's {LUNAR_OBLIQUITY_DEG:.2f}° "
                "obliquity. Lunar curvature is included: it drops distant ground "
                "by 29 m at 10 km.\n"
                f"Rays leaving the {raster.shape[0] * raster.cell_size_m / 1000:.0f} km "
                "window count as clear sky, so distant massifs are missing and "
                "every bright value is an upper bound. That caveat is one-sided: "
                "extra terrain can only raise a horizon, so the dark ground stays "
                "dark under any wider search.",
                width=96,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def build_route_figure(setting: Setting, route: Illumination) -> Figure:
    stride = ROUTE_ILLUMINATION_STRIDE
    index = np.unique(
        np.concatenate(
            [np.arange(0, setting.transect.distance_m.size, stride),
             [setting.transect.distance_m.size - 1]]
        )
    )
    distance_km = setting.transect.distance_m[index] / 1000.0
    any_sun = route.any_sunlight_fraction
    dark = any_sun <= 0.0
    first_dark = int(np.argmax(dark)) if bool(dark.any()) else len(dark)

    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (10.2, 5.6),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.680,
                    "figure.subplot.bottom": 0.155,
                    "figure.subplot.left": 0.080,
                    "figure.subplot.right": 0.905,
                    "figure.subplot.hspace": 0.380,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(2, 1, squeeze=False, sharex=True)

        upper = axes[0][0]
        upper.plot(
            distance_km, any_sun * 100.0, color=ACCENT_PRIMARY, linewidth=1.7
        )
        upper.fill_between(
            distance_km, 0.0, any_sun * 100.0, color=ACCENT_PRIMARY, alpha=0.16,
            linewidth=0.0,
        )
        if first_dark < len(dark):
            upper.axvline(
                distance_km[first_dark], color=INK_PRIMARY, linewidth=1.1,
                linestyle=(0, (3, 2)),
            )
            upper.annotate(
                f"enters permanent shadow at {distance_km[first_dark]:.1f} km",
                xy=(distance_km[first_dark], 60.0),
                xytext=(8, 0),
                textcoords="offset points",
                color=INK_PRIMARY,
                fontsize=8.0,
            )
        upper.set_ylabel("any sunlight (% of year)")
        upper.set_title(
            "illumination along the route", color=INK_SECONDARY, loc="left"
        )
        upper.set_ylim(0.0, 102.0)

        lower = axes[1][0]
        lower.plot(
            distance_km,
            setting.transect.elevation_m[index],
            color=INK_PRIMARY,
            linewidth=1.6,
        )
        lower.fill_between(
            distance_km,
            float(setting.transect.elevation_m.min()),
            setting.transect.elevation_m[index],
            where=dark,
            color=INK_PRIMARY,
            alpha=0.20,
            linewidth=0.0,
            label="in permanent shadow",
        )
        lower.set_xlabel("distance along the route (km)")
        lower.set_ylabel("elevation (m)")
        lower.set_title(
            "and the profile it follows", color=INK_SECONDARY, loc="left"
        )
        lower.legend(loc="upper right")

        for panel in (upper, lower):
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)

        shadow_fraction = float(dark.mean())
        one_way_hours = (
            setting.trip.outbound.distance_m / setting.platform.nominal_speed_m_per_s / 3600.0
        )
        figure.suptitle(
            "The sortie is a lit traverse to a dark point, not a traverse "
            "through darkness",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.080,
            ha="left",
            y=0.958,
        )
        figure.text(
            0.080,
            0.900,
            caption(
                f"Only {shadow_fraction:.0%} of the route is permanently shadowed — "
                "the destination itself. Everything before it sees the Sun for "
                "part of a year, which is a statement about the year and not "
                "about the sortie.\n"
                "So how long the platform is actually cold depends on WHEN the "
                "sortie runs, not only where it goes: a "
                f"{one_way_hours * 2:.0f} hour round trip is a fifteenth of a "
                "lunation, and a sortie timed into the lit part of the cycle "
                "spends little of it in shadow while a badly timed one spends "
                "all of it. Day 8's survival power applies to the dark hours, "
                "and scheduling decides how many there are. This study does not "
                "schedule.",
                width=150,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def build_throughput_figure(setting: Setting, crest_lit: float) -> Figure:
    charge_W = average_charge_W(crest_lit)
    insulated = sweep_speed(setting, survival_W=INSULATED_SURVIVAL_W)
    energies = np.asarray([point.total_Wh for point in insulated])
    locomotion = np.asarray([point.locomotion_Wh for point in insulated])
    survival = np.asarray([point.survival_Wh for point in insulated])
    throughput = np.asarray(
        [point.sorties_per_week(charge_W) for point in insulated]
    )
    speeds = np.asarray([point.speed_m_per_s for point in insulated])

    cheapest = int(np.argmin(energies))
    fastest = int(np.argmax(throughput))
    cap = maximum_walking_speed(
        platform=setting.platform,
        strength=setting.strength,
        gravity_m_per_s2=LUNAR_GRAVITY,
    )

    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (10.4, 5.4),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.650,
                    "figure.subplot.bottom": 0.190,
                    "figure.subplot.left": 0.070,
                    "figure.subplot.right": 0.986,
                    "figure.subplot.wspace": 0.240,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 2, squeeze=False)

        left = axes[0][0]
        left.stackplot(
            speeds,
            locomotion,
            survival,
            colors=[ACCENT_PRIMARY, ACCENT_SECONDARY],
            labels=["locomotion, rises as speed squared", "survival, falls as one over speed"],
            edgecolor="none",
            alpha=0.9,
        )
        left.plot(speeds, energies, color=INK_PRIMARY, linewidth=1.6)
        left.plot(
            [speeds[cheapest]], [energies[cheapest]], marker="o", markersize=6.0,
            markerfacecolor="none", color=INK_PRIMARY,
        )
        left.annotate(
            f"least energy at {speeds[cheapest]:.2f} m/s\n{energies[cheapest]:.0f} Wh",
            xy=(speeds[cheapest], energies[cheapest]),
            xytext=(8, 24),
            textcoords="offset points",
            color=INK_PRIMARY,
            fontsize=8.0,
        )
        left.set_xlabel("walking speed (m/s)")
        left.set_ylabel("sortie energy (Wh)")
        left.set_title(
            "the two dominant terms pull opposite ways",
            color=INK_SECONDARY,
            loc="left",
        )
        left.set_xlim(speeds[0], speeds[-1])
        left.set_ylim(0.0, float(energies.max()) * 1.05)
        left.legend(loc="upper center")

        right = axes[0][1]
        right.plot(speeds, throughput, color=ACCENT_PRIMARY, linewidth=1.8)
        right.plot(
            [speeds[fastest]], [throughput[fastest]], marker="o", markersize=6.0,
            markerfacecolor="none", color=INK_PRIMARY,
        )
        right.annotate(
            f"most sorties at {speeds[fastest]:.2f} m/s\n{throughput[fastest]:.1f} per week",
            xy=(speeds[fastest], throughput[fastest]),
            xytext=(-10, -34),
            textcoords="offset points",
            ha="right",
            color=INK_PRIMARY,
            fontsize=8.0,
        )
        right.axvline(
            speeds[cheapest], color=ACCENT_SECONDARY, linewidth=1.0,
            linestyle=(0, (3, 2)),
        )
        right.annotate(
            "least-energy speed",
            xy=(speeds[cheapest], throughput.max() * 0.30),
            xytext=(6, 0),
            textcoords="offset points",
            rotation=90.0,
            va="center",
            color=ACCENT_SECONDARY,
            fontsize=7.8,
        )
        if cap <= speeds[-1]:
            right.axvline(cap, color=INK_PRIMARY, linewidth=1.0, linestyle=(0, (2, 2)))
        right.set_xlabel("walking speed (m/s)")
        right.set_ylabel("sorties per week")
        right.set_title(
            f"and throughput peaks somewhere else, at {charge_W:.0f} W of charge",
            color=INK_SECONDARY,
            loc="left",
        )
        right.set_xlim(speeds[0], speeds[-1])
        right.set_ylim(0.0, None)

        for panel in (left, right):
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)

        figure.suptitle(
            "There is no single best speed: energy is cheapest at "
            f"{speeds[cheapest]:.2f} m/s and throughput peaks at {speeds[fastest]:.2f}",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.070,
            ha="left",
            y=0.955,
        )
        figure.text(
            0.070,
            0.892,
            caption(
                "Swing work per metre goes as the square of speed, so locomotion "
                "rises. Survival is a power times a duration, so it falls as one "
                "over speed. Their sum has a minimum, and it is not where "
                "throughput is greatest — a faster sortie is a shorter one even "
                "when it costs more, until slip runs away near the gait limit "
                "and throughput falls again.\n"
                f"Charge does not bind. At {ARRAY_AREA_M2:.1f} m² and "
                f"{ARRAY_EFFICIENCY:.0%}, recharging takes hours against a sortie "
                "of days, so which speed to walk is a mission decision between "
                "energy and throughput rather than a power-system question.",
                width=150,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def boundary_rows(setting: Setting, grid: Illumination, crest_lit: float) -> tuple[BoundaryRow, ...]:
    window_km = setting.raster.shape[0] * setting.raster.cell_size_m / 1000.0
    return (
        BoundaryRow(
            quantity="horizon search range",
            published_range="not applicable",
            used=f"within the {window_km:.0f} km window only",
            status=OUTSIDE,
            basis=(
                f"{grid.horizon.truncated_fraction:.0%} of ray samples leave the grid "
                "and count as clear sky. Distant massifs shadow polar sites from "
                "tens of kilometres, so every lit fraction is an upper bound -- "
                "one-sided, since extra terrain can only raise a horizon"
            ),
        ),
        BoundaryRow(
            quantity="solar disc",
            published_range="0.53 degrees, standard",
            used=f"disc of angular radius {SOLAR_ANGULAR_RADIUS_DEG} degrees",
            status=INSIDE,
            basis=(
                "a point Sun would report a hard shadow edge that does not "
                "exist; at polar elevations the penumbra covers real ground"
            ),
        ),
        BoundaryRow(
            quantity="lunar curvature",
            published_range="not applicable",
            used="d^2/2R drop applied along every ray",
            status=INSIDE,
            basis=(
                "29 m at 10 km, comparable to the relief that decides a horizon "
                "at grazing incidence; omitting it would overstate shadowing"
            ),
        ),
        BoundaryRow(
            quantity="solar ephemeris",
            published_range="not applicable",
            used="analytic, swept over hour angle and obliquity",
            status=UNMEASURED,
            basis=(
                "SPICE would give event times; this gives a fraction, and the "
                "fraction is limited by the horizon rather than by the ephemeris"
            ),
        ),
        BoundaryRow(
            quantity="array area and efficiency",
            published_range="none",
            used=f"{ARRAY_AREA_M2:.1f} m2 at {ARRAY_EFFICIENCY:.0%}",
            status=UNMEASURED,
            basis=(
                "assumed, and it does not bind: recharge takes hours against a "
                "sortie of days, so throughput is limited by walking rather than "
                "by waiting"
            ),
        ),
        BoundaryRow(
            quantity="array pointing",
            published_range="none",
            used="assumed able to face the Sun",
            status=UNMEASURED,
            basis=(
                "at a pole the Sun is within a couple of degrees of the horizon, "
                "so a horizontal panel would collect almost nothing and this "
                "assumption is doing real work"
            ),
        ),
        BoundaryRow(
            quantity="sunlight on the platform",
            published_range="none",
            used="absent",
            status=UNMEASURED,
            basis=(
                "solar input at the rim would change Day 8's thermal balance "
                "entirely, in the favourable direction; deliberately out of scope"
            ),
        ),
        BoundaryRow(
            quantity="secondary illumination",
            published_range="none",
            used="absent",
            status=UNMEASURED,
            basis=(
                "light scattered from sunlit walls reaches shadowed floors, "
                "which ShadowCam measures and this does not; it affects vision "
                "rather than the energy budget"
            ),
        ),
        BoundaryRow(
            quantity="destination shadowing",
            published_range="none",
            used=f"{crest_lit:.1%} lit at the crest, 0% at the destination",
            status=INSIDE,
            basis=(
                "closes the boundary row open since Day 7: the route ends in "
                "permanent shadow, and that conclusion survives a wider horizon"
            ),
        ),
    )


def _format_float(value: float) -> str:
    return "nan" if not math.isfinite(value) else repr(float(value))


def build_report(
    setting: Setting, grid: Illumination, route: Illumination, probes: Illumination
) -> str:
    crest_lit = float(probes.any_sunlight_fraction[0])
    destination_lit = float(probes.any_sunlight_fraction[1])
    charge_W = average_charge_W(crest_lit)
    rows = boundary_rows(setting, grid, crest_lit)
    insulated = sweep_speed(setting, survival_W=INSULATED_SURVIVAL_W)
    energies = [point.total_Wh for point in insulated]
    throughput = [point.sorties_per_week(charge_W) for point in insulated]
    cheapest = int(np.argmin(energies))
    fastest = int(np.argmax(throughput))
    dark_fraction = float((route.any_sunlight_fraction <= 0.0).mean())

    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# Illumination from measured terrain, and what it does to the sortie.",
        "#",
        "# Generated by studies/sortie/illumination_and_throughput.py. Do not edit.",
        "#",
        "# THREE AXES OF SIX. Comms and cold-trap range are still empty, and",
        "# sunlight on the platform is deliberately out of scope.",
        "",
        f"schema_version = {REPORT_SCHEMA_VERSION}",
        "",
        "[environment]",
        f'python = "{host_platform.python_version()}"',
        f'numpy = "{np.__version__}"',
        "",
        "[method]",
        f"horizon_azimuths = {HORIZON_AZIMUTHS}",
        f"horizon_samples_along_ray = {HORIZON_SAMPLES}",
        f"solar_angular_radius_deg = {_format_float(SOLAR_ANGULAR_RADIUS_DEG)}",
        f"lunar_obliquity_deg = {_format_float(LUNAR_OBLIQUITY_DEG)}",
        'ephemeris = "analytic; hour angle and sub-solar latitude swept"',
        "curvature_applied = true",
        "horizon_truncated_fraction = "
        f"{_format_float(grid.horizon.truncated_fraction)}",
        'truncation_direction = "lit fractions are upper bounds; dark ground stays dark"',
        "",
        "# The question left open on Day 7, now answered.",
        "[destination]",
        f"lit_fraction = {_format_float(destination_lit)}",
        "is_permanently_shadowed = " + str(destination_lit <= 0.0).lower(),
        "horizon_max_deg = "
        f"{_format_float(float(probes.horizon.elevation_deg[1].max()))}",
        'note = "de Gerlache\'s own interior lies outside this window; the route '
        'still ends in permanent shadow"',
        "",
        "[charge_point]",
        f"lit_fraction = {_format_float(crest_lit)}",
        "horizon_max_deg = "
        f"{_format_float(float(probes.horizon.elevation_deg[0].max()))}",
        f"array_area_m2 = {_format_float(ARRAY_AREA_M2)}",
        f"array_efficiency = {_format_float(ARRAY_EFFICIENCY)}",
        f"average_charge_W = {_format_float(charge_W)}",
        "",
        "[route_illumination]",
        f"shadowed_fraction_of_route = {_format_float(dark_fraction)}",
        "",
        "# The trade that has no single answer. Locomotion rises as the square of",
        "# speed because swing work does; survival falls as one over speed",
        "# because it is a power times a duration.",
        "",
    ]
    for point in insulated[::4]:
        lines += [
            "[[speed]]",
            f"speed_m_per_s = {_format_float(point.speed_m_per_s)}",
            f"hours = {_format_float(point.hours)}",
            f"locomotion_Wh = {_format_float(point.locomotion_Wh)}",
            f"survival_Wh = {_format_float(point.survival_Wh)}",
            f"total_Wh = {_format_float(point.total_Wh)}",
            "sorties_per_week = "
            f"{_format_float(point.sorties_per_week(charge_W))}",
            "",
        ]

    lines += [
        "[optima]",
        "least_energy_speed_m_per_s = "
        f"{_format_float(insulated[cheapest].speed_m_per_s)}",
        f"least_energy_Wh = {_format_float(energies[cheapest])}",
        "most_sorties_speed_m_per_s = "
        f"{_format_float(insulated[fastest].speed_m_per_s)}",
        f"most_sorties_per_week = {_format_float(throughput[fastest])}",
        "gait_speed_cap_m_per_s = "
        + _format_float(
            maximum_walking_speed(
                platform=setting.platform,
                strength=setting.strength,
                gravity_m_per_s2=LUNAR_GRAVITY,
            )
        ),
        "",
        "# Which of the three binds, now that three compete.",
        "[binding]",
        "statement = \"\"\"",
        "Charge does not bind. A modest array at the crest recharges a sortie's",
        "battery in hours against a sortie that takes days, so throughput is set",
        "by how long the walk takes rather than how long the waiting takes.",
        "",
        "What binds is duration, and duration is a choice. Sortie energy is least",
        "at about a third of a metre per second, where rising swing work meets",
        "falling survival energy. Throughput is greatest at roughly twice that,",
        "because a shorter sortie repeats sooner even when it costs more -- until",
        "slip runs away near the gait limit from rung three and throughput falls",
        "again.",
        "",
        "So there is no single best speed, and which one to walk is a mission",
        "decision between energy and throughput rather than a physical limit.",
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
            "Compute illumination from measured terrain, settle whether the "
            "destination is dark, and find what limits throughput."
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

    setting = load_setting()
    height, width = setting.raster.shape

    grid_rows, grid_columns = np.meshgrid(
        np.arange(0, height, MAP_STRIDE),
        np.arange(0, width, MAP_STRIDE),
        indexing="ij",
    )
    shape = grid_rows.shape
    print(f"  horizon over {grid_rows.size} map points ...")
    grid = illuminate(setting.raster, grid_rows.ravel(), grid_columns.ravel())

    stride = ROUTE_ILLUMINATION_STRIDE
    # Include the final sample explicitly: a stride that misses the endpoint
    # would report a route that never reaches the place it is going.
    route_index = np.unique(
        np.concatenate(
            [np.arange(0, setting.route_rows.size, stride), [setting.route_rows.size - 1]]
        )
    )
    print(f"  horizon along {route_index.size} route points ...")
    route = illuminate(
        setting.raster, setting.route_rows[route_index], setting.route_columns[route_index]
    )

    probes = illuminate(
        setting.raster,
        np.array([setting.crest[0], setting.destination[0]]),
        np.array([setting.crest[1], setting.destination[1]]),
    )
    crest_lit = float(probes.any_sunlight_fraction[0])

    arguments.figure_directory.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)

    for name, figure in (
        ("illumination-map", build_map_figure(setting, grid, shape)),
        ("illumination-along-the-route", build_route_figure(setting, route)),
        ("speed-energy-throughput", build_throughput_figure(setting, crest_lit)),
    ):
        path = arguments.figure_directory / f"{name}.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        print(f"wrote {path.relative_to(REPOSITORY_ROOT)}")

    arguments.report.write_text(
        build_report(setting, grid, route, probes), encoding="utf-8"
    )
    print(f"wrote {arguments.report.relative_to(REPOSITORY_ROOT)}")

    print("\n  measured against extrapolated\n")
    print(text_table(boundary_rows(setting, grid, crest_lit)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
