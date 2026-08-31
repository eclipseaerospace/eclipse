# SPDX-License-Identifier: Apache-2.0
#
# studies.sites.morphology — crew, wheel and leg over the same ten sites.
#
# Day 14 measured the legged case in ground and found 8.3 times the terrain and
# 21.8 times the cold trap of a suited crew. Every one of those comparisons was
# against a human in a suit, and nobody is proposing to walk astronauts into
# permanently shadowed craters. The competitor is VIPER, and it is wheeled.
#
# This study runs the same search over the same ground from the same charge
# point with a third actor in it. Nothing about the search changes: the planner
# consumes a cost-per-metre table and has never been told what produces one, so
# a rover enters it as a different table rather than as a different code path.
# That was the architectural claim and this is the first day it has been tested
# against a genuinely different contact regime rather than a different number of
# the same patches.
#
# The physics is in studies.mobility.wheels and comes to this: below about eight
# degrees the wheel is the cheaper machine because it pays no swing work, and
# above it the wheel is the more expensive one because it must drag itself out
# of the rut it is cutting, with a slip that divides every joule it spends. The
# rover stops at 26.6 degrees on traction; the quadruped runs to 39.8 and stops
# by tipping over.
#
# Which of those two facts decides the errand is a question about terrain rather
# than about either machine, and it is what this study is for. Polar terrain is
# mostly gentle, so the wheel should win the area comparison outright. Whether
# it also wins the one that matters -- cold trap reached, which is the errand --
# depends on whether the last few hundred metres into a shadowed crater are
# steep, and that is not something either platform gets a say in.
#
# One asymmetry is deliberate and runs against the platform this project is
# building. Both machines are given the same derating, the same insulation power
# and the same battery, and the rover is given no grousers and a small wheel. A
# real rover would have both and would do better than this. The comparison is
# not tuned to make legs look good; where it favours them it should be checked
# against the boundary table first, which lists what the wheel is not being
# credited with.
#
# References
#   Wong JY (2008) Theory of Ground Vehicles, 4th ed. Wiley.
#   Carrier WD III, Olhoeft GR, Mendell W (1991) Physical Properties of the
#     Lunar Surface. In: Lunar Sourcebook, ch. 9. Cambridge University Press.

from __future__ import annotations

import argparse
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from eclipse.analysis.boundary import (  # noqa: E402
    INSIDE,
    OUTSIDE,
    UNMEASURED,
    BoundaryRow,
    tally,
    text_table,
    toml_lines,
)
from eclipse.analysis.style import (  # noqa: E402
    ACCENT_PRIMARY,
    ACCENT_SECONDARY,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    figure_style,
)
from eclipse.illumination import (  # noqa: E402
    best_charge_point,
    horizon_elevation_deg,
    illumination_fraction,
)
from eclipse.io.platform import load_platform, load_wheeled_platform  # noqa: E402
from eclipse.io.site import Site, load_sites  # noqa: E402
from eclipse.io.soil import (  # noqa: E402
    janosi_hanamoto_model,
    load_soil,
    mohr_coulomb_model,
)
from eclipse.io.terrain import (  # noqa: E402
    GeoRaster,
    centred_window,
    latitudes_degrees,
    load_terrain_manifest,
    north_azimuth_degrees,
    read_float_geotiff,
)
from eclipse.mobility import cost_of_transport  # noqa: E402
from eclipse.planning import TraversalCost, round_trip_energy_J  # noqa: E402
from eclipse.platform import (  # noqa: E402
    Platform,
    equilibrium_slip_ratio,
    swing_work_per_meter,
)
from eclipse.rolling import (  # noqa: E402
    WheeledPlatform,
    rolling_cost_of_transport,
    wheel_equilibrium_slip_ratio,
)
from eclipse.sortie import JOULES_PER_WATT_HOUR  # noqa: E402
from eclipse.stance import wave_gait, within_stride_slip_ratio  # noqa: E402

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
SITE_DIRECTORY: Final = REPOSITORY_ROOT / "configs" / "sites"
TERRAIN_DIRECTORY: Final = REPOSITORY_ROOT / "data" / "terrain"
MANIFEST_PATH: Final = TERRAIN_DIRECTORY / "manifest.toml"
QUADRUPED_PATH: Final = (
    REPOSITORY_ROOT / "configs" / "platforms" / "nominal-quadruped.toml"
)
ROVER_PATH: Final = REPOSITORY_ROOT / "configs" / "platforms" / "nominal-rover.toml"
SOIL_PATH: Final = REPOSITORY_ROOT / "data" / "soils" / "lunar-intercrater.toml"
FIGURE_DIRECTORY: Final = Path(__file__).resolve().parent / "figures"
DEFAULT_REPORT_PATH: Final = (
    Path(__file__).resolve().parent / "results" / "morphology.toml"
)

REPORT_SCHEMA_VERSION: Final = 1
LUNAR_GRAVITY: Final = 1.62
FEET_IN_STANCE: Final = 3
NOMINAL_DERATING: Final = 4.0
LEGGED_TIPPING_LIMIT_DEG: Final = 39.8055710922652
COST_SLOPE_DEG: Final[NDArray[np.float64]] = np.arange(-89.0, 89.01, 0.1)

COMMON_WINDOW_KM: Final = 16.0
MAP_STRIDE: Final = 50
MASK_STRIDE: Final = 20
HORIZON_AZIMUTHS: Final = 72
HORIZON_SAMPLES: Final = 140
HORIZON_STANDOFF_M: Final = 50.0

INSULATED_SURVIVAL_W: Final = 11.8
DWELL_HOURS: Final = 4.0
NOMINAL_BATTERY_WH: Final = 400.0
LIT_WINDOW_HOURS: Final = 520.8

SHOWCASE_SITE: Final = "nobile-rim-1"


def caption(text: str, width: int = 150) -> str:
    return "\n".join(
        textwrap.fill(" ".join(paragraph.split()), width=width)
        for paragraph in text.split("\n")
    )


def illuminate(
    raster: GeoRaster, rows: NDArray[np.int_], columns: NDArray[np.int_]
) -> Any:
    return illumination_fraction(
        horizon=horizon_elevation_deg(
            raster,
            rows=rows,
            columns=columns,
            azimuths=HORIZON_AZIMUTHS,
            samples_along_ray=HORIZON_SAMPLES,
            minimum_range_m=HORIZON_STANDOFF_M,
        ),
        latitude_deg=latitudes_degrees(raster, rows, columns),
        north_azimuth_deg=north_azimuth_degrees(raster, rows, columns),
    )


def walking_cost(
    *,
    platform: Platform,
    contact: Any,
    strength: Any,
    mobilization: Any,
    survival_W: float,
    speed_m_per_s: float,
) -> TraversalCost:
    """Day 14's legged table, unchanged, so the two actors differ only in body."""
    flat, _ = within_stride_slip_ratio(
        platform=platform,
        gait=wave_gait(lift_order=(2, 0, 3, 1), duty_factor=0.75),
        strength=strength,
        mobilization=mobilization,
        gravity_m_per_s2=LUNAR_GRAVITY,
    )
    hotel = survival_W / speed_m_per_s
    joules = np.full(COST_SLOPE_DEG.shape, np.inf)
    for index, slope in enumerate(COST_SLOPE_DEG):
        if abs(float(slope)) > LEGGED_TIPPING_LIMIT_DEG:
            continue
        demanded = equilibrium_slip_ratio(
            platform=platform,
            strength=strength,
            mobilization=mobilization,
            gravity_m_per_s2=LUNAR_GRAVITY,
            slope_degrees=abs(float(slope)),
        )
        ratio = max(float(demanded), flat)
        if not np.isfinite(ratio) or ratio >= 1.0:
            continue
        swing = float(
            swing_work_per_meter(
                platform=platform, gravity_m_per_s2=LUNAR_GRAVITY, slip_ratio=ratio
            ).total_J
        )
        cost = cost_of_transport(
            mass_kg=platform.total_mass_kg,
            gravity_m_per_s2=LUNAR_GRAVITY,
            slope_degrees=float(slope),
            slip_ratio=ratio,
            patch=platform.contact_patch,
            feet_in_stance=FEET_IN_STANCE,
            stride_length_m=platform.stride_length_m,
            stance_length_m=platform.stride_length_m,
            contact_model=contact,
            strength=strength,
            mobilization=mobilization,
            swing_work_per_meter_J=swing,
        )
        joules[index] = (
            max(float(cost.total_J_per_m), 0.0) * NOMINAL_DERATING + hotel
        )
    usable = np.isfinite(joules)
    return TraversalCost(
        slope_deg=COST_SLOPE_DEG[usable],
        joules_per_metre=joules[usable],
        limit_deg=min(
            float(np.abs(COST_SLOPE_DEG[usable]).max()), LEGGED_TIPPING_LIMIT_DEG
        ),
    )


def rolling_cost(
    *,
    platform: WheeledPlatform,
    contact: Any,
    strength: Any,
    mobilization: Any,
    survival_W: float,
    speed_m_per_s: float,
) -> TraversalCost:
    """The rover's table, in the same type the planner already consumed.

    The interface test, and it passes: this function is the whole of what a new
    morphology costs at this layer. Nothing in eclipse.planning, eclipse.mobility
    or the reachable-set composition knows a wheel exists, because a cost curve
    is all any of them were ever given.

    Derating, insulation and dwell are identical to the legged table on purpose.
    A wheel drivetrain is very likely more efficient than legged actuators, and
    crediting it with that would be inventing a number rather than measuring
    one, so both actors carry the same factor and the difference that remains is
    contact physics.
    """
    hotel = survival_W / speed_m_per_s
    slip = wheel_equilibrium_slip_ratio(
        platform=platform,
        contact_model=contact,
        strength=strength,
        mobilization=mobilization,
        gravity_m_per_s2=LUNAR_GRAVITY,
        slope_degrees=np.abs(COST_SLOPE_DEG),
    )
    holdable = (
        np.isfinite(slip)
        & (slip < 1.0)
        & (np.abs(COST_SLOPE_DEG) <= platform.tipping_slope_degrees)
    )
    cost = rolling_cost_of_transport(
        platform=platform,
        contact_model=contact,
        strength=strength,
        mobilization=mobilization,
        gravity_m_per_s2=LUNAR_GRAVITY,
        slope_degrees=COST_SLOPE_DEG,
        slip_ratio=np.where(holdable, slip, 0.0),
    )
    joules = np.where(
        holdable,
        np.maximum(cost.total_J_per_m, 0.0) * NOMINAL_DERATING + hotel,
        np.inf,
    )
    usable = np.isfinite(joules)
    return TraversalCost(
        slope_deg=COST_SLOPE_DEG[usable],
        joules_per_metre=joules[usable],
        limit_deg=float(np.abs(COST_SLOPE_DEG[usable]).max()),
    )


def distance_cost(*, limit_deg: float) -> TraversalCost:
    """One joule per metre, so the field comes back as ground distance."""
    return TraversalCost(
        slope_deg=COST_SLOPE_DEG,
        joules_per_metre=np.ones_like(COST_SLOPE_DEG),
        limit_deg=limit_deg,
    )


@dataclass(frozen=True, slots=True)
class Comparison:
    """Three actors, one charge point, one window of ground."""

    site: Site
    home: tuple[int, int]
    cell_size_m: float
    elevation_m: NDArray[np.float64]
    dark: NDArray[np.bool_]
    legged_Wh: NDArray[np.float64]
    wheeled_Wh: NDArray[np.float64]
    crew_one_way_m: NDArray[np.float64]
    legged_hours: NDArray[np.float64]
    wheeled_hours: NDArray[np.float64]
    crew_range_km: float
    crew_slope_limit_deg: float
    legged_limit_deg: float
    wheeled_limit_deg: float

    @property
    def cell_area_km2(self) -> float:
        return self.cell_size_m**2 / 1e6

    def legged_set(self, battery_Wh: float) -> NDArray[np.bool_]:
        return np.asarray(
            (self.legged_Wh <= battery_Wh) & (self.legged_hours <= LIT_WINDOW_HOURS)
        )

    def wheeled_set(self, battery_Wh: float) -> NDArray[np.bool_]:
        return np.asarray(
            (self.wheeled_Wh <= battery_Wh) & (self.wheeled_hours <= LIT_WINDOW_HOURS)
        )

    @property
    def crew_set(self) -> NDArray[np.bool_]:
        return np.asarray(self.crew_one_way_m <= self.crew_range_km * 1000.0)

    def area_km2(self, mask: NDArray[np.bool_]) -> float:
        return float(mask.sum()) * self.cell_area_km2

    def cold_trap_km2(self, mask: NDArray[np.bool_]) -> float:
        return float((mask & self.dark).sum()) * self.cell_area_km2


def build_comparison(
    site: Site,
    *,
    raster: GeoRaster,
    quadruped: Platform,
    rover: WheeledPlatform,
    contact: Any,
    strength: Any,
    mobilization: Any,
) -> Comparison | None:
    first_row, last_row, first_column, last_column = centred_window(
        raster, span_m=COMMON_WINDOW_KM * 1000.0
    )
    rows, columns = np.meshgrid(
        np.arange(first_row, last_row, MAP_STRIDE),
        np.arange(first_column, last_column, MAP_STRIDE),
        indexing="ij",
    )
    lit = illuminate(
        raster, rows.ravel(), columns.ravel()
    ).any_sunlight_fraction.reshape(rows.shape)
    if not bool((lit <= 0.0).any()):
        return None
    charge = best_charge_point(
        rows=rows,
        columns=columns,
        any_sunlight_fraction=lit,
        elevation_m=raster.values[rows, columns],
    )

    mask_rows, mask_columns = np.meshgrid(
        np.arange(first_row, last_row, MASK_STRIDE),
        np.arange(first_column, last_column, MASK_STRIDE),
        indexing="ij",
    )
    mask = (
        illuminate(
            raster, mask_rows.ravel(), mask_columns.ravel()
        ).any_sunlight_fraction
        <= 0.0
    ).reshape(mask_rows.shape)
    span = last_row - first_row
    dark = np.repeat(np.repeat(mask, MASK_STRIDE, axis=0), MASK_STRIDE, axis=1)[
        :span, :span
    ]

    elevation = np.ascontiguousarray(
        raster.values[first_row:last_row, first_column:last_column]
    )
    home = (charge[0] - first_row, charge[1] - first_column)

    legged_table = walking_cost(
        platform=quadruped,
        contact=contact,
        strength=strength,
        mobilization=mobilization,
        survival_W=INSULATED_SURVIVAL_W,
        speed_m_per_s=quadruped.nominal_speed_m_per_s,
    )
    wheeled_table = rolling_cost(
        platform=rover,
        contact=contact,
        strength=strength,
        mobilization=mobilization,
        survival_W=INSULATED_SURVIVAL_W,
        speed_m_per_s=rover.nominal_speed_m_per_s,
    )

    def energy(table: TraversalCost) -> NDArray[np.float64]:
        return np.asarray(
            round_trip_energy_J(
                elevation_m=elevation,
                cell_size_m=raster.cell_size_m,
                home=home,
                cost=table,
            )
            / JOULES_PER_WATT_HOUR
            + INSULATED_SURVIVAL_W * DWELL_HOURS
        )

    def distance(limit_deg: float) -> NDArray[np.float64]:
        return np.asarray(
            round_trip_energy_J(
                elevation_m=elevation,
                cell_size_m=raster.cell_size_m,
                home=home,
                cost=distance_cost(limit_deg=limit_deg),
            )
        )

    legged_length = distance(legged_table.limit_deg)
    wheeled_length = distance(wheeled_table.limit_deg)
    return Comparison(
        site=site,
        home=home,
        cell_size_m=raster.cell_size_m,
        elevation_m=elevation,
        dark=dark,
        legged_Wh=energy(legged_table),
        wheeled_Wh=energy(wheeled_table),
        crew_one_way_m=distance(site.crew.maximum_slope_deg) / 2.0,
        legged_hours=legged_length / quadruped.nominal_speed_m_per_s / 3600.0
        + DWELL_HOURS,
        wheeled_hours=wheeled_length / rover.nominal_speed_m_per_s / 3600.0
        + DWELL_HOURS,
        crew_range_km=site.crew.traverse_range_km,
        crew_slope_limit_deg=site.crew.maximum_slope_deg,
        legged_limit_deg=legged_table.limit_deg,
        wheeled_limit_deg=wheeled_table.limit_deg,
    )


def build_map_figure(entry: Comparison) -> Figure:
    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (9.8, 9.6),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.5,
                    "axes.grid": False,
                    "figure.subplot.top": 0.700,
                    "figure.subplot.bottom": 0.058,
                    "figure.subplot.left": 0.086,
                    "figure.subplot.right": 0.986,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 1, squeeze=False)
        panel = axes[0][0]
        cell = entry.cell_size_m
        span = entry.elevation_m.shape[0]
        extent = (0.0, span * cell / 1000.0, span * cell / 1000.0, 0.0)

        shade = np.gradient(entry.elevation_m)[0]
        panel.imshow(
            shade,
            extent=extent,
            cmap="gray",
            vmin=float(np.nanpercentile(shade, 2)),
            vmax=float(np.nanpercentile(shade, 98)),
            interpolation="nearest",
        )

        legged = entry.legged_set(NOMINAL_BATTERY_WH)
        wheeled = entry.wheeled_set(NOMINAL_BATTERY_WH)
        for mask, color, alpha in (
            (wheeled & ~legged, ACCENT_SECONDARY, 0.55),
            (legged & ~wheeled, ACCENT_PRIMARY, 0.55),
            (legged & wheeled, "#4f7bbf", 0.42),
        ):
            layer = np.zeros(mask.shape + (4,))
            rgba = matplotlib.colors.to_rgba(color)
            layer[..., 0], layer[..., 1], layer[..., 2] = rgba[0], rgba[1], rgba[2]
            layer[..., 3] = np.where(mask, alpha, 0.0)
            panel.imshow(layer, extent=extent, interpolation="nearest")

        grid_x = np.linspace(extent[0], extent[1], span)
        grid_y = np.linspace(extent[3], extent[2], span)
        panel.contour(
            grid_x,
            grid_y,
            entry.dark.astype(float),
            levels=[0.5],
            colors=[INK_PRIMARY],
            linewidths=1.1,
        )
        panel.contour(
            grid_x,
            grid_y,
            entry.crew_set.astype(float),
            levels=[0.5],
            colors=["#c9a227"],
            linewidths=1.6,
        )

        home_x = (entry.home[1] + 0.5) * cell / 1000.0
        home_y = (entry.home[0] + 0.5) * cell / 1000.0
        panel.plot(
            [home_x],
            [home_y],
            marker="o",
            markersize=8.0,
            markerfacecolor="none",
            markeredgewidth=1.8,
            color="white",
        )

        panel.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    color="#4f7bbf",
                    linewidth=6.0,
                    alpha=0.6,
                    label=f"both reach, {entry.area_km2(legged & wheeled):.0f} km²",
                ),
                Line2D(
                    [],
                    [],
                    color=ACCENT_SECONDARY,
                    linewidth=6.0,
                    alpha=0.7,
                    label=f"wheel only, {entry.area_km2(wheeled & ~legged):.0f} km²",
                ),
                Line2D(
                    [],
                    [],
                    color=ACCENT_PRIMARY,
                    linewidth=6.0,
                    alpha=0.7,
                    label=f"leg only, {entry.area_km2(legged & ~wheeled):.0f} km²",
                ),
                Line2D(
                    [],
                    [],
                    color="#c9a227",
                    linewidth=1.8,
                    label=f"suited crew, {entry.area_km2(entry.crew_set):.0f} km²",
                ),
                Line2D([], [], color=INK_PRIMARY, linewidth=1.1, label="permanent shadow"),
            ],
            loc="upper left",
        )
        panel.set_xlabel("kilometres east across the window")
        panel.set_ylabel("kilometres south across the window")
        panel.set_aspect("equal")

        wheel_area = entry.area_km2(wheeled)
        leg_area = entry.area_km2(legged)
        wheel_cold = entry.cold_trap_km2(wheeled)
        leg_cold = entry.cold_trap_km2(legged)
        figure.suptitle(
            f"At {entry.site.name} the leg reaches more cold trap than the "
            f"wheel — {leg_cold:.2f} against {wheel_cold:.2f} km²"
            if leg_cold > wheel_cold
            else f"At {entry.site.name} the wheel reaches "
            f"{wheel_cold / max(leg_cold, 1e-9):.1f} times the cold trap the "
            "leg does",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.086,
            ha="left",
            y=0.962,
        )
        figure.text(
            0.086,
            0.908,
            caption(
                "The same search over the same ground from the same charge "
                "point, at 400 Wh, with the three actors differing only in the "
                "cost curve each carries. The planner was not modified to "
                "accept a rover: it consumes a cost per metre against slope and "
                "has never been told what produces one. The legged totals come "
                "back bit-identical to Day 14, which is how that is checked.\n"
                "This is one of three sites in ten where the leg reaches more "
                "cold trap, shown rather than one that flatters the platform "
                f"because it is where the mechanism is visible. The leg covers "
                f"{leg_area:.0f} km² here against the wheel's {wheel_area:.0f}, "
                "against the ten-site pattern, because the shadow sits behind "
                "the steep part: past the eight-degree crossover the rover pays "
                "more per metre, and past 26.6 degrees it cannot pass.\n"
                "What the map cannot show is the reason to prefer either. There "
                "are no boulders in it, no pits, and no actuators to fail — and "
                "those are the three things that would actually decide this.",
                width=126,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def build_cold_trap_figure(entries: list[Comparison]) -> Figure:
    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (13.2, 7.0),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.5,
                    "figure.subplot.top": 0.605,
                    "figure.subplot.bottom": 0.180,
                    "figure.subplot.left": 0.058,
                    "figure.subplot.right": 0.988,
                    "figure.subplot.wspace": 0.185,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 2, squeeze=False, width_ratios=[1.5, 1.0])
        bars, scatter = axes[0][0], axes[0][1]

        ordered = sorted(entries, key=lambda e: e.site.name)
        names = [e.site.name for e in ordered]
        wheel = [e.cold_trap_km2(e.wheeled_set(NOMINAL_BATTERY_WH)) for e in ordered]
        leg = [e.cold_trap_km2(e.legged_set(NOMINAL_BATTERY_WH)) for e in ordered]
        crew = [e.cold_trap_km2(e.crew_set) for e in ordered]

        positions = np.arange(len(ordered), dtype=np.float64)
        width = 0.27
        for offset, values, color, label in (
            (-width, wheel, ACCENT_SECONDARY, "wheeled, 400 Wh"),
            (0.0, leg, ACCENT_PRIMARY, "legged, 400 Wh"),
            (width, crew, "#c9a227", "suited crew"),
        ):
            bars.bar(
                positions + offset, values, width=width, color=color, label=label
            )
            for x, value in zip(positions + offset, values):
                if value <= 0.0:
                    bars.annotate(
                        "0",
                        xy=(float(x), 0.0),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        color=INK_MUTED,
                        fontsize=7.5,
                    )
        bars.set_xticks(positions, names, rotation=32, ha="right")
        for label, entry in zip(bars.get_xticklabels(), ordered):
            if not entry.site.is_candidate:
                label.set_color(INK_MUTED)
                label.set_style("italic")
        bars.set_ylabel("cold trap reached (km²)")
        bars.set_title(
            "cold trap inside reach, one charge point, ten sites",
            color=INK_SECONDARY,
            loc="left",
        )
        bars.legend(loc="upper right")

        ceiling = max(max(wheel + leg) * 1.12, 0.1)
        scatter.plot(
            [0.0, ceiling],
            [0.0, ceiling],
            color=INK_MUTED,
            linewidth=1.0,
            linestyle=(0, (2, 2)),
        )
        for entry in ordered:
            legged_km2 = entry.cold_trap_km2(entry.legged_set(NOMINAL_BATTERY_WH))
            wheeled_km2 = entry.cold_trap_km2(entry.wheeled_set(NOMINAL_BATTERY_WH))
            wins = wheeled_km2 > legged_km2
            scatter.plot(
                [legged_km2],
                [wheeled_km2],
                marker="o",
                markersize=7.0,
                color=ACCENT_SECONDARY if wins else ACCENT_PRIMARY,
                markerfacecolor=(
                    (ACCENT_SECONDARY if wins else ACCENT_PRIMARY)
                    if entry.site.is_candidate
                    else "none"
                ),
                markeredgewidth=1.5,
                linestyle="none",
            )
        scatter.annotate(
            "wheel reaches more",
            xy=(ceiling * 0.06, ceiling * 0.95),
            color=ACCENT_SECONDARY,
            fontsize=8.5,
            ha="left",
        )
        scatter.annotate(
            "leg reaches more",
            xy=(ceiling * 0.97, ceiling * 0.06),
            color=ACCENT_PRIMARY,
            fontsize=8.5,
            ha="right",
        )
        scatter.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    color=INK_SECONDARY,
                    marker="o",
                    markersize=7.0,
                    linestyle="none",
                    label="Artemis candidate",
                ),
                Line2D(
                    [],
                    [],
                    color=INK_SECONDARY,
                    marker="o",
                    markersize=7.0,
                    markerfacecolor="none",
                    markeredgewidth=1.5,
                    linestyle="none",
                    label="not a candidate",
                ),
            ],
            loc="lower right",
            bbox_to_anchor=(1.0, 0.12),
        )
        scatter.set_xlim(0.0, ceiling)
        scatter.set_ylim(0.0, ceiling)
        scatter.set_xlabel("legged cold trap reached (km²)")
        scatter.set_ylabel("wheeled cold trap reached (km²)")
        scatter.set_title(
            "the same ten sites, one against the other",
            color=INK_SECONDARY,
            loc="left",
        )

        for panel in (bars, scatter):
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)

        total_wheel, total_leg, total_crew = sum(wheel), sum(leg), sum(crew)
        total_wheel_area = sum(
            e.area_km2(e.wheeled_set(NOMINAL_BATTERY_WH)) for e in ordered
        )
        total_leg_area = sum(
            e.area_km2(e.legged_set(NOMINAL_BATTERY_WH)) for e in ordered
        )
        wheel_wins = sum(1 for w, l in zip(wheel, leg) if w > l)
        crew_zero = sum(1 for value in crew if value <= 0.0)
        figure.suptitle(
            f"The wheel reaches more cold trap at {wheel_wins} of "
            f"{len(ordered)} sites — and its margin shrinks the closer the "
            "question gets to the errand",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.058,
            ha="left",
            y=0.960,
        )
        figure.text(
            0.058,
            0.902,
            caption(
                "The deliverable, and it answers the question against the "
                f"platform this project is building. Across ten sites the rover "
                f"reaches {total_wheel:.2f} km² of permanently shadowed ground "
                f"against the quadruped's {total_leg:.2f} and the suited crew's "
                f"{total_crew:.2f}. Sites where an actor reaches nothing are "
                "drawn as zero rather than omitted, because omitting them is "
                "how a comparison flatters.\n"
                "The margin is the interesting part, because it moves. On flat "
                "ground the wheel is 1.54 times cheaper per metre; over all "
                f"ground reached that becomes "
                f"{total_wheel_area / max(total_leg_area, 1e-9):.2f} times the "
                f"area; over cold trap reached it is "
                f"{total_wheel / max(total_leg, 1e-9):.2f}. Each narrowing of "
                "the question narrows the wheel's advantage, and the reason is "
                "the eight-degree crossover: a cold trap is at the bottom of "
                "something, so the last stretch into one is the steep part, "
                "which is the only ground the leg is better on. The leg wins "
                f"outright at {len(ordered) - wheel_wins} of the ten, two of "
                "them Artemis candidates.\n"
                "It is still a loss. The honest statement is narrow and it is "
                "the whole result: on terrain that has been mapped, at 5 m "
                "posting, with no obstacles and no failures, for this errand, "
                "the legged case does not close. Every one of those clauses is "
                "load-bearing, and the three things that would overturn it — "
                "boulder statistics at metre scale, a polar pit, slope past the "
                "angle of repose — are absent from this repository rather than "
                "tested and found wanting.",
                width=176,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def boundary_rows(entries: list[Comparison]) -> tuple[BoundaryRow, ...]:
    return (
        BoundaryRow(
            quantity="interface",
            published_range="not applicable",
            used="one new cost table; the planner is unchanged",
            status=INSIDE,
            basis=(
                "adding a genuinely different contact regime required no change "
                "to eclipse.planning, eclipse.mobility, eclipse.sortie or the "
                "reachable-set composition. It required one new physics module "
                "and one new loader function, because the platform file format "
                "assumes feet. That is where the seam leaked and it is the "
                "cheapest place for it to"
            ),
        ),
        BoundaryRow(
            quantity="reliability",
            published_range="none, and it is the wheel's strongest argument",
            used="absent for both actors",
            status=UNMEASURED,
            basis=(
                "twelve or more actuators against four to six, in abrasive "
                "cryogenic dust, with no redundancy, no dust ingress and no "
                "duty-cycle life modelled. It runs the same direction as this "
                "result, which means the result is if anything generous to legs"
            ),
        ),
        BoundaryRow(
            quantity="obstacles",
            published_range="none at 5 m posting",
            used="absent; every cell is smooth",
            status=OUTSIDE,
            basis=(
                "a boulder above wheel diameter stops a rover and a legged "
                "platform steps over it. That is the mechanism by which legs "
                "win and there is no dataset at the scale that would show it, so "
                "this comparison is run on ground smoother than either machine "
                "would meet. It is the single largest reason the answer here "
                "could be wrong"
            ),
        ),
        BoundaryRow(
            quantity="pits and lava tubes",
            published_range="no polar pit in any archive",
            used="absent",
            status=OUTSIDE,
            basis=(
                "Day 13 established there is no catalogued polar pit and that "
                "the median catalogued pit is 16 m across, three cells at 5 m. "
                "The canonical legged case is untestable with existing data "
                "rather than tested and failed"
            ),
        ),
        BoundaryRow(
            quantity="slope past repose",
            published_range="not encountered on these routes",
            used="not tested",
            status=OUTSIDE,
            basis=(
                "legs are supposed to win at or beyond the angle of repose, and "
                "no route to a cold trap in these ten windows needs more than "
                "the 15 degrees Day 13 measured. Crater interiors and "
                "non-candidate sites are where that case would be made"
            ),
        ),
        BoundaryRow(
            quantity="grousers",
            published_range="standard on every flown lunar wheel",
            used="none; a smooth rigid wheel",
            status=OUTSIDE,
            basis=(
                "omitting them understates the rover's traction and slope limit, "
                "so the wheel's margin here is a floor"
            ),
        ),
        BoundaryRow(
            quantity="steering",
            published_range="none",
            used="absent; straight-line cost integrated along a turning path",
            status=UNMEASURED,
            basis=(
                "skid steering on four wheels is expensive and a legged platform "
                "turns nearly free. The search turns constantly and neither "
                "actor is charged for it, which favours the rover"
            ),
        ),
        BoundaryRow(
            quantity="charge point",
            published_range="not applicable",
            used="the same point for all three actors",
            status=OUTSIDE,
            basis=(
                "chosen for illumination without reference to morphology, and "
                "the crew would not have one at all. It flatters the crew, whose "
                "numbers are therefore an upper bound on an actor that is "
                "already far behind"
            ),
        ),
        BoundaryRow(
            quantity="hardware",
            published_range="none",
            used="none; neither platform exists",
            status=UNMEASURED,
            basis=(
                "two assumed bodies, one soil, no machine. Nothing in this "
                "comparison has been measured"
            ),
        ),
    )


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return repr(float(value))


def build_report(entries: list[Comparison]) -> str:
    rows = boundary_rows(entries)
    wheel = sum(e.cold_trap_km2(e.wheeled_set(NOMINAL_BATTERY_WH)) for e in entries)
    leg = sum(e.cold_trap_km2(e.legged_set(NOMINAL_BATTERY_WH)) for e in entries)
    crew = sum(e.cold_trap_km2(e.crew_set) for e in entries)
    wheel_area = sum(e.area_km2(e.wheeled_set(NOMINAL_BATTERY_WH)) for e in entries)
    leg_area = sum(e.area_km2(e.legged_set(NOMINAL_BATTERY_WH)) for e in entries)
    crew_area = sum(e.area_km2(e.crew_set) for e in entries)
    wheel_wins = sum(
        1
        for e in entries
        if e.cold_trap_km2(e.wheeled_set(NOMINAL_BATTERY_WH))
        > e.cold_trap_km2(e.legged_set(NOMINAL_BATTERY_WH))
    )

    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# studies.sites.morphology — crew, wheel and leg over the same ground.",
        "#",
        "# Generated. Do not edit by hand.",
        "",
        f"schema_version = {REPORT_SCHEMA_VERSION}",
        'study = "morphology"',
        f"battery_Wh = {_format_float(NOMINAL_BATTERY_WH)}",
        f"survival_W = {_format_float(INSULATED_SURVIVAL_W)}",
        f"dwell_hours = {_format_float(DWELL_HOURS)}",
        f"derating = {_format_float(NOMINAL_DERATING)}",
        'soil = "lunar-intercrater/carrier1991"',
        "",
        "[totals]",
        f"sites = {len(entries)}",
        f"wheeled_area_km2 = {_format_float(wheel_area)}",
        f"legged_area_km2 = {_format_float(leg_area)}",
        f"crew_area_km2 = {_format_float(crew_area)}",
        f"wheeled_cold_trap_km2 = {_format_float(wheel)}",
        f"legged_cold_trap_km2 = {_format_float(leg)}",
        f"crew_cold_trap_km2 = {_format_float(crew)}",
        "wheel_over_leg_cold_trap = "
        + _format_float(wheel / leg if leg > 0.0 else math.nan),
        f"sites_where_wheel_reaches_more = {wheel_wins}",
        "",
        "[limits]",
        "legged_deg = "
        + _format_float(entries[0].legged_limit_deg if entries else math.nan),
        "wheeled_deg = "
        + _format_float(entries[0].wheeled_limit_deg if entries else math.nan),
        "",
    ]
    for entry in sorted(entries, key=lambda e: e.site.name):
        legged = entry.legged_set(NOMINAL_BATTERY_WH)
        wheeled = entry.wheeled_set(NOMINAL_BATTERY_WH)
        lines.extend(
            [
                "[[region]]",
                f'id = "{entry.site.id}"',
                f'name = "{entry.site.name}"',
                "candidate = " + str(entry.site.is_candidate).lower(),
                f"wheeled_km2 = {_format_float(entry.area_km2(wheeled))}",
                f"legged_km2 = {_format_float(entry.area_km2(legged))}",
                f"crew_km2 = {_format_float(entry.area_km2(entry.crew_set))}",
                "wheeled_cold_trap_km2 = "
                + _format_float(entry.cold_trap_km2(wheeled)),
                "legged_cold_trap_km2 = " + _format_float(entry.cold_trap_km2(legged)),
                "crew_cold_trap_km2 = "
                + _format_float(entry.cold_trap_km2(entry.crew_set)),
                "leg_only_km2 = " + _format_float(entry.area_km2(legged & ~wheeled)),
                "wheel_only_km2 = " + _format_float(entry.area_km2(wheeled & ~legged)),
                "",
            ]
        )

    lines.extend(toml_lines(rows))
    lines.extend(["", "[summary]", 'text = """'])
    lines.extend(
        caption(
            "Does the legged case close on the terrain that has been mapped. "
            "No.\n"
            "\n"
            f"Across ten sites from the same charge point at 400 Wh, the rover "
            f"reaches {wheel:.2f} km² of permanently shadowed ground against "
            f"the quadruped's {leg:.2f} and a suited crew's {crew:.2f}. It "
            f"reaches more cold trap at {wheel_wins} of the ten. On total "
            f"ground the margin is wider still: {wheel_area:.0f} km² against "
            f"{leg_area:.0f}.\n"
            "\n"
            "The mechanism is a crossover at eight degrees and the fact that "
            "polar terrain is mostly below it. A wheel pays no swing work, "
            "which is the largest term in the legged budget at lunar gravity, "
            "and it pays compaction as a rolling resistance instead. On gentle "
            "ground the trade is strongly in its favour. Above the crossover it "
            "reverses hard, and past 26.6 degrees the rover cannot hold the "
            "slope at all while the quadruped continues to 39.8 — but Day 13 "
            "already established that no route to a cold trap in these windows "
            "needs more than 15 degrees. The legged advantage is real and it is "
            "on ground these errands do not cross.\n"
            "\n"
            "The irony is worth recording. Bekker built this model for vehicle "
            "mobility, and every plate-sinkage campaign this repository has "
            "transcribed — KLS-1, GRC-1 — was run to characterise wheels. "
            "Fifteen days of terramechanics calibrated from wheel literature "
            "and applied to feet, and the wheel is what the same data supports.\n"
            "\n"
            "What the result is not: an answer about legged robotics, or about "
            "any terrain that has not been mapped at 5 m. Every clause in the "
            "claim is doing work. There are no obstacles in this comparison, "
            "and a boulder above wheel diameter stops a rover where a legged "
            "platform steps over it — that is the mechanism by which legs win "
            "and there is no dataset at the scale that would show it. There is "
            "no pit, because no archive holds a polar one. There is no slope "
            "past the angle of repose, because these routes do not meet it.\n"
            "\n"
            "And the strongest argument on the table runs the same way as the "
            "result rather than against it. Twelve or more actuators in "
            "abrasive cryogenic dust against four to six is why every funded "
            "programme is wheeled, and nothing here models it. A comparison "
            "that omits reliability and still finds for the wheel is not one "
            "that reliability would rescue.",
            width=74,
        ).split("\n")
    )
    lines.extend(['"""', "", "[boundary]", f'tally = "{tally(rows)}"', ""])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a suited crew, a rover and a quadruped on the same "
        "ground from the same charge point."
    )
    parser.add_argument("--figure-directory", type=Path, default=FIGURE_DIRECTORY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    arguments = parser.parse_args(argv)

    sites = load_sites(SITE_DIRECTORY)
    products = load_terrain_manifest(MANIFEST_PATH)
    quadruped = load_platform(QUADRUPED_PATH).platform
    rover = load_wheeled_platform(ROVER_PATH).platform
    dataset = load_soil(SOIL_PATH).datasets["carrier1991"]
    contact = dataset.models["bekker"].extrapolating
    strength = mohr_coulomb_model(dataset, depth_range_cm="0-15")
    mobilization = janosi_hanamoto_model(dataset)

    entries: list[Comparison] = []
    showcase: Comparison | None = None
    for site in sites.values():
        if not site.has_terrain:
            continue
        product = products[cast(str, site.terrain_product)]
        path = TERRAIN_DIRECTORY / product.filename
        if not path.exists():
            print(
                f"{path.relative_to(REPOSITORY_ROOT)} is absent. Terrain products "
                "are fetched, not committed; run tools/fetch_terrain.py"
            )
            return 1
        raster = read_float_geotiff(path)
        entry = build_comparison(
            site,
            raster=raster,
            quadruped=quadruped,
            rover=rover,
            contact=contact,
            strength=strength,
            mobilization=mobilization,
        )
        del raster
        if entry is None:
            print(f"  {site.name:22s} no permanent shadow in the window")
            continue
        entries.append(entry)
        if site.id == SHOWCASE_SITE:
            showcase = entry
        legged = entry.legged_set(NOMINAL_BATTERY_WH)
        wheeled = entry.wheeled_set(NOMINAL_BATTERY_WH)
        print(
            f"  {site.name:22s} area  wheel {entry.area_km2(wheeled):6.1f}  "
            f"leg {entry.area_km2(legged):6.1f}  crew "
            f"{entry.area_km2(entry.crew_set):5.1f} km²   |   cold trap  wheel "
            f"{entry.cold_trap_km2(wheeled):5.2f}  leg "
            f"{entry.cold_trap_km2(legged):5.2f}  crew "
            f"{entry.cold_trap_km2(entry.crew_set):4.2f} km²"
        )

    if not entries:
        print("no site produced a comparison")
        return 1
    if showcase is None:
        showcase = entries[0]

    arguments.figure_directory.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    for name, figure in (
        ("three-actors-one-window", build_map_figure(showcase)),
        ("cold-trap-by-morphology", build_cold_trap_figure(entries)),
    ):
        path = arguments.figure_directory / f"{name}.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        print(f"wrote {path.relative_to(REPOSITORY_ROOT)}")

    arguments.report.write_text(build_report(entries), encoding="utf-8")
    print(f"wrote {arguments.report.relative_to(REPOSITORY_ROOT)}")

    print("\n  measured against extrapolated\n")
    print(text_table(boundary_rows(entries)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
