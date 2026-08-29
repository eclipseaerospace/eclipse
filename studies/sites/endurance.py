# SPDX-License-Identifier: Apache-2.0
#
# studies.sites.endurance — range and endurance, which is what the case turned
# out to be.
#
# Thirteen days killed the gradient argument and produced a better one in its
# place. Every deep errand runs six to nineteen kilometres; a suited crew is held
# to two. Range is the discriminator and always was, and it has been sitting in
# the margins since Day 11 while the requirements envelope stayed
# slope-versus-regions -- an envelope on the one axis that turned out not to
# bind.
#
# But range is not a capability this project models. It is an outcome of four
# things computed separately and never composed: the energy a round trip costs,
# the power insulation demands while the platform is cold, the time a sortie
# takes against the light available, and the gradient it can hold. This composes
# them.
#
# The output is a set, not a distance, and that distinction is the point. A
# legged platform's range is not a radius. It is a lobe shaped by terrain, by
# the asymmetry between climbing and descending, and by where the shadow is --
# and it is drawn against the crew's own reachable set computed the same way,
# over the same ground, with the crew's own slope limit rather than an idealised
# circle.
#
# Four results.
#
# Only two of the four constraints bind, and the day was built expecting three.
# Energy binds everywhere. Slope binds at the edges, through what the search can
# reach at all. Time never binds: the longest sortie in this study is about a
# day against lit runs of three to four weeks, so the charge window that Day 10
# measured is never the limit. And thermal does not bind as a separate condition
# at all -- it enters through the rate, as the power drawn while walking and
# while dwelling, so insulation moves the whole energy field rather than cutting
# a slice out of it.
#
# What endurance buys is about eight times the crew's ground and twenty-two
# times the cold trap, and at two of ten sites the crew reaches no permanently
# shadowed ground at all. That is the case, measured in ground rather than
# degrees, and the cold-trap multiple is the one that matters because it is what
# the mission is for.
#
# And range in the sense of a distance is not a property of a place at all. The
# furthest one-way reach comes out identical at all ten sites -- 6.4 km on the
# nominal platform -- because the furthest a battery goes is set by the cheapest
# ground it could possibly cross, and the cheapest ground is flat everywhere.
# What differs between sites is how much ground and how much shadow lies inside
# that radius, which is exactly why the answer had to be a set rather than a
# number. A project that reported range as a distance would have found ten
# identical sites and concluded terrain does not matter.
#
# Battery moves the frontier further than insulation, which reverses the
# ordering Day 11 found and is worth stating as a reversal rather than quietly.
# Twice the battery multiplies reachable area by 2.5; halving the heat loss
# multiplies it by 1.3; twice the speed by 1.2. The mechanism is that survival
# power is only about a quarter of what a metre costs on flat ground, so halving
# it cuts the cost per metre by an eighth, while doubling the battery doubles
# the budget outright -- and area grows faster than reach because it is roughly a
# disc.
#
# Day 11 is not wrong; it was asking a different question. Insulation decides
# whether one specific journey is affordable at all, because it multiplies that
# journey's cost, and a region opens or does not. Battery decides how much
# ground there is. Both orderings are real and the design lesson is that they
# answer different questions -- reach further, or reach at all.
#
# And the multi-sortie question is answered rather than swept: repeating sorties
# does not extend reach. Every sortie starts and ends at the same charge point,
# so the reachable set is the same set every time, and Day 10's sixty-five to
# eighty-eight sorties per lunation is a coverage number rather than a range
# number. Extending reach needs a cache, a mobile charge point, or surviving a
# night away from home, and none of the three is modelled here.
#
# What this does not carry, stated because the number is large enough to invite
# trust it has not earned: no obstacles, no boulders, no failure modes, no
# operational margin, one soil everywhere, and a reachable set clipped by the
# 16 km analysis window rather than by the platform.
#
# References
#   Rice JW et al. (2023) Artemis III Candidate Landing Region Geology. LPSC.

from __future__ import annotations

import argparse
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
from matplotlib.lines import Line2D
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
    best_charge_point,
    horizon_elevation_deg,
    illumination_fraction,
)
from eclipse.io.platform import load_platform
from eclipse.io.site import Site, load_sites
from eclipse.io.soil import janosi_hanamoto_model, load_soil, mohr_coulomb_model
from eclipse.io.terrain import (
    GeoRaster,
    centred_window,
    latitudes_degrees,
    load_terrain_manifest,
    north_azimuth_degrees,
    read_float_geotiff,
)
from eclipse.mobility import cost_of_transport
from eclipse.planning import TraversalCost, round_trip_energy_J
from eclipse.platform import Platform, equilibrium_slip_ratio, swing_work_per_meter
from eclipse.sortie import JOULES_PER_WATT_HOUR
from eclipse.stance import wave_gait, within_stride_slip_ratio

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
SITE_DIRECTORY: Final = REPOSITORY_ROOT / "configs" / "sites"
TERRAIN_DIRECTORY: Final = REPOSITORY_ROOT / "data" / "terrain"
MANIFEST_PATH: Final = TERRAIN_DIRECTORY / "manifest.toml"
PLATFORM_PATH: Final = (
    REPOSITORY_ROOT / "configs" / "platforms" / "nominal-quadruped.toml"
)
SOIL_PATH: Final = REPOSITORY_ROOT / "data" / "soils" / "lunar-intercrater.toml"
FIGURE_DIRECTORY: Final = Path(__file__).resolve().parent / "figures"
DEFAULT_REPORT_PATH: Final = (
    Path(__file__).resolve().parent / "results" / "endurance.toml"
)

REPORT_SCHEMA_VERSION: Final = 1
LUNAR_GRAVITY: Final = 1.62
FEET_IN_STANCE: Final = 3
NOMINAL_DERATING: Final = 4.0
TIPPING_LIMIT_DEG: Final = 39.8055710922652
COST_SLOPE_DEG: Final[NDArray[np.float64]] = np.arange(-89.0, 89.01, 0.1)

COMMON_WINDOW_KM: Final = 16.0
MAP_STRIDE: Final = 50
HORIZON_AZIMUTHS: Final = 72
HORIZON_SAMPLES: Final = 140
HORIZON_STANDOFF_M: Final = 50.0

# The cold-trap mask is sampled finer than the charge point is chosen, because
# an area is a sum over cells and a 250 m sampling would quantise it coarsely.
# It is still coarser than the 5 m the reachable set is computed on, so
# cold-trap areas carry that resolution and are reported with it.
MASK_STRIDE: Final = 20

# From Day 8 at an effective emissivity of 0.05, and the bracket that day put
# around it. Drawn continuously while walking and throughout the dwell.
INSULATED_SURVIVAL_W: Final = 11.8
INSULATION_SWEEP_W: Final = (5.0, 11.8, 30.0, 100.0)
SPEED_SWEEP_M_PER_S: Final = (0.25, 0.5, 1.0)
DWELL_HOURS: Final = 4.0

# Twice the Day 10 sortie band of 176 to 208 Wh: a platform sized with a hundred
# percent margin on the errand the project has been running. Stated rather than
# optimised, and swept either side of.
NOMINAL_BATTERY_WH: Final = 400.0
BATTERY_SWEEP_WH: Final[NDArray[np.float64]] = np.linspace(50.0, 2000.0, 157)

# The longest lit run Day 10 measured at a polar crest was about three weeks.
# A sortie must fit inside one or the platform is caught out in the dark with a
# battery it has already spent getting there.
LIT_WINDOW_HOURS: Final = 520.8

# What a suited crew is held to, from the site files. Their reachable set is
# computed over the same terrain by the same search, with their own slope limit
# rather than an idealised circle, which is the fair comparison and the one
# that flatters them least.
CREW_ORIGIN: Final = "the same charge point, which the crew would not have"


def caption(text: str, width: int = 148) -> str:
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
    """Cost of transport plus the power drawn while walking, per metre.

    Insulation enters here rather than as a separate constraint, and that is the
    finding rather than a modelling convenience: what worse insulation costs is
    a higher rate, drawn over every metre and every dwell hour, so it moves the
    whole energy field instead of cutting a region out of it.
    """
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
        if abs(float(slope)) > TIPPING_LIMIT_DEG:
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
                platform=platform,
                gravity_m_per_s2=LUNAR_GRAVITY,
                slip_ratio=ratio,
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
            max(
                float(
                    cost.gravitational_J_per_m
                    + cost.shear_J_per_m
                    + cost.compaction_J_per_m
                    + cost.swing_J_per_m
                ),
                0.0,
            )
            * NOMINAL_DERATING
            + hotel
        )
    usable = np.isfinite(joules)
    return TraversalCost(
        slope_deg=COST_SLOPE_DEG[usable],
        joules_per_metre=joules[usable],
        limit_deg=min(float(np.abs(COST_SLOPE_DEG[usable]).max()), TIPPING_LIMIT_DEG),
    )


def distance_cost(*, limit_deg: float) -> TraversalCost:
    """One joule per metre, so the field comes back as ground distance.

    Symmetric by construction -- a metre costs a metre in either direction --
    so a round trip over this table is exactly twice the one way, which is what
    makes the crew's traverse range computable from the same search.
    """
    return TraversalCost(
        slope_deg=COST_SLOPE_DEG,
        joules_per_metre=np.ones_like(COST_SLOPE_DEG),
        limit_deg=limit_deg,
    )


@dataclass(frozen=True, slots=True)
class Frontier:
    """What one platform reaches from one charge point, and what stops it."""

    site: Site
    charge: tuple[int, int]
    home: tuple[int, int]
    cell_size_m: float
    elevation_m: NDArray[np.float64]
    dark: NDArray[np.bool_]
    energy_Wh: NDArray[np.float64]
    one_way_m: NDArray[np.float64]
    crew_one_way_m: NDArray[np.float64]
    sortie_hours: NDArray[np.float64]
    crew_range_km: float
    crew_slope_limit_deg: float

    @property
    def cell_area_km2(self) -> float:
        return self.cell_size_m**2 / 1e6

    @property
    def window_area_km2(self) -> float:
        return float(self.energy_Wh.size) * self.cell_area_km2

    def reachable(self, battery_Wh: float) -> NDArray[np.bool_]:
        return np.asarray(
            (self.energy_Wh <= battery_Wh) & (self.sortie_hours <= LIT_WINDOW_HOURS)
        )

    def area_km2(self, battery_Wh: float) -> float:
        return float(self.reachable(battery_Wh).sum()) * self.cell_area_km2

    def cold_trap_km2(self, battery_Wh: float) -> float:
        return (
            float((self.reachable(battery_Wh) & self.dark).sum()) * self.cell_area_km2
        )

    def furthest_km(self, battery_Wh: float) -> float:
        within = self.reachable(battery_Wh)
        if not bool(within.any()):
            return 0.0
        return float(np.max(self.one_way_m[within])) / 1000.0

    @property
    def crew_set(self) -> NDArray[np.bool_]:
        return np.asarray(self.crew_one_way_m <= self.crew_range_km * 1000.0)

    @property
    def crew_area_km2(self) -> float:
        return float(self.crew_set.sum()) * self.cell_area_km2

    @property
    def crew_cold_trap_km2(self) -> float:
        return float((self.crew_set & self.dark).sum()) * self.cell_area_km2

    @property
    def saturates(self) -> bool:
        """Whether the window clips the frontier rather than the platform."""
        return self.area_km2(float(BATTERY_SWEEP_WH[-1])) >= 0.995 * float(
            np.isfinite(self.energy_Wh).sum()
        ) * self.cell_area_km2

    def time_limited_cells(self, battery_Wh: float) -> int:
        return int(
            ((self.energy_Wh <= battery_Wh) & (self.sortie_hours > LIT_WINDOW_HOURS)).sum()
        )


def build_frontier(
    site: Site,
    *,
    raster: GeoRaster,
    platform: Platform,
    contact: Any,
    strength: Any,
    mobilization: Any,
) -> Frontier | None:
    first_row, last_row, first_column, last_column = centred_window(
        raster, span_m=COMMON_WINDOW_KM * 1000.0
    )
    rows, columns = np.meshgrid(
        np.arange(first_row, last_row, MAP_STRIDE),
        np.arange(first_column, last_column, MAP_STRIDE),
        indexing="ij",
    )
    lit = illuminate(raster, rows.ravel(), columns.ravel()).any_sunlight_fraction.reshape(
        rows.shape
    )
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
        illuminate(raster, mask_rows.ravel(), mask_columns.ravel()).any_sunlight_fraction
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
    cost = walking_cost(
        platform=platform,
        contact=contact,
        strength=strength,
        mobilization=mobilization,
        survival_W=INSULATED_SURVIVAL_W,
        speed_m_per_s=platform.nominal_speed_m_per_s,
    )
    energy = (
        round_trip_energy_J(
            elevation_m=elevation,
            cell_size_m=raster.cell_size_m,
            home=home,
            cost=cost,
        )
        / JOULES_PER_WATT_HOUR
        + INSULATED_SURVIVAL_W * DWELL_HOURS
    )
    length = round_trip_energy_J(
        elevation_m=elevation,
        cell_size_m=raster.cell_size_m,
        home=home,
        cost=distance_cost(limit_deg=cost.limit_deg),
    )
    crew = (
        round_trip_energy_J(
            elevation_m=elevation,
            cell_size_m=raster.cell_size_m,
            home=home,
            cost=distance_cost(limit_deg=site.crew.maximum_slope_deg),
        )
        / 2.0
    )
    return Frontier(
        site=site,
        charge=charge,
        home=home,
        cell_size_m=raster.cell_size_m,
        elevation_m=elevation,
        dark=dark,
        energy_Wh=energy,
        one_way_m=length / 2.0,
        crew_one_way_m=crew,
        sortie_hours=length / platform.nominal_speed_m_per_s / 3600.0 + DWELL_HOURS,
        crew_range_km=site.crew.traverse_range_km,
        crew_slope_limit_deg=site.crew.maximum_slope_deg,
    )


@dataclass(frozen=True, slots=True)
class Sweep:
    """Reachable area against one platform parameter, the others held."""

    axis: str
    values: tuple[float, ...]
    area_km2: tuple[float, ...]
    cold_trap_km2: tuple[float, ...]

    def improvement_from_doubling(self, *, nominal: float, better_is: str) -> float:
        """Area multiplied by one doubling of this axis, from the nominal.

        Comparable across axes only if each is moved in its own improving
        direction by the same factor: twice the battery, twice the speed, half
        the heat loss. Taken from the nominal rather than across the swept
        range, because the range ends are not the same distance from it.
        """
        step = nominal * (0.5 if better_is == "lower" else 2.0)
        values = np.asarray(self.values, dtype=np.float64)
        areas = np.asarray(self.area_km2, dtype=np.float64)
        order = np.argsort(values)
        base = float(np.interp(nominal, values[order], areas[order]))
        if base <= 0.0:
            return float("nan")
        return float(np.interp(step, values[order], areas[order]) / base)


def sweep_platform(
    frontier: Frontier,
    *,
    platform: Platform,
    contact: Any,
    strength: Any,
    mobilization: Any,
    battery_Wh: float,
) -> tuple[Sweep, Sweep, Sweep]:
    """Reachable area against battery, insulation and speed, one axis at a time.

    Battery is free -- it is a threshold on a field already computed. The other
    two are not: both change what a metre costs, so both change which path is
    cheapest and the whole field has to be searched again.
    """
    cell_area = frontier.cell_area_km2
    battery = Sweep(
        axis="battery_Wh",
        values=tuple(float(value) for value in BATTERY_SWEEP_WH),
        area_km2=tuple(frontier.area_km2(float(v)) for v in BATTERY_SWEEP_WH),
        cold_trap_km2=tuple(
            frontier.cold_trap_km2(float(v)) for v in BATTERY_SWEEP_WH
        ),
    )

    def field_for(survival_W: float, speed: float) -> NDArray[np.float64]:
        cost = walking_cost(
            platform=platform,
            contact=contact,
            strength=strength,
            mobilization=mobilization,
            survival_W=survival_W,
            speed_m_per_s=speed,
        )
        return (
            round_trip_energy_J(
                elevation_m=frontier.elevation_m,
                cell_size_m=frontier.cell_size_m,
                home=frontier.home,
                cost=cost,
            )
            / JOULES_PER_WATT_HOUR
            + survival_W * DWELL_HOURS
        )

    areas, cold = [], []
    for survival_W in INSULATION_SWEEP_W:
        field = (
            frontier.energy_Wh
            if survival_W == INSULATED_SURVIVAL_W
            else field_for(survival_W, platform.nominal_speed_m_per_s)
        )
        within = field <= battery_Wh
        areas.append(float(within.sum()) * cell_area)
        cold.append(float((within & frontier.dark).sum()) * cell_area)
    insulation = Sweep(
        axis="survival_W",
        values=INSULATION_SWEEP_W,
        area_km2=tuple(areas),
        cold_trap_km2=tuple(cold),
    )

    areas, cold = [], []
    for speed in SPEED_SWEEP_M_PER_S:
        field = (
            frontier.energy_Wh
            if speed == platform.nominal_speed_m_per_s
            else field_for(INSULATED_SURVIVAL_W, speed)
        )
        within = field <= battery_Wh
        areas.append(float(within.sum()) * cell_area)
        cold.append(float((within & frontier.dark).sum()) * cell_area)
    speed_sweep = Sweep(
        axis="speed_m_per_s",
        values=SPEED_SWEEP_M_PER_S,
        area_km2=tuple(areas),
        cold_trap_km2=tuple(cold),
    )
    return battery, insulation, speed_sweep


def build_map_figure(frontier: Frontier, *, battery_Wh: float) -> Figure:
    cell = frontier.cell_size_m
    span = frontier.energy_Wh.shape[0]
    extent = (0.0, span * cell / 1000.0, span * cell / 1000.0, 0.0)

    reachable = frontier.reachable(battery_Wh)
    beyond = np.isfinite(frontier.energy_Wh) & ~reachable
    unreachable = ~np.isfinite(frontier.energy_Wh)
    layers = np.zeros(frontier.energy_Wh.shape, dtype=np.float64)
    layers[beyond] = 1.0
    layers[reachable] = 2.0
    layers[unreachable] = np.nan

    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (9.6, 9.2),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.5,
                    "axes.grid": False,
                    "figure.subplot.top": 0.688,
                    "figure.subplot.bottom": 0.058,
                    "figure.subplot.left": 0.082,
                    "figure.subplot.right": 0.986,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 1, squeeze=False)
        panel = axes[0][0]
        shade = np.gradient(frontier.elevation_m)[0]
        panel.imshow(
            shade,
            extent=extent,
            cmap="Greys_r",
            vmin=float(np.percentile(shade, 2)),
            vmax=float(np.percentile(shade, 98)),
            interpolation="bilinear",
        )
        panel.imshow(
            np.where(layers == 2.0, 1.0, np.nan),
            extent=extent,
            cmap="Blues",
            vmin=0.0,
            vmax=1.6,
            interpolation="nearest",
            alpha=0.42,
        )
        panel.imshow(
            np.where(frontier.dark, 1.0, np.nan),
            extent=extent,
            cmap="autumn",
            vmin=0.0,
            vmax=2.4,
            interpolation="nearest",
            alpha=0.55,
        )
        panel.contour(
            np.linspace(extent[0], extent[1], span),
            np.linspace(extent[3], extent[2], span),
            frontier.crew_set.astype(float),
            levels=[0.5],
            colors=[INK_PRIMARY],
            linewidths=1.6,
        )
        panel.contour(
            np.linspace(extent[0], extent[1], span),
            np.linspace(extent[3], extent[2], span),
            reachable.astype(float),
            levels=[0.5],
            colors=[ACCENT_PRIMARY],
            linewidths=1.4,
        )
        home_x = (frontier.home[1] + 0.5) * cell / 1000.0
        home_y = (frontier.home[0] + 0.5) * cell / 1000.0
        panel.plot(
            [home_x], [home_y], marker="o", markersize=8.0, markerfacecolor="none",
            markeredgewidth=1.8, color="white",
        )
        panel.annotate(
            "charge point",
            xy=(home_x, home_y),
            xytext=(10, -16),
            textcoords="offset points",
            color=INK_PRIMARY,
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7,
                  "boxstyle": "round,pad=0.2"},
        )
        handles = [
            Line2D([], [], color=ACCENT_PRIMARY, linewidth=1.6,
                       label=f"the platform on {battery_Wh:.0f} Wh, "
                             f"{frontier.area_km2(battery_Wh):.0f} km²"),
            Line2D([], [], color=INK_PRIMARY, linewidth=1.6,
                       label=f"a suited crew, {frontier.crew_range_km:.0f} km at "
                             f"{frontier.crew_slope_limit_deg:.0f}°, "
                             f"{frontier.crew_area_km2:.0f} km²"),
            Line2D([], [], color=ACCENT_SECONDARY, linewidth=6.0, alpha=0.55,
                       label="permanent shadow"),
        ]
        panel.legend(handles=handles, loc="upper left", framealpha=0.75)
        panel.set_xlabel("kilometres east across the window")
        panel.set_ylabel("kilometres south across the window")
        panel.set_aspect("equal")

        multiple = frontier.area_km2(battery_Wh) / max(frontier.crew_area_km2, 1e-9)
        figure.suptitle(
            f"At {frontier.site.name} endurance buys {multiple:.0f} times the "
            "crew's ground, and the shape of it is not a circle",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.082,
            ha="left",
            y=0.972,
        )
        figure.text(
            0.082,
            0.948,
            caption(
                "Blue is everywhere the platform can reach and return from on "
                f"{battery_Wh:.0f} Wh, composing all four constraints: the "
                "round-trip energy inside the battery, every step inside the "
                "tipping limit, the survival power drawn while walking and "
                "dwelling, and the sortie fitting inside a lit window. Black is "
                "the crew's own set, computed by the same search over the same "
                f"ground with their {frontier.crew_slope_limit_deg:.0f}° limit and "
                f"{frontier.crew_range_km:.0f} km traverse range — a set rather "
                "than a circle, because terrain takes bites out of theirs too.\n"
                "Neither boundary is a radius. Both are lobes, pulled out along "
                "gentle ground and pulled in where the platform would have to "
                "climb back, and the asymmetry between descending and climbing is "
                "what makes the two directions cost differently. Nobody in this "
                "literature draws the set; they quote a range.\n"
                "Orange is permanently shadowed ground, sampled at 100 m and so "
                "coarser than the reachable set it is intersected with. The "
                "comparison flatters the crew twice over: they are given the same "
                "charge point they would not have, and they are compared on slope "
                "and range alone rather than on consumables, dust or walkback.\n"
                "The furthest one-way reach here is "
                f"{frontier.furthest_km(battery_Wh):.1f} km, and it is the same "
                "at all ten sites in this study, because the furthest a battery "
                "goes is set by the cheapest ground it could cross and flat "
                "ground costs the same everywhere. Range as a distance is a "
                "property of the platform. What the terrain decides is how much "
                "ground and how much shadow falls inside it, which is the whole "
                "reason this had to be a set.",
                width=136,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def build_sweep_figure(
    frontier: Frontier,
    *,
    battery: Sweep,
    insulation: Sweep,
    speed: Sweep,
    battery_Wh: float,
    nominal_speed_m_per_s: float,
) -> Figure:
    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (13.0, 6.6),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.600,
                    "figure.subplot.bottom": 0.115,
                    "figure.subplot.left": 0.055,
                    "figure.subplot.right": 0.988,
                    "figure.subplot.wspace": 0.250,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 3, squeeze=False)
        left, middle, right = axes[0][0], axes[0][1], axes[0][2]

        left.plot(
            battery.values, battery.area_km2, color=ACCENT_PRIMARY, linewidth=2.0,
            label="all ground",
        )
        left.plot(
            battery.values, battery.cold_trap_km2, color=ACCENT_SECONDARY,
            linewidth=1.8, linestyle=(0, (4, 2)), label="permanent shadow",
        )
        left.axhline(
            frontier.crew_area_km2, color=INK_PRIMARY, linewidth=1.2,
            linestyle=(0, (2, 2)),
        )
        left.annotate(
            f"a crew reaches {frontier.crew_area_km2:.0f} km²",
            xy=(float(battery.values[-1]), frontier.crew_area_km2),
            xytext=(-6, 6),
            textcoords="offset points",
            ha="right",
            color=INK_PRIMARY,
            fontsize=8.0,
        )
        left.axhline(
            frontier.window_area_km2, color=INK_MUTED, linewidth=1.0,
            linestyle=(0, (1.5, 1.5)),
        )
        left.annotate(
            "the analysis window clips here",
            xy=(float(battery.values[0]), frontier.window_area_km2),
            xytext=(6, -13),
            textcoords="offset points",
            color=INK_MUTED,
            fontsize=8.0,
        )
        left.axvline(battery_Wh, color=INK_SECONDARY, linewidth=1.0, linestyle=(0, (3, 2)))
        left.set_xlabel("battery capacity (Wh)")
        left.set_ylabel("reachable ground (km²)")
        left.set_title("against battery", color=INK_SECONDARY, loc="left")
        left.set_xlim(float(battery.values[0]), float(battery.values[-1]))
        left.set_ylim(0.0, frontier.window_area_km2 * 1.08)
        left.legend(loc="center right")

        for panel, sweep, label, colour in (
            (middle, insulation, "survival power (W)", ACCENT_SECONDARY),
            (right, speed, "walking speed (m/s)", ACCENT_PRIMARY),
        ):
            panel.plot(
                sweep.values, sweep.area_km2, color=colour, linewidth=2.0,
                marker="o", markersize=5.0, label="all ground",
            )
            panel.plot(
                sweep.values, sweep.cold_trap_km2, color=INK_SECONDARY,
                linewidth=1.6, linestyle=(0, (4, 2)), marker="s", markersize=4.0,
                label="permanent shadow",
            )
            panel.axhline(
                frontier.crew_area_km2, color=INK_PRIMARY, linewidth=1.2,
                linestyle=(0, (2, 2)),
            )
            panel.set_xscale("log")
            panel.set_xlabel(label)
            panel.set_ylabel("reachable ground (km²)")
            panel.set_ylim(0.0, frontier.window_area_km2 * 1.08)
            panel.legend(loc="upper right")

        middle.set_title(
            f"against insulation, on {battery_Wh:.0f} Wh",
            color=INK_SECONDARY,
            loc="left",
        )
        right.set_title(
            f"and against speed, on {battery_Wh:.0f} Wh",
            color=INK_SECONDARY,
            loc="left",
        )
        for panel in (left, middle, right):
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)

        battery_gain = battery.improvement_from_doubling(
            nominal=battery_Wh, better_is="higher"
        )
        insulation_gain = insulation.improvement_from_doubling(
            nominal=INSULATED_SURVIVAL_W, better_is="lower"
        )
        speed_gain = speed.improvement_from_doubling(
            nominal=nominal_speed_m_per_s, better_is="higher"
        )
        figure.suptitle(
            f"Twice the battery buys {battery_gain:.1f}× the ground, half the heat "
            f"loss {insulation_gain:.1f}×, twice the speed {speed_gain:.1f}× — and "
            "that reverses Day 11's ordering",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.055,
            ha="left",
            y=0.958,
        )
        figure.text(
            0.055,
            0.900,
            caption(
                "One axis at a time, the others held at the nominal platform. "
                "Battery is a threshold on a field already computed; insulation "
                "and speed are not, because both change what a metre costs and so "
                "change which path is cheapest — every point on those two panels "
                "is a fresh search over ten million cells.\n"
                "Day 11 found insulation moved the envelope further than "
                "battery, and on area the ordering reverses. Survival power is "
                "about a quarter of what a metre costs on flat ground, so halving "
                "the heat loss cuts the cost per metre by an eighth, while "
                "doubling the battery doubles the budget outright — and area "
                "grows faster than reach because it is roughly a disc. Day 11 is "
                "not wrong; it was asking whether one journey is affordable, "
                "which insulation decides because it multiplies that journey's "
                "cost. Battery decides how much ground there is. Reach at all, or "
                "reach further.\n"
                "Speed is the axis Day 10 found had two optima — least energy at "
                "0.16 m/s and most sorties at 0.60 — and reachable area is a "
                "third objective that agrees with neither: it wants the platform "
                "to walk as fast as the gait allows. Faster shortens the sortie "
                "and so cuts the survival power drawn on the way, and that term "
                "outweighs the swing work speed adds. Three objectives, three "
                "different answers, and none of them is the speed the platform "
                "file currently specifies.",
                width=178,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def build_comparison_figure(frontiers: list[Frontier], *, battery_Wh: float) -> Figure:
    ordered = sorted(frontiers, key=lambda f: f.cold_trap_km2(battery_Wh))
    names = [f.site.name for f in ordered]
    positions = np.arange(len(ordered))
    robot = np.asarray([f.cold_trap_km2(battery_Wh) for f in ordered])
    crew = np.asarray([f.crew_cold_trap_km2 for f in ordered])

    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (11.8, 7.0),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.5,
                    "figure.subplot.top": 0.618,
                    "figure.subplot.bottom": 0.098,
                    "figure.subplot.left": 0.168,
                    "figure.subplot.right": 0.986,
                    "figure.subplot.wspace": 0.075,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 2, squeeze=False, width_ratios=[1.0, 1.0])
        left, right = axes[0][0], axes[0][1]

        height = 0.36
        left.barh(
            positions + height / 2,
            [f.area_km2(battery_Wh) for f in ordered],
            height=height,
            color=ACCENT_PRIMARY,
            label=f"the platform on {battery_Wh:.0f} Wh",
        )
        left.barh(
            positions - height / 2,
            [f.crew_area_km2 for f in ordered],
            height=height,
            color=INK_PRIMARY,
            label="a suited crew",
        )
        left.set_yticks(positions, names)
        for label, entry in zip(left.get_yticklabels(), ordered):
            if not entry.site.is_candidate:
                label.set_color(INK_MUTED)
                label.set_style("italic")
        left.set_xlabel("all reachable ground (km²)")
        left.set_title("ground reached", color=INK_SECONDARY, loc="left")
        left.set_xlim(0.0, left.get_xlim()[1] * 1.30)
        left.legend(loc="lower right")

        right.barh(
            positions + height / 2, robot, height=height, color=ACCENT_SECONDARY
        )
        right.barh(positions - height / 2, crew, height=height, color=INK_PRIMARY)
        for index, (value, reference) in enumerate(zip(robot, crew)):
            right.annotate(
                f"{value:.2f}" + (f" vs {reference:.2f}" if reference > 0.0 else " vs 0"),
                xy=(value, index + height / 2),
                xytext=(5, -3),
                textcoords="offset points",
                color=INK_SECONDARY,
                fontsize=7.8,
            )
        right.set_yticks(positions, [""] * len(positions))
        right.set_xlabel("permanently shadowed ground reached (km²)")
        right.set_title(
            "and cold trap reached", color=INK_SECONDARY, loc="left"
        )
        right.set_xlim(0.0, float(robot.max()) * 1.42 if robot.max() > 0 else 1.0)

        for panel in (left, right):
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)

        zero_crew = int((crew <= 0.0).sum())
        total_robot = float(robot.sum())
        total_crew = float(crew.sum())
        figure.suptitle(
            "The platform reaches "
            + (
                f"{total_robot / total_crew:.0f} times the cold trap a crew can"
                if total_crew > 0.0
                else "cold trap a crew reaches none of"
            )
            + f", and at {zero_crew} of {len(ordered)} sites the crew reaches none at all",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.055,
            ha="left",
            y=0.966,
        )
        figure.text(
            0.055,
            0.908,
            caption(
                "Both sets computed by the same search over the same ground, from "
                "the same charge point, differing only in the constraints each "
                "actor carries. Grey italic rows are south pole sites that are "
                "not Artemis III candidate regions.\n"
                "The left panel is the range argument in the units it should "
                "always have been in. The right panel is whether that range "
                "reaches anything worth the trip, and it is the harder test: a "
                "platform whose frontier is mostly lit ground crew could walk to "
                "in an afternoon would have a weaker case than the area alone "
                "suggests.\n"
                "Sites where the crew reaches no cold trap at all are shown as "
                "zero rather than omitted, because a zero is the strongest form "
                "of the comparison and dropping it would flatter the platform. "
                "Cold-trap area is sampled at 100 m and is therefore coarser than "
                "the reachable sets it is intersected with, and every number here "
                "is clipped by the 16 km analysis window rather than by the "
                "platform.",
                width=172,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def boundary_rows(
    frontiers: list[Frontier], *, battery_Wh: float
) -> tuple[BoundaryRow, ...]:
    clipped = [f.site.name for f in frontiers if f.saturates]
    time_limited = sum(f.time_limited_cells(battery_Wh) for f in frontiers)
    return (
        BoundaryRow(
            quantity="reachable set",
            published_range="none",
            used="round-trip energy, slope, survival power and sortie duration",
            status=INSIDE,
            basis=(
                "all four composed per cell rather than applied as a radius. "
                "Range is an outcome of these and this is the first time the "
                "project has computed it as one"
            ),
        ),
        BoundaryRow(
            quantity="binding constraint",
            published_range="not applicable",
            used="energy everywhere, slope at the edges, time nowhere",
            status=INSIDE,
            basis=(
                f"{time_limited} cells in the whole study are excluded by sortie "
                "duration against a lit window. The charge window Day 10 measured "
                "never binds, and thermal is not a separate condition at all -- "
                "it enters as the rate drawn while walking and dwelling"
            ),
        ),
        BoundaryRow(
            quantity="analysis window",
            published_range="NASA: regions are approximately 15 by 15 km",
            used=f"{COMMON_WINDOW_KM:.0f} km centred on each product",
            status=OUTSIDE,
            basis=(
                "the frontier is clipped by the window rather than by the "
                "platform at "
                + (", ".join(clipped) if clipped else "no site")
                + ". Every area at a large battery is therefore a lower bound, "
                "and the curve flattening is the window rather than the terrain"
            ),
        ),
        BoundaryRow(
            quantity="crew comparison",
            published_range="Rice et al. (2023): 20 degrees, 2 km",
            used="the same search, the same origin, their slope limit",
            status=INSIDE,
            basis=(
                "a set rather than a circle, so terrain takes bites out of theirs "
                "too. It flatters them twice: they are given a charge point they "
                f"would not have -- {CREW_ORIGIN} -- and they are compared on "
                "slope and range alone rather than on consumables, walkback, dust "
                "or thermal load"
            ),
        ),
        BoundaryRow(
            quantity="cold trap resolution",
            published_range="not applicable",
            used=f"{MASK_STRIDE * 5} m illumination mask on a 5 m reachable set",
            status=OUTSIDE,
            basis=(
                "the shadow mask is coarser than the set it is intersected with, "
                "so cold-trap areas are quantised at 100 m. Day 13 showed the "
                "nearest cold traps are hundredths of a square kilometre, which is "
                "a few mask cells, so small traps are the ones this resolution "
                "treats worst"
            ),
        ),
        BoundaryRow(
            quantity="multi-sortie range",
            published_range="not applicable",
            used="not modelled; reach is single-sortie",
            status=UNMEASURED,
            basis=(
                "repeating a sortie does not extend reach, because every sortie "
                "starts and ends at the same charge point and the set is the same "
                "set each time. Day 10's sixty-five to eighty-eight sorties per "
                "lunation is coverage rather than range. Extending reach needs a "
                "cache, a mobile charge point or surviving a night away, and none "
                "of the three is modelled"
            ),
        ),
        BoundaryRow(
            quantity="obstacles",
            published_range="none",
            used="absent",
            status=UNMEASURED,
            basis=(
                "the specification Day 6 produced and Day 13 sharpened. Every "
                "square kilometre counted here is counted as though a foot could "
                "be placed anywhere in it, and boulder statistics need imagery "
                "this project does not carry"
            ),
        ),
        BoundaryRow(
            quantity="operational margin",
            published_range="none",
            used="none; the battery is spent to the last watt-hour",
            status=UNMEASURED,
            basis=(
                "a cell is reachable if the round trip exactly fits. No reserve, "
                "no contingency, no failure modes, no degradation over a mission. "
                "Every area here is a ceiling"
            ),
        ),
        BoundaryRow(
            quantity="soil",
            published_range="Carrier et al. (1991) lunar intercrater",
            used="the same soil at every cell of every site",
            status=UNMEASURED,
            basis=(
                "carried forward unchanged. A crater floor is not the same "
                "regolith as a rim and the cost field cannot tell them apart"
            ),
        ),
        BoundaryRow(
            quantity="cold trap range",
            published_range="none",
            used="reachable cold-trap area, not only distance and depth",
            status=INSIDE,
            basis=(
                "the site files have declared this axis populated since Day 11, "
                "when it meant the distance and depth to the nearest trap. That "
                "was the weaker form: it says where the nearest one is, not how "
                "much of it a platform can work. This computes the area reachable "
                "and returnable from, which is what the axis was declared for. "
                "Comms is the one genuinely empty axis remaining"
            ),
        ),
    )


def _format_float(value: float) -> str:
    return repr(float(value))


def build_report(
    frontiers: list[Frontier],
    sweeps: dict[str, tuple[Sweep, Sweep, Sweep]],
    *,
    battery_Wh: float,
    showcase: str,
    nominal_speed_m_per_s: float,
) -> str:
    rows = boundary_rows(frontiers, battery_Wh=battery_Wh)
    total_robot = sum(f.cold_trap_km2(battery_Wh) for f in frontiers)
    total_crew = sum(f.crew_cold_trap_km2 for f in frontiers)
    battery, insulation, speed = sweeps[showcase]
    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# Range and endurance, composed from the four things that make them.",
        "#",
        "# Generated by studies/sites/endurance.py. Do not edit.",
        "#",
        "# FIVE AXES OF SIX. Cold-trap range was declared populated on Day 11 as",
        "# a distance and a depth; this gives it as a reachable area, which is the",
        "# form the axis was declared for. Comms is the one genuinely empty axis.",
        "",
        f"schema_version = {REPORT_SCHEMA_VERSION}",
        "",
        "[environment]",
        f'python = "{host_platform.python_version()}"',
        f'numpy = "{np.__version__}"',
        "",
        "[method]",
        f"common_window_km = {_format_float(COMMON_WINDOW_KM)}",
        f"nominal_battery_Wh = {_format_float(battery_Wh)}",
        f"survival_W = {_format_float(INSULATED_SURVIVAL_W)}",
        f"dwell_hours = {_format_float(DWELL_HOURS)}",
        f"tipping_limit_deg = {_format_float(TIPPING_LIMIT_DEG)}",
        f"lit_window_hours = {_format_float(LIT_WINDOW_HOURS)}",
        f"cold_trap_mask_m = {MASK_STRIDE * 5}",
        'reach = "round trip from a fixed charge point, both directions searched"',
        'crew = "the same search, the same origin, their own slope limit and range"',
        "",
        "# What endurance buys, in ground.",
        "[verdict]",
        f"sites = {len(frontiers)}",
        "robot_ground_km2 = "
        + _format_float(sum(f.area_km2(battery_Wh) for f in frontiers)),
        "crew_ground_km2 = " + _format_float(sum(f.crew_area_km2 for f in frontiers)),
        "robot_cold_trap_km2 = " + _format_float(total_robot),
        "crew_cold_trap_km2 = " + _format_float(total_crew),
        "cold_trap_multiple = "
        + (_format_float(total_robot / total_crew) if total_crew > 0.0 else "inf"),
        "sites_where_a_crew_reaches_no_cold_trap = "
        + str(sum(1 for f in frontiers if f.crew_cold_trap_km2 <= 0.0)),
        "cells_excluded_by_sortie_duration = "
        + str(sum(f.time_limited_cells(battery_Wh) for f in frontiers)),
        'binding = "energy everywhere, slope at the edges, time nowhere"',
        "",
        "# Range as a distance is a property of the platform, not of the place.",
        "# The furthest a battery reaches is set by the cheapest ground it could",
        "# cross, and flat ground costs the same everywhere, so this comes out",
        "# identical at every site. What differs is how much lies inside it.",
        "[reach]",
        "furthest_one_way_km = ["
        + ", ".join(
            _format_float(f.furthest_km(battery_Wh)) for f in frontiers
        )
        + "]",
        "identical_across_sites = "
        + str(
            float(
                np.ptp([f.furthest_km(battery_Wh) for f in frontiers])
            )
            < 0.01
        ).lower(),
        "ground_inside_it_km2 = ["
        + ", ".join(_format_float(f.area_km2(battery_Wh)) for f in frontiers)
        + "]",
        "",
    ]
    for frontier in frontiers:
        lines += [
            "[[region]]",
            f'id = "{frontier.site.id}"',
            f'name = "{frontier.site.name}"',
            "candidate = " + str(frontier.site.is_candidate).lower(),
            f"robot_ground_km2 = {_format_float(frontier.area_km2(battery_Wh))}",
            f"crew_ground_km2 = {_format_float(frontier.crew_area_km2)}",
            "ground_multiple = "
            + _format_float(
                frontier.area_km2(battery_Wh) / max(frontier.crew_area_km2, 1e-9)
            ),
            f"robot_cold_trap_km2 = {_format_float(frontier.cold_trap_km2(battery_Wh))}",
            f"crew_cold_trap_km2 = {_format_float(frontier.crew_cold_trap_km2)}",
            f"furthest_one_way_km = {_format_float(frontier.furthest_km(battery_Wh))}",
            f"window_area_km2 = {_format_float(frontier.window_area_km2)}",
            "window_clips_the_frontier = " + str(frontier.saturates).lower(),
            "",
        ]

    lines += [
        "# One axis at a time, at the showcase site. Battery is a threshold on a",
        "# field; insulation and speed each need the field searched again.",
        f'[sweep]\nsite = "{showcase}"',
        "",
    ]
    for sweep in (battery, insulation, speed):
        step = max(1, len(sweep.values) // 12)
        lines += [
            "[[sweep.axis]]",
            f'name = "{sweep.axis}"',
            "values = ["
            + ", ".join(_format_float(v) for v in sweep.values[::step])
            + "]",
            "area_km2 = ["
            + ", ".join(_format_float(v) for v in sweep.area_km2[::step])
            + "]",
            "cold_trap_km2 = ["
            + ", ".join(_format_float(v) for v in sweep.cold_trap_km2[::step])
            + "]",
            "area_multiple_from_one_doubling = "
            + _format_float(
                sweep.improvement_from_doubling(
                    nominal={
                        "battery_Wh": battery_Wh,
                        "survival_W": INSULATED_SURVIVAL_W,
                        "speed_m_per_s": nominal_speed_m_per_s,
                    }[sweep.axis],
                    better_is="lower" if sweep.axis == "survival_W" else "higher",
                )
            ),
            "",
        ]

    lines += [
        "[answer]",
        'statement = """',
        "Endurance buys ground, and the multiple is larger than anything the",
        "gradient argument ever produced. Eight times the crew's ground and",
        "twenty-two times the cold trap, and at two of ten sites the crew reaches",
        "no permanently shadowed ground at all.",
        "",
        "Range as a distance turned out not to be a property of a place. The",
        "furthest one-way reach is identical at all ten sites, because the",
        "furthest a battery goes is set by the cheapest ground it could cross and",
        "flat ground costs the same everywhere. What terrain decides is how much",
        "lies inside that radius -- a factor of four between the best and worst",
        "sites here -- which is why the answer had to be a set. A study that",
        "reported range as a number would have found ten identical sites and",
        "concluded that terrain does not matter.",
        "",
        "Composed rather than assumed: a cell counts only if the round trip fits",
        "the battery, every step holds inside the tipping limit, the survival",
        "power is paid over the whole walk and the dwell, and the sortie fits a",
        "lit window. The result is a set rather than a radius, and it is not a",
        "circle -- it is pulled out along gentle ground and pulled in wherever the",
        "platform would have to climb back, which is the ascent asymmetry showing",
        "up in a map instead of a number.",
        "",
        "Of the four constraints only two bind. Energy binds everywhere. Slope",
        "binds at the edges, through what the search can reach at all. Time never",
        "binds anywhere in the study: the longest sortie is about a day against",
        "lit runs of three to four weeks, so the charge window Day 10 measured is",
        "not a limit on reach. And thermal is not a separate condition -- it is a",
        "rate, drawn over every metre and every dwell hour, so insulation scales",
        "the whole field rather than cutting a piece out of it.",
        "",
        "The strongest design axis is battery, which reverses Day 11's ordering.",
        "Twice the battery multiplies reachable area by 2.5, halving the heat loss",
        "by 1.3, twice the speed by 1.2. Survival power is about a quarter of what",
        "a metre costs on flat ground, so halving it cuts the cost per metre by an",
        "eighth while doubling the battery doubles the budget, and area grows",
        "faster than reach because it is roughly a disc.",
        "",
        "Day 11 is not wrong and the resolution is worth keeping: insulation",
        "decides whether one journey is affordable, because it multiplies that",
        "journey's cost and a region opens or does not. Battery decides how much",
        "ground there is. Reach at all, or reach further, and the two orderings",
        "answer different questions.",
        "",
        "Reachable area also wants the platform to walk fast, which is a third",
        "objective disagreeing with both of Day 10's: least energy at 0.16 m/s,",
        "most sorties at 0.60, most ground at the gait limit. None of the three is",
        "the speed the platform file specifies.",
        "",
        "The multi-sortie question is answered rather than deferred: repeating",
        "sorties does not extend reach. Every sortie starts and ends at the same",
        "charge point, so the reachable set is the same set each time, and Day",
        "10's sixty-five to eighty-eight sorties per lunation is a coverage number",
        "rather than a range number. The project has been reporting the first when",
        "it meant the second. Extending reach needs a cache, a mobile charge point",
        "or surviving a night away from home, and none of the three is modelled.",
        "",
        "What stops this being the strong claim it looks like: no obstacles, no",
        "margin, no failure modes, one soil, a battery spent to the last",
        "watt-hour, and a frontier clipped by a 16 km window rather than by the",
        "platform. Every area here is a ceiling, and the boulder statistics that",
        "would turn it into a floor are the measurement this project keeps",
        "specifying and cannot make.",
        '"""',
        "",
        f"# {tally(rows)}",
        "",
        *toml_lines(rows),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compose the reachable set and compare it with a crew's."
    )
    parser.add_argument("--figure-directory", type=Path, default=FIGURE_DIRECTORY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    arguments = parser.parse_args(argv)

    sites = load_sites(SITE_DIRECTORY)
    products = load_terrain_manifest(MANIFEST_PATH)
    platform = load_platform(PLATFORM_PATH).platform
    dataset = load_soil(SOIL_PATH).datasets["carrier1991"]
    contact = dataset.models["bekker"].extrapolating
    strength = mohr_coulomb_model(dataset, depth_range_cm="0-15")
    mobilization = janosi_hanamoto_model(dataset)

    frontiers: list[Frontier] = []
    sweeps: dict[str, tuple[Sweep, Sweep, Sweep]] = {}
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
        frontier = build_frontier(
            site,
            raster=raster,
            platform=platform,
            contact=contact,
            strength=strength,
            mobilization=mobilization,
        )
        del raster
        if frontier is None:
            print(f"  {site.name:22s} no permanent shadow in the window")
            continue
        frontiers.append(frontier)
        print(
            f"  {site.name:22s} robot {frontier.area_km2(NOMINAL_BATTERY_WH):6.1f} km² "
            f"crew {frontier.crew_area_km2:5.1f} km² "
            f"(x{frontier.area_km2(NOMINAL_BATTERY_WH) / max(frontier.crew_area_km2, 1e-9):4.1f})  "
            f"cold trap {frontier.cold_trap_km2(NOMINAL_BATTERY_WH):5.2f} vs "
            f"{frontier.crew_cold_trap_km2:4.2f} km²  "
            f"reach {frontier.furthest_km(NOMINAL_BATTERY_WH):5.2f} km"
        )

    if not frontiers:
        print("no site produced a frontier; there is nothing to compare")
        return 1

    # The design sweep is a study at one site rather than ten, because every
    # point on it is a fresh search over ten million cells and the axis it is
    # measuring is a property of the platform rather than of the place.
    candidates = [f for f in frontiers if f.site.is_candidate] or frontiers
    showcase = max(candidates, key=lambda f: f.cold_trap_km2(NOMINAL_BATTERY_WH))
    print(f"\n  sweeping the design axes at {showcase.site.name} ...")
    sweeps[showcase.site.id] = sweep_platform(
        showcase,
        platform=platform,
        contact=contact,
        strength=strength,
        mobilization=mobilization,
        battery_Wh=NOMINAL_BATTERY_WH,
    )

    arguments.figure_directory.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)

    battery, insulation, speed = sweeps[showcase.site.id]
    for name, figure in (
        (
            "reachable-set",
            build_map_figure(showcase, battery_Wh=NOMINAL_BATTERY_WH),
        ),
        (
            "endurance-against-design",
            build_sweep_figure(
                showcase,
                battery=battery,
                insulation=insulation,
                speed=speed,
                battery_Wh=NOMINAL_BATTERY_WH,
                nominal_speed_m_per_s=platform.nominal_speed_m_per_s,
            ),
        ),
        (
            "cold-trap-reached",
            build_comparison_figure(frontiers, battery_Wh=NOMINAL_BATTERY_WH),
        ),
    ):
        target = arguments.figure_directory / f"{name}.png"
        figure.savefig(target, dpi=200)
        plt.close(figure)
        print(f"wrote {target.relative_to(REPOSITORY_ROOT)}")

    arguments.report.write_text(
        build_report(
            frontiers,
            sweeps,
            battery_Wh=NOMINAL_BATTERY_WH,
            showcase=showcase.site.id,
            nominal_speed_m_per_s=platform.nominal_speed_m_per_s,
        ),
        encoding="utf-8",
    )
    print(f"wrote {arguments.report.relative_to(REPOSITORY_ROOT)}")

    print("\n  measured against extrapolated\n")
    print(text_table(boundary_rows(frontiers, battery_Wh=NOMINAL_BATTERY_WH)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
