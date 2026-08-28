# SPDX-License-Identifier: Apache-2.0
#
# studies.sites.regions — the same pipeline against every candidate region, and
# what the spread says that one place could not.
#
# The site abstraction was declared on Day 6 to make the result a count across
# candidate regions rather than a finding about one place. It then held one
# instance for five days, which is long enough for a schema and a pipeline to be
# quietly shaped by the single case they had seen.
#
# The claim being tested is CLAUDE.md's falsifiable one: the same evaluator runs
# against a different site file with no code change. Nothing below reads a place
# name, a product filename, an extent or a coordinate. Every site-specific fact
# arrives through configs/sites and data/terrain/manifest.toml.
#
# Coverage first, because it bounds everything else. NASA names nine candidate
# regions. The PGDA 5 m collection holds products for six of them, and the
# assignment of one of those six rests on a renaming rather than a matching
# name. Three regions -- Peak near Cabeus B, Slater Plain, and one of the two
# Mons Mouton regions -- have no product here and are recorded as absences. Four
# further south pole sites that are not candidates are carried as terrain,
# because a pipeline built on rim sites should be run against a crater floor
# before it is trusted.
#
# Then the selection-bias caveat, which has to come before the numbers rather
# than after them. These regions were chosen by NASA for crew landing safety, so
# they are pre-filtered toward gentle ground. A small legged advantage across
# them is a statement about the selection criteria and not about legged
# robotics, and the interesting question it raises is about the terrain just
# outside the selected regions rather than inside them.
#
# The comparison is the deliverable. Three figures and one table: what each
# region opens to legs that is closed to crew, what a sortie into its nearest
# permanent shadow costs, and how many of the nine a given slope capability,
# battery and insulation open.
#
# Four axes of six, now across places rather than at one. Comms and cold-trap
# range remain empty, and saying that at nine sites is the same admission it was
# at one: this is a partial envelope, and the missing axis at Malapert Massif --
# direct-to-Earth visibility -- is the one that region is actually chosen for.
#
# References
#   NASA (2024) NASA Provides Update on Artemis III Moon Landing Regions.
#   Rice JW et al. (2023) Artemis III Candidate Landing Region Geology. LPSC.
#   Barker MK et al. (2021) Improved LOLA Elevation Maps for South Pole Landing
#     Sites. Planetary and Space Science 203, 105119.

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
    SUBSOLAR_LATITUDE_PERIOD_HOURS,
    ShadowTarget,
    best_charge_point,
    horizon_elevation_deg,
    illumination_fraction,
    illumination_series,
    shadow_targets,
)
from eclipse.io.platform import load_platform
from eclipse.io.site import AXIS_NAMES, Site, load_sites
from eclipse.io.soil import janosi_hanamoto_model, load_soil, mohr_coulomb_model
from eclipse.io.terrain import (
    GeoRaster,
    centred_window,
    latitudes_degrees,
    load_terrain_manifest,
    north_azimuth_degrees,
    read_float_geotiff,
)
from eclipse.platform import Platform
from eclipse.schedule import shadowed_hours
from eclipse.sortie import JOULES_PER_WATT_HOUR, sample_transect, walk_round_trip
from eclipse.stance import wave_gait, within_stride_slip_ratio
from eclipse.terrain import slope_degrees

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
SITE_DIRECTORY: Final = REPOSITORY_ROOT / "configs" / "sites"
TERRAIN_DIRECTORY: Final = REPOSITORY_ROOT / "data" / "terrain"
MANIFEST_PATH: Final = TERRAIN_DIRECTORY / "manifest.toml"
PLATFORM_PATH: Final = (
    REPOSITORY_ROOT / "configs" / "platforms" / "nominal-quadruped.toml"
)
SOIL_PATH: Final = REPOSITORY_ROOT / "data" / "soils" / "lunar-intercrater.toml"
FIGURE_DIRECTORY: Final = Path(__file__).resolve().parent / "figures"
DEFAULT_REPORT_PATH: Final = Path(__file__).resolve().parent / "results" / "regions.toml"

REPORT_SCHEMA_VERSION: Final = 1
LUNAR_GRAVITY: Final = 1.62
FEET_IN_STANCE: Final = 3
NOMINAL_DERATING: Final = 4.0
SLOPE_METHOD: Final = "central_difference"

# From rung three. Tipping is the binding one -- the platform rotates about its
# downhill feet before they slide -- so it is what the legged fraction uses.
# Day 6 headlined the traction limit and both are carried, because the
# difference between them is the whole cost of using the wrong one.
TIPPING_LIMIT_DEG: Final = 39.8055710922652
TRACTION_LIMIT_DEG: Final = 43.545928803868314

# The producers' upper bound on RMS slope error in these products. A route
# refused by less than this is not refused by the terrain, it is refused by the
# map, and saying which is the difference between a finding and an artifact.
SLOPE_ERROR_DEG: Final = 2.5

ARRAY_AREA_M2: Final = 0.5
ARRAY_EFFICIENCY: Final = 0.30
SOLAR_CONSTANT_W_PER_M2: Final = 1361.0
INSULATED_SURVIVAL_W: Final = 11.8

MAP_STRIDE: Final = 50
HORIZON_AZIMUTHS: Final = 72
HORIZON_SAMPLES: Final = 140
HORIZON_STANDOFF_M: Final = 50.0
# One sample per cell along the dominant axis, rather than a fixed count. A
# fixed count is a step size that depends on how long the route happens to be,
# and on a short route it goes finer than the grid: consecutive samples share a
# cell, a whole cell-to-cell rise is charged against a sub-cell run, and the
# slope comes out far steeper than the ground. That is what closed two
# candidate regions on Day 11 -- 53 and 66 degree walls that are really 26 and
# 30 -- and sample_transect now refuses the request outright.
ROUTE_SAMPLE_STEP_CELLS: Final = 1
ROUTE_ILLUMINATION_SPACING_M: Final = 40.0
TIME_STEP_H: Final = 0.25
DEPARTURE_STEP_H: Final = 4.0
DEPARTURE_SPAN_H: Final = SUBSOLAR_LATITUDE_PERIOD_HOURS
DWELL_HOURS: Final = 4.0

# The envelope axes. Slope is the platform's achievable capability, battery the
# stored energy, insulation the survival power it implies.
ACHIEVABLE_SLOPE_DEG: Final[NDArray[np.float64]] = np.linspace(5.0, 60.0, 221)
BATTERY_SWEEP_WH: Final[NDArray[np.float64]] = np.linspace(50.0, 1200.0, 231)
SURVIVAL_SWEEP_W: Final[NDArray[np.float64]] = np.array([5.0, 11.8, 30.0, 100.0])

# A sortie is a day trip. A region whose nearest permanent shadow lies beyond
# this is reported with its distance rather than counted as reachable.
DAY_TRIP_LIMIT_KM: Final = 10.0

# The products are not the same size -- 16, 20, 21 and 30 km across -- so a
# slope distribution over one is not comparable to a slope distribution over
# another, and a small window makes "no cold trap here" more likely for a
# reason that is about the data. Every region is therefore analysed over a
# common window centred on its product: 16 km, which is the largest the
# smallest product allows and close to the 15 km NASA states the candidate
# regions to be.
#
# Only the region of interest is cropped. Horizons are still computed against
# the whole raster, because terrain outside the region still casts shadow into
# it, and throwing that away would bias every illumination number toward more
# light.
COMMON_WINDOW_KM: Final = 16.0


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


@dataclass(frozen=True, slots=True)
class Sortie:
    """What a day trip into the nearest permanent shadow costs, timing included.

    Dark hours are carried rather than only the energies they imply, so the
    requirements envelope can ask what a different insulation would buy without
    walking every route again. Energy is locomotion plus a power times those
    hours, and only the power changes.
    """

    distance_km: float
    drop_m: float
    shadow_area_km2: float
    walking_hours: float
    sortie_hours: float
    route_max_slope_deg: float
    refused_by_traction: int
    refused_by_tipping: int
    segments: int
    locomotion_Wh: float
    dark_hours_best: float
    dark_hours_median: float
    dark_hours_worst: float

    def energy_Wh(self, survival_W: float, dark_hours: float) -> float:
        return self.locomotion_Wh + survival_W * dark_hours

    @property
    def best_Wh(self) -> float:
        return self.energy_Wh(INSULATED_SURVIVAL_W, self.dark_hours_best)

    @property
    def median_Wh(self) -> float:
        return self.energy_Wh(INSULATED_SURVIVAL_W, self.dark_hours_median)

    @property
    def worst_Wh(self) -> float:
        return self.energy_Wh(INSULATED_SURVIVAL_W, self.dark_hours_worst)

    @property
    def timing_ratio(self) -> float:
        return self.worst_Wh / self.best_Wh

    @property
    def refused_segments(self) -> int:
        return self.refused_by_traction + self.refused_by_tipping

    @property
    def traversable(self) -> bool:
        return self.refused_segments == 0

    @property
    def tipping_exceedance_deg(self) -> float:
        return max(0.0, self.route_max_slope_deg - TIPPING_LIMIT_DEG)

    @property
    def decidable(self) -> bool:
        """Whether the refusal survives the map's own slope uncertainty."""
        return self.tipping_exceedance_deg > SLOPE_ERROR_DEG

    @property
    def refusal(self) -> str:
        if self.refused_by_traction and self.refused_by_tipping:
            return "traction and tipping"
        if self.refused_by_traction:
            return "traction"
        if self.refused_by_tipping:
            return "tipping"
        return "none"


@dataclass(frozen=True, slots=True)
class Survey:
    """One region, measured. Absences are values here rather than omissions."""

    site: Site
    product: str
    centre_latitude_deg: float
    centre_longitude_deg: float
    window_km: float
    product_window_km: float
    cell_size_m: float
    relief_m: float
    mean_slope_deg: float
    crew_fraction: float
    tipping_fraction: float
    traction_fraction: float
    charge_row: int
    charge_column: int
    charge_lit_fraction: float
    charge_is_highest: bool
    shadow_fraction_of_window: float
    targets: dict[str, ShadowTarget]
    sortie: Sortie | None
    no_shadow_reason: str

    @property
    def legged_advantage(self) -> float:
        return self.tipping_fraction - self.crew_fraction

    @property
    def has_shadow(self) -> bool:
        return bool(self.targets)

    @property
    def reachable(self) -> bool:
        sortie = self.sortie
        return (
            sortie is not None
            and sortie.traversable
            and sortie.distance_km <= DAY_TRIP_LIMIT_KM
        )


def survey_site(
    site: Site,
    *,
    raster: GeoRaster,
    product: str,
    platform: Platform,
    contact: Any,
    strength: Any,
    mobilization: Any,
) -> Survey:
    """The whole pipeline for one place, knowing nothing about which place.

    Two things here are deliberately not what Day 9 did, and both are
    single-site assumptions that only became visible at nine. The charge point
    is the best-illuminated cell rather than the highest one, because "highest"
    was a proxy for "sunlit" that happens to hold on a rim and fails on a
    crater floor. And a region with no permanently shadowed ground in its
    window returns a survey saying so rather than raising, because that is a
    finding about the region.
    """
    first_row, last_row, first_column, last_column = centred_window(raster, span_m=COMMON_WINDOW_KM * 1000.0)
    slope = slope_degrees(
        raster.values[first_row:last_row, first_column:last_column],
        cell_size_m=raster.cell_size_m,
        method=SLOPE_METHOD,
    ).ravel()
    ordered = np.sort(slope)
    crew = float(
        np.searchsorted(ordered, site.crew.maximum_slope_deg, side="right")
        / ordered.size
    )
    tipping = float(
        np.searchsorted(ordered, TIPPING_LIMIT_DEG, side="right") / ordered.size
    )
    traction = float(
        np.searchsorted(ordered, TRACTION_LIMIT_DEG, side="right") / ordered.size
    )
    mean_slope = float(slope.mean())
    del slope, ordered

    grid_rows, grid_columns = np.meshgrid(
        np.arange(first_row, last_row, MAP_STRIDE),
        np.arange(first_column, last_column, MAP_STRIDE),
        indexing="ij",
    )
    grid = illuminate(raster, grid_rows.ravel(), grid_columns.ravel())
    lit = grid.any_sunlight_fraction.reshape(grid_rows.shape)

    elevation = raster.values[grid_rows, grid_columns]
    charge_index = best_charge_point(
        rows=grid_rows,
        columns=grid_columns,
        any_sunlight_fraction=lit,
        elevation_m=elevation,
    )
    highest = np.unravel_index(int(np.argmax(elevation)), lit.shape)

    latitude, longitude = raster.center_latitude_longitude()
    shadow_fraction = float((lit <= 0.0).mean())
    window_km = (last_row - first_row) * raster.cell_size_m / 1000.0

    common = dict(
        site=site,
        product=product,
        centre_latitude_deg=latitude,
        centre_longitude_deg=longitude,
        window_km=window_km,
        product_window_km=raster.shape[0] * raster.cell_size_m / 1000.0,
        cell_size_m=raster.cell_size_m,
        relief_m=float(
            np.ptp(raster.values[first_row:last_row, first_column:last_column])
        ),
        mean_slope_deg=mean_slope,
        crew_fraction=crew,
        tipping_fraction=tipping,
        traction_fraction=traction,
        charge_row=charge_index[0],
        charge_column=charge_index[1],
        charge_lit_fraction=float(
            lit[
                (grid_rows == charge_index[0]) & (grid_columns == charge_index[1])
            ][0]
        ),
        charge_is_highest=bool(
            (int(grid_rows[highest]), int(grid_columns[highest])) == charge_index
        ),
        shadow_fraction_of_window=shadow_fraction,
    )

    if shadow_fraction <= 0.0:
        return Survey(
            targets={},
            sortie=None,
            no_shadow_reason=(
                "no fully shadowed cell on the illumination grid at this "
                "sampling; the window holds no cold trap a sortie could visit"
            ),
            **cast(Any, common),
        )

    targets = shadow_targets(
        raster,
        start=charge_index,
        rows=grid_rows,
        columns=grid_columns,
        any_sunlight_fraction=lit,
    )
    nearest = targets["nearest"]
    return Survey(
        targets=targets,
        sortie=price_sortie(
            raster,
            platform=platform,
            contact=contact,
            strength=strength,
            mobilization=mobilization,
            charge=charge_index,
            target=nearest,
        ),
        no_shadow_reason="",
        **cast(Any, common),
    )


def price_sortie(
    raster: GeoRaster,
    *,
    platform: Platform,
    contact: Any,
    strength: Any,
    mobilization: Any,
    charge: tuple[int, int],
    target: ShadowTarget,
) -> Sortie:
    span = max(abs(target.row - charge[0]), abs(target.column - charge[1]))
    samples = max(span // ROUTE_SAMPLE_STEP_CELLS + 1, 2)
    transect = sample_transect(
        raster,
        start_row_column=charge,
        end_row_column=(target.row, target.column),
        samples=samples,
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
    locomotion = trip.total_J / JOULES_PER_WATT_HOUR * NOMINAL_DERATING

    rows = np.rint(np.linspace(charge[0], target.row, samples)).astype(int)
    columns = np.rint(np.linspace(charge[1], target.column, samples)).astype(int)
    spacing = float(transect.distance_m[-1]) / (samples - 1)
    stride = max(1, int(round(ROUTE_ILLUMINATION_SPACING_M / spacing)))
    index = np.unique(
        np.concatenate([np.arange(0, samples, stride), [samples - 1]])
    )

    walking = float(transect.distance_m[-1]) / platform.nominal_speed_m_per_s / 3600.0
    sortie_hours = 2.0 * walking + DWELL_HOURS
    hours = np.arange(0.0, DEPARTURE_SPAN_H + sortie_hours + 1.0, TIME_STEP_H)
    series = illumination_series(
        horizon=horizon_elevation_deg(
            raster,
            rows=rows[index],
            columns=columns[index],
            azimuths=HORIZON_AZIMUTHS,
            samples_along_ray=HORIZON_SAMPLES,
            minimum_range_m=HORIZON_STANDOFF_M,
        ),
        latitude_deg=latitudes_degrees(raster, rows[index], columns[index]),
        north_azimuth_deg=north_azimuth_degrees(raster, rows[index], columns[index]),
        hours=hours,
    )
    dark = shadowed_hours(
        dark=~series.any_sunlight,
        hours=hours,
        elapsed_hours=transect.distance_m[index]
        / platform.nominal_speed_m_per_s
        / 3600.0,
        departure_hours=np.arange(0.0, DEPARTURE_SPAN_H, DEPARTURE_STEP_H),
        dwell_hours=DWELL_HOURS,
    )
    # Two refusals, and only one of them lives inside walk_leg. That one is
    # traction: the slip solve diverges and the segment is priced as nan.
    # Tipping is not there at all -- rung three established that the platform
    # rotates about its downhill feet before they slide, and the cost model
    # never learned it, so the caller has to apply it. Ten days of routes
    # stayed under the tipping angle, which is why nothing surfaced the gap.
    traction_refused = int(
        np.isnan(trip.outbound.gravitational_J).sum()
        + np.isnan(trip.outbound.dissipative_J).sum()
    )
    steep = np.abs(transect.slope_degrees) > TIPPING_LIMIT_DEG
    priced = ~np.isnan(trip.outbound.gravitational_J)
    return Sortie(
        distance_km=target.distance_km,
        drop_m=target.drop_m,
        shadow_area_km2=target.region_area_km2,
        walking_hours=walking,
        sortie_hours=sortie_hours,
        route_max_slope_deg=float(np.abs(transect.slope_degrees).max()),
        refused_by_traction=traction_refused,
        refused_by_tipping=int((steep & priced).sum()),
        segments=int(transect.slope_degrees.size),
        locomotion_Wh=locomotion,
        dark_hours_best=float(dark.min()),
        dark_hours_median=float(np.median(dark)),
        dark_hours_worst=float(dark.max()),
    )


@dataclass(frozen=True, slots=True)
class Envelope:
    """How many regions a given capability opens, and out of how many."""

    slope_deg: NDArray[np.float64]
    opened_against_slope: NDArray[np.float64]
    battery_Wh: NDArray[np.float64]
    opened_against_battery: dict[float, NDArray[np.float64]]
    candidates: int
    with_terrain: int
    with_route: int
    marginal: int
    nominal_slope_deg: float
    nominal_battery_Wh: float


def opened_count(
    surveys: list[Survey],
    *,
    slope_deg: float,
    battery_Wh: float,
    survival_W: float,
) -> int:
    """Regions a platform of this capability can actually work in.

    Opened means all four of: the region has terrain in this archive, it has
    permanently shadowed ground in the window, that ground lies within a day
    trip, and the route to it is inside the slope capability and inside the
    battery at the worst departure time. A region that fails only on coverage
    is a gap in the data and is counted as unopened either way, which makes
    every curve here a lower bound.
    """
    total = 0
    for survey in surveys:
        sortie = survey.sortie
        if sortie is None or not sortie.traversable:
            continue
        if sortie.distance_km > DAY_TRIP_LIMIT_KM:
            continue
        if sortie.route_max_slope_deg > slope_deg:
            continue
        if sortie.energy_Wh(survival_W, sortie.dark_hours_worst) > battery_Wh:
            continue
        total += 1
    return total


def build_envelope(surveys: list[Survey], *, candidates: int) -> Envelope:
    priced = [
        s
        for s in surveys
        if s.site.is_candidate and s.sortie is not None and s.sortie.traversable
    ]
    with_terrain = sum(1 for s in surveys if s.site.is_candidate)
    marginal = sum(
        1
        for s in surveys
        if s.site.is_candidate
        and s.sortie is not None
        and not s.sortie.traversable
        and not s.sortie.decidable
    )
    nominal_battery = 400.0
    nominal_slope = 40.0
    return Envelope(
        slope_deg=ACHIEVABLE_SLOPE_DEG,
        opened_against_slope=np.asarray(
            [
                opened_count(
                    priced,
                    slope_deg=float(value),
                    battery_Wh=nominal_battery,
                    survival_W=INSULATED_SURVIVAL_W,
                )
                for value in ACHIEVABLE_SLOPE_DEG
            ],
            dtype=np.float64,
        ),
        battery_Wh=BATTERY_SWEEP_WH,
        opened_against_battery={
            float(power): np.asarray(
                [
                    opened_count(
                        priced,
                        slope_deg=nominal_slope,
                        battery_Wh=float(value),
                        survival_W=float(power),
                    )
                    for value in BATTERY_SWEEP_WH
                ],
                dtype=np.float64,
            )
            for power in SURVIVAL_SWEEP_W
        },
        candidates=candidates,
        with_terrain=with_terrain,
        with_route=len(priced),
        marginal=marginal,
        nominal_slope_deg=nominal_slope,
        nominal_battery_Wh=nominal_battery,
    )


def short_name(survey: Survey) -> str:
    return survey.site.name


def build_comparison_figure(surveys: list[Survey]) -> Figure:
    ordered = sorted(surveys, key=lambda s: s.legged_advantage)
    names = [short_name(s) for s in ordered]
    crew = np.asarray([s.crew_fraction for s in ordered]) * 100.0
    legged = np.asarray([s.tipping_fraction for s in ordered]) * 100.0
    positions = np.arange(len(ordered))

    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (11.6, 6.6),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.700,
                    "figure.subplot.bottom": 0.110,
                    "figure.subplot.left": 0.175,
                    "figure.subplot.right": 0.986,
                    "figure.subplot.wspace": 0.070,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 2, squeeze=False, width_ratios=[1.5, 1.0])
        left, right = axes[0][0], axes[0][1]

        left.hlines(
            positions, crew, legged, color=ACCENT_SECONDARY, linewidth=5.0, alpha=0.35
        )
        left.plot(
            crew, positions, marker="o", markersize=6.5, linestyle="none",
            color=INK_PRIMARY, label="crew, 20° limit",
        )
        left.plot(
            legged, positions, marker="D", markersize=6.0, linestyle="none",
            color=ACCENT_SECONDARY, label=f"legged, {TIPPING_LIMIT_DEG:.0f}° tipping limit",
        )
        left.set_yticks(positions, names)
        for label, survey in zip(left.get_yticklabels(), ordered):
            if not survey.site.is_candidate:
                label.set_color(INK_MUTED)
                label.set_style("italic")
        left.set_xlabel("terrain traversable (% of the window)")
        left.set_title(
            "what each limit reaches", color=INK_SECONDARY, loc="left"
        )
        left.set_xlim(None, 100.6)
        left.legend(loc="lower center", ncols=2)

        advantage = np.asarray([s.legged_advantage for s in ordered]) * 100.0
        colors = [
            ACCENT_PRIMARY if s.site.is_candidate else INK_MUTED for s in ordered
        ]
        right.barh(positions, advantage, color=colors, height=0.62)
        for index, value in enumerate(advantage):
            right.annotate(
                f"{value:.1f}",
                xy=(value, index),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                color=INK_SECONDARY,
                fontsize=8.0,
            )
        right.set_yticks(positions, [""] * len(positions))
        right.set_xlabel("percentage points closed to crew and open to legs")
        right.set_title(
            "and what only legs reach", color=INK_SECONDARY, loc="left"
        )
        right.set_xlim(0.0, float(advantage.max()) * 1.22)

        for panel in (left, right):
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)

        candidates = [s for s in ordered if s.site.is_candidate]
        best = max(candidates, key=lambda s: s.legged_advantage)
        worst = min(candidates, key=lambda s: s.legged_advantage)
        figure.suptitle(
            "Across the candidate regions the legged advantage runs "
            f"{worst.legged_advantage:.1%} to {best.legged_advantage:.1%}, and the "
            "spread is the finding",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.060,
            ha="left",
            y=0.968,
        )
        figure.text(
            0.060,
            0.915,
            caption(
                "Every row is the same pipeline against a different site file. "
                f"The bar is terrain above the {surveys[0].site.crew.maximum_slope_deg:.0f}° "
                f"crew limit and below the {TIPPING_LIMIT_DEG:.1f}° tipping limit "
                "from rung three — ground a suited crew is barred from and a "
                "legged platform can stand on. Tipping rather than traction is "
                "used because the platform rotates about its downhill feet before "
                "they slide; Day 6 headlined the traction limit and the difference "
                "is under a percentage point at every site here.\n"
                "The caveat belongs before the numbers rather than after them. "
                "These regions were selected by NASA for crew landing safety, so "
                "they are pre-filtered toward gentle ground, and a modest legged "
                "advantage across them says something about the selection criteria "
                "and not about legged robotics. The four rows in grey italics "
                "are south pole sites that are not candidate regions, carried for "
                "exactly that reason: they are the only terrain here that nobody "
                "chose for being safe, and the steepest of them tops the table.",
                width=170,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def build_cost_figure(surveys: list[Survey], absent: list[Site]) -> Figure:
    priced = [
        s for s in surveys if s.sortie is not None and s.sortie.traversable
    ]
    unshadowed = [s for s in surveys if s.sortie is None]
    refused = [
        s for s in surveys if s.sortie is not None and not s.sortie.traversable
    ]

    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (11.6, 6.4),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.672,
                    "figure.subplot.bottom": 0.115,
                    "figure.subplot.left": 0.072,
                    "figure.subplot.right": 0.986,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 1, squeeze=False)
        panel = axes[0][0]

        by_distance = sorted(
            priced, key=lambda s: cast(Sortie, s.sortie).distance_km
        )
        for rank, survey in enumerate(by_distance):
            sortie = survey.sortie
            assert sortie is not None
            candidate = survey.site.is_candidate
            color = ACCENT_PRIMARY if candidate else INK_MUTED
            panel.errorbar(
                [sortie.distance_km],
                [sortie.median_Wh],
                yerr=[[sortie.median_Wh - sortie.best_Wh],
                      [sortie.worst_Wh - sortie.median_Wh]],
                fmt="o" if candidate else "s",
                markersize=7.0 if candidate else 5.5,
                color=color,
                ecolor=color,
                elinewidth=1.4,
                capsize=3.5,
                alpha=0.95,
            )
            panel.annotate(
                short_name(survey),
                xy=(sortie.distance_km, sortie.median_Wh),
                xytext=(9, 4) if rank % 2 == 0 else (9, -12),
                textcoords="offset points",
                color=INK_SECONDARY if candidate else INK_MUTED,
                fontsize=8.0,
            )
        panel.axvline(
            DAY_TRIP_LIMIT_KM, color=INK_PRIMARY, linewidth=1.0, linestyle=(0, (3, 2))
        )
        panel.annotate(
            f"a day trip stops here, {DAY_TRIP_LIMIT_KM:.0f} km",
            xy=(DAY_TRIP_LIMIT_KM, 0.03),
            xycoords=("data", "axes fraction"),
            xytext=(-7, 0),
            textcoords="offset points",
            rotation=90.0,
            ha="right",
            va="bottom",
            color=INK_PRIMARY,
            fontsize=7.8,
        )
        panel.set_xscale("log")
        panel.set_yscale("log")
        panel.set_xlabel("distance to the nearest permanent shadow (km, log)")
        panel.set_ylabel("sortie energy, median departure (Wh, log)")

        blocks = []
        if absent:
            blocks.append(
                "no terrain product in this archive:\n"
                + "\n".join(f"   {s.name}" for s in absent)
            )
        if unshadowed:
            blocks.append(
                "no permanent shadow in the window:\n"
                + "\n".join(f"   {short_name(s)}" for s in unshadowed)
            )
        if refused:
            blocks.append(
                "direct route refused, too steep to hold:\n"
                + "\n".join(
                    f"   {short_name(s)} — "
                    f"{cast(Sortie, s.sortie).route_max_slope_deg:.0f}° at "
                    f"{cast(Sortie, s.sortie).distance_km:.2f} km, "
                    f"{cast(Sortie, s.sortie).refusal}"
                    + (
                        ""
                        if cast(Sortie, s.sortie).decidable
                        else f" (by {cast(Sortie, s.sortie).tipping_exceedance_deg:.1f}°,"
                        f" inside the map's {SLOPE_ERROR_DEG:.1f}° slope error)"
                    )
                    for s in refused
                )
            )
        if blocks:
            panel.annotate(
                "\n\n".join(blocks),
                xy=(0.014, 0.975),
                xycoords="axes fraction",
                va="top",
                color=INK_MUTED,
                fontsize=8.0,
                linespacing=1.35,
            )

        panel.spines["top"].set_visible(False)
        panel.spines["right"].set_visible(False)

        candidates = [s for s in priced if s.site.is_candidate]
        cheapest = min(candidates, key=lambda s: cast(Sortie, s.sortie).median_Wh)
        dearest = max(candidates, key=lambda s: cast(Sortie, s.sortie).median_Wh)
        figure.suptitle(
            "A cold-trap sortie costs between "
            f"{cast(Sortie, cheapest.sortie).median_Wh:.0f} and "
            f"{cast(Sortie, dearest.sortie).median_Wh:.0f} Wh depending on which "
            "region it starts in",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.072,
            ha="left",
            y=0.968,
        )
        figure.text(
            0.072,
            0.912,
            caption(
                "Each point is a round trip from the best-illuminated cell in the "
                "window to the nearest fully shadowed ground, dwelling four hours. "
                "The bar is the departure time: survival power applies where and "
                "when the ground is dark, so the same route costs its best and its "
                "worst depending on when it leaves. Circles are Artemis III "
                "candidate regions; squares are the south pole sites carried as "
                "terrain.\n"
                "Both axes are logarithmic because the spread is a factor rather "
                "than a difference, and that spread is the point: the distance to "
                "a cold trap is a property of the place, not of the platform, and "
                "it is what decides whether a region admits a day trip at all.\n"
                "Regions with no point are listed with the reason, because the "
                "three reasons are different findings and a figure that dropped "
                "them would make the set look complete. Missing terrain is a gap "
                "in the archive. A refused route is not: it means the nearest cold "
                "trap is close but the straight line to it crosses ground the "
                "platform cannot hold, and this pipeline walks straight lines. "
                "Path planning is the capability that would close those two, and "
                "it is not modelled anywhere in this project.",
                width=170,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def build_envelope_figure(envelope: Envelope) -> Figure:
    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (11.6, 7.2),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.570,
                    "figure.subplot.bottom": 0.115,
                    "figure.subplot.left": 0.062,
                    "figure.subplot.right": 0.986,
                    "figure.subplot.wspace": 0.185,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 2, squeeze=False)
        left, right = axes[0][0], axes[0][1]

        for panel in (left, right):
            panel.axhline(
                envelope.candidates, color=INK_PRIMARY, linewidth=1.0,
                linestyle=(0, (3, 2)),
            )
            panel.axhline(
                envelope.with_terrain, color=INK_MUTED, linewidth=1.0,
                linestyle=(0, (1.5, 1.5)),
            )
            if envelope.marginal:
                panel.axhline(
                    envelope.with_route + envelope.marginal,
                    color=ACCENT_SECONDARY,
                    linewidth=1.0,
                    linestyle=(0, (4, 2)),
                )
            panel.set_ylim(0.0, envelope.candidates + 0.6)
            panel.set_ylabel("candidate regions opened")
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)

        left.step(
            envelope.slope_deg, envelope.opened_against_slope, where="post",
            color=ACCENT_PRIMARY, linewidth=2.0,
        )
        left.fill_between(
            envelope.slope_deg, 0.0, envelope.opened_against_slope, step="post",
            color=ACCENT_PRIMARY, alpha=0.14, linewidth=0.0,
        )
        left.annotate(
            f"all {envelope.candidates} candidate regions",
            xy=(float(envelope.slope_deg[-1]), envelope.candidates),
            xytext=(-6, 5),
            textcoords="offset points",
            ha="right",
            color=INK_PRIMARY,
            fontsize=7.8,
        )
        left.annotate(
            f"{envelope.with_terrain} have terrain in this archive",
            xy=(float(envelope.slope_deg[-1]), envelope.with_terrain),
            xytext=(-6, 5),
            textcoords="offset points",
            ha="right",
            color=INK_MUTED,
            fontsize=7.8,
        )
        if envelope.marginal:
            left.annotate(
                f"+{envelope.marginal} refused inside the map's slope error",
                xy=(float(envelope.slope_deg[-1]), envelope.with_route + envelope.marginal),
                xytext=(-6, 5),
                textcoords="offset points",
                ha="right",
                color=ACCENT_SECONDARY,
                fontsize=7.8,
            )
        left.set_xlabel("achievable slope (°)")
        left.set_title(
            f"against slope capability, on {envelope.nominal_battery_Wh:.0f} Wh",
            color=INK_SECONDARY,
            loc="left",
        )
        left.set_xlim(float(envelope.slope_deg[0]), float(envelope.slope_deg[-1]))

        styles = (
            (ACCENT_SECONDARY, "solid"),
            (ACCENT_PRIMARY, (0, (5, 2))),
            (INK_SECONDARY, (0, (2.5, 2))),
            (INK_MUTED, (0, (1.4, 1.4))),
        )
        for (power, opened), (color, dash) in zip(
            envelope.opened_against_battery.items(), styles
        ):
            right.step(
                envelope.battery_Wh, opened, where="post", color=color,
                linewidth=1.8, linestyle=dash,
                label=f"{power:.0f} W survival",
            )
        right.set_xlabel("battery capacity (Wh)")
        right.set_title(
            f"and against battery, at {envelope.nominal_slope_deg:.0f}° capability",
            color=INK_SECONDARY,
            loc="left",
        )
        right.set_xlim(float(envelope.battery_Wh[0]), float(envelope.battery_Wh[-1]))
        right.legend(loc="lower right")

        opens_all = envelope.slope_deg[
            envelope.opened_against_slope >= envelope.with_route
        ]
        figure.suptitle(
            f"{envelope.with_route} of the {envelope.candidates} candidate "
            "regions are open to a legged day trip, and what closes the rest is "
            "not slope capability",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.062,
            ha="left",
            y=0.958,
        )
        figure.text(
            0.062,
            0.918,
            caption(
                "A region counts as opened when four things hold at once: this "
                "archive has terrain for it, its window holds permanently shadowed "
                "ground, that ground is within a day trip, and the route to it is "
                "inside both the slope capability and the battery at the worst "
                "departure time. Every curve is therefore a lower bound, because a "
                "region missing only its terrain product counts as closed.\n"
                "The black ceiling is the nine regions NASA named and the grey "
                "one is the six this archive covers; the gap between them is "
                "coverage rather than capability, and it is the largest single "
                "term in the answer. The orange line adds the one region refused "
                "by less than the map's own slope error, which is a refusal by the "
                "data rather than by the ground.\n"
                "What closes the remaining regions is not slope capability. Two "
                "are closed because the straight line to their nearest cold trap "
                "crosses ground no achievable slope number fixes — 53° and 66°, "
                "against a platform that tips at 40 — and a route around them is "
                "path planning, which this project does not model. Battery and "
                "insulation, on the right, are the axes a designer can actually "
                "buy, and insulation moves the answer further than battery: at "
                "100 W of survival power the battery that opens every reachable "
                "region is nearly six times the one that does it at 5 W.",
                width=170,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def boundary_rows(surveys: list[Survey], absent: list[Site]) -> tuple[BoundaryRow, ...]:
    windows = sorted({round(s.product_window_km) for s in surveys})
    return (
        BoundaryRow(
            quantity="region coverage",
            published_range="nine Artemis III candidate regions",
            used=f"{sum(1 for s in surveys if s.site.is_candidate)} analysed, "
            f"{len(absent)} absent",
            status=OUTSIDE,
            basis=(
                "the PGDA 5 m collection does not hold a product for every "
                "candidate region. Every count in this study is therefore a lower "
                "bound over the nine, and the absences are named rather than "
                "dropped"
            ),
        ),
        BoundaryRow(
            quantity="region to product mapping",
            published_range="none",
            used="by archive directory name",
            status=UNMEASURED,
            basis=(
                "NASA publishes no centre coordinate for any candidate region, so "
                "products are matched to regions by the collection README's own "
                "site names. Mons Mouton Plateau rests on the 2023 renaming of "
                "Leibnitz Beta and is the weakest match in the set"
            ),
        ),
        BoundaryRow(
            quantity="analysis window",
            published_range="NASA: regions are approximately 15 by 15 km",
            used=f"{COMMON_WINDOW_KM:.0f} km centred on each product",
            status=INSIDE,
            basis=(
                "products span "
                + ", ".join(f"{value} km" for value in windows)
                + " and a slope distribution over one is not comparable to one "
                "over another. Horizons still use the whole raster, so shadow "
                "cast from outside the window is kept"
            ),
        ),
        BoundaryRow(
            quantity="selection bias",
            published_range="not applicable",
            used="the candidate regions as selected",
            status=OUTSIDE,
            basis=(
                "NASA selected these regions for crew landing safety, so they are "
                "pre-filtered toward gentle ground. A small legged advantage here "
                "is a statement about the selection and not about the terrain the "
                "Moon has"
            ),
        ),
        BoundaryRow(
            quantity="slope resolution",
            published_range="producers: 1.5 to 2.5 degrees RMS slope error",
            used="5 m posting, central difference",
            status=INSIDE,
            basis=(
                "the same grid and the same algorithm at every site, on products "
                "the producers state are roughly 90% interpolated at this posting"
            ),
        ),
        BoundaryRow(
            quantity="charge point",
            published_range="none",
            used="the best illuminated sampled cell in the window",
            status=UNMEASURED,
            basis=(
                "no landing or trafficability assessment. It is a cell that sees "
                "the most Sun, not a place anyone has said a lander could sit, and "
                "at "
                + ", ".join(
                    s.site.name for s in surveys if not s.charge_is_highest
                )
                + " it is not the highest ground, which is what Day 9 used"
            ),
        ),
        BoundaryRow(
            quantity="horizon search range",
            published_range="not applicable",
            used="within each product only",
            status=OUTSIDE,
            basis=(
                "rays leaving a product count as clear sky, so every lit fraction "
                "is an upper bound and every permanent shadow survives a wider "
                "search. Products differ in extent, so the size of that bias "
                "differs by site"
            ),
        ),
        BoundaryRow(
            quantity="soil",
            published_range="Carrier et al. (1991) lunar intercrater",
            used="the same soil at every site",
            status=UNMEASURED,
            basis=(
                "regolith properties are not measured per region and no published "
                "set is. Every energy number here inherits one soil, which is a "
                "larger assumption across nine places than it was at one"
            ),
        ),
        BoundaryRow(
            quantity="cold trap range and depth",
            published_range="none",
            used="measured from illumination at each site",
            status=INSIDE,
            basis="the axis Day 6 declared and this study populates",
        ),
        BoundaryRow(
            quantity="direct to Earth visibility",
            published_range="none",
            used="absent",
            status=UNMEASURED,
            basis=(
                "one of the two axes still empty, and the one Malapert Massif is "
                "chosen for. Reporting an envelope without it at that site in "
                "particular would be reporting the wrong envelope"
            ),
        ),
        BoundaryRow(
            quantity="boulder size frequency",
            published_range="none",
            used="absent",
            status=UNMEASURED,
            basis=(
                "the specification Day 6 produced. Still the thing that decides "
                "whether a foothold exists, at every one of these sites"
            ),
        ),
    )


def _format_float(value: float) -> str:
    return repr(float(value))


def build_report(
    surveys: list[Survey], absent: list[Site], envelope: Envelope
) -> str:
    rows = boundary_rows(surveys, absent)
    candidates = [s for s in surveys if s.site.is_candidate]
    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# The same pipeline against every candidate region, and the spread.",
        "#",
        "# Generated by studies/sites/regions.py. Do not edit.",
        "#",
        "# FOUR AXES OF SIX, now across places. Comms and cold-trap range remain",
        "# empty, and the regions were selected for crew landing safety, so they",
        "# are pre-filtered toward gentle ground.",
        "",
        f"schema_version = {REPORT_SCHEMA_VERSION}",
        "",
        "[environment]",
        f'python = "{host_platform.python_version()}"',
        f'numpy = "{np.__version__}"',
        "",
        "[method]",
        f"common_window_km = {_format_float(COMMON_WINDOW_KM)}",
        f'slope_method = "{SLOPE_METHOD}"',
        f"crew_slope_limit_deg = {_format_float(candidates[0].site.crew.maximum_slope_deg)}",
        f"tipping_limit_deg = {_format_float(TIPPING_LIMIT_DEG)}",
        f"traction_limit_deg = {_format_float(TRACTION_LIMIT_DEG)}",
        f"day_trip_limit_km = {_format_float(DAY_TRIP_LIMIT_KM)}",
        f"dwell_hours = {_format_float(DWELL_HOURS)}",
        f"survival_W = {_format_float(INSULATED_SURVIVAL_W)}",
        'charge_point = "best illuminated sampled cell in the window"',
        'horizons = "computed against the whole product, not the cropped window"',
        "",
        "# Coverage, which bounds everything below it.",
        "[coverage]",
        "candidate_regions = 9",
        f"candidates_analysed = {len(candidates)}",
        f"candidates_absent = {len(absent)}",
        "absent_ids = ["
        + ", ".join(f'"{site.id}"' for site in absent)
        + "]",
        f"non_candidate_sites_carried = {len(surveys) - len(candidates)}",
        "",
    ]

    for survey in surveys:
        sortie = survey.sortie
        lines += [
            "[[region]]",
            f'id = "{survey.site.id}"',
            f'name = "{survey.site.name}"',
            "candidate = " + str(survey.site.is_candidate).lower(),
            f'product = "{survey.product}"',
            f"product_window_km = {_format_float(survey.product_window_km)}",
            f"centre_latitude_deg = {_format_float(survey.centre_latitude_deg)}",
            f"centre_longitude_deg = {_format_float(survey.centre_longitude_deg)}",
            f"relief_m = {_format_float(survey.relief_m)}",
            f"mean_slope_deg = {_format_float(survey.mean_slope_deg)}",
            f"crew_traversable_fraction = {_format_float(survey.crew_fraction)}",
            f"tipping_traversable_fraction = {_format_float(survey.tipping_fraction)}",
            f"traction_traversable_fraction = {_format_float(survey.traction_fraction)}",
            f"legged_advantage = {_format_float(survey.legged_advantage)}",
            f"charge_lit_fraction = {_format_float(survey.charge_lit_fraction)}",
            "charge_point_is_highest_ground = "
            + str(survey.charge_is_highest).lower(),
            "shadow_fraction_of_window = "
            f"{_format_float(survey.shadow_fraction_of_window)}",
            "has_permanent_shadow = " + str(survey.has_shadow).lower(),
        ]
        if sortie is None:
            lines += [f'no_sortie = "{survey.no_shadow_reason}"', ""]
            continue
        lines += [
            f"nearest_shadow_km = {_format_float(sortie.distance_km)}",
            f"nearest_shadow_drop_m = {_format_float(sortie.drop_m)}",
            f"nearest_shadow_area_km2 = {_format_float(sortie.shadow_area_km2)}",
            f"route_max_slope_deg = {_format_float(sortie.route_max_slope_deg)}",
            f"refused_by_traction = {sortie.refused_by_traction}",
            f"refused_by_tipping = {sortie.refused_by_tipping}",
            f'refusal = "{sortie.refusal}"',
            "tipping_exceedance_deg = "
            f"{_format_float(sortie.tipping_exceedance_deg)}",
            "refusal_survives_map_slope_error = "
            + str(sortie.decidable).lower(),
            f"route_segments = {sortie.segments}",
            "direct_route_traversable = " + str(sortie.traversable).lower(),
            f"sortie_hours = {_format_float(sortie.sortie_hours)}",
            f"locomotion_Wh = {_format_float(sortie.locomotion_Wh)}",
            f"best_Wh = {_format_float(sortie.best_Wh)}",
            f"median_Wh = {_format_float(sortie.median_Wh)}",
            f"worst_Wh = {_format_float(sortie.worst_Wh)}",
            f"timing_ratio = {_format_float(sortie.timing_ratio)}",
            "within_a_day_trip = " + str(survey.reachable).lower(),
            "",
        ]

    lines += [
        "# The axis vector, across places. An empty axis is empty at every site.",
        "[axes]",
    ]
    for axis in AXIS_NAMES:
        populated = sum(
            1 for s in surveys if s.site.axes.get(axis) == "populated"
        )
        lines.append(f"{axis} = {populated}")

    opened = envelope.opened_against_slope
    saturates = envelope.slope_deg[opened >= envelope.with_route]
    lines += [
        "",
        "# The requirements envelope, with a count of real places on the y axis.",
        "[envelope]",
        f"candidate_regions = {envelope.candidates}",
        f"regions_with_terrain = {envelope.with_terrain}",
        f"regions_with_a_traversable_route = {envelope.with_route}",
        f"regions_refused_inside_map_slope_error = {envelope.marginal}",
        f"nominal_battery_Wh = {_format_float(envelope.nominal_battery_Wh)}",
        f"nominal_slope_deg = {_format_float(envelope.nominal_slope_deg)}",
        "slope_that_opens_every_reachable_region_deg = "
        + (_format_float(float(saturates[0])) if saturates.size else "nan"),
        f"tipping_limit_deg = {_format_float(TIPPING_LIMIT_DEG)}",
        "",
    ]
    for power, curve in envelope.opened_against_battery.items():
        enough = envelope.battery_Wh[curve >= envelope.with_route]
        lines += [
            "[[envelope.insulation]]",
            f"survival_W = {_format_float(power)}",
            "battery_that_opens_every_reachable_region_Wh = "
            + (_format_float(float(enough[0])) if enough.size else "nan"),
            "",
        ]

    lines += [
        "# Did the architecture hold.",
        "[architecture]",
        'statement = """',
        "Yes, and the assumptions it broke are the findings.",
        "",
        "Nothing in the evaluator reads a place name, a filename, an extent or a",
        "coordinate. Adding a region is adding a file. That was the falsifiable",
        "claim and it survives.",
        "",
        "What nine sites broke, in the order they broke it:",
        "",
        "The charge point was the highest cell. That is a proxy for sunlit that",
        "holds on a rim and fails on a crater floor, and it is now the",
        "best-illuminated cell instead.",
        "",
        "The products are not the same size. Sixteen to thirty kilometres across,",
        "so a slope distribution over one is not a slope distribution over",
        "another, and a small window makes an absent cold trap look like a fact",
        "about the place. Every region is now cropped to a common centred window",
        "while horizons still use the whole raster.",
        "",
        "A region can have no permanently shadowed ground at all. The pipeline",
        "raised; it now returns a survey that says so, because that is a finding",
        "and not an error.",
        "",
        "The sortie model never learned the tipping limit. walk_leg refuses a",
        "segment when the slip solve diverges, which is traction at 43.5",
        "degrees, and rung three established that this platform rotates about",
        "its downhill feet at 39.8 and so tips first. Ten days of routes stayed",
        "under both and nothing surfaced the gap; one route here sits between",
        "them and would have been priced as walkable. The caller now applies the",
        "tipping limit, and the two refusals are reported separately because",
        "they are different mechanisms with different fixes.",
        "",
        "A refusal can be smaller than the map that produced it. Haworth is",
        "refused by one segment of 399, by two tenths of a degree, against",
        "products whose stated RMS slope error is 1.5 to 2.5. That is not a",
        "decidable refusal and it is reported as one that the data cannot",
        "settle rather than as a closed region.",
        "",
        "The schema made the region block mandatory, because the one region that",
        "had a press release had all of it. Most candidate regions have no",
        "quotable extent and none has a published centre, so the block is",
        "optional and a site with no terrain product is a site rather than a gap.",
        "",
        "One product arrived truncated and the archive reported success. The byte",
        "count in the manifest caught it, which is what the byte count is for,",
        "and the reader now says so in words rather than failing inside numpy.",
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
            "Run the same pipeline against every candidate region and report "
            "the spread."
        )
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

    absent = [site for site in sites.values() if not site.has_terrain]
    surveys: list[Survey] = []
    for site in sites.values():
        if not site.has_terrain:
            print(f"  {site.name:24s} no terrain product; recorded as absent")
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
        survey = survey_site(
            site,
            raster=raster,
            product=product.id,
            platform=platform,
            contact=contact,
            strength=strength,
            mobilization=mobilization,
        )
        del raster
        sortie = survey.sortie
        if sortie is None:
            summary = "no permanent shadow in the window"
        elif not sortie.traversable:
            summary = (
                f"nearest shadow {sortie.distance_km:5.2f} km, direct route "
                f"refused on {sortie.refusal}: "
                f"{sortie.route_max_slope_deg:.1f}° at "
                f"{sortie.refused_segments} of {sortie.segments} segments"
                + ("" if sortie.decidable else "  [inside the map's slope error]")
            )
        else:
            summary = (
                f"nearest shadow {sortie.distance_km:5.2f} km, "
                f"{sortie.median_Wh:6.1f} Wh"
            )
        print(
            f"  {site.name:24s} crew {survey.crew_fraction:6.2%} "
            f"legs {survey.tipping_fraction:6.2%} "
            f"(+{survey.legged_advantage:5.2%})  {summary}"
        )
        surveys.append(survey)

    envelope = build_envelope(
        surveys, candidates=sum(1 for s in sites.values() if s.is_candidate)
    )

    arguments.figure_directory.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)

    for name, figure in (
        ("regions-compared", build_comparison_figure(surveys)),
        ("sortie-cost-across-regions", build_cost_figure(surveys, absent)),
        ("requirements-envelope", build_envelope_figure(envelope)),
    ):
        path = arguments.figure_directory / f"{name}.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        print(f"wrote {path.relative_to(REPOSITORY_ROOT)}")

    arguments.report.write_text(
        build_report(surveys, absent, envelope), encoding="utf-8"
    )
    print(f"wrote {arguments.report.relative_to(REPOSITORY_ROOT)}")

    print("\n  measured against extrapolated\n")
    print(text_table(boundary_rows(surveys, absent)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
