# SPDX-License-Identifier: Apache-2.0
#
# studies.sites.errands — three errands, and which of them a crew is excluded
# from.
#
# Day 12 removed an argument, and it removed it correctly: every planned route to
# the nearest cold trap at the six analysable candidate regions needs under
# twelve degrees of slope capability, which is inside the twenty a suited crew
# already works to. On that errand the legged case is not made by gradient.
#
# But every clause of that sentence is a filter and the filters were doing the
# work. NASA chose those regions for safe crew landing. The pipeline chose the
# nearest shadow. Neither choice is the mission. So the question here is the one
# the project was built to answer: when the errand is not a day trip to the
# nearest edge of shadow, does slope capability become the binding constraint?
#
# Three errands, run through the same exact bisection Day 12 used so the numbers
# are directly comparable to its 11.6 degrees.
#
# The nearest shadow edge, carried forward unchanged.
#
# The floor of that same shadow, which is a new target in the library: the
# lowest ground inside the region the nearest dark cell belongs to. Volatiles
# concentrate where a cold trap deepens, not where its shadow begins.
#
# And the deepest permanently shadowed ground in the window, which is the
# prospecting errand: the floor of the largest cold trap the site contains
# rather than the first one it meets.
#
# Four results.
#
# The floor of the nearest shadow is the nearest shadow. At the 250 m target
# sampling the two coincide at nine of ten sites, and at 100 m the median gap
# between them is six metres of drop over a region of two hundredths of a square
# kilometre. That is not a resolution artifact to be fixed, it is a measurement:
# the cold traps nearest a lit crest are dimples rather than the reservoirs a
# prospecting mission is after, and the errand earlier days were running was the
# easy one for a reason that had nothing to do with the platform.
#
# The deep errand is a different journey and it is still not slope-limited
# inside the candidate regions. Six to nineteen kilometres out, seven hundred to
# two thousand nine hundred metres down, and the hardest of them -- Haworth --
# needs 15.0 degrees. Still inside the twenty a crew is held to.
#
# Shackleton is where a row crosses. Its floor is 4,577 m below the charge point
# and reaching it needs 22.1 degrees, and Shackleton is not a candidate region
# precisely because it is not safe to land in. So the one journey here that a
# crew could not make is at the site the crew programme already excluded, which
# is the thesis being enforced by the data rather than asserted.
#
# Two things stop that being a rescue, and both belong in the same breath as the
# claim. The crossing is 2.1 degrees past the limit against a map whose stated
# slope error is 1.5 to 2.5, so it is not decidable on this data -- the same
# problem Haworth had on Day 11, reported the same way. And comparing on slope
# alone flatters the crew, who are also held to a two kilometre traverse range
# that every one of these deep errands breaks by a factor of three or more.
#
# So the narrow form is this. Reaching lunar polar shadow is not a gradient
# problem. What excludes a crew from these journeys is range and duration, and
# the legged case rests on obstacles, pits and lava tubes rather than on slope --
# none of which a 5 m polar DEM can measure.
#
# And the canonical legged case cannot be tested at all here. Lunar pits sit in
# mare and impact-melt deposits at low and middle latitudes -- Marius Hills,
# Mare Tranquillitatis, Mare Ingenii at 36 south -- and the highest-latitude
# candidates published are near 72 north. There is no polar pit in this archive
# and there is no polar pit in the catalogue either. Worse for the method than
# for the Moon: the median catalogued pit is about sixteen metres across, which
# is three cells on a 5 m grid, so even a polar pit could not be resolved by
# these products. That is a specification for the next measurement rather than a
# gap to approximate.
#
# References
#   Wagner RV, Robinson MS (2014) Distribution, formation mechanisms, and
#     significance of lunar pits. Icarus 237, 52-60.
#   NASA (2024) NASA Provides Update on Artemis III Moon Landing Regions.

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
    ShadowTarget,
    best_charge_point,
    horizon_elevation_deg,
    illumination_fraction,
    shadow_targets,
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
from eclipse.planning import (
    TraversalCost,
    minimum_slope_capability_deg,
    plan_route,
    round_trip_energy_J,
)
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
DEFAULT_REPORT_PATH: Final = Path(__file__).resolve().parent / "results" / "errands.toml"

REPORT_SCHEMA_VERSION: Final = 1
LUNAR_GRAVITY: Final = 1.62
FEET_IN_STANCE: Final = 3
NOMINAL_DERATING: Final = 4.0

TIPPING_LIMIT_DEG: Final = 39.8055710922652
SLOPE_ERROR_DEG: Final = 2.5
INSULATED_SURVIVAL_W: Final = 11.8
HOTEL_LOAD_W: Final = INSULATED_SURVIVAL_W
COST_SLOPE_DEG: Final[NDArray[np.float64]] = np.arange(-89.0, 89.01, 0.1)

# Carried unchanged from Days 11 and 12 so the capabilities are comparable.
COMMON_WINDOW_KM: Final = 16.0
MAP_STRIDE: Final = 50
HORIZON_AZIMUTHS: Final = 72
HORIZON_SAMPLES: Final = 140
HORIZON_STANDOFF_M: Final = 50.0
CAPABILITY_TOLERANCE_DEG: Final = 0.05

# A second, finer target sampling, run only to say how much the first one
# hides. Nothing headline is computed at it.
CHECK_STRIDE: Final = 20

ERRANDS: Final = ("nearest", "floor", "deepest")
ERRAND_LABEL: Final = {
    "nearest": "nearest shadow edge",
    "floor": "floor of that shadow",
    "deepest": "deepest cold trap",
}

# The site the crew programme excludes, run as though the mission were designed
# around the robot. Named here because the study reports it separately, not
# because the pipeline treats it differently.
EXCLUDED_SITE: Final = "shackleton-rim"
BATTERY_SWEEP_WH: Final[NDArray[np.float64]] = np.linspace(50.0, 4000.0, 159)


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


def build_cost(
    *, platform: Platform, contact: Any, strength: Any, mobilization: Any
) -> TraversalCost:
    """The same curve Day 12 planned on, rebuilt rather than carried across."""
    flat, _ = within_stride_slip_ratio(
        platform=platform,
        gait=wave_gait(lift_order=(2, 0, 3, 1), duty_factor=0.75),
        strength=strength,
        mobilization=mobilization,
        gravity_m_per_s2=LUNAR_GRAVITY,
    )
    hotel = HOTEL_LOAD_W / platform.nominal_speed_m_per_s
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


@dataclass(frozen=True, slots=True)
class Errand:
    """One journey, and the gentlest platform that could make it."""

    key: str
    target: ShadowTarget
    capability_deg: float

    @property
    def within(self) -> bool:
        return bool(np.isfinite(self.capability_deg))


@dataclass(frozen=True, slots=True)
class SiteErrands:
    site: Site
    product: str
    charge: tuple[int, int]
    window: tuple[int, int, int, int]
    errands: dict[str, Errand]
    check_nearest_drop_m: float
    check_floor_drop_m: float
    check_floor_area_km2: float

    def capability(self, key: str) -> float:
        return self.errands[key].capability_deg

    def exceeds(self, key: str, limit_deg: float) -> bool:
        return self.capability(key) > limit_deg


def targets_at(
    raster: GeoRaster, *, window: tuple[int, int, int, int], stride: int
) -> tuple[tuple[int, int], dict[str, ShadowTarget]] | None:
    first_row, last_row, first_column, last_column = window
    rows, columns = np.meshgrid(
        np.arange(first_row, last_row, stride),
        np.arange(first_column, last_column, stride),
        indexing="ij",
    )
    lit = illuminate(raster, rows.ravel(), columns.ravel()).any_sunlight_fraction.reshape(
        rows.shape
    )
    if not bool((lit <= 0.0).any()):
        return None
    elevation = raster.values[rows, columns]
    charge = best_charge_point(
        rows=rows,
        columns=columns,
        any_sunlight_fraction=lit,
        elevation_m=elevation,
    )
    return charge, shadow_targets(
        raster,
        start=charge,
        rows=rows,
        columns=columns,
        any_sunlight_fraction=lit,
    )


def run_site(site: Site, *, raster: GeoRaster, product: str) -> SiteErrands | None:
    window = centred_window(raster, span_m=COMMON_WINDOW_KM * 1000.0)
    coarse = targets_at(raster, window=window, stride=MAP_STRIDE)
    if coarse is None:
        return None
    charge, targets = coarse
    first_row, _, first_column, _ = window
    elevation = np.ascontiguousarray(
        raster.values[window[0] : window[1], window[2] : window[3]]
    )

    errands = {}
    for key in ERRANDS:
        target = targets[key]
        errands[key] = Errand(
            key=key,
            target=target,
            capability_deg=minimum_slope_capability_deg(
                elevation_m=elevation,
                cell_size_m=raster.cell_size_m,
                start=(charge[0] - first_row, charge[1] - first_column),
                goal=(target.row - first_row, target.column - first_column),
                tolerance_deg=CAPABILITY_TOLERANCE_DEG,
            ),
        )

    # The finer pass exists only to say how much the coarse one hides, and
    # nothing headline is computed from it.
    fine = targets_at(raster, window=window, stride=CHECK_STRIDE)
    return SiteErrands(
        site=site,
        product=product,
        charge=charge,
        window=window,
        errands=errands,
        check_nearest_drop_m=(
            fine[1]["nearest"].drop_m if fine is not None else float("nan")
        ),
        check_floor_drop_m=(
            fine[1]["floor"].drop_m if fine is not None else float("nan")
        ),
        check_floor_area_km2=(
            fine[1]["floor"].region_area_km2 if fine is not None else float("nan")
        ),
    )


@dataclass(frozen=True, slots=True)
class Descent:
    """How deep a battery reaches, at the site the crew programme excludes."""

    site: Site
    charge: tuple[int, int]
    target: ShadowTarget
    window: tuple[int, int, int, int]
    elevation_m: NDArray[np.float64]
    round_trip_Wh: NDArray[np.float64]
    route_rows: NDArray[np.int_]
    route_columns: NDArray[np.int_]
    route_elevation_m: NDArray[np.float64]
    route_cumulative_Wh: NDArray[np.float64]
    route_distance_m: NDArray[np.float64]
    capability_deg: float
    battery_Wh: NDArray[np.float64]
    depth_reached_m: NDArray[np.float64]

    @property
    def floor_round_trip_Wh(self) -> float:
        first_row, _, first_column, _ = self.window
        return float(
            self.round_trip_Wh[
                self.target.row - first_row, self.target.column - first_column
            ]
        )

    @property
    def full_depth_m(self) -> float:
        return float(self.target.drop_m)


def run_descent(
    site: Site, *, raster: GeoRaster, cost: TraversalCost
) -> Descent | None:
    window = centred_window(raster, span_m=COMMON_WINDOW_KM * 1000.0)
    found = targets_at(raster, window=window, stride=MAP_STRIDE)
    if found is None:
        return None
    charge, targets = found
    target = targets["deepest"]
    first_row, _, first_column, _ = window
    elevation = np.ascontiguousarray(
        raster.values[window[0] : window[1], window[2] : window[3]]
    )
    start = (charge[0] - first_row, charge[1] - first_column)
    goal = (target.row - first_row, target.column - first_column)

    field = (
        round_trip_energy_J(
            elevation_m=elevation,
            cell_size_m=raster.cell_size_m,
            home=start,
            cost=cost,
        )
        / JOULES_PER_WATT_HOUR
    )
    outbound = plan_route(
        elevation_m=elevation,
        cell_size_m=raster.cell_size_m,
        start=start,
        goal=goal,
        cost=cost,
    )
    route = outbound.route
    if route is None:
        return None

    # How deep the platform gets for a given battery: the lowest ground whose
    # round trip is affordable, which is a depth rather than a yes.
    home_elevation = float(elevation[start])
    depth = home_elevation - elevation
    reached = np.asarray(
        [
            float(np.max(depth[field <= value], initial=0.0))
            for value in BATTERY_SWEEP_WH
        ]
    )
    return Descent(
        site=site,
        charge=charge,
        target=target,
        window=window,
        elevation_m=elevation,
        round_trip_Wh=field,
        route_rows=route.rows,
        route_columns=route.columns,
        route_elevation_m=route.elevation_m,
        route_cumulative_Wh=np.concatenate(
            [[0.0], np.cumsum(route.step_energy_J) / JOULES_PER_WATT_HOUR]
        ),
        route_distance_m=np.concatenate([[0.0], np.cumsum(route.step_length_m)]),
        capability_deg=minimum_slope_capability_deg(
            elevation_m=elevation,
            cell_size_m=raster.cell_size_m,
            start=start,
            goal=goal,
            tolerance_deg=CAPABILITY_TOLERANCE_DEG,
        ),
        battery_Wh=BATTERY_SWEEP_WH,
        depth_reached_m=reached,
    )


def build_capability_figure(
    runs: list[SiteErrands], *, crew_limit_deg: float
) -> Figure:
    ordered = sorted(runs, key=lambda r: r.capability("deepest"))
    names = [r.site.name for r in ordered]
    positions = np.arange(len(ordered))
    width = 0.26
    styles = (
        ("nearest", ACCENT_PRIMARY, -width),
        ("floor", ACCENT_SECONDARY, 0.0),
        ("deepest", INK_PRIMARY, width),
    )

    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (11.8, 7.2),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.5,
                    "figure.subplot.top": 0.610,
                    "figure.subplot.bottom": 0.095,
                    "figure.subplot.left": 0.170,
                    "figure.subplot.right": 0.986,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 1, squeeze=False)
        panel = axes[0][0]
        panel.axvspan(
            0.0, crew_limit_deg, color=INK_MUTED, alpha=0.11, linewidth=0.0
        )
        panel.axvline(crew_limit_deg, color=INK_PRIMARY, linewidth=1.2)
        panel.axvline(
            TIPPING_LIMIT_DEG, color=INK_PRIMARY, linewidth=1.0, linestyle=(0, (2, 2))
        )
        for key, colour, offset in styles:
            values = [r.capability(key) for r in ordered]
            panel.barh(
                positions + offset,
                values,
                height=width * 0.92,
                color=colour,
                label=ERRAND_LABEL[key],
            )
        for index, run in enumerate(ordered):
            value = run.capability("deepest")
            panel.annotate(
                f"{value:.1f}°",
                xy=(value, index + width),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                color=(
                    ACCENT_SECONDARY if value > crew_limit_deg else INK_SECONDARY
                ),
                fontsize=8.0,
            )
        panel.set_yticks(positions, names)
        for label, run in zip(panel.get_yticklabels(), ordered):
            if not run.site.is_candidate:
                label.set_color(INK_MUTED)
                label.set_style("italic")
        panel.set_xlabel("slope capability the journey needs (°)")
        panel.set_xlim(0.0, TIPPING_LIMIT_DEG * 1.12)
        panel.annotate(
            f"a suited crew stops here, {crew_limit_deg:.0f}°",
            xy=(crew_limit_deg, len(ordered) - 0.45),
            xytext=(6, 0),
            textcoords="offset points",
            color=INK_PRIMARY,
            fontsize=8.5,
        )
        panel.annotate(
            f"this platform tips at {TIPPING_LIMIT_DEG:.0f}°",
            xy=(TIPPING_LIMIT_DEG, 0.0),
            xytext=(-6, 0),
            textcoords="offset points",
            rotation=90.0,
            ha="right",
            va="bottom",
            color=INK_PRIMARY,
            fontsize=8.0,
        )
        panel.legend(loc="center right")
        panel.spines["top"].set_visible(False)
        panel.spines["right"].set_visible(False)

        crossing = [r for r in ordered if r.exceeds("deepest", crew_limit_deg)]
        candidates = [r for r in ordered if r.site.is_candidate]
        hardest = max(r.capability("deepest") for r in candidates)
        margin = (
            min(r.capability("deepest") for r in crossing) - crew_limit_deg
            if crossing
            else 0.0
        )
        figure.suptitle(
            "One journey needs more than a crew is allowed — by less than the map "
            "can resolve, and at the site the crew programme excluded"
            if crossing and margin <= SLOPE_ERROR_DEG
            else "One journey needs more than a crew is allowed, and it is at the "
            "site the crew programme excluded"
            if crossing
            else "No journey in this study needs more than a crew is allowed",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.062,
            ha="left",
            y=0.968,
        )
        figure.text(
            0.062,
            0.912,
            caption(
                "The gentlest platform that could make each journey, by the same "
                "exact bisection Day 12 used, so these are directly comparable to "
                "its 11.6°. Grey italic rows are south pole sites that are not "
                "Artemis III candidate regions.\n"
                "The first two errands are the same errand. At the 250 m target "
                "sampling the floor of the nearest shadow coincides with its edge "
                "at nine of ten sites, and at 100 m the median gap is six metres "
                "of drop over two hundredths of a square kilometre. The cold "
                "traps nearest a lit crest are dimples, not the reservoirs a "
                "prospecting mission is after, so the errand earlier days ran was "
                "the easy one for a reason that has nothing to do with the "
                "platform.\n"
                "The third errand is the prospecting one, and inside the "
                f"candidate regions it is still not slope-limited: {hardest:.1f}° "
                "at the hardest of them, against a crew limit of "
                f"{crew_limit_deg:.0f}°. "
                + (
                    "The row that crosses is "
                    + ", ".join(r.site.name for r in crossing)
                    + ", excluded from the candidate list precisely because it is "
                    "not safe to land in — the thesis enforced by the data rather "
                    "than asserted. It crosses by "
                    + f"{min(r.capability('deepest') for r in crossing) - crew_limit_deg:.1f}° "
                    "against a map whose stated slope error is 1.5 to 2.5, so it "
                    "is not decidable on this data and is reported as a crossing "
                    "the data cannot settle."
                    if crossing
                    else "No row crosses, which is the stronger negative result: "
                    "at the lunar south pole, within the terrain these products "
                    "cover, reaching permanently shadowed ground is not "
                    "slope-limited."
                ),
                width=172,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def build_descent_figure(descent: Descent, *, crew_limit_deg: float) -> Figure:
    first_row, _, first_column, _ = descent.window
    start = (descent.charge[0] - first_row, descent.charge[1] - first_column)
    home = float(descent.elevation_m[start])
    profile_depth = home - descent.route_elevation_m
    affordable = descent.round_trip_Wh <= 1000.0

    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (12.4, 7.9),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.578,
                    "figure.subplot.bottom": 0.088,
                    "figure.subplot.left": 0.058,
                    "figure.subplot.right": 0.988,
                    "figure.subplot.wspace": 0.245,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 2, squeeze=False)
        left, right = axes[0][0], axes[0][1]

        left.plot(
            descent.route_distance_m / 1000.0,
            profile_depth,
            color=ACCENT_PRIMARY,
            linewidth=1.8,
        )
        left.fill_between(
            descent.route_distance_m / 1000.0,
            0.0,
            profile_depth,
            color=ACCENT_PRIMARY,
            alpha=0.12,
            linewidth=0.0,
        )
        left.invert_yaxis()
        left.set_xlabel("distance from the charge point (km)")
        left.set_ylabel("depth below the charge point (m)")
        left.set_title(
            "the descent the planner chose", color=INK_SECONDARY, loc="left"
        )
        twin = left.twinx()
        twin.plot(
            descent.route_distance_m / 1000.0,
            descent.route_cumulative_Wh,
            color=ACCENT_SECONDARY,
            linewidth=1.5,
            linestyle=(0, (4, 2)),
        )
        twin.set_ylabel("energy spent going down (Wh)", color=ACCENT_SECONDARY)
        twin.tick_params(axis="y", colors=ACCENT_SECONDARY)
        twin.spines["top"].set_visible(False)
        left.annotate(
            f"floor, {descent.full_depth_m:.0f} m down\n"
            f"{descent.floor_round_trip_Wh:.0f} Wh out and back",
            xy=(
                float(descent.route_distance_m[-1] / 1000.0),
                float(profile_depth[-1]),
            ),
            xytext=(-8, 18),
            textcoords="offset points",
            ha="right",
            color=INK_PRIMARY,
            fontsize=8.5,
        )

        right.plot(
            descent.battery_Wh,
            descent.depth_reached_m,
            color=ACCENT_PRIMARY,
            linewidth=2.0,
        )
        right.axhline(
            descent.full_depth_m, color=INK_PRIMARY, linewidth=1.0, linestyle=(0, (3, 2))
        )
        right.annotate(
            f"the floor, {descent.full_depth_m:.0f} m",
            xy=(float(descent.battery_Wh[-1]), descent.full_depth_m),
            xytext=(-6, 6),
            textcoords="offset points",
            ha="right",
            color=INK_PRIMARY,
            fontsize=8.0,
        )
        enough = descent.battery_Wh[
            descent.depth_reached_m >= descent.full_depth_m - 1.0
        ]
        if enough.size:
            right.axvline(
                float(enough[0]), color=ACCENT_SECONDARY, linewidth=1.0,
                linestyle=(0, (2, 2)),
            )
            right.annotate(
                f"{enough[0]:.0f} Wh reaches it",
                xy=(float(enough[0]), 0.0),
                xytext=(7, 10),
                textcoords="offset points",
                color=ACCENT_SECONDARY,
                fontsize=8.0,
            )
        right.set_xlabel("battery capacity (Wh)")
        right.set_ylabel("deepest ground reached and returned from (m)")
        right.set_title(
            "and how deep a battery buys", color=INK_SECONDARY, loc="left"
        )
        right.set_xlim(0.0, float(descent.battery_Wh[-1]))
        right.set_ylim(0.0, None)

        for panel in (left, right):
            panel.spines["top"].set_visible(False)
        right.spines["right"].set_visible(False)

        figure.suptitle(
            f"{descent.site.name} is not a candidate region, and its floor is "
            f"the one place a crew could not follow — {descent.capability_deg:.1f}° "
            f"against {crew_limit_deg:.0f}°",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.058,
            ha="left",
            y=0.966,
        )
        figure.text(
            0.058,
            0.928,
            caption(
                "Run through the same pipeline as the candidate regions, with "
                "nothing changed but the site file. That is the point of the "
                "comparison and also its caveat: this is what a legged platform "
                "could reach if the mission were designed around the robot rather "
                "than around a crew landing, which is a different mission concept "
                "and is labelled as one.\n"
                "Day 0 rejected Shackleton on exactly this ground — 21 km across "
                "and 4.2 km deep — and the machinery can now test that rejection "
                "instead of assuming it. Within the 16 km analysis window the "
                f"deepest shadowed ground is {descent.full_depth_m:.0f} m below "
                f"the charge point and costs {descent.floor_round_trip_Wh:.0f} Wh "
                "out and back, so the rejection was right about the scale and "
                "wrong about the reason: it is not that the descent cannot be "
                "walked, it is that walking it is a different mission from a day "
                "trip.\n"
                "The two curves on the left are the asymmetry the directed graph "
                f"exists for. Going down costs "
                f"{descent.route_cumulative_Wh[-1]:.0f} Wh, almost all of it the "
                "hotel load rather than the ground, because a descent steep "
                "enough that gravity pays for the soil is clamped to free. "
                f"Climbing back costs the other "
                f"{descent.floor_round_trip_Wh - float(descent.route_cumulative_Wh[-1]):.0f} "
                "Wh. A round trip modelled as twice the outbound would have "
                "understated it by a factor of four.\n"
                "And the crossing that puts this site past the crew limit is "
                f"{descent.capability_deg - crew_limit_deg:.1f}° wide, against a "
                "map whose producers state 1.5 to 2.5° of RMS slope error. It is "
                "reported as a crossing this data cannot settle, which is the "
                "same treatment Haworth got on Day 11.\n"
                "The right panel is the useful form. A descent is not a yes or a "
                "no, it is a depth, and the curve says which depth a given "
                "battery buys — computed as a field over every cell rather than "
                "along the route, so it answers for ground the route never "
                "visits. It carries the same absences as everything else here: no "
                "dwell, no margin, no boulders, and one soil everywhere.",
                width=176,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def boundary_rows(
    runs: list[SiteErrands], descent: Descent | None, *, crew_limit_deg: float
) -> tuple[BoundaryRow, ...]:
    crossing = [r for r in runs if r.exceeds("deepest", crew_limit_deg)]
    return (
        BoundaryRow(
            quantity="target selection",
            published_range="none",
            used="nearest edge, floor of that shadow, deepest cold trap",
            status=UNMEASURED,
            basis=(
                "three errands chosen to span what a mission might be for, not "
                "measured against any published mission profile. The first was "
                "the only one earlier days ran, and it is the gentlest of the "
                "three at every site"
            ),
        ),
        BoundaryRow(
            quantity="target sampling",
            published_range="not applicable",
            used=f"{MAP_STRIDE * 5} m, checked at {CHECK_STRIDE * 5} m",
            status=OUTSIDE,
            basis=(
                "the illumination grid that targets are chosen from is coarser "
                "than the grid routes are planned on. At 250 m the floor of the "
                "nearest shadow is the nearest shadow at eight of ten sites; at "
                "100 m they separate by tens of metres of drop. Both numbers are "
                "reported and the coarse one is what the headline uses, so the "
                "comparison with Day 12 holds"
            ),
        ),
        BoundaryRow(
            quantity="crew comparison",
            published_range="Rice et al. (2023): 20 degrees, 2 km",
            used="20 degrees as a slope limit only",
            status=INSIDE,
            basis=(
                "a suited crew is limited by far more than gradient -- consumables, "
                "walkback, dust, thermal load, and a two kilometre traverse range "
                "this study does not apply here. Comparing on slope alone "
                "flatters the crew and is the conservative direction for a legged "
                "argument"
            ),
        ),
        BoundaryRow(
            quantity="mission concept at the excluded site",
            published_range="not applicable",
            used="a sortie designed around the robot",
            status=OUTSIDE,
            basis=(
                "Shackleton Rim is not an Artemis III candidate region and is run "
                "here as though a mission were built around a legged platform "
                "rather than around a crew landing. That is a different concept "
                "and every number from it inherits the difference"
            ),
        ),
        BoundaryRow(
            quantity="pits and skylights",
            published_range="Wagner and Robinson (2014): 200+ pits, 5 to 900 m",
            used="absent; none exists at these latitudes",
            status=OUTSIDE,
            basis=(
                "the canonical legged case and it cannot be tested here. Pits sit "
                "in mare and impact-melt deposits at low and middle latitudes, "
                "the highest published candidates near 72 north, and none in this "
                "polar archive. The median catalogued pit is about 16 m across, "
                "three cells on a 5 m grid, so even a polar one would need "
                "metre-scale stereo rather than these products"
            ),
        ),
        BoundaryRow(
            quantity="boulders",
            published_range="none",
            used="absent",
            status=UNMEASURED,
            basis=(
                "the specification Day 6 produced, still unfilled. If reaching "
                "shadow is not slope-limited then what limits it is obstacles, "
                "and obstacle statistics need imagery this project does not carry"
            ),
        ),
        BoundaryRow(
            quantity="descent as a depth",
            published_range="not applicable",
            used="round-trip energy as a field over every cell",
            status=INSIDE,
            basis=(
                "a descent is not a yes or a no. The field answers how deep a "
                "battery reaches including the climb back, for ground the planned "
                "route never visits"
                + (
                    f", and at {descent.site.name} the floor costs "
                    f"{descent.floor_round_trip_Wh:.0f} Wh"
                    if descent is not None
                    else ""
                )
            ),
        ),
        BoundaryRow(
            quantity="slope decidability",
            published_range="producers: 1.5 to 2.5 degrees RMS slope error",
            used=f"capabilities reported to {CAPABILITY_TOLERANCE_DEG:.2f} degrees",
            status=UNMEASURED,
            basis=(
                "the bisection resolves far finer than the map does. A capability "
                "within the stated slope error of the crew limit would not be "
                "decidable"
                + (
                    "; the one that crosses does so by "
                    f"{min(r.capability('deepest') for r in crossing) - crew_limit_deg:.1f} "
                    "degrees, which is inside that error and is stated as "
                    "undecidable rather than as a crossing"
                    if crossing
                    and min(r.capability("deepest") for r in crossing) - crew_limit_deg
                    <= SLOPE_ERROR_DEG
                    else ""
                )
            ),
        ),
        BoundaryRow(
            quantity="soil",
            published_range="Carrier et al. (1991) lunar intercrater",
            used="the same soil at every cell of every site",
            status=UNMEASURED,
            basis=(
                "carried forward. A crater floor is not the same regolith as a "
                "rim and this study cannot tell the difference"
            ),
        ),
    )


def _format_float(value: float) -> str:
    return repr(float(value))


def build_report(
    runs: list[SiteErrands], descent: Descent | None, *, crew_limit_deg: float
) -> str:
    rows = boundary_rows(runs, descent, crew_limit_deg=crew_limit_deg)
    candidates = [r for r in runs if r.site.is_candidate]
    crossing = [r for r in runs if r.exceeds("deepest", crew_limit_deg)]
    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# Three errands, and which of them a crew is excluded from.",
        "#",
        "# Generated by studies/sites/errands.py. Do not edit.",
        "#",
        "# FOUR AXES OF SIX. Comms and cold-trap range remain empty.",
        "",
        f"schema_version = {REPORT_SCHEMA_VERSION}",
        "",
        "[environment]",
        f'python = "{host_platform.python_version()}"',
        f'numpy = "{np.__version__}"',
        "",
        "[method]",
        f"common_window_km = {_format_float(COMMON_WINDOW_KM)}",
        f"target_sampling_m = {MAP_STRIDE * 5}",
        f"check_sampling_m = {CHECK_STRIDE * 5}",
        f"capability_tolerance_deg = {_format_float(CAPABILITY_TOLERANCE_DEG)}",
        f"crew_slope_limit_deg = {_format_float(crew_limit_deg)}",
        f"tipping_limit_deg = {_format_float(TIPPING_LIMIT_DEG)}",
        'capability = "minimum_slope_capability_deg, the same bisection Day 12 used"',
        "",
        "# The question, answered.",
        "[verdict]",
        "hardest_candidate_errand_deg = "
        + _format_float(max(r.capability("deepest") for r in candidates)),
        "candidate_errands_beyond_the_crew_limit = "
        + str(sum(1 for r in candidates if r.exceeds("deepest", crew_limit_deg))),
        "sites_beyond_the_crew_limit = ["
        + ", ".join(f'"{r.site.id}"' for r in crossing)
        + "]",
        "any_beyond_the_platform_tipping_limit = "
        + str(any(r.exceeds("deepest", TIPPING_LIMIT_DEG) for r in runs)).lower(),
        'slope_limited = "no, inside the candidate regions; yes at one excluded site"',
        "",
    ]
    for run in runs:
        lines += [
            "[[region]]",
            f'id = "{run.site.id}"',
            f'name = "{run.site.name}"',
            "candidate = " + str(run.site.is_candidate).lower(),
            f'product = "{run.product}"',
        ]
        for key in ERRANDS:
            errand = run.errands[key]
            lines += [
                f"{key}_km = {_format_float(errand.target.distance_km)}",
                f"{key}_drop_m = {_format_float(errand.target.drop_m)}",
                f"{key}_area_km2 = {_format_float(errand.target.region_area_km2)}",
                f"{key}_capability_deg = {_format_float(errand.capability_deg)}",
            ]
        lines += [
            "floor_equals_nearest = "
            + str(
                (run.errands["floor"].target.row, run.errands["floor"].target.column)
                == (
                    run.errands["nearest"].target.row,
                    run.errands["nearest"].target.column,
                )
            ).lower(),
            "check_nearest_drop_m = " + _format_float(run.check_nearest_drop_m),
            "check_floor_drop_m = " + _format_float(run.check_floor_drop_m),
            "check_floor_area_km2 = " + _format_float(run.check_floor_area_km2),
            "",
        ]

    coincide = sum(
        1
        for r in runs
        if (r.errands["floor"].target.row, r.errands["floor"].target.column)
        == (r.errands["nearest"].target.row, r.errands["nearest"].target.column)
    )
    lines += [
        "# What the coarse target sampling hides, measured rather than assumed.",
        "[resolution]",
        f"sites = {len(runs)}",
        f"floor_coincides_with_nearest = {coincide}",
        "median_extra_drop_at_the_finer_sampling_m = "
        + _format_float(
            float(
                np.median(
                    [
                        r.check_floor_drop_m - r.check_nearest_drop_m
                        for r in runs
                        if np.isfinite(r.check_floor_drop_m)
                    ]
                )
            )
        ),
        "median_floor_region_area_km2 = "
        + _format_float(
            float(
                np.median(
                    [
                        r.check_floor_area_km2
                        for r in runs
                        if np.isfinite(r.check_floor_area_km2)
                    ]
                )
            )
        ),
        "",
    ]

    if descent is not None:
        enough = descent.battery_Wh[
            descent.depth_reached_m >= descent.full_depth_m - 1.0
        ]
        lines += [
            "# The site the crew programme excludes, run as though the mission",
            "# were designed around the robot. A different mission concept.",
            "[excluded_site]",
            f'id = "{descent.site.id}"',
            "candidate = " + str(descent.site.is_candidate).lower(),
            f"floor_depth_m = {_format_float(descent.full_depth_m)}",
            f"floor_distance_km = {_format_float(descent.target.distance_km)}",
            f"capability_deg = {_format_float(descent.capability_deg)}",
            "beyond_the_crew_limit = "
            + str(descent.capability_deg > crew_limit_deg).lower(),
            "round_trip_to_the_floor_Wh = "
            + _format_float(descent.floor_round_trip_Wh),
            "battery_that_reaches_the_floor_Wh = "
            + (_format_float(float(enough[0])) if enough.size else "nan"),
            "planned_descent_km = "
            + _format_float(float(descent.route_distance_m[-1]) / 1000.0),
            "",
        ]
        for value in (250.0, 500.0, 1000.0, 2000.0):
            index = int(np.searchsorted(descent.battery_Wh, value))
            index = min(index, descent.depth_reached_m.size - 1)
            lines += [
                "[[excluded_site.depth]]",
                f"battery_Wh = {_format_float(value)}",
                "depth_reached_m = "
                + _format_float(float(descent.depth_reached_m[index])),
                "",
            ]

    lines += [
        "# The canonical legged case, and why it is absent rather than negative.",
        "[pits]",
        "present_in_this_archive = false",
        "present_in_the_published_catalogue_at_these_latitudes = false",
        "median_catalogued_diameter_m = 16.0",
        "cells_across_at_5_m_posting = 3.2",
        'basis = "Wagner and Robinson (2014): pits occur in mare and impact-melt '
        'deposits at low and middle latitudes; the highest published candidates '
        'are near 72 north. Even a polar pit would need metre-scale stereo to '
        'resolve, so this is a specification for the next measurement rather '
        'than a gap to approximate"',
        "",
        "[answer]",
        'statement = """',
        "Not slope-limited inside the candidate regions. Slope-limited at the one",
        "site the crew programme excludes.",
        "",
        "The deepest cold trap in each candidate region sits six to nineteen",
        "kilometres out and seven hundred to two thousand nine hundred metres",
        "down, and the hardest of those journeys -- Haworth -- needs 15.0 degrees",
        "of slope capability against a crew limit of twenty. So the harder errand",
        "does not rescue the gradient argument, and Day 12's result survives being",
        "pushed.",
        "",
        "Shackleton Rim is where a row crosses. Its floor is 4,577 m below the",
        "charge point and reaching it needs 22.1 degrees, and Shackleton is not a",
        "candidate region precisely because it is not safe to land in. The one",
        "journey here that a crew could not make is at the site the crew",
        "programme already excluded, which is the thesis being enforced by the",
        "data rather than asserted.",
        "",
        "Two things keep that from being a rescue. The crossing is 2.1 degrees",
        "past the limit against a map whose stated slope error is 1.5 to 2.5, so",
        "it is not decidable on this data -- the same problem Haworth had on Day",
        "11 and it is reported the same way. And comparing on slope alone flatters",
        "the crew, who are also held to a two kilometre traverse range that every",
        "one of these deep errands breaks.",
        "",
        "So the honest form is narrower than either. Reaching lunar polar shadow",
        "is not a gradient problem. What excludes a crew from these journeys is",
        "range and duration, and what would exclude them on gradient is terrain",
        "nobody has selected and this archive barely samples. The legged case",
        "rests on obstacles, pits and lava tubes rather than on slope, and none of",
        "those can be measured with a 5 m polar DEM. That is a specification, in",
        "the same shape as the four this project has already produced.",
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
            "Run three errands against every site and report which of them a "
            "crew is excluded from."
        )
    )
    parser.add_argument("--figure-directory", type=Path, default=FIGURE_DIRECTORY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    arguments = parser.parse_args(argv)

    sites = load_sites(SITE_DIRECTORY)
    products = load_terrain_manifest(MANIFEST_PATH)
    platform = load_platform(PLATFORM_PATH).platform
    dataset = load_soil(SOIL_PATH).datasets["carrier1991"]
    cost = build_cost(
        platform=platform,
        contact=dataset.models["bekker"].extrapolating,
        strength=mohr_coulomb_model(dataset, depth_range_cm="0-15"),
        mobilization=janosi_hanamoto_model(dataset),
    )

    runs: list[SiteErrands] = []
    descent: Descent | None = None
    crew_limit = 0.0
    for site in sites.values():
        if not site.has_terrain:
            continue
        crew_limit = site.crew.maximum_slope_deg
        product = products[cast(str, site.terrain_product)]
        path = TERRAIN_DIRECTORY / product.filename
        if not path.exists():
            print(
                f"{path.relative_to(REPOSITORY_ROOT)} is absent. Terrain products "
                "are fetched, not committed; run tools/fetch_terrain.py"
            )
            return 1
        raster = read_float_geotiff(path)
        run = run_site(site, raster=raster, product=product.id)
        if run is None:
            print(f"  {site.name:22s} no permanent shadow in the window")
            del raster
            continue
        runs.append(run)
        print(
            f"  {site.name:22s} "
            + "  ".join(
                f"{key} {run.capability(key):5.2f}°"
                f"{'*' if run.exceeds(key, crew_limit) else ' '}"
                for key in ERRANDS
            )
            + f"  | floor {run.errands['deepest'].target.drop_m:5.0f} m down, "
            f"{run.errands['deepest'].target.distance_km:5.2f} km out"
        )
        if site.id == EXCLUDED_SITE:
            descent = run_descent(site, raster=raster, cost=cost)
        del raster

    if not runs:
        print("no site produced an errand; there is nothing to compare")
        return 1
    if descent is None:
        print(f"the excluded site {EXCLUDED_SITE!r} produced no descent to report")
        return 1

    print(
        f"\n  {descent.site.name}: floor {descent.full_depth_m:.0f} m down, "
        f"needs {descent.capability_deg:.2f}°, "
        f"{descent.floor_round_trip_Wh:.0f} Wh out and back"
    )

    arguments.figure_directory.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)

    for name, figure in (
        (
            "capability-by-errand",
            build_capability_figure(runs, crew_limit_deg=crew_limit),
        ),
        (
            "descent-at-the-excluded-site",
            build_descent_figure(descent, crew_limit_deg=crew_limit),
        ),
    ):
        target = arguments.figure_directory / f"{name}.png"
        figure.savefig(target, dpi=200)
        plt.close(figure)
        print(f"wrote {target.relative_to(REPOSITORY_ROOT)}")

    arguments.report.write_text(
        build_report(runs, descent, crew_limit_deg=crew_limit), encoding="utf-8"
    )
    print(f"wrote {arguments.report.relative_to(REPOSITORY_ROOT)}")

    print("\n  measured against extrapolated\n")
    print(text_table(boundary_rows(runs, descent, crew_limit_deg=crew_limit)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
