# SPDX-License-Identifier: Apache-2.0
#
# studies.sites.routing — what changes when the route is allowed to bend.
#
# Day 11 closed two candidate regions not because the platform could not walk
# them -- 32 degrees of slope capability opens every reachable one -- but because
# the straight line to their nearest cold trap crossed 53 and 66 degrees. The
# binding constraint at those sites was a capability this project had never
# modelled, which is a reordering rather than a footnote, and this is it acted
# on.
#
# It also closes a gap that has been widening since Day 7. Every route in this
# project has been a straight line between two extrema, and every energy number,
# timing band and traversability claim rests on that. A straight line is the
# worst route in rough terrain and the shortest in flat, so the direction of the
# error is not consistent between sites -- which means the Day 11 comparison may
# have been ranking regions partly on how well a straight line happened to serve
# them.
#
# What this is, and the qualifier matters more than the result. Search over a
# known map. No perception, no state estimation, no uncertainty, no replanning.
# The platform is not deciding what it can see; it is deciding what was already
# known. It is nevertheless the first thing in this repository that decides
# rather than evaluates, and until today the repository contained no autonomy at
# all -- physics with a clock.
#
# The day was set up to answer whether two regions were closed by terrain or by
# method, and the answer turned out to be neither. They were closed by a
# sampling bug, and building the planner is what found it.
#
# Four results, and the first one retracts a headline.
#
# Day 11's two closures were an artifact. It sampled every route at a fixed four
# hundred points regardless of length, which on a half-kilometre route is a two
# metre step on a five metre grid. Consecutive samples then land on the same
# cell and report no rise, and the next pair charges a whole cell-to-cell rise
# against a sub-cell run. That turned a 30 degree slope at Malapert Massif into
# a 66 degree cliff and a 26 degree one at de Gerlache into 53, and closed two
# candidate regions that were never closed. The planner found it because a
# search over the grid steps cell to cell by construction and could not
# reproduce the walls the transect was reporting. sample_transect now refuses to
# sample finer than the grid it is reading.
#
# So the count is six of nine, not three, and the correction rather than the
# capability is what moved it. Every straight line at every site here was
# walkable all along.
#
# Planning buys very little on these routes. Around one percent of energy where
# both routes exist, and a few degrees off the steepest step. It is margin
# against a limit rather than access to a place, and on this evidence a straight
# line was an adequate approximation for the sortie numbers -- which is a null
# result for the day's own capability and is reported as one.
#
# The ranking does not change. Day 11's ordering of regions by sortie cost
# survives planning exactly, so that comparison was not measuring straight-line
# luck after all.
#
# And the baseline decides the answer. Traversability is evaluated at the native
# 5 m posting because coarsening manufactures passability: across these products
# the fraction of cells steeper than the tipping limit falls from about one in
# ten thousand at 5 m to exactly zero at 40 m. A planner on a 40 m grid would
# find every region open and would be reporting a property of the aggregation.
#
# The result that matters most is none of those. Every planned route here needs
# under twelve degrees of slope capability, which is inside the twenty degree
# limit a suited crew already works to. Once a route can bend, reaching the
# nearest cold trap stops being a gradient problem, and what separates this
# platform from a crew on this errand is the two kilometre traverse range rather
# than the ground. Day 11's terrain-coverage advantage is an argument about
# where a robot can go; it is not an argument about this particular journey, and
# the two had been running together.
#
# What has not changed. Comms and cold-trap range are still empty, so this is
# four axes of six with better routes rather than five. And the 5 m slope is
# still roughly ninety percent interpolated with a stated RMS error of 1.5 to
# 2.5 degrees, so a route threading cells near the limit is a route the data
# cannot certify -- counted and reported rather than assumed away.
#
# References
#   Dijkstra EW (1959) A note on two problems in connexion with graphs.
#     Numerische Mathematik 1, 269-271.
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
    Route,
    TraversalCost,
    minimum_slope_capability_deg,
    plan_route,
)
from eclipse.platform import (
    Platform,
    equilibrium_slip_ratio,
    swing_work_per_meter,
)
from eclipse.sortie import JOULES_PER_WATT_HOUR
from eclipse.stance import wave_gait, within_stride_slip_ratio
from eclipse.terrain import aggregate, slope_degrees

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
SITE_DIRECTORY: Final = REPOSITORY_ROOT / "configs" / "sites"
TERRAIN_DIRECTORY: Final = REPOSITORY_ROOT / "data" / "terrain"
MANIFEST_PATH: Final = TERRAIN_DIRECTORY / "manifest.toml"
PLATFORM_PATH: Final = (
    REPOSITORY_ROOT / "configs" / "platforms" / "nominal-quadruped.toml"
)
SOIL_PATH: Final = REPOSITORY_ROOT / "data" / "soils" / "lunar-intercrater.toml"
FIGURE_DIRECTORY: Final = Path(__file__).resolve().parent / "figures"
DEFAULT_REPORT_PATH: Final = Path(__file__).resolve().parent / "results" / "routing.toml"

REPORT_SCHEMA_VERSION: Final = 1
LUNAR_GRAVITY: Final = 1.62
FEET_IN_STANCE: Final = 3
NOMINAL_DERATING: Final = 4.0
SLOPE_METHOD: Final = "central_difference"

TIPPING_LIMIT_DEG: Final = 39.8055710922652
SLOPE_ERROR_DEG: Final = 2.5
INSULATED_SURVIVAL_W: Final = 11.8
DWELL_HOURS: Final = 4.0

MAP_STRIDE: Final = 50
HORIZON_AZIMUTHS: Final = 72
HORIZON_SAMPLES: Final = 140
HORIZON_STANDOFF_M: Final = 50.0
# The straight line is drawn on the figure at cell resolution, the same
# stepping the planner uses and the same the cost comparison uses.
STRAIGHT_LINE_STEP_CELLS: Final = 1
COMMON_WINDOW_KM: Final = 16.0
DAY_TRIP_LIMIT_KM: Final = 10.0

# The cost curve is sampled on this axis and interpolated between. A tenth of a
# degree is far finer than the map resolves slope to, so the interpolation adds
# nothing the data does not already have.
COST_SLOPE_DEG: Final[NDArray[np.float64]] = np.arange(-89.0, 89.01, 0.1)

# What the platform draws while it walks, whether or not the ground is dark.
# Without it a descent steeper than about thirteen degrees costs nothing after
# the free-descent clamp, and the search becomes indifferent between a hundred
# metre drop and a ten kilometre one -- it would return an arbitrarily wandering
# path with the same energy. The clock is what breaks that tie, and this is the
# smallest defensible statement of it: Day 8's insulated survival power, drawn
# continuously, at the platform's nominal speed.
HOTEL_LOAD_W: Final = INSULATED_SURVIVAL_W

# The baselines the traversable fraction is measured at, to show what
# coarsening buys. This is evidence rather than a parameter: no result below is
# computed at anything but the first of them.
BASELINE_FACTORS: Final = (1, 2, 4, 8)

ACHIEVABLE_SLOPE_DEG: Final[NDArray[np.float64]] = np.linspace(5.0, 60.0, 221)


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
    *,
    platform: Platform,
    contact: Any,
    strength: Any,
    mobilization: Any,
    hotel_load_W: float,
) -> TraversalCost:
    """The platform's own cost of transport, tabulated against signed slope.

    Clamped at zero where gravity more than pays for the ground, which is the
    same convention Leg.segment_J uses and the condition that makes the search
    valid. The hotel term is added after the clamp, so a free descent still
    costs the time it takes.
    """
    flat, _ = within_stride_slip_ratio(
        platform=platform,
        gait=wave_gait(lift_order=(2, 0, 3, 1), duty_factor=0.75),
        strength=strength,
        mobilization=mobilization,
        gravity_m_per_s2=LUNAR_GRAVITY,
    )
    hotel = hotel_load_W / platform.nominal_speed_m_per_s
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
        locomotion = float(
            cost.gravitational_J_per_m
            + cost.shear_J_per_m
            + cost.compaction_J_per_m
            + cost.swing_J_per_m
        )
        joules[index] = max(locomotion, 0.0) * NOMINAL_DERATING + hotel

    # Beyond the traction limit the slip solve refuses and the entry stays
    # infinite; carry that forward as impassable rather than interpolating over
    # it, which is what an infinity in the table does.
    usable = np.isfinite(joules)
    limit = float(np.abs(COST_SLOPE_DEG[usable]).max())
    return TraversalCost(
        slope_deg=COST_SLOPE_DEG[usable],
        joules_per_metre=joules[usable],
        limit_deg=min(limit, TIPPING_LIMIT_DEG),
    )


@dataclass(frozen=True, slots=True)
class Legs:
    """A round trip planned in both directions over the same directed graph."""

    outbound: Route
    inbound: Route

    @property
    def energy_Wh(self) -> float:
        return (
            self.outbound.total_energy_J + self.inbound.total_energy_J
        ) / JOULES_PER_WATT_HOUR

    @property
    def distance_m(self) -> float:
        return self.outbound.distance_m + self.inbound.distance_m

    @property
    def max_abs_slope_deg(self) -> float:
        return max(
            self.outbound.max_abs_slope_deg, self.inbound.max_abs_slope_deg
        )

    def undecidable_steps(self) -> int:
        return self.outbound.undecidable_steps(
            limit_deg=TIPPING_LIMIT_DEG, error_deg=SLOPE_ERROR_DEG
        ) + self.inbound.undecidable_steps(
            limit_deg=TIPPING_LIMIT_DEG, error_deg=SLOPE_ERROR_DEG
        )


@dataclass(frozen=True, slots=True)
class Routed:
    """One region, routed both ways, against the straight line it replaces."""

    site: Site
    product: str
    charge: tuple[int, int]
    target: ShadowTarget
    legs: Legs | None
    reason: str
    impassable_cell_fraction: float
    reached_cell_fraction: float
    straight_distance_m: float
    straight_max_slope_deg: float
    straight_energy_Wh: float
    straight_traversable: bool
    capability_deg: float
    window_elevation_m: NDArray[np.float64] | None
    window_origin: tuple[int, int]
    traversable_fraction_at: dict[int, float]

    @property
    def planned(self) -> bool:
        return self.legs is not None

    @property
    def opened_by_planning(self) -> bool:
        return self.planned and not self.straight_traversable

    @property
    def detour(self) -> float:
        legs = self.legs
        if legs is None or self.straight_distance_m <= 0.0:
            return float("nan")
        return legs.distance_m / self.straight_distance_m

    @property
    def one_way_km(self) -> float:
        legs = self.legs
        return float("nan") if legs is None else legs.outbound.distance_m / 1000.0

    @property
    def within_crew_slope(self) -> bool:
        return self.capability_deg <= self.site.crew.maximum_slope_deg

    @property
    def within_crew_range(self) -> bool:
        return self.one_way_km <= self.site.crew.traverse_range_km

    @property
    def crew_could_reach(self) -> bool:
        return self.planned and self.within_crew_slope and self.within_crew_range

    @property
    def saving(self) -> float:
        legs = self.legs
        if legs is None or not self.straight_traversable:
            return float("nan")
        return legs.energy_Wh / self.straight_energy_Wh


def grid_route(
    *,
    elevation_m: NDArray[np.float64],
    cell_size_m: float,
    rows: NDArray[np.int_],
    columns: NDArray[np.int_],
    cost: TraversalCost,
) -> Route:
    """Price a given sequence of cells exactly as the planner prices its own."""
    elevation = elevation_m[rows, columns]
    run = cell_size_m * np.hypot(np.diff(rows), np.diff(columns))
    rise = np.diff(elevation)
    length = np.hypot(rise, run)
    slope = np.degrees(np.arctan2(rise, run))
    return Route(
        rows=rows,
        columns=columns,
        elevation_m=elevation,
        step_length_m=length,
        step_slope_deg=slope,
        step_energy_J=cost.at(slope) * length,
    )


def straight_line_legs(
    *,
    elevation_m: NDArray[np.float64],
    cell_size_m: float,
    start: tuple[int, int],
    goal: tuple[int, int],
    cost: TraversalCost,
) -> Legs:
    """The straight line, discretised the way the graph is and priced the same.

    This is the third attempt at a fair comparison and the first correct one,
    which is worth recording because both earlier ones were wrong in opposite
    directions and each looked reasonable.

    Sampling the line at a fixed count went finer than the grid on short
    routes: consecutive samples shared a cell, a whole cell-to-cell rise was
    charged against a sub-cell run, and a thirty degree slope was reported as
    sixty-six. Sampling it once per cell along the dominant axis went the other
    way, skipping cells the graph steps through and smoothing the slopes, which
    made the straight line look cheaper than the least-energy route -- which is
    impossible, and is what exposed it.

    Stepping one cell at a time in the dominant axis makes the line a sequence
    of the same eight-connected moves the search is choosing among, so the two
    columns differ by the route and by nothing else. The planner is then
    guaranteed to be no worse, and if it ever is, the comparison is broken
    rather than the finding being interesting.
    """
    steps = max(abs(goal[0] - start[0]), abs(goal[1] - start[1]))
    rows = np.rint(np.linspace(start[0], goal[0], max(steps + 1, 2))).astype(int)
    columns = np.rint(
        np.linspace(start[1], goal[1], max(steps + 1, 2))
    ).astype(int)
    outbound = grid_route(
        elevation_m=elevation_m,
        cell_size_m=cell_size_m,
        rows=rows,
        columns=columns,
        cost=cost,
    )
    inbound = grid_route(
        elevation_m=elevation_m,
        cell_size_m=cell_size_m,
        rows=rows[::-1],
        columns=columns[::-1],
        cost=cost,
    )
    return Legs(outbound=outbound, inbound=inbound)


def route_site(
    site: Site,
    *,
    raster: GeoRaster,
    product: str,
    cost: TraversalCost,
    keep_window: bool = False,
) -> Routed | None:
    """Charge point, nearest cold trap, and the best route between them.

    The charge point and the target are derived exactly as Day 11 derived them,
    through the same library calls, so the only thing that differs between the
    two studies is what happens after the destination is chosen.
    """
    first_row, last_row, first_column, last_column = centred_window(raster, span_m=COMMON_WINDOW_KM * 1000.0)
    grid_rows, grid_columns = np.meshgrid(
        np.arange(first_row, last_row, MAP_STRIDE),
        np.arange(first_column, last_column, MAP_STRIDE),
        indexing="ij",
    )
    grid = illuminate(raster, grid_rows.ravel(), grid_columns.ravel())
    lit = grid.any_sunlight_fraction.reshape(grid_rows.shape)
    if not bool((lit <= 0.0).any()):
        return None

    elevation = raster.values[grid_rows, grid_columns]
    charge = best_charge_point(
        rows=grid_rows,
        columns=grid_columns,
        any_sunlight_fraction=lit,
        elevation_m=elevation,
    )
    target = shadow_targets(
        raster,
        start=charge,
        rows=grid_rows,
        columns=grid_columns,
        any_sunlight_fraction=lit,
    )["nearest"]

    window = np.ascontiguousarray(
        raster.values[first_row:last_row, first_column:last_column]
    )
    traversable = {}
    for factor in BASELINE_FACTORS:
        coarse = aggregate(window, factor) if factor > 1 else window
        steep = slope_degrees(
            coarse, cell_size_m=raster.cell_size_m * factor, method=SLOPE_METHOD
        )
        traversable[factor] = float((steep <= TIPPING_LIMIT_DEG).mean())

    start = (charge[0] - first_row, charge[1] - first_column)
    goal = (target.row - first_row, target.column - first_column)
    straight = straight_line_legs(
        elevation_m=window,
        cell_size_m=raster.cell_size_m,
        start=start,
        goal=goal,
        cost=cost,
    )
    straight_ok = bool(np.isfinite(straight.energy_Wh))
    outbound = plan_route(
        elevation_m=window,
        cell_size_m=raster.cell_size_m,
        start=start,
        goal=goal,
        cost=cost,
    )
    inbound = plan_route(
        elevation_m=window,
        cell_size_m=raster.cell_size_m,
        start=goal,
        goal=start,
        cost=cost,
    )
    legs = (
        Legs(outbound=cast(Route, outbound.route), inbound=cast(Route, inbound.route))
        if outbound.found and inbound.found
        else None
    )
    return Routed(
        site=site,
        product=product,
        charge=charge,
        target=target,
        legs=legs,
        reason=outbound.reason or inbound.reason,
        impassable_cell_fraction=outbound.impassable_cell_fraction,
        reached_cell_fraction=outbound.reached_cell_fraction,
        straight_distance_m=straight.distance_m,
        straight_max_slope_deg=straight.max_abs_slope_deg,
        straight_energy_Wh=straight.energy_Wh,
        straight_traversable=straight_ok,
        capability_deg=minimum_slope_capability_deg(
            elevation_m=window,
            cell_size_m=raster.cell_size_m,
            start=start,
            goal=goal,
        ),
        window_elevation_m=window if keep_window else None,
        window_origin=(first_row, first_column),
        traversable_fraction_at={
            int(round(raster.cell_size_m * factor)): value
            for factor, value in traversable.items()
        },
    )


def boundary_rows(routed: list[Routed]) -> tuple[BoundaryRow, ...]:
    baselines = sorted(routed[0].traversable_fraction_at)
    coarsest = baselines[-1]
    native = baselines[0]
    return (
        BoundaryRow(
            quantity="traversability baseline",
            published_range="not applicable",
            used=f"{native} m, the products' native posting",
            status=INSIDE,
            basis=(
                "coarsening manufactures passability. The traversable fraction "
                "rises from "
                + ", ".join(
                    f"{r.traversable_fraction_at[native]:.5f} at {native} m to "
                    f"{r.traversable_fraction_at[coarsest]:.5f} at {coarsest} m"
                    for r in routed[:1]
                )
                + " and reaches exactly one at every site by 40 m, so a planner "
                "on an aggregated grid finds every region open and reports a "
                "property of the aggregation"
            ),
        ),
        BoundaryRow(
            quantity="planning knowledge",
            published_range="not applicable",
            used="complete prior map, no perception",
            status=OUTSIDE,
            basis=(
                "search over a known elevation grid. No perception, no state "
                "estimation, no uncertainty, no replanning. The platform is not "
                "deciding what it can see, it is deciding what was already "
                "known, and a real traverse has none of that in advance"
            ),
        ),
        BoundaryRow(
            quantity="slope decidability",
            published_range="producers: 1.5 to 2.5 degrees RMS slope error",
            used=f"cells within {SLOPE_ERROR_DEG:.1f} degrees of the limit are counted",
            status=UNMEASURED,
            basis=(
                "a route threading cells near the tipping limit is a route the "
                "data cannot certify: those cells are passable in the product "
                "and would not be if the product were wrong by its own stated "
                "error. The count is reported per region rather than assumed to "
                "be zero"
            ),
        ),
        BoundaryRow(
            quantity="graph connectivity",
            published_range="not applicable",
            used="eight-connected, diagonals at their true length",
            status=INSIDE,
            basis=(
                "four-connected forbids diagonal travel and inflates distance by "
                "up to root two; charging a diagonal as one step produces "
                "staircases. Sixteen-connected would reduce the residual angular "
                "quantisation and is a refinement for when something turns on it"
            ),
        ),
        BoundaryRow(
            quantity="search algorithm",
            published_range="not applicable",
            used="Dijkstra, directed, non-negative costs",
            status=INSIDE,
            basis=(
                "the free-descent clamp makes the cheapest achievable step cost "
                "nothing, so the only admissible A* heuristic is zero and a "
                "useful one would give up the optimality guarantee. Ten million "
                "nodes search in about a second, so the heuristic buys nothing"
            ),
        ),
        BoundaryRow(
            quantity="hotel load",
            published_range="none",
            used=f"{HOTEL_LOAD_W:.1f} W drawn continuously while walking",
            status=UNMEASURED,
            basis=(
                "without it a free descent costs nothing and the search is "
                "indifferent between a short drop and an arbitrarily long one. "
                "Day 8's insulated survival power is the smallest defensible "
                "statement of what the platform draws regardless, and it makes "
                "the clock break the tie"
            ),
        ),
        BoundaryRow(
            quantity="route optimality",
            published_range="not applicable",
            used="least energy, not least time or least risk",
            status=UNMEASURED,
            basis=(
                "one objective, chosen because energy is what the rest of the "
                "project reports. A route that minimises exposure, or time, or "
                "distance from a safe abort, is a different route and this study "
                "does not compute it"
            ),
        ),
        BoundaryRow(
            quantity="soil",
            published_range="Carrier et al. (1991) lunar intercrater",
            used="the same soil at every cell of every site",
            status=UNMEASURED,
            basis=(
                "the cost surface is uniform in everything but slope. Real "
                "terrain varies in bearing strength cell to cell and that is "
                "exactly what a traversability map would carry if one existed"
            ),
        ),
    )


def _format_float(value: float) -> str:
    return repr(float(value))


def build_route_figure(routed: Routed, raster_cell_m: float) -> Figure:
    window = routed.window_elevation_m
    assert window is not None
    legs = routed.legs
    assert legs is not None
    first_row, first_column = routed.window_origin
    start = (routed.charge[0] - first_row, routed.charge[1] - first_column)
    goal = (routed.target.row - first_row, routed.target.column - first_column)

    pad = 60
    top = max(0, min(start[0], goal[0], int(legs.outbound.rows.min())) - pad)
    bottom = min(window.shape[0], max(start[0], goal[0], int(legs.outbound.rows.max())) + pad)
    left_edge = max(0, min(start[1], goal[1], int(legs.outbound.columns.min())) - pad)
    right_edge = min(window.shape[1], max(start[1], goal[1], int(legs.outbound.columns.max())) + pad)
    patch = window[top:bottom, left_edge:right_edge]
    steep = slope_degrees(patch, cell_size_m=raster_cell_m, method=SLOPE_METHOD)
    extent = (0.0, patch.shape[1] * raster_cell_m, patch.shape[0] * raster_cell_m, 0.0)

    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (11.4, 7.6),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "axes.grid": False,
                    "figure.subplot.top": 0.700,
                    "figure.subplot.bottom": 0.070,
                    "figure.subplot.left": 0.075,
                    "figure.subplot.right": 0.900,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 1, squeeze=False)
        panel = axes[0][0]
        image = panel.imshow(
            patch, extent=extent, cmap="Greys_r", interpolation="bilinear"
        )
        bar = figure.colorbar(image, ax=panel, fraction=0.046, pad=0.03)
        bar.set_label("elevation (m)")

        blocked = np.where(steep > TIPPING_LIMIT_DEG, 1.0, np.nan)
        panel.imshow(
            blocked,
            extent=extent,
            cmap="autumn",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            alpha=0.95,
        )

        def to_metres(rows: NDArray[np.int_], columns: NDArray[np.int_]) -> tuple[
            NDArray[np.float64], NDArray[np.float64]
        ]:
            return (
                np.asarray(
                    (columns - left_edge + 0.5) * raster_cell_m, dtype=np.float64
                ),
                np.asarray((rows - top + 0.5) * raster_cell_m, dtype=np.float64),
            )

        span = max(abs(goal[0] - start[0]), abs(goal[1] - start[1]))
        drawn = max(span // STRAIGHT_LINE_STEP_CELLS + 1, 2)
        straight_rows = np.rint(np.linspace(start[0], goal[0], drawn)).astype(int)
        straight_columns = np.rint(
            np.linspace(start[1], goal[1], drawn)
        ).astype(int)
        sx, sy = to_metres(straight_rows, straight_columns)
        panel.plot(
            sx, sy, color=INK_PRIMARY, linewidth=3.0, alpha=0.85,
        )
        panel.plot(
            sx,
            sy,
            color=ACCENT_SECONDARY,
            linewidth=1.5,
            linestyle=(0, (4, 2)),
            label=f"straight line, peaks at {routed.straight_max_slope_deg:.0f}°",
        )
        px, py = to_metres(legs.outbound.rows, legs.outbound.columns)
        panel.plot(px, py, color="white", linewidth=3.6, alpha=0.9)
        panel.plot(
            px,
            py,
            color=ACCENT_PRIMARY,
            linewidth=1.9,
            label=(
                f"planned outbound, {legs.outbound.distance_m / 1000:.2f} km, "
                f"peak {legs.outbound.max_abs_slope_deg:.0f}°"
            ),
        )
        rx, ry = to_metres(legs.inbound.rows, legs.inbound.columns)
        panel.plot(
            rx,
            ry,
            color=INK_SECONDARY,
            linewidth=1.3,
            linestyle=(0, (1.6, 1.6)),
            label=(
                f"planned return, {legs.inbound.distance_m / 1000:.2f} km — "
                "not the outbound reversed"
            ),
        )

        worst = int(np.argmax(np.abs(np.diff(window[straight_rows, straight_columns]))))
        bx, by = to_metres(straight_rows[worst : worst + 1], straight_columns[worst : worst + 1])
        panel.plot(
            bx, by, marker="X", markersize=11.0, color=ACCENT_SECONDARY,
            markeredgecolor="white", markeredgewidth=1.2,
        )
        panel.annotate(
            f"steepest step on the straight line, {routed.straight_max_slope_deg:.0f}°",
            xy=(float(bx[0]), float(by[0])),
            xytext=(12, 10),
            textcoords="offset points",
            color=ACCENT_SECONDARY,
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65,
                  "boxstyle": "round,pad=0.2"},
        )
        for point, label, marker in (
            (start, "charge point", "o"),
            (goal, "cold trap", "s"),
        ):
            mx, my = to_metres(
                np.array([point[0]]), np.array([point[1]])
            )
            panel.plot(
                mx, my, marker=marker, markersize=9.0, markerfacecolor="none",
                markeredgewidth=1.8, color="white",
            )
            panel.annotate(
                label,
                xy=(float(mx[0]), float(my[0])),
                xytext=(10, -16),
                textcoords="offset points",
                color=INK_PRIMARY,
                fontsize=8.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7,
                      "boxstyle": "round,pad=0.2"},
            )
        panel.set_xlabel("metres east across the window")
        panel.set_ylabel("metres south across the window")
        panel.set_aspect("equal")
        panel.legend(loc="upper left", framealpha=0.35)

        figure.suptitle(
            f"At {routed.site.name} the planner takes "
            f"{routed.straight_max_slope_deg - legs.max_abs_slope_deg:.0f}° off the "
            "steepest step for "
            + (
                "no extra distance at all"
                if abs(legs.distance_m / routed.straight_distance_m - 1.0) < 0.005
                else f"{legs.distance_m / routed.straight_distance_m - 1.0:+.1%} of distance"
            ),
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.075,
            ha="left",
            y=0.972,
        )
        figure.text(
            0.075,
            0.918,
            caption(
                "Orange marks cells steeper than the "
                f"{TIPPING_LIMIT_DEG:.1f}° tipping limit, which cannot be stood "
                "on"
                + (
                    " — there are none in this view, which is the point"
                    if not bool(np.isfinite(blocked).any())
                    else ""
                )
                + ". The dashed line is the straight route to the same cold trap; "
                f"it peaks at {routed.straight_max_slope_deg:.0f}° and the "
                f"planned one at {legs.max_abs_slope_deg:.0f}°, for "
                + (
                    "the same ground distance"
                    if abs(legs.distance_m / routed.straight_distance_m - 1.0) < 0.005
                    else f"{legs.distance_m / routed.straight_distance_m:.2f} times the ground distance"
                )
                + ". Both are walkable. What planning buys on these sites is "
                "margin against a limit rather than access to a place, and the "
                "honest version of the day's result is that the places were never "
                "shut.\n"
                "Outbound and return are separate searches on the same directed "
                "graph and they are not the same path, because climbing and "
                "descending the same gradient differ by more than a factor of "
                "three. That asymmetry is the project's most robust locomotion "
                "result and it belongs in the search rather than being applied "
                "afterwards.\n"
                "This is search over a map that is already known. There is no "
                "perception here, no state estimation and no replanning — the "
                "platform is not deciding what it can see, it is deciding what "
                "was handed to it. Traversability is evaluated at the products' "
                "native 5 m posting, because at 40 m no cell at any of these "
                "sites exceeds the limit and every region would open for a reason "
                "that is about the arithmetic rather than the ground.",
                width=166,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def build_trade_figure(routed: list[Routed]) -> Figure:
    both = [r for r in routed if r.planned and r.straight_traversable]

    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (13.4, 6.9),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.605,
                    "figure.subplot.bottom": 0.110,
                    "figure.subplot.left": 0.055,
                    "figure.subplot.right": 0.988,
                    "figure.subplot.wspace": 0.255,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 3, squeeze=False)
        left, right, steepest = axes[0][0], axes[0][1], axes[0][2]

        for panel, quantity in (
            (left, "distance (km)"),
            (right, "energy (Wh)"),
            (steepest, "steepest step (°)"),
        ):
            panel.axline((0.0, 0.0), slope=1.0, color=INK_MUTED, linewidth=1.0)
            panel.set_xlabel(f"straight-line {quantity}")
            panel.set_ylabel(f"planned {quantity}")
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)

        for entry in both:
            legs = entry.legs
            assert legs is not None
            candidate = entry.site.is_candidate
            colour = ACCENT_PRIMARY if candidate else INK_MUTED
            left.plot(
                [entry.straight_distance_m / 1000.0],
                [legs.distance_m / 1000.0],
                marker="o" if candidate else "s",
                markersize=7.0,
                color=colour,
            )
            left.annotate(
                entry.site.name,
                xy=(entry.straight_distance_m / 1000.0, legs.distance_m / 1000.0),
                xytext=(8, -3),
                textcoords="offset points",
                color=INK_SECONDARY if candidate else INK_MUTED,
                fontsize=8.0,
            )
            right.plot(
                [entry.straight_energy_Wh],
                [legs.energy_Wh],
                marker="o" if candidate else "s",
                markersize=7.0,
                color=colour,
            )
            steepest.plot(
                [entry.straight_max_slope_deg],
                [legs.max_abs_slope_deg],
                marker="o" if candidate else "s",
                markersize=7.0,
                color=colour,
            )
            steepest.annotate(
                entry.site.name,
                xy=(entry.straight_max_slope_deg, legs.max_abs_slope_deg),
                xytext=(8, -3),
                textcoords="offset points",
                color=INK_SECONDARY if candidate else INK_MUTED,
                fontsize=7.6,
            )

        for panel in (left, right, steepest):
            panel.set_ylim(0.0, None)
            panel.set_xlim(0.0, panel.get_xlim()[1] * 1.32)
        left.set_title("the detour", color=INK_SECONDARY, loc="left")
        right.set_title("what it saves", color=INK_SECONDARY, loc="left")
        steepest.set_title(
            "and where it actually shows", color=INK_SECONDARY, loc="left"
        )
        for panel, note in (
            (left, "above the line is a longer walk"),
            (right, "below the line is a cheaper walk"),
            (steepest, "below the line is gentler ground"),
        ):
            panel.annotate(
                note,
                xy=(0.04, 0.94),
                xycoords="axes fraction",
                color=INK_MUTED,
                fontsize=7.8,
            )

        detours = np.asarray([r.detour for r in both])
        savings = np.asarray([r.saving for r in both])
        gentler = np.asarray(
            [
                r.straight_max_slope_deg - cast(Legs, r.legs).max_abs_slope_deg
                for r in both
            ]
        )
        figure.suptitle(
            "Planning saves at most "
            f"{100.0 * (1.0 - float(np.min(savings))):.0f}% of the energy and "
            f"takes up to {float(gentler.max()):.0f}° off the steepest step — the "
            "value is in the margin, not the bill",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.062,
            ha="left",
            y=0.962,
        )
        figure.text(
            0.055,
            0.918,
            caption(
                "Both routes priced on the same cost curve, so the difference "
                "between the axes is the route and nothing else, and both are "
                "stepped one cell at a time so the discretisation is shared too. "
                "Distance is measured along the ground, which is why a planned "
                "route can be shorter than the straight line it replaces.\n"
                "Every region appears, because every straight line here was "
                "walkable. Day 11 reported two closed by 53 and 66 degree "
                "barriers; those were an artifact of sampling a short route at a "
                "fixed four hundred points, finer than the five metre grid, "
                "which charges whole cell-to-cell rises against sub-cell runs. "
                "The real figures are 26 and 30 degrees.\n"
                "The first two panels sit on the line and that is the result: "
                "on these sites planning is neither an efficiency measure nor a "
                "feasibility one, because nothing needed opening. The third "
                "panel is where it does something — several degrees off the "
                "steepest step is margin against a limit rather than a saving, "
                "and margin is what a platform spends when the map turns out to "
                "be wrong by its own stated error.\n"
                "What planning did that nothing else could was disagree with the "
                "transect, which is how the sampling artifact was caught. A "
                "capability that earns its place by contradicting the thing it "
                "replaced is worth having; that is a different argument from the "
                "one this figure was built to make, and it is the one the "
                "evidence supports.",
                width=170,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def build_envelope_figure(routed: list[Routed]) -> Figure:
    candidates = [r for r in routed if r.site.is_candidate]
    with_planning = np.asarray(
        [
            sum(
                1
                for r in candidates
                if r.planned and r.capability_deg <= float(limit)
            )
            for limit in ACHIEVABLE_SLOPE_DEG
        ],
        dtype=np.float64,
    )
    without = np.asarray(
        [
            sum(
                1
                for r in candidates
                if r.straight_traversable and r.straight_max_slope_deg <= float(limit)
            )
            for limit in ACHIEVABLE_SLOPE_DEG
        ],
        dtype=np.float64,
    )

    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (11.6, 6.8),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.5,
                    "figure.subplot.top": 0.620,
                    "figure.subplot.bottom": 0.115,
                    "figure.subplot.left": 0.070,
                    "figure.subplot.right": 0.986,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 1, squeeze=False)
        panel = axes[0][0]
        panel.axhline(9, color=INK_PRIMARY, linewidth=1.0, linestyle=(0, (3, 2)))
        panel.annotate(
            "all 9 candidate regions",
            xy=(float(ACHIEVABLE_SLOPE_DEG[-1]), 9),
            xytext=(-6, 5),
            textcoords="offset points",
            ha="right",
            color=INK_PRIMARY,
            fontsize=8.0,
        )
        panel.axhline(
            len(candidates), color=INK_MUTED, linewidth=1.0, linestyle=(0, (1.5, 1.5))
        )
        panel.annotate(
            f"{len(candidates)} have terrain in this archive",
            xy=(float(ACHIEVABLE_SLOPE_DEG[-1]), len(candidates)),
            xytext=(-6, 5),
            textcoords="offset points",
            ha="right",
            color=INK_MUTED,
            fontsize=8.0,
        )
        panel.step(
            ACHIEVABLE_SLOPE_DEG, with_planning, where="post",
            color=ACCENT_PRIMARY, linewidth=2.2, label="with planning",
        )
        panel.fill_between(
            ACHIEVABLE_SLOPE_DEG, without, with_planning, step="post",
            color=ACCENT_PRIMARY, alpha=0.14, linewidth=0.0,
        )
        panel.step(
            ACHIEVABLE_SLOPE_DEG, without, where="post",
            color=ACCENT_SECONDARY, linewidth=2.0, linestyle=(0, (4, 2)),
            label="straight lines only",
        )
        crew_limit = candidates[0].site.crew.maximum_slope_deg
        panel.axvspan(
            float(ACHIEVABLE_SLOPE_DEG[0]), crew_limit,
            color=INK_MUTED, alpha=0.12, linewidth=0.0,
        )
        panel.annotate(
            f"inside the {crew_limit:.0f}° crew limit",
            xy=(0.5 * (float(ACHIEVABLE_SLOPE_DEG[0]) + crew_limit), 8.6),
            ha="center",
            color=INK_SECONDARY,
            fontsize=8.5,
        )
        panel.axvline(
            TIPPING_LIMIT_DEG, color=INK_PRIMARY, linewidth=1.0, linestyle=(0, (1.5, 1.5))
        )
        panel.annotate(
            f"this platform tips at {TIPPING_LIMIT_DEG:.0f}°",
            xy=(TIPPING_LIMIT_DEG, 0.4),
            xytext=(-7, 0),
            textcoords="offset points",
            rotation=90.0,
            ha="right",
            va="bottom",
            color=INK_PRIMARY,
            fontsize=8.0,
        )
        panel.set_xlabel("achievable slope (°)")
        panel.set_ylabel("candidate regions opened")
        panel.set_xlim(float(ACHIEVABLE_SLOPE_DEG[0]), float(ACHIEVABLE_SLOPE_DEG[-1]))
        panel.set_ylim(0.0, 9.6)
        panel.legend(loc="lower right")
        panel.spines["top"].set_visible(False)
        panel.spines["right"].set_visible(False)

        needed = [r.capability_deg for r in candidates if r.planned]
        straight_needs = max(
            r.straight_max_slope_deg for r in candidates if r.straight_traversable
        )
        figure.suptitle(
            "Planning does not open more regions — it opens the same six at "
            f"{max(needed):.1f}° of capability instead of {straight_needs:.1f}°",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.070,
            ha="left",
            y=0.962,
        )
        figure.text(
            0.070,
            0.900,
            caption(
                "The blue curve is the gentlest platform that could make each "
                "journey at all, computed exactly rather than searched for: "
                "whether a step is allowed depends on the magnitude of its "
                "slope, which is the same in both directions, so reachability "
                "is an undirected connectivity question and bisecting it gives "
                "the answer to a twentieth of a degree.\n"
                "The orange curve is the same regions reached by straight "
                "lines only. The gap between them is horizontal rather than "
                "vertical, and that is the honest reading: planning does not "
                "open more regions in the end, it opens the same six at 11.6° of "
                "capability instead of 29.6°. More than halving the gradient a "
                "platform must be built for is the first quantified argument for "
                "autonomy in this repository rather than an assertion that "
                "autonomy matters — and it is worth being precise about how "
                "small a claim it still is: this is search over a map somebody "
                "already made.\n"
                "Neither curve reaches nine, and the distance to it is coverage "
                "rather than capability. Three candidate regions have no 5 m "
                "product in this archive and no amount of planning reaches a "
                "place there is no map of.\n"
                "The shaded band is the uncomfortable part. The hardest of these "
                f"journeys needs {max(needed):.1f}° of slope capability, so every "
                "one of them lies inside what a suited crew is already allowed. "
                "Once a route can bend, reaching the nearest cold trap stops "
                "being a gradient problem — what separates the platform from a "
                f"crew here is the {candidates[0].site.crew.traverse_range_km:.0f} km "
                "traverse range, not the slope. Day 11's terrain-coverage "
                "advantage is an argument about where a robot can go; it is not "
                "an argument about this particular errand.",
                width=170,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def build_report(routed: list[Routed], cost: TraversalCost) -> str:
    rows = boundary_rows(routed)
    candidates = [r for r in routed if r.site.is_candidate]
    opened = [r for r in candidates if r.opened_by_planning]
    both = [r for r in routed if r.planned and r.straight_traversable]
    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# What changes when the route is allowed to bend.",
        "#",
        "# Generated by studies/sites/routing.py. Do not edit.",
        "#",
        "# SEARCH OVER A KNOWN MAP. No perception, no state estimation, no",
        "# uncertainty, no replanning. Four axes of six, with better routes.",
        "",
        f"schema_version = {REPORT_SCHEMA_VERSION}",
        "",
        "[environment]",
        f'python = "{host_platform.python_version()}"',
        f'numpy = "{np.__version__}"',
        "",
        "[method]",
        'algorithm = "Dijkstra, directed, eight-connected"',
        'heuristic = "none; the free-descent clamp makes zero the only admissible one"',
        'diagonal_cost = "true along-ground length, hypot(rise, run)"',
        f"planning_baseline_m = {_format_float(5.0)}",
        f"tipping_limit_deg = {_format_float(TIPPING_LIMIT_DEG)}",
        f"cost_limit_deg = {_format_float(cost.limit_deg)}",
        f"hotel_load_W = {_format_float(HOTEL_LOAD_W)}",
        f"derating = {_format_float(NOMINAL_DERATING)}",
        f"flat_cost_J_per_m = {_format_float(float(cost.at(np.array([0.0]))[0]))}",
        "climb_20_J_per_m = "
        f"{_format_float(float(cost.at(np.array([20.0]))[0]))}",
        "descend_20_J_per_m = "
        f"{_format_float(float(cost.at(np.array([-20.0]))[0]))}",
        'energy_scope = "traverse only; no dwell, so these are not Day 11 totals"',
        "",
        "# What planning changed, which is the question the day was set.",
        "[verdict]",
        f"candidate_regions = 9",
        f"candidates_with_terrain = {len(candidates)}",
        "open_with_straight_lines = "
        + str(sum(1 for r in candidates if r.straight_traversable)),
        "open_with_planning = " + str(sum(1 for r in candidates if r.planned)),
        "opened_by_planning = [" + ", ".join(f'"{r.site.id}"' for r in opened) + "]",
        'closed_by = "a sampling artifact in Day 11, not terrain and not the straight line"',
        "straight_lines_walkable_all_along = true",
        "",
        "# The uncomfortable one. Once routes can bend, reaching the nearest cold",
        "# trap needs far less slope capability than standing on the region does,",
        "# and every planned route here is inside the crew slope limit. What is",
        "# left of the legged advantage for THIS mission is range, not gradient.",
        "[against_crew]",
        "crew_slope_limit_deg = "
        + _format_float(candidates[0].site.crew.maximum_slope_deg),
        "crew_traverse_range_km = "
        + _format_float(candidates[0].site.crew.traverse_range_km),
        "hardest_route_needs_deg = "
        + _format_float(max(r.capability_deg for r in candidates if r.planned)),
        "routes_inside_the_crew_slope_limit = "
        + str(sum(1 for r in candidates if r.planned and r.within_crew_slope)),
        "routes_inside_the_crew_traverse_range = "
        + str(sum(1 for r in candidates if r.planned and r.within_crew_range)),
        "regions_a_crew_could_reach = "
        + str(sum(1 for r in candidates if r.crew_could_reach)),
        "regions_only_a_robot_could_reach = "
        + str(sum(1 for r in candidates if r.planned and not r.crew_could_reach)),
        "limited_by = ["
        + ", ".join(
            f'"{r.site.id}: {"range" if not r.within_crew_range else "neither"}"'
            for r in candidates
            if r.planned and not r.crew_could_reach
        )
        + "]",
        "",
    ]
    for entry in routed:
        legs = entry.legs
        lines += [
            "[[region]]",
            f'id = "{entry.site.id}"',
            f'name = "{entry.site.name}"',
            "candidate = " + str(entry.site.is_candidate).lower(),
            f"nearest_shadow_km = {_format_float(entry.target.distance_km)}",
            "straight_ground_km = "
            f"{_format_float(entry.straight_distance_m / 1000.0)}",
            "straight_max_slope_deg = "
            f"{_format_float(entry.straight_max_slope_deg)}",
            "straight_traversable = " + str(entry.straight_traversable).lower(),
            "straight_energy_Wh = "
            + (
                _format_float(entry.straight_energy_Wh)
                if entry.straight_traversable
                else "nan"
            ),
            "minimum_capability_deg = " + _format_float(entry.capability_deg),
            "impassable_cell_fraction = "
            f"{_format_float(entry.impassable_cell_fraction)}",
        ]
        if legs is None:
            lines += [f'planned = false', f'no_route = "{entry.reason}"', ""]
            continue
        lines += [
            "planned = true",
            f"planned_ground_km = {_format_float(legs.distance_m / 1000.0)}",
            f"planned_max_slope_deg = {_format_float(legs.max_abs_slope_deg)}",
            f"planned_energy_Wh = {_format_float(legs.energy_Wh)}",
            "outbound_km = "
            f"{_format_float(legs.outbound.distance_m / 1000.0)}",
            f"inbound_km = {_format_float(legs.inbound.distance_m / 1000.0)}",
            "return_is_outbound_reversed = "
            + str(
                legs.outbound.steps == legs.inbound.steps
                and bool(
                    np.array_equal(legs.outbound.rows, legs.inbound.rows[::-1])
                )
            ).lower(),
            f"one_way_km = {_format_float(entry.one_way_km)}",
            "within_crew_slope_limit = " + str(entry.within_crew_slope).lower(),
            "within_crew_traverse_range = " + str(entry.within_crew_range).lower(),
            "crew_could_reach = " + str(entry.crew_could_reach).lower(),
            f"detour_ratio = {_format_float(entry.detour)}",
            f"energy_ratio = {_format_float(entry.saving)}",
            "undecidable_steps = " + str(legs.undecidable_steps()),
            "opened_by_planning = " + str(entry.opened_by_planning).lower(),
            "",
        ]

    lines += [
        "# What coarsening the baseline would have bought, which is the whole",
        "# answer and none of the terrain.",
        "",
    ]
    for entry in routed:
        lines += [
            "[[baseline]]",
            f'id = "{entry.site.id}"',
            "posting_m = ["
            + ", ".join(str(k) for k in sorted(entry.traversable_fraction_at))
            + "]",
            "traversable_fraction = ["
            + ", ".join(
                _format_float(entry.traversable_fraction_at[k])
                for k in sorted(entry.traversable_fraction_at)
            )
            + "]",
            "",
        ]

    detours = [r.detour for r in both]
    savings = [r.saving for r in both]
    lines += [
        "[trade]",
        "regions_with_both_routes = " + str(len(both)),
        f"median_detour_ratio = {_format_float(float(np.median(detours)))}",
        f"median_energy_ratio = {_format_float(float(np.median(savings)))}",
        f"worst_detour_ratio = {_format_float(float(np.max(detours)))}",
        f"best_energy_ratio = {_format_float(float(np.min(savings)))}",
        "",
        "# Did the ranking change once routes could bend.",
        "[ranking]",
        "by_straight_line = ["
        + ", ".join(
            f'"{r.site.id}"'
            for r in sorted(
                (r for r in candidates if r.straight_traversable),
                key=lambda r: r.straight_energy_Wh,
            )
        )
        + "]",
        "by_planned_route = ["
        + ", ".join(
            f'"{r.site.id}"'
            for r in sorted(
                (r for r in candidates if r.planned),
                key=lambda r: cast(Legs, r.legs).energy_Wh,
            )
        )
        + "]",
        "",
        "[answer]",
        'statement = """',
        "Neither. They were closed by a sampling artifact.",
        "",
        "Day 11 sampled every route at a fixed four hundred points regardless of",
        "its length. On a half-kilometre route that is a two metre step on a five",
        "metre grid, so consecutive samples share a cell and report no rise, and",
        "the next pair charges a whole cell-to-cell rise against a sub-cell run.",
        "The 66 degree wall at Malapert Massif is 30 degrees and the 53 degree one",
        "at de Gerlache is 26. Both were always walkable, and the count of",
        "candidate regions open to a legged day trip is six of nine rather than",
        "three.",
        "",
        "The planner found it rather than confirming it. A search over the grid",
        "steps cell to cell by construction, so it could not reproduce walls that",
        "existed only between sub-cell samples, and the disagreement was the",
        "signal. sample_transect now refuses to sample finer than the grid.",
        "",
        "That makes the day's own capability a null result, which is worth saying",
        "plainly rather than burying. Planning saves around one percent of energy",
        "where both routes exist and takes a few degrees off the steepest step. It",
        "does not reorder the regions. On these sites, for this errand, a straight",
        "line was an adequate approximation and the elaborate thing did not earn",
        "its place -- it earned its place by finding the bug.",
        "",
        "What did not change is the bound. Three candidate regions have no 5 m",
        "product in this archive, and no planner reaches a place there is no map",
        "of. Six of nine is still a lower bound and coverage is still the largest",
        "single term in it.",
        "",
        "And one thing got worse for the legged case, which is worth stating",
        "plainly because it was not the day's expected result. Every planned",
        "route here needs less than twelve degrees of slope capability, which is",
        "inside the twenty degree crew limit. Once a route can bend, reaching the",
        "nearest cold trap stops being a gradient problem at all. What separates",
        "the platform from a suited crew on this mission is the two kilometre",
        "traverse range, not the slope -- so the legged argument for these",
        "specific sorties is range and endurance, and Day 11's terrain-coverage",
        "advantage is an argument about where a robot can go rather than about",
        "this particular errand.",
        '"""',
        "",
        f"# {tally(rows)}",
        "",
        *toml_lines(rows),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-run the candidate regions with planned routes."
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
        hotel_load_W=HOTEL_LOAD_W,
    )
    print(
        f"  cost curve: flat {cost.at(np.array([0.0]))[0]:.1f} J/m, "
        f"climbing 20° {cost.at(np.array([20.0]))[0]:.1f}, "
        f"descending 20° {cost.at(np.array([-20.0]))[0]:.1f}, "
        f"impassable beyond {cost.limit_deg:.1f}°"
    )

    routed: list[Routed] = []
    showcase: Routed | None = None
    cell = 0.0
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
        cell = raster.cell_size_m
        entry = route_site(
            site, raster=raster, product=product.id, cost=cost, keep_window=True
        )
        del raster
        if entry is None:
            print(f"  {site.name:24s} no permanent shadow in the window")
            continue
        legs = entry.legs
        if legs is None:
            print(f"  {site.name:24s} no route: {entry.reason}")
        else:
            print(
                f"  {site.name:24s} straight {entry.straight_distance_m / 1000:5.2f} km "
                f"max {entry.straight_max_slope_deg:5.1f}° "
                f"{'walkable ' if entry.straight_traversable else 'REFUSED  '}"
                f"| planned {legs.distance_m / 1000:5.2f} km "
                f"max {legs.max_abs_slope_deg:4.1f}° {legs.energy_Wh:6.1f} Wh "
                f"| needs {entry.capability_deg:4.1f}°"
            )
        # The subject of the route figure is wherever the planner most
        # visibly went around something: the largest drop in peak slope
        # between the straight line and the route that replaces it.
        if legs is not None and (
            showcase is None
            or entry.straight_max_slope_deg - legs.max_abs_slope_deg
            > showcase.straight_max_slope_deg
            - cast(Legs, showcase.legs).max_abs_slope_deg
        ):
            showcase = entry
        routed.append(
            Routed(
                **{
                    **{
                        name: getattr(entry, name)
                        for name in Routed.__dataclass_fields__
                    },
                    "window_elevation_m": (
                        entry.window_elevation_m
                        if entry is showcase
                        else None
                    ),
                }
            )
        )
        if showcase is entry:
            showcase = routed[-1]

    if showcase is None:
        print("no region produced a route; the figures have no subject")
        return 1

    arguments.figure_directory.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)

    for name, figure in (
        ("planned-route", build_route_figure(showcase, cell)),
        ("planned-against-straight", build_trade_figure(routed)),
        ("envelope-with-planning", build_envelope_figure(routed)),
    ):
        target = arguments.figure_directory / f"{name}.png"
        figure.savefig(target, dpi=200)
        plt.close(figure)
        print(f"wrote {target.relative_to(REPOSITORY_ROOT)}")

    arguments.report.write_text(build_report(routed, cost), encoding="utf-8")
    print(f"wrote {arguments.report.relative_to(REPOSITORY_ROOT)}")

    print("\n  measured against extrapolated\n")
    print(text_table(boundary_rows(routed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
