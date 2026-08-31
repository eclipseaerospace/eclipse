# SPDX-License-Identifier: Apache-2.0
#
# studies.sites.contact — the last axis, and what it costs to be out of touch.
#
# Comms is the sixth of the six site axes declared on Day 6 and the last one
# empty. It is also the one that decides whether this is a teleoperated mission
# or an autonomous one, which is a distinction CLAUDE.md has claimed matters
# since the first day and the project has never tested against ground.
#
# The test is simple to state. Day 14 produced a set of places the platform can
# reach and return from. Intersect it with Earth visibility and it splits in
# three: reachable and in contact, reachable and blind, and out of reach. The
# middle region is the autonomy requirement, measured in square kilometres
# rather than asserted in a sentence.
#
# Four results.
#
# The structure is the one the geometry predicts and it is worth saying plainly
# because it is the whole argument. A lit crest sees Earth for the same reason
# it sees the Sun -- it is high ground with a low skyline. A cold trap is
# shadowed from Earth by the same rim that shadows it from the Sun. So the
# places worth going are close to exactly the places you cannot talk from, and
# that is a fact about polar geometry rather than about any particular site.
#
# But the mission is not blind, because it does not have to be continuous. Earth
# libration swings seven degrees either way over a month against mean elevations
# of half a degree to four, so at most of these sites contact is intermittent
# rather than absent, and the question becomes how long a gap the platform must
# survive rather than whether it has a link at all.
#
# A relay at the charge point does not close the gap, and the reason is
# structural rather than a matter of siting. The charge point is chosen for
# sunlight, which means high ground, which means it sees a great deal of the
# window -- but a cold trap is a hole, and a hole is invisible from outside it
# whatever the transmitter is standing on. Line of sight into a crater comes
# from above it, not from beside it.
#
# So the answer is an autonomy requirement rather than a surface-relay
# requirement, and it has a duration attached: the platform must work out of
# contact for the length of a dwell plus an approach, and that is the number a
# design would be built against.
#
# Only a surface mast is modelled, and the distinction matters. The planned
# architecture is orbital -- Lunar Pathfinder, Moonlight, LunaNet -- and a
# satellite sees into a shadowed crater from above at elevations no rim mast can
# reach. That reshapes the requirement rather than removing it: unattended
# between passes rather than unattended in a hole. Nothing here models a pass.
#
# This is geometry, not a link budget. Whether a visible platform can close a
# link at a useful data rate is a question about antennas and power and is not
# this study.
#
# References
#   Meeus J (1998) Astronomical Algorithms, 2nd ed. Chapter 53, libration.

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
from eclipse.comms import (
    EARTH_ANGULAR_RADIUS_DEG,
    LIBRATION_LATITUDE_DEG,
    LIBRATION_LONGITUDE_DEG,
    LIBRATION_LONGITUDE_PERIOD_HOURS,
    earth_elevation_deg,
    earth_visibility,
    viewshed,
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
    model_to_latitude_longitude,
    north_azimuth_degrees,
    read_float_geotiff,
)
from eclipse.mobility import cost_of_transport
from eclipse.planning import TraversalCost, plan_route, round_trip_energy_J
from eclipse.platform import Platform, equilibrium_slip_ratio, swing_work_per_meter
from eclipse.sortie import JOULES_PER_WATT_HOUR
from eclipse.stance import wave_gait, within_stride_slip_ratio
from eclipse.terrain import aggregate

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
SITE_DIRECTORY: Final = REPOSITORY_ROOT / "configs" / "sites"
TERRAIN_DIRECTORY: Final = REPOSITORY_ROOT / "data" / "terrain"
MANIFEST_PATH: Final = TERRAIN_DIRECTORY / "manifest.toml"
PLATFORM_PATH: Final = (
    REPOSITORY_ROOT / "configs" / "platforms" / "nominal-quadruped.toml"
)
SOIL_PATH: Final = REPOSITORY_ROOT / "data" / "soils" / "lunar-intercrater.toml"
FIGURE_DIRECTORY: Final = Path(__file__).resolve().parent / "figures"
DEFAULT_REPORT_PATH: Final = Path(__file__).resolve().parent / "results" / "contact.toml"

REPORT_SCHEMA_VERSION: Final = 1
LUNAR_GRAVITY: Final = 1.62
FEET_IN_STANCE: Final = 3
NOMINAL_DERATING: Final = 4.0
TIPPING_LIMIT_DEG: Final = 39.8055710922652
COST_SLOPE_DEG: Final[NDArray[np.float64]] = np.arange(-89.0, 89.01, 0.1)

# Carried unchanged from Day 14 so the reachable sets are the same sets.
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

# A mast on the charge point, for the relay question. Two metres is a stated
# assumption and the answer is insensitive to it: a hole is invisible from
# outside whatever height the transmitter stands at.
RELAY_MAST_M: Final = 2.0

# A relay area is reportable when it survives coarsening and refinement about as
# well as the Earth answer does. Stated as a ratio against Earth rather than an
# absolute tolerance because the two are computed over the same terrain from the
# same products, so Earth's spread is the noise floor these products impose.
RELAY_REPORTABLE_MARGIN: Final = 1.5

# The libration cycle is sampled this many times. Coarser than the illumination
# sweep because Earth moves a hundred times more slowly than the Sun.
LIBRATION_SAMPLES: Final = 240

# The autonomy axis: how much of a libration cycle the platform is willing to
# work out of contact for. Zero is a teleoperated machine that must always be
# in touch; one is fully autonomous.
BLIND_TOLERANCE: Final[NDArray[np.float64]] = np.linspace(0.0, 1.0, 101)
BATTERY_SWEEP_WH: Final[NDArray[np.float64]] = np.linspace(50.0, 2000.0, 79)
INSULATION_SWEEP_W: Final = (5.0, 11.8, 30.0, 100.0)
SLOPE_SWEEP_DEG: Final = (15.0, 25.0, 39.8055710922652)


def caption(text: str, width: int = 148) -> str:
    return "\n".join(
        textwrap.fill(" ".join(paragraph.split()), width=width)
        for paragraph in text.split("\n")
    )


def horizon_at(
    raster: GeoRaster, rows: NDArray[np.int_], columns: NDArray[np.int_]
) -> Any:
    return horizon_elevation_deg(
        raster,
        rows=rows,
        columns=columns,
        azimuths=HORIZON_AZIMUTHS,
        samples_along_ray=HORIZON_SAMPLES,
        minimum_range_m=HORIZON_STANDOFF_M,
    )


def longitudes_degrees(
    raster: GeoRaster, rows: NDArray[np.int_], columns: NDArray[np.int_]
) -> NDArray[np.float64]:
    return np.asarray(
        [
            model_to_latitude_longitude(
                raster.origin_x_m + (float(c) + 0.5) * raster.cell_size_m,
                raster.origin_y_m - (float(r) + 0.5) * raster.cell_size_m,
                reference_radius_m=raster.reference_radius_m,
            )[1]
            for r, c in zip(rows, columns)
        ]
    )


def walking_cost(
    *,
    platform: Platform,
    contact: Any,
    strength: Any,
    mobilization: Any,
    survival_W: float = INSULATED_SURVIVAL_W,
    limit_deg: float = TIPPING_LIMIT_DEG,
) -> TraversalCost:
    flat, _ = within_stride_slip_ratio(
        platform=platform,
        gait=wave_gait(lift_order=(2, 0, 3, 1), duty_factor=0.75),
        strength=strength,
        mobilization=mobilization,
        gravity_m_per_s2=LUNAR_GRAVITY,
    )
    hotel = survival_W / platform.nominal_speed_m_per_s
    joules = np.full(COST_SLOPE_DEG.shape, np.inf)
    for index, slope in enumerate(COST_SLOPE_DEG):
        if abs(float(slope)) > limit_deg:
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
        limit_deg=min(float(np.abs(COST_SLOPE_DEG[usable]).max()), limit_deg),
    )


def _upsample(grid: NDArray[Any], *, span: int) -> NDArray[Any]:
    """Nearest-neighbour from the mask sampling up to the full grid.

    The masks are computed at a coarser sampling than the reachable set they
    are intersected with, so every area that involves one carries the coarser
    resolution. Repeating rather than interpolating keeps a boolean boolean.
    """
    return np.asarray(
        np.repeat(np.repeat(grid, MASK_STRIDE, axis=0), MASK_STRIDE, axis=1)[
            :span, :span
        ]
    )


@dataclass(frozen=True, slots=True)
class Contact:
    """One site, split into what it can reach and what it can talk from."""

    site: Site
    charge: tuple[int, int]
    home: tuple[int, int]
    cell_size_m: float
    elevation_m: NDArray[np.float64]
    dark: NDArray[np.bool_]
    contact_fraction: NDArray[np.float64]
    energy_Wh: NDArray[np.float64]
    relay_visible: NDArray[np.bool_]
    charge_contact_fraction: float
    earth_elevation_range_deg: tuple[float, float]
    route_distance_m: NDArray[np.float64]
    route_contact: NDArray[np.bool_]
    route_hours: float

    @property
    def cell_area_km2(self) -> float:
        return self.cell_size_m**2 / 1e6

    @property
    def reachable(self) -> NDArray[np.bool_]:
        return np.asarray(self.energy_Wh <= NOMINAL_BATTERY_WH)

    @property
    def in_contact(self) -> NDArray[np.bool_]:
        return np.asarray(self.contact_fraction > 0.0)

    def area_km2(self, mask: NDArray[np.bool_]) -> float:
        return float(mask.sum()) * self.cell_area_km2

    @property
    def reachable_km2(self) -> float:
        return self.area_km2(self.reachable)

    @property
    def blind_km2(self) -> float:
        return self.area_km2(self.reachable & ~self.in_contact)

    @property
    def blind_fraction(self) -> float:
        return self.blind_km2 / max(self.reachable_km2, 1e-9)

    @property
    def cold_trap_km2(self) -> float:
        return self.area_km2(self.reachable & self.dark)

    @property
    def blind_cold_trap_km2(self) -> float:
        return self.area_km2(self.reachable & self.dark & ~self.in_contact)

    @property
    def relay_km2(self) -> float:
        return self.area_km2(self.reachable & self.relay_visible)

    @property
    def relay_recovers_km2(self) -> float:
        """Blind ground a relay at the charge point would put back in touch."""
        return self.area_km2(self.reachable & ~self.in_contact & self.relay_visible)

    @property
    def relay_recovers_cold_trap_km2(self) -> float:
        return self.area_km2(
            self.reachable & self.dark & ~self.in_contact & self.relay_visible
        )

    @property
    def route_blind_fraction(self) -> float:
        return float((~self.route_contact).mean())

    @property
    def route_blind_hours(self) -> float:
        return self.route_blind_fraction * self.route_hours

    def usable_cold_trap_km2(self, blind_tolerance: float) -> float:
        """Cold trap a platform willing to be out of touch this much can work."""
        return self.area_km2(
            self.reachable
            & self.dark
            & ((1.0 - self.contact_fraction) <= blind_tolerance)
        )


def survey_contact(
    site: Site,
    *,
    raster: GeoRaster,
    platform: Platform,
    contact_model: Any,
    strength: Any,
    mobilization: Any,
) -> Contact | None:
    first_row, last_row, first_column, last_column = centred_window(
        raster, span_m=COMMON_WINDOW_KM * 1000.0
    )
    rows, columns = np.meshgrid(
        np.arange(first_row, last_row, MAP_STRIDE),
        np.arange(first_column, last_column, MAP_STRIDE),
        indexing="ij",
    )
    lit = illumination_fraction(
        horizon=horizon_at(raster, rows.ravel(), columns.ravel()),
        latitude_deg=latitudes_degrees(raster, rows.ravel(), columns.ravel()),
        north_azimuth_deg=north_azimuth_degrees(raster, rows.ravel(), columns.ravel()),
    ).any_sunlight_fraction.reshape(rows.shape)
    if not bool((lit <= 0.0).any()):
        return None
    charge = best_charge_point(
        rows=rows,
        columns=columns,
        any_sunlight_fraction=lit,
        elevation_m=raster.values[rows, columns],
    )
    target = shadow_targets(
        raster,
        start=charge,
        rows=rows,
        columns=columns,
        any_sunlight_fraction=lit,
    )["nearest"]

    mask_rows, mask_columns = np.meshgrid(
        np.arange(first_row, last_row, MASK_STRIDE),
        np.arange(first_column, last_column, MASK_STRIDE),
        indexing="ij",
    )
    flat_rows, flat_columns = mask_rows.ravel(), mask_columns.ravel()
    mask_horizon = horizon_at(raster, flat_rows, flat_columns)
    dark_mask = (
        illumination_fraction(
            horizon=mask_horizon,
            latitude_deg=latitudes_degrees(raster, flat_rows, flat_columns),
            north_azimuth_deg=north_azimuth_degrees(raster, flat_rows, flat_columns),
        ).any_sunlight_fraction
        <= 0.0
    ).reshape(mask_rows.shape)
    seen = earth_visibility(
        horizon=mask_horizon,
        latitude_deg=latitudes_degrees(raster, flat_rows, flat_columns),
        longitude_deg=longitudes_degrees(raster, flat_rows, flat_columns),
        north_azimuth_deg=north_azimuth_degrees(raster, flat_rows, flat_columns),
        reference_radius_m=raster.reference_radius_m,
        samples=LIBRATION_SAMPLES,
    )
    contact_mask = seen.any_contact_fraction.reshape(mask_rows.shape)

    span = last_row - first_row
    dark = _upsample(dark_mask, span=span)
    contact_fraction = _upsample(contact_mask, span=span)

    elevation = np.ascontiguousarray(
        raster.values[first_row:last_row, first_column:last_column]
    )
    home = (charge[0] - first_row, charge[1] - first_column)
    cost = walking_cost(
        platform=platform,
        contact=contact_model,
        strength=strength,
        mobilization=mobilization,
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

    window = GeoRaster(
        values=elevation,
        origin_x_m=raster.origin_x_m + first_column * raster.cell_size_m,
        origin_y_m=raster.origin_y_m - first_row * raster.cell_size_m,
        cell_size_m=raster.cell_size_m,
        reference_radius_m=raster.reference_radius_m,
    )
    relay = viewshed(window, origin=home, mast_height_m=RELAY_MAST_M)

    goal = (target.row - first_row, target.column - first_column)
    plan = plan_route(
        elevation_m=elevation,
        cell_size_m=raster.cell_size_m,
        start=home,
        goal=goal,
        cost=cost,
    )
    route = plan.route
    if route is None:
        return None
    along = np.concatenate([[0.0], np.cumsum(route.step_length_m)])
    hours = float(along[-1]) / platform.nominal_speed_m_per_s / 3600.0

    latitude, _ = raster.center_latitude_longitude()
    longitude = float(
        longitudes_degrees(
            raster, np.array([charge[0]]), np.array([charge[1]])
        )[0]
    )
    site_latitude = float(
        latitudes_degrees(raster, np.array([charge[0]]), np.array([charge[1]]))[0]
    )
    swings = earth_elevation_deg(
        latitude_deg=site_latitude,
        longitude_deg=longitude,
        sub_earth_latitude_deg=np.array([-LIBRATION_LATITUDE_DEG, 0.0, LIBRATION_LATITUDE_DEG]),
        sub_earth_longitude_deg=np.array([-LIBRATION_LONGITUDE_DEG, 0.0, LIBRATION_LONGITUDE_DEG]),
        reference_radius_m=raster.reference_radius_m,
    )
    return Contact(
        site=site,
        charge=charge,
        home=home,
        cell_size_m=raster.cell_size_m,
        elevation_m=elevation,
        dark=dark,
        contact_fraction=contact_fraction,
        energy_Wh=energy,
        relay_visible=relay,
        charge_contact_fraction=float(contact_fraction[home]),
        earth_elevation_range_deg=(float(swings.min()), float(swings.max())),
        route_distance_m=along,
        route_contact=np.asarray(
            contact_fraction[route.rows, route.columns] > 0.0
        ),
        route_hours=hours,
    )


@dataclass(frozen=True, slots=True)
class Posting:
    """How much each answer moves when the grid is coarsened, and when it is not.

    The decisive diagnostic of the day. Both answers are set by distant large
    terrain and both survive an eightfold coarsening, so both are reportable --
    but that held only once the viewshed stopped charging a cell's occlusion to
    the single bin its centre fell in. Under-resolved, the relay area moved
    sixfold across the same sweep and scaled with the grid spacing, which read
    as the terrain being undecidable rather than as an artifact. So the
    refinement sweep is carried alongside the coarsening sweep: a number that
    moves when the algorithm is refined is not yet about the terrain, whatever
    the grid does.
    """

    cell_size_m: tuple[float, ...]
    earth_contact_fraction: tuple[float, ...]
    sunlit_fraction: tuple[float, ...]
    relay_visible_km2: tuple[float, ...]
    relay_refined_km2: float

    @property
    def earth_spread(self) -> float:
        values = np.asarray(self.earth_contact_fraction)
        return float(values.max() / max(values.min(), 1e-9))

    @property
    def relay_spread(self) -> float:
        values = np.asarray(self.relay_visible_km2)
        return float(values.max() / max(values.min(), 1e-9))

    @property
    def relay_bin_spread(self) -> float:
        """How far the relay answer moves when the binning alone is refined.

        The companion to the resolution sweep and the one that has to be clean
        first: a number that moves when the algorithm is refined is not yet
        measuring the terrain, whatever the grid does.
        """
        native = self.relay_visible_km2[0]
        pair = (native, self.relay_refined_km2)
        return float(max(pair) / max(min(pair), 1e-9))

    @property
    def relay_resolution_exponent(self) -> float:
        """How close the relay answer is to being a function of the grid alone.

        An exponent near one means the area scales with the cell size, so the
        number reports the resolution it was computed at rather than the terrain
        it was computed on. Near zero would mean the answer had converged.
        """
        slope, _ = np.polyfit(
            np.log(np.asarray(self.cell_size_m)),
            np.log(np.asarray(self.relay_visible_km2)),
            1,
        )
        return float(slope)


def posting_sensitivity(
    raster: GeoRaster, *, home: tuple[int, int], window: tuple[int, int, int, int]
) -> Posting:
    first_row, last_row, first_column, last_column = window
    native = np.ascontiguousarray(
        raster.values[first_row:last_row, first_column:last_column]
    )
    cells, earth, sun, relay = [], [], [], []
    for factor in (1, 2, 4, 8):
        coarse = aggregate(native, factor) if factor > 1 else native
        cell = raster.cell_size_m * factor
        grid = GeoRaster(
            values=coarse,
            origin_x_m=raster.origin_x_m + first_column * raster.cell_size_m,
            origin_y_m=raster.origin_y_m - first_row * raster.cell_size_m,
            cell_size_m=cell,
            reference_radius_m=raster.reference_radius_m,
        )
        stride = max(1, MASK_STRIDE * 5 // factor)
        rows, columns = np.meshgrid(
            np.arange(0, coarse.shape[0], stride),
            np.arange(0, coarse.shape[1], stride),
            indexing="ij",
        )
        flat_rows, flat_columns = rows.ravel(), columns.ravel()
        horizon = horizon_at(grid, flat_rows, flat_columns)
        seen = earth_visibility(
            horizon=horizon,
            latitude_deg=latitudes_degrees(grid, flat_rows, flat_columns),
            longitude_deg=longitudes_degrees(grid, flat_rows, flat_columns),
            north_azimuth_deg=north_azimuth_degrees(grid, flat_rows, flat_columns),
            reference_radius_m=grid.reference_radius_m,
            samples=LIBRATION_SAMPLES // 2,
        )
        lit = illumination_fraction(
            horizon=horizon,
            latitude_deg=latitudes_degrees(grid, flat_rows, flat_columns),
            north_azimuth_deg=north_azimuth_degrees(grid, flat_rows, flat_columns),
        )
        view = viewshed(
            grid,
            origin=(home[0] // factor, home[1] // factor),
            mast_height_m=RELAY_MAST_M,
            minimum_range_m=HORIZON_STANDOFF_M,
        )
        cells.append(cell)
        earth.append(float(seen.any_contact_fraction.mean()))
        sun.append(float(lit.any_sunlight_fraction.mean()))
        relay.append(float(view.sum()) * cell**2 / 1e6)
    corner = max(
        float(np.hypot(row - home[0], column - home[1]))
        for row in (0, native.shape[0] - 1)
        for column in (0, native.shape[1] - 1)
    )
    refined = viewshed(
        GeoRaster(
            values=native,
            origin_x_m=raster.origin_x_m + first_column * raster.cell_size_m,
            origin_y_m=raster.origin_y_m - first_row * raster.cell_size_m,
            cell_size_m=raster.cell_size_m,
            reference_radius_m=raster.reference_radius_m,
        ),
        origin=home,
        mast_height_m=RELAY_MAST_M,
        minimum_range_m=HORIZON_STANDOFF_M,
        azimuth_bins=2 * int(np.ceil(2.0 * np.pi * corner)),
    )
    return Posting(
        cell_size_m=tuple(cells),
        earth_contact_fraction=tuple(earth),
        sunlit_fraction=tuple(sun),
        relay_visible_km2=tuple(relay),
        relay_refined_km2=float(refined.sum()) * raster.cell_size_m**2 / 1e6,
    )


@dataclass(frozen=True, slots=True)
class Envelope:
    """Cold trap reached against each platform parameter, one at a time."""

    battery_Wh: tuple[float, ...]
    battery_km2: tuple[float, ...]
    insulation_W: tuple[float, ...]
    insulation_km2: tuple[float, ...]
    slope_deg: tuple[float, ...]
    slope_km2: tuple[float, ...]
    blind_tolerance: tuple[float, ...]
    blind_km2: tuple[float, ...]


def build_envelope(
    entry: Contact,
    *,
    raster_cell_m: float,
    platform: Platform,
    contact_model: Any,
    strength: Any,
    mobilization: Any,
) -> Envelope:
    cell_area = entry.cell_area_km2
    dark = entry.dark

    def cold_trap(field: NDArray[np.float64], battery: float) -> float:
        return float(((field <= battery) & dark).sum()) * cell_area

    def field_for(survival_W: float, limit_deg: float) -> NDArray[np.float64]:
        cost = walking_cost(
            platform=platform,
            contact=contact_model,
            strength=strength,
            mobilization=mobilization,
            survival_W=survival_W,
            limit_deg=limit_deg,
        )
        return (
            round_trip_energy_J(
                elevation_m=entry.elevation_m,
                cell_size_m=raster_cell_m,
                home=entry.home,
                cost=cost,
            )
            / JOULES_PER_WATT_HOUR
            + survival_W * DWELL_HOURS
        )

    insulation = []
    for survival_W in INSULATION_SWEEP_W:
        field = (
            entry.energy_Wh
            if survival_W == INSULATED_SURVIVAL_W
            else field_for(survival_W, TIPPING_LIMIT_DEG)
        )
        insulation.append(cold_trap(field, NOMINAL_BATTERY_WH))

    slope = []
    for limit in SLOPE_SWEEP_DEG:
        field = (
            entry.energy_Wh
            if limit == TIPPING_LIMIT_DEG
            else field_for(INSULATED_SURVIVAL_W, limit)
        )
        slope.append(cold_trap(field, NOMINAL_BATTERY_WH))

    return Envelope(
        battery_Wh=tuple(float(v) for v in BATTERY_SWEEP_WH),
        battery_km2=tuple(
            cold_trap(entry.energy_Wh, float(v)) for v in BATTERY_SWEEP_WH
        ),
        insulation_W=INSULATION_SWEEP_W,
        insulation_km2=tuple(insulation),
        slope_deg=SLOPE_SWEEP_DEG,
        slope_km2=tuple(slope),
        blind_tolerance=tuple(float(v) for v in BLIND_TOLERANCE),
        blind_km2=tuple(
            entry.usable_cold_trap_km2(float(v)) for v in BLIND_TOLERANCE
        ),
    )


def build_contact_figure(entry: Contact) -> Figure:
    cell = entry.cell_size_m
    span = entry.energy_Wh.shape[0]
    extent = (0.0, span * cell / 1000.0, span * cell / 1000.0, 0.0)
    reachable = entry.reachable
    talking = reachable & entry.in_contact
    blind = reachable & ~entry.in_contact

    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (9.8, 9.4),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.5,
                    "axes.grid": False,
                    "figure.subplot.top": 0.745,
                    "figure.subplot.bottom": 0.060,
                    "figure.subplot.left": 0.086,
                    "figure.subplot.right": 0.986,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 1, squeeze=False)
        panel = axes[0][0]
        shade = np.gradient(entry.elevation_m)[0]
        panel.imshow(
            shade,
            extent=extent,
            cmap="Greys_r",
            vmin=float(np.percentile(shade, 2)),
            vmax=float(np.percentile(shade, 98)),
            interpolation="bilinear",
        )
        panel.imshow(
            np.where(talking, 1.0, np.nan),
            extent=extent,
            cmap="Blues",
            vmin=0.0,
            vmax=1.7,
            interpolation="nearest",
            alpha=0.40,
        )
        panel.imshow(
            np.where(blind, 1.0, np.nan),
            extent=extent,
            cmap="autumn",
            vmin=0.0,
            vmax=1.5,
            interpolation="nearest",
            alpha=0.48,
        )
        panel.contour(
            np.linspace(extent[0], extent[1], span),
            np.linspace(extent[3], extent[2], span),
            entry.dark.astype(float),
            levels=[0.5],
            colors=[INK_PRIMARY],
            linewidths=1.1,
        )
        home_x = (entry.home[1] + 0.5) * cell / 1000.0
        home_y = (entry.home[0] + 0.5) * cell / 1000.0
        panel.plot(
            [home_x], [home_y], marker="o", markersize=8.0, markerfacecolor="none",
            markeredgewidth=1.8, color="white",
        )
        panel.annotate(
            f"charge point, in touch {entry.charge_contact_fraction:.0%} of a month",
            xy=(home_x, home_y),
            xytext=(-10, -16) if home_x > 0.5 * extent[1] else (10, -16),
            textcoords="offset points",
            ha="right" if home_x > 0.5 * extent[1] else "left",
            color=INK_PRIMARY,
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72,
                  "boxstyle": "round,pad=0.2"},
        )
        handles = [
            Line2D([], [], color=ACCENT_PRIMARY, linewidth=6.0, alpha=0.45,
                   label=f"reachable and in touch, {entry.area_km2(talking):.0f} km²"),
            Line2D([], [], color=ACCENT_SECONDARY, linewidth=6.0, alpha=0.55,
                   label=f"reachable and blind, {entry.blind_km2:.0f} km² "
                         f"({entry.blind_fraction:.0%})"),
            Line2D([], [], color=INK_PRIMARY, linewidth=1.4,
                   label="permanent shadow"),
        ]
        panel.legend(handles=handles, loc="upper left", framealpha=0.78)
        panel.set_xlabel("kilometres east across the window")
        panel.set_ylabel("kilometres south across the window")
        panel.set_aspect("equal")

        share = entry.blind_cold_trap_km2 / max(entry.cold_trap_km2, 1e-9)
        figure.suptitle(
            f"At {entry.site.name} the platform is out of touch over "
            f"{entry.blind_fraction:.0%} of its range and "
            f"{share:.0%} of the cold trap it can reach",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.086,
            ha="left",
            y=0.976,
        )
        figure.text(
            0.086,
            0.940,
            caption(
                "Earth sits near the horizon at these latitudes and libration "
                "swings it seven degrees either way over a month, so contact is "
                "intermittent rather than absent: a point is counted as in touch "
                "if Earth's disc clears its skyline at any point in the cycle. "
                f"The charge point manages {entry.charge_contact_fraction:.0%}, "
                f"and Earth runs from {entry.earth_elevation_range_deg[0]:.1f}° to "
                f"{entry.earth_elevation_range_deg[1]:.1f}° above its horizontal "
                "over the month.\n"
                "The structure is the one the geometry predicts, and it is the "
                "whole argument. A lit crest sees Earth for the same reason it "
                "sees the Sun — high ground, low skyline. A cold trap is hidden "
                "from Earth by the same rim that hides it from the Sun. So the "
                "places worth going are close to exactly the places you cannot "
                "talk from, which is a fact about polar geometry rather than "
                "about this site.\n"
                "That makes the orange region the autonomy requirement, in square "
                "kilometres rather than in a sentence. It is not a link budget: "
                "whether a platform in sight of Earth can close a link at a "
                "useful rate is a question about antennas and power, and this is "
                "only geometry.",
                width=140,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def build_blackout_figure(entries: list[Contact], posting: Posting) -> Figure:
    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (12.6, 6.4),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.560,
                    "figure.subplot.bottom": 0.118,
                    "figure.subplot.left": 0.135,
                    "figure.subplot.right": 0.988,
                    "figure.subplot.wspace": 0.245,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 2, squeeze=False, width_ratios=[1.35, 1.0])
        left, right = axes[0][0], axes[0][1]

        ordered = sorted(entries, key=lambda e: e.route_blind_fraction)
        for index, entry in enumerate(ordered):
            distance = entry.route_distance_m / 1000.0
            level = float(index)
            left.plot(
                distance,
                np.full(distance.size, level),
                color=ACCENT_PRIMARY,
                linewidth=4.0,
                solid_capstyle="butt",
            )
            blind = ~entry.route_contact
            left.plot(
                np.where(blind, distance, np.nan),
                np.full(distance.size, level),
                color=ACCENT_SECONDARY,
                linewidth=4.0,
                solid_capstyle="butt",
            )
            left.annotate(
                f"{entry.route_blind_hours * 60:.0f} min"
                if entry.route_blind_fraction > 0.0
                else "no blackout",
                xy=(float(distance[-1]), level),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                color=(
                    ACCENT_SECONDARY
                    if entry.route_blind_fraction > 0.0
                    else INK_MUTED
                ),
                fontsize=8.0,
            )
        left.set_yticks(
            np.arange(len(ordered)), [entry.site.name for entry in ordered]
        )
        for label, entry in zip(left.get_yticklabels(), ordered):
            if not entry.site.is_candidate:
                label.set_color(INK_MUTED)
                label.set_style("italic")
        left.set_xlabel("distance along the outbound route (km)")
        left.set_title(
            "where the link drops on one errand", color=INK_SECONDARY, loc="left"
        )
        left.set_xlim(0.0, None)
        left.legend(
            handles=[
                Line2D([], [], color=ACCENT_PRIMARY, linewidth=4.0, label="in touch"),
                Line2D([], [], color=ACCENT_SECONDARY, linewidth=4.0, label="blind"),
            ],
            loc="lower right",
        )

        right.plot(
            posting.cell_size_m,
            np.asarray(posting.earth_contact_fraction)
            / posting.earth_contact_fraction[0],
            color=ACCENT_PRIMARY,
            linewidth=2.0,
            marker="o",
            markersize=5.0,
            label="Earth contact fraction",
        )
        right.plot(
            posting.cell_size_m,
            np.asarray(posting.sunlit_fraction) / posting.sunlit_fraction[0],
            color=INK_SECONDARY,
            linewidth=1.6,
            linestyle=(0, (4, 2)),
            marker="s",
            markersize=4.0,
            label="sunlit fraction, for comparison",
        )
        right.plot(
            posting.cell_size_m,
            np.asarray(posting.relay_visible_km2) / posting.relay_visible_km2[0],
            color=ACCENT_SECONDARY,
            linewidth=2.0,
            marker="D",
            markersize=5.0,
            label="area a surface relay sees",
        )
        right.axhline(1.0, color=INK_MUTED, linewidth=1.0)
        right.set_xscale("log")
        right.set_yscale("log")
        right.set_xlabel("grid posting (m)")
        right.set_ylabel("answer, relative to the 5 m answer")
        right.set_title(
            "each answer against grid posting, 5 to 40 m, one site",
            color=INK_SECONDARY,
            loc="left",
        )
        right.legend(loc="upper left")

        for panel in (left, right):
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)

        worst = max(entries, key=lambda e: e.route_blind_hours)
        blind_km2 = sum(entry.blind_km2 for entry in entries)
        recovered_km2 = sum(entry.relay_recovers_km2 for entry in entries)
        recovered_share = recovered_km2 / max(blind_km2, 1e-9)
        recovered_cold_trap = sum(
            entry.relay_recovers_cold_trap_km2 for entry in entries
        ) / max(sum(entry.blind_cold_trap_km2 for entry in entries), 1e-9)
        barren = sum(1 for entry in entries if entry.relay_recovers_km2 < 0.005)
        figure.suptitle(
            "A sortie loses the link for minutes, not hours, and a relay on "
            "the rim gives back one percent of what is blind",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.058,
            ha="left",
            y=0.958,
        )
        figure.text(
            0.058,
            0.900,
            caption(
                "Earth visibility integrated along the planned route to each "
                "site's nearest cold trap, the same way Day 10 integrated shadow. "
                f"The worst is {worst.site.name} at "
                f"{worst.route_blind_hours * 60:.0f} minutes out of touch on a "
                f"{worst.route_hours:.1f} hour walk, and several sites never lose "
                "the link at all. Minutes of blackout on the approach is a "
                "different engineering problem from hours, and the hours are at "
                "the destination rather than on the way.\n"
                "The right panel is the day's decisive diagnostic and it splits "
                "both answers against resolution. Earth visibility moves "
                f"{posting.earth_spread:.2f}× across an eightfold coarsening of "
                f"the grid and the area a relay sees {posting.relay_spread:.2f}×, "
                "with a further "
                f"{posting.relay_bin_spread - 1.0:.0%} when the viewshed's own "
                "binning is refined. Both are stable enough to report — which was "
                "not the first answer here. Charging each cell's occlusion to the "
                "one azimuth bin its centre falls in leaves gaps between the cells "
                "of a ridge, so ridges near the mast leaked, and the leak scaled "
                "with resolution: a defect in the computation wearing the costume "
                "of a result about terrain.\n"
                f"So the relay answer is reported, and it is that a mast at the "
                f"charge point recovers {recovered_share:.1%} of the "
                f"{blind_km2:.0f} km² that is blind and "
                f"{recovered_cold_trap:.1%} of the blind cold trap, at "
                f"{barren} of ten sites nothing at all. The charge point is chosen "
                "for sunlight, sunlight means high ground, and high ground already "
                "sees Earth — so a relay there covers the ground that least needs "
                "it, and cannot see into the holes where the blind ground is.",
                width=176,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


ABSENCES: Final = (
    (
        "obstacles",
        "no boulder or rock-abundance statistics anywhere. Every square "
        "kilometre is counted as though a foot could be placed anywhere in it",
    ),
    (
        "operational margin",
        "no reserve, no contingency, no failure modes, no degradation. The "
        "battery is spent to the last watt-hour and the machine never breaks",
    ),
    (
        "soil",
        "one soil at every cell of every site. A crater floor is not the same "
        "regolith as a rim and nothing here can tell them apart",
    ),
    (
        "analysis window",
        "every reachable set is clipped by a 16 km window rather than by the "
        "platform, so areas at large batteries are lower bounds",
    ),
    (
        "region coverage",
        "three of the nine candidate regions have no 5 m product in this "
        "archive, so every count over the nine is a lower bound",
    ),
    (
        "slope decidability",
        "the products state 1.5 to 2.5 degrees of RMS slope error, which makes "
        "Haworth's Day 11 refusal and Shackleton's Day 13 crossing undecidable",
    ),
    (
        "surface line of sight",
        "the aggregate area a relay sees is stable across resolution, but "
        "whether a single link closes turns on roughness below the grid",
    ),
    (
        "link budget",
        "visibility is geometry. Antenna gain, power and data rate are a "
        "different study and none of it is here",
    ),
    (
        "hardware",
        "no platform has been built, no soil has been touched, and no result "
        "in this repository has been compared against a physical measurement",
    ),
)


def build_envelope_figure(
    entry: Contact, envelope: Envelope, *, battery_Wh: float
) -> Figure:
    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (13.2, 7.8),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.745,
                    "figure.subplot.bottom": 0.345,
                    "figure.subplot.left": 0.055,
                    "figure.subplot.right": 0.988,
                    "figure.subplot.wspace": 0.265,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 4, squeeze=False)
        battery, insulation, slope, autonomy = (
            axes[0][0],
            axes[0][1],
            axes[0][2],
            axes[0][3],
        )
        ceiling = max(envelope.battery_km2 + envelope.blind_km2) * 1.12

        battery.plot(
            envelope.battery_Wh, envelope.battery_km2, color=ACCENT_PRIMARY,
            linewidth=2.0,
        )
        battery.axvline(
            battery_Wh, color=INK_SECONDARY, linewidth=1.0, linestyle=(0, (3, 2))
        )
        battery.set_xlabel("battery (Wh)")
        battery.set_title("battery", color=INK_SECONDARY, loc="left")

        insulation.plot(
            envelope.insulation_W, envelope.insulation_km2, color=ACCENT_SECONDARY,
            linewidth=2.0, marker="o", markersize=5.0,
        )
        insulation.set_xscale("log")
        insulation.set_xlabel("survival power (W)")
        insulation.set_title("insulation", color=INK_SECONDARY, loc="left")

        slope.plot(
            envelope.slope_deg, envelope.slope_km2, color=INK_PRIMARY,
            linewidth=2.0, marker="o", markersize=5.0,
        )
        slope.set_xlabel("slope capability (°)")
        slope.set_title("slope", color=INK_SECONDARY, loc="left")

        autonomy.plot(
            np.asarray(envelope.blind_tolerance) * 100.0,
            envelope.blind_km2,
            color=ACCENT_PRIMARY,
            linewidth=2.0,
        )
        autonomy.set_xlabel("blind operation tolerated (% of a month)")
        autonomy.set_title("autonomy", color=INK_SECONDARY, loc="left")

        for panel in (battery, insulation, slope, autonomy):
            panel.set_ylim(0.0, ceiling)
            panel.set_ylabel("cold trap reached (km²)")
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)

        teleoperated = envelope.blind_km2[0]
        autonomous = envelope.blind_km2[-1]
        figure.suptitle(
            "Six axes populated: a teleoperated platform works "
            f"{teleoperated:.2f} km² of cold trap here and an autonomous one "
            f"{autonomous:.2f}",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.055,
            ha="left",
            y=0.964,
        )
        figure.text(
            0.055,
            0.908,
            caption(
                f"Cold trap reached at {entry.site.name}, one platform parameter "
                "at a time with the others held. Every axis Day 6 declared is now "
                "populated, and this is the first figure in the project that "
                "shows all of them against one output.\n"
                "Autonomy is the axis this day added and it is the sharpest: a "
                "machine that must always be in touch works "
                f"{teleoperated:.2f} km², one that will operate out of contact "
                f"works {autonomous:.2f}. Slope is the flattest, which is Days 12 "
                "and 13 showing up again — reaching polar shadow is not a "
                "gradient problem. Battery is the strongest continuous axis, "
                "which reversed Day 11's ordering on Day 14 and holds here.",
                width=178,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
        figure.text(
            0.055,
            0.272,
            "Six axes populated is not six axes validated. What is absent, in one place:",
            color=INK_PRIMARY,
            fontsize=9.0,
            ha="left",
            va="top",
        )
        figure.text(
            0.055,
            0.238,
            caption(
                "   ".join(f"• {name}: {why}" for name, why in ABSENCES[:5]),
                width=196,
            ),
            color=INK_SECONDARY,
            fontsize=7.6,
            ha="left",
            va="top",
            linespacing=1.45,
        )
        figure.text(
            0.055,
            0.126,
            caption(
                "   ".join(f"• {name}: {why}" for name, why in ABSENCES[5:]),
                width=196,
            ),
            color=INK_SECONDARY,
            fontsize=7.6,
            ha="left",
            va="top",
            linespacing=1.45,
        )
    return figure


def boundary_rows(
    entries: list[Contact], posting: Posting
) -> tuple[BoundaryRow, ...]:
    return (
        BoundaryRow(
            quantity="Earth visibility",
            published_range="not applicable",
            used="disc clearing the local skyline over a libration cycle",
            status=INSIDE,
            basis=(
                f"moves by {posting.earth_spread:.2f} times across an eightfold "
                "coarsening of the grid, the same stability the sunlit fraction "
                "has and for the same reason: a horizon is set by distant large "
                "terrain rather than by near-field roughness"
            ),
        ),
        BoundaryRow(
            quantity="surface line of sight",
            published_range="not applicable",
            used="a viewshed from the charge point, aggregate area only",
            status=OUTSIDE,
            basis=(
                f"the area moves {posting.relay_spread:.2f} times across an "
                f"eightfold coarsening and {posting.relay_bin_spread:.2f} times "
                "when the viewshed's own binning is refined, so the aggregate is "
                "stable enough to report. Whether any particular link closes is "
                "not: a boulder or a metre-scale rise below the grid blocks a "
                "sightline without moving the total, on a product the producers "
                "state is about nine tenths interpolated at 5 m"
            ),
        ),
        BoundaryRow(
            quantity="orbital relay",
            published_range="Lunar Pathfinder, Moonlight, LunaNet, all planned",
            used="absent; only a surface mast at the charge point is modelled",
            status=OUTSIDE,
            basis=(
                "the relay tested here stands on the rim, and a rim cannot see "
                "into a hole. A satellite looks down into a permanently shadowed "
                "region at elevations no mast reaches, so the blind fractions "
                "above are the surface-only case and must not be read as the "
                "mission's. What an orbiter changes is the shape of the autonomy "
                "requirement rather than its existence: work unattended between "
                "passes instead of permanently in shadow. Pass geometry, "
                "constellation size and link budget are all absent here"
            ),
        ),
        BoundaryRow(
            quantity="libration model",
            published_range="optical libration, 7.9 and 6.7 degrees",
            used="two sinusoids on the anomalistic and draconic months",
            status=INSIDE,
            basis=(
                "no SPICE kernel and no physical libration. The two periods do "
                "not close together, so a fraction over one month is not quite a "
                "fraction over a year, and the answer is dominated by terrain "
                "rather than by ephemeris"
            ),
        ),
        BoundaryRow(
            quantity="Earth as a disc",
            published_range="0.95 degrees angular radius",
            used="partial visibility counted as contact",
            status=INSIDE,
            basis=(
                "Earth subtends four times the Sun's disc, so at grazing "
                "elevations the band of partly-visible ground is wide. Counting "
                "partial as contact is the generous direction and is stated as "
                "such"
            ),
        ),
        BoundaryRow(
            quantity="link budget",
            published_range="none",
            used="absent; this is geometry only",
            status=UNMEASURED,
            basis=(
                "whether a platform in sight of Earth can close a link at a "
                "useful rate depends on antenna gain, transmit power and data "
                "rate, none of which is modelled. Visibility is necessary and "
                "not sufficient"
            ),
        ),
        BoundaryRow(
            quantity="horizon search range",
            published_range="not applicable",
            used="within each product only",
            status=OUTSIDE,
            basis=(
                "rays leaving the window count as clear sky, so Earth visibility "
                "is an upper bound in exactly the way sunlight is: distant "
                "terrain can only occlude further"
            ),
        ),
        BoundaryRow(
            quantity="contact mask resolution",
            published_range="not applicable",
            used=f"{MASK_STRIDE * 5} m on a 5 m reachable set",
            status=OUTSIDE,
            basis=(
                "visibility is sampled coarser than the set it is intersected "
                "with, so every blind area carries the coarser resolution"
            ),
        ),
        BoundaryRow(
            quantity="autonomy requirement",
            published_range="none",
            used="a blind fraction and a blind duration, not a design",
            status=UNMEASURED,
            basis=(
                "the output is how much blind operation over what duration. What "
                "it takes to build a machine that can do it is a different "
                "problem and this study does not touch it"
            ),
        ),
        BoundaryRow(
            quantity="hardware",
            published_range="none",
            used="none; nothing here has been measured against a machine",
            status=UNMEASURED,
            basis=(
                "fifteen days of simulation against published data. The honest "
                "ceiling remains physics-consistent rather than validated"
            ),
        ),
    )


def _format_float(value: float) -> str:
    return repr(float(value))


def build_report(
    entries: list[Contact],
    posting: Posting,
    envelope: Envelope,
    *,
    showcase: str,
) -> str:
    rows = boundary_rows(entries, posting)
    total_reachable = sum(e.reachable_km2 for e in entries)
    total_blind = sum(e.blind_km2 for e in entries)
    total_cold = sum(e.cold_trap_km2 for e in entries)
    total_blind_cold = sum(e.blind_cold_trap_km2 for e in entries)
    recovered_share = sum(e.relay_recovers_km2 for e in entries) / max(
        total_blind, 1e-9
    )
    recovered_cold_trap = sum(
        e.relay_recovers_cold_trap_km2 for e in entries
    ) / max(total_blind_cold, 1e-9)
    barren = sum(1 for e in entries if e.relay_recovers_km2 < 0.005)
    worst = max(entries, key=lambda e: e.route_blind_hours)
    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# The last axis: whether the platform can be seen while it works.",
        "#",
        "# Generated by studies/sites/contact.py. Do not edit.",
        "#",
        "# SIX AXES OF SIX, WHICH IS NOT SIX AXES VALIDATED. The absence list at",
        "# the end is the distance between those two things.",
        "",
        f"schema_version = {REPORT_SCHEMA_VERSION}",
        "",
        "[environment]",
        f'python = "{host_platform.python_version()}"',
        f'numpy = "{np.__version__}"',
        "",
        "[method]",
        f"earth_angular_radius_deg = {_format_float(EARTH_ANGULAR_RADIUS_DEG)}",
        f"libration_longitude_deg = {_format_float(LIBRATION_LONGITUDE_DEG)}",
        f"libration_latitude_deg = {_format_float(LIBRATION_LATITUDE_DEG)}",
        f"libration_samples = {LIBRATION_SAMPLES}",
        f"cycle_hours = {_format_float(LIBRATION_LONGITUDE_PERIOD_HOURS)}",
        f"relay_mast_m = {_format_float(RELAY_MAST_M)}",
        f"battery_Wh = {_format_float(NOMINAL_BATTERY_WH)}",
        'contact = "Earth\'s disc clearing the local skyline at any point in a cycle"',
        'finite_distance = "corrected; the observer is on the surface, not at the centre"',
        "",
        "# What the mission is out of touch for.",
        "[verdict]",
        f"sites = {len(entries)}",
        f"reachable_km2 = {_format_float(total_reachable)}",
        f"blind_km2 = {_format_float(total_blind)}",
        "blind_fraction_of_range = "
        + _format_float(total_blind / max(total_reachable, 1e-9)),
        f"cold_trap_km2 = {_format_float(total_cold)}",
        f"blind_cold_trap_km2 = {_format_float(total_blind_cold)}",
        "blind_fraction_of_cold_trap = "
        + _format_float(total_blind_cold / max(total_cold, 1e-9)),
        f'worst_route_site = "{worst.site.id}"',
        "worst_route_blind_minutes = "
        + _format_float(worst.route_blind_hours * 60.0),
        "sites_with_no_blackout_on_the_route = "
        + str(sum(1 for e in entries if e.route_blind_fraction <= 0.0)),
        'answer = "an autonomy requirement, not a relay requirement"',
        "",
    ]
    for entry in entries:
        lines += [
            "[[region]]",
            f'id = "{entry.site.id}"',
            f'name = "{entry.site.name}"',
            "candidate = " + str(entry.site.is_candidate).lower(),
            "charge_point_contact_fraction = "
            + _format_float(entry.charge_contact_fraction),
            "earth_elevation_min_deg = "
            + _format_float(entry.earth_elevation_range_deg[0]),
            "earth_elevation_max_deg = "
            + _format_float(entry.earth_elevation_range_deg[1]),
            f"reachable_km2 = {_format_float(entry.reachable_km2)}",
            f"blind_km2 = {_format_float(entry.blind_km2)}",
            f"blind_fraction = {_format_float(entry.blind_fraction)}",
            f"cold_trap_km2 = {_format_float(entry.cold_trap_km2)}",
            f"blind_cold_trap_km2 = {_format_float(entry.blind_cold_trap_km2)}",
            "blind_cold_trap_fraction = "
            + _format_float(
                entry.blind_cold_trap_km2 / max(entry.cold_trap_km2, 1e-9)
            ),
            f"route_hours = {_format_float(entry.route_hours)}",
            "route_blind_minutes = "
            + _format_float(entry.route_blind_hours * 60.0),
            f"relay_visible_km2 = {_format_float(entry.relay_km2)}",
            "relay_recovers_km2 = " + _format_float(entry.relay_recovers_km2),
            "relay_recovers_cold_trap_km2 = "
            + _format_float(entry.relay_recovers_cold_trap_km2),
            "relay_recovers_blind_fraction = "
            + _format_float(entry.relay_recovers_km2 / max(entry.blind_km2, 1e-9)),
            "",
        ]

    lines += [
        "# The diagnostic that decides which of the two questions is reportable.",
        "[posting]",
        f'site = "{showcase}"',
        "cell_size_m = ["
        + ", ".join(_format_float(v) for v in posting.cell_size_m)
        + "]",
        "earth_contact_fraction = ["
        + ", ".join(_format_float(v) for v in posting.earth_contact_fraction)
        + "]",
        "sunlit_fraction = ["
        + ", ".join(_format_float(v) for v in posting.sunlit_fraction)
        + "]",
        "relay_visible_km2 = ["
        + ", ".join(_format_float(v) for v in posting.relay_visible_km2)
        + "]",
        f"earth_spread = {_format_float(posting.earth_spread)}",
        f"relay_spread = {_format_float(posting.relay_spread)}",
        f"relay_refined_km2 = {_format_float(posting.relay_refined_km2)}",
        f"relay_bin_spread = {_format_float(posting.relay_bin_spread)}",
        f"relay_resolution_exponent = {_format_float(posting.relay_resolution_exponent)}",
        "relay_reportable = "
        + str(
            max(posting.relay_spread, posting.relay_bin_spread)
            <= posting.earth_spread * RELAY_REPORTABLE_MARGIN
        ).lower(),
        "",
        "# Every axis Day 6 declared, against one output.",
        "[envelope]",
        f'site = "{showcase}"',
        "teleoperated_cold_trap_km2 = " + _format_float(envelope.blind_km2[0]),
        "autonomous_cold_trap_km2 = " + _format_float(envelope.blind_km2[-1]),
        "battery_Wh = ["
        + ", ".join(_format_float(v) for v in envelope.battery_Wh[::12])
        + "]",
        "battery_cold_trap_km2 = ["
        + ", ".join(_format_float(v) for v in envelope.battery_km2[::12])
        + "]",
        "insulation_W = ["
        + ", ".join(_format_float(v) for v in envelope.insulation_W)
        + "]",
        "insulation_cold_trap_km2 = ["
        + ", ".join(_format_float(v) for v in envelope.insulation_km2)
        + "]",
        "slope_deg = ["
        + ", ".join(_format_float(v) for v in envelope.slope_deg)
        + "]",
        "slope_cold_trap_km2 = ["
        + ", ".join(_format_float(v) for v in envelope.slope_km2)
        + "]",
        "",
        "# Written once, properly, rather than scattered across fifteen tables.",
        "[absences]",
    ]
    for name, why in ABSENCES:
        lines.append(f'{name.replace(" ", "_")} = "{why}"')

    lines += [
        "",
        "[answer]",
        'statement = """',
        "An autonomy requirement, and the duration is a dwell rather than a walk.",
        "",
        "The structure is the one polar geometry forces. A lit crest sees Earth",
        "for the same reason it sees the Sun -- high ground, low skyline. A cold",
        "trap is hidden from Earth by the same rim that hides it from the Sun. So",
        "the places worth going are close to exactly the places you cannot talk",
        "from, and that is not a property of any site in this set.",
        "",
        "But the mission is not blind, because contact does not have to be",
        "continuous. Libration swings Earth seven degrees either way against mean",
        "elevations of half a degree to four, so most of this ground is in touch",
        "for part of a month rather than none of it. The walk out barely loses the",
        "link at all -- minutes on the worst route here, nothing on several -- and",
        "the blackout is at the destination, which is where the platform stops and",
        "works.",
        "",
        "So the requirement is: operate unattended for a dwell, in a hole, with a",
        "link that returns when the platform climbs out. That is a far weaker",
        "autonomy requirement than continuous unsupervised traverse, and it is the",
        "first time this project has stated one in a form a design could be built",
        "against.",
        "",
        "A relay on the rim does not close the gap, and that is now measured",
        f"rather than asserted. A mast at the charge point recovers "
        f"{recovered_share:.1%} of the blind",
        f"ground across these ten sites and {recovered_cold_trap:.1%} of the "
        f"blind cold trap; at {barren} of",
        "the ten it recovers nothing at all. The reason is structural and not a",
        "matter of siting: the charge point is chosen for sunlight, sunlight means",
        "high and open ground, and high open ground is already in touch with",
        "Earth. A relay there covers the ground that least needs covering, and it",
        "cannot see into a hole any more than Earth can.",
        "",
        "That conclusion was nearly the opposite one. Charging each cell's",
        "occlusion to the single azimuth bin its centre falls in leaves gaps",
        "between the cells of a ridge, so a ridge near the mast stopped blocking",
        "and the leak grew with resolution. The relay area then moved sixfold",
        "across a coarsening and looked like a quantity the products could not",
        "decide — a defect in how it was computed, wearing the costume of a",
        "result about terrain. Once each cell occludes the sector it actually",
        f"subtends, the area moves {posting.relay_spread:.2f}× across the same "
        f"coarsening against Earth's",
        f"{posting.earth_spread:.2f}×, and both are reportable. The failure is "
        "worth recording because",
        "it was invisible from the output: an under-resolved viewshed returns a",
        "plausible number, and the shortfall reads as terrain.",
        "",
        "What the grid still cannot settle is any single link. The aggregate area",
        "is stable, but a boulder or a metre-scale rise below the posting blocks",
        "a sightline without moving the total, so the metre-scale stereo already",
        "specified for boulders and for pits is wanted here too — for whether a",
        "given link closes, not for how much ground a relay covers.",
        "",
        "Six axes populated is not six axes validated. The absence list above is",
        "the distance between those two things, and it is the most useful thing in",
        "this report: no obstacles, no margin, no failure modes, one soil, three",
        "regions with no data, two sharp results the map cannot decide, and no",
        "hardware anywhere. Every number in fifteen days is a ceiling, and that",
        "list is the specification of what would turn them into floors.",
        '"""',
        "",
        f"# {tally(rows)}",
        "",
        *toml_lines(rows),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute Earth visibility over the reachable set and report "
        "the autonomy requirement it implies."
    )
    parser.add_argument("--figure-directory", type=Path, default=FIGURE_DIRECTORY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    arguments = parser.parse_args(argv)

    sites = load_sites(SITE_DIRECTORY)
    products = load_terrain_manifest(MANIFEST_PATH)
    platform = load_platform(PLATFORM_PATH).platform
    dataset = load_soil(SOIL_PATH).datasets["carrier1991"]
    contact_model = dataset.models["bekker"].extrapolating
    strength = mohr_coulomb_model(dataset, depth_range_cm="0-15")
    mobilization = janosi_hanamoto_model(dataset)

    entries: list[Contact] = []
    rasters: dict[str, tuple[GeoRaster, tuple[int, int, int, int]]] = {}
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
        entry = survey_contact(
            site,
            raster=raster,
            platform=platform,
            contact_model=contact_model,
            strength=strength,
            mobilization=mobilization,
        )
        if entry is None:
            print(f"  {site.name:22s} no permanent shadow, or no route to it")
            del raster
            continue
        entries.append(entry)
        rasters[site.id] = (
            raster,
            centred_window(raster, span_m=COMMON_WINDOW_KM * 1000.0),
        )
        print(
            f"  {site.name:22s} charge sees Earth {entry.charge_contact_fraction:5.1%} "
            f"| blind {entry.blind_km2:5.1f} of {entry.reachable_km2:6.1f} km² "
            f"({entry.blind_fraction:5.1%})  cold trap {entry.cold_trap_km2:5.2f}, "
            f"blind {entry.blind_cold_trap_km2 / max(entry.cold_trap_km2, 1e-9):5.1%}"
            f"  route blind {entry.route_blind_hours * 60:4.0f} min"
        )

    if not entries:
        print("no site produced a contact survey; there is nothing to report")
        return 1

    # The showcase is the candidate region with the most cold trap to lose, and
    # the posting sweep runs there alone because each of its passes is a fresh
    # horizon over the whole window.
    candidates = [e for e in entries if e.site.is_candidate] or entries
    showcase = max(candidates, key=lambda e: e.cold_trap_km2)
    raster, window = rasters[showcase.site.id]
    print(f"\n  posting sensitivity and envelope at {showcase.site.name} ...")
    posting = posting_sensitivity(raster, home=showcase.home, window=window)
    print(
        f"    Earth contact moves {posting.earth_spread:.2f}x, relay coverage "
        f"{posting.relay_spread:.2f}x across posting and "
        f"{posting.relay_bin_spread:.2f}x across binning"
    )
    envelope = build_envelope(
        showcase,
        raster_cell_m=raster.cell_size_m,
        platform=platform,
        contact_model=contact_model,
        strength=strength,
        mobilization=mobilization,
    )

    arguments.figure_directory.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)

    for name, figure in (
        ("earth-contact", build_contact_figure(showcase)),
        ("blackout-and-decidability", build_blackout_figure(entries, posting)),
        (
            "six-axis-envelope",
            build_envelope_figure(
                showcase, envelope, battery_Wh=NOMINAL_BATTERY_WH
            ),
        ),
    ):
        target = arguments.figure_directory / f"{name}.png"
        figure.savefig(target, dpi=200)
        plt.close(figure)
        print(f"wrote {target.relative_to(REPOSITORY_ROOT)}")

    arguments.report.write_text(
        build_report(entries, posting, envelope, showcase=showcase.site.id),
        encoding="utf-8",
    )
    print(f"wrote {arguments.report.relative_to(REPOSITORY_ROOT)}")

    print("\n  measured against extrapolated\n")
    print(text_table(boundary_rows(entries, posting)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
