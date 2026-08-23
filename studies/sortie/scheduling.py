# SPDX-License-Identifier: Apache-2.0
#
# studies.sortie.scheduling — when the sortie departs, what that costs, and
# whether timing is a constraint or a convenience.
#
# Every result before this one answers where and how far. None answers when,
# and Day 9 established that when is not a detail: a seven-hour round trip is a
# hundredth of a lunation, and only the final approach and the dwell sit in
# permanent shadow. So survival cost is a property of a route and a departure
# time together, and the project has been quoting one end of a range without
# knowing which end.
#
# It turns out to have quoted both ends and neither middle. Day 8 charged
# survival power across the whole sortie, which is exactly the worst departure.
# Day 9's correction charged the dwell plus the permanently shadowed share of
# the walk, which is exactly the best one. Neither was wrong about its own case
# and neither said which case it was.
#
# Five results.
#
# The distribution, which is what the project actually needed. Across a year of
# departures the sortie is dark for 4.4 to 7.1 hours of its 7.1, so survival
# runs 52 to 84 Wh and total sortie energy 176 to 208. Day 9's 176 Wh is the
# best-timed sortie and Day 8's 208 the worst, and the median departure costs
# 189.
#
# Timing is a convenience, not a constraint, and the ratio is the argument.
# Worst to best is a factor of 1.18 on sortie energy, because locomotion
# dominates and the destination is permanently dark, so most of the cost cannot
# be timed away at all.
#
# With one exception, which is the sharp part. Below 176 Wh of battery no
# departure is affordable; above 208 Wh every departure is. Between them --
# a band 32 Wh wide -- the mission is genuinely window-limited, and inside it
# the feasible fraction of a lunation runs from nothing to everything. So
# timing binds only for a battery sized inside an 18% band, and a designer who
# sizes 20% above the best case removes scheduling from the problem. That is a
# requirement rather than a verdict, in the same shape as Day 8's emissivity.
#
# Timing matters less as the dwell grows, which is backwards from the
# intuition. The destination is permanently shadowed, so dwell is an immovable
# cost; lengthening it adds to the part no departure time can help. Worst to
# best falls from 1.25 at zero dwell to 1.06 at thirty-two hours. Wanting to
# work longer at the cold trap is what makes the clock stop mattering.
#
# And the crest's light is seasonal rather than ragged. Six lit runs a year of
# 520 to 675 hours, separated by dark spells of about a week, with five
# consecutive lunations of unbroken sunlight. A seven-hour sortie fits inside
# any of that with room to spare. The expectation going in was a ragged trace,
# because a polar point can be occulted by a distant massif and lit again
# within one rotation -- but this crest is the highest ground in the window, so
# there is no massif inside the data to do it. That makes the smoothness an
# upper bound in the same one-sided way every lit fraction here is: a wider
# search can add occultations and cannot remove them.
#
# What limits throughput is none of the three candidates the day set out to
# distinguish. At a battery above the band the platform walks essentially
# continuously -- 65 to 88 sorties per lunation, limited by how long a sortie
# takes -- and neither charge rate nor departure availability ever binds. That
# number is an upper bound with no operational margin in it at all: no
# maintenance, no contingency, no reason a real programme would run a machine
# back to back for a month. It is reported as a ceiling, not a plan.
#
# Four axes of six. Comms and cold-trap range remain empty, and four is not
# six: this is the closest the project has come to a sortie envelope and it is
# still missing the axis that decides whether anyone can talk to the platform.
#
# References
#   Mazarico E et al. (2011) Illumination conditions of the lunar polar regions
#     using LOLA topography. Icarus 211, 1066-1081.

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
    LUNAR_OBLIQUITY_DEG,
    LUNATION_HOURS,
    SOLAR_ANGULAR_RADIUS_DEG,
    SUBSOLAR_LATITUDE_PERIOD_HOURS,
    HorizonMap,
    IlluminationSeries,
    ShadowTarget,
    contiguous_interval_hours,
    horizon_elevation_deg,
    illumination_fraction,
    illumination_series,
    shadow_targets,
)
from eclipse.io.platform import load_platform
from eclipse.io.soil import janosi_hanamoto_model, load_soil, mohr_coulomb_model
from eclipse.io.terrain import GeoRaster, model_to_latitude_longitude, read_float_geotiff
from eclipse.platform import Platform
from eclipse.schedule import OperatingCycle, run_lunation, shadowed_hours
from eclipse.sortie import JOULES_PER_WATT_HOUR, sample_transect, walk_round_trip
from eclipse.stance import wave_gait, within_stride_slip_ratio

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
    Path(__file__).resolve().parent / "results" / "scheduling.toml"
)

REPORT_SCHEMA_VERSION: Final = 1
LUNAR_GRAVITY: Final = 1.62
FEET_IN_STANCE: Final = 3
NOMINAL_DERATING: Final = 4.0

# Carried unchanged from Day 9 so the two studies describe the same site.
ARRAY_AREA_M2: Final = 0.5
ARRAY_EFFICIENCY: Final = 0.30
SOLAR_CONSTANT_W_PER_M2: Final = 1361.0
MAP_STRIDE: Final = 50
HORIZON_AZIMUTHS: Final = 72
HORIZON_SAMPLES: Final = 140
HORIZON_STANDOFF_M: Final = 50.0
ROUTE_SAMPLES: Final = 400
ROUTE_ILLUMINATION_SPACING_M: Final = 40.0

# From Day 8, at an effective emissivity of 0.05.
INSULATED_SURVIVAL_W: Final = 11.8

# The clock. A quarter hour resolves a seven-hour sortie into thirty steps,
# which is finer than the illumination changes over that span.
TIME_STEP_H: Final = 0.25
DEPARTURE_STEP_H: Final = 1.0

# The mission parameter nobody has set, now explicit and swept. Day 9 assumed
# four hours at the cold trap without saying why, and the sweep is what makes
# that assumption visible rather than load-bearing.
DWELL_HOURS: Final = 4.0
DWELL_SWEEP_H: Final[NDArray[np.float64]] = np.array(
    [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]
)

BATTERY_SWEEP_WH: Final[NDArray[np.float64]] = np.linspace(150.0, 260.0, 111)
BATTERY_LINES_WH: Final = (210.0, 250.0, 400.0)
SEASONAL_BATTERY_WH: Final = 250.0
TRACE_LUNATIONS: Final = 2.0


def caption(text: str, width: int = 148) -> str:
    return "\n".join(
        textwrap.fill(" ".join(paragraph.split()), width=width)
        for paragraph in text.split("\n")
    )


def north_azimuth_deg(
    raster: GeoRaster, rows: NDArray[np.int_], columns: NDArray[np.int_]
) -> NDArray[np.float64]:
    x = raster.origin_x_m + (columns.astype(np.float64) + 0.5) * raster.cell_size_m
    y = raster.origin_y_m - (rows.astype(np.float64) + 0.5) * raster.cell_size_m
    return np.asarray(np.degrees(np.arctan2(x, -y)) % 360.0)


def latitudes(
    raster: GeoRaster, rows: NDArray[np.int_], columns: NDArray[np.int_]
) -> NDArray[np.float64]:
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


def horizon_at(
    raster: GeoRaster, rows: NDArray[np.int_], columns: NDArray[np.int_]
) -> HorizonMap:
    return horizon_elevation_deg(
        raster,
        rows=rows,
        columns=columns,
        azimuths=HORIZON_AZIMUTHS,
        samples_along_ray=HORIZON_SAMPLES,
        minimum_range_m=HORIZON_STANDOFF_M,
    )


def series_at(
    raster: GeoRaster,
    rows: NDArray[np.int_],
    columns: NDArray[np.int_],
    hours: NDArray[np.float64],
) -> IlluminationSeries:
    return illumination_series(
        horizon=horizon_at(raster, rows, columns),
        latitude_deg=latitudes(raster, rows, columns),
        north_azimuth_deg=north_azimuth_deg(raster, rows, columns),
        hours=hours,
    )


def average_charge_W(lit_fraction: float) -> float:
    return SOLAR_CONSTANT_W_PER_M2 * ARRAY_AREA_M2 * ARRAY_EFFICIENCY * lit_fraction


@dataclass(frozen=True, slots=True)
class Setting:
    raster: GeoRaster
    platform: Platform
    crest: tuple[int, int]
    target: ShadowTarget
    route_rows: NDArray[np.int_]
    route_columns: NDArray[np.int_]
    elapsed_hours: NDArray[np.float64]
    locomotion_Wh: float
    crest_lit_fraction: float

    @property
    def walking_hours(self) -> float:
        return float(self.elapsed_hours[-1])

    def sortie_hours(self, dwell_hours: float) -> float:
        return 2.0 * self.walking_hours + dwell_hours


def load_setting() -> Setting:
    raster = read_float_geotiff(ELEVATION_PATH)
    platform = load_platform(PLATFORM_PATH).platform
    dataset = load_soil(SOIL_PATH).datasets["carrier1991"]
    contact = dataset.models["bekker"].extrapolating
    strength = mohr_coulomb_model(dataset, depth_range_cm="0-15")
    mobilization = janosi_hanamoto_model(dataset)

    highest = np.unravel_index(int(np.argmax(raster.values)), raster.values.shape)
    crest = (int(highest[0]), int(highest[1]))

    height, width = raster.shape
    grid_rows, grid_columns = np.meshgrid(
        np.arange(0, height, MAP_STRIDE),
        np.arange(0, width, MAP_STRIDE),
        indexing="ij",
    )
    print(f"  horizon over {grid_rows.size} map points ...")
    grid = illumination_fraction(
        horizon=horizon_at(raster, grid_rows.ravel(), grid_columns.ravel()),
        latitude_deg=latitudes(raster, grid_rows.ravel(), grid_columns.ravel()),
        north_azimuth_deg=north_azimuth_deg(
            raster, grid_rows.ravel(), grid_columns.ravel()
        ),
    )
    target = shadow_targets(
        raster,
        start=crest,
        rows=grid_rows,
        columns=grid_columns,
        any_sunlight_fraction=grid.any_sunlight_fraction.reshape(grid_rows.shape),
    )["nearest"]

    transect = sample_transect(
        raster,
        start_row_column=crest,
        end_row_column=(target.row, target.column),
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

    rows = np.rint(np.linspace(crest[0], target.row, ROUTE_SAMPLES)).astype(int)
    columns = np.rint(np.linspace(crest[1], target.column, ROUTE_SAMPLES)).astype(int)
    spacing = float(transect.distance_m[-1]) / (ROUTE_SAMPLES - 1)
    stride = max(1, int(round(ROUTE_ILLUMINATION_SPACING_M / spacing)))
    index = np.unique(
        np.concatenate([np.arange(0, ROUTE_SAMPLES, stride), [ROUTE_SAMPLES - 1]])
    )

    probe = illumination_fraction(
        horizon=horizon_at(raster, np.array([crest[0]]), np.array([crest[1]])),
        latitude_deg=latitudes(raster, np.array([crest[0]]), np.array([crest[1]])),
        north_azimuth_deg=north_azimuth_deg(
            raster, np.array([crest[0]]), np.array([crest[1]])
        ),
    )
    return Setting(
        raster=raster,
        platform=platform,
        crest=crest,
        target=target,
        route_rows=rows[index],
        route_columns=columns[index],
        elapsed_hours=transect.distance_m[index]
        / platform.nominal_speed_m_per_s
        / 3600.0,
        locomotion_Wh=trip.total_J / JOULES_PER_WATT_HOUR * NOMINAL_DERATING,
        crest_lit_fraction=float(probe.any_sunlight_fraction[0]),
    )


@dataclass(frozen=True, slots=True)
class DepartureSweep:
    dwell_hours: float
    departure_hours: NDArray[np.float64]
    shadowed_hours: NDArray[np.float64]
    locomotion_Wh: float

    @property
    def survival_Wh(self) -> NDArray[np.float64]:
        return np.asarray(INSULATED_SURVIVAL_W * self.shadowed_hours)

    @property
    def total_Wh(self) -> NDArray[np.float64]:
        return np.asarray(self.locomotion_Wh + self.survival_Wh)

    def feasible_fraction(self, battery_Wh: float) -> float:
        return float((self.total_Wh <= battery_Wh).mean())


def sweep_departures(
    setting: Setting,
    *,
    dwell_hours: float,
    span_hours: float,
    route: IlluminationSeries | None = None,
) -> tuple[DepartureSweep, IlluminationSeries]:
    """Sortie energy for every departure across a span, at a given dwell.

    The illumination series is reusable across dwells only while the sortie
    still fits inside it, so it is passed back and passed in rather than
    recomputed; a longer dwell needs a longer series and gets one.
    """
    sortie = setting.sortie_hours(dwell_hours)
    needed = span_hours + sortie + TIME_STEP_H
    if route is None or float(route.hours[-1]) < needed:
        hours = np.arange(0.0, needed, TIME_STEP_H)
        route = series_at(
            setting.raster, setting.route_rows, setting.route_columns, hours
        )
    departures = np.arange(0.0, span_hours, DEPARTURE_STEP_H)
    return (
        DepartureSweep(
            dwell_hours=dwell_hours,
            departure_hours=departures,
            shadowed_hours=shadowed_hours(
                dark=~route.any_sunlight,
                hours=route.hours,
                elapsed_hours=setting.elapsed_hours,
                departure_hours=departures,
                dwell_hours=dwell_hours,
            ),
            locomotion_Wh=setting.locomotion_Wh,
        ),
        route,
    )


@dataclass(frozen=True, slots=True)
class DwellPoint:
    dwell_hours: float
    sortie_hours: float
    best_Wh: float
    worst_Wh: float
    survival_share: float
    sorties: dict[float, int]
    limited_by: dict[float, str]

    @property
    def timing_ratio(self) -> float:
        return self.worst_Wh / self.best_Wh


def sweep_dwell(setting: Setting, crest: IlluminationSeries) -> tuple[DwellPoint, ...]:
    charge = np.where(
        crest.any_sunlight[0],
        SOLAR_CONSTANT_W_PER_M2 * ARRAY_AREA_M2 * ARRAY_EFFICIENCY,
        0.0,
    )
    points = []
    for dwell in DWELL_SWEEP_H:
        sweep, _ = sweep_departures(
            setting, dwell_hours=float(dwell), span_hours=LUNATION_HOURS
        )
        sortie = setting.sortie_hours(float(dwell))
        energy = np.interp(
            crest.hours, sweep.departure_hours, sweep.total_Wh,
            left=float(sweep.total_Wh[0]), right=float(sweep.total_Wh[-1]),
        )
        sorties: dict[float, int] = {}
        limited: dict[float, str] = {}
        for battery in BATTERY_LINES_WH:
            cycle = run_lunation(
                hours=crest.hours,
                energy_Wh=energy,
                sortie_hours=sortie,
                charge_W=charge,
                battery_Wh=battery,
                lunation_hours=LUNATION_HOURS,
            )
            sorties[battery] = cycle.sorties
            limited[battery] = cycle.limited_by
        points.append(
            DwellPoint(
                dwell_hours=float(dwell),
                sortie_hours=sortie,
                best_Wh=float(sweep.total_Wh.min()),
                worst_Wh=float(sweep.total_Wh.max()),
                survival_share=float(
                    np.median(sweep.survival_Wh / sweep.total_Wh)
                ),
                sorties=sorties,
                limited_by=limited,
            )
        )
    return tuple(points)


def sweep_seasons(
    setting: Setting, probe: IlluminationSeries, year: DepartureSweep
) -> tuple[OperatingCycle, ...]:
    """The same route and the same battery, run in each lunation of a year.

    The route's own cost barely moves between seasons. What moves is the
    charge point: for half the year it is in unbroken sunlight and for the
    other half a week-long dark spell stops the array, and the greedy pass
    turns that into a throughput difference without being told about it.
    """
    charge = np.where(
        probe.any_sunlight[0],
        SOLAR_CONSTANT_W_PER_M2 * ARRAY_AREA_M2 * ARRAY_EFFICIENCY,
        0.0,
    )
    sortie = setting.sortie_hours(DWELL_HOURS)
    energy = np.interp(
        probe.hours,
        year.departure_hours,
        year.total_Wh,
        left=float(year.total_Wh[0]),
        right=float(year.total_Wh[-1]),
    )
    span = LUNATION_HOURS + sortie
    cycles = []
    start = 0.0
    while start + span <= float(probe.hours[-1]):
        window = (probe.hours >= start) & (probe.hours < start + span)
        cycles.append(
            run_lunation(
                hours=probe.hours[window] - start,
                energy_Wh=energy[window],
                sortie_hours=sortie,
                charge_W=charge[window],
                battery_Wh=SEASONAL_BATTERY_WH,
                lunation_hours=LUNATION_HOURS,
            )
        )
        start += LUNATION_HOURS
    return tuple(cycles)


def build_light_figure(
    setting: Setting, probe: IlluminationSeries, seasons: tuple[OperatingCycle, ...]
) -> Figure:
    trace = probe.hours <= TRACE_LUNATIONS * LUNATION_HOURS
    days = probe.hours[trace] / 24.0
    crest_dark = ~probe.any_sunlight[0][trace]
    lit_runs = contiguous_interval_hours(mask=probe.any_sunlight[0], hours=probe.hours)
    dark_runs = contiguous_interval_hours(mask=~probe.any_sunlight[0], hours=probe.hours)
    sortie = setting.sortie_hours(DWELL_HOURS)

    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (11.6, 6.1),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.596,
                    "figure.subplot.bottom": 0.150,
                    "figure.subplot.left": 0.060,
                    "figure.subplot.right": 0.986,
                    "figure.subplot.wspace": 0.210,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 2, squeeze=False, width_ratios=[1.85, 1.0])
        left, right = axes[0][0], axes[0][1]

        left.fill_between(
            days,
            0.0,
            1.0,
            where=crest_dark,
            transform=left.get_xaxis_transform(),
            color=INK_PRIMARY,
            alpha=0.075,
            linewidth=0.0,
            label="the crest is dark",
        )
        left.axhspan(
            -SOLAR_ANGULAR_RADIUS_DEG,
            SOLAR_ANGULAR_RADIUS_DEG,
            color=ACCENT_SECONDARY,
            alpha=0.30,
            linewidth=0.0,
            label="the Sun's disc, half a degree wide",
        )
        left.plot(
            days,
            probe.clearance_deg[0][trace],
            color=ACCENT_PRIMARY,
            linewidth=1.3,
            label="charge point on the crest",
        )
        left.plot(
            days,
            probe.clearance_deg[1][trace],
            color=INK_MUTED,
            linewidth=1.3,
            linestyle=(0, (4, 2)),
            label="destination in permanent shadow",
        )
        left.set_xlabel("days from an arbitrary epoch")
        left.set_ylabel("Sun's centre above the horizon (°)")
        left.set_title(
            f"clearance over {TRACE_LUNATIONS:.0f} lunations",
            color=INK_SECONDARY,
            loc="left",
        )
        left.set_xlim(float(days[0]), float(days[-1]))
        left.set_ylim(float(probe.clearance_deg[:, trace].min()) - 5.0, None)
        left.legend(loc="lower left", ncols=2)

        for runs, color, label, height in (
            (lit_runs, ACCENT_PRIMARY, f"{lit_runs.size} lit runs", 1.0),
            (dark_runs, ACCENT_SECONDARY, f"{dark_runs.size} dark runs", 0.0),
        ):
            right.plot(
                runs / 24.0,
                np.full(runs.size, height),
                marker="o",
                markersize=7.0,
                markerfacecolor=color,
                markeredgecolor="none",
                linestyle="none",
                alpha=0.75,
                label=label,
            )
            if runs.size:
                right.annotate(
                    f"{runs.min() / 24.0:.1f} to {runs.max() / 24.0:.0f} days",
                    xy=(float(np.median(runs) / 24.0), height),
                    xytext=(0, 13),
                    textcoords="offset points",
                    ha="center",
                    color=color,
                    fontsize=8.0,
                )
        right.axvline(sortie / 24.0, color=INK_PRIMARY, linewidth=1.2)
        right.annotate(
            f"one sortie,\n{sortie:.1f} h",
            xy=(sortie / 24.0, 0.5),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=INK_PRIMARY,
            fontsize=8.0,
        )
        right.set_xscale("log")
        right.set_xlim(sortie / 24.0 / 3.0, 60.0)
        right.set_ylim(-0.6, 1.6)
        right.set_yticks([0.0, 1.0], ["dark", "lit"])
        right.set_xlabel("run length (days, log)")
        right.set_title(
            "every run of light and dark, against one sortie",
            color=INK_SECONDARY,
            loc="left",
        )

        for panel in (left, right):
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)

        figure.suptitle(
            "The crest's light is seasonal rather than ragged, and the shortest "
            f"run of it is {lit_runs.min() / sortie:.0f} times a sortie",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.060,
            ha="left",
            y=0.958,
        )
        figure.text(
            0.060,
            0.912,
            caption(
                f"Over a year the crest has {lit_runs.size} runs of sunlight "
                f"lasting {lit_runs.min() / 24.0:.0f} to "
                f"{lit_runs.max() / 24.0:.0f} days, separated by dark spells of "
                f"{dark_runs.min() / 24.0:.1f} to {dark_runs.max() / 24.0:.0f}. "
                "No sortie is ever cut short by nightfall and none has to wait for "
                "dawn. What a dark spell does stop is charging, and that is where "
                "the difference between a winter lunation and a summer one comes "
                "from: the same platform on the same route manages "
                f"{min(cycle.sorties for cycle in seasons)} sorties in one and "
                f"{max(cycle.sorties for cycle in seasons)} in the other, on a "
                f"{SEASONAL_BATTERY_WH:.0f} Wh battery.\n"
                "A ragged trace was the expectation, because a polar point can be "
                "occulted by a distant massif and lit again within one rotation. "
                "This crest is the highest ground in the window, so nothing inside "
                "the data occults it and its darkness is the Sun setting rather "
                "than terrain intervening. The smoothness inherits the same "
                "one-sided caveat as every lit fraction here: a wider search can "
                "only add occultations, never remove them.\n"
                "The destination's clearance is the check, and it is not flat — it "
                f"rises and falls by "
                f"{probe.clearance_deg[1].max() - probe.clearance_deg[1].min():.0f} "
                "degrees as the Sun circles. What makes it a permanent shadow is "
                "that across a year it never comes within "
                f"{-probe.clearance_deg[1].max():.1f} degrees of the disc, at any "
                "hour of any lunation.",
                width=170,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def build_departure_figure(
    setting: Setting, sweep: DepartureSweep, year: DepartureSweep
) -> Figure:
    days = sweep.departure_hours / 24.0
    total = sweep.total_Wh
    best = int(np.argmin(total))
    worst = int(np.argmax(total))
    floor = setting.locomotion_Wh + INSULATED_SURVIVAL_W * sweep.dwell_hours
    battery = float(np.ceil(year.total_Wh.max() / 10.0) * 10.0)
    feasible = np.asarray(
        [year.feasible_fraction(float(b)) for b in BATTERY_SWEEP_WH]
    )

    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (11.8, 5.6),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.628,
                    "figure.subplot.bottom": 0.168,
                    "figure.subplot.left": 0.060,
                    "figure.subplot.right": 0.986,
                    "figure.subplot.wspace": 0.230,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 2, squeeze=False, width_ratios=[1.7, 1.0])
        left, right = axes[0][0], axes[0][1]

        shown = float(np.percentile(year.total_Wh, 50.0))
        left.fill_between(
            days,
            0.0,
            total,
            where=total <= shown,
            color=ACCENT_PRIMARY,
            alpha=0.14,
            linewidth=0.0,
            label=f"affordable on {shown:.0f} Wh",
        )
        left.plot(days, total, color=ACCENT_PRIMARY, linewidth=1.4)
        left.axhline(shown, color=ACCENT_PRIMARY, linewidth=1.0, linestyle=(0, (3, 2)))
        left.axhline(floor, color=INK_MUTED, linewidth=1.0, linestyle=(0, (1.5, 1.5)))
        left.annotate(
            "dwell and locomotion alone: no departure reaches it",
            xy=(float(days[0]), floor),
            xytext=(6, -13),
            textcoords="offset points",
            ha="left",
            color=INK_MUTED,
            fontsize=7.8,
        )
        for index, label, offset in (
            (best, f"best {total[best]:.0f} Wh", (8.0, 8.0)),
            (worst, f"worst {total[worst]:.0f} Wh", (0.0, -22.0)),
        ):
            left.plot(
                [days[index]], [total[index]], marker="o", markersize=6.0,
                markerfacecolor="none", color=INK_PRIMARY,
            )
            left.annotate(
                label,
                xy=(float(days[index]), float(total[index])),
                xytext=offset,
                textcoords="offset points",
                color=INK_PRIMARY,
                fontsize=8.0,
            )
        left.set_xlabel("departure, days into a lunation")
        left.set_ylabel("sortie energy (Wh)")
        left.set_title(
            f"energy against departure, at a {sweep.dwell_hours:.0f} h dwell",
            color=INK_SECONDARY,
            loc="left",
        )
        left.set_xlim(float(days[0]), float(days[-1]))
        left.set_ylim(float(floor) * 0.94, float(total.max()) * 1.04)
        left.legend(loc="upper left")

        right.plot(BATTERY_SWEEP_WH, feasible * 100.0, color=ACCENT_SECONDARY, linewidth=1.8)
        low = float(year.total_Wh.min())
        high = float(year.total_Wh.max())
        right.axvspan(low, high, color=ACCENT_SECONDARY, alpha=0.12, linewidth=0.0)
        right.annotate(
            f"{high - low:.0f} Wh band",
            xy=(0.5 * (low + high), 50.0),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            color=INK_SECONDARY,
            fontsize=8.0,
        )
        right.set_xlabel("battery capacity (Wh)")
        right.set_ylabel("departures that are affordable (%)")
        right.set_title(
            "and the band where the choice matters at all",
            color=INK_SECONDARY,
            loc="left",
        )
        right.set_xlim(float(BATTERY_SWEEP_WH[0]), float(BATTERY_SWEEP_WH[-1]))
        right.set_ylim(-3.0, 103.0)

        for panel in (left, right):
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)

        figure.suptitle(
            "Timing is worth "
            f"{year.total_Wh.max() / year.total_Wh.min():.2f}× on sortie energy, "
            f"and it decides anything only inside a {high - low:.0f} Wh band of battery",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.060,
            ha="left",
            y=0.958,
        )
        figure.text(
            0.060,
            0.898,
            caption(
                "Survival power applies where and when the ground is dark, read at "
                "the time the platform is standing on it rather than at departure. "
                f"Across a year the sortie is cold for "
                f"{year.shadowed_hours.min():.1f} to "
                f"{year.shadowed_hours.max():.1f} of its "
                f"{setting.sortie_hours(sweep.dwell_hours):.1f} hours, so total "
                f"energy runs {low:.0f} to {high:.0f} Wh. Day 9 quoted "
                f"{low:.0f} — the best-timed sortie — and Day 8 quoted "
                f"{high:.0f}, the worst. Neither said which it was.\n"
                f"Below {low:.0f} Wh of battery nothing flies; above {high:.0f} Wh "
                "everything does, and the departure time stops being a decision. "
                "So scheduling is a design consequence rather than an operational "
                "one: size the battery a fifth above the best case and the clock "
                "disappears from the problem. The dashed floor is what the sortie "
                "would cost if only the permanently shadowed ground were dark, "
                "which no real departure achieves, because part of the traverse is "
                "in shadow at any hour a sortie might leave.",
                width=172,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def build_dwell_figure(points: tuple[DwellPoint, ...]) -> Figure:
    dwell = np.asarray([point.dwell_hours for point in points])
    ratio = np.asarray([point.timing_ratio for point in points])

    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (11.8, 5.6),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.640,
                    "figure.subplot.bottom": 0.168,
                    "figure.subplot.left": 0.060,
                    "figure.subplot.right": 0.986,
                    "figure.subplot.wspace": 0.230,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 2, squeeze=False)
        left, right = axes[0][0], axes[0][1]

        styles = (
            (BATTERY_LINES_WH[0], ACCENT_SECONDARY, "solid"),
            (BATTERY_LINES_WH[1], ACCENT_PRIMARY, (0, (5, 2))),
            (BATTERY_LINES_WH[2], INK_MUTED, (0, (1.6, 1.6))),
        )
        for battery, color, dash in styles:
            counts = np.asarray([point.sorties[battery] for point in points])
            left.plot(
                dwell, counts, color=color, linewidth=1.7, linestyle=dash,
                marker="o", markersize=3.4, label=f"{battery:.0f} Wh battery",
            )
        annotated: set[str] = set()
        for point in points:
            limit = point.limited_by[BATTERY_LINES_WH[1]]
            if limit in annotated:
                continue
            annotated.add(limit)
            count = point.sorties[BATTERY_LINES_WH[1]]
            left.annotate(
                limit,
                xy=(point.dwell_hours, count),
                xytext=(10, 12 if count == 0 else -14),
                textcoords="offset points",
                color=INK_SECONDARY,
                fontsize=7.8,
            )
        left.set_xlabel("dwell at the cold trap (h)")
        left.set_ylabel("sorties per lunation")
        left.set_title(
            "throughput, and what stops more of it", color=INK_SECONDARY, loc="left"
        )
        left.legend(loc="upper right")

        right.plot(dwell, ratio, color=ACCENT_PRIMARY, linewidth=1.8)
        right.fill_between(
            dwell, 1.0, ratio, color=ACCENT_PRIMARY, alpha=0.14, linewidth=0.0
        )
        right.axhline(1.0, color=INK_MUTED, linewidth=1.0)
        right.annotate(
            "timing buys nothing",
            xy=(float(dwell[-1]), 1.0),
            xytext=(-6, 6),
            textcoords="offset points",
            ha="right",
            color=INK_MUTED,
            fontsize=7.8,
        )
        right.set_xlabel("dwell at the cold trap (h)")
        right.set_ylabel("worst departure ÷ best departure")
        right.set_title(
            "and how much the departure time can still change",
            color=INK_SECONDARY,
            loc="left",
        )
        right.set_ylim(1.0, float(ratio.max()) * 1.03)

        for panel in (left, right):
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)

        figure.suptitle(
            "Working longer at the cold trap is what makes the clock stop "
            "mattering",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.060,
            ha="left",
            y=0.958,
        )
        figure.text(
            0.060,
            0.898,
            caption(
                "The destination is permanently shadowed, so dwell is a cost no "
                "departure time can move. Lengthening it adds only to the "
                "immovable part, and what timing can still buy falls from "
                f"{ratio[0]:.2f}× at no dwell to {ratio[-1]:.2f}× at "
                f"{dwell[-1]:.0f} hours. The intuition runs the other way — a "
                "longer sortie sounds like more exposure to bad timing — and it is "
                "wrong for the same reason the sortie is affordable at all: the "
                "expensive part of being cold is arriving, not travelling.\n"
                "Throughput is a greedy pass over one lunation and a ceiling "
                "rather than a plan: go when the battery allows it, otherwise "
                "charge, otherwise wait. It carries no maintenance, no "
                "contingency and no reason a programme would run a machine back to "
                "back for a month. What it does say is which constraint is live, "
                "and that changes across the sweep — sortie duration while the "
                "sorties are cheap, then departure windows, then the battery "
                "outright.",
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
    setting: Setting, year: DepartureSweep, probe: IlluminationSeries
) -> tuple[BoundaryRow, ...]:
    window_km = setting.raster.shape[0] * setting.raster.cell_size_m / 1000.0
    return (
        BoundaryRow(
            quantity="horizon search range",
            published_range="not applicable",
            used=f"within the {window_km:.0f} km window only",
            status=OUTSIDE,
            basis=(
                f"{probe.horizon.truncated_fraction:.0%} of ray samples leave the "
                "grid and count as clear sky. Carried from Day 9 and now with a "
                "second edge: a wider search can only add occultations, so the "
                "crest's uninterrupted lit runs are an upper bound too"
            ),
        ),
        BoundaryRow(
            quantity="solar ephemeris",
            published_range="not applicable",
            used="analytic; hour angle per lunation, declination per year",
            status=UNMEASURED,
            basis=(
                "no SPICE kernel. The 18.6-year nodal precession is absent, which "
                "moves the obliquity amplitude rather than the period, and the "
                "amplitude is already the least certain input here"
            ),
        ),
        BoundaryRow(
            quantity="lunation period",
            published_range="29.530589 d, synodic month",
            used="29.530589 d",
            status=INSIDE,
            basis="the clock a schedule runs on, and the only one that is exact",
        ),
        BoundaryRow(
            quantity="dwell at the destination",
            published_range="none",
            used=f"{DWELL_HOURS:.0f} h nominal, swept "
            f"{DWELL_SWEEP_H[0]:.0f} to {DWELL_SWEEP_H[-1]:.0f} h",
            status=UNMEASURED,
            basis=(
                "a mission parameter nobody has set. Day 9 assumed four hours "
                "without saying why; it sets the ratio of immovable to timeable "
                "cost, so it decides how much timing is worth"
            ),
        ),
        BoundaryRow(
            quantity="operational margin",
            published_range="none",
            used="none; sorties run back to back",
            status=UNMEASURED,
            basis=(
                "no maintenance, no contingency, no dust or thermal duty cycle. "
                "Sorties per lunation is a ceiling and should never be read as a "
                "plan"
            ),
        ),
        BoundaryRow(
            quantity="thermal transient",
            published_range="none",
            used="steady state on entering shadow",
            status=UNMEASURED,
            basis=(
                "survival power applies the instant the ground goes dark. Day 8's "
                "lumped model has a cooling time and could carry the lag, which "
                "would reduce the cost of short crossings; a full transient is a "
                "later rung"
            ),
        ),
        BoundaryRow(
            quantity="array area and efficiency",
            published_range="none",
            used=f"{ARRAY_AREA_M2:.1f} m2 at {ARRAY_EFFICIENCY:.0%}",
            status=UNMEASURED,
            basis=(
                "assumed, and it does not bind at any dwell swept: the charge rate "
                "never becomes the live constraint"
            ),
        ),
        BoundaryRow(
            quantity="departure timing",
            published_range="none",
            used=f"swept hourly across a year, {year.departure_hours.size} departures",
            status=INSIDE,
            basis=(
                f"worst departure costs {year.total_Wh.max() / year.total_Wh.min():.2f} "
                "times the best. The distribution is the result; Day 8 reported "
                "its maximum and Day 9 its minimum"
            ),
        ),
        BoundaryRow(
            quantity="survival power",
            published_range="none",
            used=f"{INSULATED_SURVIVAL_W:.1f} W",
            status=UNMEASURED,
            basis=(
                "from Day 8 at an effective emissivity of 0.05, which is a "
                "requirement rather than a measurement and carries forward as one"
            ),
        ),
    )


def _format_float(value: float) -> str:
    return repr(float(value))


def build_report(
    setting: Setting,
    probe: IlluminationSeries,
    lunation: DepartureSweep,
    year: DepartureSweep,
    dwell: tuple[DwellPoint, ...],
    seasons: tuple[OperatingCycle, ...],
) -> str:
    rows = boundary_rows(setting, year, probe)
    lit_runs = contiguous_interval_hours(
        mask=probe.any_sunlight[0], hours=probe.hours
    )
    dark_runs = contiguous_interval_hours(
        mask=~probe.any_sunlight[0], hours=probe.hours
    )
    low = float(year.total_Wh.min())
    high = float(year.total_Wh.max())

    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# When the sortie departs, and what that costs.",
        "#",
        "# Generated by studies/sortie/scheduling.py. Do not edit.",
        "#",
        "# FOUR AXES OF SIX. Comms and cold-trap range are still empty.",
        "",
        f"schema_version = {REPORT_SCHEMA_VERSION}",
        "",
        "[environment]",
        f'python = "{host_platform.python_version()}"',
        f'numpy = "{np.__version__}"',
        "",
        "[method]",
        f"time_step_h = {_format_float(TIME_STEP_H)}",
        f"departure_step_h = {_format_float(DEPARTURE_STEP_H)}",
        f"lunation_hours = {_format_float(LUNATION_HOURS)}",
        f"subsolar_latitude_period_hours = {_format_float(SUBSOLAR_LATITUDE_PERIOD_HOURS)}",
        f"lunar_obliquity_deg = {_format_float(LUNAR_OBLIQUITY_DEG)}",
        f"solar_angular_radius_deg = {_format_float(SOLAR_ANGULAR_RADIUS_DEG)}",
        'ephemeris = "analytic; hour angle per lunation, declination per year"',
        "horizon_truncated_fraction = "
        f"{_format_float(probe.horizon.truncated_fraction)}",
        "",
        "[route]",
        f"distance_km = {_format_float(setting.target.distance_km)}",
        f"drop_m = {_format_float(setting.target.drop_m)}",
        f"speed_m_per_s = {_format_float(setting.platform.nominal_speed_m_per_s)}",
        f"walking_hours_one_way = {_format_float(setting.walking_hours)}",
        f"locomotion_Wh = {_format_float(setting.locomotion_Wh)}",
        "",
        "# The light at the charge point, as a history rather than a fraction.",
        "[charge_point]",
        f"lit_fraction_of_year = {_format_float(setting.crest_lit_fraction)}",
        "lit_runs_per_year = " + str(lit_runs.size),
        f"lit_run_shortest_h = {_format_float(float(lit_runs.min()))}",
        f"lit_run_longest_h = {_format_float(float(lit_runs.max()))}",
        "dark_runs_per_year = " + str(dark_runs.size),
        f"dark_run_longest_h = {_format_float(float(dark_runs.max()))}",
        'structure = "seasonal, not ragged; this crest is the highest ground in '
        'the window so nothing inside the data occults it"',
        "",
        "[destination]",
        "dark_at_every_sample = "
        + str(bool((~probe.any_sunlight[1]).all())).lower(),
        "peak_clearance_deg = "
        f"{_format_float(float(probe.clearance_deg[1].max()))}",
        "",
        "# The result. Day 8 charged survival across the whole sortie, which is",
        "# the worst departure. Day 9 charged the dwell plus the permanently",
        "# shadowed share, which is the best. Both are in this range and neither",
        "# said which end it was.",
        "[departure]",
        f"dwell_hours = {_format_float(DWELL_HOURS)}",
        f"sortie_hours = {_format_float(setting.sortie_hours(DWELL_HOURS))}",
        "departures_swept = " + str(year.departure_hours.size),
        f"shadowed_hours_best = {_format_float(float(year.shadowed_hours.min()))}",
        f"shadowed_hours_median = {_format_float(float(np.median(year.shadowed_hours)))}",
        f"shadowed_hours_worst = {_format_float(float(year.shadowed_hours.max()))}",
        f"total_Wh_best = {_format_float(low)}",
        f"total_Wh_median = {_format_float(float(np.median(year.total_Wh)))}",
        f"total_Wh_worst = {_format_float(high)}",
        f"timing_ratio = {_format_float(high / low)}",
        "",
        "# Where timing decides anything, which is a narrower place than the",
        "# ratio alone suggests.",
        "[window]",
        f"battery_below_which_nothing_flies_Wh = {_format_float(low)}",
        f"battery_above_which_everything_flies_Wh = {_format_float(high)}",
        f"band_width_Wh = {_format_float(high - low)}",
        f"band_width_fraction = {_format_float((high - low) / low)}",
        "",
    ]
    for battery in (low, 0.5 * (low + high), high):
        lines += [
            "[[feasible]]",
            f"battery_Wh = {_format_float(battery)}",
            "fraction_of_departures = "
            f"{_format_float(year.feasible_fraction(float(battery)))}",
            "",
        ]

    lines += [
        "# The same route in every lunation of a year. What varies is not the",
        "# sortie but the charge point: half the year it is in unbroken sunlight",
        "# and half it loses a week to a dark spell.",
        "[season]",
        f"battery_Wh = {_format_float(SEASONAL_BATTERY_WH)}",
        "lunations = " + str(len(seasons)),
        "sorties_per_lunation = ["
        + ", ".join(str(cycle.sorties) for cycle in seasons)
        + "]",
        "fewest_sorties = " + str(min(cycle.sorties for cycle in seasons)),
        "most_sorties = " + str(max(cycle.sorties for cycle in seasons)),
        "limited_by = ["
        + ", ".join(f'"{cycle.limited_by}"' for cycle in seasons)
        + "]",
        "",
        "# Dwell is the mission parameter that decides how much timing is worth,",
        "# and it runs the opposite way to the intuition.",
        "",
    ]
    for point in dwell:
        lines += [
            "[[dwell]]",
            f"dwell_hours = {_format_float(point.dwell_hours)}",
            f"sortie_hours = {_format_float(point.sortie_hours)}",
            f"best_Wh = {_format_float(point.best_Wh)}",
            f"worst_Wh = {_format_float(point.worst_Wh)}",
            f"timing_ratio = {_format_float(point.timing_ratio)}",
            f"survival_share = {_format_float(point.survival_share)}",
            "sorties_per_lunation = ["
            + ", ".join(str(point.sorties[b]) for b in BATTERY_LINES_WH)
            + "]",
            "limited_by = ["
            + ", ".join(f'"{point.limited_by[b]}"' for b in BATTERY_LINES_WH)
            + "]",
            "",
        ]

    lines += [
        "batteries_Wh = ["
        + ", ".join(_format_float(b) for b in BATTERY_LINES_WH)
        + "]",
        "",
        "# Is timing a constraint or a convenience.",
        "[verdict]",
        'statement = """',
        "A convenience, for a battery sized anywhere sensible.",
        "",
        "The worst departure costs 1.18 times the best, which sounds like a",
        "schedule until it is put beside what it is a fraction of. Locomotion is",
        "seventy percent of the sortie and does not care what time it is, and the",
        "destination is permanently shadowed, so the dwell is dark whenever it",
        "happens. What is left to time is the traverse, and the traverse is short.",
        "",
        "The exception is real and narrow. Below the best-case energy no departure",
        "is affordable and above the worst-case every one is, so timing decides",
        "something only for a battery inside that band -- 32 Wh, or eighteen",
        "percent. Inside it the feasible fraction of a lunation moves from nothing",
        "to everything, and a mission sized there would be genuinely",
        "window-limited. Outside it the clock is not a constraint at all. That is",
        "a sizing requirement rather than an operational finding, and it is the",
        "useful form of the answer.",
        "",
        "Two things push the same way and are worth stating because both invert",
        "an expectation. A longer dwell makes timing matter less, not more,",
        "because it adds to the cost no departure can move. And the crest's light",
        "does not come in windows that need catching: six runs a year of three to",
        "four weeks each, against a sortie of seven hours.",
        "",
        "What limits throughput is none of the three candidates. Above the band",
        "the platform walks continuously and sortie duration is the only live",
        "constraint. That is a ceiling with no operational margin in it, and the",
        "number is reported for what it bounds rather than what it plans.",
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
            "Compute illumination as a time series, price the sortie against "
            "departure time, and settle whether timing is a constraint."
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
    print(
        f"  crest {setting.crest} -> nearest PSR "
        f"({setting.target.row}, {setting.target.column}), "
        f"{setting.target.distance_km:.2f} km, "
        f"{setting.sortie_hours(DWELL_HOURS):.2f} h round trip"
    )

    year_hours = np.arange(
        0.0,
        SUBSOLAR_LATITUDE_PERIOD_HOURS + setting.sortie_hours(DWELL_HOURS) + 1.0,
        TIME_STEP_H,
    )
    print(f"  clearance at two points over {year_hours.size} samples ...")
    probe = series_at(
        setting.raster,
        np.array([setting.crest[0], setting.target.row]),
        np.array([setting.crest[1], setting.target.column]),
        year_hours,
    )

    print("  departures across a year ...")
    year, route = sweep_departures(
        setting,
        dwell_hours=DWELL_HOURS,
        span_hours=SUBSOLAR_LATITUDE_PERIOD_HOURS,
    )
    lunation, _ = sweep_departures(
        setting, dwell_hours=DWELL_HOURS, span_hours=LUNATION_HOURS, route=route
    )
    print(
        f"    shadowed {year.shadowed_hours.min():.2f}-"
        f"{year.shadowed_hours.max():.2f} h, "
        f"total {year.total_Wh.min():.1f}-{year.total_Wh.max():.1f} Wh, "
        f"ratio {year.total_Wh.max() / year.total_Wh.min():.3f}"
    )

    seasons = sweep_seasons(setting, probe, year)
    print(
        f"  {len(seasons)} lunations on {SEASONAL_BATTERY_WH:.0f} Wh: "
        + " ".join(str(cycle.sorties) for cycle in seasons)
        + " sorties"
    )

    print(f"  dwell sweep over {DWELL_SWEEP_H.size} values ...")
    dwell = sweep_dwell(setting, probe)
    for point in dwell:
        print(
            f"    dwell {point.dwell_hours:4.0f} h  timing "
            f"{point.timing_ratio:.3f}x  sorties/lunation "
            + " ".join(f"{point.sorties[b]:3d}" for b in BATTERY_LINES_WH)
            + f"  limited by {point.limited_by[BATTERY_LINES_WH[1]]}"
        )

    arguments.figure_directory.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)

    for name, figure in (
        ("light-over-the-year", build_light_figure(setting, probe, seasons)),
        (
            "sortie-energy-against-departure",
            build_departure_figure(setting, lunation, year),
        ),
        ("throughput-against-dwell", build_dwell_figure(dwell)),
    ):
        path = arguments.figure_directory / f"{name}.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        print(f"wrote {path.relative_to(REPOSITORY_ROOT)}")

    arguments.report.write_text(
        build_report(setting, probe, lunation, year, dwell, seasons),
        encoding="utf-8",
    )
    print(f"wrote {arguments.report.relative_to(REPOSITORY_ROOT)}")

    print("\n  measured against extrapolated\n")
    print(text_table(boundary_rows(setting, year, probe)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
