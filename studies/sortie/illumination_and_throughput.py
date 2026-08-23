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
# Five results.
#
# The destination is a permanently shadowed region. The route's endpoint sees no
# sunlight at any point in a year of lunations, because it sits under a horizon
# that rises to twenty-seven degrees. That was an open boundary row since Day 7
# and it closes in the favourable direction: de Gerlache's own interior is
# outside the window, but the route still ends somewhere genuinely dark.
#
# The rim crest is a viable charge point. It is lit about nine tenths of the
# time, with a horizon that falls away in every direction. Which makes the
# truncation caveat one-sided and worth stating precisely: rays leaving the
# window are treated as clear sky, so extra terrain can only raise a horizon.
# Darkness is therefore robust and sunlight is an upper bound. The destination
# being a PSR survives any wider search; the rim's nine tenths does not.
#
# Charge does not bind. A modest array at the rim recharges a sortie's battery
# in hours against a sortie that also takes hours, so throughput is limited by
# how long the walk takes and not by how long the waiting takes.
#
# Which shadow to visit is a mission decision, and it is now priced rather than
# argued. The nearest permanent shadow is 2.8 km out and costs 176 Wh round
# trip; the largest is 3.9 km and 234 Wh, for thirteen times the ground to work
# over; the deepest is 19.8 km and 908 Wh, which is a different undertaking
# entirely. All three are computed and all three are drawn, because the point of
# the machinery is to answer that question rather than to assume it.
#
# And the fifth is the one worth having, in a narrower form than it first got
# stated. Sortie energy has a minimum in walking speed, because swing work per
# metre rises as the square of speed while survival is a power times a duration
# and falls with it. But most of that duration is a fixed dwell, so the falling
# term is weak, the minimum is shallow, and it exists at all only where the
# shadowed share of the walk is large enough -- which across these three routes
# means only on the shortest. Throughput peaks elsewhere, at 0.60 m/s against
# 0.16, and crossing between them costs about twice the energy for about twice
# the sorties.
#
# So there is still no single best speed on the route that gets flown, and
# saying which one is being optimised is a mission decision rather than a
# modelling detail. It is a smaller effect than the first version claimed.
#
# On the route, because the first version of this study got it wrong in a way
# worth recording. It ran from the highest cell in the window to the lowest --
# corner to corner, twenty-five kilometres, three and a half thousand metres of
# descent, fifty-six hours of walking. That is not a sortie, it is the expedition
# this project ruled out on its first day when it rejected a descent to
# Shackleton's floor, and it had been rebuilt at de Gerlache without anyone
# noticing.
#
# The mission concept is charge on a lit rim, drop into a nearby cold trap, come
# back, and candidate regions are scoped on having permanent shadow within
# roughly two kilometres. So the route criterion is nearest permanent shadow
# from the best charge point, not deepest ground in the window -- and the
# illumination map has plenty of darker ground far closer than the far corner.
#
# The old route did earn its keep as an instrument test: full elevation range,
# every slope regime, maximum stress on the integration. It proved the
# machinery. It was not a mission.
#
# One defect found and fixed while adding the second and third routes, worth
# recording because of how it hid. illumination_fraction took a single latitude
# for a whole batch of points, and this runner passed the batch mean. A twenty
# kilometre window near the pole spans two thirds of a degree of latitude
# against an obliquity of 1.54, so the approximation was not small -- and it
# made a point's illumination depend on which other points were computed
# alongside it. The crest read 87.7% when it was probed with the far corner and
# 90.4% when it was probed with the nearest shadow, for no reason on the ground.
# Nothing caught it while one route was computed, because the number was only
# ever wrong the same way. Three routes made it inconsistent instead of merely
# biased, and inconsistency is visible. Latitude is now per point and the crest
# is 90.8% whoever it is measured beside.
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
ROUTE_SAMPLES: Final = 400

# How long the platform works at the destination. Survival power multiplies the
# hours spent cold, and on a short sortie through lit ground those are the dwell
# hours rather than the whole traverse -- which is where the first version of
# this study went wrong by a factor of ten.
DWELL_HOURS: Final = 4.0
ROUTE_ILLUMINATION_SPACING_M: Final = 40.0
TARGET_ORDER: Final = ("nearest", "largest", "deepest")
CHOSEN_TARGET: Final = "nearest"
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
        latitude_deg=latitudes(raster, rows, columns),
        north_azimuth_deg=north_azimuth_deg(raster, rows, columns),
    )


@dataclass(frozen=True, slots=True)
class Target:
    """One candidate destination, and why it was chosen.

    Which permanent shadow to visit is a genuine mission-design question rather
    than a modelling detail: the nearest is cheapest, the largest gives the most
    ground to work over, and the deepest is a different kind of expedition.
    """

    id: str
    row: int
    column: int
    distance_km: float
    drop_m: float
    region_area_km2: float


@dataclass(frozen=True, slots=True)
class TargetStyle:
    color: str
    dash: Any
    offset: tuple[float, float]
    align: str


TARGET_STYLE: Final[dict[str, TargetStyle]] = {
    "nearest": TargetStyle(ACCENT_SECONDARY, "solid", (-12.0, 10.0), "right"),
    "largest": TargetStyle(ACCENT_PRIMARY, (0, (5, 2)), (-14.0, 8.0), "right"),
    "deepest": TargetStyle(INK_MUTED, (0, (1.6, 1.6)), (14.0, 10.0), "left"),
}


def find_targets(
    setting_raster: GeoRaster,
    *,
    crest: tuple[int, int],
    grid_rows: NDArray[np.int_],
    grid_columns: NDArray[np.int_],
    lit: NDArray[np.float64],
) -> dict[str, Target]:
    """Nearest, largest and deepest permanent shadow, from the illumination map.

    Regions are four-connected components of the fully dark cells. The largest
    one is entered at its nearest member rather than its centroid, because a
    platform walks to the edge of a shadow and not to the middle of it.
    """
    dark = lit <= 0.0
    if not bool(dark.any()):
        raise ValueError(
            "no fully shadowed cell on the illumination grid; there is nowhere "
            "for this mission concept to go"
        )
    cell = setting_raster.cell_size_m
    distance_m = np.hypot(
        (grid_rows - crest[0]) * cell, (grid_columns - crest[1]) * cell
    )
    elevation = setting_raster.values[grid_rows, grid_columns]
    drop = setting_raster.values[crest[0], crest[1]] - elevation

    label = np.full(dark.shape, -1, dtype=int)
    count = 0
    for i in range(dark.shape[0]):
        for j in range(dark.shape[1]):
            if dark[i, j] and label[i, j] < 0:
                stack = [(i, j)]
                label[i, j] = count
                while stack:
                    a, b = stack.pop()
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        p, q = a + da, b + db
                        if (
                            0 <= p < dark.shape[0]
                            and 0 <= q < dark.shape[1]
                            and dark[p, q]
                            and label[p, q] < 0
                        ):
                            label[p, q] = count
                            stack.append((p, q))
                count += 1
    sizes = np.array([int((label == k).sum()) for k in range(count)])
    cell_area_km2 = (grid_rows[1, 0] - grid_rows[0, 0]) ** 2 * cell**2 / 1e6

    def make(identifier: str, index: tuple[int, int]) -> Target:
        region = int(label[index])
        return Target(
            id=identifier,
            row=int(grid_rows[index]),
            column=int(grid_columns[index]),
            distance_km=float(distance_m[index]) / 1000.0,
            drop_m=float(drop[index]),
            region_area_km2=float(sizes[region]) * cell_area_km2,
        )

    nearest = np.unravel_index(
        int(np.argmin(np.where(dark, distance_m, np.inf))), dark.shape
    )
    deepest = np.unravel_index(
        int(np.argmax(np.where(dark, drop, -np.inf))), dark.shape
    )
    biggest = int(np.argmax(sizes))
    members = np.argwhere(label == biggest)
    entry = members[int(np.argmin([distance_m[a, b] for a, b in members]))]
    return {
        "nearest": make("nearest", (int(nearest[0]), int(nearest[1]))),
        "largest": make("largest", (int(entry[0]), int(entry[1]))),
        "deepest": make("deepest", (int(deepest[0]), int(deepest[1]))),
    }


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


def load_setting(destination: tuple[int, int] | None = None) -> Setting:
    raster = read_float_geotiff(ELEVATION_PATH)
    platform = load_platform(PLATFORM_PATH).platform
    dataset = load_soil(SOIL_PATH).datasets["carrier1991"]
    contact = dataset.models["bekker"].extrapolating
    strength = mohr_coulomb_model(dataset, depth_range_cm="0-15")
    mobilization = janosi_hanamoto_model(dataset)

    highest = np.unravel_index(int(np.argmax(raster.values)), raster.values.shape)
    # The crest is the charge point. The destination is chosen for being dark
    # and close, not for being low: routing to the lowest cell in the window
    # builds an expedition rather than a sortie.
    lowest = (
        destination
        if destination is not None
        else np.unravel_index(int(np.argmin(raster.values)), raster.values.shape)
    )
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
    rows = np.rint(np.linspace(highest[0], lowest[0], ROUTE_SAMPLES)).astype(int)
    columns = np.rint(np.linspace(highest[1], lowest[1], ROUTE_SAMPLES)).astype(int)
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
    walking_hours: float
    locomotion_Wh: float
    survival_Wh: float

    @property
    def hours(self) -> float:
        return self.walking_hours + DWELL_HOURS

    @property
    def total_Wh(self) -> float:
        return self.locomotion_Wh + self.survival_Wh

    def sorties_per_week(self, charge_W: float) -> float:
        return HOURS_PER_WEEK / (self.hours + self.total_Wh / charge_W)


def sweep_speed(
    setting: Setting, *, survival_W: float, dark_route_fraction: float
) -> tuple[SpeedPoint, ...]:
    """Energy against speed, with survival charged for the hours actually cold.

    Those are the dwell hours plus whatever share of the traverse is shadowed.
    Charging survival over the whole sortie -- which the first version of this
    study did -- overstates it by the ratio of sortie to dwell, and on a short
    route through lit ground that is an order of magnitude.
    """
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
        walking_hours = distance / float(speed) / 3600.0
        cold_hours = DWELL_HOURS + dark_route_fraction * walking_hours
        points.append(
            SpeedPoint(
                speed_m_per_s=float(speed),
                walking_hours=walking_hours,
                locomotion_Wh=trip.total_J
                / JOULES_PER_WATT_HOUR
                * NOMINAL_DERATING,
                survival_Wh=survival_W * cold_hours,
            )
        )
    return tuple(points)


@dataclass(frozen=True, slots=True)
class Route:
    """One target, the walk to it, and what that walk costs across speed."""

    target: Target
    setting: Setting
    sampled_index: NDArray[np.int_]
    illumination: Illumination
    speed: tuple[SpeedPoint, ...]

    @property
    def dark_fraction(self) -> float:
        return float((self.illumination.any_sunlight_fraction <= 0.0).mean())

    @property
    def walking_hours(self) -> float:
        walked = (
            self.setting.trip.outbound.distance_m + self.setting.trip.inbound.distance_m
        )
        return walked / self.setting.platform.nominal_speed_m_per_s / 3600.0

    @property
    def sortie_hours(self) -> float:
        return self.walking_hours + DWELL_HOURS

    @property
    def locomotion_Wh(self) -> float:
        return self.setting.trip.total_J / JOULES_PER_WATT_HOUR * NOMINAL_DERATING

    @property
    def survival_Wh(self) -> float:
        cold = DWELL_HOURS + self.dark_fraction * self.walking_hours
        return INSULATED_SURVIVAL_W * cold

    @property
    def total_Wh(self) -> float:
        return self.locomotion_Wh + self.survival_Wh


def walk_to(target: Target) -> Route:
    setting = load_setting(destination=(target.row, target.column))
    spacing = float(setting.transect.distance_m[-1]) / (setting.route_rows.size - 1)
    stride = max(1, int(round(ROUTE_ILLUMINATION_SPACING_M / spacing)))
    index = np.unique(
        np.concatenate(
            [
                np.arange(0, setting.route_rows.size, stride),
                [setting.route_rows.size - 1],
            ]
        )
    )
    illumination = illuminate(
        setting.raster, setting.route_rows[index], setting.route_columns[index]
    )
    dark = float((illumination.any_sunlight_fraction <= 0.0).mean())
    return Route(
        target=target,
        setting=setting,
        sampled_index=index,
        illumination=illumination,
        speed=sweep_speed(
            setting,
            survival_W=INSULATED_SURVIVAL_W,
            dark_route_fraction=dark,
        ),
    )


def build_map_figure(
    routes: dict[str, Route], grid: Illumination, shape: tuple[int, int]
) -> Figure:
    raster = routes[CHOSEN_TARGET].setting.raster
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
                    "figure.figsize": (7.6, 8.8),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "axes.grid": False,
                    "figure.subplot.top": 0.744,
                    "figure.subplot.bottom": 0.060,
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

        for name in TARGET_ORDER:
            route = routes[name]
            style = TARGET_STYLE[name]
            setting = route.setting
            xs = [
                to_km(int(r), int(c))[0]
                for r, c in zip(setting.route_rows, setting.route_columns)
            ]
            ys = [
                to_km(int(r), int(c))[1]
                for r, c in zip(setting.route_rows, setting.route_columns)
            ]
            panel.plot(xs, ys, color="white", linewidth=3.0, alpha=0.85)
            panel.plot(
                xs,
                ys,
                color=style.color,
                linewidth=1.6 if name == CHOSEN_TARGET else 1.2,
                linestyle=style.dash,
                label=(
                    f"{name}, {route.target.distance_km:.1f} km, "
                    f"{route.total_Wh:.0f} Wh"
                    + (" — the mission" if name == CHOSEN_TARGET else "")
                ),
            )
            end_x, end_y = to_km(route.target.row, route.target.column)
            panel.plot(
                [end_x],
                [end_y],
                marker="s",
                markersize=7.0,
                markerfacecolor="none",
                markeredgewidth=1.6,
                color="white",
            )
            panel.annotate(
                name,
                xy=(end_x, end_y),
                xytext=style.offset,
                textcoords="offset points",
                ha=style.align,
                color="white",
                fontsize=8.0,
            )

        crest_x, crest_y = to_km(*routes[CHOSEN_TARGET].setting.crest)
        panel.plot(
            [crest_x],
            [crest_y],
            marker="o",
            markersize=8.0,
            markerfacecolor="none",
            markeredgewidth=1.8,
            color="white",
        )
        panel.annotate(
            "charge point\non the crest",
            xy=(crest_x, crest_y),
            xytext=(-10, -34),
            textcoords="offset points",
            ha="center",
            color="white",
            fontsize=8.0,
        )
        legend = panel.legend(loc="lower left", labelcolor="white", framealpha=0.0)
        for text in legend.get_texts():
            text.set_color("white")
        panel.set_xlabel("polar stereographic x (km)")
        panel.set_ylabel("polar stereographic y (km)")
        panel.set_aspect("equal")

        figure.suptitle(
            "Three permanent shadows reachable from one lit crest, and they are "
            "not equivalent",
            color=INK_PRIMARY,
            fontsize=11.0,
            x=0.108,
            ha="left",
            y=0.976,
        )
        figure.text(
            0.108,
            0.936,
            caption(
                "Horizon computed from the 5 m LOLA grid at every point shown, "
                "swept over a year of lunations with the Sun treated as a disc "
                f"of {SOLAR_ANGULAR_RADIUS_DEG * 2:.2f}° and the sub-solar latitude "
                f"oscillating within the Moon's {LUNAR_OBLIQUITY_DEG:.2f}° "
                "obliquity. Lunar curvature is included: it drops distant ground "
                "by 29 m at 10 km.\n"
                "Targets are the nearest, the largest and the deepest fully dark "
                "ground on this map. Which one a mission goes to is a real choice "
                "and the machinery now prices it; the nearest is walked here "
                "because the concept is a day trip from a charge point. Energies "
                "in the legend are round trips at the platform's nominal "
                f"{routes[CHOSEN_TARGET].setting.platform.nominal_speed_m_per_s:.2f} m/s; "
                "what speed costs is the third figure.\n"
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


def build_route_figure(routes: dict[str, Route]) -> Figure:
    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (10.2, 5.8),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.660,
                    "figure.subplot.bottom": 0.150,
                    "figure.subplot.left": 0.080,
                    "figure.subplot.right": 0.905,
                    "figure.subplot.hspace": 0.380,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(2, 1, squeeze=False, sharex=True)
        upper, lower = axes[0][0], axes[1][0]

        for name in TARGET_ORDER:
            route = routes[name]
            style = TARGET_STYLE[name]
            transect = route.setting.transect
            distance_km = transect.distance_m[route.sampled_index] / 1000.0
            any_sun = route.illumination.any_sunlight_fraction
            dark = any_sun <= 0.0
            width = 1.9 if name == CHOSEN_TARGET else 1.3

            upper.plot(
                distance_km,
                any_sun * 100.0,
                color=style.color,
                linewidth=width,
                linestyle=style.dash,
                label=f"{name}, {route.dark_fraction:.0%} of the walk in shadow",
            )
            elevation = transect.elevation_m[route.sampled_index]
            lower.plot(
                distance_km,
                elevation,
                color=style.color,
                linewidth=width,
                linestyle=style.dash,
            )
            lower.plot(
                np.where(dark, distance_km, np.nan),
                np.where(dark, elevation, np.nan),
                color=style.color,
                linewidth=width + 3.4,
                alpha=0.30,
                solid_capstyle="round",
            )
            if bool(dark.any()):
                entry = int(np.argmax(dark))
                upper.plot(
                    [distance_km[entry]],
                    [any_sun[entry] * 100.0],
                    marker="v",
                    markersize=5.5,
                    color=style.color,
                )

        upper.set_ylabel("any sunlight (% of year)")
        upper.set_title(
            "illumination along each route", color=INK_SECONDARY, loc="left"
        )
        upper.set_ylim(-4.0, 102.0)
        upper.legend(loc="upper right")

        lower.set_xlabel("distance from the charge point (km)")
        lower.set_ylabel("elevation (m)")
        lower.set_title(
            "and the profile each one follows", color=INK_SECONDARY, loc="left"
        )

        for panel in (upper, lower):
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)

        chosen = routes[CHOSEN_TARGET]
        figure.suptitle(
            "Every route is a lit traverse into a dark end, and the short one "
            "spends the largest share of itself cold",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.080,
            ha="left",
            y=0.962,
        )
        figure.text(
            0.080,
            0.908,
            caption(
                "Shading marks the shadowed stretches. Each route ends in "
                "permanent shadow and is otherwise lit, apart from short "
                "crossings: the mission route dips into one at 1.6 km before "
                "reaching its destination at 2.6. So survival power is mostly a "
                "cost of arriving rather than of travelling, and the share is not "
                "the same across the three, because a shadowed approach is a "
                f"fixed length while the walk is not — {chosen.dark_fraction:.0%} "
                f"of the {chosen.target.distance_km:.1f} km route against "
                f"{routes['deepest'].dark_fraction:.0%} of the "
                f"{routes['deepest'].target.distance_km:.1f} km one. Illumination "
                f"is sampled every {ROUTE_ILLUMINATION_SPACING_M:.0f} m on all "
                "three so those shares are comparable; crossings shorter than "
                "that are still missed.\n"
                "How long the platform is actually cold therefore depends on WHEN "
                "the sortie runs as well as where it goes: a "
                f"{chosen.sortie_hours:.0f} hour round trip is a hundredth of a "
                "lunation, and one timed into the lit part of the cycle spends "
                "little of it in shadow while a badly timed one spends all of it. "
                "Day 8's survival power applies to the dark hours, and scheduling "
                "decides how many there are. This study does not schedule.",
                width=150,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def build_throughput_figure(routes: dict[str, Route], crest_lit: float) -> Figure:
    charge_W = average_charge_W(crest_lit)
    chosen = routes[CHOSEN_TARGET]
    speeds = np.asarray([point.speed_m_per_s for point in chosen.speed])
    locomotion = np.asarray([point.locomotion_Wh for point in chosen.speed])
    survival = np.asarray([point.survival_Wh for point in chosen.speed])
    cap = maximum_walking_speed(
        platform=chosen.setting.platform,
        strength=chosen.setting.strength,
        gravity_m_per_s2=LUNAR_GRAVITY,
    )

    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (12.2, 6.1),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.610,
                    "figure.subplot.bottom": 0.168,
                    "figure.subplot.left": 0.060,
                    "figure.subplot.right": 0.985,
                    "figure.subplot.wspace": 0.250,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 3, squeeze=False)
        left, middle, right = axes[0][0], axes[0][1], axes[0][2]

        left.stackplot(
            speeds,
            locomotion,
            survival,
            colors=[ACCENT_PRIMARY, ACCENT_SECONDARY],
            labels=[
                "locomotion, rises as speed squared",
                "survival, a dwell plus a shadowed approach",
            ],
            edgecolor="none",
            alpha=0.9,
        )
        left.plot(
            speeds,
            np.asarray([point.total_Wh for point in chosen.speed]),
            color=INK_PRIMARY,
            linewidth=1.6,
        )
        left.set_xlabel("walking speed (m/s)")
        left.set_ylabel("sortie energy (Wh)")
        left.set_title(
            f"what the {CHOSEN_TARGET} sortie is made of",
            color=INK_SECONDARY,
            loc="left",
        )
        left.set_xlim(speeds[0], speeds[-1])
        left.set_ylim(0.0, float(locomotion[-1] + survival[-1]) * 1.05)
        left.legend(loc="upper center")

        least_energy: dict[str, float] = {}
        most_sorties: dict[str, float] = {}
        for name in TARGET_ORDER:
            route = routes[name]
            style = TARGET_STYLE[name]
            width = 1.9 if name == CHOSEN_TARGET else 1.3
            energies = np.asarray([point.total_Wh for point in route.speed])
            throughput = np.asarray(
                [point.sorties_per_week(charge_W) for point in route.speed]
            )
            cheapest = int(np.argmin(energies))
            fastest = int(np.argmax(throughput))
            least_energy[name] = float(speeds[cheapest])
            most_sorties[name] = float(speeds[fastest])

            middle.plot(
                speeds,
                energies,
                color=style.color,
                linewidth=width,
                linestyle=style.dash,
                label=f"{name}, least {energies[cheapest]:.0f} Wh",
            )
            middle.plot(
                [speeds[cheapest]],
                [energies[cheapest]],
                marker="o",
                markersize=5.5,
                markerfacecolor="none",
                color=style.color,
            )
            right.plot(
                speeds,
                throughput,
                color=style.color,
                linewidth=width,
                linestyle=style.dash,
                label=f"{name}, peak {throughput[fastest]:.1f} per week",
            )
            right.plot(
                [speeds[fastest]],
                [throughput[fastest]],
                marker="o",
                markersize=5.5,
                markerfacecolor="none",
                color=style.color,
            )

        middle.set_xlabel("walking speed (m/s)")
        middle.set_ylabel("sortie energy (Wh)")
        middle.set_title(
            "and only the shortest has a minimum at all",
            color=INK_SECONDARY,
            loc="left",
        )
        middle.set_xlim(speeds[0], speeds[-1])
        middle.set_yscale("log")
        middle.legend(loc="upper left")

        right.set_xlabel("walking speed (m/s)")
        right.set_ylabel("sorties per week")
        right.set_title(
            f"and throughput peaks elsewhere, on {charge_W:.0f} W",
            color=INK_SECONDARY,
            loc="left",
        )
        right.set_xlim(speeds[0], speeds[-1])
        right.set_ylim(0.0, None)
        right.legend(loc="lower center")

        for panel in (left, middle, right):
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)

        figure.suptitle(
            "Energy and throughput want different speeds, and only the short "
            "sortie has a cheapest one at all",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.060,
            ha="left",
            y=0.955,
        )
        figure.text(
            0.060,
            0.912,
            caption(
                "Swing work per metre goes as the square of speed, so locomotion "
                "rises. Survival is a power times a duration and most of that "
                "duration is a fixed dwell, so it falls toward a floor rather than "
                "as one over speed. Their sum has a minimum, but a shallow one, "
                "and not where throughput is greatest — a faster sortie is a "
                "shorter one even when it costs more, until slip runs away and "
                f"throughput turns over below the {cap:.2f} m/s gait limit from "
                "rung three.\n"
                "Whether that minimum exists at all depends on the target. A "
                "shadowed approach is roughly a fixed length, so it is "
                f"{routes[CHOSEN_TARGET].dark_fraction:.0%} of the mission route "
                f"and {routes['deepest'].dark_fraction:.1%} of the deepest; on the "
                "two longer routes survival is too nearly constant to bend the "
                "sum, and energy simply rises with speed. On the mission route "
                "the two "
                f"optima sit at {least_energy[CHOSEN_TARGET]:.2f} and "
                f"{most_sorties[CHOSEN_TARGET]:.2f} m/s, and crossing between them "
                "costs about twice the energy for about twice the sorties.\n"
                f"Charge does not bind on any of the three. At {ARRAY_AREA_M2:.1f} m² "
                f"and {ARRAY_EFFICIENCY:.0%} the crest returns {charge_W:.0f} W, so "
                "recharging takes hours against a sortie measured in hours too. "
                "Energy is on a log axis because the deepest target costs five "
                "times the nearest.",
                width=178,
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
                "sortie that also takes hours, so throughput is limited by "
                "walking rather than by waiting"
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
    routes: dict[str, Route],
    grid: Illumination,
    probes: Illumination,
) -> str:
    chosen = routes[CHOSEN_TARGET]
    crest_lit = float(probes.any_sunlight_fraction[0])
    destination_lit = float(probes.any_sunlight_fraction[1])
    charge_W = average_charge_W(crest_lit)
    rows = boundary_rows(chosen.setting, grid, crest_lit)
    insulated = chosen.speed
    energies = [point.total_Wh for point in insulated]
    throughput = [point.sorties_per_week(charge_W) for point in insulated]
    cheapest = int(np.argmin(energies))
    fastest = int(np.argmax(throughput))
    dark_fraction = chosen.dark_fraction

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
        f"dwell_hours = {_format_float(DWELL_HOURS)}",
        'survival_applies_to = "dwell hours plus the shadowed share of the '
        'traverse, not the whole sortie"',
        "",
        "# Which permanent shadow to visit, priced. The nearest is the mission;",
        "# the deepest is an expedition and is included to show the difference.",
        "# An earlier version of this study routed to the lowest cell in the",
        "# window and built the second by accident.",
        "",
        "# The trade that has no single answer. Locomotion rises as the square of",
        "# speed because swing work does; survival falls as one over speed",
        "# because it is a power times a duration.",
        "",
    ]
    for name in TARGET_ORDER:
        route = routes[name]
        target = route.target
        energies_here = [point.total_Wh for point in route.speed]
        throughput_here = [
            point.sorties_per_week(charge_W) for point in route.speed
        ]
        least = int(np.argmin(energies_here))
        most = int(np.argmax(throughput_here))
        lines += [
            "[[target]]",
            f'id = "{target.id}"',
            f"distance_km = {_format_float(target.distance_km)}",
            f"drop_m = {_format_float(target.drop_m)}",
            f"shadow_area_km2 = {_format_float(target.region_area_km2)}",
            "shadowed_fraction_of_route = "
            f"{_format_float(route.dark_fraction)}",
            f"walking_hours = {_format_float(route.walking_hours)}",
            f"sortie_hours = {_format_float(route.sortie_hours)}",
            f"locomotion_Wh = {_format_float(route.locomotion_Wh)}",
            f"survival_Wh = {_format_float(route.survival_Wh)}",
            f"total_Wh = {_format_float(route.total_Wh)}",
            "least_energy_speed_m_per_s = "
            f"{_format_float(route.speed[least].speed_m_per_s)}",
            f"least_energy_Wh = {_format_float(energies_here[least])}",
            "most_sorties_speed_m_per_s = "
            f"{_format_float(route.speed[most].speed_m_per_s)}",
            f"most_sorties_per_week = {_format_float(throughput_here[most])}",
            "chosen = " + str(name == CHOSEN_TARGET).lower(),
            "",
        ]

    lines += [
        "# Energy against speed, on the chosen route.",
        "",
    ]
    for point in insulated[::4]:
        lines += [
            "[[speed]]",
            f"speed_m_per_s = {_format_float(point.speed_m_per_s)}",
            f"walking_hours = {_format_float(point.walking_hours)}",
            f"sortie_hours = {_format_float(point.hours)}",
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
                platform=chosen.setting.platform,
                strength=chosen.setting.strength,
                gravity_m_per_s2=LUNAR_GRAVITY,
            )
        ),
        "",
        "# Which of the three binds, now that three compete.",
        "[binding]",
        "statement = \"\"\"",
        "Charge does not bind. A modest array at the crest recharges a sortie's",
        "battery in hours against a sortie that also takes hours, so throughput",
        "is set by how long the walk takes rather than how long the waiting",
        "takes.",
        "",
        "What binds is duration, and duration is a choice. Survival is charged",
        "for the hours actually spent cold: the dwell, plus the shadowed final",
        "approach, which on this route is about an eighth of the walk. So it is",
        "mostly a fixed cost with a small speed-dependent part, and the energy",
        "minimum it produces is shallow and sits near a sixth of a metre per",
        "second. Throughput rises with speed until slip runs away near the gait",
        "limit from rung three, and then falls.",
        "",
        "That minimum is a property of the route rather than of the platform.",
        "A shadowed approach is roughly a fixed length, so it is an eighth of the",
        "nearest route and under two percent of the deepest, and on the two",
        "longer routes survival is too nearly constant in speed to bend the sum",
        "at all. Their least-energy speed is simply the slowest speed swept, and",
        "the per-target blocks above report it as such rather than as an",
        "optimum.",
        "",
        "An earlier version of this study put that minimum at twice the speed",
        "and made much more of it. That was an artifact of the route: charging",
        "survival power over a fifty-six hour traverse that should never have",
        "been the mission made the falling term far larger than it is. The trade",
        "is real but it is shallow, and between the least-energy and",
        "most-throughput speeds the energy penalty is about a factor of two for",
        "roughly a doubling of sorties.",
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

    # Pass one: the illumination map, which is what target selection needs.
    scout = load_setting()
    height, width = scout.raster.shape
    grid_rows, grid_columns = np.meshgrid(
        np.arange(0, height, MAP_STRIDE),
        np.arange(0, width, MAP_STRIDE),
        indexing="ij",
    )
    shape = grid_rows.shape
    print(f"  horizon over {grid_rows.size} map points ...")
    grid = illuminate(scout.raster, grid_rows.ravel(), grid_columns.ravel())
    lit_map = grid.any_sunlight_fraction.reshape(shape)

    targets = find_targets(
        scout.raster,
        crest=scout.crest,
        grid_rows=grid_rows,
        grid_columns=grid_columns,
        lit=lit_map,
    )
    for target in targets.values():
        print(
            f"  {target.id:8s} PSR: {target.distance_km:5.2f} km, "
            f"{target.drop_m:6.0f} m below the crest, "
            f"{target.region_area_km2:.2f} km2 of shadow"
        )

    # Pass two: walk to every candidate, because which shadow to visit is a
    # real choice and pricing one of them cannot answer it.
    routes = {name: walk_to(target) for name, target in targets.items()}
    chosen = routes[CHOSEN_TARGET]

    probes = illuminate(
        chosen.setting.raster,
        np.array([chosen.setting.crest[0], chosen.target.row]),
        np.array([chosen.setting.crest[1], chosen.target.column]),
    )
    crest_lit = float(probes.any_sunlight_fraction[0])

    arguments.figure_directory.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)

    for name, figure in (
        ("illumination-map", build_map_figure(routes, grid, shape)),
        ("illumination-along-the-route", build_route_figure(routes)),
        ("speed-energy-throughput", build_throughput_figure(routes, crest_lit)),
    ):
        path = arguments.figure_directory / f"{name}.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        print(f"wrote {path.relative_to(REPOSITORY_ROOT)}")

    arguments.report.write_text(
        build_report(routes, grid, probes), encoding="utf-8"
    )
    print(f"wrote {arguments.report.relative_to(REPOSITORY_ROOT)}")

    print("\n  measured against extrapolated\n")
    print(text_table(boundary_rows(chosen.setting, grid, crest_lit)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
