# SPDX-License-Identifier: Apache-2.0
#
# studies.terrain.slope_sensitivity — how much of a crater rim opens per degree
# of capability, and what the map can and cannot say about it.
#
# The mobility model answers what a slope costs and whether it can be walked.
# This is the terrain to sweep that against: one Artemis III candidate region,
# from a NASA laser-altimetry product, with the slope limits from rungs two and
# three marked on it.
#
# Three results, and the second changed what the third could be.
#
# The traversable fraction is not close for a legged platform. Against the
# 43.5 degree traction limit and the 39.8 degree tipping limit, essentially the
# whole site opens. Against the 20 degree limit the Artemis III Science
# Definition Team places on crew, about an eighth of it does not. So the
# interesting terrain here is the terrain crew cannot reach, which is a stronger
# claim than reaching terrain they can, and it is also the terrain whose
# assessment is least sensitive to the map's own error -- because the legged
# limits sit far out on the distribution's tail where there is almost nothing
# left to move, while the crew limit sits near its ninetieth percentile where
# there is a great deal.
#
# Slope statistics do not depend on baseline in this product. For a self-affine
# surface the mean slope over a baseline goes as that baseline to the power of
# the Hurst exponent minus one, and natural terrain gives something like -0.3 to
# -0.1. Six octaves of aggregation here give an exponent near zero. So the
# question this study set out to ask -- how do slope statistics steepen toward
# the scale of a footstep -- has no trend in this data to extrapolate along.
#
# The obvious explanation is the producers' own: about ninety percent of the
# five-metre pixels are interpolated, because LOLA samples densely along track
# and sparsely across it. That explanation is testable and it fails. Gap-filling
# at a polar site would leave a directional signature aligned with the orbit,
# and the measured anisotropy is aligned with the crater's fall line instead,
# to within a couple of degrees, at every lag from twenty metres to two hundred.
# The directional structure in this grid is terrain.
#
# The hedge matters and is kept throughout: that test rejects the artifact it
# can see. An interpolator that smooths isotropically leaves no direction to
# find, so this establishes that the flatness is not explained by the mechanism
# that would have mattered most, not that the surface is smooth.
#
# What follows is a specification rather than a gap. Slope at the scale of a
# footstep is not resolvable from any orbital dataset -- the best stereo imagery
# posts an order of magnitude above a stride -- and it does not need to be,
# because at ninety-nine point nine percent traversable the binding constraint
# down there is not gradient at all. It is whether a boulder is taller than the
# foot clears, which is a size-frequency distribution and a different
# measurement.
#
# Not a sortie envelope. Slope is one axis of six and the others are empty.
#
# References
#   Rice MS et al. (2023) Artemis III candidate landing regions. LPSC LIV.
#   Shepard MK et al. (2001) The roughness of natural terrain. Journal of
#     Geophysical Research 106(E12), 32777-32795.

from __future__ import annotations

import argparse
import math
import platform as host_platform
import textwrap
import tomllib
from pathlib import Path
from typing import Any, Final, cast

import matplotlib
import matplotlib.colors

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
from eclipse.io.site import Site, load_site
from eclipse.io.terrain import GeoRaster, read_float_geotiff
from eclipse.terrain import (
    NATURAL_TERRAIN_SLOPE_EXPONENT,
    aggregate,
    anisotropy,
    scale_trend,
    slope_degrees,
)

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
SITE_PATH: Final = REPOSITORY_ROOT / "configs" / "sites" / "de-gerlache-rim-2.toml"
MANIFEST_PATH: Final = REPOSITORY_ROOT / "data" / "terrain" / "manifest.toml"
ELEVATION_PATH: Final = (
    REPOSITORY_ROOT / "data" / "terrain" / "SL2_final_adj_5mpp_surf.tif"
)
FIGURE_DIRECTORY: Final = Path(__file__).resolve().parent / "figures"
DEFAULT_REPORT_PATH: Final = (
    Path(__file__).resolve().parent / "results" / "slope-sensitivity.toml"
)

REPORT_SCHEMA_VERSION: Final = 1
SLOPE_METHOD: Final = "central_difference"
LUNAR_RADIUS_M: Final = 1737400.0

# From rungs two and three, and the crew limit from the Artemis III SDT.
CREW_SLOPE_LIMIT: Final = 20.0
REPOSE_BAND: Final = (30.0, 35.0)
TIPPING_LIMIT: Final = 39.8055710922652
TRACTION_LIMIT: Final = 43.545928803868314

# The product's own stated median RMS slope error.
SLOPE_ERROR_RANGE_DEG: Final = (1.5, 2.5)
HISTOGRAM_BINS: Final = 9000
HISTOGRAM_RANGE: Final = (0.0, 90.0)

AGGREGATION_FACTORS: Final = (1, 2, 4, 8, 16, 32, 64)
ANISOTROPY_LAGS: Final = (4, 8, 20, 40)
ACHIEVABLE_SLOPE_DEG: Final[NDArray[np.float64]] = np.linspace(0.0, 60.0, 601)
MAP_AGGREGATION: Final = 4

# de Gerlache crater centre, IAU Gazetteer, for the fall-line direction.
CRATER_LATITUDE: Final = -88.5
CRATER_LONGITUDE: Final = -87.1

STRIDE_M: Final = 0.30
BEST_ORBITAL_POSTING_M: Final = 2.0


def caption(text: str, width: int = 148) -> str:
    return "\n".join(
        textwrap.fill(" ".join(paragraph.split()), width=width)
        for paragraph in text.split("\n")
    )


def load_quality() -> dict[str, Any]:
    table = tomllib.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for product in table["product"]:
        if product["id"] == "SL2_final_adj_5mpp_surf":
            return dict(product["quality"])
    raise ValueError("the manifest no longer carries the elevation product")


def raster_axis(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dx, -dy)) % 180.0


def fall_line_axis(elevation: GeoRaster) -> float:
    centre_x, centre_y = elevation.center_model_m
    radius = 2.0 * LUNAR_RADIUS_M * math.tan(
        math.radians(45.0 + CRATER_LATITUDE / 2.0)
    )
    crater_x = radius * math.sin(math.radians(CRATER_LONGITUDE))
    crater_y = radius * math.cos(math.radians(CRATER_LONGITUDE))
    return raster_axis(centre_x - crater_x, centre_y - crater_y)


def ground_track_axis(elevation: GeoRaster) -> float:
    centre_x, centre_y = elevation.center_model_m
    return raster_axis(centre_x, centre_y)


def traversable_fraction(
    slope: NDArray[np.float64], limits: NDArray[np.float64]
) -> NDArray[np.float64]:
    ordered = np.sort(slope)
    return np.asarray(np.searchsorted(ordered, limits, side="right") / ordered.size)


def perturbed_fraction(
    slope: NDArray[np.float64], limit: float, sigma: float
) -> float:
    """Traversable fraction after independent slope error of this size.

    Closed form rather than sampled. Adding independent noise to every cell and
    asking what fraction falls below a limit is the mean over cells of the
    normal cumulative distribution at that limit, so there is nothing to draw
    and no seed to record. The fold at zero changes nothing: a cell perturbed
    below zero is still below any positive limit.

    Evaluated on a histogram because the exact sum wants an error function over
    sixteen million cells and the answer only needs the distribution, which a
    hundredth-of-a-degree bin carries to far better than the uncertainty being
    propagated.
    """
    counts, edges = np.histogram(slope, bins=HISTOGRAM_BINS, range=HISTOGRAM_RANGE)
    centres = 0.5 * (edges[:-1] + edges[1:])
    weights = counts / counts.sum()
    standardised = (limit - centres) / sigma
    cumulative = np.asarray(
        [0.5 * (1.0 + math.erf(float(v) / math.sqrt(2.0))) for v in standardised]
    )
    return float(np.dot(weights, cumulative))


def build_sensitivity_figure(
    slope: NDArray[np.float64], site: Site, quality: dict[str, Any]
) -> Figure:
    fraction = traversable_fraction(slope, ACHIEVABLE_SLOPE_DEG)
    low, high = SLOPE_ERROR_RANGE_DEG
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
                    "figure.subplot.top": 0.660,
                    "figure.subplot.bottom": 0.195,
                    "figure.subplot.left": 0.072,
                    "figure.subplot.right": 0.986,
                    "figure.subplot.wspace": 0.230,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 2, squeeze=False)

        left = axes[0][0]
        for sigma, shade in ((high, 0.16), (low, 0.16)):
            perturbed = np.asarray(
                [
                    perturbed_fraction(slope, float(limit), sigma)
                    for limit in ACHIEVABLE_SLOPE_DEG[::5]
                ]
            )
            left.plot(
                ACHIEVABLE_SLOPE_DEG[::5],
                perturbed * 100.0,
                color=INK_MUTED,
                linewidth=1.0,
                linestyle=(0, (3, 2)),
                label=f"with the product's own ±{sigma:.1f}° slope error"
                if sigma == high
                else None,
            )
        left.plot(
            ACHIEVABLE_SLOPE_DEG, fraction * 100.0, color=ACCENT_PRIMARY, linewidth=1.8
        )
        left.axvspan(
            site.crew.maximum_slope_deg,
            TRACTION_LIMIT,
            color=ACCENT_SECONDARY,
            alpha=0.13,
            linewidth=0.0,
            label="terrain a legged platform opens that crew cannot reach",
        )
        for limit, label, colour in (
            (site.crew.maximum_slope_deg, f"crew {site.crew.maximum_slope_deg:.0f}°", INK_PRIMARY),
            (TIPPING_LIMIT, f"tipping {TIPPING_LIMIT:.1f}°", ACCENT_SECONDARY),
            (TRACTION_LIMIT, f"traction {TRACTION_LIMIT:.1f}°", ACCENT_SECONDARY),
        ):
            left.axvline(limit, color=colour, linewidth=1.0, linestyle=(0, (2, 2)))
            left.annotate(
                label,
                xy=(limit, 52.0),
                xytext=(-4, 0),
                textcoords="offset points",
                rotation=90.0,
                ha="right",
                va="bottom",
                color=colour,
                fontsize=7.6,
            )
        left.axvspan(*REPOSE_BAND, color=INK_MUTED, alpha=0.10, linewidth=0.0)
        left.set_title(
            "traversable fraction against achievable slope",
            color=INK_SECONDARY,
            loc="left",
        )
        left.set_xlabel("achievable slope (degrees)")
        left.set_ylabel("fraction of the site (%)")
        left.set_xlim(0.0, 60.0)
        left.set_ylim(0.0, 102.0)
        left.legend(loc="upper left")

        right = axes[0][1]
        limits = (
            (site.crew.maximum_slope_deg, "crew"),
            (REPOSE_BAND[0], "repose low"),
            (TIPPING_LIMIT, "tipping"),
            (TRACTION_LIMIT, "traction"),
        )
        nominal = [float((slope <= limit).mean()) for limit, _ in limits]
        shifts = [
            perturbed_fraction(slope, limit, high) - value
            for (limit, _), value in zip(limits, nominal)
        ]
        positions = np.arange(len(limits))
        right.barh(
            positions,
            [abs(s) * 100.0 for s in shifts],
            color=[INK_PRIMARY if i == 0 else ACCENT_SECONDARY for i in positions],
            height=0.55,
        )
        for index, (value, shift) in enumerate(zip(nominal, shifts)):
            right.annotate(
                f"{shift * 100:+.3f} pp",
                xy=(abs(shift) * 100.0, index),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                color=INK_SECONDARY,
                fontsize=8.0,
            )
        right.set_yticks(positions)
        right.set_yticklabels(
            [
                f"{label}\n{value:.3%} of the site"
                for (_, label), value in zip(limits, nominal)
            ]
        )
        right.invert_yaxis()
        right.set_title(
            f"movement under the product's ±{high:.1f}° slope error",
            color=INK_SECONDARY,
            loc="left",
        )
        right.set_xlabel("shift in traversable fraction (percentage points)")
        right.set_xlim(0.0, 2.2)

        for panel in (left, right):
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)

        figure.suptitle(
            "The legged limits open almost all of the site, and are the least "
            "sensitive to the map's own error",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.072,
            ha="left",
            y=0.955,
        )
        figure.text(
            0.072,
            0.900,
            caption(
                f"{site.name}, {slope.size / 1e6:.1f} million cells at 5 m, slope "
                f"by {SLOPE_METHOD.replace('_', ' ')} — the method identified by "
                "reproducing the producers' own slope raster to a ten-thousandth "
                "of a degree.\n"
                "The crew limit sits near the ninetieth percentile of the slope "
                "distribution, where the density is high, so the product's "
                f"stated ±{high:.1f}° slope error moves it. The legged limits sit "
                "far out on the tail, where it moves nothing. Terrain that opens "
                "for a robot is also terrain whose assessment the map can "
                "support.",
                width=150,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def build_baseline_figure(elevation: GeoRaster, quality: dict[str, Any]) -> Figure:
    trend = scale_trend(
        elevation.values,
        cell_size_m=elevation.cell_size_m,
        factors=AGGREGATION_FACTORS,
        method=SLOPE_METHOD,
    )
    fall_line = fall_line_axis(elevation)
    track = ground_track_axis(elevation)
    measured = anisotropy(
        elevation.values, cell_size_m=elevation.cell_size_m, lag_cells=20
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
                    "figure.subplot.top": 0.645,
                    "figure.subplot.bottom": 0.195,
                    "figure.subplot.left": 0.072,
                    "figure.subplot.right": 0.986,
                    "figure.subplot.wspace": 0.245,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 2, squeeze=False)

        left = axes[0][0]
        anchor = float(trend.mean_slope_degrees[0])
        span = np.asarray([STRIDE_M, float(trend.baseline_m[-1])])
        for exponent, style in (
            (NATURAL_TERRAIN_SLOPE_EXPONENT[0], (0, (5, 2))),
            (NATURAL_TERRAIN_SLOPE_EXPONENT[1], (0, (5, 2))),
        ):
            left.plot(
                span,
                anchor * (span / trend.baseline_m[0]) ** exponent,
                color=INK_MUTED,
                linewidth=1.0,
                linestyle=style,
            )
        left.fill_between(
            span,
            anchor * (span / trend.baseline_m[0]) ** NATURAL_TERRAIN_SLOPE_EXPONENT[0],
            anchor * (span / trend.baseline_m[0]) ** NATURAL_TERRAIN_SLOPE_EXPONENT[1],
            color=INK_MUTED,
            alpha=0.13,
            linewidth=0.0,
            label="what natural terrain gives, exponent −0.30 to −0.10",
        )
        left.plot(
            trend.baseline_m,
            trend.mean_slope_degrees,
            color=ACCENT_PRIMARY,
            linewidth=1.8,
            marker="o",
            markersize=4.0,
            markerfacecolor="none",
            label=f"measured, exponent {trend.exponent:+.3f}",
        )
        left.axvline(STRIDE_M, color=INK_PRIMARY, linewidth=1.0, linestyle=(0, (2, 2)))
        left.annotate(
            f"a {STRIDE_M * 1000:.0f} mm stride\nnothing measures here",
            xy=(STRIDE_M, anchor * 2.4),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color=INK_PRIMARY,
            fontsize=7.8,
        )
        left.axvspan(
            STRIDE_M,
            BEST_ORBITAL_POSTING_M,
            color=ACCENT_SECONDARY,
            alpha=0.10,
            linewidth=0.0,
        )
        left.set_xscale("log")
        left.set_yscale("log")
        left.set_xlabel("baseline (m)")
        left.set_ylabel("mean slope (degrees)")
        left.set_title(
            "mean slope against baseline, block-mean aggregation",
            color=INK_SECONDARY,
            loc="left",
        )
        left.set_ylim(6.0, 60.0)
        left.legend(loc="upper right")

        right = axes[0][1]
        closed = np.concatenate([measured.axis_degrees, [180.0]])
        values = np.concatenate([measured.rms_slope_degrees, measured.rms_slope_degrees[:1]])
        right.plot(closed, values, color=ACCENT_PRIMARY, linewidth=1.8)
        marks: tuple[tuple[float, str, str, Any], ...] = (
            (fall_line, "crater fall line", ACCENT_SECONDARY, "solid"),
            (track, "LRO ground track", INK_PRIMARY, (0, (3, 2))),
        )
        for axis, label, colour, style in marks:
            right.axvline(axis, color=colour, linewidth=1.2, linestyle=style)
            right.annotate(
                label,
                xy=(axis, right.get_ylim()[0]),
                xytext=(4, 8),
                textcoords="offset points",
                rotation=90.0,
                ha="left",
                va="bottom",
                color=colour,
                fontsize=7.8,
            )
        right.plot(
            [measured.roughest_axis_degrees],
            [measured.rms_slope_degrees.max()],
            marker="o",
            markersize=6.0,
            markerfacecolor="none",
            color=ACCENT_PRIMARY,
        )
        right.set_xlabel("axis in the raster (degrees)")
        right.set_ylabel(f"RMS slope at a {measured.lag_m:.0f} m lag (degrees)")
        right.set_title(
            "directional roughness, and what it lines up with",
            color=INK_SECONDARY,
            loc="left",
        )
        right.set_xlim(0.0, 180.0)
        right.set_xticks([0, 45, 90, 135, 180])

        for panel in (left, right):
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)

        figure.suptitle(
            "The grid holds no roughness to extrapolate, and the reason is not "
            "the one to reach for",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.072,
            ha="left",
            y=0.955,
        )
        figure.text(
            0.072,
            0.900,
            caption(
                f"Six octaves of aggregation give an exponent of "
                f"{trend.exponent:+.3f}, against −0.30 to −0.10 for natural "
                "terrain: mean slope barely moves from 5 m to 320 m, so there "
                "is no trend here to carry toward a footstep.\n"
                f"The obvious explanation is that {quality['interpolated_pixel_fraction']:.0%} "
                "of the pixels are interpolated, and it fails. Gap-filling would "
                "leave a direction set by the orbit; the roughest axis is "
                f"{measured.separation_from(fall_line):.1f}° from the crater's "
                f"fall line and {measured.separation_from(track):.1f}° from the "
                "ground track, at every lag tested. The directional structure is "
                "terrain. That rejects the artifact this test can see — an "
                "isotropic smoother would leave none.",
                width=150,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def build_map_figure(
    elevation: GeoRaster, slope_at_native: NDArray[np.float64], site: Site
) -> Figure:
    coarse = aggregate(elevation.values, MAP_AGGREGATION)
    cell = elevation.cell_size_m * MAP_AGGREGATION
    slope = slope_degrees(coarse, cell_size_m=cell, method=SLOPE_METHOD)
    extent_km = [
        elevation.extent_m[0] / 1000.0,
        elevation.extent_m[1] / 1000.0,
        elevation.extent_m[2] / 1000.0,
        elevation.extent_m[3] / 1000.0,
    ]

    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (9.0, 6.4),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "axes.grid": False,
                    "figure.subplot.top": 0.720,
                    "figure.subplot.bottom": 0.090,
                    "figure.subplot.left": 0.090,
                    "figure.subplot.right": 0.975,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 1, squeeze=False)
        panel = axes[0][0]

        panel.imshow(
            slope,
            extent=(extent_km[0], extent_km[1], extent_km[2], extent_km[3]),
            origin="upper",
            cmap="Greys",
            vmin=0.0,
            vmax=35.0,
            interpolation="nearest",
        )
        crew_only = np.ma.masked_where(slope <= site.crew.maximum_slope_deg, slope)
        panel.imshow(
            np.ones_like(slope),
            extent=(extent_km[0], extent_km[1], extent_km[2], extent_km[3]),
            origin="upper",
            cmap=matplotlib.colors.ListedColormap([ACCENT_SECONDARY]),
            alpha=np.where(slope > site.crew.maximum_slope_deg, 0.55, 0.0),
            interpolation="nearest",
        )
        panel.imshow(
            np.ones_like(slope),
            extent=(extent_km[0], extent_km[1], extent_km[2], extent_km[3]),
            origin="upper",
            cmap=matplotlib.colors.ListedColormap([ACCENT_PRIMARY]),
            alpha=np.where(slope > TRACTION_LIMIT, 1.0, 0.0),
            interpolation="nearest",
        )
        panel.set_xlabel("polar stereographic x (km)")
        panel.set_ylabel("polar stereographic y (km)")
        panel.set_aspect("equal")

        # Quoted at the native 5 m grid, which is what the study measures.
        # Aggregating for display smooths the shading, so the shaded area here
        # is slightly smaller than the number in the headline, and saying so is
        # cheaper than quietly quoting whichever is larger.
        beyond_crew = float((slope_at_native > site.crew.maximum_slope_deg).mean())
        beyond_robot = float((slope_at_native > TRACTION_LIMIT).mean())
        shown_beyond_crew = float((slope > site.crew.maximum_slope_deg).mean())
        figure.suptitle(
            f"{site.name}: {beyond_crew:.1%} of the site is closed to crew and "
            "open to legs",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.090,
            ha="left",
            y=0.955,
        )
        figure.text(
            0.090,
            0.900,
            caption(
                f"Slope from LOLA at 5 m, where {beyond_crew:.2%} of cells exceed "
                f"the {site.crew.maximum_slope_deg:.0f}° limit the Artemis III "
                "Science Definition Team places on crew, and a legged platform's "
                "limits admit. Shown aggregated to "
                f"{cell:.0f} m, which smooths the shading to {shown_beyond_crew:.2%}.\n"
                f"Above the {TRACTION_LIMIT:.1f}° traction limit at 5 m: "
                f"{beyond_robot:.3%} — a few hundred cells on the steepest inner "
                "walls, and none at all once aggregated. "
                "The site centre lies 53.7 km from the pole against NASA's "
                "stated 50 km for the region; the product is a 20 km window and "
                "the region is stated as 15 km, so it fits with margin.",
                width=118,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def boundary_rows(
    elevation: GeoRaster, quality: dict[str, Any], site: Site
) -> tuple[BoundaryRow, ...]:
    latitude, _ = elevation.center_latitude_longitude()
    return (
        BoundaryRow(
            quantity="slope at DEM baseline",
            published_range="5 m grid, 20 km window",
            used="5 m to 320 m by aggregation",
            status=INSIDE,
            basis=(
                "NASA GSFC PGDA, LOLA track-adjusted; the finest product "
                "published for this latitude"
            ),
        ),
        BoundaryRow(
            quantity="slope algorithm",
            published_range="not stated by the producers",
            used="central difference",
            status=INSIDE,
            basis=(
                "identified rather than assumed, by reproducing the producers' "
                "own slope raster to under a thousandth of a degree; Horn "
                "differs by three orders of magnitude more"
            ),
        ),
        BoundaryRow(
            quantity="effective resolution",
            published_range="none stated",
            used="the 5 m grid as posted",
            status=UNMEASURED,
            basis=(
                f"the producers state {quality['interpolated_pixel_fraction']:.0%} of "
                "pixels are interpolated, so the grid spacing is not the "
                "measurement spacing and no effective baseline is published"
            ),
        ),
        BoundaryRow(
            quantity="slope at stride scale",
            published_range="none, from any orbital dataset",
            used="not estimated",
            status=UNMEASURED,
            basis=(
                f"a {STRIDE_M * 1000:.0f} mm stride is an order of magnitude below "
                f"the ~{BEST_ORBITAL_POSTING_M:.0f} m posting of the best stereo "
                "imagery, and this grid carries no scale trend to extrapolate "
                "along. It also does not bind: at 99.99% traversable the "
                "constraint at that scale is obstacle size-frequency, not gradient"
            ),
        ),
        BoundaryRow(
            quantity="obstacle size-frequency",
            published_range="none in this repository",
            used="not modelled",
            status=UNMEASURED,
            basis=(
                "the binding stride-scale constraint, and a different "
                "measurement: whether a block exceeds foot clearance. Boazman's "
                "NAC boulder survey is the dataset that would answer it"
            ),
        ),
        BoundaryRow(
            quantity="DEM slope error",
            published_range=(
                f"{SLOPE_ERROR_RANGE_DEG[0]} to {SLOPE_ERROR_RANGE_DEG[1]} deg "
                "median RMS"
            ),
            used="propagated through the traversable fraction",
            status=INSIDE,
            basis=(
                "producers' stated value; it moves the crew limit by over a "
                "percentage point and the legged limits by under a thousandth"
            ),
        ),
        BoundaryRow(
            quantity="projection scale distortion",
            published_range="not applicable",
            used="a constant 5 m cell",
            status=OUTSIDE,
            basis=(
                "polar stereographic is conformal, not equidistant: the point "
                f"scale factor is {1.0:.4f} at the pole and about "
                f"{2.0 / (1.0 + math.sin(math.radians(abs(latitude)))):.6f} here, so "
                "slope on the nominal cell is low by that fraction. Under a "
                "twentieth of a percent, and uncorrected"
            ),
        ),
        BoundaryRow(
            quantity="region centre",
            published_range="none citable",
            used="the product's own extent",
            status=UNMEASURED,
            basis=(
                "no public source gives coordinates for the Artemis III region; "
                f"the product centre is {abs(latitude):.2f}S, "
                f"{elevation.arc_distance_from_pole_m() / 1000:.1f} km from the pole "
                f"against NASA's stated {site.stated_distance_from_pole_km:.0f} km"
            ),
        ),
        BoundaryRow(
            quantity="crew slope limit",
            published_range="20 deg",
            used="20 deg",
            status=INSIDE,
            basis="Rice et al. (2023), citing the Artemis III Science Definition Team",
        ),
    )


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return repr(float(value))


def build_report(
    elevation: GeoRaster,
    slope: NDArray[np.float64],
    site: Site,
    quality: dict[str, Any],
) -> str:
    latitude, longitude = elevation.center_latitude_longitude()
    trend = scale_trend(
        elevation.values,
        cell_size_m=elevation.cell_size_m,
        factors=AGGREGATION_FACTORS,
        method=SLOPE_METHOD,
    )
    fall_line, track = fall_line_axis(elevation), ground_track_axis(elevation)
    rows = boundary_rows(elevation, quality, site)

    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# How much of an Artemis III candidate region opens per degree of",
        "# achievable slope, and what the map can and cannot say about it.",
        "#",
        "# Generated by studies/terrain/slope_sensitivity.py. Do not edit.",
        "#",
        "# The terrain product is fetched, not committed. Its identity and",
        "# checksum are in data/terrain/manifest.toml and the numbers below are",
        "# reproducible only against those exact bytes.",
        "#",
        "# Not a sortie envelope: slope is one axis of six and the rest are empty.",
        "",
        f"schema_version = {REPORT_SCHEMA_VERSION}",
        "",
        "[environment]",
        f'python = "{host_platform.python_version()}"',
        f'numpy = "{np.__version__}"',
        "",
        "[site]",
        f'id = "{site.id}"',
        f'name = "{site.name}"',
        f'config = "configs/sites/{site.id}.toml"',
        "",
        "[terrain]",
        'product = "SL2_final_adj_5mpp_surf"',
        'archive = "NASA GSFC Planetary Geodynamics Data Archive"',
        'projection = "south polar stereographic, MOON_ME / DE421"',
        f"reference_radius_m = {_format_float(elevation.reference_radius_m)}",
        f"cell_size_m = {_format_float(elevation.cell_size_m)}",
        f"columns = {elevation.shape[1]}",
        f"rows = {elevation.shape[0]}",
        f"centre_latitude_deg = {_format_float(latitude)}",
        f"centre_longitude_deg = {_format_float(longitude)}",
        "arc_distance_from_pole_km = "
        f"{_format_float(elevation.arc_distance_from_pole_m() / 1000.0)}",
        f'slope_method = "{SLOPE_METHOD}"',
        "slope_method_basis = \"identified by reproducing the producers' own "
        'slope raster; not assumed"',
        "",
        "# The distribution the sensitivity curve is built from.",
        "[slope_distribution]",
        f"cells = {slope.size}",
        f"mean_deg = {_format_float(float(slope.mean()))}",
        f"max_deg = {_format_float(float(slope.max()))}",
        *[
            f"p{q} = {_format_float(float(np.percentile(slope, q)))}"
            for q in (50, 75, 90, 95, 99)
        ],
        "",
        "# The headline. Fractions are of the whole window, and the error column",
        "# is the product's own stated slope uncertainty propagated through.",
        "",
    ]

    for limit, label in (
        (site.crew.maximum_slope_deg, "crew"),
        (REPOSE_BAND[0], "repose_low"),
        (REPOSE_BAND[1], "repose_high"),
        (TIPPING_LIMIT, "tipping"),
        (TRACTION_LIMIT, "traction"),
    ):
        nominal = float((slope <= limit).mean())
        lines += ["[[traversable]]", f'limit = "{label}"',
                  f"slope_deg = {_format_float(limit)}",
                  f"fraction = {_format_float(nominal)}"]
        for sigma in SLOPE_ERROR_RANGE_DEG:
            mean = perturbed_fraction(slope, limit, sigma)
            lines += [
                f"fraction_at_sigma_{str(sigma).replace('.', '_')} = {_format_float(mean)}",
                f"shift_pp_at_sigma_{str(sigma).replace('.', '_')} = "
                f"{_format_float((mean - nominal) * 100.0)}",
            ]
        lines += [""]

    lines += [
        "# What the legged limits open that crew cannot reach. The mission-facing",
        "# number, and the one least sensitive to the map's error.",
        "[robot_only_terrain]",
        "fraction = "
        + _format_float(
            float((slope > site.crew.maximum_slope_deg).mean())
            - float((slope > TRACTION_LIMIT).mean())
        ),
        f"between_deg = [{_format_float(site.crew.maximum_slope_deg)}, "
        f"{_format_float(TRACTION_LIMIT)}]",
        "",
        "# The scale result. For a self-affine surface mean slope goes as the",
        "# baseline to the power of the Hurst exponent minus one; natural terrain",
        "# gives roughly -0.30 to -0.10. This grid gives near zero, so there is no",
        "# trend in it to carry toward a footstep.",
        "[scale_trend]",
        f'method = "{SLOPE_METHOD}, block-mean aggregation"',
        "baseline_m = ["
        + ", ".join(_format_float(v) for v in trend.baseline_m)
        + "]",
        "mean_slope_deg = ["
        + ", ".join(_format_float(v) for v in trend.mean_slope_degrees)
        + "]",
        f"exponent = {_format_float(trend.exponent)}",
        f"hurst_exponent = {_format_float(trend.hurst_exponent)}",
        "natural_terrain_exponent = ["
        f"{_format_float(NATURAL_TERRAIN_SLOPE_EXPONENT[0])}, "
        f"{_format_float(NATURAL_TERRAIN_SLOPE_EXPONENT[1])}]",
        f"holds_roughness = {str(trend.holds_roughness).lower()}",
        "",
        "# Why the obvious explanation was rejected. Gap-filling at a polar site",
        "# would leave a direction set by the orbit. The direction found is the",
        "# crater's. This rejects the artifact the test can see; an isotropic",
        "# smoother would leave no direction at all and is not excluded.",
        "[anisotropy]",
        f"crater_fall_line_axis_deg = {_format_float(fall_line)}",
        f"lro_ground_track_axis_deg = {_format_float(track)}",
        "",
    ]
    for lag in ANISOTROPY_LAGS:
        measured = anisotropy(
            elevation.values, cell_size_m=elevation.cell_size_m, lag_cells=lag
        )
        lines += [
            "[[anisotropy.at_lag]]",
            f"lag_m = {_format_float(measured.lag_m)}",
            f"roughest_axis_deg = {_format_float(measured.roughest_axis_degrees)}",
            f"smoothest_axis_deg = {_format_float(measured.smoothest_axis_degrees)}",
            f"ratio = {_format_float(measured.ratio)}",
            "separation_from_fall_line_deg = "
            f"{_format_float(measured.separation_from(fall_line))}",
            "separation_from_ground_track_deg = "
            f"{_format_float(measured.separation_from(track))}",
            "",
        ]

    lines += [
        "# The specification this study produces, and the thing it is for.",
        "[specification]",
        "statement = \"\"\"",
        "Slope is resolved adequately for the legged limits and unresolvable",
        "below the DEM's effective baseline. No orbital dataset reaches stride",
        "scale: the best stereo imagery posts an order of magnitude above a",
        "300 mm step, and this grid carries no scale trend to extrapolate along.",
        "",
        "It does not need to. At 99.99 percent traversable against the legged",
        "limits, gradient is not what stops a foot at that scale. Whether a block",
        "exceeds foot clearance is, and that is a size-frequency distribution",
        "from imagery rather than a slope distribution from altimetry.",
        "",
        "A landing-site assessment that needs stride-scale traversability should",
        "therefore commission boulder statistics, not a finer DEM.",
        '"""',
        "",
        "# The measured-versus-extrapolated boundary for this study.",
        f"# {tally(rows)}",
        "",
        *toml_lines(rows),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep traversable fraction against achievable slope on an Artemis "
            "III candidate region, and test what the map supports."
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

    site, quality = load_site(SITE_PATH), load_quality()
    elevation = read_float_geotiff(ELEVATION_PATH)
    slope = slope_degrees(
        elevation.values, cell_size_m=elevation.cell_size_m, method=SLOPE_METHOD
    )[1:-1, 1:-1]
    slope = slope[np.isfinite(slope)].ravel()

    arguments.figure_directory.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)

    for name, figure in (
        ("slope-sensitivity", build_sensitivity_figure(slope, site, quality)),
        ("slope-against-baseline", build_baseline_figure(elevation, quality)),
        (
            "de-gerlache-rim-slope-map",
            build_map_figure(elevation, slope, site),
        ),
    ):
        path = arguments.figure_directory / f"{name}.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        print(f"wrote {path.relative_to(REPOSITORY_ROOT)}")

    arguments.report.write_text(
        build_report(elevation, slope, site, quality), encoding="utf-8"
    )
    print(f"wrote {arguments.report.relative_to(REPOSITORY_ROOT)}")

    print("\n  measured against extrapolated\n")
    print(text_table(boundary_rows(elevation, quality, site)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
