# SPDX-License-Identifier: Apache-2.0
#
# studies.mobility.wheels — the wheel against the leg, on the same soil.
#
# Fifteen days of this project measured a legged platform against a suited
# human. Nobody is proposing to walk astronauts into cold traps, so that was
# never the competitor. VIPER is wheeled, the Lunar Terrain Vehicle is wheeled,
# every Mars rover that has driven is wheeled, and no space agency has committed
# a legged machine to flight. This study runs the comparison that decides
# whether the legged case is worth making at all.
#
# The expectation going in was that wheels win outright, and the reasoning was
# sound: Day 3 found swing work is the largest term at lunar gravity precisely
# because it does not scale with weight, and a wheel pays none of it. Days 12
# and 13 found that reaching polar cold traps is not slope-limited, and slope is
# where legs are supposed to earn their keep.
#
# That is not what the physics returns, and the reason is a term the legged
# model deliberately does not pay. A placed foot presses its hole vertically and
# the leg pays that work downward, where it never enters the traction balance. A
# rolling wheel makes new rut continuously and forward motion pays for every
# metre of it. So the wheel deletes the leg's largest flat-ground term and
# acquires one of its own, and which is bigger depends entirely on gradient.
#
# The wheel's second cost is worse and less obvious. Shear under a wheel builds
# from zero at the entry of the contact patch to its maximum at the bottom, so
# the fraction of the soil's strength that any given slip mobilises is an
# average over the patch rather than a point value. On a 0.25 m wheel in lunar
# regolith the patch is 45 mm long against a shear deformation modulus of 18 mm,
# so even at the slip ceiling the wheel develops about two thirds of the shear
# available to it. It cannot reach its own traction limit at any slip. That is
# not a modelling artifact; it is why Spirit died in soft soil and why rover
# operational slope limits sit far below their static ones.
#
# There is an irony worth stating plainly. Bekker built this model for vehicle
# mobility. KLS-1, GRC-1 and every plate-sinkage campaign this repository has
# transcribed was run to characterise wheels. Fifteen days of terramechanics
# calibrated from wheel literature and applied to feet, and the wheel is the
# case the same data supports on most of the ground.
#
# What this study is not: a rover design, an argument about legged robotics in
# general, and above all not a reliability comparison. Twelve or more actuators
# in abrasive cryogenic dust against four to six is the strongest argument
# against legs that exists, and nothing in this repository models it. It sits in
# the boundary table beside a result that happens to favour legs on slope, which
# is the only honest place for it.
#
# References
#   Bekker MG (1956) Theory of Land Locomotion. University of Michigan Press.
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
from eclipse.io.platform import load_platform, load_wheeled_platform  # noqa: E402
from eclipse.io.soil import (  # noqa: E402
    janosi_hanamoto_model,
    load_soil,
    mohr_coulomb_model,
)
from eclipse.mobility import cost_of_transport  # noqa: E402
from eclipse.platform import (  # noqa: E402
    Platform,
    equilibrium_slip_ratio,
    maximum_traversable_slope_degrees,
    swing_work_per_meter,
)
from eclipse.rolling import (  # noqa: E402
    WheeledPlatform,
    rolling_cost_of_transport,
    wheel_equilibrium_slip_ratio,
    wheel_maximum_traversable_slope_degrees,
)
from eclipse.stance import wave_gait, within_stride_slip_ratio  # noqa: E402

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
QUADRUPED_PATH: Final = (
    REPOSITORY_ROOT / "configs" / "platforms" / "nominal-quadruped.toml"
)
ROVER_PATH: Final = REPOSITORY_ROOT / "configs" / "platforms" / "nominal-rover.toml"
SOIL_PATH: Final = REPOSITORY_ROOT / "data" / "soils" / "lunar-intercrater.toml"
FIGURE_DIRECTORY: Final = Path(__file__).resolve().parent / "figures"
DEFAULT_REPORT_PATH: Final = (
    Path(__file__).resolve().parent / "results" / "wheels.toml"
)

REPORT_SCHEMA_VERSION: Final = 1
LUNAR_GRAVITY: Final = 1.62
FEET_IN_STANCE: Final = 3

SLOPE_DEG: Final[NDArray[np.float64]] = np.arange(0.0, 45.001, 0.05)
DECOMPOSITION_SLOPE_DEG: Final = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0)

# The wheel's most consequential parameter and the one the nominal rover is
# least generous about. Swept because a reader is entitled to ask whether the
# result survives a bigger wheel, and because the answer is that it does.
DIAMETER_SWEEP_M: Final = (0.20, 0.25, 0.35, 0.50, 0.75)


def caption(text: str, width: int = 150) -> str:
    return "\n".join(
        textwrap.fill(" ".join(paragraph.split()), width=width)
        for paragraph in text.split("\n")
    )


@dataclass(frozen=True, slots=True)
class Curve:
    """Cost per metre against slope for one platform, split by term."""

    label: str
    slope_deg: NDArray[np.float64]
    gravitational_J_per_m: NDArray[np.float64]
    shear_J_per_m: NDArray[np.float64]
    compaction_J_per_m: NDArray[np.float64]
    swing_J_per_m: NDArray[np.float64]
    slip_ratio: NDArray[np.float64]
    traction_limit_deg: float
    tipping_limit_deg: float

    @property
    def total_J_per_m(self) -> NDArray[np.float64]:
        return np.asarray(
            self.gravitational_J_per_m
            + self.shear_J_per_m
            + self.compaction_J_per_m
            + self.swing_J_per_m
        )

    @property
    def holdable(self) -> NDArray[np.bool_]:
        """Where the platform can actually stand, which is not where the cost
        happens to be finite. The quadruped's traction runs to 43.5 degrees and
        it tips at 39.8, so a curve drawn to the traction limit shows four
        degrees of ground the machine has already fallen over on."""
        return np.asarray(
            np.isfinite(self.total_J_per_m)
            & (np.abs(self.slope_deg) <= self.limit_deg)
        )

    def masked(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(np.where(self.holdable, values, np.nan))

    @property
    def limit_deg(self) -> float:
        """Whichever binds first, which is not the same one for both bodies."""
        return min(self.traction_limit_deg, self.tipping_limit_deg)

    @property
    def binding_limit(self) -> str:
        return (
            "traction"
            if self.traction_limit_deg < self.tipping_limit_deg
            else "tipping"
        )

    def at(self, slope_deg: float) -> float:
        if slope_deg > self.limit_deg:
            return math.inf
        return float(np.interp(slope_deg, self.slope_deg, self.total_J_per_m))


def legged_curve(
    *,
    platform: Platform,
    contact_model: Any,
    strength: Any,
    mobilization: Any,
) -> Curve:
    flat, _ = within_stride_slip_ratio(
        platform=platform,
        gait=wave_gait(lift_order=(2, 0, 3, 1), duty_factor=0.75),
        strength=strength,
        mobilization=mobilization,
        gravity_m_per_s2=LUNAR_GRAVITY,
    )
    traction = maximum_traversable_slope_degrees(
        platform=platform, strength=strength, gravity_m_per_s2=LUNAR_GRAVITY
    )
    tipping = math.degrees(
        math.atan2(
            max(foot.x_m for foot in platform.footprint),
            platform.center_of_mass_height_m,
        )
    )

    terms = {name: np.full(SLOPE_DEG.shape, np.nan) for name in
             ("gravitational", "shear", "compaction", "swing")}
    slip = np.full(SLOPE_DEG.shape, np.nan)
    for index, slope in enumerate(SLOPE_DEG):
        demanded = equilibrium_slip_ratio(
            platform=platform,
            strength=strength,
            mobilization=mobilization,
            gravity_m_per_s2=LUNAR_GRAVITY,
            slope_degrees=float(slope),
        )
        ratio = max(float(demanded), flat)
        if not math.isfinite(ratio) or ratio >= 1.0:
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
            contact_model=contact_model,
            strength=strength,
            mobilization=mobilization,
            swing_work_per_meter_J=swing,
        )
        slip[index] = ratio
        terms["gravitational"][index] = float(cost.gravitational_J_per_m)
        terms["shear"][index] = float(cost.shear_J_per_m)
        terms["compaction"][index] = float(cost.compaction_J_per_m)
        terms["swing"][index] = float(cost.swing_J_per_m)

    return Curve(
        label="legged",
        slope_deg=SLOPE_DEG,
        gravitational_J_per_m=terms["gravitational"],
        shear_J_per_m=terms["shear"],
        compaction_J_per_m=terms["compaction"],
        swing_J_per_m=terms["swing"],
        slip_ratio=slip,
        traction_limit_deg=traction,
        tipping_limit_deg=tipping,
    )


def wheeled_curve(
    *,
    platform: WheeledPlatform,
    contact_model: Any,
    strength: Any,
    mobilization: Any,
) -> Curve:
    slip = wheel_equilibrium_slip_ratio(
        platform=platform,
        contact_model=contact_model,
        strength=strength,
        mobilization=mobilization,
        gravity_m_per_s2=LUNAR_GRAVITY,
        slope_degrees=SLOPE_DEG,
    )
    usable = np.isfinite(slip) & (slip < 1.0)
    safe = np.where(usable, slip, 0.0)
    cost = rolling_cost_of_transport(
        platform=platform,
        contact_model=contact_model,
        strength=strength,
        mobilization=mobilization,
        gravity_m_per_s2=LUNAR_GRAVITY,
        slope_degrees=SLOPE_DEG,
        slip_ratio=safe,
    )
    blank = np.where(usable, 1.0, np.nan)
    return Curve(
        label="wheeled",
        slope_deg=SLOPE_DEG,
        gravitational_J_per_m=cost.gravitational_J_per_m * blank,
        shear_J_per_m=cost.shear_J_per_m * blank,
        compaction_J_per_m=cost.compaction_J_per_m * blank,
        swing_J_per_m=cost.swing_J_per_m * blank,
        slip_ratio=np.where(usable, slip, np.inf),
        traction_limit_deg=wheel_maximum_traversable_slope_degrees(
            platform=platform,
            contact_model=contact_model,
            strength=strength,
            mobilization=mobilization,
            gravity_m_per_s2=LUNAR_GRAVITY,
        ),
        tipping_limit_deg=platform.tipping_slope_degrees,
    )


def crossover_slope_deg(wheel: Curve, leg: Curve) -> float | None:
    """The gradient at which the wheel's advantage runs out.

    Interpolated on the difference rather than read off a grid, so the answer
    does not inherit the slope spacing. Returns None if one platform is cheaper
    everywhere both can go, which is the outcome that would settle the question
    rather than complicate it.
    """
    usable = np.isfinite(wheel.total_J_per_m) & np.isfinite(leg.total_J_per_m)
    usable &= np.abs(SLOPE_DEG) <= min(wheel.limit_deg, leg.limit_deg)
    if not bool(usable.any()):
        return None
    difference = (wheel.total_J_per_m - leg.total_J_per_m)[usable]
    slope = SLOPE_DEG[usable]
    sign_change = np.flatnonzero(np.diff(np.sign(difference)) != 0.0)
    if sign_change.size == 0:
        return None
    index = int(sign_change[0])
    low, high = slope[index], slope[index + 1]
    left, right = difference[index], difference[index + 1]
    return float(low + (high - low) * (-left) / (right - left))


@dataclass(frozen=True, slots=True)
class DiameterPoint:
    diameter_m: float
    level_J_per_m: float
    traction_limit_deg: float
    crossover_deg: float | None


def sweep_diameter(
    *,
    rover: WheeledPlatform,
    leg: Curve,
    contact_model: Any,
    strength: Any,
    mobilization: Any,
) -> tuple[DiameterPoint, ...]:
    points = []
    for diameter in DIAMETER_SWEEP_M:
        variant = WheeledPlatform(
            body_mass_kg=rover.body_mass_kg,
            wheel_mass_kg=rover.wheel_mass_kg,
            wheels=rover.wheels,
            wheel_diameter_m=diameter,
            wheel_width_m=rover.wheel_width_m,
            wheelbase_m=rover.wheelbase_m,
            track_width_m=rover.track_width_m,
            center_of_mass_height_m=rover.center_of_mass_height_m,
            nominal_speed_m_per_s=rover.nominal_speed_m_per_s,
        )
        curve = wheeled_curve(
            platform=variant,
            contact_model=contact_model,
            strength=strength,
            mobilization=mobilization,
        )
        points.append(
            DiameterPoint(
                diameter_m=diameter,
                level_J_per_m=curve.at(0.0),
                traction_limit_deg=curve.traction_limit_deg,
                crossover_deg=crossover_slope_deg(curve, leg),
            )
        )
    return tuple(points)


def build_cost_figure(
    wheel: Curve, leg: Curve, crossover: float | None, sweep: tuple[DiameterPoint, ...]
) -> Figure:
    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (13.4, 6.6),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.585,
                    "figure.subplot.bottom": 0.108,
                    "figure.subplot.left": 0.055,
                    "figure.subplot.right": 0.988,
                    "figure.subplot.wspace": 0.235,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 3, squeeze=False)
        totals, legs, wheels = axes[0][0], axes[0][1], axes[0][2]

        totals.plot(
            leg.slope_deg,
            leg.masked(leg.total_J_per_m),
            color=ACCENT_PRIMARY,
            linewidth=2.0,
            label="legged, 50 kg",
        )
        totals.plot(
            wheel.slope_deg,
            wheel.masked(wheel.total_J_per_m),
            color=ACCENT_SECONDARY,
            linewidth=2.0,
            label="wheeled, 50 kg",
        )
        if crossover is not None:
            totals.axvline(
                crossover, color=INK_PRIMARY, linewidth=1.0, linestyle=(0, (2, 2))
            )
            totals.annotate(
                f"crossover {crossover:.1f}°",
                xy=(crossover, leg.at(crossover)),
                xytext=(8, 22),
                textcoords="offset points",
                color=INK_PRIMARY,
                fontsize=8.5,
            )
        for curve, color, anchor, offset, align in (
            (wheel, ACCENT_SECONDARY, 0.0, (6, 12), "left"),
            (leg, ACCENT_PRIMARY, 118.0, (-6, -2), "right"),
        ):
            totals.axvline(
                curve.limit_deg, color=color, linewidth=1.0, linestyle=(0, (1, 2))
            )
            totals.annotate(
                f"{curve.label} stops at {curve.limit_deg:.1f}°, "
                f"on {curve.binding_limit}",
                xy=(curve.limit_deg, anchor),
                xytext=offset,
                textcoords="offset points",
                color=color,
                fontsize=8.0,
                ha=align,
                va="top" if curve is leg else "bottom",
            )
        totals.set_ylim(0.0, 120.0)
        totals.set_xlim(0.0, 45.0)
        totals.set_xlabel("slope climbed (°)")
        totals.set_ylabel("energy per metre (J/m)")
        totals.set_title(
            "cost per metre, both platforms, one soil",
            color=INK_SECONDARY,
            loc="left",
        )
        totals.legend(loc="upper left")

        for panel, curve, title in (
            (legs, leg, "legged, by term"),
            (wheels, wheel, "wheeled, by term"),
        ):
            holdable = curve.holdable
            panel.stackplot(
                curve.slope_deg[holdable],
                np.nan_to_num(curve.swing_J_per_m[holdable]),
                np.nan_to_num(curve.compaction_J_per_m[holdable]),
                np.nan_to_num(curve.shear_J_per_m[holdable]),
                np.nan_to_num(curve.gravitational_J_per_m[holdable]),
                labels=["swing", "compaction", "shear (slip)", "gravitational"],
                colors=["#9db7dd", "#c9a227", "#d4570a", "#8a8880"],
            )
            panel.set_ylim(0.0, 120.0)
            panel.set_xlim(0.0, 45.0)
            panel.set_xlabel("slope climbed (°)")
            panel.set_ylabel("energy per metre (J/m)")
            panel.set_title(title, color=INK_SECONDARY, loc="left")
            panel.legend(loc="upper left")

        for panel in (totals, legs, wheels):
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)

        level_ratio = leg.at(0.0) / wheel.at(0.0)
        biggest = sweep[-1]
        figure.suptitle(
            "The wheel is cheaper on the flat and cannot follow the leg uphill"
            + (f" — they cross at {crossover:.1f}°" if crossover else ""),
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.055,
            ha="left",
            y=0.962,
        )
        figure.text(
            0.055,
            0.905,
            caption(
                f"Both platforms at 50.0 kg on the same soil at lunar gravity, "
                f"with the same 39.8° tipping limit by construction, so the "
                f"comparison is locomotion and not size or stance. On level "
                f"ground the wheel costs {wheel.at(0.0):.1f} J/m against the "
                f"leg's {leg.at(0.0):.1f}, a factor of {level_ratio:.2f}, and "
                "the middle two panels say why: the leg's bill is swing work it "
                "pays whatever the ground does, and the wheel simply does not "
                "have that term.\n"
                "It buys the saving by taking on the term the legged model "
                "refuses. A placed foot presses its hole vertically and the leg "
                "pays that work downward, out of the traction balance entirely; "
                "a rolling wheel cuts new rut every metre and must drag itself "
                "out of it before it climbs anything. So the wheel slips "
                f"{float(wheel.slip_ratio[0]):.0%} on dead level ground where "
                "the foot slips none, and that slip is multiplicative: every "
                "joule of resistance and of climb is divided by one minus it.\n"
                f"Past {crossover:.1f}° that division runs away. The wheel's "
                "contact patch is 45 mm long against a shear deformation "
                "modulus of 18 mm, so the soil under it is still gripping at the "
                "entry while it has already slid at the bottom, and even at the "
                "slip ceiling only about two thirds of the available shear is "
                f"developed. The rover therefore stops at "
                f"{wheel.limit_deg:.1f}° on traction while the quadruped runs to "
                f"{leg.limit_deg:.1f}° and stops by tipping over instead. "
                f"A bigger wheel helps and does not rescue it: at "
                f"{biggest.diameter_m:.2f} m the limit is still "
                f"{biggest.traction_limit_deg:.1f}°.",
                width=178,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def boundary_rows(
    wheel: Curve, leg: Curve, crossover: float | None
) -> tuple[BoundaryRow, ...]:
    return (
        BoundaryRow(
            quantity="compaction resistance",
            published_range="Bekker rigid-wheel form, Wong (2008) ch. 2",
            used="pressure integrated through depth over the rut width",
            status=INSIDE,
            basis=(
                "the integral form and Wong's closed form are independent routes "
                "to the same number and agree to machine precision across four "
                "sinkage exponents, three loads and three wheel geometries"
            ),
        ),
        BoundaryRow(
            quantity="reliability",
            published_range="none, and this is the wheel's strongest argument",
            used="absent entirely",
            status=UNMEASURED,
            basis=(
                "twelve or more actuators in abrasive cryogenic dust against "
                "four to six, with no redundancy analysis, no dust ingress "
                "model, no duty-cycle life and no failure modes on either "
                "platform. Nothing in this repository models it, and it sits "
                "beside a slope result that favours legs. A reader who weighted "
                "it would be right to"
            ),
        ),
        BoundaryRow(
            quantity="obstacles",
            published_range="none at 5 m posting",
            used="absent; both platforms cross a smooth surface",
            status=OUTSIDE,
            basis=(
                "a boulder above wheel diameter stops a rover and a legged "
                "platform steps over it, which is the mechanism by which legs "
                "are supposed to win. There is no boulder statistic at the scale "
                "that would show it, so the comparison here is run on ground "
                "smoother than either machine would meet"
            ),
        ),
        BoundaryRow(
            quantity="grousers",
            published_range="standard on every flown lunar wheel",
            used="none; a smooth rigid wheel",
            status=OUTSIDE,
            basis=(
                "grousers raise developed thrust substantially and are the first "
                "thing a real lunar wheel has. Omitting them understates the "
                "rover's slope limit, so the traction limit reported here is a "
                "floor rather than an estimate"
            ),
        ),
        BoundaryRow(
            quantity="wheel diameter",
            published_range="0.25 m assumed; 0.20 to 0.75 m swept",
            used="0.25 m on a 50 kg platform",
            status=UNMEASURED,
            basis=(
                "the single most consequential rover parameter, because diameter "
                "sets contact length and contact length sets how much shear any "
                "slip can mobilise. Swept because a reader is entitled to ask "
                "whether a bigger wheel overturns the result"
            ),
        ),
        BoundaryRow(
            quantity="rigid wheel",
            published_range="Bekker's form assumes it",
            used="rigid, no deflection",
            status=OUTSIDE,
            basis=(
                "a compliant or mesh wheel lengthens the contact patch at the "
                "same load, which is precisely the quantity the rover is short "
                "of here. This is the assumption most likely to be understating "
                "the wheel and it is not swept"
            ),
        ),
        BoundaryRow(
            quantity="steering and turning",
            published_range="none",
            used="absent; straight-line cost only",
            status=UNMEASURED,
            basis=(
                "skid steering on a four-wheeled rover is expensive and a legged "
                "platform turns nearly free. Neither is modelled, and the "
                "reachable-set search that consumes this curve turns constantly"
            ),
        ),
        BoundaryRow(
            quantity="crossover slope",
            published_range="not applicable",
            used=(
                f"{crossover:.2f} degrees" if crossover is not None else "none found"
            ),
            status=OUTSIDE,
            basis=(
                "an equality between two assumed platforms on one soil, so it "
                "carries every assumption in both files. It is reported because "
                "its existence is the finding, not because its value is precise"
            ),
        ),
        BoundaryRow(
            quantity="hardware",
            published_range="none",
            used="none; neither platform exists",
            status=UNMEASURED,
            basis=(
                "no wheel has been rolled, no foot has been placed, and no "
                "result here has been compared against a machine"
            ),
        ),
    )


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return repr(float(value))


def build_report(
    wheel: Curve,
    leg: Curve,
    crossover: float | None,
    sweep: tuple[DiameterPoint, ...],
) -> str:
    rows = boundary_rows(wheel, leg, crossover)
    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# studies.mobility.wheels — the wheel against the leg on one soil.",
        "#",
        "# Generated. Do not edit by hand.",
        "",
        f"schema_version = {REPORT_SCHEMA_VERSION}",
        'study = "wheels"',
        f"gravity_m_per_s2 = {_format_float(LUNAR_GRAVITY)}",
        'soil = "lunar-intercrater/carrier1991"',
        "",
        "[platforms]",
        'legged = "nominal-quadruped"',
        'wheeled = "nominal-rover"',
        f"mass_kg = {_format_float(leg_mass := 50.0)}",
        "matched = \"mass and tipping limit, so the comparison is locomotion\"",
        "",
        "[level_ground]",
        f"legged_J_per_m = {_format_float(leg.at(0.0))}",
        f"wheeled_J_per_m = {_format_float(wheel.at(0.0))}",
        f"wheel_advantage = {_format_float(leg.at(0.0) / wheel.at(0.0))}",
        f"legged_swing_J_per_m = {_format_float(float(leg.swing_J_per_m[0]))}",
        "wheeled_compaction_J_per_m = "
        + _format_float(float(wheel.compaction_J_per_m[0])),
        f"legged_slip_ratio = {_format_float(float(leg.slip_ratio[0]))}",
        f"wheeled_slip_ratio = {_format_float(float(wheel.slip_ratio[0]))}",
        "",
        "[crossover]",
        "slope_deg = "
        + (_format_float(crossover) if crossover is not None else "nan"),
        "exists = " + str(crossover is not None).lower(),
        'meaning = "below it the wheel is cheaper, above it the leg is"',
        "",
        "[limits]",
        f"legged_traction_deg = {_format_float(leg.traction_limit_deg)}",
        f"legged_tipping_deg = {_format_float(leg.tipping_limit_deg)}",
        f'legged_binding = "{leg.binding_limit}"',
        f"wheeled_traction_deg = {_format_float(wheel.traction_limit_deg)}",
        f"wheeled_tipping_deg = {_format_float(wheel.tipping_limit_deg)}",
        f'wheeled_binding = "{wheel.binding_limit}"',
        "",
        "# Cost per metre at slopes both platforms can hold.",
        "[[sample]]",
    ]
    del leg_mass
    blocks = []
    for slope in DECOMPOSITION_SLOPE_DEG:
        blocks.append(
            [
                "[[sample]]",
                f"slope_deg = {_format_float(float(slope))}",
                f"legged_J_per_m = {_format_float(leg.at(float(slope)))}",
                f"wheeled_J_per_m = {_format_float(wheel.at(float(slope)))}",
                "legged_slip_ratio = "
                + _format_float(
                    float(np.interp(slope, leg.slope_deg, leg.slip_ratio))
                ),
                "wheeled_slip_ratio = "
                + _format_float(
                    float(np.interp(slope, wheel.slope_deg, wheel.slip_ratio))
                ),
                "",
            ]
        )
    lines = lines[:-1]
    for block in blocks:
        lines.extend(block)

    lines.append("# Does a bigger wheel overturn the result.")
    for point in sweep:
        lines.extend(
            [
                "[[diameter]]",
                f"diameter_m = {_format_float(point.diameter_m)}",
                f"level_J_per_m = {_format_float(point.level_J_per_m)}",
                f"traction_limit_deg = {_format_float(point.traction_limit_deg)}",
                "crossover_deg = "
                + (
                    _format_float(point.crossover_deg)
                    if point.crossover_deg is not None
                    else "nan"
                ),
                "",
            ]
        )

    lines.extend(toml_lines(rows))
    lines.extend(["", "[summary]", 'text = """'])
    lines.extend(
        caption(
            "On mapped polar terrain at 5 m posting, with no obstacles, the "
            "wheeled platform is the cheaper machine on most of the ground and "
            "the legged one is the only machine on the rest. Those are not the "
            "same claim and the study exists to keep them apart.\n"
            "\n"
            f"On level ground the rover costs {wheel.at(0.0):.1f} J/m against "
            f"the quadruped's {leg.at(0.0):.1f}, an advantage of "
            f"{leg.at(0.0) / wheel.at(0.0):.2f} times, and the reason is a term "
            "rather than a refinement: swing work is "
            f"{float(leg.swing_J_per_m[0]) / leg.at(0.0):.0%} of what the leg "
            "spends standing still and moving, and a wheel does not pay it. "
            "That is the wheel's whole case and it is a strong one.\n"
            "\n"
            f"The case ends at {crossover:.1f} degrees. Above it the wheel is "
            "more expensive and the gap widens fast, because the resistance it "
            "must overcome is divided by one minus its slip and its slip is "
            f"climbing: {float(np.interp(20.0, wheel.slope_deg, wheel.slip_ratio)):.0%} "
            "at twenty degrees against the foot's "
            f"{float(np.interp(20.0, leg.slope_deg, leg.slip_ratio)):.0%}. At "
            f"{wheel.limit_deg:.1f} degrees the rover stops entirely, on "
            "traction, and the quadruped continues to "
            f"{leg.limit_deg:.1f} where it stops by tipping over.\n"
            "\n"
            "The mechanism behind that is worth stating precisely because it is "
            "not the one intuition offers. It is not that the wheel is short of "
            "friction. It is that shear under a wheel builds from zero at the "
            "entry of the contact patch to its maximum at the bottom, so the "
            "mobilised fraction is an average over the patch. On a 0.25 m wheel "
            "in this regolith the patch is 45 mm long against a shear "
            "deformation modulus of 18 mm, and even at the slip ceiling only "
            "about two thirds of the available shear is developed. The rover "
            "cannot reach its own traction limit at any slip. This is why rover "
            "operational slope limits sit well below their static ones, and it "
            "is the same physics that stopped Spirit.\n"
            "\n"
            "A bigger wheel is the obvious answer and it helps without "
            "rescuing: over 0.20 to 0.75 m the level cost falls by "
            f"{sweep[0].level_J_per_m / sweep[-1].level_J_per_m:.2f} times and "
            f"the traction limit rises from {sweep[0].traction_limit_deg:.1f} to "
            f"{sweep[-1].traction_limit_deg:.1f} degrees, still short of the "
            f"quadruped's {leg.limit_deg:.1f}. A 0.75 m wheel on a 50 kg vehicle "
            "is also not a rover.\n"
            "\n"
            "What would have to be true for the legged case to close is now a "
            "specification rather than a hope, and none of it is in this "
            "repository: boulder statistics at metre scale, which is the "
            "mechanism by which legs actually win and the fifth time this "
            "project has asked for the same measurement; a polar pit, when no "
            "archive holds one and the median catalogued pit is three cells "
            "across at 5 m; and slope past the angle of repose, which does not "
            "occur on the routes these errands need.\n"
            "\n"
            "And the argument that runs the other way is the strongest one on "
            "the table. Twelve or more actuators in abrasive cryogenic dust "
            "against four to six is why every funded programme is wheeled, and "
            "nothing here models it. A slope result that favours legs is worth "
            "very little beside it, and saying so is not modesty but the "
            "condition for the rest being believed.",
            width=74,
        ).split("\n")
    )
    lines.extend(['"""', "", "[boundary]", f'tally = "{tally(rows)}"', ""])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a wheeled and a legged platform on the same soil."
    )
    parser.add_argument("--figure-directory", type=Path, default=FIGURE_DIRECTORY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    arguments = parser.parse_args(argv)

    dataset = load_soil(SOIL_PATH).datasets["carrier1991"]
    contact_model = dataset.models["bekker"].extrapolating
    strength = mohr_coulomb_model(dataset, depth_range_cm="0-15")
    mobilization = janosi_hanamoto_model(dataset)

    quadruped = load_platform(QUADRUPED_PATH).platform
    rover = load_wheeled_platform(ROVER_PATH).platform

    leg = legged_curve(
        platform=quadruped,
        contact_model=contact_model,
        strength=strength,
        mobilization=mobilization,
    )
    wheel = wheeled_curve(
        platform=rover,
        contact_model=contact_model,
        strength=strength,
        mobilization=mobilization,
    )
    crossover = crossover_slope_deg(wheel, leg)
    sweep = sweep_diameter(
        rover=rover,
        leg=leg,
        contact_model=contact_model,
        strength=strength,
        mobilization=mobilization,
    )

    print(
        f"  level ground   legged {leg.at(0.0):6.2f} J/m   "
        f"wheeled {wheel.at(0.0):6.2f} J/m   "
        f"wheel advantage {leg.at(0.0) / wheel.at(0.0):.2f}x"
    )
    print(
        f"  crossover      "
        + (f"{crossover:.2f}°" if crossover is not None else "none")
        + f"   legged stops {leg.limit_deg:.1f}° ({leg.binding_limit})"
        f"   wheeled stops {wheel.limit_deg:.1f}° ({wheel.binding_limit})"
    )
    for point in sweep:
        print(
            f"  wheel {point.diameter_m:.2f} m   level {point.level_J_per_m:6.2f} J/m"
            f"   traction limit {point.traction_limit_deg:5.2f}°"
            + (
                f"   crossover {point.crossover_deg:5.2f}°"
                if point.crossover_deg is not None
                else "   crossover none"
            )
        )

    arguments.figure_directory.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    figure = build_cost_figure(wheel, leg, crossover, sweep)
    path = arguments.figure_directory / "wheel-against-leg.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    print(f"wrote {path.relative_to(REPOSITORY_ROOT)}")

    arguments.report.write_text(
        build_report(wheel, leg, crossover, sweep), encoding="utf-8"
    )
    print(f"wrote {arguments.report.relative_to(REPOSITORY_ROOT)}")

    print("\n  measured against extrapolated\n")
    print(text_table(boundary_rows(wheel, leg, crossover)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
