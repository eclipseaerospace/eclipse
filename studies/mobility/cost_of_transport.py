# SPDX-License-Identifier: Apache-2.0
#
# studies.mobility.cost_of_transport — how the energy cost of walking splits by
# term, and how each part moves with gravity, slope and soil.
#
# Rung two. The output is a mobility model in the sense the architecture asks
# for: cost per meter as a function of slope, soil state and gait, with nothing
# in it that knows what produced the contact patches. It is not a sortie
# envelope and must not be read as one -- there is no power budget, no thermal
# model, no illumination and no traverse here.
#
# Three things this study exists to separate, because they are usually reported
# as one number:
#
#   the exponents   how each term scales with gravity, which follows from the
#                     model forms and holds whatever the platform is
#   the shares      what fraction of the total each term carries, which depends
#                     on an assumed platform and is therefore much weaker
#   the boundaries  where the published parameters stop supporting the answer
#
# Two inputs are assumed and both are load-bearing. Swing work per meter is the
# obvious one: no leg inertia has been measured or even specified for this
# project, and at lunar gravity it is the largest term. Slip ratio is the quiet
# one, and it sits inside the crossover rather than beside it -- shear work is
# very nearly linear in slip over this range, so halving or doubling the assumed
# 0.10 moves the crossover threshold by about the same factor. Neither number
# has evidence behind it.
#
# So the study reports the crossover rather than a share: the swing cost at
# which swing overtakes everything the soil contributes. It still depends on the
# assumed slip, which is why the sensitivity is reported beside it rather than
# left implicit.
#
# The constant swing cost is also a simplification with a known direction.
# Swinging a leg costs inertial work, which gravity does not touch, plus lifting
# the foot to clear the ground, which scales with gravity. A real leg therefore
# gets somewhat cheaper to swing at one sixth g, so holding the cost flat
# overstates how much the normalized swing term rises. The direction of that
# error runs the same way as the headline, and saying so is the point: the
# finding is that the crossover threshold collapses, which is a property of the
# soil, not that any particular platform crosses it.
#
# Magnitudes are deliberately not compared between soils. GRC-1's plate scaling
# was fitted over half-widths of 38 to 95 mm and does not transfer to a foot,
# so a simulant-versus-Moon stiffness ratio at 30 mm would be an extrapolation
# dressed as a measurement. What does transfer is the sinkage exponent, which is
# fitted per plate from the shape of the curve rather than from the scaling
# across plates, and the compaction exponent depends on nothing else.
#
# The shear models are built here from the raw mappings the loader carries,
# rather than by the loader itself. One soil in this repository has shear
# parameters; the builder moves into eclipse.io.soil at the second, which is
# also when it will be clear what varies between them.
#
# References
#   Carrier WD III, Olhoeft GR, Mendell W (1991) Physical Properties of the
#     Lunar Surface. In: Lunar Sourcebook, ch. 9. Cambridge University Press.
#   Oravec HA (2009) Understanding mechanical behavior of lunar soils for the
#     study of vehicle mobility. PhD thesis, Case Western Reserve University.

from __future__ import annotations

import argparse
import math
import platform
import textwrap
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
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
    SURFACE,
    figure_style,
)
from eclipse.io.soil import Dataset, load_soil
from eclipse.mobility import (
    ContactPatch,
    CostOfTransport,
    compaction_work_per_footfall,
    cost_of_transport,
)
from eclipse.terramechanics import (
    BekkerModel,
    ContactModel,
    JanosiHanamotoModel,
    MohrCoulombModel,
)

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
LUNAR_SOIL_PATH: Final = REPOSITORY_ROOT / "data" / "soils" / "lunar-intercrater.toml"
GRC1_CHANNELS_PATH: Final = (
    REPOSITORY_ROOT / "data" / "literature" / "oravec2009-grc1-raw-channels.toml"
)

FIGURE_DIRECTORY: Final = Path(__file__).resolve().parent / "figures"
DEFAULT_REPORT_PATH: Final = (
    Path(__file__).resolve().parent / "results" / "cost-of-transport.toml"
)

REPORT_SCHEMA_VERSION: Final = 1

# Every value in this block is assumed. None of it is measured, none of it is
# specified by a design, and the report says so beside each one. They are held
# fixed so the sweep varies gravity, slope and soil alone.
MASS_KG: Final = 50.0
FOOT_HALF_WIDTH_M: Final = 0.030
FOOT_AREA_M2: Final = math.pi * FOOT_HALF_WIDTH_M**2
STRIDE_LENGTH_M: Final = 0.30
FEET_IN_STANCE: Final = 2
SLIP_RATIO: Final = 0.10
NOMINAL_SWING_WORK_PER_METER_J: Final = 25.0

EARTH_GRAVITY: Final = 9.81
MARS_GRAVITY: Final = 3.71
LUNAR_GRAVITY: Final = 1.62
GRAVITIES: Final = (
    ("earth", EARTH_GRAVITY),
    ("mars", MARS_GRAVITY),
    ("moon", LUNAR_GRAVITY),
)

SLOPE_DEGREES: Final[NDArray[np.float64]] = np.linspace(0.0, 40.0, 161)
SLIP_RATIO_SWEEP: Final[NDArray[np.float64]] = np.linspace(0.0, 0.40, 161)
GRAVITY_SWEEP: Final[NDArray[np.float64]] = np.linspace(1.0, 10.0, 91)

SHALLOWEST_DEPTH_RANGE: Final = "0-15"
CENTIMETERS_PER_METER: Final = 100.0
MILLIMETERS_PER_METER: Final = 1000.0

TERMS: Final = ("gravitational", "shear", "compaction", "swing")

# The gravitational term is exactly sin(slope) once normalized, identical in
# every panel of a gravity comparison, and large enough to compress everything
# that does differ. The decomposition figure stacks what is left, which is the
# cost of walking over the ideal cost of climbing.
LOCOMOTION_TERMS: Final = ("shear", "compaction", "swing")
TERM_COLORS: Final[dict[str, str]] = {
    "gravitational": INK_MUTED,
    "shear": ACCENT_SECONDARY,
    "compaction": ACCENT_PRIMARY,
    "swing": INK_PRIMARY,
}


def caption(text: str, width: int = 132) -> str:
    """Wrap to a fixed column, keeping explicit line breaks as paragraph breaks.

    Matplotlib does not wrap text placed in figure coordinates, so a caption
    that outgrows the canvas is silently clipped at the right edge rather than
    flagged. Wrapping here is what stops that being discovered by looking at
    the picture.
    """
    return "\n".join(
        textwrap.fill(" ".join(paragraph.split()), width=width)
        for paragraph in text.split("\n")
    )


@dataclass(frozen=True, slots=True)
class SoilUnderFoot:
    contact: ContactModel
    strength: MohrCoulombModel
    mobilization: JanosiHanamotoModel
    sinkage_ceiling_m: float
    half_width_range_m: tuple[float, float]


@dataclass(frozen=True, slots=True)
class ParameterSet:
    id: str
    label: str
    source: str
    contact: ContactModel
    note: str


def _shear_entry(dataset: Dataset, model_id: str) -> dict[str, Any]:
    if dataset.shear_model is None:
        raise ValueError(f"{dataset.id} carries no shear models")
    for entry in dataset.shear_model:
        if entry["id"] == model_id:
            return dict(entry)
    raise ValueError(f"{dataset.id} carries no shear model {model_id!r}")


def load_lunar_soil() -> SoilUnderFoot:
    soil = load_soil(LUNAR_SOIL_PATH)
    dataset = soil.datasets["carrier1991"]
    bekker = dataset.models["bekker"]

    row = next(
        entry
        for entry in _shear_entry(dataset, "mohr_coulomb")["by_depth"]["rows"]
        if entry["depth_range_cm"] == SHALLOWEST_DEPTH_RANGE
    )
    modulus_cm = _shear_entry(dataset, "janosi_hanamoto")["parameters"][
        "shear_deformation_modulus"
    ]["value"]

    return SoilUnderFoot(
        # Deliberately the extrapolating model. The study reports where the
        # published range ends rather than refusing to evaluate past it, and
        # going through the validated wrapper would raise instead of showing it.
        contact=bekker.extrapolating,
        strength=MohrCoulombModel(
            cohesion=row["cohesion_kPa"],
            friction_angle_degrees=row["friction_angle_deg"],
        ),
        mobilization=JanosiHanamotoModel(
            shear_deformation_modulus=modulus_cm / CENTIMETERS_PER_METER
        ),
        sinkage_ceiling_m=bekker.sinkage_validity.max,
        half_width_range_m=(
            bekker.contact_half_width_validity.min,
            bekker.contact_half_width_validity.max,
        ),
    )


def load_parameter_sets(lunar: SoilUnderFoot) -> tuple[ParameterSet, ...]:
    channels = tomllib.loads(GRC1_CHANNELS_PATH.read_text(encoding="utf-8"))
    published = {entry["window"]: entry for entry in channels["verification"]}

    sets = [
        ParameterSet(
            id="lunar-intercrater",
            label="lunar regolith, measured in situ",
            source="Heiken et al. (1991) Table 9.14",
            contact=lunar.contact,
            note="in-situ measured, not a simulant",
        )
    ]
    for window, label in (
        ("lunar_pressure_range", "GRC-1, lunar pressure window"),
        ("entire_pressure_range", "GRC-1, entire pressure window"),
    ):
        entry = published[window]
        sets.append(
            ParameterSet(
                id=f"grc1-{window}",
                label=label,
                source=f"Oravec (2009), {label}",
                contact=BekkerModel(
                    cohesive_modulus=entry["cohesive_modulus"],
                    frictional_modulus=entry["frictional_modulus"],
                    sinkage_exponent=entry["sinkage_exponent"],
                ),
                note="simulant; plate scaling does not transfer to foot scale",
            )
        )
    return tuple(sets)


def walk(
    soil: SoilUnderFoot,
    *,
    gravity: float,
    slope_degrees: Any,
    slip_ratio: Any = SLIP_RATIO,
    swing_work_per_meter_J: float = NOMINAL_SWING_WORK_PER_METER_J,
) -> CostOfTransport:
    return cost_of_transport(
        mass_kg=MASS_KG,
        gravity_m_per_s2=gravity,
        slope_degrees=slope_degrees,
        slip_ratio=slip_ratio,
        patch=ContactPatch(half_width_m=FOOT_HALF_WIDTH_M, area_m2=FOOT_AREA_M2),
        feet_in_stance=FEET_IN_STANCE,
        stride_length_m=STRIDE_LENGTH_M,
        stance_length_m=STRIDE_LENGTH_M,
        contact_model=soil.contact,
        strength=soil.strength,
        mobilization=soil.mobilization,
        swing_work_per_meter_J=swing_work_per_meter_J,
    )


def maximum_traversable_slope_degrees(soil: SoilUnderFoot, gravity: float) -> float:
    """The slope at which demanded traction equals what the patches can carry.

    Demand per foot is the along-slope component of weight; capacity is
    cohesion over the patch area plus friction on the slope-normal component.
    Setting them equal and collecting terms gives
    slope = friction_angle + asin(cohesive_reserve * cos(friction_angle)),
    where the reserve is the cohesive force as a fraction of total weight. The
    reserve goes as one over gravity, so cohesion buys a steeper slope on the
    Moon than on Earth by exactly the gravity ratio.

    This is a foot-slip criterion. It says nothing about whether the slope
    itself is stable, and for these parameters the bulk failure of loose surface
    regolith is the limit that binds first.
    """
    friction_angle = math.radians(soil.strength.friction_angle_degrees)
    cohesive_force_N = soil.strength.cohesion * 1.0e3 * FOOT_AREA_M2 * FEET_IN_STANCE
    reserve = cohesive_force_N / (MASS_KG * gravity)
    argument = reserve * math.cos(friction_angle)
    if argument >= 1.0:
        return 90.0
    return math.degrees(friction_angle + math.asin(argument))


def soil_cost_per_meter(soil: SoilUnderFoot, gravity: float) -> float:
    """What the ground costs per meter on the flat, with no platform in it.

    This is the crossover the study reports in place of a swing share: a
    platform whose swing cost exceeds this carries more of its energy budget in
    its own legs than in the terrain.
    """
    flat = walk(soil, gravity=gravity, slope_degrees=0.0, swing_work_per_meter_J=0.0)
    return float(flat.shear_J_per_m) + float(flat.compaction_J_per_m)


def gravity_exponent(quantity: Any, gravity: float) -> float:
    step = 1.0e-4
    low, high = gravity * (1.0 - step), gravity * (1.0 + step)
    return (math.log(quantity(high)) - math.log(quantity(low))) / (
        math.log(high) - math.log(low)
    )


def normalized_term_exponents(soil: SoilUnderFoot, gravity: float) -> dict[str, float]:
    def normalized(term: str) -> Any:
        def evaluate(value: float) -> float:
            result = walk(soil, gravity=value, slope_degrees=10.0)
            return abs(float(getattr(result, f"{term}_J_per_m"))) / (MASS_KG * value)

        return evaluate

    return {term: gravity_exponent(normalized(term), gravity) for term in TERMS}


def sinkage_at(soil: SoilUnderFoot, gravity: float, slope_degrees: float) -> float:
    patch = ContactPatch(half_width_m=FOOT_HALF_WIDTH_M, area_m2=FOOT_AREA_M2)
    load_N = (
        MASS_KG * gravity * math.cos(math.radians(slope_degrees)) / FEET_IN_STANCE
    )
    return float(
        soil.contact.sinkage(
            pressure=patch.normal_stress_kPa(normal_load_N=load_N),
            contact_half_width=FOOT_HALF_WIDTH_M,
        )
    )


def half_width_reaching_sinkage_ceiling(soil: SoilUnderFoot, gravity: float) -> float:
    """The patch a platform would need for its sinkage to sit on the ceiling.

    Sinkage falls as the patch grows, so this is the smallest patch that keeps
    the bearing model inside the range it was published over. Solved by
    bisection rather than in closed form because the deformation modulus carries
    its own dependence on half-width and the exponent need not be one.
    """

    def sinkage_excess(half_width: float) -> float:
        patch = ContactPatch(
            half_width_m=half_width, area_m2=math.pi * half_width**2
        )
        depth = float(
            soil.contact.sinkage(
                pressure=patch.normal_stress_kPa(
                    normal_load_N=MASS_KG * gravity / FEET_IN_STANCE
                ),
                contact_half_width=half_width,
            )
        )
        return depth - soil.sinkage_ceiling_m

    low, high = 1.0e-4, 1.0
    if sinkage_excess(high) > 0.0:
        return math.inf
    for _ in range(200):
        middle = 0.5 * (low + high)
        if sinkage_excess(middle) > 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def boundary_rows(
    soil: SoilUnderFoot, parameter_sets: tuple[ParameterSet, ...]
) -> tuple[BoundaryRow, ...]:
    ceiling_mm = soil.sinkage_ceiling_m * MILLIMETERS_PER_METER
    rows = [
        BoundaryRow(
            quantity=f"sinkage, {name}",
            published_range=f"0 to {ceiling_mm:.0f} mm",
            used=f"{sinkage_at(soil, gravity, 0.0) * MILLIMETERS_PER_METER:.1f} mm",
            status=(
                INSIDE
                if sinkage_at(soil, gravity, 0.0) <= soil.sinkage_ceiling_m
                else OUTSIDE
            ),
            basis="Heiken et al. (1991) Table 9.14, quasi-static normal bearing",
        )
        for name, gravity in GRAVITIES
    ]
    rows.append(
        BoundaryRow(
            quantity="contact half-width, lunar parameters",
            published_range=(
                f"{soil.half_width_range_m[0] * MILLIMETERS_PER_METER:.0f} to "
                f"{soil.half_width_range_m[1] * MILLIMETERS_PER_METER:.0f} mm"
            ),
            used=f"{FOOT_HALF_WIDTH_M * MILLIMETERS_PER_METER:.0f} mm",
            status=INSIDE,
            basis="Heiken et al. (1991) section 9.1.9, footings under 0.5 m",
        )
    )
    rows.append(
        BoundaryRow(
            quantity="contact half-width, GRC-1 parameters",
            published_range="38 to 95 mm",
            used=f"{FOOT_HALF_WIDTH_M * MILLIMETERS_PER_METER:.0f} mm",
            status=OUTSIDE,
            basis="Oravec (2009), three plates; only the exponent is used here",
        )
    )
    rows.append(
        BoundaryRow(
            quantity="shear strength depth",
            published_range="0 to 15 cm, shallowest row published",
            used=(
                f"{sinkage_at(soil, LUNAR_GRAVITY, 0.0) * MILLIMETERS_PER_METER:.1f}"
                " mm at lunar gravity"
            ),
            status=INSIDE,
            basis="Heiken et al. (1991) Table 9.12",
        )
    )
    rows.append(
        BoundaryRow(
            quantity="shear mobilization under gait",
            published_range="none",
            used="Janosi-Hanamoto at K = 1.8 cm, applied per footfall",
            status=UNMEASURED,
            basis=(
                "K is from in-situ slip observations under steady loading; "
                "repeated loading at gait rates on already-disturbed soil is "
                "measured nowhere in this repository"
            ),
        )
    )
    rows.append(
        BoundaryRow(
            quantity="slip ratio",
            published_range="none",
            used=f"{SLIP_RATIO:.2f}, held fixed",
            status=UNMEASURED,
            basis=(
                "assumed; shear work is very nearly linear in it, so the "
                "crossover threshold moves with it by about the same factor"
            ),
        )
    )
    rows.append(
        BoundaryRow(
            quantity="swing work per meter",
            published_range="none",
            used=f"{NOMINAL_SWING_WORK_PER_METER_J:.0f} J/m, held constant",
            status=UNMEASURED,
            basis=(
                "assumed; a real leg costs inertial work plus foot clearance "
                "against gravity, so a flat value overstates the lunar rise"
            ),
        )
    )
    rows.append(
        BoundaryRow(
            quantity="reduced-gravity granular flow",
            published_range="none",
            used="none; every model here is gravity-independent in form",
            status=UNMEASURED,
            basis=(
                "cannot be validated on Earth at all, and no model in this "
                "study claims to represent it"
            ),
        )
    )
    return tuple(rows)


def compaction_against_gravity(parameters: ParameterSet) -> NDArray[np.float64]:
    """Compaction cost per meter, normalized by weight and by its Earth value.

    Normalizing each curve to its own value at Earth gravity is what makes the
    comparison legitimate: the magnitudes depend on the plate scaling, which
    does not transfer to foot scale, while the slope depends only on the sinkage
    exponent, which is fitted from each curve's own shape.
    """
    patch = ContactPatch(half_width_m=FOOT_HALF_WIDTH_M, area_m2=FOOT_AREA_M2)
    stride_advance_m = STRIDE_LENGTH_M * (1.0 - SLIP_RATIO)
    footfalls_per_meter = FEET_IN_STANCE / stride_advance_m

    loads_N = MASS_KG * GRAVITY_SWEEP / FEET_IN_STANCE
    depths = parameters.contact.sinkage(
        pressure=patch.normal_stress_kPa(normal_load_N=loads_N),
        contact_half_width=FOOT_HALF_WIDTH_M,
    )
    work_J = compaction_work_per_footfall(
        patch=patch, contact_model=parameters.contact, sinkage_m=depths
    )
    normalized = work_J * footfalls_per_meter / (MASS_KG * GRAVITY_SWEEP)
    at_earth = np.interp(EARTH_GRAVITY, GRAVITY_SWEEP, normalized)
    return np.asarray(normalized / at_earth)


def build_slip_figure(soil: SoilUnderFoot) -> Figure:
    one_modulus_m = soil.mobilization.shear_deformation_modulus
    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (10.2, 4.9),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.715,
                    "figure.subplot.bottom": 0.230,
                    "figure.subplot.left": 0.066,
                    "figure.subplot.right": 0.986,
                    "figure.subplot.wspace": 0.215,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 2, squeeze=False)

        left = axes[0][0]
        slide_m = np.linspace(0.0, 0.080, 400)
        fraction = soil.mobilization.mobilized_fraction(shear_displacement=slide_m)
        left.plot(
            slide_m * MILLIMETERS_PER_METER,
            fraction,
            color=ACCENT_PRIMARY,
            linewidth=1.6,
        )
        for multiple, label in ((1.0, "one K"), (3.0, "three K")):
            distance_mm = multiple * one_modulus_m * MILLIMETERS_PER_METER
            reached = float(
                soil.mobilization.mobilized_fraction(
                    shear_displacement=multiple * one_modulus_m
                )
            )
            left.plot(
                [distance_mm, distance_mm],
                [0.0, reached],
                color=INK_MUTED,
                linewidth=0.8,
                linestyle=(0, (4, 3)),
            )
            left.annotate(
                f"{label}: {distance_mm:.0f} mm, {reached:.0%}\n"
                f"{distance_mm / (STRIDE_LENGTH_M * MILLIMETERS_PER_METER):.0%}"
                " of a stride",
                xy=(distance_mm, reached),
                xytext=(6, -22),
                textcoords="offset points",
                color=INK_SECONDARY,
                fontsize=7.6,
            )
        left.set_title(
            "mobilized fraction against slide distance",
            color=INK_SECONDARY,
            loc="left",
        )
        left.set_xlabel("slide distance (mm)")
        left.set_ylabel("fraction of peak shear developed")
        left.set_ylim(0.0, 1.05)

        right = axes[0][1]
        for order, (name, gravity) in enumerate(GRAVITIES):
            costs = walk(
                soil,
                gravity=gravity,
                slope_degrees=0.0,
                slip_ratio=SLIP_RATIO_SWEEP,
            ).shear_J_per_m
            right.plot(
                SLIP_RATIO_SWEEP,
                costs,
                color=ACCENT_SECONDARY,
                linewidth=1.5,
                alpha=0.35 + 0.65 * order / (len(GRAVITIES) - 1),
                label=f"{name}, {gravity:.2f} m/s²",
            )
        right.set_title(
            "shear work per meter against slip ratio, three gravities",
            color=INK_SECONDARY,
            loc="left",
        )
        right.set_xlabel("slip ratio")
        right.set_ylabel("shear work (J per meter)")
        right.legend(loc="upper left")

        for panel in (left, right):
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)

        figure.suptitle(
            "Two thirds of peak traction develops in the first six percent of a "
            f"{STRIDE_LENGTH_M * MILLIMETERS_PER_METER:.0f} mm stride",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.066,
            ha="left",
            y=0.955,
        )
        figure.text(
            0.066,
            0.900,
            caption(
                "Lunar regolith, intercrater, Janosi-Hanamoto K = 1.8 cm from "
                "Lunar Sourcebook Table 9.14. Those parameters come from "
                "pressing rather than walking: mobilization under repeated gait "
                f"loading is unmeasured. Shear work assumes a {MASS_KG:.0f} kg "
                f"platform on "
                f"{FOOT_HALF_WIDTH_M * MILLIMETERS_PER_METER:.0f} mm half-width "
                f"patches, {FEET_IN_STANCE} in stance."
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def build_decomposition_figure(soil: SoilUnderFoot) -> Figure:
    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (10.6, 5.6),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.640,
                    "figure.subplot.bottom": 0.190,
                    "figure.subplot.left": 0.058,
                    "figure.subplot.right": 0.986,
                    "figure.subplot.wspace": 0.190,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 3, squeeze=False, sharey=True)

        for column, (name, gravity) in enumerate(GRAVITIES):
            panel = axes[0][column]
            costs = walk(soil, gravity=gravity, slope_degrees=SLOPE_DEGREES)
            stack = [
                np.asarray(getattr(costs, f"{term}_J_per_m"))
                / (MASS_KG * gravity)
                for term in LOCOMOTION_TERMS
            ]
            bands = panel.stackplot(
                SLOPE_DEGREES,
                *stack,
                colors=[TERM_COLORS[term] for term in LOCOMOTION_TERMS],
                labels=list(LOCOMOTION_TERMS),
                edgecolor="none",
                alpha=0.9,
            )
            panel.set_title(
                f"{name}, {gravity:.2f} m/s²",
                color=INK_SECONDARY,
                loc="left",
            )
            panel.set_xlabel("slope (degrees)")
            panel.set_xlim(0.0, SLOPE_DEGREES[-1])
            panel.spines["top"].set_visible(False)
            panel.spines["right"].set_visible(False)

            depth_m = sinkage_at(soil, gravity, 0.0)
            if depth_m > soil.sinkage_ceiling_m:
                # Struck through rather than annotated. A panel outside the
                # model's validity range is not a weaker result of the same
                # kind; it is not a result, and a corner note lets a reader take
                # the curve at face value anyway.
                panel.add_patch(
                    Rectangle(
                        (0.0, 0.0),
                        1.0,
                        1.0,
                        transform=panel.transAxes,
                        facecolor=SURFACE,
                        alpha=0.62,
                        hatch="////",
                        edgecolor=INK_MUTED,
                        linewidth=0.0,
                        zorder=5.0,
                    )
                )
                panel.annotate(
                    "outside the model's validity range\n"
                    f"{depth_m * MILLIMETERS_PER_METER:.0f} mm sinkage against a "
                    f"{soil.sinkage_ceiling_m * MILLIMETERS_PER_METER:.0f} mm "
                    "published ceiling",
                    xy=(0.5, 0.5),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    color=INK_PRIMARY,
                    fontsize=8.0,
                    zorder=6.0,
                )
            if column == 0:
                panel.set_ylabel("cost of transport (dimensionless)")
                figure.legend(
                    handles=list(bands),
                    labels=list(LOCOMOTION_TERMS),
                    loc="upper left",
                    bbox_to_anchor=(0.052, 0.725),
                    ncol=len(LOCOMOTION_TERMS),
                )

        figure.suptitle(
            "At one foot size, only lunar loading stays inside the soil model's "
            "published range",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.058,
            ha="left",
            y=0.955,
        )
        figure.text(
            0.058,
            0.910,
            caption(
                "Where the model holds, both soil terms fall with gravity once "
                "normalized and leg swing rises. Comparing gravities at fixed "
                "foot geometry cannot avoid leaving the range: Earth would need "
                "a 136 mm pad, which is a different platform.\n"
                "The gravitational term is omitted, being exactly sin(slope) at "
                f"every gravity. Assumed: {MASS_KG:.0f} kg, "
                f"{FOOT_HALF_WIDTH_M * MILLIMETERS_PER_METER:.0f} mm half-width "
                f"patches, {FEET_IN_STANCE} in stance, "
                f"{STRIDE_LENGTH_M * MILLIMETERS_PER_METER:.0f} mm stride, slip "
                f"{SLIP_RATIO:.2f}, swing "
                f"{NOMINAL_SWING_WORK_PER_METER_J:.0f} J/m; slip and swing are "
                "both assumed, and the shares depend on them.",
                width=148,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def build_scaling_figure(parameter_sets: tuple[ParameterSet, ...]) -> Figure:
    with plt.rc_context(
        cast(
            Any,
            figure_style(
                {
                    "figure.figsize": (9.4, 5.2),
                    "axes.titlesize": 9.5,
                    "xtick.labelsize": 8.5,
                    "ytick.labelsize": 8.5,
                    "font.size": 9.5,
                    "legend.fontsize": 8.0,
                    "figure.subplot.top": 0.720,
                    "figure.subplot.bottom": 0.198,
                    "figure.subplot.left": 0.098,
                    "figure.subplot.right": 0.975,
                }
            ),
        )
    ):
        figure, axes = plt.subplots(1, 1, squeeze=False)
        panel = axes[0][0]

        styles = ((0, ()), (0, (5, 2)), (0, (1.5, 1.8)))
        markers = ("o", "s", "^")
        for order, parameters in enumerate(parameter_sets):
            curve = compaction_against_gravity(parameters)
            panel.plot(
                GRAVITY_SWEEP,
                curve,
                color=ACCENT_PRIMARY if order == 0 else ACCENT_SECONDARY,
                linewidth=1.6,
                linestyle=styles[order % len(styles)],
                alpha=1.0 if order == 0 else 0.55 + 0.30 * (order - 1),
                marker=markers[order % len(markers)],
                markevery=12,
                markersize=4.0,
                markerfacecolor="none",
                label=(
                    f"{parameters.label} — n = "
                    f"{parameters.contact.sinkage_exponent:.3f}, "
                    f"slope {1.0 / parameters.contact.sinkage_exponent:.2f}"
                ),
            )

        for _, gravity in GRAVITIES:
            panel.axvline(gravity, color=INK_MUTED, linewidth=0.7, linestyle=(0, (2, 3)))

        panel.set_xscale("log")
        panel.set_yscale("log")
        panel.set_xlabel("gravity (m/s²)")
        panel.set_ylabel("compaction cost, relative to its own value at 9.81 m/s²")
        panel.set_xticks([1.62, 3.71, 9.81])
        panel.set_xticklabels(["1.62", "3.71", "9.81"])
        panel.minorticks_off()
        panel.legend(loc="upper left")
        panel.spines["top"].set_visible(False)
        panel.spines["right"].set_visible(False)

        figure.suptitle(
            "Compaction cost falls with gravity under every published sinkage "
            "exponent",
            color=INK_PRIMARY,
            fontsize=11.5,
            x=0.098,
            ha="left",
            y=0.955,
        )
        figure.text(
            0.098,
            0.898,
            caption(
                "Each curve is normalized to its own value at Earth gravity, so "
                "the comparison is of slopes alone. Magnitudes are not "
                "comparable: GRC-1's plate scaling was fitted over 38 to 95 mm "
                "half-widths and does not transfer to a foot. The slope is one "
                "over the sinkage exponent, which is positive for any physical "
                "exponent.",
                width=112,
            ),
            color=INK_SECONDARY,
            fontsize=8.2,
            ha="left",
            va="top",
            linespacing=1.5,
        )
    return figure


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return repr(float(value))


def build_report(
    soil: SoilUnderFoot, parameter_sets: tuple[ParameterSet, ...]
) -> str:
    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        "#",
        "# Cost of transport for a legged platform on deformable ground,",
        "# decomposed by term and swept over gravity, slope and soil.",
        "#",
        "# Generated by studies/mobility/cost_of_transport.py. Do not edit.",
        "#",
        "# No timestamp and no sampling: every number here is a closed-form",
        "# evaluation, so re-running leaves this byte-identical.",
        "#",
        "# This is not a sortie envelope. There is no power budget, no thermal",
        "# model and no traverse in it.",
        "",
        f"schema_version = {REPORT_SCHEMA_VERSION}",
        "",
        "[environment]",
        f'python = "{platform.python_version()}"',
        f'numpy = "{np.__version__}"',
        "",
        "# Assumed, every one of them. No leg inertia has been measured or",
        "# specified for this project, and swing_work_per_meter_J is the term",
        "# that dominates at lunar gravity, so the shares below are conditional",
        "# on a number with no evidence behind it. The exponents are not.",
        "[platform]",
        f"mass_kg = {_format_float(MASS_KG)}",
        f"foot_half_width_m = {_format_float(FOOT_HALF_WIDTH_M)}",
        f"foot_area_m2 = {_format_float(FOOT_AREA_M2)}",
        f"stride_length_m = {_format_float(STRIDE_LENGTH_M)}",
        f"feet_in_stance = {FEET_IN_STANCE}",
        f"slip_ratio = {_format_float(SLIP_RATIO)}",
        "nominal_swing_work_per_meter_J = "
        f"{_format_float(NOMINAL_SWING_WORK_PER_METER_J)}",
        'basis = "assumed"',
        "",
        "[soil]",
        'id = "lunar-intercrater"',
        'bearing = "Heiken et al. (1991) Table 9.14, Bekker"',
        'strength = "Heiken et al. (1991) Table 9.12, 0-15 cm row"',
        'mobilization = "Heiken et al. (1991) Table 9.14, Janosi-Hanamoto"',
        f"cohesion_kPa = {_format_float(soil.strength.cohesion)}",
        "friction_angle_degrees = "
        f"{_format_float(soil.strength.friction_angle_degrees)}",
        "shear_deformation_modulus_m = "
        f"{_format_float(soil.mobilization.shear_deformation_modulus)}",
        f"sinkage_exponent = {_format_float(soil.contact.sinkage_exponent)}",
        "",
        "# Where the published parameters stop supporting the answer.",
        "[boundary]",
        f"sinkage_ceiling_m = {_format_float(soil.sinkage_ceiling_m)}",
        "half_width_range_m = ["
        f"{_format_float(soil.half_width_range_m[0])}, "
        f"{_format_float(soil.half_width_range_m[1])}]",
        "",
        "# The patch each gravity would need for the bearing model to stay",
        "# inside its published sinkage range, at the assumed mass and stance.",
        "# The lunar case is satisfied by an ordinary foot. The Earth case is",
        "# not satisfied by anything a 50 kg quadruped could carry, which is",
        "# the point of the caveat below.",
        "[boundary.half_width_reaching_ceiling_m]",
        *[
            f"{name} = "
            f"{_format_float(half_width_reaching_sinkage_ceiling(soil, gravity))}"
            for name, gravity in GRAVITIES
        ],
        "",
        "# The measured-versus-extrapolated boundary, quantity by quantity. The",
        "# distinction is not between confident and uncertain: it is between a",
        "# number interpolated inside a range somebody measured and a model form",
        "# carried into a regime nobody has. Only the first has an error bar.",
        "# Counts are generated, not written, so they cannot drift from the rows.",
        f"# {tally(boundary_rows(soil, parameter_sets))}",
        "",
        *toml_lines(boundary_rows(soil, parameter_sets)),
        "[[caveat]]",
        'id = "lunar_parameters_cannot_be_exercised_at_earth_gravity"',
        "detail = \"\"\"",
        "The published sinkage ceiling of 20 mm is a ceiling on lunar loading.",
        "Pressing this soil at Earth gravity with any patch a legged platform of",
        "this mass could plausibly carry drives sinkage past it immediately: the",
        "assumed 30 mm half-width foot reaches 100 mm, five times the ceiling and",
        "over three times the patch radius, where a surface bearing model has",
        "stopped describing anything. Staying inside the range at Earth gravity",
        "needs a 68 mm half-width, a 136 mm pad, which is well outside what this",
        "platform was given. So the Earth and Mars columns of the decomposition",
        "are a scaling illustration and not a prediction, and they are marked as",
        "such in the figure. Only the lunar column sits inside the published",
        "range, and it does so with about ten percent to spare.",
        '"""',
        "",
        "[[caveat]]",
        'id = "shear_under_gait_is_unmeasured"',
        "detail = \"\"\"",
        "The Table 9.14 parameters come from pressing, not walking. The shear",
        "deformation modulus was derived from in-situ slip observations, but",
        "mobilization under repeated gait loading, at gait rates, on soil",
        "already disturbed by the previous footfall, is not measured anywhere in",
        "this repository. The bearing term is interpolation inside a published",
        "range; the shear and slip terms are a model form carried into a regime",
        "nobody has measured. They are not equally supported and should not be",
        "reported as though they were.",
        '"""',
        "",
        "[[caveat]]",
        'id = "foot_slip_is_not_slope_stability"',
        "detail = \"\"\"",
        "maximum_traversable_slope_degrees is a foot-slip criterion: the slope",
        "at which demanded traction equals what the patches can carry. It says",
        "nothing about whether the slope itself stands. It lands just above the",
        "friction angle at every gravity, which is above the repose angle of",
        "loose surface regolith, so bulk slope failure is the binding limit and",
        "this model does not represent it. Treat these values as an upper bound",
        "that another mechanism cuts before it is reached.",
        '"""',
        "",
        "[[caveat]]",
        'id = "magnitudes_across_soils_are_not_compared"',
        "detail = \"\"\"",
        "GRC-1's plate scaling was fitted over half-widths of 38 to 95 mm. A",
        "30 mm patch is outside that, so a simulant-versus-Moon stiffness ratio",
        "at foot scale would be extrapolation presented as measurement. Only the",
        "sinkage exponent is compared here, because it is fitted from the shape",
        "of each pressure-sinkage curve rather than from scaling across plates,",
        "and the compaction gravity exponent depends on nothing else.",
        '"""',
        "",
        "# How each term scales with gravity once normalized by weight. These",
        "# follow from the model forms and hold for any platform. The shear",
        "# exponent is minus the cohesive fraction of shear strength, exactly.",
        "",
    ]

    for name, gravity in GRAVITIES:
        costs = walk(soil, gravity=gravity, slope_degrees=0.0)
        exponents = normalized_term_exponents(soil, gravity)
        soil_cost = soil_cost_per_meter(soil, gravity)
        lines += [
            "[[gravity]]",
            f'id = "{name}"',
            f"gravity_m_per_s2 = {_format_float(gravity)}",
            "normal_stress_kPa = "
            + _format_float(
                float(
                    ContactPatch(
                        half_width_m=FOOT_HALF_WIDTH_M, area_m2=FOOT_AREA_M2
                    ).normal_stress_kPa(
                        normal_load_N=MASS_KG * gravity / FEET_IN_STANCE
                    )
                )
            ),
            f"sinkage_m = {_format_float(sinkage_at(soil, gravity, 0.0))}",
            "bearing_within_published_range = "
            + str(sinkage_at(soil, gravity, 0.0) <= soil.sinkage_ceiling_m).lower(),
            f"shear_J_per_m = {_format_float(float(costs.shear_J_per_m))}",
            f"compaction_J_per_m = {_format_float(float(costs.compaction_J_per_m))}",
            f"soil_cost_J_per_m = {_format_float(soil_cost)}",
            "soil_cost_dimensionless = "
            + _format_float(soil_cost / (MASS_KG * gravity)),
            "# A platform whose swing cost exceeds soil_cost_J_per_m spends more",
            "# energy moving its own legs than moving through the terrain.",
            "swing_crossover_J_per_m = " + _format_float(soil_cost),
            "flat_cost_of_transport_at_nominal_swing = "
            + _format_float(float(costs.dimensionless)),
            "maximum_traversable_slope_degrees = "
            + _format_float(maximum_traversable_slope_degrees(soil, gravity)),
            "slope_gained_from_cohesion_degrees = "
            + _format_float(
                maximum_traversable_slope_degrees(soil, gravity)
                - soil.strength.friction_angle_degrees
            ),
            "",
            "[gravity.normalized_exponent]",
        ]
        lines += [
            f"{term} = {_format_float(exponents[term])}" for term in TERMS
        ]
        lines += [""]

    lines += [
        "# The compaction exponent is one over the sinkage exponent, and it is",
        "# positive for every published parameter set, so the collapse of",
        "# compaction cost under reduced gravity is not an artifact of the",
        "# lunar set happening to have n = 1.",
        "",
    ]
    for parameters in parameter_sets:
        exponent = parameters.contact.sinkage_exponent
        lines += [
            "[[compaction_scaling]]",
            f'id = "{parameters.id}"',
            f'label = "{parameters.label}"',
            f'source = "{parameters.source}"',
            f'note = "{parameters.note}"',
            f"sinkage_exponent = {_format_float(exponent)}",
            f"normalized_gravity_exponent = {_format_float(1.0 / exponent)}",
            "cost_at_lunar_relative_to_earth = "
            + _format_float(
                float(
                    np.interp(
                        LUNAR_GRAVITY,
                        GRAVITY_SWEEP,
                        compaction_against_gravity(parameters),
                    )
                )
            ),
            "",
        ]

    lines += ["# Cost of transport against slope, at the nominal swing cost.", ""]
    for name, gravity in GRAVITIES:
        costs = walk(soil, gravity=gravity, slope_degrees=SLOPE_DEGREES)
        dimensionless = np.asarray(costs.dimensionless)
        for index in range(0, SLOPE_DEGREES.size, 20):
            lines += [
                "[[slope_point]]",
                f'gravity = "{name}"',
                f"slope_degrees = {_format_float(float(SLOPE_DEGREES[index]))}",
                "cost_of_transport = "
                f"{_format_float(float(dimensionless[index]))}",
                "",
            ]

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose cost of transport by term and sweep it over gravity, "
            "slope and soil."
        )
    )
    parser.add_argument("--figure-directory", type=Path, default=FIGURE_DIRECTORY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    arguments = parser.parse_args(argv)

    soil = load_lunar_soil()
    parameter_sets = load_parameter_sets(soil)

    arguments.figure_directory.mkdir(parents=True, exist_ok=True)
    arguments.report.parent.mkdir(parents=True, exist_ok=True)

    for name, figure in (
        ("slip-mobilization", build_slip_figure(soil)),
        ("cost-of-transport-decomposition", build_decomposition_figure(soil)),
        ("compaction-gravity-scaling", build_scaling_figure(parameter_sets)),
    ):
        path = arguments.figure_directory / f"{name}.png"
        figure.savefig(path, dpi=200)
        plt.close(figure)
        print(f"wrote {path.relative_to(REPOSITORY_ROOT)}")

    arguments.report.write_text(build_report(soil, parameter_sets), encoding="utf-8")
    print(f"wrote {arguments.report.relative_to(REPOSITORY_ROOT)}")

    print("\n  measured against extrapolated\n")
    print(text_table(boundary_rows(soil, parameter_sets)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
