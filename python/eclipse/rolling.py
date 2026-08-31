# SPDX-License-Identifier: Apache-2.0
#
# eclipse.rolling — the cost of rolling a wheeled platform over the same ground.
#
# The competitor. Every funded lunar surface-mobility programme is wheeled, and
# fifteen days of this project compared a quadruped against a suited human,
# which is not what anyone is proposing to send. This module exists so the
# comparison can be made against the thing that is actually being built.
#
# Three differences from a foot, and each is a different kind of change.
#
# There is no swing work. The largest term at lunar gravity, and the only one
# that does not scale with weight, simply is not paid: nothing is lifted and
# cycled. A wheel deletes the term that makes lunar walking expensive.
#
# Compaction stops being work paid downward and becomes a resistance to motion.
# A placed foot presses a hole and the leg pays for it vertically, so it never
# enters the traction balance. A rolling wheel makes new rut continuously and
# forward motion pays for every metre of it. This is the standard Bekker
# compaction resistance and it is derived below rather than asserted.
#
# Slip enters through the whole contact patch instead of at a point. Shear
# displacement under a wheel grows from zero at the entry to its maximum at the
# bottom, so the mobilized fraction is Janosi-Hanamoto integrated along the
# contact rather than evaluated at one displacement. That integral has no
# closed-form inverse, which is why slip here is solved for rather than written
# down, and it is the only iteration in the physics layer.
#
# THE UNITS TRAP, stated because it is silent and would flatter the wheel.
# Wong writes the compaction resistance with one symbol b in two distinct roles:
# the characteristic dimension inside k_c/b + k_phi, and the width of the rut
# the wheel cuts. In his convention both are the wheel's full width. This
# project's convention is that a contact model takes a HALF-width, so the two
# roles separate: the modulus is evaluated at width/2 and the rut is width. Pass
# the full width to deformation_modulus and k_c/b is halved, the sinkage comes
# out shallow and the resistance low -- a wheel that glides. The error is
# invisible in the output, which is why wheel_half_width_m and rut_width_m are
# separate properties with different values rather than one number used twice.
#
# The derivation, so the form can be checked rather than trusted:
#
#   A rigid wheel of diameter D and width w sinks to z0. At horizontal distance
#   x back from the lowest point the local depth is z0 - x^2/D, so the front
#   contact runs to x = sqrt(D*z0) where the depth reaches zero.
#
#   Vertical equilibrium integrates the normal pressure over that contact:
#     W = w * integral_0^sqrt(D*z0) k*(z0 - x^2/D)^n dx
#   Bekker's approximation (1 - u^2)^n ~ 1 - n*u^2 evaluates the integral as
#   (3-n)/3, giving W = w*k*(3-n)/3 * sqrt(D) * z0^(n+1/2), so
#     z0 = [ 3W / ((3-n) * w * k * sqrt(D)) ]^(2/(2n+1))
#
#   The resistance is the work of cutting a metre of that rut, which is the
#   pressure integrated through the depth over the rut width:
#     R_c = w * integral_0^z0 p dz = w * k * z0^(n+1) / (n+1)
#
# Both routes to R_c are implemented and the tests assert they agree, which is
# what makes this a verified form rather than a transcribed one.
#
# Energy per metre follows from the slip definition and needs no separate shear
# integral. In steady state the tractive force equals the resistance it must
# overcome, F = R_c + m*g*sin(slope), while the wheel rim travels 1/(1-i) metres
# for every metre the vehicle advances. So the energy per metre advanced is
# F/(1-i), and the excess over F is the slip loss. That is why slip is reported
# as a shear term here: it is the work dissipated shearing soil under a wheel
# that is turning faster than it is travelling.
#
# What is deliberately absent: bulldozing resistance, which matters only at
# sinkage comparable to wheel radius; bearing, gearing and drivetrain losses,
# which are a machine rather than a terrain; and any grouser, which is the first
# thing a real lunar wheel would have and which raises traction substantially.
# Omitting grousers understates the wheel.
#
# References
#   Bekker MG (1956) Theory of Land Locomotion. University of Michigan Press.
#   Bekker MG (1969) Introduction to Terrain-Vehicle Systems. University of
#     Michigan Press.
#   Wong JY (2008) Theory of Ground Vehicles, 4th ed. Wiley. Ch. 2, motion
#     resistance and tractive effort of rigid wheels.
#   Janosi Z, Hanamoto B (1961) The analytical determination of drawbar pull as
#     a function of slip for tracked vehicles in deformable soils. Proceedings
#     of the 1st International Conference on Terrain-Vehicle Systems.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from eclipse.mobility import CostOfTransport
from eclipse.terramechanics import (
    ContactModel,
    JanosiHanamotoModel,
    MohrCoulombModel,
)

__all__ = [
    "WheeledPlatform",
    "compaction_resistance_N",
    "contact_length_m",
    "mobilized_tractive_fraction",
    "rigid_wheel_sinkage_m",
    "rolling_cost_of_transport",
    "wheel_equilibrium_slip_ratio",
    "wheel_maximum_traversable_slope_degrees",
]

KILO: Final = 1.0e3

# Bekker's evaluation of integral_0^1 (1 - u^2)^n du under the approximation
# (1 - u^2)^n ~ 1 - n*u^2. Exact at n = 0 and n = 1, and within two percent of
# the true value over the range of sinkage exponents this project carries.
BEKKER_CONTACT_INTEGRAL_NUMERATOR: Final = 3.0

# Slip is solved by bisection on a monotone function. Fixed count rather than a
# tolerance so the result is deterministic to the bit across runs and machines.
SLIP_BISECTION_STEPS: Final = 90

# The slip ratio at which a wheel is spinning rather than driving. Beyond this
# the vehicle is not making progress in any useful sense, and the cost per metre
# has already diverged: at 0.98 the rim turns fifty metres for every one.
SLIP_CEILING: Final = 0.98


@dataclass(frozen=True, slots=True)
class WheeledPlatform:
    """One nominal rover, carrying only what a cost per metre needs.

    Not a design and not a rover that exists, in exactly the sense the nominal
    quadruped is not a robot. The mass matches that platform so the comparison
    is about locomotion rather than about size.

    There is no footprint here, unlike the legged platform, and the absence is
    the point: a rover's wheels do not redistribute load between themselves the
    way a stance does, so a per-wheel position buys nothing a wheelbase does not
    already give. Load is shared equally, which is what a rocker-bogie is for.
    """

    body_mass_kg: float
    wheel_mass_kg: float
    wheels: int
    wheel_diameter_m: float
    wheel_width_m: float
    wheelbase_m: float
    track_width_m: float
    center_of_mass_height_m: float
    nominal_speed_m_per_s: float

    def __post_init__(self) -> None:
        for name, value in (
            ("body_mass_kg", self.body_mass_kg),
            ("wheel_mass_kg", self.wheel_mass_kg),
            ("wheel_diameter_m", self.wheel_diameter_m),
            ("wheel_width_m", self.wheel_width_m),
            ("wheelbase_m", self.wheelbase_m),
            ("track_width_m", self.track_width_m),
            ("center_of_mass_height_m", self.center_of_mass_height_m),
            ("nominal_speed_m_per_s", self.nominal_speed_m_per_s),
        ):
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and positive, got {value}")
        if self.wheels < 3:
            raise ValueError(
                f"wheels must be at least three for a statically stable vehicle, "
                f"got {self.wheels}"
            )
        if self.wheels % 2 != 0:
            raise ValueError(
                f"wheels must be even so the wheelbase and track describe axle "
                f"pairs, got {self.wheels}; an odd count needs a footprint rather "
                "than two spans"
            )

    @property
    def total_mass_kg(self) -> float:
        return self.body_mass_kg + self.wheels * self.wheel_mass_kg

    @property
    def rut_width_m(self) -> float:
        """The width of soil the wheel compacts, which is the full width."""
        return self.wheel_width_m

    @property
    def wheel_half_width_m(self) -> float:
        """The dimension a contact model is evaluated at, which is half of it."""
        return 0.5 * self.wheel_width_m

    @property
    def tipping_slope_degrees(self) -> float:
        """Pitch-over about the downhill axle, climbing.

        The same criterion the legged platform is held to, so the two limits are
        comparable: the centre of mass passes over the contact furthest uphill.
        A rover is usually the wider of the two and so tips later, which is a
        real advantage and is reported rather than equalised away.
        """
        return math.degrees(
            math.atan2(0.5 * self.wheelbase_m, self.center_of_mass_height_m)
        )

    def normal_load_per_wheel_N(
        self, *, gravity_m_per_s2: float, slope_degrees: ArrayLike
    ) -> NDArray[np.float64]:
        slope = np.radians(np.asarray(slope_degrees, dtype=np.float64))
        weight_N = self.total_mass_kg * gravity_m_per_s2
        return np.asarray(weight_N * np.cos(slope) / self.wheels)


def rigid_wheel_sinkage_m(
    *,
    platform: WheeledPlatform,
    contact_model: ContactModel,
    normal_load_N: ArrayLike,
) -> NDArray[np.float64]:
    """Bekker sinkage of a rigid wheel under a vertical load.

    The modulus is evaluated at the wheel's half-width and the rut is its full
    width; see the units trap in the module header. Passing one number for both
    understates sinkage and every quantity that follows from it.
    """
    load_N = np.asarray(normal_load_N, dtype=np.float64)
    violations = np.asarray(~(np.isfinite(load_N) & (load_N > 0.0)))
    if violations.any():
        first = float(load_N.ravel()[np.argmax(violations.ravel())])
        raise ValueError(
            "normal_load_N must be finite and positive; a wheel carrying no load "
            f"has no sinkage to solve for. First offending value {first}, and "
            f"{int(violations.sum())} of {load_N.size} offend"
        )
    exponent = contact_model.sinkage_exponent
    if exponent >= BEKKER_CONTACT_INTEGRAL_NUMERATOR:
        raise ValueError(
            "Bekker's contact integral evaluates to (3 - n)/3 and is not "
            f"positive at a sinkage exponent of {exponent}; the rigid-wheel "
            "sinkage form does not apply there"
        )
    modulus_kPa = contact_model.deformation_modulus(platform.wheel_half_width_m)
    numerator = BEKKER_CONTACT_INTEGRAL_NUMERATOR * load_N / KILO
    denominator = (
        (BEKKER_CONTACT_INTEGRAL_NUMERATOR - exponent)
        * platform.rut_width_m
        * modulus_kPa
        * math.sqrt(platform.wheel_diameter_m)
    )
    return np.asarray(
        np.power(numerator / denominator, 2.0 / (2.0 * exponent + 1.0))
    )


def contact_length_m(
    *, platform: WheeledPlatform, sinkage_m: ArrayLike
) -> NDArray[np.float64]:
    """Length of the loaded arc, from first contact to the lowest point.

    Unlike a foot, a wheel's contact patch is an output of the load rather than
    a property of the body: press harder and it lengthens. Everything that scales
    with contact area -- the cohesive share of traction above all -- therefore
    moves with slope and with mass.
    """
    depth = np.asarray(sinkage_m, dtype=np.float64)
    return np.asarray(np.sqrt(platform.wheel_diameter_m * depth))


def compaction_resistance_N(
    *,
    platform: WheeledPlatform,
    contact_model: ContactModel,
    sinkage_m: ArrayLike,
) -> NDArray[np.float64]:
    """Force resisting motion because the wheel is cutting rut.

    The pressure integrated through the depth, over the width of the rut. This
    is the definition; the closed form in Wong follows from substituting the
    sinkage equation into it, and the tests check the two against each other.
    """
    depth = np.asarray(sinkage_m, dtype=np.float64)
    modulus_kPa = contact_model.deformation_modulus(platform.wheel_half_width_m)
    exponent = contact_model.sinkage_exponent
    return np.asarray(
        platform.rut_width_m
        * modulus_kPa
        * KILO
        * np.power(depth, exponent + 1.0)
        / (exponent + 1.0)
    )


def mobilized_tractive_fraction(
    *,
    mobilization: JanosiHanamotoModel,
    contact_length_m: ArrayLike,
    slip_ratio: ArrayLike,
) -> NDArray[np.float64]:
    """Share of shear capacity developed, integrated along the contact.

    Shear displacement under a wheel is j(x) = i*x measured back from the entry,
    so the mean of the Janosi-Hanamoto mobilization over the patch is
    1 - (K/(i*L))*(1 - exp(-i*L/K)) rather than the single-displacement value a
    foot uses. It is smaller: part of the patch has barely begun to grip, which
    is why a wheel needs more slip than a foot to develop the same fraction.
    """
    length = np.asarray(contact_length_m, dtype=np.float64)
    ratio = np.asarray(slip_ratio, dtype=np.float64)
    modulus = mobilization.shear_deformation_modulus
    scaled = ratio * length / modulus
    # The limit at zero slip is zero, and the series avoids 0/0 there.
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.where(
            scaled > 1.0e-8,
            1.0 - (1.0 - np.exp(-scaled)) / np.where(scaled > 0.0, scaled, 1.0),
            0.5 * scaled,
        )
    return np.asarray(np.clip(fraction, 0.0, 1.0))


def _tractive_capacity_N(
    *,
    platform: WheeledPlatform,
    strength: MohrCoulombModel,
    normal_load_N: NDArray[np.float64],
    contact_length: NDArray[np.float64],
) -> NDArray[np.float64]:
    area_m2 = platform.rut_width_m * contact_length
    cohesive_N = strength.cohesion * KILO * area_m2
    return np.asarray(cohesive_N + normal_load_N * strength.friction_coefficient)


def wheel_equilibrium_slip_ratio(
    *,
    platform: WheeledPlatform,
    contact_model: ContactModel,
    strength: MohrCoulombModel,
    mobilization: JanosiHanamotoModel,
    gravity_m_per_s2: float,
    slope_degrees: ArrayLike,
) -> NDArray[np.float64]:
    """Slip that develops exactly the tractive force the slope and rut demand.

    Unlike the legged case this cannot be inverted in closed form, because the
    mobilization is an integral along the contact. It is monotone in slip, so
    bisection converges without a starting guess and without the possibility of
    finding a second root.

    A wheel must overcome its own compaction resistance before it climbs
    anything, so slip is non-zero on level ground -- the opposite of the foot,
    whose level-ground slip comes out exactly zero because level ground demands
    no tangential force at all.
    """
    slope = np.radians(np.asarray(slope_degrees, dtype=np.float64))
    weight_N = platform.total_mass_kg * gravity_m_per_s2
    load_N = np.asarray(weight_N * np.cos(slope) / platform.wheels)

    sinkage = rigid_wheel_sinkage_m(
        platform=platform, contact_model=contact_model, normal_load_N=load_N
    )
    length = contact_length_m(platform=platform, sinkage_m=sinkage)
    resistance_N = compaction_resistance_N(
        platform=platform, contact_model=contact_model, sinkage_m=sinkage
    )
    capacity_N = _tractive_capacity_N(
        platform=platform,
        strength=strength,
        normal_load_N=load_N,
        contact_length=length,
    )

    demanded_N = resistance_N + weight_N * np.sin(slope) / platform.wheels
    required = np.where(capacity_N > 0.0, demanded_N / capacity_N, np.inf)

    low = np.zeros_like(required)
    high = np.full_like(required, SLIP_CEILING)
    for _ in range(SLIP_BISECTION_STEPS):
        middle = 0.5 * (low + high)
        developed = mobilized_tractive_fraction(
            mobilization=mobilization, contact_length_m=length, slip_ratio=middle
        )
        short = developed < required
        low = np.where(short, middle, low)
        high = np.where(short, high, middle)

    ceiling = mobilized_tractive_fraction(
        mobilization=mobilization,
        contact_length_m=length,
        slip_ratio=np.full_like(required, SLIP_CEILING),
    )
    return np.asarray(
        np.where(
            required <= 0.0,
            0.0,
            np.where(required > ceiling, np.inf, 0.5 * (low + high)),
        )
    )


def rolling_cost_of_transport(
    *,
    platform: WheeledPlatform,
    contact_model: ContactModel,
    strength: MohrCoulombModel,
    mobilization: JanosiHanamotoModel,
    gravity_m_per_s2: float,
    slope_degrees: ArrayLike,
    slip_ratio: ArrayLike,
) -> CostOfTransport:
    """Energy per metre for a rolling platform, in the legged decomposition.

    Returns the same object the legged model returns, with the swing term at
    zero. That is not a convenience: whether a wheel can be described by the
    structure built for a foot is the interface question this day exists to ask,
    and reusing the type rather than defining a parallel one is what makes the
    answer checkable.

    The terms are read as: compaction is the rut, gravitational is the climb,
    and shear is the work lost to a wheel turning faster than it travels. Their
    sum is (resistance + climb)/(1 - slip), which is the standard result.
    """
    slope = np.radians(np.asarray(slope_degrees, dtype=np.float64))
    ratio = np.asarray(slip_ratio, dtype=np.float64)
    violations = np.asarray(~((ratio >= 0.0) & (ratio < 1.0) & np.isfinite(ratio)))
    if violations.any():
        first = float(ratio.ravel()[np.argmax(violations.ravel())])
        raise ValueError(
            "slip_ratio must lie in [0, 1); at 1 the wheel spins without "
            "advancing and cost per metre is undefined. First offending value "
            f"{first}, and {int(violations.sum())} of {ratio.size} offend"
        )

    weight_N = platform.total_mass_kg * gravity_m_per_s2
    load_N = np.asarray(weight_N * np.cos(slope) / platform.wheels)
    sinkage = rigid_wheel_sinkage_m(
        platform=platform, contact_model=contact_model, normal_load_N=load_N
    )
    resistance_per_wheel_N = compaction_resistance_N(
        platform=platform, contact_model=contact_model, sinkage_m=sinkage
    )

    compaction_J_per_m = platform.wheels * resistance_per_wheel_N
    gravitational_J_per_m = np.asarray(weight_N * np.sin(slope))
    driven_J_per_m = compaction_J_per_m + gravitational_J_per_m
    shear_J_per_m = np.maximum(driven_J_per_m, 0.0) * ratio / (1.0 - ratio)

    shape = np.broadcast_shapes(np.shape(slope), np.shape(ratio))
    return CostOfTransport(
        gravitational_J_per_m=np.broadcast_to(gravitational_J_per_m, shape).copy(),
        shear_J_per_m=np.broadcast_to(shear_J_per_m, shape).copy(),
        compaction_J_per_m=np.broadcast_to(compaction_J_per_m, shape).copy(),
        swing_J_per_m=np.zeros(shape),
        mass_kg=platform.total_mass_kg,
        gravity_m_per_s2=gravity_m_per_s2,
    )


def wheel_maximum_traversable_slope_degrees(
    *,
    platform: WheeledPlatform,
    contact_model: ContactModel,
    strength: MohrCoulombModel,
    mobilization: JanosiHanamotoModel,
    gravity_m_per_s2: float,
) -> float:
    """The slope at which the wheel spins rather than climbs.

    Not the slope at which demand reaches capacity, because a wheel never
    reaches its capacity. Mobilization is integrated along a contact patch only
    a few times the shear deformation modulus in length, so most of the patch is
    still gripping when the rest has slid: at the slip ceiling this platform
    develops about two thirds of the shear the soil could give it. Reporting the
    full-capacity slope would credit the wheel with traction no amount of slip
    can produce, which is the same error as reporting a foot's traction limit
    without asking whether the platform tips first.

    The gap between the two is the wheel's version of the tipping-versus-slip
    distinction, and it runs the other way: for a foot the limit is a body
    property, for a wheel it is a contact-geometry property.

    Solved rather than written down, because compaction resistance depends on
    the normal load and so on the slope itself.
    """

    def shortfall(slope_degrees: float) -> float:
        slope = math.radians(slope_degrees)
        weight_N = platform.total_mass_kg * gravity_m_per_s2
        load_N = np.asarray(weight_N * math.cos(slope) / platform.wheels)
        sinkage = rigid_wheel_sinkage_m(
            platform=platform, contact_model=contact_model, normal_load_N=load_N
        )
        length = contact_length_m(platform=platform, sinkage_m=sinkage)
        capacity = _tractive_capacity_N(
            platform=platform,
            strength=strength,
            normal_load_N=load_N,
            contact_length=length,
        )
        demand = compaction_resistance_N(
            platform=platform, contact_model=contact_model, sinkage_m=sinkage
        ) + weight_N * math.sin(slope) / platform.wheels
        reachable = mobilized_tractive_fraction(
            mobilization=mobilization,
            contact_length_m=length,
            slip_ratio=np.asarray(SLIP_CEILING),
        )
        return float(demand / capacity - reachable)

    if shortfall(0.0) > 0.0:
        raise ValueError(
            "the platform cannot develop the traction its own compaction "
            "resistance demands on level ground, so there is no traversable "
            "slope; check the wheel width and diameter against the soil"
        )
    low, high = 0.0, 89.9
    if shortfall(high) < 0.0:
        return high
    for _ in range(SLIP_BISECTION_STEPS):
        middle = 0.5 * (low + high)
        if shortfall(middle) < 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)
