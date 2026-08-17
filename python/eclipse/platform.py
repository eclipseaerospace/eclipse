# SPDX-License-Identifier: Apache-2.0
#
# eclipse.platform — a body, and the two quantities the mobility layer had been
# assuming.
#
# Rung two left a hole. Cost of transport was real and its gravity exponents
# held generally, but slip ratio and swing cost were invented, both were
# load-bearing, and both are properties of a platform rather than of soil. This
# module supplies them, and in doing so is the first thing in the repository
# that knows a robot exists.
#
# The seam is the test. Nothing here is imported by terramechanics or mobility;
# the dependency runs one way, a platform produces a contact patch and hands it
# down, and the contact layer still cannot tell what is standing on it. If that
# had required changing the interface the mission layer consumes, the seam would
# have been in the wrong place.
#
# Swing work has two parts with different gravity dependence, which is the whole
# reason for computing rather than assuming it:
#
#   inertial    accelerating the leg through its sweep. Gravity does not enter,
#                 so this is identical on the Moon and on Earth.
#   clearance   raising the leg to lift the foot over the ground. Scales with
#                 gravity, and at one sixth g nearly vanishes.
#
# The inertial part is taken as the positive mechanical work of one acceleration
# phase. Braking is negative work: a non-regenerative drive dissipates it and
# pays for it electrically, but it is not positive mechanical work, and the soil
# terms this is added to are mechanical. An actuator with no recovery would
# roughly double the inertial term, and that is a stated sensitivity rather than
# a hidden factor of two.
#
# Slip is not a platform property either. It is an equilibrium: the foot must
# supply the along-slope component of weight, Janosi-Hanamoto says how far it
# must slide to develop that, and inverting gives the slide. So slip becomes an
# output of slope, soil and platform, and it diverges where demanded traction
# reaches capacity.
#
# Two consequences of that balance are worth stating because they are easy to
# get wrong:
#
# Compaction is not a tangential resistance here. For a wheel it is -- the wheel
# makes new rut continuously and forward motion pays for it. A walking foot is
# placed and pressed vertically, so the leg pays that work downward and it never
# appears in the traction balance. Carrying a wheel's compaction resistance into
# a legged model would inflate the demand and understate every slope.
#
# Slip on level ground comes out exactly zero, because level ground demands no
# tangential force. Real walking slips on the flat through within-stride
# acceleration, control error and foot placement, none of which is modelled. So
# the flat-ground slip here is a lower bound and should be read as one.
#
# References
#   Janosi Z, Hanamoto B (1961) The analytical determination of drawbar pull as
#     a function of slip for tracked vehicles in deformable soils. Proceedings
#     of the 1st International Conference on Terrain-Vehicle Systems.
#   Carrier WD III, Olhoeft GR, Mendell W (1991) Physical Properties of the
#     Lunar Surface. In: Lunar Sourcebook, ch. 9. Cambridge University Press.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from eclipse.mobility import ContactPatch
from eclipse.terramechanics import JanosiHanamotoModel, MohrCoulombModel

__all__ = [
    "Platform",
    "SwingWork",
    "TractionBalance",
    "equilibrium_slip_ratio",
    "maximum_traversable_slope_degrees",
    "swing_work_per_meter",
    "swing_work_per_stride",
    "traction_balance",
]

KILO: Final = 1.0e3

# A uniform rod about one end. The leg is not a uniform rod, but its inertia is
# wanted to a factor rather than a percent, and any distribution that puts more
# mass proximally lowers this. Stated so the direction of the error is known.
ROD_INERTIA_COEFFICIENT: Final = 1.0 / 3.0
ROD_CENTER_OF_MASS_FRACTION: Final = 0.5

# A triangular angular-velocity profile: constant acceleration to a peak, then
# constant deceleration. Peak is twice the mean.
TRIANGULAR_PEAK_OVER_MEAN: Final = 2.0


@dataclass(frozen=True, slots=True)
class Platform:
    """One body, carrying only what the mobility layer needs to stop assuming.

    Deliberately not a robot description. There is no actuator model, no link
    graph and no controller here, because none of those change a cost per meter
    at quasi-static stance, and inventing them would put more fabricated numbers
    underneath a result that already rests on two.
    """

    body_mass_kg: float
    leg_mass_kg: float
    leg_length_m: float
    legs: int
    feet_in_stance: int
    foot_half_width_m: float
    foot_contact_area_m2: float
    stride_length_m: float
    foot_clearance_m: float
    nominal_speed_m_per_s: float

    def __post_init__(self) -> None:
        for name, value in (
            ("body_mass_kg", self.body_mass_kg),
            ("leg_mass_kg", self.leg_mass_kg),
            ("leg_length_m", self.leg_length_m),
            ("foot_half_width_m", self.foot_half_width_m),
            ("foot_contact_area_m2", self.foot_contact_area_m2),
            ("stride_length_m", self.stride_length_m),
            ("foot_clearance_m", self.foot_clearance_m),
            ("nominal_speed_m_per_s", self.nominal_speed_m_per_s),
        ):
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and positive, got {value}")
        if self.legs < 1:
            raise ValueError(f"legs must be at least one, got {self.legs}")
        if not 1 <= self.feet_in_stance <= self.legs:
            raise ValueError(
                f"feet_in_stance must lie between one and legs ({self.legs}), got "
                f"{self.feet_in_stance}; a platform with nothing on the ground is "
                "not walking, and one with more feet down than it has is not a "
                "platform"
            )
        if self.stride_length_m > 2.0 * self.leg_length_m:
            raise ValueError(
                f"stride_length_m {self.stride_length_m} exceeds twice "
                f"leg_length_m {self.leg_length_m}, so the hip cannot sweep the "
                "foot that far; the geometry has no solution rather than a poor one"
            )

    @property
    def total_mass_kg(self) -> float:
        return self.body_mass_kg + self.legs * self.leg_mass_kg

    @property
    def duty_factor(self) -> float:
        return self.feet_in_stance / self.legs

    @property
    def contact_patch(self) -> ContactPatch:
        return ContactPatch(
            half_width_m=self.foot_half_width_m, area_m2=self.foot_contact_area_m2
        )

    @property
    def hip_sweep_radians(self) -> float:
        return 2.0 * math.asin(self.stride_length_m / (2.0 * self.leg_length_m))

    @property
    def leg_inertia_about_hip_kg_m2(self) -> float:
        return ROD_INERTIA_COEFFICIENT * self.leg_mass_kg * self.leg_length_m**2

    @property
    def stride_period_s(self) -> float:
        return self.stride_length_m / self.nominal_speed_m_per_s

    @property
    def swing_duration_s(self) -> float:
        swinging = 1.0 - self.duty_factor
        if swinging <= 0.0:
            raise ValueError(
                "every leg is in stance, so no leg swings and swing work is "
                f"undefined; legs {self.legs}, feet_in_stance {self.feet_in_stance}"
            )
        return swinging * self.stride_period_s

    def normal_load_per_foot_N(
        self, *, gravity_m_per_s2: float, slope_degrees: ArrayLike
    ) -> NDArray[np.float64]:
        slope = np.radians(np.asarray(slope_degrees, dtype=np.float64))
        weight_N = self.total_mass_kg * gravity_m_per_s2
        return np.asarray(weight_N * np.cos(slope) / self.feet_in_stance)


@dataclass(frozen=True, slots=True)
class SwingWork:
    inertial_J: NDArray[np.float64]
    clearance_J: NDArray[np.float64]

    @property
    def total_J(self) -> NDArray[np.float64]:
        return np.asarray(self.inertial_J + self.clearance_J)

    @property
    def inertial_fraction(self) -> NDArray[np.float64]:
        return np.asarray(self.inertial_J / self.total_J)


@dataclass(frozen=True, slots=True)
class TractionBalance:
    """What one foot must supply against what it can, on a slope.

    The cohesive share of the margin is the quantity worth watching. Cohesion is
    a few percent of shear strength at foot stress, which is why Day 2 concluded
    it barely matters -- but on a slope standing at its angle of repose the
    frictional capacity equals the demand identically, so the whole margin is
    cohesive. Strength and margin are different questions and cohesion answers
    them differently.
    """

    demanded_N: NDArray[np.float64]
    capacity_N: NDArray[np.float64]
    cohesive_capacity_N: NDArray[np.float64]

    @property
    def margin_N(self) -> NDArray[np.float64]:
        return np.asarray(self.capacity_N - self.demanded_N)

    @property
    def mobilized_fraction(self) -> NDArray[np.float64]:
        return np.asarray(self.demanded_N / self.capacity_N)

    @property
    def cohesive_share_of_margin(self) -> NDArray[np.float64]:
        return np.asarray(self.cohesive_capacity_N / self.margin_N)


def swing_work_per_stride(
    *, platform: Platform, gravity_m_per_s2: float
) -> SwingWork:
    """Work to swing one leg through one stride, split by gravity dependence.

    The inertial term takes a triangular angular-velocity profile and counts the
    positive mechanical work of the acceleration phase. The clearance term
    raises the leg's center of mass, which for a rod rises by half what the foot
    does.
    """
    if not (math.isfinite(gravity_m_per_s2) and gravity_m_per_s2 > 0.0):
        raise ValueError(
            f"gravity_m_per_s2 must be finite and positive, got {gravity_m_per_s2}"
        )

    mean_angular_velocity = platform.hip_sweep_radians / platform.swing_duration_s
    peak = TRIANGULAR_PEAK_OVER_MEAN * mean_angular_velocity
    inertial_J = 0.5 * platform.leg_inertia_about_hip_kg_m2 * peak**2

    center_of_mass_rise_m = (
        platform.foot_clearance_m * ROD_CENTER_OF_MASS_FRACTION
    )
    clearance_J = platform.leg_mass_kg * gravity_m_per_s2 * center_of_mass_rise_m

    return SwingWork(
        inertial_J=np.asarray(float(inertial_J)),
        clearance_J=np.asarray(float(clearance_J)),
    )


def swing_work_per_meter(
    *, platform: Platform, gravity_m_per_s2: float, slip_ratio: ArrayLike
) -> SwingWork:
    """Swing work per meter advanced, which is what cost of transport wants.

    Every leg swings once per stride and the stride advances the platform by
    less than its length when the foot slips, so slip raises swing cost as well
    as shear cost.
    """
    ratio = np.asarray(slip_ratio, dtype=np.float64)
    violations = np.asarray(~((ratio >= 0.0) & (ratio < 1.0) & np.isfinite(ratio)))
    if violations.any():
        first = float(np.asarray(ratio).ravel()[np.argmax(violations.ravel())])
        raise ValueError(
            "slip_ratio must lie in [0, 1); at 1 the platform makes no progress "
            f"and cost per meter is undefined. First offending value {first}"
        )

    per_stride = swing_work_per_stride(
        platform=platform, gravity_m_per_s2=gravity_m_per_s2
    )
    advance_m = platform.stride_length_m * (1.0 - ratio)
    swings_per_meter = platform.legs / advance_m
    return SwingWork(
        inertial_J=np.asarray(per_stride.inertial_J * swings_per_meter),
        clearance_J=np.asarray(per_stride.clearance_J * swings_per_meter),
    )


def traction_balance(
    *,
    platform: Platform,
    strength: MohrCoulombModel,
    gravity_m_per_s2: float,
    slope_degrees: ArrayLike,
) -> TractionBalance:
    slope = np.radians(np.asarray(slope_degrees, dtype=np.float64))
    weight_N = platform.total_mass_kg * gravity_m_per_s2

    demanded_N = np.abs(weight_N * np.sin(slope)) / platform.feet_in_stance
    normal_N = weight_N * np.cos(slope) / platform.feet_in_stance
    cohesive_N = strength.cohesion * KILO * platform.foot_contact_area_m2

    return TractionBalance(
        demanded_N=np.asarray(demanded_N),
        capacity_N=np.asarray(cohesive_N + normal_N * strength.friction_coefficient),
        cohesive_capacity_N=np.asarray(np.broadcast_to(cohesive_N, demanded_N.shape)),
    )


def equilibrium_slip_ratio(
    *,
    platform: Platform,
    strength: MohrCoulombModel,
    mobilization: JanosiHanamotoModel,
    gravity_m_per_s2: float,
    slope_degrees: ArrayLike,
) -> NDArray[np.float64]:
    """Slip that develops exactly the traction the slope demands.

    Inverting Janosi-Hanamoto gives the slide directly, with no iteration:
    a fraction f of capacity needs K*(-ln(1 - f)). Beyond the traction limit
    there is no equilibrium and the result is infinite, which is the honest
    answer -- the platform accelerates downhill rather than walking.
    """
    balance = traction_balance(
        platform=platform,
        strength=strength,
        gravity_m_per_s2=gravity_m_per_s2,
        slope_degrees=slope_degrees,
    )
    fraction = balance.mobilized_fraction
    # log1p rather than log(1 - f). On gentle slopes f is small, where the
    # subtraction loses the leading digits and log1p does not, and it also
    # returns a positive zero at zero demand where the other form returns a
    # negative one that would print as -0.0000 in every report downstream.
    with np.errstate(divide="ignore", invalid="ignore"):
        slide_m = -mobilization.shear_deformation_modulus * np.log1p(
            -np.minimum(fraction, 1.0)
        )
    return np.asarray(
        np.where(fraction < 1.0, slide_m / platform.stride_length_m, np.inf)
    )


def maximum_traversable_slope_degrees(
    *, platform: Platform, strength: MohrCoulombModel, gravity_m_per_s2: float
) -> float:
    """The slope at which demanded traction reaches capacity.

    Setting demand equal to capacity and collecting terms gives
    friction_angle + asin(reserve * cos(friction_angle)), where the reserve is
    the cohesive force as a fraction of total weight. The reserve goes as one
    over gravity, so cohesion buys a steeper slope on the Moon than on Earth by
    exactly the gravity ratio.

    A foot-slip criterion, and only that. It says nothing about whether the
    slope stands, and for lunar parameters it lands above the repose angle of
    loose regolith, so bulk slope failure is what binds first.
    """
    friction_angle = math.radians(strength.friction_angle_degrees)
    cohesive_N = strength.cohesion * KILO * platform.foot_contact_area_m2
    reserve = (
        cohesive_N
        * platform.feet_in_stance
        / (platform.total_mass_kg * gravity_m_per_s2)
    )
    argument = reserve * math.cos(friction_angle)
    if argument >= 1.0:
        return 90.0
    return math.degrees(friction_angle + math.asin(argument))
