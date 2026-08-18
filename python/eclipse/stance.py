# SPDX-License-Identifier: Apache-2.0
#
# eclipse.stance — how a set of feet shares the load, over a footprint and over
# a stride.
#
# Rung two modelled one contact and divided the weight by the number of feet.
# That is exact for the total and wrong for every foot, and the difference is
# not conservative: sinkage is non-linear in pressure, so the most-loaded foot
# leaves the bearing model's validity range while the mean is still comfortably
# inside it.
#
# The statics. Feet define the contact plane, so all of them sit at z = 0 and
# the body's centre of mass sits at height h above it. Uphill is +x. Balancing
# forces along the surface normal and moments about the two in-plane axes gives
# three equations in the per-foot normal loads:
#
#   sum(N_i)       = m*g*cos(slope)
#   sum(x_i * N_i) = m*g*cos(slope)*x_com - m*g*sin(slope)*h
#   sum(y_i * N_i) = m*g*cos(slope)*y_com
#
# The along-slope component of weight acts at the centre of mass rather than at
# the ground, so it contributes a pitching moment and the effective centre of
# mass shifts downhill by h*tan(slope). That term is the whole reason per-foot
# load depends on slope at all, and dropping it would leave load evenly split
# on every gradient.
#
# Three equations, one per foot unknown. So a tripod is determinate and a
# quadruped is not, and the extra degree of freedom needs a resolution rule.
#
# The rule here is minimum sum of squared normal loads, which is what a
# whole-body controller minimising contact effort would choose, and it has a
# closed form. It is a modelling decision and not a fact about robots: a real
# controller optimises something else -- friction-cone margin, actuator torque,
# or a task objective -- and would distribute differently. Stated here rather
# than buried so that the day a controller exists, this is the line it replaces.
#
# Two situations the solve has to refuse rather than approximate. A stance whose
# feet cannot balance the body at all -- the centre of mass outside the support
# polygon's spanning set -- has no solution, and a least-squares fit to it would
# return plausible numbers that do not satisfy equilibrium. And a solution with a
# negative normal load means a foot is being asked to pull the ground, which is
# tipping: the platform is rotating about an edge of its support polygon and this
# is no longer a stance problem.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from eclipse.mobility import ContactPatch
from eclipse.platform import ROD_CENTER_OF_MASS_FRACTION, FootPosition, Platform
from eclipse.terramechanics import JanosiHanamotoModel, MohrCoulombModel

__all__ = [
    "Gait",
    "LoadedPatch",
    "SwingReaction",
    "StanceDistribution",
    "UnbalanceableStanceError",
    "distribute_normal_load",
    "swing_reaction",
    "wave_gait",
    "within_stride_slip_ratio",
]

KILO: Final = 1.0e3
EQUILIBRIUM_RESIDUAL_TOLERANCE: Final = 1.0e-9


class UnbalanceableStanceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LoadedPatch:
    """A contact patch and what presses on it.

    This is the unit the mobility layer consumes. A quadruped produces four, a
    tripod three, a hopper one; nothing downstream counts them or asks what
    arranged them.
    """

    id: str
    patch: ContactPatch
    normal_load_N: float

    def normal_stress_kPa(self) -> float:
        return float(self.patch.normal_stress_kPa(normal_load_N=self.normal_load_N))


@dataclass(frozen=True, slots=True)
class StanceDistribution:
    feet: tuple[FootPosition, ...]
    patch: ContactPatch
    normal_load_N: NDArray[np.float64]

    @property
    def mean_N(self) -> NDArray[np.float64]:
        return np.asarray(self.normal_load_N.mean(axis=0))

    @property
    def maximum_N(self) -> NDArray[np.float64]:
        return np.asarray(self.normal_load_N.max(axis=0))

    @property
    def minimum_N(self) -> NDArray[np.float64]:
        return np.asarray(self.normal_load_N.min(axis=0))

    @property
    def spread(self) -> NDArray[np.float64]:
        """Most-loaded foot over the mean, which is what the single-patch model used."""
        return np.asarray(self.maximum_N / self.mean_N)

    @property
    def any_foot_unloaded(self) -> NDArray[np.bool_]:
        return np.asarray(self.minimum_N <= 0.0)

    def loaded_patches(self, index: int | None = None) -> tuple[LoadedPatch, ...]:
        column = (
            self.normal_load_N
            if index is None
            else self.normal_load_N[:, index]
        )
        return tuple(
            LoadedPatch(id=foot.id, patch=self.patch, normal_load_N=float(load))
            for foot, load in zip(self.feet, np.atleast_1d(np.asarray(column)).ravel())
        )


def distribute_normal_load(
    *,
    platform: Platform,
    stance: tuple[FootPosition, ...] | None = None,
    gravity_m_per_s2: float,
    slope_degrees: ArrayLike,
    center_of_mass_x_m: float = 0.0,
    center_of_mass_y_m: float = 0.0,
) -> StanceDistribution:
    """Per-foot normal load, resolved by minimum sum of squares.

    Returns loads with shape (feet, slopes). The minimum-norm solution of the
    three equilibrium equations is exactly the minimiser of the sum of squared
    loads subject to them, so no optimiser is involved and the answer is a
    pseudoinverse.
    """
    feet = stance if stance is not None else platform.footprint
    if not feet:
        raise UnbalanceableStanceError(
            "a stance with no feet on the ground cannot carry a body"
        )
    if not (math.isfinite(gravity_m_per_s2) and gravity_m_per_s2 > 0.0):
        raise ValueError(
            f"gravity_m_per_s2 must be finite and positive, got {gravity_m_per_s2}"
        )

    slope = np.atleast_1d(np.radians(np.asarray(slope_degrees, dtype=np.float64)))
    weight_N = platform.total_mass_kg * gravity_m_per_s2
    normal_N = weight_N * np.cos(slope)
    along_N = weight_N * np.sin(slope)

    positions = np.array([[foot.x_m, foot.y_m] for foot in feet], dtype=np.float64)
    balance = np.vstack(
        [np.ones(len(feet)), positions[:, 0], positions[:, 1]]
    )
    demand = np.vstack(
        [
            normal_N,
            normal_N * center_of_mass_x_m
            - along_N * platform.center_of_mass_height_m,
            normal_N * center_of_mass_y_m,
        ]
    )

    loads, *_ = np.linalg.lstsq(balance, demand, rcond=None)

    residual = np.abs(balance @ loads - demand).max()
    scale = max(float(np.abs(demand).max()), 1.0)
    if residual / scale > EQUILIBRIUM_RESIDUAL_TOLERANCE:
        raise UnbalanceableStanceError(
            f"these {len(feet)} feet cannot balance the body: the closest "
            f"distribution leaves a residual of {residual:.3e} N against a "
            f"demand of {scale:.3e} N. A least-squares fit would return loads "
            "that look plausible and do not satisfy equilibrium, so this is "
            "refused rather than approximated"
        )

    return StanceDistribution(
        feet=tuple(feet),
        patch=platform.contact_patch,
        normal_load_N=np.asarray(loads),
    )


@dataclass(frozen=True, slots=True)
class Gait:
    """Which feet are down when, as a schedule rather than a generator.

    Phase runs from zero to one over one stride. A leg is in stance for the
    first duty_factor of its own cycle, and its cycle is offset from the
    platform's by its phase offset. No controller and no footstep planner: this
    says when a foot is on the ground and nothing about how it got there.
    """

    duty_factor: float
    phase_offsets: tuple[float, ...]

    def __post_init__(self) -> None:
        if not (0.0 < self.duty_factor <= 1.0):
            raise ValueError(
                "duty_factor must lie in (0, 1]; at zero no foot is ever down "
                f"and above one a foot never lifts, got {self.duty_factor}"
            )
        for offset in self.phase_offsets:
            if not (math.isfinite(offset) and 0.0 <= offset < 1.0):
                raise ValueError(
                    f"phase offsets must lie in [0, 1), got {offset}"
                )

    @property
    def legs(self) -> int:
        return len(self.phase_offsets)

    def leg_phase(self, phase: ArrayLike) -> NDArray[np.float64]:
        """Each leg's own phase, shape (legs, ...)."""
        cycle = np.asarray(phase, dtype=np.float64)
        offsets = np.asarray(self.phase_offsets, dtype=np.float64)
        return np.asarray(np.mod(cycle[np.newaxis, ...] - offsets[:, np.newaxis], 1.0))

    def in_stance(self, phase: ArrayLike) -> NDArray[np.bool_]:
        return np.asarray(self.leg_phase(phase) < self.duty_factor)

    def swing_progress(self, phase: ArrayLike) -> NDArray[np.float64]:
        """How far through its swing each leg is, in [0, 1); NaN where in stance."""
        own = self.leg_phase(phase)
        swinging = own >= self.duty_factor
        return np.asarray(
            np.where(swinging, (own - self.duty_factor) / (1.0 - self.duty_factor), np.nan)
        )

    def feet_down(self, phase: ArrayLike) -> NDArray[np.int_]:
        return np.asarray(self.in_stance(phase).sum(axis=0))


def wave_gait(*, lift_order: tuple[int, ...], duty_factor: float) -> Gait:
    """The standard wave: legs lift one at a time, evenly spaced around the cycle.

    lift_order gives footprint indices in the order they leave the ground. The
    sequence matters — it is what keeps the centre of mass inside the remaining
    support polygon — and it is named rather than derived, because choosing it
    well is a gait design question this module does not answer.
    """
    legs = len(lift_order)
    if sorted(lift_order) != list(range(legs)):
        raise ValueError(
            f"lift_order must be a permutation of the leg indices, got {lift_order}"
        )
    offsets = [0.0] * legs
    for position, leg in enumerate(lift_order):
        offsets[leg] = position / legs
    return Gait(duty_factor=duty_factor, phase_offsets=tuple(offsets))


@dataclass(frozen=True, slots=True)
class SwingReaction:
    """Tangential force the stance feet must supply to accelerate the swing legs.

    An upper bound, and deliberately so. It assumes the body holds constant
    speed, which puts the whole reaction through the ground. A body left free
    would take it as a speed fluctuation instead and demand nothing of the soil,
    so the truth lies between and depends on a controller that does not exist.
    Both ends are reported so neither can be mistaken for the answer.
    """

    phase: NDArray[np.float64]
    tangential_N: NDArray[np.float64]
    body_speed_fluctuation_m_per_s: float

    @property
    def peak_N(self) -> float:
        return float(np.max(np.abs(self.tangential_N)))


def swing_reaction(
    *, platform: Platform, gait: Gait, samples: int = 721
) -> SwingReaction:
    """Reaction through one stride, sampled uniformly in phase.

    Each swinging leg accelerates through the first half of its swing and
    decelerates through the second, at constant magnitude, matching the
    triangular velocity profile the swing-work model uses. Reactions from legs
    swinging together add with their signs, so a gait that overlaps swings in
    opposite phases partly cancels its own demand.
    """
    if gait.legs != platform.legs:
        raise ValueError(
            f"gait schedules {gait.legs} legs but the platform has "
            f"{platform.legs}; a schedule that does not match the footprint "
            "cannot say which feet are down"
        )

    phase = np.linspace(0.0, 1.0, samples, endpoint=False)
    progress = gait.swing_progress(phase)

    swing_duration_s = (1.0 - gait.duty_factor) * platform.stride_period_s
    angular_acceleration = (
        4.0 * platform.hip_sweep_radians / swing_duration_s**2
    )
    center_of_mass_radius_m = ROD_CENTER_OF_MASS_FRACTION * platform.leg_length_m
    per_leg_N = (
        platform.leg_mass_kg * angular_acceleration * center_of_mass_radius_m
    )

    accelerating = np.where(np.isnan(progress), 0.0, np.where(progress < 0.5, 1.0, -1.0))
    tangential_N = per_leg_N * accelerating.sum(axis=0)

    # What the body would do instead if the ground supplied nothing: one leg's
    # momentum change shared over everything that is not that leg.
    peak_leg_speed = angular_acceleration * (swing_duration_s / 2.0) * center_of_mass_radius_m
    fluctuation = (
        platform.leg_mass_kg
        * peak_leg_speed
        / (platform.total_mass_kg - platform.leg_mass_kg)
    )

    return SwingReaction(
        phase=phase,
        tangential_N=np.asarray(tangential_N),
        body_speed_fluctuation_m_per_s=float(fluctuation),
    )


def within_stride_slip_ratio(
    *,
    platform: Platform,
    gait: Gait,
    strength: MohrCoulombModel,
    mobilization: JanosiHanamotoModel,
    gravity_m_per_s2: float,
    samples: int = 721,
) -> tuple[float, SwingReaction]:
    """Peak slip on level ground, from swing reaction alone.

    Rung two returned exactly zero here, because level ground demands no net
    traction. It does demand traction within a stride, and this is how much.
    Returns infinity where the demand exceeds what the feet can carry, which
    means the gait cannot be executed at this speed rather than that slip is
    merely large.
    """
    reaction = swing_reaction(platform=platform, gait=gait, samples=samples)

    feet_down = gait.feet_down(reaction.phase)
    if bool(np.any(feet_down == 0)):
        raise ValueError(
            "this gait leaves no foot on the ground at some point in the "
            "stride, so it has a flight phase and the quasi-static balance "
            "does not describe it"
        )

    weight_N = platform.total_mass_kg * gravity_m_per_s2
    cohesive_N = strength.cohesion * KILO * platform.foot_contact_area_m2
    capacity_N = feet_down * cohesive_N + weight_N * strength.friction_coefficient

    mobilized = np.abs(reaction.tangential_N) / capacity_N
    if bool(np.any(mobilized >= 1.0)):
        return math.inf, reaction

    slide_m = -mobilization.shear_deformation_modulus * np.log1p(-mobilized)
    return float(np.max(slide_m) / platform.stride_length_m), reaction
