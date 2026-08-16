# SPDX-License-Identifier: Apache-2.0
#
# eclipse.mobility — energy cost of moving a legged platform over deformable
# ground.
#
# This is the first layer where gravity enters the physics rather than sitting
# in the background. It enters twice, in opposite directions: the traction limit
# c + sigma*tan(phi) falls with weight because sigma does, and the cost of
# transport denominator m*g*d falls with it too. Everything below rung two would
# have read identically on Earth.
#
# Four terms, and they behave differently under reduced gravity, which is the
# whole reason for decomposing rather than reporting a total:
#
#   gravitational   m*g*d*sin(slope), recoverable in principle, negative downhill
#   shear           the work of sliding a foot that has not yet gripped
#   compaction      the work of pressing soil down, never recovered
#   swing           the work of cycling legs, set by inertia rather than weight
#
# Each is a power of gravity once divided by m*g*d, and the exponents are not
# what Earth intuition suggests. Gravitational goes as sin(slope) and does not
# depend on gravity at all. Shear goes as gravity to the minus cohesive
# fraction, because the frictional part of Mohr-Coulomb scales with weight and
# cancels while the cohesive part does not; at lunar foot stress that fraction
# is a few percent, so shear is very nearly gravity-neutral. Compaction goes as
# gravity to the one over the sinkage exponent: lighter feet sink less, and for
# any positive exponent the soil gets cheaper faster than the weight falls.
# Swing goes as gravity to the minus one, since inertia does not care what a
# body weighs.
#
# Two terms rise under reduced gravity and they rise for the same structural
# reason, that neither depends on weight — but swing rises by a factor of gravity
# and shear by only the cohesive fraction of it, so swing is a couple of orders
# of magnitude the larger effect. Compaction is the term that disappears, which
# is the opposite of the usual telling. Lunar cost of transport is still worse
# than Earth intuition predicts, but the reason is leg inertia rather than the
# ground.
#
# Swing work is a platform property, not a contact property. It is taken as an
# input here rather than modelled, because a leg inertia does not belong in a
# soil-contact layer and inventing one would put a fabricated number underneath
# a real result.
#
# Units follow the contact models: lengths in meters, pressures in kPa, and the
# shear deformation modulus in the same length unit as the displacement given to
# it. Energies come out in joules per meter travelled.
#
# References
#   Bekker MG (1956) Theory of Land Locomotion. University of Michigan Press.
#   Janosi Z, Hanamoto B (1961) The analytical determination of drawbar pull as
#     a function of slip for tracked vehicles in deformable soils. Proceedings
#     of the 1st International Conference on Terrain-Vehicle Systems.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from eclipse.terramechanics import (
    ContactModel,
    JanosiHanamotoModel,
    MohrCoulombModel,
)

__all__ = [
    "ContactPatch",
    "CostOfTransport",
    "FootfallWork",
    "compaction_work_per_footfall",
    "cost_of_transport",
    "shear_work_per_footfall",
    "slip_displacement",
]

KILO: Final = 1.0e3


@dataclass(frozen=True, slots=True)
class ContactPatch:
    """One patch of ground contact, carrying only what the soil layer needs.

    A foot is a small patch and a wheel a rolling one; nothing here knows which.
    Area and half-width are both stored because the contact models take a
    half-width and the force integrals take an area, and deriving one from the
    other would assume a shape this layer has no business assuming.
    """

    half_width_m: float
    area_m2: float

    def __post_init__(self) -> None:
        for name, value in (
            ("half_width_m", self.half_width_m),
            ("area_m2", self.area_m2),
        ):
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and positive, got {value}")

    def normal_stress_kPa(self, *, normal_load_N: ArrayLike) -> NDArray[np.float64]:
        load = np.asarray(normal_load_N, dtype=np.float64)
        return np.asarray(load / self.area_m2 / KILO)


def slip_displacement(
    *, stance_length_m: ArrayLike, slip_ratio: ArrayLike
) -> NDArray[np.float64]:
    length = np.asarray(stance_length_m, dtype=np.float64)
    ratio = np.asarray(slip_ratio, dtype=np.float64)
    violations = np.asarray(~((ratio >= 0.0) & (ratio < 1.0) & np.isfinite(ratio)))
    if violations.any():
        raise ValueError(
            "slip_ratio must lie in [0, 1); at 1 the foot slides without "
            "advancing and the platform makes no progress, so cost per meter is "
            f"undefined. First offending value {float(np.asarray(ratio).ravel()[np.argmax(violations.ravel())])}"
        )
    return np.asarray(length * ratio)


def shear_work_per_footfall(
    *,
    patch: ContactPatch,
    strength: MohrCoulombModel,
    mobilization: JanosiHanamotoModel,
    normal_load_N: ArrayLike,
    slip_displacement_m: ArrayLike,
) -> NDArray[np.float64]:
    """Work done sliding a patch that is still mobilizing its grip.

    Integrating tau(j) = tau_max*(1 - exp(-j/K)) over the slid distance gives
    tau_max*(j - K*(1 - exp(-j/K))) exactly, so no quadrature is needed. The
    subtracted term is the work not done because the soil had not yet gripped:
    at small slip it nearly cancels the first term, which is why a foot that
    barely slips costs almost nothing in shear.
    """
    slid = np.asarray(slip_displacement_m, dtype=np.float64)
    capacity_kPa = strength.maximum_shear_stress(
        normal_stress=patch.normal_stress_kPa(normal_load_N=normal_load_N)
    )
    modulus = mobilization.shear_deformation_modulus
    developed = slid - modulus * mobilization.mobilized_fraction(
        shear_displacement=slid
    )
    return np.asarray(capacity_kPa * KILO * patch.area_m2 * developed)


def compaction_work_per_footfall(
    *,
    patch: ContactPatch,
    contact_model: ContactModel,
    sinkage_m: ArrayLike,
) -> NDArray[np.float64]:
    """Work pressing the patch to a depth, which the soil never gives back.

    The integral of a Bekker or Reece pressure law over depth is
    A*k_eq*z^(n+1)/(n+1). Both the modulus and the exponent are read from the
    model rather than passed separately, so a caller cannot integrate one curve
    while the sinkage came from another.
    """
    depth = np.asarray(sinkage_m, dtype=np.float64)
    modulus_kPa = contact_model.deformation_modulus(patch.half_width_m)
    exponent = contact_model.sinkage_exponent
    return np.asarray(
        modulus_kPa
        * KILO
        * patch.area_m2
        * np.power(depth, exponent + 1.0)
        / (exponent + 1.0)
    )


@dataclass(frozen=True, slots=True)
class FootfallWork:
    shear_J: NDArray[np.float64]
    compaction_J: NDArray[np.float64]
    sinkage_m: NDArray[np.float64]
    normal_stress_kPa: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CostOfTransport:
    """Energy per meter, split by term, with the total normalized by m*g*d.

    Kept as separate terms because the point of the decomposition is that they
    scale differently with gravity, and a total hides exactly that.
    """

    gravitational_J_per_m: NDArray[np.float64]
    shear_J_per_m: NDArray[np.float64]
    compaction_J_per_m: NDArray[np.float64]
    swing_J_per_m: NDArray[np.float64]
    mass_kg: float
    gravity_m_per_s2: float

    @property
    def total_J_per_m(self) -> NDArray[np.float64]:
        return np.asarray(
            self.gravitational_J_per_m
            + self.shear_J_per_m
            + self.compaction_J_per_m
            + self.swing_J_per_m
        )

    @property
    def dimensionless(self) -> NDArray[np.float64]:
        return np.asarray(
            self.total_J_per_m / (self.mass_kg * self.gravity_m_per_s2)
        )

    def fraction(self, term: str) -> NDArray[np.float64]:
        values = {
            "gravitational": self.gravitational_J_per_m,
            "shear": self.shear_J_per_m,
            "compaction": self.compaction_J_per_m,
            "swing": self.swing_J_per_m,
        }
        if term not in values:
            raise ValueError(f"no term {term!r}; this carries {sorted(values)}")
        return np.asarray(values[term] / self.total_J_per_m)


def cost_of_transport(
    *,
    mass_kg: float,
    gravity_m_per_s2: float,
    slope_degrees: ArrayLike,
    slip_ratio: ArrayLike,
    patch: ContactPatch,
    feet_in_stance: int,
    stride_length_m: float,
    stance_length_m: float,
    contact_model: ContactModel,
    strength: MohrCoulombModel,
    mobilization: JanosiHanamotoModel,
    swing_work_per_meter_J: float,
) -> CostOfTransport:
    if feet_in_stance < 1:
        raise ValueError(
            f"feet_in_stance must be at least one, got {feet_in_stance}; a "
            "platform with nothing on the ground is not walking"
        )
    for name, value in (
        ("mass_kg", mass_kg),
        ("gravity_m_per_s2", gravity_m_per_s2),
        ("stride_length_m", stride_length_m),
        ("stance_length_m", stance_length_m),
    ):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be finite and positive, got {value}")
    if not (math.isfinite(swing_work_per_meter_J) and swing_work_per_meter_J >= 0.0):
        raise ValueError(
            "swing_work_per_meter_J must be finite and non-negative, got "
            f"{swing_work_per_meter_J}"
        )

    slope = np.radians(np.asarray(slope_degrees, dtype=np.float64))
    weight_N = mass_kg * gravity_m_per_s2
    # Only the slope-normal component presses on the soil; the along-slope
    # component is what the gravitational term accounts for.
    normal_load_per_foot_N = weight_N * np.cos(slope) / feet_in_stance

    stress_kPa = patch.normal_stress_kPa(normal_load_N=normal_load_per_foot_N)
    depth_m = contact_model.sinkage(
        pressure=stress_kPa, contact_half_width=patch.half_width_m
    )

    slid = slip_displacement(
        stance_length_m=stance_length_m, slip_ratio=slip_ratio
    )
    shear_J = shear_work_per_footfall(
        patch=patch,
        strength=strength,
        mobilization=mobilization,
        normal_load_N=normal_load_per_foot_N,
        slip_displacement_m=slid,
    )
    compaction_J = compaction_work_per_footfall(
        patch=patch,
        contact_model=contact_model,
        sinkage_m=depth_m,
    )

    # Each stride advances the platform by stride_length times what is not lost
    # to slip, and plants feet_in_stance feet.
    advance_m = stride_length_m * (1.0 - np.asarray(slip_ratio, dtype=np.float64))
    footfalls_per_meter = feet_in_stance / advance_m

    return CostOfTransport(
        gravitational_J_per_m=np.asarray(weight_N * np.sin(slope)),
        shear_J_per_m=np.asarray(shear_J * footfalls_per_meter),
        compaction_J_per_m=np.asarray(compaction_J * footfalls_per_meter),
        swing_J_per_m=np.asarray(
            np.broadcast_to(
                np.asarray(float(swing_work_per_meter_J)), np.shape(slope)
            ).copy()
        ),
        mass_kg=mass_kg,
        gravity_m_per_s2=gravity_m_per_s2,
    )
