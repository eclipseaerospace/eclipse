# SPDX-License-Identifier: Apache-2.0
#
# eclipse.stance — how a set of feet shares the load, and what averaging hides.
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
from eclipse.platform import FootPosition, Platform

__all__ = [
    "LoadedPatch",
    "StanceDistribution",
    "UnbalanceableStanceError",
    "distribute_normal_load",
]

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
