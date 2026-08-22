# SPDX-License-Identifier: Apache-2.0
#
# eclipse.thermal — losing heat in vacuum, where there are only two ways to.
#
# The first environment in this project that is hostile rather than merely
# difficult. A slope can be walked around; cold cannot, and every terrestrial
# thermal intuition is wrong here because all of them assume air. There is no
# convection, no ambient buffer, no warm floor. A body in permanent shadow loses
# heat two ways and both are one-way: it radiates to a sky at three kelvin, and
# it conducts into ground at forty through whatever touches it.
#
# One node, lumped capacitance, stated as a choice. A real platform has a
# gradient between its battery and its feet and the interesting engineering is
# in that gradient, but a single node answers the two questions that decide
# whether the mission shape exists at all -- how long can it stay, and what does
# staying cost -- and a finite-element model would answer them no differently
# while hiding which term dominates.
#
# Conduction uses the spreading resistance of an isothermal disc on a
# half-space, so the heat through one foot is 4*k*a*dT with a the contact
# radius. That is exact for the geometry and, more usefully, it needs no
# invented path length: the "thickness" of a conductive path into semi-infinite
# ground is not a quantity anyone could have supplied honestly.
#
# The conductivity it uses is measured, and it is the reason the answer comes
# out the way it does. Lunar regolith at the surface conducts about
# 0.0015 W/(m K), roughly a tenth of silica aerogel. The ground a lunar robot
# stands on is a better insulator than anything it could be wrapped in.
#
# References
#   Vaniman D et al. (1991) The Lunar Environment. In: Lunar Sourcebook, ch. 3.
#     Cambridge University Press.
#   Paige DA et al. (2010) Diviner Lunar Radiometer observations of cold traps
#     in the Moon's south polar region. Science 330, 479-482.
#     doi:10.1126/science.1187726
#   Carslaw HS, Jaeger JC (1959) Conduction of Heat in Solids, 2nd ed. Oxford.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "STEFAN_BOLTZMANN",
    "ThermalEnvelope",
    "conductive_loss_W",
    "cooling_time_s",
    "equilibrium_temperature_K",
    "radiative_loss_W",
    "survival_power_W",
]

STEFAN_BOLTZMANN: Final = 5.670374419e-8

# An isothermal disc of radius a on a semi-infinite solid of conductivity k has
# spreading resistance 1/(4ka). Classical, exact, and free of any path length.
DISC_SPREADING_COEFFICIENT: Final = 4.0


@dataclass(frozen=True, slots=True)
class ThermalEnvelope:
    """What a platform presents to a vacuum, and what it holds.

    Every field is assumed. None of them is derived from a design, because no
    design exists, and the study that consumes this reports sensitivities rather
    than a survival time for exactly that reason.
    """

    radiating_area_m2: float
    emissivity: float
    heat_capacity_J_per_K: float
    contact_radius_m: float
    contacts: int

    def __post_init__(self) -> None:
        for name, value in (
            ("radiating_area_m2", self.radiating_area_m2),
            ("heat_capacity_J_per_K", self.heat_capacity_J_per_K),
            ("contact_radius_m", self.contact_radius_m),
        ):
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and positive, got {value}")
        if not 0.0 < self.emissivity <= 1.0:
            raise ValueError(
                "emissivity must lie in (0, 1]; zero would mean a body that "
                f"cannot radiate at all, got {self.emissivity}"
            )
        if self.contacts < 0:
            raise ValueError(f"contacts must not be negative, got {self.contacts}")


def radiative_loss_W(
    *, envelope: ThermalEnvelope, temperature_K: ArrayLike, sky_K: float
) -> NDArray[np.float64]:
    temperature = np.asarray(temperature_K, dtype=np.float64)
    return np.asarray(
        envelope.emissivity
        * STEFAN_BOLTZMANN
        * envelope.radiating_area_m2
        * (temperature**4 - sky_K**4)
    )


def conductive_loss_W(
    *,
    envelope: ThermalEnvelope,
    temperature_K: ArrayLike,
    soil_K: float,
    soil_conductivity_W_per_m_K: float,
) -> NDArray[np.float64]:
    """Heat into the ground through the contacts, by spreading resistance.

    Independent of how deep anything is buried: the resistance of a disc on a
    half-space is set by the disc's radius and the medium's conductivity alone.
    """
    temperature = np.asarray(temperature_K, dtype=np.float64)
    return np.asarray(
        envelope.contacts
        * DISC_SPREADING_COEFFICIENT
        * soil_conductivity_W_per_m_K
        * envelope.contact_radius_m
        * (temperature - soil_K)
    )


def survival_power_W(
    *,
    envelope: ThermalEnvelope,
    temperature_K: ArrayLike,
    sky_K: float,
    soil_K: float,
    soil_conductivity_W_per_m_K: float,
) -> NDArray[np.float64]:
    """Continuous power to hold a temperature: total loss, by definition."""
    return np.asarray(
        radiative_loss_W(envelope=envelope, temperature_K=temperature_K, sky_K=sky_K)
        + conductive_loss_W(
            envelope=envelope,
            temperature_K=temperature_K,
            soil_K=soil_K,
            soil_conductivity_W_per_m_K=soil_conductivity_W_per_m_K,
        )
    )


def equilibrium_temperature_K(
    *,
    envelope: ThermalEnvelope,
    internal_power_W: float,
    sky_K: float,
    soil_K: float,
    soil_conductivity_W_per_m_K: float,
) -> float:
    """Where internal dissipation balances loss, by bisection.

    Loss rises monotonically with temperature -- a quartic plus a linear term,
    both increasing -- so the balance point is unique and bracketing is safe.
    """
    if internal_power_W < 0.0:
        raise ValueError(
            f"internal_power_W must not be negative, got {internal_power_W}"
        )

    def excess(temperature: float) -> float:
        return internal_power_W - float(
            survival_power_W(
                envelope=envelope,
                temperature_K=temperature,
                sky_K=sky_K,
                soil_K=soil_K,
                soil_conductivity_W_per_m_K=soil_conductivity_W_per_m_K,
            )
        )

    low, high = min(sky_K, soil_K), 1000.0
    if excess(high) > 0.0:
        raise ValueError(
            f"{internal_power_W} W does not balance below {high} K, which is far "
            "hotter than any part of this problem; check the envelope"
        )
    for _ in range(200):
        middle = 0.5 * (low + high)
        if excess(middle) > 0.0:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def cooling_time_s(
    *,
    envelope: ThermalEnvelope,
    start_K: float,
    limit_K: float,
    internal_power_W: float,
    sky_K: float,
    soil_K: float,
    soil_conductivity_W_per_m_K: float,
    steps: int = 20000,
) -> float:
    """Time to fall from one temperature to another, or infinity if it never does.

    Integrated as dt = C dT / (loss - internal) over the interval rather than
    stepped forward in time, which needs no stability argument and puts the
    singularity where it belongs: at the temperature the platform settles to.
    """
    if limit_K >= start_K:
        raise ValueError(
            f"limit_K {limit_K} must be below start_K {start_K}; this measures "
            "cooling, not warming"
        )
    temperatures = np.linspace(start_K, limit_K, steps + 1)
    loss = survival_power_W(
        envelope=envelope,
        temperature_K=temperatures,
        sky_K=sky_K,
        soil_K=soil_K,
        soil_conductivity_W_per_m_K=soil_conductivity_W_per_m_K,
    )
    net = loss - internal_power_W
    if bool(np.any(net <= 0.0)):
        return math.inf
    midpoint = 0.5 * (net[:-1] + net[1:])
    step = np.abs(np.diff(temperatures))
    return float(envelope.heat_capacity_J_per_K * np.sum(step / midpoint))
