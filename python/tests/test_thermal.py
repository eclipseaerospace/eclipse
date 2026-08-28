# SPDX-License-Identifier: Apache-2.0
#
# Tests for eclipse.thermal.
#
# The two loss channels are checked against their closed forms rather than
# against stored numbers, because both are textbook and neither needs a
# regression baseline: Stefan-Boltzmann for radiation, and the spreading
# resistance of a disc on a half-space for conduction.
#
# The cooling integral is checked against a case with an analytic answer -- a
# body losing heat by conduction alone cools exponentially -- which is the only
# way to know the quadrature is right rather than merely plausible.

from __future__ import annotations

import math

import pytest

from eclipse.io.soil import load_soil, regolith_conductivity_W_per_m_K
from eclipse.thermal import (
    STEFAN_BOLTZMANN,
    ThermalEnvelope,
    conductive_loss_W,
    cooling_time_s,
    equilibrium_temperature_K,
    radiative_loss_W,
    survival_power_W,
)

SKY_K = 3.0
SOIL_K = 38.0


@pytest.fixture(scope="module")
def envelope() -> ThermalEnvelope:
    return ThermalEnvelope(
        radiating_area_m2=1.0,
        emissivity=0.85,
        heat_capacity_J_per_K=45000.0,
        contact_radius_m=0.030,
        contacts=4,
    )


@pytest.fixture(scope="module")
def conductivity() -> float:
    dataset = load_soil(
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "data"
        / "soils"
        / "lunar-intercrater.toml"
    ).datasets["vaniman1991"]
    return regolith_conductivity_W_per_m_K(dataset, depth_range_cm="0-2")


# --- the two channels, against their closed forms


def test_radiation_is_stefan_boltzmann(envelope: ThermalEnvelope) -> None:
    temperature = 253.15
    assert float(
        radiative_loss_W(envelope=envelope, temperature_K=temperature, sky_K=SKY_K)
    ) == pytest.approx(
        envelope.emissivity
        * STEFAN_BOLTZMANN
        * envelope.radiating_area_m2
        * (temperature**4 - SKY_K**4)
    )


def test_conduction_is_the_spreading_resistance_of_a_disc(
    envelope: ThermalEnvelope, conductivity: float
) -> None:
    temperature = 253.15
    assert float(
        conductive_loss_W(
            envelope=envelope,
            temperature_K=temperature,
            soil_K=SOIL_K,
            soil_conductivity_W_per_m_K=conductivity,
        )
    ) == pytest.approx(
        envelope.contacts
        * 4.0
        * conductivity
        * envelope.contact_radius_m
        * (temperature - SOIL_K)
    )


def test_conduction_scales_with_contact_radius_not_area(
    envelope: ThermalEnvelope, conductivity: float
) -> None:
    # Spreading resistance goes as one over the radius, so doubling the foot
    # doubles the heat rather than quadrupling it. Anyone reasoning from contact
    # area would get this wrong by a factor of the radius.
    def loss(radius: float) -> float:
        wider = ThermalEnvelope(
            radiating_area_m2=envelope.radiating_area_m2,
            emissivity=envelope.emissivity,
            heat_capacity_J_per_K=envelope.heat_capacity_J_per_K,
            contact_radius_m=radius,
            contacts=envelope.contacts,
        )
        return float(
            conductive_loss_W(
                envelope=wider,
                temperature_K=253.15,
                soil_K=SOIL_K,
                soil_conductivity_W_per_m_K=conductivity,
            )
        )

    assert loss(0.060) / loss(0.030) == pytest.approx(2.0)


def test_no_contacts_means_no_conduction(
    envelope: ThermalEnvelope, conductivity: float
) -> None:
    floating = ThermalEnvelope(
        radiating_area_m2=1.0,
        emissivity=0.85,
        heat_capacity_J_per_K=45000.0,
        contact_radius_m=0.030,
        contacts=0,
    )
    assert float(
        conductive_loss_W(
            envelope=floating,
            temperature_K=253.15,
            soil_K=SOIL_K,
            soil_conductivity_W_per_m_K=conductivity,
        )
    ) == 0.0


# --- which channel dominates, which is the day's result


def test_radiation_swamps_conduction_by_three_orders_of_magnitude(
    envelope: ThermalEnvelope, conductivity: float
) -> None:
    # Regolith is a better insulator than aerogel, so the ground a lunar robot
    # stands on barely takes heat at all. Foot geometry is a bearing parameter
    # and not a thermal one, which is the opposite of what was expected.
    temperature = 253.15
    radiation = float(
        radiative_loss_W(envelope=envelope, temperature_K=temperature, sky_K=SKY_K)
    )
    conduction = float(
        conductive_loss_W(
            envelope=envelope,
            temperature_K=temperature,
            soil_K=SOIL_K,
            soil_conductivity_W_per_m_K=conductivity,
        )
    )
    assert radiation / conduction > 1000.0


def test_the_foot_that_would_make_conduction_matter_is_absurd(
    envelope: ThermalEnvelope, conductivity: float
) -> None:
    temperature = 253.15
    radiation = float(
        radiative_loss_W(envelope=envelope, temperature_K=temperature, sky_K=SKY_K)
    )
    radius = radiation / (
        envelope.contacts * 4.0 * conductivity * (temperature - SOIL_K)
    )
    assert radius > 10.0, (
        f"conduction would equal radiation at a contact radius of {radius:.1f} m, "
        "which is not a foot; the two channels never compete"
    )


# --- holding temperature, and failing to


def test_survival_power_is_the_total_loss(
    envelope: ThermalEnvelope, conductivity: float
) -> None:
    temperature = 253.15
    total = float(
        survival_power_W(
            envelope=envelope,
            temperature_K=temperature,
            sky_K=SKY_K,
            soil_K=SOIL_K,
            soil_conductivity_W_per_m_K=conductivity,
        )
    )
    parts = float(
        radiative_loss_W(envelope=envelope, temperature_K=temperature, sky_K=SKY_K)
    ) + float(
        conductive_loss_W(
            envelope=envelope,
            temperature_K=temperature,
            soil_K=SOIL_K,
            soil_conductivity_W_per_m_K=conductivity,
        )
    )
    assert total == pytest.approx(parts)


def test_equilibrium_is_where_dissipation_equals_loss(
    envelope: ThermalEnvelope, conductivity: float
) -> None:
    for power in (5.0, 50.0, 200.0):
        settled = equilibrium_temperature_K(
            envelope=envelope,
            internal_power_W=power,
            sky_K=SKY_K,
            soil_K=SOIL_K,
            soil_conductivity_W_per_m_K=conductivity,
        )
        assert float(
            survival_power_W(
                envelope=envelope,
                temperature_K=settled,
                sky_K=SKY_K,
                soil_K=SOIL_K,
                soil_conductivity_W_per_m_K=conductivity,
            )
        ) == pytest.approx(power, rel=1e-6)


def test_more_power_settles_hotter(
    envelope: ThermalEnvelope, conductivity: float
) -> None:
    settled = [
        equilibrium_temperature_K(
            envelope=envelope,
            internal_power_W=power,
            sky_K=SKY_K,
            soil_K=SOIL_K,
            soil_conductivity_W_per_m_K=conductivity,
        )
        for power in (1.0, 10.0, 100.0)
    ]
    assert settled == sorted(settled)


def test_cooling_matches_the_analytic_answer_when_only_conduction_acts(
    conductivity: float,
) -> None:
    # With radiation switched off by a vanishing emissivity, loss is linear in
    # temperature and the body cools exponentially, so the time between two
    # temperatures has a closed form. This is the only check that the quadrature
    # is right rather than merely smooth.
    envelope = ThermalEnvelope(
        radiating_area_m2=1.0,
        emissivity=1e-12,
        heat_capacity_J_per_K=45000.0,
        contact_radius_m=0.030,
        contacts=4,
    )
    conductance = envelope.contacts * 4.0 * conductivity * envelope.contact_radius_m
    start, limit = 273.15, 253.15
    analytic = (
        envelope.heat_capacity_J_per_K
        / conductance
        * math.log((start - SOIL_K) / (limit - SOIL_K))
    )
    computed = cooling_time_s(
        envelope=envelope,
        start_K=start,
        limit_K=limit,
        internal_power_W=0.0,
        sky_K=SKY_K,
        soil_K=SOIL_K,
        soil_conductivity_W_per_m_K=conductivity,
    )
    assert computed == pytest.approx(analytic, rel=1e-6)


def test_enough_power_means_it_never_reaches_the_limit(
    envelope: ThermalEnvelope, conductivity: float
) -> None:
    holding = float(
        survival_power_W(
            envelope=envelope,
            temperature_K=253.15,
            sky_K=SKY_K,
            soil_K=SOIL_K,
            soil_conductivity_W_per_m_K=conductivity,
        )
    )
    assert not math.isfinite(
        cooling_time_s(
            envelope=envelope,
            start_K=273.15,
            limit_K=253.15,
            internal_power_W=holding * 1.01,
            sky_K=SKY_K,
            soil_K=SOIL_K,
            soil_conductivity_W_per_m_K=conductivity,
        )
    )


def test_cooling_takes_longer_with_more_internal_power(
    envelope: ThermalEnvelope, conductivity: float
) -> None:
    times = [
        cooling_time_s(
            envelope=envelope,
            start_K=293.15,
            limit_K=253.15,
            internal_power_W=power,
            sky_K=SKY_K,
            soil_K=SOIL_K,
            soil_conductivity_W_per_m_K=conductivity,
        )
        for power in (0.0, 20.0, 60.0)
    ]
    assert times == sorted(times)


# --- guards


@pytest.mark.parametrize("emissivity", [0.0, -0.1, 1.5])
def test_an_impossible_emissivity_is_refused(emissivity: float) -> None:
    with pytest.raises(ValueError, match=r"emissivity must lie in \(0, 1\]"):
        ThermalEnvelope(
            radiating_area_m2=1.0,
            emissivity=emissivity,
            heat_capacity_J_per_K=1.0,
            contact_radius_m=0.03,
            contacts=4,
        )


def test_warming_is_not_cooling(envelope: ThermalEnvelope, conductivity: float) -> None:
    with pytest.raises(ValueError, match="measures cooling, not warming"):
        cooling_time_s(
            envelope=envelope,
            start_K=253.15,
            limit_K=273.15,
            internal_power_W=0.0,
            sky_K=SKY_K,
            soil_K=SOIL_K,
            soil_conductivity_W_per_m_K=conductivity,
        )
