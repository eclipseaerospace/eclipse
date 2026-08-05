# SPDX-License-Identifier: Apache-2.0
#
# biome.fitting — recover contact model parameters from measured pressure-sinkage data.
#
# Two stages, following the standard Bekker identification procedure:
#
#   1. One weighted linear least squares in log space fits a single sinkage
#      exponent shared across every plate, together with one modulus per plate
#      in pressure = modulus * sinkage^exponent. Sharing the exponent is what
#      makes the per-plate moduli comparable, and what published fits report.
#   2. The per-plate moduli are regressed against plate size. Bekker regresses
#      them on 1/half_width, giving slope k_c and intercept k_phi. Reece
#      rescales by half_width^exponent and regresses on half_width, giving
#      intercept k_c and slope k_phi.
#
# Weighting is explicit because it changes the answer. Unweighted least squares
# in log space minimises relative rather than absolute error and biases the fit
# toward small pressures; weighting each log residual by the squared pressure
# restores the linear-space criterion to first order. Published fits rarely
# state which they used, so the scheme is selectable, and recovering published
# parameters is how you find out which one was used.
#
# Fitted parameter keys match the contact model constructors, so a fit is
# directly comparable to a transcribed parameters block, key for key.
#
# References
#   Wong JY (1980) Data processing methodology in the characterization of the
#     mechanical properties of terrain. Journal of Terramechanics 17(1), 13-41.
#   Wong JY (2001) Theory of Ground Vehicles, 3rd ed. Wiley.

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from biome._validation import first_violation
from biome.terramechanics import CONTACT_MODELS, ContactModel

__all__ = [
    "DEFAULT_WEIGHTING",
    "FittedContactModel",
    "PowerLawFit",
    "PressureSinkageObservations",
    "WeightingScheme",
    "coefficient_of_determination",
    "coefficient_of_determination_by_plate",
    "coefficient_of_determination_ceiling",
    "fit_contact_model",
    "fit_shared_power_law",
    "mean_relative_residual",
    "relative_deviation",
]

WeightingScheme = Literal["uniform", "pressure_squared"]
DEFAULT_WEIGHTING: Final[WeightingScheme] = "pressure_squared"
MINIMUM_PLATES_FOR_PLATE_SCALING: Final = 2
DEFAULT_REPLICATE_TOLERANCE_M: Final = 2e-3


@dataclass(frozen=True, slots=True, eq=False)
class PressureSinkageObservations:
    contact_half_width_m: NDArray[np.float64]
    sinkage_m: NDArray[np.float64]
    pressure_kPa: NDArray[np.float64]

    def __post_init__(self) -> None:
        columns = {
            "contact_half_width_m": self.contact_half_width_m,
            "sinkage_m": self.sinkage_m,
            "pressure_kPa": self.pressure_kPa,
        }
        shapes = {name: array.shape for name, array in columns.items()}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"observation columns must share one shape, got {shapes}")
        for name, array in columns.items():
            if array.ndim != 1:
                raise ValueError(
                    f"{name} must be one-dimensional, got shape {array.shape}"
                )
            violations = np.asarray(~((array > 0.0) & np.isfinite(array)))
            if violations.any():
                count, total, first = first_violation(violations, array)
                raise ValueError(
                    f"{name} must be finite and strictly positive for a log-space "
                    f"fit; {count} of {total} values violate this, the first being "
                    f"{first}. Points at zero sinkage or zero pressure carry no "
                    "information about a power law and must be excluded by the "
                    "caller rather than dropped silently here"
                )
        if self.count < 2:
            raise ValueError(f"a fit needs at least two observations, got {self.count}")

    @property
    def count(self) -> int:
        return int(self.sinkage_m.size)

    @property
    def contact_half_widths(self) -> NDArray[np.float64]:
        return np.unique(self.contact_half_width_m)

    def for_plate(self, contact_half_width: float) -> PressureSinkageObservations:
        selected = self.contact_half_width_m == contact_half_width
        return PressureSinkageObservations(
            contact_half_width_m=self.contact_half_width_m[selected],
            sinkage_m=self.sinkage_m[selected],
            pressure_kPa=self.pressure_kPa[selected],
        )


@dataclass(frozen=True, slots=True, eq=False)
class PowerLawFit:
    sinkage_exponent: float
    contact_half_widths_m: NDArray[np.float64]
    plate_moduli: NDArray[np.float64]
    weighting: WeightingScheme
    observation_count: int


@dataclass(frozen=True, slots=True, eq=False)
class FittedContactModel:
    model_id: str
    parameters: Mapping[str, float]
    model: ContactModel
    weighting: WeightingScheme
    observation_count: int
    plate_count: int


def _log_space_weights(
    pressure_kPa: NDArray[np.float64], weighting: WeightingScheme
) -> NDArray[np.float64]:
    if weighting == "uniform":
        return np.ones_like(pressure_kPa)
    return np.square(pressure_kPa)


def fit_shared_power_law(
    observations: PressureSinkageObservations,
    *,
    weighting: WeightingScheme = DEFAULT_WEIGHTING,
) -> PowerLawFit:
    plates = observations.contact_half_widths
    design = np.zeros((observations.count, 1 + plates.size), dtype=np.float64)
    design[:, 0] = np.log(observations.sinkage_m)
    for column, plate in enumerate(plates):
        design[:, 1 + column] = observations.contact_half_width_m == plate

    root_weights = np.sqrt(_log_space_weights(observations.pressure_kPa, weighting))
    solution, _, rank, _ = np.linalg.lstsq(
        design * root_weights[:, np.newaxis],
        np.log(observations.pressure_kPa) * root_weights,
        rcond=None,
    )
    if rank < design.shape[1]:
        raise ValueError(
            f"the design matrix has rank {rank} of {design.shape[1]}, so a shared "
            "sinkage exponent and one modulus per plate are not jointly "
            "identifiable from these observations; more distinct sinkage values "
            "are needed"
        )
    return PowerLawFit(
        sinkage_exponent=float(solution[0]),
        contact_half_widths_m=plates,
        plate_moduli=np.exp(solution[1:]),
        weighting=weighting,
        observation_count=observations.count,
    )


def _least_squares_line(
    abscissa: NDArray[np.float64], ordinate: NDArray[np.float64]
) -> tuple[float, float]:
    design = np.column_stack([abscissa, np.ones_like(abscissa)])
    solution, _, rank, _ = np.linalg.lstsq(design, ordinate, rcond=None)
    if rank < 2:
        raise ValueError(
            "plate scaling needs at least two distinct plate sizes to separate a "
            "slope from an intercept"
        )
    return float(solution[0]), float(solution[1])


def _bekker_parameters(power_law: PowerLawFit) -> dict[str, float]:
    slope, intercept = _least_squares_line(
        1.0 / power_law.contact_half_widths_m, power_law.plate_moduli
    )
    return {
        "cohesive_modulus": slope,
        "frictional_modulus": intercept,
        "sinkage_exponent": power_law.sinkage_exponent,
    }


def _reece_parameters(power_law: PowerLawFit) -> dict[str, float]:
    rescaled = power_law.plate_moduli * np.power(
        power_law.contact_half_widths_m, power_law.sinkage_exponent
    )
    slope, intercept = _least_squares_line(power_law.contact_half_widths_m, rescaled)
    return {
        "cohesive_modulus": intercept,
        "frictional_modulus": slope,
        "sinkage_exponent": power_law.sinkage_exponent,
    }


PLATE_SCALINGS: Final[Mapping[str, Callable[[PowerLawFit], dict[str, float]]]] = (
    MappingProxyType(
        {
            "bekker": _bekker_parameters,
            "reece": _reece_parameters,
        }
    )
)


def fit_contact_model(
    model_id: str,
    observations: PressureSinkageObservations,
    *,
    weighting: WeightingScheme = DEFAULT_WEIGHTING,
) -> FittedContactModel:
    if model_id not in PLATE_SCALINGS:
        raise ValueError(
            f"no plate scaling is implemented for {model_id!r}; this module fits "
            f"{sorted(PLATE_SCALINGS)}"
        )
    power_law = fit_shared_power_law(observations, weighting=weighting)
    plate_count = int(power_law.contact_half_widths_m.size)
    if plate_count < MINIMUM_PLATES_FOR_PLATE_SCALING:
        raise ValueError(
            f"fitting {model_id!r} needs at least {MINIMUM_PLATES_FOR_PLATE_SCALING} "
            "distinct plate sizes to separate the cohesive from the frictional "
            f"modulus, got {plate_count}"
        )
    parameters = PLATE_SCALINGS[model_id](power_law)
    try:
        model = CONTACT_MODELS[model_id](**parameters)
    except ValueError as error:
        raise ValueError(
            f"the fit produced parameters {model_id!r} rejects, {parameters}: {error}"
        ) from error
    return FittedContactModel(
        model_id=model_id,
        parameters=MappingProxyType(parameters),
        model=model,
        weighting=weighting,
        observation_count=observations.count,
        plate_count=plate_count,
    )


def relative_deviation(
    reference: ContactModel,
    other: ContactModel,
    *,
    sinkage: ArrayLike,
    contact_half_width: ArrayLike,
) -> NDArray[np.float64]:
    reference_pressure = reference.pressure(
        sinkage=sinkage, contact_half_width=contact_half_width
    )
    if np.any(reference_pressure == 0.0):
        raise ValueError(
            "relative deviation is undefined where the reference pressure is "
            "zero, which is the case at zero sinkage; evaluate at strictly "
            "positive sinkage"
        )
    other_pressure = other.pressure(
        sinkage=sinkage, contact_half_width=contact_half_width
    )
    return np.asarray((other_pressure - reference_pressure) / reference_pressure)


def mean_relative_residual(
    model: ContactModel, observations: PressureSinkageObservations
) -> float:
    predicted = model.pressure(
        sinkage=observations.sinkage_m,
        contact_half_width=observations.contact_half_width_m,
    )
    return float(np.mean((observations.pressure_kPa - predicted) / predicted))


def coefficient_of_determination_ceiling(
    observations: PressureSinkageObservations,
    *,
    sinkage_tolerance_m: float = DEFAULT_REPLICATE_TOLERANCE_M,
) -> float:
    measured = observations.pressure_kPa
    total = float(np.sum(np.square(measured - np.mean(measured))))
    if total == 0.0:
        return float("nan")

    pure_error = 0.0
    for half_width in observations.contact_half_widths:
        plate = observations.for_plate(float(half_width))
        order = np.argsort(plate.sinkage_m)
        sinkage, pressure = plate.sinkage_m[order], plate.pressure_kPa[order]
        start = 0
        for index in range(1, sinkage.size + 1):
            if (
                index == sinkage.size
                or sinkage[index] - sinkage[start] > sinkage_tolerance_m
            ):
                group = pressure[start:index]
                pure_error += float(np.sum(np.square(group - np.mean(group))))
                start = index
    return 1.0 - pure_error / total


def coefficient_of_determination(
    model: ContactModel, observations: PressureSinkageObservations
) -> float:
    predicted = model.pressure(
        sinkage=observations.sinkage_m,
        contact_half_width=observations.contact_half_width_m,
    )
    measured = observations.pressure_kPa
    residual = float(np.sum(np.square(measured - predicted)))
    total = float(np.sum(np.square(measured - np.mean(measured))))
    if total == 0.0:
        return float("nan")
    return 1.0 - residual / total


def coefficient_of_determination_by_plate(
    model: ContactModel, observations: PressureSinkageObservations
) -> Mapping[float, float]:
    return MappingProxyType(
        {
            float(half_width): coefficient_of_determination(
                model, observations.for_plate(float(half_width))
            )
            for half_width in observations.contact_half_widths
        }
    )
