# SPDX-License-Identifier: Apache-2.0
#
# biome.fitting — recover contact model parameters from measured pressure-sinkage data.
#
# Two estimators of the same model. They are not equivalent on real data, and a
# published parameter rarely states which was used, so both are selectable and
# the choice is recorded on the fit alongside the weighting.
#
# The direct estimator fits all three parameters at once against the model as
# written. The staged estimator follows the standard Bekker identification
# procedure:
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
# The averaged_exponent estimator replaces step one only: it fits an exponent
# to each plate separately and takes the arithmetic mean, then evaluates each
# plate's modulus at that mean. That is the procedure Wong describes and that
# published bevameter analyses implement, and it is not the same estimator as
# sharing an exponent across a single joint fit. On the same observations under
# the same weighting the two differ by tens of percent in k_c, which is why
# both are selectable rather than one standing in for the other.
#
# Weighting is explicit because it changes the answer. Unweighted least squares
# in log space minimizes relative rather than absolute error and biases the fit
# toward small pressures; weighting each log residual by the squared pressure
# restores the linear-space criterion to first order. Published fits rarely
# state which they used, so the scheme is selectable, and recovering published
# parameters is how you find out which one was used.
#
# profile_cohesive_modulus fixes the cohesive modulus across a range and
# re-optimizes everything else, the sinkage exponent included. Holding the
# exponent at its joint value would profile a two-parameter model instead, and
# report a narrower interval than the three-parameter model supports.
#
# Fitted parameter keys match the contact model constructors, so a fit is
# directly comparable to a transcribed parameters block, key for key.
#
# References
#   Wong JY (1980) Data processing methodology in the characterization of the
#     mechanical properties of terrain. Journal of Terramechanics 17(1), 13-41.
#   Wong JY (2001) Theory of Ground Vehicles, 3rd ed. Wiley.

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from itertools import permutations
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from biome._validation import first_violation
from biome.terramechanics import CONTACT_MODELS, ContactModel

__all__ = [
    "DEFAULT_ESTIMATOR",
    "DEFAULT_WEIGHTING",
    "Estimator",
    "FittedContactModel",
    "PowerLawFit",
    "PressureSinkageObservations",
    "ProfileLikelihood",
    "WeightingScheme",
    "coefficient_of_determination",
    "coefficient_of_determination_by_plate",
    "coefficient_of_determination_ceiling",
    "fit_averaged_power_law",
    "fit_contact_model",
    "fit_shared_power_law",
    "mean_relative_residual",
    "parameter_bound_under_bias_permutation",
    "profile_cohesive_modulus",
    "relative_deviation",
]

WeightingScheme = Literal["uniform", "pressure_squared"]
DEFAULT_WEIGHTING: Final[WeightingScheme] = "pressure_squared"
Estimator = Literal["two_stage", "averaged_exponent", "direct"]
DEFAULT_ESTIMATOR: Final[Estimator] = "two_stage"
PROFILE_CONFIDENCE_LEVEL: Final = 0.95
_TWO_SIDED_NORMAL_QUANTILE: Final = 1.959963984540054
_BRACKET_STEP_LIMIT: Final = 200
_GOLDEN_SECTION_ITERATIONS: Final = 200
_DIRECT_SEARCH_SAMPLES: Final = 400
MINIMUM_PLATES_FOR_PLATE_SCALING: Final = 2
# Centring identical logarithms cancels to rounding error rather than to zero,
# so a per-plate exponent must reject on the spread relative to its own scale.
# Testing against zero lets a ratio of two noise terms through as a fit.
_LOG_SPREAD_RELATIVE_FLOOR: Final = 1e-12
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

    def above_sinkage(self, minimum_m: float) -> PressureSinkageObservations:
        selected = self.sinkage_m >= minimum_m
        return PressureSinkageObservations(
            contact_half_width_m=self.contact_half_width_m[selected],
            sinkage_m=self.sinkage_m[selected],
            pressure_kPa=self.pressure_kPa[selected],
        )

    def rescaled_by_plate(
        self, factors: Mapping[float, float]
    ) -> PressureSinkageObservations:
        missing = sorted(
            float(half_width)
            for half_width in self.contact_half_widths
            if float(half_width) not in factors
        )
        if missing:
            raise ValueError(
                f"no rescaling factor for plates {missing}; every plate present "
                "in the observations needs one"
            )
        scale = np.array(
            [factors[float(half_width)] for half_width in self.contact_half_width_m]
        )
        return PressureSinkageObservations(
            contact_half_width_m=self.contact_half_width_m,
            sinkage_m=self.sinkage_m,
            pressure_kPa=self.pressure_kPa / scale,
        )

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
    estimator: Estimator
    observation_count: int
    plate_count: int


@dataclass(frozen=True, slots=True, eq=False)
class ProfileLikelihood:
    model_id: str
    parameter: str
    values: NDArray[np.float64]
    residuals: NDArray[np.float64]
    weighting: WeightingScheme
    observation_count: int
    free_parameter_count: int

    @property
    def minimum_residual(self) -> float:
        return float(np.min(self.residuals))

    @property
    def minimum_value(self) -> float:
        return float(self.values[int(np.argmin(self.residuals))])

    def confidence_interval(self) -> tuple[float, float]:
        degrees = self.observation_count - self.free_parameter_count
        if degrees < 1:
            raise ValueError(
                f"{self.observation_count} observations cannot support "
                f"{self.free_parameter_count} parameters"
            )
        quantile = _TWO_SIDED_NORMAL_QUANTILE + (
            _TWO_SIDED_NORMAL_QUANTILE**3 + _TWO_SIDED_NORMAL_QUANTILE
        ) / (4.0 * degrees)
        threshold = self.minimum_residual * (1.0 + quantile**2 / degrees)
        inside = self.values[self.residuals <= threshold]
        if inside.size == 0:
            raise ValueError("no profiled value meets its own minimum residual")
        if inside[0] == self.values[0] or inside[-1] == self.values[-1]:
            raise ValueError(
                f"the {PROFILE_CONFIDENCE_LEVEL:.0%} interval reaches the edge of "
                f"the profiled range [{self.values[0]}, {self.values[-1]}]; widen it"
            )
        return float(inside.min()), float(inside.max())


@dataclass(frozen=True, slots=True)
class _LogLinearForm:
    cohesive_coefficient: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    frictional_coefficient: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    sinkage_regressor: Callable[
        [NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ]


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


def _plate_exponent(
    observations: PressureSinkageObservations, weighting: WeightingScheme
) -> float:
    distinct = np.unique(observations.sinkage_m)
    if distinct.size < 2:
        raise ValueError(
            f"the {observations.count} observations on this plate share one "
            f"sinkage value, {float(distinct[0])}, so a power law through them "
            "has no slope; a per-plate exponent needs at least two distinct "
            "sinkages"
        )
    weights = _log_space_weights(observations.pressure_kPa, weighting)
    log_sinkage = np.log(observations.sinkage_m)
    log_pressure = np.log(observations.pressure_kPa)
    total = np.sum(weights)
    centred_sinkage = log_sinkage - np.sum(weights * log_sinkage) / total
    centred_pressure = log_pressure - np.sum(weights * log_pressure) / total
    spread = np.sum(weights * centred_sinkage**2)
    scale = np.sum(weights * np.square(log_sinkage))
    if not spread > _LOG_SPREAD_RELATIVE_FLOOR * scale:
        raise ValueError(
            f"the {distinct.size} distinct sinkage values on this plate span "
            f"{float(distinct.min())} to {float(distinct.max())}, too narrow in "
            f"log space to carry an exponent: the weighted spread is "
            f"{float(spread)} against a scale of {float(scale)}. Centring "
            "near-identical logarithms leaves rounding error, and a slope "
            "through that is noise wearing the shape of a fit"
        )
    return float(np.sum(weights * centred_sinkage * centred_pressure) / spread)


def _plate_modulus(
    observations: PressureSinkageObservations,
    exponent: float,
    weighting: WeightingScheme,
) -> float:
    weights = _log_space_weights(observations.pressure_kPa, weighting)
    residual = np.log(observations.pressure_kPa) - exponent * np.log(
        observations.sinkage_m
    )
    return float(np.exp(np.sum(weights * residual) / np.sum(weights)))


def fit_averaged_power_law(
    observations: PressureSinkageObservations,
    *,
    weighting: WeightingScheme = DEFAULT_WEIGHTING,
) -> PowerLawFit:
    plates = observations.contact_half_widths
    by_plate = [observations.for_plate(plate) for plate in plates]
    exponent = float(
        np.mean([_plate_exponent(selected, weighting) for selected in by_plate])
    )
    return PowerLawFit(
        sinkage_exponent=exponent,
        contact_half_widths_m=plates,
        plate_moduli=np.array(
            [_plate_modulus(selected, exponent, weighting) for selected in by_plate]
        ),
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


_LOG_LINEAR_FORMS: Final[Mapping[str, _LogLinearForm]] = MappingProxyType(
    {
        "bekker": _LogLinearForm(
            cohesive_coefficient=lambda half_width: 1.0 / half_width,
            frictional_coefficient=np.ones_like,
            sinkage_regressor=lambda sinkage, half_width: np.log(sinkage),
        ),
        "reece": _LogLinearForm(
            cohesive_coefficient=np.ones_like,
            frictional_coefficient=lambda half_width: half_width,
            sinkage_regressor=lambda sinkage, half_width: np.log(sinkage / half_width),
        ),
    }
)


def _golden_section_minimum(
    objective: Callable[[float], float], lower: float, upper: float
) -> tuple[float, float]:
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left, right = upper - ratio * (upper - lower), lower + ratio * (upper - lower)
    at_left, at_right = objective(left), objective(right)
    for _ in range(_GOLDEN_SECTION_ITERATIONS):
        if at_left < at_right:
            upper, right, at_right = right, left, at_left
            left = upper - ratio * (upper - lower)
            at_left = objective(left)
        else:
            lower, left, at_left = left, right, at_right
            right = lower + ratio * (upper - lower)
            at_right = objective(right)
        if upper - lower < 1e-12 * max(1.0, abs(lower) + abs(upper)):
            break
    center = 0.5 * (lower + upper)
    return center, objective(center)


def _minimum_above(
    objective: Callable[[float], float], lower: float, step: float
) -> tuple[float, float]:
    left = lower + abs(lower) * 1e-9 + 1e-9
    middle = left + step
    at_left, at_middle = objective(left), objective(middle)
    for _ in range(_BRACKET_STEP_LIMIT):
        if at_middle > at_left:
            return _golden_section_minimum(objective, left, middle)
        right = middle + step
        at_right = objective(right)
        if at_right > at_middle:
            return _golden_section_minimum(objective, left, right)
        left, at_left, middle, at_middle = middle, at_middle, right, at_right
        step *= 2.0
    raise ValueError(
        "the residual keeps falling as the frictional modulus grows, so the fit "
        "does not bracket a minimum; the observations may not constrain it"
    )


def _log_space_residual(
    form: _LogLinearForm,
    observations: PressureSinkageObservations,
    weights: NDArray[np.float64],
    cohesive_modulus: float,
    frictional_modulus: float,
) -> tuple[float, float]:
    half_width = observations.contact_half_width_m
    modulus = cohesive_modulus * form.cohesive_coefficient(
        half_width
    ) + frictional_modulus * form.frictional_coefficient(half_width)
    if np.any(modulus <= 0.0):
        return math.inf, math.nan
    regressor = form.sinkage_regressor(observations.sinkage_m, half_width)
    spread = float(np.sum(weights * np.square(regressor)))
    if spread <= 0.0:
        raise ValueError(
            "every observation shares one sinkage, so the exponent is not "
            "identifiable; more distinct sinkage values are needed"
        )
    target = np.log(observations.pressure_kPa) - np.log(modulus)
    exponent = float(np.sum(weights * regressor * target) / spread)
    return float(np.sum(weights * np.square(target - exponent * regressor))), exponent


def _profile_at_cohesive_modulus(
    model_id: str,
    observations: PressureSinkageObservations,
    weights: NDArray[np.float64],
    cohesive_modulus: float,
) -> tuple[float, float, float]:
    form = _LOG_LINEAR_FORMS[model_id]
    half_width = observations.contact_half_width_m
    ratios = form.cohesive_coefficient(half_width) / form.frictional_coefficient(
        half_width
    )
    lower = float(np.max(-cohesive_modulus * ratios))
    frictional, residual = _minimum_above(
        lambda value: _log_space_residual(
            form, observations, weights, cohesive_modulus, value
        )[0],
        lower,
        step=max(1.0, abs(lower)),
    )
    return residual, frictional, _log_space_residual(
        form, observations, weights, cohesive_modulus, frictional
    )[1]


def profile_cohesive_modulus(
    model_id: str,
    observations: PressureSinkageObservations,
    values: NDArray[np.float64],
    *,
    weighting: WeightingScheme = DEFAULT_WEIGHTING,
) -> ProfileLikelihood:
    if model_id not in _LOG_LINEAR_FORMS:
        raise ValueError(
            f"no log-linear form is implemented for {model_id!r}; this module "
            f"profiles {sorted(_LOG_LINEAR_FORMS)}"
        )
    weights = _log_space_weights(observations.pressure_kPa, weighting)
    residuals = np.array(
        [
            _profile_at_cohesive_modulus(model_id, observations, weights, float(value))[0]
            for value in values
        ]
    )
    return ProfileLikelihood(
        model_id=model_id,
        parameter="cohesive_modulus",
        values=np.asarray(values, dtype=np.float64),
        residuals=residuals,
        weighting=weighting,
        observation_count=observations.count,
        free_parameter_count=3,
    )


def _fit_directly(
    model_id: str,
    observations: PressureSinkageObservations,
    weighting: WeightingScheme,
) -> dict[str, float]:
    weights = _log_space_weights(observations.pressure_kPa, weighting)
    staged = PLATE_SCALINGS[model_id](
        fit_shared_power_law(observations, weighting=weighting)
    )["cohesive_modulus"]
    span = max(abs(staged), 1.0) * 4.0
    grid = np.linspace(staged - span, staged + span, _DIRECT_SEARCH_SAMPLES)
    residuals = np.array(
        [
            _profile_at_cohesive_modulus(model_id, observations, weights, float(value))[0]
            for value in grid
        ]
    )
    best = int(np.argmin(residuals))
    if best in (0, grid.size - 1):
        raise ValueError(
            "the direct fit's minimum lies at the edge of the searched range "
            f"[{grid[0]}, {grid[-1]}]; the two-stage estimate is a poor start"
        )
    cohesive, _ = _golden_section_minimum(
        lambda value: _profile_at_cohesive_modulus(
            model_id, observations, weights, value
        )[0],
        float(grid[best - 1]),
        float(grid[best + 1]),
    )
    _, frictional, exponent = _profile_at_cohesive_modulus(
        model_id, observations, weights, cohesive
    )
    return {
        "cohesive_modulus": cohesive,
        "frictional_modulus": frictional,
        "sinkage_exponent": exponent,
    }


def fit_contact_model(
    model_id: str,
    observations: PressureSinkageObservations,
    *,
    weighting: WeightingScheme = DEFAULT_WEIGHTING,
    estimator: Estimator = DEFAULT_ESTIMATOR,
) -> FittedContactModel:
    if model_id not in PLATE_SCALINGS:
        raise ValueError(
            f"no plate scaling is implemented for {model_id!r}; this module fits "
            f"{sorted(PLATE_SCALINGS)}"
        )
    plate_count = int(observations.contact_half_widths.size)
    if plate_count < MINIMUM_PLATES_FOR_PLATE_SCALING:
        raise ValueError(
            f"fitting {model_id!r} needs at least {MINIMUM_PLATES_FOR_PLATE_SCALING} "
            "distinct plate sizes to separate the cohesive from the frictional "
            f"modulus, got {plate_count}"
        )
    if estimator == "direct":
        parameters = _fit_directly(model_id, observations, weighting)
    elif estimator == "averaged_exponent":
        parameters = PLATE_SCALINGS[model_id](
            fit_averaged_power_law(observations, weighting=weighting)
        )
    else:
        parameters = PLATE_SCALINGS[model_id](
            fit_shared_power_law(observations, weighting=weighting)
        )
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
        estimator=estimator,
        observation_count=observations.count,
        plate_count=plate_count,
    )


def parameter_bound_under_bias_permutation(
    model_id: str,
    observations: PressureSinkageObservations,
    biases: Sequence[float],
    *,
    weighting: WeightingScheme = DEFAULT_WEIGHTING,
) -> Mapping[str, tuple[float, float]]:
    plates = [float(half_width) for half_width in observations.contact_half_widths]
    if len(biases) != len(plates):
        raise ValueError(
            f"{len(biases)} biases for {len(plates)} plates; the permutation "
            "reassigns one measured bias per plate"
        )
    extremes: dict[str, list[float]] = {}
    for order in permutations(biases):
        parameters = fit_contact_model(
            model_id,
            observations.rescaled_by_plate(
                {plate: 1.0 + bias for plate, bias in zip(plates, order, strict=True)}
            ),
            weighting=weighting,
        ).parameters
        for name, value in parameters.items():
            extremes.setdefault(name, []).append(value)
    return MappingProxyType(
        {name: (min(values), max(values)) for name, values in extremes.items()}
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
