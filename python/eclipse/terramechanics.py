# SPDX-License-Identifier: Apache-2.0
#
# eclipse.terramechanics — quasi-static pressure-sinkage contact models.
#
# Pure and array-first: no I/O, no unit conversion, no parameter loading.
# Both models are monotone power laws in sinkage whose deformation modulus
# depends on the contact half-width. They differ in that dependence, and in
# whether sinkage is normalized by the half-width.
#
# Finite input yields finite output, or raises. Values outside a model's fitted
# validity range are the loader's concern, not this layer's.
#
# Units are those of the fitted parameters and are never converted here. With
# parameters as published for KLS-1, sinkage in meters yields pressure in kPa.
#
# References
#   Bekker MG (1956) Theory of Land Locomotion. University of Michigan Press.
#   Reece AR (1965) Principles of soil-vehicle mechanics. Proceedings of the
#     Institution of Mechanical Engineers: Automobile Division 180(1), 45-66.
#   Wong JY (2001) Theory of Ground Vehicles, 3rd ed. Wiley.

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from eclipse._validation import first_violation

__all__ = [
    "CONTACT_MODELS",
    "BekkerModel",
    "ContactModel",
    "DegenerateContactModelError",
    "InvertibleHalfWidthRange",
    "ReeceModel",
]


class DegenerateContactModelError(ValueError):
    pass


@runtime_checkable
class ContactModel(Protocol):
    def deformation_modulus(
        self, contact_half_width: ArrayLike
    ) -> NDArray[np.float64]: ...

    def minimum_invertible_half_width(self) -> float: ...

    def invertible_half_width_range(self) -> InvertibleHalfWidthRange: ...

    def pressure(
        self, *, sinkage: ArrayLike, contact_half_width: ArrayLike
    ) -> NDArray[np.float64]: ...

    def sinkage(
        self, *, pressure: ArrayLike, contact_half_width: ArrayLike
    ) -> NDArray[np.float64]: ...


def _as_finite_non_negative(values: ArrayLike, quantity: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    violations = np.asarray(~((array >= 0.0) & np.isfinite(array)))
    if violations.any():
        count, total, first = first_violation(violations, array)
        raise ValueError(
            f"{quantity} must be finite and non-negative; {count} of {total} "
            f"values violate this, the first being {first}"
        )
    return array


def _as_finite_positive_half_width(values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    violations = np.asarray(~((array > 0.0) & np.isfinite(array)))
    if violations.any():
        count, total, first = first_violation(violations, array)
        raise ValueError(
            f"contact_half_width must be finite and positive; {count} of "
            f"{total} values violate this, the first being {first}"
        )
    return array


def _require_finite_parameter(value: float, quantity: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{quantity} must be finite, got {value}")


def _require_invertible_sinkage_exponent(value: float) -> None:
    if not math.isfinite(value) or value <= 0.0 or not math.isfinite(1.0 / value):
        raise ValueError(
            "sinkage_exponent must be finite and positive with a finite "
            "reciprocal, because sinkage() raises pressure to the power "
            f"1/sinkage_exponent, got {value}"
        )


def _require_finite_positive_modulus(
    deformation_modulus: NDArray[np.float64], contact_half_width: NDArray[np.float64]
) -> NDArray[np.float64]:
    violations = np.asarray(
        ~((deformation_modulus > 0.0) & np.isfinite(deformation_modulus))
    )
    if violations.any():
        count, total, first = first_violation(violations, contact_half_width)
        raise DegenerateContactModelError(
            "deformation modulus must be finite and positive for pressure to "
            f"increase with sinkage; {count} of {total} contact_half_width "
            f"values violate this, the first being {first}; see "
            "invertible_half_width_range(), which is bounded below when the "
            "cohesive modulus is negative and above when the frictional one is"
        )
    return deformation_modulus


@dataclass(frozen=True, slots=True)
class InvertibleHalfWidthRange:
    minimum: float
    maximum: float

    @property
    def is_empty(self) -> bool:
        return not self.minimum < self.maximum

    def contains(self, half_width: float) -> bool:
        return self.minimum < half_width < self.maximum


def _invertible_half_width_range(
    cohesive_modulus: float, frictional_modulus: float
) -> InvertibleHalfWidthRange:
    # Both model forms carry a two-term modulus that is monotone in half-width,
    # so it crosses zero at most once, and always at -cohesive/frictional. Which
    # side of that crossing is usable depends on the signs, and a fit reports
    # nothing about which side its own plates were on.
    if cohesive_modulus >= 0.0 and frictional_modulus >= 0.0:
        empty = cohesive_modulus == 0.0 and frictional_modulus == 0.0
        return InvertibleHalfWidthRange(
            math.inf if empty else 0.0, math.inf
        )
    if cohesive_modulus <= 0.0 and frictional_modulus <= 0.0:
        return InvertibleHalfWidthRange(math.inf, math.inf)
    boundary = -cohesive_modulus / frictional_modulus
    if frictional_modulus > 0.0:
        return InvertibleHalfWidthRange(boundary, math.inf)
    return InvertibleHalfWidthRange(0.0, boundary)


@dataclass(frozen=True, slots=True)
class BekkerModel:
    cohesive_modulus: float
    frictional_modulus: float
    sinkage_exponent: float

    def __post_init__(self) -> None:
        _require_finite_parameter(self.cohesive_modulus, "cohesive_modulus")
        _require_finite_parameter(self.frictional_modulus, "frictional_modulus")
        _require_invertible_sinkage_exponent(self.sinkage_exponent)

    def _unchecked_deformation_modulus(
        self, half_width: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self.cohesive_modulus / half_width + self.frictional_modulus

    def deformation_modulus(
        self, contact_half_width: ArrayLike
    ) -> NDArray[np.float64]:
        return self._unchecked_deformation_modulus(
            _as_finite_positive_half_width(contact_half_width)
        )

    def invertible_half_width_range(self) -> InvertibleHalfWidthRange:
        return _invertible_half_width_range(
            self.cohesive_modulus, self.frictional_modulus
        )

    def minimum_invertible_half_width(self) -> float:
        return self.invertible_half_width_range().minimum

    def pressure(
        self, *, sinkage: ArrayLike, contact_half_width: ArrayLike
    ) -> NDArray[np.float64]:
        depth = _as_finite_non_negative(sinkage, "sinkage")
        half_width = _as_finite_positive_half_width(contact_half_width)
        modulus = _require_finite_positive_modulus(
            self._unchecked_deformation_modulus(half_width), half_width
        )
        return modulus * np.power(depth, self.sinkage_exponent)

    def sinkage(
        self, *, pressure: ArrayLike, contact_half_width: ArrayLike
    ) -> NDArray[np.float64]:
        load = _as_finite_non_negative(pressure, "pressure")
        half_width = _as_finite_positive_half_width(contact_half_width)
        modulus = _require_finite_positive_modulus(
            self._unchecked_deformation_modulus(half_width), half_width
        )
        return np.power(load / modulus, 1.0 / self.sinkage_exponent)


@dataclass(frozen=True, slots=True)
class ReeceModel:
    cohesive_modulus: float
    frictional_modulus: float
    sinkage_exponent: float

    def __post_init__(self) -> None:
        _require_finite_parameter(self.cohesive_modulus, "cohesive_modulus")
        _require_finite_parameter(self.frictional_modulus, "frictional_modulus")
        _require_invertible_sinkage_exponent(self.sinkage_exponent)

    def _unchecked_deformation_modulus(
        self, half_width: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self.cohesive_modulus + half_width * self.frictional_modulus

    def deformation_modulus(
        self, contact_half_width: ArrayLike
    ) -> NDArray[np.float64]:
        return self._unchecked_deformation_modulus(
            _as_finite_positive_half_width(contact_half_width)
        )

    def invertible_half_width_range(self) -> InvertibleHalfWidthRange:
        return _invertible_half_width_range(
            self.cohesive_modulus, self.frictional_modulus
        )

    def minimum_invertible_half_width(self) -> float:
        return self.invertible_half_width_range().minimum

    def pressure(
        self, *, sinkage: ArrayLike, contact_half_width: ArrayLike
    ) -> NDArray[np.float64]:
        depth = _as_finite_non_negative(sinkage, "sinkage")
        half_width = _as_finite_positive_half_width(contact_half_width)
        modulus = _require_finite_positive_modulus(
            self._unchecked_deformation_modulus(half_width), half_width
        )
        return modulus * np.power(depth / half_width, self.sinkage_exponent)

    def sinkage(
        self, *, pressure: ArrayLike, contact_half_width: ArrayLike
    ) -> NDArray[np.float64]:
        load = _as_finite_non_negative(pressure, "pressure")
        half_width = _as_finite_positive_half_width(contact_half_width)
        modulus = _require_finite_positive_modulus(
            self._unchecked_deformation_modulus(half_width), half_width
        )
        return half_width * np.power(load / modulus, 1.0 / self.sinkage_exponent)


CONTACT_MODELS: Final[Mapping[str, Callable[..., ContactModel]]] = MappingProxyType(
    {
        "bekker": BekkerModel,
        "reece": ReeceModel,
    }
)
