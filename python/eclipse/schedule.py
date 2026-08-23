# SPDX-License-Identifier: Apache-2.0
#
# eclipse.schedule — when a sortie departs, and what that costs.
#
# Every result before this one answers where and how far. None answers when,
# and Day 9 established that when is not a detail: a seven-hour round trip is a
# hundredth of a lunation, and only the final approach and the dwell are in
# permanent shadow. A sortie timed into the lit part of the cycle spends almost
# none of itself cold; a badly timed one spends all of it.
#
# So survival cost is not a property of a route. It is a property of a route and
# a departure time together, and every energy number in this project until now
# charged the worst case.
#
# Two things live here. Shadowed hours, which integrates darkness along a route
# for a given departure -- is this place dark at the time the platform is
# standing on it -- and an operating cycle, which walks a lunation and counts
# how many sorties fit.
#
# The cycle is a greedy pass, deliberately, and it is a description rather than
# a plan: go when the battery allows it, otherwise charge, otherwise wait. It
# reports where the hours went, which is what answers whether the limit is the
# battery, the charge rate or the availability of departures. Optimising a
# campaign is a different problem and is not this one.
#
# Hours are hours throughout, measured from an arbitrary epoch that only has to
# be consistent with the illumination series they index.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "OperatingCycle",
    "arrival_weights_h",
    "run_lunation",
    "shadowed_hours",
]


def arrival_weights_h(elapsed_hours: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hours of the traverse each route sample stands for, summing to the walk.

    Trapezoidal: a sample owns half the gap on either side of it. Whether a
    place is dark is known at the samples, so the duration spent in its state
    has to be attributed to them, and the ends own one half-gap rather than two.
    """
    if elapsed_hours.ndim != 1 or elapsed_hours.size < 2:
        raise ValueError(
            "arrival_weights_h needs at least two route samples in one "
            f"dimension; got shape {elapsed_hours.shape}"
        )
    gaps = np.diff(elapsed_hours)
    if bool((gaps < 0.0).any()):
        first = int(np.argmax(gaps < 0.0))
        raise ValueError(
            "elapsed_hours must not decrease along the route; it falls from "
            f"{elapsed_hours[first]:.6g} to {elapsed_hours[first + 1]:.6g} at "
            f"sample {first + 1}, and {int((gaps < 0.0).sum())} gaps do"
        )
    weights = np.zeros_like(elapsed_hours)
    weights[:-1] += 0.5 * gaps
    weights[1:] += 0.5 * gaps
    return weights


def shadowed_hours(
    *,
    dark: NDArray[np.bool_],
    hours: NDArray[np.float64],
    elapsed_hours: NDArray[np.float64],
    departure_hours: NDArray[np.float64],
    dwell_hours: float,
) -> NDArray[np.float64]:
    """Hours the platform spends in darkness, for each departure time.

    dark is indexed point-major over the same route samples elapsed_hours
    describes, against the same clock as hours. The platform passes each sample
    twice, outbound and inbound, and the state of a place is read at the time it
    is actually stood on rather than at departure.

    Dwell is charged at the destination for its own duration, which is why a
    permanently shadowed destination makes part of the cost immovable however
    well the traverse is timed.
    """
    if dark.shape[0] != elapsed_hours.size:
        raise ValueError(
            "dark must carry one row per route sample; got "
            f"{dark.shape[0]} rows against {elapsed_hours.size} samples"
        )
    if dark.shape[1] != hours.size:
        raise ValueError(
            "dark must carry one column per time sample; got "
            f"{dark.shape[1]} columns against {hours.size} hours"
        )
    if dwell_hours < 0.0:
        raise ValueError(f"dwell_hours must not be negative; got {dwell_hours}")

    step = float(np.mean(np.diff(hours)))
    walk = float(elapsed_hours[-1])
    weights = arrival_weights_h(elapsed_hours)
    points = np.arange(elapsed_hours.size)

    outbound = departure_hours[:, None] + elapsed_hours[None, :]
    inbound = (
        departure_hours[:, None] + 2.0 * walk + dwell_hours - elapsed_hours[None, :]
    )
    dwelling = departure_hours + walk + 0.5 * dwell_hours

    def state(times: NDArray[np.float64], at: NDArray[np.int_]) -> NDArray[np.bool_]:
        index = np.rint((times - hours[0]) / step)
        if bool((index < 0).any()) or bool((index > hours.size - 1).any()):
            worst = float(times.ravel()[int(np.argmax(np.abs(index - (hours.size - 1) / 2)))])
            raise ValueError(
                "a sortie runs past the illumination series it is scheduled "
                f"against; the series covers {hours[0]:.6g} to "
                f"{hours[-1]:.6g} h and a visit falls at {worst:.6g} h. Extend "
                "the series by at least one sortie duration beyond the last "
                "departure"
            )
        return np.asarray(dark[at, index.astype(int)])

    traverse = (weights[None, :] * state(outbound, points[None, :])).sum(axis=1)
    traverse += (weights[None, :] * state(inbound, points[None, :])).sum(axis=1)
    at_rest = dwell_hours * state(
        dwelling[:, None], np.full((departure_hours.size, 1), elapsed_hours.size - 1)
    ).ravel()
    return np.asarray(traverse + at_rest)


@dataclass(frozen=True, slots=True)
class OperatingCycle:
    """One lunation walked greedily, and where the hours went."""

    sorties: int
    sortie_hours: float
    charging_hours: float
    waiting_hours: float

    def __post_init__(self) -> None:
        for name in ("sortie_hours", "charging_hours", "waiting_hours"):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must not be negative; got {value}")
        if self.sorties < 0:
            raise ValueError(f"sorties must not be negative; got {self.sorties}")

    @property
    def limited_by(self) -> str:
        if self.sorties == 0:
            return "battery"
        if self.waiting_hours > self.charging_hours:
            return "departure windows"
        if self.charging_hours > self.sortie_hours:
            return "charge rate"
        return "sortie duration"


def run_lunation(
    *,
    hours: NDArray[np.float64],
    energy_Wh: NDArray[np.float64],
    sortie_hours: float,
    charge_W: NDArray[np.float64],
    battery_Wh: float,
    lunation_hours: float,
) -> OperatingCycle:
    """Count the sorties that fit in a lunation, and say what stopped more.

    Greedy and unoptimised: depart the moment the stored energy covers the
    sortie that would start now, otherwise charge on whatever the array is
    making at this instant, otherwise -- battery already full and the sortie
    still unaffordable -- wait for the cost to come down. Those last two are the
    distinction that matters, because one is answered by a bigger array and the
    other only by a bigger battery.

    energy_Wh and charge_W are given per time sample, so a departure's cost and
    the array's output both follow the same clock as the illumination they came
    from.
    """
    if not (hours.size == energy_Wh.size == charge_W.size):
        raise ValueError(
            "hours, energy_Wh and charge_W must describe the same samples; got "
            f"{hours.size}, {energy_Wh.size} and {charge_W.size}"
        )
    if battery_Wh <= 0.0:
        raise ValueError(f"battery_Wh must be positive; got {battery_Wh}")
    if sortie_hours <= 0.0:
        raise ValueError(f"sortie_hours must be positive; got {sortie_hours}")

    step = float(np.mean(np.diff(hours)))
    stored = battery_Wh
    clock = 0.0
    sorties = 0
    flying = 0.0
    charging = 0.0
    waiting = 0.0

    while clock + sortie_hours <= lunation_hours:
        index = min(int(round(clock / step)), hours.size - 1)
        cost = float(energy_Wh[index])
        if cost <= stored:
            stored -= cost
            clock += sortie_hours
            flying += sortie_hours
            sorties += 1
            continue
        if stored >= battery_Wh:
            waiting += step
        else:
            charging += step
            stored = min(battery_Wh, stored + float(charge_W[index]) * step)
        clock += step

    return OperatingCycle(
        sorties=sorties,
        sortie_hours=flying,
        charging_hours=charging,
        waiting_hours=waiting,
    )
