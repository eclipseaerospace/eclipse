# SPDX-License-Identifier: Apache-2.0
#
# eclipse.planning — the first thing in this repository that decides.
#
# Contact, cost of transport, statics, illumination and scheduling are all
# evaluation: given a route, what does it cost. Every route until now has been a
# straight line between two extrema, and a straight line is the worst route in
# rough terrain and the shortest in flat -- so the direction of that error is not
# even consistent between sites, and a comparison across sites was partly
# measuring how well a straight line happened to serve each one.
#
# What this is, stated narrowly, because "path planning" invites the reader to
# hear more than is here. This is search over a known map. There is no
# perception, no state estimation, no uncertainty and no replanning. The
# platform is not deciding what it can see; it is deciding what was already
# known. That is a rung below autonomy and several rungs below the belief-space
# version, and calling it planning is accurate only if the qualifier travels
# with it.
#
# Three things decide whether the result is honest rather than merely
# plausible.
#
# The graph is directed. Climbing and descending differ by more than a factor of
# three at the same gradient, so the return path is not the outbound path
# reversed and modelling it as one would understate a round trip. That asymmetry
# is the project's most robust locomotion result and it belongs here rather than
# being applied afterwards.
#
# Diagonal steps cost their true length. A grid that charges a diagonal as one
# step produces staircases and understates distance by up to forty percent, and
# the length that matters is along the ground rather than across the map:
# hypot(rise, run), the same convention the transect uses.
#
# And the baseline is the native posting, deliberately, because coarsening it
# manufactures passability. Measured on three of these products, the fraction of
# cells steeper than the tipping limit falls from about one in ten thousand at
# 5 m to exactly zero at 40 m. A planner on an aggregated grid would find every
# region open and would be reporting a property of the aggregation. The cost of
# that choice is a graph of ten million nodes, which turns out to be a second of
# search.
#
# Costs are non-negative by construction, which is what makes Dijkstra valid: a
# descent steep enough that gravity more than pays for the ground is clamped to
# zero rather than becoming a source. That clamp is also why A* is not used --
# the cheapest achievable step costs nothing, so the only admissible heuristic
# is the trivial one, and a heuristic worth having would have to be inadmissible.
#
# References
#   Dijkstra EW (1959) A note on two problems in connexion with graphs.
#     Numerische Mathematik 1, 269-271.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

__all__ = [
    "NEIGHBOURS",
    "Plan",
    "Route",
    "TraversalCost",
    "minimum_slope_capability_deg",
    "plan_route",
]

# Eight-connected. Four-connected forbids diagonal travel and inflates every
# distance by up to the square root of two; sixteen-connected would reduce the
# residual angular quantisation further and costs four times the edges, which
# is a refinement to make when something turns on it.
NEIGHBOURS: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


@dataclass(frozen=True, slots=True)
class TraversalCost:
    """Energy per metre of ground travelled, against signed slope.

    A table rather than a model, so this module never learns what a platform or
    a soil is. The caller evaluates its own cost of transport across a slope
    axis and hands the curve over; a synthetic curve makes the search testable
    without any physics at all.

    Positive slope climbs. Values must be non-negative -- the caller clamps a
    descent that gravity more than pays for, because a negative edge is a
    perpetual motion machine and Dijkstra would be invalid over it.
    """

    slope_deg: NDArray[np.float64]
    joules_per_metre: NDArray[np.float64]
    limit_deg: float

    def __post_init__(self) -> None:
        if self.slope_deg.shape != self.joules_per_metre.shape:
            raise ValueError(
                "the slope axis and the cost curve must have the same shape; "
                f"got {self.slope_deg.shape} and {self.joules_per_metre.shape}"
            )
        if self.slope_deg.size < 2:
            raise ValueError(
                f"a cost curve needs at least two samples; got {self.slope_deg.size}"
            )
        if bool((np.diff(self.slope_deg) <= 0.0).any()):
            raise ValueError(
                "the slope axis must increase strictly; np.interp reads it as "
                "sorted and would silently return nonsense otherwise"
            )
        negative = self.joules_per_metre < 0.0
        if bool(negative.any()):
            first = int(np.argmax(negative))
            raise ValueError(
                "cost per metre must be non-negative or the search is invalid; "
                f"it is {self.joules_per_metre[first]:.6g} J/m at "
                f"{self.slope_deg[first]:.6g} degrees, and "
                f"{int(negative.sum())} samples are negative. Clamp a free "
                "descent at zero before building the curve"
            )
        if not 0.0 < self.limit_deg < 90.0:
            raise ValueError(
                "limit_deg is the slope beyond which the platform cannot hold "
                f"and must lie strictly between 0 and 90; got {self.limit_deg}"
            )

    def at(self, slope_deg: NDArray[np.float64]) -> NDArray[np.float64]:
        """Cost per metre, infinite beyond the limit."""
        joules = np.interp(slope_deg, self.slope_deg, self.joules_per_metre)
        return np.asarray(
            np.where(np.abs(slope_deg) > self.limit_deg, np.inf, joules)
        )


@dataclass(frozen=True, slots=True)
class Route:
    """A path the platform could actually walk, and what each step of it costs."""

    rows: NDArray[np.int_]
    columns: NDArray[np.int_]
    elevation_m: NDArray[np.float64]
    step_length_m: NDArray[np.float64]
    step_slope_deg: NDArray[np.float64]
    step_energy_J: NDArray[np.float64]

    @property
    def total_energy_J(self) -> float:
        return float(self.step_energy_J.sum())

    @property
    def distance_m(self) -> float:
        return float(self.step_length_m.sum())

    @property
    def max_abs_slope_deg(self) -> float:
        return float(np.abs(self.step_slope_deg).max())

    @property
    def steps(self) -> int:
        return int(self.step_length_m.size)

    def undecidable_steps(self, *, limit_deg: float, error_deg: float) -> int:
        """Steps within the map's own slope error of the limit.

        A route threading these is a route the data cannot certify: the cells
        are passable in the product and would not be if the product were right
        to within its stated error. Reporting the count is the same habit as
        reporting a margin rather than a side.
        """
        margin = limit_deg - np.abs(self.step_slope_deg)
        return int(((margin >= 0.0) & (margin <= error_deg)).sum())


@dataclass(frozen=True, slots=True)
class Plan:
    """The outcome of a search, including the outcome of failing."""

    route: Route | None
    impassable_cell_fraction: float
    reached_cell_fraction: float
    reason: str

    @property
    def found(self) -> bool:
        return self.route is not None


def _edges(
    elevation_m: NDArray[np.float64], *, cell_size_m: float, cost: TraversalCost
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    height, width = elevation_m.shape
    index = np.arange(height * width, dtype=np.int64).reshape(height, width)
    flat = elevation_m.ravel()
    sources, targets, weights = [], [], []
    for row_step, column_step in NEIGHBOURS:
        run = cell_size_m * float(np.hypot(row_step, column_step))
        source = index[
            max(0, -row_step) : height - max(0, row_step),
            max(0, -column_step) : width - max(0, column_step),
        ].ravel()
        target = index[
            max(0, row_step) : height - max(0, -row_step),
            max(0, column_step) : width - max(0, -column_step),
        ].ravel()
        rise = flat[target] - flat[source]
        slope = np.degrees(np.arctan2(rise, run))
        length = np.hypot(rise, run)
        joules = cost.at(slope) * length
        passable = np.isfinite(joules)
        sources.append(source[passable])
        targets.append(target[passable])
        weights.append(joules[passable])
    return (
        np.concatenate(sources),
        np.concatenate(targets),
        np.concatenate(weights),
    )


def plan_route(
    *,
    elevation_m: NDArray[np.float64],
    cell_size_m: float,
    start: tuple[int, int],
    goal: tuple[int, int],
    cost: TraversalCost,
) -> Plan:
    """Least-energy path from start to goal over the grid, or why there is none.

    Dijkstra rather than A*, and the reason is in this module's header: the
    cheapest achievable step costs nothing once a free descent is clamped, so
    the only admissible heuristic is zero. Over ten million nodes the search is
    about a second, and an inadmissible heuristic would buy speed by giving up
    the guarantee that the path returned is the cheapest one.

    A grid cell is impassable when every step into it exceeds the slope limit,
    which is a property of the cell's surroundings rather than of the cell.
    """
    if elevation_m.ndim != 2:
        raise ValueError(
            "plan_route takes a two-dimensional elevation grid; got "
            f"{elevation_m.ndim} dimensions with shape {elevation_m.shape}"
        )
    height, width = elevation_m.shape
    for name, point in (("start", start), ("goal", goal)):
        if not (0 <= point[0] < height and 0 <= point[1] < width):
            raise ValueError(
                f"{name} {point} lies outside a {height} by {width} grid"
            )
    if cell_size_m <= 0.0:
        raise ValueError(f"cell_size_m must be positive; got {cell_size_m}")

    sources, targets, weights = _edges(
        elevation_m, cell_size_m=cell_size_m, cost=cost
    )
    nodes = height * width
    reachable_in = np.zeros(nodes, dtype=bool)
    reachable_in[targets] = True
    impassable = float(1.0 - reachable_in.mean())

    graph = cast(
        Any, coo_matrix((weights, (sources, targets)), shape=(nodes, nodes))
    ).tocsr()
    del sources, targets, weights

    origin = start[0] * width + start[1]
    destination = goal[0] * width + goal[1]
    found = cast(
        tuple[Any, Any],
        dijkstra(graph, directed=True, indices=origin, return_predecessors=True),
    )
    distance: NDArray[np.float64] = np.asarray(found[0], dtype=np.float64)
    predecessor: NDArray[np.int64] = np.asarray(found[1], dtype=np.int64)
    del graph, found
    reached = float(np.isfinite(distance).mean())

    if not np.isfinite(distance[destination]):
        return Plan(
            route=None,
            impassable_cell_fraction=impassable,
            reached_cell_fraction=reached,
            reason=(
                "no path exists within the slope limit; the goal is in a "
                f"component the start cannot reach, and {reached:.1%} of the "
                "grid is reachable from the start"
            ),
        )

    walk = [destination]
    while walk[-1] != origin:
        previous = int(predecessor[walk[-1]])
        if previous < 0:
            raise ValueError(
                "the predecessor chain broke before reaching the start, which "
                "means the search returned a finite distance for an "
                "unreachable node"
            )
        walk.append(previous)
    walk.reverse()

    order = np.asarray(walk, dtype=np.int64)
    rows = order // width
    columns = order % width
    elevation = elevation_m[rows, columns]
    run = cell_size_m * np.hypot(np.diff(rows), np.diff(columns))
    rise = np.diff(elevation)
    length = np.hypot(rise, run)
    slope = np.degrees(np.arctan2(rise, run))
    return Plan(
        route=Route(
            rows=rows,
            columns=columns,
            elevation_m=elevation,
            step_length_m=length,
            step_slope_deg=slope,
            step_energy_J=cost.at(slope) * length,
        ),
        impassable_cell_fraction=impassable,
        reached_cell_fraction=reached,
        reason="",
    )


def minimum_slope_capability_deg(
    *,
    elevation_m: NDArray[np.float64],
    cell_size_m: float,
    start: tuple[int, int],
    goal: tuple[int, int],
    tolerance_deg: float = 0.05,
) -> float:
    """The gentlest platform that could make this journey at all.

    A connectivity question rather than a cost one, and cheaper for it. Whether
    a step is allowed depends on the magnitude of its slope, and that magnitude
    is the same in both directions -- only the cost is asymmetric -- so the
    question of which places can be reached is undirected even though the
    question of what reaching them costs is not.

    Bisected on the limit, testing whether start and goal fall in one connected
    component. Always finite: every step on a grid of finite elevations has a
    slope short of vertical, so a platform that could climb anything could go
    anywhere, and the question is how steep rather than whether. A sealed goal
    comes back near ninety degrees, which is the same statement in the units
    the rest of the answer is in.
    """
    height, width = elevation_m.shape
    index = np.arange(height * width, dtype=np.int64).reshape(height, width)
    flat = elevation_m.ravel()
    sources, targets, steepness = [], [], []
    for row_step, column_step in NEIGHBOURS[:4]:
        run = cell_size_m * float(np.hypot(row_step, column_step))
        source = index[
            max(0, -row_step) : height - max(0, row_step),
            max(0, -column_step) : width - max(0, column_step),
        ].ravel()
        target = index[
            max(0, row_step) : height - max(0, -row_step),
            max(0, column_step) : width - max(0, -column_step),
        ].ravel()
        sources.append(source)
        targets.append(target)
        steepness.append(
            np.abs(np.degrees(np.arctan2(flat[target] - flat[source], run)))
        )
    source_index = np.concatenate(sources)
    target_index = np.concatenate(targets)
    slope = np.concatenate(steepness)
    del sources, targets, steepness

    nodes = height * width
    origin = start[0] * width + start[1]
    destination = goal[0] * width + goal[1]

    def connected(limit: float) -> bool:
        allowed = slope <= limit
        graph = cast(
            Any,
            coo_matrix(
                (
                    np.ones(int(allowed.sum()), dtype=np.int8),
                    (source_index[allowed], target_index[allowed]),
                ),
                shape=(nodes, nodes),
            ),
        ).tocsr()
        _, label = connected_components(graph, directed=False)
        return bool(label[origin] == label[destination])

    low, high = 0.0, 90.0
    while high - low > tolerance_deg:
        middle = 0.5 * (low + high)
        if connected(middle):
            high = middle
        else:
            low = middle
    return high
