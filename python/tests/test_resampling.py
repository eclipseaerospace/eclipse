# SPDX-License-Identifier: Apache-2.0
#
# Tests for biome.resampling.
#
# The interpolant is checked against the properties that define it rather than
# against stored numbers: it passes through every sample, its first and second
# derivatives are continuous across interior knots, its third derivative stays
# continuous across the second and second-to-last knots, and it reproduces any
# cubic exactly. Those four together admit only the not-a-knot spline, so a
# wrong end condition or a transposed coefficient fails one of them.

from __future__ import annotations

import numpy as np
import pytest

from biome.resampling import (
    MINIMUM_SPLINE_SAMPLES,
    ensemble,
    resample_cubic,
)

POSITIONS = np.array([0.0, 0.4, 1.1, 1.9, 3.0, 4.4, 5.0])
UNEVEN = np.array([0.0, 0.13, 0.9, 2.7, 2.95, 4.0, 6.1, 6.4])


SMOOTH = np.exp(-POSITIONS) * np.cos(POSITIONS)
PIECE_STEP = 0.05


def _piece_through(knot: float, side: int) -> np.polynomial.Polynomial:
    offsets = PIECE_STEP * np.arange(1, 5)
    positions = np.sort(knot + side * offsets)
    sampled = resample_cubic(
        sample_positions=POSITIONS, sample_values=SMOOTH, positions=positions
    )
    return np.polynomial.Polynomial.fit(positions, sampled, 3).convert()


def test_the_spline_passes_through_every_sample() -> None:
    values = np.sin(POSITIONS) + 0.3 * POSITIONS
    interpolated = resample_cubic(
        sample_positions=POSITIONS, sample_values=values, positions=POSITIONS
    )
    np.testing.assert_allclose(interpolated, values, rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize(
    "coefficients",
    [(2.0, -1.0, 0.5, 0.25), (0.0, 3.0, -2.0, 1.0), (-1.5, 0.0, 0.0, 2.0)],
)
def test_a_cubic_is_reproduced_exactly(coefficients: tuple[float, ...]) -> None:
    polynomial = np.polynomial.Polynomial(coefficients)
    dense = np.linspace(-1.0, 7.0, 401)
    interpolated = resample_cubic(
        sample_positions=UNEVEN,
        sample_values=polynomial(UNEVEN),
        positions=dense,
    )
    np.testing.assert_allclose(interpolated, polynomial(dense), rtol=1e-9, atol=1e-9)


def test_a_quartic_is_not_reproduced_exactly() -> None:
    polynomial = np.polynomial.Polynomial((0.0, 0.0, 0.0, 0.0, 1.0))
    dense = np.linspace(0.0, 6.0, 201)
    interpolated = resample_cubic(
        sample_positions=UNEVEN, sample_values=polynomial(UNEVEN), positions=dense
    )
    assert not np.allclose(interpolated, polynomial(dense), rtol=1e-6, atol=1e-6), (
        "a cubic spline that reproduced a quartic would not be interpolating, "
        "so this guards the previous test against passing vacuously"
    )


@pytest.mark.parametrize("order", [1, 2])
def test_the_first_two_derivatives_are_continuous_at_interior_knots(
    order: int,
) -> None:
    for knot in POSITIONS[1:-1]:
        left = _piece_through(float(knot), -1).deriv(order)(knot)
        right = _piece_through(float(knot), +1).deriv(order)(knot)
        assert left == pytest.approx(right, rel=1e-8, abs=1e-8)


def test_the_third_derivative_is_continuous_across_the_not_a_knot_joints() -> None:
    for knot in (POSITIONS[1], POSITIONS[-2]):
        left = _piece_through(float(knot), -1).deriv(3)(knot)
        right = _piece_through(float(knot), +1).deriv(3)(knot)
        assert left == pytest.approx(right, rel=1e-7, abs=1e-7), (
            "not-a-knot means the second and second-to-last knots are not "
            "breakpoints, so the third derivative may not jump there"
        )


def test_the_third_derivative_does_jump_at_a_genuine_knot() -> None:
    knot = float(POSITIONS[3])
    left = _piece_through(knot, -1).deriv(3)(knot)
    right = _piece_through(knot, +1).deriv(3)(knot)
    assert abs(left - right) > 1e-3, (
        "an interior knot that is a genuine breakpoint must show a third "
        "derivative jump, or the previous test proves nothing"
    )


def test_evaluation_beyond_the_samples_continues_the_end_piece() -> None:
    polynomial = np.polynomial.Polynomial((1.0, 0.5, -0.25, 0.125))
    beyond = np.array([-2.0, 8.0])
    interpolated = resample_cubic(
        sample_positions=UNEVEN, sample_values=polynomial(UNEVEN), positions=beyond
    )
    np.testing.assert_allclose(interpolated, polynomial(beyond), rtol=1e-8, atol=1e-8)


def test_too_few_samples_are_refused() -> None:
    positions = np.arange(float(MINIMUM_SPLINE_SAMPLES - 1))
    with pytest.raises(ValueError, match="at least 4"):
        resample_cubic(
            sample_positions=positions,
            sample_values=positions,
            positions=positions,
        )


@pytest.mark.parametrize(
    "positions",
    [
        np.array([0.0, 1.0, 1.0, 2.0, 3.0]),
        np.array([0.0, 2.0, 1.0, 3.0, 4.0]),
    ],
)
def test_non_increasing_positions_are_refused(positions: np.ndarray) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        resample_cubic(
            sample_positions=positions,
            sample_values=np.zeros_like(positions),
            positions=positions,
        )


def test_mismatched_sample_shapes_are_refused() -> None:
    with pytest.raises(ValueError, match="must match"):
        resample_cubic(
            sample_positions=POSITIONS,
            sample_values=POSITIONS[:-1],
            positions=POSITIONS,
        )


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_samples_are_refused(bad: float) -> None:
    values = np.zeros_like(POSITIONS)
    values[2] = bad
    with pytest.raises(ValueError, match="values must be finite"):
        resample_cubic(
            sample_positions=POSITIONS, sample_values=values, positions=POSITIONS
        )


def test_an_ensemble_averages_and_carries_its_scatter() -> None:
    grid = np.linspace(0.0, 4.0, 21)
    offsets = (-0.2, 0.0, 0.2)
    curve = ensemble(
        sample_positions=[POSITIONS] * len(offsets),
        sample_values=[POSITIONS * 2.0 + offset for offset in offsets],
        positions=grid,
    )
    np.testing.assert_allclose(curve.mean_values, grid * 2.0, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        curve.deviation, np.full(grid.shape, np.std(offsets, ddof=1)), rtol=1e-9
    )
    assert curve.curve_count == len(offsets)


def test_the_ensemble_reports_where_it_extrapolated() -> None:
    grid = np.linspace(0.0, 8.0, 33)
    curve = ensemble(
        sample_positions=[POSITIONS, POSITIONS],
        sample_values=[POSITIONS, POSITIONS * 1.1],
        positions=grid,
    )
    assert curve.sampled_maximum == pytest.approx(POSITIONS.max())
    assert curve.extrapolated.sum() == int(np.count_nonzero(grid > POSITIONS.max()))


def test_the_ensemble_takes_the_shortest_curve_as_its_sampled_bound() -> None:
    short = POSITIONS[:-2]
    curve = ensemble(
        sample_positions=[POSITIONS, short],
        sample_values=[POSITIONS, short],
        positions=np.linspace(0.0, 5.0, 11),
    )
    assert curve.sampled_maximum == pytest.approx(short.max()), (
        "a grid point beyond the shortest run is extrapolated for that run even "
        "though the others cover it, so the bound is the minimum"
    )


def test_a_single_curve_ensemble_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 2 curves"):
        ensemble(
            sample_positions=[POSITIONS],
            sample_values=[POSITIONS],
            positions=POSITIONS,
        )
