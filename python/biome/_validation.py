# SPDX-License-Identifier: Apache-2.0
#
# biome._validation — shared reporting for array guard failures.
#
# Private. Exists so that terramechanics and io.soil report a rejected array
# the same way, rather than drifting into two dialects of the same message.

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def first_violation(
    violations: NDArray[np.bool_], values: NDArray[np.float64]
) -> tuple[int, int, float]:
    flat_violations = np.ravel(violations)
    flat_values = np.ravel(np.broadcast_to(values, violations.shape))
    return (
        int(np.count_nonzero(flat_violations)),
        int(flat_violations.size),
        float(flat_values[int(np.argmax(flat_violations))]),
    )
