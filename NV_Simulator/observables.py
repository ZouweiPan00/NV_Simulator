from __future__ import annotations

import numpy as np

from .operators import BASIS_STATES


def projector_ms(ms: int) -> np.ndarray:
    if ms not in {+1, 0, -1}:
        raise ValueError("ms must be +1, 0, or -1")
    p = np.zeros((9, 9), dtype=complex)
    for idx, (ms_i, _) in enumerate(BASIS_STATES):
        if ms_i == ms:
            p[idx, idx] = 1.0
    return p


def expectation(psi: np.ndarray, operator: np.ndarray) -> float:
    return float(np.real(np.vdot(psi, operator @ psi)))


def ms_population(psi: np.ndarray, ms: int = 0) -> float:
    return expectation(psi, projector_ms(ms))

