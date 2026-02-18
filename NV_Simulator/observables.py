from __future__ import annotations

import warnings

import numpy as np

from .operators import BASIS_STATES

# Cache the three ms projectors (computed once at import).
_PROJECTOR_CACHE: dict[int, np.ndarray] = {}


def projector_ms(ms: int) -> np.ndarray:
    if ms not in {+1, 0, -1}:
        raise ValueError("ms must be +1, 0, or -1")
    if ms in _PROJECTOR_CACHE:
        return _PROJECTOR_CACHE[ms]
    p = np.zeros((9, 9), dtype=complex)
    for idx, (ms_i, _) in enumerate(BASIS_STATES):
        if ms_i == ms:
            p[idx, idx] = 1.0
    _PROJECTOR_CACHE[ms] = p
    return p


def expectation(psi: np.ndarray, operator: np.ndarray) -> float:
    val = np.vdot(psi, operator @ psi)
    if abs(val.imag) > 1e-10 * max(abs(val.real), 1.0):
        warnings.warn(
            f"Non-negligible imaginary part in expectation value: {val.imag:.3e}",
            RuntimeWarning,
        )
    return float(val.real)


def ms_population(psi: np.ndarray, ms: int = 0) -> float:
    return expectation(psi, projector_ms(ms))

