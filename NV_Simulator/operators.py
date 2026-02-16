from __future__ import annotations

from typing import Tuple

import numpy as np

# Basis order:
# |1>..|9> == |ms,mI> with
# ms in [+1, 0, -1], and for each ms: mI in [+1, 0, -1].
BASIS_STATES = [
    (+1, +1),
    (+1, 0),
    (+1, -1),
    (0, +1),
    (0, 0),
    (0, -1),
    (-1, +1),
    (-1, 0),
    (-1, -1),
]


def spin_1_matrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return spin-1 matrices (dimensionless, with hbar=1)."""
    sx = (1.0 / np.sqrt(2.0)) * np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=complex
    )
    sy = (1.0 / np.sqrt(2.0)) * np.array(
        [[0.0, -1j, 0.0], [1j, 0.0, -1j], [0.0, 1j, 0.0]], dtype=complex
    )
    sz = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=complex)
    return sx, sy, sz


def electron_operators() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return spin_1_matrices()


def nuclear_operators() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return spin_1_matrices()


def kron_op(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.kron(a, b).astype(complex)


def basis_index(ms: int, mI: int) -> int:
    """Return 0-based index in BASIS_STATES."""
    try:
        return BASIS_STATES.index((ms, mI))
    except ValueError as exc:
        raise ValueError(f"Invalid basis label (ms={ms}, mI={mI})") from exc


def ket_from_index(index_1based: int) -> np.ndarray:
    """Return basis ket |index>, index starts at 1."""
    if not 1 <= index_1based <= 9:
        raise ValueError("index_1based must be in [1, 9]")
    ket = np.zeros(9, dtype=complex)
    ket[index_1based - 1] = 1.0
    return ket


def ket_from_quantum_numbers(ms: int, mI: int) -> np.ndarray:
    ket = np.zeros(9, dtype=complex)
    ket[basis_index(ms, mI)] = 1.0
    return ket

