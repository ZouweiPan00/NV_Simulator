"""Solvers for time evolution of quantum states (ODE and matrix exponential)."""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

HamiltonianType = Union[np.ndarray, Callable[[float], np.ndarray]]


def _get_hamiltonian_callable(hamiltonian: HamiltonianType) -> Callable[[float], np.ndarray]:
    if callable(hamiltonian):
        return hamiltonian

    h_const = np.asarray(hamiltonian, dtype=complex)

    def h_of_t(_: float) -> np.ndarray:
        return h_const

    return h_of_t


def propagate_state(
    psi0: np.ndarray,
    hamiltonian: HamiltonianType,
    t_span: Tuple[float, float],
    t_eval: Optional[np.ndarray] = None,
    method: str = "DOP853",
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate pure state under Schr. equation dpsi/dt = -i H(t) psi."""
    psi0 = np.asarray(psi0, dtype=complex)
    if psi0.ndim != 1:
        raise ValueError("psi0 must be a 1D complex state vector")

    norm0 = np.linalg.norm(psi0)
    if norm0 == 0:
        raise ValueError("Initial state psi0 has zero norm")
    psi0 = psi0 / norm0

    t_start, t_stop = float(t_span[0]), float(t_span[1])
    if t_start == t_stop:
        return np.array([t_start], dtype=float), np.array([psi0], dtype=complex)

    h_of_t = _get_hamiltonian_callable(hamiltonian)

    def rhs(t: float, psi: np.ndarray) -> np.ndarray:
        return -1j * (h_of_t(t) @ psi)

    sol = solve_ivp(
        rhs,
        t_span=(t_start, t_stop),
        y0=psi0,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Time evolution failed: {sol.message}")

    states = np.asarray(sol.y, dtype=complex).T
    return sol.t, states


def propagate_expm(
    psi0: np.ndarray,
    hamiltonian: np.ndarray,
    t: Union[float, np.ndarray],
) -> np.ndarray:
    """Propagate state using matrix exponential for time-independent Hamiltonians.

    Three usage modes:

    1. H (d,d) + t scalar     -> single point, returns (d,)
    2. H (N,d,d) + t scalar   -> ODMR sweep: N different H, same time, returns (N,d)
    3. H (d,d) + t (N,)       -> Rabi sweep: same H, N different times, returns (N,d)

    Parameters
    ----------
    psi0 : ndarray (d,)
        Initial state vector.
    hamiltonian : ndarray (d,d) or (N,d,d)
        Time-independent Hamiltonian(s).
    t : float or ndarray (N,)
        Evolution time(s).

    Returns
    -------
    ndarray (d,) or (N,d)
        Final state(s).
    """
    psi0 = np.asarray(psi0, dtype=complex)
    hamiltonian = np.asarray(hamiltonian, dtype=complex)

    # --- Input validation ---
    if psi0.ndim != 1:
        raise ValueError("psi0 must be a 1D state vector")

    norm0 = np.linalg.norm(psi0)
    if norm0 == 0:
        raise ValueError("Initial state psi0 has zero norm")
    psi0 = psi0 / norm0

    d = psi0.shape[0]

    if hamiltonian.ndim == 2:
        if hamiltonian.shape != (d, d):
            raise ValueError(
                f"Hamiltonian shape {hamiltonian.shape} incompatible with psi0 length {d}"
            )
    elif hamiltonian.ndim == 3:
        if hamiltonian.shape[1:] != (d, d):
            raise ValueError(
                f"Hamiltonian shape {hamiltonian.shape} incompatible with psi0 length {d}"
            )
    else:
        raise ValueError(
            f"Hamiltonian must be 2D (d,d) or 3D (N,d,d), got {hamiltonian.ndim}D"
        )

    # --- Reject ambiguous batch H + array t (A) ---
    if hamiltonian.ndim == 3 and np.ndim(t) != 0:
        raise ValueError(
            "Cannot combine batched Hamiltonian (N,d,d) with array t; "
            "use scalar t for batch H, or single H for array t"
        )

    if np.ndim(t) == 0:
        # Modes 1 & 2: scalar time
        arg = -1j * hamiltonian * float(t)
        U = expm(arg)  # (d,d) or (N,d,d)
        return U @ psi0
    else:
        # Mode 3: H (d,d), t array (N,)
        t_arr = np.asarray(t, dtype=float)
        if t_arr.ndim != 1:
            raise ValueError(f"t must be scalar or 1D array, got {t_arr.ndim}D")
        arg = -1j * hamiltonian[None, :, :] * t_arr[:, None, None]  # (N,d,d)
        U = expm(arg)  # (N,d,d)
        return U @ psi0  # (N,d)
