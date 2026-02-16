from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from .constants import NVParams
from .operators import electron_operators, kron_op, nuclear_operators


def static_hamiltonian(params: NVParams, b0_gauss: float) -> np.ndarray:
    """Construct H0 for the nine-level system."""
    sx, sy, sz = electron_operators()
    ix, iy, iz = nuclear_operators()
    i3 = np.eye(3, dtype=complex)

    omega_e = -params.gamma_e * b0_gauss
    omega_n = -params.gamma_n * b0_gauss

    return (
        params.D * kron_op(sz @ sz, i3)
        + params.Q * kron_op(i3, iz @ iz)
        + omega_e * kron_op(sz, i3)
        + omega_n * kron_op(i3, iz)
        + params.A * kron_op(sz, iz)
    )


def drive_hamiltonian(
    params: NVParams,
    bx_gauss: float,
    by_gauss: float,
    bz_gauss: float,
) -> np.ndarray:
    """Construct H1 = -gamma_e B·S⊗1 - gamma_n B·1⊗I for a fixed B field."""
    sx, sy, sz = electron_operators()
    ix, iy, iz = nuclear_operators()
    i3 = np.eye(3, dtype=complex)

    b_dot_s = bx_gauss * sx + by_gauss * sy + bz_gauss * sz
    b_dot_i = bx_gauss * ix + by_gauss * iy + bz_gauss * iz

    return -params.gamma_e * kron_op(b_dot_s, i3) - params.gamma_n * kron_op(i3, b_dot_i)


def make_linearly_polarized_drive(
    params: NVParams,
    b1_gauss: float,
    f_drive_hz: float,
    phase_rad: float = 0.0,
    axis: str = "x",
) -> Callable[[float], np.ndarray]:
    """Return H1(t) for B1*cos(2*pi*f*t + phase) along x/y/z."""
    axis = axis.lower()
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be one of {'x', 'y', 'z'}")

    def h1_t(t: float) -> np.ndarray:
        amp = b1_gauss * np.cos(2.0 * np.pi * f_drive_hz * t + phase_rad)
        bx, by, bz = 0.0, 0.0, 0.0
        if axis == "x":
            bx = amp
        elif axis == "y":
            by = amp
        else:
            bz = amp
        return drive_hamiltonian(params=params, bx_gauss=bx, by_gauss=by, bz_gauss=bz)

    return h1_t


def total_hamiltonian(
    h0: np.ndarray, h1_t: Callable[[float], np.ndarray]
) -> Callable[[float], np.ndarray]:
    def h_t(t: float) -> np.ndarray:
        return h0 + h1_t(t)

    return h_t


def estimate_rabi_omega_rad_s(params: NVParams, b1_gauss: float) -> float:
    """Approximate electron-transition Rabi angular frequency for x/y drive."""
    return abs(params.gamma_e) * abs(b1_gauss) / np.sqrt(2.0)


def estimate_pi2_time_s(params: NVParams, b1_gauss: float) -> float:
    omega = estimate_rabi_omega_rad_s(params, b1_gauss)
    if omega <= 0:
        raise ValueError("b1_gauss must be non-zero for pi/2 pulse estimation")
    return np.pi / (2.0 * omega)

