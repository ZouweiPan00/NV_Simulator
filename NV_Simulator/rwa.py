"""Rotating-wave approximation (RWA) Hamiltonian for the 9-level NV system."""

from __future__ import annotations

import numpy as np

from .constants import NVParams
from .hamiltonian import static_hamiltonian
from .operators import electron_operators, kron_op


def rotating_frame_h0(
    h0: np.ndarray,
    f_drive_hz: float,
    branch: str = "ms_minus",
) -> np.ndarray:
    """Rotating-frame static Hamiltonian for a selected ESR branch.

    Parameters
    ----------
    h0 : ndarray (9, 9)
        Laboratory-frame static Hamiltonian.
    f_drive_hz : float
        Drive frequency in Hz.
    branch : str
        "ms_minus" targets m_s=0 <-> -1 with positive frequency (H0 + w*Sz).
        "ms_plus"  targets m_s=0 <-> +1 with positive frequency (H0 - w*Sz).

    Returns
    -------
    ndarray (9, 9)
        Rotating-frame static Hamiltonian.
    """
    _, _, sz = electron_operators()
    i3 = np.eye(3, dtype=complex)
    omega_d = 2.0 * np.pi * f_drive_hz
    if branch == "ms_minus":
        return h0 + omega_d * kron_op(sz, i3)
    if branch == "ms_plus":
        return h0 - omega_d * kron_op(sz, i3)
    raise ValueError("branch must be 'ms_minus' or 'ms_plus'")


def rwa_drive_matrix(params: NVParams, b1_gauss: float, axis: str = "x") -> np.ndarray:
    """RWA time-independent drive term: -gamma_e * B1 / 2 * (S_perp kron I3).

    Only the electron-spin coupling is included. The nuclear Zeeman drive
    term (-gamma_n * B1/2 * I_perp) is omitted since |gamma_n/gamma_e| ~ 1e-4.
    This is the sole physics difference between the RWA and ODE paths.

    The drive matrix is independent of the branch choice (ms_minus / ms_plus)
    because the RWA co-rotating term yields the same S_perp coupling for both
    ESR transitions.

    Parameters
    ----------
    params : NVParams
        NV center parameters.
    b1_gauss : float
        Drive field amplitude in Gauss.
    axis : str
        "x" or "y" ("z" returns zero matrix).

    Returns
    -------
    ndarray (9, 9)
        RWA drive matrix.
    """
    axis = axis.lower()
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be one of {'x', 'y', 'z'}")

    if axis == "z":
        return np.zeros((9, 9), dtype=complex)

    sx, sy, _ = electron_operators()
    i3 = np.eye(3, dtype=complex)
    s_perp = sx if axis == "x" else sy
    return (-params.gamma_e * b1_gauss / 2.0) * kron_op(s_perp, i3)


def rwa_hamiltonian(
    params: NVParams,
    b0_gauss: float,
    f_drive_hz: float,
    b1_gauss: float,
    axis: str = "x",
    branch: str = "ms_minus",
) -> np.ndarray:
    """Complete RWA Hamiltonian = rotating_frame_h0 + rwa_drive_matrix.

    Convenience function that builds the full time-independent 9x9 Hamiltonian
    in the rotating frame under the rotating-wave approximation.

    Parameters
    ----------
    params : NVParams
        NV center parameters.
    b0_gauss : float
        Static magnetic field along NV axis in Gauss.
    f_drive_hz : float
        Drive frequency in Hz.
    b1_gauss : float
        Drive field amplitude in Gauss.
    axis : str
        Drive polarization axis ("x", "y", or "z").
    branch : str
        ESR branch selection: "ms_minus" or "ms_plus".

    Returns
    -------
    ndarray (9, 9)
        Time-independent RWA Hamiltonian.
    """
    h0 = static_hamiltonian(params=params, b0_gauss=b0_gauss)
    h0_rot = rotating_frame_h0(h0, f_drive_hz, branch=branch)
    h1_rwa = rwa_drive_matrix(params, b1_gauss, axis)
    return h0_rot + h1_rwa
