"""Rotating-wave approximation (RWA) Hamiltonian for the 9-level NV system."""

from __future__ import annotations

import numpy as np

from .constants import NVParams
from .hamiltonian import static_hamiltonian
from .operators import basis_index, electron_operators, kron_op, nuclear_operators


# Allowed single-quantum transitions for transverse (x/y) driving.
# Electron transitions: Delta m_s = +/-1, Delta m_I = 0.
_ELECTRON_PAIRS = np.array(
    [
        [basis_index(ms1, mi), basis_index(ms2, mi)]
        for ms1, ms2 in [(+1, 0), (0, -1)]
        for mi in [+1, 0, -1]
    ],
    dtype=int,
)
# Nuclear transitions: Delta m_I = +/-1, Delta m_s = 0.
_NUCLEAR_PAIRS = np.array(
    [
        [basis_index(ms, mi1), basis_index(ms, mi2)]
        for ms in [+1, 0, -1]
        for mi1, mi2 in [(+1, 0), (0, -1)]
    ],
    dtype=int,
)


def _validate_rwa_spin(rwa_spin: str) -> str:
    rwa_spin = rwa_spin.lower()
    if rwa_spin not in {"electron", "nuclear"}:
        raise ValueError("rwa_spin must be 'electron' or 'nuclear'")
    return rwa_spin


def _validate_rwa_branch(branch: str, rwa_spin: str) -> str:
    branch = branch.lower()
    if rwa_spin == "electron":
        if branch not in {"ms_minus", "ms_plus"}:
            raise ValueError("For rwa_spin='electron', branch must be 'ms_minus' or 'ms_plus'")
        return branch
    if branch not in {"mi_minus", "mi_plus"}:
        raise ValueError("For rwa_spin='nuclear', branch must be 'mi_minus' or 'mi_plus'")
    return branch


def transition_frequencies_hz(h0: np.ndarray, spin: str) -> np.ndarray:
    """Return allowed transition frequencies (Hz) for electron or nuclear driving.

    .. note::
       This function reads only the diagonal of *h0* and therefore assumes
       *h0* is diagonal in the ``|ms, mI>`` computational basis.  The default
       ``static_hamiltonian`` satisfies this requirement.  Pass a custom *h0*
       that contains off-diagonal elements and you will get incorrect results.

    Parameters
    ----------
    h0 : ndarray (9, 9)
        Laboratory-frame static Hamiltonian (must be diagonal).
    spin : str
        "electron" or "nuclear".

    Returns
    -------
    ndarray (6,)
        Absolute transition frequencies in Hz.

    Raises
    ------
    ValueError
        If *h0* has significant off-diagonal elements.
    """
    spin = _validate_rwa_spin(spin)
    h0 = np.asarray(h0, dtype=complex)
    off_diag = np.linalg.norm(h0 - np.diag(np.diag(h0)))
    if off_diag > 1e-6 * np.linalg.norm(h0):
        raise ValueError(
            "transition_frequencies_hz requires a diagonal H0 in the |ms,mI> basis; "
            "for non-diagonal Hamiltonians use np.linalg.eigh to diagonalise first"
        )
    e_diag = np.real(np.diag(h0))
    pairs = _ELECTRON_PAIRS if spin == "electron" else _NUCLEAR_PAIRS
    return np.abs(e_diag[pairs[:, 1]] - e_diag[pairs[:, 0]]) / (2.0 * np.pi)


def cross_spin_min_detuning_hz(h0: np.ndarray, f_drive_hz: float | np.ndarray, rwa_spin: str) -> float:
    """Return minimum detuning to opposite-spin transitions for a drive frequency set."""
    rwa_spin = _validate_rwa_spin(rwa_spin)
    opposite_spin = "nuclear" if rwa_spin == "electron" else "electron"
    f_opp = transition_frequencies_hz(h0, opposite_spin)
    f_drive = np.atleast_1d(np.asarray(f_drive_hz, dtype=float))
    det = np.abs(f_drive[:, None] - f_opp[None, :])
    return float(np.min(det))


def rotating_frame_h0(
    h0: np.ndarray,
    f_drive_hz: float,
    branch: str = "ms_minus",
    rwa_spin: str = "electron",
) -> np.ndarray:
    """Rotating-frame static Hamiltonian for selected electron or nuclear branch.

    .. note::
       This transformation assumes ``[H0, Sz ⊗ I3] = 0`` (electron) or
       ``[H0, I3 ⊗ Iz] = 0`` (nuclear), i.e. the static Hamiltonian commutes
       with the generator of the rotating frame.  The default
       ``static_hamiltonian`` satisfies this.  If *h0* contains terms that
       break this symmetry the result will be incorrect.

    Parameters
    ----------
    h0 : ndarray (9, 9)
        Laboratory-frame static Hamiltonian.
    f_drive_hz : float
        Drive frequency in Hz.
    branch : str
        For ``rwa_spin='electron'``:
        - "ms_minus" targets m_s=0 <-> -1 with positive frequency (H0 + w*Sz).
        - "ms_plus"  targets m_s=0 <-> +1 with positive frequency (H0 - w*Sz).
        For ``rwa_spin='nuclear'``:
        - "mi_minus" targets m_I=0 <-> -1 with positive frequency (H0 + w*Iz).
        - "mi_plus"  targets m_I=0 <-> +1 with positive frequency (H0 - w*Iz).
    rwa_spin : str
        "electron" or "nuclear".

    Returns
    -------
    ndarray (9, 9)
        Rotating-frame static Hamiltonian.
    """
    rwa_spin = _validate_rwa_spin(rwa_spin)
    branch = _validate_rwa_branch(branch, rwa_spin)

    _, _, sz = electron_operators()
    _, _, iz = nuclear_operators()
    i3 = np.eye(3, dtype=complex)
    omega_d = 2.0 * np.pi * f_drive_hz

    if rwa_spin == "electron":
        if branch == "ms_minus":
            return h0 + omega_d * kron_op(sz, i3)
        return h0 - omega_d * kron_op(sz, i3)

    if branch == "mi_minus":
        return h0 + omega_d * kron_op(i3, iz)
    return h0 - omega_d * kron_op(i3, iz)


def rwa_drive_matrix(
    params: NVParams,
    b1_gauss: float,
    axis: str = "x",
    rwa_spin: str = "electron",
    phase_rad: float = 0.0,
) -> np.ndarray:
    """RWA time-independent drive term.

    - ``rwa_spin='electron'``: ``-gamma_e * B1/2 * (S_perp kron I3)``
    - ``rwa_spin='nuclear'``: ``-gamma_n * B1/2 * (I3 kron I_perp)``

    When ``phase_rad != 0`` the transverse operator is rotated:
    ``S_perp -> cos(phi)*S_x + sin(phi)*S_y`` (or the nuclear equivalent).

    Parameters
    ----------
    params : NVParams
        NV center parameters.
    b1_gauss : float
        Drive field amplitude in Gauss.
    axis : str
        "x" or "y" ("z" returns zero matrix).
    rwa_spin : str
        "electron" or "nuclear".
    phase_rad : float
        Drive phase in radians (default 0).

    Returns
    -------
    ndarray (9, 9)
        RWA drive matrix.
    """
    rwa_spin = _validate_rwa_spin(rwa_spin)
    axis = axis.lower()
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be one of {'x', 'y', 'z'}")

    if axis == "z":
        return np.zeros((9, 9), dtype=complex)

    sx, sy, _ = electron_operators()
    ix, iy, _ = nuclear_operators()
    i3 = np.eye(3, dtype=complex)

    # Apply phase rotation to the transverse operator
    c, s = np.cos(phase_rad), np.sin(phase_rad)
    if rwa_spin == "electron":
        if axis == "x":
            s_perp = c * sx + s * sy
        else:
            s_perp = c * sy - s * sx
        return (-params.gamma_e * b1_gauss / 2.0) * kron_op(s_perp, i3)
    if axis == "x":
        i_perp = c * ix + s * iy
    else:
        i_perp = c * iy - s * ix
    return (-params.gamma_n * b1_gauss / 2.0) * kron_op(i3, i_perp)


def rwa_hamiltonian(
    params: NVParams,
    b0_gauss: float,
    f_drive_hz: float,
    b1_gauss: float,
    axis: str = "x",
    branch: str = "ms_minus",
    rwa_spin: str = "electron",
    phase_rad: float = 0.0,
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
        Branch selection (see ``rotating_frame_h0``).
    rwa_spin : str
        "electron" or "nuclear".
    phase_rad : float
        Drive phase in radians (default 0).

    Returns
    -------
    ndarray (9, 9)
        Time-independent RWA Hamiltonian.
    """
    h0 = static_hamiltonian(params=params, b0_gauss=b0_gauss)
    h0_rot = rotating_frame_h0(h0, f_drive_hz, branch=branch, rwa_spin=rwa_spin)
    h1_rwa = rwa_drive_matrix(
        params, b1_gauss, axis, rwa_spin=rwa_spin, phase_rad=phase_rad
    )
    return h0_rot + h1_rwa
