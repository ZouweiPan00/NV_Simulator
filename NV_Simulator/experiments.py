"""High-level experiment simulations (ODMR, Rabi, T2*) with ODE and RWA solvers."""

from __future__ import annotations

import warnings
from typing import Dict

import numpy as np
from scipy.linalg import expm as _expm

from .constants import NVParams
from .hamiltonian import (
    estimate_pi2_time_s,
    make_linearly_polarized_drive,
    static_hamiltonian,
    total_hamiltonian,
)
from .observables import ms_population, projector_ms
from .operators import electron_operators, kron_op, nuclear_operators
from .rwa import (
    cross_spin_min_detuning_hz,
    rwa_drive_matrix,
    rwa_hamiltonian,
    rotating_frame_h0,
)
from .solver import propagate_expm, propagate_state


def _as_state(psi0: np.ndarray) -> np.ndarray:
    psi0 = np.asarray(psi0, dtype=complex)
    if psi0.shape != (9,):
        raise ValueError("Initial state must be a complex vector with shape (9,)")
    return psi0


def _readout_signal(psi: np.ndarray, readout_ms: int, readout_state_index: int | None) -> float:
    if readout_state_index is None:
        return ms_population(psi, ms=readout_ms)
    if not 1 <= int(readout_state_index) <= 9:
        raise ValueError("readout_state_index must be in [1, 9]")
    return float(np.abs(psi[int(readout_state_index) - 1]) ** 2)


def _readout_signal_batch(
    states: np.ndarray, readout_ms: int, readout_state_index: int | None
) -> np.ndarray:
    """Vectorized readout for a batch of states (N, d)."""
    if readout_state_index is not None:
        if not 1 <= int(readout_state_index) <= 9:
            raise ValueError("readout_state_index must be in [1, 9]")
        return np.abs(states[:, int(readout_state_index) - 1]) ** 2
    P = projector_ms(readout_ms)
    return np.real(np.einsum('ni,ij,nj->n', states.conj(), P, states))


def _validate_method(method: str) -> str:
    method = method.lower()
    if method not in {"ode", "rwa"}:
        raise ValueError(f"method must be 'ode' or 'rwa', got '{method}'")
    return method


def _validate_rwa_spin(rwa_spin: str) -> str:
    """Re-export from rwa module for internal use."""
    from .rwa import _validate_rwa_spin as _v
    return _v(rwa_spin)


def _validate_rwa_branch(rwa_branch: str, rwa_spin: str) -> str:
    """Validate rwa_branch against rwa_spin, raising on mismatch."""
    from .rwa import _validate_rwa_branch as _v
    return _v(rwa_branch, rwa_spin)


# Pre-compute Sz ⊗ I3 for vectorized ODMR frequency sweeps.
_SZ_KRON_I3 = kron_op(electron_operators()[2], np.eye(3, dtype=complex))
_I3_KRON_IZ = kron_op(np.eye(3, dtype=complex), nuclear_operators()[2])


def _rwa_branch_sign_and_generator(rwa_spin: str, rwa_branch: str) -> tuple[float, np.ndarray]:
    if rwa_spin == "electron":
        sign = 1.0 if rwa_branch == "ms_minus" else -1.0
        return sign, _SZ_KRON_I3
    sign = 1.0 if rwa_branch == "mi_minus" else -1.0
    return sign, _I3_KRON_IZ


def _rwa_cross_spin_detuning_threshold_hz(
    params: NVParams, pulse_strength_gauss: float, rwa_spin: str
) -> float:
    # Absolute floor protects against accidental near-degeneracy (e.g. ESLAC-like regimes).
    absolute_floor_hz = 1.0e5
    if rwa_spin == "electron":
        omega_opp_hz = abs(params.gamma_n) * abs(pulse_strength_gauss) / (2.0 * np.pi * np.sqrt(2.0))
    else:
        omega_opp_hz = abs(params.gamma_e) * abs(pulse_strength_gauss) / (2.0 * np.pi * np.sqrt(2.0))
    return max(absolute_floor_hz, 20.0 * omega_opp_hz)


def _rwa_needs_fallback_to_ode(
    h0: np.ndarray,
    f_drive_hz: float | np.ndarray,
    params: NVParams,
    pulse_strength_gauss: float,
    drive_axis: str,
    rwa_spin: str,
) -> tuple[bool, float, float]:
    if drive_axis.lower() == "z":
        warnings.warn(
            "RWA is not meaningful for z-axis drive (transverse component is zero). "
            "Falling back to ODE.",
            RuntimeWarning,
        )
        return True, 0.0, 0.0
    if pulse_strength_gauss == 0.0:
        return False, np.inf, 0.0
    min_detuning_hz = cross_spin_min_detuning_hz(h0, f_drive_hz, rwa_spin=rwa_spin)
    threshold_hz = _rwa_cross_spin_detuning_threshold_hz(params, pulse_strength_gauss, rwa_spin)
    return min_detuning_hz < threshold_hz, min_detuning_hz, threshold_hz


def _simulate_odmr_ode_signal(
    psi0: np.ndarray,
    params: NVParams,
    b0_gauss: float,
    freqs: np.ndarray,
    pulse_strength_gauss: float,
    pulse_duration_s: float,
    drive_axis: str,
    readout_ms: int,
    readout_state_index: int | None,
) -> np.ndarray:
    h0 = static_hamiltonian(params=params, b0_gauss=b0_gauss)
    signal = np.zeros_like(freqs, dtype=float)
    for i, f in enumerate(freqs):
        h1_t = make_linearly_polarized_drive(
            params=params,
            b1_gauss=pulse_strength_gauss,
            f_drive_hz=float(f),
            axis=drive_axis,
        )
        h_t = total_hamiltonian(h0, h1_t)
        _, states = propagate_state(
            psi0=psi0,
            hamiltonian=h_t,
            t_span=(0.0, pulse_duration_s),
            t_eval=np.array([pulse_duration_s]),
        )
        signal[i] = _readout_signal(states[-1], readout_ms, readout_state_index)
    return signal


def _simulate_rabi_ode(
    psi0: np.ndarray,
    params: NVParams,
    b0_gauss: float,
    pulse_strength_gauss: float,
    pulse_frequency_hz: float,
    drive_axis: str,
    t_stop_s: float,
    times: np.ndarray,
    readout_ms: int,
    readout_state_index: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    h0 = static_hamiltonian(params=params, b0_gauss=b0_gauss)
    h1_t = make_linearly_polarized_drive(
        params=params,
        b1_gauss=pulse_strength_gauss,
        f_drive_hz=pulse_frequency_hz,
        axis=drive_axis,
    )
    h_t = total_hamiltonian(h0, h1_t)
    _, ode_states = propagate_state(
        psi0=psi0,
        hamiltonian=h_t,
        t_span=(0.0, float(t_stop_s)),
        t_eval=times,
    )
    states = ode_states
    signal = _readout_signal_batch(states, readout_ms, readout_state_index)
    return states, signal


def _simulate_t2star_ode_signal(
    psi0: np.ndarray,
    params: NVParams,
    h0: np.ndarray,
    pulse_strength_gauss: float,
    f_eff: float,
    drive_axis: str,
    t_pi2: float,
    taus: np.ndarray,
    readout_ms: int,
    readout_state_index: int | None,
) -> np.ndarray:
    signal = np.zeros_like(taus, dtype=float)
    h1_t = make_linearly_polarized_drive(
        params=params,
        b1_gauss=pulse_strength_gauss,
        f_drive_hz=f_eff,
        axis=drive_axis,
    )
    h_pulse = total_hamiltonian(h0, h1_t)

    # Compute first pi/2 once.
    _, states_1 = propagate_state(
        psi0=psi0,
        hamiltonian=h_pulse,
        t_span=(0.0, t_pi2),
        t_eval=np.array([t_pi2]),
    )
    psi_after_pi2 = states_1[-1]

    for i, tau in enumerate(taus):
        # Free precession under static H0 (no drive)
        t1 = t_pi2
        t2 = t1 + float(tau)
        _, states_2 = propagate_state(
            psi0=psi_after_pi2,
            hamiltonian=h0,
            t_span=(0.0, float(tau)),
            t_eval=np.array([float(tau)]),
        )
        psi_2 = states_2[-1]

        # Second pi/2 pulse with continuous source phase.
        t3 = t2 + t_pi2
        _, states_3 = propagate_state(
            psi0=psi_2,
            hamiltonian=h_pulse,
            t_span=(t2, t3),
            t_eval=np.array([t3]),
        )
        psi_3 = states_3[-1]
        signal[i] = _readout_signal(psi_3, readout_ms, readout_state_index)
    return signal


def simulate_odmr(
    initial_state: np.ndarray,
    f_start_hz: float,
    f_stop_hz: float,
    pulse_strength_gauss: float,
    pulse_duration_s: float,
    b0_gauss: float,
    n_points: int = 201,
    readout_ms: int = 0,
    readout_state_index: int | None = None,
    drive_axis: str = "x",
    rwa_branch: str = "ms_minus",
    rwa_spin: str = "electron",
    rwa_auto_fallback: bool = True,
    params: NVParams | None = None,
    method: str = "rwa",
) -> Dict[str, np.ndarray]:
    method = _validate_method(method)
    params = params or NVParams()
    psi0 = _as_state(initial_state)
    if pulse_duration_s < 0:
        raise ValueError("pulse_duration_s must be non-negative")
    if pulse_strength_gauss < 0:
        raise ValueError("pulse_strength_gauss must be non-negative")

    freqs = np.linspace(f_start_hz, f_stop_hz, n_points)

    if method == "rwa":
        rwa_spin = _validate_rwa_spin(rwa_spin)
        rwa_branch = _validate_rwa_branch(rwa_branch, rwa_spin)
        h0 = static_hamiltonian(params=params, b0_gauss=b0_gauss)
        needs_fallback, min_det_hz, threshold_hz = _rwa_needs_fallback_to_ode(
            h0,
            freqs,
            params,
            pulse_strength_gauss,
            drive_axis,
            rwa_spin,
        )
        if needs_fallback and rwa_auto_fallback:
            warnings.warn(
                (
                    f"RWA may be invalid: minimum cross-spin detuning is {min_det_hz:.3e} Hz, "
                    f"below threshold {threshold_hz:.3e} Hz at B0={b0_gauss:g} G. "
                    "Falling back to ODE."
                ),
                RuntimeWarning,
            )
            method = "ode"
        elif needs_fallback:
            warnings.warn(
                (
                    f"RWA may be invalid: minimum cross-spin detuning is {min_det_hz:.3e} Hz, "
                    f"below threshold {threshold_hz:.3e} Hz at B0={b0_gauss:g} G. "
                    "Continuing with RWA because rwa_auto_fallback=False."
                ),
                RuntimeWarning,
            )

    if method == "rwa":
        h1_rwa = rwa_drive_matrix(
            params,
            pulse_strength_gauss,
            drive_axis,
            rwa_spin=rwa_spin,
        )

        # Vectorized batch construction: avoid N calls to rotating_frame_h0
        sign, rot_gen = _rwa_branch_sign_and_generator(rwa_spin, rwa_branch)
        omega_d = 2.0 * np.pi * freqs  # (N,)
        h_base = h0 + h1_rwa  # (9, 9)
        h_rwa_batch = (
            h_base[None, :, :]
            + sign * omega_d[:, None, None] * rot_gen[None, :, :]
        )  # (N, 9, 9)

        states = propagate_expm(psi0, h_rwa_batch, pulse_duration_s)  # (N, 9)
        signal = _readout_signal_batch(states, readout_ms, readout_state_index)
    else:
        signal = _simulate_odmr_ode_signal(
            psi0=psi0,
            params=params,
            b0_gauss=b0_gauss,
            freqs=freqs,
            pulse_strength_gauss=pulse_strength_gauss,
            pulse_duration_s=pulse_duration_s,
            drive_axis=drive_axis,
            readout_ms=readout_ms,
            readout_state_index=readout_state_index,
        )

    return {"frequencies_hz": freqs, "signal": signal}


def simulate_rabi(
    initial_state: np.ndarray,
    pulse_frequency_hz: float,
    t_start_s: float,
    t_stop_s: float,
    pulse_strength_gauss: float,
    b0_gauss: float,
    n_points: int = 401,
    readout_ms: int = 0,
    readout_state_index: int | None = None,
    drive_axis: str = "x",
    rwa_branch: str = "ms_minus",
    rwa_spin: str = "electron",
    rwa_auto_fallback: bool = True,
    params: NVParams | None = None,
    method: str = "rwa",
) -> Dict[str, np.ndarray]:
    """Rabi oscillation simulation.

    The drive is assumed on since t=0.  ``t_start_s`` and ``t_stop_s`` define
    the output window: both backends return the state at each time in
    ``linspace(t_start_s, t_stop_s, n_points)``.  For the ODE path the
    integrator runs from t=0 so that the drive phase ``cos(2*pi*f*t)`` is
    consistent with the RWA path.
    """
    method = _validate_method(method)
    params = params or NVParams()
    psi0 = _as_state(initial_state)
    if pulse_strength_gauss < 0:
        raise ValueError("pulse_strength_gauss must be non-negative")
    if t_stop_s < t_start_s:
        raise ValueError("t_stop_s must be >= t_start_s")

    times = np.linspace(t_start_s, t_stop_s, n_points)

    if method == "rwa":
        rwa_spin = _validate_rwa_spin(rwa_spin)
        rwa_branch = _validate_rwa_branch(rwa_branch, rwa_spin)
        h0 = static_hamiltonian(params=params, b0_gauss=b0_gauss)
        needs_fallback, min_det_hz, threshold_hz = _rwa_needs_fallback_to_ode(
            h0,
            pulse_frequency_hz,
            params,
            pulse_strength_gauss,
            drive_axis,
            rwa_spin,
        )
        if needs_fallback and rwa_auto_fallback:
            warnings.warn(
                (
                    f"RWA may be invalid: minimum cross-spin detuning is {min_det_hz:.3e} Hz, "
                    f"below threshold {threshold_hz:.3e} Hz at B0={b0_gauss:g} G. "
                    "Falling back to ODE."
                ),
                RuntimeWarning,
            )
            method = "ode"
        elif needs_fallback:
            warnings.warn(
                (
                    f"RWA may be invalid: minimum cross-spin detuning is {min_det_hz:.3e} Hz, "
                    f"below threshold {threshold_hz:.3e} Hz at B0={b0_gauss:g} G. "
                    "Continuing with RWA because rwa_auto_fallback=False."
                ),
                RuntimeWarning,
            )

    if method == "rwa":
        h_rwa = rwa_hamiltonian(
            params,
            b0_gauss,
            pulse_frequency_hz,
            pulse_strength_gauss,
            drive_axis,
            branch=rwa_branch,
            rwa_spin=rwa_spin,
        )
        states = propagate_expm(psi0, h_rwa, times)  # (N, 9)
        signal = _readout_signal_batch(states, readout_ms, readout_state_index)
    else:
        states, signal = _simulate_rabi_ode(
            psi0=psi0,
            params=params,
            b0_gauss=b0_gauss,
            pulse_strength_gauss=pulse_strength_gauss,
            pulse_frequency_hz=pulse_frequency_hz,
            drive_axis=drive_axis,
            t_stop_s=t_stop_s,
            times=times,
            readout_ms=readout_ms,
            readout_state_index=readout_state_index,
        )

    return {"times_s": times, "signal": signal, "states": states}


def simulate_t2star(
    initial_state: np.ndarray,
    detuning_hz: float,
    pulse_strength_gauss: float,
    pulse_frequency_hz: float,
    t_start_s: float,
    t_stop_s: float,
    b0_gauss: float,
    n_points: int = 201,
    readout_ms: int = 0,
    readout_state_index: int | None = None,
    drive_axis: str = "x",
    rwa_branch: str = "ms_minus",
    rwa_spin: str = "electron",
    rwa_auto_fallback: bool = True,
    params: NVParams | None = None,
    method: str = "rwa",
) -> Dict[str, np.ndarray]:
    """Ramsey-style T2* simulation: pi/2 - tau - pi/2."""
    method = _validate_method(method)
    params = params or NVParams()
    psi0 = _as_state(initial_state)
    if pulse_strength_gauss < 0:
        raise ValueError("pulse_strength_gauss must be non-negative")
    if t_stop_s < t_start_s:
        raise ValueError("t_stop_s must be >= t_start_s")
    h0 = static_hamiltonian(params=params, b0_gauss=b0_gauss)
    taus = np.linspace(t_start_s, t_stop_s, n_points)
    t_pi2 = estimate_pi2_time_s(
        params=params, b1_gauss=pulse_strength_gauss, rwa_spin=rwa_spin
    )
    f_eff = pulse_frequency_hz + detuning_hz

    if method == "rwa":
        rwa_spin = _validate_rwa_spin(rwa_spin)
        rwa_branch = _validate_rwa_branch(rwa_branch, rwa_spin)
        needs_fallback, min_det_hz, threshold_hz = _rwa_needs_fallback_to_ode(
            h0,
            f_eff,
            params,
            pulse_strength_gauss,
            drive_axis,
            rwa_spin,
        )
        if needs_fallback and rwa_auto_fallback:
            warnings.warn(
                (
                    f"RWA may be invalid: minimum cross-spin detuning is {min_det_hz:.3e} Hz, "
                    f"below threshold {threshold_hz:.3e} Hz at B0={b0_gauss:g} G. "
                    "Falling back to ODE."
                ),
                RuntimeWarning,
            )
            method = "ode"
        elif needs_fallback:
            warnings.warn(
                (
                    f"RWA may be invalid: minimum cross-spin detuning is {min_det_hz:.3e} Hz, "
                    f"below threshold {threshold_hz:.3e} Hz at B0={b0_gauss:g} G. "
                    "Continuing with RWA because rwa_auto_fallback=False."
                ),
                RuntimeWarning,
            )

    if method == "rwa":
        h_pulse_rwa = rwa_hamiltonian(
            params,
            b0_gauss,
            f_eff,
            pulse_strength_gauss,
            drive_axis,
            branch=rwa_branch,
            rwa_spin=rwa_spin,
        )
        h_free_rot = rotating_frame_h0(h0, f_eff, branch=rwa_branch, rwa_spin=rwa_spin)

        # pi/2 propagator (computed once)
        U_pi2 = _expm(-1j * h_pulse_rwa * t_pi2)  # (9, 9)

        # Sequence: pi/2 - free(tau) - pi/2
        psi1 = U_pi2 @ psi0  # (9,)
        psi2_batch = propagate_expm(psi1, h_free_rot, taus)  # (N, 9)
        psi3_batch = np.einsum('ij,nj->ni', U_pi2, psi2_batch)  # (N, 9)

        signal = _readout_signal_batch(psi3_batch, readout_ms, readout_state_index)
    else:
        signal = _simulate_t2star_ode_signal(
            psi0=psi0,
            params=params,
            h0=h0,
            pulse_strength_gauss=pulse_strength_gauss,
            f_eff=f_eff,
            drive_axis=drive_axis,
            t_pi2=t_pi2,
            taus=taus,
            readout_ms=readout_ms,
            readout_state_index=readout_state_index,
        )

    return {
        "taus_s": taus,
        "signal": signal,
        "estimated_pi2_s": np.array([t_pi2], dtype=float),
    }
