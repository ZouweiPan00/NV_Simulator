"""High-level experiment simulations (ODMR, Rabi, T2*) with ODE and RWA solvers."""

from __future__ import annotations

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
from .operators import electron_operators, kron_op
from .rwa import rwa_drive_matrix, rwa_hamiltonian, rotating_frame_h0
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


def _validate_rwa_branch(rwa_branch: str) -> str:
    rwa_branch = rwa_branch.lower()
    if rwa_branch not in {"ms_minus", "ms_plus"}:
        raise ValueError("rwa_branch must be 'ms_minus' or 'ms_plus'")
    return rwa_branch


# Pre-compute Sz ⊗ I3 for vectorized ODMR frequency sweeps.
_SZ_KRON_I3 = kron_op(electron_operators()[2], np.eye(3, dtype=complex))


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
    params: NVParams | None = None,
    method: str = "rwa",
) -> Dict[str, np.ndarray]:
    method = _validate_method(method)
    params = params or NVParams()
    psi0 = _as_state(initial_state)

    freqs = np.linspace(f_start_hz, f_stop_hz, n_points)

    if method == "rwa":
        rwa_branch = _validate_rwa_branch(rwa_branch)
        h0 = static_hamiltonian(params=params, b0_gauss=b0_gauss)
        h1_rwa = rwa_drive_matrix(params, pulse_strength_gauss, drive_axis)

        # Vectorized batch construction (fix C): avoid N calls to rotating_frame_h0
        sign = 1.0 if rwa_branch == "ms_minus" else -1.0
        omega_d = 2.0 * np.pi * freqs  # (N,)
        h_base = h0 + h1_rwa  # (9, 9)
        h_rwa_batch = (
            h_base[None, :, :]
            + sign * omega_d[:, None, None] * _SZ_KRON_I3[None, :, :]
        )  # (N, 9, 9)

        states = propagate_expm(psi0, h_rwa_batch, pulse_duration_s)  # (N, 9)
        signal = _readout_signal_batch(states, readout_ms, readout_state_index)
    else:
        h0 = static_hamiltonian(params=params, b0_gauss=b0_gauss)
        signal = np.zeros_like(freqs, dtype=float)
        for i, f in enumerate(freqs):
            h1_t = make_linearly_polarized_drive(
                params=params,
                b1_gauss=pulse_strength_gauss,
                f_drive_hz=f,
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

    times = np.linspace(t_start_s, t_stop_s, n_points)

    if method == "rwa":
        rwa_branch = _validate_rwa_branch(rwa_branch)
        h_rwa = rwa_hamiltonian(
            params,
            b0_gauss,
            pulse_frequency_hz,
            pulse_strength_gauss,
            drive_axis,
            branch=rwa_branch,
        )
        states = propagate_expm(psi0, h_rwa, times)  # (N, 9)
        signal = _readout_signal_batch(states, readout_ms, readout_state_index)
    else:
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
    params: NVParams | None = None,
    method: str = "rwa",
) -> Dict[str, np.ndarray]:
    """Ramsey-style T2* simulation: pi/2 - tau - pi/2."""
    method = _validate_method(method)
    params = params or NVParams()
    psi0 = _as_state(initial_state)
    h0 = static_hamiltonian(params=params, b0_gauss=b0_gauss)
    taus = np.linspace(t_start_s, t_stop_s, n_points)
    t_pi2 = estimate_pi2_time_s(params=params, b1_gauss=pulse_strength_gauss)
    f_eff = pulse_frequency_hz + detuning_hz

    if method == "rwa":
        rwa_branch = _validate_rwa_branch(rwa_branch)
        h_pulse_rwa = rwa_hamiltonian(
            params,
            b0_gauss,
            f_eff,
            pulse_strength_gauss,
            drive_axis,
            branch=rwa_branch,
        )
        h_free_rot = rotating_frame_h0(h0, f_eff, branch=rwa_branch)

        # pi/2 propagator (computed once)
        U_pi2 = _expm(-1j * h_pulse_rwa * t_pi2)  # (9, 9)

        # Sequence: pi/2 - free(tau) - pi/2
        psi1 = U_pi2 @ psi0  # (9,)
        psi2_batch = propagate_expm(psi1, h_free_rot, taus)  # (N, 9)
        psi3_batch = np.einsum('ij,nj->ni', U_pi2, psi2_batch)  # (N, 9)

        signal = _readout_signal_batch(psi3_batch, readout_ms, readout_state_index)
    else:
        signal = np.zeros_like(taus, dtype=float)
        h1_t = make_linearly_polarized_drive(
            params=params,
            b1_gauss=pulse_strength_gauss,
            f_drive_hz=f_eff,
            axis=drive_axis,
        )
        h_pulse = total_hamiltonian(h0, h1_t)

        # Compute first pi/2 once (fix A: was inside the loop)
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

            # Second pi/2 pulse — use correct time offset so the drive
            # phase cos(2*pi*f*(t_pi2+tau+t')) is continuous, matching
            # the physical MW source phase.
            t3 = t2 + t_pi2
            _, states_3 = propagate_state(
                psi0=psi_2,
                hamiltonian=h_pulse,
                t_span=(t2, t3),
                t_eval=np.array([t3]),
            )
            psi_3 = states_3[-1]
            signal[i] = _readout_signal(psi_3, readout_ms, readout_state_index)

    return {
        "taus_s": taus,
        "signal": signal,
        "estimated_pi2_s": np.array([t_pi2], dtype=float),
    }
