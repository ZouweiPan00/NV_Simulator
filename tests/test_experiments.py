"""Tests for high-level experiment simulations."""
from __future__ import annotations

import numpy as np
import pytest

from NV_Simulator.constants import NVParams
from NV_Simulator.experiments import simulate_odmr, simulate_rabi, simulate_t2star
from NV_Simulator.operators import ket_from_quantum_numbers


@pytest.fixture
def initial_state():
    return ket_from_quantum_numbers(0, +1)


class TestSimulateODMR:
    def test_output_keys(self, initial_state):
        result = simulate_odmr(
            initial_state=initial_state,
            f_start_hz=2.8e9,
            f_stop_hz=2.9e9,
            pulse_strength_gauss=1.0,
            pulse_duration_s=1e-6,
            b0_gauss=500.0,
            n_points=11,
            method="rwa",
        )
        assert "frequencies_hz" in result
        assert "signal" in result
        assert result["frequencies_hz"].shape == (11,)
        assert result["signal"].shape == (11,)

    def test_signal_range(self, initial_state):
        result = simulate_odmr(
            initial_state=initial_state,
            f_start_hz=2.8e9,
            f_stop_hz=2.9e9,
            pulse_strength_gauss=1.0,
            pulse_duration_s=1e-6,
            b0_gauss=500.0,
            n_points=21,
            method="rwa",
        )
        assert np.all(result["signal"] >= -0.01)
        assert np.all(result["signal"] <= 1.01)

    def test_rwa_vs_ode_consistency(self, initial_state):
        """RWA and ODE should give similar results at moderate field."""
        common = dict(
            initial_state=initial_state,
            f_start_hz=2.86e9,
            f_stop_hz=2.88e9,
            pulse_strength_gauss=0.5,
            pulse_duration_s=5e-7,
            b0_gauss=500.0,
            n_points=5,
        )
        rwa = simulate_odmr(**common, method="rwa")
        ode = simulate_odmr(**common, method="ode")
        # They should agree within ~10% for reasonable parameters
        np.testing.assert_allclose(rwa["signal"], ode["signal"], atol=0.15)


class TestSimulateRabi:
    def test_output_keys(self, initial_state):
        result = simulate_rabi(
            initial_state=initial_state,
            pulse_frequency_hz=2.87e9,
            t_start_s=0.0,
            t_stop_s=1e-6,
            pulse_strength_gauss=1.0,
            b0_gauss=500.0,
            n_points=21,
            method="rwa",
        )
        assert "times_s" in result
        assert "signal" in result
        assert "states" in result
        assert result["times_s"].shape == (21,)
        assert result["signal"].shape == (21,)
        assert result["states"].shape == (21, 9)

    def test_rabi_oscillation(self, initial_state):
        """Signal should oscillate (not be flat) when driven on resonance."""
        from NV_Simulator.hamiltonian import static_hamiltonian
        from NV_Simulator.rwa import transition_frequencies_hz

        # Find actual resonance frequency at B0=500G
        h0 = static_hamiltonian(NVParams(), b0_gauss=500.0)
        freqs = transition_frequencies_hz(h0, "electron")
        # Pair [3,6] = |0,+1> <-> |-1,+1> (ms_minus branch)
        f_res = freqs[3]

        result = simulate_rabi(
            initial_state=initial_state,
            pulse_frequency_hz=f_res,
            t_start_s=0.0,
            t_stop_s=2e-6,
            pulse_strength_gauss=1.0,
            b0_gauss=500.0,
            n_points=101,
            method="rwa",
        )
        signal = result["signal"]
        assert signal.max() - signal.min() > 0.1


class TestSimulateT2Star:
    def test_output_keys(self, initial_state):
        result = simulate_t2star(
            initial_state=initial_state,
            detuning_hz=1e6,
            pulse_strength_gauss=1.0,
            pulse_frequency_hz=2.87e9,
            t_start_s=0.0,
            t_stop_s=1e-6,
            b0_gauss=500.0,
            n_points=11,
            method="rwa",
        )
        assert "taus_s" in result
        assert "signal" in result
        assert "estimated_pi2_s" in result
        assert result["taus_s"].shape == (11,)
        assert result["signal"].shape == (11,)
