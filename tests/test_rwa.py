"""Tests for RWA Hamiltonian construction."""
from __future__ import annotations

import numpy as np
import pytest

from NV_Simulator.constants import NVParams
from NV_Simulator.hamiltonian import static_hamiltonian
from NV_Simulator.rwa import (
    cross_spin_min_detuning_hz,
    rotating_frame_h0,
    rwa_drive_matrix,
    rwa_hamiltonian,
    transition_frequencies_hz,
)


class TestTransitionFrequencies:
    def test_electron_six_transitions(self):
        h0 = static_hamiltonian(NVParams(), b0_gauss=500.0)
        freqs = transition_frequencies_hz(h0, "electron")
        assert freqs.shape == (6,)
        assert np.all(freqs > 0)

    def test_nuclear_six_transitions(self):
        h0 = static_hamiltonian(NVParams(), b0_gauss=500.0)
        freqs = transition_frequencies_hz(h0, "nuclear")
        assert freqs.shape == (6,)
        assert np.all(freqs > 0)

    def test_electron_near_d(self):
        """At moderate field, electron transitions should be near D ≈ 2.87 GHz."""
        h0 = static_hamiltonian(NVParams(), b0_gauss=500.0)
        freqs = transition_frequencies_hz(h0, "electron")
        # All should be within a few hundred MHz of 2.87 GHz
        for f in freqs:
            assert 1e9 < f < 5e9

    def test_invalid_spin(self):
        h0 = static_hamiltonian(NVParams(), b0_gauss=0.0)
        with pytest.raises(ValueError):
            transition_frequencies_hz(h0, "invalid")


class TestRotatingFrameH0:
    def test_hermitian(self):
        h0 = static_hamiltonian(NVParams(), b0_gauss=500.0)
        h0_rot = rotating_frame_h0(h0, 2.87e9, branch="ms_minus", rwa_spin="electron")
        np.testing.assert_allclose(h0_rot, h0_rot.conj().T, atol=1e-10)

    def test_invalid_branch(self):
        h0 = static_hamiltonian(NVParams(), b0_gauss=0.0)
        with pytest.raises(ValueError):
            rotating_frame_h0(h0, 2.87e9, branch="invalid")


class TestRwaDriveMatrix:
    def test_hermitian(self):
        params = NVParams()
        h1 = rwa_drive_matrix(params, b1_gauss=1.0, axis="x", rwa_spin="electron")
        np.testing.assert_allclose(h1, h1.conj().T, atol=1e-14)

    def test_z_axis_zero(self):
        """z-axis drive should give zero matrix in RWA."""
        params = NVParams()
        h1 = rwa_drive_matrix(params, b1_gauss=1.0, axis="z", rwa_spin="electron")
        np.testing.assert_allclose(h1, 0.0, atol=1e-14)

    def test_nuclear_drive(self):
        params = NVParams()
        h1 = rwa_drive_matrix(params, b1_gauss=1.0, axis="x", rwa_spin="nuclear")
        assert h1.shape == (9, 9)
        assert np.any(np.abs(h1) > 0)


class TestRwaHamiltonian:
    def test_hermitian(self):
        h = rwa_hamiltonian(NVParams(), 500.0, 2.87e9, 1.0)
        np.testing.assert_allclose(h, h.conj().T, atol=1e-10)

    def test_shape(self):
        h = rwa_hamiltonian(NVParams(), 500.0, 2.87e9, 1.0)
        assert h.shape == (9, 9)


class TestCrossSpinDetuning:
    def test_positive(self):
        h0 = static_hamiltonian(NVParams(), b0_gauss=500.0)
        det = cross_spin_min_detuning_hz(h0, 2.87e9, "electron")
        assert det > 0
