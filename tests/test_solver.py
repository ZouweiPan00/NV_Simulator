"""Tests for ODE and matrix exponential propagators."""
from __future__ import annotations

import numpy as np
import pytest

from NV_Simulator.solver import propagate_expm, propagate_state


class TestPropagateState:
    def test_unitarity(self):
        """Norm should be preserved under time evolution."""
        d = 9
        # Random Hermitian Hamiltonian
        rng = np.random.default_rng(42)
        h = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        h = (h + h.conj().T) / 2

        psi0 = np.zeros(d, dtype=complex)
        psi0[0] = 1.0

        times, states = propagate_state(psi0, h, t_span=(0.0, 1e-9))
        for psi in states:
            assert np.linalg.norm(psi) == pytest.approx(1.0, abs=1e-6)

    def test_trivial_evolution(self):
        """Zero Hamiltonian should preserve state."""
        h = np.zeros((9, 9), dtype=complex)
        psi0 = np.zeros(9, dtype=complex)
        psi0[3] = 1.0
        times, states = propagate_state(psi0, h, t_span=(0.0, 1e-6))
        np.testing.assert_allclose(np.abs(states[-1]), np.abs(psi0), atol=1e-10)

    def test_zero_norm_raises(self):
        psi0 = np.zeros(9, dtype=complex)
        h = np.eye(9, dtype=complex)
        with pytest.raises(ValueError, match="zero norm"):
            propagate_state(psi0, h, t_span=(0.0, 1e-9))

    def test_2d_state_raises(self):
        psi0 = np.zeros((9, 1), dtype=complex)
        h = np.eye(9, dtype=complex)
        with pytest.raises(ValueError):
            propagate_state(psi0, h, t_span=(0.0, 1e-9))


class TestPropagateExpm:
    def test_mode1_single_point(self):
        """H (d,d) + scalar t -> (d,)."""
        h = np.diag(np.arange(9, dtype=complex))
        psi0 = np.zeros(9, dtype=complex)
        psi0[0] = 1.0
        result = propagate_expm(psi0, h, 0.0)
        assert result.shape == (9,)
        np.testing.assert_allclose(np.abs(result), np.abs(psi0), atol=1e-14)

    def test_mode2_batch_h(self):
        """H (N,d,d) + scalar t -> (N,d)."""
        N = 5
        h_batch = np.zeros((N, 9, 9), dtype=complex)
        for i in range(N):
            h_batch[i] = np.diag(np.arange(9, dtype=complex)) * (i + 1)
        psi0 = np.zeros(9, dtype=complex)
        psi0[0] = 1.0
        result = propagate_expm(psi0, h_batch, 0.0)
        assert result.shape == (N, 9)

    def test_mode3_batch_t(self):
        """H (d,d) + t (N,) -> (N,d)."""
        h = np.diag(np.arange(9, dtype=complex))
        psi0 = np.zeros(9, dtype=complex)
        psi0[0] = 1.0
        t_arr = np.linspace(0, 1e-9, 10)
        result = propagate_expm(psi0, h, t_arr)
        assert result.shape == (10, 9)

    def test_unitarity_expm(self):
        """propagate_expm should preserve norm."""
        rng = np.random.default_rng(123)
        h = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
        h = (h + h.conj().T) / 2
        psi0 = np.zeros(9, dtype=complex)
        psi0[0] = 1.0
        t_arr = np.linspace(0, 1e-9, 20)
        states = propagate_expm(psi0, h, t_arr)
        for psi in states:
            assert np.linalg.norm(psi) == pytest.approx(1.0, abs=1e-10)

    def test_batch_h_array_t_raises(self):
        """Batch H + array t should raise."""
        h = np.zeros((5, 9, 9), dtype=complex)
        psi0 = np.zeros(9, dtype=complex)
        psi0[0] = 1.0
        with pytest.raises(ValueError, match="Cannot combine"):
            propagate_expm(psi0, h, np.array([1.0, 2.0]))
