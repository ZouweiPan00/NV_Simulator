"""Tests for measurement operators and expectation values."""
from __future__ import annotations

import numpy as np
import pytest

from NV_Simulator.observables import expectation, ms_population, projector_ms
from NV_Simulator.operators import ket_from_quantum_numbers


class TestProjectorMs:
    def test_hermitian(self):
        for ms in [+1, 0, -1]:
            p = projector_ms(ms)
            np.testing.assert_allclose(p, p.conj().T, atol=1e-14)

    def test_idempotent(self):
        for ms in [+1, 0, -1]:
            p = projector_ms(ms)
            np.testing.assert_allclose(p @ p, p, atol=1e-14)

    def test_completeness(self):
        """P(+1) + P(0) + P(-1) = I."""
        total = projector_ms(+1) + projector_ms(0) + projector_ms(-1)
        np.testing.assert_allclose(total, np.eye(9, dtype=complex), atol=1e-14)

    def test_trace(self):
        """Each projector has trace 3 (projects onto 3 mI substates)."""
        for ms in [+1, 0, -1]:
            assert np.trace(projector_ms(ms)).real == pytest.approx(3.0)

    def test_invalid_ms(self):
        with pytest.raises(ValueError):
            projector_ms(2)


class TestExpectation:
    def test_identity(self):
        psi = ket_from_quantum_numbers(0, +1)
        val = expectation(psi, np.eye(9, dtype=complex))
        assert val == pytest.approx(1.0)

    def test_projector(self):
        psi = ket_from_quantum_numbers(0, +1)
        # Should have 100% population in ms=0
        assert ms_population(psi, ms=0) == pytest.approx(1.0)
        assert ms_population(psi, ms=+1) == pytest.approx(0.0)
        assert ms_population(psi, ms=-1) == pytest.approx(0.0)
