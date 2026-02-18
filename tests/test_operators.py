"""Tests for spin-1 operators and basis state constructors."""
from __future__ import annotations

import numpy as np
import pytest

from NV_Simulator.operators import (
    BASIS_STATES,
    basis_index,
    electron_operators,
    ket_from_index,
    ket_from_quantum_numbers,
    kron_op,
    nuclear_operators,
    spin_1_matrices,
)


class TestSpin1Matrices:
    """Verify algebraic properties of spin-1 matrices."""

    def test_commutation_xy(self):
        sx, sy, sz = spin_1_matrices()
        comm = sx @ sy - sy @ sx
        expected = 1j * sz
        np.testing.assert_allclose(comm, expected, atol=1e-14)

    def test_commutation_yz(self):
        sx, sy, sz = spin_1_matrices()
        comm = sy @ sz - sz @ sy
        expected = 1j * sx
        np.testing.assert_allclose(comm, expected, atol=1e-14)

    def test_commutation_zx(self):
        sx, sy, sz = spin_1_matrices()
        comm = sz @ sx - sx @ sz
        expected = 1j * sy
        np.testing.assert_allclose(comm, expected, atol=1e-14)

    def test_s_squared(self):
        sx, sy, sz = spin_1_matrices()
        s2 = sx @ sx + sy @ sy + sz @ sz
        # S(S+1) = 1(1+1) = 2 for spin-1
        expected = 2.0 * np.eye(3, dtype=complex)
        np.testing.assert_allclose(s2, expected, atol=1e-14)

    def test_hermitian(self):
        sx, sy, sz = spin_1_matrices()
        for s in [sx, sy, sz]:
            np.testing.assert_allclose(s, s.conj().T, atol=1e-14)

    def test_sz_eigenvalues(self):
        _, _, sz = spin_1_matrices()
        eigenvalues = np.sort(np.linalg.eigvalsh(sz))
        np.testing.assert_allclose(eigenvalues, [-1, 0, 1], atol=1e-14)


class TestBasisStates:
    def test_basis_states_count(self):
        assert len(BASIS_STATES) == 9

    def test_basis_index_round_trip(self):
        for i, (ms, mI) in enumerate(BASIS_STATES):
            assert basis_index(ms, mI) == i

    def test_basis_index_invalid(self):
        with pytest.raises(ValueError):
            basis_index(2, 0)

    def test_ket_from_index(self):
        for i in range(1, 10):
            ket = ket_from_index(i)
            assert ket.shape == (9,)
            assert ket[i - 1] == 1.0
            assert np.linalg.norm(ket) == pytest.approx(1.0)

    def test_ket_from_index_invalid(self):
        with pytest.raises(ValueError):
            ket_from_index(0)
        with pytest.raises(ValueError):
            ket_from_index(10)

    def test_ket_from_quantum_numbers(self):
        ket = ket_from_quantum_numbers(0, +1)
        idx = basis_index(0, +1)
        assert ket[idx] == 1.0
        assert np.linalg.norm(ket) == pytest.approx(1.0)

    def test_kron_op_shape(self):
        i3 = np.eye(3, dtype=complex)
        result = kron_op(i3, i3)
        assert result.shape == (9, 9)
        np.testing.assert_allclose(result, np.eye(9, dtype=complex), atol=1e-14)

    def test_electron_nuclear_are_same(self):
        """electron and nuclear operators use the same spin-1 algebra."""
        e = electron_operators()
        n = nuclear_operators()
        for ei, ni in zip(e, n):
            np.testing.assert_array_equal(ei, ni)
