"""Tests for Hamiltonian construction and pulse estimation."""
from __future__ import annotations

import numpy as np
import pytest

from NV_Simulator.constants import NVParams
from NV_Simulator.hamiltonian import (
    drive_hamiltonian,
    estimate_pi2_time_s,
    estimate_rabi_omega_rad_s,
    static_hamiltonian,
)
from NV_Simulator.operators import BASIS_STATES


class TestStaticHamiltonian:
    def test_hermitian(self):
        params = NVParams()
        h0 = static_hamiltonian(params, b0_gauss=500.0)
        np.testing.assert_allclose(h0, h0.conj().T, atol=1e-10)

    def test_shape(self):
        h0 = static_hamiltonian(NVParams(), b0_gauss=0.0)
        assert h0.shape == (9, 9)

    def test_diagonal_in_default_basis(self):
        """H0 should be diagonal in |ms,mI> basis (all terms commute with Sz⊗Iz)."""
        h0 = static_hamiltonian(NVParams(), b0_gauss=500.0)
        off_diag = h0 - np.diag(np.diag(h0))
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-10)

    def test_zero_field_degeneracy(self):
        """At B0=0 with mI=0, ms=+1 and ms=-1 levels are degenerate (A*Sz*Iz=0)."""
        params = NVParams()
        h0 = static_hamiltonian(params, b0_gauss=0.0)
        evals = np.real(np.diag(h0))
        # |+1,0> and |-1,0> should be degenerate (hyperfine A*Sz*Iz vanishes for mI=0)
        e_plus = evals[1]   # |+1,0>
        e_minus = evals[7]  # |-1,0>
        assert e_plus == pytest.approx(e_minus, rel=1e-10)
        # |+1,±1> and |-1,±1> differ by 2*A*mI due to hyperfine
        e_plus_p1 = evals[0]   # |+1,+1>
        e_minus_p1 = evals[6]  # |-1,+1>
        assert abs(e_plus_p1 - e_minus_p1) == pytest.approx(
            2 * abs(params.A), rel=1e-6
        )

    def test_zero_field_splitting(self):
        """D splits ms=0 from ms=±1 at zero field."""
        params = NVParams()
        h0 = static_hamiltonian(params, b0_gauss=0.0)
        evals = np.real(np.diag(h0))
        # ms=+1,mI=0 vs ms=0,mI=0
        e_plus1 = evals[1]  # |+1,0>
        e_0 = evals[4]       # |0,0>
        diff = e_plus1 - e_0
        # Should be close to D (since Q and A contributions are small for mI=0)
        assert abs(diff - params.D) == pytest.approx(0, abs=abs(params.A))


class TestDriveHamiltonian:
    def test_hermitian(self):
        params = NVParams()
        h1 = drive_hamiltonian(params, bx_gauss=1.0, by_gauss=0.0, bz_gauss=0.0)
        np.testing.assert_allclose(h1, h1.conj().T, atol=1e-14)

    def test_zero_drive(self):
        params = NVParams()
        h1 = drive_hamiltonian(params, bx_gauss=0.0, by_gauss=0.0, bz_gauss=0.0)
        np.testing.assert_allclose(h1, 0.0, atol=1e-14)


class TestPulseEstimation:
    def test_electron_pi2(self):
        params = NVParams()
        t_pi2 = estimate_pi2_time_s(params, b1_gauss=1.0, rwa_spin="electron")
        omega = estimate_rabi_omega_rad_s(params, b1_gauss=1.0, rwa_spin="electron")
        assert t_pi2 == pytest.approx(np.pi / (2.0 * omega))
        assert t_pi2 > 0

    def test_nuclear_pi2_much_longer(self):
        """Nuclear pi/2 should be ~|gamma_e/gamma_n| times longer than electron."""
        params = NVParams()
        t_e = estimate_pi2_time_s(params, b1_gauss=1.0, rwa_spin="electron")
        t_n = estimate_pi2_time_s(params, b1_gauss=1.0, rwa_spin="nuclear")
        ratio = t_n / t_e
        expected_ratio = abs(params.gamma_e) / abs(params.gamma_n)
        assert ratio == pytest.approx(expected_ratio, rel=1e-6)

    def test_zero_field_raises(self):
        with pytest.raises(ValueError):
            estimate_pi2_time_s(NVParams(), b1_gauss=0.0)
