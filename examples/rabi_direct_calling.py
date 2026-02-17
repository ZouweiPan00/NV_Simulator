"""Rabi example using simulate_rabi() and plotting the result."""

import matplotlib.pyplot as plt
import numpy as np

from NV_Simulator import NVParams, ket_from_index, simulate_rabi
from NV_Simulator.hamiltonian import static_hamiltonian


def omega_hz_to_b1_gauss(omega_hz: float, params: NVParams) -> float:
    """Convert Rabi frequency Omega (Hz) to drive field B1 (Gauss)."""
    # 2*pi*Omega = -gamma_e*B1/sqrt(2)
    return -(2.0 * np.pi * omega_hz) * np.sqrt(2.0) / params.gamma_e


def main() -> None:
    params = NVParams()
    b0_gauss = 500.0
    omega_hz = 5.0e6
    b1_gauss = omega_hz_to_b1_gauss(omega_hz, params)

    # Initial state |4> = |ms=0, mI=+1>
    psi0 = ket_from_index(4)

    # Resonance frequency for |4> <-> |7> from static Hamiltonian diagonal.
    h0 = static_hamiltonian(params=params, b0_gauss=b0_gauss)
    e_diag = np.real(np.diag(h0))
    f_res_hz = (e_diag[6] - e_diag[3]) / (2.0 * np.pi)

    result = simulate_rabi(
        initial_state=psi0,
        pulse_frequency_hz=f_res_hz,
        t_start_s=0.0,
        t_stop_s=1.0e-6,
        pulse_strength_gauss=b1_gauss,
        b0_gauss=b0_gauss,
        n_points=501,
        readout_state_index=4,
        method="rwa",
        rwa_branch="ms_minus",
        params=params,
    )

    times_us = result["times_s"] * 1e6
    p4 = result["signal"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times_us, p4, color="C0", linewidth=1.6)
    ax.set_xlabel(r"Time ($\mu$s)")
    ax.set_ylabel(r"$P(|4\rangle)$")
    ax.set_title(
        rf"Rabi from simulate_rabi  ($B_0$={b0_gauss:.0f} G, $\Omega$={omega_hz/1e6:.1f} MHz)"
        + "\n"
        + rf"$f_{{drive}}$={f_res_hz/1e9:.6f} GHz"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

