"""ODMR example using simulate_odmr() and plotting the result."""

import matplotlib.pyplot as plt
import numpy as np

from NV_Simulator import NVParams, ket_from_index, simulate_odmr


def omega_hz_to_b1_gauss(omega_hz: float, params: NVParams) -> float:
    """Convert Rabi frequency Omega (Hz) to drive field B1 (Gauss)."""
    # 2*pi*Omega = -gamma_e*B1/sqrt(2)
    return -(2.0 * np.pi * omega_hz) * np.sqrt(2.0) / params.gamma_e


def main() -> None:
    params = NVParams()
    b0_gauss = 500.0
    omega_hz = 1.0e6
    pi_pulse_duration_s = 1/(2*omega_hz)  # s
    b1_gauss = omega_hz_to_b1_gauss(omega_hz, params)

    # Initial state |4> = |ms=0, mI=+1>
    psi0 = ket_from_index(4)

    result = simulate_odmr(
        initial_state=psi0,
        f_start_hz=1.460e9,
        f_stop_hz=1.478e9,
        pulse_strength_gauss=b1_gauss,
        pulse_duration_s=pi_pulse_duration_s,
        b0_gauss=b0_gauss,
        n_points=501,
        readout_state_index=4,
        method="rwa",
        rwa_branch="ms_minus",
        params=params,
    )

    freqs_ghz = result["frequencies_hz"] / 1e9
    p4 = result["signal"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(freqs_ghz, p4, color="C0", linewidth=1.6)
    ax.set_xlabel("Drive Frequency (GHz)")
    ax.set_ylabel(r"$P(|4\rangle)$")
    ax.set_title(
        rf"ODMR from simulate_odmr  ($B_0$={b0_gauss:.0f} G, $\Omega$={omega_hz/1e6:.1f} MHz)"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

