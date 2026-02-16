import numpy as np

from NV_Simulator import NVParams, ket_from_index, simulate_odmr, simulate_rabi, simulate_t2star


def main() -> None:
    psi0 = ket_from_index(4)  # |0,+1> = |4>
    b0 = 500.0
    omega_hz = 19.0e6  # Input Omega in Hz
    gamma_e = NVParams().gamma_e  # rad/(s*G)
    # 2*pi*Omega = -gamma_e*B1/sqrt(2)
    b1 = -(2.0 * np.pi * omega_hz) * np.sqrt(2.0) / gamma_e

    odmr = simulate_odmr(
        initial_state=psi0,
        f_start_hz=1.40e9,
        f_stop_hz=1.50e9,
        pulse_strength_gauss=b1,
        pulse_duration_s=0.4e-6,
        b0_gauss=b0,
        n_points=81,
        readout_state_index=4,
        method="rwa",
        rwa_branch="ms_minus",
    )
    print("ODMR points:", len(odmr["frequencies_hz"]))
    print("ODMR P(|4>) range:", float(np.min(odmr["signal"])), float(np.max(odmr["signal"])))

    rabi = simulate_rabi(
        initial_state=psi0,
        pulse_frequency_hz=1.470712e9,
        t_start_s=0.0,
        t_stop_s=3.0e-6,
        pulse_strength_gauss=b1,
        b0_gauss=b0,
        n_points=101,
        readout_state_index=4,
        method="rwa",
        rwa_branch="ms_minus",
    )
    print("Rabi points:", len(rabi["times_s"]))
    print("Rabi P(|4>) range:", float(np.min(rabi["signal"])), float(np.max(rabi["signal"])))

    t2star = simulate_t2star(
        initial_state=psi0,
        detuning_hz=0.6e6,
        pulse_strength_gauss=b1,
        pulse_frequency_hz=1.470712e9,
        t_start_s=0.0,
        t_stop_s=6.0e-6,
        b0_gauss=b0,
        n_points=61,
        readout_state_index=4,
        method="rwa",
        rwa_branch="ms_minus",
    )
    print("T2* points:", len(t2star["taus_s"]))
    print("T2* estimated pi/2 (s):", float(t2star["estimated_pi2_s"][0]))
    print("T2* P(|4>) range:", float(np.min(t2star["signal"])), float(np.max(t2star["signal"])))


if __name__ == "__main__":
    main()
