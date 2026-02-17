"""Rabi oscillation with 4-state population readout.

Initial state: equal superposition of |0,+1>, |0,0>, |0,-1>.
Drive frequency: on resonance with the mI=0 transition (|0,0> <-> |-1,0>).

The mI=0 component undergoes full on-resonance Rabi oscillation.
The mI=+/-1 components are detuned by +/-A (hyperfine) and show
reduced-amplitude, higher-frequency oscillations.
"""

import numpy as np
import matplotlib.pyplot as plt

from NV_Simulator import NVParams
from NV_Simulator.operators import ket_from_quantum_numbers
from NV_Simulator.hamiltonian import static_hamiltonian
from NV_Simulator.rwa import rwa_hamiltonian
from NV_Simulator.solver import propagate_expm

params = NVParams()
b0 = 20.0   # Gauss
b1 = 2.5    # Gauss  ->  Omega_R ~ 2*pi*5 MHz

# Equal superposition of ms=0 sub-levels
psi0 = (
    ket_from_quantum_numbers(0, +1)
    + ket_from_quantum_numbers(0, 0)
    + ket_from_quantum_numbers(0, -1)
) / np.sqrt(3)

# Drive at the mI=0 resonance frequency
h0 = static_hamiltonian(params=params, b0_gauss=b0)
diag = np.real(np.diag(h0))
f_res_hz = (diag[7] - diag[4]) / (2 * np.pi)   # E(|-1,0>) - E(|0,0>)

# Time sweep
n_points = 501
times_s = np.linspace(0, 1e-6, n_points)

# Propagate
h_rwa = rwa_hamiltonian(params, b0, f_res_hz, b1, axis="x", branch="ms_minus")
states = propagate_expm(psi0, h_rwa, times_s)  # (N, 9)

# Extract individual state populations
prob = np.abs(states) ** 2  # (N, 9)

fig, ax = plt.subplots(figsize=(8, 5))
times_us = times_s * 1e6

ax.plot(times_us, prob[:, 3], label=r"$|0,+1\rangle$",   linewidth=1.5)
ax.plot(times_us, prob[:, 4], label=r"$|0,\;0\rangle$",  linewidth=1.5)
ax.plot(times_us, prob[:, 5], label=r"$|0,-1\rangle$",    linewidth=1.5)
ax.plot(times_us, prob[:, 7], label=r"$|-1,0\rangle$",    linewidth=1.5, linestyle="--")

ax.set_xlabel(r"Time ($\mu$s)")
ax.set_ylabel("Population")
ax.set_title(
    f"Rabi  —  $B_0$={b0} G,  $B_1$={b1} G,  "
    f"$f_{{drive}}$={f_res_hz/1e9:.4f} GHz"
)
ax.legend()
ax.set_ylim(-0.02, 0.45)
plt.tight_layout()
plt.show()
