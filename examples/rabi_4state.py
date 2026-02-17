"""Rabi oscillation with population readout at B0 = 500 G.

Initial state: |0,+1> (state 4), optically initialized at high field.
Input drive strength is specified as Rabi frequency Omega in MHz.
Drive frequency: on resonance with |0,+1> <-> |-1,+1>.

The plot shows P(|0,+1>) oscillating with |-1,+1> at the Rabi frequency.
"""

import numpy as np
import matplotlib.pyplot as plt

from NV_Simulator import NVParams
from NV_Simulator.operators import ket_from_quantum_numbers
from NV_Simulator.hamiltonian import static_hamiltonian
from NV_Simulator.rwa import rwa_hamiltonian
from NV_Simulator.solver import propagate_expm

params = NVParams()
b0 = 500.0          # Gauss
omega_mhz = 5.0     # Rabi frequency in MHz

# Convert Omega (MHz) to B1 (Gauss):  Omega_R = |gamma_e| * B1 / sqrt(2)
b1 = (2 * np.pi * omega_mhz * 1e6) * np.sqrt(2) / abs(params.gamma_e)

# Initial state: |0,+1>
psi0 = ket_from_quantum_numbers(0, +1)

# Drive at the mI=+1 resonance frequency: |0,+1> <-> |-1,+1>
h0 = static_hamiltonian(params=params, b0_gauss=b0)
diag = np.real(np.diag(h0))
f_res_hz = (diag[6] - diag[3]) / (2 * np.pi)   # E(|-1,+1>) - E(|0,+1>)

# Time sweep
n_points = 501
times_s = np.linspace(0, 1e-6, n_points)

# Propagate
h_rwa = rwa_hamiltonian(params, b0, f_res_hz, b1, axis="x", branch="ms_minus")
states = propagate_expm(psi0, h_rwa, times_s)  # (N, 9)

# Extract population of |0,+1>
prob = np.abs(states[:, 3]) ** 2  # (N,)

fig, ax = plt.subplots(figsize=(8, 5))
times_us = times_s * 1e6

ax.plot(times_us, prob, linewidth=1.5, color="C0")

ax.set_xlabel(r"Time ($\mu$s)")
ax.set_ylabel(r"$P(|0,+1\rangle)$")
ax.set_title(
    rf"Rabi  —  $B_0$={b0:.0f} G,  $\Omega$={omega_mhz} MHz,  "
    f"$f_{{drive}}$={f_res_hz/1e9:.4f} GHz"
)
plt.tight_layout()
plt.show()
