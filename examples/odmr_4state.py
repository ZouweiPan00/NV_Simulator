"""ODMR with 4-state population readout at B0 = 500 G.

Initial state: |0,+1> (state 4), optically initialized at high field.
Input drive strength is specified as Rabi frequency Omega in MHz.

The plot shows the population of |0,+1> and the three ms=-1 sub-levels
as a function of drive frequency. Only the mI-conserving transition
|0,+1> <-> |-1,+1> produces a resonance peak.
"""

import numpy as np
import matplotlib.pyplot as plt

from NV_Simulator import NVParams
from NV_Simulator.operators import ket_from_quantum_numbers
from NV_Simulator.rwa import rwa_hamiltonian
from NV_Simulator.solver import propagate_expm

params = NVParams()
b0 = 500.0          # Gauss
omega_mhz = 5.0     # Rabi frequency in MHz
t_pulse = 1e-7      # 100 ns  (~ pi pulse at resonance)

# Convert Omega (MHz) to B1 (Gauss):  Omega_R = |gamma_e| * B1 / sqrt(2)
b1 = (2 * np.pi * omega_mhz * 1e6) * np.sqrt(2) / abs(params.gamma_e)

# Initial state: |0,+1>
psi0 = ket_from_quantum_numbers(0, +1)

# Frequency sweep around the |0,+1> <-> |-1,+1> transition (~1.4707 GHz)
n_points = 501
freqs_hz = np.linspace(1.460e9, 1.478e9, n_points)

# Build batch RWA Hamiltonians and propagate
h_batch = np.array([
    rwa_hamiltonian(params, b0, f, b1, axis="x", branch="ms_minus")
    for f in freqs_hz
])  # (N, 9, 9)
states = propagate_expm(psi0, h_batch, t_pulse)  # (N, 9)

# Extract individual state populations
# Basis: 1.|+1,+1> 2.|+1,0> 3.|+1,-1> 4.|0,+1> 5.|0,0> 6.|0,-1>
#        7.|-1,+1> 8.|-1,0> 9.|-1,-1>
prob = np.abs(states) ** 2  # (N, 9)

fig, ax = plt.subplots(figsize=(8, 5))
freqs_ghz = freqs_hz / 1e9

ax.plot(freqs_ghz, prob[:, 3], label=r"$|0,+1\rangle$",  linewidth=1.5)
ax.plot(freqs_ghz, prob[:, 6], label=r"$|-1,+1\rangle$", linewidth=1.5)
ax.plot(freqs_ghz, prob[:, 7], label=r"$|-1,0\rangle$",  linewidth=1.5)
ax.plot(freqs_ghz, prob[:, 8], label=r"$|-1,-1\rangle$", linewidth=1.5)

ax.set_xlabel("Drive frequency (GHz)")
ax.set_ylabel("Population")
ax.set_title(
    rf"ODMR  —  $B_0$={b0:.0f} G,  $\Omega$={omega_mhz} MHz,  "
    f"$t_{{pulse}}$={t_pulse*1e9:.0f} ns"
)
ax.legend()
plt.tight_layout()
plt.show()
