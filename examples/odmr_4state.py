"""ODMR with 4-state population readout at B0 = 500 G.

Initial state: equal superposition of |0,+1>, |0,0>, |0,-1>
(representing equal population across all three mI sub-levels of ms=0,
achievable via optical pumping at high field).

Since Sx drives preserve mI, each sub-level shows an independent resonance
at a hyperfine-shifted frequency. The plot reveals the three-line hyperfine
structure of the ms=0 <-> -1 transition.
"""

import numpy as np
import matplotlib.pyplot as plt

from NV_Simulator import NVParams
from NV_Simulator.operators import ket_from_quantum_numbers
from NV_Simulator.rwa import rwa_hamiltonian
from NV_Simulator.solver import propagate_expm

params = NVParams()
b0 = 500.0      # Gauss
b1 = 2.5        # Gauss  ->  Omega_R ~ 2*pi*5 MHz
t_pulse = 1e-7  # 100 ns  (~ pi pulse at resonance)

# Equal superposition of ms=0 sub-levels
psi0 = (
    ket_from_quantum_numbers(0, +1)
    + ket_from_quantum_numbers(0, 0)
    + ket_from_quantum_numbers(0, -1)
) / np.sqrt(3)

# Frequency sweep across the three hyperfine transitions (~1.466 - 1.471 GHz)
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
ax.plot(freqs_ghz, prob[:, 4], label=r"$|0,\;0\rangle$", linewidth=1.5)
ax.plot(freqs_ghz, prob[:, 5], label=r"$|0,-1\rangle$",  linewidth=1.5)
ax.plot(freqs_ghz, prob[:, 6] + prob[:, 7] + prob[:, 8],
        label=r"$P(m_s=-1)$ total", linewidth=1.5, linestyle="--", color="black")

ax.set_xlabel("Drive frequency (GHz)")
ax.set_ylabel("Population")
ax.set_title(f"ODMR  —  $B_0$={b0:.0f} G,  $B_1$={b1} G,  $t_{{pulse}}$={t_pulse*1e9:.0f} ns")
ax.legend()
ax.set_ylim(-0.02, 0.45)
plt.tight_layout()
plt.show()
