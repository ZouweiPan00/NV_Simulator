import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

omega_0 = 2*np.pi*2.87e9
omega   = 2*np.pi*2.87e9
points  = 1000
t       = np.linspace(0.0, 2.0e-6, points)

Omega = 2*np.pi*5e6
delta = omega_0 - omega

H = np.array(
    [[delta/2,  Omega/2],
     [Omega/2, -delta/2]],
    dtype=complex
)

U = la.expm(-1j * H[None, :, :] * t[:, None, None])  # (N,2,2)
psi0 = np.array([1, 0], dtype=complex)
psi_t = U @ psi0                                    # (N,2)

plt.plot(t*1e6, np.abs(psi_t[:, 0])**2, label='|0> population')
plt.plot(t*1e6, np.abs(psi_t[:, 1])**2, label='|1> population')
plt.xlabel('Time (us)')
plt.legend()
plt.show()
