import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

omega_0 = 2*np.pi*2.87e9
points  = 1000
omega   = 2*np.pi*np.linspace(2.5e9, 3.0e9, points)

Omega = 2*np.pi*5e6
t = np.pi/Omega

delta = omega_0 - omega  # (N,)

H = np.zeros((points, 2, 2), dtype=complex)
H[:, 0, 0] =  delta/2
H[:, 1, 1] = -delta/2
H[:, 0, 1] = Omega/2
H[:, 1, 0] = Omega/2

U = la.expm(-1j * H * t)     # (N,2,2)
psi0 = np.array([1, 0], dtype=complex)
psi_t = U @ psi0             # (N,2)

plt.plot(omega/(2*np.pi), np.abs(psi_t[:,0])**2, label='|0⟩ population')
plt.legend()
plt.show()
