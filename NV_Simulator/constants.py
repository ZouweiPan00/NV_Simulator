from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NVParams:
    """Physical parameters in angular-frequency units (rad/s) unless noted."""

    # Zero-field splitting: D = 2*pi*2.87 GHz
    D: float = 2.0 * np.pi * 2.87e9
    # Nuclear quadrupole: Q = -2*pi*4.945 MHz
    Q: float = -2.0 * np.pi * 4.945e6
    # Hyperfine A. Here we use angular-frequency form by default.
    A: float = -2.0 * np.pi * 2.162e6

    # Gyromagnetic ratios in angular units: rad/(s*G)
    # gamma_e(Hz/G) = -2.8029e6 -> angular form:
    gamma_e: float = 2.0 * np.pi * (-2.8029e6)
    # gamma_n(Hz/G) = 0.3077e3 -> angular form:
    gamma_n: float = 2.0 * np.pi * (0.3077e3)

