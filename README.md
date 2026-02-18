# NV\_Simulator

![Python >= 3.9](https://img.shields.io/badge/python-%3E%3D3.9-blue)
![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-green)
![NumPy >= 1.23](https://img.shields.io/badge/numpy-%3E%3D1.23-orange)
![SciPy >= 1.9](https://img.shields.io/badge/scipy-%3E%3D1.9-orange)

A Python package for simulating the **9-level nitrogen-vacancy (NV) center** in diamond: an electron spin-1 coupled to a $^{14}\text{N}$ nuclear spin-1. Two solver backends are provided:

| Solver | Method | Speed | Accuracy |
|--------|--------|-------|----------|
| **ODE** (`method="ode"`) | Full time-dependent Schrodinger equation via `scipy.integrate.solve_ivp` (DOP853) | Slow -- must resolve GHz oscillations | Exact (no approximations) |
| **RWA** (`method="rwa"`) | Rotating-wave approximation transforms to a time-independent 9x9 Hamiltonian; batch propagation via `scipy.linalg.expm` | Orders of magnitude faster | Excellent near resonance; keeps the targeted spin channel and can auto-fallback to ODE near cross-spin resonances |

Three standard NV experiments are built in: **ODMR**, **Rabi oscillations**, and **Ramsey T2\***.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Physics and Model](#physics-and-model)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [License](#license)

---

## Installation

```bash
pip install -e .
```

Requirements: Python >= 3.9, NumPy >= 1.23, SciPy >= 1.9.

---

## Quick Start

### RWA solver (fast, recommended)

```python
import numpy as np
from NV_Simulator import (
    NVParams,
    ket_from_quantum_numbers,
    simulate_odmr,
    simulate_rabi,
    simulate_t2star,
)

# Prepare initial state |ms=0, mI=0>
psi0 = ket_from_quantum_numbers(ms=0, mI=0)

# --- ODMR spectrum around the ms=0 <-> -1 transition at B0 = 20 G ---
odmr = simulate_odmr(
    initial_state=psi0,
    f_start_hz=2.81e9,
    f_stop_hz=2.93e9,
    pulse_strength_gauss=0.05,
    pulse_duration_s=0.5e-6,
    b0_gauss=20.0,
    n_points=201,
    readout_ms=0,
    method="rwa",               # fast RWA + expm solver
    rwa_branch="ms_minus",      # target the ms=0 <-> -1 transition
)
freqs = odmr["frequencies_hz"]  # (201,) array in Hz
signal = odmr["signal"]         # (201,) population of ms=0 subspace

# --- Rabi oscillation at a fixed drive frequency ---
rabi = simulate_rabi(
    initial_state=psi0,
    pulse_frequency_hz=2.87e9,
    t_start_s=0.0,
    t_stop_s=2.0e-6,
    pulse_strength_gauss=0.05,
    b0_gauss=20.0,
    n_points=401,
    readout_ms=0,
    method="rwa",
    rwa_branch="ms_minus",
)
times = rabi["times_s"]         # (401,) array in seconds
rabi_signal = rabi["signal"]    # (401,) ms=0 population vs time
states = rabi["states"]         # (401, 9) full state vectors

# --- Ramsey T2* measurement ---
t2star = simulate_t2star(
    initial_state=psi0,
    detuning_hz=1.0e6,
    pulse_strength_gauss=0.05,
    pulse_frequency_hz=2.87e9,
    t_start_s=0.0,
    t_stop_s=5.0e-6,
    b0_gauss=20.0,
    n_points=201,
    method="rwa",
    rwa_branch="ms_minus",
)
taus = t2star["taus_s"]                     # (201,) free-precession delays
t2_signal = t2star["signal"]               # (201,) ms=0 population
pi2_time = t2star["estimated_pi2_s"][0]    # estimated pi/2 pulse duration
```

### ODE solver (exact, slower)

Switch any experiment to the full time-dependent solver by passing `method="ode"`:

```python
odmr_ode = simulate_odmr(
    initial_state=psi0,
    f_start_hz=2.81e9,
    f_stop_hz=2.93e9,
    pulse_strength_gauss=0.05,
    pulse_duration_s=0.5e-6,
    b0_gauss=20.0,
    n_points=51,                # fewer points recommended -- ODE is slow
    method="ode",
)
```

The ODE solver integrates the full Schrodinger equation including both electron and nuclear drive terms. Use it to validate RWA results or when operating far from resonance.

### Custom readout

By default, experiments measure the total population in the $m_s = 0$ subspace (summed over all three nuclear states). To read out a single basis state instead, use `readout_state_index` (1-based):

```python
odmr = simulate_odmr(
    initial_state=psi0,
    f_start_hz=2.81e9,
    f_stop_hz=2.93e9,
    pulse_strength_gauss=0.05,
    pulse_duration_s=0.5e-6,
    b0_gauss=20.0,
    readout_state_index=5,      # read out P(|0,0>) specifically
    method="rwa",
)
```

---

## Physics and Model

### Hilbert space

The system lives in a 9-dimensional Hilbert space,

$$\mathcal{H} = \mathcal{H}_e \otimes \mathcal{H}_n$$

where $\mathcal{H}_e$ is the electron spin-1 space ($m_s \in \{+1, 0, -1\}$) and $\mathcal{H}_n$ is the $^{14}\text{N}$ nuclear spin-1 space ($m_I \in \{+1, 0, -1\}$).

The computational basis is ordered as:

| Index (1-based) | State |
|:---:|:---:|
| 1 | $\|+1,+1\rangle$ |
| 2 | $\|+1,0\rangle$ |
| 3 | $\|+1,-1\rangle$ |
| 4 | $\|0,+1\rangle$ |
| 5 | $\|0,0\rangle$ |
| 6 | $\|0,-1\rangle$ |
| 7 | $\|-1,+1\rangle$ |
| 8 | $\|-1,0\rangle$ |
| 9 | $\|-1,-1\rangle$ |

### Static Hamiltonian

All internal Hamiltonian terms are in angular-frequency units ($\hbar = 1$):

$$H_0 = D S_z^2 \otimes \mathbb{1}_3 + Q \mathbb{1}_3 \otimes I_z^2 + \omega_e S_z \otimes \mathbb{1}_3 + \omega_n \mathbb{1}_3 \otimes I_z + A S_z \otimes I_z$$

where $\omega_e = -\gamma_e B_0$ and $\omega_n = -\gamma_n B_0$.

### Default physical parameters

| Parameter | Symbol | Default value |
|-----------|--------|---------------|
| Zero-field splitting | $D$ | $2\pi \times 2.87$ GHz |
| Nuclear quadrupole | $Q$ | $-2\pi \times 4.945$ MHz |
| Hyperfine coupling | $A$ | $-2\pi \times 2.162$ MHz |
| Electron gyromagnetic ratio | $\gamma_e$ | $2\pi \times (-2.8029)$ MHz/G |
| Nuclear gyromagnetic ratio | $\gamma_n$ | $2\pi \times 0.3077$ kHz/G |

These are stored in the `NVParams` frozen dataclass and can be overridden:

```python
from NV_Simulator import NVParams
custom = NVParams(D=2.0 * np.pi * 2.88e9)  # custom zero-field splitting
```

### Time-dependent drive (ODE path)

The microwave drive along axis $\alpha \in \{x, y, z\}$ is:

$$H_1(t) = -\gamma_e B_1 \cos(2\pi f t) S_\alpha \otimes \mathbb{1}_3 - \gamma_n B_1 \cos(2\pi f t) \mathbb{1}_3 \otimes I_\alpha$$

The ODE solver integrates $i\hbar \partial_t |\psi\rangle = [H_0 + H_1(t)] |\psi\rangle$ using `scipy.integrate.solve_ivp` with the DOP853 (8th-order Dormand-Prince) method.

### RWA Hamiltonian (RWA path)

For microwave ESR control (electron-spin transitions), moving to the rotating frame and dropping counter-rotating terms yields a time-independent Hamiltonian:

$$H_{\text{RWA}}^{(e)} = H_0 \pm \omega_d (S_z \otimes \mathbb{1}_3) + \frac{-\gamma_e B_1}{2} (S_\perp \otimes \mathbb{1}_3)$$

- The $+$ sign selects the $m_s = 0 \leftrightarrow -1$ transition (`rwa_branch="ms_minus"`).
- The $-$ sign selects the $m_s = 0 \leftrightarrow +1$ transition (`rwa_branch="ms_plus"`).
- In ESR driving, the nuclear drive term is neglected primarily because nuclear transitions are strongly off-resonant (large detuning from the microwave drive). The small $|\gamma_n|$ only further suppresses this term.

For RF nuclear-spin control, the analogous rotating-frame form is

$$H_{\text{RWA}}^{(n)} = H_0 \pm \omega_{\rm rf} (\mathbb{1}_3 \otimes I_z) + \frac{-\gamma_n B_1}{2} (\mathbb{1}_3 \otimes I_\perp)$$

- The $-$ sign targets the $m_I=+1 \leftrightarrow 0$ branch (e.g. $|4\rangle \leftrightarrow |5\rangle$).
- The $+$ sign targets the $m_I=0 \leftrightarrow -1$ branch (e.g. $|5\rangle \leftrightarrow |6\rangle$).
- The electron transverse drive is then strongly off-resonant and can be neglected in the same spirit.

`rwa.py` supports both ESR and nuclear branches:
- `rwa_spin="electron"` with `rwa_branch in {"ms_minus", "ms_plus"}`
- `rwa_spin="nuclear"` with `rwa_branch in {"mi_minus", "mi_plus"}`

In near-crossing regimes (e.g., special high-$B_0$ points where electron and nuclear transition frequencies become close), high-level experiment APIs can auto-fallback from `method="rwa"` to `method="ode"` (`rwa_auto_fallback=True`, default).

Time evolution under a time-independent Hamiltonian is computed analytically:

$$|\psi(t)\rangle = e^{-i H_{\text{RWA}} t} |\psi(0)\rangle$$

using `scipy.linalg.expm`, which supports vectorized batch evaluation over frequency sweeps (ODMR) or time arrays (Rabi).

### Experiment protocols

- **ODMR**: Apply a fixed-duration pulse at each frequency in a sweep; read out population.
- **Rabi**: Apply a fixed-frequency drive for varying durations; read out population vs. time.
- **Ramsey T2\***: Sequence of $\pi/2$ -- free precession ($\tau$) -- $\pi/2$; read out population vs. $\tau$. The $\pi/2$ pulse duration is estimated from $\Omega_R = |\gamma_e| B_1 / \sqrt{2}$.

---

## API Reference

### Experiment functions

#### `simulate_odmr`

```python
simulate_odmr(
    initial_state: np.ndarray,             # (9,) complex state vector
    f_start_hz: float,                     # sweep start frequency (Hz)
    f_stop_hz: float,                      # sweep stop frequency (Hz)
    pulse_strength_gauss: float,           # B1 amplitude (Gauss)
    pulse_duration_s: float,               # pulse duration (seconds)
    b0_gauss: float,                       # static magnetic field (Gauss)
    n_points: int = 201,                   # number of frequency points
    readout_ms: int = 0,                   # ms subspace to measure
    readout_state_index: int | None = None,  # 1-based index (overrides readout_ms)
    drive_axis: str = "x",                 # drive polarization axis
    rwa_branch: str = "ms_minus",          # electron: ms_minus/ms_plus; nuclear: mi_minus/mi_plus
    rwa_spin: str = "electron",            # "electron" or "nuclear"
    rwa_auto_fallback: bool = True,        # auto-switch to ODE near cross-spin resonances
    params: NVParams | None = None,        # custom NV parameters
    method: str = "rwa",                   # "rwa" or "ode"
) -> Dict[str, np.ndarray]
```

**Returns:** `{"frequencies_hz": ndarray (N,), "signal": ndarray (N,)}`

#### `simulate_rabi`

```python
simulate_rabi(
    initial_state: np.ndarray,
    pulse_frequency_hz: float,             # fixed drive frequency (Hz)
    t_start_s: float,                      # start time (seconds)
    t_stop_s: float,                       # stop time (seconds)
    pulse_strength_gauss: float,
    b0_gauss: float,
    n_points: int = 401,
    readout_ms: int = 0,
    readout_state_index: int | None = None,
    drive_axis: str = "x",
    rwa_branch: str = "ms_minus",
    rwa_spin: str = "electron",
    rwa_auto_fallback: bool = True,
    params: NVParams | None = None,
    method: str = "rwa",
) -> Dict[str, np.ndarray]
```

**Returns:** `{"times_s": ndarray (N,), "signal": ndarray (N,), "states": ndarray (N, 9)}`

#### `simulate_t2star`

```python
simulate_t2star(
    initial_state: np.ndarray,
    detuning_hz: float,                    # detuning from pulse_frequency_hz (Hz)
    pulse_strength_gauss: float,
    pulse_frequency_hz: float,
    t_start_s: float,
    t_stop_s: float,
    b0_gauss: float,
    n_points: int = 201,
    readout_ms: int = 0,
    readout_state_index: int | None = None,
    drive_axis: str = "x",
    rwa_branch: str = "ms_minus",
    rwa_spin: str = "electron",
    rwa_auto_fallback: bool = True,
    params: NVParams | None = None,
    method: str = "rwa",
) -> Dict[str, np.ndarray]
```

**Returns:** `{"taus_s": ndarray (N,), "signal": ndarray (N,), "estimated_pi2_s": ndarray (1,)}`

### Solvers

#### `propagate_expm`

Batch matrix-exponential propagator for time-independent Hamiltonians. Three modes:

| Hamiltonian shape | Time `t` | Output shape | Use case |
|---|---|---|---|
| `(d, d)` | scalar | `(d,)` | single time point |
| `(N, d, d)` | scalar | `(N, d)` | ODMR frequency sweep |
| `(d, d)` | `(N,)` array | `(N, d)` | Rabi time sweep |

```python
propagate_expm(
    psi0: np.ndarray,          # (d,) initial state
    hamiltonian: np.ndarray,   # (d,d) or (N,d,d) time-independent Hamiltonian(s)
    t: float | np.ndarray,    # scalar or (N,) evolution time(s)
) -> np.ndarray
```

#### `propagate_state`

ODE propagator for arbitrary (possibly time-dependent) Hamiltonians.

```python
propagate_state(
    psi0: np.ndarray,
    hamiltonian: np.ndarray | Callable[[float], np.ndarray],
    t_span: Tuple[float, float],
    t_eval: Optional[np.ndarray] = None,
    method: str = "DOP853",
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]   # (times, states) where states is (N, d)
```

### RWA construction functions

```python
rotating_frame_h0(h0, f_drive_hz, branch="ms_minus", rwa_spin="electron") -> np.ndarray   # (9, 9)
rwa_drive_matrix(params, b1_gauss, axis="x", rwa_spin="electron") -> np.ndarray            # (9, 9)
rwa_hamiltonian(
    params,
    b0_gauss,
    f_drive_hz,
    b1_gauss,
    axis="x",
    branch="ms_minus",
    rwa_spin="electron",
) -> np.ndarray  # (9, 9)
```

- `rotating_frame_h0`: Transforms the static Hamiltonian into electron or nuclear rotating frame.
- `rwa_drive_matrix`: Builds the time-independent RWA drive term for the selected spin channel.
- `rwa_hamiltonian`: Convenience function returning the complete RWA Hamiltonian.

### State constructors

```python
ket_from_index(index_1based: int) -> np.ndarray           # 1-based: |1> through |9>
ket_from_quantum_numbers(ms: int, mI: int) -> np.ndarray  # e.g. ms=0, mI=0 -> |5>
basis_index(ms: int, mI: int) -> int                       # returns 0-based index
```

### Constants

```python
from NV_Simulator import NVParams

params = NVParams()       # frozen dataclass with default NV center parameters
params.D                  # 2*pi*2.87e9    (rad/s)
params.Q                  # -2*pi*4.945e6  (rad/s)
params.A                  # -2*pi*2.162e6  (rad/s)
params.gamma_e            # 2*pi*(-2.8029e6) rad/(s*G)
params.gamma_n            # 2*pi*(0.3077e3)  rad/(s*G)
```

---

## Project Structure

```
NV_Simulator/
    __init__.py       Public API exports
    constants.py      NVParams dataclass (D, Q, A, gamma_e, gamma_n)
    operators.py      Spin-1 matrices, basis states, ket constructors
    hamiltonian.py    Static H0, time-dependent drive H1(t), total H(t)
    observables.py    Projectors, expectation values, ms population
    rwa.py            RWA rotating-frame Hamiltonian construction
    solver.py         ODE propagator (propagate_state) + expm propagator (propagate_expm)
    experiments.py    simulate_odmr, simulate_rabi, simulate_t2star
examples/
    basic_demo.py     End-to-end demo of all three experiments
pyproject.toml        Build configuration
README.md             This file
```

---

## License

This project does not yet specify a license. Add a `LICENSE` file to the repository root to declare one.
