# optlib

**A Python computational physics library for Monte Carlo simulations of electron-solid interactions, dielectric energy loss functions, and secondary electron yields.**

`optlib` is a comprehensive computational materials physics toolkit designed to simulate secondary electron emission (SEE) and electron transport in solids. By bridging the gap between empirical optical properties and complex Monte Carlo trajectory simulations, `optlib` allows researchers to accurately model Inelastic Mean Free Paths (IMFP), differential scattering cross-sections, and macroscopic electron yields (TEY, SEY, BSE) using a highly modular pipeline.

## 🚀 Key Features

* **Optical Data Processing (`dielectrics.py`):** Calculate the complex dielectric function $\epsilon(q, \omega)$ and Energy Loss Function (ELF) using Drude, Drude-Lindhard, Mermin, and MLL oscillator models.
* **Full Penn Algorithm (`fpa.py`):** High-performance, parallelized (`joblib`) engine to extend the optical ELF to finite momentum transfer. Separates single-electron and volume plasmon excitation channels.
* **Inelastic Scattering (`inelastic.py`):** Computes Differential Inelastic Mean Free Paths (DIIMFP) and Total IMFP over standard or relativistic kinematic bounds.
* **Elastic Scattering (`elastic.py`):** A clean Python wrapper for the Fortran `ELSEPA` code to calculate elastic mean free paths and angular deflection distributions (DECS).
* **Monte Carlo Engine (`seemc.py`):** Highly optimized, multi-threaded 3D trajectory simulator. Implements quantum step-barrier transmission, specular Total Internal Reflection, and phase-space secondary generation.

## ⚙️ Installation

Make sure you have Python 3.8+ [installed properly](https://docs.python-guide.org/).

**Option 1: Install directly from GitHub**
The easiest way to install the library and all its dependencies (like `numpy`, `scipy`, `nlopt`, etc.):
```bash
pip install git+[https://github.com/oridzel/optlib.git](https://github.com/oridzel/optlib.git)
```

**Option 2: Clone and install locally (Recommended for development)**
If you want to edit the code or run the examples:
```bash
git clone [https://github.com/oridzel/optlib.git](https://github.com/oridzel/optlib.git)
cd optlib
pip install -e .
```
*(Note: The `-e` flag installs the package in "editable" mode, meaning any changes you make to the source code will immediately apply without needing to reinstall).*

## 📖 Quick Start: Calculating an Energy Loss Function

Thanks to `optlib`'s modular design, you can define materials and test optical models with just a few lines of code:

```python
import numpy as np
import matplotlib.pyplot as plt
from optlib.material import Material
from optlib.dielectrics import Oscillators

# 1. Define your energy loss grid (eV) and momentum (a0^-1)
energy_grid = np.linspace(1e-5, 100, 1000) 
q_grid = np.array([0.01]) 

# 2. Define the Mermin Oscillators
mermin_osc = Oscillators(
    model='Mermin',
    A=[0.05, 0.12],       # Oscillator strengths
    gamma=[2.5, 6.0],     # Dampings (eV)
    omega=[18.0, 26.0]    # Resonance energies (eV)
)

# 3. Create the Material data container
cu = Material(
    name="Cu_Mermin",
    oscillators=mermin_osc,
    eloss=energy_grid,
    q=q_grid
)

# 4. Calculate the Dielectric Function and ELF
elf = (-1.0 / cu.epsilon).imag

# Plot the result
plt.plot(cu.eloss, np.squeeze(elf))
plt.xlabel("Energy Loss (eV)")
plt.ylabel("ELF")
plt.title(f"Energy Loss Function for {cu.name}")
plt.show()
```

## 📁 Repository Structure

* `optlib/material.py`: Central data container (`Material` and `Composition`).
* `optlib/dielectrics.py`: The optical math engine (`DielectricFunction` and `Oscillators`).
* `optlib/fpa.py`: The heavy-duty physics engine for Full Penn Algorithm maps.
* `optlib/inelastic.py`: Standard DIIMFP/IMFP integrator.
* `optlib/elastic.py`: The Fortran interface for `ELSEPA`.
* `optlib/optimfit.py`: NLopt-based fitting routines for retrieving optical constants from experimental data.
* `optlib/seemc.py`: The Monte Carlo trajectory simulator.
* `optlib/utils.py` & `constants.py`: Shared math helpers, unit conversions, and physical constants.

## 📄 License
[Insert License Here - e.g., MIT License]
