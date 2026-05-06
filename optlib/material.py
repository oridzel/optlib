import numpy as np
import math
import pandas as pd

from optlib.utils import InputError, check_list_type, is_list_of_int_float
from optlib.constants import h2ev, a0, hc, r0
from optlib.dielectrics import Oscillators, DielectricFunction


class Composition:
    """Stores the elemental composition and atomic numbers of the material."""
    def __init__(self, elements, indices, atomic_numbers):
        if not check_list_type(elements, str):
            raise InputError("The array of elements passed must be of the type string!")
        if not is_list_of_int_float(indices):
            raise InputError("The array of indices passed must be of the type int or float!")
        if isinstance(elements, list) and isinstance(indices, list) and len(elements) != len(indices):
            raise InputError("The number of elements and indices must be the same!")
        
        self.elements = elements
        self.indices = indices
        self.atomic_numbers = atomic_numbers


class Material:
    """
    Central data container for material properties, optical data, and cross-sections.
    """
    def __init__(self, name, oscillators=None, composition=None, eloss=None, q=None, xraypath=''):
        
        self.name = name
        self.oscillators = oscillators
        self.composition = composition
        self.xraypath = xraypath
        
        # Energy loss and momentum grids
        if eloss is not None:
            self.eloss = np.array(eloss, dtype=float)
            self.eloss[self.eloss == 0] = 1e-5
        else:
            self.eloss = None
            
        self._q = None
        self.size_q = 1
        if q is not None:
            self.q = q  # uses the property setter
        
        # Fundamental electronic properties
        self.e_gap = 0.0
        self.e_fermi = 0.0
        self.u = 0.0
        self.width_of_the_valence_band = None
        self.atomic_density = None
        self.electron_density = None
        self.static_refractive_index = None
        self.Z = None
        
        # Flags
        self.is_metal = True
        self.zero_gap = True
        self.use_kk_relation = False
        self.use_kk_constraint = False
        self.use_henke_for_ne = False
        
        # Optical & Dielectric Data
        self._epsilon = None
        self.elf = None
        self.optical_omega = None  
        self.optical_elf = None    
        
        # Henke Extended Data
        self.henke_limit = 100
        self.electron_density_henke = 0
        self.elf_extended_to_henke = None
        self.eloss_henke = None
        self.elf_henke = None
        self.eloss_extended_to_henke = None
        
        # Cross-section Data
        self.diimfp = None
        self.diimfp_e = None
        self.iimfp = None
        self.imfp = None
        self.imfp_e = None
        
        self.sigma_el = None
        self.emfp = None
        self.decs = None
        self.decs_mu = None
        self.decs_theta = None
        self.norm_decs = None
        self.e0 = None
        
        self.q_dependency = None

    @property
    def q(self):
        if isinstance(self._q, np.ndarray):
            return self._q
        elif isinstance(self._q, list):
            return np.array(self._q)
        else:
            return self._q
    
    @q.setter
    def q(self, q_val):
        try: 
            self.size_q = q_val.shape[1]
        except (IndexError, AttributeError):
            try:
                self.size_q = q_val.shape[0]
            except (IndexError, AttributeError):
                self.size_q = 1
        finally:
            self._q = q_val

    @property
    def epsilon(self):
        """Lazy loads the complex dielectric function from the oscillators."""
        if self._epsilon is None:
            if self.oscillators is None:
                raise ValueError("Cannot calculate epsilon: No oscillators provided to the Material.")
            df = DielectricFunction(self)
            self._epsilon = df.calculate()
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value

    # ----------------------------------------------------------------------
    # UNIT CONVERSIONS 
    # ----------------------------------------------------------------------
    def _convert2au(self):
        if self.oscillators and self.oscillators.model == 'Drude':
            self.oscillators.A = self.oscillators.A / h2ev / h2ev
        if self.oscillators:
            self.oscillators.gamma = self.oscillators.gamma / h2ev
            self.oscillators.omega = self.oscillators.omega / h2ev
            
        self.e_fermi = self.e_fermi / h2ev
        if self.eloss is not None:
            self.eloss = self.eloss / h2ev
        if self.q is not None:
            self.q = self.q * a0
        
        if self.e_gap: self.e_gap = self.e_gap / h2ev
        if self.u: self.u = self.u / h2ev
        if self.width_of_the_valence_band: self.width_of_the_valence_band = self.width_of_the_valence_band / h2ev

    def _convert2ru(self):
        if self.oscillators and self.oscillators.model == 'Drude':
            self.oscillators.A = self.oscillators.A * h2ev * h2ev
        if self.oscillators:
            self.oscillators.gamma = self.oscillators.gamma * h2ev
            self.oscillators.omega = self.oscillators.omega * h2ev
            
        self.e_fermi = self.e_fermi * h2ev
        if self.eloss is not None:
            self.eloss = self.eloss * h2ev
        if self.q is not None:
            self.q = self.q / a0
        
        if self.e_gap: self.e_gap = self.e_gap * h2ev
        if self.u: self.u = self.u * h2ev
        if self.width_of_the_valence_band: self.width_of_the_valence_band = self.width_of_the_valence_band * h2ev

    # ----------------------------------------------------------------------
    # HIGH-ENERGY EXTENSION (Henke Data)
    # ----------------------------------------------------------------------
    def extend_to_henke(self):
        if self.eloss is None:
            raise ValueError("Cannot extend to Henke: Material.eloss is not defined.")
            
        if self.elf is None:
            self.elf = (-1 / self.epsilon).imag
            self.elf[np.isnan(self.elf)] = 1e-5
            
        if self.eloss_henke is None and self.elf_henke is None:
            self.eloss_henke, self.elf_henke = self.mopt()
            
        ind = self.eloss < self.henke_limit
        self.eloss_extended_to_henke = np.concatenate((self.eloss[ind], self.eloss_henke))
        self.elf_extended_to_henke = np.concatenate((self.elf[ind], self.elf_henke))

    def mopt(self):
        if self.atomic_density is None or self.composition is None:
            raise InputError("Please specify atomic_density and composition to calculate Henke extensions.")
            
        numberOfElements = len(self.composition.elements)
        energy = np.linspace(self.henke_limit, 30000, int((30000 - self.henke_limit) / 1.0 + 1))
        
        f1sum = np.zeros_like(energy)
        f2sum = np.zeros_like(energy)

        for i in range(numberOfElements):
            datahenke = self.readhenke(self.xraypath + self.composition.elements[i])
            f1 = np.interp(energy, datahenke[:, 0], datahenke[:, 1])
            f2 = np.interp(energy, datahenke[:, 0], datahenke[:, 2])
            f1sum += f1 * self.composition.indices[i]
            f2sum += f2 * self.composition.indices[i]

        lambda_ = hc / (energy / 1000)
        
        if not self.is_metal:
            f1sum /= np.sum(self.composition.indices)
            f2sum /= np.sum(self.composition.indices)

        n = 1 - self.atomic_density * r0 * 1e10 * lambda_**2 * f1sum / (2 * math.pi)
        k = -self.atomic_density * r0 * 1e10 * lambda_**2 * f2sum / (2 * math.pi)

        eps1 = n**2 - k**2
        eps2 = 2 * n * k

        return energy, -eps2 / (eps1**2 + eps2**2)

    def readhenke(self, filename):
        return np.loadtxt(filename + '.nff', skiprows=1)

    # ----------------------------------------------------------------------
    # UTILITIES
    # ----------------------------------------------------------------------
    def write_optical_data(self):
        if self.elf is None:
            self.elf = (-1 / self.epsilon).imag
            
        n_complex = np.sqrt(self.epsilon)
        refractive_index = n_complex.real
        extinction_coefficient = n_complex.imag
        
        d = dict(
            E=np.round(self.eloss, 2),
            n=np.round(refractive_index, 2),
            k=np.round(extinction_coefficient, 2),
            eps1=np.round(self.epsilon.real, 2),
            eps2=np.round(self.epsilon.imag, 2),
            elf=np.round(self.elf, 2)
        )
        df = pd.DataFrame.from_dict(d, orient='index').transpose().fillna('')
        
        model_name = self.oscillators.model if self.oscillators else "empirical"
        filename = f'{self.name}_{model_name}_table_optical_data.csv'
        df.to_csv(filename, index=False)