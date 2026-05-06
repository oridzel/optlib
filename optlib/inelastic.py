import numpy as np
import math
from optlib.constants import h2ev, a0, c
from optlib.dielectrics import DielectricFunction
from optlib.utils import InputError

class InelasticEngine:
    """
    Computes standard inelastic scattering cross-sections (DIIMFP, IMFP)
    using dispersive dielectric models (Drude, Mermin, MLL, etc.).
    """
    def __init__(self, material):
        self.mat = material

    def calculate_diimfp(self, E_eV, de=0.5, nq=100, normalised=True):
        """
        Calculates the Differential Inelastic Mean Free Path (DIIMFP).
        
        Parameters:
        - E_eV: Primary electron energy in eV.
        - de: Energy loss step size.
        - nq: Number of q-points for integration (if dispersive).
        - normalised: Whether to normalize the DIIMFP curve to area = 1.
        """
        old_eloss = self.mat.eloss
        old_q = self.mat.q

        # 1. Setup the energy loss grid based on material type
        if self.mat.is_metal:
            start_e, stop_e = 1e-5, E_eV - self.mat.e_fermi
        else:
            if E_eV < 2 * self.mat.e_gap + self.mat.width_of_the_valence_band:
                raise InputError("Energy must be greater than 2*E_gap + valence band width")
            start_e = self.mat.e_gap
            stop_e = (E_eV - self.mat.e_gap) - self.mat.width_of_the_valence_band

        num_steps = int((stop_e - start_e) / de + 1)
        eloss = np.linspace(start_e, stop_e, num_steps)
        self.mat.eloss = eloss  # Temporarily override material grid for evaluation

        # 2. Relativistic kinematics in atomic units
        e0_au = E_eV / h2ev
        eloss_au = eloss / h2ev
        rel_coef = ((1 + e0_au / c**2)**2) / (1 + e0_au / (2 * c**2))

        # Projectile momentum limits
        qm_val = np.sqrt(e0_au * (2 + e0_au / c**2)) - np.sqrt((e0_au - eloss_au) * (2 + (e0_au - eloss_au) / c**2))
        qp_val = np.sqrt(e0_au * (2 + e0_au / c**2)) + np.sqrt((e0_au - eloss_au) * (2 + (e0_au - eloss_au) / c**2))

        # 3. Choose integration method (Optical Limit vs. Dispersive)
        is_optical = (self.mat.oscillators.alpha == 0 and 
                      self.mat.oscillators.model not in ['Mermin', 'MLL'] and 
                      getattr(self.mat, 'q_dependency', None) is None)

        if is_optical:
            # FAST PATH: No q-dispersion, use optical ELF and integrate analytically
            self.mat.extend_to_henke()
            int_limits = np.log(qp_val / qm_val)
            int_limits[np.isinf(int_limits)] = 1e-5
            
            interp_elf = np.interp(eloss, self.mat.eloss_extended_to_henke, self.mat.elf_extended_to_henke)
            interp_elf[np.isnan(interp_elf)] = 1e-5
            
            iimfp = rel_coef * (1.0 / (math.pi * e0_au)) * interp_elf * int_limits
            diimfp = iimfp / (h2ev * a0)
            
        else:
            # DISPERSIVE PATH: Evaluate ELF over a q-grid and integrate numerically
            qm_log = np.log(qm_val)
            qp_log = np.log(qp_val)
            q_log = np.linspace(qm_log, qp_log, nq, axis=1) # Shape: (len(eloss), nq)
            
            # Temporarily set q grid for the DielectricFunction to read
            self.mat.q = np.exp(q_log) / a0
            if self.mat.oscillators.model in ['Mermin', 'MLL']:
                self.mat.q[self.mat.q == 0] = 0.01

            # Request epsilon map from our dedicated engine
            df = DielectricFunction(self.mat)
            epsilon = df.calculate()
            elf = (-1 / epsilon).imag
            elf[np.isnan(elf)] = 1e-5

            if self.mat.oscillators.model in ['Mermin', 'MLL']:
                elf[self.mat.q == 0.01] = 1e-5

            # Integrate over d(log q) = dq/q
            iimfp = rel_coef * (1.0 / (math.pi * e0_au)) * np.trapezoid(elf, q_log, axis=1)
            diimfp = iimfp / (h2ev * a0)

        # 4. Cleanup and format output
        diimfp[np.isnan(diimfp)] = 1e-5
        iimfp[np.isnan(iimfp)] = 1e-5

        if normalised:
            area = np.trapezoid(diimfp, eloss)
            if area > 0:
                diimfp = diimfp / area

        # Save to material and restore old grids
        self.mat.diimfp = diimfp
        self.mat.iimfp = iimfp
        self.mat.diimfp_e = eloss
        self.mat.e0 = E_eV
        
        self.mat.eloss = old_eloss
        self.mat.q = old_q

        return diimfp

    def calculate_imfp(self, energy_array, de=0.5, nq=100):
        """
        Calculates the Total Inelastic Mean Free Path (IMFP) over an array of primary energies.
        """
        if self.mat.is_metal and self.mat.e_fermi == 0:
            raise InputError("Please specify the value of the Fermi energy (e_fermi)")
        elif not self.mat.is_metal and self.mat.e_gap == 0 and self.mat.width_of_the_valence_band == 0:
            raise InputError("Please specify e_gap and width_of_the_valence_band")

        imfp = np.zeros_like(energy_array, dtype=float)
        
        for i, E in enumerate(energy_array):
            # We don't normalize here because we need the raw area for the IMFP
            self.calculate_diimfp(E, de, nq, normalised=False)
            
            # The IMFP is the inverse of the integrated DIIMFP curve
            integrated_diimfp = np.trapezoid(self.mat.iimfp, self.mat.diimfp_e / h2ev)
            if integrated_diimfp > 0:
                imfp[i] = 1.0 / integrated_diimfp
            else:
                imfp[i] = 0.0
                
        # Convert to Angstroms
        self.mat.imfp = imfp * a0
        self.mat.imfp_e = energy_array
        
        return self.mat.imfp