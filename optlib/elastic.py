import os
import subprocess
import re
import numpy as np
from optlib.constants import a0, gas_z
from optlib.utils import Error

class ElsepaError(Error):
    """Exception raised for errors during the ELSEPA subprocess execution."""
    pass


class ElsepaWrapper:
    """
    Wrapper for the ELSEPA (Electron Elastic Scattering cross-sections) Fortran executable.
    Generates batch input files, runs the subprocess once, and parses the output .dat files
    to compute elastic and transport mean free paths.
    """
    def __init__(self, material, elsepa_path='elsepa'):
        self.mat = material
        self.elsepa_path = elsepa_path

    @staticmethod
    def rmuf_from_number_density(n_atoms_cm3: float) -> float:
        """Calculates the Wigner-Seitz radius (Muffin-tin radius) in cm."""
        return (3.0 / (4.0 * np.pi * n_atoms_cm3))**(1.0 / 3.0)

    @staticmethod
    def parse_sigma(lines, key):
        """Extracts the cross-section in a0^2 from ELSEPA output."""
        line = next(l for l in lines if key in l)
        nums = re.findall(r"[-+]?\d+\.\d+E[+-]\d+", line)
        # ELSEPA prints cm^2 then a0^2; the LAST one is a0^2
        return float(nums[-1])

    def calculate_elastic_properties(self, energy_array, mnucl=3, melec=4, mexch=1, minE=5.0, cleanup=True):
        """
        Batch processes elastic properties over an array of energies.
        Energies below `minE` are clamped to `minE` (default 5 eV) since ELSEPA 
        cannot reliably compute cross-sections at extremely low energies.
        """
        if self.mat.atomic_density is None:
            raise ValueError("Material atomic_density must be set before running ELSEPA.")

        # 1. Clamp energies and extract UNIQUE values to save Fortran compute time
        energy_array = np.asarray(energy_array)
        used_energy = np.where(energy_array >= minE, energy_array, minE)
        unique_energies = np.unique(used_energy)
        
        n_energies = len(energy_array)
        sumweights = 0.0

        # Accumulators for the final grids (mapped back to the full energy array)
        sigma_el_total = np.zeros(n_energies, dtype=float)
        sigma_tr_total = np.zeros(n_energies, dtype=float)
        decs_total = None
        decs_theta = None

        n_atoms_cm3 = self.mat.atomic_density * 1e24

        for i, element in enumerate(self.mat.composition.elements):
            atomic_num = self.mat.composition.atomic_numbers[i]
            weight = self.mat.composition.indices[i]
            sumweights += weight

            is_gas = atomic_num in gas_z
            rmuf_flag = 0 if is_gas else 1
            
            # 2. Write the batch input file using ONLY UNIQUE energies
            input_filename = 'lub.in'
            with open(input_filename, 'w+') as fd:
                fd.write(f'IZ      {atomic_num}         atomic number                               [none]\n')
                fd.write(f'MNUCL   {mnucl}         rho_n (1=P, 2=U, 3=F, 4=Uu)                 [  3]\n')
                fd.write(f'NELEC   {atomic_num}         number of bound electrons                   [ IZ]\n')
                fd.write(f'MELEC   {melec}         rho_e (1=TFM, 2=TFD, 3=DHFS, 4=DF, 5=file)  [  4]\n')
                fd.write(f'MUFFIN  {rmuf_flag}          0=free atom, 1=muffin-tin model             [  0]\n')
                
                if not is_gas:
                    rmuf_val = self.rmuf_from_number_density(n_atoms_cm3)
                    fd.write(f'RMUF    {rmuf_val:.6e}  muffin-tin radius (cm)\n')
                    
                fd.write('IELEC  -1          -1=electron, +1=positron                    [ -1]\n')
                fd.write(f'MEXCH   {mexch}          V_ex (0=none, 1=FM, 2=TF, 3=RT)             [  1]\n')
                
                # Write the unique, clamped energies to ELSEPA
                for e in unique_energies:  
                    fd.write(f'EV      {e}      kinetic energy (eV)                         [none]\n')

            # 3. Run ELSEPA once for this element
            try:
                cmd = f"{self.elsepa_path} < {input_filename}"
                subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                raise ElsepaError(f"ELSEPA failed to run. Error: {e.stderr}")

            # 4. Parse the unique output files
            unique_sig_el = {}
            unique_sig_tr = {}
            unique_decs = {}
            
            for ene in unique_energies:
                # Safely format energy string
                ene_str = '{:1.3e}'.format(int(np.round(ene))).replace('.', 'p').replace('+0', '0')
                output_dat = f"dcs_{ene_str}.dat"

                if not os.path.exists(output_dat):
                    raise ElsepaError(f"ELSEPA output '{output_dat}' not found.")

                with open(output_dat, 'r') as fd:
                    lines = fd.readlines()

                unique_sig_el[ene] = self.parse_sigma(lines, "Total elastic cross section =")
                unique_sig_tr[ene] = self.parse_sigma(lines, "1st transport cross section =")

                data = np.loadtxt(output_dat, comments="#")
                th = np.deg2rad(data[:, 0])
                de = data[:, 3]  # DECS_A column (a0^2/sr)

                # Capture theta grid on the first file read
                if decs_theta is None:
                    decs_theta = th
                    
                # Enforce theta consistency if ELSEPA changes grid size
                if th.shape != decs_theta.shape or np.max(np.abs(th - decs_theta)) > 1e-12:
                    de = np.interp(decs_theta, th, de)

                unique_decs[ene] = de

                if cleanup:
                    os.remove(output_dat)

            if cleanup:
                os.remove(input_filename)

            # 5. Map the unique computations back to the full energy array
            if decs_total is None:
                decs_total = np.zeros((decs_theta.size, n_energies), dtype=float)
            
            element_decs = np.zeros((decs_theta.size, n_energies), dtype=float)

            for j, ene in enumerate(used_energy):
                sigma_el_total[j] += unique_sig_el[ene] * weight
                sigma_tr_total[j] += unique_sig_tr[ene] * weight
                element_decs[:, j] = unique_decs[ene]

            decs_total += element_decs * weight

        # 6. Finalize weighted properties in the Material object
        self.mat.sigma_el = sigma_el_total / sumweights
        self.mat.sigma_tr = sigma_tr_total / sumweights

        self.mat.emfp = 1.0 / (self.mat.sigma_el * a0**2 * self.mat.atomic_density)
        self.mat.trmfp = 1.0 / (self.mat.sigma_tr * a0**2 * self.mat.atomic_density)

        self.mat.decs_theta = decs_theta
        self.mat.decs = decs_total / sumweights

        # Normalize the DECS columns over theta
        norm_factor = np.trapz(self.mat.decs, self.mat.decs_theta, axis=0)
        self.mat.norm_decs = self.mat.decs / np.where(norm_factor > 0, norm_factor, 1.0)

        # Update energy grid tracking
        self.mat.e0_array = used_energy
        self.mat.original_e0_array = energy_array

        return self.mat.emfp, self.mat.decs
