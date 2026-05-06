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
    Generates input files, runs the subprocess, and parses the output .dat files
    to compute elastic mean free paths (EMFP) and angular distributions (DECS).
    """
    def __init__(self, material, elsepa_path='elsepa'):
        """
        Parameters:
        - material: optlib Material instance.
        - elsepa_path: Path to the ELSEPA executable. Defaults to 'elsepa' (assuming it's in PATH).
        """
        self.mat = material
        self.elsepa_path = elsepa_path

    def calculate_elastic_properties(self, e0, mnucl=3, melec=4, mexch=1, cleanup=True):
        """
        Runs ELSEPA to calculate elastic cross sections and angular distributions.
        
        Parameters:
        - e0: Primary electron energy in eV.
        - mnucl: Nuclear charge distribution model.
        - melec: Electron density model.
        - mexch: Exchange potential model.
        - cleanup: If True, deletes the generated lub.in and .dat files after parsing.
        """
        if self.mat.atomic_density is None:
            raise ValueError("Material atomic_density must be set before running ELSEPA.")

        self.mat.e0 = e0
        sumweights = 0.0
        
        # Accumulators for the final weighted distributions
        decs_total = None
        decs_a_total = None
        decs_theta = None
        decs_mu = None
        
        # We need a weighted average of the elastic cross-section (sigma_el)
        sigma_el_weighted_sum = 0.0

        for i in range(len(self.mat.composition.elements)):
            atomic_num = self.mat.composition.atomic_numbers[i]
            weight = self.mat.composition.indices[i]
            sumweights += weight

            # Muffin-tin model toggle
            rmuf = 0 if atomic_num in gas_z else 1

            # 1. Write the input file for ELSEPA
            input_filename = 'lub.in'
            with open(input_filename, 'w+') as fd:
                fd.write(f'IZ      {atomic_num}         atomic number                               [none]\n')
                fd.write(f'MNUCL   {mnucl}         rho_n (1=P, 2=U, 3=F, 4=Uu)                 [  3]\n')
                fd.write(f'NELEC   {atomic_num}         number of bound electrons                   [ IZ]\n')
                fd.write(f'MELEC   {melec}         rho_e (1=TFM, 2=TFD, 3=DHFS, 4=DF, 5=file)  [  4]\n')
                fd.write(f'MUFFIN  {rmuf}          0=free atom, 1=muffin-tin model             [  0]\n')
                fd.write('IELEC  -1          -1=electron, +1=positron                    [ -1]\n')
                fd.write(f'MEXCH   {mexch}          V_ex (0=none, 1=FM, 2=TF, 3=RT)             [  1]\n')
                fd.write(f'EV      {round(e0)}      kinetic energy (eV)                         [none]\n')

            # 2. Run the ELSEPA Fortran executable
            try:
                # Using run with shell=True and input redirection
                cmd = f"{self.elsepa_path} < {input_filename}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                raise ElsepaError(f"ELSEPA failed to run. Ensure '{self.elsepa_path}' is accessible. Error: {e.stderr}")

            # 3. Format expected output filename
            # e.g., for 1000 eV -> 1.000E+03 -> 1p000e03 (depending on ELSEPA's exact formatting)
            energy_str = '{:1.3e}'.format(round(e0)).replace('.', 'p').replace('+0', '0')
            output_dat = f"dcs_{energy_str}.dat"

            if not os.path.exists(output_dat):
                raise ElsepaError(f"ELSEPA output file '{output_dat}' not found. Subprocess may have failed silently.")

            # 4. Parse the output for Total Elastic Cross Section
            sigma_el = 0.0
            with open(output_dat, 'r') as fd:
                lines = fd.readlines()
                for line in lines:
                    if "Total elastic cross section =" in line:
                        # Extract the scientific notation float
                        match = re.findall(r"\d+\.\d+[E][+-]\d+", line)
                        if match:
                            sigma_el = float(match[0])
                            break
            
            sigma_el_weighted_sum += sigma_el * weight

            # 5. Parse the Differential Elastic Cross Section (DECS) array
            # Assuming comments start with '#' and columns are: Theta, Mu, DECS, DECS_A...
            data = np.loadtxt(output_dat, comments="#")
            
            if i == 0:
                # Initialize arrays on the first pass
                decs_total = np.zeros_like(data[:, 0])
                decs_a_total = np.zeros_like(data[:, 0])
                decs_theta = np.deg2rad(data[:, 0])
                decs_mu = data[:, 1]
            
            decs_total += data[:, 2] * weight
            decs_a_total += data[:, 3] * weight

            # 6. Cleanup local files
            if cleanup:
                os.remove(input_filename)
                os.remove(output_dat)

        # 7. Finalize and store weighted averages in the Material object
        self.mat.sigma_el = sigma_el_weighted_sum / sumweights
        
        # Calculate Elastic Mean Free Path (EMFP) in Angstroms
        # sigma_el is expected to be in cm^2 or a0^2 depending on ELSEPA output. 
        # Your original code used: 1 / (sigma_el * a0**2 * atomic_density)
        self.mat.emfp = 1.0 / (self.mat.sigma_el * a0**2 * self.mat.atomic_density)
        
        self.mat.decs_theta = decs_theta
        self.mat.decs_mu = decs_mu
        self.mat.decs_a = decs_a_total / sumweights
        self.mat.decs = decs_total / sumweights
        
        # Normalize DECS over theta
        self.mat.norm_decs = self.mat.decs / np.trapz(self.mat.decs, self.mat.decs_theta)

        return self.mat.emfp, self.mat.decs