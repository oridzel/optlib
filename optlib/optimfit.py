import numpy as np
import math
import copy
from tqdm import tqdm
import nlopt

from optlib.utils import InputError
from optlib.constants import h2ev, a0, wpc
from optlib.dielectrics import DielectricFunction
from optlib.inelastic import InelasticEngine

class exp_data:
    """Simple container for experimental targets."""
    def __init__(self):
        self.x_elf = []
        self.y_elf = []
        self.x_ndiimfp = []
        self.y_ndiimfp = []


class OptFit:
    def __init__(self, material, exp_data, e0, de=0.5, n_q=100, fit_alpha=False):
        if e0 == 0:
            raise InputError("e0 must be non-zero")
            
        self.material = material
        self.exp_data = exp_data
        self.e0 = e0
        self.de = de
        self.n_q = n_q
        self.count = 0
        self.fit_alpha = fit_alpha
        self.lb = None
        self.ub = None
        
    def set_bounds(self):
        osc = self.material.oscillators
        n_osc = len(osc.A)
        
        osc_min_A = np.ones(n_osc) * 1e-10
        osc_min_gamma = np.ones(n_osc) * 0.025
        osc_min_omega = np.ones(n_osc) * self.material.e_gap
        
        if osc.model == 'Drude':
            osc_max_A = np.ones(n_osc) * 2e3
        else:
            osc_max_A = np.ones(n_osc)

        osc_max_gamma = np.ones(n_osc) * 100
        osc_max_omega = np.ones(n_osc) * 500

        if osc.model == 'MLL':
            osc_min_U, osc_max_U = 0.0, 10.0
            self.lb = np.concatenate([osc_min_A, osc_min_gamma, osc_min_omega, [osc_min_U]])
            self.ub = np.concatenate([osc_max_A, osc_max_gamma, osc_max_omega, [osc_max_U]])
            
        elif self.fit_alpha and osc.model != 'Mermin':
            osc_min_alpha, osc_max_alpha = 0.0, 1.0
            self.lb = np.concatenate([osc_min_A, osc_min_gamma, osc_min_omega, [osc_min_alpha]])
            self.ub = np.concatenate([osc_max_A, osc_max_gamma, osc_max_omega, [osc_max_alpha]])
            
        else:
            self.lb = np.concatenate([osc_min_A, osc_min_gamma, osc_min_omega])
            self.ub = np.concatenate([osc_max_A, osc_max_gamma, osc_max_omega])

    def struct2vec(self, osc_struct):
        osc = osc_struct.oscillators
        if osc.model == 'MLL':
            vec = np.concatenate([osc.A, osc.gamma, osc.omega, [osc_struct.u]])
        elif self.fit_alpha:
            vec = np.concatenate([osc.A, osc.gamma, osc.omega, [osc.alpha]])
        else:
            vec = np.concatenate([osc.A, osc.gamma, osc.omega])
        return vec

    def vec2struct(self, osc_vec):
        """Creates a deep copy of the material and updates it with the optimizer's current guess."""
        material = copy.deepcopy(self.material)
        
        if material.oscillators.model == 'MLL' or self.fit_alpha:
            oscillators = np.split(osc_vec[:-1], 3)
        else:
            oscillators = np.split(osc_vec, 3)          
        
        material.oscillators.A = oscillators[0]
        material.oscillators.gamma = oscillators[1]
        material.oscillators.omega = oscillators[2]
        
        if material.oscillators.model == 'MLL':
            material.u = osc_vec[-1]
        elif self.fit_alpha:
            material.oscillators.alpha = osc_vec[-1]
            
        return material

    # =====================================================================
    # OBJECTIVE FUNCTIONS
    # =====================================================================
    def objective_function_ndiimfp(self, osc_vec, grad):
        self.count += 1
        material = self.vec2struct(osc_vec)
        
        # Use our new InelasticEngine
        engine = InelasticEngine(material)
        engine.calculate_diimfp(self.e0, self.de, self.n_q, normalised=True)
        
        diimfp_interp = np.interp(self.exp_data.x_ndiimfp, material.diimfp_e, material.diimfp)
        rms = np.sum((self.exp_data.y_ndiimfp - diimfp_interp)**2) / self.exp_data.x_ndiimfp.size

        if grad.size > 0:
            grad[:] = 0  # COBYLA doesn't use gradients, but nlopt requires the array be handled safely
            
        self.bar.update(1)
        return rms

    def objective_function_elf(self, osc_vec, grad):
        self.count += 1
        material = self.vec2struct(osc_vec)
        
        # Use our new DielectricFunction engine
        df = DielectricFunction(material)
        epsilon = df.calculate()
        elf = (-1 / epsilon).imag
        elf[np.isnan(elf)] = 1e-5
        
        elf_interp = np.interp(self.exp_data.x_elf, material.eloss, elf)
        rms = np.sum((self.exp_data.y_elf - elf_interp)**2) / self.exp_data.x_elf.size

        if grad.size > 0:
            grad[:] = 0

        self.bar.update(1)
        return rms

    def objective_function(self, osc_vec, grad):
        self.count += 1
        material = self.vec2struct(osc_vec)
        
        # 1. DIIMFP Evaluation
        engine = InelasticEngine(material)
        engine.calculate_diimfp(self.e0, self.de, self.n_q, normalised=True)
        diimfp_interp = np.interp(self.exp_data.x_ndiimfp, material.diimfp_e, material.diimfp)

        # 2. ELF Evaluation
        df = DielectricFunction(material)
        epsilon = df.calculate()
        elf = (-1 / epsilon).imag
        elf[np.isnan(elf)] = 1e-5
        elf_interp = np.interp(self.exp_data.x_elf, material.eloss, elf)

        ind_ndiimfp = self.exp_data.y_ndiimfp > 0
        ind_elf = self.exp_data.y_elf > 0

        rms = (self.diimfp_coef * np.sum((self.exp_data.y_ndiimfp[ind_ndiimfp] - diimfp_interp[ind_ndiimfp])**2) / np.sum(ind_ndiimfp) + 
               self.elf_coef * np.sum((self.exp_data.y_elf[ind_elf] - elf_interp[ind_elf])**2) / np.sum(ind_elf))
        
        if grad.size > 0:
            grad[:] = 0

        self.bar.update(1)
        return rms

    # =====================================================================
    # CONSTRAINTS
    # =====================================================================
    def fsum_constraint(self, osc_vec, grad):
        material = self.vec2struct(osc_vec)
        df = DielectricFunction(material)
        fsum = df.evaluate_f_sum()
        val = np.fabs(fsum - material.Z)
        
        if grad.size > 0:
            grad[:] = 0
        return val

    def kksum_constraint(self, osc_vec, grad):
        material = self.vec2struct(osc_vec)
        df = DielectricFunction(material)
        kksum = df.evaluate_kk_sum()
        val = np.fabs(kksum - 1.0)
        
        if grad.size > 0:
            grad[:] = 0
        return val

    # =====================================================================
    # RUNNER
    # =====================================================================
    def run_optimisation(self, diimfp_coef, elf_coef, maxeval=1000, xtol_rel=1e-6, is_global=False):
        print('Starting optimisation...')
        self.bar = tqdm(total=maxeval)
        self.count = 0
        self.diimfp_coef = diimfp_coef
        self.elf_coef = elf_coef
        
        self.set_bounds()
        x0 = self.struct2vec(self.material)
        n_params = len(x0)

        # Setup local optimizer (COBYLA is derivative-free, great for this)
        opt_local = nlopt.opt(nlopt.LN_COBYLA, n_params)
        opt_local.set_maxeval(maxeval)
        opt_local.set_xtol_rel(xtol_rel)
        # opt_local.set_ftol_rel(1e-15) # Note: syntax is a method call, not assignment!
        
        if is_global:
            # AUGLAG uses the local optimizer to handle constraints
            opt = nlopt.opt(nlopt.AUGLAG, n_params)
            opt.set_local_optimizer(opt_local)
        else:
            opt = opt_local

        # Set Objective
        if diimfp_coef == 0:
            opt.set_min_objective(self.objective_function_elf)
        elif elf_coef == 0:
            opt.set_min_objective(self.objective_function_ndiimfp)
        else:
            opt.set_min_objective(self.objective_function)

        opt.set_lower_bounds(self.lb)
        opt.set_upper_bounds(self.ub)

        # Handle Henke extension pre-requisites
        if self.material.use_henke_for_ne:
            if self.material.eloss_henke is None or self.material.elf_henke is None:
                self.material.eloss_henke, self.material.elf_henke = self.material.mopt()
            
            # Pre-calculate once
            self.material.electron_density_henke = (self.material.atomic_density * self.material.Z * a0**3 - 
                1 / (2 * math.pi**2) * np.trapz(self.material.eloss_henke / h2ev * self.material.elf_henke, self.material.eloss_henke / h2ev))

        # Add Constraints
        opt.add_inequality_constraint(self.fsum_constraint)
        if self.material.use_kk_constraint:
            opt.add_inequality_constraint(self.kksum_constraint)

        opt.set_maxeval(maxeval)
        opt.set_xtol_rel(xtol_rel)

        try:
            x_opt = opt.optimize(x0)
            print(f"\nFound minimum after {self.count} evaluations")
            print(f"Minimum value = {opt.last_optimum_value()}")
            print(f"Result code = {opt.last_optimize_result()}")
        except Exception as e:
            print(f"\nNLopt Optimization failed: {e}")
            x_opt = x0
        finally:
            self.bar.close()

        return x_opt