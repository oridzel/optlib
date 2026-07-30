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

        # --- Energy-dependent weighting for the ELF objective ---
        # None = unweighted (original behaviour). Call set_energy_weights()
        # to emphasize the low-loss/optical region, where a handful of
        # low-energy oscillators otherwise carry very little influence on
        # an unweighted sum-of-squares residual dominated by the much
        # larger high-energy portion of the spectrum.
        self.elf_weight = None

        # --- Soft Neff(E) regularization (cumulative effective electron count) ---
        # This is NOT the same as fsum_constraint: fsum_constraint checks the
        # total (E -> infinity) sum rule as a single number; neff_targets
        # checks how that oscillator strength is DISTRIBUTED across energy,
        # which the total sum rule alone cannot see.
        # Format: list of (e_max, target_neff, weight) tuples, e.g.
        #   [(3.0, 0.003, 1.0), (5.0, 0.014, 1.0), (6.5, 0.034, 1.0)]
        # Leave as None / neff_coef=0 to disable (original behaviour).
        self.neff_targets = None
        self.neff_coef = 0.0

        # --- Soft optical-constant anchoring (eps1(E), eps2(E) directly) ---
        # This is the more load-bearing check of the two: Neff(E) only sees
        # the INTEGRATED area under E*ELF(E), so a broadened-but-weaker
        # oscillator can match a Neff target while still killing the actual
        # eps1 depth (this is exactly what happened in testing -- Neff(E)
        # improved while the LSPR vanished, traced to a systematic ~1.5x
        # increase in oscillator damping). Anchoring eps1/eps2 directly at a
        # few checkpoint energies constrains lineshape sharpness, not just
        # enclosed area, and catches that failure mode immediately.
        # Format: list of (E, eps1_target, eps2_target, weight) tuples.
        self.optical_targets = None
        self.optical_coef = 0.0
        
    # =====================================================================
    # ENERGY-DEPENDENT WEIGHTING (low-loss/optical emphasis)
    # =====================================================================
    def set_energy_weights(self, low_energy_boost=20.0, energy_scale=5.0, weights=None):
        """
        Builds an energy-dependent weight array for the ELF residual.

        Default scheme is a smooth exponential boost (not a hard cutoff --
        a discontinuous step can create a kink in the objective landscape
        that derivative-free optimizers like COBYLA don't love):

            weight(E) = 1 + (low_energy_boost - 1) * exp(-E / energy_scale)

        At E=0 the weight is `low_energy_boost`; it decays smoothly back to
        1 by roughly E ~ 3-4x energy_scale. Defaults (boost=20, scale=5 eV)
        give ~20x weight at E=0, ~7x at 5 eV, ~1x by ~20 eV -- adjust
        energy_scale to match where your low-energy region of interest ends.

        Pass `weights` directly (same shape as exp_data.x_elf) to override
        with a fully custom scheme instead.
        """
        if weights is not None:
            self.elf_weight = np.asarray(weights, dtype=float)
        else:
            x = self.exp_data.x_elf
            self.elf_weight = 1.0 + (low_energy_boost - 1.0) * np.exp(-x / energy_scale)

    # =====================================================================
    # NEFF(E) SOFT REGULARIZATION (cumulative effective electron count)
    # =====================================================================
    def _neff_partial(self, material, e_max):
        """
        Cumulative Neff(E <= e_max), computed with the SAME formula as
        DielectricFunction.evaluate_f_sum(), just restricted to a partial
        energy range and evaluated directly from the oscillator model
        (no Henke extension -- fine as long as e_max stays well below
        material.henke_limit, which is true for any optical-range checkpoint).

        This is a genuinely different diagnostic from the total f-sum rule:
        the total sum rule is one number (integral to infinity); this is a
        function of e_max that shows HOW oscillator strength accumulates,
        which is exactly what a single global constraint can't see.
        """
        df = DielectricFunction(material)
        epsilon = df.calculate()
        elf = np.squeeze((-1 / epsilon).imag)
        elf = np.atleast_1d(elf)
        elf[np.isnan(elf)] = 1e-5

        eloss = np.atleast_1d(material.eloss)
        ind = eloss <= e_max
        if not np.any(ind):
            return 0.0

        neff = (1.0 / (2 * math.pi**2 * (material.atomic_density * a0**3))) * np.trapezoid(
            eloss[ind] / h2ev * elf[ind], eloss[ind] / h2ev
        )
        return neff

    def _neff_penalty(self, material):
        """
        Soft penalty pulling the model's Neff(E) toward literature/expected
        checkpoints in self.neff_targets = [(e_max, target, weight), ...].
        Returned value is added directly to the RMS objective, scaled by
        self.neff_coef -- treat neff_coef as a regularization strength you
        tune, not a hard requirement, since the targets themselves usually
        come from another model's fit to data (e.g. Etchegoin's analytic
        fit to Johnson & Christy), not an exact theoretical number.
        """
        if not self.neff_targets:
            return 0.0
        total, wsum = 0.0, 0.0
        for e_max, target, w in self.neff_targets:
            neff = self._neff_partial(material, e_max)
            total += w * (neff - target) ** 2
            wsum += w
        return total / wsum if wsum > 0 else 0.0

    # =====================================================================
    # OPTICAL-CONSTANT ANCHORING (direct eps1(E), eps2(E) matching)
    # =====================================================================
    def _optical_penalty(self, eloss, epsilon):
        """
        Soft penalty pulling the model's eps1(E), eps2(E) directly toward
        reference optical constants (e.g. literature Johnson & Christy, or
        your own ellipsometry) at specific checkpoint energies.

        Pass the eloss/epsilon ALREADY COMPUTED by the calling objective
        function (don't recompute DielectricFunction again here -- that's
        wasted work, unlike the Neff helper above which needs its own call
        since it integrates over a different, wider energy range).

        self.optical_targets: list of (E, eps1_target, eps2_target, weight).
        """
        if not self.optical_targets:
            return 0.0

        eloss = np.atleast_1d(eloss)
        eps1 = np.atleast_1d(np.squeeze(epsilon.real))
        eps2 = np.atleast_1d(np.squeeze(epsilon.imag))

        targets_E = [t[0] for t in self.optical_targets]
        eps1_interp = np.interp(targets_E, eloss, eps1)
        eps2_interp = np.interp(targets_E, eloss, eps2)

        total, wsum = 0.0, 0.0
        for (E, eps1_t, eps2_t, w), e1i, e2i in zip(self.optical_targets, eps1_interp, eps2_interp):
            total += w * ((e1i - eps1_t) ** 2 + (e2i - eps2_t) ** 2)
            wsum += w
        return total / wsum if wsum > 0 else 0.0

    # =====================================================================
    # OPTIONAL: hard nlopt constraint version, if you'd rather enforce a
    # checkpoint exactly (mirrors the style of fsum_constraint/kksum_constraint
    # below). Add via opt.add_inequality_constraint(...) in run_optimisation
    # if you want this instead of / in addition to the soft penalty above.
    # =====================================================================
    def neff_constraint(self, osc_vec, grad, e_max, target):
        material = self.vec2struct(osc_vec)
        val = np.fabs(self._neff_partial(material, e_max) - target)
        if grad.size > 0:
            grad[:] = 0
        return val

    def set_bounds(self):
        osc = self.material.oscillators
        n_osc = len(osc.A)
        
        osc_min_A = np.ones(n_osc) * 1e-10
        osc_min_gamma = np.ones(n_osc) * 0.025
        # interband edges are broad; forbid sharp low-energy resonances
        low_e = np.array(osc.omega) < 10.0
        osc_min_gamma[low_e] = 1.5
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

        w = self.elf_weight if self.elf_weight is not None else np.ones_like(self.exp_data.x_elf)
        rms = np.sum(w * (self.exp_data.y_elf - elf_interp)**2) / np.sum(w)

        if self.neff_coef > 0:
            rms += self.neff_coef * self._neff_penalty(material)

        if self.optical_coef > 0:
            rms += self.optical_coef * self._optical_penalty(material.eloss, epsilon)

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

        w_full = self.elf_weight if self.elf_weight is not None else np.ones_like(self.exp_data.x_elf)
        w = w_full[ind_elf]

        rms = (self.diimfp_coef * np.sum((self.exp_data.y_ndiimfp[ind_ndiimfp] - diimfp_interp[ind_ndiimfp])**2) / np.sum(ind_ndiimfp) + 
               self.elf_coef * np.sum(w * (self.exp_data.y_elf[ind_elf] - elf_interp[ind_elf])**2) / np.sum(w))

        if self.neff_coef > 0:
            rms += self.neff_coef * self._neff_penalty(material)

        if self.optical_coef > 0:
            rms += self.optical_coef * self._optical_penalty(material.eloss, epsilon)
        
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
                1 / (2 * math.pi**2) * np.trapezoid(self.material.eloss_henke / h2ev * self.material.elf_henke, self.material.eloss_henke / h2ev))

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
