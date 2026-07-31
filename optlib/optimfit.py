import numpy as np
import math
import copy
import traceback
from tqdm import tqdm
import nlopt

from optlib.utils import InputError
from optlib.constants import h2ev, a0, wpc

# numpy 2.0 renamed np.trapz -> np.trapezoid and removed the old name.
_trapz = getattr(np, "trapezoid", None) or np.trapz
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

        # --- sum(A) normalisation (Mermin/MLL) ---------------------------
        # In the MELF-GOS/Mermin convention the weights are a partition of
        # unity. This is not cosmetic: the combination rule in
        # DielectricFunction._mermin is
        #     1/eps = sum_i A_i/eps_i + (1 - sum(A))
        # and at optical frequencies every omega_i exceeds omega, so every
        # A_i/eps_i is negligible and eps1 collapses to ~1/(1 - sum(A)).
        # sum(A) therefore controls the entire optical response, while
        # neither the f-sum rule nor the KK sum rule can see it (both
        # integrate to 30 keV, where the optical region is ~0.02% of the
        # total). set_bounds constrains each A_i <= 1 but never the sum, so
        # without this the optimiser can drift sum(A) well above 1 with all
        # other diagnostics still looking excellent.
        self.renormalise_A = True

        # --- energy-dependent weighting for the ELF residual --------------
        # None = unweighted. See set_energy_weights().
        self.elf_weight = None

        # --- optional soft penalties (both disabled by default) -----------
        # optical_targets: [(E_eV, eps1_target, eps2_target, weight), ...]
        self.optical_targets = None
        self.optical_coef = 0.0
        # neff_targets: [(E_max_eV, target_Neff, weight), ...]
        self.neff_targets = None
        self.neff_coef = 0.0
        
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
        elif self.fit_alpha and osc.model != 'Mermin':
            # Must match set_bounds exactly. The Mermin model ignores alpha,
            # and set_bounds excludes it for Mermin; previously struct2vec /
            # vec2struct did not, so Mermin + fit_alpha=True produced a
            # 3n+1-long x0 against 3n-long bounds and nlopt raised
            # invalid_argument with an empty message.
            vec = np.concatenate([osc.A, osc.gamma, osc.omega, [osc.alpha]])
        else:
            vec = np.concatenate([osc.A, osc.gamma, osc.omega])
        return vec

    def vec2struct(self, osc_vec):
        """
        Deep-copy the material and update it with the optimizer's guess.

        Two things happen here beyond the copy:

        * CACHE INVALIDATION. Material caches derived quantities
          (_epsilon via the lazy `epsilon` property, `elf`, and the Henke
          extension arrays). copy.deepcopy duplicates those caches, and
          nothing invalidates them when the oscillator parameters change.
          extend_to_henke only recomputes elf `if self.elf is None`, so any
          code path reading it sees the ORIGINAL spectrum forever. That
          silently freezes evaluate_f_sum / evaluate_kk_sum, making
          fsum_constraint and kksum_constraint return a constant for every
          parameter vector. A constant, permanently-violated constraint
          gives COBYLA a feasibility problem with no descent direction, so
          it returns the starting point after exhausting maxeval, reporting
          success with x_opt == x0 and no error raised. This bites whenever
          the caches were populated before fitting -- e.g. checking the sum
          rule in a cell above, which is a natural thing to do.
          eloss_henke/elf_henke are deliberately NOT cleared: those are the
          tabulated X-ray scattering factors, which depend on composition
          only and are expensive to reload.

        * sum(A) RENORMALISATION (see self.renormalise_A in __init__).
        """
        material = copy.deepcopy(self.material)

        if material.oscillators.model == 'MLL' or (self.fit_alpha and material.oscillators.model != 'Mermin'):
            oscillators = np.split(osc_vec[:-1], 3)
        else:
            oscillators = np.split(osc_vec, 3)

        A = np.asarray(oscillators[0], dtype=float)
        if self.renormalise_A and material.oscillators.model in ('Mermin', 'MLL'):
            total = A.sum()
            if total > 0:
                A = A / total

        material.oscillators.A = A
        material.oscillators.gamma = oscillators[1]
        material.oscillators.omega = oscillators[2]

        if material.oscillators.model == 'MLL':
            material.u = osc_vec[-1]
        elif self.fit_alpha and material.oscillators.model != 'Mermin':
            material.oscillators.alpha = osc_vec[-1]

        material._epsilon = None
        material.elf = None
        material.eloss_extended_to_henke = None
        material.elf_extended_to_henke = None

        return material

    # =====================================================================
    # OPTIONAL REGULARISATION HELPERS
    # =====================================================================
    def set_energy_weights(self, low_energy_boost=20.0, energy_scale=5.0, weights=None):
        """
        Build an energy-dependent weight array for the ELF residual.

        An unweighted sum of squares over a spectrum spanning hundreds of eV
        is dominated by the high-energy region simply because it has more
        points and larger values, so the low-loss region can be almost
        unconstrained even when the overall fit looks good.

            weight(E) = 1 + (low_energy_boost - 1) * exp(-E / energy_scale)

        The decay is smooth on purpose: a hard cutoff puts a kink in the
        objective landscape that derivative-free optimisers handle poorly.
        Pass `weights` (same shape as exp_data.x_elf) to override entirely.
        """
        if weights is not None:
            self.elf_weight = np.asarray(weights, dtype=float)
        else:
            x = np.asarray(self.exp_data.x_elf, dtype=float)
            self.elf_weight = 1.0 + (low_energy_boost - 1.0) * np.exp(-x / energy_scale)
        return self.elf_weight

    def _optical_penalty(self, eloss, epsilon):
        """
        Soft penalty pulling eps1(E), eps2(E) toward reference optical
        constants at chosen checkpoints (literature tables, or your own
        ellipsometry). Constrains lineshape directly, unlike integrated
        diagnostics which only see enclosed area.

        Uses the epsilon already computed by the caller rather than
        recomputing it.
        """
        if not self.optical_targets:
            return 0.0
        eloss = np.atleast_1d(eloss)
        eps1 = np.atleast_1d(np.squeeze(epsilon.real))
        eps2 = np.atleast_1d(np.squeeze(epsilon.imag))
        targets_E = [t[0] for t in self.optical_targets]
        e1i = np.interp(targets_E, eloss, eps1)
        e2i = np.interp(targets_E, eloss, eps2)
        total, wsum = 0.0, 0.0
        for (E, t1, t2, w), a, b in zip(self.optical_targets, e1i, e2i):
            total += w * ((a - t1)**2 + (b - t2)**2)
            wsum += w
        return total / wsum if wsum > 0 else 0.0

    def _neff_partial(self, material, e_max):
        """
        Cumulative effective electron count Neff(E <= e_max), same formula as
        evaluate_f_sum but restricted to a partial range and evaluated from
        the oscillator model directly (no Henke extension needed while
        e_max << henke_limit).

        Complements the total f-sum rule, which is a single number and says
        nothing about how oscillator strength is distributed in energy.
        Note it is an INTEGRATED quantity, so it cannot distinguish a broad
        weak oscillator from a narrow strong one of equal area; pair it with
        _optical_penalty if lineshape matters.
        """
        df = DielectricFunction(material)
        elf = np.atleast_1d(np.squeeze((-1 / df.calculate()).imag))
        elf = np.where(np.isnan(elf), 1e-5, elf)
        eloss = np.atleast_1d(material.eloss)
        ind = eloss <= e_max
        if not np.any(ind):
            return 0.0
        return (1.0 / (2 * math.pi**2 * (material.atomic_density * a0**3))) * _trapz(
            eloss[ind] / h2ev * elf[ind], eloss[ind] / h2ev)

    def _neff_penalty(self, material):
        if not self.neff_targets:
            return 0.0
        total, wsum = 0.0, 0.0
        for e_max, target, w in self.neff_targets:
            total += w * (self._neff_partial(material, e_max) - target)**2
            wsum += w
        return total / wsum if wsum > 0 else 0.0

    def _finite(self, rms):
        """
        Guard against a non-finite objective. A NaN return value stalls
        COBYLA completely -- it cannot order NaN against anything, so it
        never accepts a step and silently returns the starting point.
        Returning a large finite value instead lets the optimiser reject the
        point and carry on.
        """
        return rms if np.isfinite(rms) else 1e10

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
        return self._finite(rms)

    def objective_function_elf(self, osc_vec, grad):
        self.count += 1
        material = self.vec2struct(osc_vec)
        
        # Use our new DielectricFunction engine
        df = DielectricFunction(material)
        epsilon = df.calculate()
        elf = (-1 / epsilon).imag
        elf[np.isnan(elf)] = 1e-5
        
        elf_interp = np.interp(self.exp_data.x_elf, material.eloss, elf)

        w = self.elf_weight if self.elf_weight is not None else np.ones_like(self.exp_data.x_elf, dtype=float)
        rms = np.sum(w * (self.exp_data.y_elf - elf_interp)**2) / np.sum(w)

        if self.optical_coef > 0:
            rms += self.optical_coef * self._optical_penalty(material.eloss, epsilon)
        if self.neff_coef > 0:
            rms += self.neff_coef * self._neff_penalty(material)

        if grad.size > 0:
            grad[:] = 0

        self.bar.update(1)
        return self._finite(rms)

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

        w_full = self.elf_weight if self.elf_weight is not None else np.ones_like(self.exp_data.x_elf, dtype=float)
        w = w_full[ind_elf]

        rms = (self.diimfp_coef * np.sum((self.exp_data.y_ndiimfp[ind_ndiimfp] - diimfp_interp[ind_ndiimfp])**2) / np.sum(ind_ndiimfp) + 
               self.elf_coef * np.sum(w * (self.exp_data.y_elf[ind_elf] - elf_interp[ind_elf])**2) / np.sum(w))

        if self.optical_coef > 0:
            rms += self.optical_coef * self._optical_penalty(material.eloss, epsilon)
        if self.neff_coef > 0:
            rms += self.neff_coef * self._neff_penalty(material)
        
        if grad.size > 0:
            grad[:] = 0

        self.bar.update(1)
        return self._finite(rms)

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

    def sumA_constraint(self, osc_vec, grad):
        """
        |sum(A) - 1| for Mermin/MLL. Only meaningful when
        self.renormalise_A is False -- with renormalisation on, sum(A) is 1
        by construction and this returns ~0. Add with a NONZERO tolerance:
        |sum(A)-1| >= 0 always, so tol=0 makes the feasible set measure-zero.
        """
        n = len(self.material.oscillators.A)
        val = float(np.fabs(np.sum(osc_vec[:n]) - 1.0))
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
    def run_optimisation(self, diimfp_coef, elf_coef, maxeval=1000, xtol_rel=1e-6, is_global=False,
                         fsum_tol=0.5, kksum_tol=0.02, sumA_tol=None):
        print('Starting optimisation...')
        self.bar = tqdm(total=maxeval)
        self.count = 0
        self.diimfp_coef = diimfp_coef
        self.elf_coef = elf_coef
        
        self.set_bounds()
        x0 = self.struct2vec(self.material)
        # nlopt raises invalid_argument (with an EMPTY message) if x0 falls
        # outside the bounds -- easy to trigger after tightening a bound
        # without moving the corresponding start value.
        out_of_bounds = np.where((x0 < self.lb) | (x0 > self.ub))[0]
        if out_of_bounds.size:
            print(f"Warning: {out_of_bounds.size} start value(s) outside bounds; clipping. "
                  f"Indices: {out_of_bounds.tolist()}")
            x0 = np.clip(x0, self.lb, self.ub)
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
                1 / (2 * math.pi**2) * _trapz(self.material.eloss_henke / h2ev * self.material.elf_henke, self.material.eloss_henke / h2ev))

        # Add Constraints.
        # NOTE the explicit tolerances. These constraints have the form
        # |f(x) - target|, which is >= 0 everywhere, so with nlopt's default
        # tolerance of 0 the feasible set is exactly {f(x) == target}: a
        # measure-zero target the optimiser can never certify as satisfied.
        opt.add_inequality_constraint(self.fsum_constraint, fsum_tol)
        if self.material.use_kk_constraint:
            opt.add_inequality_constraint(self.kksum_constraint, kksum_tol)
        if sumA_tol is not None and not self.renormalise_A:
            opt.add_inequality_constraint(self.sumA_constraint, sumA_tol)

        opt.set_maxeval(maxeval)
        opt.set_xtol_rel(xtol_rel)

        try:
            x_opt = opt.optimize(x0)
            print(f"\nFound minimum after {self.count} evaluations")
            print(f"Minimum value = {opt.last_optimum_value()}")
            print(f"Result code = {opt.last_optimize_result()}")
        except Exception as e:
            # nlopt exceptions frequently have an empty str(), so print the
            # traceback rather than only the message.
            print(f"\nNLopt Optimization failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            x_opt = x0
        finally:
            self.bar.close()

        # Write the result back so self.material reflects the fit. vec2struct
        # previously only ever modified deep copies, so callers inspecting
        # self.material after fitting saw the untouched starting values and
        # could easily mistake a successful fit for a frozen one.
        self.material = self.vec2struct(x_opt)

        return x_opt
