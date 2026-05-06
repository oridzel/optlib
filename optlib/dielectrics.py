import numpy as np
import math
from optlib.constants import *
from optlib.utils import InputError

class Oscillators:
    """Stores the oscillator parameters for dielectric function modeling."""
    def __init__(self, model, A, gamma, omega, alpha=1.0, eps_b=1.0):
        self.model = model
        self.A = np.array(A, dtype=float)
        self.gamma = np.array(gamma, dtype=float)
        self.omega = np.array(omega, dtype=float)
        self.alpha = float(alpha)
        self.eps_b = float(eps_b)

        if len(self.omega) != len(self.A) or len(self.omega) != len(self.gamma):
            raise InputError("The number of oscillator parameters (A, gamma, omega) must be the same!")

    def __str__(self):
        return f'Model = {self.model}\n' \
               f'A = [{", ".join([str(i) for i in self.A.round(3).tolist()])}]\n' \
               f'gamma = [{", ".join([str(i) for i in self.gamma.round(3).tolist()])}]\n' \
               f'omega = [{", ".join([str(i) for i in self.omega.round(3).tolist()])}]\n' \
               f'alpha = {self.alpha}\n' \
               f'eps_b = {self.eps_b}'


class DielectricFunction:
    """
    Computes the complex dielectric function epsilon(q, omega) 
    using various oscillator models (Drude, DrudeLindhard, Mermin, MLL).
    """
    def __init__(self, material):
        self.mat = material
        self.osc = material.oscillators

    def kramers_kronig(self, epsilon_imag):
        """Calculates the real part of the dielectric function via Kramers-Kronig relations."""
        eps_real = np.zeros_like(self.mat.eloss)
        for i in range(self.mat.eloss.size):
            omega = self.mat.eloss[i]
            ind = np.logical_and(self.mat.eloss != omega, self.mat.eloss > self.mat.e_gap)
            
            if len(epsilon_imag.shape) > 1:
                kk_sum = np.trapz(self.mat.eloss[ind] * epsilon_imag[ind, 0] / 
                                  (self.mat.eloss[ind] ** 2 - omega ** 2), self.mat.eloss[ind])
            else:
                kk_sum = np.trapz(self.mat.eloss[ind] * epsilon_imag[ind] / 
                                  (self.mat.eloss[ind] ** 2 - omega ** 2), self.mat.eloss[ind])
            
            eps_real[i] = 2 * kk_sum / math.pi + 1
        return eps_real

    def calculate(self):
        """Routes to the correct dielectric model based on the oscillators."""
        self.mat._convert2au() 

        if self.osc.model == 'Drude':
            epsilon = self._drude()
        elif self.osc.model == 'DrudeLindhard':
            epsilon = self._drude_lindhard()
        elif self.osc.model == 'Mermin':
            epsilon = self._mermin()
        elif self.osc.model == 'MLL':
            epsilon = self._mll()
        else:
            self.mat._convert2ru()
            raise InputError(f"Invalid model name '{self.osc.model}'. Valid models: Drude, DrudeLindhard, Mermin, MLL")

        self.mat._convert2ru() 
        return epsilon

    # ----------------------------------------------------------------------
    # 1. DRUDE MODEL
    # ----------------------------------------------------------------------
    def _drude(self):
        eps_real = None
        eps_imag = None

        for i in range(len(self.osc.A)):
            e_real, e_imag = self._drude_oscillator(self.osc.omega[i], self.osc.gamma[i], self.osc.alpha)
            
            if i == 0:
                eps_real = self.osc.eps_b * np.ones_like(e_real)
                eps_imag = np.zeros_like(e_imag)
                epsilon = np.zeros_like(eps_real, dtype=complex)
                
            eps_real -= self.osc.A[i] * e_real
            eps_imag += self.osc.A[i] * e_imag

        if self.mat.e_gap > 0 and self.mat.zero_gap:
            if len(eps_imag.shape) > 1:
                eps_imag[self.mat.eloss <= self.mat.e_gap, 0] = 1e-5
            else:
                eps_imag[self.mat.eloss <= self.mat.e_gap] = 1e-5
                
        if self.mat.use_kk_relation:
            if len(eps_real.shape) > 1:
                eps_real[:, 0] = self.kramers_kronig(eps_imag)
            else:
                eps_real = self.kramers_kronig(eps_imag)

        epsilon.real = eps_real
        epsilon.imag = eps_imag
        return epsilon

    def _drude_oscillator(self, omega0, gamma, alpha):
        w_at_q = omega0 + 0.5 * alpha * self.mat.q**2
        
        if self.mat.size_q == 1:
            omega = np.squeeze(np.array([self.mat.eloss] * self.mat.size_q).transpose())
        else:
            omega = np.expand_dims(self.mat.eloss, axis=tuple(range(1, self.mat.q.ndim)))

        mm = omega**2 - w_at_q**2
        divisor = mm**2 + omega**2 * gamma**2

        return mm / divisor, (omega * gamma) / divisor

    # ----------------------------------------------------------------------
    # 2. DRUDE-LINDHARD MODEL
    # ----------------------------------------------------------------------
    def _drude_lindhard(self):
        shape = (self.mat.eloss.shape[0], self.mat.size_q)
        sum_oneover_eps = np.squeeze(np.zeros(shape, dtype=complex))

        for i in range(len(self.osc.A)):
            oneover_eps = self._dl_oscillator(self.osc.omega[i], self.osc.gamma[i], self.osc.alpha)
            sum_oneover_eps += self.osc.A[i] * (oneover_eps - complex(1))

        sum_oneover_eps += complex(1)
        epsilon = complex(1) / np.squeeze(sum_oneover_eps)

        if self.mat.use_kk_relation:
            eps_imag = epsilon.imag
            eps_real = self.kramers_kronig(eps_imag)
            epsilon.real = eps_real
            epsilon.imag = eps_imag

        return epsilon

    def _dl_oscillator(self, omega0, gamma, alpha):
        if getattr(self.mat, 'q_dependency', None) is not None:
            w_at_q = omega0 - self.mat.q_dependency(0)/h2ev + self.mat.q_dependency(self.mat.q / a0)/h2ev
        else:
            w_at_q = omega0 + 0.5 * alpha * self.mat.q**2

        omega = np.squeeze(np.array([self.mat.eloss] * self.mat.size_q).transpose())
        mm = omega**2 - w_at_q**2
        divisor = mm**2 + omega**2 * gamma**2

        one_over_eps_imag = -omega0**2 * omega * gamma / divisor
        if self.mat.e_gap > 0:
            one_over_eps_imag[self.mat.eloss <= self.mat.e_gap] = 1e-5
            
        if self.mat.use_kk_relation:
            one_over_eps_real = self.kramers_kronig(one_over_eps_imag)
        else:
            one_over_eps_real = 1.0 + omega0**2 * mm / divisor

        return np.squeeze(np.apply_along_axis(
            lambda args: [complex(*args)], 0, np.array([one_over_eps_real, one_over_eps_imag])
        ))

    # ----------------------------------------------------------------------
    # 3. MERMIN MODEL
    # ----------------------------------------------------------------------
    def _mermin(self):
        if self.mat.size_q == 1 and np.all(self.mat.q == 0):
            self.mat.q = np.array([0.01])
            
        shape = (self.mat.eloss.shape[0], self.mat.size_q)
        oneovereps = np.squeeze(np.zeros(shape, dtype=complex))

        for i in range(len(self.osc.A)):
            if all(np.abs((self.mat.eloss - self.osc.omega[i]) / self.osc.gamma[i]) < 100000):
                eps_mermin = self._mermin_oscillator(self.osc.omega[i], self.osc.gamma[i])
            else:
                eps_mermin = complex(1)
            oneovereps += self.osc.A[i] * (complex(1) / eps_mermin)
            
        oneovereps += complex(1) - complex(np.sum(self.osc.A))
        
        if self.mat.e_gap > 0:
            if oneovereps.ndim > 1:
                oneovereps.imag[self.mat.eloss <= self.mat.e_gap, :] = 1e-5
            else:
                oneovereps.imag[self.mat.eloss <= self.mat.e_gap] = 1e-5

        epsilon = complex(1) / np.squeeze(oneovereps)
        
        if self.mat.use_kk_relation:
            eps_imag = epsilon.imag
            epsilon.real = self.kramers_kronig(eps_imag)
            epsilon.imag = eps_imag

        return epsilon

    def _mermin_oscillator(self, omega0, gamma):
        omega = np.squeeze(np.array([self.mat.eloss] * self.mat.size_q).transpose())
        gamma_over_omega = gamma / omega
        
        complex_array = np.vectorize(complex)
        z1 = complex_array(1, gamma_over_omega)
        z2 = self._lindhard_oscillator(omega, gamma, omega0) - complex(1)
        z3 = self._lindhard_oscillator(np.zeros_like(omega), 0, omega0) - complex(1)
        
        top = z1 * z2
        bottom = complex(1) + complex_array(0, gamma_over_omega) * z2 / z3
        return complex(1) + top / bottom

    # ----------------------------------------------------------------------
    # 4. MLL MODEL
    # ----------------------------------------------------------------------
    def _mll(self):
        if getattr(self.mat, 'u', 0) == 0:
            raise InputError("Please specify the value of U for the MLL model.")
        if self.mat.size_q == 1 and np.all(self.mat.q == 0):
            self.mat.q = np.array([0.01])

        shape = (self.mat.eloss.shape[0], self.mat.size_q)
        oneovereps = np.squeeze(np.zeros(shape, dtype=complex))

        for i in range(len(self.osc.A)):
            eps_mll = self._mll_oscillator(self.osc.omega[i], self.osc.gamma[i])
            oneovereps += self.osc.A[i] * (complex(1) / eps_mll - complex(1))
            
        oneovereps += complex(1)
        return complex(1) / np.squeeze(oneovereps)

    def _mll_oscillator(self, omega0, gamma):
        omega = np.squeeze(np.array([self.mat.eloss] * self.mat.size_q).transpose())
        gamma_over_omega = gamma / omega
        
        complex_array = np.vectorize(complex)
        z1 = complex_array(1, gamma_over_omega)
        z2 = self._eps_llx(omega, gamma, omega0) - complex(1)
        z3 = self._eps_llx(np.zeros_like(omega), 0, omega0) - complex(1)
        
        top = z1 * z2
        bottom = complex(1) + complex_array(0, gamma_over_omega) * z2 / z3
        return complex(1) + top / bottom

    def _eps_llx(self, omega, gamma, omega0):
        complex_array = np.vectorize(complex)
        omega_minus_square = complex_array(omega**2 - self.mat.u**2 - gamma**2, 2.0 * omega * gamma)
        r = np.abs(omega_minus_square)
        theta = np.arctan2(omega_minus_square.imag, omega_minus_square.real)
        omega_minus = complex_array(np.sqrt(r) * np.cos(theta / 2.0), np.sqrt(r) * np.sin(theta / 2.0))
        
        epsilon = np.zeros_like(omega_minus)
        ind_ge = omega_minus.real >= 0
        ind_lt = omega_minus.real < 0

        if any(ind_ge.flatten()):
            epsilon[ind_ge] = self._lindhard_oscillator(omega_minus.real, omega_minus.imag, omega0)[ind_ge]      
            
        if any(ind_lt.flatten()):
            n_dens = omega0**2 / (4.0 * math.pi)
            E_f = 0.5 * (3.0 * math.pi**2 * n_dens)**(2.0 / 3.0)
            v_f = (2 * E_f)**0.5
            deltaSquare = -omega_minus_square / E_f**2
            r = abs(deltaSquare)
            theta = np.arctan2(deltaSquare.imag, deltaSquare.real)
            delta = complex_array(np.sqrt(r) * np.cos(theta / 2.0), np.sqrt(r) * np.sin(theta / 2.0))
            
            QQ = self.mat.q / v_f
            res1 = self._c_arctan((2.0 * QQ + QQ**2) / delta)
            res2 = self._c_arctan((2.0 * QQ + QQ**2) / delta) 
            res1 = res1 + res2
            res2 = res1 * delta
            
            z1 = complex_array(deltaSquare.real + (2 * QQ + QQ**2)**2, deltaSquare.imag)
            z2 = complex_array(deltaSquare.real + (2 * QQ - QQ**2)**2, deltaSquare.imag)
            z1 = np.log(z1 / z2)
            z2 = deltaSquare * z1
            
            p1 = res2.imag / (2 * QQ**3)
            p2 = z2.imag / (8 * QQ**5)
            p3 = z1.imag / (2 * QQ**3)
            p4 = z1.imag / (8 * QQ)    
            eps_imag = 2 / (math.pi * v_f) * (-p1 + p2 + p3 - p4)
            
            t1 = res2.real / (2 * QQ**3)
            t2 = z2.real / (8 * QQ**5)
            t3 = z1.real / (2 * QQ**3)
            t4 = z1.real / (8 * QQ)
            eps_real = 1 + 2 / (math.pi * v_f) * ((1 / QQ**2 - t1) + t2 + t3 - t4)
            
            epsilon[ind_lt] = complex_array(eps_real, eps_imag)[ind_lt]
            
        return epsilon

    def _c_arctan(self, z):
        complex_array = np.vectorize(complex)
        reres = np.zeros_like(z)
        x = z.real
        y = z.imag
        imres = -1.0 / 4.0 * np.log((1 - x**2 - y**2)**2 + 4 * x**2) + 1.0 / 2.0 * np.log((1 + y)**2 + x**2)
        reres[x != 0] = math.pi / 4.0 - 0.5 * np.arctan((1 - x**2 - y**2) / (2.0 * x))
        reres[np.logical_and(x > 0, x < 0)] = math.pi / 2.0 
        return complex_array(reres.real, imres.imag)

    # ----------------------------------------------------------------------
    # HELPER LINDHARD MATH
    # ----------------------------------------------------------------------
    def _lindhard_oscillator(self, omega, gamma, omega0):
        n_dens = omega0**2 / (4 * math.pi)
        E_f = 0.5 * (3 * math.pi**2 * n_dens)**(2.0 / 3.0)
        v_f = (2 * E_f)**0.5
        
        z = self.mat.q / (2 * v_f)
        chi = np.sqrt(1.0 / (math.pi * v_f))
        
        z1_1 = omega / (self.mat.q * v_f)
        z1_1[np.isnan(z1_1)] = 1e-5
        
        gq = gamma / (self.mat.q * v_f)
        vos_g_array = np.vectorize(self._vos_g)
        
        reD1, imD1 = vos_g_array(z1_1 + z, gq)
        reD2, imD2 = vos_g_array(z1_1 - z, gq)
        
        red1_d2 = reD1 - reD2
        imd1_d2 = imD1 - imD2
        
        chizzz = chi**2 / (z**3 * 4)
        epsreal = 1 + red1_d2 * chizzz
        epsimag = imd1_d2 * chizzz
        
        complex_array = np.vectorize(complex)
        return complex_array(epsreal, epsimag)

    def _vos_g(self, z, img_z):
        zplus1 = z + 1
        zminus1 = z - 1
        
        if img_z != 0:
            imgZ2 = img_z**2
            dummy1 = math.log(np.sqrt((zplus1**2 + imgZ2) / (zminus1**2 + imgZ2)))
            dummy2 = math.atan2(img_z, zplus1) - math.atan2(img_z, zminus1)
            reim1 = 1 - (z**2 - imgZ2)

            outreal = z + 0.5 * reim1 * dummy1 + z * img_z * dummy2
            outimag = img_z + 0.5 * reim1 * dummy2 - z * img_z * dummy1
        else:
            dummy1 = math.log(abs(zplus1) / abs(zminus1))
            dummy2 = math.atan2(0, zplus1) - math.atan2(0, zminus1)
            reim1 = 1 - z**2

            outreal = z + 0.5 * reim1 * dummy1
            outimag = 0.5 * reim1 * dummy2
            
        return outreal, outimag

    # ----------------------------------------------------------------------
    # SUM RULES
    # ----------------------------------------------------------------------
    def evaluate_f_sum(self):
        old_q = self.mat.q
        self.mat.q = 0
        self.mat.extend_to_henke() 
        ind = self.mat.eloss_extended_to_henke >= self.mat.e_gap
        
        fsum = 1 / (2 * math.pi**2 * (self.mat.atomic_density * a0**3)) * np.trapz(
            self.mat.eloss_extended_to_henke[ind]/h2ev * self.mat.elf_extended_to_henke[ind], 
            self.mat.eloss_extended_to_henke[ind]/h2ev
        )
        self.mat.q = old_q
        return fsum

    def evaluate_kk_sum(self):
        old_q = self.mat.q
        self.mat.eloss[self.mat.eloss < 1e-5] = 1e-5
        
        self.mat.q = 0.01 if self.osc.model == 'MLL' else 0
            
        self.mat.extend_to_henke()
        div = self.mat.elf_extended_to_henke / self.mat.eloss_extended_to_henke
        div[((div < 0) | (np.isnan(div)))] = 1e-5
        
        kksum = 2 / math.pi * np.trapz(div, self.mat.eloss_extended_to_henke)
        
        if self.mat.e_gap != 0:
            if getattr(self.mat, 'static_refractive_index', 0) == 0:
                kksum += 1 / self.mat.epsilon.real[0]
            else:
                kksum += 1 / self.mat.static_refractive_index**2
                
        self.mat.q = old_q
        return kksum