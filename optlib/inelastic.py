"""Inelastic mean-free-path calculations for metals and nonconductors.

``E_eV`` is measured from the bottom of the valence band. For a metal the
relativistic FPA uses ``T_prime = E_eV``. For a semiconductor or insulator it
uses ``T_prime = E_eV - E_g`` (Shinotsuka et al., Eq. 7), with domain

    E_g <= omega <= T_prime - E_v = E_eV - E_g - E_v.

Thus the electronic inelastic rate is zero below ``E_v + 2*E_g``. The stored
DIIMFP is in 1/(Angstrom eV), and the IMFP is obtained by integrating it over
energy loss.
"""

import math

import numpy as np

from optlib.constants import h2ev, a0, c
from optlib.dielectrics import DielectricFunction
from optlib.utils import InputError


_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


class InelasticEngine:
    """Calculate channel-independent DIIMFPs and IMFPs with the FPA."""

    def __init__(self, material):
        self.mat = material

    def _value(self, *names, default=None):
        """Read the first available material attribute (old and new aliases)."""
        for name in names:
            if hasattr(self.mat, name):
                value = getattr(self.mat, name)
                if value is not None:
                    return float(value)
        if default is not None:
            return float(default)
        raise InputError(
            "Material is missing one of the required attributes: "
            + ", ".join(names)
        )

    def _energy_domain(self, E_eV):
        """Return ``(omega_min, omega_max, T_prime)`` in eV."""
        E_eV = float(E_eV)
        if not math.isfinite(E_eV) or E_eV <= 0.0:
            raise InputError("Electron energy must be finite and positive")

        if self.mat.is_metal:
            e_fermi = self._value("e_fermi")
            if e_fermi <= 0.0:
                raise InputError("Please specify a positive e_fermi")
            return 1.0e-5, E_eV - e_fermi, E_eV

        e_gap = self._value("e_gap", "band_gap")
        e_vb = self._value("width_of_the_valence_band", "e_vb")
        if e_gap <= 0.0 or e_vb <= 0.0:
            raise InputError(
                "Please specify positive e_gap (band_gap) and "
                "width_of_the_valence_band (e_vb)"
            )
        t_prime = E_eV - e_gap
        return e_gap, t_prime - e_vb, t_prime

    def _store_zero_diimfp(self, E_eV, omega_min, de):
        """Store a well-formed, zero-strength row below excitation threshold."""
        step = max(float(de), 1.0e-6)
        eloss = np.array([omega_min, omega_min + step], dtype=float)
        diimfp = np.zeros(2, dtype=float)
        self.mat.diimfp = diimfp
        self.mat.iimfp = np.zeros(2, dtype=float)
        self.mat.diimfp_e = eloss
        self.mat.e0 = float(E_eV)
        return diimfp

    def calculate_diimfp(self, E_eV, de=0.5, nq=100, normalised=True):
        """Calculate the DIIMFP at one VB-bottom-referenced electron energy.

        Set ``normalised=False`` when building physical DIIMFP and IMFP tables.
        With ``normalised=True`` the returned/stored DIIMFP is a unit-area
        sampling PDF, while ``mat.iimfp`` retains the raw atomic-unit result.
        """
        de = float(de)
        nq = int(nq)
        if not math.isfinite(de) or de <= 0.0:
            raise InputError("de must be finite and positive")
        if nq < 2:
            raise InputError("nq must be at least 2")

        omega_min, omega_max, t_prime = self._energy_domain(E_eV)
        # omega_max >= omega_min is equivalent to E >= E_v + 2 E_g for
        # nonconductors. Zero rows let callers build a complete grid at once.
        if omega_max <= omega_min or t_prime <= 0.0:
            return self._store_zero_diimfp(E_eV, omega_min, de)

        old_eloss = self.mat.eloss
        old_q = self.mat.q
        n_steps = max(2, int(math.ceil((omega_max - omega_min) / de)) + 1)
        eloss = np.linspace(omega_min, omega_max, n_steps)

        try:
            self.mat.eloss = eloss

            # Shinotsuka Eq. (7)-(8): every kinematic factor for a
            # nonconductor uses T_prime = T - E_g.
            t_au = t_prime / h2ev
            loss_au = eloss / h2ev
            final_au = t_au - loss_au
            if np.any(final_au <= 0.0):
                raise InputError(
                    "Invalid inelastic domain: T_prime - omega must be positive"
                )

            rel_coef = ((1.0 + t_au / c**2) ** 2) / (
                1.0 + t_au / (2.0 * c**2)
            )
            k_initial = np.sqrt(t_au * (2.0 + t_au / c**2))
            k_final = np.sqrt(final_au * (2.0 + final_au / c**2))
            q_minus = k_initial - k_final
            q_plus = k_initial + k_final

            is_optical = (
                self.mat.oscillators.alpha == 0
                and self.mat.oscillators.model not in ("Mermin", "MLL")
                and getattr(self.mat, "q_dependency", None) is None
            )

            if is_optical:
                self.mat.extend_to_henke()
                log_window = np.log(q_plus / q_minus)
                optical_elf = np.interp(
                    eloss,
                    self.mat.eloss_extended_to_henke,
                    self.mat.elf_extended_to_henke,
                )
                optical_elf = np.nan_to_num(
                    optical_elf, nan=0.0, posinf=0.0, neginf=0.0
                )
                optical_elf[optical_elf < 0.0] = 0.0
                iimfp = (
                    rel_coef
                    * (1.0 / (math.pi * t_au))
                    * optical_elf
                    * log_window
                )
            else:
                q_log = np.linspace(
                    np.log(q_minus), np.log(q_plus), nq, axis=1
                )
                self.mat.q = np.exp(q_log) / a0
                if self.mat.oscillators.model in ("Mermin", "MLL"):
                    self.mat.q[self.mat.q == 0.0] = 0.01

                epsilon = DielectricFunction(self.mat).calculate()
                elf = np.imag(-1.0 / epsilon)
                elf = np.nan_to_num(elf, nan=0.0, posinf=0.0, neginf=0.0)
                elf[elf < 0.0] = 0.0
                if self.mat.oscillators.model in ("Mermin", "MLL"):
                    elf[self.mat.q == 0.01] = 0.0

                iimfp = (
                    rel_coef
                    * (1.0 / (math.pi * t_au))
                    * _trapz(elf, q_log, axis=1)
                )

            iimfp = np.nan_to_num(iimfp, nan=0.0, posinf=0.0, neginf=0.0)
            iimfp[iimfp < 0.0] = 0.0
            raw_diimfp = iimfp / (h2ev * a0)

            diimfp = raw_diimfp.copy()
            if normalised:
                area = float(_trapz(diimfp, eloss))
                if area > 0.0 and math.isfinite(area):
                    diimfp /= area

            self.mat.diimfp = diimfp
            self.mat.iimfp = iimfp
            self.mat.diimfp_e = eloss
            self.mat.e0 = float(E_eV)
            return diimfp
        finally:
            self.mat.eloss = old_eloss
            self.mat.q = old_q

    def calculate_imfp(self, energy_array, de=0.5, nq=100):
        """Calculate physical IMFPs in Angstrom on a VB-bottom energy grid."""
        energies = np.asarray(energy_array, dtype=float)
        if energies.size == 0:
            raise InputError("energy_array must not be empty")
        imfp = np.full(energies.shape, np.inf, dtype=float)

        # Validate the material once, including aliases shared with transport.
        self._energy_domain(float(np.max(energies)))

        for index, energy in np.ndenumerate(energies):
            diimfp = self.calculate_diimfp(
                float(energy), de=de, nq=nq, normalised=False
            )
            inverse_angstrom = float(_trapz(diimfp, self.mat.diimfp_e))
            if inverse_angstrom > 0.0 and math.isfinite(inverse_angstrom):
                imfp[index] = 1.0 / inverse_angstrom

        self.mat.imfp = imfp
        self.mat.imfp_e = energies.copy()
        return self.mat.imfp
