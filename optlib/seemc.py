import numpy as np
from optlib.constants import *
import pickle
from scipy import integrate
from scipy.interpolate import RectBivariateSpline
import random
import matplotlib.pyplot as plt
import time
import plotly.graph_objects as go
from tqdm import tqdm
import multiprocessing
from types import SimpleNamespace
import gc
import math
import os
from types import SimpleNamespace

_G = None  # worker-global namespace

def cumtrapz_numpy(y, x):
    """Cumulative trapezoid integral, same length as x (initial=0)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    area = 0.5 * (y[1:] + y[:-1]) * dx
    return np.concatenate(([0.0], np.cumsum(area)))


def _init_worker(sample_name, db_path, incident_angle, cb_ref, track_trajectories):
    """
    Runs once per worker process. Keeps heavy objects (Sample tables, splines, etc.)
    local to the process so nothing big gets pickled per task.
    """
    global _G
    sample = Sample(sample_name, db_path=db_path)

    _G = SimpleNamespace(
        sample=sample,
        incident_angle=float(incident_angle),
        cb_ref=bool(cb_ref),
        track=bool(track_trajectories),
    )


def _run_one_trajectory_worker(args):
    """
    Worker task: run a single trajectory for a given primary energy.
    args = (E0, traj_id, seed_base)
    Returns: (tey, sey, bse, tracks_or_None)
    """
    global _G
    E0, traj_id, seed_base = args

    # robust per-task seed (also unique across processes)
    # seed_base should already encode energy index / energy value
    seed = (seed_base + 1_000_003 * (os.getpid() & 0xFFFF) + int(traj_id)) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)

    sample = _G.sample

    tey = 0
    sey = 0
    bse = 0

    electrons = []
    E_s0 = float(E0) + sample.Ui

    electrons.append(
        Electron(
            sample, E_s0, _G.cb_ref, _G.track,
            xyz=[0.0, 0.0, 0.0],
            uvw=[math.sin(_G.incident_angle), 0.0, math.cos(_G.incident_angle)],
            gen=0, se=False, ind=-1, rng=rng
        )
    )

    traj_tracks = []

    i = 0
    while i < len(electrons):
        e = electrons[i]

        while e.inside and (not e.dead):

            e.travel()
            if e.dead:
                break

            if e.escape():
                tey += 1
                if e.is_secondary:
                    sey += 1
                else:
                    bse += 1
                break

            e.get_scattering_type()
            if e.dead:
                break

            made_inelastic = e.scatter()

            if made_inelastic and e.energy_se > 0.0:
                se_energy = e.energy_loss + e.energy_se

                if sample.is_metal and se_energy <= e.Ui:
                    pass
                else:
                    se_defl = [math.pi - e.deflection[0],
                               (e.deflection[1] + math.pi) % (2 * math.pi)]
                    se_uvw = e.change_direction(e.uvw, se_defl)
                    se_xyz = e.xyz.copy()
                    electrons.append(
                        Electron(
                            sample, se_energy, _G.cb_ref, _G.track,
                            xyz=se_xyz, uvw=se_uvw, gen=e.generation + 1, se=True, ind=i, rng=rng
                        )
                    )

        if _G.track:
            traj_tracks.append(e.coordinates)

        electrons[i] = None
        i += 1

    return tey, sey, bse, (traj_tracks if _G.track else None)
    

class Sample:
    T = 300  # K
    k_B = 8.617e-5  # eV/K

    def __init__(self, name, db_path='MaterialDatabase.pkl'):
        import pickle
        with open(db_path, 'rb') as fp:
            data = pickle.load(fp)

        # Support dict (single material) or list[dict]
        if isinstance(data, dict):
            if data.get('name') != name:
                raise ValueError(f"DB has '{data.get('name')}', requested '{name}'")
            self.material_data = data
        elif isinstance(data, list):
            names = [d.get('name') for d in data]
            if name not in names:
                raise ValueError(f"Allowed sample names are {names}")
            self.material_data = next(d for d in data if d.get('name') == name)
        else:
            raise ValueError("Unrecognized MaterialDatabase.pkl format")

        self.name = self.material_data['name']
        self.is_metal = bool(self.material_data['is_metal'])

        # Energy grid for tables
        self.Egrid = np.asarray(self.material_data['energy'], dtype=float)
        self.Emin = float(self.Egrid[0])
        self.Emax = float(self.Egrid[-1])

        # Elastic clamp (you built elastic below 5 eV by using ~5 eV)
        self.elastic_min_energy = 5.0

        # Caches
        self._elastic_theta_cdf_cache = {}   # ind -> (theta, cdf)
        self._inelastic_eloss_cdf_cache = {} # ind -> (eloss, cdf)
        self._elf_spline = None              # RectBivariateSpline for ELF
        self._dos_cdf_cache = None           # (ener_grid, cdf) for DOS sampling
        self._precompute_inelastic_cdfs()
        self._precompute_elastic_cdfs()
        self._theta_i = np.linspace(0.0, math.pi/2.0, 180)
        self._sin_theta_i = np.sin(self._theta_i)
        self.e_fermi = float(self.material_data.get("e_fermi", 0.0))
        self.work_function = float(self.material_data.get("work_function", 0.0))
        self.Ui = self.e_fermi + self.work_function
        
        # --- Channel-resolved inelastic sampler (uses diimfp_pl/se(q,omega)) ---
        # self._inel_sampler = InelasticChannelSampler(self.material_data)

        # ---Precompute ω-CDFs per energy bin per channel (fast) ---
        self._precompute_inelastic_channel_cdfs()


    # ---------- safe interpolation helpers ----------
    def _clip_E(self, E):
        return float(np.clip(E, self.Emin, self.Emax))

    def get_imfp(self, E):
        E = self._clip_E(E)
        return float(np.interp(E, self.Egrid, self.material_data['imfp']))

    # def get_emfp(self, E):
    #     # apply elastic clamp before querying elastic tables
    #     E = max(E, self.elastic_min_energy)
    #     E = self._clip_E(E)
    #     return float(np.interp(E, self.Egrid, self.material_data['emfp']))

    def get_emfp(self, E_solid):
        # Convert solid energy (VB bottom ref) -> vacuum kinetic energy
        E_vac = E_solid - self.Ui   
        E_vac = max(E_vac, self.elastic_min_energy)    
        E_vac = self._clip_E(E_vac)
        return float(np.interp(E_vac, self.Egrid, self.material_data["emfp"]))

    # ---------- fast binning ----------
    def energy_index(self, E):
        """Nearest-neighbor bin using searchsorted (fast, stable)."""
        E = self._clip_E(E)
        i = int(np.searchsorted(self.Egrid, E, side='left'))
        if i <= 0:
            return 0
        if i >= len(self.Egrid):
            return len(self.Egrid) - 1
        # pick closer of i-1 and i
        return i if (self.Egrid[i] - E) < (E - self.Egrid[i-1]) else (i - 1)

    # ---------- CDF caches ----------
    def get_elastic_theta_cdf(self, ind):
        ind = int(ind)
        return self._elastic_theta, self._elastic_cdf_all[:, ind]


    def get_inelastic_eloss_cdf(self, ind):
        ind = int(ind)
        if not self._inel_has[ind]:
            # still return an eloss grid for shape consistency
            return self._inel_eloss_all[:, ind], None
        return self._inel_eloss_all[:, ind], self._inel_cdf_all[:, ind]


    # ---------- ELF spline (build once) ----------
    def elf_spline(self):
        if self._elf_spline is None:
            omega_h = np.asarray(self.material_data['omega'], dtype=float) / h2ev
            q_a0    = np.asarray(self.material_data['q'], dtype=float) * a0
            elf     = np.asarray(self.material_data['elf'], dtype=float)
            self._elf_spline = RectBivariateSpline(omega_h, q_a0, elf)
        return self._elf_spline

    # ---------- inelastic angular distribution ----------
    def angular_iimfp(self, E, dE):
        E_h  = E / h2ev
        dE_h = dE / h2ev
        if dE_h <= 0 or E_h <= dE_h:
            return np.zeros_like(self._theta_i)

        mu = 1.0 - dE_h / E_h
        if mu <= 0.0:
            return np.zeros_like(self._theta_i)
    
        q2 = 4*E_h - 2*dE_h - 4*np.sqrt(E_h*(E_h-dE_h))*np.cos(self._theta_i)
        q2 = np.maximum(q2, 1e-12)
    
        sqrtq = np.sqrt(q2)
    
        f_rbs = self.elf_spline()
        elf_vals = np.asarray(f_rbs(dE_h, sqrtq, grid=False)).reshape(-1)
    
        ang = (1.0 / (math.pi**2 * q2)) * np.sqrt(max(1.0 - dE_h / E_h, 0.0)) * elf_vals
        return np.nan_to_num(ang, nan=0.0, posinf=0.0, neginf=0.0)


    # ---------- DOS sampling cache (for metals) ----------
    def dos_cdf(self):
        """
        Cache the DOS sampling CDF you were building in feg_dos().
        In your model it depends on e_fermi and current energy_loss, but you used:
        dist = sqrt(ener*(ener+energy_loss)).
        We can precompute ener grid; the energy_loss dependence remains,
        but we can still cache the ener grid and reuse integration logic cheaply.
        """
        if self._dos_cdf_cache is None:
            if self.is_metal:
                e_ref = self.e_fermi
            else:
                e_ref = float(self.material_data.get('e_vb', 0.0))
            e_ref = max(e_ref, 1e-6)
            ener = np.linspace(0.0, e_ref, 400)
            self._dos_cdf_cache = ener
        return self._dos_cdf_cache


    def _precompute_inelastic_channel_cdfs(self):
        """
        Precompute CDFs for diimfp_se and diimfp_pl over omega for all energy bins.
        Expects:
          diimfp_se, diimfp_pl: (Nw,2,NE) where [:,0,:]=omega grid, [:,1,:]=pdf
        """
        def build(di_key):
            di = np.asarray(self.material_data[di_key], float)  # (Nw,2,NE)
            eloss = di[:, 0, :]   # (Nw,NE)
            pdf   = np.nan_to_num(di[:, 1, :], nan=0.0, posinf=0.0, neginf=0.0)
    
            dx = np.diff(eloss, axis=0)
            area = 0.5 * (pdf[1:, :] + pdf[:-1, :]) * dx
            cdf = np.vstack([np.zeros((1, area.shape[1])), np.cumsum(area, axis=0)])
    
            total = cdf[-1, :]
            has = (total > 0) & np.isfinite(total)
    
            cdf_norm = np.zeros_like(cdf)
            cdf_norm[:, has] = cdf[:, has] / total[has]
            cdf_norm[:, ~has] = 0.0
            return eloss, cdf_norm, has
    
        self._inel_eloss_se, self._inel_cdf_se, self._inel_has_se = build("diimfp_se")
        self._inel_eloss_pl, self._inel_cdf_pl, self._inel_has_pl = build("diimfp_pl")


    def _precompute_elastic_cdfs(self):
        theta = np.asarray(self.material_data['decs_theta'], dtype=float)
        decs = np.asarray(self.material_data['decs'], dtype=float)  # (Ntheta, Nenergy)
    
        if theta.ndim != 1 or decs.ndim != 2 or decs.shape[0] != theta.size:
            raise ValueError(f"Bad elastic shapes: theta {theta.shape}, decs {decs.shape}")
        if not np.all(np.diff(theta) > 0):
            raise ValueError("decs_theta must be strictly increasing")
    
        pdf = 2*np.pi * decs * np.sin(theta)[:, None]
        pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
    
        dx = np.diff(theta)[:, None]                    # (Ntheta-1, 1)
        area = 0.5 * (pdf[1:, :] + pdf[:-1, :]) * dx    # (Ntheta-1, Nenergy)
        cdf = np.vstack([np.zeros((1, pdf.shape[1])), np.cumsum(area, axis=0)])  # (Ntheta, Nenergy)
    
        total = cdf[-1, :]
        # avoid divide-by-zero
        good = (total > 0) & np.isfinite(total)
        cdf[:, good] /= total[good]
        cdf[:, ~good] = np.linspace(0.0, 1.0, theta.size)[:, None]
    
        self._elastic_theta = theta
        self._elastic_cdf_all = cdf

    def _precompute_inelastic_cdfs(self):
        """
        Precompute CDFs for diimfp over energy loss for all energy bins.
        Stores:
          self._inel_eloss_all: (Neloss, Nenergy)
          self._inel_cdf_all:   (Neloss, Nenergy) normalized where valid
          self._inel_has:       (Nenergy,) boolean where integral>0
        """
        di = np.asarray(self.material_data['diimfp'], dtype=float)  # (Neloss, 2, Nenergy)
        if di.ndim != 3 or di.shape[1] != 2:
            raise ValueError(f"Expected diimfp shape (Neloss,2,Nenergy), got {di.shape}")
    
        eloss = di[:, 0, :]   # (Neloss, Nenergy)
        pdf   = di[:, 1, :]   # (Neloss, Nenergy)
    
        # Replace bad values
        pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
    
        # We assume eloss is nondecreasing in axis 0 for each energy bin (true in your DB build).
        # Compute trapezoid cumulative sum along axis 0 for each column.
        dx = np.diff(eloss, axis=0)                             # (Neloss-1, Nenergy)
        area = 0.5 * (pdf[1:, :] + pdf[:-1, :]) * dx           # (Neloss-1, Nenergy)
    
        cdf = np.vstack([np.zeros((1, area.shape[1])), np.cumsum(area, axis=0)])  # (Neloss, Nenergy)
    
        total = cdf[-1, :]                                      # (Nenergy,)
        has = (total > 0) & np.isfinite(total)
    
        # Normalize only where valid
        cdf_norm = np.zeros_like(cdf)
        cdf_norm[:, has] = cdf[:, has] / total[has]
        # For invalid bins, leave zeros (we'll treat as "no inelastic")
        cdf_norm[:, ~has] = 0.0
    
        self._inel_eloss_all = eloss
        self._inel_cdf_all = cdf_norm
        self._inel_has = has

    def inv_imfp_pl(self, E):
        E = self._clip_E(E)
        return float(np.interp(E, self.Egrid, self.material_data["inv_imfp_pl"]))

    def inv_imfp_se(self, E):
        E = self._clip_E(E)
        return float(np.interp(E, self.Egrid, self.material_data["inv_imfp_se"]))

    # def _precompute_inelastic_channel_cdfs(self):
    #     """
    #     Precompute per-energy-bin CDFs for omega for each channel.
    #     This makes sampling omega fast; q is still conditional on omega per event.
    #     """
    #     q = np.asarray(self.material_data["q"], float)
    #     w = np.asarray(self.material_data["omega"], float)

    #     nq, nw = q.size, w.size
    #     di_pl = _as_q_by_w(self.material_data["diimfp_pl"], nq, nw)
    #     di_se = _as_q_by_w(self.material_data["diimfp_se"], nq, nw)

    #     # If your diimfp_{pl,se} are energy-dependent, then this needs to be 3D.
    #     # From your DB description they are 2D maps (q,omega) shared across E.
    #     # So we just store CDFs once (global). If you later add E-dependence, refactor.
    #     dq = np.diff(q)
    #     dlam_dw_pl = np.sum(0.5 * (di_pl[1:, :] + di_pl[:-1, :]) * dq[:, None], axis=0)
    #     dlam_dw_se = np.sum(0.5 * (di_se[1:, :] + di_se[:-1, :]) * dq[:, None], axis=0)

    #     cdf_pl = _cdf_from_pdf(dlam_dw_pl, w)
    #     cdf_se = _cdf_from_pdf(dlam_dw_se, w)

    #     self._inel_w = w
    #     self._inel_dlam_dw_pl = dlam_dw_pl
    #     self._inel_dlam_dw_se = dlam_dw_se
    #     self._inel_cdf_w_pl = cdf_pl
    #     self._inel_cdf_w_se = cdf_se

    # def sample_inelastic_channel_w_q(self, E, rng):
    #     """
    #     Sample channel, omega, q using:
    #       - channel weights from inv_imfp_pl/se(E)
    #       - omega from marginal ∫dq diimfp_ch(q,omega), truncated by omega<=Eeff
    #       - q from conditional diimfp_ch(q|omega), restricted to projectile kinematic [q-,q+]
    #     Returns (ch, omega, q) or None.
    #     """
    #     # metal guard (same as Electron.iimfp)
    #     if self.is_metal and E <= self.e_fermi:
    #         return None

    #     # Use Eeff if you enforce omega <= E - EF for metals (recommended):
    #     Eeff = E - self.e_fermi if self.is_metal else E
    #     if Eeff <= 0.0:
    #         return None

    #     inv_pl = self.inv_imfp_pl(E)
    #     inv_se = self.inv_imfp_se(E)
    #     s = inv_pl + inv_se
    #     if not np.isfinite(s) or s <= 0.0:
    #         return None

    #     ch = "pl" if (rng.random() < inv_pl / s) else "se"

    #     # omega sampling from precomputed CDF, truncated at Eeff
    #     w = self._inel_w
    #     iwmax = int(np.searchsorted(w, Eeff, side="right") - 1)
    #     iwmax = int(np.clip(iwmax, 0, w.size - 1))
    #     if iwmax < 1:
    #         return None

    #     if ch == "pl":
    #         cdf_full = self._inel_cdf_w_pl
    #     else:
    #         cdf_full = self._inel_cdf_w_se
    #     if cdf_full is None:
    #         return None

    #     # Renormalize truncated CDF [0..iwmax] by linear scaling
    #     cdf_tr = cdf_full[:iwmax + 1]
    #     cmax = float(cdf_tr[-1])
    #     if cmax <= 0.0 or not np.isfinite(cmax):
    #         return None

    #     u = rng.random() * cmax
    #     omega = _sample_from_cdf(cdf_tr, w[:iwmax + 1], u / cmax)

    #     # q conditional on omega using your InelasticChannelSampler (no reject)
    #     # (it already handles kinematic q-bounds)
    #     res = self._inel_sampler.sample(Eeff, rng=rng)
    #     if res is None:
    #         return None
    #     ch2, omega2, q = res

    #     # ensure consistency: force channel we chose, but reuse omega from the same sample
    #     # simplest: just trust sampler’s own channel/omega sampling:
    #     return ch2, omega2, q

    def sample_inelastic_channel_w_q(self, E_solid_eV, rng):
        """
        Option A sampler:
          1) choose channel using inv_imfp_pl/se(E)
          2) sample omega from diimfp_pl/se CDF at that energy bin
          3) sample q from ELF_pl/se(omega,q) within kinematic bounds
    
        Returns (ch, omega_eV, q_a0inv) or None.
        """
        # metal guard (your model)
        if self.is_metal and E_solid_eV <= self.e_fermi:
            return None
    
        # energy bin for ω tables and inv_imfp arrays
        ind = self.energy_index(E_solid_eV)
    
        inv_pl = float(self.material_data["inv_imfp_pl"][ind])
        inv_se = float(self.material_data["inv_imfp_se"][ind])
        s = inv_pl + inv_se
        if s <= 0.0 or not np.isfinite(s):
            return None
        ch = "pl" if (rng.random() < inv_pl / s) else "se"
    
        # ω sampling (and enforce ω <= E - EF for metals)
        Eeff = E_solid_eV - self.e_fermi if self.is_metal else E_solid_eV
        if Eeff <= 0.0:
            return None
    
        if ch == "se":
            if not self._inel_has_se[ind]:
                return None
            wgrid = self._inel_eloss_se[:, ind]
            cdf   = self._inel_cdf_se[:, ind]
        else:
            if not self._inel_has_pl[ind]:
                return None
            wgrid = self._inel_eloss_pl[:, ind]
            cdf   = self._inel_cdf_pl[:, ind]
    
        # sample ω, but reject/clip beyond Eeff (simple and safe)
        omega = float(np.interp(rng.random(), cdf, wgrid))
        if omega <= 0.0 or omega >= Eeff:
            return None
    
        # q sampling from ELF at this omega (still need E for q-bounds)
        q = self.sample_q_from_elf(ch, Eeff, omega, rng)
        if q is None:
            return None
    
        return ch, omega, q

    def sample_target_k_FEG_disk_au(self, omega_eV, q_a0inv, rng):
        """
        Disk/annulus sampling in atomic units.
        Returns k_vec (a0^-1) or None.
        """
        if not self.is_metal:
            return None
        if q_a0inv <= 0.0:
            return None
    
        omega_h = omega_eV / h2ev
        ef_h    = self.e_fermi / h2ev
    
        kF = math.sqrt(max(2.0 * ef_h, 0.0))  # since E = k^2/2 (Hartree)
    
        q = float(q_a0inv)
    
        # from ω = (q^2 + 2 kz q)/2  -> kz = (2ω - q^2)/(2q)
        kz = (2.0 * omega_h - q*q) / (2.0 * q)
    
        r_out_sq = kF*kF - kz*kz
        if r_out_sq <= 0.0:
            return None
        r_out = math.sqrt(r_out_sq)
    
        # Pauli blocking: |k+q| >= kF  -> k_perp^2 + (kz+q)^2 >= kF^2
        r_in_sq = kF*kF - (kz + q)*(kz + q)
        if r_in_sq > 0.0:
            r_in = math.sqrt(r_in_sq)
            if r_in >= r_out:
                return None
        else:
            r_in = 0.0
    
        u = rng.random()
        r = math.sqrt(r_in*r_in + u*(r_out*r_out - r_in*r_in))
        phi = 2.0 * math.pi * rng.random()
    
        kx = r * math.cos(phi)
        ky = r * math.sin(phi)
        return np.array([kx, ky, kz], float)

    def sample_target_k_FEG(self, omega, q, rng):
        """
        Sample initial target electron k-vector inside Fermi sphere
        consistent with (omega, q) for single-electron excitation.
    
        Returns k_vec (3-array) or None if no allowed state.
        Units:
            q in Å^-1
            omega in eV
        """
    
        alpha = HBAR2_2M_eVA2  # ħ²/2m in eV·Å²
    
        # Fermi wavevector
        if not self.is_metal:
            return None  # only defined for metal FEG
    
        kF = math.sqrt(max(self.e_fermi, 0.0) / alpha)
    
        if q <= 0.0:
            return None
    
        # k_z plane from energy-momentum constraint:
        # k_z = (omega/alpha - q^2) / (2q)
        kz = (omega / alpha - q*q) / (2.0*q)
    
        # must lie inside Fermi sphere
        r_out_sq = kF*kF - kz*kz
        if r_out_sq <= 0.0:
            return None
    
        r_out = math.sqrt(r_out_sq)
    
        # Pauli blocking: final state k+q must lie outside Fermi sphere
        # |k+q|^2 = k_perp^2 + (kz+q)^2
        r_in_sq = kF*kF - (kz + q)*(kz + q)
    
        if r_in_sq > 0.0:
            r_in = math.sqrt(r_in_sq)
            if r_in >= r_out:
                return None  # fully blocked
        else:
            r_in = 0.0
    
        # Sample uniformly in disk or annulus
        u = rng.random()
        r = math.sqrt(r_in*r_in + u*(r_out*r_out - r_in*r_in))
        phi = 2.0 * math.pi * rng.random()
    
        kx = r * math.cos(phi)
        ky = r * math.sin(phi)
    
        return np.array([kx, ky, kz])

    def elf_channel_splines(self):
        """
        Build RectBivariateSpline on (omega_hartree, log(q_a0inv)) for elf_se and elf_pl.
        Assumes:
          material_data['omega'] in eV
          material_data['q'] in a0^-1
          material_data['elf_se'], ['elf_pl'] shaped (Nw,Nq)
        """
        if getattr(self, "_elf_se_spl", None) is None:
            omega_h = np.asarray(self.material_data["omega"], float) / h2ev  # Hartree
            q_a0inv = np.asarray(self.material_data["q"], float)            # a0^-1
            qlog = np.log(q_a0inv)
    
            elf_se = np.asarray(self.material_data["elf_se"], float)
            elf_pl = np.asarray(self.material_data["elf_pl"], float)
    
            # ensure (Nw,Nq)
            if elf_se.shape != (omega_h.size, qlog.size):
                if elf_se.shape == (qlog.size, omega_h.size):
                    elf_se = elf_se.T
                    elf_pl = elf_pl.T
                else:
                    raise ValueError(f"ELF shape mismatch: elf_se {elf_se.shape}, expected (Nw,Nq)")
    
            self._elf_se_spl = RectBivariateSpline(omega_h, qlog, elf_se, kx=1, ky=1)
            self._elf_pl_spl = RectBivariateSpline(omega_h, qlog, elf_pl, kx=1, ky=1)
    
            self._qlog_grid = qlog
    
        return self._elf_se_spl, self._elf_pl_spl

    def _qlog_bounds_rel(self, E_eV, omega_eV):
        """
        Relativistic q bounds in atomic units (a0^-1), returned as (qm_log, qp_log).
        Matches your database-build formulas.
        """
        c = 137.036
        e0 = E_eV / h2ev
        om = omega_eV / h2ev
        emw = max(e0 - om, 0.0)
        qm = np.log(np.sqrt(e0*(2 + e0/c**2)) - np.sqrt(emw*(2 + emw/c**2)))
        qp = np.log(np.sqrt(e0*(2 + e0/c**2)) + np.sqrt(emw*(2 + emw/c**2)))
        return qm, qp
    
    def sample_q_from_elf(self, ch, E_eV, omega_eV, rng):
        """
        Sample q (a0^-1) from ELF_ch(omega,q) over allowed projectile bounds [q-, q+].
        Sampling is done in log(q), consistent with how DIIMFP was integrated (d log q).
        """
        se_spl, pl_spl = self.elf_channel_splines()
        spl = se_spl if ch == "se" else pl_spl
    
        # bounds in log(q)
        qm, qp = self._qlog_bounds_rel(E_eV, omega_eV)
        if not np.isfinite(qm) or not np.isfinite(qp) or qp <= qm:
            return None
    
        qlog_grid = self._qlog_grid
        i0 = int(np.searchsorted(qlog_grid, qm, side="left"))
        i1 = int(np.searchsorted(qlog_grid, qp, side="right") - 1)
        i0 = max(i0, 0)
        i1 = min(i1, qlog_grid.size - 1)
        if i1 <= i0:
            return None
    
        qlog = qlog_grid[i0:i1+1]
    
        # evaluate ELF at this omega on the qlog slice
        omega_h = omega_eV / h2ev
        w = np.asarray(spl.ev(np.full_like(qlog, omega_h), qlog), float)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w[w < 0] = 0.0
    
        # build CDF in qlog
        cdf = cumtrapz_numpy(w, qlog)
        tot = float(cdf[-1])
        if tot <= 0.0 or not np.isfinite(tot):
            return None
        cdf /= tot
    
        qlog_s = float(np.interp(rng.random(), cdf, qlog))
        q = float(np.exp(qlog_s))  # a0^-1
        return q


class Electron:
    def __init__(self, sample: Sample, energy, cb_ref, save_coord, xyz, uvw, gen, se, ind, rng):
        self.sample = sample
        self.conduction_band_reference = cb_ref
        self.save_coordinates = save_coord
        self.xyz = [float(x) for x in xyz]
        self.uvw = [float(u) for u in uvw]
        self.generation = int(gen)
        self.is_secondary = bool(se)
        self.parent_index = int(ind)
        self.rng = rng

        # --- ENERGY CONVENTION ---
        # While inside the solid: energy is E_s referenced to valence-band bottom (VB bottom = 0).
        self.energy = float(energy)
        self.initial_energy = self.energy
        self.initial_depth = self.xyz[2]

        self.inside = True
        self.dead = False

        self.scattering_type = -1
        self.energy_loss = 0.0
        self.energy_se = 0.0
        self.n_secondaries = 0
        self.path_length = 0.0
        self.deflection = [0.0, 0.0]

        self.coordinates = []
        if self.save_coordinates:
            self.coordinates.append([round(v, 2) for v in self.xyz + [self.energy]])

        # material params
        self.work_function = float(self.sample.material_data.get('work_function', 0.0))
        self.e_fermi = float(self.sample.material_data.get('e_fermi', 0.0))

        # Full barrier from VB bottom to vacuum level for metals: Ui = Ef + phi
        self.Ui = self.e_fermi + self.work_function
    
        # Optional: store vacuum energy upon emission (None while inside)
        self.energy_vac = None
        self.inel_channel = None

    # --- rates ---
    @property
    def iemfp(self):
        emfp = self.sample.get_emfp(self.energy)
        if (not np.isfinite(emfp)) or emfp <= 0:
            return 0.0
        return 1.0 / emfp

    @property
    def iimfp(self):
        # Metal model: inelastic only if E > E_F
        if self.sample.is_metal and self.energy <= self.e_fermi:
            return 0.0

        imfp = self.sample.get_imfp(self.energy)
        if (not np.isfinite(imfp)) or imfp <= 0:
            return 0.0
        return 1.0 / imfp

    @property
    def itmfp(self):
        return self.iemfp + self.iimfp  # (phonons omitted here; add back later if needed)

    def is_dead(self):
        if (not np.isfinite(self.energy)) or self.energy <= 0.0:
            self.dead = True
            return
    
        if self.inside and self.energy <= self.Ui:
            self.dead = True
            return

    # --- transport ---
    def travel(self):
        rate = self.itmfp
        if (not np.isfinite(rate)) or rate <= 0.0:
            self.dead = True
            return

        s = -math.log(self.rng.random()) / rate

        # stop exactly at surface if would cross (vacuum is z<0)
        hit_surface = False
        if self.uvw[2] < 0.0 and abs(self.uvw[2]) > 1e-15:
            s_to_surface = -self.xyz[2] / self.uvw[2]
            if 0.0 <= s_to_surface < s:
                s = s_to_surface
                hit_surface = True
        
        self.path_length += s
        self.xyz[0] += self.uvw[0] * s
        self.xyz[1] += self.uvw[1] * s
        self.xyz[2] += self.uvw[2] * s
        
        if hit_surface:
            self.xyz[2] = 0.0  # Explicitly snap to surface

        if self.save_coordinates:
            self.coordinates.append([round(v, 2) for v in self.xyz + [self.energy]])

    def get_scattering_type(self):
        total = self.itmfp
        if total <= 0 or (not np.isfinite(total)):
            self.dead = True
            return

        r = self.rng.random()
        pel = self.iemfp / total
        # no phonons here; re-add later if desired
        self.scattering_type = 0 if (r < pel) else 1

    def scatter(self):
        self.deflection[1] = self.rng.random() * 2.0 * math.pi

        if self.scattering_type == 0:
            # elastic: use elastic-clamped energy bin
            # ind = self.sample.energy_index(max(self.energy, self.sample.elastic_min_energy))
            E_vac = self.energy - self.Ui
            ind = self.sample.energy_index(max(E_vac, self.sample.elastic_min_energy))
            theta_grid, cdf = self.sample.get_elastic_theta_cdf(ind)
            
            u = self.rng.random()
            self.deflection[0] = float(np.interp(u, cdf, theta_grid))
            self.uvw = self.change_direction(self.uvw, self.deflection)
            return False  # no SE creation

        # inelastic
        if self.sample.is_metal and self.energy <= self.e_fermi:
            # should not happen because iimfp=0, but stay safe
            return False

        # -------- inelastic (channel-resolved, Option A) --------
        triple = self.sample.sample_inelastic_channel_w_q(self.energy, self.rng)
        if triple is None:
            return False
        
        ch, omega, q_a0inv = triple
        self.inel_channel = ch
        self.energy_loss = float(omega)
        
        E_before = self.energy
        self.energy -= self.energy_loss
        self.is_dead()
        if self.dead:
            return True
        
        # projectile deflection:
        # you currently compute deflection from q in Å^-1; but here q is a0^-1.
        # Convert q(a0^-1) -> q(Å^-1) by dividing by a0(Å):
        q_Ainv = q_a0inv / a0
        
        k  = math.sqrt(max(E_before, 0.0) / HBAR2_2M_eVA2)
        kp = math.sqrt(max(self.energy, 0.0) / HBAR2_2M_eVA2)
        if k > 0 and kp > 0:
            cos_th = (k*k + kp*kp - q_Ainv*q_Ainv) / (2.0*k*kp)
            cos_th = min(1.0, max(-1.0, cos_th))
            self.deflection[0] = math.acos(cos_th)
        else:
            self.deflection[0] = 0.0
        
        self.uvw = self.change_direction(self.uvw, self.deflection)
        
        # secondary energy model:
        if ch == "se":
            k_i = self.sample.sample_target_k_FEG_disk_au(self.energy_loss, q_a0inv, self.rng)
            if k_i is None:
                self.energy_se = 0.0
                return True
        
            k_f = np.array([k_i[0], k_i[1], k_i[2] + q_a0inv])
            Ei_h = 0.5 * float(np.dot(k_i, k_i))
            # Ef_h = 0.5 * float(np.dot(k_f, k_f))  # would equal Ei_h + omega_h
            self.energy_se = Ei_h * h2ev            # VB-bottom referenced initial energy
        else:
            # plasmon: keep your existing phase-space secondary model
            self.feg_dos()
        
        return True

    def feg_dos(self):
        e_ref = self.e_fermi if self.sample.is_metal else float(self.sample.material_data.get('e_vb', 0.0))
        e_ref = max(e_ref, 1e-6)

        ener = self.sample.dos_cdf()
        # energy_loss changes per event, so CDF changes; but we avoid rebuilding ener each time
        dist = np.sqrt(np.maximum(ener * (ener + self.energy_loss), 0.0))
        cdf = cumtrapz_numpy(dist, ener)
        total = float(cdf[-1])
        if total > 0 and np.isfinite(total):
            cdf = cdf / total
            self.energy_se = float(np.interp(self.rng.random(), cdf, ener))
        else:
            self.energy_se = 0.0

    def escape(self):
        # Only call when at/above surface crossing
        # Remember to use the 1e-12 epsilon fix discussed previously!
        if self.xyz[2] > 1e-12:
            return False
    
        Ui = self.Ui
        Es = self.energy
        ux, uy, uz = self.uvw
    
        # If total energy is below barrier, it can never escape (step-barrier model)
        if Es <= Ui:
            self._specular_reflect_into_solid()
            return False
    
        # Perpendicular energy condition for having a propagating solution in vacuum
        Eperp = Es * (uz * uz)
        if Eperp <= Ui:
            self._specular_reflect_into_solid()
            return False
    
        # Quantum transmission probability for step (depends on Eperp)
        root = math.sqrt(1.0 - Ui / Eperp)
        t = 4.0 * root / ((1.0 + root) ** 2)
    
        if self.rng.random() >= t:
            self._specular_reflect_into_solid()
            return False
    
        # Transmit into vacuum
        Ev = Es - Ui
        if Ev <= 0.0:
            self.dead = True
            return False
    
        # Conserve parallel momentum -> check for total internal reflection
        Epar = Es * (ux*ux + uy*uy)
        if Ev <= Epar:
            # Cannot satisfy real uz_out -> reflect specularly
            self._specular_reflect_into_solid()
            return False
    
        # Calculate new trajectory in vacuum (conserving parallel momentum)
        s = math.sqrt(Es / Ev)
        ux_out = ux * s
        uy_out = uy * s
        # Vacuum is z < 0, so uz_out must be negative
        uz_out = -math.sqrt(1.0 - (ux_out*ux_out + uy_out*uy_out))
    
        self.inside = False
        self.uvw = [ux_out, uy_out, uz_out]
        self.energy_vac = Ev
        self.energy = Ev
        self.xyz[2] = 0.0
        return True
    
    
    def _specular_reflect_into_solid(self):
        # Specular reflection: invert only the surface-normal velocity (z-axis)
        # Since the solid is z > 0, the new uz must be positive
        self.uvw[2] = abs(self.uvw[2])
        
        # Push back a tiny distance to avoid being trapped exactly at z=0 due to floating point limits
        self.xyz[2] = 1e-6
        
        if self.save_coordinates:
            self.coordinates.append([round(v, 2) for v in self.xyz + [self.energy]])


    def change_direction(self, uvw, deflection):
        # normalized rotation (your earlier function but stable)
        new_uvw = [0.0, 0.0, 0.0]
        sin_psi = math.sin(deflection[0])
        cos_psi = math.cos(deflection[0])
        sin_fi = math.sin(deflection[1])
        cos_fi = math.cos(deflection[1])

        cos_theta = uvw[2]
        sin_theta = math.sqrt(max(uvw[0]**2 + uvw[1]**2, 0.0))
        if sin_theta > 1e-12:
            cos_phi = uvw[0] / sin_theta
            sin_phi = uvw[1] / sin_theta
        else:
            cos_phi, sin_phi = 1.0, 0.0

        h0 = sin_psi * cos_fi
        h1 = sin_theta * cos_psi + h0 * cos_theta
        h2 = sin_psi * sin_fi

        new_uvw[0] = h1 * cos_phi - h2 * sin_phi
        new_uvw[1] = h1 * sin_phi + h2 * cos_phi
        new_uvw[2] = cos_theta * cos_psi - h0 * sin_theta

        norm = math.sqrt(new_uvw[0]**2 + new_uvw[1]**2 + new_uvw[2]**2)
        if norm > 0:
            new_uvw[0] /= norm
            new_uvw[1] /= norm
            new_uvw[2] /= norm
        return new_uvw

HBAR2_2M_eVA2 = 3.80998212  # ħ²/(2m) in eV·Å²  (electron)

def _as_q_by_w(A, nq, nw):
    """Return A as shape (nq, nw), accepting (nq,nw) or (nw,nq)."""
    A = np.asarray(A, float)
    if A.shape == (nq, nw):
        return A
    if A.shape == (nw, nq):
        return A.T
    raise ValueError(f"Expected diimfp shape (nq,nw) or (nw,nq); got {A.shape}")

def _cdf_from_pdf(pdf, x):
    pdf = np.clip(np.asarray(pdf, float), 0.0, np.inf)
    x = np.asarray(x, float)
    if pdf.size != x.size:
        raise ValueError("pdf and x size mismatch")
    if pdf.size < 2:
        return None
    dx = np.diff(x)
    area = 0.5 * (pdf[1:] + pdf[:-1]) * dx
    cdf = np.concatenate(([0.0], np.cumsum(area)))
    tot = cdf[-1]
    if not np.isfinite(tot) or tot <= 0.0:
        return None
    return cdf / tot

def _sample_from_cdf(cdf, x, u):
    j = np.searchsorted(cdf, u, side="right") - 1
    j = int(np.clip(j, 0, len(x) - 2))
    c0, c1 = cdf[j], cdf[j + 1]
    x0, x1 = x[j], x[j + 1]
    if c1 <= c0:
        return float(x0)
    t = (u - c0) / (c1 - c0)
    return float(x0 + t * (x1 - x0))

def projectile_q_bounds(E, w):
    """
    q bounds from projectile kinematics (free-electron dispersion):
      q_- = |k - k'|, q_+ = k + k'
    with k = sqrt(E / (ħ²/2m)) in Å^-1 when E in eV.
    """
    if w < 0.0:
        return 0.0, 0.0
    Ep = E - w
    if Ep <= 0.0:
        return 0.0, 0.0
    k  = np.sqrt(max(E, 0.0)  / HBAR2_2M_eVA2)
    kp = np.sqrt(max(Ep, 0.0) / HBAR2_2M_eVA2)
    return abs(k - kp), (k + kp)

class InelasticChannelSampler:
    """
    Uses DB keys:
      q, omega
      diimfp_pl, diimfp_se
      inv_imfp_pl(E), inv_imfp_se(E)
    """
    def __init__(self, d):
        self.d = d
        self.q = np.asarray(d["q"], float)
        self.w = np.asarray(d["omega"], float)
        self.nq = self.q.size
        self.nw = self.w.size

        self.di_pl = _as_q_by_w(d["diimfp_pl"], self.nq, self.nw)
        self.di_se = _as_q_by_w(d["diimfp_se"], self.nq, self.nw)

    def sample(self, E, rng=np.random):
        """
        Return (channel, omega, q).
        channel in {"pl","se"}.
        """
        E = float(E)
        if E <= self.w[0]:
            return None

        # ---- 1) channel selection: weights from inv_imfp_* at this E ----
        Egrid = np.asarray(self.d["energy"], float)
        inv_pl_arr = np.asarray(self.d["inv_imfp_pl"], float)
        inv_se_arr = np.asarray(self.d["inv_imfp_se"], float)
        inv_pl = float(np.interp(E, Egrid, inv_pl_arr))
        inv_se = float(np.interp(E, Egrid, inv_se_arr))

        s = inv_pl + inv_se
        if not np.isfinite(s) or s <= 0.0:
            return None

        u = rng.random()
        ch = "pl" if (u < inv_pl / s) else "se"
        di = self.di_pl if ch == "pl" else self.di_se

        # ---- limit omega to <= E ----
        iwmax = int(np.searchsorted(self.w, E, side="right") - 1)
        iwmax = int(np.clip(iwmax, 0, self.nw - 1))
        wgrid = self.w[:iwmax + 1]
        di = di[:, :iwmax + 1]  # (nq, nw_eff)

        # ---- 2) sample omega from marginal dλ/dω = ∫ dq diimfp(q,ω) ----
        dq = np.diff(self.q)
        # trapz over q for each omega column
        dlam_dw = np.sum(0.5 * (di[1:, :] + di[:-1, :]) * dq[:, None], axis=0)  # (nw_eff,)

        cdf_w = _cdf_from_pdf(dlam_dw, wgrid)
        if cdf_w is None:
            return None
        omega = _sample_from_cdf(cdf_w, wgrid, rng.random())

        # ---- 3) sample q conditional on omega, restricted to [q-, q+] ----
        # linear interp between nearest omega bins
        j = int(np.searchsorted(wgrid, omega, side="right") - 1)
        j = int(np.clip(j, 0, wgrid.size - 2))
        w0, w1 = wgrid[j], wgrid[j + 1]
        t = 0.0 if w1 <= w0 else (omega - w0) / (w1 - w0)

        pdf_q = (1.0 - t) * di[:, j] + t * di[:, j + 1]  # (nq,)

        qmin, qmax = projectile_q_bounds(E, omega)
        i0 = int(np.searchsorted(self.q, qmin, side="left"))
        i1 = int(np.searchsorted(self.q, qmax, side="right") - 1)
        i0 = max(i0, 0)
        i1 = min(i1, self.nq - 1)
        if i1 <= i0:
            return None

        qgrid = self.q[i0:i1 + 1]
        pdfs  = pdf_q[i0:i1 + 1]
        cdf_q = _cdf_from_pdf(pdfs, qgrid)
        if cdf_q is None:
            return None
        q = _sample_from_cdf(cdf_q, qgrid, rng.random())

        return ch, omega, q
        

class SEEMC:
    def __init__(self, energy_array, sample_name, angle, n_traj, cb_ref=False, track=False, db_path='MaterialDatabase.pkl'):
        self.energy_array = np.asarray(energy_array, dtype=float)
        self.sample = Sample(sample_name, db_path=db_path)
        self.n_trajectories = int(n_traj)
        self.cb_ref = cb_ref
        self.track_trajectories = track
        self.incident_angle = float(angle)
        self.db_path = db_path

        self.tey = np.zeros(len(self.energy_array))
        self.sey = np.zeros(len(self.energy_array))
        self.bse = np.zeros(len(self.energy_array))

        # only store tracks if requested
        self.tracks = []  # list per energy if track=True

    def run_one_trajectory(self, E0, traj_id):
        seed = (os.getpid() * 1_000_003 + traj_id) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)

        traj_tracks = []
        
        tey = 0
        sey = 0
        bse = 0      
    
        electrons = []    
        E_s0 = float(E0) + self.sample.Ui
        
        xyz=[0.0, 0.0, 0.0]
        electrons.append(Electron(
            self.sample,
            E_s0,   # solid energy (VB-bottom reference)
            self.cb_ref,
            self.track_trajectories,
            xyz=xyz,
            uvw=[math.sin(self.incident_angle), 0, math.cos(self.incident_angle)],
            gen=0,
            se=False,
            ind=-1,
            rng=rng
        ))

        i = 0
        while i < len(electrons):
            e = electrons[i]
    
            while e.inside and (not e.dead):
                    
                e.travel()
                if e.dead:
                    break

                if e.escape():
                    tey += 1
                    if e.is_secondary:
                        sey += 1
                    else:
                        bse += 1
                    break
    
                e.get_scattering_type()
                if e.dead:
                    break
    
                made_inelastic = e.scatter()

                if made_inelastic and e.energy_se > 0.0:
                    se_energy = e.energy_loss + e.energy_se
    
                    # spawn criterion (metal): only above EF for this model
                    if self.sample.is_metal and se_energy <= e.Ui:
                        pass
                    else:
                        se_defl = [math.pi - e.deflection[0], (e.deflection[1] + math.pi) % (2*math.pi)]
                        se_uvw = e.change_direction(e.uvw, se_defl)
                        se_xyz = e.xyz.copy()
                        electrons.append(Electron(
                            self.sample, se_energy, self.cb_ref, self.track_trajectories,
                            xyz=se_xyz, uvw=se_uvw, gen=e.generation + 1, se=True, ind=i, rng=rng
                        ))
    
            if self.track_trajectories:
                traj_tracks.append(e.coordinates)
    
            electrons[i] = None
            i += 1

        return tey, sey, bse, (traj_tracks if self.track_trajectories else None)


    def run_simulation(self, use_parallel=False):
        import time
        import multiprocessing as mp
        from tqdm import tqdm
    
        t0 = time.time()
    
        # Serial path unchanged
        if not use_parallel:
            for k, E0 in enumerate(self.energy_array):
                t_tey = t_sey = t_bse = 0
                tracks_E = [] if self.track_trajectories else None
    
                for traj in tqdm(range(self.n_trajectories), desc=f"E={E0:.1f} eV"):
                    tey, sey, bse, trk = self.run_one_trajectory(E0, traj)
                    t_tey += tey
                    t_sey += sey
                    t_bse += bse

                    if self.track_trajectories:
                        tracks_E.append(trk)
    
                self.tey[k] = t_tey / self.n_trajectories
                self.sey[k] = t_sey / self.n_trajectories
                self.bse[k] = t_bse / self.n_trajectories
                if self.track_trajectories:
                    self.tracks.append(tracks_E)
    
            print(f"Done in {time.time() - t0:.1f} s")
            return
    
        # Parallel path: one Pool for all energies + worker globals
        nproc = mp.cpu_count()
    
        # Good default chunking: bigger chunks reduce overhead a lot
        chunksize = max(1, self.n_trajectories // (nproc * 4))
    
        # IMPORTANT: if you run on Windows, you must keep this under:
        # if __name__ == "__main__":
        ctx = mp.get_context("spawn")  # safest cross-platform; use "fork" on Linux for slightly less overhead
    
        with ctx.Pool(
            processes=nproc,
            initializer=_init_worker,
            initargs=(self.sample.name, self.db_path, self.incident_angle, self.cb_ref, self.track_trajectories),
        ) as pool:
    
            for k, E0 in enumerate(self.energy_array):
                # seed_base encodes energy so different energies don't reuse RNG streams
                seed_base = (k * 1_000_000 + int(round(float(E0) * 10))) & 0xFFFFFFFF
    
                tasks = ((float(E0), traj, seed_base) for traj in range(self.n_trajectories))
    
                t_tey = t_sey = t_bse = 0
                tracks_E = [] if self.track_trajectories else None
    
                total_inelastic = 0
                total_spawned = 0
                max_cascade = 0
                
                for tey, sey, bse, trk in tqdm(
                    pool.imap_unordered(_run_one_trajectory_worker, tasks, chunksize=chunksize),
                    total=self.n_trajectories,
                    desc=f"E={E0:.1f} eV",
                ):
                    t_tey += tey
                    t_sey += sey
                    t_bse += bse
                    if self.track_trajectories:
                        tracks_E.append(trk)
    
                self.tey[k] = t_tey / self.n_trajectories
                self.sey[k] = t_sey / self.n_trajectories
                self.bse[k] = t_bse / self.n_trajectories
                
                if self.track_trajectories:
                    self.tracks.append(tracks_E)
    
        print(f"Done in {time.time() - t0:.1f} s")

    def plot_yield(self):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(self.energy_array, self.tey, label="TEY")
        plt.plot(self.energy_array, self.sey, label="SEY")
        plt.plot(self.energy_array, self.bse, label="BSE")
        plt.xlabel("Energy (eV)")
        plt.ylabel("Yield")
        plt.title(self.sample.name)
        plt.legend()
        plt.show()

