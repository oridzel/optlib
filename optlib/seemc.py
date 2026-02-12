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
    print('_run_one_trajectory_worker')
    print(sample.Ui)
    print(E_s0)

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

            if made_inelastic:
                se_energy = e.energy_loss + e.energy_se
                print('main loop')
                print(e.Ui)

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
        if self.uvw[2] < 0.0 and abs(self.uvw[2]) > 1e-15:
            s_to_surface = -self.xyz[2] / self.uvw[2]
            if 0.0 <= s_to_surface < s:
                s = s_to_surface

        self.path_length += s
        self.xyz[0] += self.uvw[0] * s
        self.xyz[1] += self.uvw[1] * s
        self.xyz[2] += self.uvw[2] * s

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
            if E_vac <= 0:
                self.dead = True
                return False
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

        ind = self.sample.energy_index(self.energy)
        eloss_grid, cdf = self.sample.get_inelastic_eloss_cdf(ind)
        if cdf is None:
            return False

        u = self.rng.random()
        self.energy_loss = float(np.interp(u, cdf, eloss_grid))

        # apply loss
        self.energy -= self.energy_loss
        self.is_dead()
        if self.dead:
            return True  # inelastic happened but primary died

        # DOS sampling
        # self.energy_se = self.sample.e_fermi
        self.feg_dos()

        ang = self.sample.angular_iimfp(self.energy + self.energy_loss, self.energy_loss)
        # w = np.nan_to_num(ang, nan=0.0, posinf=0.0, neginf=0.0) * self.sample._sin_theta_i
        w = np.nan_to_num(ang, nan=0.0) * self.sample._sin_theta_i
        
        cdf2 = cumtrapz_numpy(w, self.sample._theta_i)
        tot = float(cdf2[-1])
        if tot > 0.0 and np.isfinite(tot):
            cdf2 /= tot
            self.deflection[0] = float(np.interp(self.rng.random(), cdf2, self.sample._theta_i))
        else:
            arg = self.energy_loss / max(self.energy + self.energy_loss, 1e-12)
            arg = min(1.0, max(0.0, arg))
            self.deflection[0] = math.asin(math.sqrt(arg))
            
        self.uvw = self.change_direction(self.uvw, self.deflection)
        self.is_dead()
        if self.dead:
            return (self.scattering_type == 1)

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
        # only call when at/above surface crossing
        if self.xyz[2] > 0.0:
            return False
    
        Ui = self.Ui
        Es = self.energy
        ux, uy, uz = self.uvw
    
        # If total energy is below barrier, it can never escape (step-barrier model)
        if Es <= Ui:
            self.dead = True
            return False
    
        # perpendicular energy condition for having a propagating solution in vacuum
        Eperp = Es * (uz * uz)
        if Eperp <= Ui:
            # reflect back into solid (diffuse)
            self._diffuse_reflect_into_solid()
            return False
    
        # quantum transmission probability for step (depends on Eperp)
        root = math.sqrt(1.0 - Ui / Eperp)
        t = 4.0 * root / ((1.0 + root) ** 2)
    
        if self.rng.random() >= t:
            # quantum reflection even though classically allowed -> reflect
            self._diffuse_reflect_into_solid()
            return False
    
        # Transmit into vacuum
        Ev = Es - Ui
        if Ev <= 0.0:
            self.dead = True
            return False
    
        # conserve parallel momentum -> check for total internal reflection
        Epar = Es * (ux*ux + uy*uy)
        if Ev <= Epar:
            # cannot satisfy real uz_out -> reflect instead (no clamping)
            self._diffuse_reflect_into_solid()
            return False
    
        s = math.sqrt(Es / Ev)
        ux_out = ux * s
        uy_out = uy * s
        uz_out = -math.sqrt(1.0 - (ux_out*ux_out + uy_out*uy_out))
    
        self.inside = False
        self.uvw = [ux_out, uy_out, uz_out]
        self.energy_vac = Ev
        self.energy = Ev
        self.xyz[2] = 0.0
        return True
    
    
    def _diffuse_reflect_into_solid(self):
        # Lambertian into solid half-space (uz > 0)
        u = self.rng.random()
        v = self.rng.random()
        cos_t = math.sqrt(u)
        sin_t = math.sqrt(max(1.0 - cos_t*cos_t, 0.0))
        phi = 2.0 * math.pi * v
        self.uvw[0] = sin_t * math.cos(phi)
        self.uvw[1] = sin_t * math.sin(phi)
        self.uvw[2] = cos_t
    
        # push back a small distance (same length unit as mfp)
        push = 1e-3 * min(self.sample.get_imfp(self.energy), self.sample.get_emfp(self.energy))
        self.xyz[2] = max(push, 1e-6)
    
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


class SEEMC:
    def __init__(self, energy_array, sample_name, angle, n_traj, cb_ref=False, track=False, db_path='MaterialDatabase.pkl'):
        self.energy_array = np.asarray(energy_array, dtype=float)
        self.sample = Sample(sample_name, db_path=db_path)
        self.n_trajectories = int(n_traj)
        self.cb_ref = cb_ref
        self.track_trajectories = track
        self.incident_angle = float(angle)

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

                if made_inelastic:
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
            initargs=(self.sample.name, "MaterialDatabase.pkl", self.incident_angle, self.cb_ref, self.track_trajectories),
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

