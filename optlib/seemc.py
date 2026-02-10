import numpy as np
from optlib.constants import *
import pickle
from scipy import integrate
from scipy.integrate import cumulative_trapezoid
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

def cumtrapz_numpy(y, x):
    """Cumulative trapezoid integral, same length as x (initial=0)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    area = 0.5 * (y[1:] + y[:-1]) * dx
    return np.concatenate(([0.0], np.cumsum(area)))

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


    # ---------- safe interpolation helpers ----------
    def _clip_E(self, E):
        return float(np.clip(E, self.Emin, self.Emax))

    def get_imfp(self, E):
        E = self._clip_E(E)
        return float(np.interp(E, self.Egrid, self.material_data['imfp']))

    def get_emfp(self, E):
        # apply elastic clamp before querying elastic tables
        E = max(E, self.elastic_min_energy)
        E = self._clip_E(E)
        return float(np.interp(E, self.Egrid, self.material_data['emfp']))

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
            # Expect omega, q, elf in DB
            omega = np.asarray(self.material_data['omega'], dtype=float)
            q = np.asarray(self.material_data['q'], dtype=float)
            elf = np.asarray(self.material_data['elf'], dtype=float)
            # Your code used omega/h2ev and q*a0 elsewhere; keep your conventions there.
            self._elf_spline = RectBivariateSpline(omega, q, elf)
        return self._elf_spline

    # ---------- inelastic angular distribution ----------
    def angular_iimfp(self, E, dE):
        """
        Return theta grid and angular distribution for a given incident kinetic E and loss dE.
        This is your existing formula with better numeric guards.
        """
        theta = np.linspace(0.0, math.pi / 2.0, 180)

        # Convert to Hartree units if your existing code expects it.
        # Keep exactly your previous conversions if needed; I’m leaving these as placeholders:
        # E_h = E / h2ev ; dE_h = dE / h2ev
        E_h = E / h2ev
        dE_h = dE / h2ev

        # guard
        if dE_h <= 0 or E_h <= dE_h:
            return theta, np.zeros_like(theta)

        q2 = 4*E_h - 2*dE_h - 4*np.sqrt(E_h*(E_h-dE_h))*np.cos(theta)
        q2 = np.maximum(q2, 1e-12)

        f_rbs = RectBivariateSpline(self.material_data['omega'] / h2ev,
                                    self.material_data['q'] * a0,
                                    self.material_data['elf'])

        x, y = np.meshgrid(np.array([dE_h]), np.sqrt(q2), indexing="ij")
        elf_vals = np.squeeze(f_rbs(x, y, grid=False))

        ang = (1.0 / (math.pi**2 * q2)) * np.sqrt(max(1.0 - dE_h / E_h, 0.0)) * elf_vals
        ang = np.nan_to_num(ang, nan=0.0, posinf=0.0, neginf=0.0)
        return theta, ang

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
                e_ref = float(self.material_data['e_fermi'])
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

        self.n_escape_calls = 0
        self.n_escape_below_barrier = 0
        self.n_escape_reflected_prob = 0
        self.n_escape_transmit = 0
        self.min_Eperp_over_Ui = float("inf")
        self.max_Eperp_over_Ui = 0.0

    # --- rates ---
    @property
    def iemfp(self):
        emfp = self.sample.get_emfp(self.energy)
        if (not np.isfinite(emfp)) or emfp <= 0:
            return 0.0
        return 1.0 / emfp

    @property
    def iimfp(self):
        # Metal model: inelastic only if E > E_F (your DB construction)
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
        # basic
        if (not np.isfinite(self.energy)) or self.energy <= 0.0:
            self.dead = True
            return
    
        # METAL THERMALIZATION RULE:
        # Once an electron's solid energy drops to EF or below, it merges into the Fermi sea.
        if self.sample.is_metal and self.inside and self.energy <= self.e_fermi:
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
            ind = self.sample.energy_index(max(self.energy, self.sample.elastic_min_energy))
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

        # DOS sampling (your model, but cached energy grid)
        self.feg_dos()

        # inelastic angular distribution: still computed per-event
        # (we can cache later by binning dE if you want maximum speed)
        theta, ang = self.sample.angular_iimfp(self.energy + self.energy_loss, self.energy_loss)
        w = np.nan_to_num(ang, nan=0.0) * np.sin(theta)
        cdf2 = cumulative_trapezoid(w, theta, initial=0.0)
        total = float(cdf2[-1])
        if total > 0 and np.isfinite(total):
            cdf2 = cdf2 / total
            self.deflection[0] = float(np.interp(self.rng.random(), cdf2, theta))
        else:
            # kinematic fallback
            arg = self.energy_loss / max(self.energy + self.energy_loss, 1e-12)
            arg = min(1.0, max(0.0, arg))
            self.deflection[0] = math.asin(math.sqrt(arg))

        self.uvw = self.change_direction(self.uvw, self.deflection)
        return True  # inelastic occurred -> may spawn SE

    def feg_dos(self):
        # Your metal model uses e_fermi as ref
        e_ref = self.e_fermi if self.sample.is_metal else float(self.sample.material_data.get('e_vb', 0.0))
        e_ref = max(e_ref, 1e-6)

        ener = self.sample.dos_cdf()
        # energy_loss changes per event, so CDF changes; but we avoid rebuilding ener each time
        dist = np.sqrt(np.maximum(ener * (ener + self.energy_loss), 0.0))
        cdf = cumulative_trapezoid(dist, ener, initial=0.0)
        total = float(cdf[-1])
        if total > 0 and np.isfinite(total):
            cdf = cdf / total
            self.energy_se = float(np.interp(self.rng.random(), cdf, ener))
        else:
            self.energy_se = 0.0

    def escape(self):
        if self.xyz[2] > 0.0:
            return False
    
        self.n_escape_calls += 1
    
        Ui = self.Ui
        Eperp = self.energy * (self.uvw[2] ** 2)
    
        ratio = Eperp / Ui if Ui > 0 else float("inf")
        self.min_Eperp_over_Ui = min(self.min_Eperp_over_Ui, ratio)
        self.max_Eperp_over_Ui = max(self.max_Eperp_over_Ui, ratio)

        if self.n_escape_calls <= 10:
            print("Es", self.energy, "uz", self.uvw[2], "Eperp/Ui", Eperp/Ui)
    
        if self.energy <= Ui or Eperp <= Ui:
            self.n_escape_below_barrier += 1   # <-- NEW
            self.uvw[2] *= -1
            self.xyz[2] = 1e-10
            if self.save_coordinates:
                self.coordinates.append([round(v, 2) for v in self.xyz + [self.energy]])
            return False
    
        root = math.sqrt(1.0 - Ui / Eperp)
        t = 4.0 * root / ((1.0 + root) ** 2)
    
        if self.rng.random() < t:
            self.inside = False
            Ev = self.energy - Ui
            self.energy_vac = Ev
            self.energy = Ev
            self.n_escape_transmit += 1        # <-- NEW
            return True
    
        self.n_escape_reflected_prob += 1      # <-- NEW
        self.uvw[2] *= -1
        self.xyz[2] = 1e-10
        return False

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
    
        if traj_id == 0:
            print("E0", E0, "E_F", self.sample.material_data.get("e_fermi"), "WF", self.sample.material_data.get("work_function"))
    
        tey = 0
        sey = 0
        bse = 0
    
        electrons = []
        Ui = self.sample.material_data['e_fermi'] + self.sample.material_data['work_function']
        E_s0 = E0 + Ui
    
        electrons.append(Electron(
            self.sample, E_s0, self.cb_ref, self.track_trajectories,
            xyz=[0, 0, 0], uvw=[math.sin(self.incident_angle), 0, math.cos(self.incident_angle)],
            gen=0, se=False, ind=-1, rng=rng
        ))

        if traj_id == 0:
            print("Init primary: E0(vac)=", E0, "Ui=", Ui, "Es0=", electrons[0].energy)
    
        traj_tracks = []

        n_scatter = 0
        max_scatter = 20000
        min_ratio = float("inf")
        max_ratio = 0.0
        below_barrier = 0
        ref_prob = 0
        transmit = 0
        escape_calls = 0

        i = 0
        while i < len(electrons):
            e = electrons[i]
    
            while e.inside and (not e.dead):
                if n_scatter >= max_scatter:
                    e.dead = True
                    break
                    
                e.travel()
                if e.dead:
                    break

                if self.sample.is_metal and e.energy <= e.e_fermi:
                    e.dead = True
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
                # thermalization in metals: below or at EF, electron is absorbed
                if self.sample.is_metal and e.energy <= e.e_fermi:
                    e.dead = True
                    break
                n_scatter += 1
                if made_inelastic:
                    se_energy = e.energy_loss + e.energy_se
    
                    # spawn criterion (metal): only above EF for this model
                    if self.sample.is_metal and se_energy <= e.e_fermi:
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
            escape_calls += e.n_escape_calls
            below_barrier += e.n_escape_below_barrier
            ref_prob += e.n_escape_reflected_prob
            transmit += e.n_escape_transmit
            
            min_ratio = min(min_ratio, e.min_Eperp_over_Ui)
            max_ratio = max(max_ratio, e.max_Eperp_over_Ui)

        if traj_id == 0:
            print("escape_calls", escape_calls,
                  "below_barrier", below_barrier,
                  "ref_prob", ref_prob,
                  "transmit", transmit,
                  "min(Eperp/Ui)", min_ratio,
                  "max(Eperp/Ui)", max_ratio)



        return tey, sey, bse, (traj_tracks if self.track_trajectories else None)


    def run_simulation(self, use_parallel=False):
        import time
        from tqdm import tqdm
        import multiprocessing as mp

        t0 = time.time()

        for k, E0 in enumerate(self.energy_array):
            if not use_parallel:
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
            else:
                # parallel: return just counts (avoid huge pickles)
                with mp.Pool(mp.cpu_count()) as pool:
                    results = list(tqdm(
                        pool.starmap(self.run_one_trajectory, [(E0, traj) for traj in range(self.n_trajectories)]),
                        total=self.n_trajectories,
                        desc=f"E={E0:.1f} eV"
                    ))

                t_tey = sum(r[0] for r in results)
                t_sey = sum(r[1] for r in results)
                t_bse = sum(r[2] for r in results)

                self.tey[k] = t_tey / self.n_trajectories
                self.sey[k] = t_sey / self.n_trajectories
                self.bse[k] = t_bse / self.n_trajectories

                if self.track_trajectories:
                    self.tracks.append([r[3] for r in results])

        print(f"Done in {time.time()-t0:.1f} s")

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

