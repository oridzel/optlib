"""
seemc.py -- Monte Carlo simulation of secondary electron emission in solids.

Part of `optlib`.  Reads a material database produced by the optlib DB builder
and transports electrons through a semi-infinite solid occupying z > 0, with
vacuum at z < 0.

=============================================================================
ENERGY CONVENTIONS  --  READ THIS BEFORE TOUCHING ANY TABLE LOOKUP
=============================================================================
Three different energy references appear in this code.  Mixing them up is the
single easiest way to get a wrong answer, so every table lookup goes through a
named converter in `Sample` and never touches `material_data` directly.

    E_s     "solid" energy, measured from the BOTTOM OF THE VALENCE BAND.
            This is the state variable carried by `Electron.energy` while the
            electron is inside the solid.  It equals T' in Shinotsuka et al.,
            Surf. Interface Anal. 47 (2015) 871, Eq. (2).

    T       = E_s - E_F, measured from the FERMI LEVEL.  This is the abscissa
            of the standard IMFP tables (Shinotsuka Table 2: "electron kinetic
            energy T with respect to the Fermi level").

    E_vac   = E_s - U_i  with  U_i = E_F + phi,  the kinetic energy the
            electron would have in vacuum.  This is the abscissa of ELSEPA
            elastic tables: ELSEPA is fed a vacuum kinetic energy, and the
            solid-state optical-model potential is flat (= -Delta_E) outside
            the muffin-tin sphere, so the inner potential is folded into the
            potential rather than into the energy argument
            (Salvat/Jablonski/Powell, "Atoms in solids" note, Eqs. 7-8).

Which reference each DB table uses is declared once, in `MCConfig`:

    imfp_energy_ref  : 'vb_bottom' (default, matches the optlib FPA builder)
                       or 'fermi'  (matches the published Shinotsuka tables)
    emfp_energy_ref  : 'vacuum'    (default, matches ELSEPA)

KINEMATIC INVARIANTS (Shinotsuka Eq. 2-3) -- these are asserted, not assumed:

    omega_max = T' - E_F = E_s - E_F        (maximum energy loss)
    q_bounds  are evaluated at T' = E_s     (NOT at E_s - E_F)

The relativistic momenta used for the q-bounds are also used for the
projectile deflection, so the sampled q is guaranteed to lie in
[|k - k'|, k + k'] and the law-of-cosines never needs clamping.

=============================================================================
CHANGELOG vs. the previous version
=============================================================================
Physics / correctness
  1. A truncated step at the surface no longer forces a scattering event.
     Previously every internal reflection was followed by a collision that the
     exponential never generated, piling up spurious energy loss in exactly
     the depth range that controls SEY.
  2. Secondary-electron direction is now built from the momentum transfer q
     (rotated out of the frame with z || q) instead of the ad-hoc rule
     [pi - theta, phi + pi] applied to the *already deflected* projectile
     direction.  Energy and momentum of the (projectile, SE) pair are now
     consistent by construction.
  3. `energy_se` can no longer leak from a previous collision: `scatter()`
     returns an explicit result object instead of mutating shared state.
  4. Rejected inelastic samples no longer silently become null collisions.
     omega is drawn from a CDF truncated at omega_max, and q from a grid built
     inside [q-, q+], so a valid event is produced every time.  The residual
     failure modes are counted in `diagnostics`.
  5. q-bounds evaluated at E_s (was E_s - E_F).  See above.
  6. Projectile deflection uses relativistic momenta, consistent with the
     q-bounds.
  7. The incident beam is refracted at the surface barrier (parallel momentum
     conserved), instead of keeping its vacuum direction.
  8. Energy-bin lookup is stochastically interpolated between adjacent bins
     instead of snapping to the nearest bin.
  9. Unconditional death check at the top of the transport loop plus a step
     counter, so a sub-barrier electron in a non-metal cannot loop forever.
 10. Contradictory q-unit handling removed: the DB's 'q' grid has one declared
     unit (`MCConfig.q_unit`), validated at load time.

Bookkeeping
 11. Emitted electrons are classified both by cascade flag and by the
     conventional 50 eV cut, so results are comparable to measured
     delta / eta curves and to MAST-SEY.
 12. Per-energy statistical uncertainties (standard error of the mean).
 13. Emission energy and angle spectra are collected.
 14. One shared trajectory implementation for serial and parallel runs.
 15. Reproducible seeding via numpy SeedSequence (no PID dependence).
"""

from __future__ import annotations

import math
import os
import pickle
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import RectBivariateSpline

# --------------------------------------------------------------------------
# Constants.  Defined locally so the module is self-contained and testable,
# then cross-checked against optlib.constants if that is importable.  The old
# code did `from optlib.constants import *` and *also* defined
# HBAR2_2M_eVA2 further down the file, so which value won depended on file
# order -- exactly the kind of thing that silently poisons a q conversion.
# --------------------------------------------------------------------------
H2EV = 27.211386245988          # Hartree -> eV
A0_ANG = 0.529177210903         # Bohr radius in Angstrom
HBAR2_2M_EVA2 = 0.5 * H2EV * A0_ANG ** 2   # hbar^2/2m in eV*Angstrom^2 (3.80998)
C_AU = 137.035999084            # speed of light in atomic units

# Backwards-compatible aliases (old code referenced these names)
h2ev = H2EV
a0 = A0_ANG
HBAR2_2M_eVA2 = HBAR2_2M_EVA2

# If optlib is installed, ADOPT its constants rather than argue with them: the
# DIIMFP/ELF tables were integrated using optlib's values, so the sampler must
# use the same numbers.  Self-consistency with the database matters more than
# agreement with the newest CODATA digits -- a 1e-5 difference in the Hartree
# is far below every other approximation in this model.
#
# The tolerances distinguish two very different situations:
#   * a rounding-level difference (h2ev = 27.21184 vs 27.211386...) -> adopt silently
#   * a genuinely different constant (Rydberg for Hartree, nm or m for Angstrom,
#     or a0 = 1 in atomic units) -> fatal, because it would silently corrupt
#     every q conversion in the code
_ADOPT_QUIETLY = 1e-4      # below this: same constant, different rounding
_FATAL_MISMATCH = 1e-2     # above this: a different constant entirely

try:  # pragma: no cover - depends on installation
    from optlib import constants as _optlib_constants
except ImportError:
    _optlib_constants = None

if _optlib_constants is not None:
    _adopted = {}
    for _name, _mine in (("h2ev", H2EV), ("a0", A0_ANG)):
        _ref = getattr(_optlib_constants, _name, None)
        if _ref is None:
            continue
        _rel = abs(float(_ref) - _mine) / _mine
        if _rel > _FATAL_MISMATCH:
            raise ValueError(
                f"optlib.constants.{_name} = {_ref} differs from the expected "
                f"{_mine} by {_rel:.1%}. That is a different constant, not a "
                f"different rounding (Rydberg vs Hartree, or nm/m vs Angstrom). "
                f"Fix the unit convention rather than the tolerance."
            )
        if _rel > _ADOPT_QUIETLY:
            warnings.warn(
                f"Adopting optlib.constants.{_name} = {_ref} in place of "
                f"{_mine} ({_rel:.2e} relative difference).",
                RuntimeWarning, stacklevel=2,
            )
        _adopted[_name] = float(_ref)

    H2EV = _adopted.get("h2ev", H2EV)
    A0_ANG = _adopted.get("a0", A0_ANG)
    HBAR2_2M_EVA2 = 0.5 * H2EV * A0_ANG ** 2   # derived, never imported
    h2ev, a0, HBAR2_2M_eVA2 = H2EV, A0_ANG, HBAR2_2M_EVA2


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class MCConfig:
    """Everything that used to be a magic number or an implicit assumption."""

    # --- table energy references (see module docstring) ---
    imfp_energy_ref: str = "vb_bottom"   # 'vb_bottom' (E_s) or 'fermi' (E_s - E_F)
    emfp_energy_ref: str = "vacuum"      # 'vacuum' (E_s - U_i) or 'vb_bottom'

    # --- units of material_data['q'] ---
    q_unit: str = "a0^-1"                # 'a0^-1' or 'A^-1'

    # --- elastic ---
    elastic_min_energy: float = 5.0      # ELSEPA tables clamped below this (eV, vacuum ref)

    # --- surface barrier ---
    # 'abrupt'   : T = 4r/(1+r)^2, r = sqrt(1 - Ui/E_perp).  Abrupt step.
    #              ('quantum' is accepted as a synonym.)
    # 'classical': T = 1 whenever E_perp > Ui.
    # 'sigmoid'  : JMONSEL's barrier (Villarrubia 2015 Eq. 11).  The potential
    #              rises as U(x) = dU / [1 + exp(-2x/w)] over a width w, and
    #              Schroedinger's equation is solved exactly:
    #                  T = 1 - [sinh(pi w (k1-k2)/2) / sinh(pi w (k1+k2)/2)]^2
    #              w -> 0 recovers 'abrupt'; large w recovers 'classical'.
    #              This matters: a real surface is not atomically abrupt, and
    #              the abrupt limit is the LOWEST-transmission choice, so it
    #              gives the lowest SEY of the three.
    barrier_model: str = "abrupt"
    barrier_width: float = 0.0           # w in Angstrom, used by 'sigmoid'

    # --- how the SE-generation mechanism is decided ---
    # 'mao'  : Mao et al. 2008 Eq. (9).  After (omega, q) are sampled, single
    #          electron excitation is declared if q- <= q <= q+, and plasmon
    #          damping if q < q-, with
    #              q_mp = -/+ k_F + sqrt(k_F^2 + 2 omega)      [atomic units]
    #          This is EXACTLY the condition for the Fermi-sphere disk to be
    #          non-empty, so the binary-encounter sampler can never fail and
    #          no excitation is ever lost.  RECOMMENDED.
    # 'table': trust the DB's elf_se / elf_pl split to also define the SE
    #          mechanism.  This is only equivalent to 'mao' if the tables were
    #          built with the same k_F that the DB reports as e_fermi.  For an
    #          FPA database they generally were NOT: the FPA decomposition
    #          integrates over a plasmon frequency omega_p that scans the whole
    #          optical range, so the support of elf_se is set by the LARGEST
    #          k_F(omega_p) in the decomposition, not by the material's k_F.
    #          The result is elf_se strength at q < q-, where no target state
    #          exists -- silently destroying secondaries.
    se_channel_rule: str = "mao"

    # --- FEG parameters used by the binary-encounter SE model ---
    # k_F for the struck-electron sampling is normally taken from the DB's
    # e_fermi.  For a d-band metal that is often WRONG: optlib's Penn/FPA
    # extension disperses the ELF using an electron density inferred from the
    # f-sum rule (for Cu that is ~11 e/atom, E_F ~ 35 eV), while `e_fermi` in
    # the DB is the true Fermi energy (~8.7 eV, ~1 e/atom).  If those disagree,
    # the pair continuum implied by the ELF is much wider than the one the
    # sampler enforces, and a large fraction of (omega, q) pairs get Pauli
    # rejected -- each rejection is a secondary electron that never existed.
    # Set this to the density-equivalent E_F to make them consistent.
    feg_fermi_energy: Optional[float] = None

    # What to do when the FEG kinematics forbid the sampled (omega, q):
    # 'fallback' : still create a secondary, with E_SE = E_i + omega drawn from
    #              the occupied DOS (the plasmon-channel construction).  Energy
    #              is conserved and no excitation is lost.
    # 'drop'     : create no secondary (the previous behaviour).  The projectile
    #              still loses omega, so this is a silent energy sink and a
    #              direct SEY deficit.
    on_pauli_block: str = "fallback"

    # --- secondary electron generation ---
    # 'momentum' : SE direction from k_f = k_i + q, rotated out of the q frame.
    # 'isotropic': SE emitted isotropically (debug / comparison only).
    se_direction_model: str = "momentum"
    # Plasmon decay is a Landau-damping event at q ~ q_c that is uncorrelated
    # with the incident direction; isotropic is the standard choice
    # (Ding & Shimizu).  'momentum' reuses the binary-encounter construction.
    plasmon_se_direction: str = "isotropic"

    # --- termination ---
    # An electron with E_s <= U_i can never escape through a step barrier and
    # cannot gain energy, so tracking it only costs time.  Set True if you add
    # phonon transport or want energy-deposition maps.
    track_subbarrier: bool = False
    max_steps_per_electron: int = 100_000
    max_generation: int = 100
    max_secondaries_per_trajectory: int = 100_000

    # --- classification ---
    bse_cutoff_ev: float = 50.0          # conventional SE/BSE split on emission energy

    # --- sampling resolution ---
    n_q_sample: int = 64                 # points used to build the conditional q CDF
    n_theta_dcs: int = 0                 # 0 = use the DB's native theta grid

    # --- diagnostics ---
    collect_spectra: bool = True

    def validate(self) -> None:
        if self.imfp_energy_ref not in ("vb_bottom", "fermi"):
            raise ValueError(f"bad imfp_energy_ref: {self.imfp_energy_ref}")
        if self.emfp_energy_ref not in ("vacuum", "vb_bottom"):
            raise ValueError(f"bad emfp_energy_ref: {self.emfp_energy_ref}")
        if self.q_unit not in ("a0^-1", "A^-1"):
            raise ValueError(f"bad q_unit: {self.q_unit}")
        if self.se_direction_model not in ("momentum", "isotropic"):
            raise ValueError(f"bad se_direction_model: {self.se_direction_model}")
        if self.plasmon_se_direction not in ("momentum", "isotropic"):
            raise ValueError(f"bad plasmon_se_direction: {self.plasmon_se_direction}")
        if self.barrier_model == "quantum":
            self.barrier_model = "abrupt"        # backwards-compatible synonym
        if self.barrier_model not in ("abrupt", "classical", "sigmoid"):
            raise ValueError(f"bad barrier_model: {self.barrier_model}")
        if self.barrier_model == "sigmoid" and self.barrier_width <= 0:
            raise ValueError("barrier_model='sigmoid' requires barrier_width > 0")
        if self.se_channel_rule not in ("mao", "table"):
            raise ValueError(f"bad se_channel_rule: {self.se_channel_rule}")
        if self.on_pauli_block not in ("fallback", "drop"):
            raise ValueError(f"bad on_pauli_block: {self.on_pauli_block}")


# --------------------------------------------------------------------------
# Small numerical helpers
# --------------------------------------------------------------------------
def cumtrapz_numpy(y, x):
    """Cumulative trapezoid integral, same length as x, starting at 0."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    area = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    return np.concatenate(([0.0], np.cumsum(area)))


def _invert_cdf(cdf, x, u):
    """
    Invert a monotonically non-decreasing CDF by linear interpolation.

    Unlike np.interp(u, cdf, x) this is safe when the CDF has flat regions
    (zero-probability gaps in the ELF), which np.interp resolves arbitrarily.

    Uses the ndarray.searchsorted method rather than np.searchsorted: the
    module-level function goes through numpy's dispatch wrapper, which
    dominates the runtime when it is called a few hundred times per
    trajectory on scalars.
    """
    j = int(cdf.searchsorted(u, side="right")) - 1
    n2 = len(x) - 2
    j = 0 if j < 0 else (n2 if j > n2 else j)
    c0, c1 = float(cdf[j]), float(cdf[j + 1])
    if c1 <= c0:
        return float(x[j])
    t = (u - c0) / (c1 - c0)
    return float(x[j] + t * (x[j + 1] - x[j]))


def _bin_and_fraction(grid, value):
    """Return (i, t) with grid[i] <= value <= grid[i+1] and t the fraction."""
    n = len(grid)
    if n < 2:
        return 0, 0.0
    lo, hi = grid[0], grid[-1]
    v = lo if value < lo else (hi if value > hi else float(value))
    i = int(grid.searchsorted(v, side="right")) - 1
    n2 = n - 2
    i = 0 if i < 0 else (n2 if i > n2 else i)
    span = grid[i + 1] - grid[i]
    t = 0.0 if span <= 0 else (v - grid[i]) / span
    return i, float(np.clip(t, 0.0, 1.0))


def _k_rel_au(E_ev):
    """
    Relativistic electron momentum in atomic units (a0^-1) for energy E_ev.

        k = sqrt(E (2 + E/c^2))     [Hartree atomic units]

    Same expression as Shinotsuka Eq. (2); used for BOTH the q-bounds and the
    projectile deflection so the two can never disagree.
    """
    e = max(float(E_ev), 0.0) / H2EV
    return math.sqrt(e * (2.0 + e / (C_AU ** 2)))


def barrier_transmission(E_perp, Ui, cfg):
    """
    Transmission through the surface barrier for perpendicular energy E_perp.

    'abrupt'    Abrupt step:  T = 4 r / (1 + r)^2,  r = sqrt(1 - Ui/E_perp).
    'classical' T = 1 above the barrier.
    'sigmoid'   Villarrubia et al., Ultramicroscopy 154 (2015) Eq. (11), for
                U(x) = Ui / [1 + exp(-2x/w)]:

                    T = 1 - [sinh(pi w (k1 - k2)/2) / sinh(pi w (k1 + k2)/2)]^2

                with k1, k2 the perpendicular wavenumbers inside and outside.
                Evaluated in log space because both sinh arguments overflow
                for any realistic w at a few tens of eV.
    """
    if E_perp <= Ui:
        return 0.0
    model = cfg.barrier_model
    if model == "classical":
        return 1.0
    if model == "abrupt":
        r = math.sqrt(1.0 - Ui / E_perp)
        return 4.0 * r / ((1.0 + r) ** 2)

    # sigmoid
    k1 = math.sqrt(2.0 * E_perp / H2EV) / A0_ANG          # Angstrom^-1
    k2 = math.sqrt(2.0 * (E_perp - Ui) / H2EV) / A0_ANG
    a = 0.5 * math.pi * cfg.barrier_width * (k1 - k2)
    b = 0.5 * math.pi * cfg.barrier_width * (k1 + k2)
    if b <= 0.0:
        return 0.0
    if b < 20.0:
        ratio = math.sinh(a) / math.sinh(b)
    else:
        # sinh(a)/sinh(b) = exp(a-b) * (1 - e^-2a)/(1 - e^-2b)
        ratio = math.exp(a - b) * (1.0 - math.exp(-2.0 * a)) / (1.0 - math.exp(-2.0 * b))
    return max(0.0, min(1.0, 1.0 - ratio * ratio))


def _isotropic_direction(rng):
    cos_t = 2.0 * rng.random() - 1.0
    sin_t = math.sqrt(max(1.0 - cos_t * cos_t, 0.0))
    phi = 2.0 * math.pi * rng.random()
    return [sin_t * math.cos(phi), sin_t * math.sin(phi), cos_t]


def rotate_direction(uvw, polar, azimuth):
    """
    Rotate the unit vector `uvw` by `polar` away from its own axis and
    `azimuth` about it.  (Unchanged from the original `change_direction`,
    which was correct, including the uvw ~ +/-z degenerate case.)
    """
    sin_psi = math.sin(polar)
    cos_psi = math.cos(polar)
    sin_fi = math.sin(azimuth)
    cos_fi = math.cos(azimuth)

    cos_theta = uvw[2]
    sin_theta = math.sqrt(max(uvw[0] ** 2 + uvw[1] ** 2, 0.0))
    if sin_theta > 1e-12:
        cos_phi = uvw[0] / sin_theta
        sin_phi = uvw[1] / sin_theta
    else:
        cos_phi, sin_phi = 1.0, 0.0

    h0 = sin_psi * cos_fi
    h1 = sin_theta * cos_psi + h0 * cos_theta
    h2 = sin_psi * sin_fi

    out = [
        h1 * cos_phi - h2 * sin_phi,
        h1 * sin_phi + h2 * cos_phi,
        cos_theta * cos_psi - h0 * sin_theta,
    ]
    norm = math.sqrt(out[0] ** 2 + out[1] ** 2 + out[2] ** 2)
    if norm > 0:
        out = [v / norm for v in out]
    return out


class Diagnostics(dict):
    """Counter bag.  Every silent fallback in the physics increments one."""

    _KEYS = (
        "inelastic_events",
        "elastic_events",
        "surface_encounters",
        "escapes",
        "internal_reflections",
        "se_created",
        "se_blocked_pauli",       # FEG kinematics forbade a target state
        "se_pauli_fallback",      # ... and a DOS-based secondary was made instead
        "channel_reclassified",   # table channel != Mao q-boundary channel
        "se_below_barrier",       # SE created but cannot escape -> not tracked
        "omega_cdf_empty",        # energy bin had no inelastic strength
        "q_window_clipped",       # [q-, q+] extended past the tabulated q grid
        "q_cdf_empty",            # ELF integrated to zero inside [q-, q+]
        "step_limit_hit",
        "generation_limit_hit",
    )

    def __init__(self):
        super().__init__({k: 0 for k in self._KEYS})

    def add(self, other):
        for k, v in other.items():
            self[k] = self.get(k, 0) + v

    def report(self, n_trajectories=None):
        lines = ["Diagnostics:"]
        for k in sorted(self):
            v = self[k]
            if n_trajectories:
                lines.append(f"  {k:<24s} {v:>12d}   ({v / n_trajectories:.4g}/traj)")
            else:
                lines.append(f"  {k:<24s} {v:>12d}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Sample
# --------------------------------------------------------------------------
class Sample:
    """Material tables plus every sampling routine that depends only on them."""

    def __init__(self, name, db_path="MaterialDatabase.pkl", config: Optional[MCConfig] = None):
        self.cfg = config or MCConfig()
        self.cfg.validate()

        with open(db_path, "rb") as fp:
            data = pickle.load(fp)

        if isinstance(data, dict):
            if data.get("name") != name:
                raise ValueError(f"DB holds '{data.get('name')}', requested '{name}'")
            self.material_data = data
        elif isinstance(data, list):
            names = [d.get("name") for d in data]
            if name not in names:
                raise ValueError(f"Allowed sample names are {names}")
            self.material_data = next(d for d in data if d.get("name") == name)
        else:
            raise ValueError("Unrecognized MaterialDatabase.pkl format")

        md = self.material_data
        self.name = md["name"]
        self.is_metal = bool(md["is_metal"])

        self.Egrid = np.asarray(md["energy"], dtype=float)
        if not np.all(np.diff(self.Egrid) > 0):
            raise ValueError("material_data['energy'] must be strictly increasing")
        self.Emin = float(self.Egrid[0])
        self.Emax = float(self.Egrid[-1])

        self.e_fermi = float(md.get("e_fermi", 0.0))
        self.work_function = float(md.get("work_function", 0.0))
        self.Ui = self.e_fermi + self.work_function     # VB bottom -> vacuum level
        self.e_vb = float(md.get("e_vb", 0.0))

        # Fermi energy used ONLY by the binary-encounter SE kinematics.  Kept
        # separate from self.e_fermi (which sets omega_max and the barrier)
        # because for a d-band metal they legitimately differ -- see
        # MCConfig.feg_fermi_energy.
        self.e_fermi_feg = float(
            self.cfg.feg_fermi_energy if self.cfg.feg_fermi_energy is not None
            else self.e_fermi
        )
        self.k_fermi_feg = math.sqrt(max(2.0 * self.e_fermi_feg / H2EV, 0.0))

        self.imfp_table = np.asarray(md["imfp"], dtype=float)
        self.emfp_table = np.asarray(md["emfp"], dtype=float)
        # Interpolate the inverse MFPs directly: they are what the transport
        # needs, and this avoids a division per step plus the 1/0 guards.
        with np.errstate(divide="ignore", invalid="ignore"):
            self.inv_imfp_table = np.where(self.imfp_table > 0, 1.0 / self.imfp_table, 0.0)
            self.inv_emfp_table = np.where(self.emfp_table > 0, 1.0 / self.emfp_table, 0.0)
        self.inv_imfp_table[~np.isfinite(self.inv_imfp_table)] = 0.0
        self.inv_emfp_table[~np.isfinite(self.inv_emfp_table)] = 0.0
        self._check_table_shapes()

        self._precompute_elastic_cdfs()
        self._precompute_inelastic_channel_cdfs()
        self._build_elf_channel_splines()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _check_table_shapes(self):
        n = self.Egrid.size
        for key in ("imfp", "emfp", "inv_imfp_pl", "inv_imfp_se"):
            arr = np.asarray(self.material_data[key], dtype=float)
            if arr.shape != (n,):
                raise ValueError(f"material_data['{key}'] has shape {arr.shape}, expected ({n},)")

        decs = np.asarray(self.material_data["decs"], dtype=float)
        theta = np.asarray(self.material_data["decs_theta"], dtype=float)
        if decs.shape != (theta.size, n):
            raise ValueError(
                f"decs has shape {decs.shape}, expected ({theta.size}, {n}); the elastic "
                "tables must share the 'energy' grid"
            )

        for key in ("diimfp_se", "diimfp_pl"):
            arr = np.asarray(self.material_data[key], dtype=float)
            if arr.ndim != 3 or arr.shape[1] != 2 or arr.shape[2] != n:
                raise ValueError(
                    f"material_data['{key}'] has shape {arr.shape}, expected (Nw, 2, {n})"
                )

    def consistency_report(self):
        """
        Cross-checks worth running once per material.  These catch the class of
        bug that produces a plausible-looking but wrong SEY curve.
        """
        md = self.material_data
        lines = [f"Consistency report for {self.name}", "-" * 46]

        inv_tot = np.asarray(md["inv_imfp_pl"], float) + np.asarray(md["inv_imfp_se"], float)
        # Only bins that carry inelastic strength are meaningful: below E_F + the
        # smallest excitation there is nothing to compare.
        live = inv_tot > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(inv_tot[live] * self.imfp_table[live] - 1.0)
        rel = rel[np.isfinite(rel)]
        worst = float(np.max(rel)) if rel.size else float("nan")
        lines.append(
            f"  1/imfp vs (inv_imfp_pl + inv_imfp_se): max rel. deviation {worst:.3%}"
        )
        if worst > 0.02:
            lines.append(
                "     WARNING: the transport rate and the channel decomposition disagree. "
                "The channel branching will not reproduce the tabulated IMFP."
            )

        q = np.asarray(md["q"], float)
        lines.append(
            f"  q grid: [{q.min():.4g}, {q.max():.4g}] declared as {self.cfg.q_unit}"
        )
        # A physically sensible grid must span the momentum transfers that the
        # kinematics actually demand at the top of the energy range.
        k_top = _k_rel_au(self.Emax)
        q_top = 2.0 * k_top if self.cfg.q_unit == "a0^-1" else 2.0 * k_top / A0_ANG
        if q.max() < 0.5 * q_top:
            lines.append(
                f"     WARNING: q_max = {q.max():.4g} is far below the 2k = {q_top:.4g} "
                f"required at E = {self.Emax:.4g} eV. Check q_unit."
            )

        lines.append(f"  E_F = {self.e_fermi:.3f} eV, phi = {self.work_function:.3f} eV, "
                     f"U_i = {self.Ui:.3f} eV")
        lines.append(f"  IMFP tabulated vs '{self.cfg.imfp_energy_ref}', "
                     f"EMFP vs '{self.cfg.emfp_energy_ref}'")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Energy reference conversions -- the ONLY places a reference is applied
    # ------------------------------------------------------------------
    def E_fermi_ref(self, E_s):
        return E_s - self.e_fermi

    def E_vacuum_ref(self, E_s):
        return E_s - self.Ui

    def _imfp_abscissa(self, E_s):
        return E_s if self.cfg.imfp_energy_ref == "vb_bottom" else self.E_fermi_ref(E_s)

    def _emfp_abscissa(self, E_s):
        if self.cfg.emfp_energy_ref == "vacuum":
            # ELSEPA is tabulated against vacuum kinetic energy; below the
            # tabulated minimum the DCS is frozen at elastic_min_energy.
            return max(self.E_vacuum_ref(E_s), self.cfg.elastic_min_energy)
        return E_s

    def _clip_E(self, E):
        # Plain comparisons: np.clip on a scalar costs ~8 us of dispatch
        # overhead and this is the single hottest call in the transport loop.
        if E < self.Emin:
            return self.Emin
        if E > self.Emax:
            return self.Emax
        return float(E)

    # ------------------------------------------------------------------
    # Mean free paths
    # ------------------------------------------------------------------
    def get_imfp(self, E_s):
        E = self._clip_E(self._imfp_abscissa(E_s))
        return float(np.interp(E, self.Egrid, self.imfp_table))

    def get_emfp(self, E_s):
        E = self._clip_E(self._emfp_abscissa(E_s))
        return float(np.interp(E, self.Egrid, self.emfp_table))

    def inverse_mfps(self, E_s):
        """(1/emfp, 1/imfp) at E_s.  Evaluated once per transport step."""
        inv_e = float(np.interp(self._clip_E(self._emfp_abscissa(E_s)),
                                self.Egrid, self.inv_emfp_table))
        if self.is_metal and E_s <= self.e_fermi:
            inv_i = 0.0            # no inelastic channel below E_F
        else:
            inv_i = float(np.interp(self._clip_E(self._imfp_abscissa(E_s)),
                                    self.Egrid, self.inv_imfp_table))
        return inv_e, inv_i

    def omega_max(self, E_s):
        """Shinotsuka Eq. (3): omega_max = T' - E_F for a metal."""
        return E_s - self.e_fermi if self.is_metal else E_s

    # ------------------------------------------------------------------
    # Elastic
    # ------------------------------------------------------------------
    def _precompute_elastic_cdfs(self):
        theta = np.asarray(self.material_data["decs_theta"], dtype=float)
        decs = np.asarray(self.material_data["decs"], dtype=float)

        if not np.all(np.diff(theta) > 0):
            raise ValueError("decs_theta must be strictly increasing")

        pdf = 2.0 * np.pi * decs * np.sin(theta)[:, None]
        pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
        pdf[pdf < 0] = 0.0

        area = 0.5 * (pdf[1:, :] + pdf[:-1, :]) * np.diff(theta)[:, None]
        cdf = np.vstack([np.zeros((1, pdf.shape[1])), np.cumsum(area, axis=0)])

        total = cdf[-1, :]
        good = (total > 0) & np.isfinite(total)
        if not np.all(good):
            bad = np.where(~good)[0]
            raise ValueError(
                f"Elastic DCS integrates to zero at energy bins {bad[:5].tolist()}"
                f"{' ...' if bad.size > 5 else ''}. The old code silently replaced these "
                "with a uniform-in-theta distribution, which is not isotropic and not "
                "physical; fix the table instead."
            )
        cdf /= total

        self._elastic_theta = theta
        self._elastic_cdf = cdf

    def sample_elastic_theta(self, E_s, rng):
        """Sample the elastic polar deflection, interpolating between energy bins."""
        i, t = _bin_and_fraction(self.Egrid, self._clip_E(self._emfp_abscissa(E_s)))
        j = i + 1 if (t > 0.0 and rng.random() < t) else i
        return _invert_cdf(self._elastic_cdf[:, j], self._elastic_theta, rng.random())

    # ------------------------------------------------------------------
    # Inelastic: omega CDFs per channel per energy bin
    # ------------------------------------------------------------------
    def _precompute_inelastic_channel_cdfs(self):
        def build(key):
            di = np.asarray(self.material_data[key], float)      # (Nw, 2, NE)
            eloss = di[:, 0, :]
            pdf = np.nan_to_num(di[:, 1, :], nan=0.0, posinf=0.0, neginf=0.0)
            pdf[pdf < 0] = 0.0

            area = 0.5 * (pdf[1:, :] + pdf[:-1, :]) * np.diff(eloss, axis=0)
            cdf = np.vstack([np.zeros((1, area.shape[1])), np.cumsum(area, axis=0)])
            total = cdf[-1, :]
            has = (total > 0) & np.isfinite(total)
            # NOTE: kept UNNORMALISED.  Truncating at omega_max requires the
            # absolute cumulative value; normalising here and renormalising
            # later is what made the old sampler reject-and-drop.
            return eloss, cdf, has

        self._w_se, self._cdf_se, self._has_se = build("diimfp_se")
        self._w_pl, self._cdf_pl, self._has_pl = build("diimfp_pl")

    def _channel_tables(self, ch):
        if ch == "se":
            return self._w_se, self._cdf_se, self._has_se
        return self._w_pl, self._cdf_pl, self._has_pl

    def choose_channel(self, E_s, rng):
        E = self._clip_E(self._imfp_abscissa(E_s))
        inv_pl = float(np.interp(E, self.Egrid, self.material_data["inv_imfp_pl"]))
        inv_se = float(np.interp(E, self.Egrid, self.material_data["inv_imfp_se"]))
        s = inv_pl + inv_se
        if not np.isfinite(s) or s <= 0.0:
            return None
        return "pl" if (rng.random() < inv_pl / s) else "se"

    def sample_energy_loss(self, ch, E_s, rng, diag):
        """
        Draw omega from the channel DIIMFP, with the CDF truncated at
        omega_max = E_s - E_F.  Truncating BEFORE sampling (rather than
        sampling then rejecting) keeps the realised inelastic rate equal to
        1/imfp; the old code's post-hoc rejection quietly lengthened the
        effective IMFP and softened the stopping power.
        """
        w_all, cdf_all, has = self._channel_tables(ch)

        i, t = _bin_and_fraction(self.Egrid, self._clip_E(self._imfp_abscissa(E_s)))
        j = i + 1 if (t > 0.0 and rng.random() < t) else i
        if not has[j]:
            diag["omega_cdf_empty"] += 1
            return None

        wgrid = w_all[:, j]
        cdf = cdf_all[:, j]

        w_max = min(self.omega_max(E_s), float(wgrid[-1]))
        if w_max <= float(wgrid[0]):
            diag["omega_cdf_empty"] += 1
            return None

        c_max = float(np.interp(w_max, wgrid, cdf))
        if c_max <= 0.0 or not np.isfinite(c_max):
            diag["omega_cdf_empty"] += 1
            return None

        omega = _invert_cdf(cdf, wgrid, rng.random() * c_max)
        omega = min(omega, w_max)
        return omega if omega > 0.0 else None

    # ------------------------------------------------------------------
    # Inelastic: q sampling from the channel-resolved ELF
    # ------------------------------------------------------------------
    def _build_elf_channel_splines(self):
        md = self.material_data
        omega_h = np.asarray(md["omega"], float) / H2EV
        q_raw = np.asarray(md["q"], float)
        # ONE declared unit for the whole module (the old code read this key as
        # A^-1 in elf_spline() and as a0^-1 in elf_channel_splines()).
        q_a0inv = q_raw if self.cfg.q_unit == "a0^-1" else q_raw * A0_ANG
        if np.any(q_a0inv <= 0):
            raise ValueError("material_data['q'] must be strictly positive for log-q sampling")
        qlog = np.log(q_a0inv)

        elf_se = np.asarray(md["elf_se"], float)
        elf_pl = np.asarray(md["elf_pl"], float)
        if elf_se.shape != (omega_h.size, qlog.size):
            if elf_se.shape == (qlog.size, omega_h.size):
                elf_se = elf_se.T
                elf_pl = elf_pl.T
            else:
                raise ValueError(
                    f"ELF shape {elf_se.shape} matches neither (Nw, Nq) = "
                    f"({omega_h.size}, {qlog.size}) nor its transpose"
                )

        self._omega_h_grid = omega_h
        self._qlog_grid = qlog
        self._elf_spl = {
            "se": RectBivariateSpline(omega_h, qlog, elf_se, kx=1, ky=1),
            "pl": RectBivariateSpline(omega_h, qlog, elf_pl, kx=1, ky=1),
        }

    def mao_q_boundaries(self, omega):
        """
        Mao et al. 2008, Eq. (9): the edges of the single-electron-excitation
        region, in atomic units (a0^-1).

            q_-+ = -/+ k_F + sqrt(k_F^2 + 2 omega)

        For q in [q_-, q_+] the Fermi-sphere disk of Eq. (18) is non-empty and
        a single-electron excitation is kinematically allowed.  For q < q_- the
        loss is a plasmon (the plasmon dispersion line terminates at q_-).
        """
        wh = max(float(omega), 0.0) / H2EV
        kF = self.k_fermi_feg
        root = math.sqrt(kF * kF + 2.0 * wh)
        return root - kF, root + kF

    def qlog_bounds(self, E_s, omega):
        """
        Relativistic momentum-transfer bounds in log(q / a0^-1).

        Shinotsuka Eq. (2): the bounds are evaluated at T' = E_s, the
        VB-bottom-referenced energy -- NOT at E_s - E_F.  Using E_s - E_F
        here (as the previous version did) narrows the q window and biases
        every inelastic deflection angle.
        """
        k = _k_rel_au(E_s)
        kp = _k_rel_au(max(E_s - omega, 0.0))
        q_minus = abs(k - kp)
        q_plus = k + kp
        if q_minus <= 0.0 or q_plus <= q_minus:
            return None
        return math.log(q_minus), math.log(q_plus), k, kp

    def sample_q(self, ch, E_s, omega, rng, diag):
        """
        Sample q inside [q-, q+] from ELF_ch(omega, q), in log q (the variable
        the DIIMFP was integrated over).

        The CDF is built on a fresh grid spanning the kinematic window rather
        than on whatever tabulated q points happen to fall inside it, so a
        narrow window (small omega, high E) can no longer produce an empty
        interval and a dropped collision.
        """
        bounds = self.qlog_bounds(E_s, omega)
        if bounds is None:
            diag["q_cdf_empty"] += 1
            return None
        qm_log, qp_log, k, kp = bounds

        lo, hi = self._qlog_grid[0], self._qlog_grid[-1]
        if qm_log < lo or qp_log > hi:
            diag["q_window_clipped"] += 1
        qm_log = max(qm_log, lo)
        qp_log = min(qp_log, hi)
        if qp_log <= qm_log:
            diag["q_cdf_empty"] += 1
            return None

        qlog = np.linspace(qm_log, qp_log, self.cfg.n_q_sample)
        omega_h = omega / H2EV
        elf = np.asarray(
            self._elf_spl[ch].ev(np.full_like(qlog, omega_h), qlog), float
        )
        elf = np.nan_to_num(elf, nan=0.0, posinf=0.0, neginf=0.0)
        elf[elf < 0.0] = 0.0

        cdf = cumtrapz_numpy(elf, qlog)
        total = float(cdf[-1])
        if total <= 0.0 or not np.isfinite(total):
            diag["q_cdf_empty"] += 1
            return None

        q = math.exp(_invert_cdf(cdf / total, qlog, rng.random()))
        # Guaranteed by construction, but this is the invariant that the old
        # code violated silently through the acos() clamp.
        q = min(max(q, abs(k - kp)), k + kp)
        return q, k, kp

    # ------------------------------------------------------------------
    # Free-electron-gas target sampling
    # ------------------------------------------------------------------
    def sample_target_electron(self, omega, q_a0inv, rng, diag):
        """
        Sample the initial state of the struck electron for a single-particle
        (channel 'se') excitation of a free electron gas.

        Works in a local frame with z || q, all in Hartree atomic units:
            omega = (q^2 + 2 k_z q)/2   ->   k_z = (2 omega - q^2)/(2 q)
        with |k| <= k_F (occupied) and |k + q| >= k_F (Pauli blocking).
        The allowed set is an annulus in the k_z plane; sampling uniformly in
        area is the correct weight because the FEG matrix element does not
        depend on k.

        Returns (k_perp, k_z) in a0^-1, or None if the state is blocked.
        """
        if q_a0inv <= 0.0:
            return None
        omega_h = omega / H2EV
        kF = self.k_fermi_feg
        q = float(q_a0inv)

        kz = (2.0 * omega_h - q * q) / (2.0 * q)

        r_out_sq = kF * kF - kz * kz
        if r_out_sq <= 0.0:
            diag["se_blocked_pauli"] += 1
            return None
        r_out = math.sqrt(r_out_sq)

        r_in_sq = kF * kF - (kz + q) * (kz + q)
        r_in = math.sqrt(r_in_sq) if r_in_sq > 0.0 else 0.0
        if r_in >= r_out:
            diag["se_blocked_pauli"] += 1
            return None

        u = rng.random()
        r = math.sqrt(r_in * r_in + u * (r_out * r_out - r_in * r_in))
        return r, kz

    def sample_plasmon_target_energy(self, omega, rng):
        """
        Initial energy of the electron promoted by plasmon decay, drawn from
        the free-electron joint density of states  ~ sqrt(E (E + omega))  on
        [0, E_F].  Rejection sampling (the old code rebuilt a 400-point
        trapezoid CDF on every single plasmon event).
        """
        e_ref = self.e_fermi_feg if self.is_metal else self.e_vb
        e_ref = max(e_ref, 1e-6)
        f_max = math.sqrt(e_ref * (e_ref + omega))
        if f_max <= 0.0:
            return 0.0
        for _ in range(200):
            e = rng.random() * e_ref
            if rng.random() * f_max <= math.sqrt(e * (e + omega)):
                return e
        return 0.5 * e_ref


# --------------------------------------------------------------------------
# Electron
# --------------------------------------------------------------------------
@dataclass
class Secondary:
    """A secondary electron requested by an inelastic collision."""
    energy: float               # E_s, VB-bottom referenced
    uvw: list
    xyz: list
    generation: int


@dataclass
class Emission:
    """Record of one electron leaving the surface."""
    energy: float               # vacuum kinetic energy, eV
    uz: float                   # direction cosine w.r.t. -z (outward), in (0, 1]
    is_cascade: bool            # born in the cascade (vs. the incident electron)
    generation: int
    birth_depth: float


class Electron:
    """
    One electron.  `energy` is always E_s (VB-bottom referenced) while inside
    the solid, and becomes the vacuum kinetic energy after emission.
    """

    def __init__(self, sample: Sample, energy, xyz, uvw, generation=0,
                 is_cascade=False, rng=None, save_coordinates=False):
        self.sample = sample
        self.cfg = sample.cfg
        self.rng = rng

        self.energy = float(energy)
        self.initial_energy = self.energy
        self.xyz = [float(v) for v in xyz]
        self.uvw = [float(v) for v in uvw]
        self.generation = int(generation)
        self.is_cascade = bool(is_cascade)
        self.birth_depth = self.xyz[2]

        self.inside = True
        self.dead = False
        self.path_length = 0.0
        self.save_coordinates = bool(save_coordinates)
        self.coordinates = []
        self._record()

        self.Ui = sample.Ui
        self.e_fermi = sample.e_fermi
        self._inv_e, self._inv_i = sample.inverse_mfps(self.energy)

    # -- convenience ---------------------------------------------------
    def _record(self):
        if self.save_coordinates:
            self.coordinates.append([round(v, 3) for v in self.xyz] + [round(self.energy, 3)])

    def refresh_rates(self):
        """Evaluate the inverse MFPs once per step and cache them."""
        self._inv_e, self._inv_i = self.sample.inverse_mfps(self.energy)
        return self._inv_e + self._inv_i

    @property
    def iemfp(self):
        return self.sample.inverse_mfps(self.energy)[0]

    @property
    def iimfp(self):
        return self.sample.inverse_mfps(self.energy)[1]

    @property
    def itmfp(self):
        inv_e, inv_i = self.sample.inverse_mfps(self.energy)
        return inv_e + inv_i

    def check_alive(self):
        """
        Called at the TOP of every transport step (the old code only checked
        after an inelastic loss, which let a sub-barrier electron in a
        non-metal scatter elastically forever).
        """
        if (not np.isfinite(self.energy)) or self.energy <= 0.0:
            self.dead = True
            return
        if self.inside and self.energy <= self.Ui and not self.cfg.track_subbarrier:
            # Cannot escape a step barrier and cannot gain energy: terminal.
            self.dead = True

    # -- transport -----------------------------------------------------
    def travel(self):
        """
        Advance one free path.  Returns True if the step was truncated by the
        surface, in which case NO collision may be processed for this step.
        """
        rate = self.refresh_rates()
        if (not np.isfinite(rate)) or rate <= 0.0:
            self.dead = True
            return False

        s = -math.log(max(self.rng.random(), 1e-300)) / rate

        hit_surface = False
        if self.uvw[2] < -1e-15:
            s_to_surface = -self.xyz[2] / self.uvw[2]
            if 0.0 <= s_to_surface < s:
                s = s_to_surface
                hit_surface = True

        self.path_length += s
        self.xyz[0] += self.uvw[0] * s
        self.xyz[1] += self.uvw[1] * s
        self.xyz[2] += self.uvw[2] * s
        if hit_surface:
            self.xyz[2] = 0.0
        self._record()
        return hit_surface

    def escape(self):
        """
        Attempt to cross the planar step barrier of height U_i.
        Returns True if the electron left the solid; otherwise it has been
        specularly reflected and stays inside.
        """
        Es = self.energy
        ux, uy, uz = self.uvw

        E_perp = Es * uz * uz
        if Es <= self.Ui or E_perp <= self.Ui:
            self._reflect()
            return False

        T = barrier_transmission(E_perp, self.Ui, self.cfg)
        if T < 1.0 and self.rng.random() >= T:
            self._reflect()
            return False

        Ev = Es - self.Ui
        # Parallel momentum is conserved; E_perp > U_i already guarantees a
        # real outgoing u_z, so no separate total-internal-reflection test.
        scale = math.sqrt(Es / Ev)
        ux_out = ux * scale
        uy_out = uy * scale
        uz_out = -math.sqrt(max(1.0 - (ux_out * ux_out + uy_out * uy_out), 0.0))

        self.inside = False
        self.uvw = [ux_out, uy_out, uz_out]
        self.energy = Ev
        self.xyz[2] = 0.0
        self._record()
        return True

    def _reflect(self):
        self.uvw[2] = abs(self.uvw[2])
        self.xyz[2] = 0.0          # u_z > 0 now, so the surface test cannot re-trigger
        self._record()

    # -- collisions ----------------------------------------------------
    def choose_scattering_type(self):
        # Reuses the rates computed by travel() for this very step: the
        # branching ratio must be the one that generated the free path.
        inv_e, inv_i = self._inv_e, self._inv_i
        total = inv_e + inv_i
        if total <= 0.0 or not np.isfinite(total):
            self.dead = True
            return None
        return "elastic" if (self.rng.random() < inv_e / total) else "inelastic"

    def scatter(self, diag):
        """
        Perform one collision.  Returns a `Secondary` to be queued, or None.
        All per-collision state is local: nothing can leak into the next event.
        """
        kind = self.choose_scattering_type()
        if kind is None:
            return None

        if kind == "elastic":
            diag["elastic_events"] += 1
            theta = self.sample.sample_elastic_theta(self.energy, self.rng)
            phi = 2.0 * math.pi * self.rng.random()
            self.uvw = rotate_direction(self.uvw, theta, phi)
            return None

        return self._inelastic(diag)

    def _inelastic(self, diag):
        smp = self.sample
        rng = self.rng

        ch = smp.choose_channel(self.energy, rng)
        if ch is None:
            return None

        omega = smp.sample_energy_loss(ch, self.energy, rng, diag)
        if omega is None:
            return None

        qres = smp.sample_q(ch, self.energy, omega, rng, diag)
        if qres is None:
            return None
        q, k, kp = qres

        diag["inelastic_events"] += 1

        # Mao Eq. (9): the SE MECHANISM is decided by where (omega, q) sits
        # relative to the single-electron-excitation window, NOT by which
        # table the pair happened to be drawn from.  The transport (energy
        # loss, deflection) is unaffected because the tables sum to the total;
        # only the secondary-electron construction changes.
        if self.cfg.se_channel_rule == "mao":
            q_minus, q_plus = smp.mao_q_boundaries(omega)
            mech = "se" if (q_minus <= q <= q_plus) else "pl"
            if mech != ch:
                diag["channel_reclassified"] += 1
            ch = mech

        # --- projectile deflection (relativistic momenta, same as the bounds)
        cos_theta_p = (k * k + kp * kp - q * q) / (2.0 * k * kp)
        cos_theta_p = min(1.0, max(-1.0, cos_theta_p))
        theta_p = math.acos(cos_theta_p)
        phi_p = 2.0 * math.pi * rng.random()

        uvw_before = list(self.uvw)          # <-- the SE frame, captured BEFORE rotating
        self.uvw = rotate_direction(uvw_before, theta_p, phi_p)
        self.energy -= omega
        self._record()

        # --- secondary electron
        if ch == "se":
            return self._secondary_from_binary_encounter(
                uvw_before, theta_p, phi_p, omega, q, k, kp, diag
            )
        return self._secondary_from_plasmon(
            uvw_before, theta_p, phi_p, omega, q, k, kp, diag
        )

    # -- secondary construction ---------------------------------------
    def _q_hat(self, uvw_before, theta_p, phi_p, k, kp):
        """
        Unit vector along the momentum transfer q = k - k'.

        In the frame with z || k:
            q_perp = k' sin(theta_p),  q_z = k - k' cos(theta_p),  azimuth = phi_p + pi
        """
        theta_q = math.atan2(kp * math.sin(theta_p), k - kp * math.cos(theta_p))
        return rotate_direction(uvw_before, theta_q, (phi_p + math.pi) % (2.0 * math.pi))

    def _secondary_from_binary_encounter(self, uvw_before, theta_p, phi_p,
                                         omega, q, k, kp, diag):
        smp = self.sample
        target = smp.sample_target_electron(omega, q, self.rng, diag)
        if target is None:
            if self.cfg.on_pauli_block == "drop":
                return None
            # The ELF said this (omega, q) carries single-particle strength but
            # the FEG kinematics say no state is available -- i.e. the model
            # used to disperse the ELF and the model used here disagree.
            # Dropping the secondary would destroy the excitation while keeping
            # the energy loss, so fall back to the DOS construction instead.
            diag["se_pauli_fallback"] += 1
            return self._secondary_from_plasmon(
                uvw_before, theta_p, phi_p, omega, q, k, kp, diag
            )
        r, kz = target

        # Final state of the struck electron, in the frame with z || q:
        #   k_f = k_i + q z_hat   ->   E_f = E_i + omega  exactly.
        kfz = kz + q
        E_se = 0.5 * (r * r + kfz * kfz) * H2EV

        if self.cfg.se_direction_model == "isotropic":
            uvw = _isotropic_direction(self.rng)
        else:
            q_hat = self._q_hat(uvw_before, theta_p, phi_p, k, kp)
            psi = 2.0 * math.pi * self.rng.random()      # azimuth about q
            theta_f = math.atan2(r, kfz)
            uvw = rotate_direction(q_hat, theta_f, psi)

        return Secondary(E_se, uvw, list(self.xyz), self.generation + 1)

    def _secondary_from_plasmon(self, uvw_before, theta_p, phi_p,
                                omega, q, k, kp, diag):
        """
        Plasmon decay.  The plasmon carries q << k_F and decays by Landau
        damping at a wavevector uncorrelated with the incident direction, so
        the emitted direction is taken isotropic by default; the initial state
        is drawn from the free-electron joint DOS.
        """
        E_i = self.sample.sample_plasmon_target_energy(omega, self.rng)
        E_se = E_i + omega

        if self.cfg.plasmon_se_direction == "isotropic":
            uvw = _isotropic_direction(self.rng)
        else:
            q_hat = self._q_hat(uvw_before, theta_p, phi_p, k, kp)
            k_i = math.sqrt(max(2.0 * E_i / H2EV, 0.0))
            mu = 2.0 * self.rng.random() - 1.0
            kz = k_i * mu + q
            r = k_i * math.sqrt(max(1.0 - mu * mu, 0.0))
            psi = 2.0 * math.pi * self.rng.random()
            uvw = rotate_direction(q_hat, math.atan2(r, kz), psi)

        return Secondary(E_se, uvw, list(self.xyz), self.generation + 1)


# --------------------------------------------------------------------------
# Trajectory  (ONE implementation, shared by the serial and parallel paths)
# --------------------------------------------------------------------------
@dataclass
class TrajectoryResult:
    tey: int = 0
    sey_cascade: int = 0        # split by "was born in the cascade"
    bse_cascade: int = 0
    sey_50ev: int = 0           # split by the conventional 50 eV emission cut
    bse_50ev: int = 0
    emissions: list = field(default_factory=list)
    tracks: list = field(default_factory=list)
    diagnostics: Diagnostics = field(default_factory=Diagnostics)


def incident_direction(E0, sample: Sample, angle_rad):
    """
    Direction of the primary just inside the surface.

    The barrier accelerates the electron from E0 to E_s = E0 + U_i while
    conserving parallel momentum, so it is refracted towards the normal:

        sin(theta_solid) = sqrt(E0 / E_s) * sin(theta_vacuum)

    The previous version added U_i to the energy but kept the vacuum angle,
    which only agrees at normal incidence.
    """
    E_s = E0 + sample.Ui
    sin_in = math.sqrt(max(E0, 0.0) / E_s) * math.sin(angle_rad)
    sin_in = min(sin_in, 1.0)
    cos_in = math.sqrt(max(1.0 - sin_in * sin_in, 0.0))
    return E_s, [sin_in, 0.0, cos_in]


def simulate_trajectory(sample: Sample, E0, angle_rad, rng, track=False):
    """Transport one primary electron and its whole cascade."""
    cfg = sample.cfg
    res = TrajectoryResult()
    diag = res.diagnostics

    E_s, uvw0 = incident_direction(float(E0), sample, angle_rad)
    queue = [Electron(sample, E_s, [0.0, 0.0, 0.0], uvw0,
                      generation=0, is_cascade=False, rng=rng,
                      save_coordinates=track)]

    i = 0
    while i < len(queue):
        e = queue[i]
        steps = 0

        while True:
            e.check_alive()
            if e.dead:
                break

            steps += 1
            if steps > cfg.max_steps_per_electron:
                diag["step_limit_hit"] += 1
                e.dead = True
                break

            hit_surface = e.travel()
            if e.dead:
                break

            if hit_surface:
                diag["surface_encounters"] += 1
                if e.escape():
                    diag["escapes"] += 1
                    res.tey += 1
                    if e.is_cascade:
                        res.sey_cascade += 1
                    else:
                        res.bse_cascade += 1
                    if e.energy < cfg.bse_cutoff_ev:
                        res.sey_50ev += 1
                    else:
                        res.bse_50ev += 1
                    if cfg.collect_spectra:
                        res.emissions.append(
                            Emission(e.energy, abs(e.uvw[2]), e.is_cascade,
                                     e.generation, e.birth_depth)
                        )
                    break
                diag["internal_reflections"] += 1
                # KEY FIX: a step truncated at the surface produced no
                # collision.  Draw a fresh free path instead of forcing one.
                continue

            secondary = e.scatter(diag)
            if secondary is None:
                continue

            if secondary.generation > cfg.max_generation:
                diag["generation_limit_hit"] += 1
                continue
            if len(queue) >= cfg.max_secondaries_per_trajectory:
                continue
            if secondary.energy <= sample.Ui and not cfg.track_subbarrier:
                # Cannot escape a step barrier; tracking it changes no yield.
                diag["se_below_barrier"] += 1
                continue

            diag["se_created"] += 1
            queue.append(
                Electron(sample, secondary.energy, secondary.xyz, secondary.uvw,
                         generation=secondary.generation, is_cascade=True,
                         rng=rng, save_coordinates=track)
            )

        if track:
            res.tracks.append(e.coordinates)
        queue[i] = None            # release the cascade as we go
        i += 1

    return res


# --- multiprocessing plumbing --------------------------------------------
_G = None


def _init_worker(sample_name, db_path, config, angle_rad, track):
    global _G
    from types import SimpleNamespace
    _G = SimpleNamespace(
        sample=Sample(sample_name, db_path=db_path, config=config),
        angle=float(angle_rad),
        track=bool(track),
    )


def _worker_task(args):
    E0, seed_entropy = args
    rng = np.random.default_rng(np.random.SeedSequence(seed_entropy))
    r = simulate_trajectory(_G.sample, E0, _G.angle, rng, track=_G.track)
    return (r.tey, r.sey_cascade, r.bse_cascade, r.sey_50ev, r.bse_50ev,
            r.emissions, r.tracks if _G.track else None, dict(r.diagnostics))


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
class SEEMC:
    """
    Yields as a function of primary energy.

    `cb_ref` is accepted for backwards compatibility but is not used by the
    transport: the previous version stored it on every Electron and never read
    it.  Pass a `MCConfig` instead to change model options.
    """

    def __init__(self, energy_array, sample_name, angle, n_traj,
                 cb_ref=False, track=False, db_path="MaterialDatabase.pkl",
                 config: Optional[MCConfig] = None, seed=12345):
        self.energy_array = np.asarray(energy_array, dtype=float)
        self.cfg = config or MCConfig()
        self.cfg.validate()
        self.sample = Sample(sample_name, db_path=db_path, config=self.cfg)
        self.n_trajectories = int(n_traj)
        self.incident_angle = float(angle)
        self.db_path = db_path
        self.track_trajectories = bool(track)
        self.cb_ref = cb_ref
        self.seed = int(seed)

        n = len(self.energy_array)
        self.tey = np.zeros(n)
        self.sey = np.zeros(n)          # cascade-flag split (delta)
        self.bse = np.zeros(n)
        self.sey_50ev = np.zeros(n)     # conventional 50 eV split
        self.bse_50ev = np.zeros(n)
        self.tey_err = np.zeros(n)
        self.sey_err = np.zeros(n)
        self.bse_err = np.zeros(n)

        self.emissions = [[] for _ in range(n)]
        self.tracks = []
        self.diagnostics = Diagnostics()

    def _seed_for(self, k, traj):
        """Deterministic, PID-independent, collision-free by construction."""
        return [self.seed, int(k), int(traj)]

    # ------------------------------------------------------------------
    def run_simulation(self, use_parallel=False, progress=True):
        """
        Run all energies.  NOTE: `use_parallel=True` uses the 'spawn' start
        method, so the calling code must be guarded:

            if __name__ == "__main__":
                mc.run_simulation(use_parallel=True)

        Serial and parallel runs with the same `seed` give bitwise-identical
        yields: the per-trajectory stream is SeedSequence([seed, k, traj]),
        which does not depend on process id or completion order.
        """
        import time
        t0 = time.time()

        try:
            from tqdm import tqdm
        except ImportError:                       # pragma: no cover
            def tqdm(x, **kw):
                return x

        n_traj = self.n_trajectories

        if use_parallel:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            nproc = mp.cpu_count()
            chunksize = max(1, n_traj // (nproc * 8))
            pool = ctx.Pool(
                processes=nproc,
                initializer=_init_worker,
                initargs=(self.sample.name, self.db_path, self.cfg,
                          self.incident_angle, self.track_trajectories),
            )
        else:
            pool = None

        try:
            for k, E0 in enumerate(self.energy_array):
                acc = np.zeros(5)          # tey, sey, bse, sey50, bse50
                acc_sq = np.zeros(5)
                tracks_E = [] if self.track_trajectories else None

                if pool is None:
                    it = (
                        self._run_one(E0, k, traj)
                        for traj in range(n_traj)
                    )
                else:
                    tasks = ((float(E0), self._seed_for(k, traj))
                             for traj in range(n_traj))
                    it = pool.imap_unordered(_worker_task, tasks, chunksize=chunksize)

                iterator = tqdm(it, total=n_traj, desc=f"E={E0:.1f} eV") if progress else it

                for tey, sey, bse, sey50, bse50, emis, trk, diag in iterator:
                    vals = np.array([tey, sey, bse, sey50, bse50], dtype=float)
                    acc += vals
                    acc_sq += vals * vals
                    if self.cfg.collect_spectra:
                        self.emissions[k].extend(emis)
                    if self.track_trajectories and trk is not None:
                        tracks_E.append(trk)
                    self.diagnostics.add(diag)

                mean = acc / n_traj
                var = np.maximum(acc_sq / n_traj - mean ** 2, 0.0)
                sem = np.sqrt(var / n_traj)

                self.tey[k], self.sey[k], self.bse[k] = mean[0], mean[1], mean[2]
                self.sey_50ev[k], self.bse_50ev[k] = mean[3], mean[4]
                self.tey_err[k], self.sey_err[k], self.bse_err[k] = sem[0], sem[1], sem[2]

                if self.track_trajectories:
                    self.tracks.append(tracks_E)
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        print(f"Done in {time.time() - t0:.1f} s")
        return self

    def _run_one(self, E0, k, traj):
        rng = np.random.default_rng(np.random.SeedSequence(self._seed_for(k, traj)))
        r = simulate_trajectory(self.sample, float(E0), self.incident_angle,
                                rng, track=self.track_trajectories)
        return (r.tey, r.sey_cascade, r.bse_cascade, r.sey_50ev, r.bse_50ev,
                r.emissions, r.tracks if self.track_trajectories else None,
                dict(r.diagnostics))

    # ------------------------------------------------------------------
    def emission_spectrum(self, k, bins=100, e_max=None):
        """Energy distribution of emitted electrons at energy index k."""
        e = np.array([em.energy for em in self.emissions[k]], dtype=float)
        if e.size == 0:
            return np.zeros(bins), np.linspace(0, 1, bins + 1)
        e_max = e_max if e_max is not None else float(np.percentile(e, 99.5))
        counts, edges = np.histogram(e, bins=bins, range=(0.0, e_max))
        return counts / self.n_trajectories, edges

    def summary(self):
        lines = [f"{self.sample.name}: {self.n_trajectories} trajectories/energy",
                 f"{'E0 (eV)':>9} {'TEY':>10} {'+/-':>9} "
                 f"{'SEY(<50eV)':>11} {'BSE(>50eV)':>11}"]
        for k, E0 in enumerate(self.energy_array):
            lines.append(
                f"{E0:9.1f} {self.tey[k]:10.4f} {self.tey_err[k]:9.4f} "
                f"{self.sey_50ev[k]:11.4f} {self.bse_50ev[k]:11.4f}"
            )
        lines.append("")
        lines.append(self.diagnostics.report(self.n_trajectories * len(self.energy_array)))
        return "\n".join(lines)

    def plot_yield(self, use_50ev_split=True):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.errorbar(self.energy_array, self.tey, yerr=self.tey_err,
                     label="TEY", marker="o", capsize=3)
        if use_50ev_split:
            plt.plot(self.energy_array, self.sey_50ev, "s--", label="SEY (<50 eV)")
            plt.plot(self.energy_array, self.bse_50ev, "^--", label="BSE (>50 eV)")
        else:
            plt.errorbar(self.energy_array, self.sey, yerr=self.sey_err,
                         label="SEY (cascade)", marker="s")
            plt.errorbar(self.energy_array, self.bse, yerr=self.bse_err,
                         label="BSE (primary)", marker="^")
        plt.xlabel("Primary energy (eV)")
        plt.ylabel("Yield (electrons/primary)")
        plt.title(self.sample.name)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()


# ==========================================================================
# Validation.  Run these once per material before trusting a yield curve --
# each one targets a specific class of bug that is invisible in the final
# SEY curve but shifts it by tens of percent.
# ==========================================================================
def check_null_collisions(sample: Sample, E_s, n=200_000, seed=1):
    """
    (i) Every collision the transport loop starts must end in a real event.

    A "null" collision -- free path consumed, nothing happened -- silently
    lengthens the effective mean free path.  The old sampler produced one
    every time omega or q sampling failed.  This should report 0.
    """
    rng = np.random.default_rng(seed)
    diag = Diagnostics()
    e = Electron(sample, E_s, [0, 0, 1e3], [0, 0, 1.0], rng=rng)
    for _ in range(n):
        e.energy = E_s
        e.uvw = [0.0, 0.0, 1.0]
        e.refresh_rates()
        e.scatter(diag)

    e.energy = E_s          # scatter() lowered it; reset before reading the rates
    real = diag["elastic_events"] + diag["inelastic_events"]
    null_frac = 1.0 - real / n
    inel_frac = diag["inelastic_events"] / n
    expected_inel = e.iimfp / e.itmfp if e.itmfp > 0 else float("nan")
    return {
        "null_fraction": null_frac,
        "inelastic_fraction_measured": inel_frac,
        "inelastic_fraction_expected": expected_inel,
        "effective_imfp_inflation": (1.0 / (1.0 - null_frac)) if null_frac < 1 else np.inf,
        "diagnostics": dict(diag),
    }


def check_energy_loss_spectrum(sample: Sample, E_s, n=200_000, bins=120, seed=2):
    """
    (ii) The sampled energy-loss distribution must reproduce the tabulated
    DIIMFP, truncated at omega_max = E_s - E_F.

    Compared through the CDF (Kolmogorov-Smirnov distance) and through the
    mean energy loss rather than through a binned density: the omega tables
    are log-spaced, so any linear histogram disagrees at the first and last
    bin for reasons that have nothing to do with the sampler.  The mean loss
    is the quantity that actually propagates into the yield -- it is the
    stopping power per collision.
    """
    rng = np.random.default_rng(seed)
    diag = Diagnostics()
    losses = []
    for _ in range(n):
        ch = sample.choose_channel(E_s, rng)
        if ch is None:
            continue
        w = sample.sample_energy_loss(ch, E_s, rng, diag)
        if w is not None:
            losses.append(w)
    losses = np.sort(np.asarray(losses))
    if losses.size == 0:
        return {"n_sampled": 0, "ks_distance": np.nan, "mean_loss_error": np.nan}

    # Reference pdf: linear blend of the two bracketing energy bins, exactly
    # what the stochastic bin choice reproduces on average.
    i, t = _bin_and_fraction(sample.Egrid, sample._clip_E(sample._imfp_abscissa(E_s)))
    w_max = sample.omega_max(E_s)
    grid = np.unique(np.concatenate([
        np.asarray(sample.material_data["diimfp_se"], float)[:, 0, i],
        np.linspace(0.0, w_max, 2000),
    ]))
    grid = grid[(grid >= 0.0) & (grid <= w_max)]

    pdf = np.zeros_like(grid)
    for key in ("diimfp_se", "diimfp_pl"):
        tab = np.asarray(sample.material_data[key], float)
        lo = np.interp(grid, tab[:, 0, i], tab[:, 1, i], left=0.0, right=0.0)
        hi = np.interp(grid, tab[:, 0, i + 1], tab[:, 1, i + 1], left=0.0, right=0.0)
        pdf += (1.0 - t) * lo + t * hi

    cdf_ref = cumtrapz_numpy(pdf, grid)
    total = float(cdf_ref[-1])
    if total <= 0:
        return {"n_sampled": int(losses.size), "ks_distance": np.nan,
                "mean_loss_error": np.nan}
    cdf_ref /= total

    emp = np.arange(1, losses.size + 1) / losses.size
    ref_at = np.interp(losses, grid, cdf_ref)
    ks = float(np.max(np.abs(emp - ref_at)))
    ks_crit = 1.36 / math.sqrt(losses.size)          # 95% critical value

    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    mean_ref = float(trapz(pdf * grid, grid) / total)
    mean_mc = float(losses.mean())

    counts, edges = np.histogram(losses, bins=bins, range=(0.0, w_max), density=True)
    return {"n_sampled": int(losses.size),
            "ks_distance": ks, "ks_critical_95": ks_crit, "ks_pass": ks < ks_crit,
            "mean_loss_mc": mean_mc, "mean_loss_table": mean_ref,
            "mean_loss_error": (mean_mc - mean_ref) / mean_ref,
            "omega": 0.5 * (edges[1:] + edges[:-1]), "sampled": counts,
            "pdf_grid": grid, "pdf_ref": pdf / total}


def check_escape_probability(sample: Sample, E_s, n=200_000, seed=3):
    """
    (iii) Barrier test, decoupled from transport.  Release electrons at the
    surface with isotropic directions and compare the escaped fraction with

        P = (1/2) * integral_0^1 T(E_s mu^2) d mu

    This exercises the transmission coefficient, the parallel-momentum
    refraction and the reflection bookkeeping without any table lookups.
    """
    rng = np.random.default_rng(seed)
    escaped = 0
    for _ in range(n):
        uvw = _isotropic_direction(rng)
        e = Electron(sample, E_s, [0.0, 0.0, 0.0], uvw, rng=rng)
        if uvw[2] < 0 and e.escape():
            escaped += 1

    mu = np.linspace(0.0, 1.0, 20001)
    Eperp = E_s * mu * mu
    T = np.zeros_like(mu)
    ok = Eperp > sample.Ui
    T[ok] = [barrier_transmission(e, sample.Ui, sample.cfg) for e in Eperp[ok]]
    integ = np.trapezoid(T, mu) if hasattr(np, "trapezoid") else np.trapz(T, mu)
    analytic = 0.5 * integ

    mc = escaped / n
    err = math.sqrt(max(mc * (1 - mc), 0.0) / n)
    return {"monte_carlo": mc, "analytic": analytic, "sigma": err,
            "pulls": (mc - analytic) / err if err > 0 else np.nan}


def check_collision_kinematics(sample: Sample, E_s, n=20_000, seed=4):
    """
    (iv) Energy and momentum bookkeeping of the inelastic vertex.

    Checks, per event:
      * q lies inside [|k - k'|, k + k']            (q-bound consistency)
      * |k u_before - k' u_after| equals q          (the q-hat construction)
      * E_SE = E_i + omega for the binary-encounter channel
    """
    rng = np.random.default_rng(seed)
    diag = Diagnostics()
    worst_q, worst_vec, worst_e = 0.0, 0.0, 0.0
    n_checked = 0

    for _ in range(n):
        e = Electron(sample, E_s, [0.0, 0.0, 1e3], [0.0, 0.0, 1.0], rng=rng)
        ch = sample.choose_channel(E_s, rng)
        if ch is None:
            continue
        omega = sample.sample_energy_loss(ch, E_s, rng, diag)
        if omega is None:
            continue
        qres = sample.sample_q(ch, E_s, omega, rng, diag)
        if qres is None:
            continue
        q, k, kp = qres
        n_checked += 1

        worst_q = max(worst_q, max(abs(k - kp) - q, q - (k + kp), 0.0) / q)

        cos_tp = min(1.0, max(-1.0, (k * k + kp * kp - q * q) / (2 * k * kp)))
        tp = math.acos(cos_tp)
        phip = 2 * math.pi * rng.random()
        u0 = [0.0, 0.0, 1.0]
        u1 = rotate_direction(u0, tp, phip)
        qvec = np.array(u0) * k - np.array(u1) * kp
        worst_vec = max(worst_vec, abs(np.linalg.norm(qvec) - q) / q)

        q_hat = e._q_hat(u0, tp, phip, k, kp)
        worst_vec = max(worst_vec, float(np.linalg.norm(
            qvec / np.linalg.norm(qvec) - np.array(q_hat))))

        if ch == "se":
            t = sample.sample_target_electron(omega, q, rng, diag)
            if t is not None:
                r, kz = t
                E_i = 0.5 * (r * r + kz * kz) * H2EV
                E_f = 0.5 * (r * r + (kz + q) ** 2) * H2EV
                worst_e = max(worst_e, abs(E_f - (E_i + omega)) / max(omega, 1e-9))

    return {"n_checked": n_checked,
            "max_q_bound_violation": worst_q,
            "max_q_vector_error": worst_vec,
            "max_energy_closure_error": worst_e,
            "diagnostics": dict(diag)}


def run_all_checks(sample: Sample, energies=(50.0, 200.0, 1000.0), verbose=True):
    out = {}
    if verbose:
        print(sample.consistency_report())
        print()
    for E_vac in energies:
        E_s = E_vac + sample.Ui
        res = {
            "null": check_null_collisions(sample, E_s, n=50_000),
            "loss": check_energy_loss_spectrum(sample, E_s, n=50_000),
            "escape": check_escape_probability(sample, E_s, n=50_000),
            "kinematics": check_collision_kinematics(sample, E_s, n=5_000),
        }
        out[E_vac] = res
        if verbose:
            print(f"E_vac = {E_vac:g} eV  (E_s = {E_s:g} eV)")
            print(f"  null collision fraction     : {res['null']['null_fraction']:.4%}")
            print(f"  inelastic fraction meas/exp : "
                  f"{res['null']['inelastic_fraction_measured']:.4f} / "
                  f"{res['null']['inelastic_fraction_expected']:.4f}")
            print(f"  loss spectrum KS / crit     : "
                  f"{res['loss']['ks_distance']:.4f} / {res['loss']['ks_critical_95']:.4f}"
                  f"  {'PASS' if res['loss']['ks_pass'] else 'FAIL'}")
            print(f"  mean loss MC / table        : "
                  f"{res['loss']['mean_loss_mc']:.3f} / {res['loss']['mean_loss_table']:.3f} eV "
                  f"({res['loss']['mean_loss_error']:+.3%})")
            print(f"  escape prob MC / analytic   : "
                  f"{res['escape']['monte_carlo']:.5f} / {res['escape']['analytic']:.5f} "
                  f"({res['escape']['pulls']:+.2f} sigma)")
            print(f"  q-bound violation           : {res['kinematics']['max_q_bound_violation']:.2e}")
            print(f"  q-vector construction error : {res['kinematics']['max_q_vector_error']:.2e}")
            print(f"  energy closure error        : {res['kinematics']['max_energy_closure_error']:.2e}")
            print()
    return out


def check_channel_boundaries(sample: Sample, omegas=None, threshold=1e-3):
    """
    (v) Are the DB's channel-resolved ELFs consistent with the Fermi energy
    the DB reports?

    Mao et al. 2008 Eq. (9) puts single-electron excitation in
    q- <= q <= q+ with q_-+ = -/+ k_F + sqrt(k_F^2 + 2 omega).  That window is
    exactly where the Fermi-sphere disk of Eq. (18) is non-empty, so:

      * elf_se strength at q < q-  ->  losses with NO available target state.
        Under se_channel_rule='table' those become dropped or fallback
        secondaries; they are really plasmon losses that the FPA decomposition
        assigned to the single-particle channel because it integrates over a
        scanning omega_p whose k_F(omega_p) exceeds the material's k_F.
      * elf_pl strength at q > q-  ->  the mirror image.

    This routine measures the actual support of each channel table and inverts
    the boundary to recover the k_F that the tables were built with:

        from the lower edge of elf_se:  k_F = (2 omega - q-^2) / (2 q-)
        from the upper edge of elf_se:  k_F = (q+^2 - 2 omega) / (2 q+)

    A k_F_eff that is consistent across omega but differs from the DB's
    sqrt(2 E_F) tells you exactly which value to put in
    MCConfig.feg_fermi_energy -- or, better, that you should use
    se_channel_rule='mao' and stop relying on the split.
    """
    q = np.exp(sample._qlog_grid)                     # a0^-1
    w_grid = sample._omega_h_grid * H2EV              # eV
    if omegas is None:
        lo = max(float(w_grid[0]), 1.0)
        hi = min(float(w_grid[-1]), 200.0)
        omegas = np.geomspace(lo, hi, 12)

    kF_db = sample.k_fermi_feg
    rows = []
    for w in omegas:
        j = int(np.argmin(np.abs(w_grid - w)))
        se = np.asarray(sample._elf_spl["se"].ev(
            np.full_like(q, w_grid[j] / H2EV), sample._qlog_grid), float)
        pl = np.asarray(sample._elf_spl["pl"].ev(
            np.full_like(q, w_grid[j] / H2EV), sample._qlog_grid), float)
        se = np.clip(np.nan_to_num(se), 0.0, None)
        pl = np.clip(np.nan_to_num(pl), 0.0, None)

        qm_th, qp_th = sample.mao_q_boundaries(w_grid[j])

        row = {"omega": float(w_grid[j]), "q_minus_theory": qm_th,
               "q_plus_theory": qp_th, "kF_db": kF_db}

        if se.max() > 0:
            sup = q[se > threshold * se.max()]
            qlo, qhi = float(sup[0]), float(sup[-1])
            wh = w_grid[j] / H2EV
            row["se_q_lo"] = qlo
            row["se_q_hi"] = qhi
            row["kF_from_lower_edge"] = (2 * wh - qlo * qlo) / (2 * qlo)
            row["kF_from_upper_edge"] = (qhi * qhi - 2 * wh) / (2 * qhi)
            below = q < qm_th
            trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            tot = trapz(se, np.log(q))
            row["se_frac_below_qminus"] = (
                float(trapz(se[below], np.log(q[below])) / tot)
                if below.sum() > 1 and tot > 0 else 0.0
            )
        if pl.max() > 0:
            trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            above = q > qm_th
            tot = trapz(pl, np.log(q))
            row["pl_frac_above_qminus"] = (
                float(trapz(pl[above], np.log(q[above])) / tot)
                if above.sum() > 1 and tot > 0 else 0.0
            )
        rows.append(row)
    return rows


def se_strength_lost(sample: Sample, E_s, n_omega=60):
    """
    DIIMFP-weighted estimate of how many single-particle secondaries the
    'table' channel rule loses at primary energy E_s.

    For each omega, the fraction of elf_se strength sitting at q < q-(omega)
    is weighted by that omega's contribution to the inelastic rate.  Under
    se_channel_rule='table' with on_pauli_block='drop' (the original code) this
    fraction of 'se' events produced NO secondary at all.
    """
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    i, _ = _bin_and_fraction(sample.Egrid, sample._clip_E(sample._imfp_abscissa(E_s)))
    tab = np.asarray(sample.material_data["diimfp_se"], float)
    w_tab, d_tab = tab[:, 0, i], tab[:, 1, i]

    w_max = sample.omega_max(E_s)
    if w_max <= 0:
        return float("nan"), float("nan")
    omegas = np.geomspace(max(w_tab[w_tab > 0].min(), 0.5), w_max, n_omega)
    q = np.exp(sample._qlog_grid)

    weight, lost = [], []
    for w in omegas:
        elf = np.clip(np.nan_to_num(np.asarray(
            sample._elf_spl["se"].ev(np.full_like(q, w / H2EV), sample._qlog_grid),
            float)), 0.0, None)
        tot = trapz(elf, sample._qlog_grid)
        if tot <= 0:
            weight.append(0.0); lost.append(0.0); continue
        qm, _ = sample.mao_q_boundaries(w)
        m = q < qm
        below = trapz(elf[m], sample._qlog_grid[m]) if m.sum() > 1 else 0.0
        weight.append(float(np.interp(w, w_tab, d_tab, left=0.0, right=0.0)))
        lost.append(below / tot)

    weight = np.asarray(weight); lost = np.asarray(lost)
    denom = trapz(weight, omegas)
    frac = float(trapz(weight * lost, omegas) / denom) if denom > 0 else float("nan")

    inv_se = float(np.interp(sample._clip_E(sample._imfp_abscissa(E_s)),
                             sample.Egrid, sample.material_data["inv_imfp_se"]))
    inv_pl = float(np.interp(sample._clip_E(sample._imfp_abscissa(E_s)),
                             sample.Egrid, sample.material_data["inv_imfp_pl"]))
    share = inv_se / max(inv_se + inv_pl, 1e-30)
    return frac, frac * share

def report_channel_boundaries(sample: Sample, energies=(100.0, 500.0, 2000.0), **kw):
    rows = check_channel_boundaries(sample, **kw)
    kF_db = sample.k_fermi_feg
    print(f"Channel-boundary check for {sample.name}")
    print(f"  DB e_fermi = {sample.e_fermi_feg:.3f} eV  ->  k_F = {kF_db:.4f} a0^-1")
    print(f"  {'omega':>8} {'q-(th)':>8} {'se_q_lo':>8} {'kF(lo)':>8} "
          f"{'kF(hi)':>8} {'se<q-':>7} {'pl>q-':>7}")
    kfs, ws = [], []
    for r in rows:
        lo = r.get("se_q_lo", float("nan"))
        klo = r.get("kF_from_lower_edge", float("nan"))
        khi = r.get("kF_from_upper_edge", float("nan"))
        fb = r.get("se_frac_below_qminus", 0.0)
        fa = r.get("pl_frac_above_qminus", 0.0)
        if np.isfinite(klo) and klo > 0:
            kfs.append(klo)
            ws.append(r["omega"])
        print(f"  {r['omega']:8.2f} {r['q_minus_theory']:8.4f} {lo:8.4f} "
              f"{klo:8.4f} {khi:8.4f} {fb:7.1%} {fa:7.1%}")

    if len(kfs) >= 4:
        kfs = np.asarray(kfs)
        ws = np.asarray(ws)
        spread = float(kfs.max() / max(kfs.min(), 1e-12))
        slope = float(np.polyfit(np.log(ws), np.log(kfs), 1)[0])
        print()
        print(f"  k_F inferred from the elf_se lower edge: "
              f"{kfs.min():.2f} - {kfs.max():.2f} a0^-1 "
              f"(spread {spread:.1f}x, ~omega^{slope:.2f})")
        if spread > 1.5:
            print()
            print("  The inferred k_F is NOT constant, so the elf_se lower edge is not a")
            print("  pair-continuum edge, and NO single feg_fermi_energy can repair the")
            print("  table split. This is the expected signature of an FPA database:")
            print("  Mao Eq. (8) integrates over a scanning omega_p, so the support of")
            print("  elf_se is a UNION of continua with different k_F(omega_p), not one")
            print("  continuum.")
            print("    => Keep se_channel_rule='mao' (the default). It classifies each")
            print("       sampled (omega, q) by Mao Eq. (9) and cannot lose a secondary.")
            print("    => Do NOT set feg_fermi_energy from this table.")
        else:
            ef = 0.5 * float(np.median(kfs)) ** 2 * H2EV
            print(f"  Consistent with a single continuum: consider "
                  f"MCConfig(feg_fermi_energy={ef:.2f})")

    print()
    print("  DIIMFP-weighted single-particle strength at q < q-")
    print("  (this fraction produced NO secondary under 'table' + 'drop'):")
    for E_vac in energies:
        f_se, f_all = se_strength_lost(sample, E_vac + sample.Ui)
        print(f"    E0 = {E_vac:7.0f} eV : {f_se:6.1%} of 'se' events, "
              f"{f_all:6.1%} of all inelastic events")
    return rows
