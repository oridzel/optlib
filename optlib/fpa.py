import numpy as np
import math
import time
from scipy.interpolate import interp1d, RectBivariateSpline
from joblib import Parallel, delayed
from optlib.constants import h2ev, a0

# =====================================================================
# 1. STANDALONE LINDHARD MATH & WORKERS (Required for Joblib parallelization)
# =====================================================================

def k_f(omega_pl):
    return (3.0 * math.pi / 4.0)**(1.0 / 3.0) * omega_pl**(2.0 / 3.0)

def f_lindhard(t):
    # Stable analytic form around t = ±1
    eps = 1e-15
    t = np.asarray(t)
    tp = np.where(np.abs(t - 1.0) < eps, 1.0 + eps, t)
    tm = np.where(np.abs(tp + 1.0) < eps, -1.0 - eps, tp)
    return (1.0 - tm**2) * np.log(np.abs((tm + 1.0) / (tm - 1.0)))

def g_arr(omega_pl_arr, opt_omega_eV, opt_elf):
    w = np.asarray(omega_pl_arr)
    w_safe = np.maximum(w, 1e-12)  # avoid divide-by-zero
    w_eV = (w_safe * h2ev).ravel()
    interp_vals = np.interp(w_eV, opt_omega_eV, opt_elf).reshape(w.shape)
    return 2.0 / (math.pi * w_safe) * interp_vals

def q_plus_surface(omega, omega_pl):
    kf = k_f(omega_pl)
    return kf + np.sqrt(kf**2 + 2 * omega[:, None])

def q_minus_surface(omega, omega_pl):
    kf = k_f(omega_pl)
    return -kf + np.sqrt(kf**2 + 2 * omega[:, None])

def q_minus_pair_vec(omega_au, omega0_au_vec):
    kf = k_f(omega0_au_vec)
    return -kf + np.sqrt(kf**2 + 2.0 * omega_au)

def lindhard_qcol_batch(qcols_au, omega_au, omega_pl_au):
    """Vectorized calculation of the complex Lindhard dielectric function."""
    B, Nw = qcols_au.shape
    Npl = omega_pl_au.size

    kf = np.clip(k_f(omega_pl_au), 1e-30, None)
    KF = kf[None, None, :]

    x = 2.0 * omega_au[:, None] / (kf[None, :]**2)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.broadcast_to(x[None, :, :], (B, Nw, Npl))

    Q = qcols_au[:, :, None]
    Z = Q / (2.0 * KF)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    z0 = (Z == 0.0)
    X0 = (X == 0.0)
    main = (~z0) & (~X0)
    Zs = np.where(z0, 1.0, Z)

    eps_real = np.ones((B, Nw, Npl), dtype=np.float64)
    eps_imag = np.zeros((B, Nw, Npl), dtype=np.float64)

    eps_real_z0 = 1.0 - 16.0 / (3.0 * KF * math.pi * (X**2))
    eps_real = np.where(z0 & (~X0), eps_real_z0, eps_real)

    u = X / (4.0 * Zs)
    coef = 1.0 / (8.0 * KF * (Zs**3))

    ind_general = ~((u < 0.01) | (u / (Zs + 1.0) > 100.0))
    eps_real_base = 1.0 + 1.0 / (math.pi * KF * (Zs**2)) * (
        0.5 + 1.0 / (8.0 * Zs) * (f_lindhard(Zs - u) + f_lindhard(Zs + u))
    )
    eps_real = np.where(main & ind_general, eps_real_base, eps_real)

    ind_1 = (X > 0.0) & (X < 4.0 * Zs * (1.0 - Zs))
    ind_2 = (X > np.abs(4.0 * Zs * (1.0 - Zs))) & (X < 4.0 * Zs * (1.0 + Zs))

    eps_imag = np.where(main & ind_1, coef * X, eps_imag)
    eps_imag = np.where(main & ind_2, coef * (1.0 - (Zs - u)**2), eps_imag)

    QQ = np.broadcast_to(qcols_au[:, :, None], (B, Nw, Npl))
    den = QQ * Zs
    den_safe = np.where(np.abs(den) > 0.0, den, 1.0)
    eps_imag_smallu = u / den_safe
    mask_smallu_imag = main & (u < 0.01) & (np.abs(QQ) > 0.0)
    eps_imag = np.where(mask_smallu_imag, eps_imag_smallu, eps_imag)

    eps_real = np.where(X0, 1.0, eps_real)
    eps_imag = np.where(X0, 0.0, eps_imag)

    eps_real = np.nan_to_num(eps_real, nan=1.0, posinf=1.0, neginf=1.0)
    eps_imag = np.nan_to_num(eps_imag, nan=0.0, posinf=0.0, neginf=0.0)

    return eps_real + 1j * eps_imag

def lindhard_derivative_vec(q_au_vec, omega_au, omega0_au_vec):
    kf = np.clip(k_f(omega0_au_vec), 1e-30, None)
    x  = 2.0 * omega_au / (kf**2)
    z  = q_au_vec / (2.0 * kf)

    z_safe = np.where(np.abs(z) > 0.0, z, 1.0)
    u = x / (4.0 * z_safe)
    y_plus  = z_safe + u
    y_minus = z_safe - u

    ind = ~((x > 100.0 * z_safe) | (z_safe > 100.0 * x))
    de = np.where(
        ind,
        (np.log(np.abs((y_minus+1)/(y_minus-1))) + np.log(np.abs((y_plus+1)/(y_plus-1))))
        / (4.0 * kf**(5/2) * math.sqrt(3*math.pi) * z_safe**3),
        0.0
    )

    a = z_safe / np.where(np.abs(x) > 0.0, x, 1.0)
    de = np.where(
        x > 100.0 * z_safe,
        16.0 / (kf**(5/2) * math.sqrt(3*math.pi) * x**2) *
        (-1 - 16*a**2 - 16*a**4*(16 + x**2) - (512/3)*a**6*(24 + 5*x**2)),
        de
    )
    return de


def _elf_qblock_worker(k0, k1, omega_eV_grid, q_grid_a0inv, omega_pl_eV, opt_omega, opt_elf, chunk_pl, clip_deps):
    """Joblib worker that computes ELF maps for a specific subset of momentum transfer (q) values."""
    omega_eV = np.asarray(omega_eV_grid, float)
    omega_au = omega_eV / h2ev
    Nw = omega_au.size

    qb = np.asarray(q_grid_a0inv, float)[k0:k1]
    Bq = qb.size

    omega_pl_au = np.asarray(omega_pl_eV, float) / h2ev
    Npl = omega_pl_au.size

    g_wpl = g_arr(omega_pl_au, opt_omega, opt_elf)
    g_interp = interp1d(omega_pl_au, g_wpl, kind="linear", bounds_error=False, fill_value=0.0, assume_sorted=True)

    elf_se_block = np.zeros((Nw, Bq), dtype=np.float64)
    elf_pl_block = np.zeros((Nw, Bq), dtype=np.float64)

    inner_batch_q = min(16, Bq)

    for kk0 in range(0, Bq, inner_batch_q):
        kk1 = min(kk0 + inner_batch_q, Bq)
        qb_small = qb[kk0:kk1]
        B = qb_small.size
        qcols = np.broadcast_to(qb_small[:, None], (B, Nw))

        se_acc = np.zeros((B, Nw), dtype=np.float64)
        M = B * Nw
        w0_rows = np.zeros(M, dtype=np.float64)
        ok_rows = np.zeros(M, dtype=bool)
        done_rows = np.zeros(M, dtype=bool)

        prev_wpl, prev_se, prev_eps_real = None, None, None

        for p0 in range(0, Npl, chunk_pl):
            p1 = min(p0 + chunk_pl, Npl)
            wpl_chunk = omega_pl_au[p0:p1]
            g_chunk = g_wpl[p0:p1]
            Nc = wpl_chunk.size

            boundary_wpl, boundary_se, boundary_eps = prev_wpl, prev_se, prev_eps_real

            eps_chunk = lindhard_qcol_batch(qcols, omega_au, wpl_chunk)
            q_m_chunk = q_minus_surface(omega_au, wpl_chunk)
            q_p_chunk = q_plus_surface(omega_au, wpl_chunk)

            im_m1 = (-1.0 / eps_chunk).imag
            se_chunk = im_m1 * g_chunk[None, None, :]
            mask = (q_p_chunk[None, :, :] >= qcols[:, :, None]) & (q_m_chunk[None, :, :] <= qcols[:, :, None])
            se_chunk *= mask
            se_chunk = np.nan_to_num(se_chunk, nan=0.0, posinf=0.0, neginf=0.0)

            if boundary_se is None:
                se_acc += np.trapezoid(se_chunk, wpl_chunk, axis=2)
            else:
                w_cat = np.concatenate(([boundary_wpl], wpl_chunk))
                se_cat = np.concatenate((boundary_se[:, :, None], se_chunk), axis=2)
                se_acc += np.trapezoid(se_cat, w_cat, axis=2)

            eps_real_chunk = eps_chunk.real.reshape(M, Nc)
            if boundary_eps is None:
                eps_cat = eps_real_chunk
                w_cat_r = wpl_chunk
            else:
                eps_cat = np.concatenate((boundary_eps[:, None], eps_real_chunk), axis=1)
                w_cat_r = np.concatenate(([boundary_wpl], wpl_chunk))

            s = np.sign(eps_cat)
            sc = (s[:, :-1] * s[:, 1:]) < 0

            eligible = ~done_rows
            if np.any(eligible):
                sc_e = sc[eligible]
                has = sc_e.any(axis=1)
                if np.any(has):
                    idx_e = np.flatnonzero(eligible)
                    j = np.argmax(sc_e, axis=1)
                    rows = idx_e[has]
                    jj = j[has]
                    y1, y2 = eps_cat[rows, jj], eps_cat[rows, jj+1]
                    denom = y2 - y1
                    good = np.abs(denom) > 1e-14

                    if np.any(good):
                        rowsg, jjg = rows[good], jj[good]
                        x1, x2 = w_cat_r[jjg], w_cat_r[jjg+1]
                        y1g, y2g = eps_cat[rowsg, jjg], eps_cat[rowsg, jjg+1]
                        w_est = x1 - y1g * (x2 - x1) / (y2g - y1g)

                        w0_rows[rowsg] = w_est
                        ok_rows[rowsg] = True
                        done_rows[rowsg] = True

            prev_wpl = wpl_chunk[-1]
            prev_se = se_chunk[:, :, -1]
            prev_eps_real = eps_chunk.real[:, :, -1].reshape(M)

        elf_se_block[:, kk0:kk1] = se_acc.T
        w0 = w0_rows.reshape(B, Nw)
        ok = ok_rows.reshape(B, Nw)
        pl = np.zeros((B, Nw), dtype=np.float64)

        for bi in range(B):
            valid = ok[bi, :]
            qv = qcols[bi, :]
            w0v = w0[bi, :]

            gcoef = g_interp(w0v)
            der = np.abs(lindhard_derivative_vec(qv, omega_au, w0v))
            der = np.clip(der, clip_deps, None)

            gate = (q_minus_pair_vec(omega_au, w0v) - qv) >= 0.0
            pl[bi, :] = gcoef * (math.pi / der) * gate * valid

        elf_pl_block[:, kk0:kk1] = pl.T

    return k0, k1, elf_se_block, elf_pl_block


# =====================================================================
# 2. FPA ENGINE CLASS (Stateful Manager)
# =====================================================================

class FPAEngine:
    def __init__(self, material, n_jobs=-1):
        """
        Initializes the Full Penn Algorithm engine.
        material: An optlib Material instance (must have .eloss and .elf defined).
        n_jobs: Number of CPU cores to use for parallel map building (-1 means all cores).
        """
        self.mat = material
        self.n_jobs = n_jobs
        self.qlog_grid = None
        self.se_spl = None
        self.pl_spl = None

    def build_elf_maps(self, qmax=45.0, qsplit=2.7, omega_pl_max=30000, n_log=260, chunk_pl=1024):
        """Builds the (omega, qlog) interpolators using parallel execution."""
        print("Building FPA ELF Maps... This may take a moment.")
        t0 = time.time()
        
        # Build grids
        q_grid = self._make_q_grid_hybrid_a0inv(qmax, qsplit, n_log=n_log)
        omega_pl_eV = self._make_omega_pl_grid_fast(omega_pl_max)
        
        Nq = q_grid.size
        q_block = 64
        tasks = [(k0, min(k0 + q_block, Nq)) for k0 in range(0, Nq, q_block)]

        results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(_elf_qblock_worker)(
                k0, k1,
                self.mat.eloss, q_grid, omega_pl_eV, 
                self.mat.eloss, self.mat.elf, # Assuming material.elf contains optical limit
                chunk_pl, 1e-14
            )
            for (k0, k1) in tasks
        )

        Nw = self.mat.eloss.size
        elf_se_map = np.zeros((Nw, Nq), dtype=np.float64)
        elf_pl_map = np.zeros((Nw, Nq), dtype=np.float64)

        for k0, k1, se_blk, pl_blk in results:
            elf_se_map[:, k0:k1] = se_blk
            elf_pl_map[:, k0:k1] = pl_blk

        self.qlog_grid = np.log(q_grid)
        self.se_spl = RectBivariateSpline(self.mat.eloss, self.qlog_grid, elf_se_map, kx=1, ky=1)
        self.pl_spl = RectBivariateSpline(self.mat.eloss, self.qlog_grid, elf_pl_map, kx=1, ky=1)
        
        print(f"FPA maps built successfully in {time.time()-t0:.2f}s")

    def calculate_diimfp(self, E_eV, nq=100):
        """
        Calculates the Differential Inelastic Mean Free Path (DIIMFP)
        and Total IMFP for a specific primary energy E0.
        """
        if self.se_spl is None or self.pl_spl is None:
            raise ValueError("ELF maps not built! Call build_elf_maps() first.")

        c = 137.036
        e0 = E_eV / h2ev

        # Enforce omega <= E - E_fermi
        wmax = E_eV - self.mat.e_fermi
        ok_w = self.mat.eloss <= max(wmax, 0.0)

        if not np.any(ok_w):
            z = np.zeros_like(self.mat.eloss, dtype=float)
            return z, z, z, z, z, z

        omega_ok = self.mat.eloss[ok_w]
        om = omega_ok / h2ev

        emw = np.clip(e0 - om, 0.0, None)
        k  = np.sqrt(e0 * (2 + e0/c**2))
        kp = np.sqrt(emw * (2 + emw/c**2))

        qminus = np.maximum(k - kp, 0.01) # Enforce a0^-1 minimum
        qplus  = k + kp

        qm = np.log(qminus)
        qp = np.minimum(np.log(qplus), self.qlog_grid[-1]) # Clamp to max map grid

        t = (np.arange(nq) + 0.5) / nq
        qlog_mat = qm[:, None] + (qp - qm)[:, None] * t[None, :]

        # Interpolate rapidly over kinematic constraints
        w_flat = np.repeat(omega_ok, nq)
        q_flat = qlog_mat.ravel()

        elf_se = self.se_spl.ev(w_flat, q_flat).reshape(omega_ok.size, nq)
        elf_pl = self.pl_spl.ev(w_flat, q_flat).reshape(omega_ok.size, nq)

        rel = ((1 + e0/(c**2))**2) / (1 + e0/(2*c**2))
        pref = rel * 1.0 / (math.pi * e0)

        iimfp_se_ok = pref * np.trapezoid(elf_se, qlog_mat, axis=1)
        iimfp_pl_ok = pref * np.trapezoid(elf_pl, qlog_mat, axis=1)
        iimfp_tot_ok = iimfp_se_ok + iimfp_pl_ok

        diimfp_se = np.zeros_like(self.mat.eloss, dtype=float)
        diimfp_pl = np.zeros_like(self.mat.eloss, dtype=float)
        diimfp_tot = np.zeros_like(self.mat.eloss, dtype=float)

        diimfp_se[ok_w] = iimfp_se_ok / (h2ev * a0)
        diimfp_pl[ok_w] = iimfp_pl_ok / (h2ev * a0)
        diimfp_tot[ok_w] = iimfp_tot_ok / (h2ev * a0)

        return diimfp_tot, diimfp_pl, diimfp_se

    # --- Grid Generators ---
    def _make_q_grid_hybrid_a0inv(self, qmax, qsplit, dq_low=0.01, n_log=260):
        q1 = np.arange(0.01, qsplit + 0.5*dq_low, dq_low)
        q2 = np.geomspace(qsplit, qmax, n_log) if qmax > qsplit else np.array([], float)
        return np.unique(np.concatenate([q1, q2]))

    def _make_omega_pl_grid_fast(self, max_eV, split_eV=100.0, d_low=0.02, n_log=3000):
        w1 = np.arange(1e-5, split_eV, d_low)
        w2 = np.geomspace(split_eV, max_eV, n_log)
        return np.unique(np.concatenate([w1, w2]))