"""Fused, float64 Full Penn map builder for Linux CPU/CUDA.

This module is imported lazily by fpa.py. All physics branches and grids follow
the supplied NumPy reference; this is an acceleration backend, not a new model.
No Joblib workers or global 3-D dielectric arrays are used here.
"""

import math
import os
import time

import numpy as np
from joblib import cpu_count

try:
    import taichi as ti
except ImportError as exc:
    if isinstance(exc, ModuleNotFoundError) and exc.name == "taichi":
        message = (
            "The Taichi backend requires Taichi, which is not installed in this "
            "Python environment. See README.md for the CPU or CUDA requirements. "
            "The NumPy backend does not require Taichi."
        )
    elif "GLIBC_" in str(exc):
        message = (
            "The installed Taichi binary requires a glibc version missing from "
            "this environment. Use a compatible container, or the isolated "
            "Taichi 1.7.3 setup tested on ASC (README.md, CUDA section). "
            "Reinstalling the same binary does not change glibc. "
            f"Original import error: {exc}"
        )
    else:
        message = (
            "Taichi import failed while loading a dependency or native library. "
            f"Original import error: {exc}"
        )
    raise ImportError(message) from exc


_PI = math.pi
_SQRT_3PI = math.sqrt(3.0 * math.pi)
_KF_COEFF = (3.0 * math.pi / 4.0) ** (1.0 / 3.0)
_INF = float("inf")


def resolve_cpu_threads(n_jobs=-1, cpu_threads=None):
    """Respect CPU affinity/quota and SLURM_CPUS_PER_TASK by default."""
    available = cpu_count()
    allocation = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocation:
        try:
            available = min(available, max(1, int(allocation)))
        except ValueError:
            pass
    if cpu_threads is not None:
        if int(cpu_threads) != cpu_threads or cpu_threads < 1:
            raise ValueError("cpu_threads must be a positive integer")
        return int(cpu_threads)
    if n_jobs is None:
        return 1
    if int(n_jobs) != n_jobs or n_jobs == 0:
        raise ValueError("n_jobs must be a nonzero integer or None")
    return min(available, int(n_jobs)) if n_jobs > 0 else max(1, available + 1 + int(n_jobs))


def ensure_runtime(arch="cpu", n_jobs=-1, cpu_threads=None):
    """Use one process-wide runtime, without resetting another Taichi user."""
    if arch not in {"cpu", "cuda"}:
        raise ValueError("This float64 backend supports arch='cpu' or arch='cuda'")
    target = ti.cpu if arch == "cpu" else ti.cuda
    threads = resolve_cpu_threads(n_jobs, cpu_threads)
    # Taichi 1.7.3/1.7.4 expose runtime initialization state through lang.impl.
    if ti.lang.impl.get_runtime().prog is None:
        ti.init(
            arch=target,
            default_fp=ti.f64,
            fast_math=False,
            cpu_max_num_threads=threads,
            enable_fallback=False,
        )
    if ti.cfg.arch != target or ti.cfg.default_fp != ti.f64 or ti.cfg.fast_math:
        raise RuntimeError(
            "An incompatible Taichi runtime is active. Use a fresh process "
            f"with arch={arch!r}, default_fp=ti.f64, fast_math=False. "
            "FPA does not reset an existing runtime."
        )
    if arch == "cpu" and ti.cfg.cpu_max_num_threads != threads:
        raise RuntimeError(
            "The active Taichi CPU thread count differs from the requested "
            f"{threads}. Use a fresh process or matching cpu_threads."
        )
    return {"arch": arch, "precision": "float64", "cpu_threads": threads,
            "fast_math": False, "taichi_version": ".".join(map(str, ti.__version__))}


@ti.func
def _finite_or(value, replacement):
    result = value
    if ti.math.isnan(value) or ti.math.isinf(value):
        result = replacement
    return result


@ti.func
def _lindhard_f(t):
    # Preserve the reference's float64 endpoint convention exactly.
    tm = t
    if ti.abs(tm - 1.0) < 1e-15:
        tm = 1.0 + 1e-15
    if ti.abs(tm + 1.0) < 1e-15:
        tm = -1.0 - 1e-15
    return (1.0 - tm * tm) * ti.log(ti.abs((tm + 1.0) / (tm - 1.0)))


@ti.func
def _epsilon(q, omega, kf):
    x = _finite_or(2.0 * omega / (kf * kf), 0.0)
    z = _finite_or(q / (2.0 * kf), 0.0)
    er = 1.0
    ei = 0.0
    if x != 0.0:
        if z == 0.0:
            er = 1.0 - 16.0 / (3.0 * kf * _PI * x * x)
        else:
            u = x / (4.0 * z)
            # These branches intentionally retain the original approximations.
            if not ((u < 0.01) or (u / (z + 1.0) > 100.0)):
                er = 1.0 + 1.0 / (_PI * kf * z * z) * (
                    0.5 + (_lindhard_f(z - u) + _lindhard_f(z + u)) / (8.0 * z)
                )
            coefficient = 1.0 / (8.0 * kf * z * z * z)
            edge = 4.0 * z * (1.0 - z)
            if x > 0.0 and x < edge:
                ei = coefficient * x
            if x > ti.abs(edge) and x < 4.0 * z * (1.0 + z):
                ei = coefficient * (1.0 - (z - u) * (z - u))
            if u < 0.01 and ti.abs(q) > 0.0:
                ei = u / (q * z)
    return _finite_or(er, 1.0), _finite_or(ei, 0.0)


@ti.func
def _log_ratio_abs(t):
    # NumPy's derivative produces infinities at these endpoints; preserve them.
    value = 0.0
    if t == 1.0:
        value = _INF
    elif t == -1.0:
        value = -_INF
    else:
        value = ti.log(ti.abs((t + 1.0) / (t - 1.0)))
    return value


@ti.func
def _derivative(q, omega, omega0):
    kf = ti.max(_KF_COEFF * omega0 ** (2.0 / 3.0), 1e-30)
    x = 2.0 * omega / (kf * kf)
    z = q / (2.0 * kf)
    zs = z
    if z == 0.0:
        zs = 1.0
    u = x / (4.0 * zs)
    de = 0.0
    if not ((x > 100.0 * zs) or (zs > 100.0 * x)):
        de = (_log_ratio_abs(zs - u) + _log_ratio_abs(zs + u)) / (
            4.0 * kf ** 2.5 * _SQRT_3PI * zs ** 3
        )
    if x > 100.0 * zs:
        a = zs / x
        de = 16.0 / (kf ** 2.5 * _SQRT_3PI * x * x) * (
            -1.0 - 16.0 * a ** 2 - 16.0 * a ** 4 * (16.0 + x * x)
            - (512.0 / 3.0) * a ** 6 * (24.0 + 5.0 * x * x)
        )
    return de


@ti.kernel
def _build_maps(
    loss_au: ti.types.ndarray(dtype=ti.f64, ndim=1),
    q_grid: ti.types.ndarray(dtype=ti.f64, ndim=1),
    plasma_au: ti.types.ndarray(dtype=ti.f64, ndim=1),
    kf_grid: ti.types.ndarray(dtype=ti.f64, ndim=1),
    g_grid: ti.types.ndarray(dtype=ti.f64, ndim=1),
    se_out: ti.types.ndarray(dtype=ti.f64, ndim=2),
    pl_out: ti.types.ndarray(dtype=ti.f64, ndim=2),
    chunk_pl: ti.i32,
    clip_deps: ti.f64,
):
    # Only the outer loop is parallel. Each cell owns its accumulator/root,
    # so there are no floating-point atomics or races in the plasma integral.
    for iw, iq in ti.ndrange(loss_au.shape[0], q_grid.shape[0]):
        omega = loss_au[iw]
        q = q_grid[iq]
        total = 0.0
        previous_se = 0.0
        previous_er = 1.0
        previous_w = 0.0
        previous_g = 0.0
        root = 0.0
        root_g = 0.0
        root_found = 0
        seen_crossing_in_chunk = 0
        for ip in range(plasma_au.shape[0]):
            if ip % chunk_pl == 0:
                seen_crossing_in_chunk = 0
            wpl = plasma_au[ip]
            kf = kf_grid[ip]
            g = g_grid[ip]
            er, ei = _epsilon(q, omega, kf)
            se = 0.0
            surface = ti.sqrt(kf * kf + 2.0 * omega)
            if (kf + surface >= q) and (-kf + surface <= q):
                denominator = er * er + ei * ei
                if denominator != 0.0:
                    se = _finite_or(ei / denominator * g, 0.0)
            if ip > 0:
                total += 0.5 * (previous_se + se) * (wpl - previous_w)
                crosses = ((previous_er < 0.0 and er > 0.0)
                           or (previous_er > 0.0 and er < 0.0))
                if root_found == 0 and seen_crossing_in_chunk == 0 and crosses:
                    # Reference accepts only the first crossing in each chunk,
                    # even when its denominator is too small to accept a root.
                    seen_crossing_in_chunk = 1
                    difference = er - previous_er
                    if ti.abs(difference) > 1e-14:
                        root = previous_w - previous_er * (wpl - previous_w) / difference
                        root_g = previous_g + (g - previous_g) * (
                            (root - previous_w) / (wpl - previous_w)
                        )
                        root_found = 1
            previous_se = se
            previous_er = er
            previous_w = wpl
            previous_g = g
        plasmon = 0.0
        if root_found != 0:
            kf_root = _KF_COEFF * root ** (2.0 / 3.0)
            qm_root = -kf_root + ti.sqrt(kf_root * kf_root + 2.0 * omega)
            if qm_root - q >= 0.0:
                derivative = ti.max(ti.abs(_derivative(q, omega, root)), clip_deps)
                plasmon = root_g * (_PI / derivative)
        se_out[iw, iq] = total
        pl_out[iw, iq] = plasmon


def _vector(values, name, *, nonnegative=False):
    result = np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(-1))
    if result.size < 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain at least two finite points")
    if np.any(np.diff(result) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    if np.any(result < 0.0 if nonnegative else result <= 0.0):
        raise ValueError(f"{name} must be {'nonnegative' if nonnegative else 'positive'}")
    return result


def compute_elf_maps(loss_eV, q_grid, plasma_eV, g_grid, *, h2ev,
                     arch="cpu", n_jobs=-1, cpu_threads=None,
                     chunk_pl=1024, clip_deps=1e-14):
    """Return (continuum map, plasmon map, timing/runtime metadata).

    g_grid is computed by the unchanged NumPy g_arr() on the plasma grid.
    Timings include initialization, transfers and result readback. The first
    kernel call can include compilation or loading an existing offline cache.
    """
    started = time.perf_counter()
    loss = _vector(loss_eV, "loss_eV", nonnegative=True)
    q = _vector(q_grid, "q_grid")
    plasma = _vector(plasma_eV, "plasma_eV")
    g = np.ascontiguousarray(np.asarray(g_grid, dtype=np.float64).reshape(-1))
    if g.shape != plasma.shape or not np.all(np.isfinite(g)):
        raise ValueError("g_grid must be finite and match plasma_eV")
    if int(chunk_pl) != chunk_pl or chunk_pl < 1:
        raise ValueError("chunk_pl must be a positive integer")
    if not np.isfinite(h2ev) or h2ev <= 0 or not np.isfinite(clip_deps) or clip_deps <= 0:
        raise ValueError("h2ev and clip_deps must be positive and finite")
    info = ensure_runtime(arch, n_jobs, cpu_threads)
    loss_au = np.ascontiguousarray(loss / h2ev)
    plasma_au = np.ascontiguousarray(plasma / h2ev)
    kf = np.ascontiguousarray(np.maximum(_KF_COEFF * plasma_au ** (2.0 / 3.0), 1e-30))
    input_arrays = []
    for values in (loss_au, q, plasma_au, kf, g):
        array = ti.ndarray(dtype=ti.f64, shape=values.shape)
        array.from_numpy(values)
        input_arrays.append(array)
    se = ti.ndarray(dtype=ti.f64, shape=(loss.size, q.size))
    pl = ti.ndarray(dtype=ti.f64, shape=(loss.size, q.size))
    ti.sync()
    kernel_started = time.perf_counter()
    _build_maps(*input_arrays, se, pl, int(chunk_pl), float(clip_deps))
    ti.sync()
    kernel_seconds = time.perf_counter() - kernel_started
    se_result, pl_result = se.to_numpy(), pl.to_numpy()
    if not np.all(np.isfinite(se_result)) or not np.all(np.isfinite(pl_result)):
        raise FloatingPointError("Taichi produced nonfinite ELF values; compare this input against NumPy")
    info.update(kernel_call_seconds=kernel_seconds,
                total_seconds=time.perf_counter() - started,
                shape=[int(loss.size), int(q.size)],
                plasma_points=int(plasma.size))
    return se_result, pl_result, info
