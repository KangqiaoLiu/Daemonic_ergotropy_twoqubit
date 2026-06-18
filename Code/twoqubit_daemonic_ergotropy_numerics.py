"""Numerical calculations for the two-qubit daemonic-ergotropy paper.

The script evaluates the analytic checks, finite-outcome POVM comparisons, and
figures from the formulas stated in the paper.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange, set_num_threads
from tqdm.auto import tqdm

from figure_io import (
    FIG4_FONT_RC,
    FIGURE_DIRS,
    OKABE_ITO,
    RESULTS,
    configure_plot_style,
    ensure_output_dirs,
    file_sha256,
    refresh_output_manifest,
    save_named_figure,
)

OUTCOME_WEIGHT_THRESHOLD = 1.0e-8

I2 = np.eye(2, dtype=np.complex128)
SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)
PAULIS = (SX, SY, SZ)


@dataclass(frozen=True)
class NumericalSettings:
    werner_points: int = 17
    werner_seed: int = 1234
    werner_projective_samples: int = 15_000
    werner_local_starts: int = 28
    werner_local_steps: int = 220

    pure_states: int = 20
    pure_seed: int = 2026
    pure_projective_samples: int = 20_000
    pure_local_starts: int = 35
    pure_local_steps: int = 260

    mixed_states: int = 30
    mixed_master_seed: int = 7
    mixed_projective_seed_base: int = 10_000_000
    mixed_antipodal_seed_base: int = 30_000_000
    mixed_povm_seed_base: int = 20_000_000
    mixed_projective_batches: int = 4
    mixed_projective_samples: int = 18_000
    mixed_local_starts: int = 35
    mixed_local_steps: int = 280
    mixed_linear_samples: int = 1_500
    mixed_stiefel_samples: int = 1_200
    mixed_stiefel_restarts: int = 3
    mixed_stiefel_steps: int = 320


def configure_runtime() -> None:
    configure_plot_style()
    set_num_threads(min(32, os.cpu_count() or 8))


def ensure_dirs() -> None:
    ensure_output_dirs()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def ketbra(v: np.ndarray) -> np.ndarray:
    return np.outer(v, v.conj())


def bell_singlet() -> np.ndarray:
    v = np.zeros(4, dtype=np.complex128)
    v[1] = 1 / np.sqrt(2)
    v[2] = -1 / np.sqrt(2)
    return v


def werner_state(p: float) -> np.ndarray:
    v = bell_singlet()
    rho = p * ketbra(v) + (1.0 - p) * np.eye(4, dtype=np.complex128) / 4.0
    return 0.5 * (rho + rho.conj().T)


def random_pure_state(dim: int, rng: np.random.Generator) -> np.ndarray:
    v = (rng.normal(size=dim) + 1j * rng.normal(size=dim)) / np.sqrt(2.0)
    return v / np.linalg.norm(v)


def random_density_matrix(dim: int, rng: np.random.Generator) -> np.ndarray:
    g = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / np.sqrt(2.0)
    rho = g @ g.conj().T
    rho = rho / np.trace(rho)
    return 0.5 * (rho + rho.conj().T)


def bloch_params_twoqubit(rho_sa: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    a = np.zeros(3, dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    tmat = np.zeros((3, 3), dtype=np.float64)
    max_imag = 0.0
    for i, si in enumerate(PAULIS):
        zai = np.trace(rho_sa @ np.kron(si, I2))
        zbi = np.trace(rho_sa @ np.kron(I2, si))
        a[i] = float(zai.real)
        b[i] = float(zbi.real)
        max_imag = max(max_imag, abs(float(zai.imag)), abs(float(zbi.imag)))
        for j, sj in enumerate(PAULIS):
            zt = np.trace(rho_sa @ np.kron(si, sj))
            tmat[i, j] = float(zt.real)
            max_imag = max(max_imag, abs(float(zt.imag)))
    return a, b, tmat, max_imag


def sample_unit_vectors(count: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.normal(size=(count, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype(np.float64)


@njit(parallel=True)
def lproj_many(a: np.ndarray, tmat: np.ndarray, ns: np.ndarray) -> np.ndarray:
    count = ns.shape[0]
    out = np.empty(count, dtype=np.float64)
    for k in prange(count):
        n0, n1, n2 = ns[k, 0], ns[k, 1], ns[k, 2]
        t0 = tmat[0, 0] * n0 + tmat[0, 1] * n1 + tmat[0, 2] * n2
        t1 = tmat[1, 0] * n0 + tmat[1, 1] * n1 + tmat[1, 2] * n2
        t2 = tmat[2, 0] * n0 + tmat[2, 1] * n1 + tmat[2, 2] * n2
        u0, u1, u2 = a[0] + t0, a[1] + t1, a[2] + t2
        v0, v1, v2 = a[0] - t0, a[1] - t1, a[2] - t2
        out[k] = 0.5 * (
            math.sqrt(u0 * u0 + u1 * u1 + u2 * u2)
            + math.sqrt(v0 * v0 + v1 * v1 + v2 * v2)
        )
    return out


@njit
def normalize3(n: np.ndarray) -> np.ndarray:
    norm = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
    if norm < 1e-18:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return np.array([n[0] / norm, n[1] / norm, n[2] / norm], dtype=np.float64)


@njit
def lproj_single(a: np.ndarray, tmat: np.ndarray, n: np.ndarray) -> float:
    n = normalize3(n)
    t0 = tmat[0, 0] * n[0] + tmat[0, 1] * n[1] + tmat[0, 2] * n[2]
    t1 = tmat[1, 0] * n[0] + tmat[1, 1] * n[1] + tmat[1, 2] * n[2]
    t2 = tmat[2, 0] * n[0] + tmat[2, 1] * n[1] + tmat[2, 2] * n[2]
    u0, u1, u2 = a[0] + t0, a[1] + t1, a[2] + t2
    v0, v1, v2 = a[0] - t0, a[1] - t1, a[2] - t2
    return 0.5 * (
        math.sqrt(u0 * u0 + u1 * u1 + u2 * u2)
        + math.sqrt(v0 * v0 + v1 * v1 + v2 * v2)
    )


@njit
def grad_lproj(a: np.ndarray, tmat: np.ndarray, n: np.ndarray) -> np.ndarray:
    n0, n1, n2 = n[0], n[1], n[2]
    t0 = tmat[0, 0] * n0 + tmat[0, 1] * n1 + tmat[0, 2] * n2
    t1 = tmat[1, 0] * n0 + tmat[1, 1] * n1 + tmat[1, 2] * n2
    t2 = tmat[2, 0] * n0 + tmat[2, 1] * n1 + tmat[2, 2] * n2
    u0, u1, u2 = a[0] + t0, a[1] + t1, a[2] + t2
    v0, v1, v2 = a[0] - t0, a[1] - t1, a[2] - t2
    nu = max(math.sqrt(u0 * u0 + u1 * u1 + u2 * u2), 1e-14)
    nv = max(math.sqrt(v0 * v0 + v1 * v1 + v2 * v2), 1e-14)
    tu0 = tmat[0, 0] * u0 + tmat[1, 0] * u1 + tmat[2, 0] * u2
    tu1 = tmat[0, 1] * u0 + tmat[1, 1] * u1 + tmat[2, 1] * u2
    tu2 = tmat[0, 2] * u0 + tmat[1, 2] * u1 + tmat[2, 2] * u2
    tv0 = tmat[0, 0] * v0 + tmat[1, 0] * v1 + tmat[2, 0] * v2
    tv1 = tmat[0, 1] * v0 + tmat[1, 1] * v1 + tmat[2, 1] * v2
    tv2 = tmat[0, 2] * v0 + tmat[1, 2] * v1 + tmat[2, 2] * v2
    return np.array([
        0.5 * (tu0 / nu - tv0 / nv),
        0.5 * (tu1 / nu - tv1 / nv),
        0.5 * (tu2 / nu - tv2 / nv),
    ], dtype=np.float64)


@njit
def refine_projective(a: np.ndarray, tmat: np.ndarray, n0: np.ndarray, n_steps: int, eta0: float = 1.0):
    n = normalize3(n0)
    current_val = lproj_single(a, tmat, n)
    step_size = eta0
    for _ in range(n_steps):
        grad = grad_lproj(a, tmat, n)
        dot = n[0] * grad[0] + n[1] * grad[1] + n[2] * grad[2]
        tangent_grad = np.array([
            grad[0] - dot * n[0],
            grad[1] - dot * n[1],
            grad[2] - dot * n[2],
        ], dtype=np.float64)
        grad_norm = math.sqrt(
            tangent_grad[0] * tangent_grad[0]
            + tangent_grad[1] * tangent_grad[1]
            + tangent_grad[2] * tangent_grad[2]
        )
        if grad_norm < 1e-12:
            break

        accepted = False
        trial_step = step_size
        for _ in range(40):
            candidate = normalize3(n + trial_step * tangent_grad)
            val = lproj_single(a, tmat, candidate)
            if val > current_val:
                n = candidate
                current_val = val
                step_size = min(2.0 * trial_step, 16.0)
                accepted = True
                break
            trial_step *= 0.5

        if not accepted:
            break
    return current_val, n


def projective_max(
    a: np.ndarray,
    tmat: np.ndarray,
    rng: np.random.Generator,
    n_global: int,
    n_refine_starts: int,
    refine_steps: int,
) -> tuple[float, np.ndarray]:
    ns = sample_unit_vectors(n_global, rng)
    vals = lproj_many(a, tmat, ns)
    take = min(n_refine_starts, len(vals))
    idx = np.argsort(vals)[-take:][::-1]
    best_val = float(vals[idx[0]])
    best_n = ns[idx[0]].copy()
    for j in idx:
        val, n = refine_projective(a, tmat, ns[j], n_steps=refine_steps)
        if float(val) > best_val:
            best_val = float(val)
            best_n = np.asarray(n, dtype=np.float64)
    return best_val, best_n


def projective_reference_search(
    a: np.ndarray,
    tmat: np.ndarray,
    settings: NumericalSettings,
    state_index: int,
) -> tuple[float, np.ndarray, list[float]]:
    batch_values: list[float] = []
    best_val = -np.inf
    best_n: np.ndarray | None = None
    for batch in range(settings.mixed_projective_batches):
        rng = np.random.default_rng(settings.mixed_projective_seed_base + 1009 * state_index + batch)
        val, n = projective_max(
            a, tmat, rng,
            n_global=settings.mixed_projective_samples,
            n_refine_starts=settings.mixed_local_starts,
            refine_steps=settings.mixed_local_steps,
        )
        batch_values.append(float(val))
        if float(val) > best_val:
            best_val = float(val)
            best_n = np.asarray(n, dtype=np.float64)
    if best_n is None:
        raise RuntimeError("Projective reference search did not produce a direction.")
    return best_val, best_n, batch_values


@njit
def lpovm4(a: np.ndarray, tmat: np.ndarray, alpha: np.ndarray, directions: np.ndarray) -> float:
    total = 0.0
    for k in range(4):
        n0, n1, n2 = directions[k, 0], directions[k, 1], directions[k, 2]
        t0 = tmat[0, 0] * n0 + tmat[0, 1] * n1 + tmat[0, 2] * n2
        t1 = tmat[1, 0] * n0 + tmat[1, 1] * n1 + tmat[1, 2] * n2
        t2 = tmat[2, 0] * n0 + tmat[2, 1] * n1 + tmat[2, 2] * n2
        u0, u1, u2 = a[0] + t0, a[1] + t1, a[2] + t2
        total += alpha[k] * math.sqrt(u0 * u0 + u1 * u1 + u2 * u2)
    return 0.5 * total


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return (v / np.linalg.norm(v)).astype(np.float64)


def random_rank1_povm4(rng: np.random.Generator, max_tries: int = 2000, eps: float = 1e-10):
    target = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for _ in range(max_tries):
        directions = np.stack([random_unit_vector(rng) for _ in range(4)], axis=0)
        amat = np.vstack([np.ones(4), directions.T])
        if abs(np.linalg.det(amat)) < 1e-7:
            continue
        alpha = np.linalg.solve(amat, target)
        if np.all(np.isfinite(alpha)) and np.min(alpha) > eps:
            alpha = 2.0 * alpha / np.sum(alpha)
            if np.linalg.norm((alpha[:, None] * directions).sum(axis=0)) < 1e-7:
                return alpha.astype(np.float64), directions.astype(np.float64)
    return None


def povm4_random_search(a: np.ndarray, tmat: np.ndarray, rng: np.random.Generator, n_samples: int):
    best_val = -np.inf
    best = None
    attempts = 0
    while attempts < n_samples:
        res = random_rank1_povm4(rng)
        if res is None:
            continue
        alpha, directions = res
        val = float(lpovm4(a, tmat, alpha, directions))
        attempts += 1
        if val > best_val:
            best_val = val
            best = (alpha.copy(), directions.copy())
    return best_val, best


def antipodal_four_outcome(
    n: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(n, dtype=np.float64)
    n = n / np.linalg.norm(n)
    if rng is None:
        alpha = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    else:
        plus = rng.random(2)
        minus = rng.random(2)
        plus = plus / plus.sum()
        minus = minus / minus.sum()
        alpha = np.array([plus[0], plus[1], minus[0], minus[1]], dtype=np.float64)
    directions = np.stack([n, n, -n, -n], axis=0).astype(np.float64)
    return alpha, directions


def feasibility_from_alpha(alpha: np.ndarray, directions: np.ndarray) -> dict[str, float | int]:
    return {
        "closure_residual": float(np.linalg.norm((alpha[:, None] * directions).sum(axis=0))),
        "weight_sum_residual": float(abs(alpha.sum() - 2.0)),
        "active_outcomes": int(np.sum(alpha > OUTCOME_WEIGHT_THRESHOLD)),
        "min_weight": float(alpha.min()),
        "max_weight": float(alpha.max()),
    }


def inv_sqrt_2x2(hmat: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    vals, vecs = np.linalg.eigh(hmat)
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.conj().T


def retract_stiefel(amat: np.ndarray) -> np.ndarray:
    return inv_sqrt_2x2(amat @ amat.conj().T) @ amat


def random_stiefel_frame(rng: np.random.Generator) -> np.ndarray:
    amat = (rng.normal(size=(2, 4)) + 1j * rng.normal(size=(2, 4))) / np.sqrt(2.0)
    return retract_stiefel(amat.astype(np.complex128))


def alpha_directions_from_frame(frame: np.ndarray, tiny: float = 1e-14) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.zeros(4, dtype=np.float64)
    directions = np.zeros((4, 3), dtype=np.float64)
    for k in range(4):
        vk = frame[:, k]
        ak = float(np.vdot(vk, vk).real)
        alpha[k] = ak
        if ak > tiny:
            psi = vk / np.sqrt(ak)
            directions[k, 0] = float(np.vdot(psi, SX @ psi).real)
            directions[k, 1] = float(np.vdot(psi, SY @ psi).real)
            directions[k, 2] = float(np.vdot(psi, SZ @ psi).real)
        else:
            directions[k, 2] = 1.0
    return alpha, directions


def lpovm_from_frame(a: np.ndarray, tmat: np.ndarray, frame: np.ndarray) -> float:
    alpha, directions = alpha_directions_from_frame(frame)
    return float(lpovm4(a, tmat, alpha, directions))


def feasibility_from_frame(frame: np.ndarray) -> dict[str, float | int]:
    alpha, directions = alpha_directions_from_frame(frame)
    out = feasibility_from_alpha(alpha, directions)
    out["operator_residual"] = float(np.linalg.norm(frame @ frame.conj().T - np.eye(2)))
    return out


def stiefel_local_refine(
    a: np.ndarray,
    tmat: np.ndarray,
    frame0: np.ndarray,
    rng: np.random.Generator,
    n_steps: int,
    sigma0: float = 0.30,
    sigma_min: float = 0.015,
) -> tuple[float, np.ndarray]:
    frame = frame0.copy()
    best_val = lpovm_from_frame(a, tmat, frame)
    for step in range(n_steps):
        sigma = max(sigma_min, sigma0 * (0.995 ** step))
        noise = (rng.normal(size=(2, 4)) + 1j * rng.normal(size=(2, 4))) / np.sqrt(2.0)
        trial = retract_stiefel(frame + sigma * noise)
        val = lpovm_from_frame(a, tmat, trial)
        if val > best_val:
            best_val = val
            frame = trial
    return float(best_val), frame


def stiefel_search(
    a: np.ndarray,
    tmat: np.ndarray,
    rng: np.random.Generator,
    n_random: int,
    n_restarts: int,
    refine_steps: int,
) -> tuple[float, np.ndarray]:
    best_val = -np.inf
    best_frame = None
    for _ in range(n_random):
        frame = random_stiefel_frame(rng)
        val = lpovm_from_frame(a, tmat, frame)
        if val > best_val:
            best_val = val
            best_frame = frame
    if best_frame is None:
        raise RuntimeError("Stiefel search did not produce a frame.")
    for _ in range(n_restarts):
        seed_frame = random_stiefel_frame(rng)
        seed_val = lpovm_from_frame(a, tmat, seed_frame)
        start_frame = seed_frame if seed_val > best_val else best_frame
        val, frame = stiefel_local_refine(a, tmat, start_frame, rng, n_steps=refine_steps)
        if val > best_val:
            best_val = val
            best_frame = frame
    return float(best_val), best_frame


def run_werner(settings: NumericalSettings) -> np.ndarray:
    rng = np.random.default_rng(settings.werner_seed)
    ps = np.linspace(-1.0 / 3.0, 1.0, settings.werner_points)
    rows = []
    for p in tqdm(ps, desc="Werner line"):
        rho = werner_state(float(p))
        a, _, tmat, max_imag = bloch_params_twoqubit(rho)
        lnum, _ = projective_max(
            a, tmat, rng,
            n_global=settings.werner_projective_samples,
            n_refine_starts=settings.werner_local_starts,
            refine_steps=settings.werner_local_steps,
        )
        lexact = abs(float(p))
        rows.append((float(p), lnum, lexact, lnum - lexact, max_imag))
    data = np.array(rows, dtype=np.float64)
    np.savetxt(
        RESULTS / "werner_line_projective_optimum.csv",
        data,
        delimiter=",",
        header="p,L_projective_numerical,L_projective_exact,residual,max_imag_bloch",
        comments="",
    )

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(3.45, 3.15), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.18},
    )
    ax.plot(data[:, 0], data[:, 2], color=OKABE_ITO["black"], linestyle="-", label=r"$|p|$")
    ax.plot(data[:, 0], data[:, 1], color=OKABE_ITO["blue"], linestyle="none",
            marker="o", markerfacecolor="none", label="numerical")
    ax.set_ylabel(r"$\mathcal{L}_{\rm proj}^{\star}$")
    ax.legend(frameon=False, loc="upper center", ncol=2, handlelength=2.2)
    ax.text(0.02, 0.94, "(a)", transform=ax.transAxes, va="top", ha="left")
    axr.axhline(0.0, color=OKABE_ITO["black"], linewidth=0.8)
    axr.plot(data[:, 0], data[:, 3] * 1e16, color=OKABE_ITO["vermillion"],
             linestyle="none", marker="s", markerfacecolor="none")
    axr.set_xlabel(r"$p$")
    axr.set_ylabel(r"res. ($10^{-16}$)")
    axr.text(0.02, 0.88, "(b)", transform=axr.transAxes, va="top", ha="left")
    save_named_figure(fig, "figure3.png")
    plt.close(fig)
    return data


def run_aligned_diagonal() -> dict[str, np.ndarray | float]:
    a_abs = 0.35
    x = np.linspace(0.0, 1.15, 320)
    y = np.linspace(0.0, 1.15, 320)
    t3_abs_grid, tperp_grid = np.meshgrid(x, y, indexing="xy")
    l_equatorial = np.sqrt(a_abs**2 + tperp_grid**2)
    l_axial = np.maximum(a_abs, t3_abs_grid)
    margin = l_equatorial - l_axial
    tol = 1.0e-12
    tie_mask = np.abs(margin) <= tol
    region = np.where(l_axial > l_equatorial + tol, 1, 0).astype(np.int32)
    region_code = np.where(tie_mask, 2, region).astype(np.int32)
    l_star = np.maximum(l_equatorial, l_axial)
    threshold_t3_abs_values = np.unique(np.concatenate(([a_abs], x[x > a_abs])))
    threshold_tperp_values = np.sqrt(np.maximum(threshold_t3_abs_values**2 - a_abs**2, 0.0))
    nontrivial_tie_t3_abs_values = threshold_t3_abs_values
    nontrivial_tie_tperp_values = threshold_tperp_values
    trivial_tie_t3_abs_values = x[x <= a_abs]
    trivial_tie_tperp_values = np.zeros_like(trivial_tie_t3_abs_values)

    np.savez(
        RESULTS / "aligned_diagonal_projective_regions.npz",
        a_abs=a_abs,
        t3_abs_values=x,
        tperp_values=y,
        t3_abs_grid=t3_abs_grid,
        tperp_grid=tperp_grid,
        L_equatorial=l_equatorial,
        L_axial=l_axial,
        L_star=l_star,
        region=region,
        region_code=region_code,
        tie_mask=tie_mask,
        margin=margin,
        threshold_t3_abs_values=threshold_t3_abs_values,
        threshold_tperp_values=threshold_tperp_values,
        nontrivial_tie_t3_abs_values=nontrivial_tie_t3_abs_values,
        nontrivial_tie_tperp_values=nontrivial_tie_tperp_values,
        trivial_tie_t3_abs_values=trivial_tie_t3_abs_values,
        trivial_tie_tperp_values=trivial_tie_tperp_values,
        is_physical_domain_diagram=False,
    )

    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    ax.contourf(
        t3_abs_grid, tperp_grid, region,
        levels=[-0.5, 0.5, 1.5],
        colors=["#E8E8E8", "#C8DCEC"],
        alpha=1.0,
    )
    ax.plot(threshold_t3_abs_values, threshold_tperp_values, color=OKABE_ITO["black"], linestyle="--",
            linewidth=1.2, label=r"$t_\perp^2=t_3^2-a^2$")
    ax.plot([a_abs, a_abs], [0, 0.02], color=OKABE_ITO["black"], linewidth=0.8)
    ax.text(0.17, 0.86, "equatorial\nmaximizer", transform=ax.transAxes,
            ha="center", va="center")
    ax.text(0.76, 0.17, "axial\nmaximizer", transform=ax.transAxes,
            ha="center", va="center")
    ax.set_xlabel(r"$|t_3|$")
    ax.set_ylabel(r"$t_\perp$")
    ax.set_xlim(0, 1.15)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, loc="upper right", handlelength=2.4)
    save_named_figure(fig, "figure1.png")
    plt.close(fig)
    return {
        "a_abs": a_abs,
        "t3_abs_grid": t3_abs_grid,
        "tperp_grid": tperp_grid,
        "L_equatorial": l_equatorial,
        "L_axial": l_axial,
        "L_star": l_star,
        "region": region,
        "region_code": region_code,
        "tie_mask": tie_mask,
        "margin": margin,
    }


def run_pure_states(settings: NumericalSettings) -> np.ndarray:
    rng = np.random.default_rng(settings.pure_seed)
    rows = []
    for index in tqdm(range(settings.pure_states), desc="Pure states"):
        rho = ketbra(random_pure_state(4, rng))
        a, _, tmat, max_imag = bloch_params_twoqubit(rho)
        lnum, _ = projective_max(
            a, tmat, rng,
            n_global=settings.pure_projective_samples,
            n_refine_starts=settings.pure_local_starts,
            refine_steps=settings.pure_local_steps,
        )
        rows.append((index, lnum, lnum - 1.0, np.linalg.norm(a), max_imag))
    data = np.array(rows, dtype=np.float64)
    np.savetxt(
        RESULTS / "pure_state_projective_identity.csv",
        data,
        delimiter=",",
        header="index,L_projective_numerical,residual,norm_a,max_imag_bloch",
        comments="",
    )
    return data


def run_random_mixed(settings: NumericalSettings) -> list[dict[str, float | int | str]]:
    state_sequences = np.random.SeedSequence(settings.mixed_master_seed).spawn(settings.mixed_states)
    rows: list[dict[str, float | int | str]] = []
    for index in tqdm(range(settings.mixed_states), desc="Random mixed states"):
        rng_state = np.random.default_rng(state_sequences[index])
        rho = random_density_matrix(4, rng_state)
        a, _, tmat, max_imag = bloch_params_twoqubit(rho)

        lproj, nproj, projective_batch_values = projective_reference_search(a, tmat, settings, index)

        rng_antipodal = np.random.default_rng(settings.mixed_antipodal_seed_base + 131 * index)
        lpartial_search, npartial = projective_max(
            a, tmat, rng_antipodal,
            n_global=settings.mixed_projective_samples,
            n_refine_starts=settings.mixed_local_starts,
            refine_steps=settings.mixed_local_steps,
        )
        alpha_partial, dirs_partial = antipodal_four_outcome(npartial, rng_antipodal)
        lpartial = float(lpovm4(a, tmat, alpha_partial, dirs_partial))
        partial_feas = feasibility_from_alpha(alpha_partial, dirs_partial)

        rng_povm = np.random.default_rng(settings.mixed_povm_seed_base + 131 * index)
        llinear, linear_wm = povm4_random_search(a, tmat, rng_povm, settings.mixed_linear_samples)
        if linear_wm is None:
            raise RuntimeError("Linear exact-completeness search failed to generate a POVM.")
        alpha_linear, dirs_linear = linear_wm
        linear_feas = feasibility_from_alpha(alpha_linear, dirs_linear)

        lstiefel, frame = stiefel_search(
            a, tmat, rng_povm,
            n_random=settings.mixed_stiefel_samples,
            n_restarts=settings.mixed_stiefel_restarts,
            refine_steps=settings.mixed_stiefel_steps,
        )
        stiefel_feas = feasibility_from_frame(frame)

        alpha_embed, dirs_embed = antipodal_four_outcome(nproj)
        lembed = float(lpovm4(a, tmat, alpha_embed, dirs_embed))
        embed_feas = feasibility_from_alpha(alpha_embed, dirs_embed)

        finite_no_embed_best = max(lpartial, llinear, lstiefel)
        finite_candidates = {
            "partial4_antipodal": lpartial,
            "linear4": llinear,
            "stiefel4": lstiefel,
        }
        finite_source = max(finite_candidates, key=finite_candidates.get)
        rows.append({
            "index": index,
            "rho_sha256": array_sha256(rho),
            "mixed_master_seed": settings.mixed_master_seed,
            "projective_seed_base": settings.mixed_projective_seed_base + 1009 * index,
            "partial4_seed": settings.mixed_antipodal_seed_base + 131 * index,
            "povm_seed": settings.mixed_povm_seed_base + 131 * index,
            "L_projective": lproj,
            "L_projective_batch_min": float(min(projective_batch_values)),
            "L_projective_batch_max": float(max(projective_batch_values)),
            "projective_batches": settings.mixed_projective_batches,
            "projective_reference_source": "projective_only_search",
            "projective_reference_policy": "projective_only_not_updated_by_finite_search",
            "projective_samples_per_batch": settings.mixed_projective_samples,
            "projective_local_starts": settings.mixed_local_starts,
            "projective_local_steps": settings.mixed_local_steps,
            "n_projective_x": float(nproj[0]),
            "n_projective_y": float(nproj[1]),
            "n_projective_z": float(nproj[2]),
            "L_partial4_antipodal": lpartial,
            "n_partial4_x": float(npartial[0]),
            "n_partial4_y": float(npartial[1]),
            "n_partial4_z": float(npartial[2]),
            "partial4_delta_L": lpartial - lproj,
            "partial4_abs_delta_L": abs(lpartial - lproj),
            "delta_L": lpartial - lproj,
            "abs_delta_L": abs(lpartial - lproj),
            "partial4_closure_residual": partial_feas["closure_residual"],
            "partial4_weight_sum_residual": partial_feas["weight_sum_residual"],
            "partial4_active_outcomes": partial_feas["active_outcomes"],
            "partial4_min_weight": partial_feas["min_weight"],
            "partial4_max_weight": partial_feas["max_weight"],
            "partial4_alpha_scheme": "random_pair_split",
            "partial4_uses_projective_direction": False,
            "L_linear4": llinear,
            "linear4_delta_L": llinear - lproj,
            "linear4_closure_residual": linear_feas["closure_residual"],
            "linear4_weight_sum_residual": linear_feas["weight_sum_residual"],
            "linear4_active_outcomes": linear_feas["active_outcomes"],
            "linear4_min_weight": linear_feas["min_weight"],
            "linear4_max_weight": linear_feas["max_weight"],
            "linear4_accepted_samples": settings.mixed_linear_samples,
            "L_stiefel4": lstiefel,
            "stiefel4_delta_L": lstiefel - lproj,
            "stiefel4_operator_residual": stiefel_feas["operator_residual"],
            "stiefel4_closure_residual": stiefel_feas["closure_residual"],
            "stiefel4_weight_sum_residual": stiefel_feas["weight_sum_residual"],
            "stiefel4_active_outcomes": stiefel_feas["active_outcomes"],
            "stiefel4_min_weight": stiefel_feas["min_weight"],
            "stiefel4_max_weight": stiefel_feas["max_weight"],
            "L_embedded_projective": lembed,
            "embedded_delta_L": lembed - lproj,
            "embedded_closure_residual": embed_feas["closure_residual"],
            "embedded_weight_sum_residual": embed_feas["weight_sum_residual"],
            "embedded_active_outcomes": embed_feas["active_outcomes"],
            "embedded_min_weight": embed_feas["min_weight"],
            "embedded_max_weight": embed_feas["max_weight"],
            "embedded_included_in_reported_delta": False,
            "embedded_included_in_no_exceed_test": False,
            "finite_no_embed_best": finite_no_embed_best,
            "finite_no_embed_excess_over_projective": finite_no_embed_best - lproj,
            "max_independent_finite_value": finite_no_embed_best,
            "max_independent_finite_source": finite_source,
            "max_independent_finite_minus_projective": finite_no_embed_best - lproj,
            "finite_exceeds_projective_tol_1e-10": finite_no_embed_best - lproj > 1.0e-10,
            "max_imag_bloch": max_imag,
        })

    csv_path = RESULTS / "random_mixed_state_povm_equality.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    index = np.array([row["index"] for row in rows], dtype=int)
    abs_delta = np.maximum(np.array([row["abs_delta_L"] for row in rows], dtype=float), 1e-18)
    closure = np.maximum(np.array([row["partial4_closure_residual"] for row in rows], dtype=float), 1e-18)

    with plt.rc_context(FIG4_FONT_RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.06, 2.795), sharex=False)
        ax1.semilogy(index, abs_delta, color=OKABE_ITO["blue"], marker="o",
                     markerfacecolor="none", linestyle="-", linewidth=1.0)
        ax1.axhline(1e-15, color=OKABE_ITO["gray"], linestyle="--", linewidth=0.9)
        ax1.set_xlabel("state index")
        ax1.set_ylabel(r"$|\Delta\mathcal{L}|$")
        ax1.text(0.02, 0.94, "(a)", transform=ax1.transAxes, va="top", ha="left")

        ax2.semilogy(index, closure, color=OKABE_ITO["green"], marker="s",
                     markerfacecolor="none", linestyle="-", linewidth=1.0)
        ax2.axhline(1e-15, color=OKABE_ITO["gray"], linestyle="--", linewidth=0.9)
        ax2.set_xlabel("state index")
        ax2.set_ylabel(r"$\varepsilon_{\rm comp}$")
        ax2.text(0.02, 0.94, "(b)", transform=ax2.transAxes, va="top", ha="left")
        fig.subplots_adjust(wspace=0.56)
        save_named_figure(fig, "figure4.png")
        plt.close(fig)
    return rows


def write_summary(werner: np.ndarray, pure: np.ndarray, mixed_rows: list[dict[str, float | int | str]]) -> dict[str, float | int | str]:
    partial_or_four_active_excess = np.array(
        [float(row["max_independent_finite_minus_projective"]) for row in mixed_rows],
        dtype=float,
    )
    four_active_excess = np.array(
        [max(float(row["linear4_delta_L"]), float(row["stiefel4_delta_L"])) for row in mixed_rows],
        dtype=float,
    )
    summary = {
        "werner_max_abs_residual": float(np.max(np.abs(werner[:, 3]))),
        "pure_state_max_abs_residual": float(np.max(np.abs(pure[:, 2]))),
        "random_mixed_max_abs_delta_L": float(max(row["abs_delta_L"] for row in mixed_rows)),
        "random_mixed_max_abs_delta_partial4": float(max(row["partial4_abs_delta_L"] for row in mixed_rows)),
        "random_mixed_max_signed_delta_L": float(max(row["delta_L"] for row in mixed_rows)),
        "random_mixed_max_signed_delta_partial4": float(max(row["partial4_delta_L"] for row in mixed_rows)),
        "random_mixed_min_signed_delta_partial4": float(min(row["partial4_delta_L"] for row in mixed_rows)),
        "random_mixed_max_partial4_closure_residual": float(max(row["partial4_closure_residual"] for row in mixed_rows)),
        "random_mixed_max_partial4_weight_residual": float(max(row["partial4_weight_sum_residual"] for row in mixed_rows)),
        "random_mixed_max_linear4_delta_L": float(max(row["linear4_delta_L"] for row in mixed_rows)),
        "random_mixed_max_stiefel4_delta_L": float(max(row["stiefel4_delta_L"] for row in mixed_rows)),
        "random_mixed_max_embedded_abs_delta_L": float(max(abs(row["embedded_delta_L"]) for row in mixed_rows)),
        "random_mixed_max_partial4_or_four_active_minus_projective": float(np.max(partial_or_four_active_excess)),
        "random_mixed_max_four_active_minus_projective": float(np.max(four_active_excess)),
        "random_mixed_num_partial4_or_four_active_exceeds_projective_tol_1e-10": int(np.sum(partial_or_four_active_excess > 1.0e-10)),
        "random_mixed_num_four_active_exceeds_projective_tol_1e-10": int(np.sum(four_active_excess > 1.0e-10)),
        "random_mixed_min_linear4_weight": float(min(row["linear4_min_weight"] for row in mixed_rows)),
        "random_mixed_min_stiefel4_weight": float(min(row["stiefel4_min_weight"] for row in mixed_rows)),
        "random_mixed_min_partial4_weight": float(min(row["partial4_min_weight"] for row in mixed_rows)),
        "projective_reference_policy": "projective_only_not_updated_by_finite_search",
        "figure4_source_csv_sha256": file_sha256(RESULTS / "random_mixed_state_povm_equality.csv"),
        "figure4_png_sha256": file_sha256(FIGURE_DIRS[0] / "figure4.png"),
    }
    with open(RESULTS / "numerical_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    refresh_output_manifest()
    return summary


def run_all_calculations() -> dict[str, float | int | str]:
    configure_runtime()
    ensure_dirs()
    settings = NumericalSettings()
    with open(RESULTS / "numerical_settings.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(settings), handle, indent=2)

    run_aligned_diagonal()
    werner = run_werner(settings)
    pure = run_pure_states(settings)
    mixed = run_random_mixed(settings)
    summary = write_summary(werner, pure, mixed)
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    if argv:
        raise SystemExit("This script does not take command-line arguments.")
    run_all_calculations()


if __name__ == "__main__":
    main(sys.argv[1:])
