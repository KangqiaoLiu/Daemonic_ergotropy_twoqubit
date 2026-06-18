"""Closed-form routines for two-qubit daemonic ergotropy.

The conventions match the paper.  A two-qubit state is represented by
Bloch data (a,b,T), where a is the battery Bloch vector and T is the 3 x 3
correlation tensor.  The optimized average conditional battery Bloch length is

    L_star = sqrt(|a|^2 + max(0, lambda_max(T T^T - a a^T))).

The optimized daemonic ergotropy for H = (omega/2)(I - h.sigma) is

    W_D^star = (omega/2)(L_star - a.h).

The optimized daemonic gain is

    Delta W_D^star = (omega/2)(L_star - |a|).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
from typing import Iterable

import numpy as np

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = (SX, SY, SZ)


@dataclass(frozen=True)
class ClosedFormResult:
    L_star: float
    lambda_max: float
    positive_part: float
    optimal_axis: np.ndarray | None
    contact_vector: np.ndarray | None


def random_density_matrix(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Hilbert-Schmidt random density matrix."""
    g = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / np.sqrt(2.0)
    rho = g @ g.conj().T
    rho = rho / np.trace(rho)
    return 0.5 * (rho + rho.conj().T)


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    n = rng.normal(size=3)
    return n / np.linalg.norm(n)


def bloch_params_twoqubit(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (a,b,T) for a two-qubit density matrix."""
    rho = np.asarray(rho, dtype=complex).reshape(4, 4)
    a = np.zeros(3, dtype=float)
    b = np.zeros(3, dtype=float)
    T = np.zeros((3, 3), dtype=float)
    for i, si in enumerate(PAULIS):
        a[i] = np.trace(rho @ np.kron(si, I2)).real
        b[i] = np.trace(rho @ np.kron(I2, si)).real
        for j, sj in enumerate(PAULIS):
            T[i, j] = np.trace(rho @ np.kron(si, sj)).real
    return a, b, T


def projective_L(a: np.ndarray, T: np.ndarray, n: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(3)
    T = np.asarray(T, dtype=float).reshape(3, 3)
    n = np.asarray(n, dtype=float).reshape(3)
    n = n / np.linalg.norm(n)
    v = T @ n
    return 0.5 * (np.linalg.norm(a + v) + np.linalg.norm(a - v))


def closed_form(a: np.ndarray, T: np.ndarray, tol: float = 1.0e-12) -> ClosedFormResult:
    a = np.asarray(a, dtype=float).reshape(3)
    T = np.asarray(T, dtype=float).reshape(3, 3)
    M = T @ T.T - np.outer(a, a)
    vals, vecs = np.linalg.eigh(0.5 * (M + M.T))
    lam = float(vals[-1])
    s = max(0.0, lam)
    L_star = float(np.sqrt(np.dot(a, a) + s))

    axis = None
    contact = None
    if lam > tol:
        u = vecs[:, -1]
        n = T.T @ u
        n_norm = np.linalg.norm(n)
        if n_norm > tol:
            axis = n / n_norm
            contact = T @ axis
    return ClosedFormResult(L_star=L_star, lambda_max=lam, positive_part=s,
                            optimal_axis=axis, contact_vector=contact)


def L_star(a: np.ndarray, T: np.ndarray) -> float:
    return closed_form(a, T).L_star


def daemonic_ergotropy_star(a: np.ndarray, T: np.ndarray,
                            h: np.ndarray | None = None,
                            omega: float = 1.0) -> float:
    a = np.asarray(a, dtype=float).reshape(3)
    if h is None:
        h = np.array([0.0, 0.0, 1.0])
    h = np.asarray(h, dtype=float).reshape(3)
    h = h / np.linalg.norm(h)
    return 0.5 * omega * (L_star(a, T) - float(np.dot(a, h)))


def daemonic_gain_star(a: np.ndarray, T: np.ndarray, omega: float = 1.0) -> float:
    a = np.asarray(a, dtype=float).reshape(3)
    return 0.5 * omega * (L_star(a, T) - np.linalg.norm(a))


def L_star_visibility(a: np.ndarray, T: np.ndarray, eta: float) -> float:
    return L_star(a, eta * np.asarray(T, dtype=float))


def daemonic_gain_visibility(a: np.ndarray, T: np.ndarray, eta: float,
                             omega: float = 1.0) -> float:
    a = np.asarray(a, dtype=float).reshape(3)
    return 0.5 * omega * (L_star_visibility(a, T, eta) - np.linalg.norm(a))


def pure_schmidt_bloch(concurrence: float) -> tuple[np.ndarray, np.ndarray]:
    """Bloch data for sqrt(q)|00> + sqrt(1-q)|11>, parameterized by concurrence."""
    C = float(concurrence)
    if C < -1.0e-12 or C > 1.0 + 1.0e-12:
        raise ValueError("concurrence must be in [0,1]")
    C = min(1.0, max(0.0, C))
    z = np.sqrt(max(0.0, 1.0 - C * C))
    return np.array([0.0, 0.0, z]), np.diag([C, -C, 1.0])


def save_csv(path: Path, header: Iterable[str], rows: Iterable[Iterable[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(list(header))
        writer.writerows(rows)


def validate_against_random_axes(results_dir: Path, *, n_states: int = 80,
                                  n_axes: int = 20000, seed: int = 314159) -> dict[str, float]:
    """Check that random projective-axis sampling never exceeds the closed form."""
    rng = np.random.default_rng(seed)
    rows = []
    worst_overshoot = -np.inf
    largest_gap = 0.0
    for index in range(n_states):
        rho = random_density_matrix(4, rng)
        a, _b, T = bloch_params_twoqubit(rho)
        exact = L_star(a, T)
        sampled = 0.0
        for _ in range(n_axes):
            sampled = max(sampled, projective_L(a, T, random_unit_vector(rng)))
        gap = exact - sampled
        worst_overshoot = max(worst_overshoot, sampled - exact)
        largest_gap = max(largest_gap, gap)
        rows.append((index, exact, sampled, gap))

    save_csv(results_dir / "closed_form_random_axis_validation.csv",
             ["index", "L_closed_form", "L_best_random_axis", "closed_minus_sampled"], rows)
    summary = {
        "seed": seed,
        "n_states": n_states,
        "n_random_axes_per_state": n_axes,
        "largest_closed_minus_sampled": float(largest_gap),
        "largest_sampled_minus_closed": float(worst_overshoot),
    }
    with open(results_dir / "closed_form_validation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    summary = validate_against_random_axes(root / "Results")
    print(json.dumps(summary, indent=2))
