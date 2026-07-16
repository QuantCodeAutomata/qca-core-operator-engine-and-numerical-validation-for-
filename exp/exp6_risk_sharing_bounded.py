"""Experiment 6 – Risk Sharing with Bounded Total Risk.

Verifies risk-sharing results for averaged quantiles when the total risk X
is bounded above.  The inf-convolution of agent risk measures R_{I_i} equals
R_{[0,beta]}(X), and the explicit optimal allocation is constructed and
verified numerically.

Run directly:
    python exp/exp6_risk_sharing_bounded.py

Results are saved to /workspace/project/results/exp6_results.csv.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import integrate, stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Core operators (scipy frozen-distribution interface)
# ---------------------------------------------------------------------------


def avg_quantile_interval(dist, a: float, b: float) -> float:
    a = max(a, 1e-10)
    b = min(b, 1 - 1e-10)
    if a >= b:
        return float(dist.ppf(a))
    result, _ = integrate.quad(lambda t: float(dist.ppf(t)), a, b, limit=200, epsabs=1e-6)
    return result / (b - a)


def avg_quantile_union(dist, intervals: list) -> float:
    intervals = [(max(a, 1e-10), min(b, 1 - 1e-10)) for a, b in intervals if b > a]
    total_len = sum(b - a for a, b in intervals)
    if total_len < 1e-15:
        return 0.0
    total_int = sum(
        integrate.quad(lambda t: float(dist.ppf(t)), a, b, limit=200, epsabs=1e-6)[0]
        for a, b in intervals
    )
    return total_int / total_len


# ---------------------------------------------------------------------------
# TotalRisk container
# ---------------------------------------------------------------------------


class TotalRisk:
    """Container for the risk-sharing problem setup.

    Parameters
    ----------
    dist   : scipy frozen distribution for X.
    n      : number of agents.
    betas  : list of n floats, beta_i parameters (each in (0,1), sum < 1).
    lam    : distortion-mix weight lambda (default 0.4).
    """

    def __init__(self, dist, n: int, betas: list, lam: float = 0.4):
        self.dist = dist
        self.n = n
        self.betas = list(betas)
        self.lam = lam

    @property
    def beta(self) -> float:
        return sum(self.betas)

    def I_i(self, i: int) -> list:
        """Return the interval set I_i = [0, beta_i] ∪ [1-beta+beta_i, 1]."""
        bi = self.betas[i]
        beta = self.beta
        return [(0.0, bi), (1.0 - beta + bi, 1.0)]


# ---------------------------------------------------------------------------
# Probability transform
# ---------------------------------------------------------------------------


def compute_prob_transform(dist, x_vals: np.ndarray) -> np.ndarray:
    """U_X = F_X(X) for a continuous distribution."""
    return dist.cdf(x_vals)


# ---------------------------------------------------------------------------
# Allocation construction
# ---------------------------------------------------------------------------


def construct_allocation(dist, n: int, betas: list, N: int = 5000) -> tuple:
    """Construct the optimal X_i allocations on a uniform quantile grid.

    Grid of N equal-probability scenarios:
        U = linspace(1/(2N), 1-1/(2N), N)
        X = dist.ppf(U)

    Upper bound approximation:
        t = dist.support()[1] if finite, else dist.ppf(1 - 1e-8)

    For each agent i:
        lower_i = sum(betas[:i]),  upper_i = sum(betas[:i+1])
        A_i     = (U > lower_i) & (U <= upper_i)          ← subset of A
        A       = U <= beta
        X_i     = (X - t)*1_{A_i} + (X/n)*1_{A^c} + (t/(n-1))*1_{A\\A_i}

    Returns
    -------
    (Xi_list, U, X)
        Xi_list : list of n numpy arrays, each of length N
        U       : probability grid, shape (N,)
        X       : scenario values, shape (N,)
    """
    U = np.linspace(1.0 / (2 * N), 1.0 - 1.0 / (2 * N), N)
    X = dist.ppf(U)

    sup = dist.support()[1]
    t = float(sup) if np.isfinite(sup) else float(dist.ppf(1 - 1e-8))

    beta = sum(betas)
    A = U <= beta

    Xi_list = []
    for i in range(n):
        lower_i = sum(betas[:i])
        upper_i = sum(betas[:i + 1])
        A_i = (U > lower_i) & (U <= upper_i)
        # X_i = (X-t)*1_{A_i} + (X/n)*1_{A^c} + (t/(n-1))*1_{A\A_i}
        Xi = np.where(A_i, X - t, np.where(~A, X / n, t / (n - 1)))
        Xi_list.append(Xi)

    return Xi_list, U, X


# ---------------------------------------------------------------------------
# Feasibility check
# ---------------------------------------------------------------------------


def verify_feasibility(X_arrays: list, X: np.ndarray) -> float:
    """Return max |sum_i X_i - X| (should be near machine epsilon)."""
    return float(np.max(np.abs(sum(X_arrays) - X)))


# ---------------------------------------------------------------------------
# Empirical risk measure R_{I_i}(X_i)
# ---------------------------------------------------------------------------


def compute_agent_objective(
    Xi: np.ndarray, U: np.ndarray, beta_i: float, beta: float, N: int = 5000
) -> float:
    """R_{I_i}(X_i) = (1/beta) * [integral_0^{beta_i} + integral_{1-beta+beta_i}^1] q_t(X_i) dt.

    Approximated empirically via a sorted uniform-grid sample of size N:
        lower_part = sorted_Xi[:int(beta_i * N)]        (bottom beta_i fraction)
        upper_part = sorted_Xi[int((1-beta+beta_i)*N):] (top (beta-beta_i) fraction)
        R_{I_i} = (sum(lower_part) + sum(upper_part)) / (beta * N)
    """
    sorted_Xi = np.sort(Xi)
    lo_idx = int(beta_i * N)
    up_idx = int((1.0 - beta + beta_i) * N)
    lower_part = sorted_Xi[:lo_idx]
    upper_part = sorted_Xi[up_idx:]
    return (np.sum(lower_part) + np.sum(upper_part)) / (beta * N)


# ---------------------------------------------------------------------------
# Theoretical inf-convolution target
# ---------------------------------------------------------------------------


def compute_inf_conv_target(dist, beta: float) -> float:
    """R_{[0,beta]}(X) = (1/beta) * integral_0^beta q_t^-(X) dt = LES_beta(X)."""
    return avg_quantile_interval(dist, 0.0, beta)


# ---------------------------------------------------------------------------
# Distortion risk measure
# ---------------------------------------------------------------------------


def compute_distortion_risk(
    Xi: np.ndarray, U: np.ndarray, lam: float, beta_i: float, beta: float
) -> float:
    """rho_{g_{lambda,i}}(X_i) = lambda*E[X_i] + (1-lambda)*R_{I_i}(X_i)."""
    E_Xi = float(np.mean(Xi))
    R_Ii = compute_agent_objective(Xi, U, beta_i, beta, len(Xi))
    return lam * E_Xi + (1.0 - lam) * R_Ii


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def run_exp6():
    """Run all bounded risk-sharing scenarios and save results to CSV."""
    dist_configs = [
        {
            "dist": stats.uniform(0, 1),
            "name": "Uniform01",
            "n": 2,
            "betas": [0.15, 0.20],
            "lam": 0.4,
        },
        {
            "dist": stats.beta(2, 0.5),
            "name": "Beta25",
            "n": 3,
            "betas": [0.10, 0.15, 0.20],
            "lam": 0.3,
        },
        {
            "dist": stats.beta(0.5, 0.5),
            "name": "Beta05",
            "n": 2,
            "betas": [0.20, 0.25],
            "lam": 0.4,
        },
    ]

    N = 5000
    rows = []

    for cfg in dist_configs:
        dist = cfg["dist"]
        name = cfg["name"]
        n = cfg["n"]
        betas = cfg["betas"]
        lam = cfg["lam"]
        beta = sum(betas)

        # 1. Inf-convolution target R_{[0,beta]}(X)
        target = compute_inf_conv_target(dist, beta)

        # 2. Dual worst-case target R_{[beta,1]}(X) (upper expected shortfall)
        dual_target = avg_quantile_interval(dist, beta, 1.0)

        # 3. Construct optimal allocation
        Xi_list, U, X = construct_allocation(dist, n, betas, N=N)

        # 4. Feasibility check
        feasibility_err = verify_feasibility(Xi_list, X)
        assert feasibility_err < 1e-10, (
            f"{name}: feasibility error {feasibility_err:.2e} exceeds 1e-10"
        )

        # 5. Agent risk objectives R_{I_i}(X_i)
        R_Ii_vals = [
            compute_agent_objective(Xi_list[i], U, betas[i], beta, N)
            for i in range(n)
        ]
        R_Ii_sum = sum(R_Ii_vals)
        obj_diff = abs(R_Ii_sum - target)

        # 6. Distortion objectives
        dist_vals = [
            compute_distortion_risk(Xi_list[i], U, lam, betas[i], beta)
            for i in range(n)
        ]
        distortion_sum = sum(dist_vals)

        # 7. Theoretical distortion formula: lambda*E[X] + (1-lambda)*R_{[0,beta]}(X)
        E_X = float(np.mean(X))
        distortion_formula = lam * E_X + (1.0 - lam) * target
        distortion_diff = abs(distortion_sum - distortion_formula)

        row = {
            "dist_name": name,
            "n": n,
            "beta": beta,
            "lam": lam,
            "target": target,
            "dual_target": dual_target,
            "feasibility_err": feasibility_err,
            "R_Ii_sum": R_Ii_sum,
            "obj_diff": obj_diff,
            "E_X": E_X,
            "distortion_sum": distortion_sum,
            "distortion_formula": distortion_formula,
            "distortion_diff": distortion_diff,
        }
        for i, (r, d) in enumerate(zip(R_Ii_vals, dist_vals)):
            row[f"R_I{i}"] = r
            row[f"dist_obj{i}"] = d

        rows.append(row)

        print(
            f"[{name}] beta={beta:.2f}  target={target:.6f}  "
            f"R_sum={R_Ii_sum:.6f}  diff={obj_diff:.2e}  "
            f"feasErr={feasibility_err:.2e}  distDiff={distortion_diff:.2e}"
        )

    df = pd.DataFrame(rows)
    os.makedirs("/workspace/project/results", exist_ok=True)
    df.to_csv("/workspace/project/results/exp6_results.csv", index=False)
    print("Saved: /workspace/project/results/exp6_results.csv")
    return df


if __name__ == "__main__":
    run_exp6()
