"""Experiment 7 – Asymptotic Risk Sharing for Unbounded Total Risk.

Verifies the asymptotic optimality of threshold allocations X_i^(m) for
unbounded-above distributions.  The gap formula

    sum_i R_{I_i}(X_i^(m)) - R_{[0,beta]}(X) = (1/beta) * E[(X - a_m) * 1_{X > a_m}]

is verified numerically for increasing threshold m, and convergence to zero
is demonstrated.

Run directly:
    python exp/exp7_risk_sharing_unbounded.py

Results saved to /workspace/project/results/exp7_results.csv.
Convergence plot saved to /workspace/project/results/exp7_convergence.png.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate, stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exp.exp6_risk_sharing_bounded import compute_agent_objective

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
# Gap formula
# ---------------------------------------------------------------------------


def compute_gap_formula(dist, beta: float, m: float, n: int) -> float:
    """Exact theorem gap: (1/beta) * E[(X - a_m) * 1_{X > a_m}].

    a_m = n * m / (n - 1)

    Computed via numerical integration of (x - a_m) * f(x) over [a_m, inf).
    """
    if n <= 1:
        raise ValueError("n must be > 1")
    a_m = n * m / (n - 1)
    try:
        val, _ = integrate.quad(
            lambda x: (x - a_m) * dist.pdf(x),
            a_m,
            np.inf,
            limit=500,
        )
    except Exception:
        val = 0.0
    return val / beta


# ---------------------------------------------------------------------------
# Threshold allocation construction
# ---------------------------------------------------------------------------


def construct_allocation_unbounded(
    dist, n: int, betas: list, m: float, N: int = 5000
) -> tuple:
    """Construct the threshold allocation X_i^(m).

    Grid of N equal-probability scenarios:
        U = linspace(1/(2N), 1-1/(2N), N)
        X = dist.ppf(U)

    For each agent i:
        lower_i = sum(betas[:i]),  upper_i = sum(betas[:i+1])
        A_i     = (U > lower_i) & (U <= upper_i)
        A       = U <= beta
        X_i^(m) = (X - m)*1_{A_i} + (X/n)*1_{A^c} + (m/(n-1))*1_{A\\A_i}

    Returns
    -------
    (Xi_list, U, X)
    """
    U = np.linspace(1.0 / (2 * N), 1.0 - 1.0 / (2 * N), N)
    X = dist.ppf(U)

    beta = sum(betas)
    A = U <= beta

    Xi_list = []
    for i in range(n):
        lower_i = sum(betas[:i])
        upper_i = sum(betas[:i + 1])
        A_i = (U > lower_i) & (U <= upper_i)
        Xi = np.where(A_i, X - m, np.where(~A, X / n, m / (n - 1)))
        Xi_list.append(Xi)

    return Xi_list, U, X


# ---------------------------------------------------------------------------
# Total empirical objective
# ---------------------------------------------------------------------------


def compute_total_objective(
    Xi_list: list, U: np.ndarray, betas: list, beta: float
) -> float:
    """sum_i R_{I_i}(X_i^(m)) computed empirically."""
    N = len(U)
    return sum(
        compute_agent_objective(Xi_list[i], U, betas[i], beta, N)
        for i in range(len(Xi_list))
    )


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def run_exp7():
    """Run all unbounded risk-sharing scenarios and save results + plot."""
    configs = [
        {
            "dist": stats.expon(scale=1),
            "name": "Exponential1",
            "n": 2,
            "betas": [0.15, 0.20],
            "lam": 0.4,
        },
        {
            "dist": stats.lognorm(s=1, scale=1),
            "name": "Lognormal01",
            "n": 2,
            "betas": [0.15, 0.20],
            "lam": 0.4,
        },
        {
            "dist": stats.pareto(b=2),
            "name": "Pareto2",
            "n": 2,
            "betas": [0.10, 0.15],
            "lam": 0.3,
        },
    ]

    m_vals_all = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0, 25.0]
    N = 5000
    rows = []

    fig, axes = plt.subplots(1, len(configs), figsize=(15, 5))
    if len(configs) == 1:
        axes = [axes]

    for ax, cfg in zip(axes, configs):
        dist = cfg["dist"]
        name = cfg["name"]
        n = cfg["n"]
        betas = cfg["betas"]
        beta = sum(betas)

        # Compute analytical inf-convolution target
        target = avg_quantile_interval(dist, 0.0, beta)

        # Minimum admissible m: m >= max(q_beta(X), max_i q_{1-beta+bi}(X))_+
        min_m = max(
            0.0,
            float(dist.ppf(beta)),
            max(float(dist.ppf(1.0 - beta + bi)) for bi in betas),
        )
        m_vals = [m for m in m_vals_all if m >= min_m]

        print(f"\n[{name}]  beta={beta:.3f}  target={target:.6f}  min_m={min_m:.3f}")
        print(f"  m values: {m_vals}")

        realized_gaps = []
        predicted_gaps = []

        for m in m_vals:
            Xi_list, U, X = construct_allocation_unbounded(dist, n, betas, m, N=N)

            # Feasibility
            feas_err = float(np.max(np.abs(sum(Xi_list) - X)))
            assert feas_err < 1e-10, f"{name} m={m}: feasibility error {feas_err:.2e}"

            total_obj = compute_total_objective(Xi_list, U, betas, beta)
            realized_gap = total_obj - target
            predicted_gap = compute_gap_formula(dist, beta, m, n)

            assert realized_gap > -1e-6, (
                f"{name} m={m}: realized_gap={realized_gap:.6f} should be >= 0"
            )

            gap_error = abs(realized_gap - predicted_gap)
            realized_gaps.append(realized_gap)
            predicted_gaps.append(predicted_gap)

            rows.append(
                {
                    "dist_name": name,
                    "n": n,
                    "beta": beta,
                    "m": m,
                    "total_obj": total_obj,
                    "target": target,
                    "realized_gap": realized_gap,
                    "predicted_gap": predicted_gap,
                    "gap_error": gap_error,
                    "feasibility_err": feas_err,
                }
            )
            print(
                f"  m={m:5.1f}  total_obj={total_obj:.6f}  "
                f"realized_gap={realized_gap:.6f}  predicted_gap={predicted_gap:.6f}  "
                f"gap_err={gap_error:.4f}"
            )

        # Verify convergence: gaps decrease as m increases
        for i in range(len(realized_gaps) - 1):
            assert realized_gaps[i] >= realized_gaps[i + 1] - 0.05, (
                f"{name}: realized gap not decreasing at m index {i}"
            )

        # Convergence plot
        ax.semilogy(m_vals, predicted_gaps, "b-o", label="predicted gap", markersize=5)
        ax.semilogy(m_vals, [max(g, 1e-10) for g in realized_gaps],
                    "r--s", label="realized gap", markersize=5)
        ax.set_xlabel("threshold m")
        ax.set_ylabel("gap (log scale)")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.4)

    fig.suptitle("Exp 7 – Gap Convergence as m → ∞", fontsize=13)
    fig.tight_layout()

    os.makedirs("/workspace/project/results", exist_ok=True)
    fig.savefig("/workspace/project/results/exp7_convergence.png", dpi=120)
    plt.close(fig)
    print("\nSaved: /workspace/project/results/exp7_convergence.png")

    df = pd.DataFrame(rows)
    df.to_csv("/workspace/project/results/exp7_results.csv", index=False)
    print("Saved: /workspace/project/results/exp7_results.csv")
    return df


if __name__ == "__main__":
    run_exp7()
