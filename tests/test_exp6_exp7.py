"""Tests for Experiments 6 and 7 – Risk Sharing (Bounded and Unbounded)."""

import numpy as np
import pytest
from scipy import stats

from exp.exp6_risk_sharing_bounded import (
    avg_quantile_interval,
    compute_agent_objective,
    compute_distortion_risk,
    compute_inf_conv_target,
    construct_allocation,
    verify_feasibility,
)
from exp.exp7_risk_sharing_unbounded import (
    compute_gap_formula,
    compute_total_objective,
    construct_allocation_unbounded,
)
from src.operators import avg_quantile_interval  # noqa: F811 – same function


def test_feasibility_uniform():
    """sum_i X_i = X exactly (up to floating-point noise) for uniform."""
    N = 1000
    dist = stats.uniform(0, 1)
    betas = [0.15, 0.20]
    X_arrays, U, X = construct_allocation(dist, 2, betas, N=N)
    err = verify_feasibility(X_arrays, X)
    assert err < 1e-8


def test_inf_conv_target_uniform():
    """R_{[0,0.35]}(Uniform[0,1]) = 0.35/2 = 0.175."""
    dist = stats.uniform(0, 1)
    beta = 0.35
    target = compute_inf_conv_target(dist, beta)
    assert abs(target - beta / 2) < 0.01


def test_sum_objectives_equals_target():
    """sum_i R_{I_i}(X_i) ≈ R_{[0,beta]}(X) for the optimal bounded allocation."""
    N = 3000
    dist = stats.uniform(0, 1)
    n, betas = 2, [0.15, 0.20]
    beta = sum(betas)
    X_arrays, U, X = construct_allocation(dist, n, betas, N=N)
    target = compute_inf_conv_target(dist, beta)
    total_obj = sum(
        compute_agent_objective(Xi, U, betas[i], beta, N)
        for i, Xi in enumerate(X_arrays)
    )
    assert abs(total_obj - target) < 0.05


def test_gap_formula_positive():
    """Gap is strictly positive for finite m with unbounded X."""
    dist = stats.expon(scale=1)
    n, betas = 2, [0.15, 0.20]
    beta = sum(betas)
    m = 2.0
    gap = compute_gap_formula(dist, beta, m, n)
    assert gap > 0


def test_gap_convergence():
    """Gap decreases monotonically as m increases."""
    dist = stats.expon(scale=1)
    n, betas = 2, [0.15, 0.20]
    beta = sum(betas)
    gaps = [compute_gap_formula(dist, beta, m, n) for m in [1.0, 2.0, 5.0, 10.0]]
    for i in range(len(gaps) - 1):
        assert gaps[i] >= gaps[i + 1] - 1e-6, f"Gap not decreasing at m index {i}"


def test_gap_zero_limit():
    """As m → ∞ the gap vanishes for exponential distribution."""
    dist = stats.expon(scale=1)
    n, betas = 2, [0.15, 0.20]
    beta = sum(betas)
    gap_large = compute_gap_formula(dist, beta, 100.0, n)
    assert gap_large < 1e-3


def test_distortion_risk_formula():
    """sum_i rho_{g_{lam,i}}(X_i) ≈ lam*E[X] + (1-lam)*R_{[0,beta]}(X)."""
    N = 2000
    dist = stats.uniform(0, 1)
    n, betas, lam = 2, [0.15, 0.20], 0.4
    beta = sum(betas)
    X_arrays, U, X = construct_allocation(dist, n, betas, N=N)
    dist_sum = sum(
        compute_distortion_risk(Xi, U, lam, betas[i], beta)
        for i, Xi in enumerate(X_arrays)
    )
    E_X = np.mean(X)
    target = avg_quantile_interval(dist, 0, beta)
    formula = lam * E_X + (1 - lam) * target
    assert abs(dist_sum - formula) < 0.1
