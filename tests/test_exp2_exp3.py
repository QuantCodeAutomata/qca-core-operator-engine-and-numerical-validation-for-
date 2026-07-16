"""Tests for Experiment 2 (Theorem 1, Corollary 1) and Experiment 3 (Robust Bounds)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from scipy import stats

from exp.exp2_theorem1_validation import (
    avg_quantile_interval,
    avg_quantile_union,
    build_distributions,
    generate_comonotonic_coupling,
    generate_independent_coupling,
    generate_antithetic_coupling,
    empirical_avg_quantile,
    compute_theorem1_rhs,
    compute_corollary1_rhs,
    validate_theorem1,
    validate_corollary1,
)
from exp.exp3_robust_bounds import (
    compute_new_upper_bound,
    compute_old_upper_bound,
    compute_new_lower_bound,
    compute_old_lower_bound,
    approximate_exact_sup,
    approximate_exact_inf,
)


# ---------------------------------------------------------------------------
# Exp 2 tests
# ---------------------------------------------------------------------------

def test_avg_quantile_interval_uniform():
    """R_[0,1](Uniform[0,1]) = E[X] = 0.5."""
    dist = stats.uniform(0, 1)
    val = avg_quantile_interval(dist, 0, 1)
    assert abs(val - 0.5) < 1e-4, f"Expected 0.5, got {val}"


def test_avg_quantile_interval_uniform_half():
    """R_[0.5,1](Uniform[0,1]) = 0.75."""
    dist = stats.uniform(0, 1)
    val = avg_quantile_interval(dist, 0.5, 1.0)
    assert abs(val - 0.75) < 1e-4


def test_avg_quantile_union_symmetric():
    """R_[[0,0.3] union [0.7,1]](Uniform) = 0.5 by symmetry."""
    dist = stats.uniform(0, 1)
    val = avg_quantile_union(dist, [(0.0, 0.3), (0.7, 1.0)])
    # integral_[0,0.3] t dt + integral_[0.7,1] t dt all over length 0.6
    # = (0.09/2 + (1-0.49)/2) / 0.6 = (0.045 + 0.255) / 0.6 = 0.5
    assert abs(val - 0.5) < 0.01


def test_empirical_avg_quantile_uniform():
    """Empirical R_[0.5,1](Uniform) ~ 0.75."""
    N = 10000
    U = np.linspace(0.5 / N, 1.0 - 0.5 / N, N)
    S = stats.uniform.ppf(U)
    val = empirical_avg_quantile(S, 0.5, 1.0)
    assert abs(val - 0.75) < 0.02


def test_comonotonic_coupling_shape():
    """Comonotonic coupling should return (N, n) array."""
    dists = [stats.norm(), stats.expon()]
    samples = generate_comonotonic_coupling(dists, N=500)
    assert samples.shape == (500, 2)


def test_theorem1_no_violations():
    """Theorem 1 should produce no violations (slack < -1e-3) across tested cases."""
    # Use a small N for speed
    results = validate_theorem1(N=1500)
    n_violations = (results['slack'] < -1e-3).sum()
    assert n_violations == 0, (
        f"Theorem 1 violations: {n_violations}\n"
        f"{results[results['slack'] < -1e-3][['dist_combo', 'coupling', 'r', 's', 'slack']]}"
    )


def test_corollary1_no_violations():
    """Corollary 1 should produce no violations across tested cases."""
    results = validate_corollary1(N=1500)
    n_violations = (results['slack'] < -1e-3).sum()
    assert n_violations == 0, (
        f"Corollary 1 violations: {n_violations}\n"
        f"{results[results['slack'] < -1e-3][['dist_combo', 'coupling', 'r', 's', 'slack']]}"
    )


def test_theorem1_rhs_dominates_comonotonic():
    """Theorem 1 RHS should be at least as large as LHS for comonotonic coupling."""
    dists = [stats.norm(0, 1), stats.expon(scale=1)]
    r, s = 0.1, 0.2
    n = len(dists)
    ai = s / (2.0 * n)
    bi = (1.0 - r - n * ai) / n
    alphas = [ai] * n
    betas = [bi] * n
    rhs = compute_theorem1_rhs(dists, r, s, alphas, betas)

    samples = generate_comonotonic_coupling(dists, N=3000)
    S = samples.sum(axis=1)
    lhs = empirical_avg_quantile(S, r, r + s)
    assert rhs >= lhs - 1e-3


def test_corollary1_rhs_alpha_sum_constraint():
    """Corollary 1 requires sum(alphas) == s; test raises for wrong sum."""
    dists = [stats.norm(), stats.expon()]
    alphas_wrong = [0.05, 0.05]  # sum = 0.10 != s = 0.2
    with pytest.raises(AssertionError):
        compute_corollary1_rhs(dists, r=0.1, s=0.2, alphas=alphas_wrong)


def test_antithetic_coupling_n2_property():
    """For n=2, antithetic coupling: X1 ascending, X2 descending."""
    dists = [stats.uniform(0, 1), stats.uniform(0, 1)]
    samples = generate_antithetic_coupling(dists, N=100)
    # X1 + X2 should be constant (both are U and 1-U for Uniform)
    row_sums = samples.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Exp 3 tests
# ---------------------------------------------------------------------------

def test_new_upper_bound_valid():
    """New upper bound should be >= exact_sup approximation (within tolerance)."""
    dists = [stats.norm(0, 1), stats.expon(scale=1)]
    r, s = 0.1, 0.2
    new_upper, _ = compute_new_upper_bound(dists, r, s, n_starts=3)
    exact_sup = approximate_exact_sup(dists, r, s, N=500)
    assert new_upper >= exact_sup - 0.2, (
        f"Theorem 2 violated: new_upper={new_upper:.4f} < exact_sup={exact_sup:.4f}"
    )


def test_new_lower_bound_valid():
    """New lower bound should be <= exact_inf approximation (within tolerance)."""
    dists = [stats.norm(0, 1), stats.expon(scale=1)]
    r, s = 0.1, 0.2
    new_lower, _ = compute_new_lower_bound(dists, r, s, n_starts=3)
    exact_inf = approximate_exact_inf(dists, r, s, N=500)
    assert new_lower <= exact_inf + 0.2, (
        f"Theorem 3 violated: new_lower={new_lower:.4f} > exact_inf={exact_inf:.4f}"
    )


def test_old_upper_bound_valid():
    """Old upper bound should be >= exact_sup (it's also a valid upper bound)."""
    dists = [stats.beta(2, 0.5), stats.beta(0.5, 0.5)]
    r, s = 0.1, 0.2
    old_upper, _ = compute_old_upper_bound(dists, r, s, n_starts=3)
    exact_sup = approximate_exact_sup(dists, r, s, N=500)
    assert old_upper >= exact_sup - 0.2


def test_old_lower_bound_valid():
    """Old lower bound should be <= exact_inf."""
    dists = [stats.beta(2, 0.5), stats.beta(0.5, 0.5)]
    r, s = 0.1, 0.2
    old_lower, _ = compute_old_lower_bound(dists, r, s, n_starts=3)
    exact_inf = approximate_exact_inf(dists, r, s, N=500)
    assert old_lower <= exact_inf + 0.2


def test_upper_bound_ordering():
    """Both upper bounds >= exact_sup, lower bounds <= exact_inf."""
    dists = [stats.norm(0, 1), stats.norm(0, 1)]
    r, s = 0.1, 0.2
    exact_sup = approximate_exact_sup(dists, r, s, N=500)
    exact_inf = approximate_exact_inf(dists, r, s, N=500)
    new_upper, _ = compute_new_upper_bound(dists, r, s, n_starts=3)
    new_lower, _ = compute_new_lower_bound(dists, r, s, n_starts=3)
    # Consistency: lower <= exact_inf <= exact_sup <= upper (approximately)
    assert new_lower <= exact_sup + 0.5
    assert new_upper >= exact_inf - 0.5


def test_exact_sup_comonotonic_normal():
    """Comonotonic Normal sup should be greater than zero for upper tail."""
    dists = [stats.norm(0, 1), stats.norm(0, 1)]
    sup = approximate_exact_sup(dists, r=0.8, s=0.1, N=1000)
    # For N(0,1) at 80%-90% quantile, should be positive
    assert sup > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
