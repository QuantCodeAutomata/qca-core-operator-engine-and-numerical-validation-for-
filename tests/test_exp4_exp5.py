"""
Tests for Experiments 4 and 5: homogeneous-marginal robust RVaR and IRD/IQD.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exp"))

from exp4_homogeneous_marginals import (
    compute_c_n,
    compute_proposition1_rhs,
    compute_proposition2_rhs,
    approximate_robust_sup_rearrangement,
    approximate_robust_inf_rearrangement,
)
from exp5_ird_iqd import (
    avg_quantile_union,
    validate_theorem4_corollary2,
)
from scipy import stats


def test_c_n_boundary():
    # c_n should be in (0, 1/n]
    dist = stats.expon(scale=1)
    n = 2
    cn = compute_c_n(dist, n)
    assert 0 < cn <= 1 / n + 1e-6


def test_c_n_uniform():
    # For Uniform[0,1], verify c_n is computable
    dist = stats.uniform(0, 1)
    cn = compute_c_n(dist, 2)
    assert 0 < cn <= 0.5 + 1e-6


def test_proposition2_rhs_exponential():
    # Prop2 closed form should be close to inf approx
    dist = stats.expon(scale=1)
    n, r, s = 2, 0.1, 0.2
    prop2 = compute_proposition2_rhs(dist, n, r, s)
    inf_approx = approximate_robust_inf_rearrangement(dist, n, r, s, N=1000)
    # Should be reasonably close
    assert abs(prop2 - inf_approx) < 2.0  # loose tolerance


def test_proposition1_rhs_beta():
    dist = stats.beta(2, 0.5)
    n, r, s = 2, 0.1, 0.2
    prop1 = compute_proposition1_rhs(dist, n, r, s)
    sup_approx = approximate_robust_sup_rearrangement(dist, n, r, s, N=1000)
    assert abs(prop1 - sup_approx) < 2.0


def test_theorem4_decomposition_no_violations():
    results = validate_theorem4_corollary2()
    assert (results["decomp_error"].abs() < 0.5).all()  # decomposition approximately holds


def test_avg_quantile_union_two_intervals():
    dist = stats.uniform(0, 1)
    # R_[[0,0.3] union [0.7,1]](Uniform) = mean of [mean(0,0.3)*0.3 + mean(0.7,1)*0.3] / 0.6
    # = (0.15*0.3 + 0.85*0.3) / 0.6 = (0.045 + 0.255) / 0.6 = 0.3/0.6 = 0.5
    val = avg_quantile_union(dist, [(0, 0.3), (0.7, 1.0)])
    assert abs(val - 0.5) < 0.01
