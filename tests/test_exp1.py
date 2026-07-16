"""Pytest tests for Experiment 1: quantile operator validation."""

import pytest

from src.distributions import SCENARIO_CATALOG
from src.operators import (
    ES,
    LES,
    RVaR,
    average_quantile_interval,
    average_quantile_union,
    expectation,
    left_quantile,
    lower_tail_transform,
    right_quantile,
    sign_flip,
    upper_tail_transform,
)


def test_left_quantile_uniform():
    # q_u^-(Uniform[0,1]) = u
    dist = SCENARIO_CATALOG['Uniform01']
    for u in [0.1, 0.5, 0.9]:
        assert abs(left_quantile(dist, u) - u) < 1e-6


def test_average_quantile_uniform_full():
    # R_[0,1](Uniform[0,1]) = E[X] = 0.5
    dist = SCENARIO_CATALOG['Uniform01']
    val = average_quantile_interval(dist, 0, 1)
    assert abs(val - 0.5) < 1e-4


def test_es_special_case():
    # ES_0.5(Uniform[0,1]) = R_[0.5,1](Uniform) = 0.75
    dist = SCENARIO_CATALOG['Uniform01']
    assert abs(ES(dist, 0.5) - 0.75) < 1e-4


def test_tail_transform_upper_quantile_map():
    # q_u^-(mu^{r+}) = q_{r+(1-r)u}^-(mu)
    dist = SCENARIO_CATALOG['Exponential1']
    r = 0.3
    tail = upper_tail_transform(dist, r)
    for u in [0.2, 0.5, 0.8]:
        lhs = left_quantile(tail, u)
        rhs = left_quantile(dist, r + (1-r)*u)
        assert abs(lhs - rhs) < 1e-5


def test_sign_flip_identity():
    # R_[a,b](-X) = -R_[1-b,1-a](X) for Normal
    dist = SCENARIO_CATALOG['Normal01']
    flip = sign_flip(dist)
    a, b = 0.2, 0.6
    lhs = average_quantile_interval(flip, a, b)
    rhs = -average_quantile_interval(dist, 1-b, 1-a)
    assert abs(lhs - rhs) < 1e-4


def test_rvar_es_les_identities():
    dist = SCENARIO_CATALOG['Uniform01']
    p = 0.2
    assert abs(ES(dist, p) - average_quantile_interval(dist, 1-p, 1)) < 1e-4
    assert abs(LES(dist, p) - average_quantile_interval(dist, 0, p)) < 1e-4


def test_discrete_quantile_jumps():
    dist = SCENARIO_CATALOG['Bernoulli03']
    # P(X=0)=0.7, P(X=1)=0.3
    # left_quantile(0.5) = 0 (since F(0)=0.7 >= 0.5)
    assert left_quantile(dist, 0.5) == 0
    # right_quantile at u=0.7 should be 0 (F(0-)=0 <= 0.7)
    # right_quantile at u=0.71 should be 1
    assert right_quantile(dist, 0.69) == 0
    assert right_quantile(dist, 0.71) == 1


def test_lower_tail_transform():
    dist = SCENARIO_CATALOG['Uniform01']
    r = 0.4
    tail = lower_tail_transform(dist, r)
    for u in [0.1, 0.5, 0.9]:
        lhs = left_quantile(tail, u)
        rhs = left_quantile(dist, r*u)
        assert abs(lhs - rhs) < 1e-5


def test_expectation_normal():
    dist = SCENARIO_CATALOG['Normal01']
    # E[N(0,1)] = 0
    e = expectation(dist)
    assert abs(e) < 1e-3


def test_zero_length_interval_union():
    dist = SCENARIO_CATALOG['Uniform01']
    # Single zero-length interval should return left_quantile at that point
    result = average_quantile_union(dist, [(0.5, 0.5)])
    # Should return left_quantile at 0.5 = 0.5 for Uniform
    assert abs(result - 0.5) < 1e-4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
