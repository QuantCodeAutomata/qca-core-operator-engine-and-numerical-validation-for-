"""Core quantile operator engine.

Implements left/right quantile functions, the average-quantile (Rangel)
functional R_I(mu), Expected Shortfall, Lower Expected Shortfall, Range
Value at Risk, and tail-transform operators.

All integration is performed via scipy.integrate.quad for smooth
(ScipyDistribution) integrands and via exact closed-form sums for
DiscreteDistribution integrands.  Derived distributions (tail transforms,
sign flip) inherit the quad-based path.
"""

import numpy as np
from scipy.integrate import quad

from src.distributions import (
    Distribution,
    DiscreteDistribution,
    ScipyDistribution,
)


# ---------------------------------------------------------------------------
# Derived distribution classes (defined here to keep operators.py self-contained)
# ---------------------------------------------------------------------------


class UpperTailDistribution(Distribution):
    """Conditional upper-tail distribution mu^{r+}.

    CDF: F_{mu^{r+}}(x) = max((F_mu(x) - r) / (1-r), 0)

    Equivalently, mu^{r+} is the law of X | X > VaR_r(mu) (continuous case).
    """

    def __init__(self, base_dist: Distribution, r: float):
        if not (0.0 <= r < 1.0):
            raise ValueError(f"r must be in [0, 1), got {r}")
        self.base_dist = base_dist
        self.r = r
        self._name = f"UpperTail({base_dist.name}, r={r})"
        self.bounded_above = base_dist.is_bounded_above()
        self.is_continuous = getattr(base_dist, "is_continuous", True)
        self.has_atoms = getattr(base_dist, "has_atoms", False)

    @property
    def name(self) -> str:
        return self._name

    def cdf(self, x: float) -> float:
        return max((self.base_dist.cdf(x) - self.r) / (1.0 - self.r), 0.0)

    def pdf(self, x: float) -> float:
        if self.base_dist.cdf(x) <= self.r:
            return 0.0
        return self.base_dist.pdf(x) / (1.0 - self.r)

    def left_quantile(self, u: float) -> float:
        """q_u^-(mu^{r+}) = q_{r+(1-r)*u}^-(mu)."""
        return self.base_dist.left_quantile(self.r + (1.0 - self.r) * u)

    def right_quantile(self, u: float) -> float:
        """q_u^+(mu^{r+}) = q_{r+(1-r)*u}^+(mu)."""
        return self.base_dist.right_quantile(self.r + (1.0 - self.r) * u)

    def support(self) -> tuple:
        _, hi = self.base_dist.support()
        new_lo = self.base_dist.left_quantile(self.r)
        return (new_lo, hi)

    def is_bounded_above(self) -> bool:
        return self.bounded_above

    def has_finite_mean(self) -> bool:
        return self.base_dist.has_finite_mean()


class LowerTailDistribution(Distribution):
    """Conditional lower-tail distribution mu^{r-}.

    CDF: F_{mu^{r-}}(x) = min(F_mu(x) / r, 1)

    Equivalently, mu^{r-} is the law of X | X <= VaR_r(mu) (continuous case).
    """

    def __init__(self, base_dist: Distribution, r: float):
        if not (0.0 < r <= 1.0):
            raise ValueError(f"r must be in (0, 1], got {r}")
        self.base_dist = base_dist
        self.r = r
        self._name = f"LowerTail({base_dist.name}, r={r})"
        self.bounded_above = True  # upper tail is truncated
        self.is_continuous = getattr(base_dist, "is_continuous", True)
        self.has_atoms = getattr(base_dist, "has_atoms", False)

    @property
    def name(self) -> str:
        return self._name

    def cdf(self, x: float) -> float:
        return min(self.base_dist.cdf(x) / self.r, 1.0)

    def pdf(self, x: float) -> float:
        if self.base_dist.cdf(x) >= self.r:
            return 0.0
        return self.base_dist.pdf(x) / self.r

    def left_quantile(self, u: float) -> float:
        """q_u^-(mu^{r-}) = q_{r*u}^-(mu)."""
        return self.base_dist.left_quantile(self.r * u)

    def right_quantile(self, u: float) -> float:
        """q_u^+(mu^{r-}) = q_{r*u}^+(mu)."""
        return self.base_dist.right_quantile(self.r * u)

    def support(self) -> tuple:
        lo, _ = self.base_dist.support()
        new_hi = self.base_dist.left_quantile(self.r)
        return (lo, new_hi)

    def is_bounded_above(self) -> bool:
        return True

    def has_finite_mean(self) -> bool:
        return self.base_dist.has_finite_mean()


class SignFlipDistribution(Distribution):
    """Law of -X when X ~ mu.

    Quantile relations (general):
        q_u^-(-X) = -q_{1-u}^+(mu)
    For continuous mu:
        q_u^-(-X) = -q_{1-u}^-(mu) = -ppf_{mu}(1-u)
    """

    def __init__(self, base_dist: Distribution):
        self.base_dist = base_dist
        self._name = f"SignFlip({base_dist.name})"
        lo, _ = base_dist.support()
        self.bounded_above = not np.isinf(lo)
        self.is_continuous = getattr(base_dist, "is_continuous", True)
        self.has_atoms = getattr(base_dist, "has_atoms", False)

    @property
    def name(self) -> str:
        return self._name

    def cdf(self, x: float) -> float:
        # F_{-X}(x) = P(-X <= x) = P(X >= -x) = 1 - F_X(-x-)
        # For continuous base: 1 - F_X(-x)
        if self.is_continuous:
            return 1.0 - self.base_dist.cdf(-x)
        # Approximate F(-x-) as F(-x) minus any atom at -x
        return 1.0 - self.base_dist.cdf(-x) + self.base_dist.pdf(-x)

    def pdf(self, x: float) -> float:
        return self.base_dist.pdf(-x)

    def left_quantile(self, u: float) -> float:
        """q_u^-(-X) = -q_{1-u}^+(mu)."""
        return -self.base_dist.right_quantile(1.0 - u)

    def right_quantile(self, u: float) -> float:
        """q_u^+(-X) = -q_{1-u}^-(mu)."""
        return -self.base_dist.left_quantile(1.0 - u)

    def support(self) -> tuple:
        lo, hi = self.base_dist.support()
        return (-hi, -lo)

    def is_bounded_above(self) -> bool:
        return self.bounded_above

    def has_finite_mean(self) -> bool:
        return self.base_dist.has_finite_mean()


# ---------------------------------------------------------------------------
# Helper: discrete exact integration
# ---------------------------------------------------------------------------


def _discrete_avg_quantile(dist: DiscreteDistribution, a: float, b: float) -> float:
    """Exact integral of q^-(u) on [a,b] for a DiscreteDistribution.

    q^-(u) = values[k]  for u in (cumprobs_before[k], cumprobs[k]].
    The integral is sum_k values[k] * overlap([cumprobs_before[k], cumprobs[k]], [a,b]).
    """
    total = 0.0
    for k in range(len(dist.values)):
        lo_k = dist.cumprobs_before[k]
        hi_k = dist.cumprobs[k]
        overlap = max(0.0, min(hi_k, b) - max(lo_k, a))
        total += dist.values[k] * overlap
    return total / (b - a)


# ---------------------------------------------------------------------------
# Core quantile operators
# ---------------------------------------------------------------------------


def left_quantile(dist: Distribution, u: float) -> float:
    """Left (lower) quantile: q_u^-(mu) = inf{x : F(x) >= u}.

    Handles endpoints:
      u=0 -> lower support bound (may be -inf for unbounded distributions)
      u=1 -> upper support bound (may be +inf)
    """
    if u <= 0.0:
        return dist.support()[0]
    if u >= 1.0:
        return dist.support()[1]
    return dist.left_quantile(u)


def right_quantile(dist: Distribution, u: float) -> float:
    """Right (upper) quantile: q_u^+(mu) = sup{x : F(x-) <= u}.

    For continuous distributions this equals left_quantile.
    Handles endpoints:
      u=0 -> lower support bound
      u=1 -> upper support bound
    """
    if u <= 0.0:
        return dist.support()[0]
    if u >= 1.0:
        return dist.support()[1]
    return dist.right_quantile(u)


def average_quantile_interval(
    dist: Distribution, a: float, b: float, n_points: int = 2000
) -> float:
    """Average quantile functional on [a, b]:

        R_{[a,b]}(mu) = (1 / (b-a)) * integral_a^b q_t^-(mu) dt

    Special cases:
      a == b  -> left_quantile(dist, a)   (limiting degenerate interval)
      a >  b  -> raises ValueError

    For ScipyDistribution (continuous): scipy.integrate.quad on ppf.
    For DiscreteDistribution: exact closed-form sum over jump intervals.
    For derived distributions: quad on dist.left_quantile.
    """
    if a > b:
        raise ValueError(f"Require a <= b, got a={a}, b={b}")
    if a == b:
        return left_quantile(dist, a)

    if isinstance(dist, DiscreteDistribution):
        return _discrete_avg_quantile(dist, a, b)

    # Continuous / derived: use quad, nudging endpoints away from 0 and 1
    # to avoid ppf(0) = -inf or ppf(1) = +inf at the integration boundary.
    _a = a + 1e-13 if a == 0.0 else a
    _b = b - 1e-13 if b == 1.0 else b

    val, _ = quad(
        lambda u: dist.left_quantile(u),
        _a,
        _b,
        limit=200,
        epsabs=1e-8,
        epsrel=1e-8,
    )
    return val / (b - a)


def average_quantile_union(
    dist: Distribution, intervals: list, n_points: int = 2000
) -> float:
    """Average quantile functional on a union of intervals I = ∪ [a_i, b_i]:

        R_I(mu) = (1 / |I|) * sum_i integral_{a_i}^{b_i} q_t^-(mu) dt

    where |I| = sum_i (b_i - a_i).

    Zero-length intervals are skipped when computing the weighted sum.
    If ALL intervals have zero length the function returns the arithmetic
    mean of left_quantile at each such point.
    """
    positive = [(a, b) for a, b in intervals if b > a]
    total_length = sum(b - a for a, b in positive)

    if total_length == 0.0:
        # All zero-length: return average of pointwise left quantile.
        points = [a for a, b in intervals if a == b]
        if not points:
            return 0.0
        return sum(left_quantile(dist, p) for p in points) / len(points)

    weighted_sum = sum(
        average_quantile_interval(dist, a, b, n_points) * (b - a)
        for a, b in positive
    )
    return weighted_sum / total_length


def ES(dist: Distribution, p: float) -> float:
    """Expected Shortfall at level p:

        ES_p(mu) = R_{[1-p, 1]}(mu)

    Represents the average of the top-p fraction of the loss distribution.
    """
    return average_quantile_interval(dist, 1.0 - p, 1.0)


def LES(dist: Distribution, p: float) -> float:
    """Lower Expected Shortfall at level p:

        LES_p(mu) = R_{[0, p]}(mu)

    Represents the average of the bottom-p fraction of the distribution.
    """
    return average_quantile_interval(dist, 0.0, p)


def RVaR(dist: Distribution, beta: float, alpha: float) -> float:
    """Range Value at Risk:

        RVaR_{beta, alpha}(mu) = R_{[1-beta-alpha, 1-beta]}(mu)
    """
    return average_quantile_interval(dist, 1.0 - beta - alpha, 1.0 - beta)


def expectation(dist: Distribution, n_quad: int = 5000) -> float:
    """Expected value E[X] = R_{[0,1]}(mu).

    Computed as the full-range average quantile integral.
    """
    return average_quantile_interval(dist, 0.0, 1.0, n_points=n_quad)


def upper_tail_transform(dist: Distribution, r: float) -> UpperTailDistribution:
    """Return mu^{r+}, the upper-tail transform of mu at level r.

    The resulting distribution has CDF:
        F_{mu^{r+}}(x) = max((F_mu(x) - r) / (1-r), 0)

    Quantile mapping: q_u^-(mu^{r+}) = q_{r+(1-r)*u}^-(mu).
    """
    return UpperTailDistribution(dist, r)


def lower_tail_transform(dist: Distribution, r: float) -> LowerTailDistribution:
    """Return mu^{r-}, the lower-tail transform of mu at level r.

    The resulting distribution has CDF:
        F_{mu^{r-}}(x) = min(F_mu(x) / r, 1)

    Quantile mapping: q_u^-(mu^{r-}) = q_{r*u}^-(mu).
    """
    return LowerTailDistribution(dist, r)


def sign_flip(dist: Distribution) -> SignFlipDistribution:
    """Return the law of -X when X ~ mu.

    Quantile relation (general):   q_u^-(-X) = -q_{1-u}^+(mu)
    For continuous distributions:  q_u^-(-X) = -q_{1-u}^-(mu)

    Identity: R_{[a,b]}(-X) = -R_{[1-b, 1-a]}(X)
    """
    return SignFlipDistribution(dist)


# ---------------------------------------------------------------------------
# Scipy-frozen-distribution variants (used in exp6 / exp7)
# ---------------------------------------------------------------------------


def avg_quantile_interval(dist, a: float, b: float) -> float:
    """Average quantile on [a, b] for a scipy frozen distribution.

    R_{[a,b]}(dist) = (1/(b-a)) * integral_a^b ppf(t) dt

    Works with any scipy frozen distribution (or object exposing .ppf).
    Endpoints are nudged away from 0 and 1 to avoid ±inf in the integrand.
    """
    a = max(a, 1e-10)
    b = min(b, 1 - 1e-10)
    if a >= b:
        return float(dist.ppf(a))
    result, _ = quad(lambda t: float(dist.ppf(t)), a, b, limit=200, epsabs=1e-6)
    return result / (b - a)


def avg_quantile_union(dist, intervals: list) -> float:
    """Average quantile on a union of intervals for a scipy frozen distribution.

    R_I(dist) = (1/|I|) * sum_k integral_{a_k}^{b_k} ppf(t) dt
    """
    intervals = [(max(a, 1e-10), min(b, 1 - 1e-10)) for a, b in intervals if b > a]
    total_len = sum(b - a for a, b in intervals)
    if total_len < 1e-15:
        return 0.0
    total_int = sum(
        quad(lambda t: float(dist.ppf(t)), a, b, limit=200, epsabs=1e-6)[0]
        for a, b in intervals
    )
    return total_int / total_len
