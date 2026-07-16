"""Distribution catalog for quantitative finance experiments.

Provides abstract base class Distribution and concrete subclasses:
ScipyDistribution, DiscreteDistribution, MixedDistribution.
Exports SCENARIO_CATALOG dict of named distribution instances.
"""

from abc import ABC, abstractmethod

import numpy as np
import scipy.stats as stats


class Distribution(ABC):
    """Abstract base class for probability distributions used in quantile analysis."""

    @abstractmethod
    def cdf(self, x: float) -> float:
        """Cumulative distribution function F(x) = P(X <= x)."""

    @abstractmethod
    def pdf(self, x: float) -> float:
        """Probability density (or mass) function at x."""

    @abstractmethod
    def left_quantile(self, u: float) -> float:
        """Left (lower) quantile: q^-(u) = inf{x : F(x) >= u}."""

    @abstractmethod
    def right_quantile(self, u: float) -> float:
        """Right (upper) quantile: q^+(u) = sup{x : F(x-) <= u}."""

    @abstractmethod
    def support(self) -> tuple:
        """Return (lower, upper) bounds of the support (may be ±inf)."""

    @abstractmethod
    def is_bounded_above(self) -> bool:
        """True if the distribution has a finite upper support bound."""

    @abstractmethod
    def has_finite_mean(self) -> bool:
        """True if E[|X|] < infinity."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this distribution."""


class ScipyDistribution(Distribution):
    """Wraps a frozen scipy.stats distribution.

    For continuous distributions left_quantile == right_quantile == ppf.
    The is_continuous flag should be True for all continuously-distributed
    random variables (no point masses).
    """

    def __init__(
        self,
        dist,
        name: str,
        bounded_above: bool = False,
        finite_mean: bool = True,
        is_continuous: bool = True,
        has_atoms: bool = False,
    ):
        self._dist = dist
        self._name = name
        self.bounded_above = bounded_above
        self._finite_mean = finite_mean
        self.is_continuous = is_continuous
        self.has_atoms = has_atoms

    @property
    def name(self) -> str:
        return self._name

    def cdf(self, x: float) -> float:
        return float(self._dist.cdf(x))

    def pdf(self, x: float) -> float:
        return float(self._dist.pdf(x))

    def left_quantile(self, u: float) -> float:
        return float(self._dist.ppf(u))

    def right_quantile(self, u: float) -> float:
        # Continuous distributions: left == right quantile a.e.
        if self.is_continuous:
            return float(self._dist.ppf(u))
        # Discrete scipy distributions: search backwards from ppf(u) for
        # the supremum of {x : F(x-) <= u}.
        x = float(self._dist.ppf(u))
        lo = float(self._dist.ppf(0))
        step = 1e-9
        while x > lo and float(self._dist.cdf(x - step)) > u:
            x -= step
        return x

    def support(self) -> tuple:
        lo = float(self._dist.ppf(0))
        hi = float(self._dist.ppf(1))
        return (lo, hi)

    def is_bounded_above(self) -> bool:
        return self.bounded_above

    def has_finite_mean(self) -> bool:
        return self._finite_mean


class DiscreteDistribution(Distribution):
    """Finite discrete distribution defined by (value, probability) pairs.

    Implements exact left and right quantiles using cumulative sums and
    binary search.  The probability vector is normalised to sum to 1.
    """

    def __init__(self, pairs: list, name: str = "Discrete"):
        sorted_pairs = sorted(pairs, key=lambda vp: vp[0])
        self.values = np.array([v for v, _ in sorted_pairs], dtype=float)
        raw_probs = np.array([p for _, p in sorted_pairs], dtype=float)
        self.probs = raw_probs / raw_probs.sum()
        # cumprobs[i] = F(values[i])
        self.cumprobs = np.cumsum(self.probs)
        # cumprobs_before[i] = F(values[i]-) = P(X < values[i])
        self.cumprobs_before = np.concatenate([[0.0], self.cumprobs[:-1]])
        self._name = name
        self.bounded_above = True
        self.is_continuous = False
        self.has_atoms = True
        self._finite_mean = True

    @property
    def name(self) -> str:
        return self._name

    def cdf(self, x: float) -> float:
        idx = int(np.searchsorted(self.values, x, side="right")) - 1
        if idx < 0:
            return 0.0
        return float(self.cumprobs[idx])

    def pdf(self, x: float) -> float:
        idx = int(np.searchsorted(self.values, x, side="left"))
        if idx < len(self.values) and self.values[idx] == x:
            return float(self.probs[idx])
        return 0.0

    def left_quantile(self, u: float) -> float:
        """inf{x : F(x) >= u} — first atom whose cumulative prob >= u."""
        k = int(np.searchsorted(self.cumprobs, u, side="left"))
        k = min(k, len(self.values) - 1)
        return float(self.values[k])

    def right_quantile(self, u: float) -> float:
        """sup{x : F(x-) <= u} — last atom whose left-limit CDF <= u."""
        k = int(np.searchsorted(self.cumprobs_before, u, side="right")) - 1
        k = max(0, min(k, len(self.values) - 1))
        return float(self.values[k])

    def support(self) -> tuple:
        return (float(self.values[0]), float(self.values[-1]))

    def is_bounded_above(self) -> bool:
        return True

    def has_finite_mean(self) -> bool:
        return True


class MixedDistribution(Distribution):
    """Distribution with both a continuous component and point masses.

    Parameters
    ----------
    continuous_dist : ScipyDistribution
        The continuous component (will be weighted by ``cont_weight``).
    atoms : list of (value, prob)
        Point masses (probabilities are absolute, not conditional).
    name : str
    """

    def __init__(self, continuous_dist: ScipyDistribution, atoms: list, name: str = "Mixed"):
        self._cont = continuous_dist
        self._cont_weight = 1.0 - sum(p for _, p in atoms)
        sorted_atoms = sorted(atoms, key=lambda vp: vp[0])
        self._atom_vals = np.array([v for v, _ in sorted_atoms], dtype=float)
        self._atom_probs = np.array([p for _, p in sorted_atoms], dtype=float)
        self._name = name
        self.bounded_above = continuous_dist.bounded_above and True
        self.is_continuous = False
        self.has_atoms = True
        self._finite_mean = continuous_dist.has_finite_mean()

    @property
    def name(self) -> str:
        return self._name

    def cdf(self, x: float) -> float:
        cont_part = self._cont_weight * self._cont.cdf(x)
        atom_part = float(np.sum(self._atom_probs[self._atom_vals <= x]))
        return cont_part + atom_part

    def pdf(self, x: float) -> float:
        return self._cont_weight * self._cont.pdf(x)

    def left_quantile(self, u: float) -> float:
        # Bisect: find inf{x : F(x) >= u}
        lo_sup, hi_sup = self.support()
        lo = lo_sup if not np.isinf(lo_sup) else -1e15
        hi = hi_sup if not np.isinf(hi_sup) else 1e15
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if self.cdf(mid) >= u:
                hi = mid
            else:
                lo = mid
        return hi

    def right_quantile(self, u: float) -> float:
        lo_sup, hi_sup = self.support()
        lo = lo_sup if not np.isinf(lo_sup) else -1e15
        hi = hi_sup if not np.isinf(hi_sup) else 1e15
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            # F(mid-) <= u ?
            f_left = self.cdf(mid) - self.pdf(mid)
            if f_left <= u:
                lo = mid
            else:
                hi = mid
        return lo

    def support(self) -> tuple:
        cont_lo, cont_hi = self._cont.support()
        lo = min(cont_lo, float(self._atom_vals[0])) if len(self._atom_vals) else cont_lo
        hi = max(cont_hi, float(self._atom_vals[-1])) if len(self._atom_vals) else cont_hi
        return (lo, hi)

    def is_bounded_above(self) -> bool:
        return self.bounded_above

    def has_finite_mean(self) -> bool:
        return self._finite_mean


# ---------------------------------------------------------------------------
# Scenario catalog
# ---------------------------------------------------------------------------

SCENARIO_CATALOG: dict = {
    "Uniform01": ScipyDistribution(
        stats.uniform(0, 1),
        name="Uniform01",
        bounded_above=True,
        finite_mean=True,
        is_continuous=True,
        has_atoms=False,
    ),
    "Beta25": ScipyDistribution(
        stats.beta(2, 0.5),
        name="Beta25",
        bounded_above=True,
        finite_mean=True,
        is_continuous=True,
        has_atoms=False,
    ),
    "Beta05": ScipyDistribution(
        stats.beta(0.5, 0.5),
        name="Beta05",
        bounded_above=True,
        finite_mean=True,
        is_continuous=True,
        has_atoms=False,
    ),
    "Normal01": ScipyDistribution(
        stats.norm(0, 1),
        name="Normal01",
        bounded_above=False,
        finite_mean=True,
        is_continuous=True,
        has_atoms=False,
    ),
    "Exponential1": ScipyDistribution(
        stats.expon(scale=1),
        name="Exponential1",
        bounded_above=False,
        finite_mean=True,
        is_continuous=True,
        has_atoms=False,
    ),
    "Lognormal01": ScipyDistribution(
        stats.lognorm(s=1, scale=1),
        name="Lognormal01",
        bounded_above=False,
        finite_mean=True,
        is_continuous=True,
        has_atoms=False,
    ),
    "StudentT3": ScipyDistribution(
        stats.t(df=3),
        name="StudentT3",
        bounded_above=False,
        finite_mean=True,
        is_continuous=True,
        has_atoms=False,
    ),
    "Pareto2": ScipyDistribution(
        stats.pareto(b=2),
        name="Pareto2",
        bounded_above=False,
        finite_mean=True,
        is_continuous=True,
        has_atoms=False,
    ),
    "Bernoulli03": DiscreteDistribution(
        [(0, 0.7), (1, 0.3)],
        name="Bernoulli03",
    ),
    "Discrete3": DiscreteDistribution(
        [(0, 0.2), (1, 0.5), (2, 0.3)],
        name="Discrete3",
    ),
}
