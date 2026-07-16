"""Data loading module for the QCA Core Operator Engine repository.

This project is primarily theoretical and uses synthetic distributions.
No empirical market data is required for the core experiments.

For completeness, this module provides utilities for loading market data
via the Massive API if needed for future extensions.
"""

import os
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Synthetic distribution generators (primary data source for all experiments)
# ---------------------------------------------------------------------------

def get_synthetic_distributions() -> dict:
    """Return the canonical set of synthetic distributions for experiments.

    Returns:
        Dictionary mapping distribution name to scipy frozen distribution.
    """
    return {
        # Continuous bounded
        'Uniform01': stats.uniform(0, 1),
        'Beta25': stats.beta(2, 0.5),
        'Beta05': stats.beta(0.5, 0.5),
        # Continuous unbounded integrable
        'Normal01': stats.norm(0, 1),
        'Exponential1': stats.expon(scale=1),
        'Lognormal01': stats.lognorm(s=1, scale=1),
        'StudentT3': stats.t(df=3),
        'Pareto2': stats.pareto(b=2),
        'Gamma2': stats.gamma(a=2, scale=1),
    }


def sample_distribution(dist_name: str, n_samples: int = 10000, seed: int = 42) -> np.ndarray:
    """Sample from a named distribution.

    Args:
        dist_name: name in the synthetic distributions catalog
        n_samples: number of samples
        seed: random seed

    Returns:
        Array of samples.
    """
    dists = get_synthetic_distributions()
    if dist_name not in dists:
        raise ValueError(f"Unknown distribution: {dist_name}. "
                         f"Available: {list(dists.keys())}")
    dist = dists[dist_name]
    rng = np.random.default_rng(seed)
    return dist.ppf(rng.uniform(0, 1, n_samples))


def get_quantile_grid(dist_name: str, n_points: int = 5000) -> tuple:
    """Get a dense quantile grid for a distribution.

    Uses logit-transformed spacing for higher density near endpoints.

    Args:
        dist_name: distribution name
        n_points: number of grid points

    Returns:
        Tuple (u_grid, x_grid) of probability levels and quantile values.
    """
    dists = get_synthetic_distributions()
    if dist_name not in dists:
        raise ValueError(f"Unknown distribution: {dist_name}")
    dist = dists[dist_name]
    u_grid = np.linspace(1.0 / (2 * n_points), 1.0 - 1.0 / (2 * n_points), n_points)
    x_grid = dist.ppf(u_grid)
    return u_grid, x_grid


# ---------------------------------------------------------------------------
# Optional: Massive API data loading (for extensions requiring market data)
# ---------------------------------------------------------------------------

def load_market_data(
    ticker: str,
    from_date: str,
    to_date: str,
    timespan: str = 'day',
    multiplier: int = 1
) -> list:
    """Load OHLCV bars from Massive API.

    Requires MASSIVE_TOKEN environment variable.

    Args:
        ticker: asset ticker symbol
        from_date: start date string 'YYYY-MM-DD'
        to_date: end date string 'YYYY-MM-DD'
        timespan: bar timespan ('day', 'minute', etc.)
        multiplier: bar size multiplier

    Returns:
        List of aggregate bar objects, or empty list if API unavailable.
    """
    api_key = os.getenv('MASSIVE_TOKEN')
    if not api_key:
        print("Warning: MASSIVE_TOKEN not set. Returning empty data.")
        return []

    try:
        from massive import RESTClient
        client = RESTClient(api_key=api_key)
        aggs = []
        for a in client.list_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=from_date,
            to=to_date,
            limit=50000
        ):
            aggs.append(a)
        return aggs
    except ImportError:
        print("Warning: massive library not available.")
        return []
    except Exception as e:
        print(f"Warning: Failed to load market data: {e}")
        return []


if __name__ == '__main__':
    print("Synthetic distributions available:")
    for name, dist in get_synthetic_distributions().items():
        u, x = get_quantile_grid(name, n_points=100)
        print(f"  {name}: support=[{x.min():.3f}, {x.max():.3f}], mean~{x.mean():.3f}")
