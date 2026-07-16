"""Experiment 2: Validation of Theorem 1 and Corollary 1 for the New RVaR Inequality.

Tests the paper's pointwise RVaR inequality for fixed marginal distributions and
explicitly constructed couplings. Validates both Theorem 1 (general form) and
Corollary 1 (simplified form when sum_i alpha_i = s).

Methodology reference: Section on Theorem 1 and Corollary 1.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy import stats, integrate

# ---------------------------------------------------------------------------
# Core quantile operators (self-contained, no src dependency required)
# Using scipy.integrate.quad - Context7 confirmed scipy is appropriate here
# ---------------------------------------------------------------------------

def avg_quantile_interval(dist, a: float, b: float) -> float:
    """Compute R_[a,b](mu) = (1/(b-a)) * integral_a^b ppf(t) dt.

    Args:
        dist: scipy frozen distribution with .ppf method
        a: lower bound of integration interval (probability level)
        b: upper bound of integration interval (probability level)

    Returns:
        Average quantile over [a, b].
    """
    a = max(a, 1e-10)
    b = min(b, 1.0 - 1e-10)
    if a >= b:
        return float(dist.ppf(a))
    result, _ = integrate.quad(
        lambda t: float(dist.ppf(t)), a, b,
        limit=200, epsabs=1e-7, epsrel=1e-7
    )
    return result / (b - a)


def avg_quantile_union(dist, intervals: list) -> float:
    """Compute R_I(mu) for I = union of disjoint intervals [a_i, b_i].

    Args:
        dist: scipy frozen distribution with .ppf method
        intervals: list of (a, b) tuples defining the interval union

    Returns:
        Weighted average quantile over the union of intervals.
    """
    clipped = [(max(a, 1e-10), min(b, 1.0 - 1e-10)) for a, b in intervals if b > a]
    total_len = sum(b - a for a, b in clipped)
    if total_len < 1e-15:
        return 0.0
    total_int = sum(
        integrate.quad(
            lambda t: float(dist.ppf(t)), a, b,
            limit=200, epsabs=1e-7, epsrel=1e-7
        )[0]
        for a, b in clipped
    )
    return total_int / total_len


# ---------------------------------------------------------------------------
# Distribution catalog
# ---------------------------------------------------------------------------

def build_distributions() -> dict:
    """Build catalog of scipy frozen distributions for testing.

    Returns:
        Dictionary mapping name to scipy frozen distribution.
    """
    return {
        'Uniform': stats.uniform(0, 1),
        'Beta25': stats.beta(2, 0.5),
        'Beta05': stats.beta(0.5, 0.5),
        'Normal': stats.norm(0, 1),
        'Exponential': stats.expon(scale=1),
        'Lognormal': stats.lognorm(s=1, scale=1),
        'StudentT3': stats.t(df=3),
    }


# ---------------------------------------------------------------------------
# Coupling constructors
# ---------------------------------------------------------------------------

def generate_comonotonic_coupling(dists: list, N: int = 5000) -> np.ndarray:
    """Generate comonotonic coupling via common uniform draw.

    X_i = ppf_i(U) for common U ~ Uniform on grid.

    Args:
        dists: list of scipy frozen distributions (marginals)
        N: grid size

    Returns:
        Array of shape (N, n) with coupled samples.
    """
    U = np.linspace(0.5 / N, 1.0 - 0.5 / N, N)
    return np.column_stack([d.ppf(U) for d in dists])


def generate_independent_coupling(dists: list, N: int = 5000, seed: int = 42) -> np.ndarray:
    """Generate independent coupling with independent uniform draws.

    Args:
        dists: list of scipy frozen distributions
        N: number of samples
        seed: random seed for reproducibility

    Returns:
        Array of shape (N, n).
    """
    rng = np.random.default_rng(seed)
    cols = []
    for d in dists:
        U_i = rng.uniform(0, 1, N)
        cols.append(d.ppf(U_i))
    return np.column_stack(cols)


def generate_antithetic_coupling(dists: list, N: int = 5000) -> np.ndarray:
    """Generate antithetic (approximately negatively dependent) coupling.

    For n=2: X_1 = ppf_1(U), X_2 = ppf_2(1-U).
    For n>2: alternate ascending/descending.

    Args:
        dists: list of scipy frozen distributions
        N: grid size

    Returns:
        Array of shape (N, n).
    """
    n = len(dists)
    U = np.linspace(0.5 / N, 1.0 - 0.5 / N, N)
    cols = []
    for i, d in enumerate(dists):
        if i % 2 == 0:
            cols.append(d.ppf(U))
        else:
            cols.append(d.ppf(1.0 - U))
    return np.column_stack(cols)


def generate_random_coupling(dists: list, N: int = 5000, seed: int = 123) -> np.ndarray:
    """Generate randomized coupling via random permutations of quantile grid.

    Args:
        dists: list of scipy frozen distributions
        N: grid size
        seed: random seed

    Returns:
        Array of shape (N, n).
    """
    rng = np.random.default_rng(seed)
    U_base = np.linspace(0.5 / N, 1.0 - 0.5 / N, N)
    cols = []
    for i, d in enumerate(dists):
        perm = rng.permutation(N)
        cols.append(d.ppf(U_base[perm]))
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Empirical RVaR estimator
# ---------------------------------------------------------------------------

def empirical_avg_quantile(samples: np.ndarray, a: float, b: float) -> float:
    """Estimate R_[a,b](nu) from empirical samples of S = sum_i X_i.

    Uses the mean of samples in the [a, b] quantile range.

    Args:
        samples: 1D array of aggregate S values
        a: lower probability level
        b: upper probability level

    Returns:
        Empirical average quantile estimate.
    """
    N = len(samples)
    sorted_S = np.sort(samples)
    lo = max(0, int(np.floor(a * N)))
    hi = min(N, int(np.ceil(b * N)))
    if lo >= hi:
        idx = min(lo, N - 1)
        return float(sorted_S[idx])
    return float(np.mean(sorted_S[lo:hi]))


# ---------------------------------------------------------------------------
# Theorem 1 right-hand side
# ---------------------------------------------------------------------------

def compute_theorem1_rhs(
    dists: list,
    r: float,
    s: float,
    alphas: list,
    betas: list
) -> float:
    """Compute RHS of Theorem 1.

    RHS = sum_i [ coeff_i * R_{union_i}(mu_i) + (1 - coeff_i) * R_{inner_i}(mu_i) ]

    where coeff_i = (1-r-betas[i])/s,
          union_i = [r, r+alphas[i]] union [r+alphas[i]+betas[i], 1],
          inner_i = [r+alphas[i], r+alphas[i]+betas[i]].

    Edge case: if alphas[i]==0 and r+alphas[i]+betas[i]==1, the union term
    contributes 0 when coeff_i is 0.

    Args:
        dists: list of scipy frozen distributions (marginals mu_i)
        r: lower tail level parameter
        s: window width parameter
        alphas: list of alpha_i parameters (alpha_i >= 0)
        betas: list of beta_i parameters (beta_i > 0)

    Returns:
        Theorem 1 upper bound value.
    """
    total = 0.0
    n = len(dists)
    for i in range(n):
        d = dists[i]
        ai, bi = alphas[i], betas[i]
        coeff_i = (1.0 - r - bi) / s

        # Inner interval: [r+ai, r+ai+bi]
        inner_lo = r + ai
        inner_hi = r + ai + bi
        if inner_hi > inner_lo:
            R_inner = avg_quantile_interval(d, inner_lo, inner_hi)
        else:
            R_inner = float(d.ppf(max(inner_lo, 1e-10)))

        # Union interval: [r, r+ai] union [r+ai+bi, 1]
        # Edge case: if ai==0 and r+ai+bi==1, union is empty and coeff_i should be 0
        edge_case = (ai == 0.0 and abs(r + ai + bi - 1.0) < 1e-12)
        if edge_case and abs(coeff_i) < 1e-12:
            R_union = 0.0
        else:
            union_intervals = []
            if ai > 1e-12:
                union_intervals.append((r, r + ai))
            if r + ai + bi < 1.0 - 1e-12:
                union_intervals.append((r + ai + bi, 1.0))
            if union_intervals:
                R_union = avg_quantile_union(d, union_intervals)
            else:
                R_union = 0.0

        total += coeff_i * R_union + (1.0 - coeff_i) * R_inner
    return total


# ---------------------------------------------------------------------------
# Corollary 1 right-hand side
# ---------------------------------------------------------------------------

def compute_corollary1_rhs(dists: list, r: float, s: float, alphas: list) -> float:
    """Compute RHS of Corollary 1 (requires sum(alphas) == s).

    RHS = sum_i R_[[r, r+alphas[i]] union [1-s+alphas[i], 1]](mu_i)

    Args:
        dists: list of scipy frozen distributions
        r: lower tail level
        s: window width
        alphas: list of alpha_i with sum(alphas) == s

    Returns:
        Corollary 1 upper bound value.
    """
    assert abs(sum(alphas) - s) < 1e-8, f"sum(alphas)={sum(alphas)} must equal s={s}"
    total = 0.0
    for i, (d, ai) in enumerate(zip(dists, alphas)):
        intervals = []
        if ai > 1e-12:
            intervals.append((r, r + ai))
        hi_lo = 1.0 - s + ai
        hi_hi = 1.0
        if hi_hi > hi_lo + 1e-12:
            intervals.append((hi_lo, hi_hi))
        if intervals:
            total += avg_quantile_union(d, intervals)
        else:
            total += float(d.ppf(max(r + ai, 1e-10)))
    return total


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------

def _build_theorem1_param_sets(n: int, s: float = 0.2) -> list:
    """Build admissible (r, s, alphas, betas) configurations for Theorem 1.

    Conditions: 0 < r < r+s <= 1, alpha_i >= 0, beta_i > 0,
    sum_i alpha_i + max_i beta_i <= 1-r, sum_i alpha_i <= s.

    Returns:
        List of (r, s, alphas, betas) tuples.
    """
    configs = []
    for r, s in [(0.1, 0.2), (0.05, 0.3), (0.2, 0.3), (0.3, 0.2)]:
        if r + s > 1.0:
            continue
        # Equal split: alpha_i = s / (2*n), beta_i = (1-r-s) / n
        ai = s / (2.0 * n)
        bi = (1.0 - r - n * ai) / n
        bi = max(bi, 0.02)
        # Check admissibility: sum(ai) + max(bi) <= 1-r and sum(ai) <= s
        if n * ai + bi <= 1.0 - r and n * ai <= s and bi > 0:
            configs.append((r, s, [ai] * n, [bi] * n))

        # Asymmetric: first agent larger
        ai2 = [s / (2.0 * n)] * n
        ai2[0] = s / 2.0 - (n - 1) * s / (4.0 * n)
        bi2 = (1.0 - r - sum(ai2)) / (n + 1)
        bi2 = max(bi2, 0.02)
        if sum(ai2) + bi2 <= 1.0 - r and sum(ai2) <= s and bi2 > 0 and all(a >= 0 for a in ai2):
            configs.append((r, s, ai2, [bi2] * n))

    return configs


def validate_theorem1(N: int = 3000) -> pd.DataFrame:
    """Validate Theorem 1 across distributions, couplings, and parameter sets.

    Args:
        N: grid size for empirical quantile estimation

    Returns:
        DataFrame with columns: dist_combo, coupling, r, s, LHS, RHS, slack, passed
    """
    DISTS = build_distributions()
    dist_combos = {
        'Normal_Exp': ['Normal', 'Exponential'],
        'Beta25_Beta05': ['Beta25', 'Beta05'],
        'Beta25_Beta05_Normal': ['Beta25', 'Beta05', 'Normal'],
        'Uniform_StudentT3': ['Uniform', 'StudentT3'],
    }
    coupling_funcs = {
        'comonotonic': generate_comonotonic_coupling,
        'independent': generate_independent_coupling,
        'antithetic': generate_antithetic_coupling,
        'random': generate_random_coupling,
    }

    records = []
    for combo_name, dist_names in dist_combos.items():
        n = len(dist_names)
        dists_i = [DISTS[name] for name in dist_names]
        param_sets = _build_theorem1_param_sets(n)

        for (r, s, alphas, betas) in param_sets:
            rhs = compute_theorem1_rhs(dists_i, r, s, alphas, betas)

            for coup_name, coup_fn in coupling_funcs.items():
                samples = coup_fn(dists_i, N)
                S = samples.sum(axis=1)
                lhs = empirical_avg_quantile(S, r, r + s)
                slack = rhs - lhs
                records.append({
                    'dist_combo': combo_name,
                    'coupling': coup_name,
                    'r': r, 's': s,
                    'alphas': str(alphas),
                    'betas': str(betas),
                    'LHS': lhs, 'RHS': rhs,
                    'slack': slack,
                    'passed': slack >= -1e-3
                })

    return pd.DataFrame(records)


def validate_corollary1(N: int = 3000) -> pd.DataFrame:
    """Validate Corollary 1 (sum_i alpha_i = s) across distributions and couplings.

    Args:
        N: grid size

    Returns:
        DataFrame with columns: dist_combo, coupling, r, s, LHS, RHS, slack, passed
    """
    DISTS = build_distributions()
    dist_combos = {
        'Normal_Exp': ['Normal', 'Exponential'],
        'Beta25_Beta05': ['Beta25', 'Beta05'],
        'Uniform_Normal': ['Uniform', 'Normal'],
    }
    coupling_funcs = {
        'comonotonic': generate_comonotonic_coupling,
        'independent': generate_independent_coupling,
        'antithetic': generate_antithetic_coupling,
    }

    records = []
    for combo_name, dist_names in dist_combos.items():
        n = len(dist_names)
        dists_i = [DISTS[name] for name in dist_names]

        for r, s in [(0.1, 0.2), (0.05, 0.3), (0.2, 0.3)]:
            if r + s > 1.0:
                continue
            # alphas must sum to s: equal split
            alphas = [s / n] * n
            rhs = compute_corollary1_rhs(dists_i, r, s, alphas)

            for coup_name, coup_fn in coupling_funcs.items():
                samples = coup_fn(dists_i, N)
                S = samples.sum(axis=1)
                lhs = empirical_avg_quantile(S, r, r + s)
                slack = rhs - lhs
                records.append({
                    'dist_combo': combo_name,
                    'coupling': coup_name,
                    'r': r, 's': s,
                    'alphas': str(alphas),
                    'LHS': lhs, 'RHS': rhs,
                    'slack': slack,
                    'passed': slack >= -1e-3
                })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_exp2() -> dict:
    """Run Experiment 2: Theorem 1 and Corollary 1 validation.

    Returns:
        Summary dict with violation counts and DataFrames.
    """
    os.makedirs('results', exist_ok=True)

    print("=" * 70)
    print("Experiment 2: Theorem 1 & Corollary 1 Validation")
    print("=" * 70)

    print("\n--- Theorem 1 Validation ---")
    thm1_results = validate_theorem1(N=3000)
    n_violations = (thm1_results['slack'] < -1e-3).sum()
    print(f"Total cases: {len(thm1_results)}")
    print(f"Violations (slack < -1e-3): {n_violations}")
    print(f"Min slack: {thm1_results['slack'].min():.6f}")
    print(f"Mean slack: {thm1_results['slack'].mean():.6f}")

    print("\n--- Corollary 1 Validation ---")
    cor1_results = validate_corollary1(N=3000)
    n_cor_violations = (cor1_results['slack'] < -1e-3).sum()
    print(f"Total cases: {len(cor1_results)}")
    print(f"Violations (slack < -1e-3): {n_cor_violations}")
    print(f"Min slack: {cor1_results['slack'].min():.6f}")

    # Save results
    thm1_results.to_csv('results/exp2_theorem1_results.csv', index=False)
    cor1_results.to_csv('results/exp2_corollary1_results.csv', index=False)
    print("\nResults saved to results/exp2_theorem1_results.csv and results/exp2_corollary1_results.csv")

    summary = {
        'theorem1_cases': len(thm1_results),
        'theorem1_violations': int(n_violations),
        'theorem1_min_slack': float(thm1_results['slack'].min()),
        'corollary1_cases': len(cor1_results),
        'corollary1_violations': int(n_cor_violations),
        'corollary1_min_slack': float(cor1_results['slack'].min()),
    }
    return summary


if __name__ == '__main__':
    result = run_exp2()
    print("\nSummary:", result)
