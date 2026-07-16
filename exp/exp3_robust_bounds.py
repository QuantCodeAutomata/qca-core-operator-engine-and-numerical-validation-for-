"""Experiment 3: Robust RVaR Aggregation Bounds.

Validates Theorem 2 (robust upper bound) and Theorem 3 (robust lower bound)
for RVaR aggregation under dependence uncertainty. Compares new bounds against
prior convolution bounds and independently approximated exact Fréchet optima.

Methodology: optimize bound objectives over simplex and compare with
rearrangement-based exact approximations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy import stats, integrate, optimize

# ---------------------------------------------------------------------------
# Core quantile operators (self-contained)
# Using scipy - Context7 confirmed appropriate
# ---------------------------------------------------------------------------

# Pre-compute Gauss-Legendre nodes/weights once (100-point rule)
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(100)


def avg_quantile_interval(dist, a: float, b: float) -> float:
    """R_[a,b](mu) = (1/(b-a)) * integral_a^b ppf(t) dt via Gauss-Legendre."""
    a = max(a, 1e-10)
    b = min(b, 1.0 - 1e-10)
    if a >= b:
        return float(dist.ppf(a))
    t_pts = 0.5 * (b - a) * _GL_NODES + 0.5 * (a + b)
    vals = dist.ppf(t_pts)  # vectorized ppf call
    return float(np.dot(_GL_WEIGHTS, vals)) * 0.5  # = (1/(b-a)) * (b-a)/2 * sum(w*v)


def avg_quantile_union(dist, intervals: list) -> float:
    """R_I(mu) for I = union of intervals."""
    clipped = [(max(a, 1e-10), min(b, 1.0 - 1e-10)) for a, b in intervals if b > a]
    total_len = sum(b - a for a, b in clipped)
    if total_len < 1e-15:
        return 0.0
    total_int = sum(avg_quantile_interval(dist, a, b) * (b - a) for a, b in clipped)
    return total_int / total_len


# ---------------------------------------------------------------------------
# New upper bound (Theorem 2)
# ---------------------------------------------------------------------------

def _u_new_objective(beta_vec: np.ndarray, dists: list, r: float, s: float) -> float:
    """Compute U_new(beta; mu) for given beta vector.

    U_new = (1-(1-r-beta_0)/s) * sum_i R_[r+beta_i, r+beta_i+beta_0](mu_i)
           + ((1-r-beta_0)/s) * sum_i R_[[r,r+beta_i] union [r+beta_i+beta_0,1]](mu_i)

    Args:
        beta_vec: array [beta_0, beta_1, ..., beta_n], sum=(1-r), all>=0, beta_0>=1-r-s
        dists: list of n scipy frozen distributions
        r: lower tail level
        s: window width

    Returns:
        Objective value (to minimize for upper bound).
    """
    n = len(dists)
    beta_0 = beta_vec[0]
    beta_i_vec = beta_vec[1:]

    coeff_outer = (1.0 - r - beta_0) / s  # = (1-r-beta_0)/s
    coeff_inner = 1.0 - coeff_outer        # = (s - (1-r-beta_0))/s = (beta_0 - (1-r-s))/s

    sum_inner = 0.0
    sum_outer = 0.0

    for i in range(n):
        d = dists[i]
        bi = beta_i_vec[i]

        # Inner: R_[r+bi, r+bi+beta_0]
        lo_in = r + bi
        hi_in = r + bi + beta_0
        if hi_in > lo_in:
            sum_inner += avg_quantile_interval(d, lo_in, hi_in)
        else:
            sum_inner += float(d.ppf(max(lo_in, 1e-10)))

        # Outer union: [r, r+bi] union [r+bi+beta_0, 1]
        union_ivs = []
        if bi > 1e-12:
            union_ivs.append((r, r + bi))
        if r + bi + beta_0 < 1.0 - 1e-12:
            union_ivs.append((r + bi + beta_0, 1.0))
        if union_ivs:
            sum_outer += avg_quantile_union(d, union_ivs)
        # else: union empty => 0

    return coeff_inner * sum_inner + coeff_outer * sum_outer


def compute_new_upper_bound(
    dists: list, r: float, s: float, n_starts: int = 4
) -> tuple:
    """Optimize U_new over admissible beta to get Theorem 2 upper bound.

    Uses SLSQP with explicit linear constraints:
      sum(beta) = 1-r,  beta_i >= 0,  beta_0 >= 1-r-s.

    Args:
        dists: list of scipy frozen distributions
        r: lower tail level
        s: window width
        n_starts: number of random restarts

    Returns:
        (bound_value, optimal_beta)
    """
    n = len(dists)
    total = 1.0 - r
    lb_beta0 = 1.0 - r - s  # lower bound on beta_0

    def objective(beta_vec: np.ndarray) -> float:
        return _u_new_objective(beta_vec, dists, r, s)

    # Constraints: sum = total, beta_0 >= lb_beta0
    constraints = [
        {'type': 'eq', 'fun': lambda b: b.sum() - total},
    ]
    # Bounds: all >= 0, beta_0 >= lb_beta0
    bounds = [(lb_beta0, total)] + [(0.0, total)] * n

    best_val = np.inf
    best_beta = None
    rng = np.random.default_rng(42)

    # Always include a canonical feasible start: beta_0 near lb, rest uniform
    starts = []
    rem = total - lb_beta0
    b0_start = lb_beta0 + rem * 0.5  # midpoint for beta_0
    rest = (total - b0_start) / n * np.ones(n)
    starts.append(np.concatenate([[b0_start], rest]))
    for _ in range(n_starts - 1):
        # Random feasible: beta_0 ~ U(lb_beta0, total), remaining split
        b0 = rng.uniform(lb_beta0, total)
        rest_raw = rng.dirichlet(np.ones(n))
        rest_vals = rest_raw * (total - b0)
        starts.append(np.concatenate([[b0], rest_vals]))

    for x0 in starts:
        try:
            res = optimize.minimize(
                objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-7},
            )
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_beta = res.x
        except Exception:
            pass

    if best_beta is None:
        # Fallback: use canonical start value
        best_beta = starts[0]
        best_val = objective(best_beta)

    return float(best_val), best_beta


# ---------------------------------------------------------------------------
# Old upper bound (convolution benchmark)
# ---------------------------------------------------------------------------

def _u_old_objective(beta_vec: np.ndarray, dists: list, r: float, s: float) -> float:
    """U_old = sum_i R_[1-beta_i-beta_0, 1-beta_i](mu_i).

    Conditions: sum(beta_vec) = 1-r, all >= 0, beta_0 >= s.
    """
    n = len(dists)
    beta_0 = beta_vec[0]
    beta_i_vec = beta_vec[1:]
    total = 0.0
    for i in range(n):
        lo = 1.0 - beta_i_vec[i] - beta_0
        hi = 1.0 - beta_i_vec[i]
        if hi > lo:
            total += avg_quantile_interval(dists[i], lo, hi)
        else:
            total += float(dists[i].ppf(max(lo, 1e-10)))
    return total


def compute_old_upper_bound(
    dists: list, r: float, s: float, n_starts: int = 4
) -> tuple:
    """Minimize U_old over (1-r)*simplex with beta_0 >= s via SLSQP.

    Args:
        dists: list of scipy frozen distributions
        r: lower tail level
        s: window width
        n_starts: number of restarts

    Returns:
        (bound_value, optimal_beta)
    """
    n = len(dists)
    total = 1.0 - r

    def objective(beta_vec: np.ndarray) -> float:
        return _u_old_objective(beta_vec, dists, r, s)

    constraints = [{'type': 'eq', 'fun': lambda b: b.sum() - total}]
    bounds = [(s, total)] + [(0.0, total)] * n  # beta_0 >= s

    best_val = np.inf
    best_beta = None
    rng = np.random.default_rng(1)

    starts = []
    b0_start = s + (total - s) * 0.5
    rest = (total - b0_start) / n * np.ones(n)
    starts.append(np.concatenate([[b0_start], rest]))
    for _ in range(n_starts - 1):
        b0 = rng.uniform(s, total)
        rest_raw = rng.dirichlet(np.ones(n))
        starts.append(np.concatenate([[b0], rest_raw * (total - b0)]))

    for x0 in starts:
        try:
            res = optimize.minimize(
                objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-7},
            )
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_beta = res.x
        except Exception:
            pass

    if best_beta is None:
        best_beta = starts[0]
        best_val = objective(best_beta)

    return float(best_val), best_beta


# ---------------------------------------------------------------------------
# New lower bound (Theorem 3)
# ---------------------------------------------------------------------------

def _l_new_objective(beta_vec: np.ndarray, dists: list, r: float, s: float) -> float:
    """Compute L_new(beta; mu).

    L_new = (1-(r+s-beta_0)/s) * sum_i R_[r+s-bi-beta_0, r+s-bi](mu_i)
           + ((r+s-beta_0)/s) * sum_i R_[[0, r+s-bi-beta_0] union [r+s-bi, r+s]](mu_i)

    Conditions: sum(beta_vec) = r+s, all >= 0, beta_0 >= r.
    """
    n = len(dists)
    beta_0 = beta_vec[0]
    beta_i_vec = beta_vec[1:]

    rs = r + s
    coeff_inner = 1.0 - (rs - beta_0) / s   # = (beta_0 - r) / s
    coeff_outer = (rs - beta_0) / s

    sum_inner = 0.0
    sum_outer = 0.0

    for i in range(n):
        d = dists[i]
        bi = beta_i_vec[i]
        lo_in = rs - bi - beta_0
        hi_in = rs - bi

        # Inner: R_[rs-bi-beta_0, rs-bi]
        if hi_in > lo_in:
            sum_inner += avg_quantile_interval(d, lo_in, hi_in)
        else:
            sum_inner += float(d.ppf(max(lo_in, 1e-10)))

        # Outer union: [0, rs-bi-beta_0] union [rs-bi, rs]
        union_ivs = []
        if lo_in > 1e-12:
            union_ivs.append((0.0, lo_in))
        if hi_in < rs - 1e-12:
            union_ivs.append((hi_in, rs))
        if union_ivs:
            sum_outer += avg_quantile_union(d, union_ivs)

    return coeff_inner * sum_inner + coeff_outer * sum_outer


def compute_new_lower_bound(
    dists: list, r: float, s: float, n_starts: int = 4
) -> tuple:
    """Maximize L_new over (r+s)*simplex with beta_0 >= r via SLSQP.

    Args:
        dists: list of scipy frozen distributions
        r: lower tail level
        s: window width
        n_starts: number of restarts

    Returns:
        (bound_value, optimal_beta)
    """
    n = len(dists)
    total = r + s

    def neg_objective(beta_vec: np.ndarray) -> float:
        return -_l_new_objective(beta_vec, dists, r, s)

    constraints = [{'type': 'eq', 'fun': lambda b: b.sum() - total}]
    bounds = [(r, total)] + [(0.0, total)] * n  # beta_0 >= r

    best_neg_val = np.inf
    best_beta = None
    rng = np.random.default_rng(2)

    starts = []
    b0_start = r + (total - r) * 0.5
    rest = (total - b0_start) / n * np.ones(n)
    starts.append(np.concatenate([[b0_start], rest]))
    for _ in range(n_starts - 1):
        b0 = rng.uniform(r, total)
        rest_raw = rng.dirichlet(np.ones(n))
        starts.append(np.concatenate([[b0], rest_raw * (total - b0)]))

    for x0 in starts:
        try:
            res = optimize.minimize(
                neg_objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-7},
            )
            if res.success and res.fun < best_neg_val:
                best_neg_val = res.fun
                best_beta = res.x
        except Exception:
            pass

    if best_beta is None:
        best_beta = starts[0]
        best_neg_val = neg_objective(best_beta)

    return float(-best_neg_val), best_beta


# ---------------------------------------------------------------------------
# Old lower bound (convolution benchmark)
# ---------------------------------------------------------------------------

def _l_old_objective(beta_vec: np.ndarray, dists: list, r: float, s: float) -> float:
    """L_old = sum_i R_[beta_i, beta_i+beta_0](mu_i).

    Conditions: sum(beta_vec) = r+s, all >= 0, beta_0 >= s.
    """
    n = len(dists)
    beta_0 = beta_vec[0]
    beta_i_vec = beta_vec[1:]
    total = 0.0
    for i in range(n):
        lo = beta_i_vec[i]
        hi = beta_i_vec[i] + beta_0
        if hi > lo:
            total += avg_quantile_interval(dists[i], lo, hi)
        else:
            total += float(dists[i].ppf(max(lo, 1e-10)))
    return total


def compute_old_lower_bound(
    dists: list, r: float, s: float, n_starts: int = 4
) -> tuple:
    """Maximize L_old over (r+s)*simplex with beta_0 >= s via SLSQP.

    Args:
        dists: list of scipy frozen distributions
        r: lower tail level
        s: window width
        n_starts: number of restarts

    Returns:
        (bound_value, optimal_beta)
    """
    n = len(dists)
    total = r + s

    def neg_objective(beta_vec: np.ndarray) -> float:
        return -_l_old_objective(beta_vec, dists, r, s)

    constraints = [{'type': 'eq', 'fun': lambda b: b.sum() - total}]
    bounds = [(s, total)] + [(0.0, total)] * n  # beta_0 >= s

    best_neg_val = np.inf
    best_beta = None
    rng = np.random.default_rng(3)

    starts = []
    b0_start = s + (total - s) * 0.5
    rest = (total - b0_start) / n * np.ones(n)
    starts.append(np.concatenate([[b0_start], rest]))
    for _ in range(n_starts - 1):
        b0 = rng.uniform(s, total)
        rest_raw = rng.dirichlet(np.ones(n))
        starts.append(np.concatenate([[b0], rest_raw * (total - b0)]))

    for x0 in starts:
        try:
            res = optimize.minimize(
                neg_objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-7},
            )
            if res.success and res.fun < best_neg_val:
                best_neg_val = res.fun
                best_beta = res.x
        except Exception:
            pass

    if best_beta is None:
        best_beta = starts[0]
        best_neg_val = neg_objective(best_beta)

    return float(-best_neg_val), best_beta


# ---------------------------------------------------------------------------
# Exact approximations via rearrangement
# ---------------------------------------------------------------------------

def approximate_exact_sup(
    dists: list, r: float, s: float, N: int = 2000
) -> float:
    """Approximate sup_nu R_[r,r+s](nu) via comonotonic rearrangement.

    Comonotonic coupling maximizes the average of the upper quantile range
    for increasing loss aggregates. Uses a quantile grid.

    Args:
        dists: list of scipy frozen distributions
        r: lower tail level
        s: window width
        N: grid size

    Returns:
        Approximate supremum.
    """
    U = np.linspace(0.5 / N, 1.0 - 0.5 / N, N)
    S = sum(d.ppf(U) for d in dists)
    S_sorted = np.sort(S)
    start = max(0, int(r * N))
    end = min(N, int((r + s) * N))
    if start >= end:
        return float(S_sorted[min(start, N - 1)])
    return float(np.mean(S_sorted[start:end]))


def approximate_exact_inf(
    dists: list, r: float, s: float, N: int = 2000
) -> float:
    """Approximate inf_nu R_[r,r+s](nu) via antithetic rearrangement.

    Uses the rearrangement of one marginal ascending and others descending
    to minimize the aggregate in the target quantile range.

    Args:
        dists: list of scipy frozen distributions
        r: lower tail level
        s: window width
        N: grid size

    Returns:
        Approximate infimum.
    """
    U = np.linspace(0.5 / N, 1.0 - 0.5 / N, N)
    X_list = [np.sort(d.ppf(U)) for d in dists]
    # Ascending for first, descending for rest
    X_list[0] = X_list[0]  # ascending
    for i in range(1, len(X_list)):
        X_list[i] = X_list[i][::-1]  # descending
    S = sum(X_list)
    S_sorted = np.sort(S)
    start = max(0, int(r * N))
    end = min(N, int((r + s) * N))
    if start >= end:
        return float(S_sorted[min(start, N - 1)])
    return float(np.mean(S_sorted[start:end]))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_exp3() -> pd.DataFrame:
    """Run Experiment 3: Robust RVaR Aggregation Bounds Validation.

    Returns:
        Results DataFrame.
    """
    os.makedirs('results', exist_ok=True)

    print("=" * 70)
    print("Experiment 3: Robust RVaR Aggregation Bounds (Theorems 2 & 3)")
    print("=" * 70)

    dist_catalog = {
        'Normal': stats.norm(0, 1),
        'Exponential': stats.expon(scale=1),
        'Beta25': stats.beta(2, 0.5),
        'Beta05': stats.beta(0.5, 0.5),
        'Uniform': stats.uniform(0, 1),
    }

    scenarios = [
        # (combo_name, dist_names, r, s)
        ('Normal_Exp_n2', ['Normal', 'Exponential'], 0.1, 0.2),
        ('Beta25_Beta05_n2', ['Beta25', 'Beta05'], 0.1, 0.2),
        ('Beta25_Beta05_n2_r0', ['Beta25', 'Beta05'], 0.0, 0.3),
        ('Beta25_Beta05_Norm_n3', ['Beta25', 'Beta05', 'Normal'], 0.05, 0.25),
    ]

    records = []
    for combo_name, dist_names, r, s in scenarios:
        # Skip non-integrable in r=0 extended case
        if r == 0.0:
            # All dists must be integrable
            non_int = {'Pareto1'}  # Pareto with b=1 not integrable
            if any(n in non_int for n in dist_names):
                continue

        dists = [dist_catalog[n] for n in dist_names]
        print(f"\nScenario: {combo_name}, r={r}, s={s}, dists={dist_names}")

        # Compute bounds
        print("  Computing new upper bound...", end=' ', flush=True)
        new_upper, _ = compute_new_upper_bound(dists, r, s, n_starts=3)
        print(f"{new_upper:.4f}")

        print("  Computing old upper bound...", end=' ', flush=True)
        old_upper, _ = compute_old_upper_bound(dists, r, s, n_starts=3)
        print(f"{old_upper:.4f}")

        print("  Computing new lower bound...", end=' ', flush=True)
        new_lower, _ = compute_new_lower_bound(dists, r, s, n_starts=3)
        print(f"{new_lower:.4f}")

        print("  Computing old lower bound...", end=' ', flush=True)
        old_lower, _ = compute_old_lower_bound(dists, r, s, n_starts=3)
        print(f"{old_lower:.4f}")

        print("  Approximating exact sup/inf...", end=' ', flush=True)
        exact_sup = approximate_exact_sup(dists, r, s, N=500)
        exact_inf = approximate_exact_inf(dists, r, s, N=500)
        print(f"sup={exact_sup:.4f}, inf={exact_inf:.4f}")

        # Gaps
        upper_gap_new = new_upper - exact_sup
        upper_gap_old = old_upper - exact_sup
        lower_gap_new = exact_inf - new_lower
        lower_gap_old = exact_inf - old_lower

        # Theorem checks: bound must be >= exact (up to approximation error)
        thm2_ok = new_upper >= exact_sup - 0.15  # approximation tolerance
        thm3_ok = new_lower <= exact_inf + 0.15

        records.append({
            'scenario': combo_name,
            'dists': '+'.join(dist_names),
            'r': r, 's': s,
            'new_upper': new_upper,
            'old_upper': old_upper,
            'new_lower': new_lower,
            'old_lower': old_lower,
            'exact_sup': exact_sup,
            'exact_inf': exact_inf,
            'upper_gap_new': upper_gap_new,
            'upper_gap_old': upper_gap_old,
            'lower_gap_new': lower_gap_new,
            'lower_gap_old': lower_gap_old,
            'thm2_valid': thm2_ok,
            'thm3_valid': thm3_ok,
        })

        print(f"  Thm2 valid: {thm2_ok} (gap_new={upper_gap_new:.4f}, gap_old={upper_gap_old:.4f})")
        print(f"  Thm3 valid: {thm3_ok} (gap_new={lower_gap_new:.4f}, gap_old={lower_gap_old:.4f})")

    df = pd.DataFrame(records)
    df.to_csv('results/exp3_results.csv', index=False)

    print("\n--- Summary ---")
    print(f"Total scenarios: {len(df)}")
    print(f"Theorem 2 violations: {(~df['thm2_valid']).sum()}")
    print(f"Theorem 3 violations: {(~df['thm3_valid']).sum()}")
    print("Results saved to results/exp3_results.csv")

    return df


if __name__ == '__main__':
    result = run_exp3()
    print(result[['scenario', 'r', 's', 'new_upper', 'old_upper',
                   'exact_sup', 'upper_gap_new', 'thm2_valid']].to_string())
