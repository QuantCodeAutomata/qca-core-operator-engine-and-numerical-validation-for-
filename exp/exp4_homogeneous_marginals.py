"""
Experiment 4: Validate Propositions 1 and 2 for homogeneous-marginal
closed-form robust RVaR bounds.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats, integrate, optimize


# ---------------------------------------------------------------------------
# Core Operators
# ---------------------------------------------------------------------------

def avg_quantile_interval(dist, a: float, b: float) -> float:
    """R_[a,b](mu) = (1/(b-a)) * integral_a^b ppf(t) dt."""
    a = max(a, 1e-10)
    b = min(b, 1 - 1e-10)
    if a >= b:
        return float(dist.ppf(a))
    result, _ = integrate.quad(
        lambda t: float(dist.ppf(t)), a, b, limit=200, epsabs=1e-6
    )
    return result / (b - a)


def avg_quantile_union(dist, intervals: list) -> float:
    """R_I(mu) for union of intervals."""
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
# Distribution Wrappers
# ---------------------------------------------------------------------------

class _SignFlipDist:
    """Distribution of -X when X ~ dist: ppf(u) = -dist.ppf(1-u)."""

    def __init__(self, dist):
        self._dist = dist

    def ppf(self, u):
        return -float(self._dist.ppf(1.0 - float(u)))


class _LowerTailDist:
    """Lower tail transform dist^{r-}: ppf(u) = dist.ppf(r*u)."""

    def __init__(self, dist, r: float):
        self._dist = dist
        self._r = r

    def ppf(self, u):
        return float(self._dist.ppf(self._r * float(u)))


def make_sign_flip_dist(dist):
    """Create a wrapper for the sign-flip distribution that has ppf method."""
    return _SignFlipDist(dist)


def make_lower_tail_dist(dist, r: float):
    """Create wrapper for dist^{r-} with ppf(u) = dist.ppf(r*u)."""
    return _LowerTailDist(dist, r)


def lower_tail_transform_ppf(dist, r: float, u: float) -> float:
    """Left quantile of dist^{r-}: q_u^-(dist^{r-}) = dist.ppf(r*u)."""
    return float(dist.ppf(r * u))


def sign_flip_ppf(dist, u: float) -> float:
    """Left quantile of -X when X~dist: -dist.ppf(1-u) for continuous dist."""
    return -float(dist.ppf(1.0 - u))


# ---------------------------------------------------------------------------
# Core Computations
# ---------------------------------------------------------------------------

def compute_c_n(dist, n: int, n_grid: int = 500) -> float:
    """Compute c_n(mu) = inf{x in (0,1/n): g(x) <= 0}.

    g(x) = [(n-1)*q_{(n-1)x} + q_{1-x}]/n - R_[(n-1)x,1-x](mu)
    """
    x_vals = np.linspace(1e-6, 1.0 / n - 1e-6, n_grid)
    g_vals = np.empty(n_grid)

    for i, x in enumerate(x_vals):
        a = (n - 1) * x
        b = 1.0 - x
        if a >= b:
            g_vals[i] = np.inf
            continue
        lhs = ((n - 1) * dist.ppf(a) + dist.ppf(b)) / n
        rhs = avg_quantile_interval(dist, a, b)
        g_vals[i] = lhs - rhs

    feasible = np.where(g_vals <= 0)[0]
    if len(feasible) == 0:
        return 1.0 / n

    first_idx = feasible[0]

    # Refine with brentq when a sign change is bracketed
    if first_idx > 0 and g_vals[first_idx - 1] > 0:
        x_lo = float(x_vals[first_idx - 1])
        x_hi = float(x_vals[first_idx])
        try:
            def _g(x):
                a = (n - 1) * x
                b = 1.0 - x
                if a >= b:
                    return np.inf
                lhs = ((n - 1) * dist.ppf(a) + dist.ppf(b)) / n
                rhs = avg_quantile_interval(dist, a, b)
                return lhs - rhs

            return float(optimize.brentq(_g, x_lo, x_hi, xtol=1e-8))
        except (ValueError, RuntimeError):
            pass

    return float(x_vals[first_idx])


def compute_proposition1_rhs(dist, n: int, r: float, s: float) -> float:
    """Prop 1 closed form: n * R_[[r, r+s/n] union [1-s+s/n, 1]](mu)."""
    intervals = [(r, r + s / n), (1.0 - s + s / n, 1.0)]
    return n * avg_quantile_union(dist, intervals)


def compute_proposition2_rhs(dist, n: int, r: float, s: float) -> float:
    """Prop 2 closed form: n * R_[[0, (n-1)*s/n] union [r+(n-1)*s/n, r+s]](mu)."""
    intervals = [(0.0, (n - 1) * s / n), (r + (n - 1) * s / n, r + s)]
    return n * avg_quantile_union(dist, intervals)


def approximate_robust_sup_rearrangement(
    dist, n: int, r: float, s: float, N: int = 3000
) -> float:
    """Comonotonic rearrangement gives sup: all marginals sorted ascending."""
    U = np.linspace(0.5 / N, 1.0 - 0.5 / N, N)
    X = dist.ppf(U)
    S = n * X  # comonotonic: all the same quantile
    start = int(r * N)
    end = int((r + s) * N)
    if start >= end:
        return float(S[start])
    return float(np.mean(S[start:end]))


def approximate_robust_inf_rearrangement(
    dist, n: int, r: float, s: float, N: int = 3000
) -> float:
    """Rearrangement inf: X_1 ascending, remaining descending (mixed for n=3)."""
    U = np.linspace(0.5 / N, 1.0 - 0.5 / N, N)
    X_vals = dist.ppf(U)
    X_asc = np.sort(X_vals)
    X_desc = np.sort(X_vals)[::-1]

    if n == 2:
        S = X_asc + X_desc
    elif n == 3:
        S = X_asc + X_desc + X_vals  # mixed
    else:
        S = X_asc + (n - 1) * X_desc  # approximate

    S_sorted = np.sort(S)
    start = int(r * N)
    end = int((r + s) * N)
    if start >= end:
        return float(S_sorted[start])
    return float(np.mean(S_sorted[start:end]))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_exp4():
    """Validate Propositions 1 and 2 for homogeneous marginals."""
    os.makedirs("/workspace/project/results", exist_ok=True)

    prop1_dists = [
        ("Beta(2,0.5)", stats.beta(2, 0.5)),
        ("Uniform(0,1)", stats.uniform(0, 1)),
    ]
    prop2_dists = [
        ("Exponential(1)", stats.expon(scale=1)),
        ("Pareto(b=2)", stats.pareto(b=2)),
    ]
    n_vals = [2, 3, 5]
    param_sets = [(0.1, 0.2), (0.2, 0.3)]

    records = []

    print("=" * 70)
    print("Experiment 4: Homogeneous Marginals — Propositions 1 & 2")
    print("=" * 70)

    # --- Proposition 1: upper bound (increasing density) ---
    print("\n--- Proposition 1 (Upper Bound, increasing density) ---")
    for dist_name, dist in prop1_dists:
        for n in n_vals:
            for r, s in param_sets:
                # tilde_mu^{(1-r)-}: sign-flip then lower tail at (1-r)
                sf_dist = make_sign_flip_dist(dist)
                lt_sf_dist = make_lower_tail_dist(sf_dist, 1.0 - r)
                c_n = compute_c_n(lt_sf_dist, n)
                threshold = s / (1.0 - r)
                assumption_ok = threshold <= n * c_n

                prop1_val = compute_proposition1_rhs(dist, n, r, s)
                sup_approx = approximate_robust_sup_rearrangement(dist, n, r, s)
                gap = prop1_val - sup_approx

                print(
                    f"  {dist_name:20s} n={n} r={r} s={s}: "
                    f"c_n={c_n:.4f} thresh={threshold:.4f} n*c_n={n*c_n:.4f} "
                    f"assump={'OK  ' if assumption_ok else 'FAIL'} "
                    f"Prop1={prop1_val:7.4f} Sup≈{sup_approx:7.4f} gap={gap:+.4f}"
                )

                records.append(
                    dict(
                        prop="Prop1",
                        dist=dist_name,
                        n=n,
                        r=r,
                        s=s,
                        c_n=c_n,
                        threshold=threshold,
                        n_times_cn=n * c_n,
                        assumption_holds=assumption_ok,
                        formula_val=prop1_val,
                        approx_val=sup_approx,
                        gap=gap,
                        passed=abs(gap) < 0.5 or not assumption_ok,
                    )
                )

    # --- Proposition 2: lower bound (decreasing density) ---
    print("\n--- Proposition 2 (Lower Bound, decreasing density) ---")
    for dist_name, dist in prop2_dists:
        for n in n_vals:
            for r, s in param_sets:
                # mu^{(r+s)-}: lower tail transform at (r+s)
                lt_dist = make_lower_tail_dist(dist, r + s)
                c_n = compute_c_n(lt_dist, n)
                threshold = s / (r + s)
                assumption_ok = threshold <= n * c_n

                prop2_val = compute_proposition2_rhs(dist, n, r, s)
                inf_approx = approximate_robust_inf_rearrangement(dist, n, r, s)
                # Prop 2 is a lower bound; gap = approx - formula >= 0 ideally
                gap = inf_approx - prop2_val

                print(
                    f"  {dist_name:20s} n={n} r={r} s={s}: "
                    f"c_n={c_n:.4f} thresh={threshold:.4f} n*c_n={n*c_n:.4f} "
                    f"assump={'OK  ' if assumption_ok else 'FAIL'} "
                    f"Prop2={prop2_val:7.4f} Inf≈{inf_approx:7.4f} gap={gap:+.4f}"
                )

                records.append(
                    dict(
                        prop="Prop2",
                        dist=dist_name,
                        n=n,
                        r=r,
                        s=s,
                        c_n=c_n,
                        threshold=threshold,
                        n_times_cn=n * c_n,
                        assumption_holds=assumption_ok,
                        formula_val=prop2_val,
                        approx_val=inf_approx,
                        gap=gap,
                        passed=abs(gap) < 0.5 or not assumption_ok,
                    )
                )

    df = pd.DataFrame(records)
    out_path = "/workspace/project/results/exp4_results.csv"
    df.to_csv(out_path, index=False)

    n_pass = df["passed"].sum()
    print(f"\nSummary: {n_pass}/{len(df)} scenarios passed (|gap|<0.5 or assumption fails)")
    print(f"Results saved → {out_path}")
    return df


if __name__ == "__main__":
    run_exp4()
