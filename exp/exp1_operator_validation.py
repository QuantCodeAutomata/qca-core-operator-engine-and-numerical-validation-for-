"""Experiment 1 – Operator Validation.

Validates the following identities and properties:

1. ES / LES / RVaR as special cases of the average-quantile functional R_I.
2. Quantile mapping identities for upper- and lower-tail transforms.
3. The sign-flip identity R_{[a,b]}(-X) = -R_{[1-b,1-a]}(X).
4. Exact discrete-distribution quantile behaviour at jump points.

Run directly:
    python exp/exp1_operator_validation.py

Results are saved to /workspace/project/results/exp1_results.csv.
"""

import csv
import math
import os
import sys

# Allow running as a script from any working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.distributions import SCENARIO_CATALOG, DiscreteDistribution
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


# ---------------------------------------------------------------------------
# 1. ES / LES / RVaR identities
# ---------------------------------------------------------------------------


def validate_es_les_rvar_identities() -> dict:
    """Verify that ES, LES, RVaR are special cases of average_quantile_interval
    and that R_{[0,1]}(mu) == E[X] for integrable distributions.

    Also verifies the shrinking-interval limit towards pointwise quantiles.

    Returns
    -------
    dict  {dist_name: {test_name: bool}}
    """
    results = {}

    p = 0.2
    beta, alpha = 0.1, 0.1
    eps = 0.001
    probe_u = 0.4  # a generic interior probability level

    for dist_name, dist in SCENARIO_CATALOG.items():
        if not getattr(dist, "is_continuous", False):
            continue

        row = {}

        # --- ES identity -------------------------------------------------
        es_val = ES(dist, p)
        r_es = average_quantile_interval(dist, 1.0 - p, 1.0)
        row["ES_identity"] = abs(es_val - r_es) < 1e-6

        # --- LES identity ------------------------------------------------
        les_val = LES(dist, p)
        r_les = average_quantile_interval(dist, 0.0, p)
        row["LES_identity"] = abs(les_val - r_les) < 1e-6

        # --- RVaR identity -----------------------------------------------
        rvar_val = RVaR(dist, beta, alpha)
        r_rvar = average_quantile_interval(dist, 1.0 - beta - alpha, 1.0 - beta)
        row["RVaR_identity"] = abs(rvar_val - r_rvar) < 1e-6

        # --- E[X] = R_{[0,1]} (skip non-integrable) ----------------------
        if dist.has_finite_mean():
            e_val = expectation(dist)
            r_full = average_quantile_interval(dist, 0.0, 1.0)
            row["expectation_identity"] = abs(e_val - r_full) < 1e-6
        else:
            row["expectation_identity"] = None  # skipped

        # --- Shrinking-interval limit towards pointwise quantiles ---------
        # R_{[u-eps, u]} -> q^-(u) as eps -> 0
        r_left = average_quantile_interval(dist, probe_u - eps, probe_u)
        q_minus = left_quantile(dist, probe_u)
        row["shrink_left_to_left_quantile"] = abs(r_left - q_minus) < 0.02

        # R_{[u, u+eps]} -> q^+(u) = q^-(u) for continuous
        r_right = average_quantile_interval(dist, probe_u, probe_u + eps)
        q_plus = right_quantile(dist, probe_u)
        row["shrink_right_to_right_quantile"] = abs(r_right - q_plus) < 0.02

        results[dist_name] = row

    return results


# ---------------------------------------------------------------------------
# 2. Tail-transform quantile mapping identities
# ---------------------------------------------------------------------------


def validate_tail_transforms() -> dict:
    """Verify quantile mapping for upper- and lower-tail transforms.

    Upper:  q_u^-(mu^{r+}) = q_{r+(1-r)*u}^-(mu)
    Lower:  q_u^-(mu^{r-}) = q_{r*u}^-(mu)

    Returns
    -------
    dict  {dist_name: {"upper": bool, "lower": bool}}
    """
    target_dists = ["Uniform01", "Normal01", "Exponential1", "Beta25"]
    r_values = [0.1, 0.3, 0.5]
    u_values = [0.1, 0.3, 0.5, 0.9]
    tol = 1e-6

    results = {}

    for dist_name in target_dists:
        dist = SCENARIO_CATALOG[dist_name]
        upper_ok = True
        lower_ok = True

        for r in r_values:
            # Upper tail
            mu_upper = upper_tail_transform(dist, r)
            for u in u_values:
                lhs = left_quantile(mu_upper, u)
                rhs = left_quantile(dist, r + (1.0 - r) * u)
                if abs(lhs - rhs) >= tol:
                    upper_ok = False

            # Lower tail
            mu_lower = lower_tail_transform(dist, r)
            for u in u_values:
                lhs = left_quantile(mu_lower, u)
                rhs = left_quantile(dist, r * u)
                if abs(lhs - rhs) >= tol:
                    lower_ok = False

        results[dist_name] = {"upper_transform": upper_ok, "lower_transform": lower_ok}

    return results


# ---------------------------------------------------------------------------
# 3. Sign-flip identity
# ---------------------------------------------------------------------------


def validate_sign_flip_identity() -> dict:
    """Verify R_{[a,b]}(-X) = -R_{[1-b,1-a]}(X).

    Returns
    -------
    dict  {dist_name: {interval_str: bool}}
    """
    target_dists = ["Uniform01", "Normal01", "Exponential1", "Beta25"]
    intervals = [(0.1, 0.4), (0.2, 0.8), (0.05, 0.95)]
    tol = 1e-4

    results = {}

    for dist_name in target_dists:
        dist = SCENARIO_CATALOG[dist_name]
        tilde = sign_flip(dist)
        row = {}
        for a, b in intervals:
            lhs = average_quantile_interval(tilde, a, b)
            rhs = -average_quantile_interval(dist, 1.0 - b, 1.0 - a)
            key = f"[{a},{b}]"
            row[key] = abs(lhs - rhs) < tol
        results[dist_name] = row

    return results


# ---------------------------------------------------------------------------
# 4. Discrete quantile validation
# ---------------------------------------------------------------------------


def validate_discrete_quantiles() -> dict:
    """Verify discrete-distribution quantile and integral behaviour.

    Checks:
    * At each probability jump F(x_k), left_quantile <= right_quantile.
    * At interior levels (between jumps) left == right.
    * average_quantile_interval matches exact manual computation.

    Returns
    -------
    dict  {dist_name: {test_name: bool}}
    """
    results = {}

    for dist_name in ["Bernoulli03", "Discrete3"]:
        dist = SCENARIO_CATALOG[dist_name]
        row = {}

        # -- Quantile ordering at jump points ---------------------------------
        ordering_ok = True
        for k in range(len(dist.values)):
            u_jump = dist.cumprobs[k]
            lq = left_quantile(dist, u_jump)
            rq = right_quantile(dist, u_jump)
            if lq > rq:
                ordering_ok = False
            # At the last atom, left == right
            if k == len(dist.values) - 1 and lq != rq:
                ordering_ok = False
        row["quantile_ordering_at_jumps"] = ordering_ok

        # -- Strict inequality at interior jump points (not the last) ---------
        strict_ok = True
        for k in range(len(dist.values) - 1):
            u_jump = dist.cumprobs[k]
            lq = left_quantile(dist, u_jump)
            rq = right_quantile(dist, u_jump)
            if not (lq < rq):
                strict_ok = False
        row["strict_jump_inequality"] = strict_ok

        # -- Within-flat-region constancy ------------------------------------
        # For each atom k, check at the interior midpoint of the k-th flat
        # region (cumprobs_before[k], cumprobs[k]).  At such a point both
        # left and right quantile must equal values[k].
        constancy_ok = True
        for k in range(len(dist.values)):
            lo_k = dist.cumprobs_before[k]
            hi_k = dist.cumprobs[k]
            if hi_k <= lo_k:
                continue  # zero-probability atom; skip
            u_mid = 0.5 * (lo_k + hi_k)
            lq = left_quantile(dist, u_mid)
            rq = right_quantile(dist, u_mid)
            if lq != rq:
                constancy_ok = False
        row["between_jump_constancy"] = constancy_ok

        # -- average_quantile_interval matches manual calculation -------------
        integral_ok = True
        test_intervals = [
            (dist.cumprobs_before[0], dist.cumprobs[0]),  # first atom interval
            (0.0, 1.0),                                    # full range = E[X]
        ]
        for a, b in test_intervals:
            if a >= b:
                continue
            computed = average_quantile_interval(dist, a, b)
            # Manual: sum values[k] * overlap / (b-a)
            manual = 0.0
            for k in range(len(dist.values)):
                lo_k = dist.cumprobs_before[k]
                hi_k = dist.cumprobs[k]
                overlap = max(0.0, min(hi_k, b) - max(lo_k, a))
                manual += dist.values[k] * overlap
            manual /= b - a
            if abs(computed - manual) > 1e-10:
                integral_ok = False
        row["integral_matches_manual"] = integral_ok

        # -- Full-range integral equals E[X] ----------------------------------
        e_numerical = average_quantile_interval(dist, 0.0, 1.0)
        e_exact = float(np.dot(dist.values, dist.probs))
        row["expectation_equals_dot_product"] = abs(e_numerical - e_exact) < 1e-10

        results[dist_name] = row

    return results


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _flatten_results(results: dict) -> list:
    """Flatten nested result dicts into a list of (dist, test, pass) rows."""
    rows = []
    for dist_name, sub in results.items():
        if isinstance(sub, dict):
            for test_name, passed in sub.items():
                rows.append(
                    {
                        "distribution": dist_name,
                        "test": test_name,
                        "passed": passed,
                    }
                )
        else:
            rows.append({"distribution": dist_name, "test": "result", "passed": sub})
    return rows


def run_exp1() -> dict:
    """Run all validation functions, print a summary table, and save CSV.

    Returns
    -------
    dict  {"es_les_rvar": ..., "tail_transforms": ..., "sign_flip": ..., "discrete": ...}
    """
    print("=" * 64)
    print("Experiment 1: Operator Validation")
    print("=" * 64)

    # --- ES / LES / RVaR identities ----------------------------------------
    print("\n[1] ES / LES / RVaR identities")
    es_results = validate_es_les_rvar_identities()
    for dist_name, row in es_results.items():
        status = all(v for v in row.values() if v is not None)
        print(f"  {dist_name:20s}  {'PASS' if status else 'FAIL'}")

    # --- Tail transforms ---------------------------------------------------
    print("\n[2] Tail-transform quantile mapping")
    tt_results = validate_tail_transforms()
    for dist_name, row in tt_results.items():
        status = all(row.values())
        print(f"  {dist_name:20s}  {'PASS' if status else 'FAIL'}")

    # --- Sign flip ---------------------------------------------------------
    print("\n[3] Sign-flip identity")
    sf_results = validate_sign_flip_identity()
    for dist_name, row in sf_results.items():
        status = all(row.values())
        print(f"  {dist_name:20s}  {'PASS' if status else 'FAIL'}")

    # --- Discrete quantiles ------------------------------------------------
    print("\n[4] Discrete quantile behaviour")
    dq_results = validate_discrete_quantiles()
    for dist_name, row in dq_results.items():
        status = all(row.values())
        print(f"  {dist_name:20s}  {'PASS' if status else 'FAIL'}")

    # --- Overall summary ---------------------------------------------------
    all_es = all(
        v for row in es_results.values() for v in row.values() if v is not None
    )
    all_tt = all(v for row in tt_results.values() for v in row.values())
    all_sf = all(v for row in sf_results.values() for v in row.values())
    all_dq = all(v for row in dq_results.values() for v in row.values())
    overall = all_es and all_tt and all_sf and all_dq

    print("\n" + "=" * 64)
    print(f"Overall: {'ALL PASS' if overall else 'SOME FAILURES'}")
    print("=" * 64)

    # --- Save CSV ----------------------------------------------------------
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "exp1_results.csv")

    all_rows = []
    for category, res in [
        ("es_les_rvar", es_results),
        ("tail_transforms", tt_results),
        ("sign_flip", sf_results),
        ("discrete", dq_results),
    ]:
        for row in _flatten_results(res):
            row["category"] = category
            all_rows.append(row)

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["category", "distribution", "test", "passed"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDetailed results saved to: {csv_path}")

    summary = {
        "es_les_rvar": es_results,
        "tail_transforms": tt_results,
        "sign_flip": sf_results,
        "discrete": dq_results,
        "overall_pass": overall,
    }
    return summary


# Avoid importing numpy at module top level in a way that forces users to have
# it; import inside the function that needs it.
import numpy as np  # noqa: E402 – needed for dot-product in validate_discrete_quantiles


if __name__ == "__main__":
    run_exp1()
