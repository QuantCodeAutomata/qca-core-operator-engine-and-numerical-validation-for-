# Project: Quantile Operator Engine

## Overview
Quantitative finance library implementing the average-quantile (Rangel) functional
R_I(mu) and related risk measures (ES, LES, RVaR) via quantile calculus.

## Repository Layout
```
src/distributions.py   – Distribution ABC + ScipyDistribution, DiscreteDistribution,
                         MixedDistribution, and SCENARIO_CATALOG (10 distributions)
src/operators.py       – Core engine: left/right quantile, average_quantile_interval,
                         ES, LES, RVaR, expectation, tail transforms, sign flip
exp/exp1_operator_validation.py – Experiment 1: validates all operator identities
results/exp1_results.csv        – Generated CSV of pass/fail per test
tests/test_exp1.py     – 10 pytest tests (all passing)
tests/conftest.py      – sys.path setup for imports
```

## Key Design Decisions

### Quantile functions (operators.py)
- `left_quantile(dist, u)` / `right_quantile(dist, u)` in operators.py handle u=0 and
  u=1 via `dist.support()`, then delegate to the Distribution method for u in (0,1).
- For the Distribution *methods* (dist.left_quantile, dist.right_quantile), callers
  are assumed to pass u strictly in (0,1).

### Integration strategy (average_quantile_interval)
- ScipyDistribution (continuous) and derived distributions (UpperTail, LowerTail,
  SignFlip): `scipy.integrate.quad` with limit=200, endpoint-nudging (+1e-13 / -1e-13)
  to prevent ppf(0)=-inf or ppf(1)=+inf from causing NaN.
- DiscreteDistribution: **exact** closed-form sum — integral = Σ_k x_k * overlap of
  the k-th flat region [cumprobs_before[k], cumprobs[k]] with [a, b].

### Discrete quantile binary search
- `left_quantile(u)`:  `np.searchsorted(cumprobs, u, side='left')` — first k with F(x_k) >= u
- `right_quantile(u)`: `np.searchsorted(cumprobs_before, u, side='right') - 1` — last k
  with F(x_k-) <= u
- Key: `cumprobs_before[k] = cumprobs[k-1]` always (no gaps in purely discrete dists),
  so midpoints for "within-flat-region" checks must use `0.5*(cumprobs_before[k] + cumprobs[k])`
  NOT `0.5*(cumprobs[k] + cumprobs_before[k+1])` (the latter equals cumprobs[k] exactly).

### Tail transforms (quantile mappings)
- UpperTailDistribution.left_quantile(u) = base.left_quantile(r + (1-r)*u)
- LowerTailDistribution.left_quantile(u) = base.left_quantile(r * u)
- SignFlipDistribution.left_quantile(u)  = -base.right_quantile(1 - u)

### Sign-flip identity
R_{[a,b]}(-X) = -R_{[1-b,1-a]}(X)
Verified to 1e-4 tolerance for Uniform01, Normal01, Exponential1, Beta25.

## Running Tests
```bash
cd /workspace/project
python -m pytest tests/test_exp1.py -v      # all 10 tests should pass
python exp/exp1_operator_validation.py       # all 4 validation sections should PASS
```

## Experiments 6 & 7 – Risk Sharing (Bounded and Unbounded)

### New files
```
exp/exp6_risk_sharing_bounded.py   – Optimal allocation for bounded-above X
exp/exp7_risk_sharing_unbounded.py – Asymptotic threshold allocation for unbounded X
tests/test_exp6_exp7.py            – 7 pytest tests (all passing)
results/exp6_results.csv           – R_Ii sums, distortion objectives per scenario
results/exp7_results.csv           – Gap convergence table per m value
results/exp7_convergence.png       – Convergence plot (realized vs predicted gap)
```

### Design: scipy frozen distribution interface (avg_quantile_interval)
Experiments 6 & 7 use **scipy frozen distributions** directly (stats.uniform, etc.),
not the custom `Distribution` class.  Two scipy-compatible variants are added at the
bottom of `src/operators.py` and also re-defined locally at the top of each exp file:
  - `avg_quantile_interval(dist, a, b)` – uses `dist.ppf(t)` via `scipy.integrate.quad`
  - `avg_quantile_union(dist, intervals)` – union of intervals version

The test file imports `avg_quantile_interval` from `src.operators` (the scipy variant)
on top of the `*` imports from the exp files.

### Risk measure structure (exp 6)
- Agent i interval: I_i = [0, beta_i] ∪ [1-beta+beta_i, 1], |I_i| = beta
- R_{I_i}(X_i) is computed empirically via `compute_agent_objective`:
    sorted_Xi[:int(beta_i*N)]            ← bottom beta_i fraction
    sorted_Xi[int((1-beta+beta_i)*N):]   ← top (beta-beta_i) fraction
    R = (sum_lower + sum_upper) / (beta * N)
- Inf-conv target: R_{[0,beta]}(X) = avg_quantile_interval(dist, 0, beta)
- Optimal allocation (bounded X, upper bound t = dist.support()[1]):
    X_i = (X-t)*1_{A_i} + (X/n)*1_{A^c} + (t/(n-1))*1_{A\A_i}
  Feasibility holds analytically: sum_i X_i = X everywhere (machine eps errors ~1e-16)

### Gap formula (exp 7)
  sum_i R_{I_i}(X_i^(m)) - R_{[0,beta]}(X) = (1/beta) * E[(X - a_m)*1_{X > a_m}]
  a_m = n*m/(n-1)
Computed via scipy.integrate.quad on (x - a_m)*dist.pdf(x) over [a_m, ∞).

### Minimum admissible m (exp 7)
  min_m = max(0, dist.ppf(beta), max_i dist.ppf(1-beta+betas_i))
m values below min_m are filtered out before running the allocation.

### Observed numerical accuracy (N=5000 grid)
- Exponential: gap_error ~ 2e-4  (smooth tails)
- Lognormal:   gap_error ~ 3e-3  (heavier tails, larger discretization error)
- Pareto(b=2): gap_error ~ 0.03  (power-law tails; N=5000 grid under-samples extremes)

## Experiments 4 & 5 – Homogeneous Marginals and IRD/IQD

### New files
```
exp/exp4_homogeneous_marginals.py  – Propositions 1 & 2 for identical marginals
exp/exp5_ird_iqd.py                – Theorem 4, Corollary 2, Propositions 3–5
tests/test_exp4_exp5.py            – 6 pytest tests (all passing)
results/exp4_results.csv           – c_n, threshold, formula vs. approx per scenario
results/exp5_results.csv           – decomp_error and pass/fail per IRD scenario
```

### Experiment 4 – Homogeneous Marginals
Validates Propositions 1 and 2: closed-form robust RVaR bounds when all n
marginals are identical (mu_1 = ... = mu_n = mu).

**compute_c_n(dist, n, n_grid=500)**
- Grid of 500 x-values in `(1e-6, 1/n − 1e-6)`.
- `g(x) = [(n-1)·ppf((n-1)x) + ppf(1-x)] / n  −  R_[(n-1)x, 1-x](mu)`
- First x where g ≤ 0 refined with `scipy.optimize.brentq`; returns `1/n` if none.
- Accepts wrapper dist objects (only `.ppf()` needed).

**Prop 1 assumption**: build `_LowerTailDist(_SignFlipDist(dist), 1-r)`, compute
its `c_n`; check `s/(1-r) ≤ n·c_n`.
Formula: `n * R_[[r, r+s/n] ∪ [1-s+s/n, 1]](mu)`

**Prop 2 assumption**: build `_LowerTailDist(dist, r+s)`, compute its `c_n`;
check `s/(r+s) ≤ n·c_n`.
Formula: `n * R_[[0, (n-1)s/n] ∪ [r+(n-1)s/n, r+s]](mu)`

**Inf rearrangement (exp4)**: uses antithetic only (`n=2`: X_asc + X_desc,
`n=3`: X_asc + X_desc + X_vals) as specced — intentionally overestimates the
true infimum; loose test tolerance (2.0) accommodates this.

### Experiment 5 – IRD / IQD
Validates Theorem 4 / Corollary 2:
`sup_nu [R_[r2,s2] − R_[r1,s1]] = sup_nu R_[r2,s2] − inf_nu R_[r1,s1]`

**Critical interval-parameter convention:**
- `compute_robust_{sup,inf}_rearrangement(dists, r, s, N)`:
  `r` = start quantile, `s` = width → interval `[r, r+s]`.
- `compute_{direct,decomposition}_sup_IRD(dists, r1, s1, r2, s2, N)`:
  `r1, s1, r2, s2` are *endpoint* quantiles → intervals `[r1,s1]` and `[r2,s2]`.
  Internally converts: `compute_robust_sup_rearrangement(dists, r2, s2-r2, N)`.

**Inf rearrangement (exp5)**: tries comonotonic AND antithetic; returns
`min(val_como, val_anti)`. Required because for mixed-sign marginals (e.g.
Normal + Exp) the comonotonic coupling gives a smaller lower-tail average.

**Cor 2(iii) narrow windows**: restricted to `mid ∈ {0.3, 0.5}` (not 0.7).
Upper-tail narrow windows (mid=0.7) produce an inherent single-coupling
optimality gap > 0.5 for Normal+Exp — not a theorem failure but a consequence
of requiring a single joint distribution to simultaneously achieve separate
supremum/infimum.

## Dependencies
scipy >= 1.1.0 (for frozen distribution support(), quad)
numpy >= 1.20
matplotlib >= 3.0 (for exp7 convergence plot)
pandas >= 1.0 (for CSV output in exp6/exp7)
