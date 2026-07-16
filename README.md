# QCA Core Operator Engine and Numerical Validation

A comprehensive Python library implementing and validating foundational mathematical operators for quantitative risk analysis, with focus on:

- Left/right quantile operators
- Average-quantile (Rangel) functionals R_I(μ)
- Robust RVaR aggregation bounds (Theorems 2 & 3)
- Risk sharing with averaged quantiles
- Inter-RVaR and Inter-Quantile Differences

## Repository Structure

```
qca-core/
├── data/                          # Data utilities (synthetic + optional market data)
│   └── load_data.py
├── src/                           # Core implementation
│   ├── distributions.py           # Distribution ABC, catalog (10 distributions)
│   └── operators.py               # Quantile engine: R_I, ES, LES, RVaR, tail transforms
├── exp/                           # Experiment scripts
│   ├── exp1_operator_validation.py   # Core operator identity validation
│   ├── exp2_theorem1_validation.py   # Theorem 1 & Corollary 1 RVaR inequality
│   ├── exp3_robust_bounds.py         # Robust upper/lower RVaR bounds
│   ├── exp4_homogeneous_marginals.py # Propositions 1 & 2 closed forms
│   ├── exp5_ird_iqd.py               # Theorem 4, IRD & IQD
│   ├── exp6_risk_sharing_bounded.py  # Risk sharing (bounded X)
│   └── exp7_risk_sharing_unbounded.py # Risk sharing (unbounded X)
├── tests/                         # Pytest test suite
│   ├── conftest.py
│   ├── test_exp1.py
│   ├── test_exp2_exp3.py
│   ├── test_exp4_exp5.py
│   └── test_exp6_exp7.py
├── results/                       # Generated artifacts
│   ├── RESULTS.md                 # Experiment summaries
│   ├── exp1_results.csv
│   ├── exp2_theorem1_results.csv
│   ├── exp3_results.csv
│   └── ...
├── AGENTS.md                      # Persistent memory for AI agents
└── README.md
```

## Installation

```bash
pip install numpy pandas scipy matplotlib seaborn
```

All required libraries (numpy, scipy, pandas, matplotlib) are standard and pre-installed in the target environment.

## Running Experiments

Run individual experiments:

```bash
cd /path/to/repo
python exp/exp1_operator_validation.py   # Core operator validation
python exp/exp2_theorem1_validation.py   # Theorem 1 & Corollary 1
python exp/exp3_robust_bounds.py         # Robust RVaR bounds
python exp/exp4_homogeneous_marginals.py # Homogeneous propositions
python exp/exp5_ird_iqd.py               # IRD/IQD decomposition
python exp/exp6_risk_sharing_bounded.py  # Risk sharing (bounded)
python exp/exp7_risk_sharing_unbounded.py # Risk sharing (unbounded)
```

Run all tests:

```bash
pytest tests/ -v
```

## Core Mathematical Concepts

### Average-Quantile Functional

For a distribution μ and interval I = [a,b]:

```
R_[a,b](μ) = (1/(b-a)) * ∫_a^b q_t^-(μ) dt
```

Special cases:
- ES_p(μ) = R_[1-p, 1](μ)  (Expected Shortfall)
- LES_p(μ) = R_[0, p](μ)   (Lower Expected Shortfall)
- RVaR_{β,α}(μ) = R_[1-β-α, 1-β](μ)  (Range Value-at-Risk)

### Theorem 1 (RVaR Inequality)

For admissible parameters r, s, α_i, β_i and any coupling of X_1,...,X_n:

```
R_[r,r+s](ν) ≤ Σᵢ [ cᵢ · R_{Uᵢ}(μᵢ) + (1-cᵢ) · R_{[r+αᵢ, r+αᵢ+βᵢ]}(μᵢ) ]
```

where cᵢ = (1-r-βᵢ)/s and Uᵢ = [r, r+αᵢ] ∪ [r+αᵢ+βᵢ, 1].

### Risk Sharing (Bounded Case)

For bounded-above X with β = Σᵢ βᵢ < 1:

```
⊕ᵢ R_{Iᵢ}(Xᵢ) = R_[0,β](X)
```

where I_i = [0, β_i] ∪ [1-β+β_i, 1].

## Distribution Catalog

| Name | Distribution | Bounded | Finite Mean |
|------|-------------|---------|-------------|
| Uniform01 | Uniform[0,1] | Yes | Yes |
| Beta25 | Beta(2, 0.5) | Yes | Yes |
| Beta05 | Beta(0.5, 0.5) | Yes | Yes |
| Normal01 | N(0,1) | No | Yes |
| Exponential1 | Exp(1) | No | Yes |
| Lognormal01 | LogN(0,1) | No | Yes |
| StudentT3 | t(3) | No | Yes |
| Pareto2 | Pareto(b=2) | No | Yes |
| Bernoulli03 | Bernoulli(0.3) | Yes | Yes |
| Discrete3 | Discrete({0,1,2}) | Yes | Yes |

## Results Summary

See [results/RESULTS.md](results/RESULTS.md) for experiment outcomes.
