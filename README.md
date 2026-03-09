# insurance-validation
[![Tests](https://github.com/burning-cost/insurance-validation/actions/workflows/tests.yml/badge.svg)](https://github.com/burning-cost/insurance-validation/actions/workflows/tests.yml)

Your model documentation is a Word doc. Your regulator wants evidence.

PRA SS1/23, FCA Consumer Duty, and TR24/2 require that pricing models have documented, independent validation with traceable audit trails. In practice, most UK insurers have a collection of ad-hoc scripts, Excel workbooks, and email threads that they hope will pass scrutiny. This library turns that into a structured, reproducible HTML report with a JSON sidecar for audit trail ingestion.

**Blog post:** [Your Model Validation Report Won't Survive a PRA Review](https://burning-cost.github.io/2026/03/09/insurance-validation/) — what PRA SS1/23 Principle 4 requires in practice, why exposure-weighted Gini matters, and how to structure your validation evidence pack.

## What it does

- Runs data quality checks (missing values, outliers, cardinality)
- Computes actuarial performance metrics: Gini coefficient, lift charts, actual vs. expected, calibration
- Tests for proxy discrimination (FCA Consumer Duty): correlation with protected characteristics, disparate impact ratio
- Measures population stability (PSI) between training and deployment populations
- Generates a self-contained HTML report with per-section regulatory references
- Writes a JSON sidecar for model risk system ingestion

## What it is not

A compliance guarantee. The report demonstrates what was tested and what the results were. Whether that is sufficient depends on your model's materiality tier, your firm's model risk policy, and what your supervisor considers adequate. A Tier 1 model at a large composite insurer needs more than this library alone.

## Installation

```bash
uv add insurance-validation
# or
pip install insurance-validation
```

Requires Python 3.10+.

## Quick start

```python
from datetime import date
import polars as pl
import numpy as np
from insurance_validation import (
    ModelCard,
    DataQualityReport,
    PerformanceReport,
    DiscriminationReport,
    StabilityReport,
    ReportGenerator,
)

# 1. Describe the model
card = ModelCard(
    model_name="Motor TPPD Frequency",
    version="2.1.0",
    purpose="Estimate expected claim frequency for private motor policies.",
    intended_use="Underwriting pricing. Not for reserving or claims triage.",
    developer="Pricing Team",
    development_date=date(2024, 6, 1),
    limitations=(
        "Performance degrades for vehicles older than 15 years. "
        "Validated on 2018-2023 data; pre-2018 cohorts are out of scope."
    ),
    materiality_tier=2,
    approved_by=["Jane Smith - Chief Actuary", "Model Risk Committee"],
    variables=["driver_age", "vehicle_age", "annual_mileage", "region"],
    target_variable="claim_count",
    model_type="GLM",
    distribution_family="Poisson",
    validator_name="Independent Validation Team",
    validation_date=date(2024, 9, 1),
)

# 2. Run data quality checks
training_df = pl.read_parquet("training_data.parquet")
dq = DataQualityReport(training_df, dataset_name="Motor training 2018-2023")
dq_results = [
    dq.summary_statistics(),
    *dq.missing_value_analysis(threshold=0.05),
    *dq.outlier_detection(method="iqr"),
    *dq.cardinality_check(max_categories=30),
]

# 3. Validate performance on holdout set
holdout_df = pl.read_parquet("holdout_data.parquet")
y_true = holdout_df["claim_count"].to_numpy()
y_pred = holdout_df["predicted_frequency"].to_numpy()
exposure = holdout_df["earned_exposure"].to_numpy()

perf = PerformanceReport(y_true, y_pred, exposure=exposure, model_name="Motor TPPD v2.1")
perf_results = [
    perf.gini_coefficient(min_acceptable=0.2),
    perf.actual_vs_expected(n_bands=10),
    *perf.lift_chart(n_bands=10),
    perf.lorenz_curve(),
    perf.calibration_plot_data(),
]

# 4. Fairness checks (FCA Consumer Duty)
disc = DiscriminationReport(df=holdout_df, predictions=y_pred)
disc_results = [
    *disc.proxy_correlation(
        features=["region", "vehicle_value"],
        protected_chars=["income_proxy"],
        threshold=0.3,
    ),
    disc.disparate_impact_ratio(group_col="age_band"),
    disc.subgroup_outcome_analysis(group_col="region"),
]

# 5. Stability vs. training population
train_subset = training_df.sample(n=5000, seed=42)
stab = StabilityReport()
stab_results = [
    stab.psi(
        reference=train_subset["predicted_frequency"].to_numpy(),
        current=holdout_df["predicted_frequency"].to_numpy(),
        n_bins=10,
        label="predicted_frequency",
    ),
    *stab.feature_drift(
        reference_df=train_subset,
        current_df=holdout_df,
        features=["driver_age", "vehicle_age", "region"],
    ),
]

# 6. Generate report
all_results = dq_results + perf_results + disc_results + stab_results
gen = ReportGenerator(card, all_results)
gen.write_html("motor_tppd_validation_2024.html")
gen.write_json("motor_tppd_validation_2024.json")
```

## Regulatory mapping

| Requirement | Source | Module |
|---|---|---|
| Model inventory and classification | PRA SS1/23 Principle 1 | `ModelCard.materiality_tier` |
| Development documentation | PRA SS1/23 Principle 3 | `ModelCard` |
| Independent validation | PRA SS1/23 Principle 4 | `ModelCard.validator_name` |
| Data quality assessment | PRA SS1/23 Principle 4 | `DataQualityReport` |
| Outcome analysis | PRA SS1/23 Principle 4 | `PerformanceReport` |
| Proxy discrimination / fair value | FCA Consumer Duty; FCA TR24/2 | `DiscriminationReport` |
| Population stability monitoring | PRA SS1/23 Principle 5 | `StabilityReport` |

## Key design decisions

**Exposure-weighted Gini.** The standard ML definition of Gini (2*AUC-1) treats every observation equally. For insurance frequency models, a policy with 0.1 years of exposure should not count the same as a policy with 1.0 years. The `PerformanceReport.gini_coefficient()` method uses exposure as a weight in the Lorenz curve calculation. This matches the actuarial convention and produces a different (more meaningful) result than sklearn's `roc_auc_score`.

**PSI formula.** PSI = sum( (actual% - expected%) * ln(actual% / expected%) ). Bins are defined from quantiles of the reference distribution, not the current. A small epsilon (1e-6) is added to avoid log(0) on empty bins. Thresholds: <0.10 stable, 0.10-0.25 moderate, >0.25 significant.

**No cloud dependency.** ValidMind (the closest existing tool) requires their platform for full functionality and is AGPL-licenced. This library runs entirely locally, outputs standard HTML and JSON, and is MIT-licenced.

**Pydantic model card first.** You cannot generate a report without completing the model card. This is deliberate. Incomplete documentation is the most common validation failure - the library enforces completeness at construction time.

## Other Burning Cost libraries

**Model building**

| Library | Description |
|---------|-------------|
| [shap-relativities](https://github.com/burning-cost/shap-relativities) | Extract rating relativities from GBMs using SHAP |
| [insurance-interactions](https://github.com/burning-cost/insurance-interactions) | Automated GLM interaction detection via CANN and NID scores |
| [insurance-cv](https://github.com/burning-cost/insurance-cv) | Walk-forward cross-validation respecting IBNR structure |

**Uncertainty quantification**

| Library | Description |
|---------|-------------|
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Distribution-free prediction intervals for Tweedie models |
| [bayesian-pricing](https://github.com/burning-cost/bayesian-pricing) | Hierarchical Bayesian models for thin-data segments |
| [credibility](https://github.com/burning-cost/credibility) | Bühlmann-Straub credibility weighting |

**Deployment and optimisation**

| Library | Description |
|---------|-------------|
| [insurance-deploy](https://github.com/burning-cost/insurance-deploy) | Champion/challenger framework with ENBP audit logging |
| [insurance-elasticity](https://github.com/burning-cost/insurance-elasticity) | Causal price elasticity via Double Machine Learning |
| [rate-optimiser](https://github.com/burning-cost/rate-optimiser) | Constrained rate change optimisation with FCA PS21/5 compliance |

**Governance**

| Library | Description |
|---------|-------------|
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Proxy discrimination auditing for UK insurance models |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model monitoring: PSI, A/E ratios, Gini drift test |

[All libraries and blog posts →](https://burning-cost.github.io)

## Licence

MIT. Use it, modify it, embed it in commercial products.

## References

- PRA SS1/23: Model Risk Management Principles for Banks (May 2023)
- FCA Consumer Duty: PS22/9 (July 2022)
- FCA TR24/2: Pricing Practices Review (August 2024)
- PRA 2026 Supervision Priorities: Model risk and data quality for general insurers
