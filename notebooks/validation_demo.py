# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-validation: Full Workflow Demo
# MAGIC
# MAGIC This notebook demonstrates the complete model validation workflow on synthetic
# MAGIC motor insurance data. It covers:
# MAGIC
# MAGIC 1. Generating synthetic training and holdout datasets
# MAGIC 2. Fitting a GLM frequency model
# MAGIC 3. Running data quality checks
# MAGIC 4. Validating model performance (Gini, A/E, lift charts)
# MAGIC 5. Running discrimination / proxy checks (FCA Consumer Duty)
# MAGIC 6. Measuring population stability (PSI)
# MAGIC 7. Generating the HTML validation report

# COMMAND ----------

# MAGIC %pip install insurance-validation polars scikit-learn numpy jinja2 pydantic

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate synthetic motor insurance data

# COMMAND ----------

import numpy as np
import polars as pl
from datetime import date

rng = np.random.default_rng(42)

N_TRAIN = 20_000
N_HOLDOUT = 5_000

# --- Features ---
driver_age = rng.integers(18, 80, size=N_TRAIN + N_HOLDOUT).astype(float)
vehicle_age = rng.integers(0, 20, size=N_TRAIN + N_HOLDOUT).astype(float)
annual_mileage = rng.lognormal(9.0, 0.5, size=N_TRAIN + N_HOLDOUT)
region = rng.choice(["London", "South East", "Midlands", "North", "Scotland"],
                    size=N_TRAIN + N_HOLDOUT)
vehicle_value = rng.lognormal(9.5, 0.6, size=N_TRAIN + N_HOLDOUT)
exposure = rng.uniform(0.1, 1.0, size=N_TRAIN + N_HOLDOUT)

# Age band for discrimination analysis
age_band = np.where(driver_age < 25, "17-24",
           np.where(driver_age < 40, "25-39",
           np.where(driver_age < 60, "40-59", "60+")))

# Income proxy (correlated with region for testing proxy detection)
income_proxy = np.where(
    np.isin(region, ["London", "South East"]),
    rng.uniform(50_000, 120_000, size=N_TRAIN + N_HOLDOUT),
    rng.uniform(20_000, 60_000, size=N_TRAIN + N_HOLDOUT),
)

# --- True frequency (ground truth) ---
# Log-linear model with realistic relativities
log_base_rate = -3.0
log_freq = (
    log_base_rate
    + 0.04 * np.maximum(25 - driver_age, 0)   # young driver loading
    + 0.02 * vehicle_age
    + 0.25 * np.log(annual_mileage / 10_000)
    + np.where(region == "London", 0.3,
      np.where(region == "South East", 0.15, 0.0))
    + rng.normal(0, 0.1, size=N_TRAIN + N_HOLDOUT)  # unexplained noise
)
true_frequency = np.exp(log_freq)
claim_count = rng.poisson(true_frequency * exposure).astype(float)

# --- Build Polars DataFrames ---
all_data = pl.DataFrame({
    "driver_age": driver_age,
    "vehicle_age": vehicle_age,
    "annual_mileage": annual_mileage,
    "region": region,
    "vehicle_value": vehicle_value,
    "age_band": age_band,
    "income_proxy": income_proxy,
    "exposure": exposure,
    "claim_count": claim_count,
    "true_frequency": true_frequency,
})

# Split
train_df = all_data[:N_TRAIN]
holdout_df = all_data[N_TRAIN:]

print(f"Training rows: {len(train_df):,}")
print(f"Holdout rows:  {len(holdout_df):,}")
print(f"Training claim frequency: {train_df['claim_count'].mean() / train_df['exposure'].mean():.4f}")
print(f"Holdout claim frequency:  {holdout_df['claim_count'].mean() / holdout_df['exposure'].mean():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Fit a GLM frequency model

# COMMAND ----------

from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
import warnings

# Encode features for sklearn
def prepare_features(df: pl.DataFrame) -> np.ndarray:
    region_dummies = {
        f"region_{r}": (df["region"] == r).cast(pl.Float64).to_numpy()
        for r in ["London", "South East", "Midlands", "North"]  # Scotland is reference
    }
    X = np.column_stack([
        df["driver_age"].to_numpy(),
        df["vehicle_age"].to_numpy(),
        np.log(df["annual_mileage"].to_numpy()),
        *region_dummies.values(),
    ])
    return X

X_train = prepare_features(train_df)
y_train = train_df["claim_count"].to_numpy() / train_df["exposure"].to_numpy()
w_train = train_df["exposure"].to_numpy()

X_holdout = prepare_features(holdout_df)

# Fit Poisson GLM (exposure-weighted)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_holdout_scaled = scaler.transform(X_holdout)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    glm = PoissonRegressor(alpha=0, max_iter=500)
    glm.fit(X_train_scaled, y_train, sample_weight=w_train)

# Predictions (frequency per unit exposure)
y_pred_train = glm.predict(X_train_scaled)
y_pred_holdout = glm.predict(X_holdout_scaled)

print(f"Train mean predicted frequency:   {y_pred_train.mean():.4f}")
print(f"Holdout mean predicted frequency: {y_pred_holdout.mean():.4f}")
print(f"Holdout mean actual frequency:    {(holdout_df['claim_count'].to_numpy() / holdout_df['exposure'].to_numpy()).mean():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define the model card

# COMMAND ----------

from insurance_validation import ModelCard

card = ModelCard(
    model_name="Motor TPPD Claim Frequency",
    version="1.0.0",
    purpose=(
        "Estimate expected claim frequency (claims per earned exposure year) "
        "for private motor third-party property damage. Used in policy pricing."
    ),
    intended_use=(
        "Underwriting pricing for private motor policies. "
        "Not for use in claims reserving, capital modelling, or reinsurance pricing."
    ),
    developer="Pricing Analytics Team",
    development_date=date(2024, 6, 1),
    limitations=(
        "Trained on 2018-2023 underwriting years. Performance has not been validated "
        "for vehicles older than 15 years (less than 2% of portfolio). "
        "Young driver (<21) relativities are based on limited data and should be "
        "reviewed against industry benchmarks."
    ),
    materiality_tier=2,
    approved_by=["Jane Smith - Chief Actuary", "Model Risk Committee"],
    variables=["driver_age", "vehicle_age", "annual_mileage", "region"],
    target_variable="claim_count",
    model_type="GLM",
    distribution_family="Poisson",
    validator_name="Independent Validation Unit",
    validation_date=date(2024, 9, 1),
    alternatives_considered=(
        "GBM (LightGBM) was evaluated and achieved Gini 0.41 vs GLM 0.35. "
        "GLM was selected for regulatory interpretability and stable relativities. "
        "The GBM will be reconsidered if GLM Gini falls below 0.25."
    ),
    monitoring_frequency="Quarterly",
)

print(f"Model: {card.model_name} v{card.version}")
print(f"Tier:  {card.materiality_tier}")
print(f"Variables: {', '.join(card.variables)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Data quality checks

# COMMAND ----------

from insurance_validation import DataQualityReport

# Add some artificial nulls to demonstrate the check
train_with_nulls = train_df.with_columns([
    pl.when(pl.int_range(pl.len()) % 50 == 0)
    .then(None)
    .otherwise(pl.col("annual_mileage"))
    .alias("annual_mileage")
])

dq = DataQualityReport(train_with_nulls, dataset_name="Motor training 2018-2023")

dq_results = [
    dq.summary_statistics(),
    *dq.missing_value_analysis(threshold=0.05),
    *dq.outlier_detection(method="iqr", iqr_multiplier=3.0),
    *dq.cardinality_check(max_categories=20),
]

print(f"\nData quality tests: {len(dq_results)}")
failures = [r for r in dq_results if not r.passed]
print(f"Failures: {len(failures)}")
for r in failures:
    print(f"  - {r.test_name}: {r.details[:80]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Performance validation

# COMMAND ----------

from insurance_validation import PerformanceReport

y_true = holdout_df["claim_count"].to_numpy()
exposure = holdout_df["exposure"].to_numpy()

perf = PerformanceReport(
    y_true=y_true,
    y_pred=y_pred_holdout,
    exposure=exposure,
    model_name="Motor TPPD GLM v1.0",
)

perf_results = [
    perf.gini_coefficient(min_acceptable=0.2),
    perf.actual_vs_expected(n_bands=10),
    *perf.lift_chart(n_bands=10),
    perf.lorenz_curve(),
    perf.calibration_plot_data(),
]

gini_result = next(r for r in perf_results if r.test_name == "gini_coefficient")
ae_result = next(r for r in perf_results if r.test_name == "actual_vs_expected")
print(f"Gini coefficient: {gini_result.metric_value:.4f} ({'PASS' if gini_result.passed else 'FAIL'})")
print(f"A/E ratio:        {ae_result.metric_value:.4f} ({'PASS' if ae_result.passed else 'FAIL'})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Discrimination checks (FCA Consumer Duty)

# COMMAND ----------

from insurance_validation import DiscriminationReport

disc = DiscriminationReport(
    df=holdout_df,
    predictions=y_pred_holdout,
)

disc_results = [
    *disc.proxy_correlation(
        features=["vehicle_value", "region"],
        protected_chars=["income_proxy"],
        threshold=0.3,
    ),
    disc.disparate_impact_ratio(
        group_col="age_band",
        threshold=0.8,
    ),
    disc.subgroup_outcome_analysis(
        group_col="age_band",
        outcome_col="claim_count",
    ),
    disc.subgroup_outcome_analysis(
        group_col="region",
        outcome_col="claim_count",
    ),
]

print(f"\nDiscrimination tests: {len(disc_results)}")
for r in disc_results:
    status = "PASS" if r.passed else "FAIL"
    metric = f"{r.metric_value:.4f}" if r.metric_value is not None else "N/A"
    print(f"  {status} | {r.test_name}: {metric}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Population stability (PSI)

# COMMAND ----------

from insurance_validation import StabilityReport

stab = StabilityReport()

stab_results = [
    stab.psi(
        reference=y_pred_train,
        current=y_pred_holdout,
        n_bins=10,
        label="predicted_frequency",
    ),
    *stab.feature_drift(
        reference_df=train_df,
        current_df=holdout_df,
        features=["driver_age", "vehicle_age", "annual_mileage", "region"],
        n_bins=10,
    ),
]

print(f"\nStability tests: {len(stab_results)}")
for r in stab_results:
    status = "PASS" if r.passed else "FAIL"
    metric = f"{r.metric_value:.4f}" if r.metric_value is not None else "N/A"
    print(f"  {status} | {r.test_name}: PSI = {metric}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Generate validation report

# COMMAND ----------

from insurance_validation import ReportGenerator

all_results = dq_results + perf_results + disc_results + stab_results

total = len(all_results)
passed = sum(1 for r in all_results if r.passed)
failed = total - passed

print(f"Total tests: {total}")
print(f"Passed:      {passed}")
print(f"Failed:      {failed}")

gen = ReportGenerator(card, all_results, generated_date=date(2024, 9, 1))

# Write HTML report
html_path = "/tmp/motor_tppd_validation_2024.html"
json_path = "/tmp/motor_tppd_validation_2024.json"

gen.write_html(html_path)
gen.write_json(json_path)

print(f"\nReport written to: {html_path}")
print(f"JSON sidecar:      {json_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Display report summary

# COMMAND ----------

import json

with open(json_path) as f:
    report_data = json.load(f)

summary = report_data["summary"]
print("Report summary:")
print(f"  Total tests:  {summary['total_tests']}")
print(f"  Passed:       {summary['passed']}")
print(f"  Failed:       {summary['failed']}")
print(f"  Critical:     {summary['critical']}")
print(f"  Warnings:     {summary['warnings']}")

# Show per-category breakdown
from collections import Counter
cats = Counter(r["category"] for r in report_data["results"])
print("\nTests by category:")
for cat, count in sorted(cats.items()):
    cat_results = [r for r in report_data["results"] if r["category"] == cat]
    cat_pass = sum(1 for r in cat_results if r["passed"])
    print(f"  {cat:20s}: {count:3d} tests, {cat_pass} passed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This demo covered the full validation workflow:
# MAGIC
# MAGIC - **Data quality**: {N_TRAIN:,} training rows checked for missing values, outliers, and cardinality
# MAGIC - **Performance**: Gini coefficient (exposure-weighted), A/E ratio, lift chart on {N_HOLDOUT:,} holdout policies
# MAGIC - **Discrimination**: Proxy correlation (vehicle value vs. income proxy), disparate impact by age band, subgroup outcomes by region
# MAGIC - **Stability**: PSI on predicted frequency and all 4 model features
# MAGIC - **Report**: Self-contained HTML + JSON sidecar written to /tmp/
# MAGIC
# MAGIC To use in production: replace synthetic data with your actual training/holdout datasets,
# MAGIC update the ModelCard metadata, and schedule the notebook quarterly per your monitoring plan.
