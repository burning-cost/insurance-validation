# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-validation v0.2.0: PRA SS1/23 Compliant Model Validation
# MAGIC
# MAGIC This notebook demonstrates the full validation workflow for a synthetic
# MAGIC UK motor insurance frequency model. It covers every section of a
# MAGIC PRA SS1/23 aligned validation report:
# MAGIC
# MAGIC 1. Executive Summary with RAG status
# MAGIC 2. Model Card and Governance
# MAGIC 3. Data Quality Assessment
# MAGIC 4. Model Development Documentation
# MAGIC 5. Performance Validation (Gini + CI, A/E + Poisson CI, double-lift, HL test)
# MAGIC 6. Fairness and Discrimination (FCA Consumer Duty / TR24/2)
# MAGIC 7. Population Stability (PSI, feature drift)
# MAGIC 8. Feature Importance (optional SHAP)
# MAGIC 9. Monitoring Plan and Sign-Off

# COMMAND ----------
# MAGIC %pip install insurance-validation scipy polars pydantic jinja2 scikit-learn numpy -q

# COMMAND ----------
import numpy as np
import polars as pl
from datetime import date
import tempfile
from pathlib import Path

from insurance_validation import (
    ModelValidationReport,
    ModelCard,
    PerformanceReport,
    DataQualityReport,
    DiscriminationReport,
    StabilityReport,
    ReportGenerator,
)

print("insurance-validation imported OK")

# COMMAND ----------
# MAGIC %md ## 1. Synthetic data: UK motor frequency model

# COMMAND ----------
rng = np.random.default_rng(2024)
N_TRAIN = 10_000
N_VAL = 3_000

driver_age = rng.integers(17, 80, N_TRAIN + N_VAL).astype(float)
vehicle_age = rng.integers(0, 20, N_TRAIN + N_VAL).astype(float)
area = rng.choice(["Urban", "Rural", "Suburban"], N_TRAIN + N_VAL, p=[0.45, 0.25, 0.30])
ncd_years = rng.integers(0, 9, N_TRAIN + N_VAL).astype(float)
tenure = rng.integers(0, 7, N_TRAIN + N_VAL).astype(int)
segment = rng.choice(["Standard", "Enhanced", "Premium"], N_TRAIN + N_VAL, p=[0.6, 0.3, 0.1])

log_mu = (
    -3.0
    + 0.015 * (driver_age - 35)
    - 0.005 * (driver_age - 35) ** 2 / 100
    + 0.03 * vehicle_age
    + np.where(area == "Urban", 0.3, 0.0)
    + np.where(area == "Suburban", 0.1, 0.0)
    - 0.05 * ncd_years
)
mu_true = np.exp(log_mu)
exposure = rng.uniform(0.25, 1.0, N_TRAIN + N_VAL)
claim_count = rng.poisson(mu_true * exposure)

new_pred = mu_true * np.exp(rng.normal(0, 0.08, N_TRAIN + N_VAL))
old_pred = np.exp(-3.0 + 0.015 * (driver_age - 35) - 0.005 * (driver_age - 35)**2/100
                  + 0.02 * vehicle_age - 0.04 * ncd_years + rng.normal(0, 0.2, N_TRAIN + N_VAL))

y_train = claim_count[:N_TRAIN].astype(float)
y_val = claim_count[N_TRAIN:].astype(float)
pred_train = new_pred[:N_TRAIN]
pred_val = new_pred[N_TRAIN:]
pred_val_old = old_pred[N_TRAIN:]
exp_train = exposure[:N_TRAIN]
exp_val = exposure[N_TRAIN:]

X_val = pl.DataFrame({
    "driver_age": driver_age[N_TRAIN:].tolist(),
    "vehicle_age": vehicle_age[N_TRAIN:].tolist(),
    "area": area[N_TRAIN:].tolist(),
    "ncd_years": ncd_years[N_TRAIN:].tolist(),
    "tenure": tenure[N_TRAIN:].tolist(),
    "segment": segment[N_TRAIN:].tolist(),
})
X_train = pl.DataFrame({
    "driver_age": driver_age[:N_TRAIN].tolist(),
    "vehicle_age": vehicle_age[:N_TRAIN].tolist(),
    "area": area[:N_TRAIN].tolist(),
    "ncd_years": ncd_years[:N_TRAIN].tolist(),
    "tenure": tenure[:N_TRAIN].tolist(),
    "segment": segment[:N_TRAIN].tolist(),
})

print(f"Train: {N_TRAIN:,} rows | Val: {N_VAL:,} rows")
print(f"Val claim rate: {y_val.sum():.0f} claims / {exp_val.sum():.0f} years = {y_val.sum()/exp_val.sum():.4f}")

# COMMAND ----------
# MAGIC %md ## 2. Model Card

# COMMAND ----------
card = ModelCard(
    name="Motor Third-Party Property Damage Frequency",
    version="3.2.0",
    purpose=(
        "Predict claim frequency (claims per policy year) for private motor TPPD. "
        "Used in underwriting pricing to derive the risk premium component."
    ),
    methodology="CatBoost gradient boosting, Poisson log-loss objective, 500 trees",
    target="claim_count",
    features=["driver_age", "vehicle_age", "area", "ncd_years"],
    limitations=[
        "Performance degrades for vehicles over 15 years old (sparse data)",
        "No telematics data in current specification",
        "Trained on 2020-2024 data; pre-2020 regime not represented",
    ],
    owner="Personal Lines Pricing Team",
    approved_by=["Jane Smith - Chief Actuary", "Model Risk Committee"],
    development_date=date(2024, 10, 1),
    materiality_tier=2,
    model_type="GBM",
    distribution_family="Poisson",
    validator_name="Independent Actuarial Services Ltd",
    validation_date=date(2025, 1, 15),
    alternatives_considered=(
        "GLM (Poisson) considered but rejected due to lower Gini (0.31 vs 0.44). "
        "Neural network rejected due to interpretability requirements."
    ),
    monitoring_frequency="Quarterly",
    outstanding_issues=[
        "Vehicle age >15yr performance requires banding review (Q2 2025)",
        "Telematics data integration planned for v4.0",
    ],
    monitoring_owner="Actuarial Risk & Analytics Team",
    monitoring_triggers={
        "psi_score": 0.25,
        "ae_ratio_deviation": 0.10,
        "gini_drop": 0.05,
    },
)

print(card.summary())

# COMMAND ----------
# MAGIC %md ## 3. Full validation report via high-level API

# COMMAND ----------
report = ModelValidationReport(
    model_card=card,
    y_val=y_val,
    y_pred_val=pred_val,
    exposure_val=exp_val,
    y_train=y_train,
    y_pred_train=pred_train,
    exposure_train=exp_train,
    X_train=X_train,
    X_val=X_val,
    incumbent_pred_val=pred_val_old,
    tenure_col="tenure",
    segment_col="segment",
    monitoring_owner=card.monitoring_owner,
    monitoring_triggers=card.monitoring_triggers,
)

results = report.run()
rag = report.get_rag_status()

print(f"RAG Status: {rag.value.upper()}")
print(f"Total tests: {len(results)}")
print(f"Passed: {sum(1 for r in results if r.passed)}")
print(f"Failed: {sum(1 for r in results if not r.passed)}")
print(f"Critical: {sum(1 for r in results if not r.passed and r.severity.value == 'critical')}")
print(f"Warnings: {sum(1 for r in results if not r.passed and r.severity.value == 'warning')}")

# COMMAND ----------
# MAGIC %md ## 4. Performance highlights

# COMMAND ----------
gini_ci = next(r for r in results if r.test_name == "gini_with_ci")
ae_ci = next(r for r in results if r.test_name == "ae_poisson_ci")
hl = next(r for r in results if r.test_name == "hosmer_lemeshow")
dl = next(r for r in results if r.test_name == "double_lift")

print("=== Performance Summary ===")
print(f"Gini: {gini_ci.metric_value:.4f}  (95% bootstrap CI: [{gini_ci.extra['ci_lower']:.4f}, {gini_ci.extra['ci_upper']:.4f}])")
print(f"A/E:  {ae_ci.metric_value:.4f}  (Poisson 95% CI: [{ae_ci.extra['ci_lower']:.4f}, {ae_ci.extra['ci_upper']:.4f}])")
print(f"H-L test: p-value = {hl.extra['p_value']:.4f}  ({'PASS' if hl.passed else 'FAIL'})")
print(f"Double-lift: new MAE={dl.extra['new_model_mae']:.5f}, incumbent MAE={dl.extra['incumbent_mae']:.5f}  ({'New model wins' if dl.passed else 'Incumbent wins'})")

# COMMAND ----------
# MAGIC %md ## 5. Renewal cohort and sub-segment A/E

# COMMAND ----------
disc = DiscriminationReport(df=X_val)

renewal_result = disc.renewal_cohort_ae(y_true=y_val, y_pred=pred_val, tenure_col="tenure")
print("Renewal cohort A/E by tenure band:")
for band in renewal_result.extra["bands"]:
    ae_str = f"{band['ae_ratio']:.4f}" if band["ae_ratio"] is not None else "N/A"
    status = "OK" if band["in_range"] else "FLAG"
    print(f"  {band['band']:>5}: n={band['n']:>4}, A/E={ae_str} [{status}]")

seg_result = disc.subsegment_ae(y_true=y_val, y_pred=pred_val, segment_col="segment")
print(f"\nSub-segment A/E ({'PASS' if seg_result.passed else 'FAIL'}):")
for seg in seg_result.extra["segments"]:
    ae_str = f"{seg['ae_ratio']:.4f}" if seg["ae_ratio"] is not None else "N/A"
    print(f"  {seg['segment']:>10}: n={seg['n']:>4}, A/E={ae_str}")

# COMMAND ----------
# MAGIC %md ## 6. Write HTML and JSON reports

# COMMAND ----------
with tempfile.TemporaryDirectory() as tmpdir:
    html_path = Path(tmpdir) / "motor_frequency_validation.html"
    json_path = Path(tmpdir) / "motor_frequency_validation.json"
    report.generate(html_path)
    report.to_json(json_path)
    import shutil
    shutil.copy(html_path, "/tmp/motor_frequency_validation.html")
    shutil.copy(json_path, "/tmp/motor_frequency_validation.json")
    print(f"HTML: {html_path.stat().st_size:,} bytes -> /tmp/motor_frequency_validation.html")
    print(f"JSON: {json_path.stat().st_size:,} bytes -> /tmp/motor_frequency_validation.json")

print(f"Run ID: {report._run_id}")
print("\n=== Demo complete ===")
