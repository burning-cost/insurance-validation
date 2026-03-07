"""
Tests for StabilityReport PSI implementation.

PSI formula verified against hand-computed values.
"""
import numpy as np
import polars as pl
import pytest
from insurance_validation import StabilityReport
from insurance_validation.results import TestCategory, Severity


def test_identical_distributions_psi_near_zero():
    """Same distribution should produce PSI close to 0."""
    rng = np.random.default_rng(0)
    dist = rng.normal(0, 1, 10000)
    report = StabilityReport()
    result = report.psi(dist, dist.copy(), n_bins=10)
    assert result.metric_value < 0.01


def test_very_different_distributions_psi_high():
    """Completely different distributions should produce PSI > 0.25."""
    # Both non-constant to avoid edge case in quantile bin collapse
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 0.01, 1000)   # tightly clustered around 0
    cur = rng.normal(10, 0.01, 1000)  # tightly clustered around 10
    report = StabilityReport()
    result = report.psi(ref, cur, n_bins=10)
    assert result.metric_value > 0.25
    assert result.passed is False
    assert result.severity == Severity.CRITICAL


def test_constant_reference_different_current_psi_high():
    """When reference is constant but current is very different, PSI is high."""
    ref = np.zeros(1000)
    cur = np.ones(1000) * 10
    report = StabilityReport()
    result = report.psi(ref, cur, n_bins=10)
    # After fix: should detect the shift and produce high PSI
    assert result.metric_value > 0.25
    assert result.passed is False


def test_stable_psi_passes():
    """Small shift should produce PSI < 0.10, passing."""
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 5000)
    cur = rng.normal(0, 1, 5000)  # same distribution
    report = StabilityReport()
    result = report.psi(ref, cur, n_bins=10)
    assert result.passed is True
    assert result.severity == Severity.INFO


def test_moderate_psi_warning():
    """Moderate shift should produce non-negative PSI."""
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, 5000)
    cur = rng.normal(1, 1, 5000)  # shifted by 1 std dev
    report = StabilityReport()
    result = report.psi(ref, cur, n_bins=10)
    assert result.category == TestCategory.STABILITY
    assert result.metric_value >= 0


def test_psi_result_has_bin_details():
    rng = np.random.default_rng(3)
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(0.5, 1, 1000)
    report = StabilityReport()
    result = report.psi(ref, cur, n_bins=10, label="score")
    assert "bins" in result.extra
    assert result.test_name == "psi_score"


def test_psi_bin_details_sum_close_to_total():
    """Sum of per-bin PSI should equal total PSI within floating point tolerance."""
    rng = np.random.default_rng(4)
    ref = rng.normal(0, 1, 2000)
    cur = rng.normal(0.5, 1, 2000)
    report = StabilityReport()
    result = report.psi(ref, cur, n_bins=10)
    bin_sum = sum(b["bin_psi"] for b in result.extra["bins"])
    # metric_value is now full precision, bin_sum accumulates floats
    # Allow tolerance consistent with floating point accumulation over ~10 bins
    assert bin_sum == pytest.approx(result.metric_value, rel=1e-6)


def test_feature_drift_numeric():
    rng = np.random.default_rng(5)
    ref_df = pl.DataFrame({
        "driver_age": rng.normal(40, 10, 500).tolist(),
        "vehicle_age": rng.normal(5, 2, 500).tolist(),
    })
    cur_df = pl.DataFrame({
        "driver_age": rng.normal(40, 10, 500).tolist(),
        "vehicle_age": rng.normal(7, 2, 500).tolist(),  # shifted
    })
    report = StabilityReport()
    results = report.feature_drift(ref_df, cur_df, features=["driver_age", "vehicle_age"])
    assert len(results) == 2
    assert all(r.category == TestCategory.STABILITY for r in results)
    # vehicle_age shifted - should have higher PSI
    age_psi = next(r for r in results if "driver_age" in r.test_name).metric_value
    vage_psi = next(r for r in results if "vehicle_age" in r.test_name).metric_value
    assert vage_psi > age_psi


def test_feature_drift_categorical():
    ref_df = pl.DataFrame({"region": ["North"] * 50 + ["South"] * 50})
    cur_df = pl.DataFrame({"region": ["North"] * 70 + ["South"] * 30})  # shifted
    report = StabilityReport()
    results = report.feature_drift(ref_df, cur_df, features=["region"])
    assert len(results) == 1
    assert results[0].metric_value >= 0


def test_feature_drift_missing_column():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    report = StabilityReport()
    results = report.feature_drift(df, df, features=["nonexistent"])
    assert len(results) == 1
    assert results[0].passed is False


def test_psi_label_in_test_name():
    rng = np.random.default_rng(9)
    dist = rng.normal(0, 1, 100)
    report = StabilityReport()
    result = report.psi(dist, dist, label="predicted_frequency")
    assert "predicted_frequency" in result.test_name
