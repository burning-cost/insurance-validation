"""
Tests for PerformanceReport.

Key checks:
- Gini = 2*AUC - 1 (verified against a known binary case)
- PSI formula correctness verified in test_stability.py
- Exposure-weighted Gini differs from unweighted when exposure varies
- A/E ratio = 1.0 for a perfectly calibrated model
- Lift chart returns n_bands results
"""
import numpy as np
import pytest
from insurance_validation import PerformanceReport
from insurance_validation.results import TestCategory


def test_perfect_gini_is_one():
    """A model that ranks perfectly should have Gini = 1."""
    y_true = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    y_pred = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    report = PerformanceReport(y_true, y_pred)
    result = report.gini_coefficient()
    assert result.metric_value == pytest.approx(1.0, abs=1e-6)


def test_zero_gini_for_random_model():
    """A constant predictor should have Gini close to 0."""
    rng = np.random.default_rng(42)
    y_true = rng.choice([0, 1], size=1000).astype(float)
    y_pred = np.ones(1000) * 0.5  # constant prediction
    report = PerformanceReport(y_true, y_pred)
    result = report.gini_coefficient(min_acceptable=0.0)
    assert abs(result.metric_value) < 0.05


def test_gini_equals_2_auc_minus_1():
    """Verify Gini = 2*AUC-1 for binary outcome against sklearn."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(123)
    n = 500
    y_true = rng.choice([0, 1], size=n).astype(float)
    y_pred = rng.uniform(0, 1, size=n)

    report = PerformanceReport(y_true, y_pred)
    result = report.gini_coefficient(min_acceptable=-1.0)

    sklearn_auc = roc_auc_score(y_true, y_pred)
    expected_gini = 2 * sklearn_auc - 1

    assert result.metric_value == pytest.approx(expected_gini, abs=1e-4)


def test_gini_returns_test_result():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
    report = PerformanceReport(y_true, y_pred)
    result = report.gini_coefficient()
    assert result.category == TestCategory.PERFORMANCE
    assert result.test_name == "gini_coefficient"
    assert result.metric_value is not None


def test_gini_below_threshold_fails():
    y_true = np.ones(100)
    y_pred = np.ones(100)
    report = PerformanceReport(y_true, y_pred)
    result = report.gini_coefficient(min_acceptable=0.5)
    assert result.passed is False


def test_exposure_weighted_gini_differs_from_unweighted():
    """Exposure-weighted Gini should differ when exposure is heterogeneous."""
    rng = np.random.default_rng(7)
    n = 200
    y_true = rng.poisson(0.1, size=n).astype(float)
    y_pred = rng.uniform(0.05, 0.2, size=n)
    exposure = rng.uniform(0.1, 1.0, size=n)

    unweighted = PerformanceReport(y_true, y_pred)
    weighted = PerformanceReport(y_true, y_pred, exposure=exposure)

    g_unweighted = unweighted.gini_coefficient(min_acceptable=-1.0).metric_value
    g_weighted = weighted.gini_coefficient(min_acceptable=-1.0).metric_value

    # They should not be identical (exposure introduces a different weighting)
    assert g_unweighted != pytest.approx(g_weighted, abs=1e-4)


def test_ae_ratio_perfect_calibration():
    """A/E ratio should be 1.0 when predictions = actuals."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    report = PerformanceReport(y, y.copy())
    result = report.actual_vs_expected()
    assert result.metric_value == pytest.approx(1.0, abs=1e-6)
    assert result.passed is True


def test_ae_ratio_overcalibrated():
    y_true = np.ones(100)
    y_pred = np.ones(100) * 2.0  # predicting twice the actuals
    report = PerformanceReport(y_true, y_pred)
    result = report.actual_vs_expected()
    assert result.metric_value == pytest.approx(0.5, abs=1e-6)
    assert result.passed is False


def test_ae_ratio_returns_band_data():
    y = np.linspace(1, 10, 100)
    report = PerformanceReport(y, y * 1.05)
    result = report.actual_vs_expected(n_bands=5)
    assert "bands" in result.extra
    assert len(result.extra["bands"]) == 5


def test_lift_chart_returns_list():
    rng = np.random.default_rng(1)
    y = rng.uniform(0, 1, 100)
    yhat = rng.uniform(0, 1, 100)
    report = PerformanceReport(y, yhat)
    results = report.lift_chart(n_bands=10)
    assert len(results) == 1
    assert results[0].extra["n_bands"] == 10
    assert len(results[0].extra["bands"]) == 10


def test_lorenz_curve_extra_has_xy():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
    report = PerformanceReport(y_true, y_pred)
    result = report.lorenz_curve(n_points=20)
    assert "x" in result.extra
    assert "y" in result.extra
    assert result.extra["x"][0] == pytest.approx(0.0, abs=1e-6)
    assert result.extra["y"][0] == pytest.approx(0.0, abs=1e-6)


def test_calibration_plot_data():
    y = np.linspace(1, 5, 50)
    report = PerformanceReport(y, y * 1.02)
    result = report.calibration_plot_data(n_bands=5)
    assert "points" in result.extra
    assert len(result.extra["points"]) == 5


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        PerformanceReport([1, 2, 3], [1, 2])


def test_mismatched_weights_raise():
    with pytest.raises(ValueError, match="same length"):
        PerformanceReport([1, 2, 3], [1, 2, 3], weights=[1, 2])
