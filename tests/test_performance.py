"""
Tests for PerformanceReport.

Key checks:
- Gini is in [-1, 1], higher = better discrimination
- Exposure-weighted Gini differs from unweighted when exposure varies
- A/E ratio = 1.0 for a perfectly calibrated model
- Lift chart returns n_bands results

On Gini convention: this library uses the actuarial Lorenz curve definition.
Policies are ranked descending by predicted score. The x-axis is cumulative
exposure share; y-axis is cumulative loss share. Gini = 2 * AUC - 1 where
AUC is the area under this Lorenz curve.

For this formulation, a perfect model (one with the ideal ranking) does not
necessarily give Gini = 1.0 - the theoretical maximum depends on the
distribution of y_true. Gini = 1.0 is only achievable in the degenerate
case where all loss is concentrated in a single observation. What matters
is that better-ranked models have higher Gini, and the metric is in
[-1, 1].

If you need the ROC-AUC-based Gini (= 2*roc_auc_score-1), that is a
different quantity appropriate for binary outcomes only.
"""
import numpy as np
import pytest
from insurance_validation import PerformanceReport
from insurance_validation.results import TestCategory


def test_gini_is_bounded():
    """Gini must be in [-1, 1]."""
    rng = np.random.default_rng(42)
    y_true = rng.exponential(0.1, size=200)
    y_pred = rng.uniform(0, 0.2, size=200)
    report = PerformanceReport(y_true, y_pred)
    result = report.gini_coefficient(min_acceptable=-1.0)
    assert -1.0 <= result.metric_value <= 1.0


def test_zero_gini_for_constant_prediction():
    """A constant predictor should have Gini close to 0."""
    rng = np.random.default_rng(42)
    y_true = rng.uniform(0, 1, size=1000)
    y_pred = np.ones(1000) * 0.5  # constant prediction
    report = PerformanceReport(y_true, y_pred)
    result = report.gini_coefficient(min_acceptable=-1.0)
    assert abs(result.metric_value) < 0.05


def test_gini_better_model_has_higher_gini():
    """A model with better rankings should have a higher Gini."""
    rng = np.random.default_rng(123)
    n = 500
    y_true = rng.exponential(0.1, size=n)

    # Good model: prediction is close to truth
    y_pred_good = y_true + rng.normal(0, 0.01, size=n)
    # Bad model: mostly random
    y_pred_bad = rng.uniform(0, 0.2, size=n)

    report_good = PerformanceReport(y_true, y_pred_good)
    report_bad = PerformanceReport(y_true, y_pred_bad)

    gini_good = report_good.gini_coefficient(min_acceptable=-1.0).metric_value
    gini_bad = report_bad.gini_coefficient(min_acceptable=-1.0).metric_value

    assert gini_good > gini_bad


def test_gini_known_value():
    """
    Verify Gini against a hand-computed case.

    y_true = [0, 1, 0, 2, 0], y_pred = [0.1, 0.9, 0.2, 0.8, 0.3]
    Sorted descending by y_pred: y_true = [1, 2, 0, 0, 0]
    weights = [1,1,1,1,1], total_loss = 3
    cum_loss = [1, 3, 3, 3, 3]
    x = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    y = [0, 1/3, 3/3, 1, 1, 1] = [0, 0.333, 1.0, 1.0, 1.0, 1.0]
    AUC = trapz = 0.2*(0+0.333)/2 + 0.2*(0.333+1)/2 + 0.2*(1+1)/2*3
         = 0.0333 + 0.1333 + 0.6 = 0.7667
    Gini = 2*0.7667 - 1 = 0.5333
    """
    y_true = np.array([0.0, 1.0, 0.0, 2.0, 0.0])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    report = PerformanceReport(y_true, y_pred)
    result = report.gini_coefficient(min_acceptable=-1.0)

    # Hand-computed value
    expected_gini = 2 * 0.7667 - 1
    assert result.metric_value == pytest.approx(expected_gini, abs=0.002)


def test_gini_returns_test_result():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
    report = PerformanceReport(y_true, y_pred)
    result = report.gini_coefficient()
    assert result.category == TestCategory.PERFORMANCE
    assert result.test_name == "gini_coefficient"
    assert result.metric_value is not None


def test_gini_below_threshold_fails():
    """Constant predictions give near-zero Gini, should fail a >0.5 threshold."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 1.0, 1.0])  # constant
    report = PerformanceReport(y_true, y_pred)
    result = report.gini_coefficient(min_acceptable=0.5)
    assert result.passed is False


def test_exposure_weighted_gini_differs_from_unweighted():
    """Exposure-weighted Gini should differ when exposure is heterogeneous."""
    rng = np.random.default_rng(99)
    n = 500
    # Create data where exposure is strongly correlated with outcome
    # so weighting changes the effective contribution of each observation
    y_true = rng.exponential(0.1, size=n)
    y_pred = y_true + rng.normal(0, 0.05, size=n)
    # Exposure: high-risk policies (high y_true) have much shorter exposure
    # This strongly changes the weighted Lorenz curve shape
    exposure = 1.0 / (1.0 + y_true * 20) + rng.uniform(0, 0.02, size=n)

    unweighted = PerformanceReport(y_true, y_pred)
    weighted = PerformanceReport(y_true, y_pred, exposure=exposure)

    g_unweighted = unweighted.gini_coefficient(min_acceptable=-1.0).metric_value
    g_weighted = weighted.gini_coefficient(min_acceptable=-1.0).metric_value

    # They should differ because exposure re-weights the Lorenz curve
    # Using a tight inverse relationship ensures a meaningful difference
    assert abs(g_unweighted - g_weighted) > 0.001


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


def test_gini_monotone_with_ranking_quality():
    """Better-ranked predictions should produce higher Gini."""
    rng = np.random.default_rng(77)
    n = 300
    y_true = rng.exponential(1.0, size=n)

    ginis = []
    for noise_scale in [0.01, 0.5, 2.0, 5.0]:
        y_pred = y_true + rng.normal(0, noise_scale, size=n)
        r = PerformanceReport(y_true, y_pred)
        ginis.append(r.gini_coefficient(min_acceptable=-1.0).metric_value)

    # More noise = lower Gini
    assert ginis[0] > ginis[1] > ginis[2]
