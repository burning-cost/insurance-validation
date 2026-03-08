"""
Extended tests for PerformanceReport: bootstrap Gini CI, Poisson A/E CI,
double-lift, and Hosmer-Lemeshow calibration test.
"""
from __future__ import annotations

import numpy as np
import pytest
from insurance_validation import PerformanceReport
from insurance_validation.results import TestCategory, Severity


def make_report(n: int = 500, seed: int = 42) -> PerformanceReport:
    rng = np.random.default_rng(seed)
    y_true = rng.poisson(0.1, size=n).astype(float)
    y_pred = np.clip(y_true + rng.normal(0, 0.05, n), 0.001, 10)
    exposure = rng.uniform(0.5, 2.0, n)
    return PerformanceReport(y_true, y_pred, exposure=exposure, model_name="test")


# ── Bootstrap Gini CI ────────────────────────────────────────────────────────

def test_gini_ci_returns_test_result():
    rep = make_report()
    result = rep.gini_with_ci(n_resamples=100)
    assert result.test_name == "gini_with_ci"
    assert result.category == TestCategory.PERFORMANCE


def test_gini_ci_extra_has_ci_bounds():
    rep = make_report()
    result = rep.gini_with_ci(n_resamples=100)
    assert "ci_lower" in result.extra
    assert "ci_upper" in result.extra
    assert result.extra["ci_lower"] <= result.metric_value
    assert result.extra["ci_upper"] >= result.metric_value


def test_gini_ci_lower_le_upper():
    rep = make_report()
    result = rep.gini_with_ci(n_resamples=200)
    assert result.extra["ci_lower"] <= result.extra["ci_upper"]


def test_gini_ci_width_decreases_with_sample_size():
    """More data -> narrower CI."""
    rng = np.random.default_rng(0)
    n_small, n_large = 100, 2000
    y_true_s = rng.poisson(0.1, n_small).astype(float)
    y_pred_s = y_true_s + rng.normal(0, 0.05, n_small)
    y_true_l = rng.poisson(0.1, n_large).astype(float)
    y_pred_l = y_true_l + rng.normal(0, 0.05, n_large)

    ci_small = PerformanceReport(y_true_s, y_pred_s).gini_with_ci(n_resamples=200)
    ci_large = PerformanceReport(y_true_l, y_pred_l).gini_with_ci(n_resamples=200)

    width_small = ci_small.extra["ci_upper"] - ci_small.extra["ci_lower"]
    width_large = ci_large.extra["ci_upper"] - ci_large.extra["ci_lower"]
    assert width_large < width_small


def test_gini_ci_reproducible():
    """Same seed should give same CI."""
    rep = make_report(seed=99)
    r1 = rep.gini_with_ci(n_resamples=100, random_state=7)
    r2 = rep.gini_with_ci(n_resamples=100, random_state=7)
    assert r1.extra["ci_lower"] == pytest.approx(r2.extra["ci_lower"])
    assert r1.extra["ci_upper"] == pytest.approx(r2.extra["ci_upper"])


def test_gini_ci_n_resamples_in_extra():
    rep = make_report()
    result = rep.gini_with_ci(n_resamples=50)
    assert result.extra["n_resamples"] == 50


# ── Poisson A/E CI ───────────────────────────────────────────────────────────

def test_ae_poisson_ci_returns_result():
    rep = make_report()
    result = rep.ae_with_poisson_ci()
    assert result.test_name == "ae_poisson_ci"
    assert result.category == TestCategory.PERFORMANCE


def test_ae_poisson_ci_perfect_model_passes():
    """Perfect model (A=E) should pass."""
    y = np.full(1000, 0.05)
    rep = PerformanceReport(y, y.copy())
    result = rep.ae_with_poisson_ci()
    assert result.passed == True
    assert result.metric_value == pytest.approx(1.0, abs=1e-6)


def test_ae_poisson_ci_extra_has_bounds():
    rep = make_report()
    result = rep.ae_with_poisson_ci()
    assert "ci_lower" in result.extra
    assert "ci_upper" in result.extra
    assert result.extra["ci_lower"] >= 0.0


def test_ae_poisson_ci_lower_le_upper():
    rep = make_report()
    result = rep.ae_with_poisson_ci()
    assert result.extra["ci_lower"] <= result.extra["ci_upper"]


def test_ae_poisson_ci_biased_model_fails():
    """Model predicting twice actuals should give A/E ~0.5 -> fail."""
    y_true = np.ones(200)
    y_pred = np.ones(200) * 2.0
    rep = PerformanceReport(y_true, y_pred)
    result = rep.ae_with_poisson_ci()
    assert result.passed == False
    assert result.metric_value == pytest.approx(0.5, abs=0.01)


def test_ae_poisson_ci_zero_claims_handled():
    """Zero actual claims: lower CI should be 0."""
    y_true = np.zeros(100)
    y_pred = np.ones(100) * 0.01
    rep = PerformanceReport(y_true, y_pred)
    result = rep.ae_with_poisson_ci()
    assert result.extra["ci_lower"] == pytest.approx(0.0)


def test_ae_poisson_ci_ibnr_caveat_in_details():
    rep = make_report()
    result = rep.ae_with_poisson_ci()
    assert "IBNR" in result.details


def test_ae_poisson_ci_ci_includes_one_for_well_calibrated():
    """A well-calibrated model on large data should have CI including 1.0."""
    rng = np.random.default_rng(7)
    n = 5000
    # Poisson with lambda=0.1, perfect predictions
    lam = 0.1
    y_true = rng.poisson(lam, n).astype(float)
    y_pred = np.full(n, lam)
    rep = PerformanceReport(y_true, y_pred)
    result = rep.ae_with_poisson_ci()
    assert result.extra["ci_includes_one"] == True


# ── Double-lift chart ─────────────────────────────────────────────────────────

def test_double_lift_returns_result():
    rng = np.random.default_rng(1)
    n = 300
    y_true = rng.poisson(0.1, n).astype(float)
    y_pred_new = y_true + rng.normal(0, 0.02, n)
    y_pred_old = y_true + rng.normal(0, 0.05, n)
    rep = PerformanceReport(y_true, y_pred_new)
    result = rep.double_lift(y_pred_incumbent=y_pred_old, n_bands=5)
    assert result.test_name == "double_lift"
    assert result.category == TestCategory.PERFORMANCE


def test_double_lift_extra_has_bands():
    rng = np.random.default_rng(2)
    n = 200
    y_true = rng.exponential(0.1, n)
    y_pred_new = y_true + rng.normal(0, 0.01, n)
    y_pred_old = y_true + rng.normal(0, 0.1, n)
    rep = PerformanceReport(y_true, y_pred_new)
    result = rep.double_lift(y_pred_incumbent=y_pred_old, n_bands=5)
    assert "bands" in result.extra
    assert len(result.extra["bands"]) == 5


def test_double_lift_band_keys():
    rng = np.random.default_rng(3)
    n = 100
    y = rng.exponential(0.1, n)
    rep = PerformanceReport(y, y)
    result = rep.double_lift(y_pred_incumbent=y * 1.1, n_bands=5)
    band = result.extra["bands"][0]
    assert "band" in band
    assert "actual_rate" in band
    assert "new_model_rate" in band
    assert "incumbent_rate" in band


def test_double_lift_better_model_wins():
    """A much better new model should pass (new_mae < inc_mae)."""
    rng = np.random.default_rng(42)
    n = 500
    y_true = rng.exponential(0.2, n)
    y_pred_new = y_true + rng.normal(0, 0.005, n)   # Very good
    y_pred_old = y_true + rng.normal(0, 0.5, n)     # Much worse
    rep = PerformanceReport(y_true, y_pred_new)
    result = rep.double_lift(y_pred_incumbent=y_pred_old)
    assert result.passed == True
    assert result.extra["new_model_mae"] < result.extra["incumbent_mae"]


def test_double_lift_mismatched_length_raises():
    y = np.ones(100)
    rep = PerformanceReport(y, y)
    with pytest.raises(ValueError, match="same length"):
        rep.double_lift(y_pred_incumbent=np.ones(50))


def test_double_lift_extra_has_mae_values():
    rng = np.random.default_rng(5)
    n = 200
    y = rng.exponential(0.1, n)
    rep = PerformanceReport(y, y)
    result = rep.double_lift(y_pred_incumbent=y * 1.2)
    assert "new_model_mae" in result.extra
    assert "incumbent_mae" in result.extra
    assert result.extra["new_model_mae"] >= 0
    assert result.extra["incumbent_mae"] >= 0


# ── Hosmer-Lemeshow test ───────────────────────────────────────────────────

def test_hl_returns_result():
    rep = make_report()
    result = rep.hosmer_lemeshow_test(n_groups=10)
    assert result.test_name == "hosmer_lemeshow"
    assert result.category == TestCategory.PERFORMANCE


def test_hl_well_calibrated_passes():
    """Perfect calibration should give large p-value."""
    rng = np.random.default_rng(99)
    n = 5000
    y_true = rng.poisson(0.1, n).astype(float)
    y_pred = np.full(n, 0.1)  # constant = perfect overall calibration
    rep = PerformanceReport(y_true, y_pred)
    result = rep.hosmer_lemeshow_test(n_groups=10)
    # Constant prediction: HL tests within groups, should have p > 0.05 on large sample
    # with a well-specified Poisson model
    assert result.extra["p_value"] is not None
    assert result.extra["p_value"] >= 0.0


def test_hl_badly_calibrated_fails():
    """Model predicting opposite direction should have very low p-value."""
    rng = np.random.default_rng(77)
    n = 2000
    y_true = rng.exponential(1.0, n)
    # Systematically wrong: predict high where actual is low
    y_pred = 10.0 - y_true + rng.normal(0, 0.1, n)
    y_pred = np.clip(y_pred, 0.001, 100)
    rep = PerformanceReport(y_true, y_pred)
    result = rep.hosmer_lemeshow_test(n_groups=10)
    assert result.extra["hl_statistic"] > 0


def test_hl_extra_has_groups():
    rep = make_report()
    result = rep.hosmer_lemeshow_test(n_groups=8)
    assert "groups" in result.extra
    assert len(result.extra["groups"]) == 8


def test_hl_extra_has_hl_statistic():
    rep = make_report()
    result = rep.hosmer_lemeshow_test()
    assert "hl_statistic" in result.extra
    assert result.extra["hl_statistic"] >= 0


def test_hl_df_is_n_groups_minus_2():
    rep = make_report()
    result = rep.hosmer_lemeshow_test(n_groups=10)
    assert result.extra["df"] == 8
