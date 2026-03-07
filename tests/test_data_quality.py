"""Tests for DataQualityReport."""
import polars as pl
import pytest
from insurance_validation import DataQualityReport
from insurance_validation.results import TestCategory, Severity


def make_clean_df():
    return pl.DataFrame({
        "driver_age": [25, 35, 45, 55, 30],
        "vehicle_age": [2, 5, 1, 8, 3],
        "region": ["North", "South", "East", "West", "North"],
        "claim_count": [0, 1, 0, 2, 0],
    })


def make_df_with_nulls():
    return pl.DataFrame({
        "driver_age": [25, None, 45, None, 30],
        "vehicle_age": [2, 5, 1, 8, 3],
        "region": ["North", None, "East", None, None],
    })


def test_summary_statistics_always_passes():
    df = make_clean_df()
    dqr = DataQualityReport(df)
    result = dqr.summary_statistics()
    assert result.passed is True
    assert result.category == TestCategory.DATA_QUALITY
    assert result.metric_value == 5.0  # 5 rows


def test_summary_statistics_details_contains_shape():
    df = make_clean_df()
    dqr = DataQualityReport(df)
    result = dqr.summary_statistics()
    assert "5" in result.details
    assert "4" in result.details  # 4 columns


def test_missing_value_analysis_clean_data():
    df = make_clean_df()
    dqr = DataQualityReport(df)
    results = dqr.missing_value_analysis()
    # All columns should pass
    assert all(r.passed for r in results)
    assert all(r.metric_value == 0.0 for r in results)


def test_missing_value_analysis_detects_nulls():
    df = make_df_with_nulls()
    dqr = DataQualityReport(df)
    results = dqr.missing_value_analysis(threshold=0.05)
    # driver_age: 2/5 = 40% missing - should fail
    age_result = next(r for r in results if "driver_age" in r.test_name)
    assert age_result.passed is False
    assert age_result.metric_value == pytest.approx(0.4)


def test_missing_value_analysis_returns_one_per_column():
    df = make_clean_df()
    dqr = DataQualityReport(df)
    results = dqr.missing_value_analysis()
    assert len(results) == len(df.columns)


def test_missing_value_critical_above_50pct():
    df = pl.DataFrame({"x": [1, None, None, None, None, None]})
    dqr = DataQualityReport(df)
    results = dqr.missing_value_analysis(threshold=0.05)
    assert results[0].severity == Severity.CRITICAL


def test_outlier_detection_no_outliers():
    df = pl.DataFrame({"x": list(range(1, 21))})  # 1-20
    dqr = DataQualityReport(df)
    results = dqr.outlier_detection(method="iqr", iqr_multiplier=3.0)
    assert len(results) == 1
    assert results[0].passed is True


def test_outlier_detection_finds_extreme_value():
    values = list(range(1, 20)) + [10000]
    df = pl.DataFrame({"x": values})
    dqr = DataQualityReport(df)
    results = dqr.outlier_detection(method="iqr", iqr_multiplier=3.0)
    assert results[0].passed is False
    assert results[0].metric_value > 0


def test_outlier_detection_zscore():
    values = [0.0] * 98 + [100.0, -100.0]
    df = pl.DataFrame({"x": values})
    dqr = DataQualityReport(df)
    results = dqr.outlier_detection(method="zscore", zscore_threshold=4.0)
    assert results[0].passed is False


def test_outlier_detection_skips_non_numeric():
    df = pl.DataFrame({"region": ["North", "South", "East"]})
    dqr = DataQualityReport(df)
    results = dqr.outlier_detection()
    # No numeric columns - should return empty list
    assert len(results) == 0


def test_cardinality_check_passes():
    df = pl.DataFrame({"region": ["A", "B", "C", "A", "B"]})
    dqr = DataQualityReport(df)
    results = dqr.cardinality_check(max_categories=10)
    assert all(r.passed for r in results)


def test_cardinality_check_fails_high_cardinality():
    import string
    # 100 unique categories
    cats = [str(i) for i in range(100)]
    df = pl.DataFrame({"policy_id": cats})
    dqr = DataQualityReport(df)
    results = dqr.cardinality_check(max_categories=50)
    assert any(not r.passed for r in results)


def test_cardinality_check_skips_numeric():
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    dqr = DataQualityReport(df)
    results = dqr.cardinality_check()
    # No categorical columns
    assert len(results) == 0


def test_all_results_have_correct_category():
    df = make_clean_df()
    dqr = DataQualityReport(df)
    all_results = [
        dqr.summary_statistics(),
        *dqr.missing_value_analysis(),
        *dqr.outlier_detection(),
        *dqr.cardinality_check(),
    ]
    for r in all_results:
        assert r.category == TestCategory.DATA_QUALITY
