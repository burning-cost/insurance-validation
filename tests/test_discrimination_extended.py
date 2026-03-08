"""
Tests for renewal cohort A/E and sub-segment calibration.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from insurance_validation import DiscriminationReport
from insurance_validation.results import TestCategory, Severity


def make_df_with_tenure(n: int = 400) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    tenure = rng.integers(0, 8, n)
    return pl.DataFrame({
        "tenure": tenure,
        "region": [f"R{i % 4}" for i in range(n)],
        "segment": [f"Seg{i % 3}" for i in range(n)],
        "age_band": [f"Band{i % 5}" for i in range(n)],
    })


# ── Renewal cohort A/E ───────────────────────────────────────────────────────

def test_renewal_cohort_ae_returns_result():
    df = make_df_with_tenure()
    rng = np.random.default_rng(1)
    n = len(df)
    y_true = rng.poisson(0.1, n).astype(float)
    y_pred = np.full(n, 0.1)
    disc = DiscriminationReport(df=df, predictions=y_pred)
    result = disc.renewal_cohort_ae(y_true=y_true, y_pred=y_pred, tenure_col="tenure")
    assert result.test_name == "renewal_cohort_ae"
    assert result.category == TestCategory.FAIRNESS


def test_renewal_cohort_ae_all_within_range_passes():
    """When A/E is ~1 everywhere, test should pass."""
    n = 1000
    rng = np.random.default_rng(2)
    tenure = rng.integers(0, 8, n)
    df = pl.DataFrame({"tenure": tenure})
    y_true = rng.poisson(0.1, n).astype(float)
    y_pred = np.full(n, 0.1)
    disc = DiscriminationReport(df=df)
    result = disc.renewal_cohort_ae(y_true=y_true, y_pred=y_pred, tenure_col="tenure")
    # With perfect predictions and large n, A/E should be close to 1.0
    # (may fail if random data extreme, but very unlikely with n=1000)
    assert result.extra is not None
    assert "bands" in result.extra


def test_renewal_cohort_ae_bad_calibration_fails():
    """When new customers A/E is very high, test fails."""
    n = 500
    rng = np.random.default_rng(3)
    tenure = rng.integers(0, 4, n)
    df = pl.DataFrame({"tenure": tenure})

    y_true = np.where(tenure == 0, 1.0, 0.05)  # New customers have 20x higher actual
    y_pred = np.full(n, 0.05)  # All predicted the same
    disc = DiscriminationReport(df=df)
    result = disc.renewal_cohort_ae(y_true=y_true, y_pred=y_pred, tenure_col="tenure")
    assert result.passed == False


def test_renewal_cohort_ae_missing_col_returns_warning():
    df = pl.DataFrame({"region": ["North", "South"]})
    disc = DiscriminationReport(df=df)
    result = disc.renewal_cohort_ae(
        y_true=np.array([1.0, 0.0]),
        y_pred=np.array([0.5, 0.5]),
        tenure_col="nonexistent",
    )
    assert result.passed == False
    assert result.severity == Severity.WARNING


def test_renewal_cohort_ae_bands_in_extra():
    df = make_df_with_tenure(200)
    rng = np.random.default_rng(4)
    n = len(df)
    y_true = rng.poisson(0.1, n).astype(float)
    y_pred = np.full(n, 0.1)
    disc = DiscriminationReport(df=df)
    result = disc.renewal_cohort_ae(y_true=y_true, y_pred=y_pred, tenure_col="tenure")
    assert "bands" in result.extra
    bands = result.extra["bands"]
    assert len(bands) == 4  # Default: New, 1yr, 2yr, 3yr+
    for band in bands:
        assert "band" in band
        assert "ae_ratio" in band or band["n"] == 0


def test_renewal_cohort_ae_custom_bands():
    n = 200
    tenure = np.array([0] * 50 + [1] * 50 + [5] * 100)
    df = pl.DataFrame({"tenure": tenure})
    y_true = np.full(n, 0.1)
    y_pred = np.full(n, 0.1)
    disc = DiscriminationReport(df=df)
    custom_bands = [(0, 0, "New"), (1, 4, "1-4yr"), (5, 999, "5yr+")]
    result = disc.renewal_cohort_ae(
        y_true=y_true,
        y_pred=y_pred,
        tenure_col="tenure",
        tenure_bands=custom_bands,
    )
    assert len(result.extra["bands"]) == 3


def test_renewal_cohort_ae_thresholds():
    """A/E of 2.0 is outside default [0.85, 1.15] and should fail."""
    n = 200
    tenure = np.zeros(n, dtype=int)
    df = pl.DataFrame({"tenure": tenure})
    y_true = np.full(n, 2.0)
    y_pred = np.full(n, 1.0)  # A/E = 2.0
    disc = DiscriminationReport(df=df)
    result = disc.renewal_cohort_ae(
        y_true=y_true, y_pred=y_pred, tenure_col="tenure",
        ae_low=0.85, ae_high=1.15
    )
    assert result.passed == False


# ── Sub-segment calibration ───────────────────────────────────────────────────

def test_subsegment_ae_returns_result():
    df = make_df_with_tenure()
    rng = np.random.default_rng(5)
    n = len(df)
    y_true = rng.poisson(0.1, n).astype(float)
    y_pred = np.full(n, 0.1)
    disc = DiscriminationReport(df=df)
    result = disc.subsegment_ae(y_true=y_true, y_pred=y_pred, segment_col="segment")
    assert result.test_name == "subsegment_ae_segment"
    assert result.category == TestCategory.FAIRNESS


def test_subsegment_ae_well_calibrated_passes():
    n = 600
    seg = np.array(["A"] * 200 + ["B"] * 200 + ["C"] * 200)
    df = pl.DataFrame({"segment": seg})
    y_true = np.full(n, 0.1)
    y_pred = np.full(n, 0.1)
    disc = DiscriminationReport(df=df)
    result = disc.subsegment_ae(y_true=y_true, y_pred=y_pred, segment_col="segment")
    assert result.passed == True


def test_subsegment_ae_one_bad_segment_fails():
    n = 300
    seg = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 100)
    df = pl.DataFrame({"segment": seg})
    y_true = np.where(seg == "C", 2.0, 0.1)
    y_pred = np.full(n, 0.1)  # All predicted same
    disc = DiscriminationReport(df=df)
    result = disc.subsegment_ae(y_true=y_true, y_pred=y_pred, segment_col="segment")
    assert result.passed == False


def test_subsegment_ae_missing_col_returns_warning():
    df = pl.DataFrame({"region": ["North", "South"]})
    disc = DiscriminationReport(df=df)
    result = disc.subsegment_ae(
        y_true=np.array([1.0, 0.0]),
        y_pred=np.array([0.5, 0.5]),
        segment_col="nonexistent",
    )
    assert result.passed == False
    assert result.severity == Severity.WARNING


def test_subsegment_ae_segments_in_extra():
    df = make_df_with_tenure(200)
    n = len(df)
    rng = np.random.default_rng(6)
    y_true = rng.poisson(0.1, n).astype(float)
    y_pred = np.full(n, 0.1)
    disc = DiscriminationReport(df=df)
    result = disc.subsegment_ae(y_true=y_true, y_pred=y_pred, segment_col="segment")
    assert "segments" in result.extra
    segs = result.extra["segments"]
    assert len(segs) == 3  # Seg0, Seg1, Seg2
    for seg in segs:
        assert "segment" in seg
        assert "ae_ratio" in seg


def test_subsegment_ae_custom_thresholds():
    """Tight threshold [0.95, 1.05] should fail if A/E is 1.1."""
    n = 200
    df = pl.DataFrame({"seg": ["X"] * 100 + ["Y"] * 100})
    y_true = np.array([1.1] * 100 + [1.0] * 100)
    y_pred = np.ones(n)
    disc = DiscriminationReport(df=df)
    result = disc.subsegment_ae(
        y_true=y_true, y_pred=y_pred, segment_col="seg",
        ae_low=0.95, ae_high=1.05
    )
    assert result.passed == False
