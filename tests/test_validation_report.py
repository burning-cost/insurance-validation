"""
Tests for ModelValidationReport high-level facade.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from insurance_validation import ModelCard, ModelValidationReport
from insurance_validation.results import RAGStatus


def make_card():
    return ModelCard(
        name="Motor Frequency v3.2",
        version="3.2.0",
        purpose="Predict claim frequency for UK motor portfolio.",
        methodology="CatBoost gradient boosting with Poisson objective",
        target="claim_count",
        features=["age", "vehicle_age", "area", "vehicle_group"],
        limitations=["No telematics data", "Limited fleet exposure"],
        owner="Pricing Team",
    )


def make_data(n: int = 500, seed: int = 42):
    rng = np.random.default_rng(seed)
    y_train = rng.poisson(0.1, n).astype(float)
    y_pred_train = np.clip(y_train + rng.normal(0, 0.03, n), 0.001, 10)
    y_val = rng.poisson(0.1, n).astype(float)
    y_pred_val = np.clip(y_val + rng.normal(0, 0.03, n), 0.001, 10)
    exposure = rng.uniform(0.5, 2.0, n)
    return y_train, y_pred_train, y_val, y_pred_val, exposure


# ── Construction ─────────────────────────────────────────────────────────────

def test_constructs_without_optional_args():
    y_train, y_pred_train, y_val, y_pred_val, exposure = make_data()
    report = ModelValidationReport(
        model_card=make_card(),
        y_val=y_val,
        y_pred_val=y_pred_val,
    )
    assert report is not None


def test_constructs_with_all_args():
    y_train, y_pred_train, y_val, y_pred_val, exposure = make_data()
    card = make_card()
    card.monitoring_frequency = "Quarterly"
    report = ModelValidationReport(
        model_card=card,
        y_val=y_val,
        y_pred_val=y_pred_val,
        exposure_val=exposure,
        y_train=y_train,
        y_pred_train=y_pred_train,
        exposure_train=exposure,
        monitoring_owner="Jane Smith",
        monitoring_triggers={"psi_score": 0.25, "ae_ratio": 0.15},
    )
    assert report is not None


# ── run() ─────────────────────────────────────────────────────────────────────

def test_run_returns_list_of_results():
    y_train, y_pred_train, y_val, y_pred_val, exposure = make_data()
    report = ModelValidationReport(
        model_card=make_card(),
        y_val=y_val,
        y_pred_val=y_pred_val,
        exposure_val=exposure,
        y_train=y_train,
        y_pred_train=y_pred_train,
    )
    results = report.run()
    assert isinstance(results, list)
    assert len(results) > 0


def test_run_includes_gini():
    y_train, y_pred_train, y_val, y_pred_val, exposure = make_data()
    report = ModelValidationReport(
        model_card=make_card(),
        y_val=y_val,
        y_pred_val=y_pred_val,
        exposure_val=exposure,
    )
    results = report.run()
    test_names = [r.test_name for r in results]
    assert "gini_coefficient" in test_names


def test_run_includes_gini_with_ci():
    y_train, y_pred_train, y_val, y_pred_val, exposure = make_data()
    report = ModelValidationReport(
        model_card=make_card(),
        y_val=y_val,
        y_pred_val=y_pred_val,
    )
    results = report.run()
    test_names = [r.test_name for r in results]
    assert "gini_with_ci" in test_names


def test_run_includes_ae_poisson_ci():
    y_train, y_pred_train, y_val, y_pred_val, exposure = make_data()
    report = ModelValidationReport(
        model_card=make_card(),
        y_val=y_val,
        y_pred_val=y_pred_val,
    )
    results = report.run()
    test_names = [r.test_name for r in results]
    assert "ae_poisson_ci" in test_names


def test_run_includes_hosmer_lemeshow():
    y_train, y_pred_train, y_val, y_pred_val, exposure = make_data()
    report = ModelValidationReport(
        model_card=make_card(),
        y_val=y_val,
        y_pred_val=y_pred_val,
    )
    results = report.run()
    test_names = [r.test_name for r in results]
    assert "hosmer_lemeshow" in test_names


def test_run_includes_lift_chart():
    y_train, y_pred_train, y_val, y_pred_val, exposure = make_data()
    report = ModelValidationReport(
        model_card=make_card(),
        y_val=y_val,
        y_pred_val=y_pred_val,
    )
    results = report.run()
    test_names = [r.test_name for r in results]
    assert "lift_chart" in test_names


def test_run_includes_psi_with_train_preds():
    y_train, y_pred_train, y_val, y_pred_val, exposure = make_data()
    report = ModelValidationReport(
        model_card=make_card(),
        y_val=y_val,
        y_pred_val=y_pred_val,
        y_train=y_train,
        y_pred_train=y_pred_train,
    )
    results = report.run()
    test_names = [r.test_name for r in results]
    assert "psi_score" in test_names


def test_run_includes_monitoring_plan():
    y_train, y_pred_train, y_val, y_pred_val, exposure = make_data()
    report = ModelValidationReport(
        model_card=make_card(),
        y_val=y_val,
        y_pred_val=y_pred_val,
        monitoring_owner="Risk Team",
    )
    results = report.run()
    test_names = [r.test_name for r in results]
    assert "monitoring_plan" in test_names


def test_run_with_double_lift():
    y_train, y_pred_train, y_val, y_pred_val, exposure = make_data()
    rng = np.random.default_rng(10)
    incumbent = y_pred_val + rng.normal(0, 0.05, len(y_pred_val))
    report = ModelValidationReport(
        model_card=make_card(),
        y_val=y_val,
        y_pred_val=y_pred_val,
        incumbent_pred_val=incumbent,
    )
    results = report.run()
    test_names = [r.test_name for r in results]
    assert "double_lift" in test_names


def test_run_with_x_val():
    y_train, y_pred_train, y_val, y_pred_val, exposure = make_data()
    n = len(y_val)
    rng = np.random.default_rng(11)
    X_val = pl.DataFrame({
        "age": rng.integers(18, 80, n).tolist(),
        "vehicle_age": rng.integers(0, 20, n).tolist(),
    })
    report = ModelValidationReport(
        model_card=make_card(),
        y_val=y_val,
        y_pred_val=y_pred_val,
        X_val=X_val,
    )
    results = report.run()
    test_names = [r.test_name for r in results]
    assert "summary_statistics" in test_names


def test_run_cached_same_results():
    y_train, y_pred_train, y_val, y_pred_val, exposure = make_data()
    report = ModelValidationReport(
        model_card=make_card(),
        y_val=y_val,
        y_pred_val=y_pred_val,
    )
    results1 = report.run()
    results2 = report.run()
    assert len(results1) == len(results2)


# ── RAG status ────────────────────────────────────────────────────────────────

def test_get_rag_status_is_valid():
    y_train, y_pred_train, y_val, y_pred_val, exposure = make_data()
    report = ModelValidationReport(
        model_card=make_card(),
        y_val=y_val,
        y_pred_val=y_pred_val,
    )
    status = report.get_rag_status()
    assert status in (RAGStatus.GREEN, RAGStatus.AMBER, RAGStatus.RED)


# ── generate() ───────────────────────────────────────────────────────────────

def test_generate_creates_html_file():
    y_train, y_pred_train, y_val, y_pred_val, exposure = make_data()
    report = ModelValidationReport(
        model_card=make_card(),
        y_val=y_val,
        y_pred_val=y_pred_val,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.html"
        path = report.generate(out)
        assert path.exists()
        content = path.read_text()
        assert "<!DOCTYPE html>" in content
        assert "Motor Frequency" in content


# ── to_json() ────────────────────────────────────────────────────────────────

def test_to_json_creates_valid_json():
    y_train, y_pred_train, y_val, y_pred_val, exposure = make_data()
    report = ModelValidationReport(
        model_card=make_card(),
        y_val=y_val,
        y_pred_val=y_pred_val,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "report.json"
        path = report.to_json(out)
        assert path.exists()
        data = json.loads(path.read_text())
        assert "run_id" in data
        assert "rag_status" in data
        assert "results" in data
        assert "model_card" in data
        assert "summary" in data


def test_to_dict_has_run_id():
    y_train, y_pred_train, y_val, y_pred_val, exposure = make_data()
    report = ModelValidationReport(
        model_card=make_card(),
        y_val=y_val,
        y_pred_val=y_pred_val,
    )
    d = report.to_dict()
    assert "run_id" in d
    import uuid
    uuid.UUID(d["run_id"])  # Should not raise


def test_monitoring_plan_passes_with_owner():
    y_train, y_pred_train, y_val, y_pred_val, exposure = make_data()
    report = ModelValidationReport(
        model_card=make_card(),
        y_val=y_val,
        y_pred_val=y_pred_val,
        monitoring_owner="Actuarial Risk",
        monitoring_triggers={"psi_score": 0.25},
    )
    results = report.run()
    mon = next((r for r in results if r.test_name == "monitoring_plan"), None)
    assert mon is not None
    assert mon.passed == True


def test_monitoring_plan_fails_without_owner():
    y_train, y_pred_train, y_val, y_pred_val, exposure = make_data()
    report = ModelValidationReport(
        model_card=make_card(),
        y_val=y_val,
        y_pred_val=y_pred_val,
        # No monitoring_owner
    )
    results = report.run()
    mon = next((r for r in results if r.test_name == "monitoring_plan"), None)
    assert mon is not None
    assert mon.passed == False


def test_extra_results_included():
    from insurance_validation.results import TestResult, TestCategory, Severity
    y_train, y_pred_train, y_val, y_pred_val, exposure = make_data()
    custom = TestResult(
        test_name="custom_test",
        category=TestCategory.PERFORMANCE,
        passed=True,
        details="Custom test.",
        severity=Severity.INFO,
    )
    report = ModelValidationReport(
        model_card=make_card(),
        y_val=y_val,
        y_pred_val=y_pred_val,
        extra_results=[custom],
    )
    results = report.run()
    test_names = [r.test_name for r in results]
    assert "custom_test" in test_names


def test_tenure_and_segment_cols_with_x_val():
    n = 400
    rng = np.random.default_rng(20)
    y_val = rng.poisson(0.1, n).astype(float)
    y_pred_val = np.full(n, 0.1)
    tenure = rng.integers(0, 6, n)
    segment = [f"Seg{i % 3}" for i in range(n)]
    X_val = pl.DataFrame({"tenure": tenure, "segment": segment})

    report = ModelValidationReport(
        model_card=make_card(),
        y_val=y_val,
        y_pred_val=y_pred_val,
        X_val=X_val,
        tenure_col="tenure",
        segment_col="segment",
    )
    results = report.run()
    test_names = [r.test_name for r in results]
    assert "renewal_cohort_ae" in test_names
    assert "subsegment_ae_segment" in test_names
