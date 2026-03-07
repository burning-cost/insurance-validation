"""Tests for ReportGenerator HTML and JSON output."""
import json
import tempfile
from datetime import date
from pathlib import Path

import pytest
from insurance_validation import ModelCard, ReportGenerator
from insurance_validation.results import Severity, TestCategory, TestResult


def make_card():
    return ModelCard(
        model_name="Test Model",
        version="1.0.0",
        purpose="Test model for validation report generation.",
        intended_use="Unit testing only.",
        developer="Test Team",
        development_date=date(2024, 1, 1),
        limitations="This is a test model with no real-world limitations.",
        materiality_tier=3,
        approved_by=["Test Approver"],
        variables=["x", "y"],
        target_variable="z",
        model_type="GLM",
        distribution_family="Poisson",
    )


def make_results():
    return [
        TestResult(
            test_name="summary_statistics",
            category=TestCategory.DATA_QUALITY,
            passed=True,
            metric_value=1000.0,
            details="Dataset has 1,000 rows.",
            severity=Severity.INFO,
        ),
        TestResult(
            test_name="gini_coefficient",
            category=TestCategory.PERFORMANCE,
            passed=True,
            metric_value=0.45,
            details="Gini = 0.45. Good discriminatory power.",
            severity=Severity.INFO,
        ),
        TestResult(
            test_name="psi_score",
            category=TestCategory.STABILITY,
            passed=False,
            metric_value=0.31,
            details="PSI = 0.31 - significant shift.",
            severity=Severity.CRITICAL,
        ),
    ]


def test_render_html_returns_string():
    card = make_card()
    results = make_results()
    gen = ReportGenerator(card, results)
    html = gen.render_html()
    assert isinstance(html, str)
    assert len(html) > 100


def test_html_contains_model_name():
    card = make_card()
    gen = ReportGenerator(card, [])
    html = gen.render_html()
    assert "Test Model" in html


def test_html_contains_version():
    card = make_card()
    gen = ReportGenerator(card, [])
    html = gen.render_html()
    assert "1.0.0" in html


def test_html_contains_test_names():
    card = make_card()
    results = make_results()
    gen = ReportGenerator(card, results)
    html = gen.render_html()
    assert "gini_coefficient" in html
    assert "psi_score" in html
    assert "summary_statistics" in html


def test_html_shows_fail_status_when_critical():
    card = make_card()
    results = make_results()  # contains a CRITICAL fail
    gen = ReportGenerator(card, results)
    html = gen.render_html()
    assert "FAIL" in html


def test_html_shows_pass_when_no_failures():
    card = make_card()
    results = [
        TestResult(
            test_name="gini_coefficient",
            category=TestCategory.PERFORMANCE,
            passed=True,
            metric_value=0.4,
            details="Passes.",
            severity=Severity.INFO,
        )
    ]
    gen = ReportGenerator(card, results)
    html = gen.render_html()
    assert "PASS" in html


def test_html_is_valid_html():
    card = make_card()
    gen = ReportGenerator(card, make_results())
    html = gen.render_html()
    assert "<!DOCTYPE html>" in html
    assert "<html" in html
    assert "</html>" in html


def test_write_html_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "report.html"
        card = make_card()
        gen = ReportGenerator(card, make_results())
        result_path = gen.write_html(out_path)
        assert result_path.exists()
        assert result_path.stat().st_size > 0


def test_to_dict_structure():
    card = make_card()
    results = make_results()
    gen = ReportGenerator(card, results)
    d = gen.to_dict()
    assert "model_card" in d
    assert "results" in d
    assert "summary" in d
    assert "generated_date" in d
    assert d["summary"]["total_tests"] == 3
    assert d["summary"]["passed"] == 2
    assert d["summary"]["failed"] == 1


def test_write_json_creates_valid_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "report.json"
        card = make_card()
        gen = ReportGenerator(card, make_results())
        gen.write_json(out_path)
        with open(out_path) as f:
            data = json.load(f)
        assert data["model_card"]["model_name"] == "Test Model"
        assert len(data["results"]) == 3


def test_empty_results_renders():
    card = make_card()
    gen = ReportGenerator(card, [])
    html = gen.render_html()
    assert "Test Model" in html
    assert "Total Tests" in html


def test_generated_date_default_is_today():
    card = make_card()
    gen = ReportGenerator(card, [])
    assert gen._generated_date == date.today()


def test_generated_date_custom():
    card = make_card()
    gen = ReportGenerator(card, [], generated_date=date(2025, 1, 15))
    assert gen._generated_date == date(2025, 1, 15)
    html = gen.render_html()
    assert "2025-01-15" in html
