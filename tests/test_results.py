"""Tests for results module: TestResult, RAGStatus, compute_rag_status."""
from insurance_validation.results import (
    TestResult, TestCategory, Severity, RAGStatus, compute_rag_status
)


def make_result(passed=True, severity=Severity.INFO, category=TestCategory.PERFORMANCE):
    return TestResult(
        test_name="test",
        category=category,
        passed=passed,
        severity=severity,
        details="test details",
    )


def test_testresult_to_dict():
    r = make_result()
    d = r.to_dict()
    assert d["test_name"] == "test"
    assert d["category"] == "performance"
    assert d["passed"] == True
    assert d["severity"] == "info"


def test_testresult_to_dict_all_fields():
    r = TestResult(
        test_name="gini",
        category=TestCategory.PERFORMANCE,
        passed=True,
        metric_value=0.42,
        details="Good",
        severity=Severity.INFO,
        extra={"key": "value"},
    )
    d = r.to_dict()
    assert d["metric_value"] == 0.42
    assert d["extra"]["key"] == "value"


def test_severity_enum_values():
    assert Severity.INFO == "info"
    assert Severity.WARNING == "warning"
    assert Severity.CRITICAL == "critical"


def test_rag_status_enum_values():
    assert RAGStatus.GREEN == "green"
    assert RAGStatus.AMBER == "amber"
    assert RAGStatus.RED == "red"


def test_test_category_fairness():
    assert TestCategory.FAIRNESS == "fairness"
    assert TestCategory.MONITORING == "monitoring"


def test_compute_rag_green():
    results = [
        make_result(passed=True, severity=Severity.INFO),
        make_result(passed=True, severity=Severity.INFO),
    ]
    assert compute_rag_status(results) == RAGStatus.GREEN


def test_compute_rag_amber_warning_failure():
    results = [
        make_result(passed=True, severity=Severity.INFO),
        make_result(passed=False, severity=Severity.WARNING),
    ]
    assert compute_rag_status(results) == RAGStatus.AMBER


def test_compute_rag_red_critical_failure():
    results = [
        make_result(passed=False, severity=Severity.CRITICAL),
    ]
    assert compute_rag_status(results) == RAGStatus.RED


def test_compute_rag_red_overrides_amber():
    results = [
        make_result(passed=False, severity=Severity.WARNING),
        make_result(passed=False, severity=Severity.CRITICAL),
    ]
    assert compute_rag_status(results) == RAGStatus.RED


def test_compute_rag_info_failure_is_green():
    """A failed INFO result does not trigger AMBER."""
    results = [
        make_result(passed=False, severity=Severity.INFO),
    ]
    assert compute_rag_status(results) == RAGStatus.GREEN


def test_compute_rag_empty():
    assert compute_rag_status([]) == RAGStatus.GREEN
