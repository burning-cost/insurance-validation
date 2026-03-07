"""
TestResult dataclass and enums for insurance model validation.

Every test in this library returns a TestResult. This keeps reporting
consistent regardless of which module produced the result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """Severity level for a validation finding."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class TestCategory(str, Enum):
    """Which section of the validation report this result belongs to."""

    DATA_QUALITY = "data_quality"
    PERFORMANCE = "performance"
    DISCRIMINATION = "discrimination"
    STABILITY = "stability"
    ASSUMPTIONS = "assumptions"


@dataclass
class TestResult:
    """
    Structured result from a single validation test.

    Attributes
    ----------
    test_name:
        Short identifier, e.g. ``"missing_values"`` or ``"gini_coefficient"``.
    category:
        Which report section this result belongs to.
    passed:
        True if the test passed its threshold or produced no finding.
    metric_value:
        Primary numeric output, if the test produces one. None for
        pass/fail checks with no numeric summary.
    details:
        Human-readable explanation of what was tested and what the
        result means. Written for a reviewer who was not present when
        the model was built.
    severity:
        Severity of a failure. INFO results are always informational
        regardless of the ``passed`` flag.
    extra:
        Additional structured data (e.g. per-band lift table). Not
        included in summary tables but available for detailed renders.
    """

    test_name: str
    category: TestCategory
    passed: bool
    metric_value: float | None = None
    details: str = ""
    severity: Severity = Severity.INFO
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON export."""
        return {
            "test_name": self.test_name,
            "category": self.category.value,
            "passed": self.passed,
            "metric_value": self.metric_value,
            "details": self.details,
            "severity": self.severity.value,
            "extra": self.extra,
        }
