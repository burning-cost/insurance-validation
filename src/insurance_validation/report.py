"""
HTML validation report generator.

Takes a ModelCard and a list of TestResult objects, renders them into a
self-contained HTML report, and optionally writes a JSON sidecar for
audit trail ingestion.

The HTML is completely self-contained: no external CSS frameworks, no
CDN dependencies, no JavaScript. A single file you can email, store in
SharePoint, or attach to a Jira ticket.

Usage
-----
    from insurance_validation import ModelCard, ReportGenerator
    from insurance_validation.results import TestResult

    card = ModelCard(...)
    results: list[TestResult] = [...]

    gen = ReportGenerator(card, results)
    gen.write_html("validation_report.html")
    gen.write_json("validation_report.json")  # audit trail
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from .model_card import ModelCard
from .results import TestResult


class ReportGenerator:
    """
    Generate a validation report from a ModelCard and test results.

    Parameters
    ----------
    card:
        Completed ModelCard for the model being validated.
    results:
        List of TestResult objects from any combination of
        DataQualityReport, PerformanceReport, DiscriminationReport,
        StabilityReport, or custom tests.
    generated_date:
        Date to stamp on the report. Defaults to today.
    """

    def __init__(
        self,
        card: ModelCard,
        results: list[TestResult],
        generated_date: date | None = None,
    ) -> None:
        self._card = card
        self._results = results
        self._generated_date = generated_date or date.today()

        self._env = Environment(
            loader=PackageLoader("insurance_validation", "templates"),
            autoescape=select_autoescape(["html", "j2"]),
        )

    def render_html(self) -> str:
        """
        Render the validation report to an HTML string.

        Returns
        -------
        str
            Complete, self-contained HTML document.
        """
        template = self._env.get_template("report.html.j2")

        # Convert results to dicts with string category/severity for template
        result_dicts = []
        for r in self._results:
            d = r.to_dict()
            # Jinja2 filter uses string comparison
            d["category"] = r.category.value
            d["severity"] = r.severity.value
            # Keep original passed bool
            result_dicts.append(d)

        return template.render(
            card=self._card,
            results=result_dicts,
            generated_date=str(self._generated_date),
        )

    def write_html(self, path: str | Path) -> Path:
        """
        Write the HTML report to a file.

        Parameters
        ----------
        path:
            Output file path. Parent directories must exist.

        Returns
        -------
        Path
            Resolved path to the written file.
        """
        out = Path(path).resolve()
        out.write_text(self.render_html(), encoding="utf-8")
        return out

    def to_dict(self) -> dict:
        """
        Serialise the full report to a plain dict for JSON export.

        Returns
        -------
        dict
        """
        return {
            "model_card": self._card.model_dump(mode="json"),
            "generated_date": str(self._generated_date),
            "results": [r.to_dict() for r in self._results],
            "summary": {
                "total_tests": len(self._results),
                "passed": sum(1 for r in self._results if r.passed),
                "failed": sum(1 for r in self._results if not r.passed),
                "critical": sum(
                    1 for r in self._results
                    if not r.passed and r.severity.value == "critical"
                ),
                "warnings": sum(
                    1 for r in self._results
                    if not r.passed and r.severity.value == "warning"
                ),
            },
        }

    def write_json(self, path: str | Path) -> Path:
        """
        Write a JSON sidecar for audit trail ingestion.

        The JSON contains the full model card, all test results, and a
        summary. Suitable for ingestion into a model risk management
        system or storage alongside the HTML report.

        Parameters
        ----------
        path:
            Output file path.

        Returns
        -------
        Path
        """
        out = Path(path).resolve()
        out.write_text(
            json.dumps(self.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        return out
