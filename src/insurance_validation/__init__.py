"""
insurance-validation: Model validation report generator for UK insurance pricing models.

Aligned with PRA SS1/23 model risk management principles, FCA Consumer Duty,
and FCA TR24/2 pricing governance requirements.

Quick start
-----------
    from datetime import date
    from insurance_validation import (
        ModelCard,
        DataQualityReport,
        PerformanceReport,
        DiscriminationReport,
        StabilityReport,
        ReportGenerator,
    )
    import polars as pl
    import numpy as np

    card = ModelCard(
        model_name="Motor TPPD Frequency",
        version="2.0.0",
        purpose="Estimate expected claim frequency for private motor policies.",
        intended_use="Underwriting pricing only. Not for reserving.",
        developer="Pricing Team",
        development_date=date(2024, 6, 1),
        limitations="Performance degrades for vehicles over 15 years old.",
        materiality_tier=2,
        approved_by=["Chief Actuary"],
        variables=["driver_age", "vehicle_age", "region"],
        target_variable="claim_count",
        model_type="GLM",
        distribution_family="Poisson",
    )

    dq = DataQualityReport(training_df)
    perf = PerformanceReport(y_true, y_pred, exposure=exposure)

    results = [
        dq.summary_statistics(),
        *dq.missing_value_analysis(),
        perf.gini_coefficient(),
        perf.actual_vs_expected(),
    ]

    gen = ReportGenerator(card, results)
    gen.write_html("validation_report.html")
"""
from .data_quality import DataQualityReport
from .discrimination import DiscriminationReport
from .model_card import ModelCard
from .performance import PerformanceReport
from .report import ReportGenerator
from .results import Severity, TestCategory, TestResult
from .stability import StabilityReport

__all__ = [
    "ModelCard",
    "DataQualityReport",
    "PerformanceReport",
    "DiscriminationReport",
    "StabilityReport",
    "ReportGenerator",
    "TestResult",
    "TestCategory",
    "Severity",
]

__version__ = "0.1.0"
