"""
insurance-validation: PRA SS1/23 compliant model validation report generator.

Aligned with PRA SS1/23 model risk management principles, FCA Consumer Duty,
and FCA TR24/2 pricing governance requirements.

Quick start (high-level API)
-----------------------------
    import numpy as np
    from insurance_validation import ModelValidationReport, ModelCard

    card = ModelCard(
        name="Motor Frequency v3.2",
        version="3.2.0",
        purpose="Predict claim frequency for UK motor portfolio",
        methodology="CatBoost gradient boosting with Poisson objective",
        target="claim_count",
        features=["age", "vehicle_age", "area", "vehicle_group"],
        limitations=["No telematics data", "Limited fleet exposure"],
        owner="Pricing Team",
    )

    report = ModelValidationReport(
        model_card=card,
        y_val=y_val,
        y_pred_val=y_pred_val,
        exposure_val=exposure_val,
        y_train=y_train,
        y_pred_train=y_pred_train,
    )

    report.generate("validation_report.html")
    report.to_json("validation_report.json")

Lower-level API
---------------
    from insurance_validation import (
        ModelCard,
        DataQualityReport,
        PerformanceReport,
        DiscriminationReport,
        StabilityReport,
        ReportGenerator,
    )

    card = ModelCard(...)
    perf = PerformanceReport(y_true, y_pred, exposure=exposure)
    results = [
        perf.gini_coefficient(),
        perf.gini_with_ci(),
        perf.ae_with_poisson_ci(),
        *perf.lift_chart(n_bands=10),
        perf.double_lift(y_pred_incumbent=old_preds),
        perf.hosmer_lemeshow_test(),
    ]
    gen = ReportGenerator(card, results)
    gen.write_html("report.html")
"""
from .data_quality import DataQualityReport
from .discrimination import DiscriminationReport
from .model_card import ModelCard
from .performance import PerformanceReport
from .report import ReportGenerator
from .results import RAGStatus, Severity, TestCategory, TestResult
from .stability import StabilityReport
from .validation_report import ModelValidationReport

__all__ = [
    # High-level facade
    "ModelValidationReport",
    # Lower-level components
    "ModelCard",
    "DataQualityReport",
    "PerformanceReport",
    "DiscriminationReport",
    "StabilityReport",
    "ReportGenerator",
    # Result types
    "TestResult",
    "TestCategory",
    "Severity",
    "RAGStatus",
]

__version__ = "0.2.0"
