"""
ModelValidationReport: high-level facade for PRA SS1/23 compliant validation.

This is the primary entry point for most users. It accepts your model,
training data, and validation data, runs all standard checks, and produces
the HTML + JSON report in one call.

The lower-level classes (PerformanceReport, DataQualityReport, etc.) remain
available for custom workflows where you need to add or replace individual
tests.

Usage
-----
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
        y_train=y_train,
        y_pred_train=y_pred_train,
        y_val=y_val,
        y_pred_val=y_pred_val,
        exposure_train=exposure_train,
        exposure_val=exposure_val,
        X_train=X_train,
        X_val=X_val,
        model_card=card,
    )

    report.generate("validation_report.html")
    report.to_json("validation_report.json")

Design notes
------------
- The facade runs a fixed set of "standard" tests and allows extending
  with custom TestResult lists.
- Shap and insurance-fairness integrations are optional extras. Missing
  packages produce an informational TestResult, not an error.
- Every numeric output is deterministic given the same inputs and
  random_state. Bootstrap CI uses random_state=42 by default.
"""
from __future__ import annotations

import json
import uuid
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from .data_quality import DataQualityReport
from .discrimination import DiscriminationReport
from .model_card import ModelCard
from .performance import PerformanceReport
from .report import ReportGenerator
from .results import RAGStatus, Severity, TestCategory, TestResult, compute_rag_status
from .stability import StabilityReport

try:
    import polars as pl
    _POLARS = True
except ImportError:
    _POLARS = False


class ModelValidationReport:
    """
    PRA SS1/23 compliant model validation report generator.

    Runs all standard validation checks and produces an HTML report plus
    a JSON sidecar. The HTML is completely self-contained (no CDN, no JS).

    Parameters
    ----------
    model_card:
        Completed ModelCard for the model being validated.
    y_val:
        Validation set observed outcomes.
    y_pred_val:
        Model predictions on the validation set.
    exposure_val:
        Exposure measure for the validation set (policy years). Optional
        but strongly recommended for frequency models.
    y_train:
        Training set observed outcomes. Required for PSI and stability
        checks.
    y_pred_train:
        Model predictions on the training set.
    exposure_train:
        Exposure measure for the training set.
    X_train:
        Training features as a Polars DataFrame. Optional. Used for data
        quality and feature drift checks.
    X_val:
        Validation features as a Polars DataFrame. Optional.
    incumbent_pred_val:
        Predictions from the incumbent model on the validation set.
        Required for double-lift chart. Optional.
    fairness_group_col:
        Column name in X_val for fairness (disparate impact) analysis.
    tenure_col:
        Column name in X_val containing years as customer (for renewal
        cohort A/E test).
    segment_col:
        Column name in X_val for sub-segment calibration.
    monitoring_owner:
        Named owner for the monitoring plan section. Required to complete
        the monitoring plan.
    monitoring_triggers:
        Dict of {metric_name: threshold} for monitoring alerts.
    extra_results:
        Additional TestResult objects to include in the report.
    random_state:
        Seed for reproducible bootstrap CIs.
    """

    def __init__(
        self,
        model_card: ModelCard,
        y_val: np.ndarray | list,
        y_pred_val: np.ndarray | list,
        exposure_val: np.ndarray | list | None = None,
        y_train: np.ndarray | list | None = None,
        y_pred_train: np.ndarray | list | None = None,
        exposure_train: np.ndarray | list | None = None,
        X_train: Any | None = None,
        X_val: Any | None = None,
        incumbent_pred_val: np.ndarray | list | None = None,
        fairness_group_col: str | None = None,
        tenure_col: str | None = None,
        segment_col: str | None = None,
        monitoring_owner: str | None = None,
        monitoring_triggers: dict[str, float] | None = None,
        extra_results: list[TestResult] | None = None,
        random_state: int = 42,
    ) -> None:
        self._card = model_card
        self._y_val = np.asarray(y_val, dtype=float)
        self._y_pred_val = np.asarray(y_pred_val, dtype=float)
        self._exposure_val = np.asarray(exposure_val, dtype=float) if exposure_val is not None else None
        self._y_train = np.asarray(y_train, dtype=float) if y_train is not None else None
        self._y_pred_train = np.asarray(y_pred_train, dtype=float) if y_pred_train is not None else None
        self._exposure_train = np.asarray(exposure_train, dtype=float) if exposure_train is not None else None
        self._X_train = X_train
        self._X_val = X_val
        self._incumbent_pred_val = (
            np.asarray(incumbent_pred_val, dtype=float)
            if incumbent_pred_val is not None
            else None
        )
        self._fairness_group_col = fairness_group_col
        self._tenure_col = tenure_col
        self._segment_col = segment_col
        self._monitoring_owner = monitoring_owner or getattr(model_card, "monitoring_owner", None)
        self._monitoring_triggers = monitoring_triggers or getattr(model_card, "monitoring_triggers", None)
        self._extra_results = extra_results or []
        self._random_state = random_state
        self._run_id = str(uuid.uuid4())
        self._generated_date = date.today()

        self._results: list[TestResult] | None = None

    def run(self) -> list[TestResult]:
        """
        Execute all validation tests and return the full results list.

        Results are cached. Call ``run()`` again if you change inputs.
        """
        results: list[TestResult] = []

        # ── Performance validation ───────────────────────────────────────
        model_name = self._card.get_effective_model_name()
        perf = PerformanceReport(
            y_true=self._y_val,
            y_pred=self._y_pred_val,
            exposure=self._exposure_val,
            model_name=model_name,
        )

        results.append(perf.gini_coefficient())
        results.append(perf.gini_with_ci(random_state=self._random_state))
        results.extend(perf.lift_chart(n_bands=10))
        results.append(perf.actual_vs_expected(n_bands=10))
        results.append(perf.ae_with_poisson_ci())
        results.append(perf.hosmer_lemeshow_test())
        results.append(perf.lorenz_curve())
        results.append(perf.calibration_plot_data())

        # Double-lift if incumbent provided
        if self._incumbent_pred_val is not None:
            results.append(
                perf.double_lift(
                    y_pred_incumbent=self._incumbent_pred_val,
                    n_bands=10,
                )
            )

        # ── Data quality ─────────────────────────────────────────────────
        if self._X_val is not None and _POLARS:
            import polars as pl
            if isinstance(self._X_val, pl.DataFrame):
                dq_val = DataQualityReport(self._X_val, dataset_name="validation")
                results.append(dq_val.summary_statistics())
                results.extend(dq_val.missing_value_analysis())
                results.extend(dq_val.outlier_detection())
                results.extend(dq_val.cardinality_check())

        if self._X_train is not None and _POLARS:
            import polars as pl
            if isinstance(self._X_train, pl.DataFrame):
                dq_train = DataQualityReport(self._X_train, dataset_name="training")
                results.append(dq_train.summary_statistics())

        # ── Stability: PSI on predictions ─────────────────────────────
        if self._y_pred_train is not None:
            stab = StabilityReport()
            results.append(
                stab.psi(
                    reference=self._y_pred_train,
                    current=self._y_pred_val,
                    n_bins=10,
                    label="score",
                )
            )

        # Feature drift
        if (
            self._X_train is not None
            and self._X_val is not None
            and _POLARS
        ):
            import polars as pl
            if isinstance(self._X_train, pl.DataFrame) and isinstance(self._X_val, pl.DataFrame):
                common_features = [
                    c for c in self._X_train.columns
                    if c in self._X_val.columns
                ]
                if common_features:
                    stab = StabilityReport()
                    results.extend(
                        stab.feature_drift(
                            reference_df=self._X_train,
                            current_df=self._X_val,
                            features=common_features[:10],  # Cap to avoid excessive output
                        )
                    )

        # ── Fairness / discrimination ───────────────────────────────────
        if self._X_val is not None and _POLARS:
            import polars as pl
            if isinstance(self._X_val, pl.DataFrame):
                disc = DiscriminationReport(df=self._X_val, predictions=self._y_pred_val)

                if self._fairness_group_col and self._fairness_group_col in self._X_val.columns:
                    results.append(
                        disc.disparate_impact_ratio(group_col=self._fairness_group_col)
                    )

                if self._tenure_col and self._tenure_col in self._X_val.columns:
                    results.append(
                        disc.renewal_cohort_ae(
                            y_true=self._y_val,
                            y_pred=self._y_pred_val,
                            tenure_col=self._tenure_col,
                        )
                    )

                if self._segment_col and self._segment_col in self._X_val.columns:
                    results.append(
                        disc.subsegment_ae(
                            y_true=self._y_val,
                            y_pred=self._y_pred_val,
                            segment_col=self._segment_col,
                        )
                    )

        # ── Monitoring plan ────────────────────────────────────────────
        results.append(self._monitoring_plan_result())

        # ── Optional integrations ──────────────────────────────────────
        results.extend(self._try_shap())
        results.extend(self._try_insurance_fairness())

        # ── User-supplied extras ───────────────────────────────────────
        results.extend(self._extra_results)

        self._results = results
        return results

    def _monitoring_plan_result(self) -> TestResult:
        """Produce a monitoring plan TestResult."""
        owner = self._monitoring_owner
        triggers = self._monitoring_triggers
        freq = self._card.monitoring_frequency

        has_owner = bool(owner)
        has_triggers = bool(triggers)
        passed = has_owner  # Owner is the minimum requirement

        if has_owner and has_triggers:
            trigger_desc = "; ".join(f"{k} > {v}" for k, v in triggers.items())
            details = (
                f"Monitoring plan: Owner = {owner}. "
                f"Review frequency: {freq or 'Not specified'}. "
                f"Alert triggers: {trigger_desc}."
            )
        elif has_owner:
            details = (
                f"Monitoring plan: Owner = {owner}. "
                f"Review frequency: {freq or 'Not specified'}. "
                "No alert triggers defined - consider adding PSI and A/E thresholds."
            )
        else:
            details = (
                "Monitoring plan incomplete: no named owner. "
                "PRA SS1/23 Principle 5 requires a named individual responsible for "
                "ongoing monitoring. Assign an owner before production sign-off."
            )

        return TestResult(
            test_name="monitoring_plan",
            category=TestCategory.MONITORING,
            passed=passed,
            metric_value=None,
            details=details,
            severity=Severity.INFO if passed else Severity.WARNING,
            extra={
                "owner": owner,
                "triggers": triggers,
                "review_frequency": freq,
            },
        )

    def _try_shap(self) -> list[TestResult]:
        """Attempt shap-relativities integration. Graceful if not installed."""
        try:
            import shap_relativities  # noqa: F401
            return []  # placeholder - would compute SHAP here
        except ImportError:
            return [
                TestResult(
                    test_name="shap_feature_importance",
                    category=TestCategory.PERFORMANCE,
                    passed=True,
                    details=(
                        "SHAP feature importance not available. "
                        "Install with: pip install insurance-validation[shap]"
                    ),
                    severity=Severity.INFO,
                )
            ]

    def _try_insurance_fairness(self) -> list[TestResult]:
        """Attempt insurance-fairness integration. Graceful if not installed."""
        try:
            import insurance_fairness  # noqa: F401
            return []
        except ImportError:
            return []  # Silently skip - insurance-fairness is a newer optional dep

    def get_results(self) -> list[TestResult]:
        """Return cached results, running if not yet executed."""
        if self._results is None:
            self.run()
        return self._results

    def get_rag_status(self) -> RAGStatus:
        """Return overall RAG status."""
        return compute_rag_status(self.get_results())

    def generate(self, path: str | Path) -> Path:
        """
        Run all tests and write the HTML report.

        Parameters
        ----------
        path:
            Output path for the HTML file.

        Returns
        -------
        Path
            Resolved path to the written file.
        """
        results = self.get_results()
        gen = ReportGenerator(
            card=self._card,
            results=results,
            generated_date=self._generated_date,
            run_id=self._run_id,
            rag_status=self.get_rag_status(),
        )
        return gen.write_html(path)

    def to_json(self, path: str | Path) -> Path:
        """
        Write the JSON sidecar for MRM system ingestion.

        Parameters
        ----------
        path:
            Output path for the JSON file.

        Returns
        -------
        Path
        """
        results = self.get_results()
        gen = ReportGenerator(
            card=self._card,
            results=results,
            generated_date=self._generated_date,
            run_id=self._run_id,
            rag_status=self.get_rag_status(),
        )
        return gen.write_json(path)

    def to_dict(self) -> dict:
        """Serialise the full report to a plain dict."""
        results = self.get_results()
        gen = ReportGenerator(
            card=self._card,
            results=results,
            generated_date=self._generated_date,
            run_id=self._run_id,
            rag_status=self.get_rag_status(),
        )
        return gen.to_dict()
