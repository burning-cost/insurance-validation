"""
Model performance validation for insurance pricing models.

The key design decision here: every metric is exposure-weighted by default.
Standard ML libraries compute unweighted AUC, unweighted RMSE, etc. For
insurance pricing, an unweighted Gini is nearly meaningless - a policy with
0.001 years of exposure should not count the same as a policy with 1.0 years.

Gini = 2 * AUC - 1, computed on (y_true, y_pred) ranked by y_pred. When
exposure weights are provided, the Lorenz curve areas are exposure-weighted.
This matches the actuarial convention used in UK pricing teams.

Usage
-----
    import numpy as np
    from insurance_validation import PerformanceReport

    report = PerformanceReport(
        y_true=actual_claim_counts,
        y_pred=predicted_frequencies,
        exposure=policy_years,
    )

    results = [
        report.gini_coefficient(),
        report.actual_vs_expected(n_bands=10),
        *report.lift_chart(n_bands=10),
    ]
"""
from __future__ import annotations

import numpy as np

from .results import Severity, TestCategory, TestResult


def _weighted_gini(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray | None,
) -> float:
    """
    Compute the Gini coefficient using the trapezoidal AUC of the Lorenz curve.

    This is the actuarial definition: rank policies by predicted score
    (descending), compute the cumulative share of actual losses captured.
    Gini = 2 * AUC - 1.

    For unweighted data, this is equivalent to 2 * sklearn.metrics.roc_auc_score - 1
    for binary outcomes, but this implementation handles continuous outcomes
    (e.g. claim counts, severity) correctly.
    """
    n = len(y_true)
    if weights is None:
        weights = np.ones(n)

    # Sort by predicted score descending
    order = np.argsort(-y_pred)
    y_true_sorted = y_true[order]
    weights_sorted = weights[order]

    # Cumulative shares
    cumulative_weight = np.cumsum(weights_sorted)
    cumulative_loss = np.cumsum(y_true_sorted * weights_sorted)

    total_weight = cumulative_weight[-1]
    total_loss = cumulative_loss[-1]

    if total_weight == 0 or total_loss == 0:
        return 0.0

    x = np.concatenate([[0], cumulative_weight / total_weight])
    y = np.concatenate([[0], cumulative_loss / total_loss])

    # Area under Lorenz curve via trapezoid rule
    auc = float(np.trapz(y, x))
    gini = 2 * auc - 1
    return float(np.clip(gini, -1.0, 1.0))


class PerformanceReport:
    """
    Model performance validation for a fitted insurance pricing model.

    Parameters
    ----------
    y_true:
        Observed outcomes (claim counts, claim amounts, loss ratios, etc.).
    y_pred:
        Model predictions (expected values on the same scale as y_true).
    weights:
        Sample weights (e.g. policy count for frequency models). If None,
        uniform weights are used.
    exposure:
        Exposure measure (policy years). When provided, actual vs. expected
        ratios are computed on a per-unit-exposure basis, matching actuarial
        convention. For frequency models: rate = claims / exposure.
    model_name:
        Optional label for the model in result details.
    """

    def __init__(
        self,
        y_true: np.ndarray | list,
        y_pred: np.ndarray | list,
        weights: np.ndarray | list | None = None,
        exposure: np.ndarray | list | None = None,
        model_name: str = "model",
    ) -> None:
        self._y_true = np.asarray(y_true, dtype=float)
        self._y_pred = np.asarray(y_pred, dtype=float)
        self._weights = np.asarray(weights, dtype=float) if weights is not None else None
        self._exposure = np.asarray(exposure, dtype=float) if exposure is not None else None
        self._model_name = model_name

        if len(self._y_true) != len(self._y_pred):
            raise ValueError(
                f"y_true and y_pred must have the same length. "
                f"Got {len(self._y_true)} and {len(self._y_pred)}."
            )
        if self._weights is not None and len(self._weights) != len(self._y_true):
            raise ValueError("weights must have the same length as y_true.")
        if self._exposure is not None and len(self._exposure) != len(self._y_true):
            raise ValueError("exposure must have the same length as y_true.")

    def gini_coefficient(
        self,
        min_acceptable: float = 0.1,
    ) -> TestResult:
        """
        Compute the Gini coefficient (exposure-weighted if exposure is provided).

        A Gini of 0 means the model has no discriminatory power. For UK
        private motor frequency models, Gini > 0.3 is typical for a
        well-specified model. Below 0.1 is a concern worth documenting.

        Parameters
        ----------
        min_acceptable:
            Gini values below this threshold fail the test. The default
            of 0.1 is deliberately conservative - override for your line.

        Returns
        -------
        TestResult
        """
        gini = _weighted_gini(self._y_true, self._y_pred, self._weights)

        passed = gini >= min_acceptable
        if gini >= 0.3:
            severity = Severity.INFO
            assessment = "good discriminatory power"
        elif gini >= 0.1:
            severity = Severity.INFO
            assessment = "moderate discriminatory power - acceptable for complex risks"
        else:
            severity = Severity.WARNING
            assessment = (
                "low discriminatory power - review variable selection and "
                "model specification"
            )

        details = (
            f"Gini coefficient for {self._model_name}: {gini:.4f} "
            f"({assessment}). "
            f"Minimum acceptable threshold: {min_acceptable:.2f}."
        )

        return TestResult(
            test_name="gini_coefficient",
            category=TestCategory.PERFORMANCE,
            passed=passed,
            metric_value=round(gini, 6),
            details=details,
            severity=severity if passed else Severity.WARNING,
            extra={"gini": gini, "weighted": self._weights is not None},
        )

    def lorenz_curve(self, n_points: int = 100) -> TestResult:
        """
        Compute Lorenz curve data points.

        Returns the (x, y) coordinates for the Lorenz curve at n_points
        quantiles. The diagonal represents a model with no discriminatory
        power. Always passes - this is an informational output for chart
        rendering.

        Returns
        -------
        TestResult with extra["x"] and extra["y"] lists.
        """
        w = self._weights if self._weights is not None else np.ones(len(self._y_true))
        order = np.argsort(-self._y_pred)
        y_sorted = self._y_true[order]
        w_sorted = w[order]

        cum_w = np.cumsum(w_sorted)
        cum_loss = np.cumsum(y_sorted * w_sorted)

        total_w = cum_w[-1]
        total_loss = cum_loss[-1]

        x_full = np.concatenate([[0], cum_w / total_w])
        y_full = np.concatenate([[0], cum_loss / total_loss if total_loss > 0 else cum_loss * 0])

        # Sample at n_points
        indices = np.linspace(0, len(x_full) - 1, n_points, dtype=int)
        x_sample = x_full[indices].tolist()
        y_sample = y_full[indices].tolist()

        gini = _weighted_gini(self._y_true, self._y_pred, self._weights)

        return TestResult(
            test_name="lorenz_curve",
            category=TestCategory.PERFORMANCE,
            passed=True,
            metric_value=round(gini, 6),
            details=(
                f"Lorenz curve data for {self._model_name} at {n_points} points. "
                f"Gini = {gini:.4f}."
            ),
            severity=Severity.INFO,
            extra={"x": x_sample, "y": y_sample, "gini": gini},
        )

    def lift_chart(self, n_bands: int = 10) -> list[TestResult]:
        """
        Compute lift by prediction decile.

        Ranks policies by predicted value, groups into n_bands equal-weight
        bands, and computes actual vs. predicted rate per band. This is the
        standard actuarial lift chart (not the marketing "lift" definition).

        A well-calibrated model should show actual and predicted rates
        tracking closely within each band.

        Parameters
        ----------
        n_bands:
            Number of bands (deciles by default).

        Returns
        -------
        list[TestResult]
            Summary TestResult plus one per band in extra.
        """
        w = self._weights if self._weights is not None else np.ones(len(self._y_true))
        exp = self._exposure if self._exposure is not None else np.ones(len(self._y_true))

        order = np.argsort(self._y_pred)
        y_sorted = self._y_true[order]
        pred_sorted = self._y_pred[order]
        w_sorted = w[order]
        exp_sorted = exp[order]

        # Assign to bands based on cumulative weight
        cum_w = np.cumsum(w_sorted)
        total_w = cum_w[-1]
        band_edges = np.linspace(0, total_w, n_bands + 1)
        band_ids = np.searchsorted(cum_w, band_edges[1:], side="left")
        band_ids = np.clip(band_ids, 0, len(y_sorted) - 1)

        bands = []
        prev = 0
        for band_num, edge in enumerate(band_ids):
            idx = slice(prev, edge + 1)
            actual_rate = (
                float(np.sum(y_sorted[idx] * w_sorted[idx]))
                / float(np.sum(exp_sorted[idx]))
                if np.sum(exp_sorted[idx]) > 0
                else 0.0
            )
            predicted_rate = (
                float(np.sum(pred_sorted[idx] * w_sorted[idx]))
                / float(np.sum(exp_sorted[idx]))
                if np.sum(exp_sorted[idx]) > 0
                else 0.0
            )
            n_w = float(np.sum(w_sorted[idx]))
            bands.append({
                "band": band_num + 1,
                "actual_rate": actual_rate,
                "predicted_rate": predicted_rate,
                "weight": n_w,
            })
            prev = edge + 1

        # Summary metric: mean absolute error across bands (relative)
        rel_errors = [
            abs(b["actual_rate"] - b["predicted_rate"]) / b["predicted_rate"]
            for b in bands
            if b["predicted_rate"] > 0
        ]
        mean_rel_error = float(np.mean(rel_errors)) if rel_errors else 0.0
        passed = mean_rel_error < 0.15

        return [
            TestResult(
                test_name="lift_chart",
                category=TestCategory.PERFORMANCE,
                passed=passed,
                metric_value=round(mean_rel_error, 6),
                details=(
                    f"Lift chart ({n_bands} bands) for {self._model_name}. "
                    f"Mean relative error across bands: {mean_rel_error:.2%}. "
                    f"{'Calibration is acceptable.' if passed else 'Mean relative error >15% - model may be miscalibrated.'}"
                ),
                severity=Severity.INFO if passed else Severity.WARNING,
                extra={"bands": bands, "n_bands": n_bands},
            )
        ]

    def actual_vs_expected(self, n_bands: int = 10) -> TestResult:
        """
        Overall actual vs. expected ratio and by-decile breakdown.

        The A/E ratio (actual / expected) should be close to 1.0 on the
        validation dataset. Significant deviation indicates model bias.

        Parameters
        ----------
        n_bands:
            Number of bands for the by-decile breakdown stored in extra.

        Returns
        -------
        TestResult
        """
        w = self._weights if self._weights is not None else np.ones(len(self._y_true))
        exp = self._exposure if self._exposure is not None else np.ones(len(self._y_true))

        total_actual = float(np.sum(self._y_true * w))
        total_predicted = float(np.sum(self._y_pred * w))
        total_exposure = float(np.sum(exp))

        if total_predicted == 0:
            ae_ratio = float("nan")
            passed = False
            details = f"A/E ratio undefined: total predicted is zero for {self._model_name}."
        else:
            ae_ratio = total_actual / total_predicted
            passed = 0.90 <= ae_ratio <= 1.10
            severity_desc = (
                "within acceptable range (0.90-1.10)"
                if passed
                else "outside acceptable range (0.90-1.10) - model is biased"
            )
            details = (
                f"Overall A/E ratio for {self._model_name}: {ae_ratio:.4f} "
                f"({severity_desc}). "
                f"Total actual: {total_actual:,.1f}, "
                f"total predicted: {total_predicted:,.1f}, "
                f"total exposure: {total_exposure:,.1f}."
            )

        # By-band breakdown - sort by predicted
        order = np.argsort(self._y_pred)
        y_sorted = self._y_true[order]
        pred_sorted = self._y_pred[order]
        w_sorted = w[order]

        cum_w = np.cumsum(w_sorted)
        total_w = cum_w[-1]
        band_edges = np.linspace(0, total_w, n_bands + 1)
        band_ids = np.searchsorted(cum_w, band_edges[1:], side="left")
        band_ids = np.clip(band_ids, 0, len(y_sorted) - 1)

        band_data = []
        prev = 0
        for band_num, edge in enumerate(band_ids):
            idx = slice(prev, edge + 1)
            act = float(np.sum(y_sorted[idx] * w_sorted[idx]))
            pred = float(np.sum(pred_sorted[idx] * w_sorted[idx]))
            band_data.append({
                "band": band_num + 1,
                "actual": act,
                "predicted": pred,
                "ae_ratio": act / pred if pred > 0 else None,
            })
            prev = edge + 1

        return TestResult(
            test_name="actual_vs_expected",
            category=TestCategory.PERFORMANCE,
            passed=passed,
            metric_value=round(ae_ratio, 6) if not np.isnan(ae_ratio) else None,
            details=details,
            severity=Severity.INFO if passed else Severity.WARNING,
            extra={
                "ae_ratio": ae_ratio if not np.isnan(ae_ratio) else None,
                "total_actual": total_actual,
                "total_predicted": total_predicted,
                "bands": band_data,
            },
        )

    def calibration_plot_data(self, n_bands: int = 10) -> TestResult:
        """
        Calibration data: mean predicted vs. mean actual per decile.

        Similar to lift_chart but expressed as absolute rates rather than
        relative lift. Used to draw the calibration scatter plot in the
        report.

        Returns
        -------
        TestResult with extra["points"] list of {predicted, actual} dicts.
        """
        w = self._weights if self._weights is not None else np.ones(len(self._y_true))

        order = np.argsort(self._y_pred)
        y_sorted = self._y_true[order]
        pred_sorted = self._y_pred[order]
        w_sorted = w[order]

        cum_w = np.cumsum(w_sorted)
        total_w = cum_w[-1]
        band_edges = np.linspace(0, total_w, n_bands + 1)
        band_ids = np.searchsorted(cum_w, band_edges[1:], side="left")
        band_ids = np.clip(band_ids, 0, len(y_sorted) - 1)

        points = []
        prev = 0
        for edge in band_ids:
            idx = slice(prev, edge + 1)
            w_sum = float(np.sum(w_sorted[idx]))
            if w_sum > 0:
                mean_pred = float(np.sum(pred_sorted[idx] * w_sorted[idx])) / w_sum
                mean_actual = float(np.sum(y_sorted[idx] * w_sorted[idx])) / w_sum
                points.append({"predicted": mean_pred, "actual": mean_actual, "weight": w_sum})
            prev = edge + 1

        return TestResult(
            test_name="calibration_plot",
            category=TestCategory.PERFORMANCE,
            passed=True,
            metric_value=None,
            details=(
                f"Calibration data for {self._model_name} at {n_bands} bands. "
                "Plot predicted vs. actual to assess calibration quality."
            ),
            severity=Severity.INFO,
            extra={"points": points, "n_bands": n_bands},
        )
