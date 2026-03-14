"""
Model performance validation for insurance pricing models.

The key design decision here: every metric is exposure-weighted by default.
Standard ML libraries compute unweighted AUC, unweighted RMSE, etc. For
insurance pricing, an unweighted Gini is nearly meaningless - a policy with
0.001 years of exposure should not count the same as a policy with 1.0 years.

Gini = 2 * AUC - 1, computed on (y_true, y_pred) ranked by y_pred. When
exposure weights are provided, the Lorenz curve areas are exposure-weighted.
This matches the actuarial convention used in UK pricing teams.

Bootstrap confidence intervals on Gini use 1000 resamples by default. The
Poisson exact CI on A/E ratios uses the chi-squared pivot, which is exact
for Poisson-distributed claim counts. Both are standard actuarial tools.

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
        report.gini_with_ci(),
        report.actual_vs_expected(n_bands=10),
        report.ae_with_poisson_ci(),
        *report.lift_chart(n_bands=10),
        report.double_lift(y_pred_incumbent=old_predictions),
        report.hosmer_lemeshow_test(),
    ]
"""
from __future__ import annotations

import numpy as np

# numpy.trapezoid was added in 2.0; np.trapz was removed in 2.0
try:
    _trapezoid = np.trapezoid
except AttributeError:
    _trapezoid = np.trapz  # type: ignore[attr-defined]  # NumPy < 2.0
from scipy import stats

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
    auc = float(_trapezoid(y, x))
    gini = 2 * auc - 1
    return float(np.clip(gini, -1.0, 1.0))


def _poisson_ae_ci(
    actual_claims: float,
    expected_claims: float,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Exact Poisson confidence interval on the A/E ratio.

    Uses the chi-squared pivot:
      lower = chi2(2*A, alpha/2) / (2*E)
      upper = chi2(2*A+2, 1-alpha/2) / (2*E)

    This is exact for Poisson-distributed claim counts and is the
    standard actuarial approach. See Dobson et al. (1991).

    Parameters
    ----------
    actual_claims:
        Observed claim count (A). Must be >= 0.
    expected_claims:
        Predicted claim count (E). Must be > 0.
    alpha:
        Significance level. Default 0.05 gives 95% CI.

    Returns
    -------
    (lower_ratio, upper_ratio)
        95% CI on the A/E ratio.
    """
    if expected_claims <= 0:
        return (float("nan"), float("nan"))

    A = actual_claims
    E = expected_claims

    # Handle zero claims: lower bound is 0
    if A == 0:
        lower = 0.0
    else:
        lower = stats.chi2.ppf(alpha / 2, df=2 * A) / (2 * E)

    upper = stats.chi2.ppf(1 - alpha / 2, df=2 * A + 2) / (2 * E)

    return (lower, upper)


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
        # Use exposure as weight for the Lorenz curve when no separate weights provided.
        # This is the actuarial convention: exposure-weighted Gini.
        lorenz_weights = self._weights if self._weights is not None else self._exposure
        gini = _weighted_gini(self._y_true, self._y_pred, lorenz_weights)

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

    def gini_with_ci(
        self,
        n_resamples: int = 1000,
        confidence: float = 0.95,
        min_acceptable: float = 0.1,
        random_state: int = 42,
    ) -> TestResult:
        """
        Gini coefficient with bootstrap confidence interval.

        Bootstrap CI uses stratified resampling with replacement. 1000
        resamples is industry standard for actuarial bootstrap intervals.
        The CI is reported at the percentile level (not BCa) for
        transparency.

        Parameters
        ----------
        n_resamples:
            Number of bootstrap resamples. 1000 is adequate for 95% CI.
        confidence:
            Confidence level, e.g. 0.95 for 95% CI.
        min_acceptable:
            Gini values below this threshold fail. Applied to the point
            estimate, not the lower bound.
        random_state:
            Random seed for reproducibility.

        Returns
        -------
        TestResult with extra["ci_lower"] and extra["ci_upper"].
        """
        lorenz_weights = self._weights if self._weights is not None else self._exposure
        gini_point = _weighted_gini(self._y_true, self._y_pred, lorenz_weights)

        rng = np.random.default_rng(random_state)
        n = len(self._y_true)
        bootstrap_ginis = np.empty(n_resamples)

        for i in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            w_boot = lorenz_weights[idx] if lorenz_weights is not None else None
            bootstrap_ginis[i] = _weighted_gini(
                self._y_true[idx], self._y_pred[idx], w_boot
            )

        alpha = 1 - confidence
        ci_lower = float(np.percentile(bootstrap_ginis, 100 * alpha / 2))
        ci_upper = float(np.percentile(bootstrap_ginis, 100 * (1 - alpha / 2)))

        passed = gini_point >= min_acceptable
        details = (
            f"Gini coefficient for {self._model_name}: {gini_point:.4f} "
            f"({confidence:.0%} CI: [{ci_lower:.4f}, {ci_upper:.4f}], "
            f"{n_resamples} bootstrap resamples). "
            f"Minimum acceptable: {min_acceptable:.2f}."
        )

        return TestResult(
            test_name="gini_with_ci",
            category=TestCategory.PERFORMANCE,
            passed=passed,
            metric_value=round(gini_point, 6),
            details=details,
            severity=Severity.INFO if passed else Severity.WARNING,
            extra={
                "gini": gini_point,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "confidence": confidence,
                "n_resamples": n_resamples,
            },
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
        lorenz_weights = self._weights if self._weights is not None else self._exposure
        w = lorenz_weights if lorenz_weights is not None else np.ones(len(self._y_true))
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

        gini = _weighted_gini(self._y_true, self._y_pred, lorenz_weights)

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

    def double_lift(
        self,
        y_pred_incumbent: np.ndarray | list,
        n_bands: int = 10,
        incumbent_name: str = "incumbent",
    ) -> TestResult:
        """
        Double-lift chart: compare new model vs. incumbent model.

        Policies are ranked by the ratio new_pred / incumbent_pred
        (ascending), then grouped into n_bands equal-weight bands. For
        each band we plot actual, new_pred, and incumbent_pred rates.
        Bands where the new model is lower than the incumbent show the
        new model predicting a lower risk than the old; the actual rate
        tells you which is closer to truth.

        This is the standard chart for model-vs-model comparison in UK
        pricing validation.

        Parameters
        ----------
        y_pred_incumbent:
            Predictions from the incumbent model, same length as y_true.
        n_bands:
            Number of bands.
        incumbent_name:
            Label for the incumbent model in result details.

        Returns
        -------
        TestResult with extra["bands"] list of per-band dicts.
        """
        y_pred_inc = np.asarray(y_pred_incumbent, dtype=float)
        if len(y_pred_inc) != len(self._y_true):
            raise ValueError(
                "y_pred_incumbent must have the same length as y_true."
            )

        w = self._weights if self._weights is not None else np.ones(len(self._y_true))
        exp = self._exposure if self._exposure is not None else np.ones(len(self._y_true))

        # Ratio: new / incumbent. Avoid division by zero.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                y_pred_inc > 0,
                self._y_pred / y_pred_inc,
                np.inf,
            )

        # Sort by ratio ascending (policies where new < incumbent first)
        order = np.argsort(ratio)
        y_sorted = self._y_true[order]
        new_sorted = self._y_pred[order]
        inc_sorted = y_pred_inc[order]
        w_sorted = w[order]
        exp_sorted = exp[order]

        cum_w = np.cumsum(w_sorted)
        total_w = cum_w[-1]
        band_edges = np.linspace(0, total_w, n_bands + 1)
        band_ids = np.searchsorted(cum_w, band_edges[1:], side="left")
        band_ids = np.clip(band_ids, 0, len(y_sorted) - 1)

        bands = []
        prev = 0
        for band_num, edge in enumerate(band_ids):
            idx = slice(prev, edge + 1)
            exp_sum = float(np.sum(exp_sorted[idx]))
            w_sum = float(np.sum(w_sorted[idx]))

            def rate(arr: np.ndarray) -> float:
                return float(np.sum(arr[idx] * w_sorted[idx])) / exp_sum if exp_sum > 0 else 0.0

            bands.append({
                "band": band_num + 1,
                "actual_rate": rate(y_sorted),
                "new_model_rate": rate(new_sorted),
                "incumbent_rate": rate(inc_sorted),
                "weight": w_sum,
            })
            prev = edge + 1

        # Summary: which model tracks actual better?
        new_mae = float(np.mean([
            abs(b["actual_rate"] - b["new_model_rate"])
            for b in bands
        ]))
        inc_mae = float(np.mean([
            abs(b["actual_rate"] - b["incumbent_rate"])
            for b in bands
        ]))
        new_model_wins = new_mae < inc_mae
        passed = new_model_wins  # informative - not a hard regulatory threshold

        details = (
            f"Double-lift chart ({n_bands} bands): {self._model_name} vs {incumbent_name}. "
            f"New model MAE across bands: {new_mae:.5f}. "
            f"Incumbent MAE: {inc_mae:.5f}. "
            f"{'New model tracks actual losses more closely.' if new_model_wins else 'Incumbent tracks actual losses more closely - review model improvement case.'}"
        )

        return TestResult(
            test_name="double_lift",
            category=TestCategory.PERFORMANCE,
            passed=passed,
            metric_value=round(new_mae - inc_mae, 6),
            details=details,
            severity=Severity.INFO if new_model_wins else Severity.WARNING,
            extra={
                "bands": bands,
                "n_bands": n_bands,
                "new_model_mae": new_mae,
                "incumbent_mae": inc_mae,
                "incumbent_name": incumbent_name,
            },
        )

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

    def ae_with_poisson_ci(
        self,
        alpha: float = 0.05,
        threshold_low: float = 0.90,
        threshold_high: float = 1.10,
    ) -> TestResult:
        """
        Overall A/E ratio with Poisson exact confidence interval.

        The Poisson exact CI uses the chi-squared pivot:
          lower = chi2(2A, alpha/2) / (2*E)
          upper = chi2(2A+2, 1-alpha/2) / (2*E)

        This is exact for Poisson-distributed claim counts and is the
        standard actuarial approach. The test fails if the CI does not
        include 1.0 (i.e., material bias is detectable at the given alpha
        level) OR if the point estimate is outside [threshold_low, threshold_high].

        IBNR caveat: A/E ratios on recently-incurred claims may reflect
        IBNR underdevelopment, not model error. Use fully-developed claims
        wherever possible.

        Parameters
        ----------
        alpha:
            Significance level for the CI. Default 0.05 = 95% CI.
        threshold_low:
            A/E below this fails the point estimate check.
        threshold_high:
            A/E above this fails the point estimate check.

        Returns
        -------
        TestResult
        """
        w = self._weights if self._weights is not None else np.ones(len(self._y_true))

        total_actual = float(np.sum(self._y_true * w))
        total_predicted = float(np.sum(self._y_pred * w))

        if total_predicted == 0:
            return TestResult(
                test_name="ae_poisson_ci",
                category=TestCategory.PERFORMANCE,
                passed=False,
                details=f"A/E Poisson CI undefined: total predicted is zero for {self._model_name}.",
                severity=Severity.WARNING,
            )

        ae_ratio = total_actual / total_predicted
        ci_lower, ci_upper = _poisson_ae_ci(total_actual, total_predicted, alpha=alpha)

        # Test passes if 1.0 is inside the CI and point estimate is in range
        ci_includes_one = bool(ci_lower <= 1.0 <= ci_upper)
        point_in_range = threshold_low <= ae_ratio <= threshold_high
        passed = bool(ci_includes_one and point_in_range)

        details = (
            f"A/E ratio for {self._model_name}: {ae_ratio:.4f} "
            f"(Poisson {100*(1-alpha):.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]). "
            f"Point estimate {'within' if point_in_range else 'outside'} "
            f"acceptable range [{threshold_low}, {threshold_high}]. "
            f"CI {'includes' if ci_includes_one else 'excludes'} 1.0 "
            f"(material bias {'not detected' if ci_includes_one else 'detected'}). "
            "IBNR caveat: A/E ratios on recently-incurred claims may reflect "
            "IBNR underdevelopment, not model error. Use fully-developed claims "
            "wherever possible."
        )

        return TestResult(
            test_name="ae_poisson_ci",
            category=TestCategory.PERFORMANCE,
            passed=passed,
            metric_value=round(ae_ratio, 6),
            details=details,
            severity=Severity.INFO if passed else Severity.WARNING,
            extra={
                "ae_ratio": ae_ratio,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "alpha": alpha,
                "total_actual": total_actual,
                "total_predicted": total_predicted,
                "ci_includes_one": ci_includes_one,
            },
        )

    def hosmer_lemeshow_test(
        self,
        n_groups: int = 10,
        alpha: float = 0.05,
    ) -> TestResult:
        """
        Hosmer-Lemeshow calibration test.

        Tests whether the model is calibrated: are predicted probabilities
        consistent with observed frequencies across predicted-risk groups?

        Policies are grouped into n_groups deciles of predicted risk.
        For each group we compute observed and expected counts. The
        H-L statistic is approximately chi-squared(n_groups - 2) under
        the null of perfect calibration.

        Parameters
        ----------
        n_groups:
            Number of groups (deciles by default). H-L statistic has
            n_groups - 2 degrees of freedom.
        alpha:
            Significance level. p < alpha fails the test.

        Returns
        -------
        TestResult
        """
        w = self._weights if self._weights is not None else np.ones(len(self._y_true))

        order = np.argsort(self._y_pred)
        y_sorted = self._y_true[order]
        pred_sorted = self._y_pred[order]
        w_sorted = w[order]

        cum_w = np.cumsum(w_sorted)
        total_w = cum_w[-1]
        band_edges = np.linspace(0, total_w, n_groups + 1)
        band_ids = np.searchsorted(cum_w, band_edges[1:], side="left")
        band_ids = np.clip(band_ids, 0, len(y_sorted) - 1)

        hl_stat = 0.0
        prev = 0
        groups = []
        for edge in band_ids:
            idx = slice(prev, edge + 1)
            observed = float(np.sum(y_sorted[idx] * w_sorted[idx]))
            expected = float(np.sum(pred_sorted[idx] * w_sorted[idx]))
            n_w = float(np.sum(w_sorted[idx]))
            if expected > 0:
                hl_stat += (observed - expected) ** 2 / expected
            groups.append({
                "observed": observed,
                "expected": expected,
                "n_weight": n_w,
            })
            prev = edge + 1

        df = n_groups - 2
        p_value = float(1 - stats.chi2.cdf(hl_stat, df=df)) if df > 0 else float("nan")
        passed = p_value >= alpha if not np.isnan(p_value) else False

        details = (
            f"Hosmer-Lemeshow calibration test ({n_groups} groups) for {self._model_name}: "
            f"H-L statistic = {hl_stat:.4f}, df = {df}, p-value = {p_value:.4f}. "
            f"{'Model is well-calibrated (p >= ' + str(alpha) + ').' if passed else 'Calibration rejected (p < ' + str(alpha) + ') - model predictions are systematically biased in some risk groups.'}"
        )

        return TestResult(
            test_name="hosmer_lemeshow",
            category=TestCategory.PERFORMANCE,
            passed=passed,
            metric_value=round(p_value, 6) if not np.isnan(p_value) else None,
            details=details,
            severity=Severity.INFO if passed else Severity.WARNING,
            extra={
                "hl_statistic": hl_stat,
                "df": df,
                "p_value": p_value if not np.isnan(p_value) else None,
                "alpha": alpha,
                "groups": groups,
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
