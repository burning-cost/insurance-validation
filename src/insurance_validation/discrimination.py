"""
Fairness and proxy discrimination testing for UK insurance pricing models.

This module implements the FCA Consumer Duty and TR24/2 discrimination checks.
The focus is on proxy discrimination: using variables that are correlated with
protected characteristics (age, gender, ethnicity, disability) may produce
discriminatory outcomes even if the protected characteristic itself is excluded.

Postcode is the canonical insurance example: postcodes correlate with
ethnicity, and using postcode relativities without analysis creates disparate
impact risk.

The renewal cohort A/E test is an FCA TR24/2 requirement: renewal pricing
must not systematically over-charge long-standing customers. We test A/E
by tenure band and flag divergence outside [0.85, 1.15].

Usage
-----
    import polars as pl
    from insurance_validation import DiscriminationReport

    report = DiscriminationReport(
        df=features_df,
        predictions=model_predictions,
    )

    results = [
        report.proxy_correlation(
            features=["postcode_area", "vehicle_value"],
            protected_chars=["income_proxy", "age_band"],
        ),
        report.disparate_impact_ratio(
            predictions=predictions,
            group_col="age_band",
        ),
        report.renewal_cohort_ae(
            y_true=claim_counts,
            y_pred=predictions,
            tenure_col="years_as_customer",
        ),
        report.subsegment_ae(
            y_true=claim_counts,
            y_pred=predictions,
            segment_col="product_segment",
        ),
        report.subgroup_outcome_analysis(group_col="region"),
    ]
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .results import Severity, TestCategory, TestResult


class DiscriminationReport:
    """
    Fairness and proxy discrimination checks for a pricing model.

    Parameters
    ----------
    df:
        DataFrame containing model features and any group membership
        variables.
    predictions:
        Model predictions (e.g. predicted premium or frequency) aligned
        to the rows of ``df``.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        predictions: np.ndarray | list | None = None,
    ) -> None:
        self._df = df
        self._predictions = (
            np.asarray(predictions, dtype=float)
            if predictions is not None
            else None
        )

    def proxy_correlation(
        self,
        features: list[str],
        protected_chars: list[str],
        threshold: float = 0.3,
    ) -> list[TestResult]:
        """
        Test for correlation between model features and protected characteristics.

        Uses Spearman rank correlation (robust to non-linear monotone
        relationships) for numeric pairs and Cramer's V for categorical pairs.

        Parameters
        ----------
        features:
            Model input variable names present in ``df``.
        protected_chars:
            Protected characteristic or proxy variable names present in
            ``df`` (e.g. ``"age_band"``, ``"income_decile"``).
        threshold:
            Correlation magnitude above which a pair is flagged.
            0.3 is moderate correlation - warrants documentation.

        Returns
        -------
        list[TestResult]
            One result per (feature, protected_char) pair.
        """
        results = []

        for feat in features:
            if feat not in self._df.columns:
                continue
            for prot in protected_chars:
                if prot not in self._df.columns:
                    continue

                feat_series = self._df[feat]
                prot_series = self._df[prot]

                feat_numeric = feat_series.dtype in (
                    pl.Float32, pl.Float64,
                    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                )
                prot_numeric = prot_series.dtype in (
                    pl.Float32, pl.Float64,
                    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                )

                if feat_numeric and prot_numeric:
                    # Spearman rank correlation
                    corr_result = self._spearman(feat_series, prot_series)
                    metric = corr_result
                    metric_label = "Spearman r"
                else:
                    # Cramer's V for categorical / mixed
                    corr_result = self._cramers_v(feat_series, prot_series)
                    metric = corr_result
                    metric_label = "Cramer's V"

                passed = abs(metric) < threshold
                severity = Severity.INFO if passed else (
                    Severity.WARNING if abs(metric) < 0.5 else Severity.CRITICAL
                )

                if passed:
                    details = (
                        f"'{feat}' and '{prot}': {metric_label} = {metric:.3f}, "
                        f"below threshold {threshold}. No evidence of proxy correlation."
                    )
                else:
                    details = (
                        f"'{feat}' and '{prot}': {metric_label} = {metric:.3f}, "
                        f"exceeds threshold {threshold}. "
                        "This feature may act as a proxy for the protected characteristic. "
                        "Document the actuarial justification for its inclusion."
                    )

                results.append(
                    TestResult(
                        test_name=f"proxy_correlation_{feat}_{prot}",
                        category=TestCategory.DISCRIMINATION,
                        passed=passed,
                        metric_value=round(abs(metric), 6),
                        details=details,
                        severity=severity,
                        extra={
                            "feature": feat,
                            "protected_char": prot,
                            "metric": metric_label,
                            "correlation": metric,
                        },
                    )
                )

        return results

    def _spearman(self, x: pl.Series, y: pl.Series) -> float:
        """Spearman rank correlation, handling nulls by dropping pairs."""
        df_pair = pl.DataFrame({"x": x, "y": y}).drop_nulls()
        if len(df_pair) < 3:
            return 0.0
        x_arr = df_pair["x"].to_numpy().astype(float)
        y_arr = df_pair["y"].to_numpy().astype(float)
        x_ranks = _rank(x_arr)
        y_ranks = _rank(y_arr)
        return float(_pearson(x_ranks, y_ranks))

    def _cramers_v(self, x: pl.Series, y: pl.Series) -> float:
        """Cramer's V association statistic for two categorical series."""
        df_pair = pl.DataFrame({"x": x.cast(pl.Utf8), "y": y.cast(pl.Utf8)}).drop_nulls()
        if len(df_pair) < 3:
            return 0.0
        n = len(df_pair)
        x_vals = df_pair["x"].to_list()
        y_vals = df_pair["y"].to_list()
        x_cats = sorted(set(x_vals))
        y_cats = sorted(set(y_vals))
        x_idx = {v: i for i, v in enumerate(x_cats)}
        y_idx = {v: i for i, v in enumerate(y_cats)}
        contingency = np.zeros((len(x_cats), len(y_cats)), dtype=float)
        for xv, yv in zip(x_vals, y_vals):
            contingency[x_idx[xv], y_idx[yv]] += 1
        row_sums = contingency.sum(axis=1, keepdims=True)
        col_sums = contingency.sum(axis=0, keepdims=True)
        expected = row_sums * col_sums / n
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = float(np.sum(np.where(expected > 0, (contingency - expected) ** 2 / expected, 0)))
        min_dim = min(len(x_cats) - 1, len(y_cats) - 1)
        if min_dim == 0 or n == 0:
            return 0.0
        return float(np.sqrt(chi2 / (n * min_dim)))

    def disparate_impact_ratio(
        self,
        predictions: np.ndarray | list | None = None,
        group_col: str = "",
        reference_group: str | None = None,
        threshold: float = 0.8,
    ) -> TestResult:
        """
        Compute the disparate impact ratio across groups.

        Disparate impact ratio (DIR) = mean prediction for disadvantaged group /
        mean prediction for reference group. The 80% rule (DIR < 0.8) is the
        US EEOC standard and is widely cited in FCA fair value analysis.

        For insurance, 'predictions' are typically the premium or the risk
        loading applied. A DIR < 0.8 for a protected group indicates
        potential discrimination.

        Parameters
        ----------
        predictions:
            If None, uses predictions passed at construction.
        group_col:
            Column in ``df`` containing group membership labels.
        reference_group:
            The group to use as reference (denominator). If None, uses
            the group with the highest mean prediction.
        threshold:
            DIR below this value fails the test. Default 0.8 (80% rule).

        Returns
        -------
        TestResult
        """
        preds = predictions if predictions is not None else self._predictions
        if preds is None:
            return TestResult(
                test_name="disparate_impact_ratio",
                category=TestCategory.DISCRIMINATION,
                passed=False,
                details="No predictions provided for disparate impact analysis.",
                severity=Severity.WARNING,
            )

        preds_arr = np.asarray(preds, dtype=float)

        if group_col not in self._df.columns:
            return TestResult(
                test_name="disparate_impact_ratio",
                category=TestCategory.DISCRIMINATION,
                passed=False,
                details=f"Group column '{group_col}' not found in dataset.",
                severity=Severity.WARNING,
            )

        groups = self._df[group_col].to_list()
        unique_groups = sorted(set(g for g in groups if g is not None))

        group_means: dict[str, float] = {}
        for g in unique_groups:
            mask = np.array([gi == g for gi in groups])
            if mask.sum() > 0:
                group_means[str(g)] = float(preds_arr[mask].mean())

        if not group_means:
            return TestResult(
                test_name="disparate_impact_ratio",
                category=TestCategory.DISCRIMINATION,
                passed=False,
                details="Could not compute group means.",
                severity=Severity.WARNING,
            )

        if reference_group is not None and reference_group in group_means:
            ref_mean = group_means[reference_group]
            ref_label = reference_group
        else:
            ref_label = max(group_means, key=lambda k: group_means[k])
            ref_mean = group_means[ref_label]

        ratios = {
            g: m / ref_mean if ref_mean > 0 else float("nan")
            for g, m in group_means.items()
        }

        min_dir = min(ratios.values())
        min_group = min(ratios, key=lambda k: ratios[k])
        passed = min_dir >= threshold

        details = (
            f"Disparate impact analysis on '{group_col}' using '{ref_label}' "
            f"as reference group. "
            f"Minimum DIR: {min_dir:.3f} (group: '{min_group}'). "
            f"Threshold: {threshold}. "
            f"{'All groups within threshold.' if passed else 'Group below threshold - document justification and consider mitigation.'}"
        )

        return TestResult(
            test_name="disparate_impact_ratio",
            category=TestCategory.DISCRIMINATION,
            passed=passed,
            metric_value=round(min_dir, 6),
            details=details,
            severity=Severity.INFO if passed else Severity.WARNING,
            extra={
                "group_col": group_col,
                "reference_group": ref_label,
                "group_means": group_means,
                "ratios": ratios,
                "threshold": threshold,
            },
        )

    def renewal_cohort_ae(
        self,
        y_true: np.ndarray | list,
        y_pred: np.ndarray | list,
        tenure_col: str,
        weights: np.ndarray | list | None = None,
        ae_low: float = 0.85,
        ae_high: float = 1.15,
        tenure_bands: list[tuple[int, int, str]] | None = None,
    ) -> TestResult:
        """
        Compute A/E ratio by customer tenure band.

        FCA TR24/2 requires that renewal pricing is not systematically
        biased against long-standing customers (the "loyalty penalty"
        concern). This test checks that the model A/E ratio is consistent
        across tenure bands.

        A divergence in A/E by tenure does not necessarily indicate the
        model is wrong - it may reflect genuine risk differences between
        new and renewing customers. But if predicted A/E is outside
        [ae_low, ae_high] for any band, it warrants documentation.

        Parameters
        ----------
        y_true:
            Observed outcomes (claim counts).
        y_pred:
            Model predictions (expected counts).
        tenure_col:
            Column in ``df`` containing years as customer (integer).
        weights:
            Sample weights. If None, uniform.
        ae_low:
            Lower threshold for A/E ratio. Default 0.85 (FCA guidance).
        ae_high:
            Upper threshold. Default 1.15.
        tenure_bands:
            List of (min_tenure, max_tenure, label) tuples defining the
            bands. Defaults to [(0, 0, "New"), (1, 1, "1yr"), (2, 2, "2yr"),
            (3, 999, "3yr+")].

        Returns
        -------
        TestResult. Fails if any band A/E is outside [ae_low, ae_high].
        """
        if tenure_col not in self._df.columns:
            return TestResult(
                test_name="renewal_cohort_ae",
                category=TestCategory.FAIRNESS,
                passed=False,
                details=f"Tenure column '{tenure_col}' not found in dataset.",
                severity=Severity.WARNING,
            )

        if tenure_bands is None:
            tenure_bands = [
                (0, 0, "New"),
                (1, 1, "1yr"),
                (2, 2, "2yr"),
                (3, 9999, "3yr+"),
            ]

        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        w_arr = np.asarray(weights, dtype=float) if weights is not None else np.ones(len(y_true_arr))
        tenure_vals = self._df[tenure_col].to_numpy()

        band_results = []
        any_failed = False

        for t_min, t_max, label in tenure_bands:
            mask = (tenure_vals >= t_min) & (tenure_vals <= t_max)
            n = int(mask.sum())
            if n == 0:
                band_results.append({
                    "band": label,
                    "n": 0,
                    "actual": 0.0,
                    "predicted": 0.0,
                    "ae_ratio": None,
                    "in_range": True,
                })
                continue

            actual = float(np.sum(y_true_arr[mask] * w_arr[mask]))
            predicted = float(np.sum(y_pred_arr[mask] * w_arr[mask]))
            ae = actual / predicted if predicted > 0 else float("nan")
            in_range = ae_low <= ae <= ae_high if not np.isnan(ae) else False

            if not in_range:
                any_failed = True

            band_results.append({
                "band": label,
                "n": n,
                "actual": actual,
                "predicted": predicted,
                "ae_ratio": ae if not np.isnan(ae) else None,
                "in_range": in_range,
            })

        passed = not any_failed

        failing_bands = [b["band"] for b in band_results if not b["in_range"] and b["ae_ratio"] is not None]
        if passed:
            verdict = f"All tenure bands have A/E within [{ae_low}, {ae_high}]. No renewal cohort bias detected."
        else:
            verdict = (
                f"A/E outside [{ae_low}, {ae_high}] in bands: {', '.join(failing_bands)}. "
                "Review whether the model systematically under- or over-prices renewal business. "
                "FCA TR24/2 requires documented justification for pricing differentials by tenure."
            )

        return TestResult(
            test_name="renewal_cohort_ae",
            category=TestCategory.FAIRNESS,
            passed=passed,
            metric_value=None,
            details=f"Renewal cohort A/E by tenure band (threshold [{ae_low}, {ae_high}]). {verdict}",
            severity=Severity.INFO if passed else Severity.WARNING,
            extra={
                "bands": band_results,
                "ae_low": ae_low,
                "ae_high": ae_high,
            },
        )

    def subsegment_ae(
        self,
        y_true: np.ndarray | list,
        y_pred: np.ndarray | list,
        segment_col: str,
        weights: np.ndarray | list | None = None,
        ae_low: float = 0.85,
        ae_high: float = 1.15,
    ) -> TestResult:
        """
        Compute A/E ratio by product governance segment.

        Sub-segment calibration checks whether the model is calibrated
        within each product segment (e.g. Tier 1 / Tier 2 by NCD,
        or by distribution channel). Material divergence within a segment
        suggests the model has a systematic blind spot.

        This supports PRA SS1/23 Principle 4 (outcome analysis) and FCA
        Consumer Duty's requirement to demonstrate fair value by segment.

        Parameters
        ----------
        y_true:
            Observed outcomes.
        y_pred:
            Model predictions.
        segment_col:
            Column in ``df`` containing segment labels.
        weights:
            Sample weights. If None, uniform.
        ae_low:
            Lower A/E threshold. Default 0.85.
        ae_high:
            Upper A/E threshold. Default 1.15.

        Returns
        -------
        TestResult. Fails if any segment A/E outside [ae_low, ae_high].
        """
        if segment_col not in self._df.columns:
            return TestResult(
                test_name=f"subsegment_ae_{segment_col}",
                category=TestCategory.FAIRNESS,
                passed=False,
                details=f"Segment column '{segment_col}' not found in dataset.",
                severity=Severity.WARNING,
            )

        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        w_arr = np.asarray(weights, dtype=float) if weights is not None else np.ones(len(y_true_arr))
        segment_vals = self._df[segment_col].to_list()
        unique_segs = sorted(set(s for s in segment_vals if s is not None))

        seg_results = []
        any_failed = False

        for seg in unique_segs:
            mask = np.array([s == seg for s in segment_vals])
            n = int(mask.sum())
            actual = float(np.sum(y_true_arr[mask] * w_arr[mask]))
            predicted = float(np.sum(y_pred_arr[mask] * w_arr[mask]))
            ae = actual / predicted if predicted > 0 else float("nan")
            in_range = ae_low <= ae <= ae_high if not np.isnan(ae) else False

            if not in_range:
                any_failed = True

            seg_results.append({
                "segment": str(seg),
                "n": n,
                "actual": actual,
                "predicted": predicted,
                "ae_ratio": ae if not np.isnan(ae) else None,
                "in_range": in_range,
            })

        passed = not any_failed
        failing_segs = [s["segment"] for s in seg_results if not s["in_range"] and s["ae_ratio"] is not None]

        if passed:
            verdict = f"All segments within A/E range [{ae_low}, {ae_high}]."
        else:
            verdict = (
                f"A/E outside [{ae_low}, {ae_high}] in segments: {', '.join(failing_segs)}. "
                "Sub-segment miscalibration may indicate the model is systematically "
                "wrong for a product segment. Review GLM offsets or GBM feature coverage."
            )

        return TestResult(
            test_name=f"subsegment_ae_{segment_col}",
            category=TestCategory.FAIRNESS,
            passed=passed,
            metric_value=None,
            details=f"Sub-segment A/E by '{segment_col}' (threshold [{ae_low}, {ae_high}]). {verdict}",
            severity=Severity.INFO if passed else Severity.WARNING,
            extra={
                "segment_col": segment_col,
                "segments": seg_results,
                "ae_low": ae_low,
                "ae_high": ae_high,
            },
        )

    def subgroup_outcome_analysis(
        self,
        group_col: str,
        outcome_col: str | None = None,
        predictions: np.ndarray | list | None = None,
    ) -> TestResult:
        """
        Summarise predicted and (optionally) actual outcomes by subgroup.

        This supports the SS1/23 requirement to evidence that model outcomes
        are understood and justified across customer segments.

        Parameters
        ----------
        group_col:
            Column in ``df`` containing group membership labels.
        outcome_col:
            Column in ``df`` containing actual outcomes. Optional.
        predictions:
            Model predictions. If None, uses predictions from construction.

        Returns
        -------
        TestResult (always INFO - this is descriptive, not a threshold test).
        """
        preds = predictions if predictions is not None else self._predictions

        if group_col not in self._df.columns:
            return TestResult(
                test_name=f"subgroup_outcome_{group_col}",
                category=TestCategory.DISCRIMINATION,
                passed=False,
                details=f"Group column '{group_col}' not found in dataset.",
                severity=Severity.WARNING,
            )

        groups = self._df[group_col].to_list()
        unique_groups = sorted(set(g for g in groups if g is not None))

        subgroup_data = []
        for g in unique_groups:
            mask = np.array([gi == g for gi in groups])
            n = int(mask.sum())
            row: dict = {"group": str(g), "n": n}

            if preds is not None:
                preds_arr = np.asarray(preds, dtype=float)
                row["mean_predicted"] = float(preds_arr[mask].mean())
                row["std_predicted"] = float(preds_arr[mask].std())

            if outcome_col and outcome_col in self._df.columns:
                outcomes = self._df[outcome_col].to_numpy().astype(float)
                row["mean_actual"] = float(outcomes[mask].mean())

            subgroup_data.append(row)

        details = (
            f"Subgroup outcome analysis for '{group_col}': "
            f"{len(unique_groups)} groups found. "
            "Review per-group means for systematic disparities."
        )

        return TestResult(
            test_name=f"subgroup_outcome_{group_col}",
            category=TestCategory.DISCRIMINATION,
            passed=True,
            metric_value=float(len(unique_groups)),
            details=details,
            severity=Severity.INFO,
            extra={"group_col": group_col, "subgroups": subgroup_data},
        )


def _rank(arr: np.ndarray) -> np.ndarray:
    """Return average ranks for an array (handles ties)."""
    n = len(arr)
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    # Handle ties: use average rank
    sorted_arr = arr[order]
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_arr[j + 1] == sorted_arr[j]:
            j += 1
        avg_rank = (ranks[order[i]] + ranks[order[j]]) / 2
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    if denom == 0:
        return 0.0
    return float((xm * ym).sum() / denom)
