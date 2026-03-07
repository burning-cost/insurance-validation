"""
Population Stability Index and feature drift testing.

PSI measures how much a distribution has shifted between a reference
period (training data) and a current period (deployment data). It is the
standard actuarial tool for monitoring model population stability.

PSI thresholds (industry standard):
  < 0.10  : Stable. No action required.
  0.10-0.25: Moderate shift. Monitor closely, consider investigation.
  > 0.25  : Significant shift. Model recalibration or rebuild likely needed.

The PSI formula is:
  PSI = sum( (actual_pct - expected_pct) * ln(actual_pct / expected_pct) )

where bins are defined on the reference distribution and both reference
and current are binned identically.

Usage
-----
    import polars as pl
    from insurance_validation import StabilityReport

    report = StabilityReport()

    result = report.psi(
        reference=reference_predictions,
        current=current_predictions,
        n_bins=10,
    )

    drift_results = report.feature_drift(
        reference_df=train_df,
        current_df=deploy_df,
        features=["vehicle_age", "driver_age", "region"],
    )
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .results import Severity, TestCategory, TestResult

PSI_STABLE = 0.10
PSI_MODERATE = 0.25


def _psi_score(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    bins: np.ndarray | None = None,
) -> tuple[float, list[dict]]:
    """
    Compute PSI and return per-bin breakdown.

    Parameters
    ----------
    reference:
        Reference distribution (1D array of values).
    current:
        Current distribution (1D array of values).
    n_bins:
        Number of bins. Ignored if ``bins`` is provided.
    bins:
        Pre-computed bin edges. If None, quantile bins are computed
        from the reference distribution.

    Returns
    -------
    (psi_value, bin_details)
    """
    epsilon = 1e-6  # Avoid log(0)

    if bins is None:
        # Quantile-based bins from reference
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(reference, quantiles)
        # Ensure uniqueness (can collapse for discrete/constant variables)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            # Reference is constant. Expand range to cover both distributions.
            all_vals = np.concatenate([reference, current])
            vmin, vmax = float(all_vals.min()), float(all_vals.max())
            if vmin == vmax:
                # Both distributions are identical single-value - PSI = 0
                return 0.0, []
            bin_edges = np.linspace(vmin, vmax, n_bins + 1)
    else:
        bin_edges = bins

    # Ensure the last bin edge captures max values (right edge inclusive)
    bin_edges = bin_edges.copy()
    bin_edges[-1] = max(bin_edges[-1], float(current.max()) + 1e-10, float(reference.max()) + 1e-10)

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    n_ref = float(len(reference))
    n_cur = float(len(current))
    n_bins_actual = len(ref_counts)

    ref_pct = (ref_counts + epsilon) / (n_ref + epsilon * n_bins_actual)
    cur_pct = (cur_counts + epsilon) / (n_cur + epsilon * n_bins_actual)

    bin_psi = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    psi_total = float(bin_psi.sum())

    bin_details = [
        {
            "bin": i + 1,
            "lower": float(bin_edges[i]),
            "upper": float(bin_edges[i + 1]),
            "reference_pct": float(ref_pct[i]),
            "current_pct": float(cur_pct[i]),
            "bin_psi": float(bin_psi[i]),
        }
        for i in range(n_bins_actual)
    ]

    return psi_total, bin_details


class StabilityReport:
    """
    Population stability assessment comparing reference and current data.

    This class can be instantiated without arguments. Pass data to each
    method rather than at construction, which makes it easier to run
    multiple PSI checks with different reference periods.
    """

    def psi(
        self,
        reference: np.ndarray | list,
        current: np.ndarray | list,
        n_bins: int = 10,
        label: str = "score",
    ) -> TestResult:
        """
        Compute PSI between a reference distribution and a current distribution.

        Parameters
        ----------
        reference:
            Values from the reference period (typically training or holdout).
        current:
            Values from the current period (typically live deployment).
        n_bins:
            Number of quantile bins. 10 is the convention for model score PSI.
        label:
            Name of the variable being tested, used in result details.

        Returns
        -------
        TestResult
        """
        ref_arr = np.asarray(reference, dtype=float)
        cur_arr = np.asarray(current, dtype=float)

        psi_val, bin_details = _psi_score(ref_arr, cur_arr, n_bins=n_bins)

        if psi_val < PSI_STABLE:
            passed = True
            severity = Severity.INFO
            interpretation = f"stable (PSI < {PSI_STABLE})"
        elif psi_val < PSI_MODERATE:
            passed = True
            severity = Severity.WARNING
            interpretation = (
                f"moderate shift ({PSI_STABLE} <= PSI < {PSI_MODERATE}). "
                "Monitor closely."
            )
        else:
            passed = False
            severity = Severity.CRITICAL
            interpretation = (
                f"significant shift (PSI >= {PSI_MODERATE}). "
                "Model recalibration or rebuild required."
            )

        details = (
            f"PSI for '{label}': {psi_val:.4f} - {interpretation} "
            f"(reference n={len(ref_arr):,}, current n={len(cur_arr):,}, "
            f"bins={n_bins})."
        )

        return TestResult(
            test_name=f"psi_{label}",
            category=TestCategory.STABILITY,
            passed=passed,
            metric_value=psi_val,  # Full precision - do not round here
            details=details,
            severity=severity,
            extra={"bins": bin_details, "n_bins": n_bins, "label": label},
        )

    def feature_drift(
        self,
        reference_df: pl.DataFrame,
        current_df: pl.DataFrame,
        features: list[str],
        n_bins: int = 10,
    ) -> list[TestResult]:
        """
        Compute PSI for each feature between reference and current datasets.

        Numeric features use quantile-based PSI. Categorical features use
        a value-frequency PSI (each unique value is a bin).

        Parameters
        ----------
        reference_df:
            Reference period dataset.
        current_df:
            Current period dataset.
        features:
            Column names to test. Must be present in both dataframes.
        n_bins:
            Number of bins for numeric PSI.

        Returns
        -------
        list[TestResult]
            One result per feature.
        """
        results = []

        for feat in features:
            if feat not in reference_df.columns:
                results.append(
                    TestResult(
                        test_name=f"psi_{feat}",
                        category=TestCategory.STABILITY,
                        passed=False,
                        details=f"Feature '{feat}' not found in reference dataset.",
                        severity=Severity.WARNING,
                    )
                )
                continue

            if feat not in current_df.columns:
                results.append(
                    TestResult(
                        test_name=f"feature_drift_{feat}",
                        category=TestCategory.STABILITY,
                        passed=False,
                        details=f"Feature '{feat}' not found in current dataset.",
                        severity=Severity.WARNING,
                    )
                )
                continue

            ref_series = reference_df[feat]
            cur_series = current_df[feat]

            ref_is_numeric = ref_series.dtype in (
                pl.Float32, pl.Float64,
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            )

            if ref_is_numeric:
                ref_arr = ref_series.drop_nulls().to_numpy().astype(float)
                cur_arr = cur_series.drop_nulls().to_numpy().astype(float)
                psi_val, bin_details = _psi_score(ref_arr, cur_arr, n_bins=n_bins)
            else:
                # Categorical PSI: frequency per category
                ref_counts = (
                    reference_df.select(pl.col(feat).cast(pl.Utf8))
                    .group_by(feat)
                    .len()
                    .rename({"len": "ref_count"})
                )
                cur_counts = (
                    current_df.select(pl.col(feat).cast(pl.Utf8))
                    .group_by(feat)
                    .len()
                    .rename({"len": "cur_count"})
                )
                merged = ref_counts.join(cur_counts, on=feat, how="full")
                epsilon = 1e-6
                n_ref = len(reference_df)
                n_cur = len(current_df)
                n_cats = len(merged)
                ref_pct = (merged["ref_count"].fill_null(0).to_numpy() + epsilon) / (n_ref + epsilon * n_cats)
                cur_pct = (merged["cur_count"].fill_null(0).to_numpy() + epsilon) / (n_cur + epsilon * n_cats)
                bin_psi = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
                psi_val = float(bin_psi.sum())
                bin_details = []

            if psi_val < PSI_STABLE:
                passed = True
                severity = Severity.INFO
                interp = "stable"
            elif psi_val < PSI_MODERATE:
                passed = True
                severity = Severity.WARNING
                interp = "moderate shift"
            else:
                passed = False
                severity = Severity.CRITICAL
                interp = "significant shift"

            results.append(
                TestResult(
                    test_name=f"feature_drift_{feat}",
                    category=TestCategory.STABILITY,
                    passed=passed,
                    metric_value=psi_val,  # Full precision
                    details=(
                        f"Feature drift PSI for '{feat}': {psi_val:.4f} - {interp}."
                    ),
                    severity=severity,
                    extra={"feature": feat, "psi": psi_val, "bins": bin_details},
                )
            )

        return results
