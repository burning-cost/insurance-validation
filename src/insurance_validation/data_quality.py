"""
Data quality assessment for insurance model validation.

Produces structured TestResult objects for each check. The checks here
correspond directly to the Data Quality Assessment section of a compliant
SS1/23-aligned validation report.

Usage
-----
    import polars as pl
    from insurance_validation import DataQualityReport

    df = pl.read_parquet("training_data.parquet")
    report = DataQualityReport(df)

    results = [
        *report.missing_value_analysis(threshold=0.05),
        *report.outlier_detection(method="iqr"),
        report.cardinality_check(max_categories=50),
        report.summary_statistics(),
    ]
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl

from .results import Severity, TestCategory, TestResult


class DataQualityReport:
    """
    Data quality assessment for a Polars DataFrame.

    All methods return TestResult objects or lists of them. This keeps
    the interface consistent with the rest of the validation library.

    Parameters
    ----------
    df:
        The dataset to assess. Typically the model training or validation
        dataset.
    dataset_name:
        Label for the dataset used in result details. Defaults to
        ``"dataset"``.
    """

    def __init__(self, df: pl.DataFrame, dataset_name: str = "dataset") -> None:
        self._df = df
        self._name = dataset_name

    @property
    def df(self) -> pl.DataFrame:
        return self._df

    def missing_value_analysis(
        self,
        threshold: float = 0.05,
    ) -> list[TestResult]:
        """
        Check for missing values in each column.

        Parameters
        ----------
        threshold:
            Maximum acceptable missing rate (0-1). Columns above this
            threshold produce a WARNING; above 0.5 produce CRITICAL.

        Returns
        -------
        list[TestResult]
            One result per column.
        """
        results = []
        n_rows = len(self._df)

        for col in self._df.columns:
            n_missing = self._df[col].null_count()
            rate = n_missing / n_rows if n_rows > 0 else 0.0

            if rate == 0:
                severity = Severity.INFO
                passed = True
                details = f"No missing values in '{col}'."
            elif rate <= threshold:
                severity = Severity.INFO
                passed = True
                details = (
                    f"'{col}' has {n_missing:,} missing values "
                    f"({rate:.1%}), within the {threshold:.0%} threshold."
                )
            elif rate <= 0.5:
                severity = Severity.WARNING
                passed = False
                details = (
                    f"'{col}' has {n_missing:,} missing values "
                    f"({rate:.1%}), exceeding the {threshold:.0%} threshold. "
                    "Review imputation or exclusion decisions."
                )
            else:
                severity = Severity.CRITICAL
                passed = False
                details = (
                    f"'{col}' has {n_missing:,} missing values "
                    f"({rate:.1%}). A column with >50% missing data should not "
                    "be used without documented justification."
                )

            results.append(
                TestResult(
                    test_name=f"missing_values_{col}",
                    category=TestCategory.DATA_QUALITY,
                    passed=passed,
                    metric_value=round(rate, 6),
                    details=details,
                    severity=severity,
                    extra={"column": col, "n_missing": n_missing, "n_rows": n_rows},
                )
            )

        return results

    def outlier_detection(
        self,
        method: Literal["iqr", "zscore"] = "iqr",
        iqr_multiplier: float = 3.0,
        zscore_threshold: float = 4.0,
    ) -> list[TestResult]:
        """
        Detect outliers in numeric columns.

        Parameters
        ----------
        method:
            ``"iqr"`` uses the interquartile range method (robust to
            skewed distributions, recommended for insurance data).
            ``"zscore"`` uses the standard deviation method.
        iqr_multiplier:
            Multiplier applied to IQR for fence calculation. Default 3.0
            is more conservative than the textbook 1.5, appropriate for
            the heavy tails seen in insurance data.
        zscore_threshold:
            Number of standard deviations beyond which a value is
            flagged as an outlier.

        Returns
        -------
        list[TestResult]
            One result per numeric column.
        """
        results = []
        numeric_cols = [
            col
            for col in self._df.columns
            if self._df[col].dtype in (
                pl.Float32, pl.Float64, pl.Int8, pl.Int16,
                pl.Int32, pl.Int64, pl.UInt8, pl.UInt16,
                pl.UInt32, pl.UInt64,
            )
        ]

        for col in numeric_cols:
            series = self._df[col].drop_nulls()
            n_total = len(series)

            if n_total == 0:
                continue

            if method == "iqr":
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - iqr_multiplier * iqr
                upper = q3 + iqr_multiplier * iqr
                n_outliers = int(((series < lower) | (series > upper)).sum())
                method_desc = f"IQR x{iqr_multiplier}"
            else:
                arr = series.to_numpy()
                mean = float(arr.mean())
                std = float(arr.std())
                if std == 0:
                    n_outliers = 0
                else:
                    z_scores = np.abs((arr - mean) / std)
                    n_outliers = int((z_scores > zscore_threshold).sum())
                method_desc = f"z-score > {zscore_threshold}"

            rate = n_outliers / n_total
            passed = n_outliers == 0
            severity = Severity.INFO if passed else (
                Severity.WARNING if rate < 0.01 else Severity.CRITICAL
            )

            if passed:
                details = f"No outliers detected in '{col}' using {method_desc}."
            else:
                details = (
                    f"'{col}' has {n_outliers:,} potential outliers "
                    f"({rate:.2%} of non-null values) using {method_desc}. "
                    "Verify capping/flooring decisions are documented."
                )

            results.append(
                TestResult(
                    test_name=f"outliers_{col}",
                    category=TestCategory.DATA_QUALITY,
                    passed=passed,
                    metric_value=round(rate, 6),
                    details=details,
                    severity=severity,
                    extra={"column": col, "n_outliers": n_outliers, "method": method},
                )
            )

        return results

    def cardinality_check(
        self,
        max_categories: int = 50,
    ) -> list[TestResult]:
        """
        Check cardinality of categorical (string/boolean) columns.

        Very high cardinality columns (e.g. free-text fields, policy
        numbers) are likely data leakage risks or pre-processed
        incorrectly.

        Parameters
        ----------
        max_categories:
            Columns with more unique values than this are flagged.

        Returns
        -------
        list[TestResult]
            One result per categorical column.
        """
        results = []
        cat_cols = [
            col
            for col in self._df.columns
            if self._df[col].dtype in (pl.Utf8, pl.String, pl.Boolean, pl.Categorical)
        ]

        for col in cat_cols:
            n_unique = self._df[col].n_unique()
            passed = n_unique <= max_categories
            severity = Severity.INFO if passed else Severity.WARNING

            if passed:
                details = (
                    f"'{col}' has {n_unique} unique values, "
                    f"within the {max_categories} limit."
                )
            else:
                details = (
                    f"'{col}' has {n_unique} unique values, "
                    f"exceeding the limit of {max_categories}. "
                    "If this is a categorical feature, consider grouping. "
                    "If it is an identifier, it should not be in the model."
                )

            results.append(
                TestResult(
                    test_name=f"cardinality_{col}",
                    category=TestCategory.DATA_QUALITY,
                    passed=passed,
                    metric_value=float(n_unique),
                    details=details,
                    severity=severity,
                    extra={"column": col, "n_unique": n_unique},
                )
            )

        return results

    def summary_statistics(self) -> TestResult:
        """
        Produce a summary of dataset size and column types.

        Always returns a passing INFO result. The purpose is to give the
        reviewer basic orientation before the detail checks.

        Returns
        -------
        TestResult
        """
        n_rows, n_cols = self._df.shape
        dtype_counts: dict[str, int] = {}
        for col in self._df.columns:
            dtype_name = str(self._df[col].dtype)
            dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1

        dtype_summary = ", ".join(
            f"{k}: {v}" for k, v in sorted(dtype_counts.items())
        )

        return TestResult(
            test_name="summary_statistics",
            category=TestCategory.DATA_QUALITY,
            passed=True,
            metric_value=float(n_rows),
            details=(
                f"{self._name} contains {n_rows:,} rows and {n_cols} columns. "
                f"Column types: {dtype_summary}."
            ),
            severity=Severity.INFO,
            extra={
                "n_rows": n_rows,
                "n_cols": n_cols,
                "dtype_counts": dtype_counts,
                "columns": self._df.columns,
            },
        )
