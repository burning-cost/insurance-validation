"""
Pydantic model card schema for UK insurance pricing models.

The model card is the anchor document for the validation report. It
captures what the model is, who built it, what it is allowed to do, and
what its known limitations are. This maps directly to SS1/23 Principle 3
(Model development, implementation and use) and the FCA's requirement for
Consumer Duty fair value documentation.

Usage
-----
    from insurance_validation import ModelCard

    card = ModelCard(
        model_name="Motor Third-Party Property Damage Frequency",
        version="2.1.0",
        purpose="Estimate expected claim frequency for private motor policies",
        intended_use="Underwriting pricing, not claims reserving",
        developer="Pricing Team",
        development_date="2024-09-01",
        limitations="Out-of-sample performance degrades for vehicles >10 years old",
        materiality_tier=2,
        approved_by=["Chief Actuary", "Model Risk Committee"],
        variables=["vehicle_age", "driver_age", "annual_mileage", "region"],
        target_variable="claim_count",
        model_type="GLM",
        distribution_family="Poisson",
    )
"""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ModelCard(BaseModel):
    """
    Structured metadata for an insurance pricing model.

    All fields are validated on construction. Required fields must be
    non-empty strings or non-empty lists. This forces the team to
    document their model before running validation - not as an
    afterthought.
    """

    model_name: str = Field(
        ...,
        description="Full descriptive name of the model, e.g. "
        "'Motor TPPD Frequency v2.1'",
        min_length=1,
    )
    version: str = Field(
        ...,
        description="Version string, e.g. '2.1.0'",
        min_length=1,
    )
    purpose: str = Field(
        ...,
        description="One or two sentences stating what the model does and "
        "what business decision it supports.",
        min_length=10,
    )
    intended_use: str = Field(
        ...,
        description="Scope of permitted use. Explicitly state what the model "
        "should NOT be used for.",
        min_length=5,
    )
    developer: str = Field(
        ...,
        description="Name of team or individual who built the model.",
        min_length=1,
    )
    development_date: date = Field(
        ...,
        description="Date the model was signed off for production use.",
    )
    limitations: str = Field(
        ...,
        description="Known limitations, failure modes, or out-of-scope "
        "populations. Must be explicit - omitting this field is not an option.",
        min_length=10,
    )
    materiality_tier: int = Field(
        ...,
        description="Model risk tier per internal classification (1=highest risk, "
        "3=lowest). Drives validation intensity and sign-off requirements.",
        ge=1,
        le=3,
    )
    approved_by: list[str] = Field(
        ...,
        description="List of named approvers with title, e.g. "
        "['Jane Smith - Chief Actuary', 'Model Risk Committee'].",
        min_length=1,
    )
    variables: list[str] = Field(
        ...,
        description="List of model input variables (features) used in production.",
        min_length=1,
    )
    target_variable: str = Field(
        ...,
        description="Name of the response variable, e.g. 'claim_count' or "
        "'incurred_loss'.",
        min_length=1,
    )
    model_type: Literal["GLM", "GBM", "GAM", "Neural Network", "Ensemble", "Other"] = Field(
        ...,
        description="High-level model family.",
    )
    distribution_family: str = Field(
        ...,
        description="Statistical distribution assumed for the response, "
        "e.g. 'Poisson', 'Gamma', 'Tweedie'. For GBMs, state the loss function.",
        min_length=1,
    )

    # Optional but recommended fields
    validation_date: date | None = Field(
        default=None,
        description="Date of this validation run.",
    )
    validator_name: str | None = Field(
        default=None,
        description="Name of the independent validator.",
    )
    model_description: str | None = Field(
        default=None,
        description="Extended description for the report narrative.",
    )
    alternatives_considered: str | None = Field(
        default=None,
        description="Alternative approaches evaluated during development and "
        "reasons for rejection.",
    )
    monitoring_frequency: str | None = Field(
        default=None,
        description="How often ongoing model monitoring is performed, "
        "e.g. 'Quarterly'.",
    )

    @model_validator(mode="after")
    def approved_by_must_be_non_empty_strings(self) -> ModelCard:
        for entry in self.approved_by:
            if not entry.strip():
                raise ValueError(
                    "Each entry in approved_by must be a non-empty string."
                )
        return self

    @model_validator(mode="after")
    def variables_must_be_non_empty_strings(self) -> ModelCard:
        for var in self.variables:
            if not var.strip():
                raise ValueError(
                    "Each variable name must be a non-empty string."
                )
        return self

    def summary(self) -> dict:
        """Return a flat dict suitable for the report summary table."""
        return {
            "Model name": self.model_name,
            "Version": self.version,
            "Model type": self.model_type,
            "Distribution": self.distribution_family,
            "Developer": self.developer,
            "Development date": str(self.development_date),
            "Materiality tier": self.materiality_tier,
            "Approved by": ", ".join(self.approved_by),
            "Target variable": self.target_variable,
            "Number of variables": len(self.variables),
        }
