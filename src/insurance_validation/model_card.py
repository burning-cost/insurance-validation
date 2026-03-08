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

New simplified API (also accepted)
-----------------------------------
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

    Supports two field naming conventions:
    - Legacy: model_name, developer, variables, approved_by
    - Simplified: name, owner, features (with automatic mapping)
    """

    # Primary name field (legacy)
    model_name: str | None = Field(
        default=None,
        description="Full descriptive name of the model, e.g. "
        "'Motor TPPD Frequency v2.1'",
    )
    # Simplified API alias
    name: str | None = Field(
        default=None,
        description="Model name (simplified API alias for model_name).",
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
        min_length=5,
    )
    intended_use: str | None = Field(
        default=None,
        description="Scope of permitted use. Explicitly state what the model "
        "should NOT be used for.",
    )
    # Legacy developer field
    developer: str | None = Field(
        default=None,
        description="Name of team or individual who built the model.",
    )
    # Simplified API alias
    owner: str | None = Field(
        default=None,
        description="Model owner / developer (simplified API alias for developer).",
    )
    development_date: date | None = Field(
        default=None,
        description="Date the model was signed off for production use.",
    )
    # Limitations: accepts string or list for the simplified API
    limitations: str | list[str] | None = Field(
        default=None,
        description="Known limitations, failure modes, or out-of-scope "
        "populations. Must be explicit - omitting this field is not an option.",
    )
    materiality_tier: int | None = Field(
        default=None,
        description="Model risk tier per internal classification (1=highest risk, "
        "3=lowest). Drives validation intensity and sign-off requirements.",
        ge=1,
        le=3,
    )
    approved_by: list[str] | None = Field(
        default=None,
        description="List of named approvers with title, e.g. "
        "['Jane Smith - Chief Actuary', 'Model Risk Committee'].",
    )
    # Legacy variables field
    variables: list[str] | None = Field(
        default=None,
        description="List of model input variables (features) used in production.",
    )
    # Simplified API alias
    features: list[str] | None = Field(
        default=None,
        description="Feature list (simplified API alias for variables).",
    )
    # Legacy target_variable field
    target_variable: str | None = Field(
        default=None,
        description="Name of the response variable, e.g. 'claim_count' or "
        "'incurred_loss'.",
    )
    # Simplified API alias
    target: str | None = Field(
        default=None,
        description="Target variable (simplified API alias for target_variable).",
    )
    model_type: Literal["GLM", "GBM", "GAM", "Neural Network", "Ensemble", "Other"] | None = Field(
        default=None,
        description="High-level model family.",
    )
    distribution_family: str | None = Field(
        default=None,
        description="Statistical distribution assumed for the response, "
        "e.g. 'Poisson', 'Gamma', 'Tweedie'. For GBMs, state the loss function.",
    )
    # Simplified API: methodology replaces distribution_family when provided
    methodology: str | None = Field(
        default=None,
        description="Model methodology description (used in simplified API, "
        "populates distribution_family if not set).",
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
    outstanding_issues: list[str] | None = Field(
        default=None,
        description="Known outstanding issues that require resolution before or "
        "shortly after production sign-off.",
    )
    monitoring_owner: str | None = Field(
        default=None,
        description="Named owner responsible for ongoing model monitoring.",
    )
    monitoring_triggers: dict[str, float] | None = Field(
        default=None,
        description="Metric names and threshold values that trigger a model review, "
        "e.g. {'psi_score': 0.25, 'ae_ratio_deviation': 0.10}.",
    )

    @model_validator(mode="after")
    def _normalise_aliases(self) -> "ModelCard":
        """Resolve simplified API aliases to legacy field names."""
        # model_name / name
        if self.model_name is None and self.name is not None:
            self.model_name = self.name
        if self.model_name is None:
            raise ValueError("Either 'model_name' or 'name' must be provided.")
        if not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string.")

        # developer / owner
        if self.developer is None and self.owner is not None:
            self.developer = self.owner

        # variables / features
        if self.variables is None and self.features is not None:
            self.variables = self.features

        # target_variable / target
        if self.target_variable is None and self.target is not None:
            self.target_variable = self.target

        # distribution_family / methodology
        if self.distribution_family is None and self.methodology is not None:
            self.distribution_family = self.methodology

        # Normalise limitations list -> string
        if isinstance(self.limitations, list):
            self.limitations = "; ".join(self.limitations)

        # Validate approved_by entries
        if self.approved_by is not None:
            for entry in self.approved_by:
                if not str(entry).strip():
                    raise ValueError("Each entry in approved_by must be a non-empty string.")

        # Validate variables entries
        if self.variables is not None:
            for var in self.variables:
                if not str(var).strip():
                    raise ValueError("Each variable name must be a non-empty string.")

        return self

    def get_effective_model_name(self) -> str:
        return self.model_name or self.name or "Unknown"

    def get_effective_developer(self) -> str:
        return self.developer or self.owner or "Not specified"

    def get_effective_variables(self) -> list[str]:
        return self.variables or self.features or []

    def get_effective_target(self) -> str:
        return self.target_variable or self.target or "Not specified"

    def get_effective_distribution(self) -> str:
        return self.distribution_family or self.methodology or "Not specified"

    def get_effective_limitations(self) -> str:
        if isinstance(self.limitations, list):
            return "; ".join(self.limitations)
        return self.limitations or "None documented"

    def summary(self) -> dict:
        """Return a flat dict suitable for the report summary table."""
        return {
            "Model name": self.get_effective_model_name(),
            "Version": self.version,
            "Model type": self.model_type or "Not specified",
            "Distribution": self.get_effective_distribution(),
            "Developer": self.get_effective_developer(),
            "Development date": str(self.development_date) if self.development_date else "Not specified",
            "Materiality tier": self.materiality_tier if self.materiality_tier is not None else "Not specified",
            "Approved by": ", ".join(self.approved_by) if self.approved_by else "Pending",
            "Target variable": self.get_effective_target(),
            "Number of variables": len(self.get_effective_variables()),
        }
