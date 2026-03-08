"""Tests for ModelCard Pydantic schema."""
import pytest
from datetime import date
from pydantic import ValidationError
from insurance_validation import ModelCard


def make_valid_card(**overrides):
    defaults = dict(
        model_name="Motor TPPD Frequency",
        version="1.0.0",
        purpose="Estimate expected claim frequency for private motor policies.",
        intended_use="Pricing only, not reserving.",
        developer="Pricing Team",
        development_date=date(2024, 1, 1),
        limitations="Degrades for high-mileage commercial vehicles.",
        materiality_tier=2,
        approved_by=["Chief Actuary"],
        variables=["driver_age", "vehicle_age", "region"],
        target_variable="claim_count",
        model_type="GLM",
        distribution_family="Poisson",
    )
    defaults.update(overrides)
    return ModelCard(**defaults)


def test_valid_card_constructs():
    card = make_valid_card()
    assert card.get_effective_model_name() == "Motor TPPD Frequency"
    assert card.materiality_tier == 2
    assert card.model_type == "GLM"


def test_model_type_enum_variants():
    for mt in ("GLM", "GBM", "GAM", "Neural Network", "Ensemble", "Other"):
        card = make_valid_card(model_type=mt)
        assert card.model_type == mt


def test_invalid_model_type_rejected():
    with pytest.raises(ValidationError):
        make_valid_card(model_type="RandomForest")


def test_materiality_tier_bounds():
    make_valid_card(materiality_tier=1)
    make_valid_card(materiality_tier=3)
    with pytest.raises(ValidationError):
        make_valid_card(materiality_tier=0)
    with pytest.raises(ValidationError):
        make_valid_card(materiality_tier=4)


def test_empty_model_name_rejected():
    with pytest.raises(ValidationError):
        make_valid_card(model_name="")


def test_short_purpose_rejected():
    with pytest.raises(ValidationError):
        make_valid_card(purpose="Hi")  # Too short (< 5 chars)


def test_empty_approved_by_entries():
    with pytest.raises(ValidationError):
        make_valid_card(approved_by=["  "])


def test_summary_returns_dict():
    card = make_valid_card()
    summary = card.summary()
    assert "Model name" in summary
    assert "Version" in summary
    assert "Materiality tier" in summary


def test_optional_fields_default_none():
    card = make_valid_card()
    assert card.validator_name is None
    assert card.validation_date is None
    assert card.model_description is None


def test_optional_fields_accepted():
    card = make_valid_card(
        validator_name="Jane Smith",
        validation_date=date(2024, 3, 1),
        model_description="A GLM fitted on 5 years of motor data.",
        alternatives_considered="GBM was considered but interpretability required GLM.",
        monitoring_frequency="Quarterly",
    )
    assert card.validator_name == "Jane Smith"
    assert card.validation_date == date(2024, 3, 1)


def test_date_string_coercion():
    # Pydantic v2 coerces date strings
    card = make_valid_card(development_date="2024-06-15")
    assert card.development_date == date(2024, 6, 15)


# ── Simplified API tests ────────────────────────────────────────────────────

def test_simplified_api_with_name_alias():
    card = ModelCard(
        name="Motor Frequency v3.2",
        version="3.2.0",
        purpose="Predict claim frequency for UK motor portfolio",
        methodology="CatBoost gradient boosting with Poisson objective",
        target="claim_count",
        features=["age", "vehicle_age", "area"],
        limitations=["No telematics data", "Limited fleet exposure"],
        owner="Pricing Team",
    )
    assert card.get_effective_model_name() == "Motor Frequency v3.2"
    assert card.get_effective_developer() == "Pricing Team"
    assert card.get_effective_variables() == ["age", "vehicle_age", "area"]
    assert card.get_effective_target() == "claim_count"
    assert card.get_effective_distribution() == "CatBoost gradient boosting with Poisson objective"


def test_limitations_list_coerces_to_string():
    card = ModelCard(
        name="Test Model",
        version="1.0.0",
        purpose="Test purpose long enough",
        features=["x"],
        target="y",
        limitations=["Limitation one", "Limitation two"],
        owner="Team",
    )
    lim = card.get_effective_limitations()
    assert "Limitation one" in lim
    assert "Limitation two" in lim


def test_simplified_api_summary():
    card = ModelCard(
        name="Test",
        version="1.0.0",
        purpose="Test purpose that is long enough",
        owner="Me",
        features=["a", "b"],
        target="z",
        limitations="None",
    )
    s = card.summary()
    assert s["Model name"] == "Test"
    assert s["Developer"] == "Me"


def test_no_model_name_or_name_raises():
    with pytest.raises(ValidationError):
        ModelCard(version="1.0.0", purpose="A purpose long enough")


def test_outstanding_issues_field():
    card = make_valid_card(
        outstanding_issues=["Issue 1: data gap", "Issue 2: pending sign-off"]
    )
    assert len(card.outstanding_issues) == 2


def test_monitoring_fields():
    card = make_valid_card(
        monitoring_owner="Jane Smith",
        monitoring_triggers={"psi_score": 0.25, "ae_ratio": 0.15},
        monitoring_frequency="Monthly",
    )
    assert card.monitoring_owner == "Jane Smith"
    assert card.monitoring_triggers["psi_score"] == 0.25
