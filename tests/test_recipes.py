"""Tests for the catalogue loader and the profile validator."""

import json
import os

import pytest

from memproj import config, recipes
from memproj.recipes import derive_attributes, validate, load_catalogue

CATALOGUE = str(config.CATALOGUE_PATH)


# --- derive_attributes ------------------------------------------------------

def test_plain_vegetable_dish_is_vegetarian():
    a = derive_attributes({"ingredients": ["carrot", "onion", "rice"], "tags": []})
    assert a["vegetarian"] is True


def test_meat_makes_it_not_vegetarian():
    a = derive_attributes({"ingredients": ["chicken", "rice"], "tags": []})
    assert a["vegetarian"] is False


def test_sausage_counts_as_meat():
    a = derive_attributes({"ingredients": ["sausage", "potato"], "tags": []})
    assert a["vegetarian"] is False


def test_tag_can_force_vegetarian():
    a = derive_attributes({"ingredients": ["tofu"], "tags": ["vegetarian"]})
    assert a["vegetarian"] is True


def test_chilli_raises_spice_level():
    a = derive_attributes({"ingredients": ["chilli", "tomato"], "tags": []})
    assert a["spice_level"] in ("medium", "spicy")


def test_no_chilli_is_mild():
    a = derive_attributes({"ingredients": ["carrot", "cream"], "tags": []})
    assert a["spice_level"] == "mild"


def test_spicy_tag_forces_spicy():
    a = derive_attributes({"ingredients": ["carrot"], "tags": ["spicy"]})
    assert a["spice_level"] == "spicy"


def test_ingredient_flags_are_lowercased_and_complete():
    a = derive_attributes({"ingredients": ["Mushrooms", "Peanut Butter"], "tags": []})
    assert "mushroom" in a["ingredients"]
    assert "peanut" in a["ingredients"]


# --- validate ---------------------------------------------------------------

def test_vegetarian_profile_rejects_meat():
    r = derive_attributes({"ingredients": ["chicken", "rice"], "tags": []})
    ok, violations = validate(r, {"diet": {"vegetarian"}})
    assert ok is False
    assert any("vegetarian" in v for v in violations)


def test_vegetarian_profile_accepts_veg():
    r = derive_attributes({"ingredients": ["carrot", "rice"], "tags": []})
    ok, violations = validate(r, {"diet": {"vegetarian"}})
    assert ok is True and violations == []


def test_spice_ceiling_is_respected():
    r = derive_attributes({"ingredients": ["chilli"], "tags": ["spicy"]})
    ok, violations = validate(r, {"spice_level": {"mild"}})
    assert ok is False
    assert any("spice" in v for v in violations)


def test_spice_exact_match_passes():
    r = derive_attributes({"ingredients": ["carrot"], "tags": []})  # mild
    ok, _ = validate(r, {"spice_level": {"mild"}})
    assert ok is True


def test_excluded_ingredient_is_rejected():
    r = derive_attributes({"ingredients": ["mushroom", "rice"], "tags": []})
    ok, violations = validate(r, {"excluded_ingredient": {"mushroom"}})
    assert ok is False
    assert any("mushroom" in v for v in violations)


def test_excluded_via_tag_is_rejected():
    r = derive_attributes({"ingredients": ["flour"], "tags": ["contains-nuts"]})
    ok, violations = validate(r, {"excluded_ingredient": {"peanut"}})
    assert ok is False


def test_empty_profile_accepts_anything():
    r = derive_attributes({"ingredients": ["chicken", "chilli"], "tags": []})
    ok, violations = validate(r, {})
    assert ok is True and violations == []


def test_multiple_violations_all_reported():
    r = derive_attributes({"ingredients": ["chicken", "mushroom", "chilli"],
                           "tags": ["spicy"]})
    ok, violations = validate(
        r, {"diet": {"vegetarian"}, "excluded_ingredient": {"mushroom"},
            "spice_level": {"mild"}}
    )
    assert ok is False
    assert len(violations) >= 3


# --- catalogue --------------------------------------------------------------

def test_shipped_catalogue_loads():
    cat = load_catalogue(CATALOGUE)
    assert len(cat) >= 60


def test_shipped_recipes_have_required_fields():
    for r in load_catalogue(CATALOGUE):
        assert set(r) >= {"id", "name", "ingredients", "tags"}


def test_catalogue_ids_are_unique():
    ids = [r["id"] for r in load_catalogue(CATALOGUE)]
    assert len(ids) == len(set(ids))


def test_catalogue_covers_both_diets():
    cat = [derive_attributes(r) for r in load_catalogue(CATALOGUE)]
    assert any(a["vegetarian"] for a in cat)
    assert any(not a["vegetarian"] for a in cat)


def test_catalogue_covers_all_spice_levels():
    levels = {derive_attributes(r)["spice_level"] for r in load_catalogue(CATALOGUE)}
    assert levels == {"mild", "medium", "spicy"}


def test_catalogue_offers_excludable_ingredients():
    """The corpus excludes these three by name, so the catalogue must carry
    them -- otherwise the exclusion constraint has nothing to reject."""
    flags = set()
    for r in load_catalogue(CATALOGUE):
        flags |= derive_attributes(r)["ingredients"]
    assert {"mushroom", "peanut", "aubergine"} <= flags


# --- the HUMMUS phrase defect (2026-10-29) ----------------------------------

def test_a_meat_phrase_is_not_vegetarian():
    """HUMMUS writes "chicken breast" rather than bare "chicken". An exact
    set intersection called 15% of a 20k sample vegetarian."""
    for name in ("chicken breast", "ground beef", "pork belly", "smoked bacon"):
        a = derive_attributes({"ingredients": [name], "tags": []})
        assert a["vegetarian"] is False, name


def test_a_vegetable_phrase_stays_vegetarian():
    for name in ("bell pepper", "beefsteak tomato", "graham crackers",
                 "butter beans", "goat cheese"):
        a = derive_attributes({"ingredients": [name], "tags": []})
        assert a["vegetarian"] is True, name


def test_plural_heat_sources_are_counted():
    a = derive_attributes({"ingredients": ["chillies", "jalapenos"], "tags": []})
    assert a["spice_level"] == "spicy"


def test_a_heat_source_inside_a_phrase_is_counted():
    a = derive_attributes({"ingredients": ["jalapeño peppers"], "tags": []})
    assert a["spice_level"] == "medium"
