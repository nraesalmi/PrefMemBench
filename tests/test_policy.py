"""The recommender-policy ablation: no model call may appear here.

The candidate filter is a design decision, not a fact, and the report cannot
see what it hides: a wrong profile that leaves one safe dish in the pool
scores "satisfied" exactly like a right profile. `policy.available()` asks
the question the action column cannot -- was a safe choice offered at all --
for both policies, from the profiles a run already produced.
"""

import json

import pytest

from memproj.pipeline import policy, recommend


# Derived rows, as `recipes.load_catalogue` produces them: `ingredients` is a
# set and `validate()` intersects it directly.
CATALOGUE = [
    {"id": "r1", "name": "Rice Bowl", "spice_level": "mild",
     "vegetarian": True, "ingredients": {"rice"}, "tags": []},
    {"id": "r2", "name": "Rice Hotpot", "spice_level": "spicy",
     "vegetarian": True, "ingredients": {"rice"}, "tags": []},
    {"id": "r3", "name": "Beef Stew", "spice_level": "mild",
     "vegetarian": False, "ingredients": {"beef"}, "tags": []},
    {"id": "r4", "name": "Beef Chilli", "spice_level": "spicy",
     "vegetarian": False, "ingredients": {"beef"}, "tags": []},
]

# The same four as RAW catalogue rows, for the CLI, which loads from disk and
# derives the attributes itself.
RAW_CATALOGUE = [
    {"id": "r1", "name": "Rice Bowl", "ingredients": ["rice"],
     "tags": ["vegetarian"]},
    {"id": "r2", "name": "Rice Hotpot", "ingredients": ["rice", "chilli"],
     "tags": ["vegetarian", "spicy"]},
    {"id": "r3", "name": "Beef Stew", "ingredients": ["beef"], "tags": []},
    {"id": "r4", "name": "Beef Chilli", "ingredients": ["beef", "chilli"],
     "tags": ["spicy"]},
]

GOLD = {"spice_level": ["mild"], "diet": ["vegetarian"]}   # only r1 satisfies it

# A profile that rejects the ONLY safe dish. A `spicy` ceiling would not do:
# the ceiling means "at most this hot", so `spicy` admits everything.
WRONG = {"excluded_ingredient": ["rice"]}                  # leaves r3, r4


def _row(method="proposed", profile=None, action="recommend"):
    return {"method": method, "action": action, "profile": profile,
            "checkpoint_id": "c1", "trajectory_id": "t1"}


def _checkpoint(gold=None):
    return {"checkpoint_id": "c1", "trajectory_id": "t1",
            "gold_profile": dict(GOLD if gold is None else gold)}


# --- the policy parameter itself --------------------------------------------

def test_filter_offers_only_what_the_profile_accepts():
    profile = {"spice_level": ["mild"]}
    got = recommend.recipe_search(profile, CATALOGUE, top_k=None, policy="filter")
    assert {r["id"] for r in got} == {"r1", "r3"}


def test_rank_offers_the_whole_catalogue():
    profile = {"spice_level": ["mild"]}
    got = recommend.recipe_search(profile, CATALOGUE, top_k=None, policy="rank")
    assert {r["id"] for r in got} == {"r1", "r2", "r3", "r4"}


def test_rank_puts_the_valid_recipes_first():
    profile = {"spice_level": ["mild"]}
    got = recommend.recipe_search(profile, CATALOGUE, top_k=2, policy="rank")
    assert {r["id"] for r in got} == {"r1", "r3"}


def test_an_unknown_policy_is_an_error_not_a_silent_default():
    with pytest.raises(ValueError):
        recommend.recipe_search(GOLD, CATALOGUE, policy="everything")


def test_both_policies_ignore_a_None_profile_the_same_way():
    a = recommend.recipe_search(None, CATALOGUE, top_k=None, policy="filter")
    b = recommend.recipe_search(None, CATALOGUE, top_k=None, policy="rank")
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_an_empty_profile_offers_everything_under_both_policies():
    for p in policy.POLICIES:
        got = recommend.recipe_search({}, CATALOGUE, top_k=None, policy=p)
        assert len(got) == len(CATALOGUE), p


# --- what the ablation measures ---------------------------------------------

def test_a_right_profile_is_offered_a_safe_choice_under_both():
    for p in policy.POLICIES:
        assert policy.available(_row(profile=GOLD), _checkpoint(), CATALOGUE, p) is True


def test_a_wrong_profile_can_still_be_offered_a_safe_choice_under_rank_only():
    """The whole point: a profile that rejects the only safe dish makes the
    recommendation unwinnable under `filter` and merely harder under `rank`."""
    assert policy.available(_row(profile=WRONG), _checkpoint(), CATALOGUE,
                            "filter") is False
    assert policy.available(_row(profile=WRONG), _checkpoint(), CATALOGUE,
                            "rank") is True


def test_a_clarify_row_is_not_evidence_about_the_policy():
    for p in policy.POLICIES:
        assert policy.available(_row(action="clarify", profile=GOLD),
                                _checkpoint(), CATALOGUE, p) is None


def test_a_free_text_row_is_not_evidence_about_the_policy():
    for p in policy.POLICIES:
        assert policy.available(_row(profile=None), _checkpoint(), CATALOGUE, p) is None


# --- the table ---------------------------------------------------------------

def test_the_table_has_a_row_per_structured_method_and_policy():
    rows = policy.table([_row(profile=GOLD)], [_checkpoint()], CATALOGUE)
    assert [r["method"] for r in rows] == list(policy.STRUCTURED)
    for row in rows:
        for p in policy.POLICIES:
            assert p in row and p + "_offered" in row


def test_free_text_methods_are_not_in_the_table():
    rows = policy.table([_row(method="full_history", profile=None)],
                        [_checkpoint()], CATALOGUE)
    assert {r["method"] for r in rows} == set(policy.STRUCTURED)


def test_the_table_separates_the_two_policies_on_a_bad_profile():
    rows = policy.table([_row(profile=WRONG)], [_checkpoint()], CATALOGUE)
    proposed = [r for r in rows if r["method"] == "proposed"][0]
    assert proposed["filter"] == 1 and proposed["filter_offered"] == 0
    assert proposed["rank"] == 1 and proposed["rank_offered"] == 1


def test_the_table_is_deterministic_and_renders_markdown():
    rows = policy.table([_row(profile=GOLD)], [_checkpoint()], CATALOGUE)
    assert policy.render(rows) == policy.render(rows)
    assert "| latest_wins | filter |" in policy.render(rows)


def test_a_row_with_no_matching_checkpoint_is_skipped_not_crashed():
    orphan = dict(_row(profile=GOLD), checkpoint_id="nope")
    rows = policy.table([orphan], [_checkpoint()], CATALOGUE)
    assert all(r["filter"] == 0 for r in rows)


def test_the_cli_writes_what_it_prints(tmp_path, capsys):
    preds = tmp_path / "p.jsonl"
    preds.write_text(json.dumps(_row(profile=GOLD)) + "\n", encoding="utf-8")
    cps = tmp_path / "c.jsonl"
    cps.write_text(json.dumps(_checkpoint()) + "\n", encoding="utf-8")
    cat = tmp_path / "cat.json"
    cat.write_text(json.dumps(RAW_CATALOGUE), encoding="utf-8")
    out = tmp_path / "table.md"

    code = policy.main([str(preds), "--checkpoints", str(cps),
                        "--catalogue", str(cat), "--markdown", str(out)])
    assert code == 0
    assert out.read_text(encoding="utf-8") == capsys.readouterr().out.replace(
        f"\nwrote {out}\n", "")
