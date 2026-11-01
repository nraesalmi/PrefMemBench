"""The evaluator.

The contract for this file: a copy of the gold predictions scores 100% on
every metric, and each deliberately corrupted copy fails in exactly one
known way. If a corruption moved two counters, one of them is noise and the
results table cannot be read. No test here calls a model.
"""

import json
import os

import pytest

from memproj import config
from memproj import recipes
from memproj.eval import metrics, report
from memproj.pipeline import run
from memproj.validate_predictions import validate_predictions

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
HISTORIES = os.path.join(FIXTURES, "histories.jsonl")
CHECKPOINTS = os.path.join(FIXTURES, "checkpoints.jsonl")


# --- state ------------------------------------------------------------------

def test_matching_profiles_are_exact():
    score = metrics.score({"spice_level": ["mild"]}, {"spice_level": ["mild"]})
    assert score["exact"] and score["fn"] == 0 and score["fp"] == 0
    assert score["precision"] == 1.0 and score["recall"] == 1.0


def test_a_wrong_value_costs_one_miss_and_one_false_positive():
    score = metrics.score({"spice_level": ["spicy"]}, {"spice_level": ["mild"]})
    assert not score["exact"]
    assert score["fn"] == 1 and score["fp"] == 1
    assert score["wrong_value"] == ["spice_level"]
    assert score["precision"] == 0.0 and score["recall"] == 0.0


def test_missing_and_extra_values_are_counted_separately():
    score = metrics.score(
        {"excluded_ingredient": ["peanut"], "spice_level": ["spicy"]},
        {"excluded_ingredient": ["mushroom"], "spice_level": ["mild"]},
    )
    assert score["missing"] == [("excluded_ingredient", "mushroom"),
                                ("spice_level", "mild")]
    assert score["extra"] == [("excluded_ingredient", "peanut"),
                              ("spice_level", "spicy")]


def test_saying_nothing_is_vacuously_precise_but_never_exact():
    score = metrics.score({}, {"spice_level": ["mild"]})
    assert not score["exact"]
    assert score["precision"] == 1.0
    assert score["recall"] == 0.0


def test_none_is_not_an_empty_profile():
    # None means "no structured opinion" and is the caller's signal to skip
    # the row; {} means "nothing is active" and is scored.
    assert metrics.profile_pairs(None) == set()
    assert metrics.score(None, {})["exact"] is True  # both empty


def test_a_value_replaced_by_two_others_costs_two_false_positives():
    # The step-3 copy reports tp=1 here (total - missing - wrong), crediting a
    # hit for a gold value the method never produced. Nothing matched.
    score = metrics.score({"excluded_ingredient": ["medium"]},
                          {"excluded_ingredient": ["mushroom", "spicy"]})
    assert score["tp"] == 0 and score["fn"] == 2 and score["fp"] == 1
    assert score["wrong_value"] == ["excluded_ingredient"]


def test_spelling_is_not_the_fact():
    # `recipes.validate` normalises before matching an ingredient and
    # `tools._check_value` normalises on write, so the evaluator must too --
    # otherwise a correct extraction scores as a miss, an extra and an
    # invented fact at once.
    assert metrics.score({"excluded_ingredient": ["Peanuts"]},
                         {"excluded_ingredient": ["peanut"]})["exact"]
    assert metrics.score({"spice_level": [" Mild "]},
                         {"spice_level": ["mild"]})["exact"]
    assert metrics.profile_pairs({"excluded_ingredient": ["Mushrooms"]}) == {
        ("excluded_ingredient", "mushroom")}


# --- action -----------------------------------------------------------------

CATALOGUE = [
    {"id": "r01", "name": "Mild Stew", "vegetarian": True,
     "spice_level": "mild", "ingredients": {"potato"}, "tags": []},
    {"id": "r02", "name": "Hot Curry", "vegetarian": True,
     "spice_level": "spicy", "ingredients": {"chilli"}, "tags": []},
]

MILD_CHECKPOINT = {"checkpoint_id": "c0", "gold_profile": {"spice_level": ["mild"]},
                   "gold_action": "recommend", "now": "2026-03-01T19:00:00",
                   "person": "user", "relevant_event_ids": [],
                   "distractor_event_ids": []}


def prediction(**overrides):
    row = {"trajectory_id": "p001_base", "checkpoint_id": "c0",
           "method": "gold", "model": "gold", "profile": {"spice_level": ["mild"]},
           "action": "recommend", "recipe_id": "r01", "question": None,
           "explanation": "", "cited_turns": [], "tokens": 0,
           "latency_s": None}
    row.update(overrides)
    return row


def test_a_violating_recipe_fails_satisfaction():
    result = metrics.satisfaction(prediction(recipe_id="r02"), MILD_CHECKPOINT,
                                  CATALOGUE)
    assert result["applicable"] and not result["satisfied"]
    assert "exceeds" in result["violations"][0]


def test_an_unknown_recipe_fails_rather_than_crashing():
    result = metrics.satisfaction(prediction(recipe_id="r99"), MILD_CHECKPOINT,
                                  CATALOGUE)
    assert result["satisfied"] is False


def test_a_clarification_is_not_an_unsatisfied_recommendation():
    result = metrics.satisfaction(
        prediction(action="clarify", recipe_id=None, question="Which?"),
        MILD_CHECKPOINT, CATALOGUE)
    assert result == {"applicable": False, "satisfied": None, "violations": []}


@pytest.mark.parametrize("gold, got, expected", [
    ("clarify", "clarify", "appropriate"),
    ("clarify", "recommend", "missed"),
    ("recommend", "clarify", "unnecessary"),
    ("recommend", "recommend", "none"),
])
def test_clarification_outcomes(gold, got, expected):
    assert metrics.clarification_outcome({"action": got},
                                         {"gold_action": gold}) == expected


# --- contamination ----------------------------------------------------------

EVENTS = [
    {"event_id": "e1", "type": "initial_pref", "subject": "user",
     "attribute": "spice_level", "value": "mild", "scope": "permanent",
     "ts": "2026-03-01T10:00:00", "valid_from": "2026-03-01T10:00:00",
     "valid_until": None},
    {"event_id": "e2", "type": "temporary_exception", "subject": "user",
     "attribute": "spice_level", "value": "spicy", "scope": "temporary",
     "ts": "2026-03-01T10:07:00", "valid_from": "2026-03-01T10:07:00",
     "valid_until": "2026-03-01T18:00:00"},
    {"event_id": "e3", "type": "other_person", "subject": "brother",
     "attribute": "spice_level", "value": "medium", "scope": "permanent",
     "ts": "2026-03-01T10:08:00", "valid_from": "2026-03-01T10:08:00",
     "valid_until": None},
    {"event_id": "e4", "type": "permanent_revision", "subject": "user",
     "attribute": "excluded_ingredient", "value": "peanut", "scope": "permanent",
     "ts": "2026-03-01T10:09:00", "valid_from": "2026-03-01T10:09:00",
     "valid_until": None},
]

TURNS = [
    {"turn_id": "t0001", "speaker": "user", "event_id": "e1",
     "ts": "2026-03-01T10:05:00", "text": "I like mild food."},
    {"turn_id": "t0002", "speaker": "assistant", "event_id": None,
     "ts": "2026-03-01T10:06:00", "text": "Noted."},
    {"turn_id": "t0003", "speaker": "user", "event_id": "e2",
     "ts": "2026-03-01T10:07:00", "text": "Spicy is fine tonight."},
    {"turn_id": "t0004", "speaker": "user", "event_id": "e3",
     "ts": "2026-03-01T10:08:00", "text": "My brother likes spicy."},
    {"turn_id": "t0005", "speaker": "user", "event_id": None,
     "ts": "2026-03-01T10:09:00", "text": "Nice weather."},
    {"turn_id": "t0006", "speaker": "user", "event_id": "e1",
     "ts": "2026-03-02T10:00:00", "text": "Still mild."},  # after the checkpoint
]

TRAJECTORY = {"trajectory_id": "p001_base", "pair_id": "p001", "variant": "base",
              "family": "temp_vs_perm", "flipped_field": "scope",
              "events": EVENTS, "turns": TURNS}

CHECKPOINT = {"checkpoint_id": "c0", "trajectory_id": "p001_base",
              "now": "2026-03-01T19:00:00", "person": "user",
              "request_text": "What should I cook?",
              "gold_profile": {"spice_level": ["mild"]},
              "gold_action": "recommend", "relevant_event_ids": ["e1"],
              "distractor_event_ids": ["e2", "e3"], "kind": "temp_expired"}


def contamination(profile):
    return metrics.contamination(prediction(profile=profile),
                                 CHECKPOINT, TRAJECTORY)


def test_an_expired_exception_is_named_as_such():
    findings = contamination({"spice_level": ["mild", "spicy"]})
    assert findings == [{"attribute": "spice_level", "value": "spicy",
                         "event_id": "e2", "source": "expired_exception"}]


def test_another_persons_preference_is_named_as_such():
    findings = contamination({"spice_level": ["mild", "medium"]})
    assert findings[0]["source"] == "other_person"
    assert findings[0]["event_id"] == "e3"


def test_a_superseded_value_is_named_as_such():
    findings = contamination({"spice_level": ["mild"],
                              "excluded_ingredient": ["peanut"]})
    assert findings == [{"attribute": "excluded_ingredient", "value": "peanut",
                         "event_id": "e4", "source": "superseded"}]


def test_a_fact_no_event_explains_is_unsupported():
    findings = contamination({"spice_level": ["mild"], "diet": ["vegetarian"]})
    assert findings == [{"attribute": "diet", "value": "vegetarian",
                         "event_id": None, "source": "unsupported"}]


def test_contamination_finds_the_event_through_the_spelling():
    # e4 is the superseded peanut statement; "Peanuts" is the same fact.
    findings = contamination({"excluded_ingredient": ["Peanuts"]})
    assert findings == [{"attribute": "excluded_ingredient", "value": "peanut",
                         "event_id": "e4", "source": "superseded"}]


def test_a_correct_profile_has_no_contamination():
    assert contamination({"spice_level": ["mild"]}) == []


def test_free_text_rows_have_no_profile_to_contaminate():
    assert contamination(None) == []


# --- provenance -------------------------------------------------------------

def provenance(*cited):
    return metrics.provenance(prediction(cited_turns=list(cited)),
                              CHECKPOINT, TRAJECTORY)


def test_relevant_user_turns_are_clean_provenance():
    assert provenance("t0001") == []


def test_an_assistant_turn_is_not_evidence():
    assert "assistant turn" in provenance("t0002")[0]


def test_a_filler_turn_is_not_evidence():
    assert "no preference statement" in provenance("t0005")[0]


def test_citing_a_distractor_is_a_problem():
    assert "distractor" in provenance("t0003")[0]


def test_citing_the_other_persons_statement_is_a_problem():
    assert "brother" in provenance("t0004")[0]


def test_citing_a_later_turn_is_a_problem():
    assert "after the checkpoint" in provenance("t0006")[0]


def test_an_unknown_turn_id_is_reported_not_crashed():
    assert "not a turn" in provenance("t9999")[0]


# --- end to end -------------------------------------------------------------

@pytest.fixture(scope="module")
def fixture_corpus():
    return report.load_inputs(os.path.join(FIXTURES, "predictions_valid.jsonl"),
                              CHECKPOINTS, HISTORIES)


def test_gold_predictions_score_perfectly_on_the_fixtures(tmp_path):
    path = str(tmp_path / "gold.jsonl")
    run.run(["gold"], out_path=path, histories_path=HISTORIES,
            checkpoints_path=CHECKPOINTS)
    assert validate_predictions(path, histories_path=HISTORIES,
                                checkpoints_path=CHECKPOINTS) == []

    rows = report.evaluate_path(path, CHECKPOINTS, HISTORIES)
    summary = report.summarise(rows)["gold"]
    assert summary["state_rows"] == summary["rows"]
    assert summary["state_exact"] == summary["rows"]
    assert summary["satisfied"] == summary["recommend_rows"]
    assert summary["contamination"] == 0
    assert summary["provenance_problems"] == 0


def synthetic_corpus():
    """One base trajectory with both a distractor window and a distractor
    speaker, so a corrupted copy can trip exactly one metric."""
    return {
        "predictions": [prediction(recipe_id="r01", cited_turns=["t0001"],
                                   profile={"spice_level": ["mild"]})],
        "checkpoints": [dict(CHECKPOINT)],
        "histories": [dict(TRAJECTORY)],
        "catalogue": CATALOGUE,
    }


def score_one(corpus):
    return report.evaluate(**corpus)[0]


def test_the_synthetic_baseline_is_clean():
    row = score_one(synthetic_corpus())
    assert row["state"]["exact"] and row["satisfied"]
    assert row["contamination"] == [] and row["provenance"] == []


def test_a_violating_recipe_moves_only_satisfaction():
    corpus = synthetic_corpus()
    corpus["predictions"][0]["recipe_id"] = "r02"
    row = score_one(corpus)
    assert row["state"]["exact"] is True
    assert row["satisfied"] is False
    assert row["provenance"] == [] and row["contamination"] == []


def test_an_extra_fact_moves_only_state_and_contamination():
    corpus = synthetic_corpus()
    corpus["predictions"][0]["profile"] = {"spice_level": ["mild", "spicy"]}
    row = score_one(corpus)
    assert row["state"]["exact"] is False
    assert row["satisfied"] is True  # the pick itself is still mild
    assert [f["source"] for f in row["contamination"]] == ["expired_exception"]
    assert row["provenance"] == []


def test_citing_a_distractor_moves_only_provenance():
    corpus = synthetic_corpus()
    corpus["predictions"][0]["cited_turns"] = ["t0003"]
    row = score_one(corpus)
    assert row["state"]["exact"] is True and row["satisfied"] is True
    assert "distractor" in row["provenance"][0]


def test_an_unnecessary_clarification_moves_only_the_clarify_counter():
    corpus = synthetic_corpus()
    corpus["predictions"][0].update(action="clarify", recipe_id=None,
                                    question="Which one?")
    row = score_one(corpus)
    assert row["clarification"] == "unnecessary"
    assert row["satisfied"] is None
    assert row["state"]["exact"] is True


def test_dissociation_separates_recalling_from_acting():
    corpus = synthetic_corpus()
    corpus["predictions"].append({
        **prediction(recipe_id="r02"),
        "checkpoint_id": "c0", "method": "proposed",
    })
    rows = report.evaluate(**corpus)
    cells = report.dissociation(rows)
    assert cells["gold"] == {"both": 1, "recalled_but_violated": 0,
                             "lucky": 0, "both_wrong": 0}
    assert cells["proposed"] == {"both": 0, "recalled_but_violated": 1,
                                 "lucky": 0, "both_wrong": 0}


# --- pairing ----------------------------------------------------------------

def twin_corpus():
    """A base with one checkpoint and a matched with two, mirroring
    `subject_swap`'s asymmetry (README, Data formats)."""
    base = {"trajectory_id": "p001_base", "pair_id": "p001", "variant": "base",
            "family": "subject_swap", "flipped_field": "subject",
            "events": [], "turns": []}
    matched = dict(base, trajectory_id="p001_matched", variant="matched")
    checkpoints = [
        {"checkpoint_id": "base_user", "trajectory_id": "p001_base",
         "now": CHECKPOINT["now"], "person": "user", "kind": "for_user",
         "gold_profile": {}, "gold_action": "recommend",
         "relevant_event_ids": [], "distractor_event_ids": []},
        {"checkpoint_id": "matched_user", "trajectory_id": "p001_matched",
         "now": CHECKPOINT["now"], "person": "user", "kind": "for_user",
         "gold_profile": {}, "gold_action": "recommend",
         "relevant_event_ids": [], "distractor_event_ids": []},
        {"checkpoint_id": "matched_other", "trajectory_id": "p001_matched",
         "now": CHECKPOINT["now"], "person": "brother",
         "kind": "for_other_person", "gold_profile": {},
         "gold_action": "recommend", "relevant_event_ids": [],
         "distractor_event_ids": []},
    ]
    predictions = [
        prediction(trajectory_id="p001_base", checkpoint_id="base_user",
                   profile={}),
        prediction(trajectory_id="p001_matched", checkpoint_id="matched_user",
                   profile={}),
        prediction(trajectory_id="p001_matched", checkpoint_id="matched_other",
                   profile={}),
    ]
    return {"predictions": predictions, "checkpoints": checkpoints,
            "histories": [base, matched], "catalogue": CATALOGUE}


def test_pairing_aligns_twins_on_family_pair_person_kind():
    rows = report.evaluate(**twin_corpus())
    pairs = report.paired_table(rows)
    assert len(pairs) == 1  # for_other_person has no base counterpart
    assert pairs[0]["kind"] == "for_user"
    assert pairs[0]["person"] == "user"


def test_paired_summary_tolerates_the_unpaired_kind():
    rows = report.evaluate(**twin_corpus())
    summary = report.paired_summary(rows)
    assert [(item["method"], item["family"], item["pairs"])
            for item in summary] == [("gold", "subject_swap", 1)]


def test_a_corpus_without_twins_has_no_paired_table():
    """The challenge set is one history per question, not a twin pair.

    `paired_table` pairs base against matched; a corpus whose variant is
    neither (the hand-written challenge set, variant "challenge") has
    nothing to pair, and must render every other table rather than raise.
    """
    corpus = twin_corpus()
    for history in corpus["histories"]:
        history["variant"] = "challenge"

    rows = report.evaluate(**corpus)

    assert report.paired_table(rows) == []
    assert report.paired_summary(rows) == []


def test_a_duplicated_twin_is_refused_not_silently_dropped():
    corpus = twin_corpus()
    corpus["predictions"].append(dict(corpus["predictions"][1]))
    rows = report.evaluate(**corpus)
    with pytest.raises(ValueError, match="answered this question 2 times"):
        report.paired_table(rows)


def test_a_row_from_another_trajectory_is_refused():
    corpus = synthetic_corpus()
    corpus["predictions"][0]["trajectory_id"] = "p999_base"
    with pytest.raises(ValueError, match="belongs to p001_base"):
        report.evaluate(**corpus)


# --- one file, two models ---------------------------------------------------

def test_two_models_under_one_method_are_not_merged():
    corpus = synthetic_corpus()
    corpus["predictions"].append(dict(corpus["predictions"][0],
                                      model="other-model", recipe_id="r02"))
    rows = report.evaluate(**corpus)
    summary = report.summarise(rows)
    assert sorted(summary) == ["gold", "gold [other-model]"]
    # The bad pick is attributed to the model that made it, not averaged in.
    assert (summary["gold"]["satisfied"]
            == summary["gold"]["recommend_rows"] == 1)
    assert summary["gold [other-model]"]["satisfied"] == 0


def test_a_single_model_keeps_the_plain_method_label():
    assert sorted(report.summarise(report.evaluate(**synthetic_corpus()))) \
        == ["gold"]


# --- rendering --------------------------------------------------------------

def test_render_produces_all_three_tables():
    rows = report.evaluate(**synthetic_corpus())
    rendered = report.render(rows)
    assert "## Per method" in rendered
    assert "## State vs action" in rendered
    assert "## Matched variants" in rendered
    assert "| gold |" in rendered


def test_cli_prints_and_writes_markdown(tmp_path):
    predictions = str(tmp_path / "gold.jsonl")
    run.run(["gold"], out_path=predictions, histories_path=HISTORIES,
            checkpoints_path=CHECKPOINTS)
    markdown = str(tmp_path / "report.md")
    assert report.main([predictions, "--histories", HISTORIES,
                        "--checkpoints", CHECKPOINTS,
                        "--markdown", markdown]) == 0
    with open(markdown, encoding="utf-8") as fh:
        assert "Per method" in fh.read()
