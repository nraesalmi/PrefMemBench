"""The fixed recommender and the gold baseline.

No test here calls a model. The model path is exercised through a fake chat
function, and the gold baseline is run against the hand-written fixture
corpus rather than `out/`, so these tests do not need a generated corpus.
"""

import json
import os
import random

import pytest

from memproj import config
from memproj import recipes
from memproj.pipeline import recommend, run
from memproj.validate_predictions import validate_predictions

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
HISTORIES = os.path.join(FIXTURES, "histories.jsonl")
CHECKPOINTS = os.path.join(FIXTURES, "checkpoints.jsonl")


@pytest.fixture(scope="module")
def catalogue():
    return recipes.load_derived(str(config.CATALOGUE_PATH))


@pytest.fixture(scope="module")
def corpus():
    return run.load_corpus(HISTORIES, CHECKPOINTS)


def ids(rows):
    return [row["id"] for row in rows]


def fake_chat(reply):
    """A chat callable that records the messages it was given."""
    calls = []

    def chat_fn(messages):
        calls.append(messages)
        return {"text": reply, "tokens": 123}

    chat_fn.calls = calls
    return chat_fn


# --- candidate policy (option a) -------------------------------------------

def test_free_text_profiles_offer_the_whole_catalogue(catalogue):
    for profile in (None, {}):
        everyone = recommend.recipe_search(profile, catalogue, top_k=None)
        assert len(everyone) == len(catalogue)
    assert len(recommend.recipe_search(None, catalogue)) == config.TOP_K_RECIPES


def test_structured_profile_offers_only_valid_recipes(catalogue):
    profile = {"spice_level": ["mild"]}
    candidates = recommend.recipe_search(profile, catalogue, top_k=None)
    expected = recipes.valid_recipes(catalogue, {"spice_level": {"mild"}})
    assert ids(candidates) == sorted(ids(expected))
    assert all(recipes.validate(recipe, {"spice_level": {"mild"}})[0]
               for recipe in candidates)
    assert len(candidates) < len(catalogue)


def test_exclusions_and_diet_filter_too(catalogue):
    for profile, as_sets in [
        ({"excluded_ingredient": ["mushroom", "peanut"]},
         {"excluded_ingredient": {"mushroom", "peanut"}}),
        ({"diet": ["vegetarian"], "spice_level": ["medium"]},
         {"diet": {"vegetarian"}, "spice_level": {"medium"}}),
    ]:
        candidates = recommend.recipe_search(profile, catalogue, top_k=None)
        assert candidates
        assert all(recipes.validate(recipe, as_sets)[0]
                   for recipe in candidates)


def test_headroom_is_the_untruncated_count(catalogue):
    profile = {"spice_level": ["mild"]}
    assert (recommend.headroom(profile, catalogue)
            == len(recommend.recipe_search(profile, catalogue, top_k=None)))
    assert recommend.headroom(None, catalogue) == len(catalogue)


def test_search_is_deterministic_and_order_independent(catalogue):
    profile = {"excluded_ingredient": ["aubergine"]}
    first = ids(recommend.recipe_search(profile, catalogue, top_k=None))
    shuffled = list(catalogue)
    random.Random(0).shuffle(shuffled)
    second = ids(recommend.recipe_search(profile, shuffled, top_k=None))
    assert first == second


def test_ranking_prefers_the_requested_heat_then_id(catalogue):
    mild = recommend.recipe_search({"spice_level": ["mild"]}, catalogue,
                                   top_k=None)
    assert mild[0]["spice_level"] == "mild"
    keys = [abs(recipes.SPICE_ORDER[r["spice_level"]] - 0) for r in mild]
    assert keys == sorted(keys)

    unfiltered = recommend.recipe_search(None, catalogue, top_k=None)
    assert ids(unfiltered) == sorted(ids(catalogue))


def test_validate_choice_wraps_recipes_validate():
    # The fixture catalogue, not config.CATALOGUE_PATH: this test is about the
    # wrapper, and a shipped catalogue rebuild must not break it.
    catalogue = recipes.load_derived(os.path.join(FIXTURES, "catalogue.json"))
    ok, violations = recommend.validate_choice("r01", {"spice_level": ["mild"]},
                                               catalogue)
    assert ok and violations == []
    ok, violations = recommend.validate_choice("nope", {}, catalogue)
    assert not ok
    assert "not in the catalogue" in violations[0]


def test_explain_names_every_constraint():
    recipe = {"id": "r01", "name": "Test Dish"}
    text = recommend.explain(recipe, {
        "diet": ["vegetarian"],
        "spice_level": ["mild"],
        "excluded_ingredient": ["mushroom"],
    })
    assert "vegetarian" in text
    assert "no hotter than mild" in text
    assert "mushroom" in text

    assert "no active restrictions" in recommend.explain(recipe, {})


# --- the model path, through a fake chat ------------------------------------

def test_parse_choice_accepts_fenced_json():
    reply = ('```json\n{"action": "recommend", "recipe_id": "r02", '
             '"question": null, "explanation": "Because it is mild."}\n```')
    choice = recommend.parse_choice(reply, ["r01", "r02"])
    assert choice["action"] == "recommend"
    assert choice["recipe_id"] == "r02"
    assert choice["question"] is None


def test_parse_choice_keeps_a_clarification():
    choice = recommend.parse_choice(
        '{"action": "clarify", "question": "Which restriction?", '
        '"explanation": "Two restrictions are active."}', ["r01"])
    assert choice == {"action": "clarify", "recipe_id": None,
                      "question": "Which restriction?",
                      "explanation": "Two restrictions are active."}


@pytest.mark.parametrize("reply, fragment", [
    ("no json here", "no JSON object"),
    ('{"action": "cook", "explanation": "x"}', "action must be"),
    ('{"action": "recommend", "recipe_id": "r99", "explanation": "x"}',
     "not one of the candidates"),
    ('{"action": "recommend", "recipe_id": "r01", "question": "Which?", '
     '"explanation": "x"}', "must not carry a question"),
    ('{"action": "clarify", "explanation": "x"}',
     "requires a question"),
    ('{"action": "clarify", "recipe_id": "r01", "question": "Which?", '
     '"explanation": "x"}', "must not carry a recipe_id"),
    ('{"action": "recommend", "recipe_id": "r01"}',
     "explanation must be a string"),
])
def test_parse_choice_rejects_malformed_replies(reply, fragment):
    with pytest.raises(ValueError, match=fragment):
        recommend.parse_choice(reply, ["r01", "r02"])


def test_recommend_offers_only_candidates_and_reports_tokens(catalogue):
    candidates = recommend.recipe_search({"spice_level": ["mild"]}, catalogue)
    chosen = candidates[1]["id"]
    chat_fn = fake_chat(json.dumps({
        "action": "recommend", "recipe_id": chosen,
        "question": None, "explanation": "It is mild.",
    }))

    choice = recommend.recommend({"spice_level": ["mild"]}, "context",
                                 "Got anything?", catalogue, chat_fn=chat_fn)

    assert choice["recipe_id"] == chosen
    assert choice["tokens"] == 123
    prompt = chat_fn.calls[0][1]["content"]
    assert "Got anything?" in prompt
    assert "context" in prompt
    for candidate in candidates:
        assert candidate["id"] in prompt
    for recipe in catalogue:
        if recipe not in candidates:
            assert f"- {recipe['id']}:" not in prompt


def test_recommend_can_clarify(catalogue):
    chat_fn = fake_chat('{"action": "clarify", "question": "Which one?", '
                        '"explanation": "Ambiguous."}')
    choice = recommend.recommend(None, "context", "request", catalogue,
                                 chat_fn=chat_fn)
    assert choice["action"] == "clarify"
    assert choice["recipe_id"] is None


def test_recommend_without_candidates_asks_and_makes_no_call():
    catalogue = [{"id": "r01", "name": "Chicken", "vegetarian": False,
                  "spice_level": "mild", "ingredients": {"chicken"},
                  "tags": []}]
    chat_fn = fake_chat("never called")
    choice = recommend.recommend({"diet": ["vegetarian"]}, "context", "request",
                                 catalogue, chat_fn=chat_fn)
    assert choice["action"] == "clarify"
    assert choice["tokens"] == 0
    assert chat_fn.calls == []


# --- the gold baseline -------------------------------------------------------

def test_gold_run_writes_a_valid_predictions_file(tmp_path, corpus):
    path = str(tmp_path / "predictions.jsonl")
    rows, stats = run.run(["gold"], out_path=path,
                          histories_path=HISTORIES,
                          checkpoints_path=CHECKPOINTS)
    assert stats["state_mismatches"] == []
    assert len(rows) == len(corpus[1])
    assert validate_predictions(path, histories_path=HISTORIES,
                                checkpoints_path=CHECKPOINTS,
                                catalogue_path=config.CATALOGUE_PATH) == []


def test_gold_profile_is_the_checkpoint_profile_verbatim(corpus):
    trajectories, checkpoints = corpus
    by_id = {t["trajectory_id"]: t for t in trajectories}
    catalogue = recipes.load_derived(str(config.CATALOGUE_PATH))
    for checkpoint in checkpoints:
        row = run.gold_prediction(by_id[checkpoint["trajectory_id"]],
                                  checkpoint, catalogue)
        assert row["profile"] == checkpoint["gold_profile"]
        assert row["profile"] is not checkpoint["gold_profile"]


def test_gold_recommendations_satisfy_the_profile(corpus):
    trajectories, checkpoints = corpus
    by_id = {t["trajectory_id"]: t for t in trajectories}
    catalogue = recipes.load_derived(str(config.CATALOGUE_PATH))
    for checkpoint in checkpoints:
        row = run.gold_prediction(by_id[checkpoint["trajectory_id"]],
                                  checkpoint, catalogue)
        if row["action"] == "recommend":
            ok, violations = recommend.validate_choice(
                row["recipe_id"], row["profile"], catalogue)
            assert ok, violations


def test_gold_rows_carry_no_cost(corpus):
    trajectories, checkpoints = corpus
    by_id = {t["trajectory_id"]: t for t in trajectories}
    catalogue = recipes.load_derived(str(config.CATALOGUE_PATH))
    for checkpoint in checkpoints:
        row = run.gold_prediction(by_id[checkpoint["trajectory_id"]],
                                  checkpoint, catalogue)
        assert row["tokens"] == 0
        assert row["latency_s"] is None
        assert row["model"] == "gold"


def test_gold_citations_belong_to_the_trajectory(corpus):
    trajectories, checkpoints = corpus
    by_id = {t["trajectory_id"]: t for t in trajectories}
    catalogue = recipes.load_derived(str(config.CATALOGUE_PATH))
    for checkpoint in checkpoints:
        trajectory = by_id[checkpoint["trajectory_id"]]
        row = run.gold_prediction(trajectory, checkpoint, catalogue)
        known = {turn["turn_id"] for turn in trajectory["turns"]}
        assert set(row["cited_turns"]) <= known
        assert row["cited_turns"]


def test_gold_clarify_names_the_live_restrictions(corpus):
    trajectories, _ = corpus
    trajectory = trajectories[0]
    checkpoint = {
        "checkpoint_id": "synthetic_c0",
        "trajectory_id": trajectory["trajectory_id"],
        "gold_action": "clarify",
        "gold_profile": {"excluded_ingredient": ["mushroom", "peanut"]},
        "relevant_event_ids": ["e1"],
    }
    row = run.gold_prediction(trajectory, checkpoint,
                              recipes.load_derived(str(config.CATALOGUE_PATH)))
    assert row["action"] == "clarify"
    assert row["recipe_id"] is None
    assert "mushroom" in row["question"] and "peanut" in row["question"]


def test_gold_context_implements_the_method_interface(corpus):
    trajectories, checkpoints = corpus
    trajectory, checkpoint = trajectories[0], checkpoints[0]
    context = run.gold_context(trajectory, checkpoint)
    assert set(context) == {"context_text", "profile", "evidence_turn_ids"}
    assert context["profile"] == checkpoint["gold_profile"]
    assert context["evidence_turn_ids"] == ["t0001"]
    assert "Preferences that apply now" in context["context_text"]


def test_a_structured_method_needs_its_builder_and_so_does_not_run_bare(corpus):
    # run() hands latest_wins/proposed to build_prediction only after it has
    # built an extraction cache; calling build_prediction directly
    # must say so rather than invent a profile.
    with pytest.raises(NotImplementedError, match=r"structured\(cache\)"):
        run.build_prediction({"trajectory_id": "p000_base",
                              "turns": []},
                             {"checkpoint_id": "p000_base_c0",
                              "trajectory_id": "p000_base"},
                             [], "proposed")


def test_headroom_summary_counts_open_checkpoints(catalogue):
    rows = [
        {"profile": {"spice_level": ["mild"]}},
        {"profile": None},
    ]
    summary = run.headroom_summary(rows, catalogue)
    assert summary["count"] == 2
    assert summary["fully_open"] == 1
    assert summary["min"] == recommend.headroom({"spice_level": ["mild"]},
                                               catalogue)
    assert summary["max"] == len(catalogue)


def test_cli_runs_and_reports(tmp_path, capsys):
    path = str(tmp_path / "predictions.jsonl")
    code = run.main(["--methods", "gold", "--out", path,
                     "--histories", HISTORIES,
                     "--checkpoints", CHECKPOINTS])
    assert code == 0
    out = capsys.readouterr().out
    assert "wrote 2 rows" in out
    assert "gold state" in out


# --- the transport retry (2026-10-25) ---------------------------------------
# A GPU run died on httpx.ReadTimeout after 180 s on one row, and 329 rows of
# journal went with it. The extraction loop already survives transport errors;
# `recommend.chat` is the other place a model call can stall, so it gets the
# same treatment: bounded retries, then raise.

def _fake_ollama(monkeypatch, replies):
    """Replace `ollama.Client` with a scripted one; count the attempts."""
    import sys
    import types

    attempts = []

    class FakeClient:
        def __init__(self, timeout=None):
            self.timeout = timeout

        def chat(self, **kwargs):
            attempts.append(kwargs)
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply

    monkeypatch.setitem(sys.modules, "ollama",
                        types.SimpleNamespace(Client=FakeClient))
    monkeypatch.setattr(recommend.time, "sleep", lambda seconds: None)
    return attempts


def _reply(text="It is mild."):
    return {"message": {"content": text}, "prompt_eval_count": 3,
            "eval_count": 4}


def test_a_transport_error_is_retried_and_the_call_survives(monkeypatch):
    import httpx

    attempts = _fake_ollama(monkeypatch,
                            [httpx.ReadTimeout("timed out"), _reply()])
    answer = recommend.chat([{"role": "user", "content": "hi"}])
    assert len(attempts) == 2
    assert answer == {"text": "It is mild.", "tokens": 7}
    # The retry re-asks the same question: same model tag, same options.
    assert attempts[0]["model"] == config.MODEL
    assert attempts[0]["options"] == config.ollama_options()


def test_transport_retries_are_bounded(monkeypatch):
    import httpx

    attempts = _fake_ollama(monkeypatch, [httpx.ReadTimeout("timed out")] * 10)
    with pytest.raises(httpx.ReadTimeout):
        recommend.chat([{"role": "user", "content": "hi"}])
    assert len(attempts) == config.TOOL_RETRIES + 1


# --- which role thinks (2026-10-25) ------------------------------------------
# qwen3's chain of thought does not converge on some open-ended prompts: the
# summariser was still generating after 150 s on a 434-character transcript
# and answered in 0.6 s with thinking off. The pipeline's own calls switch it
# off; the extraction loop keeps it (the extraction config). Both are uniform across
# the seven methods, so pin both call sites here.

def test_the_recommender_asks_for_no_chain_of_thought(monkeypatch):
    attempts = _fake_ollama(monkeypatch, [_reply()])
    recommend.chat([{"role": "user", "content": "hi"}])
    assert attempts[0]["think"] is config.PIPELINE_THINK


def test_a_programming_error_is_not_retried(monkeypatch):
    attempts = _fake_ollama(monkeypatch, [ValueError("bad reply")] * 10)
    with pytest.raises(ValueError):
        recommend.chat([{"role": "user", "content": "hi"}])
    assert len(attempts) == 1
