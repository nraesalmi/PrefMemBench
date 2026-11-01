"""The runner's model path, exercised through a fake chat callable.

Everything here is plumbing: does the method's context reach the one
recommender, does a malformed reply get retried, does `--resume` actually
skip work, does the row match README, Predictions format. No test calls a model.
"""

import json
import os
import re

import pytest

from memproj import config
from memproj.pipeline import run
from memproj.validate_predictions import validate_predictions

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
HISTORIES = os.path.join(FIXTURES, "histories.jsonl")
CHECKPOINTS = os.path.join(FIXTURES, "checkpoints.jsonl")


def candidate_ids(prompt):
    # Ids are opaque catalogue strings, not "r01" -- see README, Recipe catalogue.
    return re.findall(r"^- ([^\s:]+):", prompt, re.M)


def reply(messages, tokens=11):
    """A valid choice of the first candidate in the prompt."""
    chosen = candidate_ids(messages[1]["content"])[0]
    return {"text": json.dumps({"action": "recommend", "recipe_id": chosen,
                                "question": None,
                                "explanation": "The first candidate fits."}),
            "tokens": tokens}


def build(tmp_path, methods=("recent_only",), chat_fn=reply, **kwargs):
    path = str(tmp_path / "predictions.jsonl")
    rows, stats = run.run(list(methods), out_path=path,
                          histories_path=HISTORIES,
                          checkpoints_path=CHECKPOINTS,
                          chat_fn=chat_fn, **kwargs)
    return rows, stats, path


def test_free_text_rows_go_through_the_shared_recommender(tmp_path):
    rows, stats, path = build(tmp_path)
    assert stats["rows_by_method"] == {"recent_only": 2}
    for row in rows:
        assert row["profile"] is None            # README, Method interface
        assert row["model"] == config.MODEL
        assert row["tokens"] == 11
        assert row["latency_s"] is not None and row["latency_s"] >= 0
        assert row["cited_turns"]                 # user turns with an event
    assert validate_predictions(path, histories_path=HISTORIES,
                                checkpoints_path=CHECKPOINTS) == []


def test_cited_turns_belong_to_the_trajectory(tmp_path):
    rows, _, _ = build(tmp_path)
    for row in rows:
        trajectory = next(
            line for line in map(json.loads, open(HISTORIES, encoding="utf-8"))
            if line["trajectory_id"] == row["trajectory_id"]
        )
        user_turns = {turn["turn_id"] for turn in trajectory["turns"]
                      if turn["speaker"] == "user"}
        assert set(row["cited_turns"]) <= user_turns


def test_the_model_flag_is_recorded(tmp_path):
    rows, _, _ = build(tmp_path, model="tiny-test-model")
    assert {row["model"] for row in rows} == {"tiny-test-model"}


def test_a_malformed_reply_is_retried_with_the_reason(tmp_path):
    calls = []

    def chat_fn(messages):
        calls.append(messages)
        if len(calls) == 1:
            return {"text": "I would suggest something mild!", "tokens": 5}
        return reply(messages)

    rows, _, _ = build(tmp_path, limit=1, chat_fn=chat_fn)
    assert len(calls) == 2
    assert "rejected" in calls[1][-1]["content"]
    assert "no JSON object" in calls[1][-1]["content"]
    assert rows[0]["action"] == "recommend"


def test_every_reply_unusable_still_writes_an_honest_line(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TOOL_RETRIES", 1)
    calls = []

    def chat_fn(messages):
        calls.append(messages)
        return {"text": "still not json", "tokens": 3}

    rows, _, path = build(tmp_path, limit=1, chat_fn=chat_fn)
    assert len(calls) == 2                     # initial attempt + one retry
    assert rows[0]["action"] == "clarify"
    assert rows[0]["question"]
    assert "rejected 2 times" in rows[0]["explanation"]
    # A missing row would quietly flatter the method; a clarification does not.
    assert len(rows) == 1
    assert validate_predictions(path, histories_path=HISTORIES,
                                checkpoints_path=CHECKPOINTS) == []


def test_resume_skips_completed_checkpoints(tmp_path):
    calls = []

    def chat_fn(messages):
        calls.append(messages)
        return reply(messages)

    _, first, _ = build(tmp_path, limit=1, chat_fn=chat_fn)
    rows, stats, _ = build(tmp_path, limit=2, chat_fn=chat_fn, resume=True)

    assert first["written"] == 1
    assert stats["resumed"] == 1
    assert stats["written"] == 1
    assert len(rows) == 2
    assert len(calls) == 2                     # one per generated row only


def test_gold_makes_no_model_call(tmp_path):
    def explode(messages):
        raise AssertionError("the gold baseline must not call a model")

    rows, _, _ = build(tmp_path, methods=("gold",), chat_fn=explode)
    assert {row["tokens"] for row in rows} == {0}
    assert {row["latency_s"] for row in rows} == {None}


def test_a_builder_with_the_wrong_shape_is_rejected(tmp_path):
    class Wrong:
        def build_context(self, trajectory, checkpoint):
            return {"context_text": "x", "profile": None}

    with pytest.raises(ValueError, match="Method interface"):
        run.method_prediction({"trajectory_id": "p000_base"},
                              {"checkpoint_id": "p000_base_c0",
                               "request_text": "hi"},
                              [], "recent_only", Wrong(), chat_fn=reply)


def test_build_prediction_alone_cannot_build_a_structured_method():
    # Step 6 wires latest_wins/proposed in through `builders`; without them
    # the error names where they come from instead of guessing a profile.
    with pytest.raises(NotImplementedError, match=r"structured\(cache\)"):
        run.build_prediction({"trajectory_id": "p000_base", "turns": []},
                             {"checkpoint_id": "p000_base_c0",
                              "trajectory_id": "p000_base"},
                             [], "proposed")


# --- the journal ------------------------------------------------------------

def test_a_fresh_run_does_not_inherit_an_old_journal(tmp_path):
    # Without --resume this is a new run, so the journal it reads back next
    # time must contain only what this run did.
    rows, stats, path = build(tmp_path, limit=1)
    journal = path + ".journal"
    with open(journal, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({**rows[0], "checkpoint_id": "ghost",
                             "trajectory_id": "ghost"}) + "\n")

    build(tmp_path, limit=1)                     # fresh, not resumed
    with open(journal, encoding="utf-8") as fh:
        written = [json.loads(line) for line in fh if line.strip()]
    assert [row["checkpoint_id"] for row in written] == [rows[0]["checkpoint_id"]]


def test_resume_ignores_rows_for_checkpoints_not_in_the_corpus(tmp_path):
    rows, _, path = build(tmp_path, limit=1)
    with open(path + ".journal", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({**rows[0], "checkpoint_id": "ghost"}) + "\n")

    kept, stats, _ = build(tmp_path, limit=2, resume=True)
    assert stats["stale_journal_rows"] == 1
    assert "ghost" not in {row["checkpoint_id"] for row in kept}


def test_resume_refuses_a_journal_from_a_different_corpus(tmp_path):
    build(tmp_path, limit=1)
    # Same ids, different answer keys: exactly what a re-seed produces.
    changed = tmp_path / "checkpoints.jsonl"
    rows = [json.loads(line) for line in open(CHECKPOINTS, encoding="utf-8")
            if line.strip()]
    rows[0]["gold_profile"] = {"spice_level": ["spicy"]}
    with open(changed, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="different corpus"):
        run.run(["recent_only"], out_path=str(tmp_path / "predictions.jsonl"),
                histories_path=HISTORIES, checkpoints_path=str(changed),
                chat_fn=reply, limit=1, resume=True)
