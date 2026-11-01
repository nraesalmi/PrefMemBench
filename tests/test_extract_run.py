"""The corpus runner and its table: what state accuracy is, per update type.

The loop (test_extract.py) is pinned turn by turn. This file pins the part
that turns a corpus into a number:

  - a checkpoint is asked of the store built from the turns up to it, never
    later ones -- the runner reuses replay's prefix cut, so that property
    cannot drift between gold and a model
  - "missing" is a gold fact the agent does not hold, "extra" a fact it
    holds that gold does not, and an excluded_ingredient set mismatched by
    one value counts as both -- in different sets, not as two wrong values
  - the table is by checkpoint kind, because that is the update type the
    family tests (a revision and a temporary exception fail differently)

No model runs here; the scripted client from test_extract drives both.
"""

from datetime import datetime, timedelta

from memproj.memory import extract


class FakeClient:
    """Scripted ollama client, spelled out here because the tests directory
    has no package: importing test_extract by name would need one."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, model, messages, tools=None, options=None, think=None):
        self.calls.append({"model": model, "messages": list(messages)})
        nxt = self.script.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class _Call:
    def __init__(self, name, arguments):
        self.function = type("Fn", (), {"name": name,
                                        "arguments": arguments})()


def answer(*calls):
    message = type("Msg", (), {"tool_calls": [_Call(*c) for c in calls],
                               "content": ""})()
    return type("Resp", (), {"message": message})()


def repeated(*calls):
    """Last resort: the same answer to every call, however many are made.

    A scripted client whose list runs out is correct for a test about
    call counts; a test about what a store HOLDS should not double as one.
    """
    class Repeater(FakeClient):
        def chat(self, model, messages, tools=None, options=None, think=None):
            self.calls.append({"model": model, "messages": list(messages)})
            return answer(*calls)
    return Repeater([])

T0 = datetime(2026, 3, 1, 10, 0)


def _trajectory():
    return {
        "trajectory_id": "x_base", "pair_id": "x", "family": "temp_vs_perm",
        "variant": "base",
        "events": [],
        "turns": [
            {"turn_id": "t0001", "session": 0, "speaker": "user",
             "ts": T0.isoformat(), "event_id": "e1", "text": "I like mild food."},
            {"turn_id": "t0002", "session": 0, "speaker": "assistant",
             "ts": (T0 + timedelta(minutes=1)).isoformat(), "event_id": None,
             "text": "Noted."},
            {"turn_id": "t0003", "session": 1, "speaker": "user",
             "ts": (T0 + timedelta(days=1)).isoformat(), "event_id": "e2",
             "text": "Tonight spicy is fine."},
        ],
    }


def _checkpoints():
    day = T0 + timedelta(days=1)
    return [
        {"checkpoint_id": "x_c0", "trajectory_id": "x_base", "person": "user",
         "kind": "after_revision", "now": (T0 + timedelta(hours=2)).isoformat(),
         "gold_profile": {"spice_level": ["mild"]},
         "gold_action": "recommend", "family": "temp_vs_perm"},
        {"checkpoint_id": "x_c1", "trajectory_id": "x_base", "person": "user",
         "kind": "inside_window", "now": (day + timedelta(hours=2)).isoformat(),
         "gold_profile": {"spice_level": ["mild", "spicy"]},
         "gold_action": "recommend", "family": "temp_vs_perm"},
    ]


def test_a_checkpoint_is_asked_of_the_store_up_to_its_now():
    """The second statement happens after c0: c0 must not see it."""
    client = FakeClient([answer(("add_preference", {
        "subject": "user", "attribute": "spice_level",
        "value": "mild", "scope": "permanent"}))])
    rows = extract.run([_trajectory()], _checkpoints(), client=client,
                       model="fake-model")
    assert [r["checkpoint_id"] for r in rows] == ["x_c0", "x_c1"]
    # Each user turn is read once, when its checkpoint is reached: c0's
    # prompt cannot contain the turn that happens a day later. (This
    # script has one reply, so the second statement costs MAX_ATTEMPTS --
    # the count is about when turns are READ, not about retries.)
    assert len(client.calls) == extract.MAX_ATTEMPTS + 1
    assert client.calls[0]["messages"][-1]["content"] == "I like mild food."
    assert all(r["wrong_value"] == [] for r in rows)
    assert rows[0]["missing"] == []
    # c1's gold adds 'spicy' inside the window; mild-only holds both keys
    assert rows[1]["exact"] is False


def test_a_missing_gold_fact_is_named_and_scored_as_missing():
    client = repeated(("ignore_statement", {"reason": "chatter"}))
    rows = extract.run([_trajectory()], _checkpoints(), client=client,
                       model="fake-model")
    c0 = rows[0]
    assert c0["missing"] == ["spice_level:mild"] and c0["extra"] == []
    assert c0["exact"] is False
    assert (c0["tp"], c0["fn"], c0["fp"]) == (0, 1, 0)


def test_a_wrong_value_is_wrong_in_both_directions():
    client = repeated(("add_preference", {
        "subject": "user", "attribute": "spice_level",
        "value": "spicy", "scope": "permanent"}))
    c0 = extract.run([_trajectory()], _checkpoints(), client=client,
                     model="fake-model")[0]
    assert c0["wrong_value"] == [{"attribute": "spice_level",
                                  "gold": ["mild"], "got": ["spicy"]}]
    # `wrong_value` names the attribute for the failure samples; the facts
    # themselves are charged where every other fact is, as a miss and a
    # surplus value. It is one error, not a third one.
    assert c0["missing"] == ["spice_level:mild"]
    assert c0["extra"] == ["spice_level:spicy"]
    assert (c0["tp"], c0["fn"], c0["fp"]) == (0, 1, 1)   # a miss and a guess


def test_a_disjoint_multi_valued_attribute_credits_no_hit():
    """Gold {mushroom, spicy} vs got {medium}: nothing matched, so tp is 0.

    78 of 205 checkpoints carry a multi-valued attribute, so subtracting one
    "wrong" per attribute instead of one per fact would overstate precision
    on the reachable case, not an edge.
    """
    result = extract.score({"excluded_ingredient": ["medium"]},
                           {"excluded_ingredient": ["mushroom", "spicy"]})
    assert result["missing"] == ["excluded_ingredient:mushroom",
                                 "excluded_ingredient:spicy"]
    assert result["extra"] == ["excluded_ingredient:medium"]
    assert result["wrong_value"] == [{"attribute": "excluded_ingredient",
                                      "gold": ["mushroom", "spicy"],
                                      "got": ["medium"]}]
    assert (result["tp"], result["fn"], result["fp"]) == (0, 2, 1)


def test_an_empty_gold_list_is_extra_only_never_a_wrong_value():
    """A value where gold says nothing is a false positive, not a negative tp."""
    result = extract.score({"excluded_ingredient": ["mushroom"]},
                           {"excluded_ingredient": []})
    assert result["wrong_value"] == []
    assert (result["tp"], result["fn"], result["fp"]) == (0, 0, 1)


def test_an_excluded_ingredient_set_is_compared_value_by_value():
    gold = {"excluded_ingredient": ["mushroom", "peanut"]}
    got = {"excluded_ingredient": ["peanut", "fish"]}
    missing, extra, wrong = extract.diff_profiles(got, gold)
    assert missing == ["excluded_ingredient:mushroom"]
    assert extra == ["excluded_ingredient:fish"]
    assert wrong == []


def test_the_summary_table_counts_by_checkpoint_kind_and_reports_rates():
    rows = [
        {"kind": "inside_window", "exact": True, "tp": 2, "fn": 0, "fp": 0,
         "wrong_value": []},
        {"kind": "inside_window", "exact": False, "tp": 0, "fn": 1, "fp": 0,
         "wrong_value": []},
        {"kind": "after_revision", "exact": False, "tp": 1, "fn": 0, "fp": 1,
         "wrong_value": [{"attribute": "spice_level"}]},
    ]
    table = extract.summarise(rows)
    assert "inside_window" in table and "after_revision" in table
    assert "all" in table.splitlines()[-1] or "all" in table
    inside = next(l for l in table.splitlines() if l.startswith("inside_window"))
    assert " 50.0%" in inside or "50.0" in inside      # 1 of 2 exact
    assert " 0.0%" in inside or "0.0" in inside        # precision 0 of 1


def test_a_failed_statement_is_counted_not_silently_absent():
    client = FakeClient([RuntimeError("down")] * extract.MAX_ATTEMPTS * 3)
    rows = extract.run([_trajectory()], _checkpoints(), client=client,
                       model="fake-model")
    summary = extract.summarise(rows)
    assert "failed" in summary.lower()


def test_a_failed_statement_is_journaled_with_its_reason():
    """A 0% run must be explainable from the journal alone.

    The decision rows that carry the error live in an in-memory store that
    is closed when the trajectory ends, so the reason has to reach the row
    the journal keeps. `failed` is the whole path's count, not this
    checkpoint's segment: an early unreadable statement still corrupts
    every later profile.
    """
    client = FakeClient(
        [RuntimeError("model 'qwen3:8b' not found")] * extract.MAX_ATTEMPTS
        + [answer(("add_preference", {
            "subject": "user", "attribute": "spice_level",
            "value": "spicy", "scope": "temporary",
            "valid_until": (T0 + timedelta(days=1, hours=6)).isoformat()}))])
    rows = extract.run([_trajectory()], _checkpoints(), client=client,
                       model="fake-model")
    assert rows[0]["failed"] == 1
    # The loop stamps the exception type in front of the message, so the
    # journal distinguishes "the server said 404" from "the call was
    # unusable" without a second log.
    assert rows[0]["failures"] == [
        {"turn_id": "t0001",
         "error": "RuntimeError: model 'qwen3:8b' not found"}]
    # c1 read a second statement, which succeeded: the path's failure
    # count does not reset when the segment is clean.
    assert rows[1]["failed"] == 1
    assert len(rows[1]["failures"]) == 1


def test_completed_trajectories_needs_every_checkpoint():
    """A trajectory paused after c0 is not done: resuming it must redo c0
    so the journal never holds two rows for one checkpoint."""
    rows = [{"trajectory_id": "x_base", "checkpoint_id": "x_c0"},
            {"trajectory_id": "x_base", "checkpoint_id": "x_c1"},
            {"trajectory_id": "y_base", "checkpoint_id": "y_c0"}]
    cps = [{"trajectory_id": "x_base", "checkpoint_id": "x_c0"},
           {"trajectory_id": "x_base", "checkpoint_id": "x_c1"},
           {"trajectory_id": "y_base", "checkpoint_id": "y_c0"},
           {"trajectory_id": "y_base", "checkpoint_id": "y_c1"}]
    assert extract.completed_trajectories(rows, cps) == {"x_base"}


def test_run_makes_no_calls_for_a_skipped_trajectory():
    client = repeated(("ignore_statement", {"reason": "chatter"}))
    rows = extract.run([_trajectory()], _checkpoints(), client=client,
                       model="fake-model", skip={"x_base"})
    assert rows == [] and client.calls == []


def test_main_skip_leaves_the_named_trajectory_alone(tmp_path, monkeypatch):
    """`--skip` is how an ingredient-only rerun pays for 54 trajectories and
    not 120; the flag has to reach `run()`, not just its parameter list."""
    import json

    from memproj import replay

    path = tmp_path / "results.jsonl"
    client = repeated(("ignore_statement", {"reason": "chatter"}))
    monkeypatch.setattr(extract, "make_client", lambda: client)
    monkeypatch.setattr(replay, "load_corpus",
                        lambda: ([_trajectory()], _checkpoints()))
    assert extract.main(["--skip", "x_base", "--out", str(path)]) == 0
    assert path.read_text(encoding="utf-8") == ""
    assert client.calls == []


def test_resume_drops_the_interrupted_trajectory_and_keeps_the_rest(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text(
        '{"trajectory_id": "x_base", "checkpoint_id": "x_c0"}\n'
        '{"trajectory_id": "x_base", "checkpoint_id": "x_c1"}\n'
        '{"trajectory_id": "y_base", "checkpoint_id": "y_c0"}\n',
        encoding="utf-8")
    cps = _checkpoints() + [
        {"trajectory_id": "y_base", "checkpoint_id": "y_c0"},
        {"trajectory_id": "y_base", "checkpoint_id": "y_c1"}]
    kept, skip = extract.resume_state(str(path), cps)
    assert skip == {"x_base"}
    assert [r["checkpoint_id"] for r in kept] == ["x_c0", "x_c1"]
    # y_base was interrupted mid-trajectory: its half-written row is gone,
    # so a resumed run cannot leave two rows for y_c0.
    assert '"y_base"' not in path.read_text(encoding="utf-8")


def test_main_resume_keeps_the_finished_journal_and_reruns_the_rest(
        tmp_path, monkeypatch):
    """The CLI contract for a paused run: `--resume` appends rather than
    truncating, and the finished trajectories are not asked again."""
    import json

    from memproj import replay

    path = tmp_path / "results.jsonl"
    done = {"trajectory_id": "x_base", "checkpoint_id": "x_c0", "kind": "k",
            "exact": True, "tp": 0, "fn": 0, "fp": 0, "missing": [],
            "extra": [], "wrong_value": [], "failed": 0, "failures": []}
    path.write_text(json.dumps(done) + "\n", encoding="utf-8")
    client = repeated(("ignore_statement", {"reason": "chatter"}))
    monkeypatch.setattr(extract, "make_client", lambda: client)
    monkeypatch.setattr(replay, "load_corpus",
                        lambda: ([_trajectory()], _checkpoints()))
    assert extract.main(["--resume", "--out", str(path)]) == 0
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines()]
    # x_c0 came from the journal, x_c1 from the resumed run, in that order.
    assert [r["checkpoint_id"] for r in rows] == ["x_c0", "x_c1"]


def test_the_state_counters_agree_with_the_evaluator():
    """There are two copies of diff/score; they must not disagree on numbers.

    `memproj/eval/metrics.py` is the report's authority and this module is
    the extraction journal. They were allowed to stay separate, not to drift: an
    earlier version here subtracted `wrong` from `tp` and credited gold
    values the method never said. These cases are the ones that reach the
    divergence (disjoint multi-valued, empty gold, missing attribute).
    """
    from memproj.eval import metrics

    cases = [
        ({}, {}),
        ({"spice_level": ["mild"]}, {"spice_level": ["mild"]}),
        ({"spice_level": ["spicy"]}, {"spice_level": ["mild"]}),
        ({}, {"spice_level": ["mild"]}),
        ({"excluded_ingredient": ["peanut"]},
         {"excluded_ingredient": ["mushroom", "peanut"]}),
        ({"excluded_ingredient": ["medium"]},
         {"excluded_ingredient": ["mushroom", "spicy"]}),
        ({"excluded_ingredient": ["mushroom"]}, {"excluded_ingredient": []}),
        ({"diet": ["vegan"]}, {"spice_level": ["mild"]}),
    ]
    for got, gold in cases:
        ours = extract.score(got, gold)
        theirs = metrics.score(got, gold)
        assert ((ours["tp"], ours["fn"], ours["fp"], ours["exact"])
                == (theirs["tp"], theirs["fn"], theirs["fp"], theirs["exact"])), \
            (got, gold, ours, theirs)
