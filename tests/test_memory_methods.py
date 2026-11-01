"""Step 6: the memory half's two structured methods as `build_context` objects.

The contract (README, Method interface) is one method, three keys, and a
resolved profile in the same shape as `gold_profile`. What is worth pinning
here is not the shape but the split:

  - `proposed` applies every resolver rule, cites the statement behind each
    fact, and surfaces a pending question -- the only channel the interface
    has for the clarify action
  - `latest_wins` takes the newest statement per attribute and nothing else:
    no subject filter, no window, no supersede. That is the baseline's whole
    point, and each test below is one rule it should ignore
  - both read the SAME extraction: one model pass per trajectory, shared
    across checkpoints and methods, so the experiment varies the rule only

No test calls a model; the scripted client answers turn by turn.
"""

from datetime import datetime, timedelta

import pytest

from memproj.memory import methods as memory_methods
from memproj.pipeline import run
from memproj.validate_predictions import validate_predictions

T0 = datetime(2026, 3, 1, 10, 0)
FIXTURES = "tests/fixtures"


# --- the scripted client ----------------------------------------------------

class _Call:
    def __init__(self, name, arguments):
        self.function = type("Fn", (), {"name": name,
                                        "arguments": arguments})()


def answer(*calls, prompt_tokens=10, eval_tokens=5):
    message = type("Msg", (), {"tool_calls": [_Call(*c) for c in calls],
                               "content": ""})()
    return type("Resp", (), {"message": message,
                             "prompt_eval_count": prompt_tokens,
                             "eval_count": eval_tokens})()


class Client:
    """One scripted reply per user turn, in order."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, model, messages, tools=None, options=None, think=None):
        self.calls.append(list(messages))
        reply = self.script.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def add(subject, value, scope="permanent", attribute="spice_level",
        valid_until=None):
    args = {"subject": subject, "attribute": attribute, "value": value,
            "scope": scope}
    if valid_until is not None:
        args["valid_until"] = valid_until
    return answer(("add_preference", args))


def ignore():
    return answer(("ignore_statement", {"reason": "chatter"}))


# --- fixtures ---------------------------------------------------------------

def _turn(n, speaker, minutes, text, event_id=None):
    return {"turn_id": f"t{n:04d}", "session": 0, "speaker": speaker,
            "ts": (T0 + timedelta(minutes=minutes)).isoformat(),
            "event_id": event_id, "text": text}


def _trajectory(turns, tid="m_base"):
    return {"trajectory_id": tid, "pair_id": "m", "family": "temp_vs_perm",
            "variant": "base", "events": [], "turns": turns}


def _checkpoint(minutes, person="user", cid="m_c0"):
    return {"checkpoint_id": cid, "trajectory_id": "m_base", "person": person,
            "kind": "for_user",
            "now": (T0 + timedelta(minutes=minutes)).isoformat(),
            "gold_profile": {}, "gold_action": "recommend"}


def _build(turns, script, model=None):
    """(trajectory, cache) with extraction already run through `script`."""
    trajectory = _trajectory(turns)
    cache = memory_methods.ExtractionCache(Client(script), model=model)
    return trajectory, cache


# --- proposed and latest wins -----------------------------------------------

def test_proposed_returns_the_resolved_profile_and_cites_its_statement():
    turns = [_turn(1, "user", 0, "I like mild food.", "e1"),
             _turn(2, "assistant", 1, "Noted.")]
    trajectory, cache = _build(turns, [add("user", "mild")])
    try:
        context = memory_methods.Proposed(cache).build_context(
            trajectory, _checkpoint(60))
    finally:
        cache.close()
    assert context["profile"] == {"spice_level": ["mild"]}
    assert context["evidence_turn_ids"] == ["t0001"]
    assert "mild" in context["context_text"]
    assert "t0001" in context["context_text"]      # the source is named


def test_proposed_leaves_another_persons_statement_out():
    turns = [_turn(1, "user", 0, "I like mild food.", "e1"),
             _turn(2, "user", 30, "My brother likes spicy food.", "e2")]
    trajectory, cache = _build(turns, [add("user", "mild"),
                                       add("brother", "spicy")])
    try:
        proposed = memory_methods.Proposed(cache).build_context(
            trajectory, _checkpoint(60))
        latest = memory_methods.LatestWins(cache).build_context(
            trajectory, _checkpoint(60))
    finally:
        cache.close()
    assert proposed["profile"] == {"spice_level": ["mild"]}
    # The baseline has no subject rule: the newest statement wins, whoever
    # it is about. This is the contamination the metric is there to show.
    assert latest["profile"] == {"spice_level": ["spicy"]}
    assert latest["evidence_turn_ids"] == ["t0002"]


def test_proposed_drops_an_expired_exception_that_latest_wins_keeps():
    turns = [_turn(1, "user", 0, "I like mild food.", "e1"),
             _turn(2, "user", 30, "Tonight spicy is fine.", "e2")]
    until = (T0 + timedelta(hours=2)).isoformat()
    trajectory, cache = _build(turns, [add("user", "mild"),
                                       add("user", "spicy", scope="temporary",
                                           valid_until=until)])
    try:
        proposed = memory_methods.Proposed(cache).build_context(
            trajectory, _checkpoint(180))
        latest = memory_methods.LatestWins(cache).build_context(
            trajectory, _checkpoint(180))
    finally:
        cache.close()
    assert proposed["profile"] == {"spice_level": ["mild"]}
    assert latest["profile"] == {"spice_level": ["spicy"]}


def test_proposed_surfaces_a_pending_question_and_latest_wins_does_not():
    turns = [_turn(1, "user", 0, "That restriction is no longer important.",
                   "e1")]
    script = [answer(("ask_clarification",
                      {"question": "Which restriction do you mean?"}))]
    trajectory, cache = _build(turns, script)
    try:
        proposed = memory_methods.Proposed(cache).build_context(
            trajectory, _checkpoint(60))
        latest = memory_methods.LatestWins(cache).build_context(
            trajectory, _checkpoint(60))
    finally:
        cache.close()
    assert "Which restriction do you mean?" in proposed["context_text"]
    assert proposed["profile"] == {}
    assert "Which restriction" not in latest["context_text"]


def test_a_question_asked_after_the_checkpoint_is_not_pending_yet():
    turns = [_turn(1, "user", 0, "I like mild food.", "e1"),
             _turn(2, "user", 120, "That restriction is no longer important.",
                   "e2")]
    script = [add("user", "mild"),
              answer(("ask_clarification", {"question": "Which one?"}))]
    trajectory, cache = _build(turns, script)
    try:
        proposed = memory_methods.Proposed(cache).build_context(
            trajectory, _checkpoint(60))
    finally:
        cache.close()
    assert "Which one?" not in proposed["context_text"]


# --- the contract -----------------------------------------------------------

def test_build_context_returns_exactly_the_contract_keys():
    turns = [_turn(1, "user", 0, "I like mild food.", "e1")]
    trajectory, cache = _build(turns, [add("user", "mild")])
    try:
        for method in (memory_methods.Proposed(cache),
                       memory_methods.LatestWins(cache)):
            context = method.build_context(trajectory, _checkpoint(60))
            assert set(context) == {"context_text", "profile",
                                    "evidence_turn_ids"}
            assert isinstance(context["context_text"], str)
            assert isinstance(context["profile"], dict)
            assert all(isinstance(v, list)
                       for v in context["profile"].values())
            assert isinstance(context["evidence_turn_ids"], list)
    finally:
        cache.close()


def test_evidence_is_the_statement_behind_the_fact_not_a_future_turn():
    turns = [_turn(1, "user", 0, "I like mild food.", "e1"),
             _turn(2, "assistant", 1, "Noted."),
             _turn(3, "user", 120, "Make it spicy.", "e2")]
    trajectory, cache = _build(turns, [add("user", "mild"),
                                       add("user", "spicy")])
    try:
        context = memory_methods.Proposed(cache).build_context(
            trajectory, _checkpoint(60))
    finally:
        cache.close()
    assert context["evidence_turn_ids"] == ["t0001"]


# --- cost -------------------------------------------------------------------

def test_extraction_runs_once_per_trajectory_across_methods_and_checkpoints():
    turns = [_turn(1, "user", 0, "I like mild food.", "e1"),
             _turn(2, "user", 30, "And no mushrooms.", "e2")]
    script = [add("user", "mild"),
              add("user", "mushroom", attribute="excluded_ingredient")]
    trajectory, cache = _build(turns, script)
    client = cache.client
    try:
        for _ in (0, 1):
            for method in (memory_methods.Proposed(cache),
                           memory_methods.LatestWins(cache)):
                method.build_context(trajectory, _checkpoint(60))
                method.build_context(trajectory, _checkpoint(120, cid="m_c1"))
    finally:
        cache.close()
    # Two user turns, one pass: 2 methods x 2 checkpoints still cost 2 calls.
    assert len(client.calls) == 2


def test_the_rendered_profile_is_deterministic():
    turns = [_turn(1, "user", 0, "I like mild food.", "e1")]
    first, cache = _build(turns, [add("user", "mild")])
    try:
        a = memory_methods.Proposed(cache).build_context(first,
                                                         _checkpoint(60))
    finally:
        cache.close()
    second, cache = _build(turns, [add("user", "mild")])
    try:
        b = memory_methods.Proposed(cache).build_context(second,
                                                         _checkpoint(60))
    finally:
        cache.close()
    assert a["context_text"] == b["context_text"]


# --- the extraction cache on disk -------------------------------------------
# Step 7 runs five resolver ablations over the same extraction. Re-extracting
# per ablation would cost another GPU run each, so the pass is written once
# per trajectory and read back. What must NOT be reused is an extraction made
# from different inputs: a different corpus, model or prompt answers different
# questions.

class _NoModel:
    """A client that fails the test if anything asks it for a reply."""

    def __init__(self):
        self.calls = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("the extraction should have come from disk")


def _disk_cache(directory, script=(), model=None, fingerprint="corpus-a"):
    return memory_methods.ExtractionCache(
        Client(list(script)), model=model, directory=str(directory),
        fingerprint=fingerprint)


def _mild_turns():
    return [_turn(1, "user", 0, "I like mild food.", "e1")]


def test_the_extraction_is_written_once_and_reused_from_disk(tmp_path):
    first = _disk_cache(tmp_path, [add("user", "mild")])
    try:
        written = memory_methods.Proposed(first).build_context(
            _trajectory(_mild_turns()), _checkpoint(60))
    finally:
        first.close()
    assert first.stats()["extracted"] == 1

    second = memory_methods.ExtractionCache(
        _NoModel(), directory=str(tmp_path), fingerprint="corpus-a")
    try:
        read = memory_methods.Proposed(second).build_context(
            _trajectory(_mild_turns()), _checkpoint(60))
        assert second.stats()["loaded"] == 1
        assert second.stats()["extracted"] == 0
    finally:
        second.close()
    assert read == written


def test_a_different_model_does_not_reuse_the_cache(tmp_path):
    first = _disk_cache(tmp_path, [add("user", "mild")])
    try:
        memory_methods.Proposed(first).build_context(
            _trajectory(_mild_turns()), _checkpoint(60))
    finally:
        first.close()

    client = Client([add("user", "mild")])
    other = memory_methods.ExtractionCache(
        client, model="another-model", directory=str(tmp_path),
        fingerprint="corpus-a")
    try:
        memory_methods.Proposed(other).build_context(
            _trajectory(_mild_turns()), _checkpoint(60))
    finally:
        other.close()
    assert len(client.calls) == 1
    assert other.stats()["loaded"] == 0


def test_a_regenerated_corpus_does_not_reuse_the_cache(tmp_path):
    first = _disk_cache(tmp_path, [add("user", "mild")])
    try:
        memory_methods.Proposed(first).build_context(
            _trajectory(_mild_turns()), _checkpoint(60))
    finally:
        first.close()

    client = Client([add("user", "mild")])
    other = memory_methods.ExtractionCache(
        client, directory=str(tmp_path), fingerprint="corpus-b")
    try:
        memory_methods.Proposed(other).build_context(
            _trajectory(_mild_turns()), _checkpoint(60))
    finally:
        other.close()
    assert len(client.calls) == 1


def test_a_regenerated_corpus_discards_every_file_not_just_the_first(tmp_path):
    """The manifest is rewritten after the FIRST trajectory of a fresh pass.

    Checking it per file therefore lets every other file match it, which on
    2026-10-27 loaded 93 of 120 extractions from the previous corpus as if
    they answered today's. `first` and `second` stand in for those files:
    with a new fingerprint and a scripted client that answers for both, both
    must be asked again and the stale files must be gone.
    """
    first = _disk_cache(tmp_path, [add("user", "mild")])
    try:
        memory_methods.Proposed(first).build_context(_trajectory(_mild_turns()),
                                                    _checkpoint(60))
    finally:
        first.close()

    client = Client([add("user", "mild"), add("user", "mild")])
    other = memory_methods.ExtractionCache(
        client, directory=str(tmp_path), fingerprint="corpus-b")
    try:
        for tid in ("m_base", "m_matched"):
            memory_methods.Proposed(other).build_context(
                _trajectory(_mild_turns(), tid=tid), _checkpoint(60))
        stats = other.stats()
    finally:
        other.close()
    assert len(client.calls) == 2          # the stale file was not served
    assert stats["extracted"] == 2 and stats["loaded"] == 0
    assert stats["discarded"] == 1


def test_a_read_only_cache_refuses_a_stale_directory(tmp_path):
    """`ablate` opens the cache with no client: it must not serve stale rows."""
    first = _disk_cache(tmp_path, [add("user", "mild")])
    try:
        memory_methods.Proposed(first).build_context(_trajectory(_mild_turns()),
                                                    _checkpoint(60))
    finally:
        first.close()

    with pytest.raises(RuntimeError, match="rerun the pipeline"):
        memory_methods.ExtractionCache(
            None, directory=str(tmp_path), fingerprint="corpus-b")
    # Nothing was deleted by a reader that could not replace it.
    assert (tmp_path / "m_base.sqlite").exists()


def test_a_changed_extraction_prompt_does_not_reuse_the_cache(tmp_path, monkeypatch):
    first = _disk_cache(tmp_path, [add("user", "mild")])
    try:
        memory_methods.Proposed(first).build_context(
            _trajectory(_mild_turns()), _checkpoint(60))
    finally:
        first.close()

    monkeypatch.setattr(memory_methods, "_prompt_signature", lambda: "v2")
    client = Client([add("user", "mild")])
    other = memory_methods.ExtractionCache(
        client, directory=str(tmp_path), fingerprint="corpus-a")
    try:
        memory_methods.Proposed(other).build_context(
            _trajectory(_mild_turns()), _checkpoint(60))
    finally:
        other.close()
    assert len(client.calls) == 1


def test_stats_report_turns_tokens_failures_and_cost(tmp_path):
    turns = [_turn(1, "user", 0, "I like mild food.", "e1"),
             _turn(2, "user", 30, "And no mushrooms.", "e2")]
    cache = _disk_cache(
        tmp_path,
        [add("user", "mild"), RuntimeError("socket died"),
         RuntimeError("socket died"), RuntimeError("socket died")])
    try:
        memory_methods.Proposed(cache).build_context(
            _trajectory(turns), _checkpoint(60))
        stats = cache.stats()
    finally:
        cache.close()
    assert stats["trajectories"] == 1
    assert stats["turns"] == 2
    assert stats["extracted"] == 1
    assert [f["turn_id"] for f in stats["failures"]] == ["t0002"]
    assert "socket died" in stats["failures"][0]["error"]
    # Only the answered call has counts; three failed attempts cost no tokens.
    assert stats["tokens"] == 15
    assert stats["seconds"] >= 0.0


def test_a_cache_without_a_directory_keeps_working_in_memory():
    cache = memory_methods.ExtractionCache(Client([add("user", "mild")]))
    try:
        context = memory_methods.Proposed(cache).build_context(
            _trajectory(_mild_turns()), _checkpoint(60))
    finally:
        cache.close()
    assert context["profile"] == {"spice_level": ["mild"]}


# --- through the runner -----------------------------------------------------

def _reply(messages, tokens=7):
    import json
    import re
    chosen = re.findall(r"^- ([^\s:]+):", messages[1]["content"], re.M)[0]
    return {"text": json.dumps({"action": "recommend", "recipe_id": chosen,
                                "question": None,
                                "explanation": "Fits."}),
            "tokens": tokens}


def test_a_structured_method_runs_end_to_end_and_validates(tmp_path):
    client = Client([ignore(), ignore(), ignore()])   # 3 user turns in fixtures
    path = str(tmp_path / "predictions.jsonl")
    rows, stats = run.run(["proposed"], out_path=path,
                          histories_path=FIXTURES + "/histories.jsonl",
                          checkpoints_path=FIXTURES + "/checkpoints.jsonl",
                          chat_fn=_reply, client=client)
    assert stats["rows_by_method"] == {"proposed": 2}
    assert all(isinstance(row["profile"], dict) for row in rows)
    assert len(client.calls) == 3          # one extraction pass per turn
    assert validate_predictions(
        path, histories_path=FIXTURES + "/histories.jsonl",
        checkpoints_path=FIXTURES + "/checkpoints.jsonl") == []


def test_an_unknown_method_is_still_rejected():
    with pytest.raises(NotImplementedError):
        run.build_prediction({"trajectory_id": "x", "turns": []},
                             {"checkpoint_id": "c", "trajectory_id": "x"},
                             [], method="nope", builders={})


# --- what the cache manifest protects ---------------------------------------
# Step 6's contamination table (README, "Contamination by source") is only
# comparable across reruns if every structured method reads the SAME
# extraction. These two pins are the difference between a cache and a
# plausible-looking quiet bug: change the question and the answers must be
# re-asked, and the hash must actually move when the questions do.

def test_a_changed_tool_schema_re_extracts(tmp_path, monkeypatch):
    # The prompt signature hashes the tool schemas, not just the system
    # prompt. Editing a tool's description makes the model read the same
    # sentence differently, so a cached answer is from another experiment.
    first = _disk_cache(tmp_path, [add("user", "mild")])
    try:
        memory_methods.Proposed(first).build_context(
            _trajectory(_mild_turns()), _checkpoint(60))
    finally:
        first.close()

    edited = [dict(schema) for schema in memory_methods.TOOL_SCHEMAS]
    edited[0] = dict(edited[0],
                     function=dict(edited[0]["function"],
                                   description="A reworded description."))
    monkeypatch.setattr(memory_methods, "TOOL_SCHEMAS", edited)

    second = _disk_cache(tmp_path, [add("user", "mild")])
    try:
        memory_methods.Proposed(second).build_context(
            _trajectory(_mild_turns()), _checkpoint(60))
        assert second.stats()["extracted"] == 1
        assert second.stats()["loaded"] == 0
    finally:
        second.close()


def test_the_signature_sees_a_reworded_system_prompt(monkeypatch):
    # A weaker test than the one above would pass on a constant: assert the
    # hash is a function of what it claims to cover.
    before = memory_methods._prompt_signature()
    monkeypatch.setattr(memory_methods, "EXTRACTION_SYSTEM",
                        memory_methods.EXTRACTION_SYSTEM + " One more line.")
    assert memory_methods._prompt_signature() != before
