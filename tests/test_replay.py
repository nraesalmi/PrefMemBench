"""Gold replay: the tool layer reproduces the answer key, and is honest.

replay.py is the tool-layer gate in executable form. These tests are
the parts of it that the corpus-level run cannot say about itself: what one
kind of event maps to, the one case where a whole session is truncated in
the corpus, and that a checkpoint is asked of the store as it stood.

No LLM call appears here, and none may be added.
"""

import pytest

from memproj import replay
from memproj.resolver import active_profile
from memproj.semantics import STATEFUL_TYPES

@pytest.fixture(scope="module")
def corpus():
    return replay.load_corpus()


def _events(trajectory):
    return [replay._parse_event(e) for e in trajectory["events"]]


def _one(key, value):
    return [
        {"event_id": "e1", "session": 0,
                   "ts": "2026-03-01T10:00:00", "type": "initial_pref",
                   "subject": "user", "attribute": "spice_level",
                   "value": "mild", "scope": "permanent",
                   "valid_from": "2026-03-01T10:00:00", "valid_until": None,
                   "target_event_id": None, "needs_clarification": False,
                   "referent_event_id": None},
        {"event_id": "e2", "session": 1,
                   "ts": "2026-03-02T11:00:00", "type": "ambiguous_update",
                   "subject": "user", "attribute": "excluded_ingredient",
                   "value": None, "scope": "one_off",
                   "valid_from": "2026-03-02T11:00:00", "valid_until": None,
                   "target_event_id": None, "needs_clarification": False,
         "referent_event_id": "e1"},
    ]


def _calls(events):
    parsed = [replay._parse_event(e) for e in events]
    return [(name, args, e.event_id) for name, args, e in
            replay.events_to_tool_calls(parsed)]


# --- the gold mapping -------------------------------------------------------

def test_a_vague_statement_is_one_call_with_the_question_and_no_answer():
    events = _one(None, None)
    events[1]["referent_event_id"] = None
    (name, args, event_id), = [c for c in _calls(events) if c[2] == "e2"]
    assert name == "clarify_statement"
    assert args["resolves"] == {}
    assert args["question"]


def test_a_resolving_statement_carries_its_own_answer():
    # The user's reply to "which restriction?" is that this restriction is
    # over: gold reads the answer in the turn the question was asked.
    (name, args, event_id), = [c for c in _calls(_one(None, None))
                               if c[2] == "e2"]
    assert name == "clarify_statement"
    assert args["resolves"] == {"source_event_id": "e1"}


def test_a_valueless_statement_call_nothing():
    """\"That exception is over\": gold keeps no record, so replay tool-calls
    none. The call count is derived independently in replay and checked
    per event on every replay, so a drift raises rather than passes."""
    events = _one(None, None)
    events[1].update(type="temporary_exception", scope="temporary",
                              referent_event_id=None)
    assert [c for c in _calls(events) if c[2] == "e2"] == []


def test_every_corpus_event_maps_to_the_calls_its_type_implies(corpus):
    trajectories, _ = corpus
    for tr in trajectories:
        planned = replay.events_to_tool_calls(_events(tr))
        counts = {}
        for _name, _args, e in planned:
            counts[e.event_id] = counts.get(e.event_id, 0) + 1
        for e in _events(tr):
            want = 0 if (e.value is None and e.type in STATEFUL_TYPES
                         and e.type != "ambiguous_update") else 1
            assert counts.get(e.event_id, 0) == want, (tr["trajectory_id"],
                                                       e.event_id, e.type)


def test_the_vague_statement_of_a_reask_keeps_asking_without_an_answer(corpus):
    trajectories, _ = corpus
    matched = [t for t in trajectories
               if t["trajectory_id"] == "p007_matched"]
    if not matched:
        pytest.skip("corpus is not the generated one")
    calls = {e.event_id: args for n, args, e in
             replay.events_to_tool_calls(_events(matched[0]))
             if n == "clarify_statement"}
    # Both vague statements ask; neither has an answer to resolve.
    assert set(calls) == {"e3", "e4"}
    assert all(args["resolves"] == {} for args in calls.values())


# --- the store a checkpoint is asked of -------------------------------------

def test_a_checkpoint_sees_only_the_turns_before_its_now(corpus):
    trajectories, checkpoints = corpus
    by_id = {c["checkpoint_id"]: c for c in checkpoints}
    cp = by_id["p003_matched_c0"]
    tr = next(t for t in trajectories if t["trajectory_id"] == cp["trajectory_id"])
    store = replay.replay_trajectory(tr, before_turn=replay._turns_before(tr, cp["now"]))
    got = active_profile(store, cp["person"], cp["now"])
    assert replay._comparable(got) == cp["gold_profile"]
    # The answer's session lands later; asking early must not read it.
    turns = [t["turn_id"] for t in tr["turns"] if t["ts"] <= cp["now"]]
    assert turns == [t["turn_id"] for t in tr["turns"]][:len(turns)]
    store.close()


def test_the_corpus_checkpoint_inside_a_session_is_asked_of_a_prefix(corpus):
    """correction_of_temporary asks an hour into its session, before that
    session's later turns -- the one checkpoint kind where the cut matters."""
    trajectories, checkpoints = corpus
    mid = [c for c in checkpoints if c["kind"] == "inside_window"]
    assert mid
    for cp in mid:
        tr = next(t for t in trajectories
                  if t["trajectory_id"] == cp["trajectory_id"])
        cut = replay._turns_before(tr, cp["now"])
        assert cut < len(tr["turns"])
        assert all(t["ts"] <= cp["now"] for t in tr["turns"][:cut])
        assert all(t["ts"] > cp["now"] for t in tr["turns"][cut:])


def test_a_question_asked_after_a_checkpoint_is_not_pending_at_it(corpus):
    trajectories, checkpoints = corpus
    by_id = {c["checkpoint_id"]: c for c in checkpoints}
    cp = by_id["p007_matched_c0"]       # 30 minutes after the first vague ask
    later = by_id["p007_matched_c1"]    # after the second, vague reply
    earlier = [c for c in checkpoints
               if c["trajectory_id"] == cp["trajectory_id"]
               and c["now"] < cp["now"]]
    since = earlier[-1]["now"] if earlier else ""
    tr = next(t for t in trajectories if t["trajectory_id"] == cp["trajectory_id"])
    for c, want in ((cp, True), (later, True)):
        store = replay.replay_trajectory(
            tr, before_turn=replay._turns_before(tr, c["now"]))
        start = since if c is cp else cp["now"]
        got = replay._asked_since(store, start, c["now"])
        assert got == want == (c["gold_action"] == "clarify")
        store.close()


def test_replay_refuses_a_trajectory_with_an_event_that_has_no_turn():
    tr = {"trajectory_id": "x", "events": _one(None, None),
          "turns": []}
    with pytest.raises(AssertionError, match="has no turn"):
        replay.replay_trajectory(tr)
