"""The four free-text baselines.

No test here calls a model. The summariser and the embedder are the two
things that would, and both are injected, so these stay rule-logic tests.
"""

import pytest

from memproj import config
from memproj.pipeline import methods

NOW = "2026-03-01T10:30:00"
MILD_TEXT = "I like mild food."
NUT_TEXT = "No peanuts, please."
FILLER_TEXT = "Nice weather today."
FUTURE_TEXT = "Spicy is fine tonight."


def turn(turn_id, speaker, ts, event_id=None, text="something"):
    return {"turn_id": turn_id, "speaker": speaker, "ts": ts,
            "event_id": event_id, "session": 0, "text": text,
            "is_implicit": False}


TURNS = [
    turn("t0001", "user", "2026-03-01T10:05:00", "e1", MILD_TEXT),
    turn("t0002", "assistant", "2026-03-01T10:06:00", None, "Noted."),
    turn("t0003", "user", "2026-03-01T10:07:00", "e2", NUT_TEXT),
    turn("t0004", "assistant", "2026-03-01T10:08:00", None, "Okay."),
    turn("t0005", "user", "2026-03-01T10:20:00", None, FILLER_TEXT),
    # After NOW: a baseline that reads this is an oracle, not a memory method.
    turn("t0006", "user", "2026-03-01T11:00:00", "e3", FUTURE_TEXT),
]

TRAJECTORY = {
    "trajectory_id": "p900_base", "pair_id": "p900", "variant": "base",
    "family": "temp_vs_perm", "flipped_field": "scope",
    "turns": TURNS, "events": [],
}


def checkpoint(now=NOW, request="What should I cook?", checkpoint_id="p900_base_c0"):
    return {"checkpoint_id": checkpoint_id, "trajectory_id": "p900_base",
            "now": now, "person": "user", "request_text": request,
            "gold_profile": {}, "gold_action": "recommend",
            "relevant_event_ids": [], "distractor_event_ids": [],
            "kind": "final"}


# --- shared behaviour -------------------------------------------------------

def test_visible_turns_stop_at_the_checkpoint():
    visible = methods.visible_turns(TRAJECTORY, NOW)
    assert [t["turn_id"] for t in visible] == ["t0001", "t0002", "t0003",
                                               "t0004", "t0005"]
    later = methods.visible_turns(TRAJECTORY, "2026-03-01T12:00:00")
    assert later[-1]["turn_id"] == "t0006"


def test_evidence_skips_assistant_and_filler_turns():
    ids = methods.evidence_turn_ids(TURNS[:5])
    assert ids == ["t0001", "t0003"]  # not t0002/t0004 (assistant), not t0005


@pytest.mark.parametrize("builder", [
    methods.RecentOnly(budget=3),
    methods.FullHistory(),
    methods.RollingSummary(summarise_fn=lambda text: "SUMMARY", budget=3),
])
def test_every_baseline_returns_exactly_the_contract_keys(builder):
    context = builder.build_context(TRAJECTORY, checkpoint())
    assert set(context) == {"context_text", "profile", "evidence_turn_ids"}
    assert context["profile"] is None  # free text, by README, Method interface
    assert isinstance(context["context_text"], str)
    assert all(isinstance(turn_id, str)
               for turn_id in context["evidence_turn_ids"])


def test_no_baseline_ever_reports_a_future_turn():
    # Constructed here rather than via baselines(): the real Retrieval would
    # call the embedder, and this is a rule-logic test.
    builders = [
        methods.RecentOnly(budget=50),
        methods.FullHistory(),
        methods.RollingSummary(summarise_fn=lambda text: "SUMMARY", budget=2),
        methods.Retrieval(embed_fn=fake_embeddings({}), top_k=50),
    ]
    for builder in builders:
        context = builder.build_context(TRAJECTORY, checkpoint())
        assert "t0006" not in context["evidence_turn_ids"]
        assert FUTURE_TEXT not in context["context_text"]


# --- recent only ------------------------------------------------------------

def test_recent_only_keeps_the_last_n_turns():
    context = methods.RecentOnly(budget=3).build_context(TRAJECTORY,
                                                         checkpoint())
    for turn_id in ("t0003", "t0004", "t0005"):
        assert f"[{turn_id}]" in context["context_text"]
    for turn_id in ("t0001", "t0002"):
        assert f"[{turn_id}]" not in context["context_text"]
    assert context["evidence_turn_ids"] == ["t0003"]


def test_recent_only_with_a_large_budget_degenerates_to_full_history():
    recent = methods.RecentOnly(budget=50).build_context(TRAJECTORY, checkpoint())
    full = methods.FullHistory().build_context(TRAJECTORY, checkpoint())
    # Same transcript; only the heading says which baseline produced it.
    assert (recent["context_text"].split("\n", 1)[1]
            == full["context_text"].split("\n", 1)[1])


# --- full history -----------------------------------------------------------

def test_full_history_keeps_everything_before_now():
    context = methods.FullHistory().build_context(TRAJECTORY, checkpoint())
    for turn_id in ("t0001", "t0002", "t0003", "t0004", "t0005"):
        assert f"[{turn_id}]" in context["context_text"]
    assert context["evidence_turn_ids"] == ["t0001", "t0003"]


# --- rolling summary --------------------------------------------------------

def test_rolling_summary_summarises_only_the_old_prefix():
    calls = []

    def summarise(text):
        calls.append(text)
        return "SUMMARY"

    context = methods.RollingSummary(summarise_fn=summarise,
                                     budget=3).build_context(
        TRAJECTORY, checkpoint())

    assert len(calls) == 1
    assert MILD_TEXT in calls[0] and "Noted." in calls[0]
    assert NUT_TEXT not in calls[0]  # that one is still verbatim below
    assert "SUMMARY" in context["context_text"]
    assert f"[t0003]" in context["context_text"]
    assert f"[t0001]" not in context["context_text"]
    # The summariser read the prefix, so those turns stay citable.
    assert context["evidence_turn_ids"] == ["t0001", "t0003"]


def test_rolling_summary_caches_the_prefix_across_checkpoints():
    calls = []
    builder = methods.RollingSummary(
        summarise_fn=lambda text: calls.append(text) or "SUMMARY", budget=3)

    builder.build_context(TRAJECTORY, checkpoint(now="2026-03-01T10:30:00"))
    builder.build_context(TRAJECTORY,
                          checkpoint(now="2026-03-01T10:30:00",
                                     checkpoint_id="p900_base_c1"))
    builder.build_context(TRAJECTORY,
                          checkpoint(now="2026-03-01T10:40:00",
                                     checkpoint_id="p900_base_c2"))

    # Same visible prefix three times -> one model call, not three.
    assert len(calls) == 1


def test_rolling_summary_makes_no_call_when_nothing_is_old():
    def boom(text):
        raise AssertionError("no prefix, so no summary call should happen")

    context = methods.RollingSummary(summarise_fn=boom,
                                     budget=12).build_context(
        TRAJECTORY, checkpoint(now="2026-03-01T10:05:30"))
    assert "Summary" not in context["context_text"]


# --- retrieval --------------------------------------------------------------

def fake_embeddings(table):
    def embed_fn(texts):
        return [table.get(text, [0.0, 1.0]) for text in texts]
    return embed_fn


def test_retrieval_ranks_by_similarity_to_the_request():
    embed_fn = fake_embeddings({
        "What should I cook?": [1.0, 0.0],
        MILD_TEXT: [1.0, 0.0],
        NUT_TEXT: [0.5, 0.5],
    })
    context = methods.Retrieval(embed_fn=embed_fn,
                                top_k=1).build_context(TRAJECTORY, checkpoint())
    assert f"[t0001]" in context["context_text"]
    assert f"[t0003]" not in context["context_text"]
    assert context["evidence_turn_ids"] == ["t0001"]


def test_retrieval_keeps_the_retrieved_turns_in_conversation_order():
    embed_fn = fake_embeddings({
        "What should I cook?": [1.0, 0.0],
        MILD_TEXT: [1.0, 0.0],
        NUT_TEXT: [0.9, 0.1],
    })
    context = methods.Retrieval(embed_fn=embed_fn,
                                top_k=2).build_context(TRAJECTORY, checkpoint())
    assert context["context_text"].index("[t0001]") < \
        context["context_text"].index("[t0003]")


def test_retrieval_breaks_ties_by_position():
    # Every vector identical -> every score is 1.0 -> the earliest turns win.
    embed_fn = fake_embeddings({})
    context = methods.Retrieval(embed_fn=embed_fn,
                                top_k=2).build_context(TRAJECTORY, checkpoint())
    assert [f"[{turn_id}]" in context["context_text"]
            for turn_id in ("t0001", "t0002")] == [True, True]


def test_cosine_handles_a_zero_vector():
    assert methods.cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert methods.cosine([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_baselines_registry_covers_the_contract_methods():
    assert set(methods.baselines()) == {"recent_only", "full_history",
                                        "rolling_summary", "retrieval"}


# --- transport retries (2026-10-25) ----------------------------------------
# Both real model calls go through `recommend.retry_transport`. `chat` is
# covered in test_recommender.py; these pin the two that use their own client
# method (`client.embed`, and the summariser's chat), because a call that
# forgets the helper only shows up as a dead GPU run. The client is faked at
# the `ollama` module, so still no real model call.

def _fake_ollama(monkeypatch, method, replies):
    import sys
    import types

    calls = []

    class FakeClient:
        def __init__(self, timeout=None):
            self.timeout = timeout

        def __getattr__(self, name):
            def send(**kwargs):
                calls.append(kwargs)
                reply = replies.pop(0)
                if isinstance(reply, Exception):
                    raise reply
                return reply
            return send

    monkeypatch.setitem(sys.modules, "ollama",
                        types.SimpleNamespace(Client=FakeClient))
    monkeypatch.setattr(methods.recommend.time, "sleep", lambda s: None)
    return calls


def test_the_summariser_asks_for_no_chain_of_thought(monkeypatch):
    """The runaway that killed two GPU runs: same prompt, 150 s and rising
    with thinking on, 0.6 s off (2026-10-25)."""
    calls = _fake_ollama(monkeypatch, "chat",
                         [{"message": {"content": "Mild food, no nuts."}}])
    methods.summarise("user: I like mild food. ")
    assert calls[0]["think"] is config.PIPELINE_THINK


def test_embed_retries_a_transport_error(monkeypatch):
    import httpx

    calls = _fake_ollama(monkeypatch, "embed", [
        httpx.ReadTimeout("timed out"),
        {"embeddings": [[1.0, 0.0]]},
    ])
    assert methods.embed(["a"]) == [[1.0, 0.0]]
    assert len(calls) == 2
    assert calls[0]["model"] == methods.config.EMBED_MODEL


def test_summarise_retries_a_transport_error(monkeypatch):
    import httpx

    calls = _fake_ollama(monkeypatch, "chat", [
        httpx.ReadTimeout("timed out"),
        {"message": {"content": "  The user likes mild food.  "}},
    ])
    assert methods.summarise("transcript") == "The user likes mild food."
    assert len(calls) == 2
