"""The extraction loop: a model's tool calls, made safe and countable.

Step 3 replaces the gold mapping with a model. What a model gets wrong is
data; what the LOOP gets wrong is a bug. These tests drive the loop with a
scripted client, so every behaviour is pinned without a model running:

  - the harness stamps ts and turn_id; the model never supplies them
  - a tool that refuses its arguments is retried, with the error shown to
    the model, at most MAX_ATTEMPTS times per statement
  - a second call in the same turn is logged as a failure, not kept
  - chatty turns take no extraction call at all
  - the model never sees event ids, gold labels, or the evaluator's names

A live model is exercised only by the extraction runner, never here.
"""

from datetime import datetime, timedelta

import pytest

from memproj.memory import extract
from memproj.memory.extract import MAX_ATTEMPTS, Extractor
from memproj.schema import Turn
from memproj.store import Store
from memproj.tools import TOOL_SCHEMAS

T0 = datetime(2026, 3, 1, 18, 30)


# --- a scripted ollama client -----------------------------------------------

class FakeClient:
    """Answers `chat()` from a script: a list of responses per call, each a
    dict shaped like ollama's ChatResponse (message.tool_calls)."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, model, messages, tools=None, options=None, think=None):
        self.calls.append({"model": model, "messages": list(messages),
                           "tools": tools, "options": options,
                           "think": think})
        assert self.script, "the loop called the model one time too many"
        nxt = self.script.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class _Fn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _Call:
    def __init__(self, name, arguments):
        self.function = _Fn(name, arguments)


class _ToolCalls(list):
    def __init__(self, calls):
        super().__init__(_Call(*c) for c in calls)


class _Message:
    def __init__(self, calls, content=""):
        self.tool_calls = _ToolCalls(calls)
        self.content = content


class _Response:
    def __init__(self, calls, content=""):
        self.message = _Message(calls, content)


def answer(*calls):
    return _Response(list(calls))


# --- fixtures ---------------------------------------------------------------

@pytest.fixture
def store():
    s = Store(":memory:").create()
    s.add_turn(Turn("t0001", 0, T0, "user",
                    "I usually eat mild food."))
    s.add_turn(Turn("t0002", 0, T0 + timedelta(minutes=2), "assistant",
                    "Noted."))
    yield s
    s.close()


def make(client, **kwargs):
    kwargs.setdefault("model", "fake-model")
    return Extractor(client=client, **kwargs)


def say(store, text, turn_id="t0003", ts=T0):
    """Record one user turn as the harness would, and return it.

    A turn after the fixture's t0001/t0002, because a store holds a real
    conversation: the model sees the earlier exchange, not a replay of it.
    """
    turn = Turn(turn_id, 0, ts, "user", text)
    store.add_turn(turn)
    store.turn_texts = {**getattr(store, "turn_texts", {}), turn_id: text}
    return turn


def turn_row(store, turn_id):
    """The turn as the loop sees it: the store's row, with its words.

    The turns table holds who and when; the transcript holds the words. A
    live runner reads both, and so does this.
    """
    stored = store._conn.execute("SELECT * FROM turns WHERE turn_id = ?",
                                 (turn_id,)).fetchone()
    return {**dict(stored), "text": store.turn_texts.get(turn_id, "")}


ADD_MILD = ("add_preference", {"subject": "user", "attribute": "spice_level",
                               "value": "mild", "scope": "permanent"})


# --- one call per user turn -------------------------------------------------

def test_an_assistant_turn_takes_no_extraction_call(store):
    client = FakeClient([])
    ex = make(client)
    ex.run_turn(store, turn_row(store, "t0002"))
    assert client.calls == []


def test_a_user_turn_is_one_call_with_the_tool_schemas(store):
    client = FakeClient([answer(ADD_MILD)])
    ex = make(client)
    turn = say(store, "I usually eat mild food.")
    ok, result = ex.run_turn(store, turn_row(store, turn.turn_id))
    assert ok, result
    (call,) = client.calls
    assert call["tools"] == TOOL_SCHEMAS
    assert call["options"]["num_ctx"] == ex.num_ctx
    assert call["options"]["temperature"] == 0.0
    assert call["model"] == "fake-model"


def test_the_extractor_asks_for_chain_of_thought_explicitly(store):
    """Step 3's numbers were measured with thinking on while the pipeline's
    own calls turn it off (config.PIPELINE_THINK). Pinning it here keeps the
    two roles from drifting together if the client's default ever changes."""
    client = FakeClient([answer(ADD_MILD)])
    ex = make(client)
    turn = say(store, "I usually eat mild food.")
    ex.run_turn(store, turn_row(store, turn.turn_id))
    assert client.calls[0]["think"] is True


def test_the_model_sees_the_turn_text_and_harness_facts_only(store):
    client = FakeClient([answer(ADD_MILD)])
    ex = make(client)
    turn = say(store, "I usually eat mild food.")
    ex.run_turn(store, turn_row(store, turn.turn_id))
    sent = " ".join(m["content"] for m in client.calls[0]["messages"])
    assert client.calls[0]["messages"][-1]["content"] == "I usually eat mild food."
    assert T0.isoformat() in sent            # the harness's clock, not a guess
    assert "t0001" not in sent               # the harness stamps it
    for leaked in ("initial_pref", "temporary_exception", "gold"):
        assert leaked not in sent


def test_relative_phrases_are_resolved_by_the_harness(store):
    """"tonight" must reach the model as a window, not as a word to compute."""
    client = FakeClient([answer(ADD_MILD)])
    ex = make(client)
    turn = say(store, "Only tonight, spicy is fine.")
    ex.run_turn(store, turn_row(store, turn.turn_id))
    sent = " ".join(m["content"] for m in client.calls[0]["messages"])
    assert "tonight" in sent
    # tonight, resolved to the end of the day by families._temporary_window
    assert (T0.replace(hour=23, minute=59, second=0,
                       microsecond=0)).isoformat() in sent


# --- retries ----------------------------------------------------------------

def test_an_unreadable_call_is_retried_and_the_model_sees_why(store):
    client = FakeClient([
        answer(("add_preference", {"subject": "user",
                                   "attribute": "spice_level",
                                   "value": "hot", "scope": "permanent"})),
        answer(ADD_MILD),
    ])
    ex = make(client)
    turn = say(store, "I usually eat mild food.")
    ok, result = ex.run_turn(store, turn_row(store, turn.turn_id))
    assert ok, result
    assert result["tool"] == "add_preference" and result["attempt"] == 1
    retry_messages = client.calls[1]["messages"]
    feedback = [m for m in retry_messages if m["role"] == "tool"]
    assert feedback and "must be one of" in feedback[0]["content"]  # tool's words
    assert _decisions(store) == [("add_preference", 0), ("add_preference", 1)]


def test_retries_are_bounded_and_the_turn_is_recorded_as_failed(store):
    bad = answer(("supersede_preference", {"subject": "user",
                                           "attribute": "spice_level",
                                           "value": "mild"}))
    client = FakeClient([bad] * MAX_ATTEMPTS)
    ex = make(client)
    turn = say(store, "Mild for me, thanks.")
    ok, result = ex.run_turn(store, turn_row(store, turn.turn_id))
    assert not ok
    assert result["failed_after"] == MAX_ATTEMPTS
    assert len(client.calls) == MAX_ATTEMPTS
    assert "nothing to supersede" in result["error"]


def test_a_call_naming_a_tool_that_does_not_exist_is_retried(store):
    client = FakeClient([answer(("forget_everything", {})),
                         answer(("ignore_statement", {"reason": "chatter"}))])
    ex = make(client)
    turn = say(store, "Anyway, how's your day?")
    ok, result = ex.run_turn(store, turn_row(store, turn.turn_id))
    assert ok, result
    assert result["tool"] == "ignore_statement"


def test_a_turn_with_no_tool_call_is_retried_with_an_instruction(store):
    client = FakeClient([_Response([], "I think they like mild food."),
                         answer(ADD_MILD)])
    ex = make(client)
    turn = say(store, "I usually eat mild food.")
    ok, result = ex.run_turn(store, turn_row(store, turn.turn_id))
    assert ok, result
    last_user = client.calls[1]["messages"][-1]
    assert last_user["role"] == "user"
    assert "exactly one tool" in last_user["content"]


def test_an_ollama_exception_is_a_failed_attempt_not_a_crash(store):
    client = FakeClient([RuntimeError("connection reset"), answer(ADD_MILD)])
    ex = make(client)
    turn = say(store, "I usually eat mild food.")
    ok, result = ex.run_turn(store, turn_row(store, turn.turn_id))
    assert ok, result
    assert result["attempt"] == 1


# --- several calls in one turn ---------------------------------------------

def test_a_second_call_in_one_turn_is_refused_and_logged(store):
    """The store cannot carry two statements from one turn honestly; gold
    maps one statement to one call, except the question-and-answer pair,
    which is clarify_statement's own business."""
    client = FakeClient([answer(ADD_MILD,
                                ("ignore_statement", {"reason": "chatter"}))])
    ex = make(client)
    turn = say(store, "I usually eat mild food.")
    ok, result = ex.run_turn(store, turn_row(store, turn.turn_id))
    assert ok
    assert result["extra_calls"] == 1
    extra = store.decisions()[-1]
    assert extra["tool"] == "ignore_statement" and extra["ok"] == 0
    assert "second tool call" in extra["error"]


# --- the store it leaves ----------------------------------------------------

def test_the_row_is_stamped_with_the_turn_and_its_time(store):
    # `row` is the test module's helper; this test also loops over rows.
    client = FakeClient([answer(ADD_MILD)])
    ex = make(client)
    turn = say(store, "I usually eat mild food.", turn_id="t0009",
               ts=T0 + timedelta(hours=1))
    ex.run_turn(store, turn_row(store, turn.turn_id))
    (stored,) = store.all_prefs()
    assert stored["source_turn_id"] == "t0009"
    assert stored["valid_from"] == (T0 + timedelta(hours=1)).isoformat()
    assert stored["source_event_id"] is None   # only gold replay fills it


def _decisions(store):
    return [(r["tool"], r["attempt"]) for r in store.decisions()]


def test_the_system_prompt_glosses_every_attribute():
    """A new attribute must not reach the model without a definition.

    The prompt is the only place the model is told what an attribute
    MEANS; the schema gives it the name. Step 3's first full run lost 76
    excluded_ingredient facts to indirect wording, which a glossary is
    meant to prevent -- so adding a fourth attribute later has to force
    this test red until the glossary is edited too.
    """
    from memproj.tools import ATTRIBUTES, EXTRACTION_SYSTEM
    for attribute in ATTRIBUTES:
        assert attribute in EXTRACTION_SYSTEM, attribute


def test_the_corpus_runner_bounds_each_ollama_request(monkeypatch):
    """A machine that sleeps must cost one failed attempt, not the run.

    Without a timeout the ollama client waits on a socket that no longer
    has a peer, forever: observed on 2026-10-23 as a runner idle for five
    minutes with the server answering a fresh client normally.
    """
    import sys
    import types
    from memproj import config

    seen = {}

    def client(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "ollama",
                        types.SimpleNamespace(Client=client))
    extract.make_client()
    assert seen["timeout"] == config.REQUEST_TIMEOUT_S
