"""Tests for the six memory tools and their dispatch.

Every rejection here is a thing an 8B model plausibly does: names a person
that does not exist, passes 'tonight' where a date belongs, tries to
supersede nothing, corrects itself in the same turn. Each must come back as
ok=False with a message the model can act on, and must still be logged, so
extraction can count extraction failures rather than lose them.

No LLM call appears here, and none may be added.
"""

from datetime import datetime

import pytest

from memproj import resolver
from memproj.schema import Turn
from memproj.store import Store
from memproj.tools import TOOL_SCHEMAS, dispatch as _dispatch, parse_ts

T0 = datetime(2026, 3, 1, 10, 5)
T1 = datetime(2026, 3, 2, 11, 0)


@pytest.fixture
def store():
    s = Store(":memory:").create()
    s.add_turn(Turn("t0001", 0, T0, "user", "I usually eat mild food."))
    s.add_turn(Turn("t0002", 1, T1, "user", "Actually, spicy is fine."))
    yield s
    s.close()


def call(store, name, ts=T0, turn="t0001", **args):
    return dispatch(store, name, args, ts, turn)


def dispatch(store, name, args, ts, turn_id):
    """dispatch under another name: `answer` shadows it in one test below."""
    return _dispatch(store, name, args, ts, turn_id)


def active(store, subject="user"):
    """The user's live profile, not everyone's. Other people have their own."""
    return {r["attribute"]: r["value"] for r in store.all_prefs()
            if r["status"] == "active" and r["subject"] == subject}


# --- schema shape -----------------------------------------------------------

def test_six_tools_are_exposed():
    names = [s["function"]["name"] for s in TOOL_SCHEMAS]
    assert names == ["add_preference", "supersede_preference",
                     "clarify_statement", "correct_preference",
                     "ignore_statement", "ask_clarification"]


def test_no_schema_asks_the_model_for_a_timestamp_of_the_statement():
    """ts and turn_id are the harness's. A model may supply valid_until only,
    and only because the harness resolved the phrase for it."""
    for schema in TOOL_SCHEMAS:
        props = schema["function"]["parameters"].get("properties", {})
        assert "ts" not in props
        assert "turn_id" not in props
        assert "event_id" not in props


# --- add_preference ---------------------------------------------------------

def test_add_records_a_permanent_preference(store):
    ok, result = call(store, "add_preference", subject="user",
                      attribute="spice_level", value="mild", scope="permanent")
    assert ok, result
    assert store.get_pref(result["pref_id"])["valid_until"] is None
    assert active(store) == {"spice_level": "mild"}


def test_add_records_a_windowed_exception(store):
    ok, result = call(store, "add_preference", ts=T1, turn="t0002",
                      subject="user", attribute="spice_level", value="spicy",
                      scope="temporary",
                      valid_until="2026-03-02T23:59:00")
    assert ok, result
    row = store.get_pref(result["pref_id"])
    assert row["scope"] == "temporary"
    assert row["valid_until"] == "2026-03-02T23:59:00"


def test_add_temporary_without_a_window_is_refused(store):
    ok, result = call(store, "add_preference", subject="user",
                      attribute="spice_level", value="spicy", scope="temporary")
    assert not ok
    assert "valid_until" in result["error"]


def test_add_rejects_a_relative_time_phrase(store):
    """'tonight' must be resolved by the harness before the call."""
    ok, result = call(store, "add_preference", ts=T1, subject="user",
                      attribute="spice_level", value="spicy", scope="temporary",
                      valid_until="tonight")
    assert not ok
    assert "ISO 8601" in result["error"]


def test_add_rejects_a_window_that_ends_before_it_starts(store):
    ok, result = call(store, "add_preference", ts=T1, subject="user",
                      attribute="spice_level", value="spicy", scope="temporary",
                      valid_until=T1.isoformat())
    assert not ok


def test_add_rejects_an_unknown_subject(store):
    ok, result = call(store, "add_preference", subject="mother",
                      attribute="spice_level", value="mild", scope="permanent")
    assert not ok
    assert "unknown subject" in result["error"]


def test_add_rejects_a_value_outside_the_attribute_vocabulary(store):
    ok, result = call(store, "add_preference", subject="user",
                      attribute="spice_level", value="fiery", scope="permanent")
    assert not ok


def test_add_records_a_second_exclusion_beside_a_standing_one(store):
    """subject_swap's base says 'peanut', then 'mushroom': two standing
    exclusions, because excluded_ingredient is a set, not a slot."""
    call(store, "add_preference", subject="user",
         attribute="excluded_ingredient", value="peanut", scope="permanent")
    ok, result = call(store, "add_preference", ts=T1, turn="t0002",
                      subject="user", attribute="excluded_ingredient",
                      value="mushroom", scope="permanent")
    assert ok, result
    assert active(store) == {"excluded_ingredient": "mushroom"}
    assert len(store.all_prefs()) == 2


def test_add_allows_a_temporary_exception_beside_a_permanent_preference(store):
    call(store, "add_preference", subject="user", attribute="spice_level",
         value="mild", scope="permanent")
    ok, _ = call(store, "add_preference", ts=T1, turn="t0002",
                 subject="user", attribute="spice_level", value="spicy",
                 scope="temporary", valid_until="2026-03-02T23:59:00")
    assert ok


# --- supersede_preference ---------------------------------------------------

def test_supersede_marks_the_old_row_and_records_the_new_one(store):
    _, first = call(store, "add_preference", subject="user",
                    attribute="spice_level", value="mild", scope="permanent")
    ok, second = call(store, "supersede_preference", ts=T1, turn="t0002",
                      subject="user", attribute="spice_level", value="spicy")
    assert ok, second
    assert store.get_pref(first["pref_id"])["status"] == "superseded"
    assert store.get_pref(second["pref_id"])["supersedes"] == first["pref_id"]
    assert active(store) == {"spice_level": "spicy"}


def test_supersede_with_nothing_standing_is_refused(store):
    ok, result = call(store, "supersede_preference", subject="user",
                      attribute="spice_level", value="spicy")
    assert not ok
    assert "add_preference" in result["error"]


def test_supersede_to_the_same_value_is_refused(store):
    call(store, "add_preference", subject="user", attribute="spice_level",
         value="mild", scope="permanent")
    ok, result = call(store, "supersede_preference", ts=T1, turn="t0002",
                      subject="user", attribute="spice_level", value="mild")
    assert not ok


def _mild_with_a_lapsed_spicy_exception(store):
    """mild stands; spicy was granted for one evening and has lapsed.

    Both rows keep `status = 'active'` -- the exception's window, not the
    status, is what says it stopped applying. Returns the mild row and a
    moment well after the window.
    """
    _, mild = call(store, "add_preference", subject="user",
                   attribute="spice_level", value="mild", scope="permanent")
    ok, _ = call(store, "add_preference", ts=T1, turn="t0002",
                 subject="user", attribute="spice_level", value="spicy",
                 scope="temporary", valid_until="2026-03-02T23:00:00")
    assert ok
    later = datetime(2026, 3, 5, 10, 0)
    store.add_turn(Turn("t0003", 2, later, "user", "Spicy from now on."))
    return mild, later


def test_supersede_replaces_the_standing_row_not_a_lapsed_exception(store):
    """The row a revision replaces is the one in force, not the newest row.

    A lapsed exception is still `active`; superseding it instead of the
    standing preference leaves the old value beside the new one.
    """
    mild, later = _mild_with_a_lapsed_spicy_exception(store)

    ok, result = call(store, "supersede_preference", ts=later, turn="t0003",
                      subject="user", attribute="spice_level", value="medium")

    assert ok, result
    assert store.get_pref(result["pref_id"])["supersedes"] == mild["pref_id"]
    assert resolver.active_profile(store, "user", later) == {
        "spice_level": {"medium"}}


def test_supersede_may_repeat_a_value_that_has_lapsed(store):
    """"Spicy from now on" after a spicy exception has lapsed is a revision.

    `supersede_preference`'s own comment names this case; the guard used to
    refuse it because it compared against the exception's value.
    """
    _, later = _mild_with_a_lapsed_spicy_exception(store)

    ok, result = call(store, "supersede_preference", ts=later, turn="t0003",
                      subject="user", attribute="spice_level", value="spicy")

    assert ok, result
    assert resolver.active_profile(store, "user", later) == {
        "spice_level": {"spicy"}}


# --- other people never touch the user --------------------------------------

def test_a_brothers_preference_is_recorded_under_his_name(store):
    ok, result = call(store, "add_preference", subject="brother",
                      attribute="spice_level", value="spicy", scope="permanent")
    assert ok, result
    assert store.get_pref(result["pref_id"])["subject"] == "brother"
    assert active(store) == {}                                  # the user gained nothing
    assert active(store, subject="brother") == {"spice_level": "spicy"}


# --- correct_preference -----------------------------------------------------

def test_correct_rewrites_the_value_and_keeps_valid_from(store):
    _, rec = call(store, "add_preference", subject="user",
                  attribute="excluded_ingredient", value="mushroom",
                  scope="permanent")
    ok, result = call(store, "correct_preference", ts=T1, turn="t0002",
                      target_pref_id=rec["pref_id"], value="peanut")
    assert ok, result
    row = store.get_pref(rec["pref_id"])
    assert row["value"] == "peanut"
    assert row["corrected_from"] == "mushroom"
    assert row["valid_from"] == T0.isoformat()      # rule 5


def test_correct_rejects_an_unknown_pref_id(store):
    ok, result = call(store, "correct_preference", target_pref_id=999,
                      value="peanut")
    assert not ok
    assert "999" in result["error"]


def test_correct_may_not_target_the_turn_being_read(store):
    _, rec = call(store, "add_preference", turn="t0001", subject="user",
                  attribute="excluded_ingredient", value="mushroom",
                  scope="permanent")
    ok, result = call(store, "correct_preference", turn="t0001",
                      target_pref_id=rec["pref_id"], value="peanut")
    assert not ok
    assert "current turn" in result["error"]


# --- ignore_statement -------------------------------------------------------

def test_ignore_logs_and_changes_nothing(store):
    ok, result = call(store, "ignore_statement", reason="hypothetical")
    assert ok
    assert store.all_prefs() == []
    assert store.decisions()[-1]["tool"] == "ignore_statement"


def test_ignore_rejects_an_unknown_reason(store):
    ok, _ = call(store, "ignore_statement", reason="bored")
    assert not ok


# --- ask_clarification ------------------------------------------------------

def test_ask_clarification_leaves_a_pending_question(store):
    ok, result = call(store, "ask_clarification", turn="t0002", ts=T1,
                      question="Do you mean the mushroom or the peanut one?",
                      asks_about="excluded_ingredient")
    assert ok, result
    pending = store.pending_clarifications()
    assert len(pending) == 1
    assert pending[0]["clarification_id"] == result["clarification_id"]


def test_clarification_does_not_touch_the_profile(store):
    call(store, "add_preference", subject="user", attribute="spice_level",
         value="mild", scope="permanent")
    call(store, "ask_clarification", turn="t0002", ts=T1, question="Which one?")
    assert active(store) == {"spice_level": "mild"}


# --- clarify_statement ------------------------------------------------------
# One tool for a vague statement: the question the user hears, and what
# their answer resolves. Closing the answer in a second call let the store
# hold a question nobody would ask again; a checkpoint then saw two vague
# statements where the user's conversation had one open question.

T2 = datetime(2026, 3, 3, 12, 0)


def _two_restrictions(store):
    """Two exclusions, one per turn; each row starts at its statement."""
    _, peanut = call(store, "add_preference", subject="user",
                     attribute="excluded_ingredient", value="peanut",
                     scope="permanent")
    _, mushroom = call(store, "add_preference", ts=T1, turn="t0002",
                       subject="user", attribute="excluded_ingredient",
                       value="mushroom", scope="permanent")
    return store.get_pref(peanut["pref_id"]), store.get_pref(mushroom["pref_id"])


def test_clarify_statement_with_a_named_answer_drops_that_record_now(store):
    peanut, mushroom = _two_restrictions(store)
    # The answer arrives in a LATER turn: dropping it in the asking turn
    # would make the record never have stood in the store at all.
    ok, result = call(store, "clarify_statement", turn="t0002", ts=T1,
                      question="Which restriction is no longer important?",
                      resolves={"target_pref_id": mushroom["pref_id"]})
    assert ok, result
    assert result["status"] == "resolved"
    assert result["removed"]["pref_id"] == mushroom["pref_id"]
    assert result["removed"]["dropped_at"] == T1.isoformat()
    assert active(store) == {"excluded_ingredient": "peanut"}
    assert store.pending_clarifications() == []
    assert store.get_pref(mushroom["pref_id"])["status"] == "superseded"


def test_clarify_by_value_drops_only_the_standing_record(store):
    _two_restrictions(store)
    ok, result = call(store, "clarify_statement", turn="t0002", ts=T1,
                      question="Which one?",
                      resolves={"value": "mushroom"})
    assert ok, result
    assert result["removed"]["value"] == "mushroom"
    assert active(store) == {"excluded_ingredient": "peanut"}


def test_clarify_with_an_unrecognised_handle_is_refused_not_raised(store):
    """A model that writes {"pref_id": 3} instead of {"target_pref_id": 3}
    must get a refusal it can retry, not an exception that loses the turn.

    `dispatch` promises to never raise for model-supplied input; this was
    the one key lookup that broke that promise (seen as `KeyError:
    'value'` in four statements of the first full extraction run).
    """
    _two_restrictions(store)
    ok, result = _dispatch(
        store, "clarify_statement",
        {"question": "Which one?", "resolves": {"pref_id": 1}},
        T1, "t0002")
    assert not ok
    assert "pref_id" in result["error"] and "value" in result["error"]


def test_clarify_without_an_answer_asks_and_leaves_every_row(store):
    peanut, mushroom = _two_restrictions(store)
    ok, result = call(store, "clarify_statement", turn="t0002", ts=T1,
                      question="Which restriction is no longer important?")
    assert ok, result
    assert result["status"] == "pending" and result["resolved"] is False
    assert len(store.pending_clarifications()) == 1
    assert active(store) == {"excluded_ingredient": "mushroom"}
    assert store.get_pref(mushroom["pref_id"])["status"] == "active"


def test_an_answer_naming_nothing_leaves_the_question_open(store):
    """"The pepper one" when pepper was never excluded: no record, no guess."""
    _two_restrictions(store)
    ok, result = call(store, "clarify_statement", turn="t0002", ts=T1,
                      question="Which one?",
                      resolves={"value": "pepper"})
    assert ok, result
    assert result["resolved"] is False
    assert len(store.pending_clarifications()) == 1
    assert active(store) == {"excluded_ingredient": "mushroom"}


def test_a_named_removal_keeps_the_record_true_before_it(store):
    """Dropped at noon: asked at that instant, the morning's mushroom stands.

    The row is the user's, not the agent's: "that restriction is no longer
    important" changes what is true from the moment they said it, and the
    exclusion they stated an hour before was in force until then.
    """
    from datetime import timedelta
    from memproj.resolver import active_profile
    _, mushroom = _two_restrictions(store)
    answer = T1 + timedelta(hours=1)
    store.add_turn(Turn("t0003", 1, answer, "user", "The mushroom one."))
    call(store, "clarify_statement", turn="t0003", ts=answer, question="Which?",
         resolves={"target_pref_id": mushroom["pref_id"]})
    assert active_profile(store, "user", T1) == {
        "excluded_ingredient": {"peanut", "mushroom"}}
    assert active_profile(store, "user", answer) == {
        "excluded_ingredient": {"peanut"}}


def test_clarify_statement_rejects_an_unknown_attribute(store):
    ok, result = call(store, "clarify_statement", question="Which one?",
                      asks_about="mood")
    assert not ok
    assert "asks_about" in result["error"]


def test_clarify_statement_rejects_an_empty_question(store):
    ok, result = call(store, "clarify_statement", question="  ")
    assert not ok


# --- dispatch ---------------------------------------------------------------

def test_unknown_tool_is_reported_not_raised(store):
    ok, result = call(store, "forget_everything")
    assert not ok
    assert "unknown tool" in result["error"]


def test_non_object_arguments_are_reported_not_raised(store):
    ok, result = dispatch(store, "add_preference", "not a dict", T0, "t0001")
    assert not ok


def test_every_call_is_logged_including_failures(store):
    call(store, "add_preference", subject="user", attribute="spice_level",
         value="mild", scope="permanent")
    call(store, "supersede_preference", subject="user",
         attribute="spice_level", value="fiery")
    calls = store.decisions()
    assert [c["tool"] for c in calls] == ["add_preference",
                                          "supersede_preference"]
    assert [c["ok"] for c in calls] == [1, 0]
    assert calls[1]["error"]


def test_dispatch_stamps_the_harness_timestamp_not_the_args(store):
    call(store, "add_preference", ts=T1, turn="t0002", subject="user",
         attribute="spice_level", value="mild", scope="permanent")
    row = store.all_prefs()[0]
    assert row["ts"] == T1.isoformat()
    assert row["source_turn_id"] == "t0002"


# --- parse_ts ---------------------------------------------------------------

def test_parse_ts_accepts_absolute_iso():
    assert parse_ts("2026-03-01T23:59:00") == datetime(2026, 3, 1, 23, 59)


@pytest.mark.parametrize("phrase", ["tonight", "tomorrow", "this weekend", ""])
def test_parse_ts_rejects_relative_phrases(phrase):
    with pytest.raises(ValueError):
        parse_ts(phrase)


def test_parse_ts_rejects_a_non_string():
    with pytest.raises(ValueError):
        parse_ts(1759)


def test_a_free_text_ingredient_is_stored_in_catalogue_form(store):
    """'peanuts' and 'peanut' must be one fact, not two.

    `excluded_ingredient` is free text and `recipes._normalise` already
    defines how the catalogue spells it. Without this, the model's plural
    is a different value from gold's, and the extractor is charged a miss
    for a statement it recorded correctly -- 10 of 106 failing rows in the
    first full run were exactly this.
    """
    ok, result = call(store, "add_preference", subject="user",
                      attribute="excluded_ingredient", value="Peanuts",
                      scope="permanent")
    assert ok, result
    assert store.get_pref(result["pref_id"])["value"] == "peanut"
    assert active(store) == {"excluded_ingredient": "peanut"}


def test_a_plural_revision_supersedes_the_singular_record(store):
    """The two spellings must also find each other when revising."""
    call(store, "add_preference", subject="user",
         attribute="excluded_ingredient", value="peanut", scope="permanent")
    ok, result = call(store, "supersede_preference", turn="t0002", ts=T1,
                      subject="user", attribute="excluded_ingredient",
                      value="Mushrooms")
    assert ok, result
    assert active(store) == {"excluded_ingredient": "mushroom"}
