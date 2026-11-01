"""Tests for the gold-state rules.

Each test is a hand-written event list with the expected profile at one or
more `now` values, so the failure message tells you which rule broke.
"""

import pytest
from datetime import datetime

from memproj.semantics import gold_profile, history, gold_action, relevance
from memproj.schema import Event


def dt(day, hour=12, minute=0):
    return datetime(2026, 3, day, hour, minute)


def ev(event_id, ts, type, subject="user", attribute="spice_level", value="spicy",
       scope="permanent", valid_from=None, valid_until=None,
       target_event_id=None, needs_clarification=False, referent_event_id=None,
       session=1):
    return Event(
        event_id=event_id,
        session=session,
        ts=ts,
        type=type,
        subject=subject,
        attribute=attribute,
        value=value,
        scope=scope,
        valid_from=valid_from or ts,
        valid_until=valid_until,
        target_event_id=target_event_id,
        needs_clarification=needs_clarification,
        referent_event_id=referent_event_id,
    )


# --- initial_pref -----------------------------------------------------------

def test_initial_pref_visible_after_statement():
    e = ev("e1", dt(1), "initial_pref", attribute="diet", value="vegetarian")
    assert gold_profile([e], "user", dt(1, 18)) == {"diet": {"vegetarian"}}


def test_initial_pref_invisible_before_statement():
    e = ev("e1", dt(5), "initial_pref", attribute="diet", value="vegetarian")
    assert gold_profile([e], "user", dt(1)) == {}


# --- subject isolation ------------------------------------------------------

def test_other_person_never_leaks_into_user():
    e = ev("e1", dt(1), "other_person", subject="brother", value="spicy")
    assert gold_profile([e], "user", dt(2)) == {}
    assert gold_profile([e], "brother", dt(2)) == {"spice_level": {"spicy"}}


def test_user_event_never_leaks_into_other_person():
    e = ev("e1", dt(1), "initial_pref", subject="user", value="spicy")
    assert gold_profile([e], "brother", dt(2)) == {}


# --- temporary_exception ----------------------------------------------------

def make_temp():
    # Stated on day 1 at 18:00, applies through day 1 (valid_until day 2 00:00).
    return ev(
        "e1", dt(1, 18), "temporary_exception",
        scope="temporary",
        valid_from=dt(1, 18),
        valid_until=dt(2, 0),
    )


def test_temporary_not_yet_active_before_window():
    assert gold_profile([make_temp()], "user", dt(1, 12)) == {}


def test_temporary_active_inside_window():
    assert gold_profile([make_temp()], "user", dt(1, 20)) == {"spice_level": {"spicy"}}


def test_temporary_expired_exactly_at_valid_until():
    """valid_until is exclusive: at that instant the exception is gone."""
    assert gold_profile([make_temp()], "user", dt(2, 0)) == {}


def test_temporary_expired_after_window():
    assert gold_profile([make_temp()], "user", dt(2, 12)) == {}


def test_temporary_does_not_remove_the_permanent_value():
    perm = ev("e1", dt(1, 9), "initial_pref", value="mild")
    temp = ev("e2", dt(1, 18), "temporary_exception", scope="temporary",
              value="spicy", valid_from=dt(1, 18), valid_until=dt(2, 0))
    assert gold_profile([perm, temp], "user", dt(1, 20)) == {"spice_level": {"mild", "spicy"}}
    assert gold_profile([perm, temp], "user", dt(2, 12)) == {"spice_level": {"mild"}}


# --- permanent_revision -----------------------------------------------------

def test_revision_replaces_prior_value():
    old = ev("e1", dt(1), "initial_pref", value="mild")
    new = ev("e2", dt(3), "permanent_revision", value="spicy", target_event_id="e1")
    assert gold_profile([old, new], "user", dt(2)) == {"spice_level": {"mild"}}
    assert gold_profile([old, new], "user", dt(4)) == {"spice_level": {"spicy"}}


def test_revision_does_not_touch_other_attributes():
    diet = ev("e1", dt(1), "initial_pref", attribute="diet", value="vegetarian")
    old = ev("e2", dt(1), "initial_pref", value="mild")
    new = ev("e3", dt(3), "permanent_revision", value="spicy", target_event_id="e2")
    assert gold_profile([diet, old, new], "user", dt(4)) == {
        "diet": {"vegetarian"}, "spice_level": {"spicy"},
    }


def test_superseded_record_survives_in_history():
    old = ev("e1", dt(1), "initial_pref", value="mild")
    new = ev("e2", dt(3), "permanent_revision", value="spicy", target_event_id="e1")
    ids = [e.event_id for e in history([old, new], "user", dt(4))]
    assert ids == ["e1", "e2"]


# --- correction -------------------------------------------------------------

def test_correction_rewrites_value():
    orig = ev("e1", dt(1), "initial_pref", attribute="excluded_ingredient", value="peanut")
    fix = ev("e2", dt(2), "correction", attribute="excluded_ingredient",
             value="mushroom", target_event_id="e1")
    assert gold_profile([orig, fix], "user", dt(3)) == {"excluded_ingredient": {"mushroom"}}


def test_correction_keeps_original_valid_from():
    """A correction fixes a misheard value; it is not a new statement."""
    orig = ev("e1", dt(1), "initial_pref", attribute="excluded_ingredient", value="peanut")
    fix = ev("e2", dt(2), "correction", attribute="excluded_ingredient",
             value="mushroom", target_event_id="e1")
    profile, records = gold_profile([orig, fix], "user", dt(3), with_records=True)
    assert records["excluded_ingredient"][0].valid_from == dt(1)


def test_correction_not_applied_before_it_happens():
    orig = ev("e1", dt(1), "initial_pref", attribute="excluded_ingredient", value="peanut")
    fix = ev("e2", dt(5), "correction", attribute="excluded_ingredient",
             value="mushroom", target_event_id="e1")
    assert gold_profile([orig, fix], "user", dt(3)) == {"excluded_ingredient": {"peanut"}}


# --- inert types ------------------------------------------------------------

def test_hypothetical_never_changes_state():
    h = ev("e1", dt(1), "hypothetical", value="spicy")
    assert gold_profile([h], "user", dt(9)) == {}


def test_hypothetical_does_not_resurrect_anything():
    base = ev("e1", dt(1), "initial_pref", value="mild")
    h = ev("e2", dt(2), "hypothetical", value="spicy")
    assert gold_profile([base, h], "user", dt(9)) == {"spice_level": {"mild"}}


def test_filler_never_changes_state():
    f = ev("e1", dt(1), "filler", attribute=None, value=None)
    assert gold_profile([f], "user", dt(9)) == {}


# --- ambiguous_update -------------------------------------------------------

def make_ambiguity(resolved=False):
    """Two active restrictions, then an ambiguous 'that one no longer matters'.

    Unresolved: the ambiguity stands from day 2 onward.
    Resolved: a later event on day 3 names the peanut restriction as referent.
    """
    peanut = ev("e1", dt(1), "initial_pref", attribute="excluded_ingredient", value="peanut")
    mushroom = ev("e2", dt(1), "initial_pref", attribute="excluded_ingredient", value="mushroom")
    amb = ev("e3", dt(2), "ambiguous_update", attribute=None, value=None,
             needs_clarification=True, referent_event_id=None)
    events = [peanut, mushroom, amb]
    if resolved:
        events.append(ev("e4", dt(3), "ambiguous_update", attribute=None, value=None,
                         referent_event_id="e1"))
    return events


def test_ambiguous_unresolved_changes_nothing():
    e = make_ambiguity(resolved=False)
    assert gold_profile(e, "user", dt(2, 18)) == {
        "excluded_ingredient": {"peanut", "mushroom"},
    }


def test_ambiguous_unresolved_requires_clarify():
    e = make_ambiguity(resolved=False)
    assert gold_action(e, "user", dt(2, 18)) == "clarify"


def test_ambiguous_before_it_happens_recommends():
    e = make_ambiguity(resolved=False)
    assert gold_action(e, "user", dt(2) - (dt(2) - dt(1))) == "recommend"
    assert gold_action(e, "user", dt(1, 18)) == "recommend"


def test_ambiguous_resolved_removes_only_the_referent():
    e = make_ambiguity(resolved=True)
    assert gold_profile(e, "user", dt(4)) == {"excluded_ingredient": {"mushroom"}}


def test_ambiguous_resolved_returns_to_recommend():
    e = make_ambiguity(resolved=True)
    assert gold_action(e, "user", dt(2, 18)) == "clarify"
    assert gold_action(e, "user", dt(4)) == "recommend"


def test_ambiguous_with_single_active_restriction_is_not_ambiguous():
    """Ambiguity needs 2+ active restrictions; with one, the referent is clear."""
    only = ev("e1", dt(1), "initial_pref", attribute="excluded_ingredient", value="peanut")
    amb = ev("e2", dt(2), "ambiguous_update", attribute=None, value=None,
             needs_clarification=True)
    assert gold_action([only, amb], "user", dt(3)) == "recommend"


# --- relevance / distractors ------------------------------------------------

def test_expired_exception_is_a_distractor_not_relevant():
    perm = ev("e1", dt(1), "initial_pref", value="mild")
    temp = ev("e2", dt(1, 18), "temporary_exception", scope="temporary",
              value="spicy", valid_from=dt(1, 18), valid_until=dt(2, 0))
    rel, dis = relevance([perm, temp], "user", dt(2, 12))
    assert "e2" in dis and "e2" not in rel
    assert "e1" in rel


def test_active_exception_is_relevant():
    perm = ev("e1", dt(1), "initial_pref", value="mild")
    temp = ev("e2", dt(1, 18), "temporary_exception", scope="temporary",
              value="spicy", valid_from=dt(1, 18), valid_until=dt(2, 0))
    rel, dis = relevance([perm, temp], "user", dt(1, 20))
    assert "e2" in rel


def test_other_subject_is_a_distractor_for_user():
    mine = ev("e1", dt(1), "initial_pref", value="mild")
    theirs = ev("e2", dt(1), "other_person", subject="brother", value="spicy")
    rel, dis = relevance([mine, theirs], "user", dt(2))
    assert "e2" in dis and "e2" not in rel


def test_superseded_revision_is_a_distractor():
    old = ev("e1", dt(1), "initial_pref", value="mild")
    new = ev("e2", dt(3), "permanent_revision", value="spicy", target_event_id="e1")
    rel, dis = relevance([old, new], "user", dt(4))
    assert "e1" in dis and "e2" in rel


# --- history ----------------------------------------------------------------

def test_history_excludes_future_events():
    old = ev("e1", dt(1), "initial_pref", value="mild")
    new = ev("e2", dt(5), "permanent_revision", value="spicy", target_event_id="e1")
    assert [e.event_id for e in history([old, new], "user", dt(2))] == ["e1"]


def test_history_is_ordered_by_ts():
    b = ev("e1", dt(3), "initial_pref", value="mild")
    a = ev("e2", dt(1), "initial_pref", value="mild")
    assert [e.event_id for e in history([b, a], "user", dt(9))] == ["e2", "e1"]


# --- determinism ------------------------------------------------------------

def test_ties_break_on_event_id():
    a = ev("e1", dt(1), "initial_pref", value="mild")
    b = ev("e2", dt(1), "permanent_revision", value="spicy", target_event_id="e1")
    c = ev("e0", dt(1), "permanent_revision", value="medium", target_event_id="e1")
    # e0 sorts first, then e1, then e2 -> e2's "spicy" wins, deterministically.
    assert gold_profile([a, b, c], "user", dt(2)) == {"spice_level": {"spicy"}}
    assert gold_profile([c, b, a], "user", dt(2)) == {"spice_level": {"spicy"}}
