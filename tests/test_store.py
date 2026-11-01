"""Tests for the SQLite store's schema and CHECK constraints.

The tools trust the store to refuse an impossible row. These tests are what
makes that trust safe: each one tries to write something the memory design
says cannot happen and asserts the store says no.

No LLM call appears here, and none may be added: this is rule logic.
"""

import sqlite3
from datetime import datetime

import pytest

from memproj.schema import Turn
from memproj.store import Store


@pytest.fixture
def store():
    s = Store(":memory:").create()
    s.add_turn(Turn("t0001", 0, datetime(2026, 3, 1, 10, 5), "user", "hi"))
    yield s
    s.close()


def add(store, **overrides):
    row = dict(source_turn_id="t0001", subject="user", attribute="spice_level",
               value="mild", scope="permanent",
               valid_from="2026-03-01T10:05:00", valid_until=None,
               ts="2026-03-01T10:05:00")
    row.update(overrides)
    return store.insert_pref(**row)


# --- the happy path ---------------------------------------------------------

def test_all_four_tables_exist(store):
    names = {r["name"] for r in store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"turns", "prefs", "decisions", "clarifications"} <= names


def test_permanent_pref_inserts_without_a_window(store):
    assert add(store) > 0


def test_temporary_pref_inserts_with_a_closed_window(store):
    assert add(store, scope="temporary",
               valid_until="2026-03-01T23:59:00") > 0


# --- prefs: the constraints the resolver relies on --------------------------

def test_permanent_pref_may_not_carry_valid_until(store):
    with pytest.raises(sqlite3.IntegrityError):
        add(store, valid_until="2026-03-01T23:59:00")


def test_temporary_pref_must_carry_valid_until(store):
    with pytest.raises(sqlite3.IntegrityError):
        add(store, scope="temporary")


def test_window_must_be_positive(store):
    """valid_until is exclusive, so a zero-length window is never open."""
    with pytest.raises(sqlite3.IntegrityError):
        add(store, scope="temporary", valid_from="2026-03-01T10:05:00",
            valid_until="2026-03-01T10:05:00")


def test_scope_must_be_known(store):
    with pytest.raises(sqlite3.IntegrityError):
        add(store, scope="forever")


def test_pref_must_point_at_a_real_turn(store):
    with pytest.raises(sqlite3.IntegrityError):
        add(store, source_turn_id="t9999")


def test_a_row_with_an_old_value_must_be_corrected(store):
    """corrected_from is only meaningful on a corrected row; half a
    correction is a bug the database refuses."""
    pid = add(store)
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "UPDATE prefs SET corrected_from = 'spicy' WHERE pref_id = ?",
            (pid,))


def test_a_correction_may_not_be_undone_into_a_half_state(store):
    pid = add(store)
    store.correct_pref(pid, "spicy", "2026-03-02T11:00:00")
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "UPDATE prefs SET status = 'active' WHERE pref_id = ?", (pid,))


def test_correct_pref_keeps_the_old_value_and_the_original_window(store):
    pid = add(store, scope="temporary", valid_until="2026-03-01T23:59:00")
    store.correct_pref(pid, "spicy", "2026-03-02T11:00:00")
    row = store.get_pref(pid)
    assert row["value"] == "spicy"
    assert row["corrected_from"] == "mild"
    assert row["corrected_at"] == "2026-03-02T11:00:00"
    assert row["status"] == "corrected"
    assert row["valid_from"] == "2026-03-01T10:05:00"     # rule 5
    assert row["valid_until"] == "2026-03-01T23:59:00"    # window unchanged


def test_restating_the_stored_value_flags_the_row_without_an_old_value(store):
    """'No, I said spicy' after spicy was already recorded: the row stays,
    but there is no value the user would have been surprised by."""
    pid = add(store, value="spicy")
    store.correct_pref(pid, "spicy", "2026-03-02T11:00:00")
    row = store.get_pref(pid)
    assert row["edited"] == 1 and row["corrected_from"] is None
    assert row["status"] == "corrected"


def test_a_second_correction_keeps_the_first_old_value(store):
    pid = add(store)
    store.correct_pref(pid, "spicy", "2026-03-02T11:00:00")
    store.correct_pref(pid, "medium", "2026-03-03T11:00:00")
    row = store.get_pref(pid)
    assert row["value"] == "medium"
    assert row["corrected_from"] == "mild"


# --- decisions --------------------------------------------------------------

def test_a_failed_decision_must_not_carry_a_result(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.log_decision(turn_id="t0001", tool="add_preference", args_json="{}",
                           ok=False, result_json="{}", error="boom",
                           ts="2026-03-01T10:05:00")


def test_a_successful_decision_must_carry_a_result(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.log_decision(turn_id="t0001", tool="add_preference", args_json="{}",
                           ok=True, ts="2026-03-01T10:05:00")


def test_decision_must_point_at_a_real_turn(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.log_decision(turn_id="t9999", tool="ignore_statement",
                           args_json="{}", ok=True, result_json="{}",
                           ts="2026-03-01T10:05:00")


# --- clarifications ---------------------------------------------------------

def test_clarification_starts_pending(store):
    store.add_clarification(source_turn_id="t0001", question="Which one?",
                            asks_about="excluded_ingredient",
                            ts="2026-03-01T10:05:00")
    pending = store.pending_clarifications()
    assert len(pending) == 1
    assert pending[0]["question"] == "Which one?"


def test_resolved_clarification_leaves_the_pending_list(store):
    cid = store.add_clarification(source_turn_id="t0001", question="Which one?",
                                  ts="2026-03-01T10:05:00")
    store.resolve_clarification(cid, "t0002")
    assert store.pending_clarifications() == []
