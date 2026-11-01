"""SQLite store: four tables, one file per (trajectory, method) run.

The tool layer writes here; the resolver reads here. Nothing else does.

Timestamps are stored as the same naive ISO strings the generator writes
(README, Recipe catalogue), so a string comparison on `ts` is a time comparison
and nobody has to remember a timezone that does not exist.

Every table has a CHECK that enforces a rule the memory design depends on,
rather than trusting the caller. A CHECK failure is a bug in the caller, so
insert() lets sqlite3.IntegrityError escape: a run that quietly recorded an
impossible row would make every number downstream meaningless.
"""

import json
import os
import sqlite3
from typing import Dict, List, Optional

from memproj.schema import Event

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    turn_id   TEXT PRIMARY KEY,
    session   INTEGER NOT NULL,
    ts        TEXT    NOT NULL,
    speaker   TEXT    NOT NULL CHECK (speaker IN ('user', 'assistant'))
);

-- One row per preference the agent chose to remember. A row is never deleted:
-- superseding marks the old row, keeping history() answerable.
CREATE TABLE IF NOT EXISTS prefs (
    pref_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    source_turn_id  TEXT    NOT NULL REFERENCES turns(turn_id),
    source_event_id TEXT,                       -- gold replay records the event it read
    subject         TEXT    NOT NULL,
    attribute       TEXT    NOT NULL,
    value           TEXT    NOT NULL,
    scope           TEXT    NOT NULL CHECK (scope IN ('permanent', 'temporary')),
    valid_from      TEXT    NOT NULL,
    valid_until     TEXT,
    supersedes      INTEGER REFERENCES prefs(pref_id),
    closed_by       INTEGER,                    -- the pref whose revision or
                                                -- removal ended this one
    dropped_at      TEXT,                       -- the removal turn; NULL for a
                                                -- revision, which says "from now"
    status          TEXT    NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'superseded', 'corrected')),
    corrected_from  TEXT,                       -- the value before a correction
    corrected_at    TEXT,                       -- the turn that corrected it
    edited          INTEGER NOT NULL DEFAULT 0 -- the user restated this row's
                    CHECK (edited IN (0, 1)),  -- value; needs a recent turn
    ts              TEXT    NOT NULL,           -- when the agent recorded it
    CHECK ((scope = 'permanent') = (valid_until IS NULL)),
    CHECK (valid_until IS NULL OR valid_until > valid_from),
    -- A correction that only confirms the stored value leaves corrected_from
    -- NULL, so the pairing rule is one-way: an old value implies 'corrected'.
    CHECK (corrected_from IS NULL OR status = 'corrected')
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id            TEXT    NOT NULL REFERENCES turns(turn_id),
    tool               TEXT    NOT NULL,
    args_json          TEXT    NOT NULL,
    result_json        TEXT,                    -- NULL means the call failed
    ok                 INTEGER NOT NULL CHECK (ok IN (0, 1)),
    error              TEXT,
    attempt            INTEGER NOT NULL DEFAULT 0,
    ts                 TEXT    NOT NULL,
    CHECK ((ok = 1) = (result_json IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS clarifications (
    clarification_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_turn_id   TEXT    NOT NULL REFERENCES turns(turn_id),
    question         TEXT    NOT NULL,
    asks_about       TEXT,                      -- e.g. 'excluded_ingredient'
    status           TEXT    NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending', 'resolved', 'dismissed')),
    resolved_by_turn TEXT,
    ts               TEXT    NOT NULL,
    CHECK ((status = 'pending') = (resolved_by_turn IS NULL))
);

CREATE INDEX IF NOT EXISTS prefs_subject_attr ON prefs(subject, attribute);
CREATE INDEX IF NOT EXISTS decisions_turn     ON decisions(turn_id);
"""


class Store:
    """One SQLite database for one (trajectory, method) run."""

    def __init__(self, path: str, conn: Optional[sqlite3.Connection] = None):
        self.path = path
        self._conn = conn if conn is not None else sqlite3.connect(path)
        # Rows come back as dicts so callers name their columns instead of
        # counting positions in a SELECT *.
        self._conn.row_factory = _dict_rows
        # SQLite ships with foreign keys OFF. The turn references below are
        # only meaningful with them ON, so a pref can never outlive the turn
        # it was read from.
        self._conn.execute("PRAGMA foreign_keys = ON")
        # turn_id -> the words of that turn. The table keeps who and when
        # only, so the extraction loop can
        # fill a prompt with sentences by looking turns up here.
        self.turn_texts: Dict[str, str] = {}

    # --- lifecycle ----------------------------------------------------------

    def create(self) -> "Store":
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        return self

    def close(self) -> None:
        self._conn.close()

    def backup_to(self, path: str) -> None:
        """Write a consistent copy of this store to `path`, atomically.

        SQLite's backup API, not a file copy: a copied database can be read
        mid-transaction. The copy lands on a temporary name first, so a run
        that dies while saving leaves the previous good file untouched.
        """
        tmp = f"{path}.tmp"
        if os.path.exists(tmp):
            os.remove(tmp)
        destination = sqlite3.connect(tmp)
        try:
            with destination:
                self._conn.backup(destination)
        finally:
            destination.close()
        os.replace(tmp, path)

    def __enter__(self):
        return self.create()

    def __exit__(self, *exc):
        self.close()

    # --- turns --------------------------------------------------------------

    def add_turn(self, turn) -> None:
        """Copy one generated turn into the store.

        Turn ids are per-trajectory in the generator, so a store must hold
        exactly one trajectory. Replaying a second trajectory into the same
        file would silently merge two conversations.
        """
        self._conn.execute(
            "INSERT INTO turns (turn_id, session, ts, speaker) VALUES (?, ?, ?, ?)",
            (turn.turn_id, turn.session, turn.ts.isoformat(), turn.speaker),
        )
        self._conn.commit()

    def turn_ids(self) -> List[str]:
        return [r[0] for r in self._conn.execute(
            "SELECT turn_id FROM turns ORDER BY ts, turn_id")]

    # --- prefs --------------------------------------------------------------

    def insert_pref(self, *, source_turn_id: str, subject: str, attribute: str,
                    value: str, scope: str, valid_from: str,
                    valid_until: Optional[str], ts: str,
                    source_event_id: Optional[str] = None,
                    supersedes: Optional[int] = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO prefs (source_turn_id, source_event_id, subject,"
            " attribute, value, scope, valid_from, valid_until, supersedes, ts)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (source_turn_id, source_event_id, subject, attribute, value, scope,
             valid_from, valid_until, supersedes, ts),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def get_pref(self, pref_id: int):
        return self._conn.execute(
            "SELECT * FROM prefs WHERE pref_id = ?", (pref_id,)).fetchone()

    def all_prefs(self):
        return self._conn.execute(
            "SELECT * FROM prefs ORDER BY pref_id").fetchall()

    def set_status(self, pref_id: int, status: str,
                   closed_by_pref_id: Optional[int] = None,
                   dropped_at: Optional[str] = None) -> None:
        """End a row, naming the row that ended it where one exists.

        Rewriting a row (a correction) names no closer; a supersede or a
        removal names the statement that did it, which is what `history()`
        needs to show why an old preference stopped applying.

        `dropped_at` is the moment a NAMED row stops counting, and only a
        removal sets it: "that restriction is no longer important", said at
        noon, leaves the row true up to noon -- the user ate before they
        said it. A revision's new row carries the turn as its valid_from
        already, so a revision sets only `closed_by_pref_id`.
        """
        when = dropped_at is not None
        self._conn.execute(
            "UPDATE prefs SET status = ?, closed_by = ?,"
            " dropped_at = CASE WHEN ? THEN ? ELSE dropped_at END"
            " WHERE pref_id = ?",
            (status, closed_by_pref_id, int(when), dropped_at, pref_id))
        self._conn.commit()

    def correct_pref(self, pref_id: int, new_value: str, at: str) -> None:
        """Rewrite a value in place, keeping valid_from (rule 5).

        Correcting a row to the value it already holds is a legitimate call:
        the user is restating a preference they think was misheard
        ("no, I said spicy"). The row is flagged `edited` either way. When
        the value changes, corrected_from keeps the first value the user
        would have been surprised by -- NULL when a correction only confirms
        what was already recorded.

        `at` is the correcting turn's timestamp, supplied by the harness
        like every other time. The resolver reads it to decide whether the
        correction reaches back to valid_from or only starts at `at`.
        """
        self._conn.execute(
            "UPDATE prefs SET corrected_from = COALESCE(corrected_from,"
            " CASE WHEN value = ? THEN NULL ELSE value END),"
            " corrected_at = ?, value = ?, status = 'corrected', edited = 1"
            " WHERE pref_id = ?",
            (new_value, at, new_value, pref_id))
        self._conn.commit()

    # --- decisions and clarifications ---------------------------------------

    def log_decision(self, *, turn_id: str, tool: str, args_json: str,
                     ok: bool, ts: str, result_json: Optional[str] = None,
                     error: Optional[str] = None, attempt: int = 0) -> None:
        self._conn.execute(
            "INSERT INTO decisions (turn_id, tool, args_json, result_json, ok,"
            " error, attempt, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (turn_id, tool, args_json, result_json, int(ok), error, attempt, ts))
        self._conn.commit()

    def decisions(self):
        return self._conn.execute(
            "SELECT * FROM decisions ORDER BY decision_id").fetchall()

    def failed_decisions(self):
        return self._conn.execute(
            "SELECT * FROM decisions WHERE ok = 0 ORDER BY decision_id").fetchall()

    def add_clarification(self, *, source_turn_id: str, question: str, ts: str,
                          asks_about: Optional[str] = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO clarifications (source_turn_id, question, asks_about, ts)"
            " VALUES (?, ?, ?, ?)",
            (source_turn_id, question, asks_about, ts))
        self._conn.commit()
        return int(cur.lastrowid)

    def resolve_clarification(self, clarification_id: int,
                              turn_id: str) -> None:
        self._conn.execute(
            "UPDATE clarifications SET status = 'resolved', resolved_by_turn = ?"
            " WHERE clarification_id = ?",
            (turn_id, clarification_id))
        self._conn.commit()

    def pending_clarifications(self):
        return self._conn.execute(
            "SELECT * FROM clarifications WHERE status = 'pending'"
            " ORDER BY clarification_id").fetchall()

    def clarifications_with_resolution(self):
        """Every question, with the time the answering turn happened.

        `resolved_ts` is NULL both for a question nobody answered and for one
        resolved by a turn the store has no row for; the status column tells
        those apart.
        """
        return self._conn.execute(
            "SELECT c.*, t.ts AS resolved_ts FROM clarifications c"
            " LEFT JOIN turns t ON t.turn_id = c.resolved_by_turn"
            " ORDER BY c.clarification_id").fetchall()


# --- replay helpers: the store's only knowledge of the schema -----------------

def _iso(ts) -> str:
    """A datetime or an already-ISO string, as the naive string the store uses."""
    return ts if isinstance(ts, str) else ts.isoformat()


def record_event(store: Store, event: Event, turn_id: str) -> int:
    """Record one gold event as the preference rows it implies.

    This is the gold stand-in for an LLM extraction call: the mapping from
    "what the user said" to "which tool call models that" is written out
    explicitly here, and the extraction loop replaces it with model output
    that has to produce the same calls.

    The harness supplies `event` and `turn_id`; the model never does.
    """
    ts = _iso(event.ts)
    if event.type in ("filler", "hypothetical"):
        store.log_decision(turn_id=turn_id, tool="ignore_statement",
                           args_json=_json({"event_id": event.event_id}),
                           ok=True, result_json=_json({"ignored": True}), ts=ts)
        return 0

    if event.type == "ambiguous_update" and event.referent_event_id is None:
        return store.add_clarification(
            source_turn_id=turn_id,
            question=(f"Which restriction did you mean when you said "
                      f"'{event.value}' was no longer important?"),
            asks_about=event.attribute,
            ts=ts,
        )

    if event.type == "ambiguous_update":
        target = _find_pref_by_event(store, event.referent_event_id)
        if target is not None:
            store.set_status(target["pref_id"], "superseded")
        store.log_decision(turn_id=turn_id, tool="ignore_statement",
                           args_json=_json({"referent_event_id": event.referent_event_id}),
                           ok=True, result_json=_json({"removed": event.referent_event_id}),
                           ts=ts)
        return 0

    if event.type == "correction":
        target = _find_pref_by_event(store, event.target_event_id)
        if target is None:
            store.log_decision(turn_id=turn_id, tool="correct_preference",
                               args_json=_json({"target_event_id": event.target_event_id}),
                               ok=False, error="no remembered preference for that target",
                               ts=ts)
            return 0
        store.correct_pref(target["pref_id"], event.value)
        store.log_decision(turn_id=turn_id, tool="correct_preference",
                           args_json=_json({"target_event_id": event.target_event_id,
                                            "value": event.value}),
                           ok=True, result_json=_json({"pref_id": target["pref_id"]}),
                           ts=ts)
        return target["pref_id"]

    # initial_pref / permanent_revision / temporary_exception / other_person
    prior = _find_standing(store, event.subject, event.attribute)
    if event.type == "permanent_revision" and prior is not None:
        store.set_status(prior["pref_id"], "superseded")
        store.log_decision(turn_id=turn_id, tool="supersede_preference",
                           args_json=_json({"subject": event.subject,
                                            "attribute": event.attribute,
                                            "value": event.value}),
                           ok=True, result_json=_json({"superseded": prior["pref_id"]}),
                           ts=ts)
    pref_id = store.insert_pref(
        source_turn_id=turn_id,
        source_event_id=event.event_id,
        subject=event.subject,
        attribute=event.attribute,
        value=event.value,
        scope=event.scope,
        valid_from=_iso(event.valid_from),
        valid_until=_iso(event.valid_until) if event.valid_until else None,
        ts=ts,
    )
    store.log_decision(turn_id=turn_id, tool="add_preference",
                       args_json=_json({"subject": event.subject,
                                        "attribute": event.attribute,
                                        "value": event.value}),
                       ok=True, result_json=_json({"pref_id": pref_id}), ts=ts)
    return pref_id


def _find_pref_by_event(store: Store, event_id: Optional[str]):
    if event_id is None:
        return None
    row = store._conn.execute(
        "SELECT * FROM prefs WHERE source_event_id = ?", (event_id,)).fetchone()
    return row


def _find_standing(store: Store, subject: str, attribute: str):
    row = store._conn.execute(
        "SELECT * FROM prefs WHERE subject = ? AND attribute = ?"
        " AND status = 'active' ORDER BY pref_id DESC LIMIT 1",
        (subject, attribute)).fetchone()
    return row


def _dict_rows(cursor, row):
    return {d[0]: row[i] for i, d in enumerate(cursor.description)}




def _json(obj) -> str:
    return json.dumps(obj, sort_keys=True)
