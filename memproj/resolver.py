"""The resolver: store rows in, the active profile out. Pure Python.

This module is the memory half's half of the benchmark's central claim. The
LLM only ever chose WHICH tool to call; every rule that decides what is
*currently true* is applied here, deterministically, and should reproduce
`semantics.gold_profile` exactly when the tool calls were correct.

Four ablation flags turn individual rules off. They exist so the ablations can ask
which rule was doing the work, and -- just as importantly -- as a sanity
check: a flag that does not change any profile is a rule that is not
implemented, and a flag that changes the wrong families is a rule leaking.

A row is not a fact; it is a claim with a lifetime. Each row carries:
  valid_from / valid_until  when the user meant it (valid_until exclusive)
  source_turn_id            the statement behind it
  corrected_from / _at      the user said they misheard -- or re-said -- it
  supersedes / closed_by    a later statement replaced it, or named it as gone
  dropped_at                a named removal happened -- the row is true before it

Every flag removes one of those pieces, and the rule that reads it:
  use_subject    no subject filter: the brother's rows land in the user's profile
  use_scope      a temporary window never opens (temporary == permanent)
  use_time       once a value exists, its window never closes
  use_supersede  a replacing statement does not take anything out
  use_correction a restatement leaves the value it corrected standing
"""

from datetime import datetime
from typing import Dict, List, Optional

from memproj.store import Store


def active_profile(store: Store, subject: str, now,
                   use_scope: bool = True, use_time: bool = True,
                   use_subject: bool = True, use_supersede: bool = True,
                   use_correction: bool = True
                   ) -> Dict[str, set]:
    """The preferences `subject` is entitled to at time `now`.

    Returns {attribute: set(values)}, the same shape as gold_profile.
    """
    profile: Dict[str, set] = {}
    for record in active_records(store, subject, now, use_scope, use_time,
                                 use_subject, use_supersede, use_correction):
        profile.setdefault(record["attribute"], set()).add(record["value"])
    return profile


def active_records(store: Store, subject: str, now,
                   use_scope: bool = True, use_time: bool = True,
                   use_subject: bool = True, use_supersede: bool = True,
                   use_correction: bool = True
                   ) -> List[Dict]:
    """The rows behind `active_profile`, one per fact they contribute.

    A row is not a fact -- a corrected row contributes two values over time --
    so each record names the fact (`attribute`, `value`), the row it came from,
    and the moment that value started. That is what a method needs to cite its
    sources without re-deciding which rule let the row through: `proposed`'s
    evidence turn ids are `record["source_turn_id"]`.
    """
    if isinstance(now, str):
        now = datetime.fromisoformat(now)
    now_s = now.isoformat()

    rows = store.all_prefs()
    # The row that CLOSED each row, keyed by the closed row's pref_id. The
    # first draft keyed this by the closer's own closed_by, which looks up
    # the child of a row rather than its parent and hands `_applies` the
    # wrong record (usually None).
    by_id = {r["pref_id"]: r for r in rows}
    records: List[Dict] = []
    for row in rows:
        if not _applies(row, subject, now_s, use_subject, use_supersede,
                        by_id.get(row["closed_by"])):
            continue
        for start, end, value in _segments(row, use_correction):
            if _open(row, start, end, now_s, use_scope, use_time):
                records.append({"attribute": row["attribute"], "value": value,
                                "start": start, "pref_id": row["pref_id"],
                                "source_turn_id": row["source_turn_id"],
                                "row": row})
    return records


def _applies(row, subject, now_s: str, use_subject: bool,
             use_supersede: bool, closer=None) -> bool:
    """Is this row the user's as of `now_s`, and does a replacement take it
    out of play?

    Superseded normally means gone: the latest statement wins -- but only
    from the moment the replacement was spoken. A revision names that moment
    in its own valid_from, so before it the older row is still what the user
    said. Replay never meets the gap, because it reprocesses the store as of
    `now`; a store built from the WHOLE trajectory does, and dropping the row
    early would leak a later session into an earlier checkpoint.

    The exception is a one-off that has expired out of its own window: what
    ended it is the clock, so a revision that agrees with it does not erase
    it. A removal names the record and does erase it, windowed or not -- but
    only from the turn that did it: "no longer important", said at noon,
    leaves the record true before noon. A correction rewrites its row and
    never closes it.
    """
    if use_subject and row["subject"] != subject:
        return False
    if row["dropped_at"] is not None and now_s >= row["dropped_at"]:
        return False
    if (use_supersede and row["status"] == "superseded"
            and row["dropped_at"] is None):
        # A DROPPED row also carries `superseded`: its own moment above has
        # said when it stops counting, and its status must not end it early.
        if (row["valid_until"] is not None
                and closer is not None
                and closer["supersedes"] == row["pref_id"]
                and closer["valid_from"] >= row["valid_until"]):
            return True
        # A row closed off the gold path carries no closer (record_event's
        # ambiguous removal): nothing dates it, so it is gone. Otherwise the
        # revision has to have been spoken by `now`.
        return closer is not None and closer["valid_from"] > now_s
    return True


def _segments(row, use_correction: bool):
    """(start, end, value) pieces of this row, oldest first.

    An edit splits the row whatever the flag, because the answer key splits
    it: what the user had is what stood until they restated it. The
    corrected value keeps the row's window -- it is the row's own
    `valid_from`/`valid_until` that decide when the value counts -- but it
    is not known before the user says it, so it starts at `corrected_at`.
    A whole-trajectory store holds the correction while resolving a
    checkpoint that precedes it, and replay cannot see that: it rebuilds the
    store per checkpoint, so the correction is simply absent when it should
    not count. (Same blind spot as a revision moving backwards.)

    With `use_correction` off -- the ablation -- the restatement closes
    nothing: the old value keeps standing and the new one is added beside it
    from the turn it was spoken. That is what a method with no correction
    rule produces, and it is the contaminant this flag names.

    A restatement that confirmed the stored value has no `corrected_from`
    and adds nothing.
    """
    if not row["edited"] or row["corrected_from"] is None:
        return [(row["valid_from"], None, row["value"])]
    if use_correction:
        return [(row["valid_from"], row["corrected_at"], row["corrected_from"]),
                (row["corrected_at"], None, row["value"])]
    return [(row["valid_from"], None, row["corrected_from"]),
            (row["corrected_at"], None, row["value"])]


def _open(row, start: str, end, now_s: str, use_scope: bool,
          use_time: bool) -> bool:
    """Is a piece of this row, running from `start` to `end`, in force at
    `now`?

    `end` is where an edit closed the piece (exclusive), or None. It is the
    row's own history, not a window, so no flag touches it: the scope and
    time flags must move truth *outward*, and a rule that kept a corrected
    value alive past its restatement would do the same in both ablations.

    Scope says WHICH values carry a window; time says whether it is enforced.
    Without use_scope a temporary value never opens (temporary == permanent);
    without use_time an open value never closes. Both only move truth
    *outward*, so a flag can add contaminants but never remove one.
    """
    if end is not None and now_s >= end:
        return False
    if use_scope and row["scope"] == "temporary":
        if now_s < start:
            return False
        return not (use_time and row["valid_until"] is not None
                    and now_s >= row["valid_until"])
    if use_time:
        return now_s >= start
    return now_s >= row["valid_from"]


def history(store: Store, subject: str, now=None) -> List[dict]:
    """Every row that ever applied to `subject`, including superseded and
    expired ones. Corrections show the current value and what it corrected."""
    if isinstance(now, str):
        now = datetime.fromisoformat(now)
    rows = [r for r in store.all_prefs() if r["subject"] == subject]
    if now is not None:
        rows = [r for r in rows if r["valid_from"] <= now.isoformat()]
    return rows


def pending_clarification(store: Store, now=None) -> Optional[dict]:
    """The unresolved question, if the agent is waiting on one.

    With `now`, "unresolved" is judged as of that instant rather than by the
    row's final status: a question asked before a checkpoint and only answered
    after it was still pending when the checkpoint was asked. That matters to
    the proposed method, whose whole store is built in one pass over the
    trajectory, including turns a given checkpoint must not see.
    """
    if now is None:
        rows = store.pending_clarifications()
        return rows[0] if rows else None
    now_s = now.isoformat() if not isinstance(now, str) else now
    rows = [row for row in store.clarifications_with_resolution()
            if row["ts"] <= now_s
            and (row["resolved_ts"] is None or row["resolved_ts"] > now_s)]
    return rows[0] if rows else None
