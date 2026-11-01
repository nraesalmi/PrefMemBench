"""Replay one trajectory through the store and compare against the answer key.

This is the tool-layer gate, executable: for every checkpoint of every
trajectory, the tool layer must reconstruct gold_profile exactly. Gold replay
is the stand-in for an extraction model -- `events_to_tool_calls` and
`turns_to_tool_calls` produce the (tool, args) pairs a model would produce, so
swapping the gold mapping for a model changes the source of the calls and
nothing else.

    python -m memproj.replay                  # the whole corpus
    python -m memproj.replay --limit 5 -v     # first five pairs, show mismatch

Exit code 0 means zero state mismatches across every checkpoint.

Each checkpoint is asked of the store a live run would hold at that instant:
turns up to `now`, nothing later. A checkpoint whose `now` falls between
turns is asked of the store as it stands after a whole session; one that
falls inside a session (correction_of_temporary asks an hour into the
session that holds the correction) is asked of the prefix that has happened.
"""

import argparse
import json
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

from memproj import config
from memproj.resolver import active_profile
from memproj.schema import Event, Turn
from memproj.semantics import RESTRICTED_ATTRIBUTE, STATEFUL_TYPES
from memproj.store import Store
from memproj.tools import dispatch


# --- the gold mapping: event -> the tool call a perfect extractor would make --

def events_to_tool_calls(events: List[Event]) -> List[Tuple[str, dict, Event]]:
    """(tool, args, event) for every gold event, in event order.

    One call per event. A vague statement is one call to clarify_statement:
    the question and what the answer will resolve, in one step -- the
    answer arrives before the next request, so nothing is recorded about it
    at the vague statement itself. An earlier defect split this into two
    calls, the second stamped with the answer's event; that closed a
    clarification only when the answer happened to come along later, and
    let the question be asked twice. A statement that records no value
    yields no call at all: gold keeps only records with a value, so the
    correct handling of "that exception is over" is to record nothing.

    Provenance fields that only the harness knows (ts, turn_id, source_event,
    the resolved valid_until) are not in the args: replay feeds them in
    out-of-band, exactly as the extraction loop does. A model cannot supply them and a
    perfect extractor does not need to.
    """
    calls = []
    for e in events:
        if e.value is None and e.type in STATEFUL_TYPES                 and e.type != "ambiguous_update":
            continue  # nothing to record: this statement carries no value
        if e.type in ("filler", "hypothetical"):
            calls.append(("ignore_statement", {"reason": "hypothetical"
                                               if e.type == "hypothetical"
                                               else "chatter"}, e))
        elif e.type == "ambiguous_update":
            # The agent asks which restriction this is about. A resolved
            # statement's own referent is its answer: the user said it, and
            # the agent heard the reply before it asked the next question --
            # gold reads an answer as the turn the question was asked in.
            # A vague statement is answered by the next vague statement, if
            # one comes; when that one is vague too, it asks again (gold
            # keeps re-raising). Nothing dropped here means "asked it".
            resolved = e.referent_event_id is not None
            calls.append(("clarify_statement",
                          {"question": "Which restriction is no longer "
                                       "important?",
                           "asks_about": RESTRICTED_ATTRIBUTE,
                           "resolves": ({"source_event_id": e.referent_event_id}
                                        if resolved else {})}, e))
        elif e.type == "correction":
            calls.append(("_correct_target",
                          {"target_event_id": e.target_event_id,
                           "value": e.value}, e))
        elif e.type == "permanent_revision":
            # A revision whose target had a window says "not that one":
            # nothing is standing to replace, so the tool that expresses it
            # is clarify_statement taken as already answered, with `resolves`
            # naming the record it removes.
            target = next((x for x in events
                           if x.event_id == e.target_event_id), None)
            if target is not None and target.valid_until is not None:
                calls.append(("clarify_statement",
                              {"question": "Should I stop applying that "
                                           "exception?",
                               "resolves": {"source_event_id":
                                            e.target_event_id}}, e))
            else:
                calls.append(("_supersede",
                              {"subject": e.subject, "attribute": e.attribute,
                               "value": e.value}, e))
        else:  # initial_pref, temporary_exception, other_person
            args = {"subject": e.subject, "attribute": e.attribute,
                    "value": e.value, "scope": e.scope}
            if e.valid_until is not None:
                args["valid_until"] = e.valid_until.isoformat()
            calls.append(("_add", args, e))
    return calls


def _calls_expected(events: List[Event]) -> Dict[str, int]:
    """How many tool calls each event implies, derived from the event alone.

    Kept separate from events_to_tool_calls so the plan and the check cannot
    drift into agreeing on the same mistake.
    """
    wants = {}
    for e in events:
        if e.value is None and e.type in STATEFUL_TYPES                 and e.type != "ambiguous_update":
            wants[e.event_id] = 0
        else:
            wants[e.event_id] = 1
    return wants


def _apply_gold_call(store: Store, name: str, args: dict, event: Event,
                     turn_id: str) -> Tuple[bool, dict]:
    """Translate the underscore-prefixed gold calls into tool calls.

    Only `_correct_target` is not a tool: it needs the pref_id of an earlier
    record. Gold replay looks it up from the events; the extraction model names the
    pref_id it saw in a tool result. Same tool call, different source of one
    integer.

    Calls that create a row also carry `source_event_id`, so a later
    correction or resolved ambiguity can find the row the user is naming.
    It is provenance the harness knows and the model does not supply -- from
    a model, "the peanut one" would arrive as a pref_id it read in history.
    """
    ts = event.ts
    if name in ("ignore_statement", "clarify_statement"):
        return dispatch(store, name, args, ts, turn_id)
    args = {**args, "source_event_id": event.event_id}
    if name == "_add":
        return dispatch(store, "add_preference", args, ts, turn_id)
    if name == "_supersede":
        return dispatch(store, "supersede_preference", args, ts, turn_id)
    if name == "_correct_target":
        row = store._conn.execute(
            "SELECT pref_id FROM prefs WHERE source_event_id = ?",
            (args["target_event_id"],)).fetchone()
        if row is None:
            return False, {"error": f"no pref for {args['target_event_id']}"}
        return dispatch(store, "correct_preference",
                        {"target_pref_id": row["pref_id"],
                         "value": args["value"]}, ts, turn_id)
    raise AssertionError(f"unexpected gold call {name}")


# --- replaying one trajectory -----------------------------------------------

def _parse_event(d: dict) -> Event:
    def dt(key):
        v = d[key]
        return None if v is None else __import__("datetime").datetime.fromisoformat(v)
    return Event(
        event_id=d["event_id"], session=d["session"], ts=dt("ts"),
        type=d["type"], subject=d["subject"], attribute=d["attribute"],
        value=d["value"], scope=d["scope"], valid_from=dt("valid_from"),
        valid_until=dt("valid_until"), target_event_id=d["target_event_id"],
        needs_clarification=d["needs_clarification"],
        referent_event_id=d["referent_event_id"],
    )


def _parse_turn(d: dict) -> Turn:
    from datetime import datetime
    return Turn(turn_id=d["turn_id"], session=d["session"],
                ts=datetime.fromisoformat(d["ts"]), speaker=d["speaker"],
                text=d["text"], event_id=d["event_id"],
                is_implicit=d.get("is_implicit", False))


def event_turn_map(turns: List[Turn]) -> Dict[str, str]:
    """event_id -> the user turn that expressed it."""
    return {t.event_id: t.turn_id for t in turns
            if t.event_id is not None and t.speaker == "user"}


def replay_trajectory(trajectory: dict, path: str = ":memory:",
                      before_turn: int = None) -> Store:
    """Build the store the agent would have built, by replaying the events.

    The `now` axis is handled by the resolver reading rows, not by the
    replay: events are recorded in order, and each checkpoint asks the
    resolver what was true at its own time.

    `before_turn` is an index into `turns`: replay only the events expressed
    by turns before it. That is what a live run holds when the request at a
    mid-session checkpoint arrives -- the session's later turns have not
    happened yet -- and it is the only way a checkpoint sitting inside a
    session can be asked of a store without seeing the future.
    """
    events = [_parse_event(e) for e in trajectory["events"]]
    turns = [_parse_turn(t) for t in trajectory["turns"]]
    if before_turn is not None:
        live = {t.event_id for t in turns[:before_turn]
                if t.event_id is not None}
        events = [e for e in events if e.event_id in live]
    turn_of = event_turn_map(turns)

    store = Store(path).create()
    for t in turns:
        store.add_turn(t)

    planned = events_to_tool_calls(events)
    # Every event gets exactly the calls its type implies: inert statements
    # one (a real extractor must decide to ignore them), a valueless
    # statement none. Anything else means the mapping and the event
    # disagreed.
    calls_for = Counter(e.event_id for _, _, e in planned)
    wants = _calls_expected(events)
    for e in events:
        assert calls_for[e.event_id] == wants[e.event_id], (
            f"{trajectory['trajectory_id']}/{e.event_id}: "
            f"{calls_for[e.event_id]} calls for one {e.type}, "
            f"{wants[e.event_id]} expected")
    for name, args, e in planned:
        turn_id = turn_of.get(e.event_id)
        if turn_id is None:
            raise AssertionError(
                f"{trajectory['trajectory_id']}: event {e.event_id} has no turn")
        ok, result = _apply_gold_call(store, name, args, e, turn_id)
        if not ok:
            raise AssertionError(
                f"{trajectory['trajectory_id']}/{e.event_id} ({e.type}): "
                f"gold replay rejected: {result}")
    return store


# --- checking every checkpoint ----------------------------------------------

def _comparable(profile: Dict[str, set]) -> Dict[str, List[str]]:
    return {k: sorted(v) for k, v in sorted(profile.items())}


def run(trajectories: List[dict], checkpoints: List[dict],
        flags: dict = None, verbose: int = 0) -> dict:
    flags = flags or {}
    cps_by_traj: Dict[str, List[dict]] = {}
    for c in checkpoints:
        cps_by_traj.setdefault(c["trajectory_id"], []).append(c)

    mismatches, checked, per_family = [], 0, Counter()
    deferred = Counter()
    for tr in trajectories:
        fam = tr["family"]
        per_family[fam] += 0
        since_previous = ""
        # A checkpoint sits at an instant. The store it is asked about holds
        # the turns up to that instant -- nothing later, or the question is
        # answered with the future. Between checkpoints that is the same
        # store; inside a session it is a prefix of it.
        prefix = None
        store = None
        try:
            for cp in cps_by_traj.get(tr["trajectory_id"], []):
                cut = _turns_before(tr, cp["now"])
                if cut != prefix:
                    if store is not None:
                        store.close()
                    store = replay_trajectory(tr, before_turn=cut)
                    prefix = cut
                checked += 1
                per_family[fam] += 1
                got = _comparable(active_profile(
                    store, cp["person"], cp["now"], **flags))
                if got != cp["gold_profile"]:
                    mismatches.append({
                        "checkpoint_id": cp["checkpoint_id"],
                        "family": fam, "variant": tr["variant"],
                        "kind": cp["kind"], "gold": cp["gold_profile"],
                        "got": got,
                    })
                # Action, not state: an unresolved ambiguity must still be
                # pending at a checkpoint whose gold action is 'clarify'.
                # Only questions asked since the previous checkpoint count: a
                # question answered later in the session is not still hanging
                # over the user at every checkpoint between.
                asked = _asked_since(store, since_previous, cp["now"])
                if asked != (cp["gold_action"] == "clarify"):
                    mismatches.append({
                        "checkpoint_id": cp["checkpoint_id"],
                        "family": fam, "variant": tr["variant"],
                        "kind": cp["kind"] + ":action",
                        "gold": cp["gold_action"],
                        "got": "clarify" if asked else "recommend",
                    })
                since_previous = cp["now"]
        finally:
            if store is not None:
                store.close()
        if verbose and mismatches and mismatches[-1]["family"] == fam:
            for m in mismatches[-2:]:
                print(json.dumps(m, sort_keys=True))
    return {"checked": checked, "mismatches": mismatches,
            "trajectories": len(trajectories), "per_family": dict(per_family)}


def _asked_since(store: Store, since: str, now: str) -> bool:
    """Was a question asked in (since, now] and left open?

    Gold re-raises: a vague answer leaves its question open AND asks about
    the answer, so the later checkpoint that gold marks 'clarify' is asking
    about the reply. Both rows are pending in the store, the first one asked
    before `since`; the question asked inside the window is the one that
    matters. Any pending row created in the window counts.
    """
    return any(since < r["ts"] <= now for r in store.pending_clarifications())


def _turns_before(trajectory: dict, now: str) -> int:
    """How many of the trajectory's turns happen at or before `now`."""
    return sum(1 for t in trajectory["turns"] if t["ts"] <= now)


def load_corpus(histories_path=None, checkpoints_path=None):
    h = histories_path or config.HISTORIES_PATH
    c = checkpoints_path or config.CHECKPOINTS_PATH
    with open(h, encoding="utf-8") as fh:
        trajectories = [json.loads(line) for line in fh if line.strip()]
    with open(c, encoding="utf-8") as fh:
        checkpoints = [json.loads(line) for line in fh if line.strip()]
    return trajectories, checkpoints


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--histories", default=None)
    p.add_argument("--checkpoints", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="only the first N pairs (2N trajectories)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    trajectories, checkpoints = load_corpus(args.histories, args.checkpoints)
    if args.limit:
        keep = {f"p{i:03d}" for i in range(args.limit)}
        trajectories = [t for t in trajectories if t["pair_id"] in keep]
        ids = {t["trajectory_id"] for t in trajectories}
        checkpoints = [c for c in checkpoints if c["trajectory_id"] in ids]

    report = run(trajectories, checkpoints, verbose=int(args.verbose))
    print(f"trajectories: {report['trajectories']}  "
          f"checkpoints: {report['checked']}  "
          f"mismatches: {len(report['mismatches'])}")
    for m in report["mismatches"][:20]:
        print("  " + json.dumps(m, sort_keys=True))
    if len(report["mismatches"]) > 20:
        print(f"  ... {len(report['mismatches']) - 20} more")
    return 0 if not report["mismatches"] else 1


if __name__ == "__main__":
    sys.exit(main())
