"""Step 7's hand-written challenge set: the authoring file and its build.

The generated corpus is rendered from events by `templates.py`, which is what
makes it reproducible and what limits it: a phrasing no pool contains cannot
occur in it. This module carries the other half of the reality check -- 15-20
human-written histories, written as prose first, with the events declared by
hand and the answer key still computed by `semantics`, never typed in.

One authored history per line of `config.CHALLENGE_PATH`:

    {"trajectory_id": "c01", "family": "indirect_spice", "topic": "...",
     "author": "...", "reviewed_by": null, "review_notes": "",
     "turns":  [Turn fields, hand-written text, event_id, is_implicit],
     "events": [Event fields -- the label],
     "checkpoints": [{checkpoint_id, now, person, request_text, kind}]}

`gold_profile`, `gold_action`, `relevant_event_ids` and
`distractor_event_ids` are NOT authored: `load()` computes them with the same
frozen functions the generator uses. Authoring gold by hand would bypass the
one property every number in this repo rests on -- the answer key is derived
by deterministic Python.

`build()` expands the file into the two formats the runner and evaluator
already read (`histories.jsonl`, `checkpoints.jsonl`), so the challenge set is
one more corpus, not a second code path.

Usage::

    python -m memproj.challenge                        # validate, build, report
    python -m memproj.challenge --check                # also replay the corpus
    python -m memproj.challenge --strict               # fail while unreviewed
    python -m memproj.challenge --transcript c07       # text only, for labelling
    python -m memproj.challenge --diff review.jsonl    # cross-label disagreements
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from memproj import config, semantics
from memproj.schema import EVENT_TYPES, SCOPES, Event

# Fields a second labeller is expected to reproduce. Turn text and timestamps
# are the shared input, so they are not compared; the label is the event.
LABEL_FIELDS = ("type", "subject", "attribute", "value", "scope",
                "valid_from", "valid_until", "target_event_id",
                "needs_clarification", "referent_event_id")

# Checkpoint fields a labeller decides. `request_text` is the author's
# wording of the question, not a label: two labellers will never write the
# same sentence, and the run only ever reads one of them.
CHECKPOINT_FIELDS = ("now", "person", "kind")

STATEFUL_ATTRIBUTES = ("spice_level", "diet", "excluded_ingredient")


# --- reading and converting -------------------------------------------------

def read(path: Optional[str] = None) -> List[dict]:
    """The authoring file, one dict per history, in file order."""
    path = path or config.CHALLENGE_PATH
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _iso(value: datetime) -> str:
    return value.isoformat()


def _event(record: dict) -> Event:
    """One authored event dict as the frozen dataclass `semantics` reads."""
    return Event(
        event_id=record["event_id"],
        session=int(record["session"]),
        ts=_ts(record["ts"]),
        type=record["type"],
        subject=record["subject"],
        attribute=record.get("attribute"),
        value=record.get("value"),
        scope=record["scope"],
        valid_from=_ts(record["valid_from"]),
        valid_until=_ts(record["valid_until"]) if record.get("valid_until")
        else None,
        target_event_id=record.get("target_event_id"),
        needs_clarification=bool(record.get("needs_clarification", False)),
        referent_event_id=record.get("referent_event_id"),
    )


def _events(entry: dict) -> List[Event]:
    return [_event(record) for record in entry["events"]]


def _last_event_before(events: Sequence[Event], now: datetime) -> Optional[str]:
    """The event whose statement the checkpoint sits after."""
    seen = [e for e in events if e.ts <= now]
    if not seen:
        return None
    return max(seen, key=lambda e: (e.ts, e.event_id)).event_id


# --- loading into the corpus shapes ----------------------------------------

def load(path: Optional[str] = None) -> Tuple[List[dict], List[dict]]:
    """(trajectories, checkpoints) in the `out/*.jsonl` shapes.

    Raises ValueError listing every problem rather than building a corpus the
    runner would trip over one row at a time.
    """
    entries = read(path)
    problems = validate(entries)
    if problems:
        raise ValueError("the challenge file is not sound:\n  "
                         + "\n  ".join(problems))

    trajectories: List[dict] = []
    checkpoints: List[dict] = []
    for entry in entries:
        trajectory_id = entry["trajectory_id"]
        events = _events(entry)
        trajectories.append({
            "trajectory_id": trajectory_id,
            "pair_id": trajectory_id,
            "variant": "challenge",
            "family": entry["family"],
            "flipped_field": "none",
            "events": entry["events"],
            "turns": entry["turns"],
        })
        for spec in entry["checkpoints"]:
            now = _ts(spec["now"])
            person = spec["person"]
            profile = semantics.gold_profile(events, person, now)
            relevant, distractor = semantics.relevance(events, person, now)
            checkpoints.append({
                "checkpoint_id": spec["checkpoint_id"],
                "trajectory_id": trajectory_id,
                "after_event_id": _last_event_before(events, now),
                "now": _iso(now),
                "person": person,
                "request_text": spec["request_text"],
                "gold_profile": {attribute: sorted(values)
                                 for attribute, values in profile.items()},
                "gold_action": semantics.gold_action(events, person, now),
                "relevant_event_ids": relevant,
                "distractor_event_ids": distractor,
                "kind": spec["kind"],
            })
    trajectories.sort(key=lambda t: t["trajectory_id"])
    checkpoints.sort(key=lambda c: (c["trajectory_id"], c["now"],
                                    c["checkpoint_id"]))
    return trajectories, checkpoints


# --- validation -------------------------------------------------------------

def validate(entries: Sequence[dict]) -> List[str]:
    """Every reason this authoring file would not make a sound corpus."""
    problems: List[str] = []
    seen_trajectories: Dict[str, int] = {}
    seen_checkpoints: Dict[str, int] = {}

    for entry in entries:
        tid = entry.get("trajectory_id", "<missing>")
        if tid in seen_trajectories:
            problems.append(f"{tid}: duplicate trajectory id")
        seen_trajectories[tid] = seen_trajectories.get(tid, 0) + 1
        problems.extend(_validate_entry(entry, tid))
        for spec in entry.get("checkpoints", []):
            cid = spec.get("checkpoint_id", "<missing>")
            if cid in seen_checkpoints:
                problems.append(f"{tid}: duplicate checkpoint id {cid}")
            seen_checkpoints[cid] = 1
    return problems


def _validate_entry(entry: dict, tid: str) -> List[str]:
    problems: List[str] = []
    turns = entry.get("turns", [])
    events = entry.get("events", [])
    specs = entry.get("checkpoints", [])

    if not events:
        problems.append(f"{tid}: no events")
    if not specs:
        problems.append(f"{tid}: no checkpoints")

    event_ids = [record.get("event_id") for record in events]
    for event_id in sorted({e for e in event_ids if event_ids.count(e) > 1}):
        problems.append(f"{tid}: duplicate event id {event_id}")

    expressed = set()
    last_ts = None
    last_session = -1
    for turn in turns:
        turn_id = turn.get("turn_id", "<missing>")
        speaker = turn.get("speaker")
        if speaker not in ("user", "assistant"):
            problems.append(f"{tid} {turn_id}: speaker {speaker!r}")
        if speaker == "assistant":
            if turn.get("event_id") is not None:
                problems.append(f"{tid} {turn_id}: an assistant turn "
                                f"carries event {turn['event_id']}")
            if turn.get("is_implicit"):
                problems.append(f"{tid} {turn_id}: an assistant turn "
                                f"cannot be implicit")
        event_id = turn.get("event_id")
        if event_id is not None:
            if event_id not in event_ids:
                problems.append(f"{tid} {turn_id}: names event {event_id}, "
                                f"which no event declares")
            expressed.add(event_id)
        ts = _ts(turn["ts"])
        if last_ts is not None and ts < last_ts:
            problems.append(f"{tid} {turn_id}: at {turn['ts']}, before the "
                            f"previous turn")
        last_ts = ts
        session = int(turn.get("session", 0))
        if session < last_session:
            problems.append(f"{tid} {turn_id}: session {session} starts "
                            f"after session {last_session}")
        last_session = session

    for event_id in event_ids:
        if event_id not in expressed:
            problems.append(f"{tid} {event_id}: no user turn expresses it")

    by_id: Dict[str, dict] = {}
    for record in events:
        event_id = record.get("event_id")
        by_id[event_id] = record
        kind = record.get("type")
        if kind not in EVENT_TYPES:
            problems.append(f"{tid} {event_id}: unknown event type {kind!r}")
        if record.get("scope") not in SCOPES:
            problems.append(f"{tid} {event_id}: unknown scope "
                            f"{record.get('scope')!r}")
        if record.get("attribute") is not None and \
                record["attribute"] not in STATEFUL_ATTRIBUTES:
            problems.append(f"{tid} {event_id}: unknown attribute "
                            f"{record['attribute']!r}")
        valid_from = _ts(record["valid_from"])
        valid_until = record.get("valid_until")
        if valid_until is not None:
            if _ts(valid_until) <= valid_from:
                problems.append(f"{tid} {event_id}: valid_until "
                                f"{valid_until} is not after valid_from ")
        if kind == "temporary_exception" and valid_until is None:
            problems.append(f"{tid} {event_id}: a temporary exception "
                            f"needs valid_until")
        target = record.get("target_event_id")
        if kind in ("permanent_revision", "correction"):
            if target is None:
                problems.append(f"{tid} {event_id}: a {kind} needs a target")
            elif target not in by_id:
                problems.append(f"{tid} {event_id}: targets {target}, "
                                f"which no event declares")
            elif _ts(by_id[target]["ts"]) > _ts(record["ts"]):
                problems.append(f"{tid} {event_id}: targets {target}, "
                                f"which is spoken later")
        referent = record.get("referent_event_id")
        if referent is not None and referent not in by_id:
            problems.append(f"{tid} {event_id}: resolves {referent}, "
                            f"which no event declares")

    for spec in specs:
        cid = spec.get("checkpoint_id", "<missing>")
        try:
            _ts(spec["now"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"{tid} {cid}: no usable `now`")
        if not spec.get("person"):
            problems.append(f"{tid} {cid}: no person")
        if not spec.get("kind"):
            problems.append(f"{tid} {cid}: no kind")
        if not spec.get("request_text"):
            problems.append(f"{tid} {cid}: no request_text")
    return problems


# --- cross-labelling --------------------------------------------------------

def diff_labels(author: dict, reviewer: dict) -> List[str]:
    """Every label the second labeller did not reproduce, one line each.

    Events are matched by `event_id` -- not position -- so an author who
    reordered two statements is not reported as a labelling disagreement.
    """
    tid = author.get("trajectory_id", "<missing>")
    problems: List[str] = []

    theirs = {e["event_id"]: e for e in reviewer.get("events", [])}
    mine = {e["event_id"]: e for e in author.get("events", [])}
    for event_id in sorted(set(mine) | set(theirs)):
        if event_id not in theirs:
            problems.append(f"{tid} {event_id}: absent from the reviewer's "
                            f"labels")
            continue
        if event_id not in mine:
            problems.append(f"{tid} {event_id}: absent from the author's "
                            f"labels")
            continue
        for field in LABEL_FIELDS:
            mine_value = mine[event_id].get(field)
            their_value = theirs[event_id].get(field)
            if mine_value != their_value:
                problems.append(
                    f"{tid} {event_id} {field}: author {mine_value!r} != "
                    f"reviewer {their_value!r}")

    their_cps = {c["checkpoint_id"]: c for c in reviewer.get("checkpoints", [])}
    my_cps = {c["checkpoint_id"]: c for c in author.get("checkpoints", [])}
    for checkpoint_id in sorted(set(my_cps) | set(their_cps)):
        if checkpoint_id not in their_cps:
            problems.append(f"{tid} {checkpoint_id}: absent from the "
                            f"reviewer's labels")
            continue
        if checkpoint_id not in my_cps:
            problems.append(f"{tid} {checkpoint_id}: absent from the "
                            f"author's labels")
            continue
        for field in CHECKPOINT_FIELDS:
            mine_value = my_cps[checkpoint_id].get(field)
            their_value = their_cps[checkpoint_id].get(field)
            if mine_value != their_value:
                problems.append(
                    f"{tid} {checkpoint_id} {field}: author {mine_value!r} != "
                    f"reviewer {their_value!r}")
    return problems


def unreviewed(entries: Sequence[dict]) -> List[str]:
    """Histories no second labeller has signed off yet, in file order."""
    return [entry["trajectory_id"] for entry in entries
            if not entry.get("reviewed_by")]


def transcript(entry: dict) -> str:
    """The dialogue alone -- what a second labeller is given to label from.

    No event ids, no attributes, no `is_implicit`: the reviewer must read the
    prose the way any reader would, or the cross-check only measures their
    agreement with the author's notes.
    """
    lines = [f"# {entry['trajectory_id']}  ({entry.get('topic', '')})"]
    for turn in entry["turns"]:
        lines.append(f"{turn['ts']}  {turn['speaker']:9s}  {turn['text']}")
    return "\n".join(lines)


# --- building ---------------------------------------------------------------

def build(source: Optional[str] = None,
          out_dir: Optional[str] = None) -> Tuple[str, str]:
    """Write the two corpus files; returns (histories, checkpoints) paths."""
    trajectories, checkpoints = load(source)
    out_dir = out_dir or os.path.join(config.OUT_DIR, "challenge")
    os.makedirs(out_dir, exist_ok=True)
    histories_path = os.path.join(out_dir, "histories.jsonl")
    checkpoints_path = os.path.join(out_dir, "checkpoints.jsonl")
    for path, rows in ((histories_path, trajectories),
                       (checkpoints_path, checkpoints)):
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True,
                                    ensure_ascii=False) + "\n")
    return histories_path, checkpoints_path


# --- CLI --------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=None,
                        help="authoring file; default: config.CHALLENGE_PATH")
    parser.add_argument("--out", default=None,
                        help="directory for the built corpus; default: "
                             "out/challenge")
    parser.add_argument("--check", action="store_true",
                        help="also replay the built corpus (gold gate)")
    parser.add_argument("--strict", action="store_true",
                        help="fail while any history is not cross-labelled")
    parser.add_argument("--transcript", metavar="ID", default=None,
                        help="print one history's dialogue, labels hidden")
    parser.add_argument("--diff", metavar="PATH", default=None,
                        help="compare another labeller's file with this one")
    args = parser.parse_args(argv)

    entries = read(args.source)

    if args.transcript:
        match = [e for e in entries if e["trajectory_id"] == args.transcript]
        if not match:
            print(f"no history {args.transcript}", file=sys.stderr)
            return 2
        print(transcript(match[0]))
        return 0

    if args.diff:
        reviewer = read(args.diff)
        disagreements: List[str] = []
        for mine in entries:
            theirs = [r for r in reviewer
                      if r["trajectory_id"] == mine["trajectory_id"]]
            if not theirs:
                disagreements.append(f"{mine['trajectory_id']}: no labels "
                                     f"in {args.diff}")
                continue
            disagreements.extend(diff_labels(mine, theirs[0]))
        for line in disagreements:
            print(line)
        print(f"{len(disagreements)} disagreement(s)")
        return 0 if not disagreements else 1

    problems = validate(entries)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        print(f"{len(problems)} problem(s) in {args.source or config.CHALLENGE_PATH}",
              file=sys.stderr)
        return 1

    waiting = unreviewed(entries)
    if args.strict and waiting:
        print(f"{len(waiting)} histor{'y' if len(waiting) == 1 else 'ies'} "
              f"await cross-labelling: {', '.join(waiting)}", file=sys.stderr)
        return 1

    histories_path, checkpoints_path = build(args.source, args.out)
    trajectories, checkpoints = load(args.source)
    print(f"{len(trajectories)} histories, {len(checkpoints)} checkpoints")
    print(f"awaiting cross-labelling: {len(waiting)} of {len(trajectories)}")
    print(f"wrote {histories_path}")
    print(f"wrote {checkpoints_path}")

    if args.check:
        from memproj import replay
        report = replay.run(trajectories, checkpoints)
        print(f"replay: {report['checked']} checkpoints, "
              f"{len(report['mismatches'])} mismatches")
        if report["mismatches"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
