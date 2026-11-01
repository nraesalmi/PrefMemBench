"""The memory tools, their Ollama schemas, and dispatch.

The LLM decides WHICH tool to call and fills the fields it can read off the
utterance. It never supplies `ts` or `turn_id`: the harness passes those to
dispatch(), which stamps every row with them. A model that could name its own
timestamps could make a temporary window permanent or forge provenance by
invention, and both are exactly what this benchmark measures.

Every call returns (ok, result) rather than raising. A model that hallucinates
a subject or a date should get an error it can read and retry against, not a
traceback that ends the trajectory.
"""

import json
from datetime import datetime
from typing import Tuple

from memproj.recipes import _normalise
from memproj.store import Store

# The tool names, in the order the prompt lists them.
TOOL_NAMES = (
    "add_preference",
    "supersede_preference",
    "clarify_statement",
    "correct_preference",
    "ignore_statement",
    "ask_clarification",
)

# Subjects the generator can produce. "user" plus the other-person vocabulary;
# a subject outside this set is a typo, not a new person.
OTHER_PEOPLE = ("brother", "partner", "friend")

# Attributes and the values each admits. Validating values here means an
# extraction error shows up as ok=False at extraction time, where extraction can
# count it, instead of as a bogus profile at checkpoint time.
ATTRIBUTES = {
    "spice_level": {"mild", "medium", "spicy"},
    "excluded_ingredient": None,   # free text, normalised by recipes._normalise
    "diet": {"vegetarian"},
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "add_preference",
            "description": (
                "Record a preference the user is stating for the first time, "
                "or a temporary exception that applies only inside a window. "
                "Use scope='permanent' for an ongoing preference and "
                "scope='temporary' with valid_until for a one-off. Never use "
                "this to change a preference the user already holds."),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string",
                                "description": "Who the preference is about: "
                                               "'user', 'brother', 'partner', "
                                               "or 'friend'."},
                    "attribute": {"type": "string",
                                  "enum": sorted(ATTRIBUTES)},
                    "value": {"type": "string"},
                    "scope": {"type": "string",
                              "enum": ["permanent", "temporary"]},
                    "valid_until": {
                        "type": "string",
                        "description": "ISO 8601 time the exception stops "
                                       "applying, exclusive. Required if and "
                                       "only if scope is 'temporary'. The "
                                       "harness converts relative phrases such "
                                       "as 'tonight'; supply what it tells you."},
                },
                "required": ["subject", "attribute", "value", "scope"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "supersede_preference",
            "description": (
                "Replace a preference the user previously held. Use for a "
                "permanent revision ('I now prefer mild food'). The old row is "
                "kept for history and is no longer applied."),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "attribute": {"type": "string", "enum": sorted(ATTRIBUTES)},
                    "value": {"type": "string"},
                },
                "required": ["subject", "attribute", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clarify_statement",
            "description": (
                "Ask the user WHICH earlier preference a vague statement is "
                "about ('that restriction is no longer important'), then name "
                "the record their answer points at in `resolves`. If the "
                "answer names a record that is standing, it is dropped now; if "
                "it names nothing or is vague again, the question stays open. "
                "Never guess the record."),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "asks_about": {
                        "type": "string", "enum": sorted(ATTRIBUTES),
                        "description": "Optional hint: the attribute you "
                                       "think is meant."},
                    "resolves": {
                        "type": "object",
                        "description": (
                            "What the user's answer names, once they have "
                            "answered: the pref_id, the source_event_id of "
                            "the statement that recorded it, or its value. "
                            "Leave out keys you do not know."),
                        "properties": {
                            "target_pref_id": {"type": "integer"},
                            "source_event_id": {"type": "string"},
                            "value": {"type": "string"},
                        },
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "correct_preference",
            "description": (
                "Fix a value you recorded wrongly, because the user says you "
                "misunderstood ('I said NO mushrooms'). This rewrites the "
                "original record; it does not create a later preference and "
                "must not move the time it started applying."),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_pref_id": {"type": "integer",
                                       "description": "The pref_id you "
                                                      "recorded from the "
                                                      "misunderstood statement."},
                    "value": {"type": "string"},
                },
                "required": ["target_pref_id", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ignore_statement",
            "description": (
                "Record that an utterance carries no preference: small talk, a "
                "hypothetical ('if I liked spicy food...'), or a request about "
                "someone else's meal that changes nothing about the user. "
                "Recording the decision keeps the run auditable."),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string",
                               "enum": ["chatter", "hypothetical",
                                        "no_new_information"]},
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_clarification",
            "description": (
                "Ask the user a question when the request itself is "
                "under-specified, and record it as pending. Do not guess."),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "asks_about": {
                        "type": "string", "enum": sorted(ATTRIBUTES),
                        "description": "Optional hint: the attribute you "
                                       "think is meant."},
                },
                "required": ["question"],
            },
        },
    },
]


def parse_ts(value: str) -> datetime:
    """Absolute ISO 8601 in, datetime out. Relative phrases are rejected.

    Date arithmetic ('tonight' -> 23:59) is the harness's job. Accepting a
    phrase here would let it happen inside the model's field.
    """
    if not isinstance(value, str):
        raise ValueError(f"valid_until must be a string, got {type(value).__name__}")
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(
            f"valid_until {value!r} is not an ISO 8601 datetime; the harness "
            f"resolves relative phrases such as 'tonight' before the call"
        ) from None


def _fail(message: str) -> Tuple[bool, dict]:
    return False, {"error": message}


def _check_subject(args) -> str:
    subject = args.get("subject")
    if subject not in ("user",) + OTHER_PEOPLE:
        raise ValueError(f"unknown subject {subject!r}; expected 'user', "
                         f"'brother', 'partner' or 'friend'")
    return subject


def _check_attribute(args) -> str:
    attribute = args.get("attribute")
    if attribute not in ATTRIBUTES:
        raise ValueError(f"unknown attribute {attribute!r}; expected one of "
                         f"{sorted(ATTRIBUTES)}")
    return attribute


def _check_value(attribute: str, value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("value must be a non-empty string")
    value = value.strip()
    allowed = ATTRIBUTES[attribute]
    if allowed is not None and value.lower() not in allowed:
        raise ValueError(f"{attribute} must be one of {sorted(allowed)}, "
                         f"got {value!r}")
    if allowed is not None:
        # The vocabulary is closed, so lowercasing is the whole rule.
        return value.lower()
    # Free text is stored in the catalogue's spelling: "Peanuts" and
    # "peanut" must be one fact, or a correctly recorded statement scores
    # as both a miss and an extra. recipes._normalise is the same
    # singularisation the recommender matches ingredients with.
    return _normalise(value)


def _find_active(store: Store, subject: str, attribute: str, now):
    """The row this attribute's value comes from at `now`.

    `status = 'active'` is not the same claim as "in force": a temporary
    exception keeps its status after its window closes. A revision replaces
    the value the user would be revising *right now*, so the window decides,
    with the resolver's own half-open comparison (`valid_from <= now <
    valid_until`). Choosing the newest row instead let a lapsed exception
    stand in for the preference it had been an exception to.
    """
    now_s = now if isinstance(now, str) else now.isoformat()
    return store._conn.execute(
        "SELECT * FROM prefs WHERE subject = ? AND attribute = ?"
        " AND status IN ('active', 'corrected')"
        " AND valid_from <= ? AND (valid_until IS NULL OR valid_until > ?)"
        " ORDER BY pref_id DESC LIMIT 1",
        (subject, attribute, now_s, now_s)).fetchone()


# --- the six tools ----------------------------------------------------------
# Each takes (store, args, ts, turn_id) and returns (ok, result).
#
# A model never supplies a timestamp or a turn id in `args`. Replay does add
# one harness-side key, `source_event_id`, which points a row at the gold
# event it came from; it is not in any schema and no model can name it.

def add_preference(store, args, ts, turn_id):
    subject = _check_subject(args)
    attribute = _check_attribute(args)
    value = _check_value(attribute, args.get("value"))
    scope = args.get("scope")
    if scope not in ("permanent", "temporary"):
        raise ValueError(f"scope must be 'permanent' or 'temporary', got {scope!r}")

    if scope == "temporary":
        if not args.get("valid_until"):
            raise ValueError("a temporary exception requires valid_until")
        valid_until = parse_ts(args["valid_until"])
        if valid_until <= ts:
            raise ValueError("valid_until must be after the statement itself")
        until = valid_until.isoformat()
    else:
        until = None

    pref_id = store.insert_pref(
        source_turn_id=turn_id, subject=subject, attribute=attribute,
        value=value, scope=scope, valid_from=ts.isoformat(),
        valid_until=until, ts=ts.isoformat(),
        source_event_id=args.get("source_event_id"))
    return True, {"pref_id": pref_id, "scope": scope, "valid_until": until}


def supersede_preference(store, args, ts, turn_id):
    subject = _check_subject(args)
    attribute = _check_attribute(args)
    value = _check_value(attribute, args.get("value"))

    prior = _find_active(store, subject, attribute, ts)
    if prior is None:
        return _fail(
            f"nothing to supersede: {subject} has no standing {attribute} "
            f"preference; use add_preference")
    if prior["value"] == value:
        return _fail(f"{subject}'s {attribute} is already {value!r}; "
                     f"nothing changed")

    # A revision that repeats an earlier value is still a revision: "from now
    # on, spicy" after a spicy request that has lapsed. The row it replaces is
    # the one in force, so a lapsed exception with the same value is not
    # consulted here; the new row has no window and is permanent.
    pref_id = store.insert_pref(
        source_turn_id=turn_id, subject=subject, attribute=attribute,
        value=value, scope="permanent", valid_from=ts.isoformat(),
        valid_until=None, ts=ts.isoformat(), supersedes=prior["pref_id"],
        source_event_id=args.get("source_event_id"))
    store.set_status(prior["pref_id"], "superseded", closed_by_pref_id=pref_id)
    return True, {"pref_id": pref_id, "superseded": prior["pref_id"]}


def clarify_statement(store, args, ts, turn_id):
    """Ask WHICH record a vague statement is about, and name it.

    The question is what the user hears; `resolves` is what the agent will
    do once the answer arrives, and it is the answer -- an event read off the
    user's reply, or a sentence in the answer -- never the statement itself.

    When that answer names a record that is standing right now, it goes at
    once: the reply came in this conversation, after the question, so the
    agent hears it before it uses the profile again. The row is dropped as
    of this turn (`dropped_at`), which keeps the record true for anything
    asked about an earlier instant. Nothing is dropped when the answer names
    no record ("the mushroom one" when the user never excluded mushrooms) or
    when the statement was vague and its answer is still vague: the question
    stays pending and the agent asks about the reply.
    """
    question = args.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    asks_about = args.get("asks_about")
    if asks_about is not None and asks_about not in ATTRIBUTES:
        raise ValueError(f"asks_about must be one of {sorted(ATTRIBUTES)}")
    resolves = args.get("resolves")
    if resolves is not None and not isinstance(resolves, dict):
        raise ValueError("resolves must name the answer: a pref_id, "
                         "a source_event_id or a value")

    cid = store.add_clarification(
        source_turn_id=turn_id, question=question.strip(),
        asks_about=asks_about, ts=ts.isoformat())

    row = _resolve_target(store, resolves) if resolves else None
    if row is None or row["status"] not in ("active", "corrected"):
        return True, {"clarification_id": cid, "status": "pending",
                      "resolved": False, "resolves": resolves}
    store.set_status(row["pref_id"], "superseded", dropped_at=ts.isoformat())
    store.resolve_clarification(cid, turn_id)
    return True, {"clarification_id": cid, "status": "resolved",
                  "removed": {"pref_id": row["pref_id"],
                              "attribute": row["attribute"],
                              "value": row["value"],
                              "dropped_at": ts.isoformat()}}


def _resolve_target(store: Store, resolves: dict):
    """The standing row an answer points at, from any of the three handles."""
    if resolves.get("target_pref_id") is not None:
        target = resolves["target_pref_id"]
        if not isinstance(target, int) or isinstance(target, bool):
            raise ValueError("target_pref_id must be an integer pref_id")
        return store.get_pref(target)
    if resolves.get("source_event_id") is not None:
        return store._conn.execute(
            "SELECT * FROM prefs WHERE source_event_id = ?",
            (resolves["source_event_id"],)).fetchone()
    # A model that writes {"pref_id": 3} for {"target_pref_id": 3} names
    # nothing usable. It gets the same refusal an empty `resolves` does --
    # never a KeyError, which dispatch does not catch and which would cost
    # the whole statement.
    value = resolves.get("value")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("resolves must name a pref_id, a source_event_id "
                         f"or a value; got keys {sorted(resolves)}")
    return store._conn.execute(
        "SELECT * FROM prefs WHERE subject = 'user' AND value = ?"
        " AND status IN ('active', 'corrected') ORDER BY pref_id DESC"
        " LIMIT 1", (value.strip(),)).fetchone()


def correct_preference(store, args, ts, turn_id):
    target = args.get("target_pref_id")
    if not isinstance(target, int) or isinstance(target, bool):
        raise ValueError("target_pref_id must be an integer pref_id")
    row = store.get_pref(target)
    if row is None:
        return _fail(f"no preference with pref_id {target}")
    if row["source_turn_id"] == turn_id:
        # Correcting the statement you are currently reading means you
        # recorded it twice, not that the user changed their mind.
        return _fail("cannot correct a preference recorded from the current "
                     "turn; supersede it instead")
    value = _check_value(row["attribute"], args.get("value"))
    # Restating the stored value is not a no-op. "No, I said spicy" after a
    # temporary spicy exception is the user making a one-off permanent, and
    # the row has to stay (with its valid_from) for the profile to match.
    store.correct_pref(target, value, ts.isoformat())
    return True, {"pref_id": target, "corrected_from": row["value"],
                  "value": value, "unchanged": value == row["value"]}


def ignore_statement(store, args, ts, turn_id):
    reason = args.get("reason")
    if reason not in ("chatter", "hypothetical", "no_new_information"):
        raise ValueError(f"unknown reason {reason!r}")
    return True, {"ignored": True, "reason": reason}


def ask_clarification(store, args, ts, turn_id):
    question = args.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    asks_about = args.get("asks_about")
    if asks_about is not None and asks_about not in ATTRIBUTES:
        raise ValueError(f"asks_about must be one of {sorted(ATTRIBUTES)}")
    cid = store.add_clarification(
        source_turn_id=turn_id, question=question.strip(),
        asks_about=asks_about, ts=ts.isoformat())
    return True, {"clarification_id": cid, "status": "pending"}


DISPATCH = {
    "add_preference": add_preference,
    "supersede_preference": supersede_preference,
    "clarify_statement": clarify_statement,
    "correct_preference": correct_preference,
    "ignore_statement": ignore_statement,
    "ask_clarification": ask_clarification,
}


def dispatch(store: Store, name: str, args: dict, ts: datetime,
             turn_id: str, attempt: int = 0) -> Tuple[bool, dict]:
    """Run one tool call. Never raises for model-supplied input.

    `ts` and `turn_id` come from the harness. If a caller passed a string ts
    it is parsed, so the tool signatures are forgiving about the harness while
    the model's fields stay strict.
    """
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    if name not in DISPATCH:
        return _fail(f"unknown tool {name!r}; expected one of {list(TOOL_NAMES)}")
    if not isinstance(args, dict):
        return _fail(f"tool arguments must be an object, got {type(args).__name__}")
    try:
        ok, result = DISPATCH[name](store, args, ts, turn_id)
    except ValueError as exc:
        ok, result = _fail(str(exc))

    store.log_decision(
        turn_id=turn_id, tool=name,
        args_json=json.dumps(args, sort_keys=True, default=str),
        ok=ok,
        result_json=json.dumps(result, sort_keys=True) if ok else None,
        error=None if ok else result.get("error"),
        attempt=attempt, ts=ts.isoformat())
    return ok, result


def prompt_context(ts: datetime, catalogues: dict) -> str:
    """The harness facts a model needs to fill a tool call correctly.

    Today's date and the allowed vocabulary are supplied by Python. Asking a
    model to resolve 'tonight' itself is how a benchmark ends up measuring
    date arithmetic instead of memory.
    """
    lines = [
        f"The current time is {ts.isoformat()}.",
        "Allowed subjects: user, brother, partner, friend.",
        f"Allowed attributes and values: {catalogues}",
        "valid_until is exclusive and must be absolute ISO 8601.",
    ]
    return "\n".join(lines)


# Kept next to the schemas so a prompt change and a schema change are one edit.
#
# The glossary and the statement-level rules are part of the extraction prompt.
# The first full run lost 76 `excluded_ingredient` facts to indirect wording
# and scored 0% on every ambiguity checkpoint, and the old prompt said
# nothing about either. No sentence here quotes a preference-statement
# template: teaching the model the corpus's wording would measure recall of
# THIS corpus instead of the classification it is meant to perform. (The
# time phrases are different: `resolve_time_phrases` lists every one with
# its resolved end, because date arithmetic is the harness's job.)
EXTRACTION_SYSTEM = """You maintain a user's food preferences across sessions. For each user turn, call exactly one tool.

Attributes:
- excluded_ingredient: a food the user wants kept out of their meals. It counts whether they say so plainly or only allude to it (a habit, a nickname, something they never order); both are preferences.
- spice_level: how much heat the user wants.
- diet: only a pattern the user names outright; never infer it from an exclusion.

Rules:
- A named other person holds the preference, not the user.
- A hypothetical changes nothing.
- A phrase like 'tonight' or 'this week' makes a temporary exception, not a revision; the harness gives you its absolute valid_until.
- A correction fixes a record that was stored wrongly; a revision replaces a record the user no longer wants.
- Ask (clarify_statement) only when the user means an earlier preference without naming it. Never ask when the statement itself says what they want.
- Fill only what the statement says; never invent a subject, a value, or a date.
"""
