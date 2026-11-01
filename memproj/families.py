"""History construction: the families and their matched variants.

A family is a story shape. Each produces a base event list and a matched
event list differing in exactly one detail, so any behavioural difference
between the two is attributable to that detail alone.

No RNG lives here on purpose -- everything in this module is a pure
function of its arguments, which makes the pairing logic testable without
seeding anything.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from memproj.schema import Event


ATTRIBUTES = ("diet", "spice_level", "excluded_ingredient")

FAMILIES = (
    "temp_vs_perm",
    "subject_swap",
    "hypo_vs_real",
    "ambiguous_vs_resolved",
    "correction_vs_none",
    # --- composition families: the six update types interacting -------------
    # The five above each isolate ONE type in ONE narrative role, which is
    # what makes them clean. It also means the corpus never exercises the
    # rules composing -- a correction that retargets a windowed record, an
    # exception that expires and is then restated, an ambiguity that turns
    # out to need no resolution. These four put two types in sequence.
    "correction_of_temporary",
    "exception_then_restated",
    "ambiguous_then_clarify",
    "hypothetical_then_real",
    # --- attribute families: the two cells the corpus never exercised ------
    # Every family above states `spice_level` somewhere, and only ever with
    # `mild` or `spicy`; none states `diet` at all. That left two of
    # `recipes.validate()`'s three checks pre-satisfied, so a
    # method that hallucinated "keep it vegetarian" scored like one that did
    # not. These two add the missing cells without inventing a new rule.
    "diet_swap",
    "medium_vs_spicy",
)

# Which event type each family is "about", for the summary table.
FAMILY_TOPIC: Dict[str, str] = {
    "temp_vs_perm": "temporary_exception",
    "subject_swap": "other_person",
    "hypo_vs_real": "hypothetical",
    "ambiguous_vs_resolved": "ambiguous_update",
    "correction_vs_none": "correction",
    "correction_of_temporary": "correction x temporary_exception",
    "exception_then_restated": "expiry x re-statement",
    "ambiguous_then_clarify": "ambiguous_update x resolution",
    "hypothetical_then_real": "hypothetical x permanent_revision",
    "diet_swap": "diet",
    "medium_vs_spicy": "spice_level ceiling",
}


class GenerationError(RuntimeError):
    """Raised when a history cannot be built to spec."""


def _at_hour(day: datetime, hour: int, minute: int = 0) -> datetime:
    """Same day, at a given clock hour. Use this, not `+ timedelta(hours=18)`,
    which silently rolls into the next day."""
    return day.replace(hour=hour, minute=minute, second=0, microsecond=0)


def session_times(start: datetime, n_sessions: int,
                  gap_hours: float = 30.0) -> List[datetime]:
    """Session start times, separated by gaps of hours to days.

    Each session lands at a plausible hour: an evening slot on day 0, then
    late morning on later days, so checkpoints at "inside the window" really
    are inside it.
    """
    hours = (10, 11, 10, 11)  # per-session start hour
    out = []
    for i in range(n_sessions):
        day = start + timedelta(hours=gap_hours * i)
        out.append(_at_hour(day, hours[i % len(hours)]))
    return out


# --- helper builders --------------------------------------------------------

def _ev(event_id, session, ts, type, subject="user", attribute=None, value=None,
        scope="permanent", valid_from=None, valid_until=None,
        target_event_id=None, needs_clarification=False,
        referent_event_id=None) -> Event:
    return Event(
        event_id=event_id, session=session, ts=ts, type=type, subject=subject,
        attribute=attribute, value=value, scope=scope,
        valid_from=valid_from or ts, valid_until=valid_until,
        target_event_id=target_event_id,
        needs_clarification=needs_clarification,
        referent_event_id=referent_event_id,
    )


def _temporary_window(ts: datetime, phrase: str) -> datetime:
    """Resolve a relative phrase to a concrete end time.

    'tonight' and 'just this evening' end at midnight; the rest run whole
    days. The caller supplies the phrase; this is the only place that maps
    words to durations.
    """
    midnight = ts.replace(hour=23, minute=59, second=0, microsecond=0)
    if phrase in ("tonight", "just this evening"):
        return midnight if ts < midnight else ts + timedelta(hours=4)
    if phrase == "tomorrow":
        return (ts + timedelta(days=1)).replace(hour=23, minute=59)
    if phrase == "this weekend":
        return (ts + timedelta(days=2)).replace(hour=23, minute=59)
    if phrase == "for the next three days":
        return (ts + timedelta(days=3)).replace(hour=23, minute=59)
    if phrase == "for a couple of days":
        return (ts + timedelta(days=2)).replace(hour=23, minute=59)
    if phrase == "for the rest of the week":
        return (ts + timedelta(days=4)).replace(hour=23, minute=59)
    return (ts + timedelta(days=1)).replace(hour=23, minute=59)


# --- family: temp_vs_perm ---------------------------------------------------

def family_temp_vs_perm(sessions: List[datetime], other: str,
                        spice: str) -> Tuple[List[Event], List[Event], dict, dict, datetime]:
    """Base: spicy allowed tonight only. Matched: spicy allowed from now on."""
    s0, s1 = sessions[0], sessions[1]
    base_ts = _at_hour(s0, 18)
    phrase = "tonight"

    base = [
        _ev("e1", 0, s0, "initial_pref", attribute="spice_level", value="mild"),
        _ev("e2", 0, base_ts, "temporary_exception", attribute="spice_level",
            value=spice, scope="temporary",
            valid_from=base_ts, valid_until=_temporary_window(base_ts, phrase)),
    ]
    # The revision names the record it replaces, as every other revision in
    # the corpus does. Left unnamed, rule 4 takes nothing out (there is no
    # target to match and mild is an initial_pref, not a revision), and gold
    # would keep the old value beside the new one.
    matched = [
        _ev("e1", 0, s0, "initial_pref", attribute="spice_level", value="mild"),
        _ev("e2", 0, base_ts, "permanent_revision", attribute="spice_level",
            value=spice, target_event_id="e1"),
    ]
    slots = {"time_phrase": phrase, "attribute": "spice_level", "value": spice,
             "flip_session": 0}
    meta = {"flipped_field": "scope", "flip_event": "e2"}
    return base, matched, slots, meta, s1


# --- family: subject_swap ---------------------------------------------------

def family_subject_swap(sessions: List[datetime], other: str,
                        spice: str) -> Tuple[List[Event], List[Event], dict, dict, datetime]:
    """Base: *I* like spicy. Matched: *my brother* likes spicy."""
    s0, s1 = sessions[0], sessions[1]
    ts = s0 + timedelta(hours=2)

    # Base: the user revises their OWN preference to spicy, and names the
    # old one so the revision replaces exactly that record.
    base = [
        _ev("e1", 0, s0, "initial_pref", attribute="spice_level", value="mild"),
        _ev("e2", 0, ts, "permanent_revision", attribute="spice_level",
            value=spice, target_event_id="e1"),
    ]
    # Matched: the very same words, but attributed to the other person. The
    # user's own profile stays at mild -- the fact belongs to someone else.
    matched = [
        _ev("e1", 0, s0, "initial_pref", attribute="spice_level", value="mild"),
        _ev("e2", 0, ts, "other_person", subject=other,
            attribute="spice_level", value=spice),
    ]
    slots = {"person": other, "attribute": "spice_level", "value": spice,
             "flip_session": 0}
    meta = {"flipped_field": "subject", "flip_event": "e2"}
    return base, matched, slots, meta, s1


# --- family: hypo_vs_real ---------------------------------------------------

def family_hypo_vs_real(sessions: List[datetime], other: str,
                        spice: str) -> Tuple[List[Event], List[Event], dict, dict, datetime]:
    """Base: 'if I liked spicy...'. Matched: 'I like spicy now'."""
    s0, s1 = sessions[0], sessions[1]
    ts = s0 + timedelta(hours=2)

    base = [
        _ev("e1", 0, s0, "initial_pref", attribute="spice_level", value="mild"),
        _ev("e2", 0, ts, "hypothetical", attribute="spice_level", value=spice),
    ]
    matched = [
        _ev("e1", 0, s0, "initial_pref", attribute="spice_level", value="mild"),
        _ev("e2", 0, ts, "permanent_revision", attribute="spice_level",
            value=spice, target_event_id="e1"),
    ]
    slots = {"attribute": "spice_level", "value": spice, "flip_session": 0}
    meta = {"flipped_field": "type", "flip_event": "e2"}
    return base, matched, slots, meta, s1


# --- family: ambiguous_vs_resolved ------------------------------------------

def family_ambiguous_vs_resolved(sessions: List[datetime], other: str,
                                 spice: str) -> Tuple[List[Event], List[Event], dict, dict, datetime]:
    """Base: 'that restriction doesn't matter' -- which one? Matched: says which."""
    s0, s1 = sessions[0], sessions[1]
    ts = s1 + timedelta(hours=1)

    # Two active restrictions, so "that one" is genuinely unclear.
    common = [
        _ev("e1", 0, s0, "initial_pref", attribute="excluded_ingredient",
            value="peanut"),
        _ev("e2", 0, s0 + timedelta(minutes=40), "initial_pref",
            attribute="excluded_ingredient", value="mushroom"),
    ]
    base = common + [
        _ev("e3", 1, ts, "ambiguous_update", needs_clarification=True),
    ]
    matched = common + [
        _ev("e3", 1, ts, "ambiguous_update", needs_clarification=True,
            referent_event_id="e2"),
    ]
    slots = {"attribute": "excluded_ingredient", "value": "mushroom",
             "flip_session": 1}
    meta = {"flipped_field": "referent_event_id", "flip_event": "e3"}
    return base, matched, slots, meta, s1


# --- family: correction_vs_none ---------------------------------------------

def family_correction_vs_none(sessions: List[datetime], other: str,
                              spice: str) -> Tuple[List[Event], List[Event], dict, dict, datetime]:
    """Base: the user corrects a misheard ingredient. Matched: no correction."""
    s0, s1 = sessions[0], sessions[1]
    stated_at = s0 + timedelta(hours=1)
    corrected_at = s1 + timedelta(hours=1)

    common = [
        _ev("e1", 0, stated_at, "initial_pref",
            attribute="excluded_ingredient", value="peanut"),
    ]
    base = common + [
        _ev("e2", 1, corrected_at, "correction",
            attribute="excluded_ingredient", value="mushroom",
            target_event_id="e1"),
    ]
    matched = list(common)  # no correction at all
    slots = {"attribute": "excluded_ingredient", "value": "mushroom",
             "flip_session": 1}
    meta = {"flipped_field": "type", "flip_event": "e2"}
    return base, matched, slots, meta, s1


# --- family: correction_of_temporary ----------------------------------------
# GAP: a correction only ever retargeted a permanent record in the original
# corpus. Rule 5 (the target's valid_from is unchanged) is far sharper when
# the target is windowed: correcting a still-open exception must keep the
# window, and correcting an already-expired one must not resurrect it.

def family_correction_of_temporary(sessions: List[datetime], other: str,
                                   spice: str) -> Tuple[List[Event], List[Event], dict, dict, datetime]:
    """Base: corrects the ingredient inside an open window. Matched: corrects a permanent one.

    Both variants contain the SAME correction event at the SAME timestamp.
    Only its target differs: a windowed exception in the base, the permanent
    restriction in the matched. So the twins diverge by exactly rule 5 --
    whether the corrected value goes away when the window closes.

    The window is deliberately wide ('for the rest of the week') so the
    correction lands well inside it, in the next session.
    """
    s0, s1 = sessions[0], sessions[1]
    phrase = "for the rest of the week"          # spans several days
    exc_ts = s0 + timedelta(hours=2)
    corr_ts = s1 + timedelta(hours=1)

    first = _ev("e1", 0, s0, "initial_pref", attribute="excluded_ingredient",
                value="peanut")
    exception = _ev("e2", 0, exc_ts, "temporary_exception",
                    attribute="excluded_ingredient", value="mushroom",
                    scope="temporary", valid_from=exc_ts,
                    valid_until=_temporary_window(exc_ts, phrase))
    # Same event in both twins; only target_event_id differs.
    correction_to_window = _ev("e3", 1, corr_ts, "correction",
                               attribute="excluded_ingredient", value="aubergine",
                               target_event_id="e2")
    correction_to_perm = _ev("e3", 1, corr_ts, "correction",
                             attribute="excluded_ingredient", value="aubergine",
                             target_event_id="e1")

    base = [first, exception, correction_to_window]
    matched = [first, exception, correction_to_perm]
    slots = {"attribute": "excluded_ingredient", "value": "aubergine",
             "time_phrase": phrase, "flip_session": 1}
    meta = {"flipped_field": "target_event_id", "flip_event": "e3"}
    return base, matched, slots, meta, s1


# --- family: exception_then_restated ----------------------------------------
# GAP: nothing tested what happens when a window closes and the user says
# the same thing again. The re-statement is a NEW event with a NEW window,
# not an extension of the old one -- and the old one must not come back.

def family_exception_then_restated(sessions: List[datetime], other: str,
                                   spice: str) -> Tuple[List[Event], List[Event], dict, dict, datetime]:
    """Base: re-states tonight's exception tomorrow, as a fresh window.

    Matched: the same words, but the exception the user means is not
    tonight's -- it is TONIGHT, the one that has already lapsed -- so
    nothing new is granted.

    The twins differ in one field of e3, `value`: the re-statement in the
    base names spicy again (a new window, spicy tonight); in the matched it
    names no value -- the user is referring back to what has passed, so
    nothing is recorded and the night stays mild.

    Why not `type` or `target_event_id`: under rule 4 a permanent_revision
    also erases the standing mild, so the twin would differ in two ways at
    once; and `target_event_id` only means something on a revision -- on a
    temporary_exception it is inert, because naming a record does not make
    the new one end (a temporary_exception is not a permanent_revision, so
    rule 4's pruning never sees it).

    Under gold's rule 7 caveat this statement could be read as an
    ambiguity -- two restrictions are standing at its ts -- but the reader
    who does not is reading it the way gold's profile does, and the family
    tests only the profile. Clarification is measured in
    ambiguous_then_clarify.
    """
    s0, s1 = sessions[0], sessions[1]
    phrase = "tonight"
    first_ts = _at_hour(s0, 18)
    restate_ts = _at_hour(s1, 18)

    first = _ev("e1", 0, s0, "initial_pref", attribute="spice_level", value="mild")
    exception = _ev(
        "e2", 0, first_ts, "temporary_exception", attribute="spice_level",
        value=spice, scope="temporary", valid_from=first_ts,
        valid_until=_temporary_window(first_ts, phrase),
    )
    # Re-stated the NEXT DAY. Base: a fresh exception for the new night.
    # Matched: the same words, no value -- "tonight" means the night that
    # has ended. Gold only keeps a record that carries a value.
    said_again = dict(attribute="spice_level", scope="temporary",
                      valid_from=restate_ts,
                      valid_until=_temporary_window(restate_ts, phrase))
    base = [first, exception,
            _ev("e3", 1, restate_ts, "temporary_exception", value=spice,
                **said_again)]
    matched = [first, exception,
               _ev("e3", 1, restate_ts, "temporary_exception", value=None,
                   **said_again)]
    slots = {"time_phrase": phrase, "attribute": "spice_level", "value": spice,
             "flip_session": 1}
    meta = {"flipped_field": "value", "flip_event": "e3"}
    return base, matched, slots, meta, s1


# --- family: ambiguous_then_clarify -----------------------------------------
# GAP: the only ambiguity family resolves the ambiguity. The other branch --
# the user answers the clarification and REMOVES a restriction -- was never
# tested. Here both twins must ask (same question), and then diverge.

def family_ambiguous_then_clarify(sessions: List[datetime], other: str,
                                  spice: str) -> Tuple[List[Event], List[Event], dict, dict, datetime]:
    """Base: the user answers the clarification and drops the mushroom rule.

    Matched: the user answers but keeps both rules.

    Both variants leave the clarification unanswered, so gold_action is
    `clarify` for both at the first checkpoint -- the twins agree where they
    should, and diverge only after the answer.

    gold_action treats every vague statement as a question, not just the
    first: the matched's vague answer is still ambiguous (2+ restrictions
    standing), so the agent must ask again there. The base's answer names
    the referent and leaves one restriction, so it is answered outright.
    """
    s0, s1 = sessions[0], sessions[1]
    amb_ts = s1 + timedelta(hours=1)
    # A day after the question, so it is its own session. The answer has to
    # be spoken AFTER the pending_ambiguity checkpoint: gold says the agent
    # asks at 12:30, and by then the user cannot yet have answered.
    answer_ts = s1 + timedelta(days=1)

    common = [
        _ev("e1", 0, s0, "initial_pref", attribute="excluded_ingredient",
            value="peanut"),
        _ev("e2", 0, s0 + timedelta(minutes=40), "initial_pref",
            attribute="excluded_ingredient", value="mushroom"),
        # Unresolved in BOTH twins: this is what makes the agent ask.
        _ev("e3", 1, amb_ts, "ambiguous_update", needs_clarification=True),
    ]
    # Base: the answer names the referent, so the mushroom rule goes.
    base = common + [
        _ev("e4", 2, answer_ts, "ambiguous_update", referent_event_id="e2"),
    ]
    # Matched: the answer names nothing -- the user changed the subject.
    # Rule 7 keeps both restrictions, and because both are still standing
    # the vague answer is itself a new question (gold_action re-asks).
    matched = common + [
        _ev("e4", 2, answer_ts, "ambiguous_update", needs_clarification=True),
    ]
    slots = {"attribute": "excluded_ingredient", "value": "mushroom",
             "flip_session": 1}
    meta = {"flipped_field": "referent_event_id", "flip_event": "e4"}
    return base, matched, slots, meta, s1


# --- family: hypothetical_then_real -----------------------------------------
# GAP: a hypothetical was only ever followed by nothing. Here a hypothetical
# is followed by a REAL statement about the same attribute, testing whether
# the inert event contaminates the profile once a similar real fact exists.

def family_hypothetical_then_real(sessions: List[datetime], other: str,
                                  spice: str) -> Tuple[List[Event], List[Event], dict, dict, datetime]:
    """Base: the 'what if' is never made real. Matched: it later is.

    The hypothetical is present in BOTH twins at the same timestamp, so the
    twins differ only in whether a later permanent_revision exists.
    """
    s0, s1 = sessions[0], sessions[1]
    hypo_ts = s0 + timedelta(hours=2)
    real_ts = s1 + timedelta(hours=1)

    first = _ev("e1", 0, s0, "initial_pref", attribute="spice_level", value="mild")
    hypothesis = _ev("e2", 0, hypo_ts, "hypothetical",
                     attribute="spice_level", value=spice)
    # Gold's rule 4 keeps every older row of a revised attribute except the
    # one named here, so naming e1 is what leaves the user at mild+spicy
    # afterwards. Left unnamed, the revision would erase e1 too.
    real = _ev("e3", 1, real_ts, "permanent_revision", attribute="spice_level",
               value=spice, target_event_id="e1")
    base = [first, hypothesis]
    matched = [first, hypothesis, real]
    slots = {"attribute": "spice_level", "value": spice, "flip_session": 1}
    meta = {"flipped_field": "type", "flip_event": "e3"}
    return base, matched, slots, meta, s1


# --- family: diet_swap ------------------------------------------------------
# GAP: `diet` was never stated, by anyone, in any of the 303 events. The
# vegetarian branch of recipes.validate() therefore never fired, so no
# method could lose a point for recommending meat to a vegetarian user.
# The shape is subject_swap's -- the same words, attributed to someone else
# -- because that is the cheapest way to make the attribute load-bearing:
# the base twin must avoid every non-vegetarian recipe, the matched twin
# need not.

def family_diet_swap(sessions: List[datetime], other: str,
                     spice: str) -> Tuple[List[Event], List[Event], dict, dict, datetime]:
    """Base: *I'm* vegetarian. Matched: *my brother* is vegetarian."""
    s0, s1 = sessions[0], sessions[1]
    ts = s0 + timedelta(hours=2)

    base = [
        _ev("e1", 0, s0, "initial_pref", attribute="spice_level", value="mild"),
        _ev("e2", 0, ts, "initial_pref", attribute="diet", value="vegetarian"),
    ]
    # The same sentence about someone else. The user's own profile keeps
    # only the mild ceiling; the fact belongs to the brother.
    matched = [
        _ev("e1", 0, s0, "initial_pref", attribute="spice_level", value="mild"),
        _ev("e2", 0, ts, "other_person", subject=other,
            attribute="diet", value="vegetarian"),
    ]
    slots = {"person": other, "attribute": "diet", "value": "vegetarian",
             "flip_session": 0}
    meta = {"flipped_field": "subject", "flip_event": "e2"}
    return base, matched, slots, meta, s1


# --- family: medium_vs_spicy ------------------------------------------------
# GAP: every gold profile said `mild`, `spicy`, or both -- never `medium`.
# `spicy` cannot fail validate() (it is the top of SPICE_ORDER), so the only
# ceiling ever tested was "nothing". Here the base caps at medium: 16 spicy
# recipes must be rejected and 45 accepted. The matched raises the same
# statement to spicy, which is what makes this a matched pair rather than a
# one-sided harder question -- and it doubles as the control, because both
# twins differ in exactly the value the ceiling reads.

def family_medium_vs_spicy(sessions: List[datetime], other: str,
                           spice: str) -> Tuple[List[Event], List[Event], dict, dict, datetime]:
    """Base: the user's ceiling is medium. Matched: spicy."""
    s0, s1 = sessions[0], sessions[1]
    ts = s0 + timedelta(hours=2)

    base = [
        _ev("e1", 0, s0, "initial_pref", attribute="spice_level", value="mild"),
        _ev("e2", 0, ts, "permanent_revision", attribute="spice_level",
            value="medium", target_event_id="e1"),
    ]
    matched = [
        _ev("e1", 0, s0, "initial_pref", attribute="spice_level", value="mild"),
        _ev("e2", 0, ts, "permanent_revision", attribute="spice_level",
            value=spice, target_event_id="e1"),
    ]
    slots = {"attribute": "spice_level", "value": "medium", "flip_session": 0}
    meta = {"flipped_field": "value", "flip_event": "e2"}
    return base, matched, slots, meta, s1


BUILDERS = {
    "temp_vs_perm": family_temp_vs_perm,
    "subject_swap": family_subject_swap,
    "hypo_vs_real": family_hypo_vs_real,
    "ambiguous_vs_resolved": family_ambiguous_vs_resolved,
    "correction_vs_none": family_correction_vs_none,
    "correction_of_temporary": family_correction_of_temporary,
    "exception_then_restated": family_exception_then_restated,
    "ambiguous_then_clarify": family_ambiguous_then_clarify,
    "hypothetical_then_real": family_hypothetical_then_real,
    "diet_swap": family_diet_swap,
    "medium_vs_spicy": family_medium_vs_spicy,
}


def build_pair(family: str, sessions: List[datetime], other_person: str,
               spicy_value: str):
    """(base_events, matched_events, render_slots, meta, flip_session)."""
    if family not in BUILDERS:
        raise GenerationError(f"unknown family {family!r}")
    return BUILDERS[family](sessions, other_person, spicy_value)


# --- checkpoints ------------------------------------------------------------

def checkpoint_specs(family: str, events: List[Event],
                     sessions: List[datetime]) -> List[dict]:
    """Where to place checkpoints for this history.

    Each spec is {after, now, person, kind}. `now` is the instant the user
    asks for a suggestion, chosen to land at an informative moment: inside
    a temporary window, after it expires, right after a correction, and so
    on. The twins are asked the same questions at the same instants.

    A family that needs a different `now` to differ is a family whose
    events do not differ; the pair tests catch that.
    """
    return _specs(family, events, sessions)


def _specs(family: str, events: List[Event],
           sessions: List[datetime]) -> List[dict]:
    """The checkpoint list both twins share, before any variant tweak."""
    by_type: Dict[str, List[Event]] = {}
    for e in events:
        by_type.setdefault(e.type, []).append(e)

    # Whichever other person this history actually uses -- never a guess,
    # or the checkpoint would ask about someone the events never mention.
    other = next((e.subject for e in events if e.type == "other_person"), None)

    specs: List[dict] = []
    last_ts = events[-1].ts if events else sessions[0]
    last_id = events[-1].event_id if events else None

    if family == "temp_vs_perm":
        # The second statement is the flip: a temporary exception in the
        # base, a permanent revision in the matched. Both variants must be
        # asked the SAME two questions, or they cannot be compared:
        #   temp_inside  -- right after the statement, window still open
        #   temp_expired -- well after the window would have closed
        # For the base, expiry is real. For the matched the window never
        # existed, so the value simply persists. Both questions are asked
        # about "tonight" whatever the checkpoint's own clock says: the
        # checkpoint time is the clock the resolver reads, not the meal.
        second = (by_type.get("temporary_exception")
                  or by_type.get("permanent_revision") or [None])[0]
        if second is not None:
            inside = second.ts + timedelta(hours=1)
            # A day later is safely past any window this domain uses.
            after = second.ts + timedelta(days=1, hours=1)
            specs.append({"after": second.event_id, "now": inside,
                          "person": "user", "kind": "temp_inside"})
            specs.append({"after": second.event_id, "now": after,
                          "person": "user", "kind": "temp_expired"})

    elif family in ("subject_swap", "diet_swap"):
        # diet_swap asks exactly the same questions: the user's own meal,
        # where the twins differ, and the other person's, where they do not
        # (the base has no other_person event, so it is asked only the
        # first; `subject_swap` is the only asymmetric pair). The two variants
        # anchor on DIFFERENT statements, because the fact sits with the
        # user in one and with the brother in the other: without the diet
        # fallback the base would find nothing and fall through to a
        # `final` checkpoint with no counterpart question.
        inst = (by_type.get("other_person") or by_type.get("permanent_revision")
                or [e for e in events if e.attribute == "diet"] or [None])[0]
        if inst is not None:
            # The user's own meal: base has spicy, matched must still be mild.
            specs.append({"after": inst.event_id,
                          "now": inst.ts + timedelta(hours=1),
                          "person": "user", "kind": "for_user"})
            # And the other person's meal, where the fact actually lives.
            if other is not None:
                specs.append({"after": inst.event_id,
                              "now": inst.ts + timedelta(hours=2),
                              "person": other, "kind": "for_other_person"})

    elif family in ("hypo_vs_real", "medium_vs_spicy"):
        # medium_vs_spicy's flip is a revision in both twins, so the
        # permanent_revision lookup finds it either way.
        inst = by_type.get("hypothetical") or by_type.get("permanent_revision")
        inst = (inst or [None])[0]
        if inst is not None:
            # Base: "if I liked spicy" changes nothing. Matched: it does.
            specs.append({"after": inst.event_id,
                          "now": inst.ts + timedelta(hours=1),
                          "person": "user", "kind": "for_user"})

    elif family == "ambiguous_vs_resolved":
        amb = by_type.get("ambiguous_update", [None])[0]
        if amb is not None:
            specs.append({"after": amb.event_id, "now": amb.ts + timedelta(hours=1),
                          "person": "user", "kind": "after_ambiguity"})

    elif family == "correction_of_temporary":
        # Three moments, each asked of BOTH twins at the same instant:
        #   inside_window      -- exception live, pre-correction (twins agree)
        #   window_corrected   -- post-correction, window still open (differ)
        #   window_closed      -- window lapsed (differ: only the matched
        #                         still carries the corrected value)
        exc = (by_type.get("temporary_exception") or [None])[0]
        corr = (by_type.get("correction") or [None])[0]
        if exc is not None:
            specs.append({"after": exc.event_id,
                          "now": exc.ts + timedelta(hours=1),
                          "person": "user", "kind": "inside_window"})
        if corr is not None and exc is not None:
            specs.append({"after": corr.event_id,
                          "now": corr.ts + timedelta(hours=1),
                          "person": "user", "kind": "window_corrected"})
        if exc is not None and exc.valid_until is not None:
            specs.append({"after": exc.event_id,
                          "now": exc.valid_until + timedelta(hours=1),
                          "person": "user", "kind": "window_closed"})

    elif family == "exception_then_restated":
        # Asked of both twins at the same two instants:
        #   first_window_open -- the original exception is live (twins agree)
        #   after_restate     -- the re-statement has happened (twins differ)
        exc = (by_type.get("temporary_exception") or [None])[0]
        third = (events[2] if len(events) > 2 else None)
        if exc is not None:
            specs.append({"after": exc.event_id,
                          "now": exc.ts + timedelta(hours=1),
                          "person": "user", "kind": "first_window_open"})
        if third is not None:
            # An hour after the re-statement: its window is open, and the
            # lapsed first-night exception must not be revived alongside it.
            specs.append({"after": third.event_id,
                          "now": third.ts + timedelta(hours=1),
                          "person": "user", "kind": "after_restate"})

    elif family == "ambiguous_then_clarify":
        #   pending_ambiguity  -- both twins must ask (gold_action == clarify)
        #   after_answer       -- base has dropped a restriction and asks no
        #                         more; matched still holds both and re-asks
        amb = (by_type.get("ambiguous_update") or [None])[0]
        if amb is not None:
            specs.append({"after": amb.event_id,
                          "now": amb.ts + timedelta(minutes=30),
                          "person": "user", "kind": "pending_ambiguity"})
        if len(events) > 3:
            answer = events[3]
            specs.append({"after": answer.event_id,
                          "now": answer.ts + timedelta(hours=1),
                          "person": "user", "kind": "after_answer"})

    elif family == "hypothetical_then_real":
        #   after_hypothetical   -- identical in both twins (both inert)
        #   after_real_statement -- only the matched has revised anything
        hypo = (by_type.get("hypothetical") or [None])[0]
        if hypo is not None:
            specs.append({"after": hypo.event_id,
                          "now": hypo.ts + timedelta(hours=1),
                          "person": "user", "kind": "after_hypothetical"})
        # The real statement exists only in the matched, so the BASE cannot
        # find it by event type. Ask both twins at the same wall-clock
        # instant instead: the moment the matched's revision lands.
        real = (by_type.get("permanent_revision") or [None])[0]
        real_ts = real.ts if real is not None else _at_hour(sessions[1], 12)
        specs.append({"after": last_id,
                      "now": real_ts + timedelta(hours=1),
                      "person": "user", "kind": "after_real_statement"})

    elif family == "correction_vs_none":
        corr = by_type.get("correction", [None])[0]
        anchor = corr or by_type.get("initial_pref", [None])[0]
        if anchor is not None:
            specs.append({"after": anchor.event_id,
                          "now": anchor.ts + timedelta(hours=1),
                          "person": "user", "kind": "after_statement"})
    if not specs:
        specs.append({"after": last_id, "now": last_ts + timedelta(hours=1),
                      "person": "user", "kind": "final"})
    return specs
