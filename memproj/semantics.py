"""Gold-state rules -- the single source of truth for the benchmark.

ANNOTATION RULES
================
1. CUTOFF      Only events with ts <= now exist. Later statements have not
               been made yet.
2. SUBJECT     Only events whose subject matches exactly are considered.
               An other_person event can never reach the user's profile.
3. TEMPORARY   A temporary event contributes only while
               valid_from <= now < valid_until. valid_until is EXCLUSIVE:
               at that instant the exception has already expired. An
               expired exception is a distractor, not a fact.
4. REVISION    A permanent_revision replaces the standing value for its
               (subject, attribute). The superseded record leaves the
               current profile but stays visible to history().
5. CORRECTION  A correction rewrites the value of its target_event_id. The
               target's valid_from is unchanged -- the user is fixing a
               misheard value, not stating a new preference later.
6. INERT       hypothetical and filler never change state, at any now.
7. AMBIGUITY   An ambiguous_update with no referent_event_id changes
               nothing and makes gold_action() return "clarify" -- but only
               while 2+ restricted attributes are simultaneously active,
               since that is what makes the referent genuinely unclear.
               Once a resolving event sets the referent, the referent's
               value is removed from the profile as of the resolver's ts.

Order of application is by ts, ties broken on event_id, for determinism.
"""

from typing import Dict, List, Optional, Sequence, Set, Tuple

from memproj.schema import Event


# Types that carry preference content. Everything else is noise.
STATEFUL_TYPES = frozenset({
    "initial_pref",
    "permanent_revision",
    "temporary_exception",
    "other_person",
    "correction",
    "ambiguous_update",
})

# The attribute whose removal an ambiguous_update refers to in this domain.
RESTRICTED_ATTRIBUTE = "excluded_ingredient"


def _ordered(events: Sequence[Event]) -> List[Event]:
    """Chronological order, ties broken on event_id. Never uses set order."""
    return sorted(events, key=lambda e: (e.ts, e.event_id))


def _applies_now(event: Event, now) -> bool:
    """Rule 3, isolated so it reads the same everywhere."""
    if event.scope != "temporary":
        return True
    if event.valid_from > now:
        return False
    if event.valid_until is not None and now >= event.valid_until:
        return False
    return True


def _patch_corrections(events: Sequence[Event], subject: str, now) -> List[Event]:
    """Rule 5: fold each correction into the record it targets.

    A correction whose target is not visible by `now` is itself not yet
    known, so it is dropped rather than applied blindly.

    Later corrections win: if a record is corrected twice, the last one
    before `now` supplies the value.
    """
    # target event_id -> the value it ends up with
    patches: Dict[str, str] = {}
    for c in _ordered(events):
        if (c.type == "correction" and c.subject == subject
                and c.target_event_id is not None and c.ts <= now):
            patches[c.target_event_id] = c.value

    out = []
    for e in events:
        if e.type == "correction":
            continue  # applied via its target below
        patch = patches.get(e.event_id)
        if patch is None:
            out.append(e)
        else:
            out.append(Event(
                event_id=e.event_id,
                session=e.session,
                ts=e.ts,
                type=e.type,
                subject=e.subject,
                attribute=e.attribute,
                value=patch,                # the corrected value
                scope=e.scope,
                valid_from=e.valid_from,    # unchanged: rule 5
                valid_until=e.valid_until,
                target_event_id=e.target_event_id,
                needs_clarification=e.needs_clarification,
                referent_event_id=e.referent_event_id,
            ))
    return out


def _resolved_referents(events: Sequence[Event], subject: str, now) -> Dict[str, Event]:
    """Resolvers: ambiguous_update events that DO name a referent."""
    return {
        e.event_id: e
        for e in events
        if e.type == "ambiguous_update"
        and e.subject == subject
        and e.referent_event_id is not None
        and e.ts <= now
    }


def _standing_records(events: Sequence[Event], subject: str, now) -> Dict[str, List[Event]]:
    """attribute -> surviving record(s) after applying rules 1-7.

    Returned in application order so callers can pick the winner or report
    the full picture.
    """
    ordered = _ordered([e for e in events if e.subject == subject and e.ts <= now])
    ordered = _patch_corrections(ordered, subject, now)

    # Rule 7: collect referents removed by a resolver, and when.
    removals: Dict[str, object] = {}
    for resolver in _resolved_referents(ordered, subject, now).values():
        removals[resolver.referent_event_id] = resolver.ts

    standing: Dict[str, List[Event]] = {}
    for e in ordered:
        if e.type not in STATEFUL_TYPES or e.attribute is None:
            continue  # rules 6 and the unresolved-ambiguity case
        if e.type == "ambiguous_update":
            continue  # handled via removals above
        if e.type == "permanent_revision":
            # Rule 4: the revision replaces whatever is standing for this
            # subject+attribute. Named target goes; any earlier revision for
            # the same attribute goes too, so the latest one wins outright.
            recs = standing.get(e.attribute, [])
            recs = [r for r in recs
                    if r.event_id != e.target_event_id
                    and r.type != "permanent_revision"]
            standing[e.attribute] = recs
        if not _applies_now(e, now):
            continue  # rule 3: window not open, or already expired
        if e.event_id in removals:
            continue  # rule 7: this record was later made unimportant
        standing.setdefault(e.attribute, []).append(e)
    return standing


def gold_profile(
    events: Sequence[Event],
    subject: str,
    now,
    with_records: bool = False,
):
    """The preference state for `subject` at time `now`.

    Returns {attribute: set(value)}. With with_records=True also returns
    {attribute: [records]} so callers can inspect valid_from/valid_until --
    used by tests and by relevance().
    """
    standing = _standing_records(events, subject, now)
    profile: Dict[str, Set[str]] = {}
    for attribute, records in standing.items():
        values = {r.value for r in records if r.value is not None}
        if values:
            profile[attribute] = values
    if with_records:
        return profile, standing
    return profile


def history(events: Sequence[Event], subject: str, now) -> List[Event]:
    """Rule: every stateful event that ever applied to `subject` up to `now`,
    including superseded and expired ones. Correction patches are applied.
    """
    ordered = _ordered([e for e in events if e.subject == subject and e.ts <= now])
    ordered = _patch_corrections(ordered, subject, now)
    return [e for e in ordered if e.type in STATEFUL_TYPES and e.attribute is not None]


def gold_action(events: Sequence[Event], subject: str, now) -> str:
    """'clarify' when an unresolved ambiguity is pending, else 'recommend'."""
    pending = False
    for e in _ordered(events):
        if e.subject != subject or e.type not in ("ambiguous_update",):
            continue  # other subjects' events cannot resolve ours either
        if e.ts > now:
            break  # nothing later has happened yet
        if e.referent_event_id is not None:
            pending = False  # resolved
        else:
            # Rule 7 caveat: only genuinely ambiguous when 2+ restricted
            # attributes are active at this moment.
            restricted = gold_profile(events, subject, e.ts).get(
                RESTRICTED_ATTRIBUTE, set()
            )
            if len(restricted) >= 2:
                pending = True
    return "clarify" if pending else "recommend"


def relevance(
    events: Sequence[Event], subject: str, now
) -> Tuple[List[str], List[str]]:
    """Split past events into (relevant, distractor) ids for a checkpoint.

    Relevant = materially shaped the current profile: the winning record per
    attribute, plus any correction that rewrote one.
    Distractor = looks citable but must not be cited.

    Returned in chronological order.
    """
    standing = _standing_records(events, subject, now)
    relevant: Set[str] = set()
    for records in standing.values():
        for r in records:
            relevant.add(r.event_id)

    # A correction is citable only if it actually rewrote a live record.
    for c in events:
        if (c.type == "correction" and c.subject == subject and c.ts <= now
                and c.target_event_id in relevant):
            relevant.add(c.event_id)

    distractors: Set[str] = set()
    for e in _ordered([e for e in events if e.subject != subject and e.ts <= now]):
        if e.type in STATEFUL_TYPES:
            distractors.add(e.event_id)
    for e in _ordered([e for e in events if e.subject == subject and e.ts <= now]):
        if e.event_id in relevant:
            continue
        if e.type in ("hypothetical", "filler"):
            continue  # inert chatter is neither citable nor a trap
        if e.type == "correction":
            distractors.add(e.event_id)
        elif e.type == "ambiguous_update" and e.referent_event_id is None:
            continue
        else:
            distractors.add(e.event_id)  # expired, superseded, or removed

    order = {e.event_id: i for i, e in enumerate(_ordered(events))}
    by_time = lambda ids: sorted(ids, key=lambda i: order.get(i, 0))  # noqa: E731
    return by_time(relevant), by_time(distractors)
