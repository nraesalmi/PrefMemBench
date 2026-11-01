"""Data structures for the preference-memory benchmark.

An Event is ground truth. A Turn is how that ground truth was said out loud.
Nothing here derives meaning -- that is semantics.py's job.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


EVENT_TYPES = (
    "initial_pref",
    "permanent_revision",
    "temporary_exception",
    "other_person",
    "hypothetical",
    "correction",
    "ambiguous_update",
    "filler",
)

SCOPES = ("permanent", "temporary")


@dataclass(frozen=True)
class Event:
    """One preference statement (or one piece of inert chatter)."""

    event_id: str
    session: int
    ts: datetime
    type: str
    subject: str
    attribute: Optional[str]
    value: Optional[str]
    scope: str
    valid_from: datetime
    valid_until: Optional[datetime] = None
    # permanent_revision / correction point at the record they act on.
    target_event_id: Optional[str] = None
    # ambiguous_update: True while the referent is unknown.
    needs_clarification: bool = False
    # ambiguous_update: set by the later event that resolves it.
    referent_event_id: Optional[str] = None


@dataclass(frozen=True)
class Turn:
    """One utterance. event_id links it back to the event it expresses."""

    turn_id: str
    session: int
    ts: datetime
    speaker: str  # "user" | "assistant"
    text: str
    event_id: Optional[str] = None
    # True when the user implied the preference rather than stating it.
    # Always False for filler and assistant turns. Recorded here so the
    # explicitness ablation does not have to re-parse the text.
    is_implicit: bool = False


@dataclass(frozen=True)
class Checkpoint:
    """A moment where the agent must decide what to do."""

    checkpoint_id: str
    trajectory_id: str
    after_event_id: Optional[str]
    now: datetime
    person: str
    request_text: str
    gold_profile: dict
    gold_action: str  # "recommend" | "clarify"
    relevant_event_ids: tuple = ()
    distractor_event_ids: tuple = ()
    # What kind of moment this checkpoint tests, e.g. "temp_inside",
    # "temp_expired", "after_ambiguity". Answers the same question across a
    # base/matched pair so the two can be compared directly.
    kind: str = "final"


@dataclass(frozen=True)
class Trajectory:
    """A whole conversation history, plus its pairing metadata."""

    trajectory_id: str
    pair_id: str
    variant: str  # "base" | "matched"
    family: str
    flipped_field: str
    events: tuple = ()
    turns: tuple = ()
