"""The metrics. Pure functions over parsed dicts; no file I/O here.

Two things are scored separately, because the whole point of the benchmark
is that they come apart:

* **state** -- did the method reconstruct the active profile?
* **action** -- did the recommendation satisfy the user's actual profile,
  and did the citations support it?

A method can pass the first and fail the second (it recalls "mild" and then
recommends a spicy dish), and that cell is the one the results table has to
make visible. See `report.dissociation`.

**State is only defined for structured methods.** README, Method interface
gives the four free-text baselines `profile=None`, which means "no structured
opinion", not "no preferences". Scoring that as an empty profile would
report a false state failure for a method that never made the claim. So
`score()` is called only on rows whose profile is a dict, and the report
shows `n/a` for the rest; their action is still scored against the
checkpoint's answer key, which is the fair comparison.

**Values are compared canonically**, through `canonical()`: stripped, case-
folded, and singularised for `excluded_ingredient` with
`recipes._normalise`. Two halves of the system already agree that spelling
is not the fact -- `recipes.validate()` normalises before matching an
ingredient, and `tools._check_value` normalises on the way into the store --
so an evaluator comparing raw strings would fail a method that extracted
"Peanuts" correctly, on both state and contamination. `canonical()` is that
same rule, applied where equality is decided.

`diff_profiles`/`score` are duplicated in `memproj/memory/extract.py` (the
step-3 journal), and the two copies must agree on numbers while keeping
different representations: this one speaks `(attribute, value)` tuples and
names a disjoint attribute in `wrong_value`, that one speaks `"attr:value"`
strings and reports `wrong_value` as attribute names. That is deliberate --
do not "consolidate" by adopting one. The step-3 copy used to charge a
disjoint attribute twice (`tp = total - missing - wrong`, which credited a
hit for every gold value the method did *not* produce); it was fixed on
2026-10-25 to the same rule as this one -- facts always land in
`missing`/`extra`, `wrong_value` is a label only, and
`tp = gold facts - len(missing)` (never negative).
`tests/test_extract_run.py::test_the_state_counters_agree_with_the_evaluator`
pins the two together, so a change to either has to be made in both.
"""

from typing import Dict, List, Optional, Sequence, Set, Tuple

from memproj import recipes
from memproj.pipeline import recommend

PROFILE_ATTRIBUTES = ("diet", "spice_level", "excluded_ingredient")

# The attribute whose values are free text rather than a closed vocabulary.
FREE_TEXT_ATTRIBUTES = ("excluded_ingredient",)

# A recommended recipe that violates the profile is the headline failure.
# A missing recommendation is scored by `clarification_outcome` instead,
# so the two do not double-count the same checkpoint.


# --- value equality ---------------------------------------------------------

def canonical(attribute: str, value: str) -> str:
    """The one spelling of a fact, for comparing a profile with the answer key.

    `diet`/`spice_level` come from a closed vocabulary, so case and padding
    are the only noise. An `excluded_ingredient` is free text, and
    `recipes._normalise` is already the catalogue's rule for it -- the same
    function `tools._check_value` applies on write and `recipes.validate`
    applies before matching an ingredient. Using it here means the state
    metric, the action metric and the store all agree on what "the same
    ingredient" means.
    """
    text = str(value).strip()
    if attribute in FREE_TEXT_ATTRIBUTES:
        return recipes._normalise(text)
    return text.lower()


# --- state ------------------------------------------------------------------

def profile_pairs(profile: Optional[dict]) -> Set[Tuple[str, str]]:
    """`{attr: [values]}` as a set of canonical (attribute, value) pairs."""
    return {(attribute, canonical(attribute, value))
            for attribute, values in (profile or {}).items()
            for value in values}


def _attribute_sets(profile: Optional[dict]) -> Dict[str, Set[str]]:
    sets: Dict[str, Set[str]] = {}
    for attribute, value in profile_pairs(profile):
        sets.setdefault(attribute, set()).add(value)
    return sets


def diff_profiles(predicted: Optional[dict],
                  gold: Optional[dict]) -> Tuple[list, list, list]:
    """(missing, extra, wrong_value) between a predicted and a gold profile.

    *missing* and *extra* are (attribute, value) pairs, sorted.
    *wrong_value* names the attributes where both sides said something and
    nothing matched -- the "they said mild, gold says spicy" case, which the
    pair lists already charge as one miss for every gold value and one false
    positive for every value the method actually said. Naming the attribute
    separately is what gives the failure samples their one-line reason; it
    is not a third error.
    """
    predicted_sets = _attribute_sets(predicted)
    gold_sets = _attribute_sets(gold)

    missing = [(attribute, value)
               for attribute in sorted(gold_sets)
               for value in sorted(gold_sets[attribute]
                                   - predicted_sets.get(attribute, set()))]
    extra = [(attribute, value)
             for attribute in sorted(predicted_sets)
             for value in sorted(predicted_sets[attribute]
                                 - gold_sets.get(attribute, set()))]
    # Disjoint means something was said and something else was expected. An
    # attribute the method left empty is only missing, never wrong.
    wrong = [attribute for attribute in sorted(gold_sets)
             if gold_sets[attribute]
             and predicted_sets.get(attribute)
             and not (gold_sets[attribute] & predicted_sets[attribute])]
    return missing, extra, wrong


def score(predicted: Optional[dict], gold: Optional[dict]) -> dict:
    """State precision, recall and exact match for one checkpoint."""
    missing, extra, wrong = diff_profiles(predicted, gold)
    tp = len(profile_pairs(gold)) - len(missing)
    fn, fp = len(missing), len(extra)
    return {
        "missing": missing,
        "extra": extra,
        "wrong_value": wrong,
        "exact": not missing and not extra,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        # No claim and nothing missed is vacuously perfect; a method that
        # says nothing scores 1.0 precision and 0.0 recall, which is the
        # standard reading and keeps `exact` as the headline number.
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
    }


# --- action -----------------------------------------------------------------

def satisfaction(prediction: dict, checkpoint: dict,
                 catalogue: Sequence[dict]) -> dict:
    """Did the recommended recipe satisfy the checkpoint's answer key?

    Validated against `checkpoint["gold_profile"]`, never against the
    method's own profile: a method that is confidently wrong about the state
    does not get to grade its own homework.
    """
    if prediction["action"] != "recommend":
        return {"applicable": False, "satisfied": None, "violations": []}
    ok, violations = recommend.validate_choice(
        prediction["recipe_id"], checkpoint["gold_profile"], catalogue)
    return {"applicable": True, "satisfied": ok, "violations": violations}


def clarification_outcome(prediction: dict, checkpoint: dict) -> str:
    """One of: appropriate, unnecessary, missed, none.

    `missed` is the failure the benchmark cares about -- the state was
    genuinely ambiguous and the method guessed instead of asking.
    """
    gold = checkpoint["gold_action"]
    got = prediction["action"]
    if gold == "clarify":
        return "appropriate" if got == "clarify" else "missed"
    return "unnecessary" if got == "clarify" else "none"


# --- contamination ----------------------------------------------------------

def _source_label(event: dict, person: str, now: str,
                  distractors: Set[str], relevant: Set[str]) -> str:
    """Why a wrong fact is in the profile. Names the trap, not the symptom."""
    if event["subject"] != person:
        return "other_person"
    if event["type"] == "hypothetical":
        return "hypothetical"
    if event["type"] == "temporary_exception":
        until = event.get("valid_until")
        if until is not None and now >= until:
            return "expired_exception"
        if event["valid_from"] > now:
            return "future_exception"
        return "removed_restriction"
    if event["type"] in ("permanent_revision", "initial_pref"):
        return "superseded"
    if event["type"] == "correction":
        return "inapplicable_correction"
    if event["type"] == "ambiguous_update":
        return "ambiguous_update"
    if event["event_id"] in relevant:
        return "unsupported"
    if event["event_id"] in distractors:
        return "distractor"
    return "stale"


def contamination(prediction: dict, checkpoint: dict,
                  trajectory: dict) -> List[dict]:
    """Every extra profile fact, with the event that explains it.

    Only meaningful for structured methods; a free-text row has no profile
    to contaminate, and returns [].

    An extra fact that no event explains is `unsupported`: the method
    invented it. That is a different failure from applying the brother's
    preference to the user, and the table keeps them apart.
    """
    profile = prediction.get("profile")
    if profile is None:
        return []

    gold = profile_pairs(checkpoint["gold_profile"])
    extra = sorted(profile_pairs(profile) - gold)
    if not extra:
        return []

    person, now = checkpoint["person"], checkpoint["now"]
    distractors = set(checkpoint.get("distractor_event_ids", ()))
    relevant = set(checkpoint.get("relevant_event_ids", ()))
    events = [event for event in trajectory.get("events", ())
              if event["ts"] <= now]

    findings = []
    for attribute, value in extra:
        # Same canonical equality as the state metric: a method that stored
        # "Peanuts" from the peanut event has an extra fact, but it is not an
        # invented one. Matching raw strings would call it `unsupported`.
        matches = [event for event in events
                   if event.get("attribute") == attribute
                   and event.get("value") is not None
                   and canonical(attribute, event["value"]) == value]
        # Prefer a declared trap: it is the answer key's own account of why
        # this fact is in the history but must not be used.
        trapping = [event for event in matches
                    if event["event_id"] in distractors]
        event = (trapping or matches or [None])[0]
        findings.append({
            "attribute": attribute,
            "value": value,
            "event_id": event["event_id"] if event else None,
            "source": (_source_label(event, person, now, distractors, relevant)
                       if event else "unsupported"),
        })
    return findings


# --- provenance -------------------------------------------------------------

def provenance(prediction: dict, checkpoint: dict,
               trajectory: dict) -> List[str]:
    """Problems with the cited turns. Empty means every citation is support.

    The contract already guarantees each cited id exists in the trajectory
    (`validate_predictions.py`), so this checks the four things the contract
    cannot: the turn is the user's, it happened before the checkpoint, it
    carries a preference statement, and that statement is still applicable
    to this person.
    """
    turns = {turn["turn_id"]: turn
             for turn in trajectory.get("turns", ())}
    events = {event["event_id"]: event
              for event in trajectory.get("events", ())}
    distractors = set(checkpoint.get("distractor_event_ids", ()))
    person, now = checkpoint["person"], checkpoint["now"]

    problems: List[str] = []
    for turn_id in prediction.get("cited_turns", ()):
        turn = turns.get(turn_id)
        if turn is None:
            problems.append(f"{turn_id}: not a turn of this trajectory")
            continue
        if turn["speaker"] != "user":
            problems.append(f"{turn_id}: an assistant turn is not evidence")
            continue
        if turn["ts"] > now:
            problems.append(f"{turn_id}: happens after the checkpoint")
            continue
        event_id = turn.get("event_id")
        if not event_id:
            problems.append(f"{turn_id}: carries no preference statement")
            continue
        event = events.get(event_id)
        if event is None:
            problems.append(f"{turn_id}: event {event_id!r} is missing")
            continue
        if event["subject"] != person:
            problems.append(
                f"{turn_id}: {event['subject']}'s preference, not {person}'s")
            continue
        if event["type"] == "hypothetical":
            problems.append(f"{turn_id}: a hypothetical is not evidence")
            continue
        if event_id in distractors:
            problems.append(
                f"{turn_id}: cites a distractor ({event['type']})")
    return problems


# --- pairing ----------------------------------------------------------------

def pair_key(row: dict) -> tuple:
    """The twin-alignment key from README, Data formats.

    `kind` is not globally unique (`for_user` belongs to two families), so
    the family has to be in the key; the pair id separates one base/matched
    twin from the next pair in the same family.
    """
    return (row["family"], row["pair_id"], row["person"], row["kind"])
