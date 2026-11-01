"""Check a predictions file against the contract in README, Predictions format.

Two ways in:

    from memproj.validate_predictions import validate_predictions
    errors = validate_predictions("out/predictions.jsonl")   # [] means valid

    python -m memproj.validate_predictions out/predictions.jsonl

Errors are plain strings of the form "path:line: message", so the CLI output is
greppable and the tests can assert on fragments. Nothing here raises on bad
input: an unreadable corpus, a malformed line or a duplicate entry is reported
like any other problem.

The evaluator reads only predictions.jsonl and checkpoints.jsonl (section 4), so
this module is the only thing standing between a drifted field name and a
results table that looks fine.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Set

from memproj import config

# The legal `method` values, mirroring README, Method interface's table.
METHODS = frozenset({
    "recent_only",
    "full_history",
    "rolling_summary",
    "retrieval",
    "latest_wins",
    "proposed",
    "gold",
})

# Methods that return a resolved profile. The rest are free text and must
# return None: None means "no structured opinion", {} means "the user has no
# active preferences", and the evaluator scores those differently.
STRUCTURED_METHODS = frozenset({"latest_wins", "proposed", "gold"})

# Attributes the generator can produce. A key outside this set is a typo that
# would silently score as a missing constraint.
PROFILE_ATTRIBUTES = frozenset({"diet", "spice_level", "excluded_ingredient"})

ACTIONS = ("recommend", "clarify")
GOLD = "gold"

FIELDS = frozenset({
    "trajectory_id", "checkpoint_id", "method", "model", "profile", "action",
    "recipe_id", "question", "explanation", "cited_turns", "tokens",
    "latency_s",
})


# --- loading ----------------------------------------------------------------

def _read_jsonl(path, label) -> tuple:
    """Read a JSONL file into dicts. Returns (rows, errors)."""
    if not os.path.exists(path):
        return None, [f"{label} file not found: {path}"]
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for n, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rows.append(json.loads(raw))
                except ValueError as exc:
                    return None, [f"{path}:{n}: invalid JSON in {label}: {exc}"]
    except OSError as exc:
        return None, [f"{label} file unreadable: {path}: {exc}"]
    return rows, []


def _read_catalogue(path) -> tuple:
    """The catalogue is a JSON array, not JSONL."""
    if not os.path.exists(path):
        return None, [f"catalogue file not found: {path}"]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            rows = json.load(fh)
    except (OSError, ValueError) as exc:
        return None, [f"catalogue file unreadable: {path}: {exc}"]
    if not isinstance(rows, list):
        return None, [f"catalogue must be a JSON list: {path}"]
    return rows, []


def _index_corpus(histories_path, checkpoints_path, catalogue_path) -> tuple:
    """Build the lookups the per-line checks need.

    Turn ids are per-trajectory (the generator restarts at t0001 in every
    history), so the turn lookup is keyed by trajectory; a bare set of all turn
    ids would accept a citation from the wrong conversation.
    """
    errors: List[str] = []

    turn_ids: Dict[str, Set[str]] = {}
    histories, errs = _read_jsonl(histories_path, "histories")
    errors.extend(errs)
    for row in histories or []:
        turn_ids[row["trajectory_id"]] = {
            t["turn_id"] for t in row.get("turns", [])
        }

    checkpoint_owner: Dict[str, str] = {}
    checkpoints, errs = _read_jsonl(checkpoints_path, "checkpoints")
    errors.extend(errs)
    for row in checkpoints or []:
        checkpoint_owner[row["checkpoint_id"]] = row["trajectory_id"]

    recipe_ids: Set[str] = set()
    catalogue, errs = _read_catalogue(catalogue_path)
    errors.extend(errs)
    for row in catalogue or []:
        recipe_ids.add(row["id"])

    return turn_ids, checkpoint_owner, recipe_ids, errors


# --- the per-line rules -----------------------------------------------------

def _check_exclusivity(line, errors) -> None:
    """A line recommends or clarifies, never both and never neither."""
    action = line["action"]
    if action not in ACTIONS:
        errors.append(
            f"action must be 'recommend' or 'clarify', got {action!r}"
        )
        return
    recipe_id, question = line["recipe_id"], line["question"]
    if action == "recommend":
        if recipe_id is None:
            errors.append("action 'recommend' requires a recipe_id")
        if question is not None:
            errors.append("action 'recommend' must not carry a question")
    else:
        if recipe_id is not None:
            errors.append("action 'clarify' must not carry a recipe_id")
        if question is None:
            errors.append("action 'clarify' requires a question")


def _check_profile(profile, errors) -> None:
    if not isinstance(profile, dict):
        errors.append("profile must be an object or null")
        return
    for attribute in sorted(profile):
        if attribute not in PROFILE_ATTRIBUTES:
            errors.append(f"unknown profile attribute: {attribute}")
            continue
        values = profile[attribute]
        if (not isinstance(values, list)
                or not all(isinstance(v, str) for v in values)):
            errors.append(f"profile[{attribute!r}] must be a list of strings")
            continue
        if values != sorted(values):
            errors.append(f"profile[{attribute!r}] values are not sorted")


def _check_method_and_model(line, errors) -> None:
    method, model = line["method"], line["model"]
    if method not in METHODS:
        errors.append(f"unknown method: {method!r}")
    if method == GOLD and model != GOLD:
        errors.append(f"method 'gold' requires model 'gold', got {model!r}")
    elif method != GOLD and model == GOLD:
        errors.append("model 'gold' is reserved for method 'gold'")

    profile = line["profile"]
    if method in STRUCTURED_METHODS:
        if profile is None:
            errors.append(f"method {method!r} is structured but profile is null")
        else:
            _check_profile(profile, errors)
    elif profile is not None:
        errors.append(f"method {method!r} is free-text but carries a profile")


def _check_references(line, turn_ids, checkpoint_owner, recipe_ids,
                      errors) -> None:
    trajectory_id = line["trajectory_id"]

    if line["recipe_id"] is not None and line["recipe_id"] not in recipe_ids:
        errors.append(
            f"recipe_id {line['recipe_id']!r} is not in the catalogue"
        )

    if trajectory_id not in turn_ids:
        errors.append(f"trajectory_id {trajectory_id!r} is not in histories")
    else:
        known = turn_ids[trajectory_id]
        for cited in line["cited_turns"]:
            if cited not in known:
                errors.append(
                    f"cited_turns entry {cited!r} is not a turn_id of "
                    f"{trajectory_id}"
                )

    checkpoint_id = line["checkpoint_id"]
    if checkpoint_id not in checkpoint_owner:
        errors.append(f"checkpoint_id {checkpoint_id!r} is not in checkpoints")
    elif checkpoint_owner[checkpoint_id] != trajectory_id:
        errors.append(
            f"checkpoint {checkpoint_id!r} does not belong to trajectory "
            f"{trajectory_id!r}"
        )


def _check_types(line, errors) -> None:
    tokens = line["tokens"]
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        errors.append(f"tokens must be a non-negative integer, got {tokens!r}")
    # null is allowed only for the gold baseline (README, Predictions format): it
    # does real work but no model call, so its cost is not comparable with the
    # methods that do. A number there would average in as if it were instant.
    latency = line["latency_s"]
    if latency is None:
        if line["method"] != GOLD:
            errors.append(
                f"latency_s may only be null for method {GOLD!r}, got null "
                f"for {line['method']!r}"
            )
    elif (isinstance(latency, bool)
            or not isinstance(latency, (int, float)) or latency < 0):
        errors.append(
            f"latency_s must be a non-negative number, got {latency!r}"
        )
    if not isinstance(line["explanation"], str):
        errors.append("explanation must be a string")
    cited = line["cited_turns"]
    if not isinstance(cited, list) or not all(isinstance(c, str) for c in cited):
        errors.append("cited_turns must be a list of strings")


def validate_line(line, turn_ids, checkpoint_owner, recipe_ids) -> List[str]:
    """Check one prediction dict. Returns a list of problems (empty if fine)."""
    if not isinstance(line, dict):
        return ["line is not a JSON object"]

    errors: List[str] = []
    keys = set(line)
    for missing in sorted(FIELDS - keys):
        errors.append(f"missing key: {missing}")
    for extra in sorted(keys - FIELDS):
        errors.append(f"extra key: {extra}")
    if errors:
        # Every rule below reads a specific field; without the full key set
        # the report would be noise rather than a diagnosis.
        return errors

    _check_method_and_model(line, errors)
    _check_exclusivity(line, errors)
    _check_types(line, errors)
    _check_references(line, turn_ids, checkpoint_owner, recipe_ids, errors)
    return errors


# --- the file-level rules ---------------------------------------------------

def validate_predictions(predictions_path,
                         histories_path=None,
                         checkpoints_path=None,
                         catalogue_path=None) -> List[str]:
    """Validate a whole predictions file. Returns errors; empty means valid."""
    turn_ids, checkpoint_owner, recipe_ids, corpus_errors = _index_corpus(
        histories_path or config.HISTORIES_PATH,
        checkpoints_path or config.CHECKPOINTS_PATH,
        catalogue_path or config.CATALOGUE_PATH,
    )
    if corpus_errors:
        # Without the corpora no line can be checked, so report the cause
        # rather than one reference error per line.
        return corpus_errors

    if not os.path.exists(predictions_path):
        return [f"predictions file not found: {predictions_path}"]

    errors: List[str] = []
    seen: Dict[tuple, int] = {}
    with open(predictions_path, "r", encoding="utf-8") as fh:
        for n, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                line = json.loads(raw)
            except ValueError as exc:
                errors.append(f"{predictions_path}:{n}: invalid JSON: {exc}")
                continue

            for problem in validate_line(line, turn_ids, checkpoint_owner,
                                         recipe_ids):
                errors.append(f"{predictions_path}:{n}: {problem}")

            if isinstance(line, dict):
                key = (line.get("trajectory_id"), line.get("checkpoint_id"),
                       line.get("method"), line.get("model"))
                if key in seen:
                    errors.append(
                        f"{predictions_path}:{n}: duplicate (trajectory_id, "
                        f"checkpoint_id, method, model) {key}, first seen at "
                        f"line {seen[key]}"
                    )
                else:
                    seen[key] = n
    return errors


# --- CLI --------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("predictions", help="path to predictions.jsonl")
    parser.add_argument("--histories", default=None,
                        help="default: config.HISTORIES_PATH")
    parser.add_argument("--checkpoints", default=None,
                        help="default: config.CHECKPOINTS_PATH")
    parser.add_argument("--catalogue", default=None,
                        help="default: config.CATALOGUE_PATH")
    args = parser.parse_args(argv)

    errors = validate_predictions(
        args.predictions,
        histories_path=args.histories,
        checkpoints_path=args.checkpoints,
        catalogue_path=args.catalogue,
    )
    if errors:
        for line in errors:
            print(line)
        print(f"{len(errors)} problem(s) found", file=sys.stderr)
        return 1
    print(f"OK: {args.predictions} is valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
