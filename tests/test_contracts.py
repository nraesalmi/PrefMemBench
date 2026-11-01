"""Contract tests for the predictions format (README, Predictions format).

The validator exists so contract drift breaks a test rather than a results
table. These tests therefore spend most of their effort proving it *rejects*
things: a validator that never fails is indistinguishable from no validator.

Every rejection case is produced by mutating the one known-good line from
`fixtures/predictions_valid.jsonl` in exactly one way.
"""

import json
import os
import subprocess
import sys

import pytest

from memproj import config
from memproj.validate_predictions import validate_predictions

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
HISTORIES = os.path.join(FIXTURES, "histories.jsonl")
CHECKPOINTS = os.path.join(FIXTURES, "checkpoints.jsonl")
VALID = os.path.join(FIXTURES, "predictions_valid.jsonl")
BROKEN = os.path.join(FIXTURES, "predictions_broken.jsonl")
# A catalogue of its own, so a contract fixture cannot be broken by
# rebuilding the shipped catalogue (2026-10-29: the HUMMUS swap changed every
# recipe id, and "r02" stopped resolving).
CATALOGUE = os.path.join(FIXTURES, "catalogue.json")


def validate(path):
    """validate_predictions against the hand-written fixture corpus."""
    return validate_predictions(
        path,
        histories_path=HISTORIES,
        checkpoints_path=CHECKPOINTS,
        catalogue_path=CATALOGUE,
    )


def valid_line():
    with open(VALID, encoding="utf-8") as fh:
        return json.loads(fh.readline())


def write(tmp_path, lines):
    """Serialise dicts (or raw strings) to a predictions file."""
    path = tmp_path / "predictions.jsonl"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for line in lines:
            fh.write(line if isinstance(line, str) else json.dumps(line,
                                                                   sort_keys=True))
            fh.write("\n")
    return str(path)


# --- the happy path ---------------------------------------------------------
# Without this, every rejection test below could pass against a validator that
# rejects everything.

def test_valid_fixture_line_passes():
    assert validate(VALID) == []


def test_valid_line_written_to_a_temp_file_passes(tmp_path):
    assert validate(write(tmp_path, [valid_line()])) == []


def test_gold_line_may_omit_latency(tmp_path):
    """README, Predictions format: the gold baseline does no model call, so it has no
    comparable wall-clock and reports null rather than a number that would be
    averaged in with the methods that do."""
    line = valid_line()
    line.update(method="gold", model="gold", tokens=0, latency_s=None,
                profile={"spice_level": ["mild"]})
    assert validate(write(tmp_path, [line])) == [], validate(write(tmp_path, [line]))


# --- the whole point: a broken line must be caught --------------------------

def test_broken_fixture_line_is_rejected():
    errors = validate(BROKEN)
    assert len(errors) == 1, errors
    assert "must not carry a question" in errors[0]


def test_error_message_names_the_line(tmp_path):
    path = write(tmp_path, [valid_line(), valid_line()])
    errors = validate(path)
    assert all(str(path) in e for e in errors)
    assert any(":2:" in e for e in errors)


# --- one mutation, one rule -------------------------------------------------

def _drop(key):
    def mutate(line):
        del line[key]
        return line
    return mutate


def _extra(key, value):
    def mutate(line):
        line[key] = value
        return line
    return mutate


def _recommend_with_question(line):
    line["question"] = "Which cuisine?"
    return line


def _recommend_without_recipe(line):
    line["recipe_id"] = None
    return line


def _clarify_with_recipe(line):
    line["action"] = "clarify"
    line["question"] = "Which one?"
    return line


def _clarify_without_question(line):
    line["action"] = "clarify"
    line["recipe_id"] = None
    line["question"] = None
    return line


def _structured_method_with_null_profile(line):
    line["profile"] = None
    return line


def _free_text_method_with_profile(line):
    line["method"] = "recent_only"
    return line


def _gold_method_wrong_model(line):
    line["method"] = "gold"
    return line


def _non_gold_method_with_gold_model(line):
    line["model"] = "gold"
    return line


def _unknown_profile_attribute(line):
    line["profile"] = {"colour": ["red"]}
    return line


def _unsorted_profile_values(line):
    line["profile"] = {"spice_level": ["spicy", "mild"]}
    return line


def _unknown_trajectory(line):
    line["trajectory_id"] = "p999_base"
    return line


def _unknown_checkpoint(line):
    line["checkpoint_id"] = "p000_base_c99"
    return line


def _checkpoint_of_another_trajectory(line):
    line["checkpoint_id"] = "p001_base_c0"
    return line


def _cited_turn_of_another_trajectory(line):
    # t0003 exists, but only on p001_base. The line stays on p000_base, so the
    # citation is real yet does not belong to the trajectory being scored --
    # turn ids restart at t0001 within each history.
    line["cited_turns"] = ["t0003"]
    return line


MUTATIONS = [
    pytest.param(_drop("explanation"), "missing key", id="missing-key"),
    pytest.param(_extra("unexpected", 1), "extra key", id="extra-key"),
    pytest.param(_extra("method", "magic_memory"), "unknown method",
                 id="unknown-method"),
    pytest.param(_extra("action", "maybe"), "action must be",
                 id="action-outside-enum"),
    pytest.param(_extra("recipe_id", "r999"), "not in the catalogue",
                 id="recipe-not-in-catalogue"),
    pytest.param(_extra("cited_turns", ["t9999"]), "is not a turn_id",
                 id="cited-turn-not-in-trajectory"),
    pytest.param(_recommend_with_question, "must not carry a question",
                 id="recommend-with-question"),
    pytest.param(_recommend_without_recipe, "requires a recipe_id",
                 id="recommend-without-recipe"),
    pytest.param(_clarify_with_recipe, "must not carry a recipe_id",
                 id="clarify-with-recipe"),
    pytest.param(_clarify_without_question, "requires a question",
                 id="clarify-without-question"),
    pytest.param(_structured_method_with_null_profile, "structured",
                 id="null-profile-for-structured-method"),
    pytest.param(_free_text_method_with_profile, "free-text",
                 id="profile-for-free-text-method"),
    pytest.param(_gold_method_wrong_model, "model 'gold'",
                 id="gold-method-with-ollama-model"),
    pytest.param(_non_gold_method_with_gold_model, "reserved for method 'gold'",
                 id="gold-model-on-non-gold-method"),
    pytest.param(_unknown_profile_attribute, "unknown profile attribute",
                 id="unknown-profile-attribute"),
    pytest.param(_unsorted_profile_values, "not sorted",
                 id="unsorted-profile-values"),
    pytest.param(_extra("tokens", -1), "tokens must be",
                 id="negative-tokens"),
    pytest.param(_extra("latency_s", -0.5), "latency_s must be",
                 id="negative-latency"),
    pytest.param(_extra("latency_s", None), "only be null for method 'gold'",
                 id="null-latency-on-a-model-method"),
    pytest.param(_unknown_trajectory, "is not in histories",
                 id="unknown-trajectory"),
    pytest.param(_unknown_checkpoint, "is not in checkpoints",
                 id="unknown-checkpoint"),
    pytest.param(_checkpoint_of_another_trajectory, "does not belong to",
                 id="checkpoint-of-another-trajectory"),
    pytest.param(_cited_turn_of_another_trajectory, "is not a turn_id",
                 id="turn-of-another-trajectory"),
]


@pytest.mark.parametrize("mutate,expected", MUTATIONS)
def test_single_mutation_is_rejected(tmp_path, mutate, expected):
    line = mutate(valid_line())
    errors = validate(write(tmp_path, [line]))
    assert errors, f"expected rejection containing {expected!r}"
    assert any(expected in e for e in errors), errors


@pytest.mark.parametrize("mutate,expected", MUTATIONS)
def test_mutation_does_not_produce_a_flood_of_errors(tmp_path, mutate, expected):
    """One mutation should trip its own rule, not cascade into noise."""
    line = mutate(valid_line())
    errors = validate(write(tmp_path, [line]))
    assert len(errors) <= 2, errors


# --- file-level rules -------------------------------------------------------

def test_duplicate_tuple_is_rejected(tmp_path):
    path = write(tmp_path, [valid_line(), valid_line()])
    errors = validate(path)
    assert any("duplicate" in e for e in errors), errors


def test_different_method_is_not_a_duplicate(tmp_path):
    other = valid_line()
    other["method"] = "latest_wins"
    assert validate(write(tmp_path, [valid_line(), other])) == []


def test_malformed_json_is_reported_not_raised(tmp_path):
    errors = validate(write(tmp_path, ["{not json"]))
    assert len(errors) == 1
    assert "JSON" in errors[0]


def test_empty_file_is_valid(tmp_path):
    assert validate(write(tmp_path, [])) == []


def test_missing_corpus_file_is_reported_not_raised(tmp_path):
    errors = validate_predictions(
        VALID,
        histories_path=str(tmp_path / "nope.jsonl"),
        checkpoints_path=CHECKPOINTS,
        catalogue_path=CATALOGUE,
    )
    assert len(errors) == 1
    assert "histories" in errors[0]


# --- the CLI ----------------------------------------------------------------

def _cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "memproj.validate_predictions", *args],
        cwd=str(config.REPO_ROOT), capture_output=True, text=True,
    )


def test_cli_exits_zero_on_a_valid_file():
    result = _cli(VALID, "--histories", HISTORIES,
                  "--checkpoints", CHECKPOINTS, "--catalogue", CATALOGUE)
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_exits_non_zero_on_a_broken_file():
    result = _cli(BROKEN, "--histories", HISTORIES,
                  "--checkpoints", CHECKPOINTS)
    assert result.returncode != 0
    assert "must not carry a question" in result.stdout + result.stderr
