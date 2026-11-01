"""The resolver's ablation flags: each one breaks only the rule it turns off.

Step 2's sanity test. The corpus is replayed with every flag on (no
mismatch is replay.py's job); here the same corpus is replayed with ONE
flag on, and the checkpoint kinds allowed to move are named per flag. A
flag that leaves every checkpoint alone is a rule nobody implemented; one
that moves a checkpoint outside its list is a rule leaking into another.

No LLM call appears here, and none may be added.
"""

import json

import pytest

from memproj import config, replay
from memproj.resolver import active_profile, history
from memproj.store import Store
from memproj.schema import Turn
from memproj.tools import dispatch
from datetime import datetime

T0 = datetime(2026, 3, 1, 10, 5)
T1 = datetime(2026, 3, 2, 11, 0)

# Checkpoint kinds each flag may break, as (family, kind). Read off the
# corpus in one run, each list answering "which rule does this checkpoint
# depend on": the scope list and the time list must mirror each other, since
# an exception kept forever and an exception never opened are one behaviour.
ALLOWED = {
    "use_scope": {("temp_vs_perm", "temp_expired"),
                  ("correction_of_temporary", "window_closed"),
                  ("exception_then_restated", "after_restate")},
    "use_time": {("temp_vs_perm", "temp_expired"),
                 ("correction_of_temporary", "window_closed"),
                 ("exception_then_restated", "after_restate")},
    "use_subject": {("subject_swap", "for_user"),
                    ("subject_swap", "for_other_person"),
                    # diet_swap is subject_swap's shape with the attribute
                    # that actually gates the catalogue: ignoring the subject
                    # feeds the brother's vegetarian diet to the user.
                    ("diet_swap", "for_user"),
                    ("diet_swap", "for_other_person")},
    "use_supersede": {("temp_vs_perm", "temp_inside"),
                      ("temp_vs_perm", "temp_expired"),
                      ("subject_swap", "for_user"),
                      ("hypo_vs_real", "for_user"),
                      ("hypothetical_then_real", "after_real_statement"),
                      # medium_vs_spicy reads the revision: without it the
                      # profile keeps the superseded mild and the medium
                      # ceiling is never applied.
                      ("medium_vs_spicy", "for_user")},
    # Without the rule the restatement closes nothing, so the value it
    # corrected keeps standing beside the new one -- after the restatement,
    # never before it (dating the edit is not this flag's job).
    "use_correction": {("correction_of_temporary", "window_corrected"),
                       ("correction_of_temporary", "window_closed"),
                       ("correction_vs_none", "after_statement")},
}


@pytest.fixture(scope="module")
def corpus():
    return replay.load_corpus()


def _broken(corpus, flag):
    trajectories, checkpoints = corpus
    report = replay.run(trajectories, checkpoints, flags={flag: False})
    return {(m["family"], m["kind"].split(":")[0]) for m in report["mismatches"]}


def test_every_flag_is_exercised_by_the_corpus(corpus):
    for flag, allowed in ALLOWED.items():
        assert _broken(corpus, flag), f"{flag} off changed nothing: the rule " \
                                      f"it turns off is not implemented"
    assert len(ALLOWED) == 5


@pytest.mark.parametrize("flag", sorted(ALLOWED))
def test_a_flag_breaks_only_the_checkpoints_its_rule_governs(corpus, flag):
    """Every broken checkpoint must be one whose answer its rule decides.

    Both halves matter: a flag that breaks nothing is a rule that does not
    exist, and one that breaks a checkpoint outside its list is a rule
    leaking into a decision it does not own.
    """
    broken = _broken(corpus, flag)
    leaked = broken - ALLOWED[flag]
    assert not leaked, f"{flag} also moved {sorted(leaked)}"
    assert broken == ALLOWED[flag],         f"{flag} left {sorted(ALLOWED[flag] - broken)} untouched"


def test_scope_and_time_are_the_same_rule_from_two_sides(corpus):
    assert _broken(corpus, "use_scope") == _broken(corpus, "use_time")


def test_no_flag_on_is_exact(corpus):
    trajectories, checkpoints = corpus
    assert replay.run(trajectories, checkpoints)["mismatches"] == []


# --- history() --------------------------------------------------------------

def _ledger():
    """Two turns and a small store: corpus-free, in-memory."""
    s = Store(":memory:").create()
    s.add_turn(Turn("t0001", 0, T0, "user", "I usually eat mild food."))
    s.add_turn(Turn("t0002", 1, T1, "user", "Actually, spicy is fine."))
    return s


def _call(store, name, ts, turn, **args):
    ok, result = dispatch(store, name, args, ts, turn)
    assert ok, result
    return result


def test_history_keeps_superseded_rows_with_the_live_one():
    s = _ledger()
    old = _call(s, "add_preference", T0, "t0001", subject="user",
                attribute="spice_level", value="mild", scope="permanent")
    _call(s, "supersede_preference", T1, "t0002", subject="user",
          attribute="spice_level", value="spicy")
    rows = history(s, "user")
    assert [r["pref_id"] for r in rows] == [old["pref_id"], 2]
    assert [r["status"] for r in rows] == ["superseded", "active"]
    assert active_profile(s, "user", T1) == {"spice_level": {"spicy"}}
    s.close()


def test_a_revision_takes_effect_from_its_own_turn():
    """A whole-trajectory store must not leak a later session backwards.

    Replay reprocesses the store per checkpoint, so it never meets this; a
    method that extracts once and reads the same store at every `now` does.
    Before the fix a superseded row left the profile at once, so `proposed`
    answered {} at p017_matched_c0 where the answer key said mild, and the
    revision to spicy only arrives in a later session.
    """
    s = _ledger()
    _call(s, "add_preference", T0, "t0001", subject="user",
          attribute="spice_level", value="mild", scope="permanent")
    _call(s, "supersede_preference", T1, "t0002", subject="user",
          attribute="spice_level", value="spicy")
    between = datetime(2026, 3, 1, 18, 0)          # before the revision
    assert active_profile(s, "user", between) == {"spice_level": {"mild"}}
    assert active_profile(s, "user", T0) == {"spice_level": {"mild"}}
    assert active_profile(s, "user", T1) == {"spice_level": {"spicy"}}
    s.close()


def test_a_revision_never_merges_two_standing_values():
    """The fix must not keep the old value alive after the revision lands."""
    s = _ledger()
    _call(s, "add_preference", T0, "t0001", subject="user",
          attribute="excluded_ingredient", value="peanut", scope="permanent")
    _call(s, "supersede_preference", T1, "t0002", subject="user",
          attribute="excluded_ingredient", value="mushroom")
    assert active_profile(s, "user", T1)[ "excluded_ingredient"] == {"mushroom"}
    assert active_profile(s, "user", T1)["excluded_ingredient"] != {
        "peanut", "mushroom"}
    s.close()


def test_a_restatement_takes_effect_from_its_own_turn():
    """Same blind spot as a revision: replay rebuilds, a cache does not.

    Gold never shows a corrected value before the user says it -- p005_base
    answers {mushroom, peanut} at c0, an hour after the statement and a day
    before the correction that rewrites it. The control used to start the
    corrected value at the row's `valid_from`, so a whole-trajectory store
    answered {aubergine} there, one session early.
    """
    s = _ledger()
    first = _call(s, "add_preference", T0, "t0001", subject="user",
                  attribute="excluded_ingredient", value="mushroom",
                  scope="permanent")
    _call(s, "correct_preference", T1, "t0002",
          target_pref_id=first["pref_id"], value="aubergine")
    between = datetime(2026, 3, 1, 18, 0)          # before the restatement
    assert active_profile(s, "user", between) == {
        "excluded_ingredient": {"mushroom"}}
    assert active_profile(s, "user", T1) == {
        "excluded_ingredient": {"aubergine"}}
    s.close()


def test_a_restatement_retires_the_value_it_corrected():
    """Dating the edit must not leave both values standing after it."""
    s = _ledger()
    first = _call(s, "add_preference", T0, "t0001", subject="user",
                  attribute="spice_level", value="mild", scope="permanent")
    _call(s, "correct_preference", T1, "t0002",
          target_pref_id=first["pref_id"], value="medium")
    assert active_profile(s, "user", T1) == {"spice_level": {"medium"}}
    # And the ablation adds exactly the contaminant its name promises.
    assert active_profile(s, "user", T1, use_correction=False) == {
        "spice_level": {"mild", "medium"}}
    s.close()


def test_history_now_cuts_at_the_instant_asked_about():
    s = _ledger()
    _call(s, "add_preference", T0, "t0001", subject="user",
          attribute="spice_level", value="mild", scope="permanent")
    _call(s, "add_preference", T1, "t0002", subject="user",
          attribute="excluded_ingredient", value="peanut", scope="permanent")
    assert len(history(s, "user", T0)) == 1
    assert len(history(s, "user", T1)) == 2
    s.close()


def test_history_shows_the_current_value_and_what_it_corrected():
    s = _ledger()
    first = _call(s, "add_preference", T0, "t0001", subject="user",
                  attribute="spice_level", value="mild", scope="permanent")
    _call(s, "correct_preference", T1, "t0002", target_pref_id=first["pref_id"],
          value="medium")
    (row,) = history(s, "user")
    assert row["value"] == "medium"
    assert row["corrected_from"] == "mild"
    # A correction rewrites the row: it does not move when it started.
    assert row["valid_from"] == T0.isoformat()
    s.close()


def test_history_is_per_subject():
    s = _ledger()
    _call(s, "add_preference", T0, "t0001", subject="brother",
          attribute="spice_level", value="spicy", scope="permanent")
    assert history(s, "user") == []
    assert len(history(s, "brother")) == 1
    s.close()


# --- the corpus the flags are checked against -------------------------------

def test_the_checked_corpus_is_the_one_config_points_at(corpus):
    trajectories, checkpoints = corpus
    with open(config.CHECKPOINTS_PATH, encoding="utf-8") as fh:
        on_disk = [json.loads(line) for line in fh if line.strip()]
    assert len(checkpoints) == len(on_disk)


# --- two guards the ingredient families lean on -----------------------------
# Step 6's contamination table showed `proposed` keeping a restriction the
# gold profile had dropped ("superseded" 11 vs latest_wins' 8). Tracing the
# eleven rows (2026-10-27) found no resolver bug: every one came from an
# extraction that stored the wrong value on an ingredient turn. What keeps
# that from choosing an unsafe plate is the pair of properties below. Neither
# adds a rule; both are pins, so a later "improvement" cannot quietly remove
# them.

def _ingredients(*calls):
    """A store whose turns are excluded_ingredient statements."""
    s = Store(":memory:").create()
    for i, (ts, name, args) in enumerate(calls, 1):
        s.add_turn(Turn(f"t{i:04d}", i, ts, "user", "said"))
        ok, result = dispatch(s, name, args, ts, f"t{i:04d}")
        assert ok, result
        if i == len(calls):
            s.last = result
    return s


def _exclude(value, **extra):
    return dict(subject="user", attribute="excluded_ingredient",
                value=value, scope="permanent", **extra)


def test_a_revision_does_not_merge_two_standing_exclusions():
    # gold's rule 4 evicts only the named target plus earlier revisions, so
    # "actually aubergine" over {peanut, mushroom} leaves three exclusions.
    # The tool must not invent a single-valued attribute: that would drop a
    # gold restriction, the one error direction that can suggest a banned
    # ingredient. It replaces the newest row and the older one survives.
    s = _ingredients((T0, "add_preference", _exclude("peanut")),
                     (T1, "add_preference", _exclude("mushroom")),
                     (T1, "supersede_preference",
                      dict(subject="user", attribute="excluded_ingredient",
                           value="aubergine")))
    assert s.last["superseded"] == 2          # the newest standing row
    assert active_profile(s, "user", T1) == {
        "excluded_ingredient": {"aubergine", "peanut"}}
    s.close()


def test_the_supersede_tool_ignores_a_named_target():
    # `target_pref_id` is not in the tool's schema. A model that supplies it
    # anyway must replace the same row it would replace without it -- never
    # the row it named -- otherwise one badly-shaped argument could remove
    # either restriction. The newest row (mushroom) is the one that goes in
    # both stores, and peanut survives in both.
    with_target = _ingredients(
        (T0, "add_preference", _exclude("peanut")),
        (T1, "add_preference", _exclude("mushroom")),
        (T1, "supersede_preference", _exclude("aubergine", target_pref_id=1)))
    plain = _ingredients((T0, "add_preference", _exclude("peanut")),
                         (T1, "add_preference", _exclude("mushroom")),
                         (T1, "supersede_preference", _exclude("aubergine")))
    assert with_target.last == plain.last
    assert active_profile(with_target, "user", T1) ==         active_profile(plain, "user", T1) == {
            "excluded_ingredient": {"aubergine", "peanut"}}
    with_target.close()
    plain.close()


def test_a_resolved_answer_removes_its_restriction_from_the_answer_on():
    # The removal leg is what lets `proposed` shed a rule (rule 7); it must
    # cut at the answer, not earlier, so the record stays true for history.
    s = _ingredients((T0, "add_preference", _exclude("peanut")),
                     (T1, "add_preference", _exclude("mushroom")),
                     (T1, "clarify_statement",
                      dict(question="Which one?", resolves={"value": "mushroom"})))
    assert s.last["status"] == "resolved"
    assert active_profile(s, "user", T0) == {"excluded_ingredient": {"peanut"}}
    assert active_profile(s, "user", T1) == {"excluded_ingredient": {"peanut"}}
    s.close()
