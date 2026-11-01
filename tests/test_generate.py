"""End-to-end tests: determinism, matched pairs, coverage, no leakage."""

import json
import os

import pytest

from memproj import config, families, recipes, templates
from memproj.generate import build_pair, generate, write_output
from memproj.semantics import gold_action, gold_profile

# The shipped catalogue, not a fixture: coverage (>=3 valid recipes per
# checkpoint profile) is a property of this file, and these tests are the
# only place it is checked against a real generation run.
CATALOGUE = str(config.CATALOGUE_PATH)

@pytest.fixture(scope="module")
def run_small():
    return generate(n_base=12, seed=7, n_sessions=3, turns_per_session=8,
                    catalogue_path=CATALOGUE)


@pytest.fixture(scope="module")
def run_small_again():
    return generate(n_base=12, seed=7, n_sessions=3, turns_per_session=8,
                    catalogue_path=CATALOGUE)


# --- determinism ------------------------------------------------------------

def test_same_seed_gives_identical_trajectories(run_small, run_small_again):
    a, _ = run_small
    b, _ = run_small_again
    assert [t.trajectory_id for t in a] == [t.trajectory_id for t in b]
    assert [t.turns for t in a] == [t.turns for t in b]
    assert [t.events for t in a] == [t.events for t in b]


def test_same_seed_gives_identical_checkpoints(run_small, run_small_again):
    _, a = run_small
    _, b = run_small_again
    assert a == b


def test_same_seed_gives_byte_identical_files(tmp_path):
    d1, d2 = tmp_path / "a", tmp_path / "b"
    for d in (d1, d2):
        traj, cps = generate(6, 3, 3, 8, CATALOGUE)
        write_output(str(d), traj, cps)
    for name in ("histories.jsonl", "checkpoints.jsonl"):
        assert (d1 / name).read_bytes() == (d2 / name).read_bytes()


def test_different_seed_changes_output(tmp_path):
    traj_a, _ = generate(6, 1, 3, 8, CATALOGUE)
    traj_b, _ = generate(6, 2, 3, 8, CATALOGUE)
    assert [t.turns for t in traj_a] != [t.turns for t in traj_b]


# --- shape ------------------------------------------------------------------

def test_every_base_has_exactly_one_matched_twin(run_small):
    traj, _ = run_small
    pairs = {}
    for t in traj:
        pairs.setdefault(t.pair_id, []).append(t)
    assert pairs, "expected some pairs"
    for pid, members in pairs.items():
        assert len(members) == 2, pid
        assert {m.variant for m in members} == {"base", "matched"}


def test_every_family_appears(run_small):
    traj, _ = run_small
    assert {t.family for t in traj} == set(families.FAMILIES)


def test_trajectory_ids_are_unique(run_small):
    traj, _ = run_small
    ids = [t.trajectory_id for t in traj]
    assert len(ids) == len(set(ids))


def test_checkpoints_reference_real_trajectories(run_small):
    traj, cps = run_small
    ids = {t.trajectory_id for t in traj}
    for c in cps:
        assert c.trajectory_id in ids


def test_each_trajectory_has_at_least_one_checkpoint(run_small):
    traj, cps = run_small
    covered = {c.trajectory_id for c in cps}
    for t in traj:
        assert t.trajectory_id in covered


# --- matched pairs ----------------------------------------------------------

def _pairs(traj):
    out = {}
    for t in traj:
        out.setdefault(t.pair_id, {})[t.variant] = t
    return out


# Families differ by one event, but "differ" can mean any of:
#   flip    -- same event_id, a different field value (most families)
#   remove  -- present in base, absent in matched (correction_vs_none)
#   add     -- absent in base, present in matched (hypothetical_then_real)
#
# All three are legitimate: the invariant we actually need is that the twins
# are identical EXCEPT for the one declared detail, whatever mechanism
# expresses it. Asserting a single mechanism would test the implementation
# choice rather than the experimental property.

def _twin_delta(pair):
    """(kind, base_event, matched_event) for the one differing event.

    `kind` is "flip", "remove", or "add".
    """
    b = {e.event_id: e for e in pair["base"].events}
    m = {e.event_id: e for e in pair["matched"].events}
    only_base = set(b) - set(m)
    only_matched = set(m) - set(b)
    differing = [k for k in set(b) & set(m) if b[k] != m[k]]

    assert len(only_base) + len(only_matched) + len(differing) == 1, (
        f"{pair['base'].pair_id}: expected exactly one differing event, got "
        f"only_base={sorted(only_base)} only_matched={sorted(only_matched)} "
        f"differing={sorted(differing)}"
    )
    if differing:
        return "flip", b[differing[0]], m[differing[0]]
    if only_base:
        return "remove", b[sorted(only_base)[0]], None
    return "add", None, m[sorted(only_matched)[0]]


def test_matched_twin_differs_in_exactly_one_event(run_small):
    """Twin pairs must be identical apart from the declared flip."""
    traj, _ = run_small
    for pid, pair in _pairs(traj).items():
        _twin_delta(pair)  # the helper asserts the count itself


def test_every_flip_is_expressed_by_the_declared_field(run_small):
    """The declared `flipped_field` must be among the fields that changed.

    It may not be the ONLY change: some flips are causal. Declaring `scope`
    necessarily drags `type` and `valid_until` along, because a
    permanent_revision is not the same event as a temporary_exception. What
    must never happen is a declared field that did not move at all.
    """
    traj, _ = run_small
    for pid, pair in _pairs(traj).items():
        field = pair["base"].flipped_field
        assert pair["matched"].flipped_field == field, (
            f"{pid}: twins declare different flipped_field"
        )
        assert field in _EVENT_FIELDS, f"{pid}: unknown field {field!r}"
        kind, b_event, m_event = _twin_delta(pair)
        if kind == "flip":
            changed = {f for f in _EVENT_FIELDS
                       if getattr(b_event, f) != getattr(m_event, f)}
            assert field in changed, (
                f"{pid}: declared flip {field!r} did not change; "
                f"changed fields were {sorted(changed)}"
            )
            assert b_event.event_id == m_event.event_id
        else:
            # An added or removed event has nothing to compare field-wise;
            # the declaration still has to name a real Event field.
            assert kind in ("add", "remove"), kind


def test_every_family_declares_a_flip_somewhere(run_small):
    """No family may slip through with an empty or missing declaration."""
    traj, _ = run_small
    families_seen = {}
    for pid, pair in _pairs(traj).items():
        fam = pair["base"].family
        kind, _, _ = _twin_delta(pair)
        families_seen.setdefault(fam, set()).add(kind)
    assert set(families_seen) == set(families.FAMILIES), (
        "run_small did not cover every family"
    )
    for fam, kinds in families_seen.items():
        assert len(kinds) == 1, f"{fam} expressed its flip inconsistently: {kinds}"


_EVENT_FIELDS = (
    "session", "ts", "type", "subject", "attribute", "value", "scope",
    "valid_from", "valid_until", "target_event_id", "needs_clarification",
    "referent_event_id",
)


def _profiles_for(pair):
    """Gold profile at each checkpoint, keyed by checkpoint kind position."""
    return (pair["base"].events, pair["matched"].events)


def test_gold_state_differs_between_twins_for_every_family(run_small):
    """The flip must change the gold answer for at least one aligned question.

    Checkpoints are matched by (person, question kind), not by position: the
    twins can answer a different number of questions. For
    exception_then_restated the flip lives in the question's timing, not in
    a field of the events, so the pair differs without the events differing.
    """
    traj, cps = run_small
    for pid, pair in _pairs(traj).items():
        fam = pair["base"].family
        base_cps = [c for c in cps if c.trajectory_id == pair["base"].trajectory_id]
        matched_cps = [c for c in cps
                       if c.trajectory_id == pair["matched"].trajectory_id]
        assert base_cps and matched_cps, pid
        # Align on (person, kind) -- the same question, twice. Event ids
        # cannot be used: correction_vs_none anchors on a different event in
        # each variant.
        b_by_key = {(c.person, c.kind): c for c in base_cps}
        m_by_key = {(c.person, c.kind): c for c in matched_cps}
        shared = set(b_by_key) & set(m_by_key)
        assert shared, f"{pid} ({fam}): twins share no comparable question"
        differs = any(
            (b_by_key[k].gold_profile, b_by_key[k].gold_action)
            != (m_by_key[k].gold_profile, m_by_key[k].gold_action)
            for k in shared
        )
        assert differs, (
            f"{fam}: matched variant produced no behavioural difference on "
            f"{sorted(shared)}"
        )


def test_temp_vs_perm_expires_but_permanent_does_not(run_small):
    traj, cps = run_small
    for pair in _pairs(traj).values():
        if pair["base"].family != "temp_vs_perm":
            continue
        base, matched = pair["base"], pair["matched"]
        # At the last checkpoint the exception is over: base loses spicy.
        b_last = [c for c in cps if c.trajectory_id == base.trajectory_id][-1]
        m_last = [c for c in cps if c.trajectory_id == matched.trajectory_id][-1]
        assert "spicy" not in b_last.gold_profile.get("spice_level", [])
        assert "spicy" in m_last.gold_profile.get("spice_level", [])


def test_ambiguous_base_clarifies_and_matched_does_not(run_small):
    traj, cps = run_small
    for pair in _pairs(traj).values():
        if pair["base"].family != "ambiguous_vs_resolved":
            continue
        b = [c for c in cps if c.trajectory_id == pair["base"].trajectory_id][0]
        m = [c for c in cps if c.trajectory_id == pair["matched"].trajectory_id][0]
        assert b.gold_action == "clarify"
        assert m.gold_action == "recommend"


def test_subject_swap_keeps_user_clean(run_small):
    traj, cps = run_small
    for pair in _pairs(traj).values():
        if pair["base"].family != "subject_swap":
            continue
        for c in cps:
            if c.trajectory_id != pair["matched"].trajectory_id:
                continue
            if c.person != "user":
                continue
            assert "spicy" not in c.gold_profile.get("spice_level", [])


# --- composition families ---------------------------------------------------
# Each of these asserts the SPECIFIC behavioural difference the family was
# built to create. Without them the new families only prove that the twins
# differ somehow, which the general tests already cover.

def _cps_for(traj, cps, pair, variant):
    tid = pair[variant].trajectory_id
    return {c.kind: c for c in cps if c.trajectory_id == tid}


def _only_pair(traj, family):
    for pair in _pairs(traj).values():
        if pair["base"].family == family:
            return pair
    raise AssertionError(f"no pair of family {family} in this run")


def test_correction_of_temporary_keeps_the_window(run_small):
    """Rule 5: correcting a live windowed record must not extend its life.

    Inside the window the corrected value applies. Once the window closes it
    must be gone -- and the permanent record corrected in the twin must NOT
    be gone, which is what makes the target field load-bearing.
    """
    traj, cps = run_small
    pair = _only_pair(traj, "correction_of_temporary")
    base = _cps_for(traj, cps, pair, "base")
    matched = _cps_for(traj, cps, pair, "matched")

    # Before the correction both twins agree entirely.
    assert base["inside_window"].gold_profile == matched["inside_window"].gold_profile

    # After it, they differ: the corrected value sits on a different record.
    assert (base["window_corrected"].gold_profile
            != matched["window_corrected"].gold_profile)

    # The corrected ingredient is LIVE in the base after the correction...
    assert "aubergine" in base["window_corrected"].gold_profile.get(
        "excluded_ingredient", [])
    # ...and DEAD once the window closes, while the twin keeps it forever.
    assert "aubergine" not in base["window_closed"].gold_profile.get(
        "excluded_ingredient", [])
    assert "aubergine" in matched["window_closed"].gold_profile.get(
        "excluded_ingredient", [])


def test_exception_then_restated_does_not_resurrect_the_old_window(run_small):
    """A re-statement is a new record, not an extension of the expired one."""
    traj, cps = run_small
    pair = _only_pair(traj, "exception_then_restated")
    base = _cps_for(traj, cps, pair, "base")
    matched = _cps_for(traj, cps, pair, "matched")

    # The original window is open for both: the twins agree where they should.
    assert (base["first_window_open"].gold_profile
            == matched["first_window_open"].gold_profile)
    assert "spicy" in base["first_window_open"].gold_profile["spice_level"]

    # The re-statement, asked at the same instant of both twins: the base's
    # fresh window is open, the matched's names tonight's and grants nothing.
    assert "spicy" in base["after_restate"].gold_profile["spice_level"]
    assert "spicy" not in matched["after_restate"].gold_profile["spice_level"]
    assert base["after_restate"].now == matched["after_restate"].now


def test_a_named_statement_cannot_ask_for_a_permanent_revision(run_small):
    """Rule 4 removes every earlier permanent_revision of the attribute, so a
    revision that names ANY earlier event erases that event's effect too.

    Two families name a record with a permanent_revision and both name an
    initial_pref, which leave no earlier revision behind them. A revision
    naming a revision (a second revision, a revision of another person's
    preference) would lose the first one's effect as well as make the
    change it names -- a second behaviour wrapped inside the one the family
    declares. If a future family needs that shape, rule 4's "latest one
    wins outright" clause has to be decided first.
    """
    traj, _ = run_small
    for pid, pair in _pairs(traj).items():
        for variant in ("base", "matched"):
            events = getattr(pair[variant], "events")
            by_id = {e.event_id: e for e in events}
            for e in events:
                if e.type != "permanent_revision" or not e.target_event_id:
                    continue
                target = by_id[e.target_event_id]
                assert target.type == "initial_pref", (
                    f"{pid}/{variant}/{e.event_id} revises {target.event_id} "
                    f"({target.type}); rule 4 would erase that record's "
                    f"effect as well as the one being revised")


def test_ambiguous_then_clarify_asks_again_when_the_answer_is_vague(run_small):
    """Both twins must request clarification; only the answer differs.

    Gold treats EVERY vague statement as a fresh question, not just the
    first. That is the reading the matched twin is built around: its answer
    names nothing, so the user is asked again (gold_action == clarify) and
    the restriction stays pending. A family whose matched answer was meant
    to close the question would contradict gold_action, and the replay
    would flag it rather than hide it.
    """
    traj, cps = run_small
    pair = _only_pair(traj, "ambiguous_then_clarify")
    base = _cps_for(traj, cps, pair, "base")
    matched = _cps_for(traj, cps, pair, "matched")

    # Same obligation before the user answers -- this is the point of the
    # family: an agent that never clarifies fails visibly HERE.
    assert base["pending_ambiguity"].gold_action == "clarify"
    assert matched["pending_ambiguity"].gold_action == "clarify"
    assert (base["pending_ambiguity"].gold_profile
            == matched["pending_ambiguity"].gold_profile)

    # Base: the answer names the referent, so a restriction is dropped and
    # nothing is left to ask about.
    assert base["after_answer"].gold_action == "recommend"
    assert "mushroom" not in base["after_answer"].gold_profile.get(
        "excluded_ingredient", [])
    # Matched: the answer names nothing, so both restrictions stand and the
    # vague answer re-raises the question.
    assert "mushroom" in matched["after_answer"].gold_profile.get(
        "excluded_ingredient", [])
    assert matched["after_answer"].gold_action == "clarify"


def test_hypothetical_then_real_stays_inert_until_it_is_real(run_small):
    """A hypothetical must not become true when a matching real fact lands."""
    traj, cps = run_small
    pair = _only_pair(traj, "hypothetical_then_real")
    base = _cps_for(traj, cps, pair, "base")
    matched = _cps_for(traj, cps, pair, "matched")

    # While it is only hypothetical, both twins must be at mild.
    assert (base["after_hypothetical"].gold_profile
            == matched["after_hypothetical"].gold_profile)
    assert base["after_hypothetical"].gold_profile["spice_level"] == ["mild"]

    # Afterwards the base is STILL mild -- the hypothetical never took.
    assert base["after_real_statement"].gold_profile["spice_level"] == ["mild"]
    assert matched["after_real_statement"].gold_profile["spice_level"] == ["spicy"]


# --- recipe coverage --------------------------------------------------------

def test_every_checkpoint_has_three_valid_recipes(run_small):
    catalogue = recipes.load_derived(CATALOGUE)
    _, cps = run_small
    for c in cps:
        profile = {k: set(v) for k, v in c.gold_profile.items()}
        ok = recipes.valid_recipes(catalogue, profile)
        assert len(ok) >= 3, f"{c.checkpoint_id}: only {len(ok)} recipes for {profile}"


def test_gold_profile_is_recomputable_from_events(run_small):
    """The stored profile must equal what semantics computes, not a copy."""
    traj, cps = run_small
    by_id = {t.trajectory_id: t for t in traj}
    for c in cps:
        events = by_id[c.trajectory_id].events
        recomputed = gold_profile(events, c.person, c.now)
        assert c.gold_profile == {k: sorted(v) for k, v in recomputed.items()}
        assert c.gold_action == gold_action(events, c.person, c.now)


# --- explicitness flag ------------------------------------------------------

def test_implicit_flag_is_recorded_on_both_surface_pools(run_small):
    """The flag must actually distinguish the two pools, not sit at False.

    render() picks the IMPLICIT pool ~65% of the time, so a corpus where no
    turn is ever marked implicit means the value never reached the Turn.
    """
    traj, _ = run_small
    marked = [t.is_implicit for tr in traj for t in tr.turns if t.speaker == "user"]
    assert any(marked), "no user turn was ever marked implicit"
    assert not all(marked), "every user turn was marked implicit"


def test_filler_and_assistant_turns_are_never_implicit(run_small):
    """Only event-bearing user turns can imply a preference; chatter cannot."""
    traj, _ = run_small
    for tr in traj:
        for turn in tr.turns:
            if turn.speaker == "assistant" or turn.event_id is None:
                assert turn.is_implicit is False, (
                    f"{tr.trajectory_id} {turn.turn_id}: {turn.text!r}"
                )


def test_implicit_flag_survives_serialisation(tmp_path, run_small):
    """A flag only in memory cannot drive a later ablation."""
    traj, cps = run_small
    write_output(str(tmp_path), traj, cps)
    lines = (tmp_path / "histories.jsonl").read_text(encoding="utf-8").splitlines()
    flags = [turn["is_implicit"] for line in lines
             for turn in json.loads(line)["turns"]]
    assert any(flags), "is_implicit never true after a round trip"
    assert not all(flags), "is_implicit always true after a round trip"


# --- timestamps -------------------------------------------------------------

def test_turn_timestamps_increase_within_a_session(run_small):
    traj, _ = run_small
    for t in traj:
        last = {}
        for turn in t.turns:
            if turn.session in last:
                assert turn.ts > last[turn.session], (
                    f"{t.trajectory_id}: {turn.turn_id} not after previous"
                )
            last[turn.session] = turn.ts


def test_sessions_are_separated_by_gaps(run_small):
    traj, _ = run_small
    for t in traj:
        starts = {}
        for turn in t.turns:
            starts.setdefault(turn.session, turn.ts)
        ordered = [starts[s] for s in sorted(starts)]
        for a, b in zip(ordered, ordered[1:]):
            assert (b - a).total_seconds() >= 3600, "sessions must be hours apart"


def test_valid_until_is_never_before_valid_from(run_small):
    traj, _ = run_small
    for t in traj:
        for e in t.events:
            if e.valid_until is not None:
                assert e.valid_until >= e.valid_from, e.event_id


def test_event_timestamps_are_ordered_within_a_session(run_small):
    traj, _ = run_small
    for t in traj:
        per_session = {}
        for e in t.events:
            per_session.setdefault(e.session, []).append(e.ts)
        for s, times in per_session.items():
            assert times == sorted(times), f"{t.trajectory_id} session {s}"


# --- no leakage -------------------------------------------------------------

def _all_preference_values(run):
    traj, _ = run
    values = set()
    for t in traj:
        for e in t.events:
            if e.value:
                values.add(e.value.lower())
    return values


def test_no_assistant_turn_contains_a_preference_value(run_small):
    traj, _ = run_small
    values = _all_preference_values(run_small)
    for t in traj:
        for turn in t.turns:
            if turn.speaker != "assistant":
                continue
            low = turn.text.lower()
            for v in values:
                assert v not in low, (
                    f"{t.trajectory_id} {turn.turn_id}: assistant leaked {v!r}"
                )


def test_assistant_turns_never_carry_an_event_id(run_small):
    traj, _ = run_small
    for t in traj:
        for turn in t.turns:
            if turn.speaker == "assistant":
                assert turn.event_id is None


def test_other_person_never_appears_in_user_gold_profile(run_small):
    traj, cps = run_small
    by_id = {t.trajectory_id: t for t in traj}
    for c in cps:
        if c.person != "user":
            continue
        events = by_id[c.trajectory_id].events
        others = {e.subject for e in events
                  if e.type == "other_person" and e.ts <= c.now}
        if not others:
            continue
        # Any value only ever stated about someone else must not be in the
        # user's profile unless the user stated it separately.
        user_values = {e.value for e in events
                       if e.subject == "user" and e.value is not None}
        for attr, vals in c.gold_profile.items():
            for v in vals:
                assert v in user_values or v not in _other_only_values(events), (
                    f"{c.checkpoint_id}: other person's {v!r} leaked into user"
                )


def _other_only_values(events):
    """Values stated for a non-user subject and never for the user."""
    other = {}
    user = set()
    for e in events:
        if e.value is None:
            continue
        if e.subject == "user":
            user.add(e.value)
        else:
            other[e.value] = True
    return {v for v in other if v not in user}
