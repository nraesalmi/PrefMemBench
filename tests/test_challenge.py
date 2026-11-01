"""Step 7's hand-written challenge set: the authoring file and its build.

The challenge set is the reality check on the template corpus: prose the
generator's pools cannot produce, labelled by hand rather than rendered. The
one rule that keeps it honest is the same one the generated corpus follows --
humans declare the events, Python computes the answer key -- so most of these
tests are about the data contract, not the prose.
"""

import json

import pytest

from memproj import challenge, config, replay


def _event(event_id="e1", **over):
    event = {
        "event_id": event_id,
        "session": 0,
        "ts": "2026-04-01T10:00:00",
        "type": "initial_pref",
        "subject": "user",
        "attribute": "spice_level",
        "value": "mild",
        "scope": "permanent",
        "valid_from": "2026-04-01T10:00:00",
        "valid_until": None,
        "target_event_id": None,
        "needs_clarification": False,
        "referent_event_id": None,
    }
    event.update(over)
    return event


def _turn(turn_id="t0001", **over):
    turn = {
        "turn_id": turn_id,
        "session": 0,
        "ts": "2026-04-01T10:00:00",
        "speaker": "user",
        "text": "That curry last week was more than I can handle.",
        "event_id": "e1",
        "is_implicit": True,
    }
    turn.update(over)
    return turn


def _entry(trajectory_id="c01", **over):
    """One authored history: a heat ceiling stated as a complaint."""
    entry = {
        "trajectory_id": trajectory_id,
        "family": "indirect_spice",
        "topic": "a heat ceiling stated as a complaint",
        "author": "hand-written",
        "reviewed_by": None,
        "review_notes": "",
        "turns": [
            _turn(),
            _turn(turn_id="t0002", ts="2026-04-01T10:01:00",
                  speaker="assistant", text="Something gentler, then.",
                  event_id=None, is_implicit=False),
        ],
        "events": [_event()],
        "checkpoints": [
            {"checkpoint_id": f"{trajectory_id}_c0",
             "now": "2026-04-02T19:00:00", "person": "user",
             "request_text": "What should I cook tonight?",
             "kind": "after_complaint"},
        ],
    }
    entry.update(over)
    return entry


def _two_event_entry():
    """Mild, then a revision to medium: gold must follow the later event."""
    entry = _entry()
    entry["events"].append(
        _event("e2", ts="2026-04-01T11:00:00", type="permanent_revision",
               value="medium", target_event_id="e1"))
    entry["turns"].insert(1, _turn(turn_id="t0002", ts="2026-04-01T11:00:00",
                                   text="Actually, middle of the road is fine.",
                                   event_id="e2", is_implicit=False))
    entry["turns"][2]["turn_id"] = "t0003"
    entry["turns"][2]["ts"] = "2026-04-01T11:01:00"
    return entry


def _write(tmp_path, entries, name="challenge.jsonl"):
    path = tmp_path / name
    path.write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries),
        encoding="utf-8")
    return str(path)


# --- loading into the corpus shapes ----------------------------------------

def test_load_expands_a_history_into_the_corpus_shapes(tmp_path):
    trajectories, checkpoints = challenge.load(_write(tmp_path, [_entry()]))

    assert [t["trajectory_id"] for t in trajectories] == ["c01"]
    trajectory = trajectories[0]
    assert trajectory["family"] == "indirect_spice"
    assert trajectory["pair_id"] == "c01"
    assert trajectory["variant"] == "challenge"
    assert [e["event_id"] for e in trajectory["events"]] == ["e1"]
    assert [t["turn_id"] for t in trajectory["turns"]] == ["t0001", "t0002"]

    checkpoint = checkpoints[0]
    assert checkpoint["trajectory_id"] == "c01"
    assert checkpoint["after_event_id"] == "e1"
    assert checkpoint["gold_profile"] == {"spice_level": ["mild"]}
    assert checkpoint["gold_action"] == "recommend"
    assert checkpoint["relevant_event_ids"] == ["e1"]


def test_the_answer_key_is_computed_from_the_events(tmp_path):
    """A source line carries no gold: what `semantics` says is the answer."""
    entry = _entry()
    entry["checkpoints"][0]["gold_profile"] = {"spice_level": ["spicy"]}
    entry["events"][0]["value"] = "medium"

    _, checkpoints = challenge.load(_write(tmp_path, [entry]))

    assert checkpoints[0]["gold_profile"] == {"spice_level": ["medium"]}


def test_two_checkpoints_of_one_history_both_get_their_own_answer(tmp_path):
    entry = _two_event_entry()
    entry["checkpoints"].append(
        {"checkpoint_id": "c01_c1", "now": "2026-04-01T10:30:00",
         "person": "user", "request_text": "Anything quick?",
         "kind": "before_revision"})

    _, checkpoints = challenge.load(_write(tmp_path, [entry]))

    by_id = {c["checkpoint_id"]: c for c in checkpoints}
    assert by_id["c01_c0"]["gold_profile"] == {"spice_level": ["medium"]}
    assert by_id["c01_c1"]["gold_profile"] == {"spice_level": ["mild"]}


# --- validation -------------------------------------------------------------

def test_validate_accepts_a_sound_history():
    assert challenge.validate([_entry()]) == []


def test_validate_reports_an_event_that_no_turn_expresses():
    entry = _entry()
    entry["events"].append(
        _event("e2", ts="2026-04-01T10:05:00", value="spicy"))

    problems = challenge.validate([entry])

    assert any("e2" in problem and "turn" in problem for problem in problems)


def test_validate_reports_a_turn_that_names_a_missing_event():
    entry = _entry()
    entry["turns"][0]["event_id"] = "e9"

    problems = challenge.validate([entry])

    assert any("e9" in problem for problem in problems)


def test_validate_reports_timestamps_that_go_backwards():
    entry = _entry()
    entry["turns"][1]["ts"] = "2026-04-01T09:00:00"

    problems = challenge.validate([entry])

    assert any("t0002" in problem and "before" in problem
               for problem in problems)


def test_validate_reports_a_window_that_closes_before_it_opens():
    entry = _entry()
    entry["events"][0].update({"type": "temporary_exception",
                               "scope": "temporary",
                               "valid_until": "2026-04-01T09:00:00"})

    problems = challenge.validate([entry])

    assert any("valid_until" in problem for problem in problems)


def test_validate_reports_a_revision_with_no_target():
    entry = _entry()
    entry["events"][0].update({"type": "permanent_revision",
                               "target_event_id": "e7"})

    problems = challenge.validate([entry])

    assert any("e7" in problem for problem in problems)


def test_validate_reports_an_unknown_event_type():
    entry = _entry()
    entry["events"][0]["type"] = "musing"

    problems = challenge.validate([entry])

    assert any("musing" in problem for problem in problems)


def test_validate_reports_duplicate_trajectory_ids():
    problems = challenge.validate([_entry(), _entry()])

    assert any("c01" in problem and "duplicate" in problem
               for problem in problems)


# --- building the two corpus files -----------------------------------------

def test_build_writes_the_two_files_the_runner_reads(tmp_path):
    source = _write(tmp_path, [_entry()])

    histories, checkpoints = challenge.build(source, tmp_path / "built")

    with open(histories, encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh]
    assert lines[0]["trajectory_id"] == "c01"
    assert set(lines[0]) == {"events", "family", "flipped_field", "pair_id",
                             "trajectory_id", "variant", "turns"}
    with open(checkpoints, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    assert rows[0]["gold_action"] == "recommend"
    assert rows[0]["gold_profile"] == {"spice_level": ["mild"]}


def test_build_is_byte_identical_across_runs(tmp_path):
    source = _write(tmp_path, [_entry(), _entry("c02")])

    first = challenge.build(source, tmp_path / "a")
    second = challenge.build(source, tmp_path / "b")

    for a, b in zip(first, second):
        with open(a, "rb") as fa, open(b, "rb") as fb:
            assert fa.read() == fb.read()


# --- cross-labelling --------------------------------------------------------

def test_diff_is_silent_when_the_two_labels_agree():
    assert challenge.diff_labels(_entry(), _entry()) == []


def test_diff_reports_a_disagreement_on_an_event_field():
    reviewer = _entry()
    reviewer["events"][0]["value"] = "medium"

    problems = challenge.diff_labels(_entry(), reviewer)

    assert any("e1" in problem and "value" in problem
               for problem in problems)


def test_diff_reports_a_whole_event_only_one_side_has():
    reviewer = _entry()
    reviewer["events"].append(_event("e2", value="spicy"))
    reviewer["turns"].append(_turn(turn_id="t0003", event_id="e2"))

    problems = challenge.diff_labels(_entry(), reviewer)

    assert any("e2" in problem and "absent" in problem for problem in problems)


def test_diff_reports_a_checkpoint_only_one_side_has():
    reviewer = _entry()
    reviewer["checkpoints"].append(
        {"checkpoint_id": "c01_c1", "now": "2026-04-03T19:00:00",
         "person": "user", "request_text": "And tomorrow?",
         "kind": "after_complaint"})

    problems = challenge.diff_labels(_entry(), reviewer)

    assert any("c01_c1" in problem for problem in problems)


def test_unreviewed_lists_the_histories_still_waiting():
    reviewed = _entry("c02", reviewed_by="second reviewer")

    assert challenge.unreviewed([_entry(), reviewed]) == ["c01"]


def test_the_transcript_hides_the_labels():
    text = challenge.transcript(_entry())

    assert "That curry" in text
    for label in ("spice_level", "mild", "e1", "is_implicit"):
        assert label not in text


# --- the gates --------------------------------------------------------------

def test_replay_reconstructs_gold_from_the_authored_events(tmp_path):
    trajectories, checkpoints = challenge.load(
        _write(tmp_path, [_two_event_entry()]))

    report = replay.run(trajectories, checkpoints)

    assert report["mismatches"] == []
    assert report["checked"] == 1


def test_the_committed_challenge_set_is_sound():
    trajectories, checkpoints = challenge.load()

    assert len(trajectories) >= 15, "the challenge set is 15-20 histories"
    assert len(checkpoints) >= 30, "2-3 checkpoints per history"
    assert challenge.validate(challenge.read()) == []
    assert replay.run(trajectories, checkpoints)["mismatches"] == []


def test_the_committed_challenge_histories_are_individually_labelled():
    for entry in challenge.read():
        assert entry["author"], entry["trajectory_id"]
        assert entry["topic"], entry["trajectory_id"]


def test_strict_mode_fails_while_a_history_awaits_cross_labelling(tmp_path):
    source = _write(tmp_path, [_entry()])

    assert challenge.main(["--source", source, "--strict",
                           "--out", str(tmp_path / "built")]) != 0


def test_the_runner_can_target_a_separate_extraction_directory(monkeypatch):
    seen = {}

    def fake_run(methods=None, **kwargs):
        seen.update(kwargs)
        return [], {"rows": 0, "checkpoints": 0, "methods": methods,
                    "path": None, "journal": None, "written": 0,
                    "resumed": 0, "stale_journal_rows": 0,
                    "rows_by_method": {}, "headroom": {}, "catalogue_size": 0,
                    "state_mismatches": [], "extraction": None}

    from memproj.pipeline import run as runner
    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(runner, "validate_predictions", lambda *a, **k: [])

    runner.main(["--methods", "gold", "--extraction-dir", "out/x"])

    assert seen["extraction_dir"] == "out/x"


def test_the_challenge_path_is_where_the_module_looks():
    assert config.CHALLENGE_PATH.name == "challenge.jsonl"
