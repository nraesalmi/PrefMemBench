"""Step 7's ablation tool: each resolver rule, off, over a hand-built store.

`memproj/memory/ablate.py` scores one rule at a time against the answer key
without asking a model. These tests pin the two halves that make the table
worth reading: `profile_at()` moves exactly the fact its flag governs, and
`cell()` reports a profile that rejects the whole catalogue as `empty`
instead of hiding it behind a fallback. No Ollama call appears here, and
none may be added.
"""

from datetime import datetime

from memproj.memory import ablate
from memproj.schema import Turn
from memproj.store import Store
from memproj.tools import dispatch

T0 = datetime(2026, 3, 1, 10, 5)
T1 = datetime(2026, 3, 2, 11, 0)
END = datetime(2026, 3, 2, 23, 59)
NOW = datetime(2026, 3, 3, 9, 0)

# Three recipes, one per exclusion below, so a profile can rule them all out.
CATALOGUE = [
    {"id": "r01", "name": "Mild Stew", "vegetarian": True,
     "spice_level": "mild", "ingredients": {"potato"}, "tags": []},
    {"id": "r02", "name": "Hot Curry", "vegetarian": True,
     "spice_level": "spicy", "ingredients": {"chilli"}, "tags": []},
    {"id": "r04", "name": "Roast Bird", "vegetarian": False,
     "spice_level": "medium", "ingredients": {"chicken"}, "tags": []},
]


def checkpoint(gold, *, action="recommend", cid="c0", tid="p001_base"):
    return {"checkpoint_id": cid, "trajectory_id": tid, "person": "user",
            "now": NOW.isoformat(), "gold_profile": gold,
            "gold_action": action}


def mild(**extra):
    return dict(subject="user", attribute="spice_level", value="mild",
                scope="permanent", **extra)


def _store(*calls):
    """An in-memory store whose turns are the statements given, in order."""
    s = Store(":memory:").create()
    for i, (ts, name, args) in enumerate(calls, 1):
        s.add_turn(Turn(f"t{i:04d}", i, ts, "user", "said"))
        ok, result = dispatch(s, name, args, ts, f"t{i:04d}")
        assert ok, result
    return s


# --- which rule is switched off ---------------------------------------------

def test_the_control_row_is_every_rule_on():
    assert ablate.ABLATIONS[0] is None
    assert ablate.ABLATIONS[1:] == ablate.ALL_FLAGS
    assert ablate.label(None) == "all rules on"
    assert ablate.label("use_time") == "no_time"
    assert ablate.label("use_supersede") == "no_supersede"


def test_only_the_named_rule_is_switched_off():
    assert ablate._flags(None) == {name: True for name in ablate.ALL_FLAGS}
    for off in ablate.ALL_FLAGS:
        flags = ablate._flags(off)
        assert [n for n, on in flags.items() if not on] == [off]


# --- profile_at: one flag, one fact -----------------------------------------

def test_an_expired_exception_is_gone_until_the_time_rule_is_off():
    s = _store((T0, "add_preference", mild()),
               (T1, "add_preference", dict(subject="user",
                                           attribute="spice_level",
                                           value="spicy", scope="temporary",
                                           valid_until=END.isoformat())))
    cp = checkpoint({"spice_level": ["mild"]})
    assert ablate.profile_at(s, cp) == {"spice_level": {"mild"}}
    # Without the clock an open exception never closes, so the contaminant
    # comes back. Both scope and time only ever ADD.
    assert ablate.profile_at(s, cp, "use_time") == {
        "spice_level": {"mild", "spicy"}}
    s.close()


def test_the_scope_rule_keeps_a_temporary_value_to_its_window():
    s = _store((T0, "add_preference", mild()),
               (T1, "add_preference", dict(subject="user",
                                           attribute="spice_level",
                                           value="spicy", scope="temporary",
                                           valid_until=END.isoformat())))
    cp = checkpoint({"spice_level": ["mild"]})
    # Off, the temporary value is treated as an ordinary standing one.
    assert ablate.profile_at(s, cp, "use_scope") == {
        "spice_level": {"mild", "spicy"}}
    s.close()


def test_the_supersede_rule_is_what_retires_a_revision():
    s = _store((T0, "add_preference", mild()),
               (T1, "supersede_preference", dict(subject="user",
                                                 attribute="spice_level",
                                                 value="spicy")))
    cp = checkpoint({"spice_level": ["spicy"]})
    assert ablate.profile_at(s, cp) == {"spice_level": {"spicy"}}
    assert ablate.profile_at(s, cp, "use_supersede") == {
        "spice_level": {"mild", "spicy"}}
    s.close()


def test_a_correction_rewrites_its_row_until_the_correction_rule_is_off():
    s = _store((T0, "add_preference", mild()),
               (T1, "correct_preference", dict(target_pref_id=1,
                                               value="medium")))
    cp = checkpoint({"spice_level": ["medium"]})
    assert ablate.profile_at(s, cp) == {"spice_level": {"medium"}}
    # Without the rule the row splits at its restatement instead of being
    # rewritten in place, and the old value survives beside the new one.
    assert ablate.profile_at(s, cp, "use_correction") == {
        "spice_level": {"mild", "medium"}}
    s.close()


def test_the_subject_rule_is_what_keeps_the_brother_out_of_the_profile():
    s = _store((T0, "add_preference", dict(subject="brother",
                                           attribute="diet",
                                           value="vegetarian",
                                           scope="permanent")))
    cp = checkpoint({})
    assert ablate.profile_at(s, cp) == {}
    assert ablate.profile_at(s, cp, "use_subject") == {
        "diet": {"vegetarian"}}
    s.close()


def test_turning_off_an_unrelated_rule_changes_nothing():
    s = _store((T0, "add_preference", mild()))
    cp = checkpoint({"spice_level": ["mild"]})
    for off in ablate.ALL_FLAGS:
        assert ablate.profile_at(s, cp, off) == ablate.profile_at(s, cp)
    s.close()


# --- cell: state, action, and the empty profile ------------------------------

def test_cell_scores_the_profile_it_resolved():
    s = _store((T0, "add_preference", mild()))
    good = ablate.cell(s, checkpoint({"spice_level": ["mild"]}), CATALOGUE)
    assert good["exact"] and good["precision"] == 1.0 and good["recall"] == 1.0
    assert good["missing"] == 0 and good["extra"] == 0
    assert good["satisfied"] is True and good["handled"] is True

    wrong = ablate.cell(s, checkpoint({"spice_level": ["spicy"]}), CATALOGUE)
    assert not wrong["exact"]
    assert wrong["missing"] == 1 and wrong["extra"] == 1
    s.close()


def test_a_profile_that_rejects_the_catalogue_is_counted_not_hidden():
    s = _store(*[(T0, "add_preference",
                  dict(subject="user", attribute="excluded_ingredient",
                       value=value, scope="permanent"))
                 for value in ("potato", "chilli", "chicken")])
    profile = {"excluded_ingredient": ["potato", "chilli", "chicken"]}
    result = ablate.cell(s, checkpoint(profile), CATALOGUE)
    assert result["empty"] is True
    assert result["satisfied"] is None      # no plate was available to judge
    assert result["action"] == "recommend"
    assert result["handled"] is False
    s.close()


def test_a_pending_question_is_the_action_and_only_counts_when_asked_for():
    s = _store((T0, "add_preference", mild()),
               (T1, "ask_clarification", dict(question="Which one?")))
    asks = ablate.cell(s, checkpoint({"spice_level": ["mild"]},
                                     action="clarify"), CATALOGUE)
    assert asks["action"] == "clarify" and asks["satisfied"] is None
    assert asks["appropriate"] is True and asks["handled"] is True

    guess = ablate.cell(s, checkpoint({"spice_level": ["mild"]}), CATALOGUE)
    assert guess["action"] == "clarify"
    assert guess["appropriate"] is False and guess["handled"] is False
    s.close()


# --- summarise / the whole run ----------------------------------------------

def test_summarise_adds_up_the_row():
    s = _store((T0, "add_preference", mild()))
    cells = [ablate.cell(s, checkpoint({"spice_level": ["mild"]}, cid="c0"),
                         CATALOGUE),
             ablate.cell(s, checkpoint({"spice_level": ["spicy"]}, cid="c1"),
                         CATALOGUE)]
    row = ablate.summarise(cells, None)
    assert row["ablation"] == "all rules on" and row["off"] is None
    assert row["checkpoints"] == 2 and row["exact"] == 1
    assert row["recommendable"] == 2 and row["satisfied"] == 2
    assert row["empty"] == 0 and row["appropriate"] == 0
    assert row["handled"] == 2
    s.close()


def test_run_ablations_scores_every_checkpoint_for_every_flag():
    s = _store((T0, "add_preference", mild()))
    asked = []

    def store_for(trajectory_id):
        asked.append(trajectory_id)
        return s

    checkpoints = [checkpoint({"spice_level": ["mild"]}, cid="c0", tid="a"),
                   checkpoint({"spice_level": ["spicy"]}, cid="c1", tid="b")]
    rows = ablate.run_ablations(store_for, checkpoints, CATALOGUE)
    assert [r["off"] for r in rows] == list(ablate.ABLATIONS)
    assert {r["checkpoints"] for r in rows} == {2}
    # Each ablation reads the store for every checkpoint: no checkpoint is
    # skipped because another flag already asked about it.
    assert asked == [t for _off in ablate.ABLATIONS for t in ("a", "b")]
    s.close()


def test_by_family_groups_the_checkpoints_of_one_trajectory():
    cells = [{"trajectory_id": "a", "exact": True, "handled": True},
             {"trajectory_id": "b", "exact": False, "handled": True}]
    rows = ablate.by_family({None: cells}, {"a": "diet_swap",
                                            "b": "temp_vs_perm"})
    assert [r["family"] for r in rows] == ["diet_swap", "temp_vs_perm"]
    assert rows[0] == {"ablation": "all rules on", "family": "diet_swap",
                       "n": 1, "exact": 1, "handled": 1}
    assert rows[1]["exact"] == 0


# --- a cache it cannot ask the model to fill --------------------------------

def test_a_trajectory_with_no_cached_extraction_is_named_not_waited_for(tmp_path):
    (tmp_path / "a.sqlite").write_bytes(b"")
    assert ablate.missing_extractions(str(tmp_path), ["a", "b"]) == ["b"]
    assert ablate.missing_extractions(str(tmp_path), ["a"]) == []


def test_main_refuses_an_incomplete_cache(tmp_path, monkeypatch, capsys):
    # A pipeline run still in progress leaves holes; this table must say so
    # rather than crash inside the extractor with a None client.
    from memproj import config
    monkeypatch.setattr(config, "EXTRACTION_DIR", str(tmp_path))
    assert ablate.main(["--limit", "1"]) == 2
    assert "no cached extraction" in capsys.readouterr().err


# --- rendering ---------------------------------------------------------------

def _row(off, exact=1, handled=1):
    return {"ablation": ablate.label(off), "off": off, "checkpoints": 2,
            "exact": exact, "precision": 0.5, "recall": 0.5, "satisfied": 1,
            "recommendable": 2, "empty": 0, "appropriate": 0,
            "handled": handled}


def test_an_inert_ablation_row_is_explained():
    # A row identical to the control reads as an unimplemented rule unless
    # the table says why it could not move.
    control = _row(None)
    same = _row("use_correction")
    moved = _row("use_time", exact=0, handled=0)
    notes = ablate.idle_notes([control, same, moved], edited=0,
                              trajectories=120)
    assert len(notes) == 1 and "no_correction" in notes[0]
    assert "`edited = 1`" in notes[0] and "120 extractions" in notes[0]
    # With edited rows present the rule had something to date, so the note
    # must not claim the cache was empty.
    notes = ablate.idle_notes([control, same], edited=3, trajectories=120)
    assert len(notes) == 1 and "nothing to date" not in notes[0]


def test_render_table_has_one_row_per_ablation():
    lines = ablate.render_table([_row(off) for off in ablate.ABLATIONS]
                                ).splitlines()
    assert len(lines) == 2 + len(ablate.ABLATIONS)
    assert lines[0].startswith("| ablation |")
    assert lines[1].startswith("|---")
    assert lines[2].startswith("| all rules on | 2 | 1/2 (50.0%) | 50.0% |")


def test_render_family_shows_only_what_the_rule_moved():
    # A family's control cell carries the extraction's own mistakes, so the
    # table must compare against it, not against a perfect cell.
    rows = [
        {"ablation": "all rules on", "family": "diet_swap", "n": 4,
         "exact": 1, "handled": 2},          # control: the first table
        {"ablation": "all rules on", "family": "temp_vs_perm", "n": 4,
         "exact": 3, "handled": 3},
        {"ablation": "no_time", "family": "temp_vs_perm", "n": 4,
         "exact": 3, "handled": 3},          # untouched: nothing to say
        {"ablation": "no_time", "family": "diet_swap", "n": 4,
         "exact": 2, "handled": 2},          # one fact lost: printed
    ]
    lines = ablate.render_family(rows).splitlines()
    assert len(lines) == 3
    assert lines[2] == ("| no_time | diet_swap | 4 | 1/4 -> 2/4 | "
                        "2/4 -> 2/4 |")
    text = chr(10).join(lines)
    assert "temp_vs_perm" not in text
    assert "all rules on" not in text


def test_render_family_skips_a_family_with_no_control_row():
    # Without the control there is nothing to subtract, and an absolute
    # count would read as the rule's doing.
    rows = [{"ablation": "no_time", "family": "diet_swap", "n": 4,
             "exact": 0, "handled": 0}]
    assert len(ablate.render_family(rows).splitlines()) == 2
