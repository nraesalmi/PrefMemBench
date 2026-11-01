"""Step 7's intervals: histories are the unit, and the comparison is paired.

Every test here is arithmetic over synthetic rows. No model call appears, and
none may be added -- the module reads what the evaluator already scored.
"""

from memproj.eval import bootstrap


def row(method="proposed", tid="a", *, exact=None, satisfied=None,
        clarify="none", action="recommend"):
    state = None if exact is None else {"exact": exact}
    return {"method": method, "trajectory_id": tid, "checkpoint_id": f"{tid}x",
            "state": state, "satisfied": satisfied, "clarification": clarify,
            "action": action}


# --- what a row contributes --------------------------------------------------

def test_exact_and_satisfied_skip_the_rows_that_are_not_scored():
    # A free-text method has profile None: its state is undefined, not zero.
    free = per = bootstrap.per_trajectory(
        [row(exact=None), row(exact=False), row(exact=True)], "exact")
    assert free == {  # first row skipped, not counted as a miss
        "a": (1, 2)}
    assert per["a"] == (1, 2)
    assert bootstrap.per_trajectory(
        [row(satisfied=None, exact=True)], "satisfied") == {}


def test_handled_is_satisfied_plate_or_appropriate_clarification():
    rows = [row(satisfied=True),
            row(satisfied=False, clarify="appropriate"),
            row(satisfied=False, clarify="missed"),
            row(satisfied=None, clarify="unnecessary")]
    assert bootstrap.per_trajectory(rows, "handled") == {"a": (2, 4)}


def test_an_unknown_stat_is_an_error_rather_than_a_silent_zero():
    try:
        bootstrap.per_trajectory([row(satisfied=True)], "vibes")
    except ValueError as exc:
        assert "unknown stat" in str(exc)
    else:
        raise AssertionError("expected ValueError")


# --- the interval ------------------------------------------------------------

def test_the_interval_brackets_the_point_estimate():
    units = {"a": (3, 3), "b": (1, 2), "c": (2, 4), "d": (0, 2)}
    low, high = bootstrap.interval(units, resamples=500, seed=1)
    assert low <= 6 / 11 <= high
    assert low < high


def test_a_perfect_method_has_a_degenerate_interval():
    # Every resample is 100%, so the corpus says exactly one thing.
    assert bootstrap.interval({"a": (2, 2), "b": (1, 1)}, 200, 1) == (1.0, 1.0)


def test_the_same_seed_gives_the_same_interval():
    units = {"a": (3, 3), "b": (1, 2), "c": (2, 4)}
    assert (bootstrap.interval(units, 300, 1)
            == bootstrap.interval(units, 300, 1))


def test_percentile_is_nearest_rank():
    assert bootstrap.percentile([0.0, 1.0, 2.0, 3.0, 4.0], 0.5) == 2.0
    assert bootstrap.percentile([1.0], 0.025) == 1.0


# --- the paired delta --------------------------------------------------------

def test_identical_methods_differ_by_zero_on_every_draw():
    units = {"a": (3, 3), "b": (1, 2), "c": (2, 4)}
    assert bootstrap.delta(units, dict(units), 500, 1) == (0.0, 0.0, 0.0)


def test_the_pair_is_read_from_the_same_draw():
    # Both methods score 1 on every row, except that `b` misses history "b".
    a = {"a": (1, 1), "b": (1, 1)}
    b = {"a": (1, 1), "b": (0, 1)}
    point, low, high = bootstrap.delta(a, b, 2000, 1)
    assert point == 0.5
    # The draw resamples DRAW ids, so the difference is 0, 0.5 or 1 with known
    # probabilities -- never the unpaired 0.5 +- noise of two separate draws.
    assert low == 0.0 and high == 1.0


def test_histories_missing_from_one_side_are_dropped():
    a = {"a": (1, 1), "b": (1, 1)}
    b = {"a": (0, 1)}
    point, low, high = bootstrap.delta(a, b, 200, 1)
    assert point == 1.0                 # only "a" is compared
    assert (low, high) == (1.0, 1.0)
    assert bootstrap.delta(a, {"c": (1, 1)}, 10, 1) is None


def test_a_history_with_more_checkpoints_is_still_one_unit():
    # "big" has 4 checkpoints and is worth one draw, not four: that is the
    # whole reason the resampling is over trajectories.
    units = bootstrap.per_trajectory(
        [row(tid="big", satisfied=True) for _ in range(4)]
        + [row(tid="small", satisfied=False)], "satisfied")
    assert units == {"big": (4, 4), "small": (0, 1)}
    low, high = bootstrap.interval(units, 2000, 1)
    assert low < 0.8 <= high          # 4/5 is not treated as four draws of 1


# --- compare and the table ---------------------------------------------------

def test_compare_reports_a_free_text_method_as_undefined_not_zero():
    rows = [row(method="proposed", exact=True, satisfied=True),
            row(method="recent_only", exact=None, satisfied=True)]
    out = {r["method"]: r for r in bootstrap.compare(rows, stat="exact",
                                                     reference=None)}
    assert out["proposed"]["point"] == 1.0
    assert out["recent_only"]["point"] is None
    assert bootstrap.compare([row(exact=None)], stat="exact")[0]["point"] is None


def test_compare_pairs_each_method_against_the_reference():
    rows = [row(method="gold", satisfied=True, tid="a"),
            row(method="gold", satisfied=True, tid="b"),
            row(method="proposed", satisfied=True, tid="a"),
            row(method="proposed", satisfied=False, tid="b")]
    out = {r["method"]: r for r in bootstrap.compare(rows, stat="satisfied")}
    assert out["gold"]["delta"] == 0.0
    assert out["proposed"]["delta"] == -0.5
    assert out["proposed"]["units"] == 2


def test_render_table_prints_one_line_per_method_plus_a_header():
    rows = bootstrap.compare(
        [row(method="gold", satisfied=True), row(method="x", satisfied=False)],
        stat="satisfied")
    lines = bootstrap.render_table(rows, "gold").splitlines()
    assert len(lines) == 2 + len(rows)
    assert lines[0].startswith("| method |")
    assert "| gold | 1 | 1 | 100.0% |" in lines[2]
    assert "n/a" in lines[1] or "|" in lines[1]


# --- the command line -------------------------------------------------------

def test_the_cli_can_score_a_corpus_other_than_the_generated_one(monkeypatch):
    """The challenge run writes its own histories; the intervals must read them.

    `report.evaluate_path` defaults to `config.CHECKPOINTS_PATH` /
    `config.HISTORIES_PATH`, which are the generated corpus. A challenge
    predictions file scored against those paths would fail on every
    checkpoint id it does not know -- or, worse, silently look up nothing.
    """
    from memproj.eval import report

    seen = {}

    def fake_evaluate_path(predictions_path, checkpoints_path=None,
                           histories_path=None, catalogue_path=None):
        seen.update(predictions=str(predictions_path),
                    checkpoints=checkpoints_path,
                    histories=histories_path)
        return [row(method="gold", satisfied=True)]

    monkeypatch.setattr(report, "evaluate_path", fake_evaluate_path)

    assert bootstrap.main(["p.jsonl", "--checkpoints", "c.jsonl",
                           "--histories", "h.jsonl"]) == 0
    assert seen == {"predictions": "p.jsonl", "checkpoints": "c.jsonl",
                    "histories": "h.jsonl"}
