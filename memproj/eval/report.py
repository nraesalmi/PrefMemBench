"""Load a predictions file, score it, and render the tables.

    python -m memproj.eval.report out/predictions.jsonl
    python -m memproj.eval.report out/predictions.jsonl --markdown out/report.md

Three tables, in the order the argument needs them:

1. **per method** -- state and action side by side. If a method is better at
   remembering than at acting, this is where it shows.
2. **dissociation** -- the 2x2 of (state exact, recommendation satisfied),
   still only over rows that made a recommendation. The off-diagonal cells
   are the finding: `recalled but violated` is an agent that knew the
   preference and ignored it.
3. **paired** -- base vs matched, per family, aligned on
   `(family, pair_id, person, kind)`. Families whose flip only matters to
   one twin's question set (`subject_swap`'s `for_other_person`) leave an
   unpaired kind; those pairs are counted as skipped, not as zero.

Nothing here calls a model or imports a memory method.
"""

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from memproj import config
from memproj import recipes
from memproj.eval import metrics


# --- loading ----------------------------------------------------------------

def read_jsonl(path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_inputs(predictions_path, checkpoints_path=None, histories_path=None,
                catalogue_path=None):
    """Everything the evaluator is allowed to read (README, Evaluator boundary)."""
    return {
        "predictions": read_jsonl(predictions_path),
        "checkpoints": read_jsonl(checkpoints_path or config.CHECKPOINTS_PATH),
        "histories": read_jsonl(histories_path or config.HISTORIES_PATH),
        "catalogue": recipes.load_derived(str(catalogue_path
                                              or config.CATALOGUE_PATH)),
    }


# --- scoring ----------------------------------------------------------------

def evaluate(predictions: Sequence[dict], checkpoints: Sequence[dict],
             histories: Sequence[dict], catalogue: Sequence[dict]) -> List[dict]:
    """Score every predictions row against its checkpoint and trajectory."""
    checkpoints_by_id = {checkpoint["checkpoint_id"]: checkpoint
                         for checkpoint in checkpoints}
    trajectories_by_id = {trajectory["trajectory_id"]: trajectory
                          for trajectory in histories}

    scored = []
    for prediction in predictions:
        checkpoint_id = prediction["checkpoint_id"]
        if checkpoint_id not in checkpoints_by_id:
            raise KeyError(f"{checkpoint_id!r} is not in checkpoints")
        checkpoint = checkpoints_by_id[checkpoint_id]
        # A row that names another trajectory's checkpoint would otherwise be
        # scored against the wrong history -- and the citation check would
        # pass, because turn ids restart in every trajectory.
        if checkpoint["trajectory_id"] != prediction["trajectory_id"]:
            raise ValueError(
                f"{checkpoint_id} belongs to {checkpoint['trajectory_id']}, "
                f"not {prediction['trajectory_id']}"
            )
        trajectory = trajectories_by_id[prediction["trajectory_id"]]

        # None means "this method has no structured opinion" -- undefined,
        # not wrong. Only a dict gets scored.
        state = None
        if prediction["profile"] is not None:
            state = metrics.score(prediction["profile"],
                                  checkpoint["gold_profile"])

        satisfied = metrics.satisfaction(prediction, checkpoint, catalogue)
        scored.append({
            "trajectory_id": prediction["trajectory_id"],
            "checkpoint_id": checkpoint_id,
            "method": prediction["method"],
            "model": prediction["model"],
            "pair_id": trajectory["pair_id"],
            "family": trajectory["family"],
            "variant": trajectory["variant"],
            "person": checkpoint["person"],
            "kind": checkpoint["kind"],
            "gold_action": checkpoint["gold_action"],
            "action": prediction["action"],
            "state": state,
            "satisfied": satisfied["satisfied"],
            "violations": satisfied["violations"],
            "clarification": metrics.clarification_outcome(prediction, checkpoint),
            "contamination": metrics.contamination(prediction, checkpoint,
                                                   trajectory),
            "provenance": metrics.provenance(prediction, checkpoint, trajectory),
            "tokens": prediction["tokens"],
            "latency_s": prediction["latency_s"],
        })
    return scored


def evaluate_path(predictions_path, checkpoints_path=None, histories_path=None,
                  catalogue_path=None) -> List[dict]:
    return evaluate(**load_inputs(predictions_path, checkpoints_path,
                                  histories_path, catalogue_path))


# --- aggregation ------------------------------------------------------------

def _mean(values: Sequence[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def _labels(rows: Sequence[dict]) -> Dict[tuple, str]:
    """`(method, model)` -> the name the tables print.

    A normal run writes one model per method, and the label is then the
    method alone. When a file holds two models for one method -- `--resume`
    after a model change, or two runs concatenated -- the rows are kept
    apart and the model is printed instead of being silently merged or
    silently dropped by taking the first row that matched.
    """
    models: Dict[str, set] = defaultdict(set)
    for row in rows:
        models[row["method"]].add(row["model"])
    # `gold` records itself as its model; that label stays plain so the
    # ceiling reads the same in every table.
    return {(method, model): (method if len(names) == 1 or model == method
                             else f"{method} [{model}]")
            for method, names in models.items()
            for model in names}


def summarise(rows: Sequence[dict]) -> Dict[str, dict]:
    """One summary dict per method (per method+model if a file mixes them)."""
    labels = _labels(rows)
    groups: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        groups[labels[(row["method"], row["model"])]].append(row)

    summary = {}
    for method, group in sorted(groups.items()):
        structured = [row for row in group if row["state"] is not None]
        recommended = [row for row in group if row["satisfied"] is not None]
        clarify = defaultdict(int)
        for row in group:
            clarify[row["clarification"]] += 1
        sources = defaultdict(int)
        for row in group:
            for finding in row["contamination"]:
                sources[finding["source"]] += 1
        latencies = [row["latency_s"] for row in group
                     if row["latency_s"] is not None]

        summary[method] = {
            "models": sorted({row["model"] for row in group}),
            "rows": len(group),
            "state_rows": len(structured),
            "state_exact": sum(1 for row in structured
                               if row["state"]["exact"]),
            "state_precision": _mean([row["state"]["precision"]
                                      for row in structured]),
            "state_recall": _mean([row["state"]["recall"]
                                   for row in structured]),
            "recommend_rows": len(recommended),
            "satisfied": sum(1 for row in recommended if row["satisfied"]),
            "violations": sum(len(row["violations"]) for row in recommended),
            "clarify": dict(clarify),
            "contamination": sum(sources.values()),
            "contamination_by_source": dict(sources),
            "provenance_rows": sum(1 for row in group if row["provenance"]),
            "provenance_problems": sum(len(row["provenance"]) for row in group),
            "tokens_mean": _mean([row["tokens"] for row in group]),
            "latency_mean": _mean(latencies),
        }
    return summary


def dissociation(rows: Sequence[dict]) -> Dict[str, dict]:
    """(state exact, satisfied) over rows that made both a claim and a pick.

    The four cells:
        both            recalled the state and acted on it
        recalled_but_violated   knew the preference, recommended against it
        lucky           wrong state, pick still passed the answer key
        both_wrong      wrong state and a pick that breaks it

    Only structured methods appear: a free-text row has no state, so there is
    no 2x2 to put it in.

    `recalled_but_violated` should be near zero when the recommender filters
    the candidate set on a resolved profile (the candidate policy above); it is the
    cell to watch if the candidate policy ever changes.
    """
    labels = _labels(rows)
    cells: Dict[str, Dict[str, int]] = {}
    for row in rows:
        if row["state"] is None or row["satisfied"] is None:
            continue
        method = labels[(row["method"], row["model"])]
        exact, ok = row["state"]["exact"], row["satisfied"]
        bucket = ("both" if exact and ok else
                  "recalled_but_violated" if exact else
                  "lucky" if ok else "both_wrong")
        cells.setdefault(method, {}).setdefault(bucket, 0)
        cells[method][bucket] += 1
    return {method: {name: counts.get(name, 0)
                     for name in ("both", "recalled_but_violated",
                                  "lucky", "both_wrong")}
            for method, counts in sorted(cells.items())}


def paired_table(rows: Sequence[dict]) -> List[dict]:
    """Base/matched twin pairs, one entry per aligned checkpoint.

    A checkpoint whose `(family, pair_id, person, kind)` has no counterpart
    in the other variant is dropped: an unpaired kind means "this variant has
    no comparable question here", not "this variant failed".
    """
    labels = _labels(rows)
    groups: Dict[tuple, Dict[str, List[dict]]] = defaultdict(
        lambda: {"base": [], "matched": []})
    for row in rows:
        if row["variant"] not in ("base", "matched"):
            # A corpus with no twins (the hand-written challenge set, variant
            # "challenge") has nothing to pair; it still gets every other
            # table. Pairing is for matched variants, not a row requirement.
            continue
        key = (labels[(row["method"], row["model"])],) + metrics.pair_key(row)
        groups[key][row["variant"]].append(row)

    pairs = []
    for (method, family, pair_id, person, kind), sides in sorted(groups.items()):
        if not sides["base"] or not sides["matched"]:
            # An unpaired kind means this variant has no comparable question
            # here, not that this variant failed.
            continue
        for side in ("base", "matched"):
            if len(sides[side]) > 1:
                raise ValueError(
                    f"{method}/{family}/{pair_id}/{kind}: the {side} variant "
                    f"answered this question {len(sides[side])} times; pairing "
                    f"one of them silently would hide the duplicate"
                )
        base, matched = sides["base"][0], sides["matched"][0]
        pairs.append({
            "method": method,
            "family": family,
            "pair_id": pair_id,
            "person": person,
            "kind": kind,
            "base_checkpoint": base["checkpoint_id"],
            "matched_checkpoint": matched["checkpoint_id"],
            "base_state": base["state"]["exact"] if base["state"] else None,
            "matched_state": (matched["state"]["exact"]
                              if matched["state"] else None),
            "base_satisfied": base["satisfied"],
            "matched_satisfied": matched["satisfied"],
        })
    return pairs


def paired_summary(rows: Sequence[dict]) -> List[dict]:
    """`paired_table` aggregated per (method, family)."""
    grouped: Dict[tuple, List[dict]] = defaultdict(list)
    for pair in paired_table(rows):
        grouped[(pair["method"], pair["family"])].append(pair)

    summary = []
    for (method, family), pairs in sorted(grouped.items()):
        state_pairs = [pair for pair in pairs
                       if pair["base_state"] is not None
                       and pair["matched_state"] is not None]
        sat_pairs = [pair for pair in pairs
                     if pair["base_satisfied"] is not None
                     and pair["matched_satisfied"] is not None]

        def rate(items, side, field):
            if not items:
                return None
            return (sum(1 for item in items if item[side]) / len(items))

        base_state = rate(state_pairs, "base_state", "base_state")
        matched_state = rate(state_pairs, "matched_state", "matched_state")
        base_sat = rate(sat_pairs, "base_satisfied", "base_satisfied")
        matched_sat = rate(sat_pairs, "matched_satisfied", "matched_satisfied")
        summary.append({
            "method": method,
            "family": family,
            "pairs": len(pairs),
            "state_pairs": len(state_pairs),
            "base_state": base_state,
            "matched_state": matched_state,
            "state_delta": (None if base_state is None
                            else matched_state - base_state),
            "sat_pairs": len(sat_pairs),
            "base_satisfied": base_sat,
            "matched_satisfied": matched_sat,
            "sat_delta": None if base_sat is None else matched_sat - base_sat,
        })
    return summary


# --- rendering --------------------------------------------------------------

def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _rate(numerator: int, denominator: int) -> str:
    if not denominator:
        return "n/a"
    return f"{numerator}/{denominator} ({numerator / denominator * 100:.1f}%)"


def _delta(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:+.1f}pp"


def render_summary(summary: Dict[str, dict]) -> str:
    lines = [
        "| method | model | rows | state exact | P | R | recommendations "
        "satisfied | clarify | contamination | rows with provenance problems |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for method, item in summary.items():
        clarify = item["clarify"]
        asked = (f"{clarify.get('appropriate', 0)} ok / "
                 f"{clarify.get('unnecessary', 0)} unnec / "
                 f"{clarify.get('missed', 0)} missed")
        lines.append(
            f"| {method} | {', '.join(item['models'])} | {item['rows']} | "
            f"{_rate(item['state_exact'], item['state_rows'])} | "
            f"{_pct(item['state_precision'])} | {_pct(item['state_recall'])} | "
            f"{_rate(item['satisfied'], item['recommend_rows'])} | {asked} | "
            f"{item['contamination']} | "
            f"{_rate(item['provenance_rows'], item['rows'])} |"
        )
    return "\n".join(lines)


def render_dissociation(cells: Dict[str, dict]) -> str:
    lines = [
        "| method | both | recalled but violated | lucky | both wrong |",
        "|---|---|---|---|---|",
    ]
    for method, counts in cells.items():
        lines.append(f"| {method} | {counts['both']} | "
                     f"{counts['recalled_but_violated']} | {counts['lucky']} | "
                     f"{counts['both_wrong']} |")
    return "\n".join(lines)


def render_paired(summary: Sequence[dict]) -> str:
    lines = [
        "| method | family | pairs | state base | state matched | delta | "
        "satisfied base | satisfied matched | delta |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for item in summary:
        lines.append(
            f"| {item['method']} | {item['family']} | {item['pairs']} | "
            f"{_pct(item['base_state'])} | {_pct(item['matched_state'])} | "
            f"{_delta(item['state_delta'])} | {_pct(item['base_satisfied'])} | "
            f"{_pct(item['matched_satisfied'])} | "
            f"{_delta(item['sat_delta'])} |"
        )
    return "\n".join(lines)


def render(rows: Sequence[dict]) -> str:
    summary = summarise(rows)
    return "\n\n".join([
        "## Per method\n\n" + render_summary(summary),
        "## State vs action (recommendations only)\n\n"
        + render_dissociation(dissociation(rows)),
        "## Matched variants (base vs matched, per family)\n\n"
        + render_paired(paired_summary(rows)),
    ])


# --- CLI --------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("predictions", help="path to predictions.jsonl")
    parser.add_argument("--checkpoints", default=None,
                        help="default: config.CHECKPOINTS_PATH")
    parser.add_argument("--histories", default=None,
                        help="default: config.HISTORIES_PATH")
    parser.add_argument("--catalogue", default=None,
                        help="default: config.CATALOGUE_PATH")
    parser.add_argument("--markdown", default=None,
                        help="also write the rendered tables to this path")
    args = parser.parse_args(argv)

    rows = evaluate_path(args.predictions, args.checkpoints, args.histories,
                         args.catalogue)
    if not rows:
        print(f"no predictions to score in {args.predictions}", file=sys.stderr)
        return 1

    rendered = render(rows)
    print(rendered)
    if args.markdown:
        parent = os.path.dirname(args.markdown)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.markdown, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(rendered + "\n")
        print(f"\nwrote {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
