"""Step 7's ablation table: one resolver rule off at a time.

The extraction is the expensive half of the benchmark, so this never asks the
model again. It reads the same persisted `ExtractionCache` the pipeline run
built, re-resolves every checkpoint with one flag off, and scores what comes
out against the answer key.

**No model call is made here, by design.** The question is what a *rule*
contributes on top of the extraction the model actually produced; putting a
second model in the loop would mix two sources of noise and make a rule's
effect indistinguishable from a sampled reply. The plate is picked by the
same deterministic policy as the gold baseline -- first candidate from
`recipe_search` -- and judged against the checkpoint's own gold profile, so
the action column says "would a sensible choice made from this profile have
been safe", not "did the model choose well" (the main run answers that).

Usage::

    python -m memproj.memory.ablate                 # table to stdout
    python -m memproj.memory.ablate --markdown out/ablations.md
"""

import argparse
import os
import sys
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence

from memproj import config, replay
from memproj.eval import metrics
from memproj.pipeline import recommend, run
from memproj.resolver import active_records, pending_clarification
from memproj.store import Store

ALL_FLAGS = ("use_scope", "use_time", "use_subject", "use_supersede",
             "use_correction")

# `None` is the control: every rule on, over the model's own extraction.
# That row is not the gold ceiling -- it is `proposed`'s memory with a
# deterministic plate, so it shows how much of `proposed`'s gap is memory and
# how much is the model's choice.
ABLATIONS = (None,) + ALL_FLAGS

# Label printed for the flag-off row. "no_scope" reads as the rule removed,
# which is what the row measures.
NAMES = {None: "all rules on"}


def label(off: Optional[str]) -> str:
    return NAMES.get(off) or off.replace("use_", "no_")


def _flags(off: Optional[str]) -> Dict[str, bool]:
    return {name: name != off for name in ALL_FLAGS}


def profile_at(store: Store, checkpoint: dict,
               off: Optional[str] = None) -> Dict[str, set]:
    """The profile the resolver builds with one rule disabled."""
    records = active_records(store, checkpoint["person"], checkpoint["now"],
                             **_flags(off))
    profile: Dict[str, set] = {}
    for record in records:
        profile.setdefault(record["attribute"], set()).add(record["value"])
    return profile


def cell(store: Store, checkpoint: dict, catalogue: Sequence[dict],
         off: Optional[str] = None) -> dict:
    """One (ablation, checkpoint) observation. No model, no file I/O."""
    profile = profile_at(store, checkpoint, off)
    state = metrics.score(profile, checkpoint["gold_profile"])

    action = ("clarify"
              if pending_clarification(store, checkpoint["now"]) is not None
              else "recommend")
    satisfied = None
    empty = False
    if action == "recommend":
        candidates = recommend.recipe_search(profile, catalogue)
        if not candidates:
            # The profile rejected the whole catalogue. That is a real
            # outcome of dropping a rule -- most often a scope rule that
            # keeps an exception ruling everything out -- and it must be
            # counted, not crash or be hidden behind a fallback.
            empty = True
        else:
            ok, _violations = recommend.validate_choice(
                candidates[0]["id"], checkpoint["gold_profile"], catalogue)
            satisfied = ok

    appropriate = (action == "clarify"
                   and checkpoint["gold_action"] == "clarify")
    return {
        "checkpoint_id": checkpoint["checkpoint_id"],
        "trajectory_id": checkpoint["trajectory_id"],
        "exact": state["exact"],
        "precision": state["precision"],
        "recall": state["recall"],
        "missing": len(state["missing"]),
        "extra": len(state["extra"]),
        "action": action,
        "satisfied": satisfied,
        "empty": empty,
        "appropriate": appropriate,
        "handled": bool(appropriate or satisfied),
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarise(cells: Sequence[dict], off: Optional[str]) -> dict:
    recommendable = [c for c in cells if c["satisfied"] is not None]
    return {
        "ablation": label(off),
        "off": off,
        "checkpoints": len(cells),
        "exact": sum(1 for c in cells if c["exact"]),
        "precision": _mean([c["precision"] for c in cells]),
        "recall": _mean([c["recall"] for c in cells]),
        "satisfied": sum(1 for c in recommendable if c["satisfied"]),
        "recommendable": len(recommendable),
        "empty": sum(1 for c in cells if c["empty"]),
        "appropriate": sum(1 for c in cells if c["appropriate"]),
        "handled": sum(1 for c in cells if c["handled"]),
    }


def cells_for(store_for: Callable[[str], Store],
              checkpoints: Sequence[dict],
              catalogue: Sequence[dict],
              off: Optional[str] = None) -> List[dict]:
    return [cell(store_for(cp["trajectory_id"]), cp, catalogue, off)
            for cp in checkpoints]


def run_ablations(store_for: Callable[[str], Store],
                  checkpoints: Sequence[dict],
                  catalogue: Sequence[dict]) -> List[dict]:
    """One summary row per ablation, over every checkpoint."""
    return [summarise(cells_for(store_for, checkpoints, catalogue, off), off)
            for off in ABLATIONS]


def by_family(cells_by_ablation: Dict[Optional[str], Sequence[dict]],
              families: Dict[str, str]) -> List[dict]:
    """Exact-state and handled counts per (ablation, family)."""
    out = []
    for off, cells in cells_by_ablation.items():
        groups: Dict[str, List[dict]] = defaultdict(list)
        for observation in cells:
            groups[families[observation["trajectory_id"]]].append(observation)
        for family in sorted(groups):
            group = groups[family]
            out.append({
                "ablation": label(off),
                "family": family,
                "n": len(group),
                "exact": sum(1 for c in group if c["exact"]),
                "handled": sum(1 for c in group if c["handled"]),
            })
    return out


def idle_notes(rows: Sequence[dict], edited: int,
               trajectories: int) -> List[str]:
    """A row that moves nothing needs a reason, or a reader calls it a bug."""
    control = next(r for r in rows if r["off"] is None)
    notes = []
    for r in rows:
        if r["off"] is None:
            continue
        if (r["exact"], r["handled"], r["satisfied"]) != (
                control["exact"], control["handled"], control["satisfied"]):
            continue
        notes.append(
            f"- `{r['ablation']}` moves no checkpoint on this extraction"
            + ("" if edited else
               f": no stored row was restated (`edited = 1`) in any of the "
               f"{trajectories} extractions, so the rule had nothing to date")
            + ".")
    return notes


# --- rendering --------------------------------------------------------------

def render_table(rows: Sequence[dict]) -> str:
    lines = [
        "| ablation | checkpoints | state exact | P | R | satisfied | "
        "empty profile | clarify ok | handled |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['ablation']} | {r['checkpoints']} | "
            f"{r['exact']}/{r['checkpoints']} "
            f"({100 * r['exact'] / r['checkpoints']:.1f}%) | "
            f"{100 * r['precision']:.1f}% | {100 * r['recall']:.1f}% | "
            f"{r['satisfied']}/{r['recommendable']} | {r['empty']} | "
            f"{r['appropriate']} | {r['handled']}/{r['checkpoints']} |")
    return "\n".join(lines)


def render_family(rows: Sequence[dict]) -> str:
    """Only the cells an ablation moved, measured against the control.

    A family's control cell already carries the extraction's own mistakes,
    so printing absolute counts would list almost every family under every
    flag -- the model misses a fact wherever it likes -- and say nothing
    about the rule. The question is what switching a rule off COSTS, so a
    cell is printed only when it moves, with both readings.
    """
    control = {r["family"]: r for r in rows if r["ablation"] == label(None)}
    lines = [
        "| ablation | family | n | exact (all on -> off) | "
        "handled (all on -> off) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        if r["ablation"] == label(None) or r["family"] not in control:
            continue                      # the control is the first table
        base = control[r["family"]]
        if r["exact"] == base["exact"] and r["handled"] == base["handled"]:
            continue                      # the rule does not bite here
        lines.append(
            f"| {r['ablation']} | {r['family']} | {r['n']} | "
            f"{base['exact']}/{r['n']} -> {r['exact']}/{r['n']} | "
            f"{base['handled']}/{r['n']} -> {r['handled']}/{r['n']} |")
    return chr(10).join(lines)


def edited_rows(store: Store) -> int:
    """Rows a restatement rewrote in place -- what the correction rule dates.

    The store exposes no counter for it and this table is the only reader
    that wants one: without an edited row, `use_correction` cannot move a
    checkpoint, and the reader deserves to be told that rather than left to
    wonder whether the rule was implemented.
    """
    row = store._conn.execute(
        "SELECT COUNT(*) AS n FROM prefs WHERE edited = 1").fetchone()
    return row["n"]


def missing_extractions(directory: str,
                        trajectory_ids: Sequence[str]) -> List[str]:
    """Trajectories with no cached extraction, in the order asked for.

    This module never asks the model, so a hole in the cache cannot be
    filled -- it can only be reported. A partially built cache (a pipeline
    run still in progress) would otherwise fail inside the extractor with a
    `None` client, or worse, answer for the trajectories that happen to be
    on disk.
    """
    return [tid for tid in trajectory_ids
            if not os.path.exists(os.path.join(directory, f"{tid}.sqlite"))]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolver ablations over the persisted extraction.")
    parser.add_argument("--markdown", metavar="PATH",
                        help="also write the tables to this file")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="first N trajectories only")
    args = parser.parse_args(argv)

    trajectories, checkpoints = replay.load_corpus()
    wanted = [t["trajectory_id"] for t in trajectories]
    if args.limit:
        wanted = wanted[:args.limit]
        checkpoints = [c for c in checkpoints
                       if c["trajectory_id"] in set(wanted)]
    missing = missing_extractions(config.EXTRACTION_DIR, wanted)
    if missing:
        print(f"{len(missing)} of {len(wanted)} trajectories have no cached "
              f"extraction (e.g. {missing[0]}); run the pipeline first -- "
              f"this table never asks the model", file=sys.stderr)
        return 2
    catalogue = run.load_catalogue()
    families = {t["trajectory_id"]: t["family"] for t in trajectories}

    from memproj.memory.methods import ExtractionCache
    fingerprint = run.corpus_fingerprint()
    with ExtractionCache(client=None, directory=config.EXTRACTION_DIR,
                         fingerprint=fingerprint) as cache:
        by_id = {t["trajectory_id"]: t for t in trajectories}
        stores: Dict[str, Store] = {}

        def store_for(trajectory_id: str) -> Store:
            if trajectory_id not in stores:
                stores[trajectory_id] = cache.trace(
                    by_id[trajectory_id]).store
            return stores[trajectory_id]

        rows = run_ablations(store_for, checkpoints, catalogue)
        cells_by_ablation = {off: cells_for(store_for, checkpoints, catalogue,
                                            off)
                             for off in ABLATIONS}
        fam_rows = by_family(cells_by_ablation, families)
        stats = cache.stats()
        edited = sum(edited_rows(store_for(tid)) for tid in wanted)

    source = (f"Extraction: {stats['loaded']} trajectories loaded, "
              f"{stats['extracted']} extracted, "
              f"{len(stats['failures'])} failed turn(s).")
    if stats.get("discarded"):
        source += f" Discarded {stats['discarded']} file(s) from other inputs."
    notes = idle_notes(rows, edited, len(wanted))
    text = "\n\n".join([
        source,
        "### Resolver ablations", render_table(rows),
        *(["Notes:"] + notes if notes else []),
        "### Where each rule bites", render_family(fam_rows),
    ])
    print(text)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text + "\n")
        print(f"wrote {args.markdown}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
