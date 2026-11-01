"""Run one or more memory methods over the corpus and write predictions.jsonl.

One loop, one predictions line per (checkpoint, method). What differs between
methods is only `build_context()` (README, Method interface); search, ranking,
the prompt and the parsing are shared, so a difference in the output is
attributable to the memory method and nothing else.

Two kinds of method run here:

* `gold` makes no model call. It injects `checkpoint["gold_profile"]`
  verbatim, so README, Predictions format pins `tokens=0` and `latency_s=null`.
* every other method returns a context (and a profile, or `None`) and then
  goes through the one recommender. `cited_turns` is the method's
  `evidence_turn_ids`: the model is not asked to cite, because that would
  add a second failure mode and stop the recommender being identical across
  methods. `latency_s` is wall-clock and covers the whole checkpoint.

A malformed model reply is retried, bounded by `config.TOOL_RETRIES`, with
the rejection reason appended to the prompt. If every attempt is unusable,
the line is written as a `clarify` explaining that, rather than left out:
a missing line would make the method look better than it is.

`--resume` replays the per-row journal a long GPU run leaves behind, so a
machine that sleeps halfway through costs only the unfinished checkpoints.
The journal is only trustworthy for the corpus that wrote it -- checkpoint
ids are positional and survive a re-seed -- so a fingerprint of the two
input files is stored beside it and a mismatch refuses rather than mixes.
A run without `--resume` starts the journal over.

    python -m memproj.pipeline.run --methods gold
    python -m memproj.pipeline.run --methods recent_only full_history
    python -m memproj.validate_predictions out/predictions.jsonl
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from memproj import config
from memproj import recipes

from memproj.pipeline import methods as memory_methods
from memproj.pipeline import recommend
from memproj.validate_predictions import validate_predictions


# --- corpus -----------------------------------------------------------------

def read_jsonl(path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def load_corpus(histories_path=None, checkpoints_path=None) -> Tuple[List[dict], List[dict]]:
    return (read_jsonl(histories_path or config.HISTORIES_PATH),
            read_jsonl(checkpoints_path or config.CHECKPOINTS_PATH))


def load_catalogue(path=None) -> List[dict]:
    """Derived rows, the shape `recipes.validate()` reasons about."""
    return recipes.load_derived(str(path or config.CATALOGUE_PATH))


def event_turn_index(trajectory: dict) -> Dict[str, str]:
    """event_id -> the first turn_id that carries it. User turns only.

    The generator restarts turn ids in every trajectory, which is why this
    index is per trajectory and never merged across the corpus.
    """
    index: Dict[str, str] = {}
    for turn in trajectory.get("turns", []):
        event_id = turn.get("event_id")
        if event_id and turn.get("speaker") == "user" and event_id not in index:
            index[event_id] = turn["turn_id"]
    return index


def evidence_turn_ids(trajectory: dict, checkpoint: dict) -> List[str]:
    """Turn ids for the events the checkpoint declares relevant.

    The gold baseline cites what the answer key says shaped the profile, so
    its provenance is correct by construction; whether the *method* cites
    correctly is what the evaluator audits.
    """
    index = event_turn_index(trajectory)
    cited = []
    for event_id in checkpoint.get("relevant_event_ids", ()):
        turn_id = index.get(event_id)
        if turn_id is not None and turn_id not in cited:
            cited.append(turn_id)
    return cited


def gold_context(trajectory: dict, checkpoint: dict) -> dict:
    """README, Method interface's return shape, with the answer key inside.

    The gold baseline does not need the text, but implementing the interface
    keeps it interchangeable with the other methods and gives the structured methods a
    reference implementation that is free of memory errors.
    """
    profile = checkpoint["gold_profile"]
    texts = {turn["turn_id"]: turn["text"] for turn in trajectory.get("turns", [])}
    cited = evidence_turn_ids(trajectory, checkpoint)
    lines = ["Preferences that apply now:"]
    for attribute in sorted(profile):
        lines.append(f"- {attribute}: {', '.join(profile[attribute])}")
    if not profile:
        lines.append("- (none)")
    if cited:
        lines += ["", "Evidence turns:"]
        lines += [f"- {turn_id}: {texts[turn_id]}" for turn_id in cited]
    return {"context_text": "\n".join(lines),
            "profile": profile,
            "evidence_turn_ids": cited}


# --- the gold baseline ------------------------------------------------------

def clarify_question(profile: Dict[str, Sequence[str]]) -> str:
    """A deterministic question naming what the ambiguity could be about.

    Gold only says `clarify`; the wording is not graded, but naming the live
    restrictions keeps the line honest and readable in the failure samples.
    """
    values = [value for attribute in sorted(profile)
              for value in profile[attribute]]
    if len(values) > 1:
        return ("Which restriction did you mean: "
                + " or ".join(values) + "?")
    return "Which restriction did you mean?"


def gold_prediction(trajectory: dict, checkpoint: dict,
                    catalogue: Sequence[dict]) -> dict:
    """One predictions line with the answer key injected. No model call."""
    profile = {attribute: sorted(values)
               for attribute, values in checkpoint["gold_profile"].items()}
    cited = evidence_turn_ids(trajectory, checkpoint)

    if checkpoint["gold_action"] == "clarify":
        action, recipe_id, question = "clarify", None, clarify_question(profile)
        explanation = ("Your statement did not say which restriction it was "
                       "about, so I need to ask before recommending.")
    else:
        candidates = recommend.recipe_search(profile, catalogue)
        if not candidates:
            # The generator guarantees at least three valid recipes per
            # checkpoint; an empty set here is a plumbing bug, not data.
            raise RuntimeError(
                f"{checkpoint['checkpoint_id']}: no recipe satisfies the gold "
                f"profile {profile!r}"
            )
        chosen = candidates[0]
        action, recipe_id, question = "recommend", chosen["id"], None
        explanation = recommend.explain(chosen, profile)

    return {
        "trajectory_id": trajectory["trajectory_id"],
        "checkpoint_id": checkpoint["checkpoint_id"],
        "method": "gold",
        "model": "gold",
        "profile": profile,
        "action": action,
        "recipe_id": recipe_id,
        "question": question,
        "explanation": explanation,
        "cited_turns": cited,
        "tokens": 0,
        "latency_s": None,
    }


# --- the model path (every method except gold) ------------------------------

# The methods this runner builds itself. `latest_wins` and `proposed` are
# README, Method interface methods too, but their build_context belongs to the
# memory half; `run()` gets them from `memproj.memory.methods.structured()`.
FREE_TEXT_METHODS = ("full_history", "recent_only", "retrieval", "rolling_summary")
STRUCTURED_METHODS = ("latest_wins", "proposed")
METHODS = ("gold",) + FREE_TEXT_METHODS


def _recommend_with_retries(profile, context_text: str, request_text: str,
                            catalogue: Sequence[dict],
                            chat_fn: Callable[[List[dict]], dict],
                            retries: Optional[int] = None
                            ) -> Tuple[dict, float]:
    """One recommendation, retried on a malformed reply. Returns (choice, seconds).

    The retry re-prompts with the rejection reason instead of resending the
    same message: at temperature 0 the same prompt mostly reproduces the same
    broken JSON. When every attempt is rejected the checkpoint still gets a
    line -- a clarification saying so -- because a missing row would silently
    improve the method's apparent accuracy.
    """
    retries = config.TOOL_RETRIES if retries is None else retries
    started = time.perf_counter()
    rejection = None
    for _ in range(retries + 1):
        try:
            choice = recommend.recommend(
                profile, context_text, request_text, catalogue,
                chat_fn=chat_fn, retry=rejection,
            )
        except ValueError as exc:
            rejection = str(exc)
            continue
        return choice, time.perf_counter() - started
    return {
        "action": "clarify",
        "recipe_id": None,
        "question": ("I could not turn the model's reply into a usable "
                     "recommendation. Could you rephrase your request?"),
        "explanation": (f"The model's reply was rejected {retries + 1} times; "
                        f"last reason: {rejection}"),
        "tokens": 0,
    }, time.perf_counter() - started


def method_prediction(trajectory: dict, checkpoint: dict,
                      catalogue: Sequence[dict], method: str, builder,
                      chat_fn: Callable[[List[dict]], dict] = recommend.chat,
                      model: Optional[str] = None) -> dict:
    """One predictions line for a method whose profile is None or resolved.

    `cited_turns` is the method's `evidence_turn_ids`. Asking the model to
    cite would introduce a second failure mode and make the recommender
    differ between methods, which is exactly what the experiment controls
    for.
    """
    context = builder.build_context(trajectory, checkpoint)
    if set(context) != {"context_text", "profile", "evidence_turn_ids"}:
        raise ValueError(
            f"method {method!r} returned {sorted(context)}; README, "
            f"Method interface allows context_text, profile, evidence_turn_ids"
        )

    choice, latency = _recommend_with_retries(
        context["profile"], context["context_text"], checkpoint["request_text"],
        catalogue, chat_fn,
    )
    return {
        "trajectory_id": trajectory["trajectory_id"],
        "checkpoint_id": checkpoint["checkpoint_id"],
        "method": method,
        "model": model or config.MODEL,
        "profile": context["profile"],
        "action": choice["action"],
        "recipe_id": choice["recipe_id"],
        "question": choice["question"],
        "explanation": choice["explanation"],
        "cited_turns": list(context["evidence_turn_ids"]),
        "tokens": int(choice.get("tokens") or 0),
        "latency_s": round(latency, 6),
    }


def build_prediction(trajectory: dict, checkpoint: dict,
                     catalogue: Sequence[dict], method: str = "gold",
                     builders: Optional[Dict[str, object]] = None,
                     chat_fn: Optional[Callable[[List[dict]], dict]] = None,
                     model: Optional[str] = None) -> dict:
    """Dispatch one checkpoint to the method that owns it."""
    if method == "gold":
        return gold_prediction(trajectory, checkpoint, catalogue)
    if builders is None:
        builders = memory_methods.baselines()
    if method in builders:
        return method_prediction(trajectory, checkpoint, catalogue, method,
                                 builders[method],
                                 chat_fn=chat_fn or recommend.chat,
                                 model=model)
    raise NotImplementedError(
        f"method {method!r} has no build_context in this run: the pipeline "
        f"half builds gold and baselines, and latest_wins/proposed arrive "
        f"through `memproj.memory.methods.structured(cache)` as builders"
    )


# --- the run ----------------------------------------------------------------

def headroom_summary(rows: Sequence[dict], catalogue: Sequence[dict]) -> dict:
    """How much room the corpus leaves the constraint-satisfaction metric.

    A checkpoint whose profile accepts the whole catalogue cannot be failed,
    so this is the number that says whether a results table means anything.
    """
    counts = [recommend.headroom(row["profile"], catalogue) for row in rows]
    if not counts:
        return {"count": 0}
    return {
        "count": len(counts),
        "min": min(counts),
        "median": int(statistics.median(counts)),
        "max": max(counts),
        "fully_open": sum(1 for n in counts if n == len(catalogue)),
        "starved": sum(1 for n in counts if n < config.TOP_K_RECIPES),
    }


def _row_key(row: dict) -> tuple:
    """The predictions contract's uniqueness key (README, Predictions format)."""
    return (row["trajectory_id"], row["checkpoint_id"], row["method"],
            row["model"])


def corpus_fingerprint(histories_path=None, checkpoints_path=None) -> str:
    """A short hash of the two corpus files, recorded beside the journal.

    Checkpoint ids are positional (`p000_base_c0`), so regenerating with a
    different `--seed` keeps every id and changes every answer key. Resuming
    across that would write a predictions file whose rows were scored against
    a corpus that no longer exists, with nothing to show for it.
    """
    digest = hashlib.sha256()
    for path in (histories_path or config.HISTORIES_PATH,
                 checkpoints_path or config.CHECKPOINTS_PATH):
        with open(path, "rb") as fh:
            digest.update(fh.read())
    return digest.hexdigest()[:16]


def _read_text(path) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read().strip() or None



def run(methods: Sequence[str] = ("gold",), out_path=None, limit: Optional[int] = None,
        histories_path=None, checkpoints_path=None, catalogue_path=None,
        model: Optional[str] = None,
        chat_fn: Optional[Callable[[List[dict]], dict]] = None,
        client=None, resume: bool = False,
        extraction_dir=None) -> Tuple[List[dict], dict]:
    """Write one predictions line per (checkpoint, method). Returns (rows, stats).

    Rows are sorted before writing so the file's order is stable. The content
    is not byte-identical across runs for any method that calls a model:
    `latency_s` is wall-clock, which README, Predictions format explicitly allows.

    Every completed row is appended to `<out>.journal` as it finishes. With
    `resume=True` the journal is read first and its rows are skipped, so a
    GPU run that dies at checkpoint 140 only pays for 65 more.

    `client` is the ollama-shaped client the structured methods extract with
    (tests pass a script). It is created only when a structured method is
    asked for, so a gold or baseline run never needs one, and the extraction
    cache it feeds is shared by both structured methods and closed with the
    run.
    """
    trajectories, checkpoints = load_corpus(histories_path, checkpoints_path)
    by_id = {trajectory["trajectory_id"]: trajectory for trajectory in trajectories}
    catalogue = load_catalogue(catalogue_path)
    # Before `limit`: a limited run is a slice of the same corpus, and its
    # journal rows for the checkpoints it did not ask for are not stale.
    known = {(checkpoint["trajectory_id"], checkpoint["checkpoint_id"])
             for checkpoint in checkpoints}
    if limit is not None:
        checkpoints = checkpoints[:limit]

    path = str(out_path or config.PREDICTIONS_PATH)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    journal = path + ".journal"
    meta = journal + ".meta"
    fingerprint = corpus_fingerprint(histories_path, checkpoints_path)

    rows_by_key: Dict[tuple, dict] = {}
    stale = 0
    if resume and os.path.exists(journal):
        recorded = _read_text(meta)
        if recorded and recorded != fingerprint:
            raise ValueError(
                f"{journal} was written for a different corpus "
                f"({recorded} != {fingerprint}): the ids match but the answer "
                f"keys do not. Delete {journal} and {meta} to start over, or "
                f"drop --resume."
            )
        for row in read_jsonl(journal):
            if (row["trajectory_id"], row["checkpoint_id"]) not in known:
                stale += 1
                continue
            rows_by_key[_row_key(row)] = row
    resumed = len(rows_by_key)

    builders = memory_methods.baselines()
    cache = None
    if any(method in STRUCTURED_METHODS for method in methods):
        from memproj.memory import extract as extraction
        from memproj.memory import methods as structured_methods

        cache = structured_methods.ExtractionCache(
            client or extraction.make_client(), model=model,
            directory=extraction_dir, fingerprint=fingerprint)
        builders.update(structured_methods.structured(cache))
    chat_fn = chat_fn or recommend.chat
    written = 0
    # Without --resume this is a fresh start, so the journal must not keep
    # yesterday's lines: they would be read back by the *next* resumed run
    # and counted as work this run did.
    with open(meta, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(fingerprint + "\n")
    try:
        with open(journal, "a" if resume else "w", encoding="utf-8",
                  newline="\n") as log:
            for method in methods:
                for checkpoint in checkpoints:
                    trajectory_id = checkpoint["trajectory_id"]
                    if trajectory_id not in by_id:
                        raise KeyError(
                            f"checkpoint {checkpoint['checkpoint_id']} names "
                            f"trajectory {trajectory_id!r}, which is not in histories"
                        )
                    key = (trajectory_id, checkpoint["checkpoint_id"], method,
                           "gold" if method == "gold" else (model or config.MODEL))
                    if key in rows_by_key:
                        continue
                    row = build_prediction(by_id[trajectory_id], checkpoint,
                                           catalogue, method, builders=builders,
                                           chat_fn=chat_fn, model=model)
                    rows_by_key[key] = row
                    log.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
                    log.write("\n")
                    log.flush()
                    written += 1
    finally:
        if cache is not None:
            cache.close()

    rows = sorted(rows_by_key.values(),
                  key=lambda row: (row["trajectory_id"], row["checkpoint_id"],
                                   row["method"], row["model"]))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
            fh.write("\n")

    profile_by_id = {checkpoint["checkpoint_id"]: checkpoint["gold_profile"]
                     for checkpoint in checkpoints}
    mismatches = [
        row["checkpoint_id"] for row in rows if row["method"] == "gold"
        and row["profile"] != profile_by_id[row["checkpoint_id"]]
    ]
    # Headroom is a property of the checkpoint's answer key, not of a row:
    # a free-text row has profile=None, which would otherwise read as "the
    # whole catalogue satisfies it".
    ceiling = headroom_summary(
        [{"profile": checkpoint["gold_profile"]} for checkpoint in checkpoints],
        catalogue)
    by_method: Dict[str, int] = {}
    for row in rows:
        by_method[row["method"]] = by_method.get(row["method"], 0) + 1
    stats = {
        "path": path,
        "journal": journal,
        "rows": len(rows),
        "written": written,
        "resumed": resumed,
        "stale_journal_rows": stale,
        "checkpoints": len(checkpoints),
        "methods": list(methods),
        "rows_by_method": by_method,
        "headroom": ceiling,
        "catalogue_size": len(catalogue),
        "state_mismatches": mismatches,
        "extraction": None if cache is None else cache.stats(),
    }
    return rows, stats


# --- CLI --------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="+", default=["gold"],
                        choices=list(METHODS) + ["latest_wins", "proposed"],
                        help="one or more of " + ", ".join(METHODS)
                             + ", latest_wins, proposed")
    parser.add_argument("--model", default=None,
                        help="Ollama tag to record and use; default: config.MODEL")
    parser.add_argument("--out", default=None,
                        help="default: config.PREDICTIONS_PATH")
    parser.add_argument("--resume", action="store_true",
                        help="skip (trajectory, checkpoint, method, model) "
                             "rows already in <out>.journal")
    parser.add_argument("--limit", type=int, default=None,
                        help="first N checkpoints, for a cheap loop")
    parser.add_argument("--histories", default=None)
    parser.add_argument("--checkpoints", default=None)
    parser.add_argument("--catalogue", default=None)
    parser.add_argument("--extraction-dir", default=None,
                        help="where the per-trajectory extraction cache "
                             "lives; default: config.EXTRACTION_DIR. Give a "
                             "second corpus its own directory -- the cache "
                             "discards files built for other inputs, so "
                             "sharing one would throw the first corpus's "
                             "extraction away.")
    args = parser.parse_args(argv)

    _, stats = run(args.methods, out_path=args.out, limit=args.limit,
                   histories_path=args.histories,
                   checkpoints_path=args.checkpoints,
                   catalogue_path=args.catalogue, model=args.model,
                   resume=args.resume,
                   extraction_dir=args.extraction_dir or config.EXTRACTION_DIR)

    print(f"wrote {stats['rows']} rows for {stats['checkpoints']} checkpoints "
          f"({', '.join(stats['methods'])}) -> {stats['path']}")
    if stats["resumed"]:
        print(f"resumed {stats['resumed']} rows from {stats['journal']}; "
              f"generated {stats['written']} this run")
    if stats["stale_journal_rows"]:
        print(f"ignored {stats['stale_journal_rows']} journal rows whose "
              f"checkpoint is not in this corpus", file=sys.stderr)
    if len(stats["rows_by_method"]) > 1:
        breakdown = ", ".join(f"{name} {count}" for name, count in
                              sorted(stats["rows_by_method"].items()))
        print(f"rows by method: {breakdown}")
    extraction = stats["extraction"]
    if extraction:
        print(f"extraction: {extraction['extracted']} trajectories "
              f"extracted, {extraction['loaded']} loaded from disk, "
              f"{extraction['turns']} turns, {extraction['tokens']} "
              f"tokens, {extraction['seconds']:.0f}s, "
              f"{len(extraction['failures'])} failed turn(s)")
        if extraction.get("discarded"):
            print(f"  discarded {extraction['discarded']} extraction file(s) "
                  f"built for other inputs")
        for failure in extraction["failures"][:20]:
            print(f"  extraction failed: {failure['trajectory_id']} "
                  f"{failure['turn_id']}: {failure['error']}",
                  file=sys.stderr)
    head = stats["headroom"]
    if head.get("count"):
        print(f"candidate headroom: min {head['min']}, median {head['median']}, "
              f"max {head['max']} of {stats['catalogue_size']} recipes")
        print(f"checkpoints the whole catalogue satisfies: "
              f"{head['fully_open']}/{head['count']}")
        print(f"checkpoints with fewer than {config.TOP_K_RECIPES} candidates: "
              f"{head['starved']}")
    if stats["state_mismatches"]:
        print(f"gold state mismatches: {stats['state_mismatches']}",
              file=sys.stderr)
        return 1
    if "gold" in stats["methods"]:
        print("gold state: every prediction profile equals the checkpoint profile")

    # A drifted field name is the failure mode README, Predictions format warns
    # about, and it is silent until the results table looks odd. Check on the
    # way out, not in a separate command someone has to remember.
    problems = validate_predictions(stats["path"],
                                    histories_path=args.histories,
                                    checkpoints_path=args.checkpoints,
                                    catalogue_path=args.catalogue)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
