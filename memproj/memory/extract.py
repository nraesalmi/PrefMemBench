"""Step 3: the model reads each user turn and emits one tool call.

The loop here is the same one `replay.events_to_tool_calls` stands in for:
turn -> (tool, args) -> `dispatch` -> store. Only the source of the calls
changes. Everything that makes a call SAFE is still the harness's: `ts` and
`turn_id` come from the turn being read, never from the model, and a
relative phrase is resolved by Python before the model is asked anything.

Three rules the loop enforces where a model alone would not:

  - one tool call per user turn. A second call is refused and logged: one
    turn is one statement, and the store has no honest way to carry two.
    (A vague statement is one call too -- `clarify_statement` carries the
    question and its answer.)
  - a refused call is retried, bounded, with the tool's error shown to the
    model. The error is the tool's own message, not a paraphrase, so what
    the model sees on retry is what a developer would see.
  - every attempt is logged as a decision, failures included, so the
    extraction failure rate is a number and not a shrug.
"""

import json
import os
from datetime import datetime
from typing import Optional, Tuple

from memproj import config, families, templates
from memproj.store import Store
from memproj.tools import (ATTRIBUTES, EXTRACTION_SYSTEM, OTHER_PEOPLE,
                           TOOL_SCHEMAS, dispatch, prompt_context)

MAX_ATTEMPTS = config.TOOL_RETRIES

# How many past exchanges the model sees. Enough for "that restriction" to
# have visible referents, small enough to stay trivial next to NUM_CTX.
HISTORY_TURNS = 12


def resolve_time_phrases(ts: datetime) -> str:
    """Every phrase the generator can write, resolved at `ts` by Python.

    This is the harness's job for the same reason `valid_until` is: a model
    doing date arithmetic turns a memory benchmark into a clock benchmark.
    The mapping itself is `families._temporary_window`, so the words mean
    the same thing to the model as they do to the answer key.
    """
    lines = ["Relative phrases, resolved (valid_until is exclusive):"]
    for phrase, _days in templates.TIME_PHRASES:
        end = families._temporary_window(ts, phrase)
        lines.append(f'  "{phrase}" ends at {end.isoformat()}')
    return "\n".join(lines)


def recent_exchanges(store: Store, before: str, then: str,
                     limit: int = HISTORY_TURNS):
    """The last few turns before `before`, plus the turn being read.

    The turns table holds who and when, not the words. The store therefore
    carries a `turn_texts` map of turn_id -> text that the runner fills as
    it reads; a model reading "that restriction" needs the sentences, and
    nothing else in the store has them. A turn with no text (an assistant
    acknowledgement, or a store the runner never fed) is shown as a short
    neutral line rather than an empty message.
    """
    rows = store._conn.execute(
        "SELECT turn_id, ts, speaker FROM turns WHERE ts < ?"
        " ORDER BY ts DESC, turn_id DESC LIMIT ?", (before, limit)).fetchall()
    texts = store.turn_texts
    messages = []
    for r in rows:
        if r["speaker"] == "assistant":
            content = texts.get(r["turn_id"], "") or ACK
        else:
            content = texts.get(r["turn_id"], "") or "(earlier message)"
        messages.append({"role": r["speaker"], "content": content})
    messages.reverse()
    return messages + [{"role": "user", "content": then}]


def _catalogues() -> str:
    """Allowed attributes and values, in the shape prompt_context expects."""
    return {attribute: sorted(values) if values else None
            for attribute, values in ATTRIBUTES.items()}


class Extractor:
    """One model, one store, one call per user turn.

    `client` is anything with ollama's `chat(model, messages, tools=,
    options=)` shape, so tests drive it with a script and the runner with
    `ollama.Client`. `model`, `num_ctx` and the temperature come from
    config.py unless passed -- the runner may pass a different tag as a CLI
    argument, which is the one legitimate override.
    """

    def __init__(self, client, model: str = None, num_ctx: int = None,
                 temperature: float = None, seed: int = None):
        self.client = client
        self.model = model or config.MODEL
        self.num_ctx = num_ctx or config.NUM_CTX
        self.temperature = (config.TEMPERATURE if temperature is None
                            else temperature)
        self.seed = config.SEED if seed is None else seed
        # Tokens this extractor has spent, summed over every answered call.
        # The counts ride along on the reply, so this is free; a failed
        # attempt (a timeout, a refused call) produced no reply and none.
        self.tokens = 0

    # --- one turn -----------------------------------------------------------

    def run_turn(self, store: Store, turn) -> Tuple[bool, dict]:
        """Extract from one turn. Assistant turns take no call.

        `turn` is a row of the store's `turns` table (or a Turn), because
        that is what a live runner has: the turn the transcript just added.
        Returns (ok, result) where result names what happened for the
        runner's counting: tool, attempt, extra_calls, or the failure.
        """
        if _field(turn, "speaker") != "user":
            return True, {"skipped": "assistant"}
        ts = _field(turn, "ts")
        ts = ts if isinstance(ts, datetime) else datetime.fromisoformat(ts)

        messages = self._opening_messages(store, turn, ts)
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = self._chat(messages)
            except Exception as exc:            # ollama's own exceptions too
                # A transport failure is this attempt's failure, not the
                # run's: one dropped connection must not end a corpus replay.
                failure = f"{type(exc).__name__}: {exc}"
                self._log(store, turn, "(model error)", {}, False,
                          failure, attempt, ts)
                continue
            self.tokens += _reply_tokens(response)
            calls = _tool_calls(response)
            if not calls:
                # A reply with no call is a refused attempt like any other:
                # the model said nothing usable, so it is told so and asked
                # again, and an answer that never comes is a failure.
                messages.append({"role": "assistant",
                                 "content": (response.message.content
                                             or "(no tool call)")})
                messages.append({"role": "user", "content": NO_CALL_RETRY})
                self._log(store, turn, "(no tool call)", {}, False,
                          "no tool call in the reply", attempt, ts)
                failure = "no tool call in the reply"
                continue

            name = calls[0].function.name
            args = _arguments(calls[0])
            if args is None:
                # dispatch would reject this too; building the tool reply
                # ourselves keeps the feedback a native tool_call sequence
                # (the model called, the tool answered) instead of the model
                # seeing its own refusal as a bare user instruction.
                error = "arguments were not a JSON object"
                self._log(store, turn, name, {}, False, error, attempt, ts)
                messages.append(_tool_message(name, {"error": error}))
                messages.append({"role": "user", "content": RETRY_INSTRUCTION})
                failure = error
                continue

            try:
                ok, result = dispatch(store, name, args, ts,
                                      _field(turn, "turn_id"), attempt=attempt)
            except Exception as exc:   # a caller bug, but still only one turn
                self._log(store, turn, name, args, False,
                          f"{type(exc).__name__}: {exc}", attempt, ts)
                return False, {"error": f"{type(exc).__name__}: {exc}"}
            # dispatch logs the refusal itself, with its error. Logging it
            # again here would make one statement look like two failures.
            if ok:
                extra = len(calls) - 1
                if extra:
                    for call in calls[1:]:
                        store.log_decision(
                            turn_id=_field(turn, "turn_id"),
                            tool=call.function.name,
                            args_json=json.dumps(_arguments(call) or {},
                                                 sort_keys=True, default=str),
                            ok=False, ts=ts.isoformat(), attempt=attempt,
                            error="second tool call in one turn was not run")
                return True, {"tool": name, "attempt": attempt,
                              "extra_calls": extra, "result": result}

            messages.append(_tool_message(name, result))
            messages.append({"role": "user", "content": RETRY_INSTRUCTION})
            failure = result.get("error")
        return False, {"failed_after": MAX_ATTEMPTS, "error": failure}

    # --- the prompt ---------------------------------------------------------

    def _opening_messages(self, store: Store, turn, ts: datetime):
        now = ts.isoformat()
        system = "\n".join([
            EXTRACTION_SYSTEM,
            "",
            prompt_context(ts, _catalogues()),
            "",
            resolve_time_phrases(ts),
            "",
            f"People besides the user who may hold preferences: "
            f"{', '.join(OTHER_PEOPLE)}.",
        ])
        return ([{"role": "system", "content": system}]
                + recent_exchanges(store, now, turn_text(turn)))

    def _chat(self, messages):
        options = {"num_ctx": self.num_ctx, "temperature": self.temperature,
                   "seed": self.seed}
        # Thinking stays ON here: the published extraction numbers were measured
        # with it, and the tool-calling prompt does not hit the runaway the
        # pipeline's prose prompts do. Pinned rather than left to the client
        # default so the two roles cannot drift together.
        return self.client.chat(self.model, messages, tools=TOOL_SCHEMAS,
                                options=options, think=True)

    def _log(self, store, turn, tool, args, ok, error, attempt, ts):
        store.log_decision(
            turn_id=_field(turn, "turn_id"), tool=tool,
            args_json=json.dumps(args, sort_keys=True, default=str),
            ok=ok, ts=ts.isoformat(), error=error, attempt=attempt)


ACK = "Got it."
NO_CALL_RETRY = (
    "You did not call a tool. Call exactly one tool for this message: "
    "ignore_statement if it states no preference.")
RETRY_INSTRUCTION = (
    "The tool refused that call: see the error above. Call the tool again "
    "with corrected arguments, or call a different one. Do not repeat the "
    "same call.")


def _tool_message(name, result) -> dict:
    return {"role": "tool", "name": name,
            "content": json.dumps(result, sort_keys=True)}


def _field(turn, name):
    """Read a turn field from either the Turn object or the store's row."""
    return getattr(turn, name) if hasattr(turn, name) else turn[name]


def turn_text(turn) -> str:
    """The words of a turn: on the Turn or the turn row, whichever is held."""
    if isinstance(turn, dict) or hasattr(turn, "keys"):
        return turn.get("text") or ""
    return getattr(turn, "text", "") or ""


def _reply_tokens(response) -> int:
    """`prompt_eval_count + eval_count` from an ollama reply, dict or object."""
    total = 0
    for name in ("prompt_eval_count", "eval_count"):
        value = getattr(response, name, None)
        if value is None and isinstance(response, dict):
            value = response.get(name)
        total += value or 0
    return total


def _tool_calls(response):
    """ollama's tool_calls, normalised, or [] when the reply has none.

    Older ollama clients returned dicts; newer ones return objects. The loop
    should not care which.
    """
    message = getattr(response, "message", response)
    calls = getattr(message, "tool_calls", None) or []
    return list(calls)


def _arguments(call) -> Optional[dict]:
    args = call.function.arguments
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    return args if isinstance(args, dict) else None

# --- the corpus run ---------------------------------------------------------

def diff_profiles(got: dict, gold: dict):
    """(missing, extra, wrong_value) between an active profile and gold.

    An attribute is compared as a SET of (key, value) facts: `excluded_
    ingredient` is a set, `spice_level` a slot that happened to hold one
    value at a time. Comparing facts rather than attributes means a set
    mismatched by one member is one missing fact, not one wrong attribute.
    A slot filled with a value that shares nothing with gold is also named
    in `wrong_value` -- "they said medium, gold says mushroom and spicy" --
    but that entry is a label for the failure samples, not a charge: the
    facts are already in `missing` (every gold value) and `extra` (every
    value the method said), so precision and recall pay per fact and the
    same error is counted once. Charging the attribute a second time there
    is what made an earlier version credit a hit for gold values the method
    never produced. An attribute gold leaves empty has nothing to be wrong
    about: a value said where nothing was expected is extra, full stop.
    """
    missing, extra, wrong = [], [], []
    for attribute in gold:
        gold_values = set(gold.get(attribute) or [])
        got_values = set(got.get(attribute) or [])
        missing.extend(f"{attribute}:{v}"
                       for v in sorted(gold_values - got_values))
        extra.extend(f"{attribute}:{v}"
                     for v in sorted(got_values - gold_values))
        if gold_values and got_values and not (gold_values & got_values):
            wrong.append({"attribute": attribute,
                          "gold": sorted(gold_values),
                          "got": sorted(got_values)})
    for attribute in got:
        if attribute not in gold and got[attribute]:
            extra.extend(f"{attribute}:{v}"
                         for v in sorted(set(got[attribute])))
    return missing, extra, wrong


def score(got: dict, gold: dict) -> dict:
    """State P/R/exact for one checkpoint, counted per fact.

    `tp` is every gold fact the profile produced, so it is never negative
    and a disjoint multi-valued attribute (62 of 193 checkpoints have one)
    credits nothing. `wrong_value` is a reason, not a term: its facts are
    in `missing`/`extra` already.
    """
    missing, extra, wrong = diff_profiles(got, gold)
    total = sum(len(set(gold.get(attribute) or [])) for attribute in gold)
    return {"missing": missing, "extra": extra, "wrong_value": wrong,
            "exact": not missing and not extra,
            "tp": total - len(missing),
            "fn": len(missing),
            "fp": len(extra)}


def completed_trajectories(rows, checkpoints) -> set:
    """Trajectory ids whose every checkpoint already has a journal row.

    A trajectory is all-or-nothing: a row is written only after its
    checkpoint is scored, so a checkpoint on the far side of a chosen
    `--limit` is not "missing", it was never asked for. The caller passes
    the checkpoints it means to cover, which is what makes this work for
    a limited run too.
    """
    need = {}
    for c in checkpoints:
        need.setdefault(c["trajectory_id"], set()).add(c["checkpoint_id"])
    have = {}
    for r in rows:
        have.setdefault(r["trajectory_id"], set()).add(r["checkpoint_id"])
    return {tid for tid, ids in need.items() if ids <= have.get(tid, set())}


def resume_state(path, checkpoints):
    """(rows kept, trajectory ids to skip) for a journal that was paused.

    A trajectory interrupted mid-way has some of its rows already written.
    Those rows are dropped and the trajectory is rerun from its first
    checkpoint, because the run is sequential: a checkpoint asked of a
    half-filled store would score a profile the model never would have
    produced at that point. Keeping the complete trajectories means the
    same checkpoint is never journalled twice.

    The kept rows are written back to `path` before anything is appended,
    so a second interruption cannot leave duplicates behind.
    """
    if not os.path.exists(path):
        return [], set()
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    skip = completed_trajectories(rows, checkpoints)
    kept = [r for r in rows if r["trajectory_id"] in skip]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in kept:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    return kept, skip


def run(trajectories, checkpoints, client, model: str = None,
        limit: int = None, verbose: bool = False, journal=None,
        skip=frozenset()):
    """Read every user turn of each trajectory, then ask every checkpoint.

    The store is built the way replay builds it: turns copied first, then
    each turn read at the instant it happened. A checkpoint is asked of the
    store as it stood at `now` -- replay's `_turns_before` cut, reused so
    the gold run and the model run can never disagree about what "before"
    means. With `limit`, only the first N pairs are run. With `skip`, the
    named trajectories are not touched and their checkpoints are ignored.

    Each row carries `failures`: the statements the loop gave up on before
    this checkpoint, with the turn and the error that ended them. The
    decision rows that hold the same error are in the store, which is
    closed at the end of the trajectory; this is the copy that survives.
    """
    from memproj import replay   # replay imports store/tools; keep it lazy
    from memproj.resolver import active_profile

    if skip:
        trajectories = [t for t in trajectories
                        if t["trajectory_id"] not in skip]
        live = {t["trajectory_id"] for t in trajectories}
        checkpoints = [c for c in checkpoints if c["trajectory_id"] in live]

    if limit:
        keep = {f"p{i:03d}" for i in range(limit)}
        trajectories = [t for t in trajectories if t["pair_id"] in keep]
        live = {t["trajectory_id"] for t in trajectories}
        checkpoints = [c for c in checkpoints if c["trajectory_id"] in live]

    cps_by_traj = {}
    for c in checkpoints:
        cps_by_traj.setdefault(c["trajectory_id"], []).append(c)

    extractor = Extractor(client=client, model=model)
    rows = []
    for tr in trajectories:
        cps = cps_by_traj.get(tr["trajectory_id"], [])
        cuts = [(replay._turns_before(tr, c["now"]), c) for c in cps]
        turns = [replay._parse_turn(t) for t in tr["turns"]]
        store = Store(":memory:").create()
        try:
            store.turn_texts = {t.turn_id: t.text for t in turns}
            for t in turns:
                store.add_turn(t)
            read = 0                     # how many turns have been read
            # Every statement the loop has given up on so far, with the
            # error that killed it. The store's decision log dies with the
            # store, so a 0% run would otherwise say "nothing was stored"
            # and not why.
            failures = []
            for cut, cp in cuts:
                for turn in turns[read:cut]:
                    if turn.speaker != "user":
                        continue
                    ok, result = extractor.run_turn(store, turn)
                    if not ok:
                        failures.append({"turn_id": turn.turn_id,
                                         "error": result.get("error")})
                read = max(read, cut)
                got = active_profile(store, cp["person"], cp["now"])
                row = {"checkpoint_id": cp["checkpoint_id"],
                       "trajectory_id": tr["trajectory_id"],
                       "family": tr["family"], "variant": tr["variant"],
                       "kind": cp["kind"], "failed": len(failures),
                       "failures": list(failures),
                       **score(got, cp["gold_profile"])}
                rows.append(row)
                if journal is not None:
                    # One line per checkpoint as it is scored: a run that
                    # is interrupted leaves a table instead of nothing.
                    journal.write(json.dumps(row, sort_keys=True) + "\n")
                    journal.flush()
                if verbose and not row["exact"]:
                    keep = ("checkpoint_id", "kind", "missing", "extra",
                            "wrong_value")
                    print(json.dumps({k: row[k] for k in keep if k in row},
                                     sort_keys=True))
        finally:
            store.close()
    return rows


def summarise(rows) -> str:
    """State accuracy by checkpoint kind, plus the all-rows total.

    Precision is tp/(tp+fp) over facts, recall tp/(tp+fn), exact the share
    of checkpoints whose profile matched gold exactly. The `failed` column
    counts statements the loop gave up on along a checkpoint's path: those
    reach the state table as misses, not as silent gaps.
    """
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r)
    lines = [_table_line("kind", None)]
    for kind in sorted(by_kind):
        lines.append(_table_line(kind, by_kind[kind]))
    if len(by_kind) > 1:
        lines.append(_table_line("all", rows))
    return "\n".join(lines)


def _table_line(label, rows) -> str:
    if rows is None:
        return (f"{'kind':<26}{'n':>4}{'exact':>8}{'prec':>8}{'recall':>8}"
                f"{'missing':>9}{'extra':>7}{'wrong':>7}{'failed':>8}")
    tp = sum(r["tp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    exact = sum(bool(r["exact"]) for r in rows)
    n = len(rows)
    prec = f"{tp / (tp + fp):>7.1%}" if tp + fp else f"{'-':>8}"
    rec = f"{tp / (tp + fn):>7.1%}" if tp + fn else f"{'-':>8}"
    return (f"{label:<26}{n:>4}{exact / n:>7.1%} {prec} {rec} "
            f"{fn:>8}{fp:>7}"
            f"{sum(len(r['wrong_value']) for r in rows):>7}"
            f"{sum(r.get('failed', 0) for r in rows):>8}")


def make_client():
    """The one Ollama client a corpus run makes.

    Bounded on purpose: a dead socket after a sleep raises instead of
    waiting, which run_turn already counts as a failed attempt.
    """
    import ollama
    return ollama.Client(timeout=config.REQUEST_TIMEOUT_S)


def main(argv=None) -> int:
    import argparse
    import subprocess

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--limit", type=int, default=None,
                   help="only the first N pairs (2N trajectories)")
    p.add_argument("--skip", nargs="+", default=[], metavar="TRAJECTORY_ID",
                   help="leave these trajectories (and their checkpoints) out")
    p.add_argument("--model", default=None, help=f"default {config.MODEL}")
    p.add_argument("--out", default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="continue the journal at --out instead of starting over")
    args = p.parse_args(argv)

    from memproj import replay
    trajectories, checkpoints = replay.load_corpus()
    out = args.out or str(config.OUT_DIR / "extraction_results.jsonl")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # A resume keeps every trajectory the journal finished and reruns the
    # one that was interrupted, so no checkpoint is journalled twice.
    prior, skip = ([], set())
    if args.resume:
        prior, skip = resume_state(out, checkpoints)
    with open(out, "a" if args.resume else "w",
              encoding="utf-8", newline="\n") as fh:
        rows = prior + run(trajectories, checkpoints, make_client(),
                           model=args.model, limit=args.limit,
                           verbose=args.verbose, journal=fh,
                           skip=skip | set(args.skip))

    try:
        version = subprocess.run(["ollama", "--version"], capture_output=True,
                                 text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        version = "unknown"
    print(json.dumps({"model": args.model or config.MODEL,
                      "num_ctx": config.NUM_CTX,
                      "temperature": config.TEMPERATURE,
                      "seed": config.SEED, "limit": args.limit,
                      "resumed_trajectories": len(skip),
                      "ollama": version}, sort_keys=True))
    print(summarise(rows))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
