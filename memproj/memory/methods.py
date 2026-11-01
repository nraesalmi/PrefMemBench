"""The two structured methods: `proposed` and `latest_wins`.

README, Method interface gives every method one entry point,
`build_context(trajectory, checkpoint) -> {context_text, profile,
evidence_turn_ids}`. This module implements it twice, over the same
extraction, so the experiment varies the resolution rule and nothing else:

  proposed     every resolver rule: subject, scope, time, supersede, correction
  latest_wins  the newest statement per attribute, whoever it is about,
               whenever it was said -- the baseline that has no rules at all

**One extraction pass per trajectory.** `build_context` is called once per
(checkpoint, method), but the extraction it depends on is per turn and does
not depend on the checkpoint: each user turn is read at its own timestamp,
and `recent_exchanges` only ever shows the model earlier turns. So
`ExtractionCache` keeps one store per trajectory, built the first time a
method asks for it, and both methods and all of that trajectory's checkpoints
share it. Extracting per checkpoint would cost ~410 model passes instead of
120 and re-ask the same question at temperature 0, which Ollama does not
answer identically twice.

The store is built from the WHOLE trajectory, including turns after a given
checkpoint's `now`. That is safe because every read is cut at `now` --
`active_profile` by `_applies`, which keeps a superseded row until the turn
that replaced it, and by `_open`, which keeps a window shut until it opens;
`latest_profile` by `valid_from`, `pending_clarification` by the question's
`ts`. It is also deliberate: the
alternative, rebuilding the store per checkpoint, would re-extract and lose
the property this module exists to keep.
"""

import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Sequence

from memproj.memory.extract import HISTORY_TURNS, Extractor
from memproj.resolver import active_records, pending_clarification
from memproj.store import Store
from memproj.tools import EXTRACTION_SYSTEM, TOOL_SCHEMAS


# --- the shared extraction ---------------------------------------------------

class Trace:
    """One trajectory, extracted once: a store plus what went wrong in it.

    `failures` mirrors the extraction journal's field of the same name: the
    statements the loop gave up on. They are counted here and not retried --
    a method that forgot a statement should look like a method that forgot a
    statement, not like one the harness quietly repaired.

    With `path`, the finished store is written there, atomically, so a later
    run can `Trace.load` it instead of asking the model the same questions
    again. An extraction is the expensive half of the ablations: five
    resolver variants are five readings of ONE pass, not five passes.
    """

    def __init__(self, trajectory: dict, extractor: Extractor,
                 path: Optional[str] = None):
        turns, texts = _turns_and_texts(trajectory)
        self.failures: List[dict] = []
        self.store = Store(":memory:").create()
        try:
            self.store.turn_texts = texts
            for turn in turns:
                self.store.add_turn(turn)
            for turn in turns:
                if turn.speaker != "user":
                    continue
                ok, result = extractor.run_turn(self.store, turn)
                if not ok:
                    self.failures.append({"turn_id": turn.turn_id,
                                          "error": result.get("error")})
            if path is not None:
                self.store.backup_to(path)
        except Exception:
            self.store.close()
            raise

    @classmethod
    def load(cls, trajectory: dict, path: str) -> "Trace":
        """An extraction written by an earlier run. Makes no model call."""
        trace = cls.__new__(cls)
        _turns, texts = _turns_and_texts(trajectory)
        trace.failures = []
        trace.store = Store(path)
        trace.store.turn_texts = texts
        return trace

    def close(self) -> None:
        self.store.close()


def _turns_and_texts(trajectory: dict):
    """(parsed turns, turn_id -> text). The turns table keeps no words."""
    from memproj import replay     # replay imports store/tools; keep it lazy

    turns = [replay._parse_turn(turn) for turn in trajectory["turns"]]
    return turns, {turn.turn_id: turn.text for turn in turns}


def _prompt_signature() -> str:
    """A hash of everything the extraction model is shown before the turn.

    A reworded prompt answers differently, so an extraction cached before
    the edit is not the same experiment. Hashing the prompt -- rather than
    bumping a version number by hand -- is what makes that impossible to
    forget.
    """
    payload = json.dumps({"system": EXTRACTION_SYSTEM, "tools": TOOL_SCHEMAS,
                          "history_turns": HISTORY_TURNS}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class ExtractionCache:
    """`trajectory_id -> Trace`, built on first use and closed as a unit.

    With `directory`, every extraction is also written to
    `<directory>/<trajectory_id>.sqlite` beside one manifest. A later cache
    reads those files back when -- and only when -- the manifest still
    matches: same corpus, model, num_ctx, temperature, seed and extraction
    prompt. Anything else answers a different question and is re-extracted,
    because a cache that silently serves yesterday's answers under today's
    settings is worse than no cache.

    A mismatch DELETES the files before the pass starts. Checking the
    manifest is not enough on its own: it is rewritten after the first
    trajectory of a fresh pass, so a file left over from another corpus
    matches it from then on. That happened once during development: a corpus
    change re-extracted one trajectory, stamped the new manifest, and every
    later file then "matched" it.

    `stats()` is what a run reports: how much it had to pay for and what it
    gave up on. Failures are carried, not retried -- see `Trace`.
    """

    def __init__(self, client, model: Optional[str] = None,
                 directory: Optional[str] = None,
                 fingerprint: Optional[str] = None):
        if directory and not fingerprint:
            raise ValueError(
                "a cached extraction needs the corpus fingerprint: the same "
                "trajectory ids answer different questions after a re-seed")
        self.client = client
        self.extractor = Extractor(client=client, model=model)
        self.directory = str(directory) if directory else None
        self.fingerprint = fingerprint
        self._traces: Dict[str, Trace] = {}
        self._manifest: Optional[dict] = None
        self._stats = {"trajectories": 0, "loaded": 0, "extracted": 0,
                       "turns": 0, "failures": [], "tokens": 0,
                       "seconds": 0.0, "discarded": 0}
        if self.directory:
            os.makedirs(self.directory, exist_ok=True)
            self._manifest = _read_json(self._manifest_path())
            if self._manifest != self._payload():
                self._discard_stale()

    def trace(self, trajectory: dict) -> Trace:
        trajectory_id = trajectory["trajectory_id"]
        if trajectory_id not in self._traces:
            self._traces[trajectory_id] = self._build(trajectory)
        return self._traces[trajectory_id]

    def stats(self) -> dict:
        """What the extraction cost, and what it gave up on."""
        return dict(self._stats, failures=list(self._stats["failures"]))

    def close(self) -> None:
        for trace in self._traces.values():
            trace.close()
        self._traces.clear()

    def __enter__(self) -> "ExtractionCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- on disk ----------------------------------------------------------

    def _build(self, trajectory: dict) -> Trace:
        trajectory_id = trajectory["trajectory_id"]
        path = self._path(trajectory_id)
        if path and os.path.exists(path) and self._manifest_matches():
            trace = Trace.load(trajectory, path)
            self._stats["loaded"] += 1
        else:
            spent, started = self.extractor.tokens, time.monotonic()
            trace = Trace(trajectory, self.extractor, path=path)
            self._stats["extracted"] += 1
            self._stats["tokens"] += self.extractor.tokens - spent
            self._stats["seconds"] += time.monotonic() - started
            if path:
                self._write_manifest()
        self._stats["trajectories"] += 1
        self._stats["turns"] += sum(1 for turn in trajectory["turns"]
                                    if turn.get("speaker") == "user")
        for failure in trace.failures:
            self._stats["failures"].append(
                dict(failure, trajectory_id=trajectory_id))
        return trace

    def _path(self, trajectory_id: str) -> Optional[str]:
        if not self.directory:
            return None
        return os.path.join(self.directory, f"{trajectory_id}.sqlite")

    def _manifest_path(self) -> str:
        return os.path.join(self.directory, "manifest.json")

    def _payload(self) -> dict:
        return {"corpus": self.fingerprint,
                "model": self.extractor.model,
                "num_ctx": self.extractor.num_ctx,
                "temperature": self.extractor.temperature,
                "seed": self.extractor.seed,
                "prompt": _prompt_signature()}

    def _manifest_matches(self) -> bool:
        if self._manifest is None:
            self._manifest = _read_json(self._manifest_path()) or {}
        return self._manifest == self._payload()

    def _discard_stale(self) -> None:
        """Delete extractions that were made from other inputs.

        They are worthless here and dangerous to keep: `_write_manifest`
        runs after the first trajectory of the new pass, so any file it did
        not write would match from then on. Read-only users (the ablation
        table opens the cache with `client=None`) get an error instead, not
        a quiet re-extraction they cannot pay for.
        """
        if self.client is None:
            raise RuntimeError(
                f"the extraction cache in {self.directory} was built for "
                f"other inputs ({self._mismatch()}); rerun the pipeline to "
                f"re-extract it")
        removed = 0
        for name in sorted(os.listdir(self.directory)):
            if name.endswith(".sqlite"):
                os.remove(os.path.join(self.directory, name))
                removed += 1
        if os.path.exists(self._manifest_path()):
            os.remove(self._manifest_path())
        self._manifest = {}
        self._stats["discarded"] = removed

    def _mismatch(self) -> str:
        """Which manifest fields differ, for the error a read-only user gets."""
        want = self._payload()
        have = self._manifest or {}
        return ", ".join(f"{key} {have.get(key)!r} != {want[key]!r}"
                         for key in sorted(want) if have.get(key) != want[key])

    def _write_manifest(self) -> None:
        self._manifest = self._payload()
        path = self._manifest_path()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8",
                  newline=chr(10)) as fh:
            json.dump(self._manifest, fh, sort_keys=True)
            fh.write(chr(10))
        os.replace(tmp, path)


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --- latest wins ------------------------------------------------------------

def latest_rows(store: Store, now: str) -> Dict[str, dict]:
    """The most recent row per attribute as of `now`, rules ignored.

    "Most recent" is `(valid_from, pref_id)`: a statement that starts later
    wins, and two that start at the same instant are ordered the way the
    store wrote them so the answer never depends on row order. Nothing here
    looks at subject, scope, `valid_until`, `status` or `edited` -- that
    absence IS the baseline. A statement that has not started yet (`valid_from`
    after `now`) has not been made as of this checkpoint and is skipped.
    """
    best: Dict[str, tuple] = {}
    for row in store.all_prefs():
        if row["valid_from"] > now:
            continue
        key = (row["valid_from"], row["pref_id"])
        if row["attribute"] not in best or key > best[row["attribute"]][0]:
            best[row["attribute"]] = (key, row)
    return {attribute: row for attribute, (_key, row) in best.items()}


# --- rendering --------------------------------------------------------------

def render(profile: Dict[str, Sequence[str]], now: str,
           sources: Optional[Dict[tuple, Sequence[str]]] = None,
           pending: Optional[dict] = None) -> str:
    """The profile as the recommender sees it. Deterministic, no wall clock.

    One line per fact, attributes and values sorted, the source turn ids
    appended where the method has them. `now` is the checkpoint's own time,
    so two runs of the same checkpoint render the same bytes.
    """
    lines = [f"Active preferences as of {now}:"]
    if profile:
        for attribute in sorted(profile):
            for value in sorted(profile[attribute]):
                cited = sorted((sources or {}).get((attribute, value), ()))
                suffix = f" [{', '.join(cited)}]" if cited else ""
                lines.append(f"- {attribute}: {value}{suffix}")
    else:
        lines.append("- none stated")
    if pending is not None:
        lines.append(f"Unresolved question ({pending['source_turn_id']}): "
                     f"{pending['question']}")
    return "\n".join(lines)


def _as_lists(profile: Dict[str, set]) -> Dict[str, List[str]]:
    return {attribute: sorted(values)
            for attribute, values in sorted(profile.items()) if values}


# --- the methods ------------------------------------------------------------

class Proposed:
    """Every rule, plus provenance and the pending question.

    `context_text` is the resolved profile with each fact's source turn, and
    one line for an unanswered question. The recommender cannot ask what it
    cannot see, and the method interface has no field for a question -- so
    the pending line is how a clarify action reaches it.
    """

    def __init__(self, cache: ExtractionCache):
        self.cache = cache

    def build_context(self, trajectory: dict, checkpoint: dict) -> dict:
        trace = self.cache.trace(trajectory)
        now, person = checkpoint["now"], checkpoint["person"]
        records = active_records(trace.store, person, now)
        sources: Dict[tuple, set] = {}
        for record in records:
            sources.setdefault((record["attribute"], record["value"]),
                               set()).add(record["source_turn_id"])
        profile = _as_lists(_profile_from(records))
        pending = pending_clarification(trace.store, now)
        return {
            "context_text": render(profile, now, sources=sources, pending=pending),
            "profile": profile,
            "evidence_turn_ids": sorted(set().union(*sources.values())
                                        if sources else set()),
        }


class LatestWins:
    """The newest statement per attribute, and nothing else.

    No profile sources: this baseline has no notion of provenance, and giving
    it one would be giving it a rule. Evidence is still returned -- the
    statement that supplied each value -- because README, Method interface
    requires the key.
    """

    def __init__(self, cache: ExtractionCache):
        self.cache = cache

    def build_context(self, trajectory: dict, checkpoint: dict) -> dict:
        trace = self.cache.trace(trajectory)
        now = checkpoint["now"]
        rows = latest_rows(trace.store, now)
        profile = {attribute: [row["value"]]
                   for attribute, row in sorted(rows.items())}
        return {
            "context_text": render(profile, now),
            "profile": profile,
            "evidence_turn_ids": sorted({row["source_turn_id"]
                                         for row in rows.values()}),
        }


def _profile_from(records: Sequence[dict]) -> Dict[str, set]:
    profile: Dict[str, set] = {}
    for record in records:
        profile.setdefault(record["attribute"], set()).add(record["value"])
    return profile


def structured(cache: ExtractionCache) -> Dict[str, object]:
    """Fresh instances over one cache, the shape `run.build_prediction` wants."""
    return {"latest_wins": LatestWins(cache), "proposed": Proposed(cache)}
