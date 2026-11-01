"""The four free-text baselines.

Each class implements README, Method interface's `build_context()` and returns
`profile=None`. That is what makes them baselines: they hand the recommender
prose, so the recommender has to do the remembering inside its own context
window, while `latest_wins`/`proposed`/`gold` hand it a resolved profile.

What must be identical across every method is the recommender; what must
differ is only this. So each baseline makes exactly the choices its name
implies and nothing else:

    recent_only      the last `RECENT_ONLY_TURNS` turns, verbatim
    full_history     every turn up to the checkpoint, verbatim
    rolling_summary  a model-written summary of everything older than the
                     recent window, then the recent window verbatim
    retrieval        the `RETRIEVAL_TOP_K` turns most similar to the request

Two properties are load-bearing:

* **The cutoff is `turn["ts"] <= checkpoint["now"]`.** A baseline that can
  see a turn the checkpoint has not reached is not a memory method, it is a
  oracle. The comparison is on the ISO strings, which is why README
  section 5 fixes the format.
* **The corpus is small** (18-30 turns per history), so `full_history` fits
  in `NUM_CTX` with room to spare. There is deliberately no truncation path
  here: a silent truncation would quietly turn full_history into a second
  recent_only, and the comparison would mean nothing.

The two model calls this module can make -- the summariser and the embedder
-- are injected. Tests pass fakes; no rule-logic test touches a model.
"""

from math import sqrt
from typing import Callable, Dict, List, Optional, Sequence

from memproj import config
from memproj.pipeline import recommend


# --- shared helpers ---------------------------------------------------------

def visible_turns(trajectory: dict, now: str) -> List[dict]:
    """Every turn at or before `now`, in generation order.

    String comparison, not parsing: README, Recipe catalogue pins the naive
    ISO format precisely so this is safe.
    """
    return [turn for turn in trajectory.get("turns", ()) if turn["ts"] <= now]


def format_turns(turns: Sequence[dict]) -> str:
    """One line per turn, with the turn id so a transcript is debuggable."""
    return "\n".join(
        f"[{turn['turn_id']}] {turn['speaker']}: {turn['text']}"
        for turn in turns
    ) or "(none)"


def evidence_turn_ids(turns: Sequence[dict]) -> List[str]:
    """User turns that carry a preference statement.

    Assistant turns and filler carry no event, so citing them would claim
    support that is not there. Everything the method actually read *does*
    appear here, distractors included -- whether it should have cited them
    is what the provenance metric decides.
    """
    return [turn["turn_id"] for turn in turns
            if turn["speaker"] == "user" and turn.get("event_id")]


def _context(text: str, evidence: List[str]) -> dict:
    """The exact three keys README, Method interface allows. Nothing else."""
    return {"context_text": text,
            "profile": None,
            "evidence_turn_ids": evidence}


# --- 1. recent only ---------------------------------------------------------

class RecentOnly:
    """The last N turns and nothing else. The "no memory" control."""

    def __init__(self, budget: int = config.RECENT_ONLY_TURNS):
        self.budget = budget

    def build_context(self, trajectory: dict, checkpoint: dict) -> dict:
        turns = visible_turns(trajectory, checkpoint["now"])[-self.budget:]
        return _context("Recent conversation:\n" + format_turns(turns),
                        evidence_turn_ids(turns))


# --- 2. full history --------------------------------------------------------

class FullHistory:
    """Every turn up to the checkpoint. The "no forgetting" control."""

    def build_context(self, trajectory: dict, checkpoint: dict) -> dict:
        turns = visible_turns(trajectory, checkpoint["now"])
        return _context("Full conversation:\n" + format_turns(turns),
                        evidence_turn_ids(turns))


# --- 3. rolling summary -----------------------------------------------------

SUMMARY_SYSTEM = (
    "You compress a conversation for later personalisation. Preserve every "
    "stated preference, restriction, and the person it belongs to, together "
    "with anything that limits when it applies. Drop small talk. Write plain "
    "prose, no headings, at most 120 words."
)


def summarise(transcript: str) -> str:
    """One Ollama call. Imported lazily so rule-logic tests need no client."""
    import ollama

    def send():
        client = ollama.Client(timeout=config.REQUEST_TIMEOUT_S)
        return client.chat(
            model=config.MODEL,
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": transcript},
            ],
            options=config.ollama_options(),
            think=config.PIPELINE_THINK,
        )

    response = recommend.retry_transport(send)
    return response["message"]["content"].strip()


class RollingSummary:
    """A summary of the old turns plus the recent window verbatim.

    The summary is cached per (trajectory, prefix length) because the
    checkpoints of one trajectory share their prefix: without the cache the
    same turns would be summarised once per checkpoint, and a 193-checkpoint
    run would pay for ~400 redundant model calls.
    """

    def __init__(self, summarise_fn: Callable[[str], str] = summarise,
                 budget: int = config.RECENT_ONLY_TURNS):
        self.summarise_fn = summarise_fn
        self.budget = budget
        self._cache: Dict[tuple, str] = {}

    def build_context(self, trajectory: dict, checkpoint: dict) -> dict:
        turns = visible_turns(trajectory, checkpoint["now"])
        if self.budget > 0:
            prefix, recent = turns[:-self.budget], turns[-self.budget:]
        else:
            prefix, recent = turns, []
        if not prefix:
            text = "Recent conversation:\n" + format_turns(recent)
        else:
            key = (trajectory["trajectory_id"], len(prefix))
            if key not in self._cache:
                self._cache[key] = self.summarise_fn(format_turns(prefix))
            text = ("Summary of the earlier conversation:\n"
                    f"{self._cache[key]}\n\n"
                    "Recent conversation:\n" + format_turns(recent))
        # The summariser read the prefix, so those turns are citable too.
        return _context(text, evidence_turn_ids(turns))


# --- 4. retrieval -----------------------------------------------------------

def embed(texts: Sequence[str], model: Optional[str] = None) -> List[List[float]]:
    """One Ollama embedding call. Imported lazily, like `summarise`."""
    import ollama

    def send():
        client = ollama.Client(timeout=config.REQUEST_TIMEOUT_S)
        return client.embed(model=model or config.EMBED_MODEL,
                            input=list(texts))

    response = recommend.retry_transport(send)
    return [[float(x) for x in vector] for vector in response["embeddings"]]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Plain cosine similarity. The vectors are small enough to skip numpy."""
    dot = sum(a * b for a, b in zip(left, right))
    norm = sqrt(sum(a * a for a in left)) * sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


class Retrieval:
    """The turns most similar to the checkpoint's request.

    Similarity is the whole method, so it ranks on the request text alone --
    no preference keywords are injected, or the baseline would be reading
    the answer key.
    """

    def __init__(self, embed_fn: Callable[[Sequence[str]], List[List[float]]] = embed,
                 top_k: int = config.RETRIEVAL_TOP_K):
        self.embed_fn = embed_fn
        self.top_k = top_k
        self._index: Dict[tuple, List[List[float]]] = {}
        self._queries: Dict[str, List[float]] = {}

    def _vectors(self, trajectory: dict, turns: Sequence[dict]) -> List[List[float]]:
        # `turns` is always a prefix of the trajectory's turns, so its length
        # identifies it exactly.
        key = (trajectory["trajectory_id"], len(turns))
        if key not in self._index:
            self._index[key] = self.embed_fn([turn["text"] for turn in turns])
        return self._index[key]

    def _query(self, request_text: str) -> List[float]:
        if request_text not in self._queries:
            self._queries[request_text] = self.embed_fn([request_text])[0]
        return self._queries[request_text]

    def build_context(self, trajectory: dict, checkpoint: dict) -> dict:
        turns = visible_turns(trajectory, checkpoint["now"])
        if not turns:
            return _context("Retrieved context:\n(none)", [])

        vectors = self._vectors(trajectory, turns)
        query = self._query(checkpoint["request_text"])
        # Descending score, ties broken on position, then restored to
        # chronological order so the transcript reads like a conversation.
        ranked = sorted(range(len(turns)),
                        key=lambda i: (-cosine(query, vectors[i]), i))
        chosen = sorted(ranked[:self.top_k])
        selected = [turns[i] for i in chosen]
        return _context(
            "Moments retrieved from earlier in this conversation:\n"
            + format_turns(selected),
            evidence_turn_ids(selected),
        )


# --- the registry -----------------------------------------------------------

def baselines() -> Dict[str, object]:
    """Fresh instances, so each run gets empty caches.

    Construct these once per run, not once per checkpoint: the caches are
    what keep a 193-checkpoint run from paying for the same summary or the
    same embedding twice.
    """
    return {
        "recent_only": RecentOnly(),
        "full_history": FullHistory(),
        "rolling_summary": RollingSummary(),
        "retrieval": Retrieval(),
    }
