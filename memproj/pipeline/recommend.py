"""The fixed recommender: search, validate, choose.

This module is the same for every memory method. Only `build_context()`
changes between experiments (README, Method interface), so a difference in the
recommendation can be attributed to the memory method and nothing else.

Candidate policy -- decided 2026-10-21, recorded here and in README.md:

    A resolved profile (from latest_wins, proposed or gold) FILTERS the
    catalogue before ranking: only recipes that `recipes.validate()` accepts
    are offered. `profile is None` (the four free-text baselines) and `{}`
    (a structured method that believes nothing is active) both offer the
    whole catalogue.

The consequence is that the gold baseline scores 100% on constraint
satisfaction by construction, which is what makes it a ceiling: the number
is not "how good the model is" but "can the catalogue satisfy this profile at
all", and `headroom()` reports how many recipes survived the filter.
"""

import json
import time
from typing import Callable, Dict, List, Optional, Sequence

from memproj import config
from memproj import recipes
from memproj.recipes import validate as validate_recipe

# Attributes the recommender ranks or explains by, in the order it names them.
PROFILE_ATTRIBUTES = ("diet", "spice_level", "excluded_ingredient")

# Candidate policies. "filter" is the one every published number was measured
# under; "rank" is the ablation (see `memproj.pipeline.policy`).
POLICIES = ("filter", "rank")


def _as_sets(profile: Dict[str, Sequence[str]]) -> Dict[str, set]:
    """The checkpoint format ({attr: sorted list}) as sets for validate()."""
    return {attribute: set(values) for attribute, values in profile.items()}


def _spice_ceiling(profile: Dict[str, Sequence[str]]) -> Optional[int]:
    """The hottest level the profile allows, or None if it says nothing."""
    allowed = set(profile.get("spice_level", ())) & set(recipes.SPICE_ORDER)
    if not allowed:
        return None
    return max(recipes.SPICE_ORDER[value] for value in allowed)


def _rank_key(recipe: dict, ceiling: Optional[int]):
    """Closest to the requested heat first, then id. Deterministic.

    The ceiling is "at most this hot", not "exactly this hot", so this only
    breaks ties: every candidate already passed validate(). Ordering the
    milder options last keeps a mild request from being answered with the
    blandest dish in the catalogue, without hiding any candidate.
    """
    if ceiling is None:
        return (0, recipe["id"])
    distance = abs(recipes.SPICE_ORDER.get(recipe.get("spice_level"), 0)
                   - ceiling)
    return (distance, recipe["id"])


def _violations(recipe: dict, profile: Dict[str, Sequence[str]]) -> int:
    """How many profile constraints this recipe breaks. 0 means valid."""
    _ok, violations = validate_recipe(recipe, _as_sets(profile))
    return len(violations)


def recipe_search(profile: Optional[Dict[str, Sequence[str]]],
                  catalogue: Sequence[dict],
                  top_k: Optional[int] = config.TOP_K_RECIPES,
                  policy: str = "filter") -> List[dict]:
    """Candidate recipes for one checkpoint, best first.

    `profile is None` -> every recipe is a candidate (free-text method).
    `profile == {}`   -> every recipe is a candidate (nothing active).
    Otherwise, `policy` decides what the model is offered:

    ``"filter"`` (default) -> only recipes satisfying `profile`.
    ``"rank"``             -> the WHOLE catalogue, ordered by how many of the
                              profile's constraints each recipe breaks (fewest
                              first), then by heat distance, then id.

    ``"rank"`` exists so the candidate filter's contribution can be measured
    rather than assumed: under ``"filter"`` a wrong profile removes the right
    dish from the candidate set, which turns a memory error into an
    unwinnable recommendation and hides it behind a "lucky" score. Under
    ``"rank"`` the constraint information is still in the order, but nothing
    is excluded. See `memproj.pipeline.policy`.

    `top_k=None` returns every candidate; that is what `headroom()` counts.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
    if profile is None or not profile:
        candidates = list(catalogue)
    elif policy == "filter":
        candidates = recipes.valid_recipes(catalogue, _as_sets(profile))
    else:
        candidates = list(catalogue)
    ceiling = _spice_ceiling(profile or {})
    if policy == "rank" and profile:
        candidates.sort(key=lambda r: (_violations(r, profile),
                                       _rank_key(r, ceiling)))
    else:
        candidates.sort(key=lambda r: _rank_key(r, ceiling))
    if top_k is not None:
        candidates = candidates[:top_k]
    return candidates


def headroom(profile: Optional[Dict[str, Sequence[str]]],
             catalogue: Sequence[dict]) -> int:
    """How many recipes satisfy the profile before TOP_K truncation.

    The metric's ceiling, not the recommender's choice set: a checkpoint with
    only three survivors can only be failed three ways.
    """
    return len(recipe_search(profile, catalogue, top_k=None))


def validate_choice(recipe_id: str, profile: Dict[str, Sequence[str]],
                    catalogue: Sequence[dict]) -> tuple:
    """`recipes.validate` for one recipe id. Returns (ok, violations)."""
    for recipe in catalogue:
        if recipe["id"] == recipe_id:
            return validate_recipe(recipe, _as_sets(profile))
    return False, [f"{recipe_id}: not in the catalogue"]


def explain(recipe: dict, profile: Optional[Dict[str, Sequence[str]]]) -> str:
    """A mechanical explanation naming every constraint that was applied.

    Built from the profile, never from the model: the provenance audit reads
    this text, so it may only claim what the profile and the recipe support.
    """
    profile = profile or {}
    reasons = []
    if "vegetarian" in profile.get("diet", ()):
        reasons.append("it is vegetarian")
    ceiling = _spice_ceiling(profile)
    if ceiling is not None:
        hottest = max((v for v in profile["spice_level"] if v in recipes.SPICE_ORDER),
                      key=recipes.SPICE_ORDER.get)
        reasons.append(f"it is no hotter than {hottest}")
    excluded = sorted(profile.get("excluded_ingredient", ()))
    if excluded:
        reasons.append("it contains none of " + ", ".join(excluded))
    if not reasons:
        return f"I picked {recipe['name']} because you have no active restrictions."
    return f"I picked {recipe['name']} because " + "; ".join(reasons) + "."


# --- the model path ---------------------------------------------------------
# Used by every method whose profile is None (the free-text baselines). The gold
# baseline below never reaches it: README, Predictions format pins tokens=0 for
# gold, which means no inference.

CHOICE_SYSTEM = (
    "You recommend one meal. You receive the user's request, a memory "
    "context, and a numbered list of candidate recipes. Reply with ONE JSON "
    "object and nothing else, with keys: action ('recommend' or 'clarify'), "
    "recipe_id (an id from the candidate list, only when recommending), "
    "question (only when clarifying), explanation (one sentence). Recommend "
    "exactly one candidate. Ask a question only when the context cannot be "
    "resolved; never guess."
)


def build_prompt(request_text: str, context_text: str,
                 candidates: Sequence[dict]) -> str:
    """The one prompt builder every method shares.

    The candidate list is rendered with the attributes validate() checks, so
    the model can reason without a second tool call; their order is the
    ranking, so the first line is the recommended default.
    """
    lines = [
        f"User request: {request_text}",
        "",
        "Memory context:",
        context_text.strip() or "(none)",
        "",
        "Candidates:",
    ]
    for recipe in candidates:
        ingredients = ", ".join(sorted(recipe.get("ingredients", ())))
        lines.append(
            f"- {recipe['id']}: {recipe['name']} "
            f"(spice={recipe.get('spice_level', '?')}, "
            f"vegetarian={'yes' if recipe.get('vegetarian') else 'no'}, "
            f"ingredients: {ingredients})"
        )
    return "\n".join(lines)


def parse_choice(text: str, candidate_ids: Sequence[str]) -> dict:
    """Parse the model's JSON reply, or raise ValueError.

    Strict on purpose: a malformed reply is a data point for the
    failure analysis, not something to paper over with a default pick.
    """
    if not isinstance(text, str):
        raise ValueError("model reply must be a string")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError(f"model reply contains no JSON object: {text!r}")
    try:
        choice = json.loads(text[start:end + 1])
    except ValueError as exc:
        raise ValueError(f"model reply is not valid JSON: {exc}") from None
    if not isinstance(choice, dict):
        raise ValueError("model reply must be a JSON object")

    action = choice.get("action")
    if action not in ("recommend", "clarify"):
        raise ValueError(f"action must be 'recommend' or 'clarify', got {action!r}")
    explanation = choice.get("explanation")
    if not isinstance(explanation, str):
        raise ValueError("explanation must be a string")

    if action == "recommend":
        recipe_id = choice.get("recipe_id")
        if recipe_id not in candidate_ids:
            raise ValueError(
                f"recipe_id {recipe_id!r} is not one of the candidates "
                f"{list(candidate_ids)}"
            )
        if choice.get("question") not in (None, ""):
            raise ValueError("a recommendation must not carry a question")
        return {"action": action, "recipe_id": recipe_id, "question": None,
                "explanation": explanation}

    question = choice.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("a clarification requires a question")
    if choice.get("recipe_id") not in (None, ""):
        raise ValueError("a clarification must not carry a recipe_id")
    return {"action": action, "recipe_id": None, "question": question.strip(),
            "explanation": explanation}


def retry_transport(call: Callable[[], dict]) -> dict:
    """Run one Ollama call, retrying a transport failure a bounded number of times.

    `config.REQUEST_TIMEOUT_S` makes a stalled call fail; this keeps the
    failure from ending the run. The retry re-asks rather than resumes, which
    is safe for every call the pipeline makes: none has a side effect to
    duplicate. A programming error inside `call` is not caught -- only
    httpx's transport family is (timeouts included, since httpx derives them).
    """
    import httpx

    for attempt in range(config.TOOL_RETRIES + 1):
        try:
            return call()
        except httpx.TransportError:
            if attempt == config.TOOL_RETRIES:
                raise
            time.sleep(min(2 ** attempt, 10))


def chat(messages: List[dict], model: Optional[str] = None) -> dict:
    """One Ollama call. Imported lazily so rule-logic tests need no client.

    The client carries `config.REQUEST_TIMEOUT_S` so a stuck GPU job fails
    instead of blocking the run; a per-call timeout is not part of the
    library's request surface.

    A TRANSPORT error is retried `config.TOOL_RETRIES` times with a fresh
    client: the timeout fires when the machine stalls (a sleeping host woke
    mid-request, a model swap), and one such row must not end a six-hour run.
    The retry restarts the call, so a reply whose bytes were lost is simply
    asked for again -- these calls have no side effects to duplicate.
    """
    import ollama

    def send():
        client = ollama.Client(timeout=config.REQUEST_TIMEOUT_S)
        return client.chat(
            model=model or config.MODEL,
            messages=messages,
            options=config.ollama_options(),
            think=config.PIPELINE_THINK,
        )

    response = retry_transport(send)
    tokens = ((response.get("prompt_eval_count") or 0)
              + (response.get("eval_count") or 0))
    return {"text": response["message"]["content"], "tokens": tokens}


def recommend(profile: Optional[Dict[str, Sequence[str]]],
              context_text: str,
              request_text: str,
              catalogue: Sequence[dict],
              chat_fn: Callable[[List[dict]], dict] = chat,
              top_k: int = config.TOP_K_RECIPES,
              retry: Optional[str] = None) -> dict:
    """Search, ask the model to choose, and return the choice.

    Returns {action, recipe_id, question, explanation, tokens}. Latency is
    the caller's to measure. If the profile leaves no candidate at all, the
    only honest action is to ask which constraint to relax -- no model call
    is made, so `tokens` is 0.

    `retry` is the rejection reason from a previous malformed reply. It is
    appended as one extra message so the retry is a different prompt rather
    than the same one at temperature 0, which would usually reproduce the
    same broken JSON.
    """
    candidates = recipe_search(profile, catalogue, top_k=top_k)
    if not candidates:
        return {
            "action": "clarify",
            "recipe_id": None,
            "question": ("Nothing in the catalogue fits your current "
                         "restrictions. Which one should I relax?"),
            "explanation": "Every candidate was rejected by the active profile.",
            "tokens": 0,
        }

    prompt = build_prompt(request_text, context_text, candidates)
    messages = [
        {"role": "system", "content": CHOICE_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    if retry is not None:
        messages.append({
            "role": "user",
            "content": (f"Your previous reply was rejected: {retry}. Reply "
                        f"again with exactly one JSON object."),
        })
    reply = chat_fn(messages)
    choice = parse_choice(reply["text"], [r["id"] for r in candidates])
    choice["tokens"] = int(reply.get("tokens") or 0)
    return choice
