"""Every knob the experiment depends on, in one place.

Nothing else in the repo hardcodes a model name, a context length, a path, or
a seed. If you find yourself typing a literal that belongs here, import it
instead -- `tests/test_config.py` asserts that the model-calling modules have
no string literals that look like Ollama tags or model names.
"""

from pathlib import Path

# --- package paths ----------------------------------------------------------

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent

DATA_DIR = PKG_DIR / "data"
OUT_DIR = REPO_ROOT / "out"
RUNS_DIR = OUT_DIR / "runs"          # one SQLite file per (trajectory, method)

CATALOGUE_PATH = DATA_DIR / "recipes_hummus.json"
# The HUMMUS download the catalogue was cut from (507,336 rows, 317 MB,
# gitignored -- its licence allows research use but not redistribution).
# `python -m memproj.catalogue` rebuilds CATALOGUE_PATH from this.
HUMMUS_RECIPES_PATH = OUT_DIR / "hummus" / "pp_recipes.zip"
CATALOGUE_SEED = 1
HISTORIES_PATH = OUT_DIR / "histories.jsonl"
CHECKPOINTS_PATH = OUT_DIR / "checkpoints.jsonl"
PREDICTIONS_PATH = OUT_DIR / "predictions.jsonl"
CHALLENGE_PATH = DATA_DIR / "challenge.jsonl"

# One SQLite file per trajectory plus a manifest, written by ExtractionCache.
# Step 7's ablations read it instead of paying for the extraction again.
EXTRACTION_DIR = OUT_DIR / "extraction"

# --- generation -------------------------------------------------------------

# Matched with the generator's own defaults so a bare `python -m
# memproj.generate` reproduces the committed data.
N_BASE = 60
SESSIONS = 3
TURNS_PER_SESSION = 8
GEN_SEED = 1

# The wall-clock anchor every generated history counts from. A fixed literal
# (not `today`) because the output has to be reproducible and comparable
# across runs months apart. Session start hours are absolute, not offsets
# from midnight -- see families.session_times.
BASE_YEAR, BASE_MONTH, BASE_DAY, BASE_HOUR = 2026, 3, 1, 10

# --- model ------------------------------------------------------------------

# One RTX 3080 at 10GB VRAM. A 7-8B instruct model at Q4 is the ceiling;
# do not plan on 14B+. Verify tool-calling support for the tag before
# relying on it:  ollama show <tag>
MODEL = "qwen3:8b"

# The Ollama default is small and truncates silently. Set it explicitly on
# EVERY call, including the extraction loop and the summariser.
NUM_CTX = 8192

# All experiment runs are greedy and seeded so numbers are comparable.
TEMPERATURE = 0.0
SEED = 1

# --- extraction loop --------------------------------------------------------

# A tool call that returns ok=False gets retried this many times before the
# turn is recorded as an extraction failure.
TOOL_RETRIES = 3

# A GPU job that hangs must fail rather than block the whole run: every
# Ollama call (extraction, summary, embedding, recommendation) passes this
# to the client. A sleeping machine drops the connection mid-call, and the
# default client waits on that socket with no deadline -- it hung an
# unattended run on 2026-10-23.
#
# 300, not the original 180 (2026-10-25): qwen3:8b's thinking sometimes runs
# past three minutes on an ordinary prompt (GPU at 89%, four identical
# attempts logged at exactly 3m0s). The timeout is a liveness knob, not a
# semantic one -- it decides whether a slow call is waited for, never what
# the model produces -- so raising it cannot make rows incomparable.
REQUEST_TIMEOUT_S = 300

# --- recommendation ---------------------------------------------------------

# The recommender returns this many candidates from recipe_search before
# generation picks one. Fixed across every memory method.
TOP_K_RECIPES = 5

# Baselines that pass free text need a turn budget so "full history" cannot
# silently exceed NUM_CTX and lose the earliest -- and often most important --
# preference statements.
RECENT_ONLY_TURNS = 12

RETRIEVAL_TOP_K = 8
EMBED_MODEL = "nomic-embed-text"

# qwen3's chain of thought is ON by default, and on some open-ended prompts
# ("compress this conversation") it does not converge: a 434-character
# transcript still ran past 150s on 2026-10-25 while the identical prompt
# answered in 0.6s with thinking off. The pipeline's own calls -- the rolling
# summary and the recommendation -- switch it off; the extraction loop keeps
# it, because that is the configuration the published extraction numbers
# were measured under. Both roles are uniform across all seven methods, so the
# comparison the benchmark makes is unaffected. `think` is a top-level chat
# parameter, not an entry in `ollama_options()`.
PIPELINE_THINK = False


def ollama_options() -> dict:
    """The options dict to pass on every Ollama call. Use this, not a literal."""
    return {
        "num_ctx": NUM_CTX,
        "temperature": TEMPERATURE,
        "seed": SEED,
    }
