# PrefMemBench

**A preference memory benchmark.** When should an AI agent remember, update, or
ignore a stated preference?

A user says, across several sessions:

1. "I usually prefer vegetarian meals."
2. "For tonight, suggest a chicken dish for my brother."
3. "Tomorrow, recommend lunch for me."

A reliable agent should understand that (2) is about another person and one
occasion, and must not let it overwrite the user's own preference. This
repository builds the benchmark for that question and a reference memory method
to run against it.

---

## The contribution

Adding memory is not, by itself, a contribution. The contribution here is
narrower:

> A benchmark of **context-dependent** preference changes, with **matched
> conversation variants**, that traces errors from memory updates through to
> downstream decisions **and explanations**.

The distinction that matters is between a *new preference* and a *statement that
applies only in a particular context*. Everything in this repository exists to
make that distinction measurable.

### Research questions

| | question |
|---|---|
| **RQ1** | Which types of conversational updates cause agents to overwrite, retain, or apply preferences incorrectly? |
| **RQ2** | Does storing scope and provenance improve recommendations beyond chronological memory or conversation summaries? |
| **RQ3** | Can an agent recall a preference correctly but still fail to apply it during tool use? |
| **RQ4** | When do explanations incorrectly attribute a recommendation to something the user supposedly said? |

### Two things are measured, at every checkpoint

1. **State** — did the method reconstruct the correct active profile?
2. **Action** — did the recommendation satisfy that profile, and is the
   explanation honest?

The diagnostic that motivates the split is the *dissociation*: an agent can
recall a preference correctly and still recommend something that violates it, or
let a temporary request contaminate later personalization. Aggregate recall
accuracy hides both. The results section reports "recalled but violated" and
"lucky" (satisfied the answer key despite a wrong profile) separately for this
reason.

---

## Architecture

```
OFFLINE (benchmark build)
  seeded event sequence -> gold state per event -> templates -> dialogue
  matched variant = same events with ONE field flipped
  out/histories.jsonl, out/checkpoints.jsonl, out/challenge/

ONLINE (per trajectory, per method)
  replay turns -> memory agent (LLM, extraction only) -> memory tools -> SQLite
  at a checkpoint: resolver (pure Python) -> active profile
  -> fixed recommender (search + validate + LLM) -> predictions.jsonl
  -> evaluator -> metrics + paired tables
```

The rule that holds everywhere: **the LLM classifies and fills fields; Python
applies the rules.** Date arithmetic, subject and scope enforcement, supersede
bookkeeping, and active-profile filtering are all deterministic code. The model
never supplies a timestamp, a turn id, or a gold label.

Retrieval, ranking and generation in the recommender are identical across all
memory methods; only the memory method changes between experiments. That is what
makes the comparison a comparison of memory designs rather than of prompts.

---

## Repository layout

```
README.md  LICENSE  NOTICE  requirements.txt  pyproject.toml
memproj/
  config.py        MODEL, NUM_CTX, TEMPERATURE, SEED, TOOL_RETRIES and every path
  schema.py        Event, Turn, Checkpoint, Trajectory dataclasses
  semantics.py     gold_profile(), gold_action(), history() -- the answer key
  generate.py      CLI: histories, checkpoints, matched variants
  templates.py     surface forms per event type (explicit and implicit pools)
  families.py      the eleven matched-variant story shapes
  recipes.py       attribute derivation and validate()
  catalogue.py     HUMMUS reader + seeded selection -> data/recipes_hummus.json
  store.py         SQLite: turns, prefs, decisions, clarifications
  tools.py         six memory tools + TOOL_SCHEMAS + dispatch
  resolver.py      active_profile() with ablation flags, history()
  replay.py        gold events -> tool calls (the tool-layer gate)
  challenge.py     the hand-written challenge set: validate, build, diff
  validate_predictions.py   the predictions contract, as code
  data/            challenge.jsonl (authored events) and recipes_hummus.json
  memory/          extraction loop, proposed method, latest-wins, ablations
  pipeline/        recommender, the four free-text baselines, gold, runner, policy
  eval/            metrics, report rendering, paired bootstrap
tests/             one file per module; tests/fixtures/ for the small corpus
results/           the predictions and rendered tables behind the paper's numbers
out/               generated data (gitignored -- regenerate it, do not commit it)
```

---

## The corpus

`python -m memproj.generate` builds 60 base histories, each with a matched twin
that differs in exactly one event, for 120 trajectories and 193 checkpoints.
Every history runs three sessions. Dialogue is rendered *from* events, so the
answer key is computed by deterministic Python and never by a model.

Eleven families, each a matched pair:

| family | what it isolates | the flip |
|---|---|---|
| `temp_vs_perm` | a temporary exception vs a permanent revision | `scope` |
| `subject_swap` | the user's preference vs another person's | `subject` |
| `hypo_vs_real` | a hypothetical vs a real statement | `type` |
| `ambiguous_vs_resolved` | ambiguity that is resolved vs not | `referent_event_id` |
| `correction_vs_none` | a correction vs no correction | remove an event |
| `correction_of_temporary` | a correction that retargets a windowed record | `target_event_id` |
| `exception_then_restated` | a window closing and the thing being said again | `value` |
| `ambiguous_then_clarify` | a pending ambiguity and its answer | `referent_event_id` |
| `hypothetical_then_real` | a hypothetical followed by the real fact | add an event |
| `diet_swap` | the user is vegetarian vs the brother is | `subject` |
| `medium_vs_spicy` | a revision to `medium` vs to `spicy` | `value` |

A flip is expressed three ways — a changed field, a removed event, or an added
event — and the tests derive which from the two event lists rather than assuming
one convention.

Verified shape at the default `--n-base 60 --seed 1`:

| quantity | value |
|---|---|
| trajectories | 120 (60 pairs) |
| events | 291 |
| checkpoints | 193 |
| turns per history | min 18, max 30, mean 24.0 |
| `gold_action` split | 172 `recommend`, 21 `clarify` |

Generation is deterministic **per pair**: each pair seeds its own RNG stream from
`f"{seed}:{pair_id}"`, so raising `--n-base` cannot perturb the histories already
generated, and the same seed gives byte-identical files months apart.

### The annotation rules

`semantics.gold_profile` is a thin wrapper over `_standing_records`, and the order
inside it is load-bearing:

1. sort by `(ts, event_id)` so ties never depend on input order;
2. fold each `correction` into its target's `value`, keeping the target's
   `valid_from`;
3. apply supersede — a `permanent_revision` evicts its named target *and* any
   earlier revision of the same attribute;
4. apply the exclusive-window check (**`valid_from <= now < valid_until`**), so an
   expired exception is a distractor rather than a fact;
5. remove the record an `ambiguous_update` with a `referent_event_id` points at,
   as of the resolver's `ts`;
6. `hypothetical` and `filler` never enter a profile;
7. an unresolved ambiguity produces `gold_action == "clarify"` rather than a
   profile guess.

---

## Data formats

These are the interfaces between the generator, the memory methods and the
evaluator. `memproj/schema.py` is the source of truth for the dataclasses;
`memproj/validate_predictions.py` enforces the predictions format and runs in the
test suite.

### `schema.py`

```python
EVENT_TYPES = ("initial_pref", "permanent_revision", "temporary_exception",
               "other_person", "hypothetical", "correction",
               "ambiguous_update", "filler")
SCOPES = ("permanent", "temporary")

@dataclass(frozen=True)
class Event:
    event_id: str
    session: int
    ts: datetime
    type: str
    subject: str
    attribute: Optional[str]
    value: Optional[str]
    scope: str
    valid_from: datetime
    valid_until: Optional[datetime] = None
    target_event_id: Optional[str] = None
    needs_clarification: bool = False
    referent_event_id: Optional[str] = None

@dataclass(frozen=True)
class Turn:
    turn_id: str
    session: int
    ts: datetime
    speaker: str                      # "user" | "assistant"
    text: str
    event_id: Optional[str] = None    # None on assistant turns, always

@dataclass(frozen=True)
class Checkpoint:
    checkpoint_id: str
    trajectory_id: str
    after_event_id: Optional[str]
    now: datetime
    person: str
    request_text: str
    gold_profile: dict                # {attribute: sorted list of values}
    gold_action: str                  # "recommend" | "clarify"
    relevant_event_ids: tuple = ()
    distractor_event_ids: tuple = ()
    kind: str = "final"
```

Three properties that are easy to get wrong:

- **There is no `time_phrase` field.** Relative phrases ("tonight", "this
  weekend") are resolved to absolute datetimes at generation time
  (`families._temporary_window`), so `semantics.py` only ever sees absolute
  times.
- **The `subject` of an `other_person` event is the person's name** — `brother`,
  `partner` or `friend` — never the literal string `"other_person"`. Filter on
  `subject == "user"`.
- **`event_id` and `turn_id` are strings and are unique only inside one
  trajectory.** Each history numbers from `e1` / `t0001`, so a `turn_id` must be
  resolved against the trajectory on the same line.

### `out/histories.jsonl`

One trajectory per line. Top-level keys: `events`, `family`, `flipped_field`,
`pair_id`, `trajectory_id`, `variant`, `turns`. Written with sorted keys,
`ensure_ascii=False`, LF endings.

`is_implicit` records whether the user *implied* the preference ("Ugh, fine,
spicy — but just tonight") or *stated* it ("Only for tonight, spicy is fine").
`templates.render` draws from the EXPLICIT and IMPLICIT pools at
`implicit_ratio=0.65`. The flag is a static property of the surface form chosen at
generation time, not something recoverable from the text, which is why it is
stored rather than re-derived.

### `out/checkpoints.jsonl`

One checkpoint per line, carrying `gold_profile`, `gold_action`,
`relevant_event_ids`, `distractor_event_ids` and `request_text`. `gold_profile` is
always **recomputed from that variant's own events**, never copied from its twin —
copying it would destroy the property the benchmark rests on.

`kind` is **not globally unique**: `for_user` is emitted by both `subject_swap`
and `hypo_vs_real`. The alignment key for a twin pair is
**`(family, person, kind)`**. `subject_swap` is the only asymmetric pair — its
base has one checkpoint and its matched twin two — so an unpaired `kind` means
"this variant has no comparable question here", not missing data.

### Method interface

Every memory method exposes exactly one entry point:

```python
def build_context(self, trajectory: dict, checkpoint: dict) -> dict:
    """Return {"context_text": str, "profile": dict | None,
               "evidence_turn_ids": list[str]}"""
```

- `trajectory` and `checkpoint` are the parsed dicts from the two JSONL files.
- `context_text` is what the recommender's model call sees.
- `profile` is `None` for the four free-text baselines, or a resolved profile of
  the same shape as `gold_profile`. An empty profile is `{}`, **not** `None`:
  `None` means "this method has no structured opinion", `{}` means "this method
  believes nothing is active". The evaluator treats them differently.
- `evidence_turn_ids` are `turn_id` strings, not integers and not `event_id`s.

| `method` | `profile` |
|---|---|
| `recent_only` | `None` |
| `full_history` | `None` |
| `rolling_summary` | `None` |
| `retrieval` | `None` |
| `latest_wins` | resolved |
| `proposed` | resolved |
| `gold` | `checkpoint["gold_profile"]` verbatim |

`gold` is the ceiling and the plumbing test: it injects the answer key, so it must
score 100% on state.

### Predictions format

One JSON line per `(trajectory_id, checkpoint_id, method, model)` in
`out/predictions.jsonl`:

```json
{"trajectory_id": "p000_base", "checkpoint_id": "p000_base_c0",
 "method": "proposed", "model": "qwen3:8b",
 "profile": {"spice_level": ["mild", "spicy"]},
 "action": "recommend",
 "recipe_id": "107579",
 "question": null,
 "explanation": "I included spicy options because you said spicy was fine tonight.",
 "cited_turns": ["t0003"],
 "tokens": 812,
 "latency_s": 3.41}
```

| field | type | rule |
|---|---|---|
| `trajectory_id` | str | must exist in `histories.jsonl` |
| `checkpoint_id` | str | must exist in `checkpoints.jsonl` and belong to that trajectory |
| `method` | str | one of the seven above |
| `model` | str | the Ollama tag actually used; `"gold"` for the gold baseline |
| `profile` | dict \| null | `null` only for the free-text baselines; values are sorted lists |
| `action` | str | exactly `"recommend"` or `"clarify"` |
| `recipe_id` | str \| null | required with `recommend`; must exist in the catalogue |
| `question` | str \| null | required with `clarify` |
| `explanation` | str | always present |
| `cited_turns` | list[str] | `turn_id`s only; may be empty |
| `tokens` | int | prompt + completion; `0` for the gold baseline, which makes no model call |
| `latency_s` | float \| null | `null` for the gold baseline, which does real work but no generation |

`action` and the two payload fields are mutually exclusive by construction and
checked mechanically. Run `python -m memproj.validate_predictions <file>` to check
a file; it prints one line per error and exits non-zero if there are any.

### Evaluator boundary

The evaluator reads only the predictions file, `checkpoints.jsonl` and
`histories.jsonl`. It never imports a memory method, never reads a SQLite file,
and never sees the generator's dataclasses — so it cannot reward a method for its
own internal state. `histories.jsonl` is a required third input because
checkpoints name *event* ids and events carry no turn ids, so provenance and
contamination cannot be scored from the other two files alone.

---

## Recipe catalogue

`memproj/data/recipes_hummus.json` holds 155 recipes cut from HUMMUS, a linked
recipe dataset published at RecSys 2023
([ACM](https://dl.acm.org/doi/abs/10.1145/3604915.3609491),
[source](https://gitlab.com/felix134/connected-recipe-data-set)). The full table
is 507,336 recipes; `memproj/catalogue.py` selects a deterministic subset stratified
by derived diet and heat, plus exclusion-carriers for peanut, mushroom and
aubergine. Rebuilding the catalogue requires `out/hummus/pp_recipes.zip` and
changes recipe ids, not the corpus: the generator reads only the spice vocabulary
and the valid recipe set.

Recipe attributes are *derived*, not looked up. HUMMUS has no `spice_level`,
`vegetarian` or `diet` column; it has free-text ingredients and a `tags` field
that is empty on 46% of rows. `recipes.derive_attributes()` therefore matches heat
and meat markers against the **word tokens** of each ingredient phrase. The token
test matters: an exact whole-name intersection calls `chicken breast` vegetarian,
and on a 20,000-row sample that misclassified about 15% of rows — enough that a
vegetarian profile could be offered meat and `validate()` would pass it.

`recipe_id` values are the HUMMUS `recipe_id` as strings. They are **opaque to
every consumer**: nothing may parse them or assume a shape.

### Data licence

The HUMMUS code is GPLv3. The **data** was crawled from food.com and may only be
used for **educational or scientific purposes**. That covers this benchmark, but
it means the source table must not be redistributed. Only the derived 155-recipe
selection ships here, as a benchmark fixture; download the full table from the
source repository if you want to rebuild it. See `NOTICE` for the full statement.

---

## Running it

Python 3.11+; the suite also runs on 3.10. Everything except the baselines needs
only the standard library and pytest.

```bash
pip install -r requirements.txt

python -m memproj.generate --n-base 60 --seed 1   # required on a fresh clone
python -m pytest tests/ -q                        # 521 tests, no model call
python -m memproj.replay                          # tool-layer gate: 0 mismatches
```

Thirteen tests in `tests/test_resolver.py` and `tests/test_replay.py` read the
generated corpus; without `out/` they error rather than skip, so generate first.
The recommender, runner and evaluator tests use `tests/fixtures/` and do not need
`out/`.

The gold baseline and the evaluator need no model either:

```bash
python -m memproj.pipeline.run --methods gold     # 193 rows, state 193/193
python -m memproj.validate_predictions out/predictions.jsonl
python -m memproj.eval.report out/predictions.jsonl
python -m memproj.eval.bootstrap out/predictions.jsonl --stat handled --reference latest_wins
python -m memproj.pipeline.policy                 # candidate policy, no model
python -m memproj.challenge --check               # build + replay the challenge set
```

The four free-text baselines and the two structured methods need Ollama with
`qwen3:8b` (and `nomic-embed-text` for retrieval):

```bash
python -m memproj.pipeline.run --methods gold recent_only full_history \
  rolling_summary retrieval latest_wins proposed
python -m memproj.pipeline.run --methods recent_only --resume   # continue a killed run
python -m memproj.memory.ablate                   # five resolver ablations from one extraction
```

Only one heavy job should run at a time. A run that dies halfway can be restarted
with `--resume`; completed rows are journalled beside the output file. The
extraction pass is cached under `out/extraction/` and is invalidated
automatically when the corpus, the model or the extraction prompt changes.

To reproduce the paper's numbers **without a GPU**, generate the corpus and score
the shipped predictions:

```bash
python -m memproj.generate --n-base 60 --seed 1
python -m memproj.eval.report results/predictions.jsonl
python -m memproj.eval.bootstrap results/predictions.jsonl --stat handled --reference latest_wins
python -m memproj.challenge --check
python -m memproj.eval.report results/challenge-predictions.jsonl \
  --histories out/challenge/histories.jsonl --checkpoints out/challenge/checkpoints.jsonl
```

Generator flags — these are CLI parameters, not constants:

| flag | default | meaning |
|---|---|---|
| `--n-base` | 60 | base histories; each gets a matched twin, so trajectories = 2x |
| `--seed` | 1 | RNG seed; controls template choice and filler length |
| `--sessions` | 3 | sessions per history |
| `--turns-per-session` | 8 | target turns per session, jittered +/-2 |
| `--catalogue` | `config.CATALOGUE_PATH` | recipe catalogue |
| `--out` | `config.OUT_DIR` | output directory |

Model settings live in `memproj/config.py` (`MODEL`, `NUM_CTX`, `TEMPERATURE`,
`SEED`) and nothing else hardcodes them; `config.ollama_options()` is the required
options dict for every Ollama call.

---

## Results

Everything below is one run: **qwen3:8b, `num_ctx` 8192, temperature 0, seed 1**,
seven methods over the 120-trajectory corpus, 1,351 prediction rows. The raw files
and rendered tables are in `results/`.

### Main table

| method | state exact | state P | state R | plate satisfies the key | clarify ok / unnecessary / missed | contamination | provenance problems | handled |
|---|---|---|---|---|---|---|---|---|
| `gold` | **193/193 (100%)** | 100.0% | 100.0% | 172/172 (100%) | 21 / 0 / 0 | 0 | 0/193 | **100%** |
| `proposed` | **158/193 (81.9%)** | 91.3% | 97.9% | 183/188 (97.3%) | 2 / 3 / 19 | 38 | 19/193 | **95.9%** |
| `rolling_summary` | n/a | n/a | n/a | 152/191 (79.6%) | 2 / 0 / 19 | 0 | 119/193 | 79.8% |
| `full_history` | n/a | n/a | n/a | 152/193 (78.8%) | 0 / 0 / 21 | 0 | 119/193 | 78.8% |
| `retrieval` | n/a | n/a | n/a | 145/188 (77.1%) | 3 / 2 / 18 | 0 | 85/193 | 76.7% |
| `recent_only` | n/a | n/a | n/a | 140/189 (74.1%) | 2 / 2 / 19 | 0 | 88/193 | 73.6% |
| `latest_wins` | **88/193 (45.6%)** | 79.3% | 66.3% | 138/193 (71.5%) | 0 / 0 / 21 | 45 | 40/193 | **71.5%** |

*handled* means the checkpoint got the right action — a satisfying plate or an
appropriate clarification. Free-text baselines have no profile, so their state
scores are `n/a`; their *action* is still scored, against the checkpoint's gold
profile rather than the method's own. The `handled` column here is a point
estimate; the paired bootstrap below it reports the same seven numbers with
intervals and a reference-relative delta, and that is the table to quote when
the question is whether a gap survives sampling noise.

### State versus action

Scored on the checkpoints where the method made a recommendation:

| method | both right | recalled but violated | lucky (satisfied despite a wrong profile) | both wrong |
|---|---|---|---|---|
| `gold` | 172 | 0 | 0 | 0 |
| `proposed` | 154 | **0** | 29 | 5 |
| `latest_wins` | 88 | **0** | 50 | 55 |

Neither structured method shows the dissociation directly, because the candidate
filter removes recipes the method's own profile rejects — a wrong profile cannot
produce a violating plate. It shows up instead as **lucky** rows: `latest_wins`
satisfies the answer key on 50 checkpoints while holding the wrong profile, more
than half its successes. `proposed` is lucky on 29 of 154.

### Paired bootstrap over histories

Checkpoints inside one history are not independent, so intervals resample
histories (2,000 resamples, seed 1). `handled`, with a paired delta against
`latest_wins`, ordered by rate:

| method | handled | 95% CI | vs latest_wins (pp) |
|---|---|---|---|
| `gold` | 100.0% | [100.0%, 100.0%] | +28.5 [+22.0, +34.9] |
| `proposed` | 95.9% | [93.0%, 98.4%] | **+24.4 [+17.6, +31.3]** |
| `rolling_summary` | 79.8% | [74.3%, 84.9%] | +8.3 [-1.1, +17.6] |
| `full_history` | 78.8% | [73.0%, 84.2%] | +7.3 [-2.3, +17.0] |
| `retrieval` | 76.7% | [70.6%, 83.0%] | +5.2 [-4.1, +15.3] |
| `recent_only` | 73.6% | [67.6%, 79.4%] | +2.1 [-6.8, +11.2] |
| `latest_wins` | 71.5% | [65.1%, 78.0%] | +0.0 [+0.0, +0.0] |

`gold` and `proposed` are the only rows whose paired interval excludes zero. The
four free-text baselines sit within a few points of `latest_wins` and every delta
straddles zero. On exact state, `proposed` is 81.9% [75.3%, 87.9] against
`latest_wins` 45.6% [37.4%, 54.1]; the free-text baselines have no state to
measure.

### Resolver ablations

One resolver rule off at a time, scored from the *same* persisted extraction so the
comparison isolates the rule rather than the model's run-to-run noise:

| ablation | state exact | satisfied | handled |
|---|---|---|---|
| all rules on | 158/193 (81.9%) | 172/178 | 181/193 |
| `no_scope` | 150/193 (77.7%) | 164/178 | 173/193 |
| `no_time` | 149/193 (77.2%) | 164/178 | 173/193 |
| `no_subject` | 136/193 (70.5%) | 166/178 | 175/193 |
| `no_supersede` | **116/193 (60.1%)** | 175/178 | 184/193 |
| `no_correction` | 155/193 (80.3%) | 172/178 | 181/193 |

Every rule that fires costs state, and supersede costs the most. Note the middle
column: removing supersede handling *raises* satisfaction, because a profile that
keeps stale values still produces plates the candidate filter accepts — the action
column alone would have hidden the error. `medium_vs_spicy` collapses from 10/10 to
0/10 exact without supersede, the sharpest single cell in the table.

### Candidate policy

Filter (offer only recipes the profile accepts) versus rank (rank the whole
catalogue) changes no count on this corpus: `latest_wins` 176 of 193 safe choices
under both, `proposed` 185 of 188 under both, `gold` 172 of 172. The filter is the
policy every reported number was measured under.

### The hand-written challenge set

18 histories (36 checkpoints) whose prose the template pools cannot produce. Only
the events are authored; gold is computed with the same `semantics` functions, and
the corpus was independently re-labelled by a second reviewer from the transcript
alone — 10 field differences, all inert, 0 gold changes.

| method | state exact | plate satisfies the key | handled |
|---|---|---|---|
| `gold` | 36/36 | 34/34 | 36/36 |
| `proposed` | 24/36 (66.7%) | 33/35 (94.3%) | 33/36 (91.7%) |
| `recent_only` | n/a | 32/36 (88.9%) | 32/36 |
| `rolling_summary` | n/a | 31/36 (86.1%) | 31/36 |
| `latest_wins` | 19/36 (52.8%) | 31/36 (86.1%) | 31/36 (86.1%) |
| `retrieval` | n/a | 30/36 (83.3%) | 30/36 |
| `full_history` | n/a | 30/36 (83.3%) | 30/36 |

The challenge set is harder and less forgiving than the generated corpus: only 13
of its 36 checkpoints are satisfiable by the whole catalogue, against 61 of 193 on
the generated corpus, so a coverage error has more room to become a wrong plate.

---

## Tests

`python -m pytest tests/ -q` — 521 tests, about four seconds, no model call in any
test of rule logic.

- `test_semantics.py` — every event type at multiple `now` values: before, during
  and after a temporary window, before and after a revision, corrections,
  other-person isolation, hypotheticals, unresolved vs resolved ambiguity.
- `test_recipes.py` — attribute derivation and each validation rule.
- `test_templates.py` — pool sizes, slot filling, the implicit/explicit split.
- `test_generate.py` — determinism (including byte-identical files), matched-pair
  integrity, each family's behavioural difference, recipe coverage, timestamp
  monotonicity, and no assistant leakage into the answer key.
- `test_store.py` / `test_tools.py` — schema, and every rejection an 8B model
  plausibly makes; each comes back `ok=False` and is logged.
- `test_config.py` — no module but `config.py` hardcodes a model tag or a
  `num_ctx`, and `ollama_options()` carries the three pinned settings.
- `test_resolver.py` — each ablation flag breaks exactly the checkpoint kinds whose
  answer its rule decides, in both directions.
- `test_contracts.py` — the predictions contract, one parametrised case per rule.
- `test_eval.py` / `test_bootstrap.py` — every metric, and that a corrupted copy of
  the gold file moves exactly one counter.
- `test_challenge.py` / `test_ablate.py` / `test_policy.py` — the challenge set,
  the ablations, and the candidate policy.

`python -m memproj.replay` is the tool-layer gate as an executable check: replaying
the gold events through the tool layer must reproduce `gold_profile` at *every*
checkpoint with zero mismatches. If the tool layer cannot reconstruct the answer
key from perfect input, no measured failure downstream means anything.

---

## Design decisions

Recorded here because a reader of the code would otherwise have to guess:

- **Ambiguity resolution is a separate later event** carrying `referent_event_id`,
  not a mutation of the original. Events stay immutable, and the resolver has a
  timestamp, which is what lets `gold_action` return `clarify` in between.
- **A resolved profile filters the candidate set.** Structured methods offer only
  recipes `recipes.validate()` accepts; `profile is None` and `{}` offer the whole
  catalogue. That makes `gold` a true ceiling — 100% satisfaction by construction.
- **Satisfaction is scored against the checkpoint's profile, never the row's own.**
  Scoring against the method's own profile would reward it for consistently
  believing the wrong thing.
- **A free-text row has no state score.** `profile: null` means "no structured
  opinion", so state is computed only over rows with a profile. The action is still
  scored.
- **Spelling is not the fact.** Profile values are compared canonically — stripped,
  case-folded, and singularised for `excluded_ingredient` — matching the rule the
  tool layer applies on write and `validate()` applies before matching.
- **The two structured methods share one extraction pass.** Both read one store per
  trajectory, so the experiment varies the resolution rule only. Extracting per
  method would double the model calls and confound the comparison with run-to-run
  noise.
- **`latest_wins` ignores the subject, not just the clock.** "Newest statement per
  attribute" means the brother's preference can become the user's. That is the
  contamination the metric exists to show; a baseline with a subject filter would
  be a rule under test granted for free.
- **Citation is the method's `evidence_turn_ids`, not the model's.** Asking the
  model to cite would add a second failure mode. For the baselines, every user turn
  carrying an event that the method *read* is cited, distractors included, so
  provenance measures the memory choice rather than a citation instruction.
- **The extraction is cached by a manifest of its inputs**, not by file presence:
  corpus fingerprint, model, `num_ctx`, temperature, seed and prompt hash. Anything
  else re-extracts. The ablations must read the *same* extraction, because Ollama at
  temperature 0 with a fixed seed still changes answers between runs.
- **Thinking is off for the pipeline's own calls.** qwen3:8b thinks by default and
  on long prose prompts can stall indefinitely; the summariser runs with
  `think=False`, while the extraction loop keeps `think=True`, which is what its
  numbers were measured with.

---

## Limitations

- **Template-generated data** makes easy update types look easy. The 18-history
  challenge set is the reality check, and every design decision above was checked
  against it.
- **A weak 8B extractor** means some of the measured gap is the model, not the
  memory design. The gold baseline separates memory errors from action errors, and
  the extraction is scored alone in `memproj/memory/extract.py`.
- **Ollama is not bit-deterministic** at temperature 0 with a fixed seed: reruns
  change a few rows in both directions. The generator's byte-identical guarantee
  does not extend to model output; the bootstrap covers that variance.
- **`gold` recommends few distinct recipes** — the first candidate from a
  deterministic ranking, so the answer is a function of the spice ceiling alone.
  Deterministic and correct, not a coverage bug; recipe diversity only becomes
  meaningful when a model chooses among top-k.
- **The catalogue uses US ingredient names** (`eggplant`, `zucchini`) where the
  corpus's exclusion vocabulary is British (`aubergine`, `courgette`). The generated
  corpus is unaffected; a hand-authored history excluding a British-only word would
  reject fewer dishes than intended.
- **`tokens` and `latency_s` cover the recommender only.** The extraction pass is
  amortised per trajectory and is not attributed to individual rows.

---

## Licence

Code: Apache-2.0 — see `LICENSE`.
Recipe catalogue: derived from HUMMUS / food.com, educational and scientific use
only — see `NOTICE`.
