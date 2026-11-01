# results/

The predictions and rendered tables behind the numbers in the top-level README.

Every file here comes from one run:

| | |
|---|---|
| model | `qwen3:8b` via Ollama |
| `num_ctx` | 8192 |
| temperature | 0 |
| seed | 1 |
| corpus | `python -m memproj.generate --n-base 60 --seed 1` |
| trajectories | 120 (60 matched pairs) |
| checkpoints | 193 |
| methods | `gold`, `proposed`, `latest_wins`, `recent_only`, `full_history`, `rolling_summary`, `retrieval` |

## Files

| file | what it is |
|---|---|
| `predictions.jsonl` | 1,351 rows (193 checkpoints x 7 methods), the scored artefact |
| `report.md` | per-method table, state-vs-action, matched-variant deltas |
| `bootstrap-handled.md` | paired bootstrap intervals for `handled`, against `latest_wins` |
| `bootstrap-exact.md` | paired bootstrap intervals for exact state |
| `ablations.md` | each resolver rule turned off, scored from the same extraction |
| `policy.md` | candidate filter vs rank |
| `challenge-predictions.jsonl` | 252 rows (36 checkpoints x 7 methods) on the hand-written set |
| `challenge-report.md` | per-method table on the challenge set |

`histories.jsonl` and `checkpoints.jsonl` are **not** shipped: the generator
reproduces them byte-for-byte from the seed, and the tests assert that.

## Reproducing without a GPU

```bash
python -m memproj.generate --n-base 60 --seed 1
python -m memproj.validate_predictions results/predictions.jsonl
python -m memproj.eval.report results/predictions.jsonl
python -m memproj.eval.bootstrap results/predictions.jsonl --stat handled --reference latest_wins
python -m memproj.challenge --check
python -m memproj.eval.report results/challenge-predictions.jsonl \
  --histories out/challenge/histories.jsonl --checkpoints out/challenge/checkpoints.jsonl
```

Rerunning the models themselves needs Ollama plus the extraction pass; see the
top-level README, "Running it".

## A note on reproducibility

Ollama is not bit-deterministic at temperature 0 with a fixed seed — a rerun
changes a few rows in both directions. The generator's byte-identical guarantee
covers the corpus, not model output. Treat these files as the recorded run, and
use the paired bootstrap to read how much of a gap is history variance.
