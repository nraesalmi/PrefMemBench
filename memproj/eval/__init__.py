"""Evaluation: scores predictions files, never builds them.

Everything here reads `out/predictions.jsonl`, `out/checkpoints.jsonl`, the
catalogue, and -- for attribution and provenance -- `out/histories.jsonl`.
It imports no memory method and opens no SQLite file (README, Evaluator boundary).
That is what lets the same evaluator score the memory half's methods and the
pipeline half's baselines without knowing how either was built.

`metrics.py` is pure functions over parsed dicts; `report.py` does the file
I/O and the rendering.
"""
