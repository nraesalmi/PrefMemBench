"""Step 7's recommender-policy ablation: what does the candidate filter hide?

The report already shows that the candidate filter masks memory errors: a
row can hold a WRONG profile and still be scored satisfied because the wrong
profile happened to leave a safe dish in the pool ("lucky" rows). That makes
the action column optimistic in a way the state column does not expose.

`recipe_search()` filters the catalogue down to recipes the profile accepts
(policy ``"filter"``, decided 2026-10-21). This module measures the
alternative (policy ``"rank"``) without asking the model anything: it takes
the profiles a run already produced and asks, per checkpoint,

    "was a choice that satisfies the CHECKPOINT'S GOLD profile available
     in the candidate set at all?"

Under ``"filter"`` that question depends on the method's own profile, so a
memory error can make the right answer unreachable. Under ``"rank"`` the
whole catalogue is offered and the answer depends only on the catalogue.
The gap between the two columns is the filter's contribution -- the number
of checkpoints where a memory error silently removed every safe choice.

**No model call is made here, by design** (same reason as
`memproj.memory.ablate`: a second model would mix two noise sources). The
plate the run actually chose is scored separately, by `eval.metrics`.

Usage::

    python -m memproj.pipeline.policy                        # table to stdout
    python -m memproj.pipeline.policy --markdown out/policy.md
"""

import argparse
import sys
from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Sequence

from memproj import config, recipes
from memproj.pipeline import recommend, run

POLICIES = recommend.POLICIES

# Only a method that resolves a profile can be affected by the filter: a
# free-text method passes `None` and both policies are identical for it.
STRUCTURED = ("latest_wins", "proposed", "gold")


def _as_sets(profile: Mapping) -> Dict[str, set]:
    return {attribute: set(values) for attribute, values in profile.items()}


def available(row: dict, checkpoint: dict, catalogue: Sequence[dict],
              policy: str) -> Optional[bool]:
    """Is some gold-satisfying recipe in this row's candidate set?

    `None` for a row that does not recommend (a clarify) or carries no
    profile -- those are not evidence about the candidate policy.
    """
    profile = row.get("profile")
    if row.get("action") != "recommend" or not isinstance(profile, dict):
        return None
    safe = {recipe["id"]
            for recipe in recipes.valid_recipes(
                catalogue,
                _as_sets(checkpoint["gold_profile"]))}
    offered = {recipe["id"]
               for recipe in recommend.recipe_search(profile, catalogue,
                                                     policy=policy)}
    return bool(safe & offered)


def table(predictions: Sequence[dict], checkpoints: Sequence[dict],
          catalogue: Sequence[dict]) -> List[dict]:
    """One row per (method, policy): how often a safe choice was offered."""
    by_checkpoint = {c["checkpoint_id"]: c for c in checkpoints}
    counts: Dict = defaultdict(lambda: defaultdict(int))
    for row in predictions:
        method = row["method"]
        if method not in STRUCTURED:
            continue
        checkpoint = by_checkpoint.get(row["checkpoint_id"])
        if checkpoint is None:
            continue
        for policy in POLICIES:
            ok = available(row, checkpoint, catalogue, policy)
            if ok is None:
                continue
            counts[(method, policy)]["rows"] += 1
            counts[(method, policy)]["offered"] += bool(ok)
    out = []
    for method in STRUCTURED:
        row = {"method": method}
        for policy in POLICIES:
            c = counts[(method, policy)]
            row[policy] = c["rows"]
            row[policy + "_offered"] = c["offered"]
            row[policy + "_pct"] = (100.0 * c["offered"] / c["rows"]
                                    if c["rows"] else None)
        out.append(row)
    return out


def render(rows: Sequence[dict]) -> str:
    lines = ["| method | policy | recommend rows | safe choice offered |",
             "|---|---|---:|---:|"]
    for row in rows:
        for policy in POLICIES:
            pct = row[policy + "_pct"]
            shown = "n/a" if pct is None else f"{row[policy + '_offered']} ({pct:.1f}%)"
            lines.append(f"| {row['method']} | {policy} | {row[policy]} | {shown} |")
    lines.append("")
    lines.append(
        "`filter` is the policy every published number was measured under. "
        "A `filter` row below its `rank` twin means the method's own profile "
        "removed every choice that would have satisfied the answer key -- a "
        "memory error the action column cannot see.")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m memproj.pipeline.policy")
    parser.add_argument("predictions", nargs="?", default=str(config.PREDICTIONS_PATH))
    parser.add_argument("--checkpoints", default=None,
                        help="defaults to config.CHECKPOINTS_PATH")
    parser.add_argument("--catalogue", default=None,
                        help="defaults to config.CATALOGUE_PATH")
    parser.add_argument("--markdown", metavar="PATH", default=None,
                        help="also write the table here")
    args = parser.parse_args(argv)

    predictions = run.read_jsonl(args.predictions)
    _histories, checkpoints = run.load_corpus(checkpoints_path=args.checkpoints)
    catalogue = run.load_catalogue(args.catalogue)

    rows = table(predictions, checkpoints, catalogue)
    text = render(rows)
    sys.stdout.write(text)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        sys.stdout.write(f"\nwrote {args.markdown}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
