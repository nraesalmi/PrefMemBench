"""Step 7's confidence intervals: resample histories, not checkpoints.

Checkpoints in one trajectory are not independent -- they read the same store,
the same sentences and mostly the same preferences -- so a rate computed over
193 checkpoints behaves like one over roughly 120 observations, not 193.
Resampling trajectories (with replacement) is the cheap honest interval: it
answers "how much would this number move if the corpus had drawn different
histories", which is the question a seminar audience asks first.

The comparison is PAIRED. Both methods are read on the same resampled
trajectories, so a method that is lucky on one history does not look better
merely because the draw happened to include it.

No model call appears here, and none may be added: this reads the rows
`memproj.eval.report.evaluate()` already produced.
"""

import argparse
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from memproj.eval import report

# The three rates the seminar quotes. `exact` and `satisfied` are undefined
# where the method has no profile / no plate, and those rows are skipped
# rather than counted as failures -- the same rule the report uses.
STATS = ("handled", "satisfied", "exact")

DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 1


def _score(row: dict, stat: str) -> Optional[Tuple[int, int]]:
    """(numerator, denominator) this row contributes, or None if not scored."""
    if stat == "exact":
        state = row.get("state")
        return None if state is None else (int(bool(state["exact"])), 1)
    if stat == "satisfied":
        satisfied = row.get("satisfied")
        return None if satisfied is None else (int(bool(satisfied)), 1)
    if stat == "handled":
        if (row.get("satisfied") is True
                or row.get("clarification") == "appropriate"):
            return 1, 1
        return 0, 1
    raise ValueError(f"unknown stat {stat!r}; expected one of {STATS}")


def per_trajectory(rows: Sequence[dict], stat: str) -> Dict[str, Tuple[int, int]]:
    """{trajectory_id: (num, den)} for ONE method's rows."""
    units: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        scored = _score(row, stat)
        if scored is None:
            continue
        units[row["trajectory_id"]][0] += scored[0]
        units[row["trajectory_id"]][1] += scored[1]
    return {tid: (num, den) for tid, (num, den) in units.items() if den}


def _ratio(units: Dict[str, Tuple[int, int]],
           draw: Sequence[str]) -> Optional[float]:
    num = sum(units[tid][0] for tid in draw)
    den = sum(units[tid][1] for tid in draw)
    return None if not den else num / den


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile: no interpolation, so a rerun cannot drift."""
    ordered = sorted(values)
    index = round(q * (len(ordered) - 1))
    return ordered[index]


def interval(units: Dict[str, Tuple[int, int]],
             resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED,
             lo: float = 0.025, hi: float = 0.975
             ) -> Optional[Tuple[float, float]]:
    """Percentile interval for one method's rate over histories."""
    if not units:
        return None
    ids = sorted(units)
    rng = random.Random(seed)
    values = []
    for _ in range(resamples):
        draw = [rng.choice(ids) for _ in ids]
        value = _ratio(units, draw)
        if value is not None:
            values.append(value)
    if not values:
        return None
    return percentile(values, lo), percentile(values, hi)


def delta(a: Dict[str, Tuple[int, int]], b: Dict[str, Tuple[int, int]],
          resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED,
          lo: float = 0.025, hi: float = 0.975
          ) -> Optional[Tuple[float, float, float]]:
    """Paired difference a - b: one draw, read by both methods.

    Histories missing from either side are dropped; a pair needs both terms.
    """
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    rng = random.Random(seed)
    deltas = []
    for _ in range(resamples):
        draw = [rng.choice(shared) for _ in shared]
        va, vb = _ratio(a, draw), _ratio(b, draw)
        if va is not None and vb is not None:
            deltas.append(va - vb)
    if not deltas:
        return None
    point = _ratio(a, shared) - _ratio(b, shared)
    return point, percentile(deltas, lo), percentile(deltas, hi)


def compare(rows: Sequence[dict], methods: Sequence[str] = (),
            stat: str = "handled", resamples: int = DEFAULT_RESAMPLES,
            seed: int = DEFAULT_SEED,
            reference: Optional[str] = "gold") -> List[dict]:
    """One line per method: the rate, its interval, and the paired delta."""
    by_method: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    order = [m for m in methods if m in by_method] or sorted(by_method)
    units = {method: per_trajectory(by_method[method], stat)
             for method in order}
    out = []
    for method in order:
        values = units[method]
        if not values:
            out.append({"method": method, "stat": stat, "n": 0, "units": 0,
                        "point": None, "lo": None, "hi": None,
                        "delta": None, "delta_lo": None, "delta_hi": None})
            continue
        point = (sum(n for n, _ in values.values())
                 / sum(d for _, d in values.values()))
        low, high = interval(values, resamples, seed)
        pair = (delta(values, units[reference], resamples, seed)
                if reference in units else None)
        out.append({
            "method": method, "stat": stat,
            "n": sum(d for _, d in values.values()), "units": len(values),
            "point": point, "lo": low, "hi": high,
            "delta": None if pair is None else pair[0],
            "delta_lo": None if pair is None else pair[1],
            "delta_hi": None if pair is None else pair[2],
        })
    return out


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def render_table(rows: Sequence[dict],
                 reference: Optional[str] = "gold") -> str:
    lines = ["| method | rows | histories | rate | 95% CI | vs "
             f"{reference or '-'} (pp) |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        pair = ("--" if r["delta"] is None else
                f"{100 * r['delta']:+.1f} "
                f"[{100 * r['delta_lo']:+.1f}, {100 * r['delta_hi']:+.1f}]")
        lines.append(f"| {r['method']} | {r['n']} | {r['units']} | "
                     f"{_fmt(r['point'])} | "
                     f"[{_fmt(r['lo'])}, {_fmt(r['hi'])}] | {pair} |")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap intervals over histories, not checkpoints.")
    parser.add_argument("predictions", help="out/predictions.jsonl")
    parser.add_argument("--stat", choices=STATS, default="handled")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--reference", default="gold")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--checkpoints", default=None,
                        help="default: config.CHECKPOINTS_PATH")
    parser.add_argument("--histories", default=None,
                        help="default: config.HISTORIES_PATH")
    parser.add_argument("--markdown", metavar="PATH")
    args = parser.parse_args(argv)

    rows = report.evaluate_path(args.predictions, args.checkpoints,
                                args.histories)
    table = render_table(compare(rows, args.methods or (), args.stat,
                                 args.resamples, args.seed, args.reference),
                         args.reference)
    text = (f"### {args.stat}: rate with a paired bootstrap over histories "
            f"({args.resamples} resamples, seed {args.seed})" + chr(10) * 2
            + table)
    print(text)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8",
                  newline=chr(10)) as fh:
            fh.write(text + chr(10))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
