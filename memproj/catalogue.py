"""Cut the benchmark's recipe catalogue out of HUMMUS.

HUMMUS is 507,336 recipes. Its licence (data crawled from food.com) allows
educational and scientific use but not redistribution, so the full table stays
under `out/hummus/` -- gitignored -- and this module writes the small,
deterministic selection the benchmark actually ships:
`memproj/data/recipes_hummus.json`, as `{id, name, ingredients, tags}`.

Why a selection at all, and why this one:

  * `derive_attributes()` decides diet and heat from the ingredient list, so
    the catalogue must contain every (diet, heat) cell a gold profile can ask
    for. A cell with too few recipes fails `generate.py`'s coverage check and
    the pair re-seeds forever.
  * the corpus excludes peanut, mushroom and aubergine, so each cell also
    needs dishes that carry them -- otherwise the exclusion constraint is
    vacuous and `validate()` never has anything to reject.
  * every other property of the selection is arbitrary, so it is seeded
    reservoir sampling rather than "the first N rows", and the quotas are
    written down here instead of emerging from the file order.

Nothing in the benchmark imports this module: it is a one-off build step.
The README's "Recipe catalogue" section records what the source rows
contain.
"""

import argparse
import ast
import csv
import io
import json
import random
import re
import sys
import zipfile
from typing import Dict, Iterator, List, Sequence, Tuple

from memproj import config
from memproj.recipes import derive_attributes

# Diet x heat. "omnivore" means "not derived vegetarian"; it says nothing
# about whether meat is in every dish (a vegetable side dish lands here too).
CELLS: Tuple[Tuple[str, str], ...] = tuple(
    (diet, heat)
    for diet in ("vegetarian", "omnivore")
    for heat in ("mild", "medium", "spicy")
)

# The corpus's three exclusions (`families.py` hardcodes these values).
TARGET_INGREDIENTS: Tuple[str, ...] = ("peanut", "mushroom", "aubergine")

# Per cell. 88% of HUMMUS derives as mild,
# so an unstratified sample would give the corpus almost no medium/spicy
# candidates, and the diet/medium checkpoints need a mix inside one profile.
QUOTA: Dict[Tuple[str, str], int] = {
    ("vegetarian", "mild"): 40,
    ("vegetarian", "medium"): 16,
    ("vegetarian", "spicy"): 8,
    ("omnivore", "mild"): 32,
    ("omnivore", "medium"): 16,
    ("omnivore", "spicy"): 8,
}

# On top of the cell quota: this many recipes carrying each target ingredient
# in each cell, so a vegetarian+mild+peanut-excluded profile still has
# candidates and its exclusion still has something to reject.
TARGETS_PER_CELL = 2

_QUOTED = re.compile(r"'([^']*)'")


# --- reading HUMMUS ---------------------------------------------------------

def _literal_list(text: str) -> List[str]:
    """Parse a Python-list-shaped column ('['a', 'b']') into lower-case names."""
    text = (text or "").strip()
    if not text:
        return []
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [m.strip().lower() for m in _QUOTED.findall(text) if m.strip()]
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(v).strip().lower() for v in value if str(v).strip()]


def _ingredient_names(text: str) -> List[str]:
    """The names from a row's `ingredients` column.

    The column is dict-shaped, with section headings as keys and
    `(name, quantity)` pairs as values:
    `{'': [('water', '1 1/4 time(s) cups '), ...], 'For the sauce': [...]}`.
    Not JSON -- `json.loads` fails -- so it needs `literal_eval`; rows that
    trip on it fall back to a quoted-string scan.

    **This is the authentic column and `ingredient_food_kg_names` is not.**
    The two agree on under 1% of rows: recipe 100024 "Chocolate Cherry Bread"
    lists water, white bread flour, wheat bread flour and dry milk here, and
    avocados, black beans and chicken broth in `ingredient_food_kg_names`.
    This column matches the title (a dish titled "...Chicken" lists chicken)
    on 84-100% of rows.
    """
    text = (text or "").strip()
    if not text:
        return []
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [m.strip().lower() for m in _QUOTED.findall(text) if m.strip()]
    if not isinstance(value, dict):
        return []
    names: List[str] = []
    for pairs in value.values():
        for pair in pairs or []:
            item = pair[0] if isinstance(pair, (list, tuple)) and pair else pair
            name = str(item).strip().lower()
            if name:
                names.append(name)
    return names


def open_rows(path: str) -> Iterator[dict]:
    """Yield `{id, name, ingredients, tags}` from a HUMMUS recipes table.

    Accepts the zipped preprocessed table (`pp_recipes.zip`). Three traps are
    handled here: the header begins with an empty column name, so rows are
    read by name and never by position; `recipe_id` repeats, so the caller
    sees one row per id; and `ingredient_food_kg_names` is misjoined, so
    ingredients come from the `ingredients` column instead.
    """
    with zipfile.ZipFile(path) as archive:
        member = next(
            (n for n in archive.namelist() if n.lower().endswith(".csv")), None
        )
        if member is None:
            raise ValueError(f"{path}: no .csv inside the archive")
        with io.TextIOWrapper(archive.open(member), encoding="utf-8",
                              errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                recipe_id = (row.get("recipe_id") or "").strip()
                title = (row.get("title") or "").strip()
                if not recipe_id or not title:
                    continue
                ingredients = _ingredient_names(row.get("ingredients"))
                if not ingredients:
                    continue
                yield {
                    "id": recipe_id,
                    "name": title,
                    "ingredients": ingredients,
                    "tags": _literal_list(row.get("tags")),
                }


# --- choosing which rows to keep --------------------------------------------

def _cell_of(derived: dict) -> Tuple[str, str]:
    diet = "vegetarian" if derived["vegetarian"] else "omnivore"
    return (diet, derived["spice_level"])


def _keep(bucket: List[dict], seen: int, limit: int, item: dict, rng) -> None:
    """Algorithm R: a uniform `limit`-sized sample of a stream, deterministically."""
    if len(bucket) < limit:
        bucket.append(item)
        return
    pick = rng.randrange(seen)
    if pick < limit:
        bucket[pick] = item


def select(rows: Iterator[dict], seed: int = 1,
           quota: Dict[Tuple[str, str], int] = None,
           targets: Sequence[str] = None,
           targets_per_cell: int = TARGETS_PER_CELL) -> List[dict]:
    """Pick the benchmark's catalogue: per-cell quotas plus exclusion carriers.

    Deterministic for a given `rows` order and `seed`. Returns rows sorted by
    id, so a rebuild produces a byte-identical file.
    """
    quota = QUOTA if quota is None else quota
    targets = TARGET_INGREDIENTS if targets is None else targets
    rng = random.Random(seed)

    cells: Dict[Tuple[str, str], List[dict]] = {cell: [] for cell in quota}
    carriers: Dict[Tuple[Tuple[str, str], str], List[dict]] = {
        (cell, target): [] for cell in quota for target in targets
    }
    cell_seen: Dict[Tuple[str, str], int] = {cell: 0 for cell in quota}
    carrier_seen: Dict[Tuple[Tuple[str, str], str], int] = {
        key: 0 for key in carriers
    }
    first_seen: set = set()

    for row in rows:
        if row["id"] in first_seen:
            continue
        first_seen.add(row["id"])

        derived = derive_attributes(row)
        cell = _cell_of(derived)
        if cell not in cells:
            continue

        cell_seen[cell] += 1
        _keep(cells[cell], cell_seen[cell], quota[cell], row, rng)

        present = derived["ingredients"]
        for target in targets:
            if target not in present:
                continue
            key = (cell, target)
            carrier_seen[key] += 1
            _keep(carriers[key], carrier_seen[key], targets_per_cell, row, rng)

    chosen = {row["id"]: row for cell in cells for row in cells[cell]}
    for bucket in carriers.values():
        for row in bucket:
            chosen[row["id"]] = row
    return sorted(chosen.values(), key=lambda row: row["id"])


# --- the CLI ----------------------------------------------------------------

def build(source: str = None, out: str = None, seed: int = None) -> dict:
    """Read HUMMUS, select, write the catalogue. Returns a summary dict."""
    source = str(source or config.HUMMUS_RECIPES_PATH)
    out = str(out or config.CATALOGUE_PATH)
    seed = config.CATALOGUE_SEED if seed is None else seed

    rows_read = 0

    def counting(iterator: Iterator[dict]) -> Iterator[dict]:
        nonlocal rows_read
        for row in iterator:
            rows_read += 1
            yield row

    selected = select(counting(open_rows(source)), seed=seed)

    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(selected, handle, indent=1, ensure_ascii=False, sort_keys=True)
        handle.write("\n")

    counts: Dict[str, int] = {}
    for row in selected:
        cell = " ".join(_cell_of(derive_attributes(row)))
        counts[cell] = counts.get(cell, 0) + 1
    carriers = {
        target: sum(1 for row in selected
                    if target in derive_attributes(row)["ingredients"])
        for target in TARGET_INGREDIENTS
    }
    return {"read": rows_read, "selected": len(selected), "out": out,
            "cells": counts, "carriers": carriers}


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cut memproj/data/recipes_hummus.json out of HUMMUS."
    )
    parser.add_argument("--source", default=None,
                        help="HUMMUS pp_recipes.zip (default: "
                             "config.HUMMUS_RECIPES_PATH)")
    parser.add_argument("--out", default=None,
                        help="default: config.CATALOGUE_PATH")
    parser.add_argument("--seed", type=int, default=None,
                        help="reservoir seed (default: config.CATALOGUE_SEED)")
    args = parser.parse_args(argv)

    summary = build(args.source, args.out, args.seed)
    print(f"read     {summary['read']:,} rows")
    print(f"selected {summary['selected']:,} recipes -> {summary['out']}")
    for cell, n in sorted(summary["cells"].items()):
        print(f"  {cell:<18} {n:>4}")
    for target, n in sorted(summary["carriers"].items()):
        print(f"  carries {target:<10} {n:>4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
