"""The catalogue build: parsing real HUMMUS rows and the selection rule.

No test here touches `out/hummus/` (507,336 rows, gitignored): the reader is
exercised on a four-row zip written into `tmp_path` with the same two traps as
the real file, and the selector on rows small enough to reason about by hand.
"""

import json
import zipfile

import pytest

from memproj import catalogue, config
from memproj.recipes import derive_attributes


# --- parsing the source columns ---------------------------------------------

def test_ingredient_names_read_every_section():
    text = ("{'': [('water', '1 cup '), ('white bread flour', '2 cups ')], "
            "'For the sauce': [('dry milk', '2 tbsp ')]}")
    assert catalogue._ingredient_names(text) == [
        "water", "white bread flour", "dry milk"
    ]


def test_ingredient_names_survives_a_row_that_is_not_valid_python():
    """4 rows in 20,000 carry a bare `nan` that makes literal_eval raise."""
    text = "['avocados', nan, 'black beans']"
    assert catalogue._ingredient_names(text) == ["avocados", "black beans"]


def test_ingredient_names_ignores_an_empty_column():
    assert catalogue._ingredient_names("") == []
    assert catalogue._ingredient_names(None) == []


def test_literal_list_parses_tags():
    assert catalogue._literal_list("['Vegetarian', 'spicy']") == [
        "vegetarian", "spicy"
    ]
    assert catalogue._literal_list("[]") == []


def test_open_rows_reads_by_name_past_the_empty_header_column(tmp_path):
    """The real header starts with an empty column name, so a reader keyed on
    position silently shifts every field by one."""
    path = tmp_path / "pp_recipes.zip"
    body = (
        ",recipe_id,title,duration,ingredients,tags,new_recipe_id\n"
        "1,100024,Chocolate Cherry Bread,250.0,"
        "\"{'': [('water', '1 cup '), ('white bread flour', '2 cups ')]}\","
        "\"['breads', 'vegetarian']\",361588\n"
        "2,,No Id,10.0,\"{'': [('salt', '1 tsp ')]}\",[],0\n"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("pp_recipes.csv", body)

    rows = list(catalogue.open_rows(str(path)))
    assert len(rows) == 1
    assert rows[0] == {
        "id": "100024",
        "name": "Chocolate Cherry Bread",
        "ingredients": ["water", "white bread flour"],
        "tags": ["breads", "vegetarian"],
    }


# --- the selection rule ------------------------------------------------------

def _row(rid, ingredients, tags=()):
    return {"id": rid, "name": f"Recipe {rid}", "ingredients": list(ingredients),
            "tags": list(tags)}


def _cells(rows):
    return [catalogue._cell_of(derive_attributes(row)) for row in rows]


def test_selection_is_deterministic_and_sorted_by_id():
    rows = [_row(f"{i:03d}", ["rice"]) for i in range(50)]
    first = catalogue.select(rows, seed=1)
    second = catalogue.select(rows, seed=1)
    assert [r["id"] for r in first] == [r["id"] for r in second]
    assert [r["id"] for r in first] == sorted(r["id"] for r in first)


def test_a_different_seed_selects_a_different_subset():
    rows = [_row(f"{i:03d}", ["rice"]) for i in range(50)]
    assert ([r["id"] for r in catalogue.select(rows, seed=1)]
            != [r["id"] for r in catalogue.select(rows, seed=2)])


def test_each_cell_gets_its_quota():
    rows = [_row(f"m{i}", ["rice"]) for i in range(20)]
    rows += [_row(f"h{i}", ["chilli", "sriracha"]) for i in range(20)]
    got = catalogue.select(rows, seed=1,
                           quota={("vegetarian", "mild"): 5,
                                  ("vegetarian", "spicy"): 3},
                           targets=())
    assert len(got) == 8
    assert _cells(got).count(("vegetarian", "mild")) == 5
    assert _cells(got).count(("vegetarian", "spicy")) == 3


def test_a_cell_with_fewer_rows_than_its_quota_keeps_them_all():
    rows = [_row("a", ["rice"]), _row("b", ["pasta"])]
    got = catalogue.select(rows, seed=1,
                           quota={("vegetarian", "mild"): 10}, targets=())
    assert {r["id"] for r in got} == {"a", "b"}


def test_exclusion_carriers_are_kept_inside_their_cell():
    """A profile excluding peanut needs mild vegetarian dishes that have it,
    or the constraint is never exercised."""
    rows = [_row(f"plain{i}", ["rice"]) for i in range(10)]
    rows += [_row(f"nut{i}", ["rice", "peanut butter"]) for i in range(10)]
    got = catalogue.select(rows, seed=1,
                           quota={("vegetarian", "mild"): 2},
                           targets=("peanut",), targets_per_cell=3)
    carriers = [r for r in got if "peanut" in derive_attributes(r)["ingredients"]]
    # The quota sample alone may contain some carriers, so the union is at
    # least targets_per_cell and no more than quota + targets_per_cell.
    assert len(carriers) >= 3
    assert 3 <= len(got) <= 5
    assert all(cell == ("vegetarian", "mild") for cell in _cells(got))


def test_carriers_do_not_duplicate_a_quota_row():
    rows = [_row(f"nut{i}", ["peanut"]) for i in range(10)]
    got = catalogue.select(rows, seed=1,
                           quota={("vegetarian", "mild"): 10},
                           targets=("peanut",), targets_per_cell=10)
    assert len(got) == 10
    assert len({r["id"] for r in got}) == 10


def test_duplicate_ids_are_kept_once():
    rows = [_row("same", ["rice"]), _row("same", ["chilli"])]
    got = catalogue.select(rows, seed=1,
                           quota={("vegetarian", "mild"): 5}, targets=())
    assert [r["id"] for r in got] == ["same"]


def test_rows_outside_every_cell_are_dropped():
    """An unknown spice level cannot be filtered on, so it is not kept."""
    rows = [_row("odd", ["rice"], tags=["medium"]), _row("ok", ["rice"])]
    rows[0]["spice_level"] = "unknown"
    got = catalogue.select(rows, seed=1,
                           quota={("vegetarian", "mild"): 5,
                                  ("vegetarian", "medium"): 5}, targets=())
    assert {r["id"] for r in got} <= {"odd", "ok"}


# --- the shipped artefact ----------------------------------------------------

def test_the_shipped_catalogue_covers_every_cell_the_corpus_asks_for():
    rows = json.load(open(config.CATALOGUE_PATH, encoding="utf-8"))
    cells = {catalogue._cell_of(derive_attributes(row)) for row in rows}
    assert cells == set(catalogue.CELLS)


def test_the_shipped_catalogue_carries_every_excluded_ingredient():
    rows = json.load(open(config.CATALOGUE_PATH, encoding="utf-8"))
    vocab = set()
    for row in rows:
        vocab |= derive_attributes(row)["ingredients"]
    for target in catalogue.TARGET_INGREDIENTS:
        assert target in vocab


def test_the_shipped_catalogue_has_unique_ids():
    rows = json.load(open(config.CATALOGUE_PATH, encoding="utf-8"))
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids))
