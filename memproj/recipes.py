"""Recipe catalogue and profile validation.

The catalogue is `memproj/data/recipes_hummus.json`, a 155-recipe selection
from HUMMUS (see `memproj/catalogue.py` for how it was cut down from 507,336
rows, and the README's "Recipe catalogue" section for what the source rows
contain).
`derive_attributes()` is the single place that knows how a raw catalogue row
maps onto {vegetarian, spice_level, ingredients}.
"""

import json
import re
from typing import Dict, List, Sequence, Set, Tuple


# --- the one place that knows the catalogue's vocabulary --------------------

# Bare words, matched against the *word tokens* of an ingredient phrase:
# HUMMUS says "chicken breast" and "ground beef" rather than bare "chicken",
# so an exact set intersection over whole names misses them.
MEAT_MARKERS = frozenset({
    # poultry and red meat
    "chicken", "beef", "pork", "lamb", "bacon", "ham", "sausage", "turkey",
    "duck", "veal", "venison", "rabbit", "quail", "bison", "steak",
    "mutton", "pheasant", "goose", "meat", "meatball", "meatloaf",
    # charcuterie
    "prosciutto", "chorizo", "salami", "pepperoni", "pancetta", "coppa",
    "bratwurst", "kielbasa", "andouille", "serrano", "jamon",
    # fish and seafood
    "fish", "salmon", "tuna", "anchovy", "anchovies", "cod", "haddock",
    "halibut", "tilapia", "trout", "mackerel", "sardine", "herring", "bass",
    "prawn", "prawns", "shrimp", "crab", "lobster", "clam", "mussel",
    "oyster", "scallop", "squid", "octopus", "calamari", "caviar", "bonito",
    "pollock", "snapper", "swordfish", "eel", "whitebait",
    # animal-derived setting agents and fats
    "gelatin", "gelatine", "lard", "suet", "tallow", "rennet", "isinglass",
})

SPICE_MARKERS = frozenset({
    "chilli", "chillies", "chili", "chile", "jalapeno", "jalapeño", "harissa",
    "sriracha", "cayenne", "paprika", "curry", "ginger", "sichuan", "wasabi",
    "sambal", "tabasco", "chipotle", "habanero", "gochujang", "piri", "peri",
})
# Deliberately absent: "pepper" (bell and black pepper are not heat sources in
# the mild/medium/spicy sense the benchmark uses) and "scotch" ("scotch
# bonnet" is spicy, "scotch whisky" is not). Both would push most of the
# catalogue to medium or above and flatten the spice vocabulary.

# Tags that name an ingredient class we may need to exclude even when the
# ingredient list does not spell it out.
TAG_INGREDIENT_ALIASES = {
    "contains-nuts": "peanut",
    "nuts": "peanut",
    "contains-mushrooms": "mushroom",
    "contains-dairy": "dairy",
    "contains-gluten": "gluten",
}


_WORD_SPLIT = re.compile(r"[\W_]+", re.UNICODE)


def _singular(word: str) -> str:
    """'mushrooms' -> 'mushroom'; 'tomatoes' -> 'tomato'. Crude, on purpose."""
    if word.endswith("oes"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _normalise(token: str) -> str:
    """'Peanut Butter' -> 'peanut'; 'Mushrooms' -> 'mushroom'.

    Crude singularisation, sufficient for this catalogue's vocabulary.
    `tools._check_value` and `eval.metrics.canonical` apply the same rule, so
    changing it changes how stored facts compare -- not a local decision.
    """
    t = token.strip().lower()
    for suffix in (" butter", " paste", " sauce", " oil", " powder", " seeds",
                   " seed", " nuts", " nut"):
        if t.endswith(suffix):
            t = t[: -len(suffix)]
            break
    return _singular(t)


def _words(name: str) -> Set[str]:
    """The singularised word tokens of one ingredient name.

    'ground beef' -> {'ground', 'beef'}, so a marker can match a phrase the
    catalogue never spells as a bare word.
    """
    return {_singular(w) for w in _WORD_SPLIT.split(str(name).lower()) if w}


# Both marker sets are compared against `_words()` output, which is
# singularised, so they are singularised the same way -- 'chillies' and
# 'chilli' have to land on one token.
_MEAT_WORDS = frozenset(_singular(w) for w in MEAT_MARKERS)
_SPICE_WORDS = frozenset(_singular(w) for w in SPICE_MARKERS)


def derive_attributes(recipe: dict) -> dict:
    """Map a raw catalogue row to the fields validate() reasons about.

    This is the one function that knows HUMMUS's raw vocabulary; every other
    module reasons about the derived {vegetarian, spice_level, ingredients}.
    """
    ingredient_names = list(recipe.get("ingredients", []))
    ingredients = {_normalise(i) for i in ingredient_names}
    tags = {t.strip().lower() for t in recipe.get("tags", [])}

    # An ingredient class named by a tag still counts as present.
    for tag, ingredient in TAG_INGREDIENT_ALIASES.items():
        if tag in tags:
            ingredients.add(ingredient)

    words: Set[str] = set()
    for name in ingredient_names:
        words |= _words(name)

    # A tag that says vegetarian is trusted over the ingredient list; without
    # one, meat is decided by word, not by whole-name match, because HUMMUS
    # writes "chicken breast" and "ground beef".
    vegetarian = "vegetarian" in tags or "vegan" in tags
    if not vegetarian:
        vegetarian = not (words & _MEAT_WORDS)

    heat = words & _SPICE_WORDS
    if "spicy" in tags:
        spice_level = "spicy"
    elif "medium" in tags:
        spice_level = "medium"
    elif heat:
        # Two or more heat sources reads as genuinely spicy.
        spice_level = "spicy" if len(heat) >= 2 else "medium"
    else:
        spice_level = "mild"

    return {
        "vegetarian": vegetarian,
        "spice_level": spice_level,
        "ingredients": ingredients,
    }


# --- validation -------------------------------------------------------------

# A recipe is acceptable if its heat is at or below the requested level.
SPICE_ORDER = {"mild": 0, "medium": 1, "spicy": 2}


def validate(recipe: dict, profile: Dict[str, Set[str]]) -> Tuple[bool, List[str]]:
    """Check one already-derived recipe against a gold profile.

    Pure function. Every violation is a human-readable string so the
    generator's failure message says which constraint it could not meet.
    """
    violations: List[str] = []

    if "diet" in profile:
        wants_vegetarian = "vegetarian" in profile["diet"]
        if wants_vegetarian and not recipe.get("vegetarian", False):
            violations.append(
                f"{recipe.get('id', '?')}: not vegetarian but profile requires it"
            )

    if "spice_level" in profile:
        allowed = profile["spice_level"] & set(SPICE_ORDER)
        if allowed:
            ceiling = max(SPICE_ORDER[v] for v in allowed)
            actual = recipe.get("spice_level", "mild")
            if SPICE_ORDER.get(actual, 0) > ceiling:
                violations.append(
                    f"{recipe.get('id', '?')}: spice_level '{actual}' exceeds "
                    f"allowed {'/'.join(sorted(allowed, key=SPICE_ORDER.get))}"
                )

    excluded = profile.get("excluded_ingredient", set())
    if excluded:
        present = {_normalise(e) for e in excluded} & recipe.get("ingredients", set())
        for hit in sorted(present):
            violations.append(
                f"{recipe.get('id', '?')}: contains excluded ingredient '{hit}'"
            )

    return (not violations), violations


# --- loading ----------------------------------------------------------------

def load_catalogue(path: str) -> List[dict]:
    """Read the raw JSON catalogue. Rows keep their original shape; callers
    derive attributes via derive_attributes()."""
    with open(path, "r", encoding="utf-8") as fh:
        rows = json.load(fh)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON list of recipes")
    return rows


def load_derived(path: str) -> List[dict]:
    """Catalogue rows with derived attributes merged in, ready for validate().

    `id` and `name` are preserved so violations name the offending recipe.
    """
    out = []
    for row in load_catalogue(path):
        derived = derive_attributes(row)
        out.append({
            "id": row["id"],
            "name": row["name"],
            "vegetarian": derived["vegetarian"],
            "spice_level": derived["spice_level"],
            "ingredients": derived["ingredients"],
            "tags": list(row.get("tags", [])),
        })
    return out


def valid_recipes(catalogue: Sequence[dict], profile: Dict[str, Set[str]]) -> List[dict]:
    """Every derived recipe that satisfies `profile`."""
    return [r for r in catalogue if validate(r, profile)[0]]


def ingredient_vocabulary(catalogue: Sequence[dict]) -> Set[str]:
    """Every ingredient name appearing anywhere, for the generator to sample."""
    vocab: Set[str] = set()
    for r in catalogue:
        vocab |= r["ingredients"]
    return vocab
