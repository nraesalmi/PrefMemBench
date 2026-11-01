"""The surface forms: a vague statement must not read as a resolved one.

Templates are chosen by the generator and rendered from events; nobody is
expected to recover an event from the wording later. But the wording still
decides whether the sentence a model reads is genuinely vague, and a form in
the vague pool that names its referent hands the extractor the answer gold
deliberately withholds -- a "clarify" case any model can solve by turning
"that restriction" into "the mushroom one".

No LLM call appears here, and none may be added.
"""

import random

from memproj import templates
from memproj.templates import EXPLICIT, IMPLICIT, TIME_PHRASES, render_ack
from memproj.templates import _fill, render, surfaces_for

ALL_TYPES = list(EXPLICIT)

# Words that name WHICH restriction is meant. The user's excluded ingredients
# are always food words; a vague statement may only point by referring back
# ("that restriction", "that one"), never by name.
REFERENT_NOUNS = {"mushroom", "peanut", "aubergine", "milk", "cheese", "fish",
                  "onion", "garlic", "tomato", "egg", "prawn", "chilli"}


def test_every_type_has_explicit_forms():
    for t in ALL_TYPES:
        assert len(EXPLICIT[t]) >= 3, t


def test_filler_pool_is_large_enough_to_pad_sessions():
    assert len(EXPLICIT["filler"]) >= 8


def test_every_type_has_an_implicit_pool_not_shared_with_explicit():
    for t in ALL_TYPES:
        if t == "filler":
            continue
        assert len(IMPLICIT[t]) >= 3, t
        assert IMPLICIT[t] is not EXPLICIT[t]
        assert not set(IMPLICIT[t]) & set(EXPLICIT[t])


def test_placeholders_only_appear_in_types_that_have_them():
    """A {...} slot in the wrong pool renders a value into a sentence that
    should not carry one (a filler with a spice level, an ambiguity naming
    its referent)."""
    # `{referent}` names WHICH restriction a resolved ambiguity drops, so it
    # may appear only in that pool; `{old_value}` is the value a correction is
    # replacing, so it may appear only in the two correction pools.
    _OWNERS = {"{referent}": ("ambiguous_update_resolved",),
               "{old_value}": ("correction", "correction_exclusion")}
    for t, forms in list(EXPLICIT.items()) + list(IMPLICIT.items()):
        for f in forms:
            for slot, owners in _OWNERS.items():
                assert slot not in f or t in owners, (t, f, slot)
            assert "{time_phrase}" not in f or t == "temporary_exception", (t, f)
            # `diet_other` is the second pool allowed to name a person: it
            # is the diet statement about someone else ("my brother is
            # vegetarian"), and no user-subject pool may draw from it.
            assert "{person}" not in f or t in ("other_person",
                                                "diet_other"), (t, f)
            if t == "filler":
                assert "{" not in f, f
            if t == "ambiguous_update":
                for slot in ("{value}", "{attribute}", "{person}"):
                    assert slot not in f, f
            for noun in REFERENT_NOUNS:
                assert noun not in f.lower() or "{value}" in f, (t, f)


def test_ambiguous_forms_name_no_referent():
    """The vague pool must stay vague: nothing in it says WHICH restriction."""
    for t in ("ambiguous_update",):
        forms = EXPLICIT[t] + IMPLICIT[t]
        for f in forms:
            low = f.lower()
            for name in ("mushroom", "peanut", "aubergine"):
                assert name not in low, f
            assert "{value}" not in f, f


def test_render_is_deterministic_for_a_seed_and_draws_only_from_its_pool():
    for t in ALL_TYPES:
        if t == "filler":
            continue
        for seed in range(20):
            a = render(t, random.Random(seed), attribute="spice_level",
                       value="spicy", person="brother", time_phrase="tonight")
            b = render(t, random.Random(seed), attribute="spice_level",
                       value="spicy", person="brother", time_phrase="tonight")
            assert a == b
            text, is_implicit = a
            pool = IMPLICIT[t] if is_implicit else EXPLICIT[t]
            assert text in {templates._fill(f, attribute="spice_level",
                                            value="spicy", person="brother",
                                            time_phrase="tonight")
                            for f in pool}, (t, text)


def test_unknown_type_renders_nothing_rather_than_crashing():
    assert render("not_a_type", random.Random(0)) == ("", False)


def test_render_ack_comes_from_the_acknowledgement_pool():
    for seed in range(10):
        assert render_ack(random.Random(seed)) in templates.ASSISTANT_ACKS


def test_surfaces_for_returns_the_pools():
    for t in ALL_TYPES:
        assert surfaces_for(t) == (EXPLICIT[t], IMPLICIT[t])
    assert surfaces_for("not_a_type") == ((), ())


def test_time_phrases_are_the_ones_the_resolver_understands():
    """_temporary_window maps exactly these phrases to windows."""
    from datetime import datetime
    from memproj.families import _temporary_window
    base = datetime(2026, 3, 1, 18, 0)
    for phrase, _days in TIME_PHRASES:
        assert _temporary_window(base, phrase) > base, phrase
    assert len({p for p, _ in TIME_PHRASES}) == len(TIME_PHRASES)


def test_fill_renders_every_slot_the_generator_supplies():
    """`attribute` and `person` are always supplied, so a form that drops its
    {value} renders an empty sentence ("Honestly,  is just how I eat")."""
    for t in ALL_TYPES:
        if t == "filler":
            continue
        for form in EXPLICIT[t] + IMPLICIT[t]:
            text = _fill(form, attribute="spice_level", value="spicy",
                         person="brother", time_phrase="tomorrow")
            assert "{" not in text and "}" not in text, (t, form)
            assert "  " not in text, (t, form)


def test_value_bearing_types_use_their_value():
    """Except the vague and what-if pools, every statement names its value:
    a form that drops the slot discards the event's content."""
    for t in ALL_TYPES:
        if t in ("filler", "hypothetical", "ambiguous_update",
                 "ambiguous_update_resolved"):
            continue
        for form in EXPLICIT[t] + IMPLICIT[t]:
            assert "{value}" in form, (t, form)


# --- the ingredient pool (2026-10-27) ---------------------------------------
# Step 6's contamination table showed the extraction storing "peanut
# excluded" for a sentence whose answer key says peanut is *wanted*, and
# keeping a mushroom exclusion after the user dropped it. The cause was
# wording: one pool served both attributes, so "Honestly, peanut is just how
# I eat" was an exclusion and "I usually prefer mild meals" was a preference.
# An ingredient statement now gets its own pool, and every form in it must
# say the ingredient is out.

_POSITIVE = ("I usually prefer", "I normally eat", "My default is",
             "I try to stick to", "Generally I like", "I lean",
             "People know me as", "Whatever you make", "is just how I eat",
             "kind of person", "usual choice")


def test_an_ingredient_statement_gets_the_exclusion_pool():
    # Every ingredient pool form, rendered with a value, must read as an
    # exclusion -- not merely avoid the banned words, but contain one.
    forms = EXPLICIT["ingredient_exclusion"] + IMPLICIT["ingredient_exclusion"]
    assert forms
    for form in forms:
        text = _fill(form, value="mushroom")
        assert any(word in text.lower() for word in
                   ("out", "avoid", "won't", "never", "don't", "can't",
                    "not touching", "clear of", "off the table", "no ",
                    "stomach turn")), text
        assert not any(p in text for p in _POSITIVE), text


def test_render_never_gives_an_ingredient_a_positive_form():
    # render() is where the routing lives; draw many times so a leak has no
    # way to hide behind a lucky RNG.
    rng = random.Random(1)
    for _ in range(200):
        text, implicit = render("initial_pref", rng,
                                attribute="excluded_ingredient", value="peanut")
        assert not any(p in text for p in _POSITIVE), text
        assert isinstance(implicit, bool)


def test_spice_level_keeps_the_general_pool():
    # "I usually prefer mild meals" is a preference and the model reads it
    # correctly, so spice_level is untouched by the re-routings.
    rng = random.Random(1)
    seen = {render("initial_pref", rng, attribute="spice_level",
                   value="mild")[0] for _ in range(200)}
    expected = {_fill(form, attribute="spice_level", value="mild")
                for form in EXPLICIT["initial_pref"] + IMPLICIT["initial_pref"]}
    assert seen <= expected


# --- heat polarity (2026-10-29) ---------------------------------------------
# The general implicit pool's "I'm not really a {value} kind of person" is the
# same polarity defect the ingredient_exclusion pool prevents, one attribute over:
# rendered with value=mild it says the user is NOT mild, while the answer key
# says mild. 14 turns read that way on the 2026-10-29 corpus. The pool must
# never negate the thing it is stating, for any attribute.

_NEGATES_VALUE = ("not really a", "not much of a", "isn't my thing",
                  "never been a", "said no ", "no {value}")


def test_no_general_form_negates_the_value_it_states():
    for t in ("initial_pref", "permanent_revision"):
        for form in EXPLICIT[t] + IMPLICIT[t]:
            low = form.lower()
            assert not any(n in low for n in _NEGATES_VALUE), (t, form)


def test_render_never_negates_a_heat_statement():
    rng = random.Random(7)
    for _ in range(300):
        for value in ("mild", "medium", "spicy"):
            text, _ = render("initial_pref", rng, attribute="spice_level",
                             value=value)
            assert not any(n in text.lower() for n in _NEGATES_VALUE), text


# --- the diet pools (2026-10-27) --------------------------------------------
# Same defect as the ingredient pool, one attribute over: the general
# implicit pool says "I'm not really a {value} kind of person" and its
# correction pool says "I said no {value}", so a vegetarian diet rendered
# from them can say the user is NOT vegetarian. The gold profile never moved
# (it is computed from the event), which is exactly the silent inversion the
# corpus is supposed to avoid.

_NEGATED_DIET = ("not really a", "not much of a", "said no ", "never been a",
                 "isn't my thing", "no {value}")


def test_a_diet_statement_gets_the_diet_pool():
    forms = EXPLICIT["diet"] + IMPLICIT["diet"]
    assert forms
    for form in forms:
        assert "{value}" in form, form
        low = form.lower()
        assert not any(n in low for n in _NEGATED_DIET), form
        assert "{person}" not in form, form


def test_a_diet_statement_about_someone_else_gets_its_own_pool():
    forms = EXPLICIT["diet_other"] + IMPLICIT["diet_other"]
    assert forms
    for form in forms:
        assert "{value}" in form and "{person}" in form, form
        assert not any(n in form.lower() for n in _NEGATED_DIET), form


# --- corrections (2026-10-29) -----------------------------------------------
# `pool_for` used to send every excluded_ingredient event to the exclusion
# pool, whatever its type, so all 16 correction turns read as fresh exclusion
# statements ("I don't do aubergine, full stop"). `correct_preference` was
# therefore unreachable from the prose and the `no_correction` ablation
# row was flat. A correction must say it is correcting, and must name the
# record it fixes.

_CORRECTIVE = ("misunderstood", "correct", "wrong", "heard", "meant",
               "told you", "for the last time", "how many times",
               "get it right", "get it straight", "said")


def test_a_correction_says_it_is_correcting():
    for key in ("correction", "correction_exclusion"):
        forms = EXPLICIT[key] + IMPLICIT[key]
        assert forms, key
        for form in forms:
            assert "{value}" in form, (key, form)
            assert any(c in form.lower() for c in _CORRECTIVE), (key, form)


def test_an_ingredient_correction_keeps_the_exclusion_polarity():
    """The corrected value is still the thing to AVOID -- a correction that
    reads as a new preference would invert the fact it corrects."""
    forms = EXPLICIT["correction_exclusion"] + IMPLICIT["correction_exclusion"]
    for form in forms:
        text = _fill(form, value="aubergine", old_value="peanut")
        assert not any(p in text for p in _POSITIVE), text


def test_render_routes_a_correction_away_from_the_exclusion_pool():
    rng = random.Random(3)
    ingredient_forms = {_fill(f, value="aubergine", old_value="peanut")
                        for f in EXPLICIT["ingredient_exclusion"]
                        + IMPLICIT["ingredient_exclusion"]}
    for _ in range(200):
        text, _ = render("correction", rng, attribute="excluded_ingredient",
                         value="aubergine", old_value="peanut")
        assert text not in ingredient_forms, text
        assert "aubergine" in text, text


# --- a resolved ambiguity names its referent (2026-10-29) -------------------
# 11 `ambiguous_update` events carry a `referent_event_id` and 0 of them said
# which restriction they meant, so the matched twin of `ambiguous_vs_resolved`
# was unanswerable: correctly asking "which restriction?" was scored wrong.
# A resolved statement names it; a genuinely vague one still must not.

def test_the_resolved_pool_names_the_referent():
    for key in ("ambiguous_update_resolved",):
        forms = EXPLICIT[key] + IMPLICIT[key]
        assert forms, key
        for form in forms:
            assert "{referent}" in form, (key, form)
            for noun in REFERENT_NOUNS:
                assert noun not in form.lower(), (key, form)


def test_the_vague_pool_stays_vague_next_to_the_resolved_one():
    vague = set(EXPLICIT["ambiguous_update"] + IMPLICIT["ambiguous_update"])
    resolved = set(EXPLICIT["ambiguous_update_resolved"]
                   + IMPLICIT["ambiguous_update_resolved"])
    assert not vague & resolved


def test_render_names_the_referent_only_when_there_is_one():
    rng = random.Random(5)
    for _ in range(200):
        named, _ = render("ambiguous_update", rng, referent="mushroom")
        assert "mushroom" in named.lower(), named
        vague, _ = render("ambiguous_update", rng)
        assert "mushroom" not in vague.lower(), vague


def test_render_routes_diet_by_subject():
    rng = random.Random(1)
    for _ in range(200):
        text, _implicit = render("initial_pref", rng, attribute="diet",
                                 value="vegetarian", person="brother")
        assert "brother" not in text, text
        user_forms = {_fill(f, attribute="diet", value="vegetarian",
                            person="brother")
                      for f in EXPLICIT["diet"] + IMPLICIT["diet"]}
        assert text in user_forms, text

        other, _implicit = render("other_person", rng, attribute="diet",
                                  value="vegetarian", person="brother")
        other_forms = {_fill(f, attribute="diet", value="vegetarian",
                             person="brother")
                       for f in EXPLICIT["diet_other"] + IMPLICIT["diet_other"]}
        assert other in other_forms, other
