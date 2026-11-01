"""Surface forms: turn an Event into the words a person would say.

Two lists per event type. EXPLICIT forms state the preference plainly;
IMPLICIT forms imply it ("not mushrooms again"). Keeping them apart lets a
later experiment separate the two without re-tagging anything.

Assistant turns are acknowledgement-only. They must never carry a
preference value -- the leakage assertions live in tests/test_generate.py.
"""

import random
from typing import Dict, List, Sequence, Tuple


# --- user turns -------------------------------------------------------------

EXPLICIT: Dict[str, List[str]] = {
    "initial_pref": [
        "I usually prefer {value} meals.",
        "Just so you know, I tend to go for {value} food.",
        "For what it's worth, I'm {value} when it comes to {attribute}.",
        "I normally eat {value}.",
        "Most of the time I'd say I'm {value}.",
        "My default is {value}.",
        "I try to stick to {value} food.",
        "Generally I like {value} cooking.",
        "I'd say {value} is my usual choice.",
        "I lean {value} most days.",
    ],
    "permanent_revision": [
        "Actually, I now prefer {value}.",
        "I've changed my mind -- from now on, {value}.",
        "These days I'm {value}, not what I said before.",
        "I want to switch to {value} going forward.",
        "Update that: I'm {value} permanently now.",
        "Let's make it {value} from here on.",
        "I've gone off the old one, I'm {value} now.",
        "New rule: {value}.",
        "I'd like to change my {attribute} to {value}.",
        "Forget the earlier one, I'm {value} these days.",
    ],
    "temporary_exception": [
        "Only for {time_phrase}, {value} is fine.",
        "Just {time_phrase} I'll make an exception -- {value} works.",
        "For {time_phrase} only, give me {value}.",
        "{time_phrase} I'm fine with {value}, but only then.",
        "Tonight aside, ignore this: {time_phrase} {value} is okay.",
        "I'll allow {value} just {time_phrase}.",
        "Special case: {time_phrase} I want {value}.",
        "{time_phrase} I'd take {value}, don't read into it.",
        "An exception {time_phrase}: {value}.",
        "Throw caution out {time_phrase}, {value} please.",
    ],
    "other_person": [
        "My {person} likes {value} food.",
        "My {person} is really into {value}.",
        "By the way, my {person} prefers {value}.",
        "My {person} would want {value}.",
        "You should know my {person} is {value}.",
        "My {person} tends to go for {value}.",
        "For my {person}, keep it {value}.",
        "My {person} always asks for {value}.",
        "My {person} won't eat anything but {value}.",
        "My {person} is a {value} person.",
    ],
    "hypothetical": [
        "If I liked {value} food, what would you suggest?",
        "Suppose I were {value} -- what then?",
        "Hypothetically, if my {attribute} were {value}, what fits?",
        "What would you pick if I wanted {value}?",
        "Just curious: what if I were {value}?",
        "Imagine I was {value} for a day, what would you make?",
        "Say I suddenly became {value}, what's on the menu?",
        "In theory, if I asked for {value}, what would you say?",
        "If I weren't {value}-averse, what would you offer?",
        "Playing devil's advocate: {value} -- what would you do?",
    ],
    # A correction must SOUND like a correction, and it used to not: the
    # ingredient pool served it (pool_for routed on attribute alone), so all
    # 16 correction turns read as fresh exclusions ("I don't do aubergine,
    # full stop") and `correct_preference` was unreachable from the prose.
    # Every form here says the user is correcting a record and names both the
    # value that is right and, where it is known, the value it replaces.
    # Polarity is handled by the pool, not by the wording: this one is the
    # attribute-neutral version, `correction_exclusion` is the exclusion one.
    "correction": [
        "You misunderstood -- I meant {value}, not {old_value}.",
        "That's not what I said; I said {value}.",
        "Let me correct that: {value}, not {old_value}.",
        "You've got it wrong -- it's {value}.",
        "To be clear, I meant {value}.",
        "Wrong one. I said {value}.",
        "No, no -- I said {value}.",
        "Careful -- I meant {value}, not {old_value}.",
        "I said it before: {value}.",
        "That's wrong -- fix it to {value}.",
    ],
    # A correction on an excluded ingredient. It still has to say the value
    # is the thing to AVOID -- a correction that reads as a new preference
    # would invert the fact it is correcting -- the same polarity defect the
    # ingredient_exclusion pool exists to prevent, one event type over.
    "correction_exclusion": [
        "You misunderstood -- it's {value} I avoid, not {old_value}.",
        "No, I said no {value} -- {old_value} is fine.",
        "Let me correct that: keep {value} out, not {old_value}.",
        "You heard wrong. {value} is the one I can't eat.",
        "That's not what I meant -- {value} is off the table.",
        "I said avoid {value}, not {old_value}.",
    ],
    "ambiguous_update": [
        "That restriction is no longer important.",
        "You can drop that one.",
        "That's not a thing anymore.",
        "Don't worry about that restriction.",
        "I've let go of that.",
        "That one doesn't matter now.",
        "Scrap that restriction.",
        "It's fine to ignore that.",
        "That's relaxed now.",
        "You can forget about that.",
    ],
    # A resolved ambiguity -- one that carries a `referent_event_id` -- must
    # name which restriction it drops. For two years of measurement it did
    # not: 11 of 11 resolved events read exactly like the vague pool, so the
    # matched twin of `ambiguous_vs_resolved` was unanswerable and a model
    # that correctly asked "which restriction?" was scored wrong. The vague
    # pool above is still the only one a genuinely unresolved event may use.
    "ambiguous_update_resolved": [
        "The {referent} one is no longer important.",
        "You can drop the {referent} restriction.",
        "Don't worry about the {referent} one anymore.",
        "I've let go of the {referent} rule.",
        "That {referent} restriction doesn't matter now.",
        "The {referent} thing is fine to ignore.",
    ],
    "filler": [
        "Terrible weather today, isn't it?",
        "I've been cooking a lot lately.",
        "My kitchen is a disaster right now.",
        "Do you ever burn things by accident?",
        "I keep meaning to buy a new pan.",
        "The shop was out of everything yesterday.",
        "I spent ages queueing this morning.",
        "I think my oven runs a bit hot.",
        "Do you like grocery shopping?",
        "I've got far too many cookbooks.",
    ],
    # `excluded_ingredient` only. A positive form ("I usually prefer {} meals")
    # means an ingredient is wanted, so it can never state an exclusion: with
    # one sentence for both polarities the model read "Honestly, peanut is
    # just how I eat" as "peanut is fine" while the answer key meant "no
    # peanut". `render()` picks this pool whenever the attribute is an
    # exclusion, so these are the only words an excluded_ingredient statement
    # is ever given.
    "ingredient_exclusion": [
        "I don't eat {value}, so please leave it out.",
        "Please leave {value} out entirely.",
        "Keep {value} out -- I don't eat it at all.",
        "You should know I avoid {value} completely.",
        "I never touch {value}, that's off the table for me.",
        "Steer clear of {value} when you cook for me.",
        "No {value} for me, ever.",
        "I can't eat {value}, so it has to go.",
    ],
    # `diet` statements, for the same reason: the general pool reads wrong
    # for a diet. Its implicit forms literally invert it -- "I'm not really
    # a {value} kind of person" states the OPPOSITE of a diet event -- and
    # its correction forms ("I said no vegetarian") invert it outright.
    # Every form has to carry {value}, because that is the fact the event
    # records.
    "diet": [
        "I'm {value}, so please keep that in mind.",
        "Just so you know, I'm strictly {value}.",
        "Everything I eat is {value} — that's my diet.",
        "I stick to {value} food at every meal.",
        "My diet is {value}, and that's the one rule.",
        "For me it's {value} all the way.",
    ],
    # The same fact about someone else. A separate key because these forms
    # need {person}, and a sentence with a {person} slot must never be
    # drawn for a statement about the user.
    "diet_other": [
        "My {person} is {value}.",
        "By the way, my {person} eats only {value} food.",
        "Keep it {value} for my {person}.",
        "My {person} has been {value} for years.",
        "My {person} is strictly {value}, so bear that in mind.",
    ],
}

IMPLICIT: Dict[str, List[str]] = {
    # `excluded_ingredient` only, for the same reason as the explicit pool.
    "ingredient_exclusion": [
        "Ugh, {value} -- I'm not touching that.",
        "The one thing I won't touch is {value}.",
        "Fair warning: I steer well clear of {value}.",
        "Honestly, {value} makes my stomach turn.",
        "I don't do {value}, full stop.",
        "Never {value} -- I mean never.",
    ],
    "initial_pref": [
        # Was "I'm not really a {value} kind of person", which states the
        # OPPOSITE of the event for every attribute that still uses this
        # pool: rendered with value=mild it said the user is not mild while
        # the answer key said mild (14 turns, 2026-10-29). The same inversion
        # is avoided for excluded_ingredient and diet by giving them their own
        # pools; heat has no separate pool, so the general one has to carry
        # only forms that do not negate their own value.
        "I'm a {value} person through and through, if that helps.",
        "Honestly, {value} is just how I eat.",
        "You'll notice I don't do much else besides {value}.",
        "Whatever you make, keep it {value} on my end.",
        "People know me as the {value} one.",
    ],
    "permanent_revision": [
        "Honestly I've gone right off the old stuff, I'm {value} now.",
        "Turns out I was wrong about myself -- {value} it is.",
        "I've grown into {value}, funnily enough.",
        "The old preference is dead, long live {value}.",
    ],
    "temporary_exception": [
        "Ugh, fine, {value} -- but just {time_phrase}.",
        "I'll regret this, but {value} {time_phrase}.",
        "Don't get used to it: {value} {time_phrase}.",
    ],
    "other_person": [
        "Meanwhile my {person} would inhale anything {value}.",
        "It's my {person} who's the {value} one, not me.",
        "You'd better ask my {person}, they're all about {value}.",
    ],
    "hypothetical": [
        "Weird question -- what if {value} was my thing?",
        "Humour me: {value}, what would that look like?",
        "Purely as a thought experiment, {value}?",
    ],
    "correction": [
        "Ugh, not {old_value} again -- {value}, I told you.",
        "I've said it twice now: {value}.",
        "Do we have to go through this? I said {value}.",
        "For the last time, {value}.",
        "You keep hearing it wrong -- {value}.",
    ],
    "correction_exclusion": [
        "I've told you twice -- it's {value} I don't eat.",
        "Not {old_value}, {value}. How many times?",
        "For the last time, {value} is the one to keep out.",
        "I said {value}, not {old_value}.",
        "Wrong one -- {value} is what I steer clear of.",
    ],
    "diet": [
        "Honestly, {value} is the only way I eat.",
        "Meat never really features -- {value} is my whole diet.",
        "{value} for me, always has been.",
        "You'll never see me eat otherwise -- {value}, full stop.",
    ],
    "diet_other": [
        "It's my {person} who's {value}, not me.",
        "My {person} won't eat anything that isn't {value}.",
        "You'd better ask my {person} -- {value} only.",
        "The {value} one in the family is my {person}.",
    ],
    "ambiguous_update": [
        "Honestly, that whole thing? Over it.",
        "Anyway. That's sorted now.",
        "That's water under the bridge.",
    ],
    "ambiguous_update_resolved": [
        "Honestly, the whole {referent} thing? Over it.",
        "The {referent} business is sorted now.",
        "Forget the {referent} one.",
        "That {referent} restriction can go.",
    ],
    "filler": [
        "Not that it matters, just making conversation.",
        "Anyway, how's your day going?",
        "I'm just killing time, ignore me.",
    ],
}


# Every inert user remark, for the generator to shuffle into sessions.
FILLER_FORMS: List[str] = EXPLICIT["filler"] + IMPLICIT["filler"]


# --- assistant turns --------------------------------------------------------

# Acknowledgements only. No values, no attributes, no names.
ASSISTANT_ACKS: List[str] = [
    "Got it.",
    "Noted.",
    "Understood.",
    "Okay, I'll keep that in mind.",
    "Thanks for telling me.",
    "Right, I've made a note.",
    "Good to know.",
    "Alright.",
    "Sure thing.",
    "I'll remember that.",
]

ASSISTANT_QUESTIONS: List[str] = [
    "Would you like some suggestions?",
    "Anything in particular you're after?",
    "Shall I put something together?",
    "Happy to help whenever you're ready.",
    "Let me know what you'd like.",
]

ASSISTANT_CLARIFY: List[str] = [
    "Which one did you mean?",
    "Sorry -- which restriction are we dropping?",
    "Could you say which one?",
    "I want to get this right: which of them?",
]


# --- time phrases -----------------------------------------------------------

# Relative phrase -> how many days the window lasts, in days (float).
# "tonight" ends at midnight of the same day and is handled specially.
TIME_PHRASES: List[Tuple[str, float]] = [
    ("tonight",     0.0),
    ("tomorrow",    1.0),
    ("this weekend", 2.0),
    ("for the next three days", 3.0),
    ("for a couple of days", 2.0),
    ("just this evening", 0.0),
    ("for the rest of the week", 4.0),
]


# --- rendering --------------------------------------------------------------

def _fill(template: str, *, attribute=None, value=None, person=None,
          time_phrase=None, old_value=None, referent=None) -> str:
    """Substitute only the slots a template actually uses.

    `old_value` and `referent` are the two slots that name the record a
    statement is about: the value a correction replaces, and the restriction
    a resolved ambiguity drops. Both degrade to a vague noun rather than to
    an empty string, so a form never renders as "I meant mushroom, not .".
    """
    out = template
    if "{attribute}" in out:
        out = out.replace("{attribute}", _pretty_attribute(attribute))
    # Before {value}: the two slots are distinct, but replacing the short one
    # first would leave "{old_that}" if they ever overlapped.
    if "{old_value}" in out:
        out = out.replace("{old_value}", old_value or "that")
    if "{referent}" in out:
        out = out.replace("{referent}", referent or "other")
    if "{value}" in out:
        out = out.replace("{value}", value or "")
    if "{person}" in out:
        out = out.replace("{person}", person or "brother")
    if "{time_phrase}" in out:
        out = out.replace("{time_phrase}", time_phrase or "tonight")
    return out


def _pretty_attribute(attribute) -> str:
    if attribute is None:
        return "preferences"
    return {
        "diet": "diet",
        "spice_level": "spice level",
        "excluded_ingredient": "ingredients to avoid",
    }.get(attribute, attribute.replace("_", " "))


def pool_for(event_type: str, attribute, has_referent: bool = False) -> str:
    """Which pool says what this event means for this attribute.

    Two attributes need their own words. A positive form used for
    `spice_level` ("I usually prefer mild meals") is a *preference*, while
    the same sentence used for `excluded_ingredient` ("I usually prefer
    peanut meals") is its opposite -- the food becomes the thing to keep,
    not the thing to avoid. The corpus rendered both with one pool, so an
    ingredient exclusion sounded like a compliment. An exclusion now gets
    the exclusion pool and nothing else.

    `diet` was the same trap one attribute over: the general implicit pool
    said "I'm not really a {value} kind of person", which makes the diet the
    one thing the user does NOT follow, and the correction pool said "I said
    no {value}". A diet statement now gets the diet pool; a diet statement
    about someone else gets `diet_other`, because only that pool may carry
    {person}. Every other attribute keeps the general pools.

    A correction is routed on EVENT TYPE first, not on attribute: routing an
    excluded_ingredient correction to the exclusion pool is what made all 16
    correction turns sound like fresh exclusions, so `correct_preference` was
    unreachable from the prose. Within corrections the attribute still picks
    the polarity (value-is-wanted vs value-is-avoided), and within ambiguities
    the presence of a referent picks the pool.
    """
    if event_type == "correction":
        if attribute == "excluded_ingredient":
            return "correction_exclusion"
        return "correction"
    if event_type == "ambiguous_update" and has_referent:
        return "ambiguous_update_resolved"
    if attribute == "excluded_ingredient":
        return "ingredient_exclusion"
    if attribute == "diet":
        return "diet_other" if event_type == "other_person" else "diet"
    return event_type


def render(
    event_type: str,
    rng: random.Random,
    *,
    attribute=None,
    value=None,
    person=None,
    time_phrase=None,
    old_value=None,
    referent=None,
    implicit_ratio: float = 0.65,
) -> Tuple[str, bool]:
    """Render one user utterance for `event_type`.

    Returns (text, is_implicit). `is_implicit` is recorded on the Turn so a
    later experiment can split explicit from implicit without re-parsing.
    """
    pool_key = pool_for(event_type, attribute, has_referent=referent is not None)
    primary = EXPLICIT.get(pool_key, [])
    secondary = IMPLICIT.get(pool_key, [])
    if not primary and not secondary:
        return "", False  # unknown type: inert, never a crash

    use_implicit = bool(secondary) and rng.random() < implicit_ratio
    if use_implicit:
        pool, is_implicit = secondary, True
    else:
        pool, is_implicit = primary or secondary, False

    template = rng.choice(pool)
    return _fill(template, attribute=attribute, value=value,
                 person=person, time_phrase=time_phrase,
                 old_value=old_value, referent=referent), is_implicit


def render_ack(rng: random.Random) -> str:
    return rng.choice(ASSISTANT_ACKS)


def render_question(rng: random.Random) -> str:
    return rng.choice(ASSISTANT_QUESTIONS)


def render_clarify(rng: random.Random) -> str:
    return rng.choice(ASSISTANT_CLARIFY)


def render_filler(rng: random.Random) -> str:
    """An inert user remark."""
    return rng.choice(FILLER_FORMS)


def surfaces_for(event_type: str) -> Tuple[Sequence[str], Sequence[str]]:
    """(explicit_forms, implicit_forms) for one event type."""
    return EXPLICIT.get(event_type, ()), IMPLICIT.get(event_type, ())
