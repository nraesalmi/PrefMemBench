"""CLI: build histories, checkpoints, and matched variants.

Everything here is driven by one seeded RNG, so the same --seed gives
byte-identical output. Gold state is never written by hand -- it is always
recomputed through semantics.gold_profile / gold_action.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timedelta

from memproj import config, families, recipes, templates
from memproj.families import GenerationError
from memproj.schema import Checkpoint, Trajectory, Turn
from memproj.semantics import gold_action, gold_profile, relevance

HERE = os.path.dirname(os.path.abspath(__file__))
# The CLI defaults below read config so that a catalogue or generation-knob
# change lands in one place. As literals they could silently pin an old
# catalogue and make a swap inert.
DEFAULT_CATALOGUE = str(config.CATALOGUE_PATH)

# Every history counts from this instant. Fixed, not "now": the generated
# corpus must be reproducible across runs months apart.
BASE_DATE = datetime(
    config.BASE_YEAR, config.BASE_MONTH, config.BASE_DAY, config.BASE_HOUR, 0
)

# Values the generator is allowed to draw from. Spice levels come straight
# from the catalogue's vocabulary rather than being invented here.
SUBJECTS_FOR_OTHER = ("brother", "partner", "friend")
MIN_RECIPES_PER_CHECKPOINT = 3


# --- value pools drawn from the catalogue -----------------------------------

def spice_vocabulary(catalogue) -> list:
    """Spice levels that actually exist in the catalogue."""
    return sorted({r["spice_level"] for r in catalogue})


def excludable_ingredients(catalogue, min_recipes: int = 3) -> list:
    """Ingredients that appear in some but not all recipes.

    An ingredient in every recipe cannot be excluded without emptying the
    catalogue, and one in no recipe is useless as an exclusion.
    """
    counts = {}
    for r in catalogue:
        for i in r["ingredients"]:
            counts[i] = counts.get(i, 0) + 1
    n = len(catalogue)
    return sorted(i for i, c in counts.items() if min_recipes <= c < n)


# --- rendering a history ----------------------------------------------------

def render_history(events, sessions, rng, slots, turns_per_session,
                   other_person) -> list:
    """Turn events into dialogue, padding each session with filler.

    One event becomes a user turn plus an assistant acknowledgement. If the
    event needs clarification the assistant asks instead.

    Each session's target is jittered around turns_per_session so histories
    vary in length rather than all landing on the same count.
    """
    by_session = {}
    for e in events:
        by_session.setdefault(e.session, []).append(e)
    by_id = {e.event_id: e for e in events}

    turns = []
    counter = [0]

    def next_id():
        counter[0] += 1
        return f"t{counter[0]:04d}"

    for s_index, start in enumerate(sessions):
        session_events = by_session.get(s_index, [])
        cursor = start + timedelta(minutes=5)
        # Vary length: jitter this session's target by +/- one exchange.
        target = max(len(session_events) * 2,
                     turns_per_session + rng.choice((-2, 0, 2)))
        ack_pool = _shuffled(templates.ASSISTANT_ACKS, rng)

        for e in sorted(session_events, key=lambda x: (x.ts, x.event_id)):
            person = other_person if e.subject == other_person else None
            # A correction has to name the value it replaces, and a resolved
            # ambiguity the restriction it drops. Both live on other events,
            # so the wording is looked up here rather than invented by a
            # template: the sentence names a real record or it says "that".
            corrected = (by_id.get(e.target_event_id)
                         if e.target_event_id else None)
            referent = (by_id.get(e.referent_event_id)
                        if e.referent_event_id else None)
            text, implicit = templates.render(
                e.type, rng,
                attribute=e.attribute,
                value=e.value,
                person=person,
                time_phrase=slots.get("time_phrase"),
                old_value=corrected.value if corrected else None,
                referent=referent.value if referent else None,
            )
            if not text:
                continue
            # A statement is spoken when the user says it -- at its event's
            # ts -- so a checkpoint whose ask lands later in the evening
            # still sits after the turn that carried the statement.
            spoken_at = max(cursor, e.ts)
            if spoken_at >= start + timedelta(hours=8):
                spoken_at = cursor  # an event set after the session; keep order
            turns.append(Turn(next_id(), s_index, spoken_at, "user", text,
                              e.event_id, is_implicit=implicit))
            cursor = spoken_at + timedelta(minutes=1)

            if e.type == "ambiguous_update" and e.referent_event_id is None:
                ack = templates.render_clarify(rng)
            else:
                ack = _draw(ack_pool, templates.ASSISTANT_ACKS, rng)
            turns.append(Turn(next_id(), s_index, cursor, "assistant", ack))
            cursor += timedelta(minutes=1)

        # Pad to this session's target with inert filler. Both loops draw
        # from shuffled pools so a session never repeats itself.
        filler_pool = _shuffled(templates.FILLER_FORMS, rng)
        while len([t for t in turns if t.session == s_index]) < target:
            text = _draw(filler_pool, templates.FILLER_FORMS, rng)
            turns.append(Turn(next_id(), s_index, cursor, "user", text))
            cursor += timedelta(minutes=1)
            turns.append(Turn(next_id(), s_index, cursor, "assistant",
                              _draw(ack_pool, templates.ASSISTANT_ACKS, rng)))
            cursor += timedelta(minutes=1)

    return turns


def _shuffled(pool, rng) -> list:
    out = list(pool)
    rng.shuffle(out)
    return out


def _draw(pool: list, refill, rng) -> str:
    """Pop from a shuffled pool, reshuffling when exhausted (never empty)."""
    if not pool:
        pool.extend(_shuffled(refill, rng))
    return pool.pop()


# --- building one pair ------------------------------------------------------

def build_pair(pair_id, family, index, rng, n_sessions, turns_per_session,
               catalogue):
    sessions = families.session_times(BASE_DATE, n_sessions)
    other_person = SUBJECTS_FOR_OTHER[index % len(SUBJECTS_FOR_OTHER)]

    spice_levels = spice_vocabulary(catalogue)
    spicy_value = "spicy" if "spicy" in spice_levels else spice_levels[-1]

    base_events, matched_events, slots, meta, _ = families.build_pair(
        family, sessions, other_person, spicy_value
    )

    trajectories, checkpoints = [], []
    for variant, events in (("base", base_events), ("matched", matched_events)):
        tid = f"{pair_id}_{variant}"

        # Recipe availability is checked per checkpoint; if any checkpoint
        # cannot be served, the caller regenerates this pair.
        for spec in families.checkpoint_specs(family, events, sessions):
            profile = gold_profile(events, spec["person"], spec["now"])
            action = gold_action(events, spec["person"], spec["now"])
            available = recipes.valid_recipes(catalogue, profile)
            if len(available) < MIN_RECIPES_PER_CHECKPOINT:
                raise GenerationError(
                    f"{tid}/{spec['kind']}: only {len(available)} recipes "
                    f"satisfy {profile}"
                )

        turns = render_history(events, sessions, rng, slots,
                               turns_per_session, other_person)
        trajectory = Trajectory(
            trajectory_id=tid, pair_id=pair_id, variant=variant,
            family=family, flipped_field=meta["flipped_field"],
            events=tuple(events), turns=tuple(turns),
        )
        trajectories.append(trajectory)

        for n, spec in enumerate(families.checkpoint_specs(
                family, events, sessions)):
            profile = gold_profile(events, spec["person"], spec["now"])
            action = gold_action(events, spec["person"], spec["now"])
            relevant, distractors = relevance(events, spec["person"], spec["now"])
            checkpoints.append(Checkpoint(
                checkpoint_id=f"{tid}_c{n}",
                trajectory_id=tid,
                after_event_id=spec["after"],
                now=spec["now"],
                person=spec["person"],
                request_text=_request_for(spec["person"], spec["kind"], rng),
                gold_profile={k: sorted(v) for k, v in sorted(profile.items())},
                gold_action=action,
                relevant_event_ids=tuple(relevant),
                distractor_event_ids=tuple(distractors),
                kind=spec["kind"],
            ))
    return trajectories, checkpoints


def _request_for(person, kind, rng) -> str:
    who = "me" if person == "user" else f"my {person}"
    # Each opener already reads grammatically with "for {who}" appended.
    opener = rng.choice([
        "What should I make for", "Any ideas for", "What would you suggest for",
        "Got anything for", "What's good for",
    ])
    if kind == "temp_inside":
        return f"{opener} {who} tonight?"
    if kind == "temp_expired":
        return f"{opener} {who} tonight?"
    return f"{opener} {who}?"


# --- driving the whole run --------------------------------------------------

def generate(n_base, seed, n_sessions, turns_per_session, catalogue_path):
    catalogue = recipes.load_derived(catalogue_path)
    family_names = list(families.FAMILIES)

    trajectories, checkpoints = [], []
    for i in range(n_base):
        family = family_names[i % len(family_names)]
        pair_id = f"p{i:03d}"
        # Each pair gets its own RNG stream so adding pairs cannot shift
        # earlier ones -- the output stays stable as --n-base grows.
        rng = random.Random(f"{seed}:{pair_id}")
        for attempt in range(20):
            try:
                t, c = build_pair(pair_id, family, i, rng, n_sessions,
                                  turns_per_session, catalogue)
                trajectories.extend(t)
                checkpoints.extend(c)
                break
            except GenerationError:
                if attempt == 19:
                    raise
                rng = random.Random(f"{seed}:{pair_id}:retry{attempt}")
    return trajectories, checkpoints


# --- output -----------------------------------------------------------------

def _event_json(e):
    return {
        "event_id": e.event_id, "session": e.session, "ts": e.ts.isoformat(),
        "type": e.type, "subject": e.subject, "attribute": e.attribute,
        "value": e.value, "scope": e.scope,
        "valid_from": e.valid_from.isoformat(),
        "valid_until": e.valid_until.isoformat() if e.valid_until else None,
        "target_event_id": e.target_event_id,
        "needs_clarification": e.needs_clarification,
        "referent_event_id": e.referent_event_id,
    }


def _turn_json(t):
    return {"turn_id": t.turn_id, "session": t.session,
            "ts": t.ts.isoformat(), "speaker": t.speaker, "text": t.text,
            "event_id": t.event_id, "is_implicit": t.is_implicit}


def _trajectory_json(tr):
    return {
        "trajectory_id": tr.trajectory_id, "pair_id": tr.pair_id,
        "variant": tr.variant, "family": tr.family,
        "flipped_field": tr.flipped_field,
        "events": [_event_json(e) for e in tr.events],
        "turns": [_turn_json(t) for t in tr.turns],
    }


def _checkpoint_json(c):
    return {
        "checkpoint_id": c.checkpoint_id, "trajectory_id": c.trajectory_id,
        "kind": c.kind,
        "after_event_id": c.after_event_id, "now": c.now.isoformat(),
        "person": c.person, "request_text": c.request_text,
        "gold_profile": c.gold_profile, "gold_action": c.gold_action,
        "relevant_event_ids": list(c.relevant_event_ids),
        "distractor_event_ids": list(c.distractor_event_ids),
    }


def write_output(out_dir, trajectories, checkpoints):
    os.makedirs(out_dir, exist_ok=True)
    h_path = os.path.join(out_dir, "histories.jsonl")
    c_path = os.path.join(out_dir, "checkpoints.jsonl")
    with open(h_path, "w", encoding="utf-8", newline="\n") as fh:
        for tr in trajectories:
            fh.write(json.dumps(_trajectory_json(tr), sort_keys=True,
                                ensure_ascii=False) + "\n")
    with open(c_path, "w", encoding="utf-8", newline="\n") as fh:
        for c in checkpoints:
            fh.write(json.dumps(_checkpoint_json(c), sort_keys=True,
                                ensure_ascii=False) + "\n")
    return h_path, c_path


def summarise(trajectories, checkpoints) -> str:
    lines = []
    lines.append(f"trajectories: {len(trajectories)}  "
                 f"checkpoints: {len(checkpoints)}")

    per_family = {}
    for tr in trajectories:
        per_family.setdefault(tr.family, 0)
        per_family[tr.family] += 1
    lines.append("")
    lines.append("by family            trajectories")
    for fam in sorted(per_family):
        lines.append(f"  {fam:22s} {per_family[fam]:>4d}")

    per_type = {}
    for tr in trajectories:
        for e in tr.events:
            per_type[e.type] = per_type.get(e.type, 0) + 1
    lines.append("")
    lines.append("by event type        occurrences")
    for t in sorted(per_type):
        lines.append(f"  {t:22s} {per_type[t]:>4d}")

    turn_counts = [len(tr.turns) for tr in trajectories]
    if turn_counts:
        lines.append("")
        lines.append(f"turns per history: min {min(turn_counts)} "
                     f"max {max(turn_counts)} "
                     f"mean {sum(turn_counts)/len(turn_counts):.1f}")

    per_action = {}
    for c in checkpoints:
        per_action[c.gold_action] = per_action.get(c.gold_action, 0) + 1
    lines.append("")
    lines.append("by gold_action       checkpoints")
    for a in sorted(per_action):
        lines.append(f"  {a:22s} {per_action[a]:>4d}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-base", type=int, default=config.N_BASE)
    p.add_argument("--seed", type=int, default=config.GEN_SEED)
    p.add_argument("--sessions", type=int, default=config.SESSIONS)
    p.add_argument("--turns-per-session", type=int,
                   default=config.TURNS_PER_SESSION)
    p.add_argument("--catalogue", default=DEFAULT_CATALOGUE)
    p.add_argument("--out", default=str(config.OUT_DIR))
    args = p.parse_args(argv)

    trajectories, checkpoints = generate(
        args.n_base, args.seed, args.sessions, args.turns_per_session,
        args.catalogue,
    )
    h_path, c_path = write_output(args.out, trajectories, checkpoints)
    print(summarise(trajectories, checkpoints))
    print()
    print(f"wrote {h_path}")
    print(f"wrote {c_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
