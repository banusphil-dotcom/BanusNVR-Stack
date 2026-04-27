"""BanusNas — Narrative generator for detection events.

Generates concise descriptions: a factual template fallback (who/where/when),
plus optional LLM-based description offloaded to the external ML GPU server.
Descriptions are action-focused: what the person is DOING, not their appearance.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from core.config import settings

logger = logging.getLogger("banusnas.narrative_generator")

# ── Action-focused prompt for vision model ─────────────────────────
#
# Both prompts are written so the vision model produces a concrete,
# household-sized description of what is happening — naming objects the
# subject is interacting with and the apparent purpose of the activity.
#
# Forbidden phrasings (these were producing useless, abstract narratives
# like "The person begins standing and facing right, then transitions to a
# seated position…"):
#   - direction words referring to the camera frame (left, right, toward
#     the camera)
#   - posture transitions described in isolation (standing, sitting,
#     bending) without saying WHY
#   - meta phrases like "across the frames", "transitions to", "appears to"

_ACTION_PROMPT = (
    "You are describing what {subject} is doing right now in a home CCTV "
    "still from {where}. Write ONE natural sentence (under 20 words) that "
    "names the activity and any objects {pronoun} is handling — for example: "
    "\"hanging a shirt on the line to dry\", \"carrying shopping into the "
    "kitchen\", \"watering the plants\", \"playing fetch with the dog\". "
    "Start with {subject}'s name. "
    "Do NOT describe clothing, appearance, mood, or camera framing. "
    "Do NOT use words like 'facing', 'left', 'right', 'transitions', "
    "'appears to', 'seems to', or 'across the frames'. "
    "If you genuinely cannot tell what {pronoun} is doing, reply with the "
    "single word: unclear."
)

_ACTIVITY_PROMPT = (
    "You are describing a short sequence of CCTV frames showing {subject} "
    "at {where}. Write ONE natural sentence (under 25 words) summarising "
    "the activity {pronoun} performed — name objects {pronoun} interacted "
    "with and the purpose of the action, e.g. \"hung two shirts on the "
    "washing line\", \"carried a parcel to the front door and went back "
    "inside\", \"played fetch with the dog\". "
    "Start with {subject}'s name. "
    "Do NOT describe posture changes in isolation, clothing, mood, or "
    "camera direction. "
    "Do NOT use 'facing', 'left', 'right', 'transitions to', 'begins', "
    "'across the frames', or 'appears to'. "
    "If the activity is genuinely unclear, reply with the single word: "
    "unclear."
)


# ── Forbidden-phrase guard ────────────────────────────────────────
#
# These substrings indicate the remote vision server fell back to its
# generic posture/direction prompt and ignored our `instructions` field.
# When detected, callers drop the result and use the factual template.

_FORBIDDEN_NARRATIVE_PHRASES: tuple[str, ...] = (
    "facing right",
    "facing left",
    "facing the camera",
    "facing away",
    "transitions to",
    "transitions back",
    "across the frames",
    "appears to be",
    "seems to be",
    "begins standing",
    "begins sitting",
    "moving towards the right",
    "moving towards the left",
    "to the right side of the frame",
    "to the left side of the frame",
)


def _is_forbidden_narrative(text: str) -> bool:
    """Return True if the narrative contains any forbidden phrasing."""
    if not text:
        return False
    lower = text.lower()
    return any(phrase in lower for phrase in _FORBIDDEN_NARRATIVE_PHRASES)


def _format_describe_prompt(
    template: str,
    *,
    named_object_name: Optional[str],
    object_type: str,
    camera_name: str,
) -> str:
    """Substitute subject/pronoun/where placeholders into a description prompt."""
    if named_object_name:
        subject = named_object_name
        pronoun = "they"
    elif object_type.lower() in ("cat", "dog"):
        subject = f"the {object_type.lower()}"
        pronoun = "it"
    elif object_type.lower() == "person":
        subject = "the person"
        pronoun = "they"
    else:
        subject = f"the {object_type.lower()}"
        pronoun = "it"

    where = _match_location(camera_name) or camera_name or "the camera location"
    return template.format(subject=subject, pronoun=pronoun, where=where)


# ── Camera name → friendly location ───────────────────────────────────

_LOCATION_MAP: dict[str, str] = {
    "living room": "the living room",
    "kitchen": "the kitchen",
    "driveway": "the driveway",
    "front": "the front door",
    "studio": "the studio",
    "flood": "the garden",
    "test": "the test camera",
}


def _match_location(camera_name: str) -> Optional[str]:
    """Match a camera name to a friendly location phrase."""
    cam_lower = camera_name.lower()
    for key, phrase in _LOCATION_MAP.items():
        if key in cam_lower:
            return phrase
    return None


def _get_time_label(hour: int) -> str:
    """Return a short time-of-day label."""
    if hour < 6:
        return "early morning"
    if hour < 12:
        return "morning"
    if hour < 14:
        return "midday"
    if hour < 17:
        return "afternoon"
    if hour < 21:
        return "evening"
    return "night"


def _friendly_type(object_type: str) -> str:
    """Return a user-friendly label for the object type."""
    return {
        "person": "Person",
        "cat": "Cat",
        "dog": "Dog",
        "car": "Car",
        "truck": "Truck",
        "motorcycle": "Motorcycle",
        "bicycle": "Bicycle",
        "bird": "Bird",
    }.get(object_type.lower(), object_type.capitalize())


def generate_narrative(
    *,
    named_object_name: Optional[str],
    object_type: str,
    camera_name: str,
    timestamp: Optional[datetime] = None,
    posture: Optional[str] = None,
    zones: Optional[list[str]] = None,
    seed: Optional[str] = None,
) -> str:
    """Generate a brief, natural narrative for a detection event.

    Uses verified facts with varied phrasing for natural reading.
    Example: "Alison was spotted in the kitchen this afternoon"
    """
    import hashlib

    location = _match_location(camera_name)
    ts = timestamp or datetime.now(timezone.utc)
    time_label = _get_time_label(ts.hour)

    # Build subject: use recognised name, else "Someone"/"A cat"/etc.
    if named_object_name:
        subject = named_object_name
    else:
        subject = {
            "person": "Someone",
            "cat": "A cat",
            "dog": "A dog",
            "car": "A car",
            "truck": "A truck",
            "motorcycle": "A motorcycle",
            "bicycle": "A bicycle",
            "bird": "A bird",
        }.get(object_type.lower(), f"A {object_type}")

    # Build location phrase
    where = f"in {location}" if location else f"on {camera_name}"

    # Deterministic variation using a hash of event details
    hash_input = f"{named_object_name or ''}{object_type}{camera_name}{ts.isoformat()}"
    variant = int(hashlib.md5(hash_input.encode()).hexdigest(), 16) % 5

    templates = [
        f"{subject} was spotted {where} this {time_label}.",
        f"{subject} appeared {where} this {time_label}.",
        f"{subject} was seen {where} this {time_label}.",
        f"{subject} was noticed {where} this {time_label}.",
        f"{subject} was detected {where} this {time_label}.",
    ]

    return templates[variant]


def generate_group_narrative(
    *,
    names: list[str],
    object_types: list[str],
    camera_name: str,
    camera_names: Optional[list[str]] = None,
    started_at: Optional[datetime] = None,
    ended_at: Optional[datetime] = None,
) -> str:
    """Generate a narrative for a group of events, potentially across cameras.

    Examples:
      "George, Alison and Xuefei were seen in the kitchen this morning for 10 minutes"
      "Philip moved from the kitchen to the driveway over 15 minutes"
      "Someone and a cat were spotted in the living room this evening"
    """
    # Build multi-camera location string
    if camera_names and len(camera_names) > 1:
        # Multi-room: "kitchen → living room → driveway"
        locs = []
        for cn in camera_names:
            loc = _match_location(cn)
            locs.append(loc if loc else cn)
        # Deduplicate while preserving order
        seen = set()
        unique_locs = []
        for loc in locs:
            if loc not in seen:
                seen.add(loc)
                unique_locs.append(loc)
        where = " → ".join(unique_locs)
    else:
        location = _match_location(camera_name)
        where = f"in {location}" if location else f"on {camera_name}"

    ts = started_at or datetime.now(timezone.utc)
    time_label = _get_time_label(ts.hour)

    # Build duration string
    duration_str = ""
    if started_at and ended_at:
        secs = (ended_at - started_at).total_seconds()
        if secs >= 3600:
            hrs = int(secs // 3600)
            duration_str = f" for about {hrs} hour{'s' if hrs > 1 else ''}"
        elif secs >= 60:
            mins = int(secs // 60)
            duration_str = f" for {mins} minute{'s' if mins > 1 else ''}"

    # Deduplicate names; unknowns are empty strings
    known_names = []
    unknown_counts: dict[str, int] = {}
    for name, otype in zip(names, object_types):
        if name:
            if name not in known_names:
                known_names.append(name)
        else:
            friendly = _friendly_type(otype).lower()
            unknown_counts[friendly] = unknown_counts.get(friendly, 0) + 1

    # Build subject parts
    parts: list[str] = list(known_names)
    for friendly, count in unknown_counts.items():
        if count == 1:
            article = "an" if friendly[0] in "aeiou" else "a"
            parts.append(f"{article} {friendly}")
        else:
            parts.append(f"{count} {friendly}s" if not friendly.endswith("s") else f"{count} {friendly}")

    if not parts:
        parts = ["Activity"]

    # Join names naturally: "A, B and C"
    if len(parts) == 1:
        subject = parts[0]
    elif len(parts) == 2:
        subject = f"{parts[0]} and {parts[1]}"
    else:
        subject = ", ".join(parts[:-1]) + f" and {parts[-1]}"

    # Pick verb — singular for solo subjects, plural for multiple
    import hashlib
    hash_input = f"{subject}{camera_name}{ts.isoformat()}"
    variant = int(hashlib.md5(hash_input.encode()).hexdigest(), 16) % 3
    is_singular = len(parts) == 1 and not parts[0][0].isdigit()
    is_multi_room = camera_names and len(camera_names) > 1
    if is_multi_room:
        # Multi-room: use movement verbs
        verb = "moved through" if is_singular else "were seen in"
    elif is_singular:
        verbs = ["was seen", "was spotted", "appeared"]
        verb = verbs[variant]
    else:
        verbs = ["were seen", "were spotted", "appeared"]
        verb = verbs[variant]

    # Capitalize first letter of subject
    subject = subject[0].upper() + subject[1:] if subject else subject

    return f"{subject} {verb} {where} this {time_label}{duration_str}."


# ── LLM-based description via external ML server ──────────────────


async def describe_snapshot_with_vision(
    frame: np.ndarray,
    *,
    camera_name: str,
    object_type: str,
    named_object_name: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> Optional[str]:
    """Send a snapshot to the external ML GPU server for LLM description.

    Returns a short action-focused description string, or None if ML offload is
    unavailable / disabled / fails.
    """
    from services.ml_client import remote_describe

    instructions = _format_describe_prompt(
        _ACTION_PROMPT,
        named_object_name=named_object_name,
        object_type=object_type,
        camera_name=camera_name,
    )

    desc = await remote_describe(
        frame,
        camera_name=camera_name,
        object_type=object_type,
        named_object_name=named_object_name,
        timestamp=timestamp,
        instructions=instructions,
    )

    # Treat the explicit "unclear" sentinel as no result so the template
    # narrative remains in place rather than displaying a useless phrase.
    if desc and desc.strip().lower().rstrip(".") == "unclear":
        return None

    # ── Reject narratives that violate our forbidden-phrase rules ──
    # The remote vision server sometimes ignores the `instructions` field and
    # falls back to a generic "describes posture and direction" prompt that
    # produces useless output like "The person is standing and facing right".
    # If that happens, drop the result so the factual template fallback is
    # used instead of a misleading narrative.
    if desc and _is_forbidden_narrative(desc):
        logger.info(
            "Rejecting vision narrative (forbidden phrasing): %s",
            desc[:120],
        )
        return None

    return desc


async def describe_with_text_llm(
    *,
    camera_name: str,
    object_type: str,
    named_object_name: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> Optional[str]:
    """Text-only LLM narrative — disabled.

    All LLM work is now offloaded to the external ML server via
    describe_snapshot_with_vision(). This stub remains for API compat.
    """
    return None
