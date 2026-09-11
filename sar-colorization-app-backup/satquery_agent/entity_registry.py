"""Controlled remote-sensing entities shared by routing and evidence adapters.

The registry describes supported language and evidence capabilities.  It does
not contain dataset labels, image identifiers, or benchmark answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RemoteSensingEntity:
    canonical_name: str
    aliases: tuple[str, ...]
    scene_prior_labels: tuple[str, ...]
    grounding_prompts: tuple[str, ...]
    count_meaningful: bool
    presence_uses_semantics: bool
    comparison_allowed: bool


REMOTE_SENSING_ENTITIES: tuple[RemoteSensingEntity, ...] = (
    RemoteSensingEntity(
        "water",
        (
            "marine water", "water bodies", "water body", "water areas", "water area",
            "rectangular water area", "medium water areas", "river", "rivers", "lake",
            "lakes", "pond", "ponds", "canal", "canals", "stream", "streams",
            "reservoir", "reservoirs", "water",
        ),
        ("inland water", "marine water", "wetland"),
        ("water body", "rectangular water area", "river", "lake", "pond", "canal", "stream"),
        True,
        True,
        True,
    ),
    RemoteSensingEntity(
        "road",
        (
            "rectangular roads", "rectangular road", "circular roads", "circular road",
            "medium roads", "medium road", "large roads", "large road", "road network",
            "streets", "street", "roads", "road",
        ),
        ("road network", "urban or built-up area", "residential area"),
        ("road", "rectangular road", "circular road", "medium road", "large road"),
        True,
        True,
        True,
    ),
    RemoteSensingEntity(
        "building",
        (
            "medium commercial buildings", "medium commercial building",
            "large commercial buildings", "large commercial building",
            "commercial buildings", "commercial building", "medium residential buildings",
            "medium residential building", "small residential buildings",
            "small residential building", "residential buildings", "residential building",
            "industrial buildings", "industrial building", "square buildings", "square building",
            "medium buildings", "medium building", "large buildings", "large building",
            "built structures", "built structure", "built-up area", "built up area", "built-up", "built up", "buildings", "building",
        ),
        ("urban or built-up area", "residential area", "industrial area"),
        (
            "building", "medium commercial building", "large commercial building", "commercial building",
            "medium residential building", "small residential building", "residential building",
            "industrial building", "square building", "medium building", "large building",
        ),
        True,
        True,
        True,
    ),
    RemoteSensingEntity(
        "grassland",
        ("small grass area", "medium grass area", "grass areas", "grass area", "grasslands", "grassland", "grass"),
        ("grassland", "vegetation", "pasture"),
        (),
        False,
        True,
        False,
    ),
    RemoteSensingEntity(
        "forest",
        ("woodlands", "woodland", "woods", "trees", "forests", "forest"),
        ("forest", "vegetation"),
        (),
        False,
        True,
        False,
    ),
    RemoteSensingEntity(
        "heath",
        ("heathland", "heath"),
        ("grassland", "vegetation", "barren land"),
        (),
        False,
        True,
        False,
    ),
    RemoteSensingEntity(
        "agriculture",
        ("agricultural land", "arable land", "farmlands", "farmland", "pasture", "fields", "field", "agriculture"),
        ("agricultural land", "arable land", "pasture"),
        (),
        False,
        True,
        False,
    ),
    RemoteSensingEntity(
        "wetland",
        ("wetlands", "wetland"),
        ("wetland", "inland water", "vegetation"),
        (),
        False,
        True,
        False,
    ),
    RemoteSensingEntity(
        "scrub",
        ("circular scrubs", "circular scrub", "scrublands", "scrubland", "scrubs", "scrub"),
        ("grassland", "vegetation", "barren land"),
        (),
        False,
        True,
        False,
    ),
    RemoteSensingEntity(
        "industrial_area",
        ("rectangular industrial area", "rectangular industrial", "industrial areas", "industrial area", "industrial"),
        ("industrial area", "urban or built-up area"),
        (),
        False,
        True,
        False,
    ),
    RemoteSensingEntity(
        "vegetation",
        ("vegetated areas", "vegetated area", "vegetation"),
        ("vegetation", "forest", "grassland"),
        ("vegetation",),
        False,
        True,
        False,
    ),
    # Existing non-RSVQA grounding concepts remain registered for backwards
    # compatibility with the public localization workflow.
    RemoteSensingEntity("aircraft", ("airplanes", "airplane", "planes", "plane", "aircraft"), ("airport",), ("aircraft",), True, False, False),
    RemoteSensingEntity("stadium", ("stadiums", "stadium"), ("urban or built-up area",), ("stadium",), True, False, False),
    RemoteSensingEntity("bridge", ("bridges", "bridge"), ("road network", "inland water"), ("bridge",), True, False, False),
)

_BY_CANONICAL = {entity.canonical_name: entity for entity in REMOTE_SENSING_ENTITIES}


def normalize_entity_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def resolve_remote_sensing_entity(value: str) -> Optional[RemoteSensingEntity]:
    """Resolve only explicit controlled aliases; unknown terms stay unknown."""
    normalized = normalize_entity_text(value)
    if not normalized:
        return None
    matches: list[tuple[int, int, RemoteSensingEntity]] = []
    for entity in REMOTE_SENSING_ENTITIES:
        for alias in entity.aliases:
            match = re.search(rf"\b{re.escape(normalize_entity_text(alias))}\b", normalized)
            if match:
                matches.append((match.start(), -len(alias), entity))
    return min(matches, key=lambda item: (item[0], item[1]))[2] if matches else None


def get_remote_sensing_entity(canonical_name: Optional[str]) -> Optional[RemoteSensingEntity]:
    return _BY_CANONICAL.get(canonical_name or "")


def grounding_prompt_for(value: str) -> Optional[str]:
    """Return the most-specific validated grounding prompt for an entity phrase."""
    entity = resolve_remote_sensing_entity(value)
    if entity is None or not entity.grounding_prompts:
        return None
    normalized = normalize_entity_text(value)
    def stems(phrase: str) -> set[str]:
        output = set()
        for token in normalize_entity_text(phrase).split():
            if token.endswith("ies") and len(token) > 3:
                token = token[:-3] + "y"
            elif token.endswith("s") and len(token) > 3:
                token = token[:-1]
            output.add(token)
        return output

    value_stems = stems(normalized)
    matching = [prompt for prompt in entity.grounding_prompts if stems(prompt).issubset(value_stems)]
    return max(matching, key=len) if matching else entity.grounding_prompts[0]


def canonical_grounding_prompt(value: str) -> Optional[str]:
    """Validate an already extracted target phrase and return its prompt token."""
    entity = resolve_remote_sensing_entity(value)
    if entity is None or not entity.grounding_prompts:
        return None
    normalized = normalize_entity_text(value)
    exact = [prompt for prompt in entity.grounding_prompts if normalized == normalize_entity_text(prompt)]
    return exact[0] if exact else entity.grounding_prompts[0]
