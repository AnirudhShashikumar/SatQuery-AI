"""Conservative local semantic interpretation of existing bi-temporal evidence.

This module does not inspect model internals, alter masks, or invoke inference.  It
turns already-published change geometry, deterministic differences, and optional
SVE scene-prior deltas into a structured intermediate representation and prose.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence

from ..models import BuiltUpChangeAssessment, ChangeAnalysisResponse, SemanticChangeSummary, SemanticTransition


SEMANTIC_INTERPRETER_VERSION = "bitemporal-semantic-1.0"


def _float_env(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        return min(maximum, max(minimum, float(os.getenv(name, str(default)))))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class SemanticInterpreterConfig:
    minimal_change_percent: float = 0.5
    localized_change_percent: float = 2.0
    noticeable_change_percent: float = 10.0
    widespread_change_percent: float = 30.0
    semantic_delta_threshold: float = 0.035
    strong_semantic_delta: float = 0.08
    disagreement_gap_percent: float = 10.0
    disagreement_ratio: float = 4.0


DEFAULT_CONFIG = SemanticInterpreterConfig(
    semantic_delta_threshold=_float_env("SATQUERY_SEMANTIC_DELTA_THRESHOLD", 0.035, 0.001, 0.5),
    strong_semantic_delta=_float_env("SATQUERY_SEMANTIC_STRONG_DELTA", 0.08, 0.001, 1.0),
    disagreement_gap_percent=_float_env("SATQUERY_CHANGE_DISAGREEMENT_GAP", 10.0, 0.0, 100.0),
    disagreement_ratio=_float_env("SATQUERY_CHANGE_DISAGREEMENT_RATIO", 4.0, 1.0, 100.0),
)


_FAMILIES: Dict[str, tuple[str, ...]] = {
    "built_up": ("urban or built-up area", "residential area", "industrial area"),
    "vegetation": ("vegetation", "forest", "grassland"),
    "water": ("inland water", "marine water", "wetland"),
    "infrastructure": ("road network", "airport", "harbor"),
}


def classify_change_intent(question: str) -> str:
    """Classify a bounded semantic intent without changing agent routing."""

    normalized = re.sub(r"[^a-z0-9]+", " ", (question or "").lower()).strip()
    if not normalized:
        return "unknown"
    if any(term in normalized for term in ("what caused", "why did", "who caused", "reason for")):
        return "unknown"
    if "largest" in normalized and any(term in normalized for term in ("where", "location", "occur")):
        return "largest_change"
    if any(term in normalized for term in ("how much", "percentage", "percent", "amount of change")):
        return "amount_of_change"
    if any(term in normalized for term in ("built up", "builtup", "urban", "building", "construction", "developed")):
        return "built_up_change"
    if "structural" in normalized:
        return "structural_change"
    if any(term in normalized for term in ("vegetation", "forest", "trees", "green cover", "grassland")):
        return "vegetation_change"
    if any(term in normalized for term in ("water", "river", "lake", "coast", "shoreline", "wetland")):
        return "water_change"
    if any(term in normalized for term in ("road", "infrastructure", "airport", "harbor")):
        return "infrastructure_change"
    if any(term in normalized for term in ("no change", "unchanged", "remain the same", "stayed the same", "mostly similar")):
        return "no_change_check"
    if any(term in normalized for term in ("where", "location", "which part", "which area")) and "change" in normalized:
        return "change_location"
    if any(term in normalized for term in ("compare", "difference between", "how different")):
        return "general_comparison"
    if any(term in normalized for term in ("what changed", "describe the change", "change between", "changes between")):
        return "change_summary"
    return "unknown"


def _requested_direction(question: str) -> Optional[str]:
    normalized = (question or "").lower()
    if any(term in normalized for term in ("increase", "increased", "expand", "expanded", "growth", "grew", "more")):
        return "increase"
    if any(term in normalized for term in ("decrease", "decreased", "shrink", "shrunk", "loss", "lost", "less", "reduced")):
        return "decrease"
    return None


def _finite_number(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _location(box: Any, width: int, height: int) -> Optional[str]:
    try:
        x = (float(box.left) + float(box.right)) / (2.0 * max(width, 1))
        y = (float(box.top) + float(box.bottom)) / (2.0 * max(height, 1))
    except (AttributeError, TypeError, ValueError):
        return None
    horizontal = "left" if x < 1 / 3 else "right" if x > 2 / 3 else "central"
    vertical = "upper" if y < 1 / 3 else "lower" if y > 2 / 3 else "central"
    if horizontal == vertical == "central":
        return "center"
    if horizontal == "central":
        return f"{vertical}-central"
    if vertical == "central":
        return f"central-{horizontal}"
    return f"{vertical}-{horizontal}"


def _magnitude(value: float, config: SemanticInterpreterConfig) -> tuple[str, str]:
    if value < config.minimal_change_percent:
        return "minimal", "very limited change"
    if value < config.localized_change_percent:
        return "localized", "localized change"
    if value < config.noticeable_change_percent:
        return "moderate", "noticeable but limited change"
    if value < config.widespread_change_percent:
        return "moderate", "substantial change"
    return "widespread", "widespread visual difference"


def _prior_changes(change: ChangeAnalysisResponse) -> list[dict[str, Any]]:
    sve = getattr(change, "sve_result", None)
    comparison = getattr(sve, "semantic_comparison", None) if sve and getattr(sve, "available", False) else None
    if comparison is None or getattr(comparison, "status", "") != "supporting_evidence":
        return []
    output = []
    for item in getattr(comparison, "prior_changes", []) or []:
        if not isinstance(item, dict) or not isinstance(item.get("label"), str):
            continue
        difference = _finite_number(item.get("difference"))
        if difference is None:
            continue
        output.append({"label": item["label"], "difference": difference})
    return output


def _transition_for_family(
    family: str,
    changes: Sequence[dict[str, Any]],
    config: SemanticInterpreterConfig,
) -> Optional[SemanticTransition]:
    labels = set(_FAMILIES[family])
    matching = [item for item in changes if item["label"] in labels and abs(item["difference"]) >= config.semantic_delta_threshold]
    if not matching:
        return None
    strongest = max(matching, key=lambda item: abs(item["difference"]))
    direction = "increase" if strongest["difference"] > 0 else "decrease"
    aligned = [item for item in matching if (item["difference"] > 0) == (strongest["difference"] > 0)]
    confidence = "strong" if len(aligned) >= 2 and abs(strongest["difference"]) >= config.strong_semantic_delta else "moderate" if len(aligned) >= 2 or abs(strongest["difference"]) >= config.strong_semantic_delta else "weak"
    transition_type = "infrastructure_change" if family == "infrastructure" else f"{family}_{direction}"
    evidence = [
        f"SVE relative scene evidence for {item['label']} moved {'up' if item['difference'] > 0 else 'down'} between observations."
        for item in aligned[:3]
    ]
    return SemanticTransition(type=transition_type, confidence=confidence, evidence=evidence)


def _claim_sentence(transition: SemanticTransition) -> str:
    prefix = {"strong": "indicates", "moderate": "appears to show", "weak": "may indicate"}[transition.confidence]
    sentences = {
        "built_up_increase": f"Scene-level evidence {prefix} stronger built-up characteristics in the later observation.",
        "built_up_decrease": f"Scene-level evidence {prefix} weaker built-up characteristics in the later observation.",
        "vegetation_increase": f"Scene-level evidence {prefix} more vegetation in the later observation.",
        "vegetation_decrease": f"Scene-level evidence {prefix} less vegetation in the later observation.",
        "water_increase": f"Scene-level evidence {prefix} greater water-related presence in the later observation.",
        "water_decrease": f"Scene-level evidence {prefix} reduced water-related presence in the later observation.",
        "infrastructure_change": f"Scene-level evidence {prefix} a change in infrastructure-related characteristics.",
    }
    return sentences.get(transition.type, "The semantic type of change remains uncertain.")


def _family_for_intent(intent: str) -> Optional[str]:
    return {
        "built_up_change": "built_up",
        "vegetation_change": "vegetation",
        "water_change": "water",
        "infrastructure_change": "infrastructure",
    }.get(intent)


def _stable_regions(changed_locations: Iterable[str], changed_percent: float) -> list[str]:
    locations = set(changed_locations)
    if changed_percent >= 30:
        return []
    candidates = ["central", "left", "right", "upper", "lower"]
    blocked = set()
    for location in locations:
        if "left" in location:
            blocked.add("left")
        if "right" in location:
            blocked.add("right")
        if "upper" in location:
            blocked.add("upper")
        if "lower" in location:
            blocked.add("lower")
        if "center" in location or "central" in location:
            blocked.add("central")
    return [item for item in candidates if item not in blocked][:2]


def interpret_change(
    question: str,
    change: ChangeAnalysisResponse,
    *,
    config: SemanticInterpreterConfig = DEFAULT_CONFIG,
    built_up_assessment: Optional[BuiltUpChangeAssessment] = None,
) -> SemanticChangeSummary:
    """Build structured facts first, then deterministic query-aware prose."""

    intent = classify_change_intent(question)
    statistics = getattr(change, "statistics", None)
    engine = getattr(change, "change_engine", None)
    learned = bool(engine and getattr(engine, "mode", "") in {"hybrid", "ttp"})
    fallback = bool(engine and getattr(engine, "fallback_used", False))
    caveats = ["The binary change detector does not identify the real-world cause of change."]
    supporting: list[str] = []
    if statistics is None:
        answer = "The image pair could not be compared reliably, so no change interpretation was generated."
        return SemanticChangeSummary(
            short_answer=answer,
            expanded_answer=answer,
            query_intent=intent,
            overall_change_level="minimal",
            evidence_strength="limited",
            caveats=caveats + ["Comparable-pixel change statistics were unavailable."],
        )

    changed_percent = _finite_number(getattr(statistics, "percentage_changed", None))
    changed_percent = min(100.0, max(0.0, changed_percent or 0.0))
    level, magnitude_phrase = _magnitude(changed_percent, config)
    regions = list(getattr(statistics, "regions", None) or [])
    regions.sort(key=lambda item: getattr(item, "area_pixels", 0), reverse=True)
    locations = [value for value in (_location(getattr(region, "bounding_box", None), statistics.analysis_width, statistics.analysis_height) for region in regions) if value]
    dominant_location = locations[0] if locations else None
    changed_regions = list(dict.fromkeys(locations[:4]))
    stable_regions = _stable_regions(changed_regions, changed_percent)
    where = f"the {dominant_location} part of the scene" if dominant_location and dominant_location != "center" else "the center of the scene" if dominant_location else None

    if changed_percent <= 0 or not regions:
        supporting.append("No connected changed region was present in the authoritative change statistics.")
    else:
        supporting.append(f"The primary change evidence identifies {magnitude_phrase}.")
        if where:
            supporting.append(f"The largest connected region is centered in {where}.")

    deterministic = getattr(change, "deterministic_statistics", None)
    deterministic_percent = _finite_number(getattr(deterministic, "percentage_changed", None)) if deterministic else None
    disagreement = bool(
        learned
        and deterministic_percent is not None
        and deterministic_percent - changed_percent >= config.disagreement_gap_percent
        and deterministic_percent >= max(changed_percent, 0.01) * config.disagreement_ratio
    )
    if disagreement:
        supporting.append("The deterministic analyzer reports much broader low-level visual difference than the learned mask.")

    changes = _prior_changes(change)
    transitions = [value for family in _FAMILIES if (value := _transition_for_family(family, changes, config))]
    # A semantic transition requires both spatial change evidence and compatible scene-level evidence.
    if not learned or changed_percent <= 0:
        transitions = []
    requested_family = _family_for_intent(intent)
    selected = next((item for item in transitions if item.type.startswith(requested_family or "__none__")), None)
    if selected is None and intent in {"change_summary", "general_comparison", "unknown"} and transitions:
        order = {"strong": 2, "moderate": 1, "weak": 0}
        selected = max(transitions, key=lambda item: order[item.confidence])
    if changes:
        caveats.append("SVE similarities are relative scene-level evidence, not calibrated probabilities or spatial segmentation.")
    else:
        caveats.append("No usable before/after scene-level semantic evidence was available, so no land-cover transition was assigned.")
    if fallback:
        caveats.append("The learned change specialist was unavailable; interpretation is limited to deterministic visual difference.")
    before_metadata = getattr(change, "before_metadata", None)
    if before_metadata is not None and not getattr(before_metadata, "is_georeferenced", False):
        caveats.append("Locations are coarse image directions, not geographic coordinates.")

    stable_summary = None
    if changed_percent < config.noticeable_change_percent:
        stable_summary = "Most of the scene remains comparatively stable outside the detected change areas."
        if stable_regions:
            stable_summary = f"The {', '.join(stable_regions)} parts show no comparable concentration of detected change."

    if changed_percent <= 0 or not regions:
        base = "The two observations are largely consistent, with no significant localized change detected."
    elif learned:
        base = f"The learned evidence identifies {magnitude_phrase}{f' concentrated in {where}' if where else ''}."
    else:
        base = f"The available visual-difference evidence identifies {magnitude_phrase}{f' concentrated in {where}' if where else ''}."

    semantic_sentence = _claim_sentence(selected) if selected else "The available evidence is not strong enough to determine a reliable land-cover transition."
    requested_direction = _requested_direction(question)
    if requested_family and selected:
        actual_direction = "increase" if selected.type.endswith("increase") else "decrease" if selected.type.endswith("decrease") else None
        if requested_direction and actual_direction and requested_direction != actual_direction:
            semantic_sentence = f"The available evidence does not support a {requested_direction}; it instead {('suggests' if selected.confidence != 'strong' else 'indicates')} a {actual_direction} in {requested_family.replace('_', '-')} characteristics."

    disagreement_sentence = "The images differ much more in overall appearance than in the learned structural mask; seasonal, illumination, sensor, or registration effects may contribute to the broader visual difference." if disagreement else None
    location_answer = f"The most significant detected change is in {where}." if where else "No dominant change location can be stated from the available geometry."

    if intent in {"largest_change", "change_location"}:
        short = location_answer
    elif intent == "amount_of_change":
        short = f"The primary evidence indicates {magnitude_phrase}, covering approximately {changed_percent:.1f}% of the analysis grid."
    elif intent == "no_change_check":
        short = "The observations are largely consistent, with no significant change detected." if changed_percent < config.minimal_change_percent else f"No. The evidence identifies {magnitude_phrase}{f' in {where}' if where else ''}."
    elif requested_family:
        short = semantic_sentence
        if where and selected:
            short = f"{short.rstrip('.')} The main detected change is concentrated in {where}."
    else:
        short = " ".join(item for item in (base, semantic_sentence if selected else None) if item)

    expanded_parts = [base]
    if selected:
        expanded_parts.append(semantic_sentence)
        supporting.extend(selected.evidence)
    elif intent not in {"amount_of_change", "largest_change", "change_location", "no_change_check"}:
        expanded_parts.append(semantic_sentence)
    if stable_summary:
        expanded_parts.append(stable_summary)
    if disagreement_sentence:
        expanded_parts.append(disagreement_sentence)
    expanded = " ".join(dict.fromkeys(expanded_parts))

    evidence_strength = "limited"
    if learned:
        evidence_strength = "moderate"
    if selected and selected.confidence == "strong":
        evidence_strength = "strong"
    likely_type = selected.type if selected else "unknown_change" if changed_percent > 0 else None

    if built_up_assessment is not None and intent in {"built_up_change", "structural_change"}:
        assessment = built_up_assessment
        relevant = [region for region in assessment.regions if region.semantic_support_level == "supported"] or assessment.regions
        region_names = list(dict.fromkeys(region.relative_location for region in relevant))
        where_text = " and ".join(region_names[:2]) if region_names else None
        strongest = assessment.regions[0] if assessment.regions else None
        if intent == "structural_change":
            short = (
                f"The strongest structural change is in the {strongest.relative_location} part of the scene."
                if strongest else "No meaningful structural change region can be localized from the available evidence."
            )
            likely_type = "structural_change" if strongest else None
        elif assessment.state == "INCREASE_SUPPORTED":
            short = f"Built-up expansion is suggested{f' in the {where_text} parts of the scene' if where_text else ''}."
            likely_type = "built_up_increase"
        elif assessment.state == "DECREASE_SUPPORTED":
            short = f"A decrease in built-up-like structural evidence is suggested{f' in the {where_text} parts of the scene' if where_text else ''}."
            likely_type = "built_up_decrease"
        elif assessment.state == "MIXED_CHANGE":
            short = f"Built-up-like evidence shows mixed directional change{f' across the {where_text} regions' if where_text else ''}."
            likely_type = "built_up_mixed_change"
        elif assessment.state == "NO_MEANINGFUL_EVIDENCE":
            short = "No meaningful evidence of built-up-region change was found."
            likely_type = None
        else:
            short = "Change is present, but the evidence is insufficient to establish whether built-up area increased or decreased."
            likely_type = "unknown_change"
        where_sentence = (
            f"The largest relevant changed cluster is in the {strongest.relative_location} part of the image; up to {len(assessment.regions)} meaningful region{'s were' if len(assessment.regions) != 1 else ' was'} compared."
            if strongest else "No meaningful connected region passed the region-size filter."
        )
        magnitude_sentence = f"The pattern is {assessment.magnitude}, based on changed-area share and region distribution."
        why_sentence = (
            "This direction combines ChangerEx geometry, independent deterministic overlap, and before/after edge-texture structural evidence. "
            "It is a heuristic interpretation consistent with built-up terrain, not a ground-truth land-cover measurement."
        )
        expanded = " ".join((short, where_sentence, magnitude_sentence, why_sentence))
        dominant_location = strongest.relative_location if strongest else dominant_location
        changed_regions = [region.relative_location for region in assessment.regions]
        evidence_strength = assessment.confidence
        supporting = list(dict.fromkeys(supporting + assessment.confidence_factors))
        caveats = list(dict.fromkeys(caveats + assessment.limitations))

    return SemanticChangeSummary(
        short_answer=short,
        expanded_answer=expanded,
        query_intent=intent,
        overall_change_level=level,
        dominant_location=dominant_location,
        likely_change_type=likely_type,
        evidence_strength=evidence_strength,
        stable_area_summary=stable_summary,
        changed_regions=changed_regions,
        stable_regions=stable_regions,
        likely_transitions=transitions,
        supporting_facts=list(dict.fromkeys(supporting)),
        caveats=list(dict.fromkeys(caveats)),
        visual_structural_disagreement=disagreement,
        built_up_assessment=built_up_assessment,
    )
