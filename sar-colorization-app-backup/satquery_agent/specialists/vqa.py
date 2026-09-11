"""Controlled answer templates grounded in deterministic GeoVision outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..entity_registry import get_remote_sensing_entity, resolve_remote_sensing_entity
from ..models import (
    ChangeAnalysisResponse,
    Confidence,
    ConfidenceLevel,
    ControlledVQAMethod,
    ControlledVQAResult,
    CrossModalResult,
    ImageMetadata,
    QuestionCategory,
    SingleImageEvidenceResult,
)
from .single_image_evidence import DEFAULT_THRESHOLDS, METHOD_ASSUMPTIONS, METHOD_LIMITATIONS


VQA_METHOD_NAME = "Controlled GeoVision evidence-grounded VQA"
VQA_METHOD_VERSION = "1.0"
UNSUPPORTED_CAUSAL = ("why", "cause", "caused", "flood", "construction", "deforestation", "crop type", "which crop", "exact location", "place name")


@dataclass(frozen=True)
class QuestionIntent:
    category: QuestionCategory
    target: Optional[str] = None
    secondary_target: Optional[str] = None
    comparison_relation: Optional[str] = None
    raw_target: Optional[str] = None
    raw_secondary_target: Optional[str] = None
    spatial_relation: Optional[str] = None


@dataclass
class ControlledAnswer:
    answer: Optional[str]
    details: ControlledVQAResult


def _normalize(question: str) -> str:
    return re.sub(r"[^a-z0-9%]+", " ", question.lower()).strip()


def _contains_any(normalized: str, terms: tuple[str, ...]) -> bool:
    words = set(normalized.split())
    return any((term in normalized) if " " in term else (term in words) for term in terms)


def resolve_controlled_concept(value: str) -> Optional[str]:
    """Resolve an RSVQA wording variant to a supported controlled concept."""
    entity = resolve_remote_sensing_entity(value)
    return entity.canonical_name if entity else None


def is_groundable_count_target(target: Optional[str]) -> bool:
    """Whether the current Grounding DINO adapter accepts this count concept."""
    entity = get_remote_sensing_entity(target)
    return bool(entity and entity.count_meaningful and entity.grounding_prompts)


def normalize_benchmark_answer(value: Any, answer_kind: Optional[str] = None) -> Optional[str]:
    """Return only an allowed RSVQA answer token, never a descriptive guess.

    ``answer_kind`` can constrain normalization to ``yes_no``,
    ``rural_urban``, or ``integer``.  Integer extraction deliberately accepts
    only a standalone number or an explicit controlled count phrase; it never
    mines incidental digits from arbitrary prose.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value) if value >= 0 and answer_kind in {None, "integer"} else None
    if not isinstance(value, str):
        return None
    normalized = _normalize(value)
    if not normalized:
        return None

    if answer_kind in {None, "yes_no"}:
        if normalized in {"yes", "y", "true", "affirmative", "present"}:
            return "yes"
        if normalized in {"no", "n", "false", "negative", "absent"}:
            return "no"
    if answer_kind in {None, "rural_urban"}:
        words = set(normalized.split())
        if "urban" in words and "rural" not in words:
            return "urban"
        if "rural" in words and "urban" not in words:
            return "rural"
    if answer_kind in {None, "integer"}:
        number = re.fullmatch(r"0|[1-9]\d*", normalized)
        if number:
            return number.group(0)
        controlled_count = re.fullmatch(
            r"(?:there (?:are|is) |number of (?:accepted )?(?:regions|detections) is )?(0|[1-9]\d*) (?:accepted )?(?:regions?|detections?)",
            normalized,
        )
        if controlled_count:
            return controlled_count.group(1)
    return None


def _is_rural_urban_question(normalized: str) -> bool:
    words = set(normalized.split())
    if not words.intersection({"rural", "urban"}):
        return False
    return (
        "rural or urban" in normalized
        or "urban or rural" in normalized
        or normalized.startswith(("is it ", "is this ", "is the scene ", "is the area ", "would you describe", "what type of area"))
    )


def _trim_count_target(value: str) -> str:
    value = re.sub(r"\b(?:are|is)\s+(?:there|visible|present)\b.*$", "", value).strip()
    return value.strip()


def _count_intent(normalized: str) -> Optional[QuestionIntent]:
    if any(term in normalized for term in (" less ", " fewer ", " more ", " greater ", " equal ", " same as ")):
        return None
    match = re.match(r"^(?:how many|what is (?:the )?(?:number|amount) of|(?:the )?number of)\s+(.+)$", normalized)
    if match is None:
        return None
    raw_target = _trim_count_target(match.group(1))
    if not raw_target:
        return None
    spatial_match = re.search(r"\b(next to|near|adjacent to|beside|around)\b", raw_target)
    return QuestionIntent(
        QuestionCategory.COUNT_VQA,
        target=resolve_controlled_concept(raw_target),
        raw_target=raw_target,
        spatial_relation=spatial_match.group(1) if spatial_match else None,
    )


def _comparison_intent(normalized: str) -> Optional[QuestionIntent]:
    relation: Optional[str] = None
    first: Optional[str] = None
    second: Optional[str] = None

    less_more = re.search(r"\b(less|fewer|more|greater)\s+(?:number of |amount of )?(.+?)\s+than\s+(.+)$", normalized)
    if less_more:
        relation = "less" if less_more.group(1) in {"less", "fewer"} else "more"
        first, second = less_more.group(2), less_more.group(3)
    else:
        subject_relation = re.search(
            r"\b(?:number|amount) of\s+(.+?)\s+(?:is )?(less|fewer|more|greater)\s+than\s+(?:(?:the )?(?:number|amount) of\s+)?(.+)$",
            normalized,
        )
        if subject_relation:
            relation = "less" if subject_relation.group(2) in {"less", "fewer"} else "more"
            first, second = subject_relation.group(1), subject_relation.group(3)
        else:
            equal = re.search(
                r"\b(?:number|amount) of\s+(.+?)\s+(?:is )?(?:equal to|the same as|same as)\s+(?:(?:the )?(?:number|amount) of\s+)?(.+)$",
                normalized,
            )
            if equal:
                relation, first, second = "equal", equal.group(1), equal.group(2)
    if relation is None or not first or not second:
        return None
    first = re.sub(r"^(?:are there|is there|is the|the)\s+", "", first).strip()
    second = re.sub(r"^(?:the)\s+", "", second).strip()
    return QuestionIntent(
        QuestionCategory.COMPARISON_VQA,
        target=resolve_controlled_concept(first),
        secondary_target=resolve_controlled_concept(second),
        comparison_relation=relation,
        raw_target=first,
        raw_secondary_target=second,
    )


def _presence_target(normalized: str) -> Optional[str]:
    """Extract the entity phrase from supported RSVQA presence constructions."""
    prefix = re.match(
        r"^(?:is there|are there)\s+(?:a|an|any)?\s*(.+)$|"
        r"^(?:does\s+(?:the|this)\s+(?:image|scene)\s+contain)\s+(?:a|an|any)?\s*(.+)$|"
        r"^(?:can you see)\s+(?:a|an|any)?\s*(.+)$",
        normalized,
    )
    if prefix:
        return next((group for group in prefix.groups() if group), None)
    suffix = re.match(
        r"^(?:is\s+(?:a|an|any)?|are\s+(?:a|an|any)?)\s*(.+?)\s+(?:present|visible)(?:\s+in\s+(?:the|this)\s+(?:image|scene))?$",
        normalized,
    )
    return suffix.group(1) if suffix else None


def _is_presence_question(normalized: str) -> bool:
    return _presence_target(normalized) is not None


def _method(source: str, assumptions: list[str], limitations: list[str]) -> ControlledVQAMethod:
    return ControlledVQAMethod(
        name=VQA_METHOD_NAME,
        version=VQA_METHOD_VERSION,
        method_type=source,
        uses_language_model=False,
        remote_sensing_adapted=False,
        assumptions=assumptions,
        limitations=limitations,
    )


class ControlledRemoteSensingVQA:
    def classify_question(self, question: str) -> QuestionIntent:
        normalized = _normalize(question)
        if not normalized or any(term in normalized for term in UNSUPPORTED_CAUSAL):
            return QuestionIntent(QuestionCategory.UNSUPPORTED)
        # RSVQA-LR routing precedes all descriptive/caption-like fallbacks.  Keep
        # the older exact templates below for backwards-compatible agent users.
        if _is_rural_urban_question(normalized) and normalized != "is this mainly urban or rural":
            return QuestionIntent(QuestionCategory.RURAL_URBAN_CLASSIFICATION, "scene")
        count = _count_intent(normalized)
        if count is not None:
            return count
        comparison = _comparison_intent(normalized)
        if comparison is not None:
            return comparison
        # Preserve the original public controlled-VQA templates.  New RSVQA
        # constructions are handled by the generic presence parser below.
        legacy_presence = {
            "is a water body visible": (QuestionCategory.PRESENCE_WATER, "water"),
            "can you see water": (QuestionCategory.PRESENCE_WATER, "water"),
            "are buildings present": (QuestionCategory.PRESENCE_BUILDINGS, "built_up"),
            "is vegetation visible": (QuestionCategory.PRESENCE_VEGETATION, "vegetation"),
            "does this scene contain agricultural land": (QuestionCategory.PRESENCE_AGRICULTURE, "agriculture"),
            "does this contain agricultural land": (QuestionCategory.PRESENCE_AGRICULTURE, "agriculture"),
        }.get(normalized)
        if legacy_presence is not None:
            return QuestionIntent(*legacy_presence)
        if _is_presence_question(normalized):
            raw_target = _presence_target(normalized) or ""
            return QuestionIntent(
                QuestionCategory.PRESENCE_VQA,
                target=resolve_controlled_concept(raw_target),
                raw_target=raw_target,
            )
        if any(term in normalized for term in ("resolution", "dimensions", "image size", "how many pixels")):
            return QuestionIntent(QuestionCategory.METADATA_QUESTION, "dimensions")
        if any(term in normalized for term in ("how many bands", "band count", "number of bands")):
            return QuestionIntent(QuestionCategory.METADATA_QUESTION, "band_count")
        if any(term in normalized for term in ("what crs", "coordinate reference", "georeferenced")):
            return QuestionIntent(QuestionCategory.METADATA_QUESTION, "georeferencing")
        if any(term in normalized for term in ("how much", "what percentage", "percent", "coverage")):
            for target, terms in {
                "vegetation": ("vegetation", "vegetated", "green"),
                "water": ("water", "water like"),
                "built_up": ("built up", "building", "structural", "urban"),
                "agriculture": ("agriculture", "agricultural", "field"),
                "barren": ("barren", "bare"),
            }.items():
                if any(term in normalized for term in terms):
                    return QuestionIntent(QuestionCategory.RELATIVE_COVERAGE, target)
            return QuestionIntent(QuestionCategory.UNSUPPORTED)
        if "dominant" in normalized and any(term in normalized for term in ("land cover", "scene", "type")):
            return QuestionIntent(QuestionCategory.DOMINANT_LAND_COVER, "scene")
        if any(term in normalized for term in ("mostly built up", "mainly built up", "predominantly built up")):
            return QuestionIntent(QuestionCategory.COMPOSITION_BUILT_UP, "built_up")
        if any(term in normalized for term in ("what kind of scene", "scene type", "mainly urban", "urban or rural", "urban agricultural", "forested water covered", "classify the scene")):
            return QuestionIntent(QuestionCategory.SCENE_TYPE, "scene")
        if "water" in normalized and _contains_any(normalized, ("is", "visible", "see", "present", "contain", "show")):
            return QuestionIntent(QuestionCategory.PRESENCE_WATER, "water")
        if any(term in normalized for term in ("building", "built up", "structurally complex")) and _contains_any(normalized, ("is", "are", "present", "visible", "contain", "show")):
            return QuestionIntent(QuestionCategory.PRESENCE_BUILDINGS, "built_up")
        if any(term in normalized for term in ("vegetation", "vegetated", "forest")) and _contains_any(normalized, ("is", "visible", "present", "contain", "show")):
            return QuestionIntent(QuestionCategory.PRESENCE_VEGETATION, "vegetation")
        if any(term in normalized for term in ("agriculture", "agricultural", "farmland", "field")) and _contains_any(normalized, ("is", "visible", "present", "contain", "show", "does")):
            return QuestionIntent(QuestionCategory.PRESENCE_AGRICULTURE, "agriculture")
        return QuestionIntent(QuestionCategory.UNSUPPORTED)

    def answer(
        self,
        question: str,
        evidence: SingleImageEvidenceResult,
        metadata: ImageMetadata,
        intent: Optional[QuestionIntent] = None,
    ) -> ControlledAnswer:
        intent = intent or self.classify_question(question)
        statistics = evidence.statistics
        category = intent.category
        target = intent.target
        used: Dict[str, Any] = {}
        references: list[str] = []
        supported = category != QuestionCategory.UNSUPPORTED

        if category == QuestionCategory.RURAL_URBAN_CLASSIFICATION:
            answer, used = self._rural_urban_answer(evidence)
            references = [reference for reference in (
                evidence.previews.built_up_support,
                evidence.previews.vegetation_support,
                evidence.previews.agriculture_support,
            ) if reference]
        elif category == QuestionCategory.PRESENCE_VQA:
            answer, used, references = self._presence_token_answer(evidence, target)
        elif category == QuestionCategory.COUNT_VQA:
            # Counting uses accepted Grounding DINO regions in the compatibility
            # endpoint.  The deterministic evidence maps must not be presented
            # as object counts.
            answer = None
            used = {
                "target": target,
                "grounding_required": True,
                "grounding_supported": is_groundable_count_target(target),
            }
        elif category == QuestionCategory.COMPARISON_VQA:
            answer = None
            used = {
                "first_target": target,
                "second_target": intent.secondary_target,
                "relation": intent.comparison_relation,
                "grounding_required": True,
                "first_grounding_supported": is_groundable_count_target(target),
                "second_grounding_supported": is_groundable_count_target(intent.secondary_target),
            }
        elif category == QuestionCategory.METADATA_QUESTION:
            if target == "dimensions":
                answer = f"The image dimensions are {metadata.width} × {metadata.height} pixels."
                used = {"width": metadata.width, "height": metadata.height}
            elif target == "band_count":
                answer = f"The uploaded image contains {metadata.band_count} raster band(s)."
                used = {"band_count": metadata.band_count}
            else:
                answer = f"The image is {'georeferenced' if metadata.is_georeferenced else 'not verifiably georeferenced'}; its CRS is {metadata.crs or 'unavailable'}."
                used = {"is_georeferenced": metadata.is_georeferenced, "crs": metadata.crs}
        elif category == QuestionCategory.DOMINANT_LAND_COVER:
            answer = self._scene_answer(evidence.dominant_scene, dominant=True)
            used = self._scene_statistics(evidence)
        elif category == QuestionCategory.SCENE_TYPE:
            answer = self._scene_answer(evidence.dominant_scene, dominant=False)
            used = self._scene_statistics(evidence)
        elif category == QuestionCategory.PRESENCE_WATER:
            value = statistics.water_support_percent
            answer = self._water_answer(value)
            used = {"water_support_percent": value}
            references = [evidence.previews.water_support] if evidence.previews.water_support else []
        elif category == QuestionCategory.PRESENCE_BUILDINGS:
            value = statistics.built_up_support_percent
            answer = self._built_answer(value, statistics.edge_density_percent)
            used = {"built_up_support_percent": value, "edge_density_percent": statistics.edge_density_percent}
            references = [evidence.previews.built_up_support] if evidence.previews.built_up_support else []
        elif category == QuestionCategory.PRESENCE_VEGETATION:
            value = statistics.vegetation_support_percent
            answer = self._vegetation_answer(value)
            used = {"vegetation_support_percent": value}
            references = [evidence.previews.vegetation_support] if evidence.previews.vegetation_support else []
        elif category == QuestionCategory.PRESENCE_AGRICULTURE:
            value = statistics.agriculture_support_percent
            answer = self._agriculture_answer(value, statistics.vegetation_support_percent)
            used = {"agriculture_support_percent": value, "vegetation_support_percent": statistics.vegetation_support_percent}
            references = [evidence.previews.agriculture_support] if evidence.previews.agriculture_support else []
        elif category == QuestionCategory.COMPOSITION_BUILT_UP:
            value = statistics.built_up_support_percent
            answer = (
                f"The scene has strong built-up or structural-complexity support over approximately {value:.1f}% of valid pixels, so it appears mostly built-up under this heuristic."
                if value >= 50.0
                else f"The scene is not mostly built-up under this heuristic; structural support covers approximately {value:.1f}% of valid pixels."
            )
            used = {"built_up_support_percent": value, "edge_density_percent": statistics.edge_density_percent}
            references = [evidence.previews.built_up_support] if evidence.previews.built_up_support else []
        elif category == QuestionCategory.RELATIVE_COVERAGE and target:
            key = {
                "vegetation": "vegetation_support_percent",
                "water": "water_support_percent",
                "built_up": "built_up_support_percent",
                "agriculture": "agriculture_support_percent",
                "barren": "barren_support_percent",
            }[target]
            value = float(getattr(statistics, key))
            answer = f"Approximately {value:.1f}% of valid pixels meet the current {target.replace('_', '-')} support rules. This is heuristic evidence, not a calibrated class probability."
            used = {key: value}
            preview = getattr(evidence.previews, key.replace("_percent", ""), None)
            references = [preview] if preview else []
        else:
            answer = "This question is outside the controlled VQA taxonomy. Ask about scene type, water, vegetation, structural complexity, possible agriculture, relative coverage, or image metadata."
            supported = False

        confidence = self._confidence(evidence, category, used)
        method = _method("deterministic evidence-grounded VQA", METHOD_ASSUMPTIONS, METHOD_LIMITATIONS)
        return ControlledAnswer(
            answer=answer,
            details=ControlledVQAResult(
                original_question=question,
                question_category=category,
                target_concept=target,
                answer_source="computed optical support maps and metadata",
                statistics_used=used,
                evidence_references=references,
                method=method,
                confidence=confidence,
                supported=supported,
                limitations=METHOD_LIMITATIONS,
                single_image_evidence=evidence,
            ),
        )

    @staticmethod
    def _strength(value: float) -> str:
        if value >= DEFAULT_THRESHOLDS.strong_evidence_percent:
            return "strong"
        if value >= DEFAULT_THRESHOLDS.moderate_evidence_percent:
            return "moderate"
        if value >= DEFAULT_THRESHOLDS.weak_evidence_percent:
            return "weak"
        return "negligible"

    def _water_answer(self, value: float) -> str:
        strength = self._strength(value)
        if strength == "strong":
            return f"There is strong visible-spectrum support for water-like regions, covering approximately {value:.1f}% of valid pixels."
        if strength in {"moderate", "weak"}:
            return f"Some water-like regions may be present, covering approximately {value:.1f}% of valid pixels."
        return "No substantial water-like region was detected by the current visible-spectrum baseline."

    def _built_answer(self, value: float, edge_density: float) -> str:
        if value >= DEFAULT_THRESHOLDS.moderate_evidence_percent and edge_density >= DEFAULT_THRESHOLDS.weak_evidence_percent:
            return f"Built-up or structurally complex regions are likely present, with heuristic support over approximately {value:.1f}% of valid pixels. Buildings are not confirmed."
        if value >= DEFAULT_THRESHOLDS.weak_evidence_percent:
            return f"Limited structural-complexity evidence is present over approximately {value:.1f}% of valid pixels; this is insufficient to confirm buildings."
        return "No substantial built-up or structural-complexity support was detected by the current baseline."

    def _vegetation_answer(self, value: float) -> str:
        if value >= DEFAULT_THRESHOLDS.weak_evidence_percent:
            return f"Visible-spectrum vegetation support is present over approximately {value:.1f}% of valid pixels. This is not an NDVI measurement."
        return "No substantial visible-spectrum vegetation support was detected by the current baseline."

    def _agriculture_answer(self, value: float, vegetation: float) -> str:
        if value >= DEFAULT_THRESHOLDS.moderate_evidence_percent and vegetation >= DEFAULT_THRESHOLDS.moderate_evidence_percent:
            return f"The image contains field-like and vegetation-supported patterns consistent with possible agricultural land over approximately {value:.1f}% of valid pixels. Crop type is not inferred."
        if value >= DEFAULT_THRESHOLDS.weak_evidence_percent:
            return f"Limited field-like support covers approximately {value:.1f}% of valid pixels, but agricultural land is uncertain."
        return "No substantial field-like, vegetation-supported agricultural pattern was detected by the current heuristic."

    @staticmethod
    def _rural_urban_answer(evidence: SingleImageEvidenceResult) -> tuple[Optional[str], Dict[str, Any]]:
        """Classify only when deterministic scene evidence is meaningfully separated."""
        statistics = evidence.statistics
        rural_support = max(
            statistics.vegetation_support_percent,
            statistics.agriculture_support_percent,
        )
        built = statistics.built_up_support_percent
        used = {
            "built_up_support_percent": built,
            "rural_support_percent": rural_support,
            "vegetation_support_percent": statistics.vegetation_support_percent,
            "agriculture_support_percent": statistics.agriculture_support_percent,
            "dominant_scene": evidence.dominant_scene,
        }
        if evidence.low_information:
            return None, used
        margin = DEFAULT_THRESHOLDS.dominant_class_margin_percent
        if built >= DEFAULT_THRESHOLDS.moderate_evidence_percent and built - rural_support >= margin:
            return "urban", used
        if rural_support >= DEFAULT_THRESHOLDS.moderate_evidence_percent and rural_support - built >= margin:
            return "rural", used
        return None, used

    @staticmethod
    def _presence_token_answer(
        evidence: SingleImageEvidenceResult,
        target: Optional[str],
    ) -> tuple[Optional[str], Dict[str, Any], list[str]]:
        if target is None or evidence.low_information:
            return None, {"target": target, "evidence_available": not evidence.low_information}, []
        statistics = evidence.statistics
        source = {
            "water": (statistics.water_support_percent, evidence.previews.water_support),
            "building": (statistics.built_up_support_percent, evidence.previews.built_up_support),
            "road": (statistics.built_up_support_percent, evidence.previews.built_up_support),
            "grassland": (statistics.vegetation_support_percent, evidence.previews.vegetation_support),
            "forest": (statistics.vegetation_support_percent, evidence.previews.vegetation_support),
            "vegetation": (statistics.vegetation_support_percent, evidence.previews.vegetation_support),
            "agriculture": (statistics.agriculture_support_percent, evidence.previews.agriculture_support),
            "heath": (statistics.vegetation_support_percent, evidence.previews.vegetation_support),
            "wetland": (max(statistics.water_support_percent, statistics.vegetation_support_percent), evidence.previews.water_support or evidence.previews.vegetation_support),
            "scrub": (statistics.vegetation_support_percent, evidence.previews.vegetation_support),
            "industrial_area": (statistics.built_up_support_percent, evidence.previews.built_up_support),
        }.get(target)
        if source is None:
            return None, {"target": target, "evidence_available": False}, []
        value, preview = source
        # These are deterministic support thresholds, not confirmation of the
        # semantic class named in the question.
        answer = "yes" if value >= DEFAULT_THRESHOLDS.weak_evidence_percent else "no"
        used = {"target": target, "support_percent": value, "threshold_percent": DEFAULT_THRESHOLDS.weak_evidence_percent}
        return answer, used, [preview] if preview else []

    @staticmethod
    def _scene_answer(scene: str, dominant: bool) -> str:
        if scene == "mixed":
            return "The scene appears mixed; no single land-cover category clearly dominates the measured support maps."
        if scene == "uncertain":
            return "The scene type is uncertain because no category has sufficiently strong, separated evidence."
        label = scene.replace("_", "-")
        prefix = "The strongest separated evidence supports" if dominant else "The controlled scene label is"
        return f"{prefix} a {label} scene. This label is heuristic rather than a calibrated classification."

    @staticmethod
    def _scene_statistics(evidence: SingleImageEvidenceResult) -> Dict[str, Any]:
        item = evidence.statistics
        return {
            "dominant_scene": evidence.dominant_scene,
            "water_support_percent": item.water_support_percent,
            "vegetation_support_percent": item.vegetation_support_percent,
            "built_up_support_percent": item.built_up_support_percent,
            "barren_support_percent": item.barren_support_percent,
            "agriculture_support_percent": item.agriculture_support_percent,
        }

    @staticmethod
    def _confidence(evidence: SingleImageEvidenceResult, category: QuestionCategory, used: Dict[str, Any]) -> Confidence:
        ambiguous = evidence.dominant_scene in {"mixed", "uncertain"} and category in {
            QuestionCategory.DOMINANT_LAND_COVER,
            QuestionCategory.SCENE_TYPE,
            QuestionCategory.RURAL_URBAN_CLASSIFICATION,
        }
        near_threshold = any(
            isinstance(value, (int, float)) and min(abs(float(value) - threshold) for threshold in (2.0, 8.0, 20.0)) < 1.0
            for value in used.values()
        )
        low = evidence.low_information or ambiguous or near_threshold or category in {
            QuestionCategory.UNSUPPORTED,
            QuestionCategory.COUNT_VQA,
            QuestionCategory.COMPARISON_VQA,
        }
        reason = (
            "The image is low-information or the measured evidence is ambiguous, and the method is not a calibrated semantic classifier."
            if low
            else "The answer is derived from consistent visible-spectrum evidence, but the method is heuristic and not a calibrated semantic classifier."
        )
        return Confidence(level=ConfidenceLevel.LOW if low else ConfidenceLevel.MODERATE, score=None, reason=reason)


CHANGE_LIMITATIONS = [
    "Answers describe normalized visual difference only and do not identify the cause or semantic type of change.",
    "Directions are coarse positions computed from connected-component centroids on the analysis grid.",
    "No registration, reprojection, or resampling is performed automatically.",
]


def classify_change_question(question: str) -> QuestionIntent:
    normalized = _normalize(question)
    if any(term in normalized for term in UNSUPPORTED_CAUSAL):
        return QuestionIntent(QuestionCategory.UNSUPPORTED)
    if "largest" in normalized and any(term in normalized for term in ("where", "location", "occur")):
        return QuestionIntent(QuestionCategory.LARGEST_CHANGE, "largest_region")
    if "how many" in normalized and any(term in normalized for term in ("change", "region")):
        return QuestionIntent(QuestionCategory.CHANGE_REGION_COUNT, "regions")
    if any(term in normalized for term in ("how much", "percentage", "percent")) and "change" in normalized:
        return QuestionIntent(QuestionCategory.CHANGE_PERCENTAGE, "changed_pixels")
    if any(term in normalized for term in ("mostly unchanged", "amount of change", "small moderate or large")):
        return QuestionIntent(QuestionCategory.CHANGE_MAGNITUDE, "magnitude")
    if any(term in normalized for term in ("what changed", "describe the change", "compare these dates")):
        return QuestionIntent(QuestionCategory.CHANGE_SUMMARY, "change")
    return QuestionIntent(QuestionCategory.UNSUPPORTED)


def _direction(box: Any, width: int, height: int) -> str:
    x = (box.left + box.right) / 2.0 / max(width, 1)
    y = (box.top + box.bottom) / 2.0 / max(height, 1)
    vertical = "north" if y < 1 / 3 else "south" if y > 2 / 3 else "central"
    horizontal = "west" if x < 1 / 3 else "east" if x > 2 / 3 else "central"
    if vertical == "central" and horizontal == "central":
        return "the center"
    if vertical == "central":
        return f"the {horizontal}"
    if horizontal == "central":
        return f"the {vertical}"
    return f"the {vertical}{horizontal}"


def answer_change_question(question: str, change: ChangeAnalysisResponse) -> ControlledAnswer:
    intent = classify_change_question(question)
    statistics = change.statistics
    hybrid = bool(change.change_engine and change.change_engine.mode in {"hybrid", "ttp"} and change.ttp_result)
    primary_label = "TTP learned mask" if hybrid else "deterministic change mask"
    confidence_reason = (
        f"{change.evidence_consistency.label}. This is not a calibrated probability."
        if hybrid and change.evidence_consistency else
        "The answer uses measured binary change evidence; it does not identify semantic cause or provide a calibrated probability."
    )
    supported = intent.category != QuestionCategory.UNSUPPORTED
    used: Dict[str, Any] = {}
    if statistics is None:
        answer = "Pixel-level change answering is unavailable because the pair requires alignment or is incompatible. No change statistic was fabricated."
        confidence = Confidence(level=ConfidenceLevel.LOW, score=None, reason="The protected change-analysis engine did not produce comparable-pixel statistics.")
    elif intent.category == QuestionCategory.CHANGE_PERCENTAGE:
        used = {"percentage_changed": statistics.percentage_changed}
        answer = f"The {primary_label} identified approximately {statistics.percentage_changed:.1f}% changed pixels on the analysis grid."
        confidence = Confidence(level=ConfidenceLevel.MODERATE, score=None, reason=confidence_reason)
    elif intent.category == QuestionCategory.CHANGE_REGION_COUNT:
        used = {"number_of_regions": statistics.number_of_regions}
        answer = f"The {primary_label} contains {statistics.number_of_regions} connected changed region(s)."
        confidence = Confidence(level=ConfidenceLevel.MODERATE, score=None, reason=confidence_reason)
    elif intent.category == QuestionCategory.LARGEST_CHANGE:
        if statistics.regions:
            region = statistics.regions[0]
            location = _direction(region.bounding_box, statistics.analysis_width, statistics.analysis_height)
            used = {"largest_connected_region": region.area_pixels, "largest_region_percent": region.percentage_of_image, "bbox_pixels": region.bounding_box.model_dump()}
            answer = f"The largest detected change occupies {region.percentage_of_image:.1f}% of the analysis grid and is located in {location}; its pixel bounding box is ({region.bounding_box.left}, {region.bounding_box.top}) to ({region.bounding_box.right}, {region.bounding_box.bottom})."
        else:
            answer = "No connected changed region was detected, so there is no largest change location."
            used = {"largest_connected_region": 0}
        confidence = Confidence(level=ConfidenceLevel.MODERATE, score=None, reason=confidence_reason + " Location is a coarse pixel-grid direction.")
    elif intent.category == QuestionCategory.CHANGE_MAGNITUDE:
        value = statistics.percentage_changed
        magnitude = "small" if value < 10.0 else "moderate" if value < 30.0 else "large"
        used = {"percentage_changed": value, "magnitude_rule": magnitude}
        mostly = "mostly unchanged" if value < 50.0 else "substantially changed"
        answer = f"The measured amount of change is {magnitude}: {value:.1f}% of the analysis grid changed, so the pair is {mostly}."
        confidence = Confidence(level=ConfidenceLevel.MODERATE, score=None, reason=confidence_reason + " Magnitude uses fixed descriptive ranges.")
    elif intent.category == QuestionCategory.CHANGE_SUMMARY:
        used = {"percentage_changed": statistics.percentage_changed, "number_of_regions": statistics.number_of_regions, "largest_connected_region": statistics.largest_connected_region}
        if hybrid and change.deterministic_statistics and change.mask_comparison:
            answer = (
                f"The TTP learned detector identified {statistics.percentage_changed:.1f}% changed pixels. "
                f"The deterministic evidence engine identified {change.deterministic_statistics.percentage_changed:.1f}%. "
                f"Their masks overlap with an IoU of {change.mask_comparison.iou:.2f}. "
                f"The primary mask contains {statistics.number_of_regions} connected region(s); the largest contains {statistics.largest_connected_region} pixels. "
                "The cause and semantic type of the change are not inferred."
            )
        else:
            answer = f"Approximately {statistics.percentage_changed:.1f}% of the analysis grid changed across {statistics.number_of_regions} connected region(s). The largest region contains {statistics.largest_connected_region} pixels. The cause and land-cover type of change are not inferred."
        confidence = Confidence(level=ConfidenceLevel.MODERATE, score=None, reason=confidence_reason)
    else:
        answer = "This change question is unsupported. Ask about changed percentage, region count, largest-region location, overall magnitude, or a non-causal change summary."
        supported = False
        confidence = Confidence(level=ConfidenceLevel.LOW, score=None, reason="The question requests information outside the controlled change taxonomy.")
    method_name = "controlled answers over TTP primary and deterministic supporting change statistics" if hybrid else "controlled answers over deterministic change statistics"
    assumptions = ["The TTP learned mask is primary and deterministic evidence is supporting; agreement is not ground-truth accuracy."] if hybrid else ["The protected deterministic change mask is the evidence source."]
    limitations = CHANGE_LIMITATIONS + (["TTP was trained on LEVIR-CD building-change imagery and may generalize weakly outside that domain.", "TTP output is not ground truth."] if hybrid else [])
    method = _method(method_name, assumptions, limitations)
    return ControlledAnswer(answer, ControlledVQAResult(
        original_question=question, question_category=intent.category, target_concept=intent.target,
        answer_source="hybrid TTP-primary change statistics" if hybrid else "deterministic bi-temporal change statistics", statistics_used=used,
        evidence_references=[path for path in (change.previews.difference, change.previews.ttp_mask, change.previews.deterministic_mask, change.previews.agreement, change.previews.disagreement, change.previews.mask, change.previews.overlay) if path],
        method=method, confidence=confidence, supported=supported, limitations=limitations,
    ))


CROSS_LIMITATIONS = [
    "Optical and SAR percentages are deterministic candidate evidence, not calibrated semantic probabilities.",
    "Water-like and structural-likelihood evidence can be confused by shadow, terrain, speckle, and strong scatterers.",
    "Full joint answering requires exact pixel and geospatial alignment.",
]


def classify_cross_modal_question(question: str) -> QuestionIntent:
    normalized = _normalize(question)
    if any(term in normalized for term in UNSUPPORTED_CAUSAL):
        return QuestionIntent(QuestionCategory.UNSUPPORTED)
    if "how many" in normalized and any(term in normalized for term in ("joint", "region")):
        return QuestionIntent(QuestionCategory.CROSS_MODAL_REGION_COUNT, "joint_regions")
    if "water" in normalized and any(term in normalized for term in ("percentage", "percent", "how much", "appears")):
        return QuestionIntent(QuestionCategory.CROSS_MODAL_WATER, "water")
    if any(term in normalized for term in ("structurally complex", "structural", "built up")):
        return QuestionIntent(QuestionCategory.CROSS_MODAL_STRUCTURE, "built_up")
    if "disagree" in normalized or "disagreement" in normalized:
        return QuestionIntent(QuestionCategory.CROSS_MODAL_DISAGREEMENT, "disagreement")
    if "agree" in normalized or "both modalities" in normalized or "supported by both" in normalized or "use both" in normalized or "combine the images" in normalized or "optical and sar together" in normalized or "analyse the optical and sar" in normalized or "analyze the optical and sar" in normalized:
        return QuestionIntent(QuestionCategory.CROSS_MODAL_AGREEMENT, "agreement")
    return QuestionIntent(QuestionCategory.UNSUPPORTED)


def answer_cross_modal_question(question: str, result: CrossModalResult) -> ControlledAnswer:
    intent = classify_cross_modal_question(question)
    statistics = result.statistics
    supported = intent.category != QuestionCategory.UNSUPPORTED
    used: Dict[str, Any] = {}
    references = [path for path in (result.previews.joint_evidence, result.previews.agreement, result.previews.disagreement, result.previews.joint_overlay) if path]
    if statistics is None:
        answer = "Pixel-level optical–SAR answering is unavailable because exact co-registration was not established. No joint percentage was fabricated."
        confidence = Confidence(level=ConfidenceLevel.LOW, score=None, reason="The protected cross-modal engine did not produce joint statistics.")
    elif intent.category == QuestionCategory.CROSS_MODAL_WATER:
        value = statistics.water_likelihood_percent or 0.0
        used = {"water_likelihood_percent": value}
        answer = f"Both modalities support water-like conditions over approximately {value:.1f}% of valid pixels. This is candidate evidence, not confirmed water segmentation."
        confidence = result.confidence
    elif intent.category == QuestionCategory.CROSS_MODAL_STRUCTURE:
        value = statistics.built_up_likelihood_percent or 0.0
        used = {"built_up_likelihood_percent": value}
        answer = f"Joint optical–SAR structural-likelihood support covers approximately {value:.1f}% of valid pixels. Structurally complex regions are {'present' if value >= 2.0 else 'not substantial'}, but buildings are not confirmed."
        confidence = result.confidence
    elif intent.category == QuestionCategory.CROSS_MODAL_DISAGREEMENT:
        value = statistics.disagreement_percent
        used = {"disagreement_percent": value}
        answer = "No candidate pixels were available for an agreement calculation." if value is None else f"Optical and SAR candidate evidence disagrees over approximately {value:.1f}% of evaluated candidate-evidence pixels."
        confidence = result.confidence
    elif intent.category == QuestionCategory.CROSS_MODAL_AGREEMENT:
        value = statistics.agreement_percent
        joint = [region for region in result.regions if region.type != "disagreement"]
        used = {"agreement_percent": value, "joint_region_count": len(joint)}
        valid = statistics.valid_pixel_percent or 0.0
        used["valid_pixel_percent"] = valid
        answer = "No candidate pixels were available for an agreement calculation." if value is None else f"The modalities positively agree over approximately {value:.1f}% of evaluated candidate-evidence pixels across {len(joint)} joint evidence region(s); {valid:.1f}% of source pixels were valid pixels."
        confidence = result.confidence
    elif intent.category == QuestionCategory.CROSS_MODAL_REGION_COUNT:
        count = sum(region.type != "disagreement" for region in result.regions)
        used = {"joint_region_count": count}
        answer = f"The deterministic fusion produced {count} joint water-like or structural-likelihood region(s)."
        confidence = result.confidence
    else:
        answer = "This optical–SAR question is unsupported. Ask about agreement, disagreement, water-like percentage, structural likelihood, or joint region count."
        supported = False
        confidence = Confidence(level=ConfidenceLevel.LOW, score=None, reason="The question requests information outside the controlled optical–SAR taxonomy.")
    method = _method("controlled answers over deterministic optical-SAR fusion", ["The protected fusion statistics are the sole answer source."], CROSS_LIMITATIONS)
    return ControlledAnswer(answer, ControlledVQAResult(
        original_question=question, question_category=intent.category, target_concept=intent.target,
        answer_source="deterministic optical-SAR evidence statistics", statistics_used=used,
        evidence_references=references, method=method, confidence=confidence, supported=supported,
        limitations=CROSS_LIMITATIONS,
    ))


_VQA = ControlledRemoteSensingVQA()


def get_vqa() -> ControlledRemoteSensingVQA:
    return _VQA
