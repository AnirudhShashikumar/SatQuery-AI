"""Structured region evidence tests; prose is intentionally not snapshot-tested."""

from __future__ import annotations

import numpy as np

from satquery_agent.models import ChangeRegion, ChangeStatistics, PixelBoundingBox
from satquery_agent.specialists.builtup_change import assess_built_up_change


def _pattern(size: int = 100, boxes=((10, 10, 45, 45),)) -> tuple[np.ndarray, np.ndarray, np.ndarray, ChangeStatistics]:
    before = np.full((size, size, 3), 120, dtype=np.uint8)
    after = before.copy()
    mask = np.zeros((size, size), dtype=bool)
    regions = []
    for region_id, (left, top, right, bottom) in enumerate(boxes, 1):
        mask[top:bottom, left:right] = True
        for y in range(top, bottom, 4):
            after[y:y + 2, left:right] = 225
        for x in range(left, right, 4):
            after[top:bottom, x:x + 2] = 35
        area = (right - left) * (bottom - top)
        box = PixelBoundingBox(left=left, top=top, right=right, bottom=bottom, area_pixels=area)
        regions.append(ChangeRegion(region_id=region_id, area_pixels=area, percentage_of_image=area * 100 / (size * size), bounding_box=box))
    statistics = ChangeStatistics(
        analysis_width=size, analysis_height=size, source_width=size, source_height=size,
        total_pixels=size * size, changed_pixels=int(mask.sum()), percentage_changed=float(mask.mean() * 100),
        largest_connected_region=max(item.area_pixels for item in regions), number_of_regions=len(regions),
        bounding_boxes=[item.bounding_box for item in regions], regions=regions, normalized_threshold=.5,
    )
    return before, after, mask, statistics


def test_clear_structural_increase_is_region_specific_and_supported() -> None:
    before, after, mask, statistics = _pattern()
    result = assess_built_up_change(before, after, mask, statistics, mask.copy())
    assert result.state == "INCREASE_SUPPORTED"
    assert result.regions[0].relative_location == "northwest"
    assert result.regions[0].after_built_up_evidence > result.regions[0].before_built_up_evidence
    assert result.regions[0].deterministic_overlap_percent == 100
    assert "ground-truth" in " ".join(result.limitations).lower()


def test_clear_structural_decrease_uses_opposite_comparison() -> None:
    smooth, structured, mask, statistics = _pattern()
    result = assess_built_up_change(structured, smooth, mask, statistics, mask.copy())
    assert result.state == "DECREASE_SUPPORTED"
    assert result.regions[0].before_structural_evidence > result.regions[0].after_structural_evidence


def test_mixed_regions_and_top_three_ranking() -> None:
    before, after, mask, statistics = _pattern(boxes=((5, 5, 45, 45), (60, 60, 90, 90), (60, 5, 82, 27), (5, 65, 15, 75)))
    # Reverse the second component so one real region supports decrease.
    before[60:90, 60:90] = after[60:90, 60:90]
    after[60:90, 60:90] = 120
    result = assess_built_up_change(before, after, mask, statistics, mask.copy())
    assert result.state == "MIXED_CHANGE"
    assert len(result.regions) == 3
    assert [item.pixel_area for item in result.regions] == sorted((item.pixel_area for item in result.regions), reverse=True)


def test_change_without_structural_direction_remains_insufficient() -> None:
    before, _, mask, statistics = _pattern()
    result = assess_built_up_change(before, before.copy(), mask, statistics, np.zeros_like(mask))
    assert result.state == "INSUFFICIENT_EVIDENCE"
    assert result.confidence == "low"
    assert all(item.semantic_support_level != "supported" for item in result.regions)


def test_tiny_noisy_components_are_filtered() -> None:
    before, after, mask, statistics = _pattern(boxes=((1, 1, 2, 2),))
    result = assess_built_up_change(before, after, mask, statistics, mask)
    assert result.state == "NO_MEANINGFUL_EVIDENCE"
    assert result.regions == []
