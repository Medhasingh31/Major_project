"""Phase 3 skeleton refinement checks."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_extractor.geometry import (
    extract_geometry,
    prune_tiny_internal_branches,
)


def _parallel_centerlines() -> np.ndarray:
    skeleton = np.zeros((64, 64), dtype=np.uint8)
    skeleton[20, 8:57] = 1
    skeleton[24, 8:57] = 1
    skeleton[20:25, 32] = 1
    return skeleton


def test_weak_tiny_internal_connector_is_removed() -> None:
    skeleton = _parallel_centerlines()
    mask = np.zeros_like(skeleton)
    mask[20, 8:57] = 1
    mask[24, 8:57] = 1
    refined, removed = prune_tiny_internal_branches(skeleton, mask)
    assert removed > 0
    assert int(refined[22, 32]) == 0


def test_supported_short_internal_connector_is_preserved() -> None:
    skeleton = _parallel_centerlines()
    mask = cv2.dilate(
        skeleton,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    refined, removed = prune_tiny_internal_branches(skeleton, mask)
    assert removed == 0
    assert int(refined[22, 32]) == 1


def test_intersection_and_curved_road_remain_structured() -> None:
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[48, 15:82] = 1
    mask[20:49, 48] = 1
    curve = [(60, 20), (62, 24), (66, 28), (72, 31), (80, 34)]
    for y, x in curve:
        mask[y, x] = 1
    geometry = extract_geometry(np.zeros((*mask.shape, 3), dtype=np.uint8), mask)
    assert geometry.junction_count() >= 1
    assert len(geometry.endpoints) >= 3
    assert geometry.segment_count() >= 3


if __name__ == "__main__":
    test_weak_tiny_internal_connector_is_removed()
    test_supported_short_internal_connector_is_preserved()
    test_intersection_and_curved_road_remain_structured()
    print("PASS: Phase 3 Geometry skeleton checks")
