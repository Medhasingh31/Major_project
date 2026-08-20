"""Phase 2 Geometry connection-safety checks."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_extractor.geometry import bridge_small_gaps, clean_binary_mask


def _two_segments(y0: int, y1: int) -> np.ndarray:
    skeleton = np.zeros((64, 64), dtype=np.uint8)
    skeleton[y0, 8:30] = 1
    skeleton[y1, 35:57] = 1
    return skeleton


def test_aligned_gap_with_mask_support_is_bridged() -> None:
    skeleton = _two_segments(20, 20)
    mask = cv2.dilate(
        skeleton,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    repaired, bridges = bridge_small_gaps(skeleton, mask)
    assert bridges == 1
    assert int(repaired[20, 32]) == 1


def test_unsupported_gap_is_not_bridged() -> None:
    skeleton = _two_segments(20, 20)
    repaired, bridges = bridge_small_gaps(
        skeleton,
        np.zeros_like(skeleton),
    )
    assert bridges == 0
    assert int(repaired[20, 32]) == 0


def test_incompatible_approach_is_not_bridged() -> None:
    skeleton = _two_segments(20, 24)
    mask = cv2.dilate(
        skeleton,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    repaired, bridges = bridge_small_gaps(skeleton, mask)
    assert bridges == 0
    assert int(repaired[22, 32]) == 0


def test_component_local_closing_does_not_create_contact() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[18, 8:56] = 1
    mask[22, 8:56] = 1
    cleaned = clean_binary_mask(mask, min_area=20, close_radius=2)
    components, _ = cv2.connectedComponents(cleaned, connectivity=8)
    assert components - 1 == 2


if __name__ == "__main__":
    test_aligned_gap_with_mask_support_is_bridged()
    test_unsupported_gap_is_not_bridged()
    test_incompatible_approach_is_not_bridged()
    test_component_local_closing_does_not_create_contact()
    print("PASS: Phase 2 Geometry connection checks")
