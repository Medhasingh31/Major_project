"""Phase 4 conservative gap-bridging checks."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_extractor.geometry import bridge_small_gaps


def _supported_aligned_gap() -> tuple[np.ndarray, np.ndarray]:
    skeleton = np.zeros((64, 64), dtype=np.uint8)
    skeleton[20, 8:30] = 1
    skeleton[20, 35:57] = 1
    mask = cv2.dilate(
        skeleton,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    return skeleton, mask


def test_aligned_projected_continuation_is_bridged() -> None:
    skeleton, mask = _supported_aligned_gap()
    repaired, bridges = bridge_small_gaps(skeleton, mask, max_gap=7)
    assert bridges == 1
    assert int(repaired[20, 32]) == 1


def test_parallel_separate_roads_are_not_joined() -> None:
    skeleton = np.zeros((64, 64), dtype=np.uint8)
    skeleton[20, 8:30] = 1
    skeleton[24, 35:57] = 1
    mask = cv2.dilate(
        skeleton,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    repaired, bridges = bridge_small_gaps(skeleton, mask, max_gap=7)
    assert bridges == 0
    assert np.array_equal(repaired, skeleton)


def test_candidate_crossing_unrelated_road_is_rejected() -> None:
    skeleton, mask = _supported_aligned_gap()
    # A separate road crosses the projected gap.  Its centerline is not an
    # endpoint of the candidate pair and must block the bridge.
    skeleton[10:31, 32] = 1
    mask = cv2.dilate(
        skeleton,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    before = int(np.sum(skeleton))
    repaired, bridges = bridge_small_gaps(skeleton, mask, max_gap=7)
    assert bridges == 0
    assert int(np.sum(repaired)) == before


def test_mask_unsupported_gap_is_rejected() -> None:
    skeleton, _ = _supported_aligned_gap()
    repaired, bridges = bridge_small_gaps(
        skeleton,
        np.zeros_like(skeleton),
        max_gap=7,
    )
    assert bridges == 0
    assert int(repaired[20, 32]) == 0


if __name__ == "__main__":
    test_aligned_projected_continuation_is_bridged()
    test_parallel_separate_roads_are_not_joined()
    test_candidate_crossing_unrelated_road_is_rejected()
    test_mask_unsupported_gap_is_rejected()
    print("PASS: Phase 4 Geometry gap-bridging checks")
