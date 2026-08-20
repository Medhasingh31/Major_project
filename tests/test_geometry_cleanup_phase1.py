"""Phase 1 Geometry cleanup checks; synthetic masks are test fixtures only."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_extractor.geometry import (
    _remove_small_skeleton_components,
    clean_binary_mask,
    extract_geometry,
    prune_skeleton_spurs,
)


def _rgb(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros((*shape, 3), dtype=np.uint8)


def test_compact_blob_removed_thin_road_preserved() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:15, 10:15] = 1  # compact false-positive blob: area 25
    mask[30, 8:28] = 1      # narrow isolated road-like component
    cleaned = clean_binary_mask(
        mask,
        min_area=50,
        close_radius=0,
        min_small_component_span=12,
        min_small_component_aspect=2.5,
        context_radius=4,
    )
    assert int(cleaned[12, 12]) == 0
    assert int(cleaned[30, 18]) == 1


def test_closing_does_not_join_nearby_roads() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[18, 8:56] = 1
    mask[22, 8:56] = 1
    cleaned = clean_binary_mask(
        mask,
        min_area=20,
        close_radius=2,
        min_small_component_span=12,
        min_small_component_aspect=2.5,
    )
    components, _ = cv2.connectedComponents(cleaned, connectivity=8)
    assert components - 1 == 2
    assert int(cleaned[20, 32]) == 0


def test_line_like_skeleton_preserved_compact_fragment_removed() -> None:
    skeleton = np.zeros((48, 48), dtype=np.uint8)
    skeleton[10, 5:20] = 1
    skeleton[30:33, 30:33] = 1
    cleaned, removed = _remove_small_skeleton_components(
        skeleton,
        min_size=22,
        min_road_like_pixels=10,
        min_road_like_span=10,
    )
    assert int(cleaned[10, 12]) == 1
    assert int(cleaned[31, 31]) == 0
    assert removed > 0


def test_extremely_short_spur_removed_long_side_road_preserved() -> None:
    short = np.zeros((64, 64), dtype=np.uint8)
    short[32, 8:52] = 1
    short[29:33, 30] = 1
    pruned_short, removed_short = prune_skeleton_spurs(
        short,
        min_branch_length=14,
        iterations=4,
        clean_mask=short,
    )
    assert removed_short > 0
    assert int(pruned_short[29, 30]) == 0

    legitimate = np.zeros((64, 64), dtype=np.uint8)
    legitimate[32, 8:52] = 1
    legitimate[20:33, 30] = 1
    # The centerline is thin, but the supporting road mask has width.  This
    # exercises the independent mask-evidence guard used for short side roads.
    legitimate_mask = cv2.dilate(
        legitimate,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    pruned_legitimate, removed_legitimate = prune_skeleton_spurs(
        legitimate,
        min_branch_length=14,
        iterations=4,
        clean_mask=legitimate_mask,
        legitimate_branch_angle=80.0,
        legitimate_branch_mask_support=0.70,
        legitimate_branch_min_fraction=0.75,
    )
    assert int(pruned_legitimate[20, 30]) == 1
    assert removed_legitimate == 0


def test_intersection_dead_end_and_curve_survive_geometry() -> None:
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[48, 15:82] = 1
    mask[20:49, 48] = 1
    curve = [(60, 20), (62, 24), (66, 28), (72, 31), (80, 34)]
    for y, x in curve:
        mask[y, x] = 1
    geometry = extract_geometry(
        _rgb(mask.shape),
        mask,
    )
    assert geometry.junction_count() >= 1
    assert len(geometry.endpoints) >= 3
    assert geometry.segment_count() >= 3


if __name__ == "__main__":
    test_compact_blob_removed_thin_road_preserved()
    test_line_like_skeleton_preserved_compact_fragment_removed()
    test_extremely_short_spur_removed_long_side_road_preserved()
    test_intersection_dead_end_and_curve_survive_geometry()
    print("PASS: Phase 1 Geometry cleanup checks")
