"""Phase 10 quantitative comparison checks."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_extractor.quantitative import METRIC_NAMES, compare_metrics


def test_all_required_metrics_and_neutral_interpretation_are_reported() -> None:
    before = {
        "road_pixels": 100,
        "skeleton_pixels": 40,
        "connected_components": 3,
        "endpoints": 8,
        "intersections": 2,
        "edges": 7,
        "suspicious_disconnected_segments": [1, 2],
        "centerline_length_pixels": 90.5,
    }
    after = {
        "road_pixels": 98,
        "skeleton_pixels": 36,
        "connected_components": 2,
        "endpoints": 6,
        "intersections": 2,
        "edges": 6,
        "suspicious_disconnected_segments": [2],
        "centerline_length_pixels": 91.0,
    }
    comparison = compare_metrics(before, after, before_label="Phase 9", after_label="Phase 10")
    assert set(comparison["before"]) == set(METRIC_NAMES)
    assert set(comparison["after"]) == set(METRIC_NAMES)
    assert comparison["after"]["suspicious_disconnected_segments"] == 1
    assert comparison["delta_after_minus_before"]["total_centerline_length"] == 0.5
    assert comparison["interpretation"]["fewer_objects_not_assumed_better"] is True


if __name__ == "__main__":
    test_all_required_metrics_and_neutral_interpretation_are_reported()
    print("PASS: Phase 10 quantitative-comparison checks")
