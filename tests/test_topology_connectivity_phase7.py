"""Phase 7 non-forcing topology connectivity validation checks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_extractor.geometry import RoadGeometry, RoadSegment
from road_extractor.topology import build_topology


def _segment(segment_id: int, y: int, x0: int, x1: int) -> RoadSegment:
    path = [(y, x) for x in range(x0, x1 + 1)]
    return RoadSegment(
        segment_id=segment_id,
        pixel_path=path,
        length_pixels=float(x1 - x0),
        direction_deg=0.0,
        curvature=0.0,
        width_pixels=4.0,
    )


def test_separate_road_components_are_reported_not_merged() -> None:
    geometry = RoadGeometry(
        segments=[
            _segment(0, 10, 5, 25),
            _segment(1, 50, 35, 55),
        ],
        endpoints=[(10, 5), (10, 25), (50, 35), (50, 55)],
        clean_mask=np.ones((64, 64), dtype=np.uint8),
    )
    topology = build_topology(geometry)
    validation = topology.metadata["connectivity_validation"]
    assert validation["total_nodes"] == 4
    assert validation["total_edges"] == 2
    assert validation["connected_components"] == 2
    assert validation["separate_components_preserved"] is True


def test_isolated_nodes_and_short_edges_are_diagnosed() -> None:
    geometry = RoadGeometry(
        segments=[_segment(0, 20, 20, 21)],
        endpoints=[(20, 20), (20, 21)],
        clean_mask=np.zeros((64, 64), dtype=np.uint8),
    )
    topology = build_topology(geometry)
    validation = topology.metadata["connectivity_validation"]
    assert validation["isolated_nodes"] == 0
    assert topology.metadata["graph_cleanup"]["isolated_nodes_removed"] == 2
    assert validation["connected_components"] == 0
    assert topology.metadata["graph_cleanup"]["tiny_edges_removed"] == 1
    assert validation["total_edges"] == 0


if __name__ == "__main__":
    test_separate_road_components_are_reported_not_merged()
    test_isolated_nodes_and_short_edges_are_diagnosed()
    print("PASS: Phase 7 topology-connectivity checks")
