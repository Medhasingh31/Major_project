"""Phase 8 pre-export graph cleanup checks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_extractor.geometry import RoadGeometry, RoadSegment
from road_extractor.topology import build_topology


def _segment(segment_id: int, y: int, x0: int, x1: int) -> RoadSegment:
    return RoadSegment(
        segment_id=segment_id,
        pixel_path=[(y, x) for x in range(x0, x1 + 1)],
        length_pixels=float(x1 - x0),
        direction_deg=0.0,
        curvature=0.0,
        width_pixels=4.0,
    )


def test_invalid_isolated_nodes_are_removed_before_export() -> None:
    geometry = RoadGeometry(
        segments=[_segment(0, 20, 20, 21)],
        endpoints=[(20, 20), (20, 21)],
        clean_mask=np.zeros((64, 64), dtype=np.uint8),
    )
    topology = build_topology(geometry)
    assert topology.graph.number_of_nodes() == 0
    assert topology.graph.number_of_edges() == 0
    assert topology.metadata["graph_cleanup"]["isolated_nodes_removed"] == 2


def test_valid_disconnected_roads_are_preserved() -> None:
    geometry = RoadGeometry(
        segments=[
            _segment(0, 10, 5, 25),
            _segment(1, 50, 35, 55),
        ],
        endpoints=[(10, 5), (10, 25), (50, 35), (50, 55)],
        clean_mask=np.ones((64, 64), dtype=np.uint8),
    )
    topology = build_topology(geometry)
    assert topology.graph.number_of_nodes() == 4
    assert topology.graph.number_of_edges() == 2
    assert len(topology.components) == 2
    assert topology.metadata["graph_cleanup"]["real_components_preserved"] is True


if __name__ == "__main__":
    test_invalid_isolated_nodes_are_removed_before_export()
    test_valid_disconnected_roads_are_preserved()
    print("PASS: Phase 8 graph-cleanup checks")
