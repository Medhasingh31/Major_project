"""Phase 6 topology edge-construction checks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_extractor.geometry import RoadGeometry, RoadSegment
from road_extractor.topology import TopologyConfig, build_topology


def _segment(segment_id: int, path: list[tuple[int, int]], length: float) -> RoadSegment:
    return RoadSegment(
        segment_id=segment_id,
        pixel_path=path,
        length_pixels=length,
        direction_deg=0.0,
        curvature=0.0,
        width_pixels=4.0,
    )


def test_duplicate_node_pair_edges_are_deduplicated() -> None:
    endpoints = [(20, 8), (20, 52)]
    geometry = RoadGeometry(
        segments=[
            _segment(0, [(20, x) for x in range(8, 53)], 44.0),
            _segment(1, [(20, x) for x in range(8, 53)], 44.0),
        ],
        endpoints=endpoints,
        clean_mask=np.ones((64, 64), dtype=np.uint8),
    )
    topology = build_topology(geometry)
    assert topology.graph.number_of_edges() == 1
    assert topology.metadata["duplicate_edges_removed"] == 1


def test_unsupported_tiny_edge_is_not_exported() -> None:
    geometry = RoadGeometry(
        segments=[_segment(0, [(20, 20), (20, 21)], 1.0)],
        endpoints=[(20, 20), (20, 21)],
        clean_mask=np.zeros((64, 64), dtype=np.uint8),
    )
    topology = build_topology(geometry)
    assert topology.graph.number_of_edges() == 0
    assert topology.metadata["tiny_edges_removed"] == 1


def test_supported_short_edge_is_retained() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20, 20:22] = 1
    geometry = RoadGeometry(
        segments=[_segment(0, [(20, 20), (20, 21)], 1.0)],
        endpoints=[(20, 20), (20, 21)],
        clean_mask=mask,
    )
    topology = build_topology(
        geometry,
        config=TopologyConfig(tiny_edge_mask_support=0.80),
    )
    assert topology.graph.number_of_edges() == 1


if __name__ == "__main__":
    test_duplicate_node_pair_edges_are_deduplicated()
    test_unsupported_tiny_edge_is_not_exported()
    test_supported_short_edge_is_retained()
    print("PASS: Phase 6 topology-edge checks")
