"""Phase 5 meaningful topology-node checks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_extractor.geometry import extract_geometry
from road_extractor.topology import build_topology


def _geometry_with_t_junction_and_tiny_contact():
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[48, 15:82] = 1
    mask[20:49, 48] = 1
    # A short branch near the road should not become a second intersection.
    mask[45:49, 60] = 1
    return extract_geometry(np.zeros((*mask.shape, 3), dtype=np.uint8), mask)


def test_t_junction_is_meaningful_but_tiny_irregularity_is_not_intersection() -> None:
    geometry = _geometry_with_t_junction_and_tiny_contact()
    topology = build_topology(geometry)
    assert len(topology.intersections) == 1
    assert topology.intersections[0].kind == "T"


def test_only_explicit_junction_nodes_are_reported_as_intersections() -> None:
    geometry = _geometry_with_t_junction_and_tiny_contact()
    topology = build_topology(geometry)
    for intersection in topology.intersections:
        node = topology.graph.nodes[intersection.node_id]
        assert node["kind"] == "junction"
        assert intersection.degree >= 3
        assert len(intersection.segment_ids) >= 3


if __name__ == "__main__":
    test_t_junction_is_meaningful_but_tiny_irregularity_is_not_intersection()
    test_only_explicit_junction_nodes_are_reported_as_intersections()
    print("PASS: Phase 5 topology-node checks")
