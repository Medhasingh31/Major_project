"""Phase 9 visual diagnostic artifact checks."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_extractor.geometry import RoadGeometry, RoadSegment
from road_extractor.pipeline import _prepare_graph_exports
from road_extractor.topology import build_topology
from road_extractor.visualize import (
    save_final_graph_overlay,
    save_mask,
    save_rgb_image,
    save_topology_overlay,
)


def test_required_visual_diagnostics_are_readable() -> None:
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[24, 8:56] = 1
    skeleton = mask.copy()
    segment = RoadSegment(
        segment_id=0,
        pixel_path=[(24, x) for x in range(8, 56)],
        length_pixels=47.0,
        direction_deg=0.0,
        curvature=0.0,
        width_pixels=4.0,
    )
    geometry = RoadGeometry(
        segments=[segment],
        endpoints=[(24, 8), (24, 55)],
        skeleton=skeleton,
        clean_mask=mask,
    )
    topology = build_topology(geometry)
    _prepare_graph_exports(topology, geometry)

    writable_test_root = Path(
        r"C:\Users\MEDHA SINGH\.codex\visualizations\2026\08\20\01a01f10-da08-7622-ba65-cc75ef7bed36"
    )
    writable_test_root.mkdir(parents=True, exist_ok=True)
    output = writable_test_root / "phase9_diagnostic_outputs"
    output.mkdir(parents=True, exist_ok=True)
    paths = {
            "original_rgb": output / "original_rgb.png",
            "raw_mask": output / "raw_mask.png",
            "cleaned_mask": output / "cleaned_mask.png",
            "raw_skeleton": output / "raw_skeleton.png",
            "cleaned_skeleton": output / "cleaned_skeleton.png",
            "topology": output / "topology_overlay.png",
            "graph": output / "road_graph.png",
    }
    save_rgb_image(rgb, paths["original_rgb"])
    save_mask(mask, paths["raw_mask"])
    save_mask(mask, paths["cleaned_mask"])
    save_mask(mask, paths["raw_skeleton"])
    save_mask(skeleton, paths["cleaned_skeleton"])
    save_topology_overlay(rgb, geometry, topology, paths["topology"])
    save_final_graph_overlay(rgb, topology.graph, paths["graph"])

    for path in paths.values():
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert path.exists() and path.stat().st_size > 0
        assert image is not None and image.size > 0


if __name__ == "__main__":
    test_required_visual_diagnostics_are_readable()
    print("PASS: Phase 9 visual-diagnostic checks")
