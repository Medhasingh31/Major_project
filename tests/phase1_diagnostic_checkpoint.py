"""Read-only checkpoint instrumentation for existing diagnostic artifacts.

This script does not import or execute Geometry, Topology, Graph, or model
code. It reads the existing Phase 14 masks, summaries, and GraphML files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import networkx as nx


CASES = (
    "straight_roads",
    "curved_roads",
    "intersections",
    "t_junctions",
    "crossroads",
    "disconnected_occluded",
    "dense_urban",
)

STAGES = (
    "original_mask",
    "repaired_mask",
    "skeleton",
    "connected_components",
    "endpoint_detection",
    "junction_detection",
    "graph_construction",
    "edge_length_calculation",
)


def _mask_stats(path: Path) -> tuple[int, int]:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    pixels = int((mask > 0).sum())
    components = max(int(cv2.connectedComponents((mask > 0).astype("uint8"))[0]) - 1, 0)
    return pixels, components


def _row(case: str, stage: str, source: list[str], **values: object) -> dict:
    row = {
        "case": case,
        "stage": stage,
        "source": source,
        "road_pixel_count": None,
        "skeleton_pixel_count": None,
        "connected_component_count": None,
        "endpoint_count": None,
        "junction_count": None,
        "node_count": None,
        "edge_count": None,
        "total_graph_length": None,
    }
    row.update(values)
    return row


def collect_case(case_dir: Path, case: str) -> list[dict]:
    geometry_path = case_dir / "geometry_summary.json"
    topology_path = case_dir / "topology_summary.json"
    graphml_path = case_dir / "road_network.graphml"
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    topology = json.loads(topology_path.read_text(encoding="utf-8"))
    metadata = geometry["metadata"]
    raw_pixels, raw_components = _mask_stats(case_dir / "raw_mask.png")
    repaired_pixels, repaired_components = _mask_stats(case_dir / "repaired_mask.png")

    graph = nx.read_graphml(graphml_path)
    graph_length = sum(float(data.get("length", 0.0)) for _, _, data in graph.edges(data=True))
    graph_components = nx.number_connected_components(graph)
    graph_isolated = sum(1 for node in graph if graph.degree(node) == 0)

    return [
        _row(case, "original_mask", ["raw_mask.png"], road_pixel_count=raw_pixels, connected_component_count=raw_components),
        _row(case, "repaired_mask", ["repaired_mask.png"], road_pixel_count=repaired_pixels, connected_component_count=repaired_components),
        _row(
            case,
            "skeleton",
            ["geometry_summary.json", "geometry_skeleton.png"],
            skeleton_pixel_count=int(metadata["pruned_skeleton_pixels"]),
            connected_component_count=int(metadata["skeleton_connected_components"]),
        ),
        _row(
            case,
            "connected_components",
            ["geometry_summary.json", "topology_summary.json"],
            road_pixel_count=int(metadata["clean_mask_pixels"]),
            skeleton_pixel_count=int(metadata["pruned_skeleton_pixels"]),
            connected_component_count=int(metadata["skeleton_connected_components"]),
            node_count=int(topology["node_count"]),
            edge_count=int(topology["edge_count"]),
        ),
        _row(
            case,
            "endpoint_detection",
            ["geometry_summary.json", "topology_summary.json"],
            skeleton_pixel_count=int(metadata["pruned_skeleton_pixels"]),
            endpoint_count=int(metadata["endpoint_count"]),
            node_count=int(topology["node_count"]),
        ),
        _row(
            case,
            "junction_detection",
            ["geometry_summary.json", "topology_summary.json"],
            skeleton_pixel_count=int(metadata["pruned_skeleton_pixels"]),
            junction_count=int(topology["intersection_count"]),
            node_count=int(topology["node_count"]),
        ),
        _row(
            case,
            "graph_construction",
            ["road_network.graphml", "topology_summary.json"],
            connected_component_count=int(graph_components),
            endpoint_count=int(topology.get("endpoint_count", 0)),
            junction_count=int(topology.get("intersection_count", 0)),
            node_count=int(graph.number_of_nodes()),
            edge_count=int(graph.number_of_edges()),
        ),
        _row(
            case,
            "edge_length_calculation",
            ["geometry_summary.json", "road_network.graphml"],
            node_count=int(graph.number_of_nodes()),
            edge_count=int(graph.number_of_edges()),
            total_graph_length=round(graph_length, 2),
        ),
    ]


def run(input_root: Path, output_path: Path) -> dict:
    rows = []
    for case in CASES:
        rows.extend(collect_case(input_root / case, case))
    report = {
        "description": "Read-only diagnostic checkpoint instrumentation; no pipeline algorithms executed or modified.",
        "input_root": str(input_root),
        "cases": list(CASES),
        "stages": list(STAGES),
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("outputs/phase14_validation"))
    parser.add_argument("--output", type=Path, default=Path("outputs/phase1_diagnostic_checkpoint_report.json"))
    args = parser.parse_args()
    report = run(args.input_root, args.output)
    print("Phase 1 diagnostic checkpoint")
    print("case                  stage                    nodes edges endpoints junctions components length")
    for case in report["cases"]:
        rows = [row for row in report["rows"] if row["case"] == case]
        graph_row = next(row for row in rows if row["stage"] == "graph_construction")
        length_row = next(row for row in rows if row["stage"] == "edge_length_calculation")
        print(
            f"{case:22} graph_construction       "
            f"{graph_row['node_count']:5} {graph_row['edge_count']:5} "
            f"{graph_row['endpoint_count']:9} {graph_row['junction_count']:9} "
            f"{graph_row['connected_component_count']:10} {length_row['total_graph_length']:7.2f}"
        )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
