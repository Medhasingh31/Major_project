"""Phase 9 validation harness for the existing Geometry + Topology pipeline.

This module deliberately wraps ``extract_roads`` without changing extraction
algorithms or export formats. It creates per-case graph overlays and a JSON
report containing measurable checks plus review flags for unlabeled behaviors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from road_extractor.config import ExtractionConfig
from road_extractor.data import read_rgb
from road_extractor.pipeline import extract_roads


CASES = {
    "straight_roads": "114656_sat.jpg",
    "curved_roads": "117974_sat.jpg",
    "intersections": "102867_sat.jpg",
    "t_junctions": "12005_sat.jpg",
    "crossroads": "106553_sat.jpg",
    "disconnected_occluded": "100794_sat.jpg",
    "dense_urban": "115141_sat.jpg",
}


def _draw_graph_diagnostic(image_path: Path, geojson_path: Path, output_path: Path) -> None:
    """Draw exported graph edges and nodes on the original RGB image."""
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    graph = json.loads(geojson_path.read_text(encoding="utf-8"))
    for feature in graph.get("features", []):
        geometry = feature.get("geometry", {})
        if geometry.get("type") == "LineString":
            points = np.asarray(geometry.get("coordinates", []), dtype=np.int32)
            if len(points) >= 2:
                cv2.polylines(image, [points[:, [0, 1]]], False, (30, 150, 255), 2)
        elif geometry.get("type") == "Point":
            x, y = [int(round(v)) for v in geometry.get("coordinates", [0, 0])]
            cv2.circle(image, (x, y), 5, (235, 45, 45), -1)
    cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _graph_checks(geojson_path: Path, graphml_path: Path, expected_nodes: int, expected_edges: int) -> dict:
    data = json.loads(geojson_path.read_text(encoding="utf-8"))
    features = data.get("features", [])
    node_ids = {
        int(f["properties"]["id"])
        for f in features
        if f.get("geometry", {}).get("type") == "Point"
    }
    edges = [
        f for f in features
        if f.get("geometry", {}).get("type") == "LineString"
    ]
    valid_edge_refs = all(
        int(f["properties"]["source"]) in node_ids
        and int(f["properties"]["target"]) in node_ids
        for f in edges
    )
    return {
        "node_count_matches_pipeline": len(node_ids) == expected_nodes,
        "edge_count_matches_pipeline": len(edges) == expected_edges,
        "edge_endpoint_references_valid": valid_edge_refs,
        "graphml_exists": graphml_path.exists() and graphml_path.stat().st_size > 0,
        "graph_valid": (
            len(node_ids) == expected_nodes
            and len(edges) == expected_edges
            and valid_edge_refs
            and graphml_path.exists()
        ),
    }


def _make_contact_sheet(case_dirs: list[tuple[str, Path]], output_path: Path) -> None:
    columns = 2
    cell_w, cell_h = 520, 420
    sheet = Image.new("RGB", (columns * cell_w, ((len(case_dirs) + 1) // 2) * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (name, directory) in enumerate(case_dirs):
        image_path = directory / "graph_diagnostic.png"
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((cell_w - 20, cell_h - 55))
        x = (index % columns) * cell_w + 10
        y = (index // columns) * cell_h + 30
        sheet.paste(image, (x, y))
        draw.text((x, 8 + (index // columns) * cell_h), name, fill="black")
    sheet.save(output_path)


def validate(images_dir: Path, weights: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    case_dirs = []
    for case, filename in CASES.items():
        image_path = images_dir / filename
        case_dir = output_dir / case
        result = extract_roads(
            image_path=image_path,
            output_dir=case_dir,
            weights_path=weights,
            config=ExtractionConfig(),
        )
        graph_diagnostic = case_dir / "graph_diagnostic.png"
        _draw_graph_diagnostic(image_path, Path(result["geojson_path"]), graph_diagnostic)
        graph_checks = _graph_checks(
            Path(result["geojson_path"]),
            Path(result["graphml_path"]),
            int(result["topology_nodes"]),
            int(result["topology_edges"]),
        )
        row = {
            "case": case,
            "image": str(image_path),
            "centerline_continuity_review": "inspect graph_diagnostic.png",
            "false_branches_review": "inspect endpoint/junction overlay",
            "missed_branches_review": "requires labeled topology",
            "endpoint_detection_review": "inspect endpoint markers",
            "intersection_detection_review": "inspect junction markers",
            "false_connections_proxy": int(result["topology_bridge_edges"]),
            "disconnected_components": int(result["topology_connected_components"]),
            "suspicious_disconnected_segments": result["suspicious_disconnected_segments"],
            "nodes": int(result["topology_nodes"]),
            "edges": int(result["topology_edges"]),
            "intersections": int(result["topology_intersections"]),
            "endpoints": int(result["topology_endpoints"]),
            "segments": int(result["segments"]),
            "centerline_length_pixels": float(result["total_length_pixels"]),
            "graph_checks": graph_checks,
            "diagnostics": [
                str(case_dir / "geometry_diagnostic.png"),
                str(graph_diagnostic),
                str(case_dir / "topology_summary.json"),
            ],
        }
        rows.append(row)
        case_dirs.append((case, case_dir))

    report = {
        "description": "Phase 9 structural validation; algorithms and exports unchanged.",
        "weights": str(weights),
        "cases": rows,
        "limitations": [
            "No per-layout topology ground truth was supplied; branch and intersection judgments are review flags.",
            "False-connection proxy is zero because topology foundation creates no inferred bridge edges.",
        ],
        "contact_sheet": str(output_dir / "validation_contact_sheet.png"),
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _make_contact_sheet(case_dirs, output_dir / "validation_contact_sheet.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Geometry + Topology on representative layouts")
    parser.add_argument("--images-dir", type=Path, default=Path("deepglobe/valid"))
    parser.add_argument("--weights", type=Path, default=Path("models/road_unet_occlusion.keras"))
    parser.add_argument("--output", type=Path, default=Path("outputs/phase9_validation"))
    args = parser.parse_args()
    report = validate(args.images_dir, args.weights, args.output)
    for row in report["cases"]:
        print(
            f"{row['case']}: nodes={row['nodes']} edges={row['edges']} "
            f"intersections={row['intersections']} endpoints={row['endpoints']} "
            f"components={row['disconnected_components']} graph_valid={row['graph_checks']['graph_valid']}"
        )
    print(f"Report: {args.output / 'validation_report.json'}")
    print(f"Contact sheet: {args.output / 'validation_contact_sheet.png'}")


if __name__ == "__main__":
    main()
