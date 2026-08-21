"""Phase 14 end-to-end validation; no extraction algorithms are changed."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from road_extractor.config import ExtractionConfig
from road_extractor.pipeline import extract_roads
from road_extractor.validate_pipeline import _draw_graph_diagnostic


VALIDATION_CASES = {
    "straight_roads": "114656_sat.jpg",
    "curved_roads": "117974_sat.jpg",
    "intersections": "102867_sat.jpg",
    "t_junctions": "12005_sat.jpg",
    "crossroads": "106553_sat.jpg",
    "disconnected_occluded": "100794_sat.jpg",
    "dense_urban": "115141_sat.jpg",
}

EXTRA_CASES = {
    "representative_aerial_1": "100905_sat.jpg",
    "representative_aerial_2": "117532_sat.jpg",
    "representative_aerial_3": "125414_sat.jpg",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_outputs(result: dict, output_dir: Path) -> dict:
    raw = cv2.imread(str(output_dir / "raw_mask.png"), cv2.IMREAD_GRAYSCALE)
    repaired = cv2.imread(str(output_dir / "repaired_mask.png"), cv2.IMREAD_GRAYSCALE)
    mask_valid = raw is not None and repaired is not None and raw.size > 0 and repaired.size > 0
    mask_values_valid = bool(mask_valid and set(np.unique(raw)).issubset({0, 255}) and set(np.unique(repaired)).issubset({0, 255}))

    geojson = json.loads(Path(result["geojson_path"]).read_text(encoding="utf-8"))
    features = geojson.get("features", [])
    nodes = [f for f in features if f.get("geometry", {}).get("type") == "Point"]
    edges = [f for f in features if f.get("geometry", {}).get("type") == "LineString"]
    node_ids = {int(f["properties"]["id"]) for f in nodes}
    edge_refs_valid = all(
        int(f["properties"]["source"]) in node_ids
        and int(f["properties"]["target"]) in node_ids
        for f in edges
    )

    graphml_path = Path(result["graphml_path"])
    try:
        graphml = nx.read_graphml(graphml_path)
        graphml_valid = graphml.number_of_nodes() == len(nodes) and graphml.number_of_edges() == len(edges)
    except Exception:
        graphml_valid = False

    required_diagnostics = [
        "raw_mask.png", "repaired_mask.png", "geometry_diagnostic.png",
        "geometry_skeleton.png", "topology_summary.json", "road_network.geojson",
        "road_network.graphml",
    ]
    diagnostics_valid = all((output_dir / name).exists() and (output_dir / name).stat().st_size > 0 for name in required_diagnostics)
    return {
        "mask_valid": mask_valid,
        "mask_values_valid": mask_values_valid,
        "raw_road_fraction": round(float(np.mean(raw > 0)), 6) if mask_valid else 0.0,
        "repaired_road_fraction": round(float(np.mean(repaired > 0)), 6) if mask_valid else 0.0,
        "geojson_node_count_matches": len(nodes) == int(result["topology_nodes"]),
        "geojson_edge_count_matches": len(edges) == int(result["topology_edges"]),
        "geojson_edge_references_valid": edge_refs_valid,
        "graphml_valid": graphml_valid,
        "diagnostics_valid": diagnostics_valid,
        "graph_valid": (
            len(nodes) == int(result["topology_nodes"])
            and len(edges) == int(result["topology_edges"])
            and edge_refs_valid
            and graphml_valid
        ),
    }


def run_case(name: str, filename: str, images_dir: Path, weights: Path, output_root: Path) -> dict:
    image_path = images_dir / filename
    output_dir = output_root / name
    started = time.perf_counter()
    result = extract_roads(
        image_path=image_path,
        output_dir=output_dir,
        weights_path=weights,
        config=ExtractionConfig(),
    )
    elapsed = time.perf_counter() - started
    _draw_graph_diagnostic(image_path, Path(result["geojson_path"]), output_dir / "graph_diagnostic.png")
    checks = _check_outputs(result, output_dir)
    return {
        "case": name,
        "image": filename,
        "runtime_seconds": round(elapsed, 3),
        "road_pixels": int(result["road_pixels"]),
        "skeleton_pixels": int(result["skeleton_pixels"]),
        "segments": int(result["segments"]),
        "nodes": int(result["topology_nodes"]),
        "edges": int(result["topology_edges"]),
        "bridges": int(result["topology_bridge_edges"]),
        "endpoints": int(result["topology_endpoints"]),
        "intersections": int(result["topology_intersections"]),
        "connected_components": int(result["topology_connected_components"]),
        "suspicious_disconnected_segments": result["suspicious_disconnected_segments"],
        "checks": checks,
        "all_checks_passed": all(checks.values()) and int(result["topology_bridge_edges"]) == 0,
        "output_dir": str(output_dir),
    }


def main() -> None:
    images_dir = ROOT / "deepglobe" / "valid"
    weights = ROOT / "models" / "road_unet_occlusion.keras"
    output_root = ROOT / "outputs" / "phase14_validation"
    output_root.mkdir(parents=True, exist_ok=True)
    before_hashes = {"weights": _sha256(weights)}
    all_cases = {**VALIDATION_CASES, **EXTRA_CASES}
    rows = [run_case(name, filename, images_dir, weights, output_root) for name, filename in all_cases.items()]
    after_hashes = {"weights": _sha256(weights)}

    baseline = json.loads((ROOT / "outputs" / "phase9_validation" / "validation_report.json").read_text(encoding="utf-8"))
    phase13 = json.loads((ROOT / "outputs" / "phase13_validation" / "validation_report.json").read_text(encoding="utf-8"))
    report = {
        "description": "Phase 14 final end-to-end validation; algorithms and export formats unchanged.",
        "weights": str(weights),
        "model_state_unchanged": before_hashes == after_hashes,
        "cases": rows,
        "summary": {
            "case_count": len(rows),
            "passed_cases": sum(row["all_checks_passed"] for row in rows),
            "graph_valid_cases": sum(row["checks"]["graph_valid"] for row in rows),
            "diagnostics_valid_cases": sum(row["checks"]["diagnostics_valid"] for row in rows),
            "total_bridges": sum(row["bridges"] for row in rows),
            "total_runtime_seconds": round(sum(row["runtime_seconds"] for row in rows), 3),
        },
        "comparison": {
            "phase9_validation_cases": len(baseline["cases"]),
            "phase9_graph_valid_cases": sum(row["graph_checks"]["graph_valid"] for row in baseline["cases"]),
            "phase13_validation_cases": len(phase13["cases"]),
            "phase13_graph_valid_cases": sum(row["graph_checks"]["graph_valid"] for row in phase13["cases"]),
            "phase13_false_connection_proxy": sum(row["false_connections_proxy"] for row in phase13["cases"]),
        },
        "limitations": [
            "No labeled topology ground truth was supplied; continuity and false-branch judgments are structural proxies/review items.",
            "A zero bridge count confirms no unsupported bridges were added in these runs; it does not prove all missing roads were recovered.",
        ],
    }
    report_path = output_root / "phase14_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in rows:
        print(f"{row['case']}: runtime={row['runtime_seconds']}s nodes={row['nodes']} edges={row['edges']} bridges={row['bridges']} components={row['connected_components']} graph_valid={row['checks']['graph_valid']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
