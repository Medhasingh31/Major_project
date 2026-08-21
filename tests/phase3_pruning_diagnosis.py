"""External instrumentation of Raw Skeleton -> Pruned Skeleton operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_extractor.geometry import (
    GeometryConfig,
    _remove_small_skeleton_components,
    bridge_small_gaps,
    extract_skeleton,
    prune_skeleton_spurs,
    prune_tiny_internal_branches,
    prune_unsupported_branches,
)


CASES = (
    "straight_roads",
    "curved_roads",
    "intersections",
    "t_junctions",
    "crossroads",
    "disconnected_occluded",
    "dense_urban",
)


def _mask(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise FileNotFoundError(path)
    return (value > 0).astype(np.uint8)


def _record(case: str, operation: str, before: np.ndarray, after: np.ndarray) -> dict:
    before_pixels = int(np.sum(before > 0))
    after_pixels = int(np.sum(after > 0))
    return {
        "case": case,
        "operation": operation,
        "pixels_before": before_pixels,
        "pixels_after": after_pixels,
        "removed": max(before_pixels - after_pixels, 0),
        "added": max(after_pixels - before_pixels, 0),
    }


def diagnose_case(case_dir: Path, case: str, cfg: GeometryConfig) -> list[dict]:
    clean_mask = _mask(case_dir / "geometry_clean_mask.png")
    skeleton = extract_skeleton(clean_mask)
    rows = [{
        "case": case,
        "operation": "raw_skeleton_baseline",
        "pixels_before": int(np.sum(skeleton > 0)),
        "pixels_after": int(np.sum(skeleton > 0)),
        "removed": 0,
        "added": 0,
    }]

    next_skeleton, _ = prune_skeleton_spurs(
        skeleton,
        min_branch_length=cfg.min_branch_length_px,
        iterations=cfg.prune_iterations,
        clean_mask=clean_mask,
        support_radius=cfg.branch_support_radius_px,
        hard_spur_length=cfg.hard_spur_length_px,
        legitimate_branch_angle=cfg.legitimate_branch_angle_deg,
        legitimate_branch_mask_support=cfg.legitimate_branch_mask_support,
        legitimate_branch_min_fraction=cfg.legitimate_branch_min_fraction,
    )
    rows.append(_record(case, "prune_skeleton_spurs", skeleton, next_skeleton))
    skeleton = next_skeleton

    next_skeleton, _ = prune_unsupported_branches(
        skeleton,
        clean_mask,
        max_branch_length=cfg.max_unsupported_branch_length_px,
        angle_tolerance_deg=cfg.unsupported_branch_angle_deg,
        support_radius=cfg.branch_support_radius_px,
    )
    rows.append(_record(case, "prune_unsupported_branches", skeleton, next_skeleton))
    skeleton = next_skeleton

    next_skeleton, _ = prune_tiny_internal_branches(
        skeleton,
        clean_mask,
        max_branch_length=cfg.max_internal_branch_length_px,
        min_mask_support=cfg.internal_branch_mask_support,
        support_radius=cfg.branch_support_radius_px,
    )
    rows.append(_record(case, "prune_tiny_internal_branches", skeleton, next_skeleton))
    skeleton = next_skeleton

    next_skeleton, _ = _remove_small_skeleton_components(
        skeleton,
        min_size=cfg.min_skeleton_component_px,
        min_road_like_pixels=max(10, cfg.min_skeleton_component_px // 2),
        min_road_like_span=cfg.min_small_component_span_px,
    )
    rows.append(_record(case, "remove_small_skeleton_components", skeleton, next_skeleton))
    skeleton = next_skeleton

    next_skeleton, _ = bridge_small_gaps(
        skeleton,
        clean_mask,
        max_gap=cfg.max_gap_bridge_px,
        angle_tolerance_deg=cfg.bridge_angle_tolerance_deg,
        mask_support_ratio=cfg.bridge_mask_support_ratio,
        crossing_clearance=cfg.bridge_crossing_clearance_px,
    )
    rows.append(_record(case, "bridge_small_gaps", skeleton, next_skeleton))
    return rows


def run(input_root: Path, output_root: Path) -> dict:
    cfg = GeometryConfig()
    rows = []
    for case in CASES:
        rows.extend(diagnose_case(input_root / case, case, cfg))
    first_removal = {}
    for case in CASES:
        case_rows = [row for row in rows if row["case"] == case]
        first_removal[case] = next(
            (row["operation"] for row in case_rows if row["removed"] > 0),
            None,
        )
    report = {
        "description": "External pruning instrumentation; no core algorithm changes.",
        "operations_in_pipeline_order": [
            "raw_skeleton_baseline",
            "prune_skeleton_spurs",
            "prune_unsupported_branches",
            "prune_tiny_internal_branches",
            "remove_small_skeleton_components",
            "bridge_small_gaps",
        ],
        "first_removal_by_case": first_removal,
        "rows": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "phase3_pruning_diagnosis_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("outputs/phase14_validation"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/phase3_pruning_diagnosis"))
    args = parser.parse_args()
    report = run(args.input_root, args.output_root)
    print("Phase 3 pruning diagnosis")
    print("case                  operation                         before after removed")
    for case in CASES:
        for row in report["rows"]:
            if row["case"] == case and row["removed"] > 0:
                print(f"{case:22} {row['operation']:32} {row['pixels_before']:6} {row['pixels_after']:5} {row['removed']:7}")
        print(f"  first removal: {report['first_removal_by_case'][case]}")
    print(f"Report: {args.output_root / 'phase3_pruning_diagnosis_report.json'}")


if __name__ == "__main__":
    main()
