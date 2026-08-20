"""Diagnosis-only replay of _remove_small_skeleton_components."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

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


def _component_sizes(binary: np.ndarray) -> list[int]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return sorted((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)), reverse=True)


def _classification(area: int, line_like: bool, adjacent: bool) -> str:
    if not line_like and not adjacent:
        return "Invalid isolated component"
    if line_like:
        return "Valid road geometry"
    return "Unclear"


def analyze_case(case_dir: Path, case: str, cfg: GeometryConfig) -> list[dict]:
    clean = _mask(case_dir / "geometry_clean_mask.png")
    skeleton = extract_skeleton(clean)
    skeleton, _ = prune_skeleton_spurs(
        skeleton,
        min_branch_length=cfg.min_branch_length_px,
        iterations=cfg.prune_iterations,
        clean_mask=clean,
        support_radius=cfg.branch_support_radius_px,
        hard_spur_length=cfg.hard_spur_length_px,
        legitimate_branch_angle=cfg.legitimate_branch_angle_deg,
        legitimate_branch_mask_support=cfg.legitimate_branch_mask_support,
        legitimate_branch_min_fraction=cfg.legitimate_branch_min_fraction,
    )
    skeleton, _ = prune_unsupported_branches(
        skeleton, clean,
        max_branch_length=cfg.max_unsupported_branch_length_px,
        angle_tolerance_deg=cfg.unsupported_branch_angle_deg,
        support_radius=cfg.branch_support_radius_px,
    )
    skeleton, _ = prune_tiny_internal_branches(
        skeleton, clean,
        max_branch_length=cfg.max_internal_branch_length_px,
        min_mask_support=cfg.internal_branch_mask_support,
        support_radius=cfg.branch_support_radius_px,
    )

    filtered, _ = _remove_small_skeleton_components(
        skeleton,
        min_size=cfg.min_skeleton_component_px,
        min_road_like_pixels=max(10, cfg.min_skeleton_component_px // 2),
        min_road_like_span=cfg.min_small_component_span_px,
    )
    removed = (skeleton > 0) & (filtered == 0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        removed.astype(np.uint8), connectivity=8
    )
    kept_dilated = cv2.dilate(filtered, np.ones((3, 3), dtype=np.uint8)) > 0
    rows = []
    overlay = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    overlay[skeleton > 0] = (255, 255, 255)
    overlay[removed] = (255, 0, 0)
    overlay[filtered > 0] = (180, 180, 180)

    for label in range(1, count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        ys, xs = np.where(component)
        component_img = component.astype(np.uint8)
        endpoints = sum(
            len([
                (yy, xx)
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
                if (dy or dx)
                and 0 <= (yy := int(yy0 + dy)) < component_img.shape[0]
                and 0 <= (xx := int(xx0 + dx)) < component_img.shape[1]
                and component_img[yy, xx] > 0
            ]) == 1
            for yy0, xx0 in zip(ys.tolist(), xs.tolist())
        )
        line_like = (
            area >= max(10, cfg.min_skeleton_component_px // 2)
            and max(width, height) >= cfg.min_small_component_span_px
            and endpoints == 2
        )
        adjacent = bool(np.any(component & kept_dilated))
        rows.append({
            "case": case,
            "component_label": label,
            "component_size": area,
            "bbox_xywh": [x, y, width, height],
            "threshold": cfg.min_skeleton_component_px,
            "min_road_like_pixels": max(10, cfg.min_skeleton_component_px // 2),
            "min_road_like_span": cfg.min_small_component_span_px,
            "endpoint_count": int(endpoints),
            "line_like": bool(line_like),
            "connected_to_main_road": False,
            "adjacent_to_kept_skeleton": adjacent,
            "classification": _classification(area, line_like, adjacent),
        })

    for yy, xx in zip(*np.where(removed)):
        cv2.circle(overlay, (int(xx), int(yy)), 1, (255, 0, 0), -1)
    overlay_path = case_dir.parent.parent / "phase6_component_diagnosis" / f"{case}_component_difference.png"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return rows


def run(input_root: Path) -> dict:
    cfg = GeometryConfig()
    rows = []
    for case in CASES:
        rows.extend(analyze_case(input_root / case, case, cfg))
    report = {
        "description": "Diagnosis-only replay; no component-filter, spur, skeleton, topology, or graph changes.",
        "thresholds": {
            "min_skeleton_component_px": cfg.min_skeleton_component_px,
            "min_road_like_pixels": max(10, cfg.min_skeleton_component_px // 2),
            "min_road_like_span_px": cfg.min_small_component_span_px,
        },
        "rows": rows,
    }
    output_root = input_root.parent / "phase6_component_diagnosis"
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "phase6_component_diagnosis_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("outputs/phase14_validation"))
    args = parser.parse_args()
    report = run(args.input_root)
    print("case                  pixels_removed components classifications")
    for case in CASES:
        entries = [x for x in report["rows"] if x["case"] == case]
        print(case, sum(x["component_size"] for x in entries), len(entries), [x["classification"] for x in entries])
    print("Report: outputs/phase6_component_diagnosis/phase6_component_diagnosis_report.json")
