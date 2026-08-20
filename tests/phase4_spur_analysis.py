"""Non-mutating analysis of branches removed by prune_skeleton_spurs."""

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
    _angle_between_vectors,
    _arc_length,
    _branch_angle_at_junction,
    _classify_pixels,
    _local_mask_support,
    _skeleton_neighbors,
    _trace_from_endpoint,
    extract_skeleton,
)


CASES = (
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


def _classify(
    branch_length: float,
    mask_support: float,
    branch_angle: float,
    cfg: GeometryConfig,
) -> str:
    if mask_support < 0.34:
        return "Genuine noise"
    if branch_length <= cfg.hard_spur_length_px:
        return "Invalid short branch"
    if mask_support >= cfg.legitimate_branch_mask_support and branch_angle >= cfg.legitimate_branch_angle_deg:
        return "Valid junction branch"
    return "Invalid short branch"


def analyze_case(case_dir: Path, case: str, cfg: GeometryConfig) -> list[dict]:
    clean_mask = _mask(case_dir / "geometry_clean_mask.png")
    skeleton = extract_skeleton(clean_mask)
    original = skeleton.copy()
    removed_overlay = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    removed_overlay[original > 0] = (255, 255, 255)
    records: list[dict] = []
    skel = skeleton.copy()

    for iteration in range(cfg.prune_iterations):
        changed = False
        ys, xs = np.where(skel > 0)
        tips = [
            (int(y), int(x))
            for y, x in zip(ys.tolist(), xs.tolist())
            if len(_skeleton_neighbors(int(y), int(x), skel)) == 1
        ]
        for ty, tx in tips:
            if skel[ty, tx] == 0:
                continue
            path, terminal = _trace_from_endpoint(
                ty, tx, skel, max_steps=cfg.min_branch_length_px
            )
            branch_length = _arc_length(path)
            if terminal != "junction" or branch_length >= cfg.min_branch_length_px:
                continue

            branch_angle = _branch_angle_at_junction(path, skel)
            mask_support = _local_mask_support(
                path, clean_mask, cfg.branch_support_radius_px
            )
            preserve = (
                branch_length > cfg.hard_spur_length_px
                and branch_length >= cfg.min_branch_length_px * cfg.legitimate_branch_min_fraction
                and branch_angle >= cfg.legitimate_branch_angle_deg
                and mask_support >= cfg.legitimate_branch_mask_support
            )
            if preserve:
                continue

            removed_pixels = [
                (py, px) for py, px in path[:-1] if skel[py, px] > 0
            ]
            for py, px in removed_pixels:
                skel[py, px] = 0
                removed_overlay[py, px] = (255, 0, 0)  # red in RGB output
            if not removed_pixels:
                continue
            changed = True
            records.append({
                "case": case,
                "iteration": iteration,
                "starting_endpoint": [ty, tx],
                "nearest_junction": list(path[-1]),
                "spur_length_pixels": round(float(branch_length), 3),
                "removed_pixels": len(removed_pixels),
                "touches_junction": terminal == "junction",
                "branch_angle_deg": round(float(branch_angle), 3),
                "mask_support": round(float(mask_support), 4),
                "classification": _classify(branch_length, mask_support, branch_angle, cfg),
            })
        if not changed:
            break

    _, endpoints = _classify_pixels(original)
    junctions, _ = _classify_pixels(original)
    for y, x in endpoints:
        cv2.circle(removed_overlay, (x, y), 3, (0, 255, 0), -1)
    for y, x in junctions:
        cv2.circle(removed_overlay, (x, y), 4, (0, 0, 255), 1)
    overlay_path = case_dir.parent.parent / "phase4_spur_analysis" / f"{case}_removed_spurs.png"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_path), cv2.cvtColor(removed_overlay, cv2.COLOR_RGB2BGR))
    return records


def run(input_root: Path) -> dict:
    cfg = GeometryConfig()
    records = []
    for case in CASES:
        records.extend(analyze_case(input_root / case, case, cfg))
    report = {
        "description": "Diagnosis-only replay of prune_skeleton_spurs; no code or thresholds changed.",
        "operations": {
            "function": "prune_skeleton_spurs",
            "file": "road_extractor/geometry.py",
            "min_branch_length_px": cfg.min_branch_length_px,
            "hard_spur_length_px": cfg.hard_spur_length_px,
            "legitimate_branch_angle_deg": cfg.legitimate_branch_angle_deg,
            "legitimate_branch_mask_support": cfg.legitimate_branch_mask_support,
            "legitimate_branch_min_fraction": cfg.legitimate_branch_min_fraction,
        },
        "cases": list(CASES),
        "removed_spurs": records,
    }
    output_root = input_root.parent / "phase4_spur_analysis"
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "phase4_spur_analysis_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("outputs/phase14_validation"))
    args = parser.parse_args()
    report = run(args.input_root)
    print("Phase 4 spur diagnosis")
    print("case                  pixels_removed spurs avg_length first_classification")
    for case in CASES:
        rows = [row for row in report["removed_spurs"] if row["case"] == case]
        total = sum(row["removed_pixels"] for row in rows)
        avg = sum(row["spur_length_pixels"] for row in rows) / len(rows) if rows else 0.0
        first = rows[0]["classification"] if rows else "None"
        print(f"{case:22} {total:14} {len(rows):5} {avg:10.2f} {first}")
    print("Report: outputs/phase4_spur_analysis/phase4_spur_analysis_report.json")


if __name__ == "__main__":
    main()
