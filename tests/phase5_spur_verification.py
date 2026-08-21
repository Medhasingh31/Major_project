"""Verification for the conservative prune_skeleton_spurs decision update."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_extractor.geometry import GeometryConfig, extract_skeleton, prune_skeleton_spurs


CASES = (
    "intersections",
    "t_junctions",
    "crossroads",
    "disconnected_occluded",
    "dense_urban",
    "straight_roads",
    "curved_roads",
)


def _mask(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise FileNotFoundError(path)
    return (value > 0).astype(np.uint8)


def run() -> dict:
    root = Path("outputs")
    previous = json.loads(
        (root / "phase4_spur_analysis" / "phase4_spur_analysis_report.json").read_text(
            encoding="utf-8"
        )
    )
    cfg = GeometryConfig()
    rows = []
    for case in CASES:
        case_rows = [row for row in previous["removed_spurs"] if row["case"] == case]
        clean = _mask(root / "phase14_validation" / case / "geometry_clean_mask.png")
        raw = extract_skeleton(clean)
        pruned, removed = prune_skeleton_spurs(
            raw,
            min_branch_length=cfg.min_branch_length_px,
            iterations=cfg.prune_iterations,
            clean_mask=clean,
            support_radius=cfg.branch_support_radius_px,
            hard_spur_length=cfg.hard_spur_length_px,
            legitimate_branch_angle=cfg.legitimate_branch_angle_deg,
            legitimate_branch_mask_support=cfg.legitimate_branch_mask_support,
            legitimate_branch_min_fraction=cfg.legitimate_branch_min_fraction,
        )
        valid_candidates = [
            row for row in case_rows if row["classification"] == "Valid junction branch"
        ]
        preserved = sum(
            1
            for row in valid_candidates
            if pruned[tuple(row["starting_endpoint"])] > 0
        )
        rows.append({
            "case": case,
            "pixels_removed_before": int(sum(row["removed_pixels"] for row in case_rows)),
            "pixels_removed_after": int(removed),
            "valid_junction_branches_identified": len(valid_candidates),
            "valid_junction_branches_preserved": int(preserved),
            "noise_removed_pixels": int(
                sum(row["removed_pixels"] for row in case_rows if row["classification"] != "Valid junction branch")
            ),
            "raw_skeleton_pixels": int(np.sum(raw > 0)),
            "pruned_skeleton_pixels": int(np.sum(pruned > 0)),
            "added_skeleton_pixels": int(np.sum((raw == 0) & (pruned > 0))),
        })
    report = {
        "description": "Verification of conservative multi-evidence spur pruning.",
        "rows": rows,
    }
    output = root / "phase5_spur_verification_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    report = run()
    print("case                  removed_before removed_after valid_preserved noise_removed added")
    for row in report["rows"]:
        print(
            f"{row['case']:22} {row['pixels_removed_before']:15} "
            f"{row['pixels_removed_after']:14} {row['valid_junction_branches_preserved']:15} "
            f"{row['noise_removed_pixels']:13} {row['added_skeleton_pixels']:5}"
        )
    print("Report: outputs/phase5_spur_verification_report.json")
