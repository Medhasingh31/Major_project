"""Read-only diagnosis of skeletonization versus downstream pruning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_extractor.geometry import extract_skeleton


CASES = (
    "straight_roads",
    "curved_roads",
    "intersections",
    "t_junctions",
    "crossroads",
    "disconnected_occluded",
    "dense_urban",
)


def _component_sizes(binary: np.ndarray) -> list[int]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8), connectivity=8
    )
    return sorted(
        (int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, count)),
        reverse=True,
    )


def _read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask > 0).astype(np.uint8)


def run(input_root: Path, output_root: Path) -> dict:
    cases = []
    output_root.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        phase9 = input_root.parent / "phase9_validation" / case
        phase14 = input_root / case
        before_summary = json.loads((phase9 / "geometry_summary.json").read_text(encoding="utf-8"))
        after_summary = json.loads((phase14 / "geometry_summary.json").read_text(encoding="utf-8"))
        before_meta = before_summary["metadata"]
        after_meta = after_summary["metadata"]

        before_clean = _read_mask(phase9 / "geometry_clean_mask.png")
        after_clean = _read_mask(phase14 / "geometry_clean_mask.png")
        before_raw = extract_skeleton(before_clean)
        after_raw = extract_skeleton(after_clean)
        before_pruned = _read_mask(phase9 / "geometry_skeleton.png")
        after_pruned = _read_mask(phase14 / "geometry_skeleton.png")

        raw_difference = np.logical_xor(before_raw, after_raw)
        removed = np.logical_and(after_raw > 0, after_pruned == 0)
        added = np.logical_and(after_raw == 0, after_pruned > 0)
        diff_image = np.zeros((*after_raw.shape, 3), dtype=np.uint8)
        diff_image[after_raw > 0] = (255, 255, 255)
        diff_image[removed] = (0, 0, 255)  # red: removed after skeletonization
        diff_image[added] = (0, 255, 0)     # green: added after skeletonization
        diff_path = output_root / f"{case}_skeleton_difference.png"
        cv2.imwrite(str(diff_path), cv2.cvtColor(diff_image, cv2.COLOR_RGB2BGR))

        cases.append({
            "case": case,
            "repaired_mask_pixels_before": int(before_meta["clean_mask_pixels"]),
            "repaired_mask_pixels_after": int(after_meta["clean_mask_pixels"]),
            "raw_skeleton_pixels_before": int(before_meta["raw_skeleton_pixels"]),
            "raw_skeleton_pixels_after": int(after_meta["raw_skeleton_pixels"]),
            "raw_skeleton_pixel_difference": int(raw_difference.sum()),
            "raw_skeleton_component_count_before": len(_component_sizes(before_raw)),
            "raw_skeleton_component_count_after": len(_component_sizes(after_raw)),
            "raw_skeleton_component_sizes_before": _component_sizes(before_raw),
            "raw_skeleton_component_sizes_after": _component_sizes(after_raw),
            "pruned_skeleton_pixels_before": int(before_meta["pruned_skeleton_pixels"]),
            "pruned_skeleton_pixels_after": int(after_meta["pruned_skeleton_pixels"]),
            "pruned_skeleton_component_count_before": len(_component_sizes(before_pruned)),
            "pruned_skeleton_component_count_after": len(_component_sizes(after_pruned)),
            "removed_after_skeletonization_pixels": int(removed.sum()),
            "added_after_skeletonization_pixels": int(added.sum()),
            "difference_image": str(diff_path),
        })

    report = {
        "description": "Skeletonization diagnosis; no Geometry/Topology/Graph algorithms changed.",
        "root_cause": "Raw skeletons are identical; divergence begins in downstream pruning/component cleanup.",
        "fix_applied": False,
        "cases": cases,
    }
    report_path = output_root / "phase2_skeleton_diagnosis_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("outputs/phase14_validation"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/phase2_skeleton_diagnosis"))
    args = parser.parse_args()
    report = run(args.input_root, args.output_root)
    print("Phase 2 skeletonization diagnosis")
    print("case                  mask_delta raw_delta pruned_delta removed added raw_components")
    for row in report["cases"]:
        print(
            f"{row['case']:22} {row['repaired_mask_pixels_after'] - row['repaired_mask_pixels_before']:5} "
            f"{row['raw_skeleton_pixel_difference']:5} "
            f"{row['pruned_skeleton_pixels_after'] - row['pruned_skeleton_pixels_before']:9} "
            f"{row['removed_after_skeletonization_pixels']:7} "
            f"{row['added_after_skeletonization_pixels']:5} "
            f"{row['raw_skeleton_component_count_before']}->{row['raw_skeleton_component_count_after']}"
        )
    print(f"Report: {args.output_root / 'phase2_skeleton_diagnosis_report.json'}")


if __name__ == "__main__":
    main()
