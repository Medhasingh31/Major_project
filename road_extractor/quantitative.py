"""Quantitative before/after reporting for Geometry + Topology refinement."""

from __future__ import annotations

from typing import Mapping, Any


METRIC_NAMES = (
    "road_pixels",
    "skeleton_pixels",
    "connected_components",
    "endpoints",
    "junctions_intersections",
    "edges",
    "suspicious_disconnected_segments",
    "total_centerline_length",
)


def _as_metric_record(record: Mapping[str, Any]) -> dict[str, float | int]:
    """Normalize pipeline results or report rows to the Phase 10 schema."""
    suspicious = record.get("suspicious_disconnected_segments", [])
    if isinstance(suspicious, (int, float)):
        suspicious_count = int(suspicious)
    else:
        suspicious_count = len(suspicious)
    return {
        "road_pixels": int(record.get("road_pixels", 0)),
        "skeleton_pixels": int(record.get("skeleton_pixels", 0)),
        "connected_components": int(
            record.get("connected_components", record.get("disconnected_components", 0))
        ),
        "endpoints": int(record.get("endpoints", 0)),
        "junctions_intersections": int(
            record.get("junctions_intersections", record.get("intersections", record.get("junctions", 0)))
        ),
        "edges": int(record.get("edges", record.get("topology_edges", 0))),
        "suspicious_disconnected_segments": suspicious_count,
        "total_centerline_length": round(float(
            record.get("total_centerline_length", record.get("centerline_length_pixels", record.get("total_length_pixels", 0.0)))
        ), 2),
    }


def compare_metrics(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    before_label: str = "before",
    after_label: str = "after",
) -> dict[str, Any]:
    """Return metric values, deltas, and neutral quality-review guidance."""
    before_metrics = _as_metric_record(before)
    after_metrics = _as_metric_record(after)
    deltas = {
        name: round(float(after_metrics[name]) - float(before_metrics[name]), 2)
        for name in METRIC_NAMES
    }
    return {
        "before_label": before_label,
        "after_label": after_label,
        "before": before_metrics,
        "after": after_metrics,
        "delta_after_minus_before": deltas,
        "interpretation": {
            "fewer_objects_not_assumed_better": True,
            "continuity_requires_visual_or_labeled_review": True,
            "separate_components_may_be_legitimate": True,
        },
    }
