"""
confidence.py — Road Extraction Confidence Scoring Module
==========================================================
Part of the Geometry + Topology pipeline (separate from the U-Net segmentation pipeline).

INPUT
-----
- rgb_image    : np.ndarray    — original RGB image
- repaired_mask: np.ndarray    — output of repair_road_mask()
- geometry     : RoadGeometry  — output of geometry.extract_geometry()
- topology     : RoadTopology  — output of topology.build_topology()

This module does NOT touch the U-Net, raw_mask, or training code.

OUTPUT
------
Returns a ConfidenceReport dataclass containing:
- segment_scores   : dict[segment_id → SegmentConfidence]
- network_score    : float  [0.0, 1.0]  — overall network-level confidence
- flags            : list[ConfidenceFlag] — anomalies / warnings
- metadata         : dict

SCORING DIMENSIONS
------------------
Each segment is scored on five independent dimensions (each in [0, 1]):

  1. mask_coverage   — fraction of segment pixels that are positive in repaired_mask
  2. color_contrast  — normalised stddev of brightness along the segment
                       (high contrast → likely a real road edge)
  3. straightness    — 1 − normalised curvature  (straight roads score higher)
  4. connectivity    — proportion of segment endpoints that connect to the network
                       (isolated dead-ends score lower)
  5. length_score    — sigmoid-like score; very short fragments score low

A weighted mean of these five dimensions gives the overall segment confidence.

PIPELINE POSITION
-----------------
    RoadGeometry  ─┐
    RoadTopology  ─┤─► confidence.score_topology() ──► ConfidenceReport
    rgb_image     ─┘
    repaired_mask ─┘
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from road_extractor.geometry import RoadGeometry, RoadSegment
from road_extractor.topology import RoadTopology


# ---------------------------------------------------------------------------
# Weights for the five scoring dimensions
# (must sum to 1.0 — adjustable without touching logic)
# ---------------------------------------------------------------------------
DIMENSION_WEIGHTS: dict[str, float] = {
    "mask_coverage":  0.30,
    "color_contrast": 0.20,
    "straightness":   0.15,
    "connectivity":   0.25,
    "length_score":   0.10,
}

# Minimum segment length (pixels) below which length_score is penalised
MIN_SEGMENT_LENGTH: float = 10.0

# Curvature above which straightness_score approaches 0
MAX_CURVATURE: float = 0.5  # radians/pixel

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SegmentConfidence:
    """Per-segment confidence report."""

    segment_id: int
    overall: float                         # weighted composite score [0, 1]

    # Individual dimension scores [0, 1]
    mask_coverage: float = 0.0
    color_contrast: float = 0.0
    straightness: float = 0.0
    connectivity: float = 0.0
    length_score: float = 0.0

    # Human-readable notes added by scoring logic
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "overall": round(self.overall, 4),
            "dimensions": {
                "mask_coverage": round(self.mask_coverage, 4),
                "color_contrast": round(self.color_contrast, 4),
                "straightness": round(self.straightness, 4),
                "connectivity": round(self.connectivity, 4),
                "length_score": round(self.length_score, 4),
            },
            "notes": self.notes,
        }


@dataclass
class ConfidenceFlag:
    """
    An anomaly or warning raised during scoring.

    severity : 'info' | 'warning' | 'error'
    """
    severity: str          # 'info' | 'warning' | 'error'
    kind: str              # short machine-readable code
    message: str
    segment_id: Optional[int] = None
    node_id: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "kind": self.kind,
            "message": self.message,
            "segment_id": self.segment_id,
            "node_id": self.node_id,
        }


@dataclass
class ConfidenceReport:
    """Container for all confidence scoring results."""

    segment_scores: dict[int, SegmentConfidence] = field(default_factory=dict)
    network_score: float = 0.0         # mean of all segment overall scores
    flags: list[ConfidenceFlag] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # -----------------------------------------------------------------------
    # Convenience accessors
    # -----------------------------------------------------------------------

    def low_confidence_segments(self, threshold: float = 0.4) -> list[int]:
        """Return segment_ids whose overall score is below *threshold*."""
        return [
            sid for sid, sc in self.segment_scores.items()
            if sc.overall < threshold
        ]

    def high_confidence_segments(self, threshold: float = 0.7) -> list[int]:
        """Return segment_ids whose overall score is at or above *threshold*."""
        return [
            sid for sid, sc in self.segment_scores.items()
            if sc.overall >= threshold
        ]

    def to_summary(self) -> dict:
        return {
            "network_score": round(self.network_score, 4),
            "segment_count": len(self.segment_scores),
            "high_confidence": len(self.high_confidence_segments()),
            "low_confidence": len(self.low_confidence_segments()),
            "flag_count": len(self.flags),
            "flags": [f.to_dict() for f in self.flags],
            "segments": {
                str(sid): sc.to_dict()
                for sid, sc in self.segment_scores.items()
            },
            **self.metadata,
        }


# ---------------------------------------------------------------------------
# Internal scoring helpers
# ---------------------------------------------------------------------------


def _score_mask_coverage(
    seg: RoadSegment,
    repaired_mask: np.ndarray,
) -> float:
    """
    Fraction of segment pixels that are positive (== 1) in the repaired_mask.
    Perfect coverage → 1.0; all pixels missing → 0.0.
    """
    h, w = repaired_mask.shape
    hits = sum(
        1
        for y, x in seg.pixel_path
        if 0 <= y < h and 0 <= x < w and repaired_mask[y, x] > 0
    )
    if not seg.pixel_path:
        return 0.0
    return hits / len(seg.pixel_path)


def _score_color_contrast(
    seg: RoadSegment,
    rgb_image: np.ndarray,
) -> float:
    """
    Normalised brightness standard deviation along the segment.
    Road surfaces tend to have a different (often more uniform) brightness
    than surroundings, but with clear edges.
    Score is stddev(brightness) / 128 clamped to [0, 1].
    Higher contrast may indicate a well-defined road edge.
    """
    h, w = rgb_image.shape[:2]
    gray_vals = []
    for y, x in seg.pixel_path:
        if 0 <= y < h and 0 <= x < w:
            r, g, b = rgb_image[y, x].astype(float)
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            gray_vals.append(brightness)

    if not gray_vals:
        return 0.0

    stddev = float(np.std(gray_vals))
    return min(stddev / 128.0, 1.0)


def _score_straightness(seg: RoadSegment) -> float:
    """
    1.0 for perfectly straight segments; approaches 0.0 for highly curved ones.
    Uses the segment's pre-computed curvature (radians/pixel).
    """
    normalised = min(seg.curvature / MAX_CURVATURE, 1.0)
    return 1.0 - normalised


def _score_connectivity(
    seg: RoadSegment,
    topology: RoadTopology,
) -> float:
    """
    Score based on how well-connected the segment's endpoints are in the
    topology graph.

    - Both endpoints connected to other segments  → 1.0
    - One endpoint connected                       → 0.5
    - Isolated (no graph or missing nodes)         → 0.0

    Uses topology.disconnected_segs for a fast O(1) check first, then
    falls back to graph degree inspection.
    An endpoint is "connected" if its graph node degree is ≥ 2.
    """
    if topology.graph is None:
        return 0.0

    # Fast path: segment is known-disconnected from topology analysis
    if hasattr(topology, "disconnected_segs") and seg.segment_id in topology.disconnected_segs:
        return 0.0

    G = topology.graph

    # Build a pixel→node lookup on first use (cached via a closure-free scan)
    pixel_to_node: dict[tuple[int, int], int] = {
        (data["y"], data["x"]): nid
        for nid, data in G.nodes(data=True)
    }

    connected_ends = 0
    for pixel in (seg.pixel_path[0], seg.pixel_path[-1]):
        node_id = pixel_to_node.get(pixel)
        if node_id is not None and G.degree(node_id) >= 2:
            connected_ends += 1

    return connected_ends / 2.0


def _score_length(seg: RoadSegment) -> float:
    """
    Sigmoid-like score penalising very short segments (likely noise).

    length ≥ MIN_SEGMENT_LENGTH * 4  → ~1.0
    length == MIN_SEGMENT_LENGTH     → 0.5
    length → 0                       → 0.0
    """
    if seg.length_pixels <= 0:
        return 0.0
    # Smooth step: uses logistic function centred at MIN_SEGMENT_LENGTH
    x = seg.length_pixels / MIN_SEGMENT_LENGTH - 1.0   # 0 at threshold
    return 1.0 / (1.0 + math.exp(-2.0 * x))


def _weighted_score(dimensions: dict[str, float]) -> float:
    """Compute weighted mean using DIMENSION_WEIGHTS."""
    total = 0.0
    for key, weight in DIMENSION_WEIGHTS.items():
        total += dimensions.get(key, 0.0) * weight
    return min(max(total, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Flag generation
# ---------------------------------------------------------------------------


def _generate_flags(
    segment_scores: dict[int, SegmentConfidence],
    topology: RoadTopology,
    geometry: RoadGeometry,
) -> list[ConfidenceFlag]:
    """
    Generate ConfidenceFlags for known anomaly patterns.
    """
    flags: list[ConfidenceFlag] = []

    # -- Low-confidence segment warnings
    for sid, sc in segment_scores.items():
        if sc.overall < 0.3:
            flags.append(ConfidenceFlag(
                severity="warning",
                kind="LOW_CONFIDENCE_SEGMENT",
                message=f"Segment {sid} has very low confidence ({sc.overall:.2f}). "
                        "Possible noise or occlusion artefact.",
                segment_id=sid,
            ))
        if sc.mask_coverage < 0.5:
            flags.append(ConfidenceFlag(
                severity="info",
                kind="LOW_MASK_COVERAGE",
                message=f"Segment {sid} has low mask coverage ({sc.mask_coverage:.2f}). "
                        "Segment may not be fully supported by the repaired mask.",
                segment_id=sid,
            ))

    # -- Disconnected segments (both endpoints unconnected)
    if hasattr(topology, "disconnected_segs"):
        for sid in topology.disconnected_segs:
            flags.append(ConfidenceFlag(
                severity="warning",
                kind="DISCONNECTED_SEGMENT",
                message=f"Segment {sid} has no connections to the road network. "
                        "Isolated fragment — may be noise or occluded road.",
                segment_id=sid,
            ))

    # -- Rejected bridge candidates with high scores (near-misses worth inspecting)
    if hasattr(topology, "rejected"):
        near_misses = [
            c for c in topology.rejected
            if c.score >= 0.45
        ]
        for conn in near_misses[:5]:  # cap at 5 to keep flags readable
            flags.append(ConfidenceFlag(
                severity="info",
                kind="NEAR_MISS_CONNECTION",
                message=(
                    f"Segments {conn.seg_a}↔{conn.seg_b} scored {conn.score:.3f} "
                    f"(gap={conn.gap_px:.1f}px) — just below connection threshold. "
                    "Consider lowering TopologyConfig.connection_threshold."
                ),
            ))

    # -- Isolated components (disconnected road islands)
    if topology.components:
        for comp_idx, comp in enumerate(topology.components):
            if len(comp) < 3:
                flags.append(ConfidenceFlag(
                    severity="info",
                    kind="ISOLATED_COMPONENT",
                    message=f"Connected component #{comp_idx} has only {len(comp)} node(s). "
                            "May be an isolated road fragment.",
                ))

    # -- High curvature segments
    for seg in geometry.segments:
        if seg.curvature > MAX_CURVATURE * 0.8:
            flags.append(ConfidenceFlag(
                severity="info",
                kind="HIGH_CURVATURE",
                message=f"Segment {seg.segment_id} has high curvature "
                        f"({seg.curvature:.4f} rad/px). "
                        "May indicate noise or a sharp turn.",
                segment_id=seg.segment_id,
            ))

    return flags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_topology(
    rgb_image: np.ndarray,
    repaired_mask: np.ndarray,
    geometry: RoadGeometry,
    topology: RoadTopology,
) -> ConfidenceReport:
    """
    Main entry point for the Confidence module.

    Parameters
    ----------
    rgb_image : np.ndarray
        Original RGB image, shape (H, W, 3), dtype uint8.
    repaired_mask : np.ndarray
        Binary road mask, shape (H, W), dtype uint8.
        Output of road_extractor.postprocess.repair_road_mask().
    geometry : RoadGeometry
        Output of geometry.extract_geometry().
    topology : RoadTopology
        Output of topology.build_topology().

    Returns
    -------
    ConfidenceReport
        Per-segment scores, network-level score, and diagnostic flags.
    """

    segment_scores: dict[int, SegmentConfidence] = {}

    for seg in geometry.segments:
        # ---------------------------------------------------------
        # Score each dimension independently
        # ---------------------------------------------------------
        dims: dict[str, float] = {
            "mask_coverage":  _score_mask_coverage(seg, repaired_mask),
            "color_contrast": _score_color_contrast(seg, rgb_image),
            "straightness":   _score_straightness(seg),
            "connectivity":   _score_connectivity(seg, topology),
            "length_score":   _score_length(seg),
        }

        overall = _weighted_score(dims)
        notes: list[str] = []

        if dims["mask_coverage"] < 0.5:
            notes.append("low mask coverage")
        if dims["connectivity"] == 0.0:
            notes.append("isolated segment")
        if dims["length_score"] < 0.4:
            notes.append("very short fragment")

        segment_scores[seg.segment_id] = SegmentConfidence(
            segment_id=seg.segment_id,
            overall=overall,
            **dims,
            notes=notes,
        )

    # ---------------------------------------------------------
    # Network-level score: mean of all segment scores
    # ---------------------------------------------------------
    if segment_scores:
        network_score = float(np.mean([sc.overall for sc in segment_scores.values()]))
    else:
        network_score = 0.0

    # ---------------------------------------------------------
    # Generate diagnostic flags
    # ---------------------------------------------------------
    flags = _generate_flags(segment_scores, topology, geometry)

    metadata = {
        "segment_count": len(segment_scores),
        "network_score": round(network_score, 4),
        "dimension_weights": DIMENSION_WEIGHTS,
        "thresholds": {
            "min_segment_length_px": MIN_SEGMENT_LENGTH,
            "max_curvature_rad_per_px": MAX_CURVATURE,
        },
    }

    return ConfidenceReport(
        segment_scores=segment_scores,
        network_score=network_score,
        flags=flags,
        metadata=metadata,
    )
