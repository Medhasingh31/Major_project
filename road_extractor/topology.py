"""
topology.py — Road Topology Module
====================================
Independent layer over RoadGeometry. Does NOT touch the U-Net, its weights,
training code, or inference code.

INPUT
-----
  geometry     : RoadGeometry  — output of geometry.extract_geometry()

DESIGN PHILOSOPHY
-----------------
Topology is a structural interpretation of Geometry. This foundation phase
does not infer missing roads, does not use masks as evidence, and does not
invent ground-truth topology. Nodes come from geometry endpoints and
intersections; edges come from geometry road segments.

DETECTED TOPOLOGY ELEMENTS
---------------------------
  • road endpoints       — degree-1 nodes (dead-ends / road tips)
  • intersections        — degree ≥ 3 nodes, classified T / X / Y / complex
  • valid connections    — geometry segments between endpoint/intersection nodes
  • disconnected segs    — segments whose both endpoints remain unconnected

OUTPUT
------
  RoadTopology dataclass:
    .graph              nx.Graph   nodes=endpoints/junctions, edges=connections
    .intersections      list[Intersection]
    .continuity_links   list[ContinuityLink]
    .connections        list[Connection]   — all accepted endpoint bridges
    .rejected           list[Connection]   — evaluated but below threshold
    .disconnected_segs  list[int]          — segment_ids with no connections
    .components         list[list[int]]    — connected component node groups
    .metadata           dict

PIPELINE POSITION
-----------------
  RoadGeometry ──► topology.build_topology() ──► RoadTopology
                                                                        │
                                                                        ▼
                                                               confidence.score_topology()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json

import cv2
import numpy as np

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False

from road_extractor.geometry import RoadGeometry, RoadSegment


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TopologyConfig:
    """
    All tunable thresholds for the topology pipeline.
    Pass a custom instance to build_topology() to override defaults.
    """

    # ── candidate search ──────────────────────────────────────────────────
    max_gap_px: float = 18.0
    """Maximum Euclidean distance (pixels) between two endpoints to be
    considered as a connection candidate at all.  Pairs further apart are
    never evaluated, saving compute and preventing absurd bridges."""

    # ── per-signal weights (must sum to 1.0) ──────────────────────────────
    w_endpoint_distance:     float = 0.20
    w_direction_compat:      float = 0.20
    w_approach_align:        float = 0.15
    w_mask_evidence:         float = 0.20
    w_width_consistency:     float = 0.10
    w_centerline_continuity: float = 0.10
    w_gap_penalty:           float = 0.05

    # ── decision threshold ─────────────────────────────────────────────────
    connection_threshold: float = 0.55
    """Minimum weighted score for an endpoint pair to become a graph edge."""

    # ── signal tuning ──────────────────────────────────────────────────────
    direction_tolerance_deg: float = 35.0
    """Orientation diff above which direction_compat begins to fall.
    At 2× this value the score reaches 0."""

    approach_tail_px: int = 8
    """Number of pixels from the segment end used to estimate the local
    approach direction (the 'tail vector')."""

    mask_probe_samples: int = 20
    """Number of equally-spaced samples along the straight-line probe
    between two endpoints when computing mask_evidence."""

    max_width_ratio: float = 3.0
    """Width ratio above which width_consistency = 0."""

    continuity_bend_limit_deg: float = 60.0
    """Predicted bend angle above which centerline_continuity = 0."""

    # ── continuity links (post-graph, within a shared node) ───────────────
    continuity_angle_tolerance_deg: float = 30.0
    """Max direction difference for two segments sharing a node to be
    labelled a 'continuity link' (same physical road through junction)."""

    node_snap_radius_px: float = 8.0
    """Maximum distance for snapping a segment terminal to a geometry node."""

    suspicious_short_segment_px: float = 12.0
    """Flag isolated segments shorter than this as suspicious fragments."""

    min_topology_edge_length_px: float = 3.0
    """Reject only sub-pixel-contact-sized graph edges."""

    tiny_edge_mask_support: float = 0.80
    """Support required to retain an exceptionally short centerline edge."""

    min_geometry_bridge_score: float = 0.74
    """Minimum composite score for a geometry-only endpoint bridge."""

    min_geometry_mask_support: float = 0.70
    """Minimum clean-mask support along a proposed bridge."""

    min_geometry_approach_alignment: float = 0.72
    """Minimum opposing endpoint tangent alignment."""

    min_geometry_continuity_score: float = 0.62
    """Minimum local bend/continuity score for a proposed bridge."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Connection:
    """
    A scored candidate connection between two segment endpoints.

    Accepted connections (score ≥ threshold) become edges in the graph.
    Rejected ones are stored in RoadTopology.rejected for inspection.
    """
    seg_a: int              # segment_id of first segment
    seg_b: int              # segment_id of second segment
    end_a: tuple[int, int]  # (y, x) endpoint on seg_a
    end_b: tuple[int, int]  # (y, x) endpoint on seg_b

    # composite score and per-signal breakdown
    score: float = 0.0
    signals: dict[str, float] = field(default_factory=dict)

    # bookkeeping
    gap_px: float = 0.0
    accepted: bool = False

    # node IDs assigned after graph construction (-1 = not yet assigned)
    node_a: int = -1
    node_b: int = -1

    # Phase 13 decision diagnostics. These are topology-summary metadata;
    # graph/export schemas are unchanged.
    evidence_count: int = 0
    evidence_required: int = 0
    decision_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "seg_a": self.seg_a,
            "seg_b": self.seg_b,
            "end_a": list(self.end_a),
            "end_b": list(self.end_b),
            "score": round(self.score, 4),
            "gap_px": round(self.gap_px, 2),
            "accepted": self.accepted,
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
            "diagnostics": {
                "evidence_count": self.evidence_count,
                "evidence_required": self.evidence_required,
                "reasons": list(self.decision_reasons),
            },
        }


@dataclass
class Intersection:
    """A node in the topology graph where ≥ 3 road segments meet."""
    location: tuple[int, int]
    degree: int
    kind: str               # 'T' | 'X' | 'Y' | 'complex' | 'endpoint' | 'passthrough'
    segment_ids: list[int] = field(default_factory=list)
    node_id: int = -1

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "location": list(self.location),
            "degree": self.degree,
            "kind": self.kind,
            "segment_ids": self.segment_ids,
        }


@dataclass
class ContinuityLink:
    """
    Two segments that share a node AND whose orientations are within
    continuity_angle_tolerance_deg — i.e. likely the same physical road.
    """
    segment_a: int
    segment_b: int
    shared_node: int
    angle_diff_deg: float

    def to_dict(self) -> dict:
        return {
            "segment_a": self.segment_a,
            "segment_b": self.segment_b,
            "shared_node": self.shared_node,
            "angle_diff_deg": round(self.angle_diff_deg, 2),
        }


@dataclass
class RoadTopology:
    """Container for all topological information derived from RoadGeometry."""

    graph: Optional[object] = field(default=None, repr=False)  # nx.Graph | None

    intersections:    list[Intersection]    = field(default_factory=list)
    continuity_links: list[ContinuityLink]  = field(default_factory=list)
    connections:      list[Connection]      = field(default_factory=list)  # accepted
    rejected:         list[Connection]      = field(default_factory=list)  # below threshold
    disconnected_segs: list[int]            = field(default_factory=list)  # no connections
    components:       list[list[int]]       = field(default_factory=list)

    metadata: dict = field(default_factory=dict)

    # ── accessors ───────────────────────────────────────────────────────────

    def node_count(self) -> int:
        return self.graph.number_of_nodes() if self.graph is not None else 0

    def edge_count(self) -> int:
        return self.graph.number_of_edges() if self.graph is not None else 0

    def component_count(self) -> int:
        return len(self.components)

    def to_summary(self) -> dict:
        return {
            "node_count":             self.node_count(),
            "edge_count":             self.edge_count(),
            "component_count":        self.component_count(),
            "intersection_count":     len(self.intersections),
            "continuity_link_count":  len(self.continuity_links),
            "accepted_connections":   len(self.connections),
            "rejected_connections":   len(self.rejected),
            "disconnected_segments":  self.disconnected_segs,
            "intersections":          [i.to_dict() for i in self.intersections],
            "continuity_links":       [c.to_dict() for c in self.continuity_links],
            "connections":            [c.to_dict() for c in self.connections],
            **self.metadata,
        }


# ---------------------------------------------------------------------------
# Signal 1 — Endpoint distance score
# ---------------------------------------------------------------------------

def _sig_endpoint_distance(gap_px: float, max_gap: float) -> float:
    """
    Smooth score that is 1.0 at gap=0 and decays to 0.0 at gap=max_gap.
    Uses a cosine taper so the decay is gradual near 0 and accelerates
    toward the hard cut-off, avoiding a cliff at the threshold.

        score = 0.5 × (1 + cos(π × gap / max_gap))   for gap ≤ max_gap
                0.0                                    for gap > max_gap
    """
    if gap_px <= 0.0:
        return 1.0
    if gap_px >= max_gap:
        return 0.0
    return 0.5 * (1.0 + math.cos(math.pi * gap_px / max_gap))


# ---------------------------------------------------------------------------
# Signal 2 — Direction compatibility
# ---------------------------------------------------------------------------

def _undirected_angle_diff(a_deg: float, b_deg: float) -> float:
    """Minimum angle between two undirected orientations, result in [0, 90]."""
    diff = abs(a_deg - b_deg) % 180.0
    return min(diff, 180.0 - diff)


def _sig_direction_compat(
    seg_a: RoadSegment,
    seg_b: RoadSegment,
    tolerance_deg: float,
) -> float:
    """
    Score based on how similar the two segments' dominant orientations are.
    At diff=0 → 1.0.  At diff=tolerance_deg → 0.5.  At diff=2×tolerance → 0.0.
    Linear interpolation between these points.
    """
    diff = _undirected_angle_diff(seg_a.direction_deg, seg_b.direction_deg)
    if diff <= tolerance_deg:
        # linear from 1.0 at 0 to 0.5 at tolerance
        return 1.0 - 0.5 * (diff / tolerance_deg)
    elif diff <= 2.0 * tolerance_deg:
        # linear from 0.5 at tolerance to 0.0 at 2×tolerance
        return 0.5 * (1.0 - (diff - tolerance_deg) / tolerance_deg)
    return 0.0


# ---------------------------------------------------------------------------
# Signal 3 — Approach alignment
# ---------------------------------------------------------------------------

def _tail_vector(
    path: list[tuple[int, int]],
    end_is_last: bool,
    tail_px: int,
) -> tuple[float, float]:
    """
    Compute the direction vector of the 'tail' of a segment — the last
    (or first) *tail_px* pixels, pointing away from the segment body.

    end_is_last=True  → tail is at path[-1], vector points outward (path end)
    end_is_last=False → tail is at path[0],  vector points outward (path start)

    Returns a unit (dy, dx) vector.  (0, 1) fallback if path is degenerate.
    """
    n = min(tail_px, len(path) - 1)
    if n < 1:
        return (0.0, 1.0)

    if end_is_last:
        p0 = path[-(n + 1)]
        p1 = path[-1]
    else:
        p0 = path[n]
        p1 = path[0]

    dy = float(p1[0] - p0[0])
    dx = float(p1[1] - p0[1])
    mag = math.sqrt(dy * dy + dx * dx)
    if mag < 1e-6:
        return (0.0, 1.0)
    return (dy / mag, dx / mag)


def _sig_approach_align(
    seg_a: RoadSegment,
    end_a: tuple[int, int],
    seg_b: RoadSegment,
    end_b: tuple[int, int],
    tail_px: int,
) -> float:
    """
    Do the two segment tails actually point toward each other?

    For a clean connection the tail of A should roughly face end_b, and the
    tail of B should roughly face end_a.

    The bridge direction vector is (end_b - end_a).  We compute the cosine
    similarity between each tail vector and the bridge direction, then
    average them and clamp to [0, 1].

    cos = 1.0  → tail points perfectly toward the other endpoint (ideal)
    cos = 0.0  → perpendicular
    cos < 0    → tail points away (anti-aligned) → score 0.0
    """
    # Bridge direction: from end_a toward end_b
    bdy = float(end_b[0] - end_a[0])
    bdx = float(end_b[1] - end_a[1])
    bmag = math.sqrt(bdy * bdy + bdx * bdx)
    if bmag < 1e-6:
        return 1.0   # same pixel → trivially aligned

    bdy /= bmag
    bdx /= bmag

    # Tail of seg_a at end_a — pointing outward from the segment
    a_is_last = (end_a == seg_a.pixel_path[-1])
    tay, tax = _tail_vector(seg_a.pixel_path, end_is_last=a_is_last, tail_px=tail_px)

    # Tail of seg_b at end_b — pointing outward, so flip bridge direction
    b_is_last = (end_b == seg_b.pixel_path[-1])
    tby, tbx = _tail_vector(seg_b.pixel_path, end_is_last=b_is_last, tail_px=tail_px)

    cos_a = tay * bdy + tax * bdx          # tail_a · bridge
    cos_b = -(tby * bdy + tbx * bdx)       # tail_b · (-bridge) = tail_b faces A

    score = (max(cos_a, 0.0) + max(cos_b, 0.0)) / 2.0
    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Signal 4 — Mask evidence along the gap
# ---------------------------------------------------------------------------

def _sig_mask_evidence(
    end_a: tuple[int, int],
    end_b: tuple[int, int],
    repaired_mask: np.ndarray,
    n_samples: int,
) -> float:
    """
    Sample the repaired_mask along the straight-line probe between end_a and
    end_b.  Returns the fraction of positive (road) pixels along the probe.

    A high fraction means the segmentation mask already shows road material
    bridging the gap — strong evidence for a real connection.
    A low fraction means the gap is genuinely empty in the mask.
    """
    h, w = repaired_mask.shape
    hits = 0
    total = 0
    for i in range(n_samples + 1):
        t = i / max(n_samples, 1)
        y = int(round(end_a[0] + t * (end_b[0] - end_a[0])))
        x = int(round(end_a[1] + t * (end_b[1] - end_a[1])))
        if 0 <= y < h and 0 <= x < w:
            total += 1
            if repaired_mask[y, x] > 0:
                hits += 1
    return hits / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Signal 5 — Width consistency
# ---------------------------------------------------------------------------

def _sig_width_consistency(seg_a: RoadSegment, seg_b: RoadSegment, max_ratio: float) -> float:
    """
    Roads of very different widths are unlikely to be the same road.

    Score = 1 − (ratio − 1) / (max_ratio − 1)  where ratio = max/min.
    ratio=1 → 1.0;  ratio=max_ratio → 0.0;  clamped to [0, 1].
    """
    wa = max(seg_a.width_pixels, 1.0)
    wb = max(seg_b.width_pixels, 1.0)
    ratio = max(wa, wb) / min(wa, wb)
    if ratio >= max_ratio:
        return 0.0
    return 1.0 - (ratio - 1.0) / (max_ratio - 1.0)


# ---------------------------------------------------------------------------
# Signal 6 — Centerline continuity
# ---------------------------------------------------------------------------

def _sig_centerline_continuity(
    seg_a: RoadSegment,
    end_a: tuple[int, int],
    seg_b: RoadSegment,
    end_b: tuple[int, int],
    tail_px: int,
    bend_limit_deg: float,
) -> float:
    """
    Estimate the bend angle introduced by bridging end_a to end_b.

    For each segment we compute the local exit angle at the relevant endpoint
    (the angle of the tail vector).  The bridge introduces a kink equal to
    the angle between the tail's outward direction and the bridge vector.
    We average the kink from both sides.

        kink_a = angle between (outward tail of A) and (bridge direction)
        kink_b = angle between (outward tail of B) and (-bridge direction)
        mean_kink = (kink_a + kink_b) / 2

    score = 1 − mean_kink / bend_limit_deg,  clamped to [0, 1].
    """
    bdy = float(end_b[0] - end_a[0])
    bdx = float(end_b[1] - end_a[1])
    bmag = math.sqrt(bdy * bdy + bdx * bdx)
    if bmag < 1e-6:
        return 1.0

    bdy /= bmag
    bdx /= bmag

    a_is_last = (end_a == seg_a.pixel_path[-1])
    b_is_last = (end_b == seg_b.pixel_path[-1])

    tay, tax = _tail_vector(seg_a.pixel_path, end_is_last=a_is_last, tail_px=tail_px)
    tby, tbx = _tail_vector(seg_b.pixel_path, end_is_last=b_is_last, tail_px=tail_px)

    # cos between tail_a and bridge direction
    cos_a = max(-1.0, min(1.0, tay * bdy + tax * bdx))
    # cos between tail_b and reverse bridge direction
    cos_b = max(-1.0, min(1.0, -(tby * bdy + tbx * bdx)))

    kink_a = math.degrees(math.acos(cos_a))
    kink_b = math.degrees(math.acos(cos_b))
    mean_kink = (kink_a + kink_b) / 2.0

    return max(0.0, 1.0 - mean_kink / bend_limit_deg)


# ---------------------------------------------------------------------------
# Signal 7 — Gap penalty relative to road width
# ---------------------------------------------------------------------------

def _sig_gap_penalty(
    gap_px: float,
    seg_a: RoadSegment,
    seg_b: RoadSegment,
    max_gap: float,
) -> float:
    """
    A gap larger than a few road-widths is physically implausible.
    This signal normalises gap_px against mean road width.

        ratio  = gap_px / mean_width
        score  = max(0, 1 − ratio / (max_gap / mean_width))

    Short gaps relative to road width → 1.0.
    Gaps beyond max_gap → 0.0.
    """
    mean_width = (seg_a.width_pixels + seg_b.width_pixels) / 2.0
    mean_width = max(mean_width, 1.0)
    relative_gap = gap_px / mean_width
    max_relative = max_gap / mean_width
    return max(0.0, 1.0 - relative_gap / max_relative)


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def _score_candidate(
    seg_a: RoadSegment,
    end_a: tuple[int, int],
    seg_b: RoadSegment,
    end_b: tuple[int, int],
    repaired_mask: np.ndarray,
    cfg: TopologyConfig,
) -> Connection:
    """
    Compute all seven signals and their weighted composite for one
    endpoint-pair candidate.  Returns a Connection (not yet accepted/rejected).
    """
    gap = math.sqrt(
        (end_b[0] - end_a[0]) ** 2 + (end_b[1] - end_a[1]) ** 2
    )

    s1 = _sig_endpoint_distance(gap, cfg.max_gap_px)
    s2 = _sig_direction_compat(seg_a, seg_b, cfg.direction_tolerance_deg)
    s3 = _sig_approach_align(seg_a, end_a, seg_b, end_b, cfg.approach_tail_px)
    s4 = _sig_mask_evidence(end_a, end_b, repaired_mask, cfg.mask_probe_samples)
    s5 = _sig_width_consistency(seg_a, seg_b, cfg.max_width_ratio)
    s6 = _sig_centerline_continuity(
        seg_a, end_a, seg_b, end_b,
        cfg.approach_tail_px, cfg.continuity_bend_limit_deg,
    )
    s7 = _sig_gap_penalty(gap, seg_a, seg_b, cfg.max_gap_px)

    signals = {
        "endpoint_distance":     s1,
        "direction_compat":      s2,
        "approach_align":        s3,
        "mask_evidence":         s4,
        "width_consistency":     s5,
        "centerline_continuity": s6,
        "gap_penalty":           s7,
    }

    score = (
        cfg.w_endpoint_distance     * s1 +
        cfg.w_direction_compat      * s2 +
        cfg.w_approach_align        * s3 +
        cfg.w_mask_evidence         * s4 +
        cfg.w_width_consistency     * s5 +
        cfg.w_centerline_continuity * s6 +
        cfg.w_gap_penalty           * s7
    )

    return Connection(
        seg_a=seg_a.segment_id,
        seg_b=seg_b.segment_id,
        end_a=end_a,
        end_b=end_b,
        score=score,
        signals=signals,
        gap_px=gap,
        accepted=False,
    )


# ---------------------------------------------------------------------------
# Endpoint enumeration
# ---------------------------------------------------------------------------

def _segment_endpoints(seg: RoadSegment) -> list[tuple[int, int]]:
    """
    Return the two physical endpoints of a segment (start and end of path).
    If path has only one pixel, return it twice.
    """
    if len(seg.pixel_path) < 2:
        return [seg.pixel_path[0], seg.pixel_path[0]]
    return [seg.pixel_path[0], seg.pixel_path[-1]]


def _point_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    dy = float(a[0] - b[0])
    dx = float(a[1] - b[1])
    return math.sqrt(dy * dy + dx * dx)


def _nearest_node(
    point: tuple[int, int],
    node_points: dict[int, tuple[int, int]],
    max_distance: float,
) -> tuple[Optional[int], float]:
    best_node: Optional[int] = None
    best_distance = float("inf")
    for node_id, node_point in node_points.items():
        distance = _point_distance(point, node_point)
        if distance < best_distance:
            best_node = node_id
            best_distance = distance
    if best_node is None or best_distance > max_distance:
        return None, best_distance
    return best_node, best_distance


def _add_geometry_node(
    graph: "nx.Graph",
    node_points: dict[int, tuple[int, int]],
    point: tuple[int, int],
    kind: str,
    node_id: int,
) -> int:
    y, x = point
    graph.add_node(node_id, y=int(y), x=int(x), kind=kind)
    node_points[node_id] = (int(y), int(x))
    return node_id + 1


def _build_geometry_graph(
    geometry: RoadGeometry,
    cfg: TopologyConfig,
) -> tuple["nx.Graph", dict[tuple[int, int], int], list[int]]:
    """
    Build a NetworkX graph strictly from RoadGeometry.

    Nodes are geometry endpoints and junction/intersection representatives.
    Edges are geometry segments snapped to those nodes. No inferred bridge
    edges are created here.
    """
    if not _HAS_NX:
        raise ImportError(
            "networkx is required for topology.build_topology(). "
            "Install it with: pip install networkx"
        )

    graph = nx.Graph()
    graph.graph["duplicate_edges_removed"] = 0
    graph.graph["tiny_edges_removed"] = 0
    graph.graph["self_loop_edges_removed"] = 0
    node_points: dict[int, tuple[int, int]] = {}
    pixel_to_node: dict[tuple[int, int], int] = {}
    next_node_id = 0

    for point in sorted(set(geometry.endpoints)):
        next_node_id = _add_geometry_node(
            graph,
            node_points,
            point,
            kind="endpoint",
            node_id=next_node_id,
        )
        pixel_to_node[point] = next_node_id - 1

    for point in sorted(set(geometry.junctions)):
        if point in pixel_to_node:
            graph.nodes[pixel_to_node[point]]["kind"] = "junction"
            continue
        next_node_id = _add_geometry_node(
            graph,
            node_points,
            point,
            kind="junction",
            node_id=next_node_id,
        )
        pixel_to_node[point] = next_node_id - 1

    suspicious_segments: list[int] = []

    for seg in geometry.segments:
        if len(seg.pixel_path) < 2:
            suspicious_segments.append(seg.segment_id)
            continue

        if seg.length_pixels < cfg.min_topology_edge_length_px:
            support = 0.0
            if geometry.clean_mask is not None:
                support = float(np.mean([
                    geometry.clean_mask[y, x] > 0
                    for y, x in seg.pixel_path
                    if 0 <= y < geometry.clean_mask.shape[0]
                    and 0 <= x < geometry.clean_mask.shape[1]
                ]))
            if support < cfg.tiny_edge_mask_support:
                suspicious_segments.append(seg.segment_id)
                graph.graph["tiny_edges_removed"] += 1
                continue

        start_px, end_px = _segment_endpoints(seg)
        start_node, start_snap = _nearest_node(
            start_px,
            node_points,
            cfg.node_snap_radius_px,
        )
        end_node, end_snap = _nearest_node(
            end_px,
            node_points,
            cfg.node_snap_radius_px,
        )

        if start_node is None:
            next_node_id = _add_geometry_node(
                graph,
                node_points,
                start_px,
                kind="endpoint",
                node_id=next_node_id,
            )
            start_node = next_node_id - 1
            pixel_to_node[start_px] = start_node
            start_snap = 0.0

        if end_node is None:
            next_node_id = _add_geometry_node(
                graph,
                node_points,
                end_px,
                kind="endpoint",
                node_id=next_node_id,
            )
            end_node = next_node_id - 1
            pixel_to_node[end_px] = end_node
            end_snap = 0.0

        if start_node == end_node:
            suspicious_segments.append(seg.segment_id)
            graph.graph["self_loop_edges_removed"] += 1
            continue

        edge_data = {
            "segment_id": seg.segment_id,
            "length_pixels": seg.length_pixels,
            "direction_deg": seg.direction_deg,
            "curvature": seg.curvature,
            "width_pixels": seg.width_pixels,
            "gap_px": 0.0,
            "connection_score": 1.0,
            "edge_kind": "geometry_segment",
            "start_snap_px": round(float(start_snap), 3),
            "end_snap_px": round(float(end_snap), 3),
        }
        if graph.has_edge(start_node, end_node):
            existing = graph.edges[start_node, end_node]
            existing_quality = (
                float(existing.get("length_pixels", 0.0)),
                float(existing.get("width_pixels", 0.0)),
            )
            new_quality = (float(seg.length_pixels), float(seg.width_pixels))
            suspicious_segments.append(seg.segment_id)
            graph.graph["duplicate_edges_removed"] += 1
            if new_quality > existing_quality:
                old_segment = int(existing.get("segment_id", -1))
                if old_segment >= 0:
                    suspicious_segments.append(old_segment)
                graph.remove_edge(start_node, end_node)
            else:
                continue

        graph.add_edge(
            start_node,
            end_node,
            **edge_data,
        )

    return graph, pixel_to_node, sorted(set(suspicious_segments))


def _add_conservative_geometry_connections(
    graph: "nx.Graph",
    geometry: RoadGeometry,
    cfg: TopologyConfig,
) -> tuple[list[Connection], list[Connection]]:
    """Add only well-supported bridges between currently terminal nodes.

    This uses Geometry's cleaned mask as the nearby-road evidence source. It
    never joins a junction node, never joins two endpoints from one segment,
    and greedily accepts each endpoint at most once.
    """
    if geometry.clean_mask is None:
        return [], []

    endpoint_records: list[tuple[RoadSegment, tuple[int, int], int]] = []
    for segment in geometry.segments:
        for endpoint in _segment_endpoints(segment):
            candidates = [
                node_id
                for node_id, data in graph.nodes(data=True)
                if data.get("kind") == "endpoint"
                and _point_distance(endpoint, (data["y"], data["x"])) <= cfg.node_snap_radius_px
            ]
            if not candidates:
                continue
            node_id = min(
                candidates,
                key=lambda candidate: _point_distance(
                    endpoint,
                    (graph.nodes[candidate]["y"], graph.nodes[candidate]["x"]),
                ),
            )
            if graph.degree(node_id) == 1:
                endpoint_records.append((segment, endpoint, int(node_id)))

    candidates: list[tuple[Connection, int, int]] = []
    rejected: list[Connection] = []
    for index, (seg_a, end_a, node_a) in enumerate(endpoint_records):
        for seg_b, end_b, node_b in endpoint_records[index + 1:]:
            if seg_a.segment_id == seg_b.segment_id or node_a == node_b:
                continue
            gap = _point_distance(end_a, end_b)
            if gap < 1.5 or gap > cfg.max_gap_px:
                continue
            candidate = _score_candidate(
                seg_a,
                end_a,
                seg_b,
                end_b,
                geometry.clean_mask,
                cfg,
            )
            accepted_by_evidence, reasons, evidence_count, evidence_required = _evaluate_multi_evidence(
                candidate, cfg
            )
            candidate.evidence_count = evidence_count
            candidate.evidence_required = evidence_required
            candidate.decision_reasons = reasons
            if not accepted_by_evidence:
                rejected.append(candidate)
                continue
            candidates.append((candidate, node_a, node_b))

    candidates.sort(key=lambda item: item[0].score, reverse=True)
    accepted: list[Connection] = []
    matched_nodes: set[int] = set()
    for candidate, node_a, node_b in candidates:
        if node_a in matched_nodes or node_b in matched_nodes:
            candidate.decision_reasons = [
                "rejected after ranking because one endpoint was already matched"
            ]
            rejected.append(candidate)
            continue
        if graph.has_edge(node_a, node_b):
            candidate.decision_reasons = [
                "rejected because the graph already contains this edge"
            ]
            rejected.append(candidate)
            continue

        candidate.accepted = True
        if not candidate.decision_reasons:
            candidate.decision_reasons = ["accepted by multi-evidence gate"]
        candidate.node_a = node_a
        candidate.node_b = node_b
        matched_nodes.update((node_a, node_b))
        graph.add_edge(
            node_a,
            node_b,
            segment_id=-1,
            length_pixels=candidate.gap_px,
            direction_deg=-1.0,
            curvature=0.0,
            width_pixels=0.0,
            gap_px=candidate.gap_px,
            connection_score=candidate.score,
            edge_kind="geometry_bridge",
            pixels=json.dumps([
                [int(candidate.end_a[1]), int(candidate.end_a[0])],
                [int(candidate.end_b[1]), int(candidate.end_b[0])],
            ]),
        )
        accepted.append(candidate)

    return accepted, rejected


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_scored_graph(
    geometry: RoadGeometry,
    repaired_mask: np.ndarray,
    cfg: TopologyConfig,
) -> tuple["nx.Graph", list[Connection], list[Connection], dict]:
    """
    1. Enumerate all endpoint-pairs within max_gap_px.
    2. Score each pair with the seven signals.
    3. Accept pairs that pass connection_threshold.
    4. Build a NetworkX graph:
         • Nodes for every unique endpoint / junction pixel.
         • Skeleton edges (from geometry, zero gap, score=1.0) always added.
         • Bridge edges (gap > 0) added only when accepted.

    Returns (graph, accepted_connections, rejected_connections, node_pixel_map).
    """
    if not _HAS_NX:
        raise ImportError(
            "networkx is required for topology.build_topology(). "
            "Install it with: pip install networkx"
        )

    G = nx.Graph()

    # ── collect all endpoint pixels ─────────────────────────────────────────
    # Each segment contributes two endpoints; junctions from geometry are also
    # added so skeleton-adjacent connections are always in the graph.
    endpoint_pixels: set[tuple[int, int]] = set()
    for seg in geometry.segments:
        for ep in _segment_endpoints(seg):
            endpoint_pixels.add(ep)
    for jy, jx in geometry.junctions:
        endpoint_pixels.add((int(jy), int(jx)))

    pixel_to_node: dict[tuple[int, int], int] = {}
    junction_set = set(geometry.junctions)

    for node_id, px in enumerate(sorted(endpoint_pixels)):
        kind = "junction" if px in junction_set else "endpoint"
        G.add_node(node_id, y=int(px[0]), x=int(px[1]), kind=kind)
        pixel_to_node[px] = node_id

    # ── skeleton edges (segments from geometry — already connected) ─────────
    for seg in geometry.segments:
        if len(seg.pixel_path) < 2:
            continue
        start_px = seg.pixel_path[0]
        end_px   = seg.pixel_path[-1]
        src = pixel_to_node.get(start_px)
        dst = pixel_to_node.get(end_px)
        if src is None or dst is None:
            continue
        # Self-loops are stored but flagged
        G.add_edge(
            src, dst,
            segment_id=seg.segment_id,
            length_pixels=seg.length_pixels,
            direction_deg=seg.direction_deg,
            curvature=seg.curvature,
            width_pixels=seg.width_pixels,
            gap_px=0.0,
            connection_score=1.0,
            edge_kind="skeleton",
        )

    # ── candidate bridge edges ───────────────────────────────────────────────
    segments = geometry.segments
    accepted: list[Connection] = []
    rejected: list[Connection] = []

    # Build a spatial index: map each segment to its endpoint pixels
    seg_endpoints: dict[int, list[tuple[int, int]]] = {
        seg.segment_id: _segment_endpoints(seg) for seg in segments
    }

    # Track which endpoints have already been matched to avoid duplicate bridges
    # (one endpoint should connect to at most one bridge partner)
    matched_endpoints: set[tuple[int, int]] = set()

    # Collect all candidates first, then greedily accept best-scoring ones
    all_candidates: list[Connection] = []

    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            seg_a = segments[i]
            seg_b = segments[j]

            for end_a in seg_endpoints[seg_a.segment_id]:
                for end_b in seg_endpoints[seg_b.segment_id]:
                    gap = math.sqrt(
                        (end_b[0] - end_a[0]) ** 2 +
                        (end_b[1] - end_a[1]) ** 2
                    )
                    # Skip the trivially-zero gap that is the skeleton edge
                    # (same pixel = already in the graph above)
                    if gap < 0.5:
                        continue
                    # Hard distance cut-off
                    if gap > cfg.max_gap_px:
                        continue

                    cand = _score_candidate(
                        seg_a, end_a, seg_b, end_b, repaired_mask, cfg
                    )
                    all_candidates.append(cand)

    # Sort by score descending, then greedily accept
    all_candidates.sort(key=lambda c: c.score, reverse=True)

    for cand in all_candidates:
        if cand.score < cfg.connection_threshold:
            cand.accepted = False
            rejected.append(cand)
            continue

        # Each endpoint can only be bridged once
        if cand.end_a in matched_endpoints or cand.end_b in matched_endpoints:
            cand.accepted = False
            rejected.append(cand)
            continue

        # Don't add a duplicate edge between the same node pair
        node_a = pixel_to_node.get(cand.end_a)
        node_b = pixel_to_node.get(cand.end_b)
        if node_a is None or node_b is None:
            rejected.append(cand)
            continue
        if G.has_edge(node_a, node_b):
            rejected.append(cand)
            continue

        # Accept
        matched_endpoints.add(cand.end_a)
        matched_endpoints.add(cand.end_b)
        cand.accepted = True
        cand.node_a = node_a
        cand.node_b = node_b

        G.add_edge(
            node_a, node_b,
            segment_id=-1,          # bridge has no source segment
            length_pixels=cand.gap_px,
            direction_deg=-1.0,
            curvature=0.0,
            width_pixels=0.0,
            gap_px=cand.gap_px,
            connection_score=cand.score,
            edge_kind="bridge",
        )
        accepted.append(cand)

    return G, accepted, rejected, pixel_to_node


# ---------------------------------------------------------------------------
# Intersection classification
# ---------------------------------------------------------------------------

def _classify_kind(degree: int) -> str:
    return {
        0: "isolated",
        1: "endpoint",
        2: "passthrough",
        3: "T",
        4: "X",
    }.get(degree, "complex")


def _detect_intersections(
    G: "nx.Graph",
    geometry: RoadGeometry,
) -> list[Intersection]:
    intersections: list[Intersection] = []
    for node_id, data in G.nodes(data=True):
        # A topology intersection must originate from Geometry's explicit
        # junction classification.  Do not promote an endpoint node merely
        # because a later bridge or incidental graph edge raises its degree.
        if data.get("kind") != "junction":
            continue
        geometry_edges = [
            edata
            for _, _, edata in G.edges(node_id, data=True)
            if edata.get("edge_kind") in {"geometry_segment", "skeleton"}
            and edata.get("segment_id", -1) >= 0
        ]
        if len(geometry_edges) < 3:
            continue
        degree = len(geometry_edges)
        kind = _classify_kind(degree)
        y, x = data["y"], data["x"]
        incident_seg_ids = [
            edata["segment_id"]
            for edata in geometry_edges
        ]
        intersections.append(
            Intersection(
                location=(y, x),
                degree=degree,
                kind=kind,
                segment_ids=incident_seg_ids,
                node_id=node_id,
            )
        )
    return intersections


# ---------------------------------------------------------------------------
# Continuity links
# ---------------------------------------------------------------------------

def _find_continuity_links(
    G: "nx.Graph",
    geometry: RoadGeometry,
    angle_tolerance_deg: float,
) -> list[ContinuityLink]:
    """
    At every node with degree ≥ 2, check all pairs of incident skeleton edges.
    If their segment direction angles are within tolerance, tag them as
    continuity links (same road continuing through the node).
    """
    links: list[ContinuityLink] = []
    seg_map: dict[int, RoadSegment] = {s.segment_id: s for s in geometry.segments}

    for node_id in G.nodes():
        incident = [
            (u, v, d)
            for u, v, d in G.edges(node_id, data=True)
            if d.get("edge_kind") in {"skeleton", "geometry_segment"}
            and d.get("segment_id", -1) >= 0
        ]
        if len(incident) < 2:
            continue

        for i in range(len(incident)):
            for j in range(i + 1, len(incident)):
                sid_i = incident[i][2]["segment_id"]
                sid_j = incident[j][2]["segment_id"]
                seg_i = seg_map.get(sid_i)
                seg_j = seg_map.get(sid_j)
                if seg_i is None or seg_j is None:
                    continue
                diff = _undirected_angle_diff(seg_i.direction_deg, seg_j.direction_deg)
                if diff <= angle_tolerance_deg:
                    links.append(
                        ContinuityLink(
                            segment_a=sid_i,
                            segment_b=sid_j,
                            shared_node=node_id,
                            angle_diff_deg=diff,
                        )
                    )
    return links


# ---------------------------------------------------------------------------
# Disconnected segment detection
# ---------------------------------------------------------------------------

def _find_disconnected_segments(
    geometry: RoadGeometry,
    G: "nx.Graph",
    pixel_to_node: dict[tuple[int, int], int],
) -> list[int]:
    """
    A segment is 'disconnected' if BOTH of its endpoint nodes have graph
    degree == 1 (i.e. only the skeleton edge connecting them to each other,
    with no bridges or junction connections on either side).

    These are isolated road fragments worth flagging.
    """
    isolated: list[int] = []
    component_by_node: dict[int, int] = {}
    for idx, component in enumerate(nx.connected_components(G)):
        for node_id in component:
            component_by_node[int(node_id)] = idx

    component_edge_counts: dict[int, int] = {}
    for u, v in G.edges():
        component_id = component_by_node.get(int(u))
        if component_id is None:
            continue
        component_edge_counts[component_id] = component_edge_counts.get(component_id, 0) + 1

    for u, v, data in G.edges(data=True):
        segment_id = data.get("segment_id", -1)
        if segment_id < 0:
            continue
        component_id = component_by_node.get(int(u))
        if component_id is not None and component_edge_counts.get(component_id, 0) == 1:
            isolated.append(int(segment_id))
    return isolated


def _evaluate_multi_evidence(
    candidate: Connection,
    cfg: TopologyConfig,
) -> tuple[bool, list[str], int, int]:
    """Require independent evidence channels before adding a bridge."""
    signals = candidate.signals
    strong = {
        "endpoint_distance": signals["endpoint_distance"] >= 0.85,
        "direction_compat": signals["direction_compat"] >= 0.80,
        "approach_align": signals["approach_align"] >= cfg.min_geometry_approach_alignment,
        "centerline_continuity": signals["centerline_continuity"] >= cfg.min_geometry_continuity_score,
        "mask_evidence": signals["mask_evidence"] >= cfg.min_geometry_mask_support,
        "width_consistency": signals["width_consistency"] >= 0.80,
    }
    geometric_names = (
        "endpoint_distance", "direction_compat", "approach_align",
        "centerline_continuity", "width_consistency",
    )
    geometry_count = sum(strong[name] for name in geometric_names)
    evidence_count = sum(strong.values())
    evidence_required = 5  # mask + four of five geometric signals
    reasons: list[str] = []
    if not strong["mask_evidence"]:
        reasons.append(
            f"mask support {signals['mask_evidence']:.2f} is below the unchanged "
            f"{cfg.min_geometry_mask_support:.2f} minimum"
        )
    if geometry_count < 4:
        weak = [name for name in geometric_names if not strong[name]]
        reasons.append(f"only {geometry_count}/5 geometric signals are strong; weak={','.join(weak)}")
    if not strong["approach_align"]:
        reasons.append("opposing approach-angle safety floor failed")
    if not strong["centerline_continuity"]:
        reasons.append("centerline bend/continuity safety floor failed")
    if candidate.score < cfg.min_geometry_bridge_score:
        reasons.append(
            f"composite score {candidate.score:.2f} is below {cfg.min_geometry_bridge_score:.2f}"
        )
    accepted = (
        candidate.score >= cfg.min_geometry_bridge_score
        and strong["mask_evidence"]
        and geometry_count >= 4
        and strong["approach_align"]
        and strong["centerline_continuity"]
    )
    if accepted:
        reasons = [
            f"accepted with {evidence_count}/{evidence_required} strong evidence channels",
            "mask support independently confirms the geometric continuation",
        ]
    return accepted, reasons, evidence_count, evidence_required


def _connectivity_diagnostics(
    graph: "nx.Graph",
    intersections: list[Intersection],
    rejected_connections: list[Connection],
    cfg: TopologyConfig,
) -> dict:
    """Summarize graph connectivity without forcing components together."""
    components = [sorted(int(node) for node in component) for component in nx.connected_components(graph)]
    very_short_edges: list[dict] = []
    suspicious_connections: list[dict] = []
    for u, v, data in graph.edges(data=True):
        length = float(data.get("length_pixels", 0.0))
        if length < cfg.suspicious_short_segment_px:
            very_short_edges.append({
                "source": int(u),
                "target": int(v),
                "length_pixels": round(length, 3),
                "edge_kind": data.get("edge_kind", "unknown"),
                "segment_id": int(data.get("segment_id", -1)),
            })
        if data.get("edge_kind") == "geometry_bridge":
            score = float(data.get("connection_score", 0.0))
            if score < cfg.min_geometry_bridge_score + 0.08:
                suspicious_connections.append({
                    "source": int(u),
                    "target": int(v),
                    "score": round(score, 4),
                    "gap_pixels": round(float(data.get("gap_px", 0.0)), 3),
                    "reason": "accepted bridge is close to the confidence floor",
                })

    suspicious_components = []
    for component_id, component in enumerate(components):
        edge_count = sum(
            1
            for u, v in graph.edges(component)
            if u in component and v in component
        )
        if edge_count == 0:
            suspicious_components.append({
                "component_id": component_id,
                "nodes": component,
                "edge_count": 0,
                "reason": "isolated node component",
            })

    return {
        "total_nodes": int(graph.number_of_nodes()),
        "total_edges": int(graph.number_of_edges()),
        "intersections": int(len(intersections)),
        "endpoints": int(sum(1 for node in graph.nodes() if graph.degree(node) == 1)),
        "connected_components": int(len(components)),
        "isolated_nodes": int(sum(1 for node in graph.nodes() if graph.degree(node) == 0)),
        "very_short_edges": very_short_edges,
        "very_short_edge_count": int(len(very_short_edges)),
        "suspicious_connections": suspicious_connections,
        "suspicious_connection_count": int(
            len(suspicious_connections) + len(rejected_connections)
        ),
        "rejected_connection_count": int(len(rejected_connections)),
        "suspicious_disconnected_components": suspicious_components,
        "separate_components_preserved": True,
    }


def _cleanup_graph_for_export(graph: "nx.Graph") -> dict:
    """Remove invalid graph artifacts while preserving real components."""
    removed_self_loops = 0
    for u, v in list(nx.selfloop_edges(graph)):
        graph.remove_edge(u, v)
        removed_self_loops += 1

    removed_isolated_nodes = 0
    for node_id in list(graph.nodes()):
        if graph.degree(node_id) == 0:
            graph.remove_node(node_id)
            removed_isolated_nodes += 1

    accidental_junctions: list[int] = []
    for node_id, data in graph.nodes(data=True):
        if data.get("kind") == "junction" and graph.degree(node_id) < 3:
            # Keep the segmentation point for edge bookkeeping, but do not
            # export it as an intersection/network event.
            data["kind"] = "passthrough"
            accidental_junctions.append(int(node_id))

    return {
        "isolated_nodes_removed": removed_isolated_nodes,
        "self_loop_edges_removed": removed_self_loops,
        "accidental_junctions_flagged": accidental_junctions,
        "duplicate_edges_removed": int(graph.graph.get("duplicate_edges_removed", 0)),
        "tiny_edges_removed": int(graph.graph.get("tiny_edges_removed", 0)),
        "real_components_preserved": True,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_topology(
    geometry: RoadGeometry,
    repaired_mask: Optional[np.ndarray] = None,
    config: Optional[TopologyConfig] = None,
) -> RoadTopology:
    """
    Main entry point for the Topology module.

    Builds a topology graph strictly from RoadGeometry. Nodes come from
    geometry endpoints and intersections; edges come from geometry road
    segments plus conservative bridges supported by Geometry's clean mask.
    The repaired_mask parameter is accepted for backward compatibility but is
    intentionally ignored in this phase.

    Parameters
    ----------
    geometry : RoadGeometry
        Output of geometry.extract_geometry().
    repaired_mask : np.ndarray
        Ignored. Topology foundation uses RoadGeometry as its only input.
    config : TopologyConfig | None
        Tuning parameters.  Uses defaults when None.

    Returns
    -------
    RoadTopology
    """
    cfg = config or TopologyConfig()

    if not _HAS_NX:
        return RoadTopology(
            metadata={"error": "networkx not installed; topology skipped"}
        )

    if not geometry.segments:
        return RoadTopology(
            metadata={"warning": "no segments in geometry; topology is empty"}
        )

    # ------------------------------------------------------------------
    # 1. Build the graph from geometry nodes and geometry segments only
    # ------------------------------------------------------------------
    G, pixel_to_node, suspicious = _build_geometry_graph(geometry, cfg)
    accepted_connections, rejected_connections = _add_conservative_geometry_connections(
        G, geometry, cfg
    )
    cleanup = _cleanup_graph_for_export(G)

    # ------------------------------------------------------------------
    # 2. Classify intersection nodes
    # ------------------------------------------------------------------
    intersections = _detect_intersections(G, geometry)

    # ------------------------------------------------------------------
    # 3. Detect directionally continuous segment pairs at shared nodes
    # ------------------------------------------------------------------
    continuity_links = _find_continuity_links(
        G, geometry, cfg.continuity_angle_tolerance_deg
    )

    # ------------------------------------------------------------------
    # 4. Find isolated (disconnected) segments
    # ------------------------------------------------------------------
    disconnected = sorted(
        set(_find_disconnected_segments(geometry, G, pixel_to_node)) | set(suspicious)
    )

    # ------------------------------------------------------------------
    # 5. Connected components
    # ------------------------------------------------------------------
    components = [
        sorted(c)
        for c in sorted(nx.connected_components(G), key=len, reverse=True)
    ]

    # ------------------------------------------------------------------
    # 6. Assemble metadata
    # ------------------------------------------------------------------
    signal_stats: dict[str, float] = {}
    endpoint_nodes = [
        node_id for node_id in G.nodes()
        if G.degree(node_id) == 1
    ]
    isolated_nodes = [
        node_id for node_id in G.nodes()
        if G.degree(node_id) == 0
    ]
    connectivity = _connectivity_diagnostics(
        G,
        intersections,
        rejected_connections,
        cfg,
    )

    metadata = {
        "node_count":            G.number_of_nodes(),
        "edge_count":            G.number_of_edges(),
        "skeleton_edges":        sum(
            1
            for _, _, d in G.edges(data=True)
            if d.get("edge_kind") == "geometry_segment"
        ),
        "duplicate_edges_removed": int(G.graph.get("duplicate_edges_removed", 0)),
        "tiny_edges_removed":      int(G.graph.get("tiny_edges_removed", 0)),
        "self_loop_edges_removed": int(G.graph.get("self_loop_edges_removed", 0)),
        "bridge_edges":          len(accepted_connections),
        "rejected_candidates":   len(rejected_connections),
        "intersection_count":    len(intersections),
        "endpoint_count":        len(endpoint_nodes),
        "isolated_node_count":   len(isolated_nodes),
        "disconnected_segment_count": len(disconnected),
        "suspicious_segments":   disconnected,
        "component_count":       len(components),
        "node_snap_radius_px":   cfg.node_snap_radius_px,
        "bridge_max_gap_px":     cfg.max_gap_px,
        "signal_stats":          signal_stats,
        "connectivity_validation": connectivity,
        "graph_cleanup": cleanup,
    }

    return RoadTopology(
        graph=G,
        intersections=intersections,
        continuity_links=continuity_links,
        connections=accepted_connections,
        rejected=rejected_connections,
        disconnected_segs=disconnected,
        components=components,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_topology_json(topology: RoadTopology, path: str | Path) -> None:
    """Write a human-readable JSON summary of a RoadTopology to *path*."""
    Path(path).write_text(
        json.dumps(topology.to_summary(), indent=2), encoding="utf-8"
    )
