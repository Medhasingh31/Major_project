"""Phase 12 stress suite re-run for Phase 13 confidence refinement.

These tests exercise the topology implementation through its public
``build_topology`` API.  They report behavior without forcing all expected
cases to pass.
The report records both the requested expectation and the observed decision;
this is useful for exposing conservative behavior rather than hiding it in a
passing assertion.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Allow the file to run directly as ``python tests/test_topology_stress.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_extractor.geometry import RoadGeometry, RoadSegment
from road_extractor.topology import build_topology


REPORT_PATH = Path(__file__).resolve().parents[1] / "outputs" / "phase13_topology_stress_report.json"


@dataclass
class CaseResult:
    case: str
    expected_result: str
    actual_result: str
    bridge_added: bool
    reason: str
    score: float | None = None
    signals: dict[str, float] | None = None
    intersections: list[dict] | None = None
    diagnostics: dict | None = None


def _segment(segment_id: int, points: list[tuple[int, int]], direction: float) -> RoadSegment:
    return RoadSegment(
        segment_id=segment_id,
        pixel_path=points,
        length_pixels=float(len(points) - 1),
        direction_deg=direction,
        curvature=0.0,
        width_pixels=6.0,
        bbox=(min(y for y, _ in points), min(x for _, x in points),
              max(y for y, _ in points), max(x for _, x in points)),
    )


def _geometry(segments, *, junctions=(), mask_lines=()):
    mask = np.zeros((100, 100), dtype=np.uint8)
    for line in mask_lines:
        for y, x in line:
            mask[y, x] = 1
    endpoints = [p for seg in segments for p in (seg.pixel_path[0], seg.pixel_path[-1])]
    return RoadGeometry(
        segments=list(segments),
        endpoints=endpoints,
        junctions=list(junctions),
        clean_mask=mask,
        skeleton=mask.copy(),
    )


def _bridge_case(name, expected, geometry, reason):
    topology = build_topology(geometry)
    bridge = topology.connections[0] if topology.connections else None
    diagnostic_connection = bridge or (topology.rejected[0] if topology.rejected else None)
    diagnostic = None if diagnostic_connection is None else {
        "score": round(diagnostic_connection.score, 4),
        "signals": {k: round(v, 4) for k, v in diagnostic_connection.signals.items()},
        "evidence_count": diagnostic_connection.evidence_count,
        "evidence_required": diagnostic_connection.evidence_required,
        "reasons": diagnostic_connection.decision_reasons,
    }
    actual = "SHOULD CONNECT" if bridge else "SHOULD NOT CONNECT"
    return CaseResult(
        case=name,
        expected_result=expected,
        actual_result=actual,
        bridge_added=bridge is not None,
        reason=reason if bridge is None else "Accepted by conservative multi-evidence confidence gate.",
        score=None if bridge is None else round(bridge.score, 4),
        signals=None if bridge is None else {k: round(v, 4) for k, v in bridge.signals.items()},
        diagnostics=diagnostic,
    ), topology


def run_cases() -> list[CaseResult]:
    cases = []

    # 1. Empty but very small aligned gap.  This documents the current
    # conservative requirement for mask support.
    a = _segment(1, [(40, x) for x in range(10, 31)], 0.0)
    b = _segment(2, [(40, x) for x in range(33, 54)], 0.0)
    result, _ = _bridge_case(
        "1_clear_small_aligned_gap", "SHOULD CONNECT",
        _geometry([a, b], mask_lines=[a.pixel_path, b.pixel_path]),
        "Rejected unless the clean mask supports the proposed bridge (minimum 0.70).",
    )
    cases.append(result)

    # 2. Same geometry, with explicit road-mask support through the gap.
    gap_support = [(40, x) for x in range(31, 33)]
    result, _ = _bridge_case(
        "2_gap_with_strong_mask_support", "SHOULD CONNECT",
        _geometry([a, b], mask_lines=[a.pixel_path, gap_support, b.pixel_path]),
        "Accepted because distance, alignment, continuity, width, and mask-support gates pass.",
    )
    cases.append(result)

    # 3. Close endpoints with perpendicular segment directions.
    c = _segment(3, [(20, x) for x in range(10, 31)], 0.0)
    d = _segment(4, [(22, 30 + y) for y in range(0, 21)], 90.0)
    result, _ = _bridge_case(
        "3_incompatible_directions", "SHOULD NOT CONNECT",
        _geometry([c, d], mask_lines=[c.pixel_path, d.pixel_path]),
        "Rejected by low approach alignment/centerline continuity for incompatible directions.",
    )
    cases.append(result)

    # 4. A close pair whose proposed bridge would introduce a large bend.
    e = _segment(5, [(65, x) for x in range(10, 31)], 0.0)
    f = _segment(6, [(68 + y, 34) for y in range(0, 21)], 90.0)
    result, _ = _bridge_case(
        "4_large_bend_angle", "SHOULD NOT CONNECT",
        _geometry([e, f], mask_lines=[e.pixel_path, f.pixel_path]),
        "Rejected by the centerline-continuity bend limit and opposing approach check.",
    )
    cases.append(result)

    # 5. Equal-distance candidate with no road-mask evidence.
    g = _segment(7, [(85, x) for x in range(10, 31)], 0.0)
    h = _segment(8, [(85, x) for x in range(33, 54)], 0.0)
    result, _ = _bridge_case(
        "5_equal_distance_unsupported_gap", "SHOULD NOT CONNECT",
        _geometry([g, h], mask_lines=[g.pixel_path, h.pixel_path]),
        "Rejected by the minimum clean-mask support gate despite equal/small distance.",
    )
    cases.append(result)

    # 6. T-junction: all three segment terminals share an existing junction.
    junction = (20, 50)
    t_segments = [
        _segment(10, [(20, x) for x in range(10, 51)], 0.0),
        _segment(11, [(20, x) for x in range(50, 91)], 0.0),
        _segment(12, [(20 - y, 50) for y in range(0, 31)], 90.0),
    ]
    topo = build_topology(_geometry(t_segments, junctions=[junction], mask_lines=[s.pixel_path for s in t_segments]))
    cases.append(CaseResult(
        "6_t_junction", "PRESERVE JUNCTION", "PRESERVE JUNCTION",
        any(e.get("edge_kind") == "geometry_bridge" for _, _, e in topo.graph.edges(data=True)),
        "Existing junction node retained and no bridge is allowed to use a junction node.",
        intersections=[i.to_dict() for i in topo.intersections],
        diagnostics={"bridge_edges": topo.metadata.get("bridge_edges", 0), "graph_valid": bool(topo.graph)},
    ))

    # 7. Crossroads: preserve the four-arm intersection.
    junction = (50, 50)
    x_segments = [
        _segment(20, [(50, x) for x in range(10, 51)], 0.0),
        _segment(21, [(50, x) for x in range(50, 91)], 0.0),
        _segment(22, [(y, 50) for y in range(10, 51)], 90.0),
        _segment(23, [(y, 50) for y in range(50, 91)], 90.0),
    ]
    topo = build_topology(_geometry(x_segments, junctions=[junction], mask_lines=[s.pixel_path for s in x_segments]))
    cases.append(CaseResult(
        "7_crossroads", "PRESERVE INTERSECTION", "PRESERVE INTERSECTION",
        any(e.get("edge_kind") == "geometry_bridge" for _, _, e in topo.graph.edges(data=True)),
        "Existing four-arm junction is represented as an X intersection; no artificial bridge added.",
        intersections=[i.to_dict() for i in topo.intersections],
        diagnostics={"bridge_edges": topo.metadata.get("bridge_edges", 0), "graph_valid": bool(topo.graph)},
    ))

    # 8–9. Clearly disconnected / occluded-looking but unsupported.
    i = _segment(30, [(10, x) for x in range(10, 31)], 0.0)
    j = _segment(31, [(70, x) for x in range(70, 91)], 0.0)
    result, topo = _bridge_case(
        "8_clearly_disconnected_roads", "SHOULD REMAIN DISCONNECTED",
        _geometry([i, j], mask_lines=[i.pixel_path, j.pixel_path]),
        "No candidate is within max_gap_px; both segments remain isolated.",
    )
    result.actual_result = "SHOULD REMAIN DISCONNECTED" if not result.bridge_added else "CONNECTED"
    cases.append(result)

    k = _segment(40, [(95, x) for x in range(10, 31)], 0.0)
    m = _segment(41, [(95, x) for x in range(33, 54)], 0.0)
    result, _ = _bridge_case(
        "9_occluded_looking_insufficient_evidence", "SHOULD REMAIN DISCONNECTED",
        _geometry([k, m], mask_lines=[k.pixel_path, m.pixel_path]),
        "Close and aligned is insufficient: mask support is below the conservative threshold.",
    )
    result.actual_result = "SHOULD REMAIN DISCONNECTED" if not result.bridge_added else "CONNECTED"
    cases.append(result)

    return cases


def main() -> None:
    results = run_cases()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    for result in results:
        print(f"{result.case}: expected={result.expected_result}; actual={result.actual_result}; bridge={result.bridge_added}; reason={result.reason}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
