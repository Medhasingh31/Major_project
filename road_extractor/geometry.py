"""
geometry.py — Road Geometry Extraction Module
==============================================
Part of the Geometry + Topology pipeline.
Completely independent of the U-Net, its training code, and its inference code.

INPUT
-----
  rgb_image    : np.ndarray  (H, W, 3)  uint8   – original RGB image
  repaired_mask: np.ndarray  (H, W)     uint8   – output of repair_road_mask()
                                                   treated as READ-ONLY here

PROCESSING STEPS  (all self-contained, no U-Net calls)
-------------------------------------------------------
  1. clean_binary_mask()        – denoise & morphologically clean the mask
  2. extract_skeleton()         – skimage / OpenCV thinning → 1-px centerline
  3. prune_skeleton_spurs()     – iterative removal of short dead-end branches
  4. _classify_pixels()         – label every skeleton pixel (junction / endpoint)
  5. _trace_segment()           – walk skeleton between special pixels
  6. _estimate_width()          – local distance-transform width per segment
  7. per-segment geometry       – arc-length, PCA direction, mean curvature
  8. extract_geometry()         – orchestrates all steps, returns RoadGeometry

OUTPUT
------
  RoadGeometry
    .segments    : list[RoadSegment]       – ordered pixel paths + properties
    .junctions   : list[(y,x)]             – branching / crossing points
    .endpoints   : list[(y,x)]             – dead-ends
    .skeleton    : np.ndarray              – pruned 1-px skeleton (for topology)
    .clean_mask  : np.ndarray              – cleaned binary mask (for debug)
    .metadata    : dict

PIPELINE POSITION
-----------------
  repaired_mask ──► geometry.extract_geometry() ──► RoadGeometry
                                                          │
                                                          ▼
                                                   topology.build_topology()
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GeometryConfig:
    """
    All tunable thresholds for the geometry pipeline.
    Pass a custom instance to extract_geometry() to override defaults.
    """
    # --- mask cleaning ---
    min_road_area_px: int = 50
    """Remove connected components smaller than this (pixel area)."""

    morphology_close_radius: int = 2
    """Disk radius for morphological closing on the cleaned mask."""

    min_small_component_span_px: int = 12
    """Minimum major-axis span for preserving a small road-like component."""

    min_small_component_aspect: float = 6.0
    """Minimum bounding-box aspect ratio for a small elongated component."""

    small_component_context_radius_px: int = 4
    """Radius used to detect nearby road support around small components."""

    # --- spur pruning ---
    min_branch_length_px: int = 14
    """Skeleton branches shorter than this (in pixels) are removed as spurs."""

    hard_spur_length_px: float = 7.0
    """Branches at or below this length are removed when attached to a junction."""

    legitimate_branch_angle_deg: float = 80.0
    """Minimum branch angle used as evidence for a legitimate short side road."""

    legitimate_branch_mask_support: float = 0.70
    """Minimum local mask support used to preserve a short side road."""

    legitimate_branch_min_fraction: float = 0.99
    """Minimum fraction of the spur cutoff for preserving a short side road."""

    prune_iterations: int = 10
    """Maximum number of pruning passes (each pass may shorten or remove spurs)."""

    min_skeleton_component_px: int = 22
    """Remove isolated skeleton components smaller than this pixel count."""

    max_unsupported_branch_length_px: float = 28.0
    """Upper length bound for the conservative unsupported-branch pass."""

    unsupported_branch_angle_deg: float = 32.0
    """Near-collinearity threshold for identifying duplicate short arms."""

    branch_support_radius_px: int = 4
    """Radius used to measure nearby mask/skeleton support around a branch."""

    max_internal_branch_length_px: float = 6.0
    """Maximum length considered for a suspicious junction-to-junction branch."""

    internal_branch_mask_support: float = 0.45
    """Minimum independent mask support required to keep a tiny internal branch."""

    # --- gap bridging ---
    max_gap_bridge_px: float = 7.0
    """Maximum endpoint-to-endpoint gap to bridge in skeleton pixels."""

    bridge_angle_tolerance_deg: float = 35.0
    """Maximum tangent angle mismatch allowed for endpoint gap bridging."""

    bridge_mask_support_ratio: float = 0.55
    """Minimum fraction of a bridge line supported by the dilated road mask."""

    bridge_crossing_clearance_px: int = 1
    """Ignore expected endpoint neighborhoods when checking bridge crossings."""

    # --- collinear long-range bridging ---
    collinear_max_gap_px: float = 400.0
    """Maximum distance for collinear dead-end bridging across large occlusions."""

    collinear_max_angle_deg: float = 22.0
    """Strict angular alignment tolerance for collinear bridging."""
    # --- segment filtering ---
    min_segment_length_px: float = 10.0
    """Segments shorter than this after tracing are discarded as noise."""

    # --- width estimation ---
    width_sample_stride: int = 5
    """Sample the distance transform every N pixels along the path for width."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RoadSegment:
    """
    A single ordered road centerline segment.

    All coordinates use (row, col) = (y, x) image convention.
    """

    segment_id: int

    # Ordered pixel path from one special pixel to the next
    # Each element is (y, x)  →  row, column
    pixel_path: list[tuple[int, int]]

    # ---- geometric properties ----
    length_pixels: float
    """Euclidean arc length (sum of step distances along the path)."""

    direction_deg: float
    """Dominant orientation in degrees, range [0, 180).
    Computed via PCA on path pixel coordinates so it is robust to curves."""

    curvature: float
    """Mean absolute angular change in radians per pixel.
    0.0 for a perfectly straight segment."""

    width_pixels: float
    """Approximate road width at the centerline (mean of local distance-
    transform values × 2).  Based on the *cleaned* binary mask."""

    # ---- bounding box: (y_min, x_min, y_max, x_max) ----
    bbox: tuple[int, int, int, int] = field(default_factory=lambda: (0, 0, 0, 0))

    # ---- appearance ----
    mean_color: tuple[float, float, float] = field(
        default_factory=lambda: (0.0, 0.0, 0.0)
    )
    """Mean RGB sampled from the original image along the centerline."""

    # ----------------------------------------------------------------
    # Convenience helpers
    # ----------------------------------------------------------------

    def start_point(self) -> tuple[int, int]:
        return self.pixel_path[0]

    def end_point(self) -> tuple[int, int]:
        return self.pixel_path[-1]

    def midpoint(self) -> tuple[float, float]:
        mid = len(self.pixel_path) // 2
        return float(self.pixel_path[mid][0]), float(self.pixel_path[mid][1])

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "num_pixels": len(self.pixel_path),
            "length_pixels": round(self.length_pixels, 2),
            "direction_deg": round(self.direction_deg, 2),
            "curvature": round(self.curvature, 6),
            "width_pixels": round(self.width_pixels, 2),
            "bbox": list(self.bbox),
            "mean_color": [round(c, 1) for c in self.mean_color],
            "start": list(self.start_point()),
            "end": list(self.end_point()),
        }


@dataclass
class RoadGeometry:
    """Container for all geometry extracted from one (mask, image) pair."""

    segments: list[RoadSegment] = field(default_factory=list)
    junctions: list[tuple[int, int]] = field(default_factory=list)
    endpoints: list[tuple[int, int]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # Arrays produced during extraction (kept for downstream modules)
    skeleton:   Optional[np.ndarray] = field(default=None, repr=False)
    clean_mask: Optional[np.ndarray] = field(default=None, repr=False)

    # ----------------------------------------------------------------

    def total_length(self) -> float:
        return sum(s.length_pixels for s in self.segments)

    def segment_count(self) -> int:
        return len(self.segments)

    def junction_count(self) -> int:
        return len(self.junctions)

    def to_summary(self) -> dict:
        return {
            "segment_count": self.segment_count(),
            "junction_count": self.junction_count(),
            "endpoint_count": len(self.endpoints),
            "total_length_pixels": round(self.total_length(), 2),
            "metadata": self.metadata,
            "segments": [s.to_dict() for s in self.segments],
        }


# ---------------------------------------------------------------------------
# Step 1 — Clean binary mask
# ---------------------------------------------------------------------------

def clean_binary_mask(
    repaired_mask: np.ndarray,
    min_area: int = 50,
    close_radius: int = 2,
    min_small_component_span: int = 12,
    min_small_component_aspect: float = 2.5,
    context_radius: int = 4,
) -> np.ndarray:
    """
    Produce a clean, hole-free binary road mask from the repaired U-Net output.

    Operations (independent of any model):
      1. Binarise (handle uint8 0/1 AND 0/255 inputs safely)
      2. Remove small connected components (noise blobs)
      3. Fill interior holes with morphological closing
      4. Final binarise

    Parameters
    ----------
    repaired_mask : np.ndarray
        Binary or near-binary mask, shape (H, W).
    min_area : int
        Minimum component area in pixels to keep.
    close_radius : int
        Disk radius for morphological closing.

    Returns
    -------
    np.ndarray  shape (H, W), dtype uint8, values 0 or 1.
    """
    # --- 1. safe binarise ---
    binary = (repaired_mask > 0).astype(np.uint8)

    # --- 2. remove small components ---
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    cleaned = np.zeros_like(binary)
    context_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (context_radius * 2 + 1, context_radius * 2 + 1),
    )
    for lbl in range(1, n_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == lbl] = 1
            continue

        width = int(stats[lbl, cv2.CC_STAT_WIDTH])
        height = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        major = max(width, height)
        minor = max(min(width, height), 1)
        aspect = major / minor
        component = (labels == lbl).astype(np.uint8)
        nearby = cv2.dilate(component, context_kernel) > 0
        nearby_support = bool(np.any(nearby & (binary > 0) & (component == 0)))
        elongated = (
            major >= min_small_component_span
            and aspect >= min_small_component_aspect
        )

        # Compact isolated blobs are noise. Preserve small elongated road-like
        # fragments, including fragments close to another road component.
        thin_road_like = (
            elongated
            and aspect >= min_small_component_aspect
            and area <= int(major * 1.5)
        )
        if thin_road_like or (nearby_support and elongated):
            cleaned[labels == lbl] = 1

    # --- 3. component-local closing to fill internal holes ---
    # Applying closing to the whole image can make two nearby but unrelated
    # roads touch.  Close each retained component independently so this stage
    # repairs holes without creating inter-road connections.
    if close_radius <= 0:
        closed = cleaned
    else:
        k = close_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        n_clean, clean_labels, _, _ = cv2.connectedComponentsWithStats(
            cleaned,
            connectivity=8,
        )
        closed = np.zeros_like(cleaned)
        for clean_label in range(1, n_clean):
            component = (clean_labels == clean_label).astype(np.uint8)
            closed |= cv2.morphologyEx(component, cv2.MORPH_CLOSE, kernel)

    return (closed > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Step 2 — Skeleton / centerline extraction
# ---------------------------------------------------------------------------

def extract_skeleton(clean_mask: np.ndarray) -> np.ndarray:
    """
    Reduce a filled binary mask to a 1-pixel-wide medial axis (skeleton).

    Uses skimage.morphology.skeletonize when available (Zhang-Suen thinning),
    otherwise falls back to an OpenCV iterative-erosion approach.

    Parameters
    ----------
    clean_mask : np.ndarray
        Clean binary mask, shape (H, W), dtype uint8 (values 0 or 1).

    Returns
    -------
    np.ndarray  shape (H, W), dtype uint8, values 0 or 1.
    """
    binary_bool = clean_mask > 0

    try:
        from skimage.morphology import skeletonize as sk_skeletonize
        return sk_skeletonize(binary_bool).astype(np.uint8)
    except ImportError:
        pass

    # --- OpenCV iterative thinning fallback ---
    img = binary_bool.astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        remaining = cv2.countNonZero(img)
        if remaining == 0:
            break
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        tmp = cv2.subtract(img, opened)
        img = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, tmp)
    return (skel > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Step 3 — Spur / noisy-branch pruning
# ---------------------------------------------------------------------------

_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _skeleton_neighbors(y: int, x: int, skel: np.ndarray) -> list[tuple[int, int]]:
    h, w = skel.shape
    return [
        (y + dy, x + dx)
        for dy, dx in _N8
        if 0 <= y + dy < h and 0 <= x + dx < w and skel[y + dy, x + dx] > 0
    ]


def _trace_from_endpoint(
    tip_y: int,
    tip_x: int,
    skel: np.ndarray,
    max_steps: int,
) -> tuple[list[tuple[int, int]], str]:
    """
    Walk from an endpoint until reaching a junction, another endpoint, or
    max_steps. Returns the traced path and the terminal condition.
    """
    prev = (-1, -1)
    cur = (tip_y, tip_x)
    path = [cur]
    while len(path) <= max_steps:
        nbrs = [n for n in _skeleton_neighbors(*cur, skel) if n != prev]
        if not nbrs:
            return path, "dead_end"
        nxt = nbrs[0]
        path.append(nxt)
        deg = len(_skeleton_neighbors(*nxt, skel))
        if deg >= 3:
            return path, "junction"
        if deg == 1 and nxt != (tip_y, tip_x):
            return path, "endpoint"
        prev, cur = cur, nxt
    return path, "long"


def _remove_small_skeleton_components(
    skeleton: np.ndarray,
    min_size: int,
    min_road_like_pixels: int = 10,
    min_road_like_span: int = 10,
) -> tuple[np.ndarray, int]:
    """Remove compact isolated fragments while preserving line-like roads."""
    binary = (skeleton > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    cleaned = np.zeros_like(binary)
    removed = 0
    for lbl in range(1, n_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area >= min_size:
            cleaned[labels == lbl] = 1
            continue

        width = int(stats[lbl, cv2.CC_STAT_WIDTH])
        height = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        component = (labels == lbl).astype(np.uint8)
        ys, xs = np.where(component > 0)
        endpoints = sum(
            len(_skeleton_neighbors(int(y), int(x), component)) == 1
            for y, x in zip(ys.tolist(), xs.tolist())
        )
        line_like = (
            area >= min_road_like_pixels
            and max(width, height) >= min_road_like_span
            and endpoints == 2
        )
        if line_like:
            cleaned[labels == lbl] = 1
        else:
            removed += area
    return cleaned, removed


def prune_skeleton_spurs(
    skeleton: np.ndarray,
    min_branch_length: int = 10,
    iterations: int = 10,
    clean_mask: Optional[np.ndarray] = None,
    support_radius: int = 4,
    hard_spur_length: float = 7.0,
    legitimate_branch_angle: float = 60.0,
    legitimate_branch_mask_support: float = 0.30,
    legitimate_branch_min_fraction: float = 0.80,
) -> tuple[np.ndarray, int]:
    """
    Iteratively remove short dead-end branches (spurs) from the skeleton.

    A spur is a chain of pixels that starts at a degree-1 pixel (endpoint)
    and whose length is below *min_branch_length*.  Each pass finds all such
    tips and removes the spur chain up to (but not including) the first
    junction or long-branch endpoint it meets.

    Parameters
    ----------
    skeleton : np.ndarray
        Binary skeleton, shape (H, W), dtype uint8.
    min_branch_length : int
        Branches shorter than this are removed.
    iterations : int
        Number of pruning passes.

    Returns
    -------
    np.ndarray  Pruned skeleton, same shape and dtype.
    """
    skel = skeleton.copy()
    removed_pixels = 0

    for _ in range(iterations):
        changed = False
        ys, xs = np.where(skel > 0)
        tips = [
            (int(y), int(x))
            for y, x in zip(ys, xs)
            if len(_skeleton_neighbors(int(y), int(x), skel)) == 1
        ]

        for ty, tx in tips:
            if skel[ty, tx] == 0:
                continue  # already removed in this pass

            path, terminal = _trace_from_endpoint(
                ty,
                tx,
                skel,
                max_steps=min_branch_length,
            )
            branch_length = _arc_length(path)
            if terminal != "junction" or branch_length >= min_branch_length:
                continue

            preserve_legitimate_side_road = False
            if clean_mask is not None:
                branch_angle = _branch_angle_at_junction(path, skel)
                mask_support = _local_mask_support(path, clean_mask, support_radius)
                endpoint_context = (
                    len(_skeleton_neighbors(ty, tx, skel)) == 1
                    and terminal == "junction"
                )
                relative_to_arms = _is_short_relative_to_junction_arms(path, skel)

                # A short branch is preserved only when independent evidence
                # agrees that it is a real road arm.  The hard-spur floor
                # still rejects contact-sized artifacts, while relative arm
                # length replaces the former near-full-threshold requirement.
                preserve_legitimate_side_road = (
                    branch_length > hard_spur_length
                    and endpoint_context
                    and branch_angle >= legitimate_branch_angle
                    and mask_support >= legitimate_branch_mask_support
                    and (
                        relative_to_arms
                        or branch_length >= min_branch_length * 0.50
                    )
                )

            # Remove an extremely short branch, or a longer sub-threshold
            # branch lacking independent evidence of being a real side road.
            if not preserve_legitimate_side_road:
                # Erase the spur but keep the junction pixel itself.
                for py, px in path[:-1]:
                    if skel[py, px] > 0:
                        skel[py, px] = 0
                        removed_pixels += 1
                changed = True

        if not changed:
            break

    return skel, removed_pixels


def _local_mask_support(
    path: list[tuple[int, int]],
    clean_mask: np.ndarray,
    radius: int,
) -> float:
    """Return the fraction of local disk samples supported by the road mask."""
    if not path:
        return 0.0
    size = radius * 2 + 1
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) > 0
    values: list[float] = []
    h, w = clean_mask.shape
    for y, x in path[:: max(1, len(path) // 12)]:
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        local_disk = disk[
            radius - (y - y0): radius + (y1 - y),
            radius - (x - x0): radius + (x1 - x),
        ]
        local_mask = clean_mask[y0:y1, x0:x1] > 0
        values.append(float(np.mean(local_mask[local_disk])))
    return float(np.mean(values)) if values else 0.0


def _nearby_skeleton_support(
    path: list[tuple[int, int]],
    skeleton: np.ndarray,
    radius: int,
) -> float:
    """Measure whether a short branch runs beside another centerline."""
    if len(path) < 4:
        return 0.0
    branch = np.zeros_like(skeleton, dtype=np.uint8)
    for y, x in path:
        branch[y, x] = 1
    other = ((skeleton > 0) & (branch == 0)).astype(np.uint8)
    # Do not count the ordinary convergence of arms at the junction as
    # parallel support. Only nearby centerline evidence away from the node is
    # useful for identifying a duplicated short arm.
    junction_y, junction_x = path[-1]
    cv2.circle(other, (junction_x, junction_y), radius + 1, 0, thickness=-1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    nearby = cv2.dilate(other, kernel) > 0
    interior = path[1:-2]
    return float(np.mean([nearby[y, x] for y, x in interior])) if interior else 0.0


def _branch_angle_at_junction(
    path: list[tuple[int, int]],
    skeleton: np.ndarray,
) -> float:
    """Return the smallest angle between a branch and another junction arm."""
    if len(path) < 3:
        return 180.0
    junction = path[-1]
    branch_neighbor = path[-2]
    branch_vector = np.asarray(branch_neighbor, dtype=np.float64) - np.asarray(
        junction, dtype=np.float64
    )
    angles = []
    for neighbor in _skeleton_neighbors(*junction, skeleton):
        if neighbor == branch_neighbor:
            continue
        other_vector = np.asarray(neighbor, dtype=np.float64) - np.asarray(
            junction, dtype=np.float64
        )
        angles.append(_angle_between_vectors(branch_vector, other_vector))
    return min(angles) if angles else 180.0


def _is_short_relative_to_junction_arms(
    path: list[tuple[int, int]],
    skeleton: np.ndarray,
    ratio: float = 0.70,
) -> bool:
    """Require a candidate arm to be materially shorter than its peers."""
    if len(path) < 3:
        return False
    junction = path[-1]
    branch_neighbor = path[-2]
    other_lengths: list[float] = []
    for neighbor in _skeleton_neighbors(*junction, skeleton):
        if neighbor == branch_neighbor:
            continue
        arm = [junction, neighbor]
        previous, current = junction, neighbor
        for _ in range(128):
            degree = len(_skeleton_neighbors(*current, skeleton))
            if degree != 2:
                break
            candidates = [n for n in _skeleton_neighbors(*current, skeleton) if n != previous]
            if not candidates:
                break
            nxt = candidates[0]
            arm.append(nxt)
            previous, current = current, nxt
        other_lengths.append(_arc_length(arm))
    if len(other_lengths) < 2:
        return False
    return _arc_length(path) <= ratio * float(np.median(other_lengths))


def prune_unsupported_branches(
    skeleton: np.ndarray,
    clean_mask: np.ndarray,
    max_branch_length: float = 28.0,
    angle_tolerance_deg: float = 32.0,
    support_radius: int = 4,
) -> tuple[np.ndarray, int]:
    """Remove only short branches with independent evidence of being spurious.

    A branch must terminate at a junction and be short. It is then removed
    only when it is either near-collinear with another arm while running beside
    existing skeleton support, or has weak local mask support. This keeps
    genuine side roads and non-collinear junction arms intact.
    """
    skel = skeleton.copy()
    removed_pixels = 0
    max_steps = max(32, int(math.ceil(max_branch_length * 2.0)))

    for _ in range(6):
        changed = False
        _, endpoints = _classify_pixels(skel)
        for endpoint in sorted(endpoints):
            if skel[endpoint] == 0:
                continue
            path, terminal = _trace_from_endpoint(
                endpoint[0], endpoint[1], skel, max_steps=max_steps
            )
            branch_length = _arc_length(path)
            if terminal != "junction" or branch_length > max_branch_length:
                continue

            angle = _branch_angle_at_junction(path, skel)
            nearby_support = _nearby_skeleton_support(path, skel, support_radius)
            mask_support = _local_mask_support(path, clean_mask, support_radius)
            relatively_short = _is_short_relative_to_junction_arms(path, skel)
            duplicate_arm = (
                relatively_short
                and angle <= angle_tolerance_deg
                and nearby_support >= 0.35
            )
            weak_support = (
                mask_support < 0.34
                and branch_length <= 22.0
                and angle > angle_tolerance_deg
                and relatively_short
            )
            if not (duplicate_arm or weak_support):
                continue

            for y, x in path[:-1]:
                if skel[y, x] > 0:
                    skel[y, x] = 0
                    removed_pixels += 1
            changed = True

        if not changed:
            break

    return skel, removed_pixels


def prune_tiny_internal_branches(
    skeleton: np.ndarray,
    clean_mask: np.ndarray,
    max_branch_length: float = 6.0,
    min_mask_support: float = 0.45,
    support_radius: int = 4,
) -> tuple[np.ndarray, int]:
    """Remove weak, very short branches joining two junctions.

    A spur ending at a junction is handled by ``prune_skeleton_spurs``.  A
    one/tiny-pixel accidental contact can instead turn into a junction at both
    ends, leaving no endpoint for that pass to find.  This pass considers only
    those internal paths, requires them to be very short, and removes them
    only when the road mask does not independently support their corridor.
    Paths whose ends are part of the same junction cluster are ignored.
    """
    skel = skeleton.copy()
    removed_pixels = 0

    for _ in range(3):
        junctions, _ = _classify_pixels(skel)
        candidates: list[tuple[list[tuple[int, int]], float]] = []
        seen: set[frozenset[tuple[int, int]]] = set()

        for origin in sorted(junctions):
            for first in _skeleton_neighbors(*origin, skel):
                if first in junctions:
                    continue
                path = [origin, first]
                previous, current = origin, first
                terminal: Optional[tuple[int, int]] = None
                while len(path) <= int(math.ceil(max_branch_length)) + 2:
                    degree = len(_skeleton_neighbors(*current, skel))
                    if degree >= 3:
                        terminal = current
                        break
                    if degree != 2:
                        break
                    choices = [
                        n for n in _skeleton_neighbors(*current, skel)
                        if n != previous
                    ]
                    if not choices:
                        break
                    nxt = choices[0]
                    path.append(nxt)
                    previous, current = current, nxt

                if terminal is None or terminal == origin:
                    continue
                key = frozenset((origin, terminal))
                if key in seen:
                    continue
                seen.add(key)
                length = _arc_length(path)
                if length > max_branch_length:
                    continue
                # Do not treat adjacent pixels in one thick junction as an
                # internal road branch.
                if len(path) <= 2:
                    continue
                support = _local_mask_support(path, clean_mask, support_radius)
                candidates.append((path, support))

        if not candidates:
            break

        changed = False
        for path, support in candidates:
            if support >= min_mask_support:
                continue
            for y, x in path[1:-1]:
                if skel[y, x] > 0:
                    skel[y, x] = 0
                    removed_pixels += 1
                    changed = True
        if not changed:
            break

    return skel, removed_pixels


# ---------------------------------------------------------------------------
# Step 4 — Pixel classification (junction / endpoint)
# ---------------------------------------------------------------------------

def _classify_pixels(
    skel: np.ndarray,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """
    Label every active skeleton pixel by its connectivity degree.

    Returns
    -------
    junctions : set  pixels with degree ≥ 3  (branching / crossing)
    endpoints : set  pixels with degree == 1 (dead-end / road tip)
    """
    junctions: set[tuple[int, int]] = set()
    endpoints: set[tuple[int, int]] = set()
    ys, xs = np.where(skel > 0)
    for y, x in zip(ys.tolist(), xs.tolist()):
        deg = len(_skeleton_neighbors(int(y), int(x), skel))
        if deg >= 3:
            junctions.add((int(y), int(x)))
        elif deg == 1:
            endpoints.add((int(y), int(x)))
    return junctions, endpoints


def _count_connected_components(binary: np.ndarray) -> int:
    """Count foreground connected components in a binary image."""
    n_labels, _, _, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8),
        connectivity=8,
    )
    return max(int(n_labels) - 1, 0)


def _angle_between_vectors(a: np.ndarray, b: np.ndarray) -> float:
    """Return the smaller angle between two 2D vectors in degrees."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 180.0
    cosine = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _endpoint_tangent(
    endpoint: tuple[int, int],
    skel: np.ndarray,
    lookahead: int = 6,
) -> Optional[np.ndarray]:
    """
    Estimate outward direction at an endpoint using the first few centerline
    pixels behind it. Vector order is [dy, dx] to match (y, x) coordinates.
    """
    path, _ = _trace_from_endpoint(
        endpoint[0],
        endpoint[1],
        skel,
        max_steps=lookahead,
    )
    if len(path) < 2:
        return None
    anchor = path[min(len(path) - 1, lookahead)]
    return np.array(
        [endpoint[0] - anchor[0], endpoint[1] - anchor[1]],
        dtype=np.float64,
    )


def _line_pixels(
    p0: tuple[int, int],
    p1: tuple[int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    """Rasterize an inclusive line between two (y, x) pixels."""
    canvas = np.zeros(shape, dtype=np.uint8)
    cv2.line(canvas, (p0[1], p0[0]), (p1[1], p1[0]), 1, thickness=1)
    ys, xs = np.where(canvas > 0)
    return [(int(y), int(x)) for y, x in zip(ys, xs)]


def _bridge_crosses_unrelated_skeleton(
    p0: tuple[int, int],
    p1: tuple[int, int],
    pixels: list[tuple[int, int]],
    skeleton: np.ndarray,
    clearance: int = 1,
) -> bool:
    """Return whether a proposed bridge intersects other centerline evidence."""
    obstacle_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (clearance * 2 + 1, clearance * 2 + 1),
    )
    nearby_skeleton = cv2.dilate((skeleton > 0).astype(np.uint8), obstacle_kernel)
    for y, x in pixels:
        if min(
            math.hypot(y - p0[0], x - p0[1]),
            math.hypot(y - p1[0], x - p1[1]),
        ) <= clearance + 1:
            continue
        if nearby_skeleton[y, x] > 0:
            return True
    return False


def bridge_small_gaps(
    skeleton: np.ndarray,
    clean_mask: np.ndarray,
    max_gap: float = 7.0,
    angle_tolerance_deg: float = 35.0,
    mask_support_ratio: float = 0.55,
    crossing_clearance: int = 1,
) -> tuple[np.ndarray, int]:
    """
    Bridge tiny endpoint-to-endpoint skeleton gaps when local geometry agrees.

    A bridge is accepted only when endpoints are close, their outward tangents
    face one another, and most bridge pixels lie inside or near the road mask.
    """
    skel = skeleton.copy()
    _, endpoints = _classify_pixels(skel)
    endpoint_list = sorted(endpoints)
    if len(endpoint_list) < 2:
        return skel, 0

    support_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    supported_mask = cv2.dilate((clean_mask > 0).astype(np.uint8), support_kernel)
    tangents = {pt: _endpoint_tangent(pt, skel) for pt in endpoint_list}
    used: set[tuple[int, int]] = set()
    bridges_added = 0

    for i, p0 in enumerate(endpoint_list):
        if p0 in used:
            continue
        best: Optional[tuple[float, tuple[int, int], list[tuple[int, int]]]] = None
        t0 = tangents.get(p0)
        if t0 is None:
            continue

        for p1 in endpoint_list[i + 1:]:
            if p1 in used:
                continue
            dy = p1[0] - p0[0]
            dx = p1[1] - p0[1]
            gap = math.sqrt(float(dy * dy + dx * dx))
            if gap <= 1.5 or gap > max_gap:
                continue

            t1 = tangents.get(p1)
            if t1 is None:
                continue

            gap_vec = np.array([dy, dx], dtype=np.float64)
            if _angle_between_vectors(t0, gap_vec) > angle_tolerance_deg:
                continue
            if _angle_between_vectors(t1, -gap_vec) > angle_tolerance_deg:
                continue
            if _angle_between_vectors(t0, -t1) > angle_tolerance_deg:
                continue

            pixels = _line_pixels(p0, p1, skel.shape)
            if len(pixels) <= 2:
                continue
            if _bridge_crosses_unrelated_skeleton(
                p0,
                p1,
                pixels,
                skel,
                clearance=crossing_clearance,
            ):
                continue
            support = np.mean([supported_mask[y, x] > 0 for y, x in pixels])
            if support < mask_support_ratio:
                continue
            if best is None or gap < best[0]:
                best = (gap, p1, pixels)

        if best is None:
            continue
        _, p1, pixels = best
        for y, x in pixels:
            skel[y, x] = 1
        used.add(p0)
        used.add(p1)
        bridges_added += 1

    return skel, bridges_added


def bridge_collinear_dead_ends(
    skeleton: np.ndarray,
    max_gap: float = 400.0,
    angle_tolerance_deg: float = 22.0,
    clearance: int = 2,
) -> tuple[np.ndarray, int]:
    """
    Bridge long-range occlusion gaps between dead-end endpoints that are
    strictly collinear and pointing towards each other across broken canopy/shadows.
    """
    skel = skeleton.copy()
    _, endpoints = _classify_pixels(skel)
    endpoint_list = sorted(endpoints)
    if len(endpoint_list) < 2:
        return skel, 0

    tangents = {pt: _endpoint_tangent(pt, skel) for pt in endpoint_list}
    used: set[tuple[int, int]] = set()
    bridges_added = 0

    pairs = []
    for i, p0 in enumerate(endpoint_list):
        t0 = tangents.get(p0)
        if t0 is None:
            continue
        for p1 in endpoint_list[i + 1:]:
            t1 = tangents.get(p1)
            if t1 is None:
                continue
            dy = p1[0] - p0[0]
            dx = p1[1] - p0[1]
            gap = math.sqrt(float(dy * dy + dx * dx))
            if gap <= 7.0 or gap > max_gap:
                continue
            gap_vec = np.array([dy, dx], dtype=np.float64)
            a0 = _angle_between_vectors(t0, gap_vec)
            a1 = _angle_between_vectors(t1, -gap_vec)
            ao = _angle_between_vectors(t0, -t1)
            if a0 <= angle_tolerance_deg and a1 <= angle_tolerance_deg and ao <= angle_tolerance_deg + 5.0:
                pairs.append((gap, p0, p1))

    pairs.sort(key=lambda x: x[0])
    for gap, p0, p1 in pairs:
        if p0 in used or p1 in used:
            continue
        pixels = _line_pixels(p0, p1, skel.shape)
        if len(pixels) <= 2:
            continue
        if _bridge_crosses_unrelated_skeleton(p0, p1, pixels, skel, clearance=clearance):
            continue
        for y, x in pixels:
            skel[y, x] = 1
        used.add(p0)
        used.add(p1)
        bridges_added += 1

    return skel, bridges_added

# ---------------------------------------------------------------------------
# Step 5 — Segment tracing
# ---------------------------------------------------------------------------

def _cluster_pixel_set(
    pixels: set[tuple[int, int]],
) -> tuple[list[set[tuple[int, int]]], dict[tuple[int, int], int]]:
    """Group 8-connected pixels, used to collapse junction blobs."""
    remaining = set(pixels)
    clusters: list[set[tuple[int, int]]] = []
    pixel_to_cluster: dict[tuple[int, int], int] = {}

    while remaining:
        seed = remaining.pop()
        cluster = {seed}
        stack = [seed]
        while stack:
            y, x = stack.pop()
            for dy, dx in _N8:
                nbr = (y + dy, x + dx)
                if nbr in remaining:
                    remaining.remove(nbr)
                    cluster.add(nbr)
                    stack.append(nbr)

        cluster_id = len(clusters)
        for pixel in cluster:
            pixel_to_cluster[pixel] = cluster_id
        clusters.append(cluster)

    return clusters, pixel_to_cluster


def _same_junction_cluster(
    p0: tuple[int, int],
    p1: tuple[int, int],
    junction_cluster_by_pixel: dict[tuple[int, int], int],
) -> bool:
    c0 = junction_cluster_by_pixel.get(p0)
    c1 = junction_cluster_by_pixel.get(p1)
    return c0 is not None and c0 == c1


def _junction_representatives(
    clusters: list[set[tuple[int, int]]],
) -> list[tuple[int, int]]:
    """Pick one stable center pixel for each junction cluster."""
    representatives: list[tuple[int, int]] = []
    for cluster in clusters:
        pts = np.array(sorted(cluster), dtype=np.float64)
        centroid = pts.mean(axis=0)
        best = min(
            cluster,
            key=lambda p: (p[0] - centroid[0]) ** 2 + (p[1] - centroid[1]) ** 2,
        )
        representatives.append((int(best[0]), int(best[1])))
    return sorted(representatives)


def _trace_segment(
    start: tuple[int, int],
    first_step: tuple[int, int],
    skel: np.ndarray,
    special: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    Walk the skeleton from *start* → *first_step* until hitting a special
    pixel (junction or endpoint) or a dead-end.

    Returns the ordered list of pixels including both terminal pixels.
    """
    path: list[tuple[int, int]] = [start, first_step]
    prev, cur = start, first_step

    while cur not in special:
        candidates = [
            n for n in _skeleton_neighbors(*cur, skel) if n != prev
        ]
        if not candidates:
            break
        nxt = candidates[0]
        path.append(nxt)
        prev, cur = cur, nxt

    return path


# ---------------------------------------------------------------------------
# Step 6 — Width estimation via distance transform
# ---------------------------------------------------------------------------

def _build_distance_transform(clean_mask: np.ndarray) -> np.ndarray:
    """
    Compute the Euclidean distance transform of the clean mask.
    Each foreground pixel's value = distance to the nearest background pixel.
    This is equivalent to the local half-width of the road at that pixel.
    """
    # cv2.distanceTransform needs uint8 mask with 0/255
    return cv2.distanceTransform(
        (clean_mask > 0).astype(np.uint8), cv2.DIST_L2, maskSize=5
    )


def _estimate_width(
    path: list[tuple[int, int]],
    dist_transform: np.ndarray,
    stride: int = 5,
) -> float:
    """
    Estimate the approximate road width for a segment.

    At every *stride*-th pixel along the path, the distance-transform value
    gives the distance from the centerline to the road edge.
    Width ≈ 2 × mean(distance-transform samples).

    Parameters
    ----------
    path : list[(y,x)]
    dist_transform : np.ndarray  – output of _build_distance_transform()
    stride : int                 – sample every N-th pixel

    Returns
    -------
    float  estimated width in pixels (≥ 1.0)
    """
    h, w = dist_transform.shape
    samples = [
        dist_transform[y, x]
        for idx, (y, x) in enumerate(path)
        if idx % stride == 0 and 0 <= y < h and 0 <= x < w
    ]
    if not samples:
        # Fall back to single midpoint sample
        my, mx = path[len(path) // 2]
        return float(dist_transform[my, mx]) * 2.0 if 0 <= my < h and 0 <= mx < w else 1.0

    return max(float(np.mean(samples)) * 2.0, 1.0)


# ---------------------------------------------------------------------------
# Geometric property helpers
# ---------------------------------------------------------------------------

def _arc_length(path: list[tuple[int, int]]) -> float:
    """Euclidean arc length along the ordered pixel path."""
    total = 0.0
    for i in range(1, len(path)):
        dy = path[i][0] - path[i - 1][0]
        dx = path[i][1] - path[i - 1][1]
        total += math.sqrt(dy * dy + dx * dx)
    return total


def _dominant_direction(path: list[tuple[int, int]]) -> float:
    """
    Dominant orientation angle of the segment in degrees, range [0, 180).

    Uses PCA on the pixel coordinates: the principal eigenvector of the
    2D scatter matrix gives the direction that minimises angular residuals.
    Falls back to simple start→end angle for very short paths (< 3 pixels).
    """
    if len(path) < 3:
        dy = path[-1][0] - path[0][0]
        dx = path[-1][1] - path[0][1]
        return math.degrees(math.atan2(float(dy), float(dx))) % 180.0

    pts = np.array(path, dtype=np.float64)          # (N, 2)  [y, x]
    centered = pts - pts.mean(axis=0)
    # 2×2 scatter matrix (proportional to covariance)
    cov = (centered.T @ centered) / max(len(path) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, np.argmax(eigenvalues)]   # (2,) [dy, dx]
    return math.degrees(math.atan2(float(principal[0]), float(principal[1]))) % 180.0


def _mean_curvature(path: list[tuple[int, int]]) -> float:
    """
    Mean absolute angular change per step along the path (radians / pixel).

    Computed as the mean of |θ_i − θ_{i-1}| where θ_i is the local
    step direction.  Normalised to [0, π].  Returns 0.0 for paths < 3 pixels.
    """
    if len(path) < 3:
        return 0.0

    deltas: list[float] = []
    prev_angle = math.atan2(
        path[1][0] - path[0][0], path[1][1] - path[0][1]
    )
    for i in range(2, len(path)):
        cur_angle = math.atan2(
            path[i][0] - path[i - 1][0],
            path[i][1] - path[i - 1][1],
        )
        diff = abs(cur_angle - prev_angle)
        if diff > math.pi:
            diff = 2.0 * math.pi - diff
        deltas.append(diff)
        prev_angle = cur_angle

    return float(np.mean(deltas)) if deltas else 0.0


def _segment_bbox(path: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    ys = [p[0] for p in path]
    xs = [p[1] for p in path]
    return int(min(ys)), int(min(xs)), int(max(ys)), int(max(xs))


def _sample_mean_color(
    path: list[tuple[int, int]],
    rgb: np.ndarray,
    stride: int = 3,
) -> tuple[float, float, float]:
    """
    Mean RGB colour sampled at every *stride*-th pixel along the path.
    Reduces per-segment sampling cost on long segments.
    """
    h, w = rgb.shape[:2]
    samples = [
        rgb[y, x].astype(np.float64)
        for idx, (y, x) in enumerate(path)
        if idx % stride == 0 and 0 <= y < h and 0 <= x < w
    ]
    if not samples:
        return (0.0, 0.0, 0.0)
    mean = np.mean(samples, axis=0)
    return (float(mean[0]), float(mean[1]), float(mean[2]))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_geometry(
    rgb_image: np.ndarray,
    repaired_mask: np.ndarray,
    config: Optional[GeometryConfig] = None,
) -> RoadGeometry:
    """
    Full geometry extraction pipeline.

    Completely independent of the U-Net — no model is loaded or called.
    The *repaired_mask* is treated as a read-only input.

    Parameters
    ----------
    rgb_image : np.ndarray
        Original RGB image, shape (H, W, 3), dtype uint8.
        Used for colour sampling only; not modified.
    repaired_mask : np.ndarray
        Binary road mask, shape (H, W), any dtype.
        Typically the output of road_extractor.postprocess.repair_road_mask().
    config : GeometryConfig | None
        Tuning parameters.  Uses defaults when None.

    Returns
    -------
    RoadGeometry
        Structured collection of road segments with full geometric properties.
    """
    cfg = config or GeometryConfig()

    # ------------------------------------------------------------------
    # 1. Clean the binary mask
    # ------------------------------------------------------------------
    clean_mask = clean_binary_mask(
        repaired_mask,
        min_area=cfg.min_road_area_px,
        close_radius=cfg.morphology_close_radius,
        min_small_component_span=cfg.min_small_component_span_px,
        min_small_component_aspect=cfg.min_small_component_aspect,
        context_radius=cfg.small_component_context_radius_px,
    )

    # ------------------------------------------------------------------
    # 2. Extract 1-pixel-wide centerline skeleton
    # ------------------------------------------------------------------
    raw_skeleton = extract_skeleton(clean_mask)

    # ------------------------------------------------------------------
    # 3. Prune short dead-end spurs and tiny isolated fragments
    # ------------------------------------------------------------------
    pruned_skeleton, pruned_spur_pixels = prune_skeleton_spurs(
        raw_skeleton,
        min_branch_length=cfg.min_branch_length_px,
        iterations=cfg.prune_iterations,
        clean_mask=clean_mask,
        support_radius=cfg.branch_support_radius_px,
        hard_spur_length=cfg.hard_spur_length_px,
        legitimate_branch_angle=cfg.legitimate_branch_angle_deg,
        legitimate_branch_mask_support=cfg.legitimate_branch_mask_support,
        legitimate_branch_min_fraction=cfg.legitimate_branch_min_fraction,
    )
    pruned_skeleton, unsupported_branch_pixels = prune_unsupported_branches(
        pruned_skeleton,
        clean_mask,
        max_branch_length=cfg.max_unsupported_branch_length_px,
        angle_tolerance_deg=cfg.unsupported_branch_angle_deg,
        support_radius=cfg.branch_support_radius_px,
    )
    pruned_skeleton, internal_branch_pixels = prune_tiny_internal_branches(
        pruned_skeleton,
        clean_mask,
        max_branch_length=cfg.max_internal_branch_length_px,
        min_mask_support=cfg.internal_branch_mask_support,
        support_radius=cfg.branch_support_radius_px,
    )
    pruned_skeleton, removed_component_pixels = _remove_small_skeleton_components(
        pruned_skeleton,
        min_size=cfg.min_skeleton_component_px,
        min_road_like_pixels=max(10, cfg.min_skeleton_component_px // 2),
        min_road_like_span=cfg.min_small_component_span_px,
    )

    # ------------------------------------------------------------------
    # 4. Bridge endpoint gaps when local & collinear geometry justifies it
    # ------------------------------------------------------------------
    skeleton, bridges_added = bridge_small_gaps(
        pruned_skeleton,
        clean_mask,
        max_gap=cfg.max_gap_bridge_px,
        angle_tolerance_deg=cfg.bridge_angle_tolerance_deg,
        mask_support_ratio=cfg.bridge_mask_support_ratio,
        crossing_clearance=cfg.bridge_crossing_clearance_px,
    )
    if getattr(cfg, "collinear_max_gap_px", 0.0) > 0:
        skeleton, collinear_added = bridge_collinear_dead_ends(
            skeleton,
            max_gap=cfg.collinear_max_gap_px,
            angle_tolerance_deg=cfg.collinear_max_angle_deg,
        )
        bridges_added += collinear_added

    # ------------------------------------------------------------------
    # 5. Classify skeleton pixels → junctions and endpoints
    # ------------------------------------------------------------------
    junctions, endpoints = _classify_pixels(skeleton)
    junction_clusters, junction_cluster_by_pixel = _cluster_pixel_set(junctions)
    junction_representatives = _junction_representatives(junction_clusters)
    special = junctions | endpoints

    # ------------------------------------------------------------------
    # 6. Build distance transform on clean mask (for width estimation)
    # ------------------------------------------------------------------
    dist_transform = _build_distance_transform(clean_mask)

    # ------------------------------------------------------------------
    # 7. Trace all road segments between special pixels
    #    Each edge in the skeleton graph becomes one RoadSegment.
    # ------------------------------------------------------------------
    segments: list[RoadSegment] = []
    visited_edges: set[frozenset] = set()
    seg_id = 0
    h_skel, w_skel = skeleton.shape

    for origin in special:
        oy, ox = origin
        for dy, dx in _N8:
            ny, nx = oy + dy, ox + dx
            if not (0 <= ny < h_skel and 0 <= nx < w_skel):
                continue
            if skeleton[ny, nx] == 0:
                continue
            if _same_junction_cluster(origin, (ny, nx), junction_cluster_by_pixel):
                continue

            edge_key = frozenset({origin, (ny, nx)})
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)

            path = _trace_segment(origin, (ny, nx), skeleton, special)

            # Mark every step along the traced path as visited
            for k in range(len(path) - 1):
                visited_edges.add(frozenset({path[k], path[k + 1]}))

            # Discard very short fragments unless the path is a supported
            # connection between two distinct topology nodes.  A fixed
            # length gate alone removes legitimate short junction arms and
            # short pieces of disconnected roads.
            arc = _arc_length(path)
            start_node = (
                ("junction", junction_cluster_by_pixel[path[0]])
                if path[0] in junction_cluster_by_pixel
                else ("endpoint", path[0])
                if path[0] in endpoints
                else None
            )
            end_node = (
                ("junction", junction_cluster_by_pixel[path[-1]])
                if path[-1] in junction_cluster_by_pixel
                else ("endpoint", path[-1])
                if path[-1] in endpoints
                else None
            )
            connects_distinct_nodes = (
                start_node is not None
                and end_node is not None
                and start_node != end_node
            )
            short_path_support = _local_mask_support(
                path,
                clean_mask,
                cfg.branch_support_radius_px,
            )
            preserve_supported_short = (
                len(path) >= 3
                and connects_distinct_nodes
                and short_path_support >= cfg.legitimate_branch_mask_support
            )
            if (
                (arc < cfg.min_segment_length_px or len(path) < 2)
                and not preserve_supported_short
            ):
                continue
            if _same_junction_cluster(path[0], path[-1], junction_cluster_by_pixel):
                continue

            seg = RoadSegment(
                segment_id=seg_id,
                pixel_path=path,
                length_pixels=arc,
                direction_deg=_dominant_direction(path),
                curvature=_mean_curvature(path),
                width_pixels=_estimate_width(path, dist_transform, cfg.width_sample_stride),
                bbox=_segment_bbox(path),
                mean_color=_sample_mean_color(path, rgb_image),
            )
            segments.append(seg)
            seg_id += 1

    # ------------------------------------------------------------------
    # 8. Assemble and return
    # ------------------------------------------------------------------
    metadata = {
        "image_shape":          list(rgb_image.shape[:2]),
        "input_mask_shape":     list(repaired_mask.shape[:2]),
        "input_road_pixels":    int(np.sum(repaired_mask > 0)),
        "clean_mask_connected_components": _count_connected_components(clean_mask),
        "raw_skeleton_pixels":  int(np.sum(raw_skeleton)),
        "pruned_skeleton_pixels": int(np.sum(skeleton)),
        "skeleton_connected_components": _count_connected_components(skeleton),
        "spurs_removed_pixels": int(
            pruned_spur_pixels + unsupported_branch_pixels + removed_component_pixels
        ),
        "unsupported_branch_pixels_removed": int(unsupported_branch_pixels),
        "internal_branch_pixels_removed": int(internal_branch_pixels),
        "net_skeleton_pixel_delta": int(np.sum(skeleton)) - int(np.sum(raw_skeleton)),
        "dead_end_spur_pixels_removed": int(pruned_spur_pixels),
        "small_component_pixels_removed": int(removed_component_pixels),
        "small_gaps_bridged": int(bridges_added),
        "clean_mask_pixels":    int(np.sum(clean_mask)),
        "segment_count":        len(segments),
        "junction_pixel_count": len(junctions),
        "junction_count":       len(junction_clusters),
        "endpoint_count":       len(endpoints),
        "total_length_pixels":  round(sum(s.length_pixels for s in segments), 2),
        "mean_width_pixels":    round(
            float(np.mean([s.width_pixels for s in segments])) if segments else 0.0, 2
        ),
    }

    return RoadGeometry(
        segments=segments,
        junctions=junction_representatives,
        endpoints=sorted(endpoints),
        metadata=metadata,
        skeleton=skeleton,
        clean_mask=clean_mask,
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def save_geometry_json(geometry: RoadGeometry, path: str | Path) -> None:
    """Write a human-readable JSON summary of a RoadGeometry to *path*."""
    Path(path).write_text(
        json.dumps(geometry.to_summary(), indent=2), encoding="utf-8"
    )


def save_skeleton_image(geometry: RoadGeometry, path: str | Path) -> None:
    """Save the pruned skeleton as a grayscale PNG (white on black)."""
    if geometry.skeleton is not None:
        cv2.imwrite(str(path), geometry.skeleton * 255)


def save_clean_mask_image(geometry: RoadGeometry, path: str | Path) -> None:
    """Save the cleaned binary mask as a grayscale PNG."""
    if geometry.clean_mask is not None:
        cv2.imwrite(str(path), geometry.clean_mask * 255)


def save_geometry_diagnostic(
    rgb_image: np.ndarray,
    geometry: RoadGeometry,
    path: str | Path,
) -> None:
    """Save an RGB diagnostic overlay of segments, endpoints, and junctions."""
    canvas = rgb_image.copy()

    if geometry.skeleton is not None:
        canvas[geometry.skeleton > 0] = (
            0.35 * canvas[geometry.skeleton > 0]
            + 0.65 * np.array([40, 220, 255], dtype=np.uint8)
        ).astype(np.uint8)

    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    palette = [
        (0, 255, 255),
        (255, 160, 0),
        (120, 255, 120),
        (255, 80, 220),
        (80, 160, 255),
    ]

    for segment in geometry.segments:
        color = palette[segment.segment_id % len(palette)]
        points = np.array(
            [[x, y] for y, x in segment.pixel_path],
            dtype=np.int32,
        )
        if len(points) >= 2:
            cv2.polylines(bgr, [points], isClosed=False, color=color, thickness=2)

    for y, x in geometry.endpoints:
        cv2.circle(bgr, (x, y), radius=4, color=(0, 0, 255), thickness=-1)

    for y, x in geometry.junctions:
        cv2.circle(bgr, (x, y), radius=5, color=(255, 0, 0), thickness=2)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr)
