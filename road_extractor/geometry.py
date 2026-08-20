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

    # --- spur pruning ---
    min_branch_length_px: int = 10
    """Skeleton branches shorter than this (in pixels) are removed as spurs."""

    prune_iterations: int = 10
    """Maximum number of pruning passes (each pass may shorten or remove spurs)."""

    # --- segment filtering ---
    min_segment_length_px: float = 8.0
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
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == lbl] = 1

    # --- 3. morphological closing to fill internal holes ---
    k = close_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

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


def _spur_length(
    tip_y: int,
    tip_x: int,
    skel: np.ndarray,
    max_steps: int,
) -> int:
    """
    Walk from a degree-1 pixel (tip) towards the junction, counting steps.
    Stops when a junction pixel (degree ≥ 3) or another endpoint is reached
    or after *max_steps* steps.  Returns the walk length.
    """
    prev = (-1, -1)
    cur = (tip_y, tip_x)
    steps = 0
    while steps < max_steps:
        nbrs = [n for n in _skeleton_neighbors(*cur, skel) if n != prev]
        if not nbrs:
            break
        nxt = nbrs[0]
        steps += 1
        deg = len(_skeleton_neighbors(*nxt, skel))
        if deg >= 3:
            break                 # reached a junction
        if deg == 1 and nxt != (tip_y, tip_x):
            break                 # reached another endpoint
        prev, cur = cur, nxt
    return steps


def prune_skeleton_spurs(
    skeleton: np.ndarray,
    min_branch_length: int = 10,
    iterations: int = 10,
) -> np.ndarray:
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

            length = _spur_length(ty, tx, skel, max_steps=min_branch_length)
            if length < min_branch_length:
                # Erase the spur back from the tip to the junction
                prev = (-1, -1)
                cur = (ty, tx)
                for _ in range(length):
                    skel[cur[0], cur[1]] = 0
                    nbrs = [
                        n for n in _skeleton_neighbors(*cur, skel) if n != prev
                    ]
                    if not nbrs:
                        break
                    # Stop just before a junction
                    if len(_skeleton_neighbors(*nbrs[0], skel)) >= 3:
                        break
                    prev, cur = cur, nbrs[0]
                changed = True

        if not changed:
            break

    return skel


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


# ---------------------------------------------------------------------------
# Step 5 — Segment tracing
# ---------------------------------------------------------------------------

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
    )

    # ------------------------------------------------------------------
    # 2. Extract 1-pixel-wide centerline skeleton
    # ------------------------------------------------------------------
    raw_skeleton = extract_skeleton(clean_mask)

    # ------------------------------------------------------------------
    # 3. Prune short dead-end spurs (noisy branches)
    # ------------------------------------------------------------------
    skeleton = prune_skeleton_spurs(
        raw_skeleton,
        min_branch_length=cfg.min_branch_length_px,
        iterations=cfg.prune_iterations,
    )

    # ------------------------------------------------------------------
    # 4. Classify skeleton pixels → junctions and endpoints
    # ------------------------------------------------------------------
    junctions, endpoints = _classify_pixels(skeleton)
    special = junctions | endpoints

    # ------------------------------------------------------------------
    # 5. Build distance transform on clean mask (for width estimation)
    # ------------------------------------------------------------------
    dist_transform = _build_distance_transform(clean_mask)

    # ------------------------------------------------------------------
    # 6. Trace all road segments between special pixels
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

            edge_key = frozenset({origin, (ny, nx)})
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)

            path = _trace_segment(origin, (ny, nx), skeleton, special)

            # Mark every step along the traced path as visited
            for k in range(len(path) - 1):
                visited_edges.add(frozenset({path[k], path[k + 1]}))

            # Discard very short fragments (noise)
            arc = _arc_length(path)
            if arc < cfg.min_segment_length_px or len(path) < 2:
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
    # 7. Assemble and return
    # ------------------------------------------------------------------
    metadata = {
        "image_shape":          list(rgb_image.shape[:2]),
        "raw_skeleton_pixels":  int(np.sum(raw_skeleton)),
        "pruned_skeleton_pixels": int(np.sum(skeleton)),
        "spurs_removed_pixels": int(np.sum(raw_skeleton)) - int(np.sum(skeleton)),
        "clean_mask_pixels":    int(np.sum(clean_mask)),
        "segment_count":        len(segments),
        "junction_count":       len(junctions),
        "endpoint_count":       len(endpoints),
        "total_length_pixels":  round(sum(s.length_pixels for s in segments), 2),
        "mean_width_pixels":    round(
            float(np.mean([s.width_pixels for s in segments])) if segments else 0.0, 2
        ),
    }

    return RoadGeometry(
        segments=segments,
        junctions=sorted(junctions),
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
