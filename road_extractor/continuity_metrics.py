"""
road_extractor/continuity_metrics.py
====================================
Comprehensive metric evaluation suite for road segmentation, topological continuity,
and gap bridging under occlusion.

Computes:
  - Standard pixel metrics: IoU, F1/Dice, Precision, Recall, Accuracy
  - Continuity metrics:
      * Connected Component counts (N_CC) in GT vs Prediction
      * Fragmentation Ratio: N_CC(Pred) / N_CC(GT)
      * Broken Segments Count & Percentage: GT components split into multiple fragments
      * Largest Connected Component (LCC) preservation ratio
      * False Dead-End / Endpoint count on skeletonized road graph
      * Occlusion Gap Recall & IoU: Performance exclusively inside the occluded zones
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
from skimage.morphology import skeletonize


@dataclass
class ContinuityMetrics:
    # Pixel-level metrics
    iou: float
    dice_f1: float
    precision: float
    recall: float
    accuracy: float
    
    # Continuity & Fragmentation metrics
    gt_components: int
    pred_components: int
    fragmentation_ratio: float      # Pred CC / GT CC (1.0 = ideal, > 1.0 = over-fragmented)
    broken_segments_count: int     # Number of continuous GT roads split in prediction
    broken_segments_pct: float     # Percentage of continuous GT roads split
    lcc_ratio_gt: float            # Largest CC / Total road area for GT
    lcc_ratio_pred: float          # Largest CC / Total road area for Pred
    lcc_preservation: float        # Pred LCC / GT LCC
    
    # Topological skeleton endpoints (degree 1 nodes)
    gt_endpoints: int
    pred_endpoints: int
    excess_endpoints: int          # pred_endpoints - gt_endpoints (extra false dead-ends)
    
    # Occlusion-specific metrics (if occlusion mask provided)
    has_occlusion: bool = False
    gap_recall: float = 0.0        # Recall inside occluded zones where GT road exists
    gap_iou: float = 0.0           # IoU inside occluded zones
    gap_precision: float = 0.0     # Precision inside occluded zones


def compute_pixel_metrics(
    pred_binary: np.ndarray,
    gt_binary: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[float, float, float, float, float]:
    """Compute IoU, Dice/F1, Precision, Recall, and Accuracy on binary masks."""
    p = pred_binary > 0
    g = gt_binary > 0

    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, np.logical_not(g)).sum()
    fn = np.logical_and(np.logical_not(p), g).sum()
    tn = np.logical_and(np.logical_not(p), np.logical_not(g)).sum()

    intersection = float(tp)
    union = float(tp + fp + fn)

    iou = (intersection + eps) / (union + eps)
    dice = (2.0 * intersection + eps) / (2.0 * intersection + fp + fn + eps)
    precision = (intersection + eps) / (intersection + fp + eps)
    recall = (intersection + eps) / (intersection + fn + eps)
    accuracy = float(tp + tn) / float(p.size)

    return float(iou), float(dice), float(precision), float(recall), float(accuracy)


def get_connected_components(binary_mask: np.ndarray, min_size: int = 8) -> Tuple[int, np.ndarray, List[int]]:
    """
    Find connected components in binary mask, filtering out isolated specks smaller than min_size.
    Returns (num_components, labeled_mask, list_of_areas).
    """
    mask_u8 = (binary_mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    valid_labels = []
    filtered_labeled = np.zeros_like(labels)
    areas = []
    
    new_label_idx = 1
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_size:
            valid_labels.append(label)
            filtered_labeled[labels == label] = new_label_idx
            areas.append(area)
            new_label_idx += 1

    return len(valid_labels), filtered_labeled, areas


def count_skeleton_endpoints(binary_mask: np.ndarray) -> int:
    """
    Skeletonize binary road mask and count degree-1 endpoints (dead-ends/break tips).
    Each break in a continuous road creates two new endpoints.
    """
    if binary_mask.sum() == 0:
        return 0

    skel = skeletonize(binary_mask > 0).astype(np.uint8)
    # 3x3 convolution to count 8-connected neighbours
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel, -1, kernel)
    # Degree-1 endpoints: point on skeleton with exactly 1 neighbor
    endpoints = np.logical_and(skel == 1, neighbor_count == 1)
    return int(endpoints.sum())


def evaluate_continuity_single(
    pred_prob: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 0.40,
    occlusion_mask: Optional[np.ndarray] = None,
    min_component_size: int = 8,
) -> ContinuityMetrics:
    """
    Evaluate comprehensive continuity metrics on a single image pair.

    Parameters
    ----------
    pred_prob : np.ndarray
        Predicted road probability map (H, W) in [0, 1].
    gt_mask : np.ndarray
        Ground truth binary road mask (H, W).
    threshold : float
        Decision threshold for binarizing pred_prob.
    occlusion_mask : np.ndarray, optional
        Binary mask (H, W) where 1 indicates an occluded zone.
    min_component_size : int
        Minimum pixel area to be considered a real road component.
    """
    pred_prob = np.squeeze(pred_prob)
    gt_mask = (np.squeeze(gt_mask) > 0.5).astype(np.uint8)
    pred_binary = (pred_prob >= threshold).astype(np.uint8)

    # 1. Standard pixel metrics
    iou, dice, prec, rec, acc = compute_pixel_metrics(pred_binary, gt_mask)

    # 2. Connected component analysis
    n_gt, labeled_gt, areas_gt = get_connected_components(gt_mask, min_size=min_component_size)
    n_pred, labeled_pred, areas_pred = get_connected_components(pred_binary, min_size=min_component_size)

    frag_ratio = float(n_pred) / float(max(1, n_gt))

    # Largest connected component ratios
    total_area_gt = sum(areas_gt) if areas_gt else 0
    total_area_pred = sum(areas_pred) if areas_pred else 0
    max_area_gt = max(areas_gt) if areas_gt else 0
    max_area_pred = max(areas_pred) if areas_pred else 0

    lcc_ratio_gt = (max_area_gt / total_area_gt) if total_area_gt > 0 else 0.0
    lcc_ratio_pred = (max_area_pred / total_area_pred) if total_area_pred > 0 else 0.0
    lcc_preservation = (lcc_ratio_pred / lcc_ratio_gt) if lcc_ratio_gt > 0 else 0.0

    # Broken segments: check each GT component to see how many distinct predicted CCs intersect it
    broken_segments = 0
    for gt_lbl in range(1, n_gt + 1):
        gt_comp = labeled_gt == gt_lbl
        intersecting_preds = np.unique(labeled_pred[gt_comp])
        intersecting_preds = [p for p in intersecting_preds if p > 0]
        if len(intersecting_preds) >= 2:
            broken_segments += 1

    broken_pct = (float(broken_segments) / float(n_gt) * 100.0) if n_gt > 0 else 0.0

    # 3. Skeleton endpoints
    gt_endpoints = count_skeleton_endpoints(gt_mask)
    pred_endpoints = count_skeleton_endpoints(pred_binary)
    excess_endpoints = pred_endpoints - gt_endpoints

    # 4. Occlusion gap metrics (if applicable)
    has_occ = False
    gap_rec = 0.0
    gap_iou = 0.0
    gap_prec = 0.0

    if occlusion_mask is not None:
        occ_zone = (np.squeeze(occlusion_mask) > 0.5)
        if occ_zone.any():
            has_occ = True
            gt_in_occ = np.logical_and(gt_mask > 0, occ_zone)
            pred_in_occ = np.logical_and(pred_binary > 0, occ_zone)

            tp_occ = np.logical_and(pred_in_occ, gt_in_occ).sum()
            fp_occ = np.logical_and(pred_in_occ, np.logical_not(gt_in_occ)).sum()
            fn_occ = np.logical_and(np.logical_not(pred_in_occ), gt_in_occ).sum()

            eps = 1e-6
            gap_rec = float(tp_occ + eps) / float(tp_occ + fn_occ + eps)
            gap_iou = float(tp_occ + eps) / float(tp_occ + fp_occ + fn_occ + eps)
            gap_prec = float(tp_occ + eps) / float(tp_occ + fp_occ + eps)

    return ContinuityMetrics(
        iou=iou,
        dice_f1=dice,
        precision=prec,
        recall=rec,
        accuracy=acc,
        gt_components=n_gt,
        pred_components=n_pred,
        fragmentation_ratio=frag_ratio,
        broken_segments_count=broken_segments,
        broken_segments_pct=broken_pct,
        lcc_ratio_gt=float(lcc_ratio_gt),
        lcc_ratio_pred=float(lcc_ratio_pred),
        lcc_preservation=float(lcc_preservation),
        gt_endpoints=gt_endpoints,
        pred_endpoints=pred_endpoints,
        excess_endpoints=excess_endpoints,
        has_occlusion=has_occ,
        gap_recall=gap_rec,
        gap_iou=gap_iou,
        gap_precision=gap_prec,
    )


def aggregate_metrics(metrics_list: List[ContinuityMetrics]) -> Dict[str, Any]:
    """Average metrics over an entire dataset / benchmark."""
    if not metrics_list:
        return {}

    n = len(metrics_list)
    keys_to_avg = [
        "iou", "dice_f1", "precision", "recall", "accuracy",
        "gt_components", "pred_components", "fragmentation_ratio",
        "broken_segments_pct", "lcc_ratio_gt", "lcc_ratio_pred", "lcc_preservation",
        "gt_endpoints", "pred_endpoints", "excess_endpoints",
    ]

    agg: Dict[str, Any] = {"num_samples": n}
    for k in keys_to_avg:
        vals = [getattr(m, k) for m in metrics_list]
        agg[k] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))

    # Total broken segments across entire test set
    agg["total_broken_segments"] = int(sum(m.broken_segments_count for m in metrics_list))

    # Occlusion-specific averages (for samples that had occlusions)
    occ_samples = [m for m in metrics_list if m.has_occlusion]
    agg["num_occluded_samples"] = len(occ_samples)
    if occ_samples:
        agg["gap_recall"] = float(np.mean([m.gap_recall for m in occ_samples]))
        agg["gap_iou"] = float(np.mean([m.gap_iou for m in occ_samples]))
        agg["gap_precision"] = float(np.mean([m.gap_precision for m in occ_samples]))
    else:
        agg["gap_recall"] = 0.0
        agg["gap_iou"] = 0.0
        agg["gap_precision"] = 0.0

    return agg
