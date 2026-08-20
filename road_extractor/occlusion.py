"""
road_extractor/occlusion.py
===========================
Realistic occlusion synthesis and augmentation module for satellite imagery road extraction.

Simulates real-world obstacles (tree canopies, buildings, shadows, irregular obstacles)
on satellite images while keeping the ground-truth road masks 100% intact.
This forces the neural network to learn spatial geometric continuity and road continuation
under occlusion.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import cv2
import numpy as np


def _create_foliage_texture(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a realistic foliage/canopy color patch with multi-frequency green/brown noise."""
    # Base green/olive/forest tones
    base_r = rng.integers(30, 80)
    base_g = rng.integers(60, 130)
    base_b = rng.integers(25, 75)
    
    patch = np.zeros((h, w, 3), dtype=np.float32)
    patch[..., 0] = base_r
    patch[..., 1] = base_g
    patch[..., 2] = base_b

    # Add textural noise
    noise_fine = rng.normal(0, 15, (h, w, 3)).astype(np.float32)
    noise_coarse = cv2.resize(
        rng.normal(0, 25, (max(1, h // 4), max(1, w // 4), 3)).astype(np.float32),
        (w, h),
        interpolation=cv2.INTER_CUBIC,
    )
    patch = np.clip(patch + noise_fine + noise_coarse, 0, 255)
    return patch.astype(np.uint8)


def _create_building_texture(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a realistic urban roof/building texture (concrete, terracotta, metallic gray)."""
    palette_type = rng.choice(["concrete", "terracotta", "slate", "light_gray"])
    if palette_type == "concrete":
        base = np.array([rng.integers(130, 170), rng.integers(130, 170), rng.integers(125, 165)], dtype=np.float32)
    elif palette_type == "terracotta":
        base = np.array([rng.integers(150, 200), rng.integers(70, 110), rng.integers(50, 90)], dtype=np.float32)
    elif palette_type == "slate":
        base = np.array([rng.integers(60, 90), rng.integers(70, 100), rng.integers(80, 115)], dtype=np.float32)
    else:
        base = np.array([rng.integers(170, 220), rng.integers(170, 220), rng.integers(170, 220)], dtype=np.float32)

    patch = np.tile(base, (h, w, 1))
    noise = rng.normal(0, 10, (h, w, 3)).astype(np.float32)
    patch = np.clip(patch + noise, 0, 255)
    return patch.astype(np.uint8)


def add_tree_canopy_occlusion(
    image: np.ndarray,
    center_y: int,
    center_x: int,
    radius: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthesize an organic tree canopy cluster over the image.
    Returns (modified_image, single_obstacle_mask).
    """
    h, w = image.shape[:2]
    obstacle_mask = np.zeros((h, w), dtype=np.uint8)

    # Multi-lobed cluster: 2 to 4 overlapping circles/ellipses
    num_subblobs = rng.integers(2, 5)
    for _ in range(num_subblobs):
        off_y = rng.integers(-radius // 2, radius // 2 + 1)
        off_x = rng.integers(-radius // 2, radius // 2 + 1)
        cy = np.clip(center_y + off_y, 0, h - 1)
        cx = np.clip(center_x + off_x, 0, w - 1)
        r_y = rng.integers(max(4, radius // 2), radius + 1)
        r_x = rng.integers(max(4, radius // 2), radius + 1)
        angle = rng.integers(0, 180)
        cv2.ellipse(obstacle_mask, (int(cx), int(cy)), (int(r_x), int(r_y)), int(angle), 0, 360, 255, -1)

    # Soften outer boundary with slight blur
    obstacle_mask = cv2.GaussianBlur(obstacle_mask, (5, 5), 0)
    alpha = (obstacle_mask > 20).astype(np.float32) * (obstacle_mask.astype(np.float32) / 255.0)
    alpha = np.clip(alpha * rng.uniform(0.85, 1.0), 0.0, 1.0)[..., np.newaxis]

    foliage_patch = _create_foliage_texture(h, w, rng)
    output_image = (image.astype(np.float32) * (1.0 - alpha) + foliage_patch.astype(np.float32) * alpha)
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)

    binary_mask = (obstacle_mask > 80).astype(np.uint8)
    return output_image, binary_mask


def add_building_occlusion(
    image: np.ndarray,
    center_y: int,
    center_x: int,
    size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthesize an angled rectangular/polygonal building roof structure.
    Returns (modified_image, single_obstacle_mask).
    """
    h, w = image.shape[:2]
    obstacle_mask = np.zeros((h, w), dtype=np.uint8)

    bw = rng.integers(max(8, size // 2), size + 1)
    bh = rng.integers(max(8, size // 2), size + 1)
    angle = rng.uniform(0, 180)

    # Create rotated rectangle polygon
    rect = ((float(center_x), float(center_y)), (float(bw), float(bh)), float(angle))
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.fillPoly(obstacle_mask, [box], 255)

    building_patch = _create_building_texture(h, w, rng)
    alpha = (obstacle_mask > 127).astype(np.float32)[..., np.newaxis]

    output_image = (image.astype(np.float32) * (1.0 - alpha) + building_patch.astype(np.float32) * alpha)
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)

    binary_mask = (obstacle_mask > 127).astype(np.uint8)
    return output_image, binary_mask


def add_shadow_occlusion(
    image: np.ndarray,
    center_y: int,
    center_x: int,
    length: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthesize a cast shadow strip across the image.
    Darkens existing satellite imagery by 45%-75% with soft edges.
    Returns (modified_image, single_obstacle_mask).
    """
    h, w = image.shape[:2]
    obstacle_mask = np.zeros((h, w), dtype=np.float32)

    width = rng.integers(max(6, length // 3), max(8, length // 2))
    angle = rng.uniform(0, 180)

    rect = ((float(center_x), float(center_y)), (float(width), float(length)), float(angle))
    box = np.int32(cv2.boxPoints(rect))
    cv2.fillPoly(obstacle_mask, [box], 1.0)
    obstacle_mask = cv2.GaussianBlur(obstacle_mask, (7, 7), 0)

    shadow_intensity = rng.uniform(0.35, 0.65)
    attenuation = 1.0 - (obstacle_mask * (1.0 - shadow_intensity))
    attenuation = attenuation[..., np.newaxis]

    output_image = np.clip(image.astype(np.float32) * attenuation, 0, 255).astype(np.uint8)
    binary_mask = (obstacle_mask > 0.4).astype(np.uint8)
    return output_image, binary_mask


def apply_realistic_occlusion(
    image: np.ndarray,
    road_mask: Optional[np.ndarray] = None,
    p: float = 0.40,
    max_obstacles: int = 3,
    min_size: int = 14,
    max_size: int = 36,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply realistic occlusion augmentation to a satellite image.

    Parameters
    ----------
    image : np.ndarray
        RGB satellite image, shape (H, W, 3), dtype uint8 [0, 255] or float32 [0, 1].
    road_mask : np.ndarray, optional
        Binary road mask, shape (H, W) or (H, W, 1). If provided, obstacles are
        preferentially placed across existing road segments to simulate real-world
        road occlusions.
    p : float
        Probability of applying occlusion augmentation to this image.
    max_obstacles : int
        Maximum number of distinct obstacles placed per image (1 to max_obstacles).
    min_size : int
        Minimum radius/dimension of each obstacle in pixels.
    max_size : int
        Maximum radius/dimension of each obstacle in pixels.
    rng : np.random.Generator, optional
        Random number generator instance.

    Returns
    -------
    occluded_image : np.ndarray
        The augmented image in the same dtype and range as the input.
    total_occlusion_mask : np.ndarray
        Binary mask (H, W) uint8 where 1 indicates an occluded region.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Convert to uint8 [0, 255] for OpenCV operations if float32
    is_float = (image.dtype == np.float32 or image.dtype == np.float64)
    if is_float:
        img_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    else:
        img_uint8 = image.copy()

    h, w = img_uint8.shape[:2]
    total_occ_mask = np.zeros((h, w), dtype=np.uint8)

    if rng.random() > p:
        # No augmentation applied
        return image.copy(), total_occ_mask

    # Find candidate coordinates on road if mask provided
    road_coords = []
    if road_mask is not None:
        mask_sq = np.squeeze(road_mask) > 0.5
        ys, xs = np.where(mask_sq)
        if len(ys) > 0:
            road_coords = list(zip(ys, xs))

    num_obstacles = rng.integers(1, max_obstacles + 1)

    for _ in range(num_obstacles):
        # 75% chance to target an actual road pixel, 25% random location
        if road_coords and (rng.random() < 0.75):
            idx = rng.integers(0, len(road_coords))
            cy, cx = road_coords[idx]
        else:
            cy = rng.integers(min_size, h - min_size)
            cx = rng.integers(min_size, w - min_size)

        size = rng.integers(min_size, max_size + 1)
        obs_type = rng.choice(["tree", "building", "shadow"], p=[0.45, 0.35, 0.20])

        if obs_type == "tree":
            img_uint8, occ_m = add_tree_canopy_occlusion(img_uint8, cy, cx, size, rng)
        elif obs_type == "building":
            img_uint8, occ_m = add_building_occlusion(img_uint8, cy, cx, size, rng)
        else:
            img_uint8, occ_m = add_shadow_occlusion(img_uint8, cy, cx, size * 2, rng)

        total_occ_mask = np.bitwise_or(total_occ_mask, occ_m)

    if is_float:
        result_img = img_uint8.astype(np.float32) / 255.0
    else:
        result_img = img_uint8

    return result_img, total_occ_mask


def py_realistic_occlusion_aug(
    image_np: np.ndarray,
    mask_np: np.ndarray,
    p: float = 0.40,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numpy wrapper for tf.py_function / tf.numpy_function in TensorFlow data pipelines.
    Image is float32 [0, 1], mask is float32 [0, 1].
    The mask remains unchanged; only the image is occluded.
    """
    rng = np.random.default_rng()
    occluded_image, _ = apply_realistic_occlusion(
        image_np,
        road_mask=mask_np,
        p=p,
        max_obstacles=3,
        min_size=12,
        max_size=32,
        rng=rng,
    )
    return occluded_image.astype(np.float32), mask_np.astype(np.float32)
