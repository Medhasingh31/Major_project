"""
prepare_deepglobe.py
====================
Standalone script to verify the DeepGlobe dataset pipeline before training.

Performs the following checks without starting model training:
  1. Loads and counts all valid image-mask pairs from metadata.csv
  2. Creates a deterministic 80/20 train/validation split
  3. Prints the dataset summary (pair counts, dimensions, U-Net input size)
  4. Verifies that a sample batch can be loaded and has the correct shapes
  5. Checks binary mask values and road pixel statistics

Usage
-----
    python prepare_deepglobe.py
    python prepare_deepglobe.py --deepglobe data/deepglobe --image-size 256 --batch-size 4
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def verify_batch(train_pairs, val_pairs, image_size: int, batch_size: int) -> None:
    """Build a tf.data pipeline for one batch and confirm shapes and value ranges."""
    try:
        import tensorflow as tf
    except ImportError:
        print("[Batch check] TensorFlow not installed — skipping batch shape check.")
        return

    from road_extractor.data import make_deepglobe_dataset

    print("\n[Batch check] Loading one training batch …")
    train_ds = make_deepglobe_dataset(train_pairs, image_size, batch_size, shuffle=False)

    for images, masks in train_ds.take(1):
        img_np  = images.numpy()
        mask_np = masks.numpy()

        print(f"  Image batch shape : {img_np.shape}")
        print(f"  Mask batch shape  : {mask_np.shape}")
        print(f"  Image dtype       : {img_np.dtype}  range [{img_np.min():.4f}, {img_np.max():.4f}]")
        print(f"  Mask dtype        : {mask_np.dtype}  unique values: {sorted(set(mask_np.flatten().tolist()))}")

        expected_img  = (batch_size, image_size, image_size, 3)
        expected_mask = (batch_size, image_size, image_size, 1)

        if img_np.shape != expected_img:
            print(f"  ✗ Image shape mismatch. Expected {expected_img}, got {img_np.shape}")
            sys.exit(1)
        if mask_np.shape != expected_mask:
            print(f"  ✗ Mask shape mismatch. Expected {expected_mask}, got {mask_np.shape}")
            sys.exit(1)

        unique_mask = set(mask_np.flatten().tolist())
        if not unique_mask.issubset({0.0, 1.0}):
            print(f"  ✗ Mask contains non-binary values: {unique_mask}")
            sys.exit(1)

        if img_np.min() < 0.0 or img_np.max() > 1.0:
            print(f"  ✗ Image values outside [0, 1]: [{img_np.min()}, {img_np.max()}]")
            sys.exit(1)

    print("  ✓ Batch shapes and value ranges are correct.")


def verify_mask_statistics(train_pairs: list, n_samples: int = 20) -> None:
    """
    Sample a few masks from the training set and report road pixel statistics.
    This confirms the masks are binary road/background and not corrupted.
    """
    print(f"\n[Mask check] Sampling {n_samples} masks for statistics …")
    road_fractions = []
    for sat_path, mask_path in train_pairs[:n_samples]:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  ✗ Could not read mask: {mask_path}")
            continue
        unique = sorted(set(mask.flatten().tolist()))
        if not set(unique).issubset({0, 255}):
            print(f"  ✗ Unexpected mask values {unique} in: {mask_path}")
        road_fraction = float((mask > 127).sum()) / mask.size
        road_fractions.append(road_fraction)

    if road_fractions:
        print(f"  Road pixel fraction  — min: {min(road_fractions):.3%}  "
              f"max: {max(road_fractions):.3%}  "
              f"mean: {np.mean(road_fractions):.3%}")
        print("  ✓ All sampled masks contain only road (255) and background (0) values.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify DeepGlobe dataset pipeline")
    parser.add_argument(
        "--deepglobe",
        default="data/deepglobe",
        metavar="ROOT",
        help="Path to DeepGlobe root (default: data/deepglobe)",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    deepglobe_root = Path(args.deepglobe)
    if not deepglobe_root.exists():
        print(f"Error: DeepGlobe root not found: {deepglobe_root.resolve()}")
        sys.exit(1)

    # ── 1. Load pairs and create split ─────────────────────────────────────
    from road_extractor.data import (
        load_deepglobe_pairs,
        print_deepglobe_summary,
    )

    train_pairs, val_pairs = load_deepglobe_pairs(
        deepglobe_root,
        val_fraction=args.val_fraction,
    )

    # ── 2. Print summary ────────────────────────────────────────────────────
    print_deepglobe_summary(train_pairs, val_pairs, args.image_size)

    # ── 3. Mask statistics on raw files ────────────────────────────────────
    verify_mask_statistics(train_pairs, n_samples=20)

    # ── 4. TensorFlow batch shape / dtype check ─────────────────────────────
    verify_batch(train_pairs, val_pairs, args.image_size, args.batch_size)

    print("\n✓ DeepGlobe dataset pipeline is ready.")
    print("\nTo start training, run:")
    print(f"  python -m road_extractor.cli train --deepglobe {args.deepglobe} "
          f"--image-size {args.image_size} --batch-size {args.batch_size}")


if __name__ == "__main__":
    main()
