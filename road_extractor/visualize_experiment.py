"""
road_extractor/visualize_experiment.py
======================================
Generates 5-column qualitative comparison visualizations:
  Original Satellite Image | Occluded Test Image | Ground Truth Mask | Original Baseline Model | Occlusion-Trained Model

Demonstrates how the original model breaks under occlusion while the occlusion-trained
model maintains topological continuity and infers hidden road continuation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from road_extractor.data import load_deepglobe_pairs, read_mask, read_rgb, resize_image
from road_extractor.model import build_light_unet
from road_extractor.occlusion import apply_realistic_occlusion


def load_models(baseline_path: str, occlusion_path: str, image_size: int = 256) -> Tuple[tf.keras.Model, tf.keras.Model]:
    m1 = build_light_unet(image_size=image_size, base_filters=16)
    m1.load_weights(baseline_path)

    m2 = build_light_unet(image_size=image_size, base_filters=16)
    m2.load_weights(occlusion_path)

    return m1, m2


def create_comparison_figure(
    original_img: np.ndarray,
    occluded_img: np.ndarray,
    gt_mask: np.ndarray,
    pred_baseline: np.ndarray,
    pred_occlusion: np.ndarray,
    occlusion_mask: np.ndarray,
    output_path: Path,
    title_suffix: str = "",
    threshold: float = 0.40,
) -> None:
    """
    Render a 5-column comparison grid figure:
    Col 1: Original Satellite Image
    Col 2: Occluded Test Image (Obstacle overlaid)
    Col 3: Ground Truth Road Mask (Continuous)
    Col 4: Original Model Output (Broken segment)
    Col 5: Occlusion-Trained Model Output (Continuous bridge)
    """
    bin_base = (pred_baseline >= threshold).astype(np.float32)
    bin_occ = (pred_occlusion >= threshold).astype(np.float32)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), dpi=200)

    # 1. Original Image
    axes[0].imshow(original_img)
    axes[0].set_title("(1) Original Satellite", fontsize=12, fontweight="bold", pad=8)
    axes[0].axis("off")

    # 2. Occluded Image
    axes[1].imshow(occluded_img)
    # Highlight occlusion outline with a red contour
    occ_u8 = (occlusion_mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(occ_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        pts = c.squeeze()
        if len(pts.shape) == 2 and pts.shape[0] > 2:
            axes[1].plot(pts[:, 0], pts[:, 1], color="red", linewidth=1.5, linestyle="--")
    axes[1].set_title("(2) Occluded Input (Simulated)", fontsize=12, fontweight="bold", pad=8)
    axes[1].axis("off")

    # 3. Ground Truth
    axes[2].imshow(gt_mask, cmap="gray")
    axes[2].set_title("(3) Ground Truth Road Mask", fontsize=12, fontweight="bold", pad=8)
    axes[2].axis("off")

    # 4. Baseline Prediction
    # Overlay prediction in green, highlight break gaps in red
    base_overlay = np.zeros((*bin_base.shape, 3), dtype=np.float32)
    base_overlay[..., 0] = bin_base * 0.9  # Green channel
    base_overlay[..., 1] = bin_base * 0.9
    base_overlay[..., 2] = bin_base * 0.9
    # Highlight missing road in occlusion as red
    missing_in_occ = np.logical_and(gt_mask > 0.5, bin_base < 0.5)
    missing_in_occ = np.logical_and(missing_in_occ, occlusion_mask > 0)
    
    rgb_base = np.zeros((*bin_base.shape, 3), dtype=np.float32)
    rgb_base[bin_base > 0] = [0.1, 0.9, 0.2]  # Bright green
    rgb_base[missing_in_occ] = [1.0, 0.2, 0.2] # Bright red for broken gap

    axes[3].imshow(rgb_base)
    axes[3].set_title("(4) Original Model (Broken)", fontsize=12, fontweight="bold", color="darkred", pad=8)
    axes[3].axis("off")

    # 5. Occlusion-Trained Model
    rgb_occ = np.zeros((*bin_occ.shape, 3), dtype=np.float32)
    rgb_occ[bin_occ > 0] = [0.1, 0.9, 0.2]     # Bright green
    # Restored road in occlusion zone
    restored_in_occ = np.logical_and(bin_occ > 0.5, occlusion_mask > 0)
    restored_in_occ = np.logical_and(restored_in_occ, gt_mask > 0.5)
    rgb_occ[restored_in_occ] = [0.2, 0.8, 1.0] # Cyan highlight for successfully bridged gap

    axes[5 - 1].imshow(rgb_occ)
    axes[5 - 1].set_title("(5) Occlusion Model (Connected)", fontsize=12, fontweight="bold", color="darkgreen", pad=8)
    axes[5 - 1].axis("off")

    plt.suptitle(
        f"Road Continuity under Occlusion Comparison {title_suffix}\n"
        "[Green: Predicted Road | Red: Disconnected Gap in Occlusion | Cyan: Reconnected Bridge across Occlusion]",
        fontsize=13,
        y=1.03,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()


def generate_qualitative_suite(
    deepglobe_root: str = "dataset",
    baseline_model_path: str = "models/road_unet.keras",
    occlusion_model_path: str = "models/road_unet_occlusion.keras",
    output_dir: str = "outputs/experiments/qualitative_comparisons",
    num_samples: int = 10,
    image_size: int = 256,
    threshold: float = 0.40,
    seed: int = 42,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  Generating Qualitative 5-Panel Comparison Figures")
    print("=" * 65)

    # Load validation pairs
    _, val_pairs = load_deepglobe_pairs(deepglobe_root, val_fraction=0.2, seed=seed)

    # Load models
    m_base, m_occ = load_models(baseline_model_path, occlusion_model_path, image_size)

    rng = np.random.default_rng(seed)

    # Filter for interesting samples that have clear road networks
    selected_indices = []
    for idx, (sat_p, msk_p) in enumerate(val_pairs):
        msk = cv2.imread(msk_p, cv2.IMREAD_GRAYSCALE)
        if msk is not None and (msk > 127).sum() > (256 * 256 * 0.04): # Has at least 4% road pixels
            selected_indices.append(idx)
        if len(selected_indices) >= num_samples * 2:
            break

    sampled_indices = selected_indices[:num_samples]
    print(f"Selected {len(sampled_indices)} rich road network validation samples for visualization.")

    for i, idx in enumerate(sampled_indices):
        sat_p, msk_p = val_pairs[idx]
        img_id = Path(sat_p).stem.replace("_sat", "")

        img_raw = read_rgb(sat_p)
        mask_raw = read_mask(msk_p)

        img_norm = resize_image(img_raw, image_size).astype(np.float32) / 255.0
        mask_bin = resize_image(mask_raw, image_size)
        mask_bin = (mask_bin > 0.5).astype(np.float32)

        # Apply deterministic occlusion
        occ_img, occ_mask = apply_realistic_occlusion(
            img_norm,
            road_mask=mask_bin,
            p=1.0,
            max_obstacles=3,
            min_size=16,
            max_size=36,
            rng=rng,
        )

        # Predict with both models on occluded image
        pred_base = m_base.predict(occ_img[np.newaxis, ...], verbose=0)[0, :, :, 0]
        pred_occ = m_occ.predict(occ_img[np.newaxis, ...], verbose=0)[0, :, :, 0]

        save_path = out_dir / f"comparison_{i+1:02d}_{img_id}.png"
        create_comparison_figure(
            original_img=img_norm,
            occluded_img=occ_img,
            gt_mask=mask_bin,
            pred_baseline=pred_base,
            pred_occlusion=pred_occ,
            occlusion_mask=occ_mask,
            output_path=save_path,
            title_suffix=f"(Sample #{i+1} — ID: {img_id})",
            threshold=threshold,
        )
        print(f"  [{i+1}/{len(sampled_indices)}] Generated: {save_path.name}")

    print(f"\n✓ All {len(sampled_indices)} comparison figures saved to: {out_dir}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 5-Column Qualitative Visualizations")
    parser.add_argument("--deepglobe", default="dataset")
    parser.add_argument("--baseline", default="models/road_unet.keras")
    parser.add_argument("--occlusion-model", default="models/road_unet_occlusion.keras")
    parser.add_argument("--output", default="outputs/experiments/qualitative_comparisons")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.40)
    args = parser.parse_args()

    generate_qualitative_suite(
        deepglobe_root=args.deepglobe,
        baseline_model_path=args.baseline,
        occlusion_model_path=args.occlusion_model,
        output_dir=args.output,
        num_samples=args.num_samples,
        image_size=args.image_size,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
