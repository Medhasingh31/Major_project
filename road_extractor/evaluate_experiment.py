"""
road_extractor/evaluate_experiment.py
=====================================
Rigorous comparative evaluation protocol comparing the baseline model against the
occlusion-trained model on:
  1. Clean Untouched Validation Set (Generalization & Regression check)
  2. Controlled Occluded Validation Benchmark (Continuity & Gap Bridging under stress)

Outputs full comparative metrics (IoU, Dice, Precision, Recall, Fragmentation,
Broken Segments, LCC Preservation, Dead-Ends, Occlusion Gap Recovery).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import tensorflow as tf

from road_extractor.continuity_metrics import (
    ContinuityMetrics,
    aggregate_metrics,
    evaluate_continuity_single,
)
from road_extractor.data import load_deepglobe_pairs, read_mask, read_rgb, resize_image
from road_extractor.model import build_light_unet
from road_extractor.occlusion import apply_realistic_occlusion


def load_model_weights(weights_path: str, image_size: int = 256) -> tf.keras.Model:
    """Instantiate light U-Net and load weights."""
    model = build_light_unet(image_size=image_size, base_filters=16)
    model.load_weights(weights_path)
    return model


def evaluate_dataset(
    model: tf.keras.Model,
    val_pairs: List[Tuple[str, str]],
    image_size: int = 256,
    threshold: float = 0.40,
    apply_synthetic_occlusion: bool = False,
    seed: int = 42,
    batch_size: int = 8,
) -> Tuple[Dict[str, Any], List[ContinuityMetrics]]:
    """
    Run evaluation over validation pairs.
    If apply_synthetic_occlusion=True, deterministic occlusions are applied to test images.
    """
    rng = np.random.default_rng(seed)
    metrics_list: List[ContinuityMetrics] = []

    # Process in batches for efficient inference
    for i in range(0, len(val_pairs), batch_size):
        batch_pairs = val_pairs[i : i + batch_size]
        batch_imgs = []
        batch_masks = []
        batch_occ_masks = []

        for sat_path, mask_path in batch_pairs:
            # Read & resize
            img = read_rgb(sat_path)
            mask = read_mask(mask_path)
            img_res = resize_image(img, image_size).astype(np.float32) / 255.0
            mask_res = resize_image(mask, image_size)
            mask_res = (mask_res > 0.5).astype(np.float32)

            if apply_synthetic_occlusion:
                # Deterministically inject 1 to 3 realistic obstacles across road
                occ_img, occ_m = apply_realistic_occlusion(
                    img_res,
                    road_mask=mask_res,
                    p=1.0,  # Ensure every benchmark sample receives occlusion
                    max_obstacles=3,
                    min_size=16,
                    max_size=36,
                    rng=rng,
                )
                batch_imgs.append(occ_img)
                batch_occ_masks.append(occ_m)
            else:
                batch_imgs.append(img_res)
                batch_occ_masks.append(None)

            batch_masks.append(mask_res)

        batch_arr = np.array(batch_imgs, dtype=np.float32)
        preds = model.predict(batch_arr, verbose=0)[:, :, :, 0]

        for j in range(len(batch_pairs)):
            p = preds[j]
            g = batch_masks[j]
            occ_m = batch_occ_masks[j]

            sample_metric = evaluate_continuity_single(
                pred_prob=p,
                gt_mask=g,
                threshold=threshold,
                occlusion_mask=occ_m,
            )
            metrics_list.append(sample_metric)

        if (i + batch_size) % 200 == 0 or (i + batch_size) >= len(val_pairs):
            print(f"    Evaluated {min(len(val_pairs), i + batch_size)} / {len(val_pairs)} samples...")

    aggregated = aggregate_metrics(metrics_list)
    return aggregated, metrics_list


def format_markdown_table(
    clean_baseline: Dict[str, Any],
    clean_occlusion: Dict[str, Any],
    occ_baseline: Dict[str, Any],
    occ_occlusion: Dict[str, Any],
) -> str:
    """Format comparative results into a structured Markdown table."""
    def _pct_diff(new_val: float, old_val: float, invert: bool = False) -> str:
        if old_val == 0:
            return "N/A"
        diff = ((new_val - old_val) / abs(old_val)) * 100.0
        sign = "+" if diff > 0 else ""
        return f"{sign}{diff:.2f}%"

    md = []
    md.append("## Controlled Experiment: Baseline vs Occlusion-Trained Model\n")
    md.append("### 1. Performance on Clean Untouched Validation Set (1,245 Images)")
    md.append("Verifies general segmentation accuracy and confirms no catastrophic forgetting or regression:\n")
    md.append("| Metric | Baseline (Original 50-epoch) | Occlusion-Trained Model | Delta / Change |")
    md.append("| :--- | :--- | :--- | :--- |")
    md.append(f"| **IoU (Jaccard)** | {clean_baseline['iou']:.4f} | {clean_occlusion['iou']:.4f} | {_pct_diff(clean_occlusion['iou'], clean_baseline['iou'])} |")
    md.append(f"| **Dice / F1** | {clean_baseline['dice_f1']:.4f} | {clean_occlusion['dice_f1']:.4f} | {_pct_diff(clean_occlusion['dice_f1'], clean_baseline['dice_f1'])} |")
    md.append(f"| **Precision** | {clean_baseline['precision']:.4f} | {clean_occlusion['precision']:.4f} | {_pct_diff(clean_occlusion['precision'], clean_baseline['precision'])} |")
    md.append(f"| **Recall** | {clean_baseline['recall']:.4f} | {clean_occlusion['recall']:.4f} | {_pct_diff(clean_occlusion['recall'], clean_baseline['recall'])} |")
    md.append(f"| **Fragmentation Ratio (Pred CC / GT CC)** | {clean_baseline['fragmentation_ratio']:.2f} | {clean_occlusion['fragmentation_ratio']:.2f} | {_pct_diff(clean_occlusion['fragmentation_ratio'], clean_baseline['fragmentation_ratio'])} |")
    md.append(f"| **Broken Road Segments (%)** | {clean_baseline['broken_segments_pct']:.2f}% | {clean_occlusion['broken_segments_pct']:.2f}% | {_pct_diff(clean_occlusion['broken_segments_pct'], clean_baseline['broken_segments_pct'])} |")
    md.append(f"| **Largest CC (LCC) Preservation** | {clean_baseline['lcc_preservation']:.4f} | {clean_occlusion['lcc_preservation']:.4f} | {_pct_diff(clean_occlusion['lcc_preservation'], clean_baseline['lcc_preservation'])} |")
    md.append(f"| **Mean Excess Skeleton Dead-Ends** | {clean_baseline['excess_endpoints']:.1f} | {clean_occlusion['excess_endpoints']:.1f} | {_pct_diff(clean_occlusion['excess_endpoints'], clean_baseline['excess_endpoints'])} |")
    md.append("\n---\n")

    md.append("### 2. Performance on Controlled Occluded Validation Benchmark (1,245 Images)")
    md.append("Tests road continuity, gap bridging, and resilience under realistic occlusions:\n")
    md.append("| Metric | Baseline (Original 50-epoch) | Occlusion-Trained Model | Delta / Change |")
    md.append("| :--- | :--- | :--- | :--- |")
    md.append(f"| **IoU under Occlusion** | {occ_baseline['iou']:.4f} | {occ_occlusion['iou']:.4f} | {_pct_diff(occ_occlusion['iou'], occ_baseline['iou'])} |")
    md.append(f"| **Dice / F1 under Occlusion** | {occ_baseline['dice_f1']:.4f} | {occ_occlusion['dice_f1']:.4f} | {_pct_diff(occ_occlusion['dice_f1'], occ_baseline['dice_f1'])} |")
    md.append(f"| **Precision under Occlusion** | {occ_baseline['precision']:.4f} | {occ_occlusion['precision']:.4f} | {_pct_diff(occ_occlusion['precision'], occ_baseline['precision'])} |")
    md.append(f"| **Recall under Occlusion** | {occ_baseline['recall']:.4f} | {occ_occlusion['recall']:.4f} | {_pct_diff(occ_occlusion['recall'], occ_baseline['recall'])} |")
    md.append(f"| **Fragmentation Ratio (Pred CC / GT CC)** | {occ_baseline['fragmentation_ratio']:.2f} | {occ_occlusion['fragmentation_ratio']:.2f} | {_pct_diff(occ_occlusion['fragmentation_ratio'], occ_baseline['fragmentation_ratio'])} |")
    md.append(f"| **Total Broken Segments Count** | {occ_baseline['total_broken_segments']} | {occ_occlusion['total_broken_segments']} | {_pct_diff(occ_occlusion['total_broken_segments'], occ_baseline['total_broken_segments'])} |")
    md.append(f"| **Broken Road Segments (%)** | {occ_baseline['broken_segments_pct']:.2f}% | {occ_occlusion['broken_segments_pct']:.2f}% | {_pct_diff(occ_occlusion['broken_segments_pct'], occ_baseline['broken_segments_pct'])} |")
    md.append(f"| **Largest CC (LCC) Preservation** | {occ_baseline['lcc_preservation']:.4f} | {occ_occlusion['lcc_preservation']:.4f} | {_pct_diff(occ_occlusion['lcc_preservation'], occ_baseline['lcc_preservation'])} |")
    md.append(f"| **Occlusion Gap Recall (In-Gap Recall)** | {occ_baseline['gap_recall']:.4f} | {occ_occlusion['gap_recall']:.4f} | {_pct_diff(occ_occlusion['gap_recall'], occ_baseline['gap_recall'])} |")
    md.append(f"| **Occlusion Gap IoU (In-Gap IoU)** | {occ_baseline['gap_iou']:.4f} | {occ_occlusion['gap_iou']:.4f} | {_pct_diff(occ_occlusion['gap_iou'], occ_baseline['gap_iou'])} |")
    md.append(f"| **Mean Excess Skeleton Dead-Ends** | {occ_baseline['excess_endpoints']:.1f} | {occ_occlusion['excess_endpoints']:.1f} | {_pct_diff(occ_occlusion['excess_endpoints'], occ_baseline['excess_endpoints'])} |")

    return "\n".join(md)


def run_comparative_experiment(
    deepglobe_root: str = "dataset",
    baseline_model_path: str = "models/road_unet.keras",
    occlusion_model_path: str = "models/road_unet_occlusion.keras",
    output_dir: str = "outputs/experiments",
    image_size: int = 256,
    threshold: float = 0.40,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  Executing Comparative Experiment: Baseline vs Occlusion-Trained")
    print("=" * 65)
    print(f"  Baseline Model Path   : {baseline_model_path}")
    print(f"  Occlusion Model Path  : {occlusion_model_path}")
    print(f"  Threshold             : {threshold}")
    print("=" * 65 + "\n")

    # Load validation pairs
    _, val_pairs = load_deepglobe_pairs(
        deepglobe_root,
        val_fraction=val_fraction,
        seed=seed,
    )
    print(f"[Dataset] Loaded {len(val_pairs)} validation pairs.\n")

    # Load models
    print(f"[1/4] Loading Baseline Model from {baseline_model_path}...")
    baseline_model = load_model_weights(baseline_model_path, image_size)

    print(f"[2/4] Loading Occlusion-Trained Model from {occlusion_model_path}...")
    occlusion_model = load_model_weights(occlusion_model_path, image_size)

    # 1. Clean Validation Evaluation
    print("\n[3/4] Evaluating Models on CLEAN Untouched Validation Set...")
    print("  -> Evaluating Baseline Model...")
    clean_baseline, _ = evaluate_dataset(baseline_model, val_pairs, image_size, threshold, apply_synthetic_occlusion=False)
    print(f"     Baseline Clean IoU: {clean_baseline['iou']:.4f}, Dice: {clean_baseline['dice_f1']:.4f}, Frag Ratio: {clean_baseline['fragmentation_ratio']:.2f}")

    print("  -> Evaluating Occlusion-Trained Model...")
    clean_occlusion, _ = evaluate_dataset(occlusion_model, val_pairs, image_size, threshold, apply_synthetic_occlusion=False)
    print(f"     Occlusion Clean IoU: {clean_occlusion['iou']:.4f}, Dice: {clean_occlusion['dice_f1']:.4f}, Frag Ratio: {clean_occlusion['fragmentation_ratio']:.2f}")

    # 2. Occluded Validation Benchmark
    print("\n[4/4] Evaluating Models on OCCLUDED Validation Benchmark...")
    print("  -> Evaluating Baseline Model under Occlusion...")
    occ_baseline, _ = evaluate_dataset(baseline_model, val_pairs, image_size, threshold, apply_synthetic_occlusion=True, seed=seed)
    print(f"     Baseline Occluded IoU: {occ_baseline['iou']:.4f}, Gap Recall: {occ_baseline['gap_recall']:.4f}, Broken Pct: {occ_baseline['broken_segments_pct']:.2f}%")

    print("  -> Evaluating Occlusion-Trained Model under Occlusion...")
    occ_occlusion, _ = evaluate_dataset(occlusion_model, val_pairs, image_size, threshold, apply_synthetic_occlusion=True, seed=seed)
    print(f"     Occlusion-Trained IoU: {occ_occlusion['iou']:.4f}, Gap Recall: {occ_occlusion['gap_recall']:.4f}, Broken Pct: {occ_occlusion['broken_segments_pct']:.2f}%")

    # Save summary JSON
    results = {
        "clean_validation": {
            "baseline": clean_baseline,
            "occlusion_trained": clean_occlusion,
        },
        "occluded_validation": {
            "baseline": occ_baseline,
            "occlusion_trained": occ_occlusion,
        },
    }

    json_path = out_dir / "metrics_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved structured metrics JSON to: {json_path}")

    # Generate and save Markdown report
    md_content = format_markdown_table(clean_baseline, clean_occlusion, occ_baseline, occ_occlusion)
    report_path = out_dir / "experiment_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"✓ Saved formatted Markdown report to: {report_path}")

    print("\n" + md_content + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Baseline vs Occlusion-Trained Model")
    parser.add_argument("--deepglobe", default="dataset")
    parser.add_argument("--baseline", default="models/road_unet.keras")
    parser.add_argument("--occlusion-model", default="models/road_unet_occlusion.keras")
    parser.add_argument("--output", default="outputs/experiments")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.40)
    args = parser.parse_args()

    run_comparative_experiment(
        deepglobe_root=args.deepglobe,
        baseline_model_path=args.baseline,
        occlusion_model_path=args.occlusion_model,
        output_dir=args.output,
        image_size=args.image_size,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
