"""
road_extractor/train_occlusion.py
=================================
Fine-tunes the baseline 50-epoch U-Net using realistic occlusion augmentation
to improve road continuity under occlusion.

Key guarantees:
  - Original baseline model (models/road_unet.keras) remains intact and untouched.
  - New model is saved to models/road_unet_occlusion.keras.
  - Same deterministic train/val split (seed=42, val_fraction=0.2).
  - Artificial occlusion is applied ONLY to training images.
  - Ground-truth road masks are kept 100% intact.
  - Validation set remains 100% clean and pristine.
  - Low learning rate fine-tuning (5e-5) with EarlyStopping and ReduceLROnPlateau.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from road_extractor.config import TrainingConfig
from road_extractor.data import (
    load_deepglobe_pairs,
    make_deepglobe_dataset,
    print_deepglobe_summary,
)
from road_extractor.model import build_light_unet, compile_model, dice_coefficient, iou_coefficient


def run_occlusion_finetuning(
    deepglobe_root: str = "dataset",
    base_model_path: str = "models/road_unet.keras",
    output_model_path: str = "models/road_unet_occlusion.keras",
    image_size: int = 256,
    batch_size: int = 4,
    epochs: int = 15,
    learning_rate: float = 5e-5,
    occlusion_prob: float = 0.40,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> None:
    output_path = Path(output_model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_dir = Path("outputs/experiments")
    log_dir.mkdir(parents=True, exist_ok=True)

    base_path = Path(base_model_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Baseline model not found at {base_path}. Cannot fine-tune.")

    print("\n" + "=" * 65)
    print("  Road Continuity Fine-Tuning with Occlusion Augmentation")
    print("=" * 65)
    print(f"  Baseline Checkpoint   : {base_model_path}")
    print(f"  Output Checkpoint     : {output_model_path}")
    print(f"  Occlusion Probability : {occlusion_prob:.2f} (applied to training images only)")
    print(f"  Ground-Truth Masks    : 100% Intact (True continuous road labels)")
    print(f"  Fine-Tuning LR        : {learning_rate}")
    print(f"  Max Epochs            : {epochs}")
    print(f"  Batch Size            : {batch_size}")
    print("=" * 65 + "\n")

    # 1. Load exact dataset split
    train_pairs, val_pairs = load_deepglobe_pairs(
        deepglobe_root,
        val_fraction=val_fraction,
        seed=seed,
    )
    print_deepglobe_summary(train_pairs, val_pairs, image_size)

    # 2. Build Datasets
    # Training: applies geometric, photometric, and realistic occlusion augmentation
    train_ds = make_deepglobe_dataset(
        train_pairs,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        apply_occlusion=True,
        occlusion_prob=occlusion_prob,
    )
    # Validation: pristine, clean, untouched images
    val_ds = make_deepglobe_dataset(
        val_pairs,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        apply_occlusion=False,
    )

    # 3. Build & Initialize Model from Baseline Weights
    print(f"\n[Model Setup] Initializing U-Net and loading weights from {base_model_path}...")
    model = build_light_unet(image_size=image_size, base_filters=16)
    model.load_weights(str(base_path))
    print("[Model Setup] Successfully loaded baseline weights.")

    # Compile with fine-tuning learning rate and combined Dice + BCE loss
    compile_model(model, learning_rate=learning_rate, loss_name="combined")

    # 4. Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(output_path),
            save_best_only=True,
            monitor="val_dice_coefficient",
            mode="max",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            patience=6,
            monitor="val_dice_coefficient",
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_dice_coefficient",
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(log_dir / "occlusion_training_log.csv"),
            append=False,
        ),
    ]

    # 5. Execute Fine-Tuning
    print("\n[Training] Starting fine-tuning with occlusion augmentation...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # Save final best weights
    model.save(str(output_path))
    print(f"\n✓ Fine-tuning complete. Model saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune U-Net with Occlusion Augmentation")
    parser.add_argument("--deepglobe", default="dataset", help="DeepGlobe dataset root directory")
    parser.add_argument("--base-model", default="models/road_unet.keras", help="Path to original baseline model")
    parser.add_argument("--output", default="models/road_unet_occlusion.keras", help="Path to save new model")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--occlusion-prob", type=float, default=0.40)
    args = parser.parse_args()

    run_occlusion_finetuning(
        deepglobe_root=args.deepglobe,
        base_model_path=args.base_model,
        output_model_path=args.output,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        occlusion_prob=args.occlusion_prob,
    )


if __name__ == "__main__":
    main()
