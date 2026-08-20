import argparse
from logging import config
import os
import math
from pathlib import Path

from road_extractor.config import ExtractionConfig, TrainingConfig, get_default_weights_path
from road_extractor.model import compile_model
from road_extractor.pipeline import extract_roads


def _print_training_config(
    train_pairs: list,
    val_pairs: list,
    config: "TrainingConfig",
    output_path: str,
) -> None:
    """Print a full pre-training configuration summary."""
    from road_extractor.model import LOSS_DISPLAY_NAMES

    steps_per_epoch = math.ceil(len(train_pairs) / config.batch_size)
    val_steps       = math.ceil(len(val_pairs)   / config.batch_size)

    print("\n" + "=" * 60)
    print("  Training Configuration")
    print("=" * 60)
    print(f"  Dataset")
    print(f"    Training pairs          : {len(train_pairs):,}")
    print(f"    Validation pairs        : {len(val_pairs):,}")
    print(f"  Model")
    print(f"    Input shape             : ({config.image_size}, {config.image_size}, 3)")
    print(f"    Base filters            : {config.base_filters}")
    print(f"  Training")
    print(f"    Epochs                  : {config.epochs}")
    print(f"    Batch size              : {config.batch_size}")
    print(f"    Steps per epoch         : {steps_per_epoch}")
    print(f"    Validation steps        : {val_steps}")
    print(f"    Learning rate           : {config.learning_rate}")
    print(f"    Loss function           : {LOSS_DISPLAY_NAMES.get(config.loss_name, config.loss_name)}")
    print(f"  Validation metrics")
    print(f"    Primary (checkpoint)    : val_dice_coefficient")
    print(f"    Additional              : val_iou_coefficient, val_binary_accuracy")
    print(f"  Callbacks")
    print(f"    ModelCheckpoint         : save best val_dice_coefficient -> {output_path}")
    print(f"    EarlyStopping           : patience=7  (monitor: val_dice_coefficient)")
    print(f"    ReduceLROnPlateau       : factor=0.5, patience=3  (monitor: val_dice_coefficient)")
    print("=" * 60 + "\n")


def train(args: argparse.Namespace) -> None:
    from tensorflow import keras

    from road_extractor.data import (
        load_deepglobe_pairs,
        make_dataset,
        make_deepglobe_dataset,
        print_deepglobe_summary,
    )
    from road_extractor.model import build_light_unet, compile_model

    config = TrainingConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        base_filters=args.base_filters,
        loss_name=args.loss,
        val_fraction=args.val_fraction,
    )

    if args.deepglobe:
        # ── DeepGlobe dataset path ──────────────────────────────────────────
        train_pairs, val_pairs = load_deepglobe_pairs(
            args.deepglobe,
            val_fraction=config.val_fraction,
        )
        print_deepglobe_summary(train_pairs, val_pairs, config.image_size)
        train_ds = make_deepglobe_dataset(
            train_pairs,
            config.image_size,
            config.batch_size,
            shuffle=True,
            apply_occlusion=getattr(args, "occlusion_aug", False),
            occlusion_prob=getattr(args, "occlusion_prob", 0.40),
        )
        val_ds   = make_deepglobe_dataset(val_pairs,   config.image_size, config.batch_size, shuffle=False)
    else:
        # ── Legacy separate-folder mode ────────────────────────────────────
        missing = [
            name for name, val in [
                ("--train-images", args.train_images),
                ("--train-masks",  args.train_masks),
                ("--val-images",   args.val_images),
                ("--val-masks",    args.val_masks),
            ]
            if val is None
        ]
        if missing:
            raise SystemExit(
                f"Error: the following arguments are required when --deepglobe is not set: "
                f"{', '.join(missing)}"
            )
        train_ds = make_dataset(args.train_images, args.train_masks, config.image_size, config.batch_size, shuffle=True)
        val_ds   = make_dataset(args.val_images,   args.val_masks,   config.image_size, config.batch_size, shuffle=False)
        # Build synthetic pair lists for summary counts
        from road_extractor.data import list_image_files
        from pathlib import Path as _Path
        _img_paths = list_image_files(args.train_images)
        _mask_dir  = _Path(args.train_masks)
        train_pairs = [(str(p), str(_mask_dir / p.name)) for p in _img_paths if (_mask_dir / p.name).exists()]
        _img_paths = list_image_files(args.val_images)
        _mask_dir  = _Path(args.val_masks)
        val_pairs   = [(str(p), str(_mask_dir / p.name)) for p in _img_paths if (_mask_dir / p.name).exists()]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Pre-training configuration summary ────────────────────────────────
    _print_training_config(train_pairs, val_pairs, config, str(output_path))

    # ── Build / load model ────────────────────────────────────────────────
    base_weights = getattr(args, "base_weights", None)

    if args.fine_tune:
        if not base_weights:
            raise SystemExit(
                "Error: --fine-tune requires --base-weights <model_path>"
            )
        if not Path(base_weights).exists():
            raise SystemExit(
                f"Error: Fine-tuning model not found: {base_weights}"
            )

        print(f"\n[Fine-Tune] Loading existing model from: {base_weights}")
        model = keras.models.load_model(
            str(base_weights),
            compile=False,
        )

        # Freeze all layers first, then unfreeze only the final 10 layers.
        for layer in model.layers:
            layer.trainable = False
        for layer in model.layers[-10:]:
            layer.trainable = True

        config.learning_rate = 1e-5
        trainable = sum(1 for layer in model.layers if layer.trainable)
        total = len(model.layers)
        print(f"[Fine-Tune] Total layers: {total}")
        print(f"[Fine-Tune] Trainable layers: {trainable}")
        print(f"[Fine-Tune] Frozen layers: {total - trainable}")
        print(f"[Fine-Tune] Learning rate: {config.learning_rate}")
    else:
        print("\n[Training] Building a new U-Net from scratch...")
        model = build_light_unet(
            config.image_size,
            config.base_filters,
        )

    compile_model(model, config.learning_rate, config.loss_name)

    # ── Callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        # Save the best model by validation Dice (higher = better)
        keras.callbacks.ModelCheckpoint(
            str(output_path),
            save_best_only=True,
            monitor="val_dice_coefficient",
            mode="max",
            verbose=1,
        ),
        # Stop early if val Dice stops improving for 7 epochs
        keras.callbacks.EarlyStopping(
            patience=7,
            monitor="val_dice_coefficient",
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        # Halve LR when val Dice plateaus for 3 epochs
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_dice_coefficient",
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        # CSV log for later inspection
        keras.callbacks.CSVLogger(
            str(output_path.parent / "training_log.csv"),
            append=False,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
    )
    model.save(output_path)
    print(f"\nSaved model to {output_path}")


def extract(args: argparse.Namespace) -> None:
    config = ExtractionConfig(
        image_size=args.image_size,
        tile_size=getattr(args, "tile_size", 256),
        tile_stride=getattr(args, "tile_stride", 128),
        use_tiling=not getattr(args, "no_tiling", False),
        threshold=args.threshold,
        min_object_size=args.min_object_size,
        closing_radius=args.closing_radius,
        bridge_kernel_size=args.bridge_kernel_size,
    )
    weights = None if args.no_model else args.weights
    result = extract_roads(args.image, args.output, weights_path=weights, config=config)
    print("Geometry + topology extraction complete")
    print(f"Input mask size: {result['input_mask_size']}")
    print(f"Road pixels: {result['road_pixels']}")
    print(f"Skeleton pixels: {result['skeleton_pixels']}")
    print(f"Geometry connected components: {result['connected_components']}")
    print(f"Geometry segments: {result['segments']}")
    print(f"Total centerline length: {result['total_length_pixels']}")
    print(f"Topology nodes: {result['topology_nodes']}")
    print(f"Topology edges: {result['topology_edges']}")
    print(f"Intersections: {result['topology_intersections']}")
    print(f"Endpoints: {result['topology_endpoints']}")
    print(f"Connected components: {result['topology_connected_components']}")
    print(f"Suspicious/disconnected segments: {result['suspicious_disconnected_segments']}")
    print("Major geometric changes:")
    for key, value in result["major_geometric_changes"].items():
        print(f"  {key}: {value}")
    print(f"Outputs written to: {result['output_dir']}")
    print("Output files:")
    for output_file in result["output_files"]:
        print(f"  {output_file}")


def evaluate_exp(args: argparse.Namespace) -> None:
    from road_extractor.evaluate_experiment import run_comparative_experiment
    run_comparative_experiment(
        deepglobe_root=args.deepglobe,
        baseline_model_path=args.baseline,
        occlusion_model_path=args.occlusion_model,
        output_dir=args.output,
        image_size=args.image_size,
        threshold=args.threshold,
    )


def visualize_exp(args: argparse.Namespace) -> None:
    from road_extractor.visualize_experiment import generate_qualitative_suite
    generate_qualitative_suite(
        deepglobe_root=args.deepglobe,
        baseline_model_path=args.baseline,
        occlusion_model_path=args.occlusion_model,
        output_dir=args.output,
        num_samples=args.num_samples,
        image_size=args.image_size,
        threshold=args.threshold,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight road extraction and graph generation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a small U-Net road segmentation model")
    # DeepGlobe shortcut (mutually exclusive with the legacy 4-dir approach)
    train_parser.add_argument(
        "--deepglobe",
        metavar="DEEPGLOBE_ROOT",
        default=None,
        help="Path to DeepGlobe dataset root (contains metadata.csv + train/). "
             "When set, --train-images/masks and --val-images/masks are ignored.",
    )
    train_parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of DeepGlobe train pairs to hold out as validation (default 0.2).",
    )
    # Legacy separate-folder arguments (required only when --deepglobe is not used)
    train_parser.add_argument("--train-images", default=None)
    train_parser.add_argument("--train-masks", default=None)
    train_parser.add_argument("--val-images", default=None)
    train_parser.add_argument("--val-masks", default=None)
    train_parser.add_argument("--output", default="models/road_unet.keras")
    train_parser.add_argument("--base-weights", default=None, help="Base pretrained weights to initialize from")
    train_parser.add_argument("--occlusion-aug", action="store_true", help="Apply realistic occlusion augmentation during training")
    train_parser.add_argument("--occlusion-prob", type=float, default=0.40, help="Probability of applying occlusion per training image")
    train_parser.add_argument("--image-size", type=int, default=256)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--base-filters", type=int, default=16)
    # Loss selection: keep default as the combined Dice + BCE loss
    from road_extractor.model import LOSS_REGISTRY
    train_parser.add_argument(
        "--loss",
        choices=sorted(list(LOSS_REGISTRY.keys())),
        default="combined",
        help="Loss function to use for training (default: combined = Dice + BCE)",
    )
    train_parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Fine-tune the existing model at the output path instead of training from scratch. Uses a very low learning rate.",
    )
    train_parser.set_defaults(func=train)

    extract_parser = subparsers.add_parser("extract", help="Extract roads and export graph outputs")
    extract_parser.add_argument("--image", required=True)
    extract_parser.add_argument("--output", default="outputs/run")
    extract_parser.add_argument("--weights", default=str(get_default_weights_path()), help="Path to U-Net model weights (defaults to occlusion-trained model)")
    extract_parser.add_argument("--no-model", action="store_true", help="Use classical fallback instead of model weights")
    extract_parser.add_argument("--image-size", type=int, default=256)
    extract_parser.add_argument("--threshold", type=float, default=0.30)
    extract_parser.add_argument("--min-object-size", type=int, default=64)
    extract_parser.add_argument("--closing-radius", type=int, default=3)
    extract_parser.add_argument("--bridge-kernel-size", type=int, default=9)
    extract_parser.set_defaults(func=extract)

    eval_parser = subparsers.add_parser("evaluate-experiment", help="Compare baseline vs occlusion-trained model")
    eval_parser.add_argument("--deepglobe", default="dataset")
    eval_parser.add_argument("--baseline", default="models/road_unet.keras")
    eval_parser.add_argument("--occlusion-model", default="models/road_unet_occlusion.keras")
    eval_parser.add_argument("--output", default="outputs/experiments")
    eval_parser.add_argument("--image-size", type=int, default=256)
    eval_parser.add_argument("--threshold", type=float, default=0.40)
    eval_parser.set_defaults(func=evaluate_exp)

    vis_parser = subparsers.add_parser("visualize-experiment", help="Generate 5-panel qualitative comparison figures")
    vis_parser.add_argument("--deepglobe", default="dataset")
    vis_parser.add_argument("--baseline", default="models/road_unet.keras")
    vis_parser.add_argument("--occlusion-model", default="models/road_unet_occlusion.keras")
    vis_parser.add_argument("--output", default="outputs/experiments/qualitative_comparisons")
    vis_parser.add_argument("--num-samples", type=int, default=8)
    vis_parser.add_argument("--image-size", type=int, default=256)
    vis_parser.add_argument("--threshold", type=float, default=0.40)
    vis_parser.set_defaults(func=visualize_exp)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
