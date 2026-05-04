import argparse
from pathlib import Path

from road_extractor.config import ExtractionConfig, TrainingConfig
from road_extractor.pipeline import extract_roads


def train(args: argparse.Namespace) -> None:
    from tensorflow import keras

    from road_extractor.data import make_dataset
    from road_extractor.model import build_light_unet, compile_model

    config = TrainingConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        base_filters=args.base_filters,
    )
    train_ds = make_dataset(args.train_images, args.train_masks, config.image_size, config.batch_size, shuffle=True)
    val_ds = make_dataset(args.val_images, args.val_masks, config.image_size, config.batch_size, shuffle=False)

    model = build_light_unet(config.image_size, config.base_filters)
    compile_model(model, config.learning_rate)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(output_path), save_best_only=True, monitor="val_loss"),
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=config.epochs, callbacks=callbacks)
    model.save(output_path)
    print(f"Saved model to {output_path}")


def extract(args: argparse.Namespace) -> None:
    config = ExtractionConfig(
        image_size=args.image_size,
        threshold=args.threshold,
        min_object_size=args.min_object_size,
        closing_radius=args.closing_radius,
        bridge_kernel_size=args.bridge_kernel_size,
    )
    weights = None if args.no_model else args.weights
    result = extract_roads(args.image, args.output, weights_path=weights, config=config)
    print(f"Road extraction complete: {result['nodes']} nodes, {result['edges']} edges")
    print(f"Outputs written to: {result['output_dir']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight road extraction and graph generation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a small U-Net road segmentation model")
    train_parser.add_argument("--train-images", required=True)
    train_parser.add_argument("--train-masks", required=True)
    train_parser.add_argument("--val-images", required=True)
    train_parser.add_argument("--val-masks", required=True)
    train_parser.add_argument("--output", default="models/road_unet.keras")
    train_parser.add_argument("--image-size", type=int, default=256)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--base-filters", type=int, default=16)
    train_parser.set_defaults(func=train)

    extract_parser = subparsers.add_parser("extract", help="Extract roads and export graph outputs")
    extract_parser.add_argument("--image", required=True)
    extract_parser.add_argument("--output", default="outputs/run")
    extract_parser.add_argument("--weights", default="models/road_unet.keras")
    extract_parser.add_argument("--no-model", action="store_true", help="Use classical fallback instead of model weights")
    extract_parser.add_argument("--image-size", type=int, default=256)
    extract_parser.add_argument("--threshold", type=float, default=0.5)
    extract_parser.add_argument("--min-object-size", type=int, default=64)
    extract_parser.add_argument("--closing-radius", type=int, default=3)
    extract_parser.add_argument("--bridge-kernel-size", type=int, default=9)
    extract_parser.set_defaults(func=extract)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
