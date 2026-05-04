from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def list_image_files(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def read_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(path: str | Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return (mask > 127).astype(np.float32)


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def make_dataset(
    image_dir: str | Path,
    mask_dir: str | Path,
    image_size: int,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    import tensorflow as tf

    image_paths = list_image_files(image_dir)
    mask_dir = Path(mask_dir)
    pairs = [(image_path, mask_dir / image_path.name) for image_path in image_paths]
    pairs = [(image_path, mask_path) for image_path, mask_path in pairs if mask_path.exists()]

    if not pairs:
        raise ValueError("No matching image/mask pairs found.")

    image_names = [str(image_path) for image_path, _ in pairs]
    mask_names = [str(mask_path) for _, mask_path in pairs]

    dataset = tf.data.Dataset.from_tensor_slices((image_names, mask_names))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(pairs), reshuffle_each_iteration=True)

    def load_pair(image_path: tf.Tensor, mask_path: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image_bytes = tf.io.read_file(image_path)
        mask_bytes = tf.io.read_file(mask_path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        mask = tf.io.decode_image(mask_bytes, channels=1, expand_animations=False)
        image = tf.image.resize(image, (image_size, image_size))
        mask = tf.image.resize(mask, (image_size, image_size), method="nearest")
        image = tf.cast(image, tf.float32) / 255.0
        mask = tf.cast(mask > 127, tf.float32)
        return image, mask

    return dataset.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
