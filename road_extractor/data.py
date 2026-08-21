from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def list_image_files(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def read_rgb(path: str | Path) -> np.ndarray:
    if str(path).lower().endswith(('.tif', '.tiff')):
        import rasterio
        with rasterio.open(path) as src:
            if src.count >= 3:
                img = src.read([1, 2, 3])
                img = np.moveaxis(img, 0, -1)
            else:
                img = src.read(1)
                img = np.stack([img, img, img], axis=-1)
            
            # Normalize high dynamic range or floats to standard uint8
            if img.dtype != np.uint8:
                img_min = img.min()
                img_max = img.max()
                if img_max > img_min:
                    img = ((img - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            return img

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


# ---------------------------------------------------------------------------
# DeepGlobe dataset loader
# ---------------------------------------------------------------------------

def load_deepglobe_pairs(
    deepglobe_root: str | Path,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Read metadata.csv from the DeepGlobe root and return train/val pairs.

    Only images from the 'train' split (which have ground-truth masks) are
    used.  The valid/ and test/ folders contain satellite images only and are
    never touched.

    Parameters
    ----------
    deepglobe_root : path
        Directory containing metadata.csv, train/, valid/, test/.
    val_fraction : float
        Fraction of the train split to reserve for validation (default 0.2).
    seed : int
        Random seed for the deterministic split.

    Returns
    -------
    train_pairs, val_pairs
        Each is a list of (absolute_sat_path, absolute_mask_path) strings.
    """
    root = Path(deepglobe_root)
    metadata_path = root / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"DeepGlobe metadata.csv not found at {metadata_path}")

    all_pairs: list[tuple[str, str]] = []
    missing: list[str] = []

    with open(metadata_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != "train":
                # valid/test splits have no masks — skip entirely
                continue
            sat_path = root / row["sat_image_path"]
            mask_path = root / row["mask_path"]
            if sat_path.exists() and mask_path.exists():
                all_pairs.append((str(sat_path), str(mask_path)))
            else:
                missing.append(str(row["image_id"]))

    if missing:
        print(f"[DeepGlobe] Warning: {len(missing)} entries in metadata.csv had missing files "
              f"and were skipped.")

    if not all_pairs:
        raise ValueError(
            f"No valid image-mask pairs found in {root}. "
            "Check that data/deepglobe/train/ contains both _sat.jpg and _mask.png files."
        )

    # Deterministic shuffle then split
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(all_pairs)).tolist()
    n_val = max(1, int(len(all_pairs) * val_fraction))
    val_indices = set(indices[:n_val])

    train_pairs = [all_pairs[i] for i in range(len(all_pairs)) if i not in val_indices]
    val_pairs   = [all_pairs[i] for i in range(len(all_pairs)) if i in val_indices]

    return train_pairs, val_pairs


def print_deepglobe_summary(
    train_pairs: list[tuple[str, str]],
    val_pairs: list[tuple[str, str]],
    image_size: int,
) -> None:
    """
    Print a short dataset summary before training begins.
    Reads one sample pair to confirm actual image/mask dimensions on disk.
    """
    total = len(train_pairs) + len(val_pairs)

    # Sample the first pair for dimension info
    sample_sat, sample_mask = train_pairs[0]
    sat_img  = cv2.imread(sample_sat, cv2.IMREAD_COLOR)
    mask_img = cv2.imread(sample_mask, cv2.IMREAD_GRAYSCALE)

    sat_h, sat_w = sat_img.shape[:2] if sat_img is not None else (-1, -1)
    msk_h, msk_w = mask_img.shape[:2] if mask_img is not None else (-1, -1)

    unique_vals = sorted(set(mask_img.flatten().tolist())) if mask_img is not None else []
    binary_vals = [0, 1] if set(unique_vals) <= {0, 255} else unique_vals

    print("=" * 55)
    print("  DeepGlobe Dataset Summary")
    print("=" * 55)
    print(f"  Total image-mask pairs  : {total:,}")
    print(f"  Training pairs          : {len(train_pairs):,}")
    print(f"  Validation pairs        : {len(val_pairs):,}")
    print(f"  Sample image dimensions : {sat_w} x {sat_h} (W x H)")
    print(f"  Sample mask dimensions  : {msk_w} x {msk_h} (W x H)")
    print(f"  Mask values on disk     : {unique_vals[:10]}  -> binarised to {binary_vals}")
    print(f"  U-Net input size        : {image_size} x {image_size}")
    print(f"  Image loading           : RGB (3 channels, normalised to [0, 1])")
    print(f"  Mask loading            : Grayscale -> binary (road=1, background=0)")
    print("=" * 55)


def make_deepglobe_dataset(
    pairs: list[tuple[str, str]],
    image_size: int,
    batch_size: int,
    shuffle: bool,
    apply_occlusion: bool = False,
    occlusion_prob: float = 0.40,
    cache: bool = True,
) -> "tf.data.Dataset":
    """
    Build a tf.data.Dataset from a list of (sat_path, mask_path) pairs.

    Each satellite image is:
      - decoded as RGB (3 channels)
      - resized to (image_size, image_size)
      - normalised to float32 in [0, 1]

    Each mask is:
      - decoded as grayscale (1 channel)
      - resized to (image_size, image_size) with nearest-neighbour
      - binarised: road = 1.0, background = 0.0
      - kept as shape (H, W, 1) to match U-Net output

    Training datasets (shuffle=True) apply geometric & photometric augmentation.
    If apply_occlusion=True, realistic tree/building/shadow occlusions are applied
    exclusively to training images while preserving ground truth masks.
    """
    import tensorflow as tf
    from road_extractor.occlusion import py_realistic_occlusion_aug

    sat_paths  = [p for p, _ in pairs]
    mask_paths = [m for _, m in pairs]

    dataset = tf.data.Dataset.from_tensor_slices((sat_paths, mask_paths))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(pairs), reshuffle_each_iteration=True)

    def load_pair(sat_path: tf.Tensor, mask_path: tf.Tensor):
        # --- satellite image: RGB, float32 [0, 1] ---
        image_bytes = tf.io.read_file(sat_path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, [image_size, image_size])
        image = tf.cast(image, tf.float32) / 255.0

        # --- road mask: grayscale → binary float32 ---
        mask_bytes = tf.io.read_file(mask_path)
        mask = tf.io.decode_image(mask_bytes, channels=1, expand_animations=False)
        mask = tf.image.resize(mask, [image_size, image_size], method="nearest")
        mask = tf.cast(mask > 127, tf.float32)   # road=1.0, background=0.0

        return image, mask

    def augment(image: tf.Tensor, mask: tf.Tensor):
        # Concatenate so identical transforms apply to both
        combined = tf.concat([image, mask], axis=-1)   # (H, W, 4)
        combined = tf.image.random_flip_left_right(combined)
        combined = tf.image.random_flip_up_down(combined)
        k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
        combined = tf.image.rot90(combined, k=k)
        aug_image = combined[..., :3]
        aug_mask  = combined[..., 3:]
        
        # Photometric transformations (apply to image only)
        aug_image = tf.image.random_brightness(aug_image, max_delta=0.1)
        aug_image = tf.image.random_contrast(aug_image, lower=0.9, upper=1.1)
        aug_image = tf.clip_by_value(aug_image, 0.0, 1.0)
        
        return aug_image, aug_mask

    def tf_occlusion_aug(image: tf.Tensor, mask: tf.Tensor):
        def _py_wrapper(img_np, msk_np):
            return py_realistic_occlusion_aug(img_np, msk_np, p=occlusion_prob)

        occ_img, occ_mask = tf.numpy_function(
            _py_wrapper,
            [image, mask],
            [tf.float32, tf.float32],
        )
        occ_img.set_shape([image_size, image_size, 3])
        occ_mask.set_shape([image_size, image_size, 1])
        return occ_img, occ_mask

    dataset = dataset.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        dataset = dataset.cache()
    if shuffle:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        if apply_occlusion:
            dataset = dataset.map(tf_occlusion_aug, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_dataset(
    image_dir: str | Path,
    mask_dir: str | Path,
    image_size: int,
    batch_size: int,
    shuffle: bool,
) -> "tf.data.Dataset":
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

    def augment(image: tf.Tensor, mask: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        # Stack to ensure identical transformations for both image and mask
        combined = tf.concat([image, mask], axis=-1)
        
        # Lightweight transformations
        combined = tf.image.random_flip_left_right(combined)
        combined = tf.image.random_flip_up_down(combined)
        
        # Random 90-degree rotations
        k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
        combined = tf.image.rot90(combined, k=k)
        
        # Unstack back into image and mask
        aug_image = combined[..., :3]
        aug_mask = combined[..., 3:]
        
        # Photometric transformations (apply to image only)
        aug_image = tf.image.random_brightness(aug_image, max_delta=0.1)
        aug_image = tf.image.random_contrast(aug_image, lower=0.9, upper=1.1)
        aug_image = tf.clip_by_value(aug_image, 0.0, 1.0)

        return aug_image, aug_mask

    dataset = dataset.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        # Only augment the training dataset (where shuffle=True)
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
