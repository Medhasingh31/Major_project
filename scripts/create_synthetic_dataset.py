from pathlib import Path

import cv2
import numpy as np


def draw_sample(seed: int, size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:] = rng.integers([35, 70, 40], [75, 115, 75], size=3, dtype=np.uint8)

    noise = rng.normal(0, 10, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)

    road_count = int(rng.integers(3, 6))
    for _ in range(road_count):
        x1, y1 = rng.integers(0, size, size=2)
        x2, y2 = rng.integers(0, size, size=2)
        width = int(rng.integers(5, 11))
        cv2.line(image, (x1, y1), (x2, y2), (170, 170, 165), width, lineType=cv2.LINE_AA)
        cv2.line(mask, (x1, y1), (x2, y2), 255, width, lineType=cv2.LINE_AA)

    for _ in range(2):
        x, y = rng.integers(20, size - 50, size=2)
        w, h = rng.integers(20, 45, size=2)
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (50, 90, 55), -1)
        cv2.rectangle(mask, (int(x), int(y)), (int(x + w), int(y + h)), 0, -1)

    return image, mask


def write_split(split: str, count: int, start_seed: int) -> None:
    image_dir = Path("data") / split / "images"
    mask_dir = Path("data") / split / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for index in range(count):
        image, mask = draw_sample(start_seed + index)
        name = f"synthetic_{index:03d}.png"
        cv2.imwrite(str(image_dir / name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_dir / name), mask)


def main() -> None:
    write_split("train", count=12, start_seed=100)
    write_split("val", count=4, start_seed=500)
    print("Synthetic dataset written to data/train and data/val")


if __name__ == "__main__":
    main()
