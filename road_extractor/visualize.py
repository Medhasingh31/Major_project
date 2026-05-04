from pathlib import Path

import cv2
import numpy as np


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask > 0).astype(np.uint8) * 255)


def save_overlay(rgb: np.ndarray, mask: np.ndarray, path: str | Path) -> None:
    overlay = rgb.copy()
    road_color = np.array([255, 60, 40], dtype=np.uint8)
    overlay[mask > 0] = (0.55 * overlay[mask > 0] + 0.45 * road_color).astype(np.uint8)
    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr)
