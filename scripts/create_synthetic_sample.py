from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    output = Path("data/sample")
    output.mkdir(parents=True, exist_ok=True)

    image = np.zeros((384, 384, 3), dtype=np.uint8)
    image[:] = [58, 95, 62]

    rng = np.random.default_rng(7)
    noise = rng.normal(0, 12, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    roads = [
        ((20, 70), (360, 95), 11),
        ((60, 340), (335, 40), 9),
        ((185, 20), (205, 360), 8),
        ((30, 245), (360, 260), 10),
    ]
    for start, end, width in roads:
        cv2.line(image, start, end, (178, 178, 170), width, lineType=cv2.LINE_AA)

    cv2.rectangle(image, (170, 150), (220, 185), (45, 88, 55), -1)
    cv2.imwrite(str(output / "input.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(output / "input.png")


if __name__ == "__main__":
    main()
