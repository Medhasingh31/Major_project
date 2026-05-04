import cv2
import numpy as np

try:
    from skimage import morphology
except ImportError:
    morphology = None


def _remove_small_components(binary: np.ndarray, min_size: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    cleaned = np.zeros(binary.shape, dtype=np.uint8)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == label] = 1
    return cleaned.astype(bool)


def _remove_small_holes(binary: np.ndarray, min_size: int) -> np.ndarray:
    inverted = np.logical_not(binary)
    cleaned_background = _remove_small_components(inverted, min_size)
    return np.logical_not(cleaned_background)


def _opencv_skeletonize(binary: np.ndarray) -> np.ndarray:
    image = binary.astype(np.uint8) * 255
    skeleton = np.zeros(image.shape, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while cv2.countNonZero(image) > 0:
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        temporary = cv2.subtract(image, opened)
        eroded = cv2.erode(image, element)
        skeleton = cv2.bitwise_or(skeleton, temporary)
        image = eroded

    return (skeleton > 0).astype(np.uint8)


def classical_road_candidate_mask(rgb: np.ndarray) -> np.ndarray:
    """CPU-only fallback useful for demos when no trained model is available."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    low_saturation = saturation < 65
    bright_surface = value > 80

    candidate = np.logical_or(edges > 0, np.logical_and(low_saturation, bright_surface))
    if morphology:
        candidate = morphology.remove_small_objects(candidate, min_size=80)
    else:
        candidate = _remove_small_components(candidate, min_size=80)
    return candidate.astype(np.uint8)


def repair_road_mask(
    mask: np.ndarray,
    min_object_size: int = 64,
    closing_radius: int = 3,
    bridge_kernel_size: int = 9,
) -> np.ndarray:
    """
    MAIN PROJECT NOVELTY: Post-processing to improve road continuity.
    This function takes the raw neural network mask and repairs it using:
    1. Noise Removal: Removes small isolated pixel clusters (false positives).
    2. Morphological Closing: Fills small holes inside road segments.
    3. Gap Filling: Uses elongated horizontal and vertical kernels to bridge 
       larger gaps caused by occlusions (like trees, vehicles, or shadows).
    """
    binary = mask > 0
    if morphology:
        binary = morphology.remove_small_objects(binary, min_size=min_object_size)
        binary = morphology.remove_small_holes(binary, area_threshold=min_object_size)
        selem = morphology.disk(closing_radius)
        closed = morphology.binary_closing(binary, selem)
    else:
        binary = _remove_small_components(binary, min_size=min_object_size)
        binary = _remove_small_holes(binary, min_size=min_object_size)
        kernel_size = closing_radius * 2 + 1
        selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, selem) > 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_kernel_size, 1))
    horizontal = cv2.morphologyEx(closed.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    vertical = cv2.morphologyEx(closed.astype(np.uint8), cv2.MORPH_CLOSE, kernel.T)
    repaired = np.logical_or(closed, np.logical_or(horizontal > 0, vertical > 0))

    return repaired.astype(np.uint8)


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """
    MAIN PROJECT NOVELTY: Skeletonization.
    Reduces the repaired road mask to a 1-pixel wide centerline.
    This is the crucial transition step required before generating the graph/network representation.
    """
    if morphology:
        skeleton = morphology.skeletonize(mask > 0)
        return skeleton.astype(np.uint8)
    return _opencv_skeletonize(mask > 0)
