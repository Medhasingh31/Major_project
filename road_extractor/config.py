from pathlib import Path
from dataclasses import dataclass, field


def get_default_weights_path() -> Path:
    """Return occlusion-resilient model if available, otherwise baseline model."""
    p_occ = Path("models/road_unet_occlusion.keras")
    if p_occ.exists():
        return p_occ
    return Path("models/road_unet.keras")


@dataclass
class ExtractionConfig:
    image_size: int = 256
    tile_size: int = 256
    tile_stride: int = 128          # 50% overlap for seamless sliding-window blending
    use_tiling: bool = True         # Run tiled sliding-window inference
    use_multiscale: bool = True     # Multi-scale contextual pyramid (Global context + Local detail)
    global_weight: float = 0.25     # Weight for global macroscopic semantic context
    local_weight: float = 0.75      # Weight for local high-resolution tiled detail
    threshold: float = 0.25         # Ground-truth-selected precision/recall baseline
    use_hysteresis: bool = True     # Hysteresis thresholding to track connected roads through shadows/canopy
    hysteresis_low_ratio: float = 0.50 # Ratio of lower threshold (e.g. 0.125 when high is 0.25)
    min_object_size: int = 32       # Keeps fine winding rural pathways
    closing_radius: int = 6         # Morphological closing radius across canopy breaks
    bridge_kernel_size: int = 21    # Directional multi-angle bridge kernel for tree occlusion gaps
    collinear_max_gap: float = 400.0# Maximum distance for collinear dead-end bridging
    collinear_max_angle: float = 20.0# Maximum angle deviation for collinear bridge candidates
    graph_simplify_pixels: int = 12 # Road graph simplification tolerance


@dataclass
class TrainingConfig:
    image_size: int = 256
    batch_size: int = 4
    epochs: int = 20
    learning_rate: float = 1e-3
    base_filters: int = 16
    loss_name: str = "combined"   # 'combined' | 'dice' | 'bce'
    val_fraction: float = 0.2
