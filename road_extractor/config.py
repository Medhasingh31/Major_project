from dataclasses import dataclass, field


@dataclass
class ExtractionConfig:
    image_size: int = 256
    tile_size: int = 256
    tile_stride: int = 128          # 50% overlap for seamless sliding-window blending
    use_tiling: bool = True         # Run tiled sliding-window inference
    use_multiscale: bool = True     # Multi-scale contextual pyramid (Global context + Local detail)
    global_weight: float = 0.60     # Weight for global macroscopic semantic context
    local_weight: float = 0.40      # Weight for local high-resolution tiled detail
    threshold: float = 0.24         # Balanced threshold for thin rural paths and urban grids
    min_object_size: int = 48       # Removes isolated noise specks while keeping connected road networks
    closing_radius: int = 4         # Morphological closing radius
    bridge_kernel_size: int = 11    # Directional bridge kernel size
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
