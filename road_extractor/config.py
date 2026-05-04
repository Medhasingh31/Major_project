from dataclasses import dataclass


@dataclass
class ExtractionConfig:
    image_size: int = 256
    threshold: float = 0.5
    min_object_size: int = 64
    closing_radius: int = 3
    bridge_kernel_size: int = 9
    graph_simplify_pixels: int = 8


@dataclass
class TrainingConfig:
    image_size: int = 256
    batch_size: int = 4
    epochs: int = 20
    learning_rate: float = 1e-3
    base_filters: int = 16
