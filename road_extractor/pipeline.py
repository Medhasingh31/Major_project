from pathlib import Path

import cv2
import numpy as np

from road_extractor.config import ExtractionConfig
from road_extractor.data import read_rgb, resize_image
from road_extractor.graph import export_geojson, export_graphml, skeleton_to_graph
from road_extractor.postprocess import classical_road_candidate_mask, repair_road_mask, skeletonize_mask
from road_extractor.visualize import save_graph_plot, save_mask, save_overlay, save_pipeline_summary_plot


def predict_mask_with_model(
    image: np.ndarray,
    weights_path: str | Path,
    config: ExtractionConfig,
) -> np.ndarray:
    from tensorflow import keras

    original_height, original_width = image.shape[:2]
    resized = resize_image(image, config.image_size).astype(np.float32) / 255.0
    model = keras.models.load_model(str(weights_path), compile=False)
    prediction = model.predict(resized[None, ...], verbose=0)[0, :, :, 0]
    prediction = cv2.resize(prediction, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
    return (prediction >= config.threshold).astype(np.uint8)


def extract_roads(
    image_path: str | Path,
    output_dir: str | Path,
    weights_path: str | Path | None = None,
    config: ExtractionConfig | None = None,
) -> dict[str, object]:
    config = config or ExtractionConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = read_rgb(image_path)
    if weights_path:
        raw_mask = predict_mask_with_model(image, weights_path, config)
    else:
        raw_mask = classical_road_candidate_mask(image)

    repaired_mask = repair_road_mask(
        raw_mask,
        min_object_size=config.min_object_size,
        closing_radius=config.closing_radius,
        bridge_kernel_size=config.bridge_kernel_size,
    )
    skeleton = skeletonize_mask(repaired_mask)
    graph = skeleton_to_graph(skeleton)

    save_mask(raw_mask, output_dir / "raw_mask.png")
    save_mask(repaired_mask, output_dir / "repaired_mask.png")
    save_mask(skeleton, output_dir / "skeleton.png")
    save_overlay(image, repaired_mask, output_dir / "overlay.png")
    save_graph_plot(graph, output_dir / "graph_plot.png", image.shape[:2])
    save_pipeline_summary_plot(image, raw_mask, repaired_mask, skeleton, graph, output_dir / "pipeline_summary.png")
    export_graphml(graph, output_dir / "road_graph.graphml")
    export_geojson(graph, output_dir / "road_graph.geojson")

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "output_dir": str(output_dir),
    }
