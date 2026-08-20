from pathlib import Path

import cv2
import numpy as np

from road_extractor.config import ExtractionConfig
from road_extractor.data import read_rgb, resize_image
from road_extractor.graph import export_geojson, export_graphml, skeleton_to_graph
from road_extractor.postprocess import classical_road_candidate_mask, repair_road_mask, skeletonize_mask
from road_extractor.visualize import save_graph_plot, save_mask, save_overlay, save_pipeline_summary_plot


def _predict_single_frame_tta(model, image_norm: np.ndarray) -> np.ndarray:
    """Run 4-way test-time augmentation (original, hflip, vflip, rot90) on a single frame."""
    p1 = model.predict(image_norm[None, ...], verbose=0)[0, :, :, 0]
    p2 = np.fliplr(model.predict(np.fliplr(image_norm)[None, ...], verbose=0)[0, :, :, 0])
    p3 = np.flipud(model.predict(np.flipud(image_norm)[None, ...], verbose=0)[0, :, :, 0])
    p4 = np.rot90(model.predict(np.rot90(image_norm)[None, ...], verbose=0)[0, :, :, 0], k=-1)
    return (p1 + p2 + p3 + p4) / 4.0


def _predict_tiled(
    model,
    image_rgb: np.ndarray,
    tile_size: int = 256,
    stride: int = 128,
    batch_size: int = 8,
) -> np.ndarray:
    """
    Overlapping sliding-window tiled inference at native resolution.
    Applies 2D Hann window blending to eliminate seam artifacts and batched 4-way TTA.
    """
    h, w = image_rgb.shape[:2]

    # Handle edge case where image is smaller than tile_size
    if h <= tile_size and w <= tile_size:
        resized = cv2.resize(image_rgb, (tile_size, tile_size)).astype(np.float32) / 255.0
        pred = _predict_single_frame_tta(model, resized)
        return cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)

    prob_map = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)

    # 2D Hann weighting window for smooth edge tapering
    w_hann = np.hanning(tile_size)
    window = np.outer(w_hann, w_hann)
    window = np.maximum(window, 1e-4)

    # Compute step coordinates ensuring full image coverage
    ys = list(range(0, max(1, h - tile_size + 1), stride))
    if (h - tile_size) > 0 and (h - tile_size) % stride != 0:
        ys.append(h - tile_size)
    xs = list(range(0, max(1, w - tile_size + 1), stride))
    if (w - tile_size) > 0 and (w - tile_size) % stride != 0:
        xs.append(w - tile_size)

    patches = []
    coords = []
    for y in ys:
        for x in xs:
            patch = image_rgb[y:y+tile_size, x:x+tile_size]
            if patch.shape[0] < tile_size or patch.shape[1] < tile_size:
                patch = cv2.resize(patch, (tile_size, tile_size))
            patches.append(patch.astype(np.float32) / 255.0)
            coords.append((y, x))

    patches_arr = np.array(patches)

    # Batched 4-way Test-Time Augmentation
    p1 = model.predict(patches_arr, batch_size=batch_size, verbose=0)[:, :, :, 0]
    p2 = model.predict(np.flip(patches_arr, axis=2), batch_size=batch_size, verbose=0)[:, :, :, 0]
    p2 = np.flip(p2, axis=2)
    p3 = model.predict(np.flip(patches_arr, axis=1), batch_size=batch_size, verbose=0)[:, :, :, 0]
    p3 = np.flip(p3, axis=1)
    p4 = model.predict(np.rot90(patches_arr, k=1, axes=(1, 2)), batch_size=batch_size, verbose=0)[:, :, :, 0]
    p4 = np.rot90(p4, k=-1, axes=(1, 2))

    tta_preds = (p1 + p2 + p3 + p4) / 4.0

    # Accumulate into global probability and weight maps
    for (y, x), pred in zip(coords, tta_preds):
        h_eff = min(tile_size, h - y)
        w_eff = min(tile_size, w - x)
        prob_map[y:y+h_eff, x:x+w_eff] += pred[:h_eff, :w_eff] * window[:h_eff, :w_eff]
        weight_map[y:y+h_eff, x:x+w_eff] += window[:h_eff, :w_eff]

    return prob_map / np.maximum(weight_map, 1e-6)


def predict_mask_with_model(
    image: np.ndarray,
    weights_path: str | Path,
    config: ExtractionConfig,
) -> np.ndarray:
    from tensorflow import keras
    from road_extractor.model import build_light_unet

    original_height, original_width = image.shape[:2]

    # Clear session to prevent threading/graph memory leaks
    keras.backend.clear_session()

    model = build_light_unet(image_size=config.tile_size)
    model.load_weights(str(weights_path))

    # 1. Global Context Scale (macro-level semantic prior for city blocks, road continuity, and spatial layouts)
    resized_global = resize_image(image, config.image_size).astype(np.float32) / 255.0
    pred_global = _predict_single_frame_tta(model, resized_global)
    pred_global_up = cv2.resize(pred_global, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

    if getattr(config, "use_multiscale", True) and (original_height > config.tile_size or original_width > config.tile_size):
        # 2. Local High-Resolution Tiled Scale (preserves fine linear features, thin pathways, and precise road borders)
        pred_tiled = _predict_tiled(
            model,
            image,
            tile_size=config.tile_size,
            stride=config.tile_stride,
        )
        gw = getattr(config, "global_weight", 0.60)
        lw = getattr(config, "local_weight", 0.40)
        total_w = gw + lw
        prediction = (gw * pred_global_up + lw * pred_tiled) / total_w
    elif getattr(config, "use_tiling", False) and (original_height > config.tile_size or original_width > config.tile_size):
        prediction = _predict_tiled(
            model,
            image,
            tile_size=config.tile_size,
            stride=config.tile_stride,
        )
    else:
        prediction = pred_global_up

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
