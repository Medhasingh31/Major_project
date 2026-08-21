from pathlib import Path
import json

import cv2
import numpy as np

from road_extractor.config import ExtractionConfig
from road_extractor.data import read_rgb, resize_image
from road_extractor.geometry import (
    GeometryConfig,
    extract_geometry,
    extract_skeleton,
    save_clean_mask_image,
    save_geometry_diagnostic,
    save_geometry_json,
    save_skeleton_image,
)
from road_extractor.graph import export_geojson, export_graphml
from road_extractor.postprocess import classical_road_candidate_mask, repair_road_mask
from road_extractor.topology import build_topology, save_topology_json
from road_extractor.visualize import (
    save_final_graph_overlay,
    save_mask,
    save_overlay,
    save_rgb_image,
    save_topology_overlay,
)


def _prepare_graph_exports(topology, geometry) -> None:
    """Attach Geometry paths to the Topology graph for the existing exporters."""
    if topology.graph is None:
        return

    segments = {segment.segment_id: segment for segment in geometry.segments}
    for source, target, data in topology.graph.edges(data=True):
        segment = segments.get(data.get("segment_id"))
        if segment is None:
            # Generate interpolated straight-line pixel path for bridge edges
            src_node = topology.graph.nodes.get(source, {})
            dst_node = topology.graph.nodes.get(target, {})
            if "x" in src_node and "y" in src_node and "x" in dst_node and "y" in dst_node:
                x0, y0 = int(src_node["x"]), int(src_node["y"])
                x1, y1 = int(dst_node["x"]), int(dst_node["y"])
                num_pts = max(abs(x1 - x0), abs(y1 - y0), 1) + 1
                xs = np.linspace(x0, x1, num_pts).round().astype(int)
                ys = np.linspace(y0, y1, num_pts).round().astype(int)
                data["pixels"] = json.dumps([[int(x), int(y)] for x, y in zip(xs, ys)])
            elif "pixels" not in data:
                data["pixels"] = json.dumps([])
            data["length"] = int(round(float(data.get("length_pixels", 0.0))))
            continue
        data["pixels"] = json.dumps(
            [[int(x), int(y)] for y, x in segment.pixel_path]
        )
        data["length"] = int(round(float(segment.length_pixels)))


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
    progress_file_path: str | None = None,
) -> np.ndarray:
    """
    Overlapping sliding-window tiled inference at native resolution.
    Applies 2D Hann window blending to eliminate seam artifacts and batched 4-way TTA.
    Optimized to iterate in small mini-batches to prevent memory exhaustion on large satellite images.
    """
    import gc
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

    all_coords = []
    for y in ys:
        for x in xs:
            all_coords.append((y, x))

    total_tiles = len(all_coords)

    # Process sequentially in mini-batches
    for idx in range(0, total_tiles, batch_size):
        batch_coords = all_coords[idx : idx + batch_size]
        patches = []
        for y, x in batch_coords:
            patch = image_rgb[y:y+tile_size, x:x+tile_size]
            if patch.shape[0] < tile_size or patch.shape[1] < tile_size:
                patch = cv2.resize(patch, (tile_size, tile_size))
            patches.append(patch.astype(np.float32) / 255.0)

        patches_arr = np.array(patches)

        # Batched 4-way Test-Time Augmentation
        p1 = model.predict(patches_arr, batch_size=len(patches), verbose=0)[:, :, :, 0]
        p2 = model.predict(np.flip(patches_arr, axis=2), batch_size=len(patches), verbose=0)[:, :, :, 0]
        p2 = np.flip(p2, axis=2)
        p3 = model.predict(np.flip(patches_arr, axis=1), batch_size=len(patches), verbose=0)[:, :, :, 0]
        p3 = np.flip(p3, axis=1)
        p4 = model.predict(np.rot90(patches_arr, k=1, axes=(1, 2)), batch_size=len(patches), verbose=0)[:, :, :, 0]
        p4 = np.rot90(p4, k=-1, axes=(1, 2))

        tta_preds = (p1 + p2 + p3 + p4) / 4.0

        # Accumulate into global probability and weight maps
        for (y, x), pred in zip(batch_coords, tta_preds):
            h_eff = min(tile_size, h - y)
            w_eff = min(tile_size, w - x)
            prob_map[y:y+h_eff, x:x+w_eff] += pred[:h_eff, :w_eff] * window[:h_eff, :w_eff]
            weight_map[y:y+h_eff, x:x+w_eff] += window[:h_eff, :w_eff]

        current_count = min(idx + batch_size, total_tiles)
        prog_msg = f"Processing tile {current_count} / {total_tiles}"
        print(prog_msg)
        if progress_file_path:
            try:
                with open(progress_file_path, 'w') as pf:
                    pf.write(prog_msg + "\n")
            except:
                pass

        # Deallocate memory explicitly
        del patches_arr, p1, p2, p3, p4, tta_preds
        gc.collect()

    return prob_map / np.maximum(weight_map, 1e-6)


def predict_mask_with_model(
    image: np.ndarray,
    weights_path: str | Path,
    config: ExtractionConfig,
    progress_file_path: str | None = None,
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
            progress_file_path=progress_file_path,
        )
        gw = getattr(config, "global_weight", 0.25)
        lw = getattr(config, "local_weight", 0.75)
        total_w = gw + lw
        prediction = (gw * pred_global_up + lw * pred_tiled) / total_w
    elif getattr(config, "use_tiling", False) and (original_height > config.tile_size or original_width > config.tile_size):
        prediction = _predict_tiled(
            model,
            image,
            tile_size=config.tile_size,
            stride=config.tile_stride,
            progress_file_path=progress_file_path,
        )
    else:
        prediction = pred_global_up

    # Apply Hysteresis thresholding to preserve continuous road corridors through shadows and tree cover
    if getattr(config, "use_hysteresis", True):
        high_thresh = config.threshold
        low_thresh = config.threshold * getattr(config, "hysteresis_low_ratio", 0.50)
        strong = (prediction >= high_thresh).astype(np.uint8)
        weak = (prediction >= low_thresh).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(weak, connectivity=8)
        mask = np.zeros_like(strong)
        for label in range(1, num_labels):
            component_mask = (labels == label)
            if np.any(strong[component_mask]):
                mask[component_mask] = 1
        return mask
    else:
        return (prediction >= config.threshold).astype(np.uint8)


def extract_roads(
    image_path: str | Path,
    output_dir: str | Path,
    weights_path: str | Path | None = None,
    config: ExtractionConfig | None = None,
    crs: str | None = None,
    transform: list[float] | None = None,
) -> dict[str, object]:
    config = config or ExtractionConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = read_rgb(image_path)
    progress_path = output_dir / "progress.txt"
    if weights_path:
        raw_mask = predict_mask_with_model(image, weights_path, config, progress_file_path=str(progress_path))
    else:
        raw_mask = classical_road_candidate_mask(image)

    repaired_mask = repair_road_mask(
        raw_mask,
        min_object_size=config.min_object_size,
        closing_radius=config.closing_radius,
        bridge_kernel_size=config.bridge_kernel_size,
    )
    geometry = extract_geometry(
        image,
        repaired_mask,
        config=GeometryConfig(
            min_road_area_px=config.min_object_size,
            collinear_max_gap_px=getattr(config, "collinear_max_gap", 400.0),
            collinear_max_angle_deg=getattr(config, "collinear_max_angle", 22.0),
        ),
    )
    topology = build_topology(geometry)
    if topology.graph is not None:
        if crs:
            topology.graph.graph['crs'] = crs
        if transform:
            topology.graph.graph['transform'] = json.dumps(transform)
    _prepare_graph_exports(topology, geometry)
    raw_skeleton = extract_skeleton(geometry.clean_mask) if geometry.clean_mask is not None else np.zeros_like(repaired_mask)

    from road_extractor.confidence import score_topology
    confidence_report = score_topology(image, repaired_mask, geometry, topology)

    save_rgb_image(image, output_dir / "original_rgb.png")
    save_mask(raw_mask, output_dir / "raw_mask.png")
    save_mask(repaired_mask, output_dir / "repaired_mask.png")
    save_clean_mask_image(geometry, output_dir / "geometry_clean_mask.png")
    save_mask(raw_skeleton, output_dir / "raw_skeleton.png")
    save_skeleton_image(geometry, output_dir / "geometry_skeleton.png")
    save_overlay(image, repaired_mask, output_dir / "overlay.png")
    save_geometry_diagnostic(image, geometry, output_dir / "geometry_diagnostic.png")
    save_topology_overlay(image, geometry, topology, output_dir / "topology_overlay.png")
    save_final_graph_overlay(image, topology.graph, output_dir / "road_graph.png")
    save_geometry_json(geometry, output_dir / "geometry_summary.json")
    save_topology_json(topology, output_dir / "topology_summary.json")
    with open(output_dir / "confidence_summary.json", "w") as f:
        json.dump(confidence_report.to_summary(), f, indent=4)

    geojson_path = output_dir / "road_network.geojson"
    graphml_path = output_dir / "road_network.graphml"
    export_geojson(topology.graph, geojson_path)
    export_graphml(topology.graph, graphml_path)

    output_files = [
        "original_rgb.png",
        "raw_mask.png",
        "repaired_mask.png",
        "geometry_clean_mask.png",
        "raw_skeleton.png",
        "geometry_skeleton.png",
        "overlay.png",
        "geometry_diagnostic.png",
        "topology_overlay.png",
        "road_graph.png",
        "geometry_summary.json",
        "topology_summary.json",
        "confidence_summary.json",
        "road_network.geojson",
        "road_network.graphml",
    ]
    metadata = geometry.metadata
    topology_metadata = topology.metadata

    return {
        "input_mask_size": metadata["input_mask_shape"],
        "road_pixels": metadata["input_road_pixels"],
        "skeleton_pixels": metadata["pruned_skeleton_pixels"],
        "connected_components": metadata["skeleton_connected_components"],
        "segments": geometry.segment_count(),
        "junctions": geometry.junction_count(),
        "endpoints": len(geometry.endpoints),
        "total_length_pixels": round(geometry.total_length(), 2),
        "topology_nodes": topology.node_count(),
        "topology_edges": topology.edge_count(),
        "topology_bridge_edges": topology_metadata["bridge_edges"],
        "topology_intersections": topology_metadata["intersection_count"],
        "topology_endpoints": topology_metadata["endpoint_count"],
        "topology_connected_components": topology_metadata["component_count"],
        "confidence_summary": confidence_report.to_summary(),
        "quantitative_metrics": {
            "road_pixels": int(metadata["input_road_pixels"]),
            "skeleton_pixels": int(metadata["pruned_skeleton_pixels"]),
            "connected_components": int(topology_metadata["component_count"]),
            "endpoints": int(topology_metadata["endpoint_count"]),
            "junctions_intersections": int(topology_metadata["intersection_count"]),
            "edges": int(topology.edge_count()),
            "suspicious_disconnected_segments": int(len(topology.disconnected_segs)),
            "total_centerline_length": round(float(geometry.total_length()), 2),
        },
        "geojson_path": str(geojson_path),
        "graphml_path": str(graphml_path),
        "suspicious_disconnected_segments": topology.disconnected_segs,
        "major_geometric_changes": {
            "dead_end_spur_pixels_removed": metadata["dead_end_spur_pixels_removed"],
            "small_component_pixels_removed": metadata["small_component_pixels_removed"],
            "small_gaps_bridged": metadata["small_gaps_bridged"],
            "junction_pixels_collapsed_to_junctions": [
                metadata["junction_pixel_count"],
                metadata["junction_count"],
            ],
        },
        "output_files": [str(output_dir / name) for name in output_files],
        "output_dir": str(output_dir),
    }
