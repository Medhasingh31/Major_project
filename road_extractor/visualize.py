from pathlib import Path
import json

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


def save_rgb_image(rgb: np.ndarray, path: str | Path) -> None:
    """Save an RGB image using the repository's OpenCV BGR convention."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _draw_graph_edges(canvas: np.ndarray, graph, edge_color: tuple[int, int, int]) -> None:
    """Draw graph edge pixel paths on an RGB canvas."""
    for _, _, data in graph.edges(data=True):
        try:
            pixels = json.loads(data.get("pixels", "[]"))
        except (TypeError, json.JSONDecodeError):
            pixels = []
        points = np.asarray(pixels, dtype=np.int32)
        if len(points) >= 2:
            cv2.polylines(
                canvas,
                [points.reshape(-1, 1, 2)],
                isClosed=False,
                color=edge_color,
                thickness=2,
            )


def save_topology_overlay(
    rgb: np.ndarray,
    geometry,
    topology,
    path: str | Path,
) -> None:
    """Save centerlines, graph edges, endpoints, and junction markers."""
    canvas = rgb.copy()
    if geometry.skeleton is not None:
        canvas[geometry.skeleton > 0] = np.array([80, 220, 220], dtype=np.uint8)

    if topology.graph is not None:
        # Blue = graph road edges; yellow = accepted gap bridges.
        for _, _, data in topology.graph.edges(data=True):
            color = (255, 180, 40) if data.get("edge_kind") == "geometry_bridge" else (40, 100, 255)
            try:
                pixels = json.loads(data.get("pixels", "[]"))
            except (TypeError, json.JSONDecodeError):
                pixels = []
            points = np.asarray(pixels, dtype=np.int32)
            if len(points) >= 2:
                cv2.polylines(canvas, [points.reshape(-1, 1, 2)], False, color, 2)

        for node_id, data in topology.graph.nodes(data=True):
            center = (int(data["x"]), int(data["y"]))
            kind = data.get("kind")
            if kind == "endpoint":
                color, radius = (40, 220, 40), 5       # green
            elif kind == "junction":
                color, radius = (255, 50, 50), 6       # red
            else:
                color, radius = (180, 180, 180), 3     # passthrough
            cv2.circle(canvas, center, radius, color, -1 if kind != "junction" else 2)

    save_rgb_image(canvas, path)


def save_final_graph_overlay(
    rgb: np.ndarray,
    graph,
    path: str | Path,
) -> None:
    """Save the final exported graph with distinct edge/node markers."""
    canvas = rgb.copy()
    if graph is not None:
        _draw_graph_edges(canvas, graph, (40, 100, 255))
        for _, data in graph.nodes(data=True):
            kind = data.get("kind")
            color = (40, 220, 40) if kind == "endpoint" else (255, 50, 50)
            radius = 5 if kind == "endpoint" else 6
            cv2.circle(canvas, (int(data["x"]), int(data["y"])), radius, color, -1)
    save_rgb_image(canvas, path)


def save_graph_plot(graph, path: str | Path, image_shape: tuple[int, int]) -> None:
    import matplotlib
    matplotlib.use('Agg')  # Headless backend
    import matplotlib.pyplot as plt
    import networkx as nx

    plt.figure(figsize=(8, 8))
    # Nodes have 'x' and 'y' properties. Invert Y for correct image coordinate alignment in pyplot
    pos = {node: (data["x"], image_shape[0] - data["y"]) for node, data in graph.nodes(data=True)}
    
    nx.draw_networkx_nodes(graph, pos, node_size=15, node_color="red")
    nx.draw_networkx_edges(graph, pos, edge_color="blue", width=1.5)
    
    plt.axis('equal')
    plt.axis('off')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(path), bbox_inches='tight', dpi=150)
    plt.close()
def save_pipeline_summary_plot(
    image: np.ndarray,
    raw_mask: np.ndarray,
    repaired_mask: np.ndarray,
    skeleton: np.ndarray,
    graph,
    path: str | Path
) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import networkx as nx

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("1. Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(raw_mask, cmap="gray")
    axes[1].set_title("2. Predicted Mask")
    axes[1].axis("off")
    
    axes[2].imshow(repaired_mask, cmap="gray")
    axes[2].set_title("3. Post-processed (Repaired)")
    axes[2].axis("off")
    
    axes[3].imshow(skeleton, cmap="gray")
    axes[3].set_title("4. Skeletonized")
    axes[3].axis("off")
    
    axes[4].imshow(image)
    axes[4].set_title("5. Network Graph")
    axes[4].axis("off")
    
    # Coordinates align perfectly with imshow (Y goes down)
    pos = {node: (data["x"], data["y"]) for node, data in graph.nodes(data=True)}
    nx.draw_networkx_nodes(graph, pos, ax=axes[4], node_size=15, node_color="red")
    nx.draw_networkx_edges(graph, pos, ax=axes[4], edge_color="cyan", width=1.5)
    
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(path), bbox_inches='tight', dpi=200)
    plt.close()
