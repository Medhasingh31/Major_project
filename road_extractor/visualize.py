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
