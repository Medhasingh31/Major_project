import json
from pathlib import Path

import networkx as nx
import numpy as np


NEIGHBORS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _neighbor_pixels(y: int, x: int, skeleton: np.ndarray) -> list[tuple[int, int]]:
    height, width = skeleton.shape
    points = []
    for dy, dx in NEIGHBORS:
        yy, xx = y + dy, x + dx
        if 0 <= yy < height and 0 <= xx < width and skeleton[yy, xx] > 0:
            points.append((yy, xx))
    return points


def skeleton_to_graph(skeleton: np.ndarray) -> nx.Graph:
    graph = nx.Graph()
    pixels = np.argwhere(skeleton > 0)
    skeleton_points = {tuple(point) for point in pixels}

    important = set()
    for y, x in skeleton_points:
        degree = len(_neighbor_pixels(y, x, skeleton))
        if degree != 2:
            important.add((y, x))

    for index, point in enumerate(sorted(important)):
        y, x = point
        graph.add_node(index, y=int(y), x=int(x))

    point_to_node = {(data["y"], data["x"]): node for node, data in graph.nodes(data=True)}
    visited_edges = set()

    for start in important:
        start_node = point_to_node[start]
        for neighbor in _neighbor_pixels(*start, skeleton):
            edge_key = frozenset((start, neighbor))
            if edge_key in visited_edges:
                continue

            path = [start, neighbor]
            previous = start
            current = neighbor
            visited_edges.add(edge_key)

            while current not in important:
                next_candidates = [point for point in _neighbor_pixels(*current, skeleton) if point != previous]
                if not next_candidates:
                    break
                next_point = next_candidates[0]
                visited_edges.add(frozenset((current, next_point)))
                previous, current = current, next_point
                path.append(current)

            if current in point_to_node and current != start:
                end_node = point_to_node[current]
                graph.add_edge(
                    start_node,
                    end_node,
                    length=len(path),
                    pixels=json.dumps([[int(x), int(y)] for y, x in path]),
                )

    return graph


def export_graphml(graph: nx.Graph, path: str | Path) -> None:
    nx.write_graphml(graph, path)


def export_geojson(graph: nx.Graph, path: str | Path) -> None:
    features = []

    for node, data in graph.nodes(data=True):
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [data["x"], data["y"]]},
                "properties": {"id": int(node), "kind": "junction"},
            }
        )

    for source, target, data in graph.edges(data=True):
        coordinates = json.loads(data["pixels"])
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coordinates},
                "properties": {
                    "source": int(source),
                    "target": int(target),
                    "length_pixels": int(data["length"]),
                    "kind": "road_segment",
                },
            }
        )

    geojson = {"type": "FeatureCollection", "features": features}
    Path(path).write_text(json.dumps(geojson, indent=2), encoding="utf-8")
