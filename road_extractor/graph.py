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
        degree = len(_neighbor_pixels(y, x, skeleton))
        graph.add_node(
            index,
            y=int(y),
            x=int(x),
            kind="endpoint" if degree == 1 else "junction",
        )

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
    
    transform_val_raw = graph.graph.get('transform')
    if isinstance(transform_val_raw, str):
        transform_vals = json.loads(transform_val_raw)
    else:
        transform_vals = transform_val_raw
    crs_wkt = graph.graph.get('crs')
    
    def transform_pt(x, y):
        if transform_vals and crs_wkt:
            from rasterio.transform import Affine
            import rasterio.warp
            try:
                t = Affine(*transform_vals)
                wx, wy = t * (x, y)
                xs, ys = rasterio.warp.transform(crs_wkt, 'EPSG:4326', [wx], [wy])
                return [xs[0], ys[0]]
            except Exception as e:
                print(f"CRS Transformation failed: {e}")
                if transform_vals:
                    t = Affine(*transform_vals)
                    wx, wy = t * (x, y)
                    return [wx, wy]
        return [x, y]

    for node, data in graph.nodes(data=True):
        node_kind = "endpoint" if data.get("kind") == "endpoint" else "junction"
        coords = transform_pt(data["x"], data["y"])
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": coords},
                "properties": {"id": int(node), "kind": node_kind},
            }
        )

    for source, target, data in graph.edges(data=True):
        pixel_coords = json.loads(data["pixels"])
        geo_coords = [transform_pt(pt[0], pt[1]) for pt in pixel_coords]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": geo_coords},
                "properties": {
                    "source": int(source),
                    "target": int(target),
                    "length_pixels": int(data["length"]),
                    "kind": "road_segment",
                },
            }
        )

    geojson = {"type": "FeatureCollection", "features": features}
    
    if crs_wkt:
        geojson["crs"] = {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
            }
        }
        
    Path(path).write_text(json.dumps(geojson, indent=2), encoding="utf-8")
