import os
import json
import math
import networkx as nx

def point_to_segment_distance(p, a, b):
    px, py = p
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay), a
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    dist = math.hypot(px - cx, py - cy)
    return dist, (cx, cy)

def point_to_network_distance(p, ref_paths):
    min_dist = float('inf')
    for path in ref_paths:
        for i in range(len(path) - 1):
            dist, _ = point_to_segment_distance(p, path[i], path[i+1])
            if dist < min_dist:
                min_dist = dist
    return min_dist

def get_path_length(path):
    total = 0.0
    for i in range(len(path) - 1):
        total += math.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
    return total

def load_graph_from_graphml(graphml_path):
    if not os.path.exists(graphml_path):
        raise FileNotFoundError(f"GraphML file not found: {graphml_path}")
        
    G = nx.read_graphml(graphml_path)
    cleaned_G = nx.Graph()
    cleaned_G.graph.update(G.graph)
    
    node_mapping = {}
    for node, data in G.nodes(data=True):
        x = float(data.get('x', 0.0))
        y = float(data.get('y', 0.0))
        cleaned_G.add_node(str(node), x=x, y=y, kind=data.get('kind', 'junction'))
        node_mapping[node] = str(node)
        
    for u, v, data in G.edges(data=True):
        edge_data = dict(data)
        if 'length_pixels' in edge_data:
            edge_data['length_pixels'] = float(edge_data['length_pixels'])
        elif 'length' in edge_data:
            edge_data['length_pixels'] = float(edge_data['length'])
        else:
            edge_data['length_pixels'] = 0.0
            
        cleaned_G.add_edge(node_mapping[u], node_mapping[v], **edge_data)
        
    return cleaned_G

def connect_point_to_graph(G, point, name_prefix, snap_threshold=150.0):
    px, py = point
    min_dist = float('inf')
    best_edge = None
    best_point_on_edge = None
    best_split_idx = -1
    
    nearest_node = None
    min_node_dist = float('inf')
    for node, data in G.nodes(data=True):
        dist = math.hypot(data['x'] - px, data['y'] - py)
        if dist < min_node_dist:
            min_node_dist = dist
            nearest_node = node
            
    for u, v, data in G.edges(data=True):
        pixels_str = data.get('pixels', '')
        if not pixels_str:
            continue
        try:
            pixels = json.loads(pixels_str)
        except Exception:
            continue
            
        for i in range(len(pixels) - 1):
            dist, pt = point_to_segment_distance(point, pixels[i], pixels[i+1])
            if dist < min_dist:
                min_dist = dist
                best_edge = (u, v)
                best_point_on_edge = pt
                best_split_idx = i

    # Snapping Boundary Check (Radius Validation)
    if min_node_dist > snap_threshold and min_dist > snap_threshold:
        if name_prefix == "start":
            raise ValueError("No usable road network found near the selected start point.")
        else:
            raise ValueError("No usable road network found near the selected destination point.")

    # Snapping logic: closest node vs splitting edge segment
    if min_node_dist <= 15.0 or (min_node_dist <= min_dist and min_node_dist <= snap_threshold):
        new_node_id = f"{name_prefix}_click"
        G.add_node(new_node_id, x=px, y=py, kind="click_point")
        nearest_data = G.nodes[nearest_node]
        snapped_coords = (nearest_data['x'], nearest_data['y'])
        
        connector_coords = [[px, py], [nearest_data['x'], nearest_data['y']]]
        G.add_edge(
            new_node_id,
            nearest_node,
            pixels=json.dumps(connector_coords),
            length_pixels=min_node_dist,
            edge_kind="connector_segment"
        )
        return new_node_id, snapped_coords

    # Split the closest edge segment
    u, v = best_edge
    edge_data = G[u][v]
    pixels = json.loads(edge_data['pixels'])
    
    split_x, split_y = best_point_on_edge
    left_coords = pixels[:best_split_idx + 1] + [[split_x, split_y]]
    right_coords = [[split_x, split_y]] + pixels[best_split_idx + 1:]
    
    new_node_id = f"{name_prefix}_node"
    G.add_node(new_node_id, x=split_x, y=split_y, kind="routing_pin")
    G.remove_edge(u, v)
    
    left_len = get_path_length(left_coords)
    left_data = dict(edge_data)
    left_data['pixels'] = json.dumps(left_coords)
    left_data['length_pixels'] = left_len
    left_data['edge_kind'] = 'split_segment'
    G.add_edge(u, new_node_id, **left_data)
    
    right_len = get_path_length(right_coords)
    right_data = dict(edge_data)
    right_data['pixels'] = json.dumps(right_coords)
    right_data['length_pixels'] = right_len
    right_data['edge_kind'] = 'split_segment'
    G.add_edge(new_node_id, v, **right_data)
    
    click_node_id = f"{name_prefix}_click"
    G.add_node(click_node_id, x=px, y=py, kind="click_point")
    connector_coords = [[px, py], [split_x, split_y]]
    G.add_edge(
        click_node_id,
        new_node_id,
        pixels=json.dumps(connector_coords),
        length_pixels=min_dist,
        edge_kind="connector_segment"
    )
    return click_node_id, (split_x, split_y)

def get_ordered_edge_pixels(G, u, v):
    edge_data = G[u][v]
    if 'pixels' in edge_data:
        try:
            pixels = json.loads(edge_data['pixels'])
        except Exception:
            pixels = []
    else:
        pixels = []
        
    if not pixels:
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        pixels = [[u_data['x'], u_data['y']], [v_data['x'], v_data['y']]]
        
    u_data = G.nodes[u]
    ux, uy = u_data['x'], u_data['y']
    dist_start = math.hypot(pixels[0][0] - ux, pixels[0][1] - uy)
    dist_end = math.hypot(pixels[-1][0] - ux, pixels[-1][1] - uy)
    
    if dist_end < dist_start:
        return list(reversed(pixels))
    return pixels

def path_overlap_ratio(path1_coords, path2_coords, min_separation):
    """
    Calculates what fraction of path1_coords length is within min_separation of path2_coords.
    """
    if not path1_coords or not path2_coords:
        return 0.0
    
    total_len = 0.0
    overlap_len = 0.0
    
    for i in range(len(path1_coords) - 1):
        pt1 = path1_coords[i]
        pt2 = path1_coords[i+1]
        seg_len = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        total_len += seg_len
        
        # Check midpoint of the segment for proximity to path2
        mid_pt = [(pt1[0] + pt2[0]) / 2.0, (pt1[1] + pt2[1]) / 2.0]
        
        # Compute min distance from mid_pt to all segments of path2
        min_dist = float('inf')
        for j in range(len(path2_coords) - 1):
            dist, _ = point_to_segment_distance(mid_pt, path2_coords[j], path2_coords[j+1])
            if dist < min_dist:
                min_dist = dist
        
        if min_dist <= min_separation:
            overlap_len += seg_len
            
    return (overlap_len / total_len) if total_len > 0.0 else 0.0

def discover_point_to_point_routes(
    graphml_path,
    start_pt,
    end_pt,
    ref_paths,
    tolerance=15.0,
    avoidance_weight=10.0,
    max_routes=4,
    allow_existing_roads=True,
    snap_threshold=150.0,
    min_separation=10.0,
    min_diversity=0.3
):
    """
    Snaps start and end coordinates to the network graph, applies baseline overlap penalties,
    discovers alternative paths with parallel corridor penalization, filters candidates to satisfy
    minimum diversity requirements, and scores each route options.
    """
    if math.hypot(start_pt[0] - end_pt[0], start_pt[1] - end_pt[1]) < 2.0:
        raise ValueError("Start and destination must be different.")
        
    G = load_graph_from_graphml(graphml_path)
    
    start_node, start_snapped = connect_point_to_graph(G, start_pt, "start", snap_threshold)
    end_node, end_snapped = connect_point_to_graph(G, end_pt, "end", snap_threshold)
    
    for u, v, data in G.edges(data=True):
        pixels_str = data.get('pixels', '')
        if pixels_str:
            try:
                coords = json.loads(pixels_str)
            except Exception:
                coords = []
        else:
            coords = []
            
        if not coords:
            u_data = G.nodes[u]
            v_data = G.nodes[v]
            coords = [[u_data['x'], u_data['y']], [v_data['x'], v_data['y']]]
            
        overlap_count = 0
        for pt in coords:
            dist = point_to_network_distance(pt, ref_paths)
            if dist <= tolerance:
                overlap_count += 1
        overlap_ratio = overlap_count / len(coords) if coords else 0.0
        
        edge_len = data.get('length_pixels') or get_path_length(coords)
        
        if not allow_existing_roads and overlap_ratio > 0.1:
            weight = 1e9
        else:
            weight = edge_len * (1.0 + overlap_ratio * (avoidance_weight - 1.0))
            
        G[u][v]['weight'] = weight
        G[u][v]['length_pixels'] = edge_len
        G[u][v]['overlap_ratio'] = overlap_ratio
        
    temp_G = G.copy()
    pool_size = max(15, max_routes * 3)
    candidate_paths = []
    
    for route_idx in range(pool_size):
        try:
            path_nodes = nx.shortest_path(temp_G, start_node, end_node, weight='weight')
            candidate_paths.append(path_nodes)
            
            # Penalize edges in this path for subsequent routing searches
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i+1]
                if temp_G.has_edge(u, v):
                    temp_G[u][v]['weight'] = temp_G[u][v]['weight'] * 5.0
            
            # Dynamic parallel corridor penalization
            current_pixels = []
            for i in range(len(path_nodes) - 1):
                current_pixels.extend(get_ordered_edge_pixels(G, path_nodes[i], path_nodes[i+1]))
                
            if current_pixels:
                for u, v, data in temp_G.edges(data=True):
                    if data.get('weight', 0.0) >= 1e8:
                        continue
                    
                    edge_pixels = get_ordered_edge_pixels(G, u, v)
                    if not edge_pixels:
                        continue
                        
                    mid_idx = len(edge_pixels) // 2
                    mid_pt = edge_pixels[mid_idx]
                    
                    min_d = float('inf')
                    for k in range(len(current_pixels) - 1):
                        dist, _ = point_to_segment_distance(mid_pt, current_pixels[k], current_pixels[k+1])
                        if dist < min_d:
                            min_d = dist
                            
                    if min_d <= min_separation:
                        temp_G[u][v]['weight'] = temp_G[u][v]['weight'] * 3.0
                        
        except nx.NetworkXNoPath:
            break
            
    # Case 4: No path exists
    if not candidate_paths:
        raise ValueError("No connected route was found between the selected points.")
        
    # Calculate absolute shortest path length for excessive length penalty (L_min)
    try:
        min_length_pixels = nx.shortest_path_length(G, start_node, end_node, weight='length_pixels')
    except nx.NetworkXNoPath:
        min_length_pixels = 0.0
        
    # Load metadata (confidence & topology summaries) if they exist
    confidence_data = {}
    disconnected_segs = set()
    
    dir_name = os.path.dirname(graphml_path)
    confidence_path = os.path.join(dir_name, "confidence_summary.json")
    topology_path = os.path.join(dir_name, "topology_summary.json")
    
    if os.path.exists(confidence_path):
        try:
            with open(confidence_path, 'r') as f:
                confidence_data = json.load(f).get("segments", {})
        except Exception:
            pass
            
    if os.path.exists(topology_path):
        try:
            with open(topology_path, 'r') as f:
                topo_summary = json.load(f)
                disconnected_segs = set(topo_summary.get("suspicious_segments", []))
        except Exception:
            pass
            
    candidates_list = []
    for path_nodes in candidate_paths:
        route_coords = []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i+1]
            seg_coords = get_ordered_edge_pixels(G, u, v)
            if not route_coords:
                route_coords.extend(seg_coords)
            else:
                if route_coords[-1] == seg_coords[0]:
                    route_coords.extend(seg_coords[1:])
                else:
                    route_coords.extend(seg_coords)
                    
        if not route_coords:
            continue
            
        total_len_pixels = 0.0
        total_overlap_len_pixels = 0.0
        geometry_bridge_len = 0.0
        disconnected_len = 0.0
        
        confidence_vals = []
        straightness_vals = []
        
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i+1]
            edge_data = G[u][v]
            edge_len = edge_data.get('length_pixels', 0.0)
            overlap_ratio = edge_data.get('overlap_ratio', 0.0)
            edge_kind = edge_data.get('edge_kind', '')
            
            total_len_pixels += edge_len
            total_overlap_len_pixels += edge_len * overlap_ratio
            
            if edge_kind == 'geometry_bridge':
                geometry_bridge_len += edge_len
                
            segment_id = edge_data.get('segment_id')
            if segment_id is not None:
                seg_id_str = str(int(segment_id))
                if int(segment_id) in disconnected_segs:
                    disconnected_len += edge_len
                    
                if seg_id_str in confidence_data:
                    sc = confidence_data[seg_id_str]
                    confidence_vals.append((edge_len, sc.get("overall", 1.0)))
                    dims = sc.get("dimensions", {})
                    straightness_vals.append((edge_len, dims.get("straightness", 1.0)))
            else:
                if edge_kind == 'geometry_bridge':
                    confidence_vals.append((edge_len, 0.4))
                    straightness_vals.append((edge_len, 0.8))
                elif edge_kind == 'connector_segment':
                    confidence_vals.append((edge_len, 1.0))
                    straightness_vals.append((edge_len, 1.0))
                    
        length_meters = total_len_pixels * 0.15
        overlap_length_meters = total_overlap_len_pixels * 0.15
        novel_length_meters = max(0.0, length_meters - overlap_length_meters)
        overlap_percentage = (overlap_length_meters / length_meters * 100.0) if length_meters > 0 else 0.0
        
        if confidence_vals:
            total_w = sum(w for w, _ in confidence_vals)
            extraction_confidence = sum(w * val for w, val in confidence_vals) / total_w if total_w > 0 else 0.8
        else:
            extraction_confidence = 0.8
            
        if straightness_vals:
            total_w = sum(w for w, _ in straightness_vals)
            route_smoothness = sum(w * val for w, val in straightness_vals) / total_w if total_w > 0 else 0.9
        else:
            route_smoothness = 0.9
            
        candidates_list.append({
            "path_nodes": path_nodes,
            "coordinates": route_coords,
            "length_meters": length_meters,
            "novel_length_meters": novel_length_meters,
            "overlap_length_meters": overlap_length_meters,
            "overlap_percentage": overlap_percentage,
            "total_len_pixels": total_len_pixels,
            "geometry_bridge_len": geometry_bridge_len,
            "disconnected_len": disconnected_len,
            "extraction_confidence": extraction_confidence,
            "route_smoothness": route_smoothness
        })
        
    selected_candidates = []
    for cand in candidates_list:
        if len(selected_candidates) >= max_routes:
            break
            
        too_similar = False
        for acc in selected_candidates:
            overlap = path_overlap_ratio(cand["coordinates"], acc["coordinates"], min_separation)
            overlap_rev = path_overlap_ratio(acc["coordinates"], cand["coordinates"], min_separation)
            max_mutual_overlap = max(overlap, overlap_rev)
            
            if max_mutual_overlap > (1.0 - min_diversity):
                too_similar = True
                break
                
        if not too_similar:
            selected_candidates.append(cand)
            
    # Case 5: Paths exist but all substantially overlap the existing network
    if not selected_candidates or all(c["overlap_percentage"] >= 90.0 for c in selected_candidates):
        raise ValueError("No sufficiently new route alternatives were found.")
        
    routes_list = []
    min_length_meters = min_length_pixels * 0.15
    
    for idx, cand in enumerate(selected_candidates):
        other_selected_coords = [acc["coordinates"] for i, acc in enumerate(selected_candidates) if i != idx]
        if other_selected_coords:
            max_div_overlap = max(
                max(
                    path_overlap_ratio(cand["coordinates"], other, min_separation),
                    path_overlap_ratio(other, cand["coordinates"], min_separation)
                )
                for other in other_selected_coords
            )
            route_diversity = 1.0 - max_div_overlap
        else:
            route_diversity = 1.0
            
        conn_quality = 1.0 - (cand["geometry_bridge_len"] / cand["total_len_pixels"]) if cand["total_len_pixels"] > 0 else 1.0
        prob_segments = cand["disconnected_len"] / cand["total_len_pixels"] if cand["total_len_pixels"] > 0 else 0.0
        
        excess_ratio = max(0.0, (cand["length_meters"] - min_length_meters) / min_length_meters) if min_length_meters > 0 else 0.0
        length_penalty = min(1.0, excess_ratio)
        overlap_ratio = cand["overlap_percentage"] / 100.0
        
        # Weights for composite scoring formula
        w_conn = 0.25
        w_conf = 0.25
        w_div = 0.25
        w_smooth = 0.15
        w_len_pen = 0.15
        w_overlap_pen = 0.15
        w_prob_pen = 0.10
        
        score = (
            w_conn * conn_quality
            + w_conf * cand["extraction_confidence"]
            + w_div * route_diversity
            + w_smooth * cand["route_smoothness"]
            - w_len_pen * length_penalty
            - w_overlap_pen * overlap_ratio
            - w_prob_pen * prob_segments
        )
        score = max(0.0, min(1.0, score))
        
        routes_list.append({
            "route_id": idx + 1,
            "coordinates": cand["coordinates"],
            "length_meters": cand["length_meters"],
            "novel_length_meters": cand["novel_length_meters"],
            "overlap_length_meters": cand["overlap_length_meters"],
            "overlap_percentage": cand["overlap_percentage"],
            "score": round(score, 3)
        })
        
    # Case 6: Fewer alternatives explain message
    info_message = None
    if 0 < len(routes_list) < max_routes:
        info_message = "Fewer route alternatives were available."
        
    return routes_list, start_snapped, end_snapped, info_message
