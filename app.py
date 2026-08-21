import os
import json
import datetime
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory

from road_extractor.pipeline import extract_roads
from road_extractor.config import ExtractionConfig, get_default_weights_path

# Initialize Flask App serving compiled React SPA from frontend/dist
STATIC_DIR = os.path.abspath('frontend/dist')
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Ensure required folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def format_analysis_results(result, job_id, name, study_area, image_year, filename, file_size_str, crs_wkt=None, transform_list=None, crs_name=None):
    """
    Format output from extract_roads pipeline into UI-compatible JSON schema.
    """
    res_m_px = 0.15  # standard resolution prior: 0.15 m/px
    total_length_px = result.get("total_length_pixels", 0.0)
    total_length_m = total_length_px * res_m_px
    total_length_km = round(total_length_m / 1000.0, 1)
    
    segments = result.get("segments", 0)
    avg_segment_m = round(total_length_m / segments, 1) if segments > 0 else 0.0
    
    # Connectivity calculation
    disconnected_segs = result.get("suspicious_disconnected_segments", [])
    disconnected_count = len(disconnected_segs)
    connectivity_val = round(100.0 * (1.0 - (disconnected_count / segments)), 1) if segments > 0 else 100.0
    
    # Confidence breakdown
    conf_summary = result.get("confidence_summary", {})
    network_score = conf_summary.get("network_score", 0.90)
    overall_confidence_pct = f"{round(network_score * 100, 1)}%"
    
    segments_scores = conf_summary.get("segments", {})
    high_count = 0
    mid_count = 0
    low_count = 0
    for sid, sc in segments_scores.items():
        score = sc.get("overall", 0.0)
        if score >= 0.70:
            high_count += 1
        elif score >= 0.40:
            mid_count += 1
        else:
            low_count += 1
            
    total_scored = len(segments_scores) or 1
    high_pct = int(round(high_count / total_scored * 100))
    mid_pct = int(round(mid_count / total_scored * 100))
    low_pct = max(0, 100 - high_pct - mid_pct)
    
    # Flagged issues & Review center mapping
    flags = conf_summary.get("flags", [])
    flagged_issues = []
    review_items = []
    
    mask_size = result.get("input_mask_size", [800, 800])
    img_height = mask_size[0]
    img_width = mask_size[1]
    
    # Read sub-summaries for coordinate lookups
    geom_summary_path = Path(result["output_dir"]) / "geometry_summary.json"
    geom_data = {}
    if geom_summary_path.exists():
        try:
            with open(geom_summary_path, 'r') as f:
                geom_data = json.load(f)
        except Exception:
            pass
            
    topo_summary_path = Path(result["output_dir"]) / "topology_summary.json"
    topo_data = {}
    if topo_summary_path.exists():
        try:
            with open(topo_summary_path, 'r') as f:
                topo_data = json.load(f)
        except Exception:
            pass
            
    seg_coords = {}
    if "segments" in geom_data:
        for seg in geom_data["segments"]:
            path = seg.get("pixel_path", [])
            if path:
                mid = path[len(path) // 2]
                seg_coords[seg["segment_id"]] = (mid[0], mid[1])
                
    node_coords = {}
    if "nodes" in topo_data:
        for node in topo_data["nodes"]:
            node_coords[node["node_id"]] = (node.get("y", 0), node.get("x", 0))
            
    for idx, flag in enumerate(flags):
        issue_id = f"issue-{idx+1}"
        rev_id = f"rev-{idx+1}"
        
        sid = flag.get("segment_id")
        nid = flag.get("node_id")
        
        pixel_y, pixel_x = 0, 0
        ref_str = ""
        if sid is not None:
            pixel_y, pixel_x = seg_coords.get(sid, (0, 0))
            ref_str = f"SEGMENT #{sid}"
        elif nid is not None:
            pixel_y, pixel_x = node_coords.get(nid, (0, 0))
            ref_str = f"NODE #{nid}"
            
        x_pct = round((pixel_x / img_width) * 100, 2) if img_width > 0 else 0.0
        y_pct = round((pixel_y / img_height) * 100, 2) if img_height > 0 else 0.0
        
        # Geodetic mapping mirroring frontend NetworkMap formulas to center issues perfectly
        lat = 31.97 - (pixel_y / img_height) * 0.02 if img_height > 0 else 31.97
        lng = 97.24 - (pixel_x / img_width) * 0.03 if img_width > 0 else 97.24
        lat_lng_str = f"{lat:.4f}° N, {lng:.4f}° W"
        
        confidence_val = 90.0
        if sid is not None and str(sid) in segments_scores:
            confidence_val = round(segments_scores[str(sid)].get("overall", 0.90) * 100)
            
        issue_item = {
            "id": issue_id,
            "description": flag.get("message"),
            "reference": ref_str,
            "category": "TOPOLOGY" if "TOPOLOGY" in flag.get("kind", "") or nid is not None else "GEOMETRY",
            "coords": { "x": x_pct, "y": y_pct },
            "latLng": lat_lng_str,
            "confidence": confidence_val
        }
        flagged_issues.append(issue_item)
        
        if flag.get("severity") in ("warning", "error"):
            review_items.append({
                "id": rev_id,
                "issueType": "Topology Issue" if issue_item["category"] == "TOPOLOGY" else "Geometry Issue",
                "description": flag.get("message"),
                "reference": ref_str,
                "location": f"{ref_str} ({lat_lng_str})",
                "category": issue_item["category"],
                "confidence": confidence_val,
                "reason": flag.get("message"),
                "level": "low" if confidence_val < 40 else "mid" if confidence_val < 70 else "high",
                "coords": { "x": x_pct, "y": y_pct },
                "latLng": lat_lng_str,
                "reviewed": False
            })

    components_list = []
    if "components" in topo_data:
        for c_idx, node_ids in enumerate(topo_data["components"]):
            comp_nodes_count = len(node_ids)
            comp_edges_count = 0
            if "edges" in topo_data:
                for edge in topo_data["edges"]:
                    if edge.get("source") in node_ids or edge.get("target") in node_ids:
                        comp_edges_count += 1
            connectivity_pct = "95.0%"
            components_list.append({
                "id": f"comp-{chr(65 + c_idx)}" if c_idx < 26 else f"comp-{c_idx}",
                "label": f"Component {chr(65 + c_idx)}" if c_idx < 26 else f"Component {c_idx}",
                "nodes": comp_nodes_count,
                "edges": comp_edges_count,
                "connectivity": connectivity_pct
            })
            
    formatted = {
        "projectId": job_id,
        "projectName": name or "Meridian County Corridor",
        "location": study_area or "Meridian County, TX",
        "analysisDate": datetime.date.today().isoformat(),
        "status": "complete",
        "imageYear": image_year or "2026",
        "resolution": f"{res_m_px} m/px",
        "fileName": filename,
        "fileSize": file_size_str,
        "networkSummary": {
            "totalRoadLength": { "value": total_length_km, "unit": "km" },
            "roadSegments": { "value": segments, "unit": "segments" },
            "intersections": { "value": result.get("topology_intersections", 0), "unit": "nodes" },
            "connectedComponents": { "value": result.get("topology_connected_components", 1), "unit": "components" },
            "avgSegmentLength": { "value": avg_segment_m, "unit": "m" },
            "connectivity": f"{connectivity_val}%",
            "overallConfidence": overall_confidence_pct
        },
        "geometry": {
            "totalRoadLength": f"{total_length_km} km",
            "avgSegmentLength": f"{avg_segment_m} m",
            "geometryIssues": len([f for f in flagged_issues if f["category"] == "GEOMETRY"]),
            "roadContinuity": f"{connectivity_val}%"
        },
        "topology": {
            "intersections": result.get("topology_intersections", 0),
            "deadEnds": result.get("topology_endpoints", 0),
            "connectedComponents": result.get("topology_connected_components", 1),
            "disconnectedSegments": disconnected_count,
            "topologyIssues": len([f for f in flagged_issues if f["category"] == "TOPOLOGY"])
        },
        "healthMetrics": {
            "connectivity": { "value": connectivity_val, "label": "Connectivity", "status": "optimal" if connectivity_val >= 90 else "warning" },
            "continuity": { "value": connectivity_val, "label": "Continuity", "status": "optimal" if connectivity_val >= 90 else "warning" },
            "topologyQuality": { "value": round(network_score * 100, 1), "label": "Topology Quality", "status": "optimal" if network_score >= 0.8 else "warning" },
            "confidence": { "value": round(network_score * 100, 1), "label": "Overall Confidence", "status": "optimal" if network_score >= 0.8 else "warning" }
        },
        "confidenceBreakdown": {
            "high": high_pct,
            "mid": mid_pct,
            "low": low_pct
        },
        "flaggedIssues": flagged_issues,
        "reviewItems": review_items,
        "crs": crs_name,
        "crsWkt": crs_wkt,
        "transform": transform_list,
        "georeferenced": (crs_wkt is not None),
        "networkGraph": {
            "nodes": result.get("topology_nodes", 0),
            "edges": result.get("topology_edges", 0),
            "segments": segments,
            "connectedComponents": result.get("topology_connected_components", 1),
            "connectivity": f"{connectivity_val}%",
            "components": components_list,
            "healthMetrics": {
                "connectivity": { "value": connectivity_val, "label": "Connectivity", "status": "optimal" if connectivity_val >= 90 else "warning" },
                "continuity": { "value": connectivity_val, "label": "Continuity", "status": "optimal" if connectivity_val >= 90 else "warning" },
                "topologyQuality": { "value": round(network_score * 100, 1), "label": "Topology Quality", "status": "optimal" if network_score >= 0.8 else "warning" },
                "confidence": { "value": round(network_score * 100, 1), "label": "Overall Confidence", "status": "optimal" if network_score >= 0.8 else "warning" }
            }
        }
    }
    
    return formatted

@app.route('/', methods=['GET'])
def index():
    """Renders the main React single page app."""
    if os.path.exists(os.path.join(app.static_folder, 'index.html')):
        return send_from_directory(app.static_folder, 'index.html')
    return "React application build is missing. Please run 'npm run build' inside the Frontend directory first.", 404

@app.route('/api/health', methods=['GET'])
@app.route('/api/ping', methods=['GET'])
def health_check():
    """Health check endpoint for status probes."""
    return jsonify({"status": "ok", "service": "road_extractor_backend"}), 200

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serves the standard uploads and outputs from the local static directory."""
    return send_from_directory('static', filename)

@app.route('/api/process', methods=['POST'])
def api_process():
    """Handles image upload and runs the full road extraction pipeline, returning JSON."""
    # 1. Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded", "success": False}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename", "success": False}), 400
        
    # 2. Extract parameters from request
    job_id = request.form.get('jobId', f"job-{int(datetime.datetime.now().timestamp())}")
    name = request.form.get('name', 'Meridian County Corridor')
    study_area = request.form.get('studyArea', 'Meridian County, TX')
    image_year = request.form.get('imageYear', '2026')
    
    threshold = request.form.get('threshold', 0.30, type=float)
    closing_radius = request.form.get('closing_radius', 6, type=int)
    min_object_size = request.form.get('min_object_size', 32, type=int)
    use_model = request.form.get('use_model', 'true').lower() == 'true'
    
    # 3. Create isolated folders
    job_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    job_output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    os.makedirs(job_upload_dir, exist_ok=True)
    os.makedirs(job_output_dir, exist_ok=True)
    
    orig_ext = os.path.splitext(file.filename)[1].lower()
    if orig_ext in ['.tif', '.tiff']:
        upload_path = os.path.join(job_upload_dir, "input_image.tif")
    else:
        upload_path = os.path.join(job_upload_dir, "input_image.png")
    file.save(upload_path)
    
    file_size_bytes = os.path.getsize(upload_path)
    file_size_str = f"{round(file_size_bytes / (1024 * 1024), 1)} MB"
    
    # 3.5 Read GeoTIFF Spatial parameters if available
    crs_wkt = None
    transform_list = None
    crs_name = None
    
    if orig_ext in ['.tif', '.tiff']:
        try:
            import rasterio
            with rasterio.open(upload_path) as rsrc:
                if rsrc.crs:
                    crs_wkt = rsrc.crs.to_wkt()
                    crs_name = rsrc.crs.to_string()
                if rsrc.transform:
                    transform_list = list(rsrc.transform)[:6]
        except Exception as re:
            print(f"Failed to read rasterio metadata: {re}")
            
    # 4. Setup paths for pipeline
    weights_path = get_default_weights_path()
    weights = str(weights_path) if (weights_path.exists() and use_model) else None
    
    config = ExtractionConfig(
        threshold=threshold,
        closing_radius=closing_radius,
        min_object_size=min_object_size
    )
    
    # 5. Run the Pipeline
    try:
        result = extract_roads(
            image_path=upload_path,
            output_dir=job_output_dir,
            weights_path=weights, 
            config=config,
            crs=crs_wkt,
            transform=transform_list
        )
        
        # Format metrics to frontend expectations
        formatted_result = format_analysis_results(
            result=result,
            job_id=job_id,
            name=name,
            study_area=study_area,
            image_year=image_year,
            filename=file.filename,
            file_size_str=file_size_str,
            crs_wkt=crs_wkt,
            transform_list=transform_list,
            crs_name=crs_name
        )
        
        # Save formatted summary
        summary_save_path = os.path.join(job_output_dir, "analysis_summary.json")
        with open(summary_save_path, 'w') as sf:
            json.dump(formatted_result, sf, indent=4)
            
        try:
            compute_road_classifications(job_id)
            with open(summary_save_path, 'r') as sf:
                formatted_result = json.load(sf)
        except Exception as ce:
            print(f"Auto-classification failed for job {job_id}: {ce}")
            
        return jsonify(formatted_result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/api/analysis/<job_id>', methods=['GET'])
def api_get_analysis(job_id):
    """Fetches previously saved analysis results by jobId."""
    summary_path = os.path.join(app.config['OUTPUT_FOLDER'], job_id, "analysis_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as sf:
                data = json.load(sf)
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": f"Failed to read analysis data: {str(e)}", "success": False}), 500
    return jsonify({"error": f"Analysis with ID {job_id} not found.", "success": False}), 404

def load_or_compute_epoch(epoch_id, file, job_id, temp_dir, config, year, name, study_area):
    if job_id:
        job_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
        summary_path = os.path.join(job_dir, "analysis_summary.json")
        if not os.path.exists(summary_path):
            raise Exception(f"Run ID '{job_id}' does not have an analysis summary.")
        with open(summary_path, 'r') as sf:
            formatted_data = json.load(sf)
        path_img = os.path.join(job_dir, "original_rgb.png")
        path_mask = os.path.join(job_dir, "repaired_mask.png")
        if not os.path.exists(path_img) or not os.path.exists(path_mask):
            raise Exception(f"Run ID '{job_id}' is missing required files.")
        return formatted_data, path_img, path_mask
    else:
        if not file:
            raise Exception("No file or job_id provided for epoch.")
        
        epoch_dir = os.path.join(temp_dir, f"epoch_{epoch_id}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"{temp_dir.split('/')[-1]}_{epoch_id}")
        os.makedirs(upload_dir, exist_ok=True)
        upload_path = os.path.join(upload_dir, "input_image.png")
        file.save(upload_path)
        
        file_size_bytes = os.path.getsize(upload_path)
        file_size_str = f"{round(file_size_bytes / (1024 * 1024), 1)} MB"
        
        weights_path = get_default_weights_path()
        weights = str(weights_path) if (weights_path.exists() and config.use_model) else None
        
        pipeline_result = extract_roads(
            image_path=upload_path,
            output_dir=epoch_dir,
            weights_path=weights,
            config=config
        )
        
        formatted_data = format_analysis_results(
            result=pipeline_result,
            job_id=f"epoch_{epoch_id}",
            name=name,
            study_area=study_area,
            image_year=year,
            filename=file.filename,
            file_size_str=file_size_str
        )
        
        summary_save_path = os.path.join(epoch_dir, "analysis_summary.json")
        with open(summary_save_path, 'w') as sf:
            json.dump(formatted_data, sf, indent=4)
            
        path_img = os.path.join(epoch_dir, "original_rgb.png")
        path_mask = os.path.join(epoch_dir, "repaired_mask.png")
        
        return formatted_data, path_img, path_mask

def compare_road_networks(output_dir, path_img_a, path_img_b, path_mask_a, path_mask_b, config):
    import cv2
    import numpy as np
    from road_extractor.geometry import GeometryConfig, extract_geometry
    from road_extractor.topology import build_topology
    from road_extractor.pipeline import _prepare_graph_exports
    from road_extractor.graph import export_geojson

    img_a = cv2.imread(str(path_img_a))
    img_b = cv2.imread(str(path_img_b))
    
    mask_a = cv2.imread(str(path_mask_a), cv2.IMREAD_GRAYSCALE)
    mask_b = cv2.imread(str(path_mask_b), cv2.IMREAD_GRAYSCALE)
    
    _, mask_a = cv2.threshold(mask_a, 127, 255, cv2.THRESH_BINARY)
    _, mask_b = cv2.threshold(mask_b, 127, 255, cv2.THRESH_BINARY)

    tolerance = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1))
    
    mask_a_buffered = cv2.dilate(mask_a, kernel)
    mask_b_buffered = cv2.dilate(mask_b, kernel)

    mask_added = cv2.bitwise_and(mask_b, cv2.bitwise_not(mask_a_buffered))
    mask_removed = cv2.bitwise_and(mask_a, cv2.bitwise_not(mask_b_buffered))
    mask_unchanged = cv2.bitwise_and(mask_b, mask_a_buffered)

    geom_config = GeometryConfig(
        min_road_area_px=config.min_object_size,
        collinear_max_gap_px=400.0,
        collinear_max_angle_deg=22.0,
    )

    def process_and_export_mask(rgb_img, mask, name):
        import networkx as nx
        geom = extract_geometry(rgb_img, mask, config=geom_config)
        topo = build_topology(geom)
        _prepare_graph_exports(topo, geom)
        geojson_path = os.path.join(output_dir, f"{name}_roads.geojson")
        graph = topo.graph if topo.graph is not None else nx.Graph()
        export_geojson(graph, geojson_path)
        return geom, topo


    geom_added, topo_added = process_and_export_mask(img_b, mask_added, "added")
    geom_removed, topo_removed = process_and_export_mask(img_a, mask_removed, "removed")
    geom_unchanged, topo_unchanged = process_and_export_mask(img_b, mask_unchanged, "unchanged")

    cv2.imwrite(os.path.join(output_dir, "added_mask.png"), mask_added)
    cv2.imwrite(os.path.join(output_dir, "removed_mask.png"), mask_removed)
    cv2.imwrite(os.path.join(output_dir, "unchanged_mask.png"), mask_unchanged)

    return {
        "added": (geom_added, topo_added),
        "removed": (geom_removed, topo_removed),
        "unchanged": (geom_unchanged, topo_unchanged)
    }

@app.route('/api/compare', methods=['POST'])
def api_compare():
    try:
        job_id_a = request.form.get('job_id_a')
        job_id_b = request.form.get('job_id_b')
        
        file_a = request.files.get('file_a')
        file_b = request.files.get('file_b')
        
        if not file_a and not job_id_a:
            return jsonify({"error": "Epoch A needs either an uploaded file (file_a) or a saved project run (job_id_a).", "success": False}), 400
        if not file_b and not job_id_b:
            return jsonify({"error": "Epoch B needs either an uploaded file (file_b) or a saved project run (job_id_b).", "success": False}), 400
            
        year_a = request.form.get('year_a', '2016')
        year_b = request.form.get('year_b', '2026')
        name = request.form.get('name', 'Meridian County Comparison')
        study_area = request.form.get('studyArea', 'Meridian County, TX')
        
        threshold = request.form.get('threshold', 0.30, type=float)
        closing_radius = request.form.get('closing_radius', 6, type=int)
        min_object_size = request.form.get('min_object_size', 32, type=int)
        use_model = request.form.get('use_model', 'true').lower() == 'true'
        
        config = ExtractionConfig(
            threshold=threshold,
            closing_radius=closing_radius,
            min_object_size=min_object_size
        )
        # Add dynamic attribute so it matches load_or_compute_epoch weights lookup
        config.use_model = use_model
        
        compare_id = f"compare-{int(datetime.datetime.now().timestamp())}"
        compare_dir = os.path.join(app.config['OUTPUT_FOLDER'], compare_id)
        os.makedirs(compare_dir, exist_ok=True)
        
        data_a, img_a_path, mask_a_path = load_or_compute_epoch(
            epoch_id="a",
            file=file_a,
            job_id=job_id_a,
            temp_dir=compare_dir,
            config=config,
            year=year_a,
            name=name,
            study_area=study_area
        )
        
        data_b, img_b_path, mask_b_path = load_or_compute_epoch(
            epoch_id="b",
            file=file_b,
            job_id=job_id_b,
            temp_dir=compare_dir,
            config=config,
            year=year_b,
            name=name,
            study_area=study_area
        )
        
        import cv2
        img_a = cv2.imread(img_a_path)
        img_b = cv2.imread(img_b_path)
        if img_a is None or img_b is None:
            return jsonify({"error": "Failed to load original imagery files for comparison.", "success": False}), 500
            
        h_a, w_a = img_a.shape[:2]
        h_b, w_b = img_b.shape[:2]
        if h_a != h_b or w_a != w_b:
            return jsonify({"error": f"Incompatible image dimensions: Image A is {w_a}x{h_a}, but Image B is {w_b}x{h_b}. Images must have identical dimensions for comparison.", "success": False}), 400
            
        comparison_res = compare_road_networks(
            output_dir=compare_dir,
            path_img_a=img_a_path,
            path_img_b=img_b_path,
            path_mask_a=mask_a_path,
            path_mask_b=mask_b_path,
            config=config
        )
        
        geom_added, _ = comparison_res["added"]
        geom_removed, _ = comparison_res["removed"]
        geom_unchanged, _ = comparison_res["unchanged"]
        
        added_km = round((geom_added.total_length() * 0.15) / 1000.0, 2)
        removed_km = round((geom_removed.total_length() * 0.15) / 1000.0, 2)
        unchanged_km = round((geom_unchanged.total_length() * 0.15) / 1000.0, 2)
        
        length_a_val = data_a["networkSummary"]["totalRoadLength"]["value"]
        length_b_val = data_b["networkSummary"]["totalRoadLength"]["value"]
        delta_length = round(length_b_val - length_a_val, 2)
        
        junctions_a = data_a["networkSummary"]["intersections"]["value"]
        junctions_b = data_b["networkSummary"]["intersections"]["value"]
        delta_junctions = junctions_b - junctions_a
        
        components_a = data_a["networkSummary"]["connectedComponents"]["value"]
        components_b = data_b["networkSummary"]["connectedComponents"]["value"]
        delta_components = components_b - components_a
        
        def parse_pct(val):
            try:
                return float(str(val).replace('%', ''))
            except:
                return 100.0
        
        conn_a = parse_pct(data_a["networkSummary"]["connectivity"])
        conn_b = parse_pct(data_b["networkSummary"]["connectivity"])
        delta_connectivity = round(conn_b - conn_a, 2)
        
        import shutil
        if job_id_a:
            shutil.copy(img_a_path, os.path.join(compare_dir, "epoch_a_rgb.png"))
            img_a_static = f"/static/outputs/{compare_id}/epoch_a_rgb.png"
        else:
            img_a_static = f"/static/outputs/{compare_id}/epoch_a/original_rgb.png"
            
        if job_id_b:
            shutil.copy(img_b_path, os.path.join(compare_dir, "epoch_b_rgb.png"))
            img_b_static = f"/static/outputs/{compare_id}/epoch_b_rgb.png"
        else:
            img_b_static = f"/static/outputs/{compare_id}/epoch_b/original_rgb.png"

        result_payload = {
            "projectId": compare_id,
            "projectName": name,
            "location": study_area,
            "analysisDate": datetime.date.today().isoformat(),
            "status": "complete",
            "year_a": year_a,
            "year_b": year_b,
            "addedLengthKm": added_km,
            "removedLengthKm": removed_km,
            "unchangedLengthKm": unchanged_km,
            "deltaLengthKm": delta_length,
            "deltaJunctions": delta_junctions,
            "deltaComponents": delta_components,
            "deltaConnectivity": delta_connectivity,
            "connectivityA": conn_a,
            "connectivityB": conn_b,
            "lengthA": length_a_val,
            "lengthB": length_b_val,
            "junctionsA": junctions_a,
            "junctionsB": junctions_b,
            "componentsA": components_a,
            "componentsB": components_b,
            "addedGeojsonUrl": f"/static/outputs/{compare_id}/added_roads.geojson",
            "removedGeojsonUrl": f"/static/outputs/{compare_id}/removed_roads.geojson",
            "unchangedGeojsonUrl": f"/static/outputs/{compare_id}/unchanged_roads.geojson",
            "imageAUrl": img_a_static,
            "imageBUrl": img_b_static
        }
        
        with open(os.path.join(compare_dir, "comparison_summary.json"), 'w') as csf:
            json.dump(result_payload, csf, indent=4)
            
        return jsonify(result_payload), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/api/compare/<compare_id>', methods=['GET'])
def api_get_compare(compare_id):
    summary_path = os.path.join(app.config['OUTPUT_FOLDER'], compare_id, "comparison_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as csf:
                data = json.load(csf)
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": f"Failed to read comparison summary: {str(e)}", "success": False}), 500
    return jsonify({"error": f"Comparison with ID {compare_id} not found.", "success": False}), 404

@app.route('/api/runs', methods=['GET'])
@app.route('/api/history', methods=['GET'])
def api_get_runs():
    try:
        include_archived = request.args.get('include_archived', 'false').lower() == 'true'
        output_folder = app.config['OUTPUT_FOLDER']
        runs = []
        
        if os.path.exists(output_folder):
            for entry in os.listdir(output_folder):
                entry_path = os.path.join(output_folder, entry)
                if os.path.isdir(entry_path):
                    summary_path = os.path.join(entry_path, "analysis_summary.json")
                    if os.path.exists(summary_path):
                        try:
                            with open(summary_path, 'r') as sf:
                                run_data = json.load(sf)
                            
                            is_archived = run_data.get("archived", False)
                            if is_archived and not include_archived:
                                continue
                                
                            runs.append({
                                "id": run_data.get("projectId", entry),
                                "name": run_data.get("projectName", "Recent Run"),
                                "location": run_data.get("location", "Unknown Location"),
                                "date": run_data.get("analysisDate", ""),
                                "segments": run_data.get("networkSummary", {}).get("roadSegments", {}).get("value", 0),
                                "length": f"{run_data.get('networkSummary', {}).get('totalRoadLength', {}).get('value', 0)} {run_data.get('networkSummary', {}).get('totalRoadLength', {}).get('unit', 'km')}",
                                "status": run_data.get("status", "complete"),
                                "archived": is_archived,
                                "resolution": run_data.get("resolution", ""),
                                "fileSize": run_data.get("fileSize", ""),
                                "fileName": run_data.get("fileName", "")
                            })
                        except Exception as e:
                            print(f"Skipping corrupted run directory {entry}: {e}")
                            
        runs.sort(key=lambda r: r.get("date", ""), reverse=True)
        return jsonify(runs), 200
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/api/analysis/<job_id>', methods=['DELETE'])
def api_delete_analysis(job_id):
    try:
        import re
        if not re.match(r'^[a-zA-Z0-9_\-]+$', job_id):
            return jsonify({"error": "Invalid Job ID format. Deletion denied.", "success": False}), 400
            
        output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
        
        deleted_any = False
        import shutil
        
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            deleted_any = True
            
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
            deleted_any = True
            
        if not deleted_any:
            return jsonify({"error": f"Analysis run '{job_id}' not found.", "success": False}), 404
            
        return jsonify({"message": f"Successfully deleted analysis run '{job_id}'.", "success": True}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to delete analysis run: {str(e)}", "success": False}), 500

def point_to_segment_distance(p, a, b):
    import math
    px, py = p
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    return math.hypot(px - cx, py - cy)

def point_to_network_distance(p, ref_segments):
    min_dist = float('inf')
    for path in ref_segments:
        for i in range(len(path) - 1):
            dist = point_to_segment_distance(p, path[i], path[i+1])
            if dist < min_dist:
                min_dist = dist
    return min_dist

def interpolate_path(path, step=2.0):
    import math
    interpolated = []
    if not path or len(path) < 2:
        return interpolated
    interpolated.append(path[0])
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i+1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        if dist == 0:
            continue
        num_steps = int(dist / step)
        for s in range(1, num_steps + 1):
            t = s / num_steps
            interpolated.append((p1[0] + t * dx, p1[1] + t * dy))
    return interpolated

def get_path_length(path):
    import math
    total = 0.0
    for i in range(len(path) - 1):
        total += math.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
    return total

def parse_geojson_to_paths(geojson_data):
    paths = []
    features = geojson_data.get("features", [])
    for f in features:
        geom = f.get("geometry", {})
        if not geom:
            continue
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        if gtype == "LineString":
            if len(coords) >= 2:
                paths.append(coords)
        elif gtype == "MultiLineString":
            for sub in coords:
                if len(sub) >= 2:
                    paths.append(sub)
    return paths

@app.route('/api/discovery', methods=['POST'])
def api_discovery():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "Satellite imagery file (file) is required.", "success": False}), 400
            
        ref_file = request.files.get('ref_file')
        ref_job_id = request.form.get('ref_job_id')
        
        if not ref_file and not ref_job_id:
            return jsonify({"error": "Either reference vector file (ref_file) or previous run ID (ref_job_id) is required.", "success": False}), 400
            
        tolerance = request.form.get('tolerance', 15.0, type=float)
        min_length = request.form.get('min_length', 30.0, type=float)
        
        threshold = request.form.get('threshold', 0.25, type=float)
        closing_radius = request.form.get('closing_radius', 6, type=int)
        min_object_size = request.form.get('min_object_size', 32, type=int)
        use_model = request.form.get('use_model', 'true').lower() == 'true'
        
        name = request.form.get('name', 'Meridian Route Discovery')
        study_area = request.form.get('studyArea', 'Meridian County')
        year = request.form.get('year', '2026')
        
        ref_geojson_data = None
        if ref_job_id:
            ref_geojson_path = os.path.join(app.config['OUTPUT_FOLDER'], ref_job_id, "road_network.geojson")
            if not os.path.exists(ref_geojson_path):
                return jsonify({"error": f"Road network GeoJSON not found for job '{ref_job_id}'.", "success": False}), 400
            with open(ref_geojson_path, 'r') as rf:
                ref_geojson_data = json.load(rf)
        else:
            filename = ref_file.filename.lower()
            if not (filename.endswith('.geojson') or filename.endswith('.json')):
                return jsonify({"error": "Shapefile (.shp) format is not supported directly. Please convert your shapefile to GeoJSON and upload.", "success": False}), 400
            ref_geojson_data = json.load(ref_file)
            
        ref_paths = parse_geojson_to_paths(ref_geojson_data)
        if not ref_paths:
            return jsonify({"error": "The reference road network contains no valid LineString or MultiLineString paths.", "success": False}), 400
            
        discovery_id = f"discovery-{int(datetime.datetime.now().timestamp())}"
        discovery_dir = os.path.join(app.config['OUTPUT_FOLDER'], discovery_id)
        os.makedirs(discovery_dir, exist_ok=True)
        
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], discovery_id)
        os.makedirs(upload_dir, exist_ok=True)
        upload_path = os.path.join(upload_dir, "input_image.png")
        file.save(upload_path)
        
        config = ExtractionConfig(
            threshold=threshold,
            closing_radius=closing_radius,
            min_object_size=min_object_size
        )
        
        weights_path = get_default_weights_path()
        weights = str(weights_path) if (weights_path.exists() and use_model) else None
        
        pipeline_result = extract_roads(
            image_path=upload_path,
            output_dir=discovery_dir,
            weights_path=weights,
            config=config
        )
        
        candidate_geojson_path = os.path.join(discovery_dir, "road_network.geojson")
        if not os.path.exists(candidate_geojson_path):
            return jsonify({"error": "Road network extraction failed to generate vector GeoJSON.", "success": False}), 500
            
        with open(candidate_geojson_path, 'r') as cf:
            cand_geojson_data = json.load(cf)
        cand_paths = parse_geojson_to_paths(cand_geojson_data)
        
        unmapped_paths = []
        for path in cand_paths:
            interpolated = interpolate_path(path, step=2.0)
            if not interpolated:
                continue
                
            current_subpath = []
            for p in interpolated:
                dist = point_to_network_distance(p, ref_paths)
                if dist > tolerance:
                    current_subpath.append(p)
                else:
                    if len(current_subpath) >= 2:
                        sub_len = get_path_length(current_subpath)
                        if sub_len >= min_length:
                            unmapped_paths.append(current_subpath)
                    current_subpath = []
            if len(current_subpath) >= 2:
                sub_len = get_path_length(current_subpath)
                if sub_len >= min_length:
                    unmapped_paths.append(current_subpath)
                    
        unmapped_features = []
        for idx, path in enumerate(unmapped_paths):
            unmapped_features.append({
                "type": "Feature",
                "id": idx,
                "properties": {
                    "id": idx,
                    "length_pixels": round(get_path_length(path), 1),
                    "length_meters": round(get_path_length(path) * 0.15, 1),
                    "status": "Unmapped Candidate Route"
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": path
                }
            })
            
        unmapped_geojson = {
            "type": "FeatureCollection",
            "features": unmapped_features
        }
        
        unmapped_path = os.path.join(discovery_dir, "unmapped_routes.geojson")
        with open(unmapped_path, 'w') as uf:
            json.dump(unmapped_geojson, uf, indent=4)
            
        ref_save_path = os.path.join(discovery_dir, "reference_network.geojson")
        with open(ref_save_path, 'w') as rf:
            json.dump(ref_geojson_data, rf, indent=4)
            
        total_len_km = round(sum(get_path_length(p) * 0.15 for p in unmapped_paths) / 1000.0, 2)
        total_segments = len(unmapped_paths)
        
        result_payload = {
            "projectId": discovery_id,
            "projectName": name,
            "location": study_area,
            "analysisDate": datetime.date.today().isoformat(),
            "status": "complete",
            "candidateLengthKm": total_len_km,
            "candidateSegments": total_segments,
            "unmappedGeojsonUrl": f"/static/outputs/{discovery_id}/unmapped_routes.geojson",
            "candidateGeojsonUrl": f"/static/outputs/{discovery_id}/road_network.geojson",
            "referenceGeojsonUrl": f"/static/outputs/{discovery_id}/reference_network.geojson",
            "imageBUrl": f"/static/outputs/{discovery_id}/original_rgb.png"
        }
        
        with open(os.path.join(discovery_dir, "discovery_summary.json"), 'w') as dsf:
            json.dump(result_payload, dsf, indent=4)
            
        return jsonify(result_payload), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/api/route-discovery/point-to-point', methods=['POST'])
@app.route('/api/discovery/point-to-point', methods=['POST'])
def api_route_discovery_point_to_point():
    try:
        job_id = request.form.get('job_id')
        
        start_x = request.form.get('start_x', type=float)
        start_y = request.form.get('start_y', type=float)
        end_x = request.form.get('end_x', type=float)
        end_y = request.form.get('end_y', type=float)
        
        if start_x is None or start_y is None or end_x is None or end_y is None:
            return jsonify({"error": "Start and end coordinates are required.", "success": False}), 400
            
        # Validation: check bounds
        if not (0 <= start_x <= 1024) or not (0 <= start_y <= 1024) or not (0 <= end_x <= 1024) or not (0 <= end_y <= 1024):
            return jsonify({"error": "Coordinates must lie within the bounds [0, 1024] of the extraction imagery.", "success": False}), 400
            
        tolerance = request.form.get('tolerance', 15.0, type=float)
        avoidance_weight = request.form.get('avoidance_weight', 10.0, type=float)
        allow_existing_roads = request.form.get('allow_existing_roads', 'true').lower() == 'true'
        max_routes = request.form.get('max_routes', 4, type=int)
        min_separation = request.form.get('min_separation', 10.0, type=float)
        min_diversity = request.form.get('min_diversity', 0.3, type=float)
        
        ref_geojson_data = None
        if job_id:
            discovery_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
            graphml_path = os.path.join(discovery_dir, "road_network.graphml")
            ref_geojson_path = os.path.join(discovery_dir, "reference_network.geojson")
            
            if not os.path.exists(graphml_path):
                return jsonify({"error": f"Road network graph not found for job '{job_id}'.", "success": False}), 400
            if not os.path.exists(ref_geojson_path):
                return jsonify({"error": f"Reference network not found for job '{job_id}'.", "success": False}), 400
                
            with open(ref_geojson_path, 'r') as rf:
                ref_geojson_data = json.load(rf)
        else:
            file = request.files.get('file')
            if not file:
                return jsonify({"error": "Satellite imagery file (file) is required.", "success": False}), 400
                
            ref_file = request.files.get('ref_file')
            ref_job_id = request.form.get('ref_job_id')
            
            if not ref_file and not ref_job_id:
                return jsonify({"error": "Either reference vector file (ref_file) or previous run ID (ref_job_id) is required.", "success": False}), 400
                
            threshold = request.form.get('threshold', 0.25, type=float)
            closing_radius = request.form.get('closing_radius', 6, type=int)
            min_object_size = request.form.get('min_object_size', 32, type=int)
            use_model = request.form.get('use_model', 'true').lower() == 'true'
            
            name = request.form.get('name', 'Meridian Route Discovery')
            study_area = request.form.get('studyArea', 'Meridian County')
            year = request.form.get('year', '2026')
            
            if ref_job_id:
                ref_geojson_path = os.path.join(app.config['OUTPUT_FOLDER'], ref_job_id, "road_network.geojson")
                if not os.path.exists(ref_geojson_path):
                    return jsonify({"error": f"Road network GeoJSON not found for job '{ref_job_id}'.", "success": False}), 400
                with open(ref_geojson_path, 'r') as rf:
                    ref_geojson_data = json.load(rf)
            else:
                filename = ref_file.filename.lower()
                if not (filename.endswith('.geojson') or filename.endswith('.json')):
                    return jsonify({"error": "Reference format must be GeoJSON.", "success": False}), 400
                ref_geojson_data = json.load(ref_file)
                
            job_id = f"discovery-{int(datetime.datetime.now().timestamp())}"
            discovery_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
            os.makedirs(discovery_dir, exist_ok=True)
            
            upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
            os.makedirs(upload_dir, exist_ok=True)
            upload_path = os.path.join(upload_dir, "input_image.png")
            file.save(upload_path)
            
            config = ExtractionConfig(
                threshold=threshold,
                closing_radius=closing_radius,
                min_object_size=min_object_size
            )
            
            weights_path = get_default_weights_path()
            weights = str(weights_path) if (weights_path.exists() and use_model) else None
            
            pipeline_result = extract_roads(
                image_path=upload_path,
                output_dir=discovery_dir,
                weights_path=weights,
                config=config
            )
            
            graphml_path = os.path.join(discovery_dir, "road_network.graphml")
            if not os.path.exists(graphml_path):
                return jsonify({"error": "Road network extraction failed to generate graph.", "success": False}), 500
                
            ref_save_path = os.path.join(discovery_dir, "reference_network.geojson")
            with open(ref_save_path, 'w') as rf:
                json.dump(ref_geojson_data, rf, indent=4)
                
        ref_paths = parse_geojson_to_paths(ref_geojson_data)
        if not ref_paths:
            return jsonify({"error": "The reference road network contains no valid LineString or MultiLineString paths.", "success": False}), 400
            
        from road_extractor.point_to_point_routes import discover_point_to_point_routes
        
        try:
            routes, start_snapped, end_snapped, info_message = discover_point_to_point_routes(
                graphml_path=graphml_path,
                start_pt=[start_x, start_y],
                end_pt=[end_x, end_y],
                ref_paths=ref_paths,
                tolerance=tolerance,
                avoidance_weight=avoidance_weight,
                max_routes=max_routes,
                allow_existing_roads=allow_existing_roads,
                snap_threshold=150.0,
                min_separation=min_separation,
                min_diversity=min_diversity
            )
        except ValueError as ve:
            return jsonify({"error": str(ve), "success": False}), 400
        
        routes_features = []
        custom_routes = []
        for r in routes:
            routes_features.append({
                "type": "Feature",
                "id": r["route_id"],
                "properties": {
                    "route_id": r["route_id"],
                    "length_meters": r["length_meters"],
                    "novel_length_meters": r["novel_length_meters"],
                    "overlap_length_meters": r["overlap_length_meters"],
                    "overlap_percentage": r["overlap_percentage"],
                    "score": r["score"]
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": r["coordinates"]
                }
            })
            
            # Formulate custom route response structure
            custom_routes.append({
                "id": r["route_id"],
                "coordinates": r["coordinates"],
                "length": r["length_meters"],
                "existing_overlap": r["overlap_percentage"] / 100.0,
                "score": r["score"],
                
                # Backwards compatibility fields
                "route_id": r["route_id"],
                "length_meters": r["length_meters"],
                "novel_length_meters": r["novel_length_meters"],
                "overlap_length_meters": r["overlap_length_meters"],
                "overlap_percentage": r["overlap_percentage"]
            })
            
        routes_geojson = {
            "type": "FeatureCollection",
            "features": routes_features
        }
        
        routes_geojson_filename = f"routes_{int(start_x)}_{int(start_y)}_to_{int(end_x)}_to_{int(end_y)}.geojson"
        routes_geojson_path = os.path.join(discovery_dir, routes_geojson_filename)
        with open(routes_geojson_path, 'w') as rf:
            json.dump(routes_geojson, rf, indent=4)
            
        return jsonify({
            "success": True,
            "projectId": job_id,
            "start_point": {
                "clicked": {"x": start_x, "y": start_y},
                "snapped": {"x": start_snapped[0], "y": start_snapped[1]}
            },
            "end_point": {
                "clicked": {"x": end_x, "y": end_y},
                "snapped": {"x": end_snapped[0], "y": end_snapped[1]}
            },
            "routes": custom_routes,
            "info_message": info_message,
            "routesGeojsonUrl": f"/static/outputs/{job_id}/{routes_geojson_filename}",
            "imageBUrl": f"/static/outputs/{job_id}/original_rgb.png",
            "referenceGeojsonUrl": f"/static/outputs/{job_id}/reference_network.geojson",
            "candidateGeojsonUrl": f"/static/outputs/{job_id}/road_network.geojson"
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500

def compute_road_classifications(
    job_id,
    arterial_width=12.0,
    collector_width=8.0,
    local_width=5.0,
    arterial_curvature=0.15,
    roughness_threshold=0.45
):
    import cv2
    import numpy as np
    import os
    import json
    import math

    job_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    summary_path = os.path.join(job_dir, "analysis_summary.json")
    geojson_path = os.path.join(job_dir, "road_network.geojson")
    mask_path = os.path.join(job_dir, "repaired_mask.png")

    if not os.path.exists(summary_path) or not os.path.exists(geojson_path) or not os.path.exists(mask_path):
        raise Exception("Required analysis run files not found for classification.")

    with open(geojson_path, 'r') as gf:
        geojson_data = json.load(gf)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise Exception("Failed to load binary road mask for classification.")

    h, w = mask.shape
    classified_features = []

    for feat in geojson_data.get("features", []):
        if feat.get("geometry", {}).get("type") != "LineString":
            classified_features.append(feat)
            continue
            
        coords = feat["geometry"]["coordinates"]
        if len(coords) < 2:
            classified_features.append(feat)
            continue

        widths = []
        for i in range(len(coords)):
            p = coords[i]
            px, py = int(p[0]), int(p[1])
            
            if i < len(coords) - 1:
                next_p = coords[i+1]
                dx = next_p[0] - p[0]
                dy = next_p[1] - p[1]
            else:
                prev_p = coords[i-1]
                dx = p[0] - prev_p[0]
                dy = p[1] - prev_p[1]

            length = math.hypot(dx, dy)
            if length == 0:
                nx, ny = 1.0, 0.0
            else:
                nx, ny = -dy / length, dx / length

            d1 = 0
            while True:
                rx = int(px + d1 * nx)
                ry = int(py + d1 * ny)
                if rx < 0 or rx >= w or ry < 0 or ry >= h or mask[ry, rx] == 0:
                    break
                d1 += 0.5
                if d1 > 50:
                    break

            d2 = 0
            while True:
                rx = int(px - d2 * nx)
                ry = int(py - d2 * ny)
                if rx < 0 or rx >= w or ry < 0 or ry >= h or mask[ry, rx] == 0:
                    break
                d2 += 0.5
                if d2 > 50:
                    break

            widths.append(d1 + d2)

        mean_width = np.mean(widths) if widths else 4.0
        width_std = np.std(widths) if widths else 0.0
        roughness = width_std / mean_width if mean_width > 0 else 0.0

        start_pt = coords[0]
        end_pt = coords[-1]
        straight_dist = math.hypot(end_pt[0] - start_pt[0], end_pt[1] - start_pt[1])
        actual_len = sum(math.hypot(coords[i+1][0] - coords[i][0], coords[i+1][1] - coords[i][1]) for i in range(len(coords)-1))
        
        sinuosity = actual_len / straight_dist if straight_dist > 0 else 1.0
        curvature = round(sinuosity - 1.0, 3)

        if mean_width >= arterial_width and curvature < arterial_curvature:
            road_class = "Primary Arterial"
        elif mean_width >= collector_width:
            road_class = "Secondary Collector"
        elif mean_width >= local_width:
            road_class = "Local Road"
        else:
            road_class = "Minor / Unpaved"

        if roughness > roughness_threshold and road_class in ["Secondary Collector", "Local Road"]:
            road_class = "Minor / Unpaved"

        base_score = 90.0
        base_score -= min(30.0, width_std * 5.0)
        base_score -= min(25.0, curvature * 50.0)
        if actual_len < 40.0:
            base_score -= 20.0

        quality_score = max(10.0, min(100.0, base_score))
        
        if quality_score >= 70.0:
            quality_tier = "High Quality"
        elif quality_score >= 40.0:
            quality_tier = "Moderate"
        else:
            quality_tier = "Requires Review"

        feat["properties"]["road_class"] = road_class
        feat["properties"]["mean_width"] = round(mean_width, 2)
        feat["properties"]["width_variation"] = round(width_std, 2)
        feat["properties"]["curvature"] = curvature
        feat["properties"]["roughness"] = round(roughness, 2)
        feat["properties"]["quality_score"] = round(quality_score, 1)
        feat["properties"]["quality_tier"] = quality_tier
        feat["properties"]["length_pixels"] = round(actual_len, 1)
        feat["properties"]["length_meters"] = round(actual_len * 0.15, 1)

        classified_features.append(feat)

    geojson_data["features"] = classified_features
    with open(geojson_path, 'w') as gf:
        json.dump(geojson_data, gf, indent=4)

    total_segments = len([f for f in classified_features if f.get("geometry", {}).get("type") == "LineString"])
    
    class_stats = {
        "Primary Arterial": {"count": 0, "length_px": 0.0},
        "Secondary Collector": {"count": 0, "length_px": 0.0},
        "Local Road": {"count": 0, "length_px": 0.0},
        "Minor / Unpaved": {"count": 0, "length_px": 0.0}
    }
    
    quality_stats = {
        "High Quality": 0,
        "Moderate": 0,
        "Requires Review": 0
    }

    for feat in classified_features:
        if feat.get("geometry", {}).get("type") != "LineString":
            continue
        props = feat["properties"]
        rc = props.get("road_class", "Local Road")
        qt = props.get("quality_tier", "Moderate")
        leng = props.get("length_pixels", 0.0)

        if rc in class_stats:
            class_stats[rc]["count"] += 1
            class_stats[rc]["length_px"] += leng
        if qt in quality_stats:
            quality_stats[qt] += 1

    distribution = {}
    for rc, stat in class_stats.items():
        leng_km = round((stat["length_px"] * 0.15) / 1000.0, 2)
        pct = round((stat["count"] / total_segments * 100.0), 1) if total_segments > 0 else 0.0
        distribution[rc] = {
            "count": stat["count"],
            "lengthKm": leng_km,
            "percentage": pct
        }

    q_breakdown = {}
    for qt, count in quality_stats.items():
        pct = round((count / total_segments * 100.0), 1) if total_segments > 0 else 0.0
        q_breakdown[qt] = {
            "count": count,
            "percentage": pct
        }

    with open(summary_path, 'r') as sf:
        summary_data = json.load(sf)

    summary_data["classificationAvailable"] = True
    summary_data["classificationStats"] = {
        "totalSegments": total_segments,
        "distribution": distribution,
        "qualityBreakdown": q_breakdown
    }

    with open(summary_path, 'w') as sf:
        json.dump(summary_data, sf, indent=4)

    return summary_data

@app.route('/api/classification', methods=['POST'])
def api_classification():
    try:
        job_id = request.form.get('job_id')
        if not job_id:
            return jsonify({"error": "Job ID is required.", "success": False}), 400
            
        arterial_width = request.form.get('arterial_width', 12.0, type=float)
        collector_width = request.form.get('collector_width', 8.0, type=float)
        local_width = request.form.get('local_width', 5.0, type=float)
        arterial_curvature = request.form.get('arterial_curvature', 0.15, type=float)
        roughness_threshold = request.form.get('roughness_threshold', 0.45, type=float)
        
        updated_summary = compute_road_classifications(
            job_id,
            arterial_width=arterial_width,
            collector_width=collector_width,
            local_width=local_width,
            arterial_curvature=arterial_curvature,
            roughness_threshold=roughness_threshold
        )
        return jsonify(updated_summary), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500

def get_nx_graph(job_id):
    import networkx as nx
    job_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    graphml_path = os.path.join(job_dir, "road_network.graphml")
    geojson_path = os.path.join(job_dir, "road_network.geojson")
    
    if os.path.exists(graphml_path):
        try:
            return nx.read_graphml(graphml_path)
        except Exception as e:
            print(f"Failed to read graphml: {e}. Falling back to GeoJSON builder.")
            
    if os.path.exists(geojson_path):
        try:
            with open(geojson_path, 'r') as gf:
                geojson_data = json.load(gf)
            graph = nx.Graph()
            for feat in geojson_data.get("features", []):
                gtype = feat.get("geometry", {}).get("type")
                if gtype == "LineString":
                    coords = feat["geometry"]["coordinates"]
                    for i in range(len(coords) - 1):
                        p1 = tuple(coords[i])
                        p2 = tuple(coords[i+1])
                        n1 = f"{p1[0]:.2f},{p1[1]:.2f}"
                        n2 = f"{p2[0]:.2f},{p2[1]:.2f}"
                        graph.add_node(n1, x=p1[0], y=p1[1])
                        graph.add_node(n2, x=p2[0], y=p2[1])
                        graph.add_edge(n1, n2)
            return graph
        except Exception as e:
            print(f"Failed to parse geojson: {e}")
            
    return nx.Graph()

def compute_graph_intelligence(job_id):
    import networkx as nx
    import os
    import json
    
    graph = get_nx_graph(job_id)
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    
    if num_nodes == 0:
        return {
            "projectId": job_id,
            "projectName": "Empty Run",
            "nodes": 0,
            "edges": 0,
            "avgDegree": "0.00",
            "maxDegree": 0,
            "density": "0.000",
            "componentsCount": 0,
            "deadEndsCount": 0,
            "junctionsCount": 0,
            "connectivityMetric": "0.0%",
            "connectivityDesc": "0.0% Gamma Index (Ratio of actual edges to max planar edges)",
            "components": [],
            "degreeDistribution": [],
            "d3Graph": {"nodes": [], "links": []}
        }
        
    components = list(nx.connected_components(graph))
    num_components = len(components)
    components_sorted = sorted(components, key=len, reverse=True)
    
    components_list = []
    for idx, comp in enumerate(components_sorted):
        comp_size = len(comp)
        pct = round((comp_size / num_nodes) * 100.0, 1)
        components_list.append({
            "id": idx,
            "label": f"Subgraph C_{idx}",
            "count": comp_size,
            "percentage": pct
        })
        
    degrees = dict(graph.degree())
    avg_degree = sum(degrees.values()) / num_nodes if num_nodes > 0 else 0.0
    max_degree = max(degrees.values()) if degrees else 0
    
    deg_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for deg in degrees.values():
        if deg == 1:
            deg_counts[1] += 1
        elif deg == 2:
            deg_counts[2] += 1
        elif deg == 3:
            deg_counts[3] += 1
        elif deg >= 4:
            deg_counts[4] += 1
            
    deg_dist = [
        {"label": "Dead End (Degree 1)", "count": deg_counts[1], "percentage": round(deg_counts[1] / num_nodes * 100.0, 1) if num_nodes > 0 else 0.0},
        {"label": "Continuation (Degree 2)", "count": deg_counts[2], "percentage": round(deg_counts[2] / num_nodes * 100.0, 1) if num_nodes > 0 else 0.0},
        {"label": "T-Junction (Degree 3)", "count": deg_counts[3], "percentage": round(deg_counts[3] / num_nodes * 100.0, 1) if num_nodes > 0 else 0.0},
        {"label": "Complex Intersection (Degree 4+)", "count": deg_counts[4], "percentage": round(deg_counts[4] / num_nodes * 100.0, 1) if num_nodes > 0 else 0.0}
    ]
    
    dead_ends = deg_counts[1]
    junctions = deg_counts[3] + deg_counts[4]
    
    if num_nodes > 2:
        gamma = num_edges / (3 * (num_nodes - 2))
        connectivity_pct = round(min(1.0, gamma) * 100.0, 1)
    else:
        connectivity_pct = 100.0
        
    connectivity_desc = f"{connectivity_pct}% Gamma Index (Ratio of actual edges to max planar edges)"
    
    node_to_idx = {}
    d3_nodes = []
    
    node_comp_map = {}
    for comp_idx, comp in enumerate(components_sorted):
        for n in comp:
            node_comp_map[n] = comp_idx
            
    for idx, node in enumerate(graph.nodes()):
        node_to_idx[node] = idx
        node_attrs = graph.nodes[node]
        nx_val = node_attrs.get('x', 0.0)
        ny_val = node_attrs.get('y', 0.0)
        
        deg = degrees.get(node, 2)
        d3_nodes.append({
            "id": idx,
            "name": str(node),
            "degree": deg,
            "group": node_comp_map.get(node, 0),
            "x": float(nx_val),
            "y": float(ny_val)
        })
        
    d3_links = []
    for u, v in graph.edges():
        d3_links.append({
            "source": node_to_idx[u],
            "target": node_to_idx[v]
        })
        
    summary_path = os.path.join(app.config['OUTPUT_FOLDER'], job_id, "analysis_summary.json")
    project_name = "Analysis Project"
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as sf:
                sdata = json.load(sf)
            project_name = sdata.get("projectName", project_name)
        except:
            pass

    return {
        "projectId": job_id,
        "projectName": project_name,
        "nodes": num_nodes,
        "edges": num_edges,
        "avgDegree": f"{avg_degree:.2f}",
        "maxDegree": max_degree,
        "density": f"{nx.density(graph):.4f}",
        "componentsCount": num_components,
        "deadEndsCount": dead_ends,
        "junctionsCount": junctions,
        "connectivityMetric": f"{connectivity_pct}%",
        "connectivityDesc": connectivity_desc,
        "components": components_list,
        "degreeDistribution": deg_dist,
        "d3Graph": {
            "nodes": d3_nodes,
            "links": d3_links
        }
    }

def generate_pdf_report(job_id):
    import os
    import json
    from pathlib import Path
    
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    job_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    summary_path = os.path.join(job_dir, "analysis_summary.json")
    pdf_path = os.path.join(job_dir, "analysis_report.pdf")

    if not os.path.exists(summary_path):
        raise Exception("Analysis summary not found.")

    with open(summary_path, 'r') as sf:
        summary = json.load(sf)

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0ea5e9'),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=15
    )

    h2_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    cell_style = ParagraphStyle(
        'CellText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#334155')
    )

    cell_bold_style = ParagraphStyle(
        'CellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#0f172a')
    )

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    story = []

    story.append(Paragraph("ROAD NETWORK EXTRACTION EXECUTIVE REPORT", title_style))
    story.append(Paragraph("AI-BASED CENTERLINE EXTRACTION & GEOSPATIAL TELEMETRY REPORT", subtitle_style))
    story.append(Spacer(1, 5))

    metadata_data = [
        [Paragraph("Project Name:", cell_bold_style), Paragraph(str(summary.get("projectName", "N/A")), cell_style),
         Paragraph("Analysis Date:", cell_bold_style), Paragraph(str(summary.get("analysisDate", "N/A")), cell_style)],
        [Paragraph("Location / Area:", cell_bold_style), Paragraph(str(summary.get("location", "N/A")), cell_style),
         Paragraph("Image Year:", cell_bold_style), Paragraph(str(summary.get("imageYear", "N/A")), cell_style)],
        [Paragraph("Source File:", cell_bold_style), Paragraph(str(summary.get("fileName", "N/A")), cell_style),
         Paragraph("File Size:", cell_bold_style), Paragraph(str(summary.get("fileSize", "N/A")), cell_style)],
        [Paragraph("Georeferenced CRS:", cell_bold_style), Paragraph(str(summary.get("crs", "Image Pixel Space (Non-Georeferenced)")), cell_style),
         Paragraph("Project Status:", cell_bold_style), Paragraph("COMPLETE", cell_bold_style)]
    ]
    
    t_meta = Table(metadata_data, colWidths=[90, 180, 90, 180])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 15))

    story.append(Paragraph("Network Topology Summary", h2_style))
    net = summary.get("networkSummary", {})
    topo = summary.get("topology", {})
    
    stats_data = [
        [Paragraph("Metric", cell_bold_style), Paragraph("Value", cell_bold_style), Paragraph("Metric Description", cell_bold_style)],
        [Paragraph("Total Centerline Length", cell_style), Paragraph(f"{net.get('totalRoadLength', {}).get('value', 0)} km", cell_bold_style), Paragraph("Total geodetic length of extracted centerlines.", cell_style)],
        [Paragraph("Road Segments count", cell_style), Paragraph(str(net.get('roadSegments', {}).get('value', 0)), cell_style), Paragraph("Centerline segments between junctions or end points.", cell_style)],
        [Paragraph("Junction Intersection nodes", cell_style), Paragraph(str(topo.get('intersections', 0)), cell_style), Paragraph("Coordinate nodes where three or more road segments cross.", cell_style)],
        [Paragraph("Dead End nodes", cell_style), Paragraph(str(topo.get('deadEnds', 0)), cell_style), Paragraph("Degrees 1 endpoints failing to connect loop junctions.", cell_style)],
        [Paragraph("Connected Component subgraphs", cell_style), Paragraph(str(topo.get('connectedComponents', 1)), cell_style), Paragraph("Isolated subnetworks in geodetic model.", cell_style)],
        [Paragraph("Overall Connectivity index", cell_style), Paragraph(str(net.get('connectivity', 'N/A')), cell_style), Paragraph("Graph completeness level metrics.", cell_style)],
        [Paragraph("Average Extraction Confidence", cell_style), Paragraph(str(net.get('overallConfidence', 'N/A')), cell_style), Paragraph("Geometric and topological pixel confidence score.", cell_style)]
    ]
    
    t_stats = Table(stats_data, colWidths=[150, 90, 300])
    t_stats.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_stats)
    story.append(Spacer(1, 15))

    story.append(PageBreak())

    story.append(Paragraph("Geospatial Visual Overlays", h2_style))
    story.append(Spacer(1, 5))

    orig_img_path = os.path.join(job_dir, "original_rgb.png")
    overlay_img_path = os.path.join(job_dir, "overlay.png")
    skeleton_img_path = os.path.join(job_dir, "geometry_skeleton.png")
    graph_img_path = os.path.join(job_dir, "road_graph.png")

    img_table_data = []

    def make_pdf_img(p):
        if os.path.exists(p):
            return Image(p, width=240, height=240)
        return Paragraph("[Visual Asset Missing]", cell_style)

    img_table_data.append([Paragraph("<b>Original Imagery View</b>", cell_bold_style), Paragraph("<b>Road Overlay Mask</b>", cell_bold_style)])
    img_table_data.append([make_pdf_img(orig_img_path), make_pdf_img(overlay_img_path)])
    img_table_data.append([Spacer(1, 5), Spacer(1, 5)])
    img_table_data.append([Paragraph("<b>Clean centerline skeleton</b>", cell_bold_style), Paragraph("<b>Extracted Topology Graph</b>", cell_bold_style)])
    img_table_data.append([make_pdf_img(skeleton_img_path), make_pdf_img(graph_img_path)])

    t_imgs = Table(img_table_data, colWidths=[270, 270])
    t_imgs.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_imgs)

    if summary.get("classificationAvailable"):
        story.append(PageBreak())
        story.append(Paragraph("Road Functional Classification Details", h2_style))
        story.append(Spacer(1, 5))
        
        class_stats = summary.get("classificationStats", {})
        dist = class_stats.get("distribution", {})
        
        class_table_data = [
            [Paragraph("Functional Class", cell_bold_style), Paragraph("Segments Count", cell_bold_style), Paragraph("Total Length (km)", cell_bold_style), Paragraph("Distribution Pct (%)", cell_bold_style)]
        ]
        
        for rclass, sdata in dist.items():
            class_table_data.append([
                Paragraph(rclass, cell_style),
                Paragraph(str(sdata.get("count", 0)), cell_style),
                Paragraph(f"{sdata.get('lengthKm', 0.0)} km", cell_style),
                Paragraph(f"{sdata.get('percentage', 0.0)}%", cell_style)
            ])
            
        t_class = Table(class_table_data, colWidths=[150, 120, 130, 140])
        t_class.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('PADDING', (0,0), (-1,-1), 5),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(t_class)
        story.append(Spacer(1, 15))

        story.append(Paragraph("Quality Tiers Breakdown", h2_style))
        story.append(Spacer(1, 5))
        qbreak = class_stats.get("qualityBreakdown", {})
        
        q_table_data = [
            [Paragraph("Quality Tier", cell_bold_style), Paragraph("Segments Count", cell_bold_style), Paragraph("Percentage (%)", cell_bold_style)]
        ]
        for tier, qdata in qbreak.items():
            q_table_data.append([
                Paragraph(tier, cell_style),
                Paragraph(str(qdata.get("count", 0)), cell_style),
                Paragraph(f"{qdata.get('percentage', 0.0)}%", cell_style)
            ])
            
        t_q = Table(q_table_data, colWidths=[180, 180, 180])
        t_q.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f8fafc')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('PADDING', (0,0), (-1,-1), 5),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(t_q)

    doc.build(story)

@app.route('/api/analysis/<job_id>/report', methods=['POST'])
def api_generate_report(job_id):
    try:
        import re
        if not re.match(r'^[a-zA-Z0-9_\-]+$', job_id):
            return jsonify({"error": "Invalid Job ID format.", "success": False}), 400
            
        generate_pdf_report(job_id)
        
        pdf_url = f"/static/outputs/{job_id}/analysis_report.pdf"
        return jsonify({"success": True, "reportUrl": pdf_url}), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/api/intelligence/<job_id>', methods=['GET'])
def api_intelligence(job_id):
    try:
        import re
        if not re.match(r'^[a-zA-Z0-9_\-]+$', job_id):
            return jsonify({"error": "Invalid Job ID format.", "success": False}), 400
            
        data = compute_graph_intelligence(job_id)
        return jsonify(data), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/api/analysis/<job_id>/progress', methods=['GET'])
def api_get_progress(job_id):
    try:
        import re
        if not re.match(r'^[a-zA-Z0-9_\-]+$', job_id):
            return jsonify({"error": "Invalid Job ID format.", "success": False}), 400
            
        progress_path = os.path.join(app.config['OUTPUT_FOLDER'], job_id, "progress.txt")
        if os.path.exists(progress_path):
            with open(progress_path, 'r') as pf:
                progress_str = pf.read().strip()
            return jsonify({"success": True, "progress": progress_str}), 200
        else:
            return jsonify({"success": True, "progress": "Initializing..."}), 200
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.errorhandler(404)
def not_found(e):
    """Fallback route to support React Router client-side routing."""
    if request.path.startswith('/api/'):
        return jsonify({"error": f"API route '{request.path}' not found", "success": False}), 404
    if os.path.exists(os.path.join(app.static_folder, 'index.html')):
        return send_from_directory(app.static_folder, 'index.html')
    return "Page not found", 404

if __name__ == '__main__':
    # Runs locally on http://127.0.0.1:5000 (Unified single host for API + React SPA)
    app.run(debug=True, use_reloader=False, threaded=False, host='0.0.0.0', port=5000)
