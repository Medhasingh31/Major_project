import pytest
import os
import tempfile
import json
import networkx as nx
from road_extractor.point_to_point_routes import discover_point_to_point_routes

@pytest.fixture
def temp_graphml():
    G = nx.MultiDiGraph()
    # Add a main road component: node 1 to node 2
    G.add_node("1", x=100.0, y=100.0)
    G.add_node("2", x=200.0, y=100.0)
    G.add_edge(
        "1", "2",
        pixels=json.dumps([[100.0, 100.0], [200.0, 100.0]]),
        length_pixels=100.0,
        segment_id=1
    )
    
    # Add an isolated road component: node 3 to node 4
    G.add_node("3", x=100.0, y=300.0)
    G.add_node("4", x=200.0, y=300.0)
    G.add_edge(
        "3", "4",
        pixels=json.dumps([[100.0, 300.0], [200.0, 300.0]]),
        length_pixels=100.0,
        segment_id=2
    )
    
    fd, path = tempfile.mkstemp(suffix=".graphml")
    os.close(fd)
    nx.write_graphml(G, path)
    
    yield path
    
    if os.path.exists(path):
        os.remove(path)

def test_case1_identical_points(temp_graphml):
    # Case 1: Point A and Point B are the same
    with pytest.raises(ValueError, match="Start and destination must be different."):
        discover_point_to_point_routes(
            temp_graphml,
            start_pt=[100.0, 100.0],
            end_pt=[100.0, 100.0],
            ref_paths=[]
        )

def test_case2_start_unsnappable(temp_graphml):
    # Case 2: Start point cannot be snapped to the road graph (e.g. far away)
    with pytest.raises(ValueError, match="No usable road network found near the selected start point."):
        discover_point_to_point_routes(
            temp_graphml,
            start_pt=[500.0, 500.0],
            end_pt=[200.0, 100.0],
            ref_paths=[],
            snap_threshold=50.0
        )

def test_case3_end_unsnappable(temp_graphml):
    # Case 3: End point cannot be snapped
    with pytest.raises(ValueError, match="No usable road network found near the selected destination point."):
        discover_point_to_point_routes(
            temp_graphml,
            start_pt=[100.0, 100.0],
            end_pt=[500.0, 500.0],
            ref_paths=[],
            snap_threshold=50.0
        )

def test_case4_no_path(temp_graphml):
    # Case 4: No path exists between components
    with pytest.raises(ValueError, match="No connected route was found between the selected points."):
        discover_point_to_point_routes(
            temp_graphml,
            start_pt=[100.0, 100.0],
            end_pt=[200.0, 300.0],
            ref_paths=[]
        )

def test_case5_excessive_overlap(temp_graphml):
    # Case 5: Paths exist but all substantially overlap the existing network.
    # We pass a reference path that perfectly matches the discovered path
    ref_paths = [[[100.0, 100.0], [200.0, 100.0]]]
    with pytest.raises(ValueError, match="No sufficiently new route alternatives were found."):
        discover_point_to_point_routes(
            temp_graphml,
            start_pt=[100.0, 100.0],
            end_pt=[200.0, 100.0],
            ref_paths=ref_paths,
            tolerance=15.0
        )

def test_case6_fewer_alternatives(temp_graphml):
    # Case 6: Only one meaningful route exists, return the route and explain via info message
    routes, start_snapped, end_snapped, info_message = discover_point_to_point_routes(
        temp_graphml,
        start_pt=[100.0, 100.0],
        end_pt=[200.0, 100.0],
        ref_paths=[],
        max_routes=4
    )
    assert len(routes) == 1
    assert info_message == "Fewer route alternatives were available."
