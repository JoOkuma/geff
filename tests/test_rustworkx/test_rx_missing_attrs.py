from pathlib import Path

import numpy as np
import pytest
import rustworkx as rx
import zarr

from geff.rustworkx import read, write
from geff.utils import validate


def graph_sparse_node_attrs():
    graph = rx.PyGraph()
    nodes = [1, 2, 3, 4, 5]
    positions = [
        [0, 1, 2],
        [0, 0, 0],
        [1, 1, 3],
        [1, 5, 2],
        [1, 7, 6],
    ]
    node_scores = [0.5, 0.2, None, None, 0.1]

    # Add nodes with data
    for _, pos, score in zip(nodes, positions, node_scores):
        node_data = {"position": pos}
        if score is not None:
            node_data["score"] = score
        graph.add_node(node_data)

    return graph, positions


def graph_sparse_edge_attrs():
    graph, _ = graph_sparse_node_attrs()
    edges = [
        [1, 3],
        [1, 4],
        [2, 5],
    ]
    edge_scores = [0.1, None, 0.5]

    # Add edges with data
    for edge, score in zip(edges, edge_scores):
        edge_data = {}
        if score is not None:
            edge_data["score"] = score
        graph.add_edge(edge[0] - 1, edge[1] - 1, edge_data)

    return graph


def test_sparse_node_attrs(tmp_path):
    zarr_path = Path(tmp_path) / "test.zarr"
    graph, positions = graph_sparse_node_attrs()
    write(graph, position_attr="position", path=zarr_path)
    # check that the written thing is valid
    assert Path(zarr_path).exists()
    validate(zarr_path)

    zroot = zarr.open(zarr_path, mode="r")
    node_attrs = zroot["nodes"]["attrs"]
    pos = node_attrs["position"]["values"][:]
    np.testing.assert_array_almost_equal(np.array(positions), pos)
    scores = node_attrs["score"]["values"][:]
    assert scores[0] == 0.5
    assert scores[1] == 0.2
    assert scores[4] == 0.1
    score_mask = node_attrs["score"]["missing"][:]
    np.testing.assert_array_almost_equal(score_mask, np.array([False, False, True, True, False]))

    # read it back in and check for consistency
    read_graph = read(zarr_path)
    for i, node_data in enumerate(graph.nodes()):
        read_node_data = read_graph.get_node_data(i)
        assert read_node_data == node_data


def test_sparse_edge_attrs(tmp_path):
    zarr_path = Path(tmp_path) / "test.zarr"
    graph = graph_sparse_edge_attrs()
    write(graph, position_attr="position", path=zarr_path)
    # check that the written thing is valid
    assert Path(zarr_path).exists()
    validate(zarr_path)

    zroot = zarr.open(zarr_path, mode="r")
    edge_attrs = zroot["edges"]["attrs"]
    scores = edge_attrs["score"]["values"][:]
    assert scores[0] == 0.1
    assert scores[2] == 0.5

    score_mask = edge_attrs["score"]["missing"][:]
    np.testing.assert_array_almost_equal(score_mask, np.array([False, True, False]))

    # read it back in and check for consistency
    read_graph = read(zarr_path)
    for i, edge_data in enumerate(graph.edges()):
        edge = graph.edge_list()[i]
        read_edge_data = read_graph.get_edge_data(*edge)
        assert read_edge_data == edge_data


def test_missing_pos_attr(tmp_path):
    zarr_path = Path(tmp_path) / "test.zarr"
    graph, _ = graph_sparse_node_attrs()
    # wrong attribute name
    with pytest.raises(ValueError, match="Position attribute pos not found in graph"):
        write(graph, position_attr="pos", path=zarr_path)
    # missing attribute
    # Remove position from first node
    node_data = graph.get_node_data(0)
    del node_data["position"]
    with pytest.raises(ValueError, match="Node .* does not have position attribute position"):
        write(graph, position_attr="position", path=zarr_path)
