import numpy as np
import pytest
import rustworkx as rx

from geff.rustworkx import read, write

node_dtypes = ["int8", "uint8", "int16", "uint16"]
node_attr_dtypes = [
    {"position": "double"},
    {"position": "int"},
]
edge_attr_dtypes = [
    {"score": "float64", "color": "uint8"},
    {"score": "float32", "color": "int16"},
]

# TODO: mixed dtypes?


@pytest.mark.parametrize("node_dtype", node_dtypes)
@pytest.mark.parametrize("node_attr_dtypes", node_attr_dtypes)
@pytest.mark.parametrize("edge_attr_dtypes", edge_attr_dtypes)
@pytest.mark.parametrize("directed", [True, False])
def test_read_write_consistency(tmp_path, node_dtype, node_attr_dtypes, edge_attr_dtypes, directed):
    axis_names = ["t", "z", "y", "x"]
    axis_units = ["s", "nm", "nm", "nm"]
    graph = rx.PyDiGraph() if directed else rx.PyGraph()

    nodes = np.array([10, 2, 127, 4, 5], dtype=node_dtype)
    pytest.skip("TODO: Rustworkx does not support arbitrary node attributes")
    positions = np.array(
        [
            [0.1, 0.5, 100.0, 1.0],
            [0.2, 0.4, 200.0, 0.1],
            [0.3, 0.3, 300.0, 0.1],
            [0.4, 0.2, 400.0, 0.1],
            [0.5, 0.1, 500.0, 0.1],
        ],
        dtype=node_attr_dtypes["position"],
    )
    for _, pos in zip(nodes, positions):
        graph.add_node({"pos": pos.tolist()})

    edges = np.array(
        [
            [10, 2],
            [2, 127],
            [2, 4],
            [4, 5],
        ],
        dtype=node_dtype,
    )
    scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=edge_attr_dtypes["score"])
    colors = np.array([1, 2, 3, 4], dtype=edge_attr_dtypes["color"])
    for edge, score, color in zip(edges, scores, colors):
        source, target = edge.tolist()
        graph.add_edge(source, target, {"score": score.item(), "color": color.item()})

    path = tmp_path / "rw_consistency.zarr/graph"

    write(graph, "pos", path, axis_names=axis_names, axis_units=axis_units)

    compare = read(path)

    # Check nodes
    assert set(graph.node_indices()) == set(compare.node_indices())
    # Check edges
    assert set(graph.edge_list()) == set(compare.edge_list())

    # Check node attributes
    for node in nodes:
        node_idx = list(graph.node_indices()).index(node.item())
        compare_node_idx = list(compare.node_indices()).index(node.item())
        assert (
            graph.get_node_data(node_idx)["pos"] == compare.get_node_data(compare_node_idx)["pos"]
        )

    # Check edge attributes
    for edge in edges:
        edge_tuple = tuple(edge.tolist())
        assert (
            graph.get_edge_data(*edge_tuple)["score"] == compare.get_edge_data(*edge_tuple)["score"]
        )
        assert (
            graph.get_edge_data(*edge_tuple)["color"] == compare.get_edge_data(*edge_tuple)["color"]
        )

    assert compare.attrs["axis_names"] == tuple(axis_names)
    assert compare.attrs["axis_units"] == tuple(axis_units)


def test_write_empty_graph():
    graph = rx.PyDiGraph()
    with pytest.warns(match="Graph is empty - not writing anything "):
        write(graph, position_attr="pos", path=".")
