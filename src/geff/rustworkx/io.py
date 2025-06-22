from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import rustworkx as rx
import zarr

import geff
import geff.utils
from geff.metadata_schema import GeffMetadata

if TYPE_CHECKING:
    from pathlib import Path


def get_roi(
    graph: rx.PyGraph | rx.PyDiGraph, position_attr: str
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Get the roi of a rustworkx graph.

    Args:
        graph (rx.PyGraph | rx.PyDiGraph): A non-empty rustworkx graph
        position_attr (str): All nodes on graph have this attribute holding their position

    Returns:
        tuple[tuple[float, ...], tuple[float, ...]]: A tuple with the min values in each
            spatial dim, and a tuple with the max values in each spatial dim
    """
    attr = np.asarray([data[position_attr] for data in graph.nodes()])
    min_attr = attr.min(axis=0)
    max_attr = attr.max(axis=0)
    return tuple(min_attr.tolist()), tuple(max_attr.tolist())


def get_node_attrs(graph: rx.PyGraph | rx.PyDiGraph) -> list[str]:
    """Get the attribute keys present on any node in the rustworkx graph. Does not imply
    that the attributes are present on all nodes.

    Args:
        graph (rx.PyGraph | rx.PyDiGraph): a rustworkx graph

    Returns:
        list[str]: A list of all unique node attribute keys
    """
    attr_keys = set()
    for node_data in graph.nodes():
        attr_keys.update(node_data.keys())
    return list(attr_keys)


def get_edge_attrs(graph: rx.PyGraph | rx.PyDiGraph) -> list[str]:
    """Get the attribute keys present on any edge in the rustworkx graph. Does not imply
    that the attributes are present on all edges.

    Args:
        graph (rx.PyGraph | rx.PyDiGraph): a rustworkx graph

    Returns:
        list[str]: A list of all unique edge attribute keys
    """
    attr_keys = set()
    for edge_data in graph.edges():
        attr_keys.update(edge_data.keys())
    return list(attr_keys)


def write(
    graph: rx.PyGraph | rx.PyDiGraph,
    position_attr: str,
    path: str | Path,
    axis_names: list[str] | None = None,
    axis_units: list[str] | None = None,
    zarr_format: int = 3,
    validate: bool = True,
):
    """Write a rustworkx graph to the geff file format

    Args:
        graph (rx.PyGraph | rx.PyDiGraph): a rustworkx graph where every node has a
            position attribute
        position_attr (str): the name of the position attribute present on every node
        path (str | Path): the path to the output zarr. Opens in append mode,
            so will only overwrite geff-controlled groups.
        axis_names (Optional[list[str]], optional): The names of the spatial dims
            represented in position attribute. Defaults to None. Will override
            value in graph attributes if provided.
        axis_units (Optional[list[str]], optional): The units of the spatial dims
            represented in position attribute. Defaults to None. Will override value
            in graph attributes if provided.
        zarr_format (Optional[int], optional): The version of zarr to write.
            Defaults to 3.
        validate (bool, optional): Flag indicating whether to perform validation on the
            rustworkx graph before writing anything to disk. If set to False and there are
            missing attributes, will likely fail with a KeyError, leading to an incomplete
            graph written to disk. Defaults to True.
    """
    if graph.num_nodes() == 0:
        warnings.warn(f"Graph is empty - not writing anything to {path}", stacklevel=2)
        return

    # open/create zarr container
    group: zarr.Group
    if zarr.__version__.startswith("3"):
        group = zarr.open(path, mode="a", zarr_format=zarr_format)
    else:
        group = zarr.open(path, mode="a")

    node_attrs = get_node_attrs(graph)
    if validate:
        if position_attr not in node_attrs:
            raise ValueError(f"Position attribute {position_attr} not found in graph")
        for node_data in graph.nodes():
            if position_attr not in node_data:
                raise ValueError(
                    f"Node {node_data} does not have position attribute {position_attr}"
                )

    if graph.attrs is None:
        graph.attrs = {}

    # write metadata
    roi_min, roi_max = get_roi(graph, position_attr=position_attr)
    metadata = GeffMetadata(
        geff_version=geff.__version__,
        directed=isinstance(graph, rx.PyDiGraph),
        roi_min=roi_min,
        roi_max=roi_max,
        position_attr=position_attr,
        axis_names=tuple(axis_names)
        if axis_names is not None
        else graph.attrs.get("axis_names", None),
        axis_units=tuple(axis_units)
        if axis_units is not None
        else graph.attrs.get("axis_units", None),
    )
    metadata.write(group)

    # get node and edge IDs
    nodes_arr = np.asarray(graph.node_indices())
    edges_arr = np.asarray(graph.edge_list())

    # write nodes
    group["nodes/ids"] = nodes_arr

    # write node attributes
    for name in node_attrs:
        values = []
        missing = []
        for node_idx in nodes_arr:
            data = graph.get_node_data(node_idx)
            if name in data:
                value = data[name]
                mask = False
            else:
                value = 0
                mask = True
            values.append(value)
            missing.append(mask)
        # Set position attribute to default "position", original stored in metadata
        if name == position_attr:
            name = "position"
        else:
            # Always store missing array even if all values are present
            group[f"nodes/attrs/{name}/missing"] = np.asarray(missing, dtype=bool)
        group[f"nodes/attrs/{name}/values"] = np.asarray(values)

    # write edges
    # Edge group is only created if edges are present on graph
    if len(edges_arr) > 0:
        group["edges/ids"] = edges_arr

        # write edge attributes
        for name in get_edge_attrs(graph):
            values = []
            missing = []
            for edge in edges_arr:
                data = graph.get_edge_data(*edge)
                if name in data:
                    value = data[name]
                    mask = False
                else:
                    value = 0
                    mask = True
                values.append(value)
                missing.append(mask)
            group[f"edges/attrs/{name}/missing"] = np.asarray(missing, dtype=bool)
            group[f"edges/attrs/{name}/values"] = np.asarray(values)


def _set_attribute_values(
    graph: rx.PyGraph | rx.PyDiGraph,
    ids: np.ndarray,
    graph_group: zarr.Group,
    name: str,
    nodes: bool = True,
) -> None:
    """Add attributes in-place to a rustworkx graph's nodes or edges.

    Args:
        graph (rx.PyGraph | rx.PyDiGraph): The rustworkx graph, already populated with nodes
            or edges, that needs attributes added
        ids (np.ndarray): Node or edge ids from Geff. If nodes, 1D. If edges, 2D.
        graph_group (zarr.Group): A zarr group holding the geff graph.
        name (str): The name of the attribute
        nodes (bool, optional): If True, extract and set node attributes.  If False,
            extract and set edge attributes. Defaults to True.
    """
    element = "nodes" if nodes else "edges"
    attr_group: zarr.Group = graph_group[f"{element}/attrs/{name}"]
    values = attr_group["values"][:]
    sparse = "missing" in attr_group.array_keys()
    if sparse:
        missing = attr_group["missing"][:]
    for idx in range(len(ids)):
        _id = ids[idx]
        val = values[idx]
        # If attribute is sparse and missing for this node, skip setting attribute
        ignore = missing[idx] if sparse else False
        if not ignore:
            # Get either individual item or list instead of setting with np.array
            val = val.tolist() if val.size > 1 else val.item()
            if nodes:
                if name == "position" and "position_attr" in graph.attrs:
                    graph_attr = graph.attrs["position_attr"]
                else:
                    graph_attr = name
                # Update node data in rustworkx
                current_data = graph.get_node_data(_id)
                current_data[graph_attr] = val
            else:
                # Update edge data in rustworkx
                current_data = graph.get_edge_data(*_id)
                current_data[name] = val


def read(path: Path | str, validate: bool = True) -> rx.PyGraph | rx.PyDiGraph:
    """Read a geff file into a rustworkx graph. Metadata attributes will be stored in
    the graph attributes, accessed via `G.attrs[key]` where G is a rustworkx graph.

    Args:
        path (Path | str): The path to the root of the geff zarr, where the .attrs contains
            the geff  metadata
    Returns:
        rx.PyGraph | rx.PyDiGraph: The graph that was stored in the geff file format
    """
    # zarr python 3 doesn't support Path
    path = str(path)

    # open zarr container
    if validate:
        geff.utils.validate(path)

    group: zarr.Group = zarr.open(path, mode="r")
    metadata = GeffMetadata.read(group)

    # read meta-data
    graph = rx.PyDiGraph() if metadata.directed else rx.PyGraph()
    graph.attrs = {}

    for key, val in metadata:
        graph.attrs[key] = val

    nodes = group["nodes/ids"][:]
    # Add nodes to rustworkx graph
    for _ in nodes.tolist():
        graph.add_node({})

    # collect node attributes
    for name in group["nodes/attrs"]:
        _set_attribute_values(graph, nodes, group, name, nodes=True)

    if "edges" in group.group_keys():
        edges = group["edges/ids"][:]
        # Add edges to rustworkx graph
        for source, target in edges.tolist():
            graph.add_edge(source, target, {})

        # collect edge attributes if they exist
        if "edges/attrs" in group:
            for name in group["edges/attrs"]:
                _set_attribute_values(graph, edges, group, name, nodes=False)

    return graph
