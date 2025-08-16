"""
This module provides utility functions to generate graphs.
"""

import itertools
import warnings
import random as rd

from typing import Any, Callable, Literal, Optional

from rdgc.graph.base import Graph


def null(
    size: int,
    *args: Any,
    directed: bool = False,
    **kwargs: Any,
) -> Graph:
    """
    Returns a null graph with `size` vertices.

    Args:
        size (int): The number of vertices.
        directed (bool, optional): Specifies whether the graph is directed. Defaults to False.

    Warnings:
        RuntimeWarning: If extra arguments are provided.

    Returns:
        Graph: A null graph with `size` vertices.
    """
    if args or kwargs:
        warnings.warn("Extra arguments are ignored", RuntimeWarning, 2)
    return Graph(size, directed)


def complete(
    size: int,
    *args: Any,
    directed: bool = False,
    self_loop: bool = False,
    weight_gener: Optional[Callable[[int, int], Any]] = None,
    **kwargs: Any,
) -> Graph:
    """
    Returns a complete graph with `size` vertices.

    Args:
        size (int): The number of vertices.
        directed (bool, optional): Specifies whether the graph is directed. Defaults to False.
        self_loop (bool, optional): Specifies whether self-loops are allowed. Defaults to False.
        weight_gener (Callable[[int, int], Any], optional): A function to generate edge weights. Defaults to None.

    Warnings:
        RuntimeWarning: If extra arguments are provided.

    Returns:
        Graph: A complete graph with `size` vertices.
    """
    if args or kwargs:
        warnings.warn("Extra arguments are ignored", RuntimeWarning, 2)
    if weight_gener is None:
        weight_gener = lambda u, v: None

    graph = Graph(size, directed)
    for u in range(size):
        for v in range(size) if directed else range(u, size):
            if u != v or self_loop:
                graph.add_edge(u, v, weight_gener(u, v))
    return graph


def tournament(
    size: int,
    *args: Any,
    weight_gener: Optional[Callable[[int, int], Any]] = None,
    **kwargs: Any,
) -> Graph:
    """
    Returns a tournament graph with `size` vertices.

    Args:
        size (int): The number of vertices.
        weight_gener (Callable[[int, int], Any], optional): A function to generate edge weights. Defaults to None.

    Warnings:
        RuntimeWarning: If extra arguments are provided.

    Returns:
        Graph: A tournament graph with `size` vertices.
    """
    if args or kwargs:
        warnings.warn("Extra arguments are ignored", RuntimeWarning, 2)
    if weight_gener is None:
        weight_gener = lambda u, v: None

    graph = Graph(size, True)
    for u in range(size):
        for v in range(u + 1, size):
            i, j = rd.sample((u, v), 2)
            graph.add_edge(i, j, weight_gener(i, j))
    return graph


def random_graph(
    size: int,
    edge_count: int,
    *args: Any,
    directed: bool = False,
    self_loop: bool = False,
    multiedge: bool = False,
    weight_gener: Optional[Callable[[int, int], Any]] = None,
    **kwargs: Any,
) -> Graph:
    """
    Returns a random graph with `size` vertices and `edge_count` edges.

    Args:
        size (int): The number of vertices.
        edge_count (int): The number of edges.
        directed (bool, optional): Specifies whether the graph is directed. Defaults to False.
        self_loop (bool, optional): Specifies whether self-loops are allowed. Defaults to False.
        multiedge (bool, optional): Specifies whether multiple edges are allowed. Defaults to False.
        weight_gener (Callable[[int, int], Any], optional): A function to generate edge weights. Defaults to None.

    Raises:
        ValueError: If the number of edges is invalid.

    Warnings:
        RuntimeWarning: If extra arguments are provided

    Returns:
        Graph: A random graph with `size` vertices and `edge_count` edges.
    """
    if args or kwargs:
        warnings.warn("Extra arguments are ignored", RuntimeWarning, 2)
    if weight_gener is None:
        weight_gener = lambda u, v: None

    if not multiedge:
        max_edge = Graph.calc_max_edge(size, directed, self_loop)
        if edge_count > max_edge:
            raise ValueError(f"Too many edges: {edge_count} > {max_edge}")

    graph = Graph(size, directed)
    while graph.edges < edge_count:
        u, v = (
            rd.sample(range(size), 2) if not self_loop else rd.choices(range(size), k=2)
        )
        if multiedge or graph.count_edge(u, v) == 0:
            graph.add_edge(u, v, weight_gener(u, v))
    return graph


def cycle(
    size: int,
    *args: Any,
    directed: bool = False,
    weight_gener: Optional[Callable[[int, int], Any]] = None,
    **kwargs: Any,
) -> "Graph":
    """
    Returns a cycle graph with `size` vertices.

    Args:
        size (int): The number of vertices.
        directed (bool, optional): Specifies whether the graph is directed. Defaults to False.
        weight_gener (Callable[[int, int], Any], optional): A function to generate edge weights. Defaults to None.

    Warnings:
        RuntimeWarning: If extra arguments are provided.

    Returns:
        Graph: A cycle graph with `size` vertices.
    """
    if args or kwargs:
        warnings.warn("Extra arguments are ignored", RuntimeWarning, 2)
    if weight_gener is None:
        weight_gener = lambda u, v: None

    graph = Graph(size, directed)
    for u in range(size):
        graph.add_edge(u, (u + 1) % size, weight_gener(u, (u + 1) % size))
    return graph


def wheel(
    size: int,
    *args: Any,
    directed: bool = False,
    weight_gener: Optional[Callable[[int, int], Any]] = None,
    **kwargs: Any,
) -> "Graph":
    """
    Returns a wheel graph with `size` vertices.

    Args:
        size (int): The number of vertices.
        directed (bool, optional): Specifies whether the graph is directed. Defaults to False.
        weight_gener (Callable[[int, int], Any], optional): A function to generate edge weights. Defaults to None.

    Warnings:
        RuntimeWarning: If extra arguments are provided.

    Returns:
        Graph: A wheel graph with `size` vertices.
    """
    if args or kwargs:
        warnings.warn("Extra arguments are ignored", RuntimeWarning, 2)
    if weight_gener is None:
        weight_gener = lambda u, v: None

    graph = Graph(size, directed)
    for u in range(1, size):
        graph.add_edge(0, u, weight_gener(0, u))
    for u in range(1, size):
        graph.add_edge(u, u % (size - 1) + 1, weight_gener(u, u % (size - 1) + 1))
    return graph


def union(
    *graphs: Graph,
    directed: bool = False,
    self_loop: bool = False,
    multiedge: bool = False,
    node_mapping: Optional[Callable[[int, int, tuple[Graph, ...]], int]] = None,
    default_mapping: Literal["separate", "combine", "connect"] = "separate",
) -> Graph:
    """
    Returns the union of multiple graphs.

    This function combines multiple graphs into a single graph, allowing for
    custom node mapping to handle potential conflicts in node IDs.

    The rank of nodes is preserved from the original graphs, only the last one which is not None is used.

    If `multiedge` is False, it ensures that no multiple edges are created between the same pair of nodes.
    If there are multiple edges, only the first one is kept.

    If `node_mapping` is not provided, it defaults to mapping nodes from different graphs
    to unique IDs by adding the cumulative number of vertices from previous graphs.

    Args:
        *graphs (Graph): The graphs to be unioned.
        directed (bool, optional): Specifies whether the resulting graph is directed. Defaults to False.
        self_loop (bool, optional): If True, allows self-loops in the resulting graph. Defaults to False.
        multiedge (bool, optional): If True, allows multiple edges between the same pair of nodes. Defaults to False.
        node_mapping (Callable[[int, int, tuple[Graph, ...]], int], optional):
            A function to map nodes from the original graphs to the new graph. It takes three arguments.
            + `node`: The node ID in the original graph.
            + `graph_index`: The index of the graph in the input list.
            + `graphs`: The tuple of all input graphs.
        default_mapping (Literal["separate", "combine", "connect"]):
            Specifies the default behavior for `node_mapping` if not provided.
            + If `"separate"`, it maps nodes to unique IDs by adding the cumulative number of vertices from previous graphs.
            + If `"combine"`, it keeps the original node IDs as they are.
            + If `"connect"`, it maps nodes to unique IDs except the first and last node of each graph,
              which are connected to the previous graph's last node and the next graph's first node, respectively.

    Returns:
        Graph: A new graph that is the union of the input graphs.
    """

    if not graphs:
        return Graph(0, directed)

    if node_mapping is None:
        if default_mapping == "combine":
            node_mapping = lambda node, graph_idx, graphs: node
        elif default_mapping == "separate":
            prev_sum = list(
                itertools.accumulate([0] + [graph.vertices for graph in graphs])
            )
            node_mapping = (
                lambda node, graph_index, graphs: node + prev_sum[graph_index]
            )
        elif default_mapping == "connect":
            prev_sum = list(
                itertools.accumulate([0] + [graph.vertices for graph in graphs])
            )
            node_mapping = (
                lambda node, graph_index, graphs: node
                + prev_sum[graph_index]
                - graph_index
            )
        else:
            raise ValueError(
                f"Unknown default_mapping: {default_mapping}. "
                "Use 'separate' or 'combine'."
            )

    max_node_id = max(
        node_mapping(j, i, graphs)
        for i in range(len(graphs))
        for j in range(graphs[i].vertices)
    )

    g = Graph(max_node_id + 1, directed=directed)

    for i, graph in enumerate(graphs):
        for u, v, weight in graph.get_edges(directed=directed):
            new_u = node_mapping(u, i, graphs)
            new_v = node_mapping(v, i, graphs)

            if not multiedge and g.count_edge(new_u, new_v) > 0:
                continue
            if not self_loop and new_u == new_v:
                continue

            g.add_edge(new_u, new_v, weight)

        for u in range(graph.vertices):
            new_u = node_mapping(u, i, graphs)
            u_rnk = graph.get_rank(u)
            if u_rnk is not None:
                g.set_rank(new_u, u_rnk)

    return g
