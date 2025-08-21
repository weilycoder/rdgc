"""
This module provides utility functions to generate graphs.
"""

import math
import itertools
import warnings
import random as rd

from typing import Any, Callable, Literal, Optional, Tuple, Union, Iterable, List, cast

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
        u, v = rd.sample(range(size), 2) if not self_loop else rd.choices(range(size), k=2)
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


def lattice(
    dim: Union[List[int], Tuple[int, ...]],
    nei: int = 1,
    *args: Any,
    directed: bool = False,
    mutual: bool = True,
    circular: Union[bool, Iterable[bool]] = True,
    weight_gener_by_id: Optional[Callable[[int, int], Any]] = None,
    weight_gener_by_dim: Optional[Callable[[Tuple[int, ...], Tuple[int, ...]], Any]] = None,
    **kwargs: Any,
):
    """
    Returns a lattice graph with the specified dimensions.

    The graph is constructed by connecting each vertex to its neighbors in the specified dimensions.

    Args:
        dim (Union[List[int], Tuple[int, ...]]): The dimensions of the lattice.
        nei (int, optional): The number of neighbors to connect to in each dimension. Defaults to 1.
        directed (bool, optional): If True, the graph is directed. Defaults to False.
        mutual (bool, optional): If True, adds edges in both directions for each connection. Defaults to True.
        circular (Union[bool, Iterable[bool]], optional): If True, the graph wraps around in each dimension.
            If an iterable, specifies whether each dimension is circular.
            Defaults to True, meaning all dimensions are circular.
        weight_gener_by_id (Optional[Callable[[int, int], Any]], optional): A function to generate edge weights
            based on vertex IDs. If provided, it takes two arguments: the IDs of the vertices being connected.
        weight_gener_by_dim (Optional[Callable[[Tuple[int, ...], Tuple[int, ...]], Any]], optional):
            A function to generate edge weights based on the dimensions of the vertices being connected.
            If provided, it takes two arguments: the dimensions of the vertices being connected.
            If both `weight_gener_by_id` and `weight_gener_by_dim` are provided, a ValueError is raised.
    """
    if weight_gener_by_id is not None and weight_gener_by_dim is not None:
        raise ValueError("Only one of weight_gener_by_id or weight_gener_by_dim can be provided.")

    if args or kwargs:
        warnings.warn("Extra arguments are ignored", RuntimeWarning, 2)

    def weight_gener(x: int, y: int, dim_x: Tuple[int, ...], dim_y: Tuple[int, ...]) -> Any:
        if weight_gener_by_id is not None:
            return weight_gener_by_id(x, y)
        if weight_gener_by_dim is not None:
            return weight_gener_by_dim(dim_x, dim_y)
        return None

    g = Graph(math.prod(dim), directed)

    num = len(dim)
    try:
        circular = iter(cast(Iterable[bool], circular))
        circular = itertools.chain(circular, itertools.repeat(True))
        circular = itertools.islice(circular, num)
    except TypeError:
        circular = itertools.repeat(cast(bool, circular), num)
    circular = list(circular)

    pre_prod = [1] + list(dim[0:-1])
    for i in range(1, num):
        pre_prod[i] *= pre_prod[i - 1]

    for dim_u in itertools.product(*map(range, dim)):
        u = int(math.sumprod(dim_u, pre_prod))
        for i, cir in zip(range(num), circular):
            flag, v = dim_u[i], u
            for _ in range(nei):
                flag += 1
                v += pre_prod[i]
                if flag == dim[i]:
                    if cir and dim[i] > 2:
                        v -= pre_prod[i] * flag
                        flag = 0
                    else:
                        break
                dim_v = dim_u[:i] + (flag,) + dim_u[i + 1 :]
                g.add_edge(u, v, weight_gener(u, v, dim_u, dim_v))
                if directed and mutual:
                    g.add_edge(v, u, weight_gener(v, u, dim_v, dim_u))

    return g


def union(
    *graphs: Graph,
    directed: bool = False,
    self_loop: bool = False,
    multiedge: bool = False,
    node_mapping: Optional[Callable[[int, int, Tuple[Graph, ...]], int]] = None,
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
            prev_sum = list(itertools.accumulate([0] + [graph.vertices for graph in graphs]))
            node_mapping = lambda node, graph_index, graphs: node + prev_sum[graph_index]
        elif default_mapping == "connect":
            prev_sum = list(itertools.accumulate([0] + [graph.vertices for graph in graphs]))
            node_mapping = lambda node, graph_index, graphs: node + prev_sum[graph_index] - graph_index
        else:
            raise ValueError(f"Unknown default_mapping: {default_mapping}. " "Use 'separate' or 'combine'.")

    max_node_id = max(node_mapping(j, i, graphs) for i in range(len(graphs)) for j in range(graphs[i].vertices))

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
