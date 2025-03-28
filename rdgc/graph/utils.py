"""
This module provides utility functions to generate graphs.
"""

import warnings
import random as rd

from typing import Any, Callable, Optional

from rdgc.graph.base import Graph


__all__ = ["null", "complete", "tournament", "random_graph", "cycle", "wheel"]


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
