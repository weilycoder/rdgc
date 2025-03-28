"""
This module contains functions to generate connected graphs.
"""

import random
import warnings

from typing import Any, Callable, Optional

from rdgc.graph.base import Graph
from rdgc.graph.tree import spanning_tree
from rdgc.graph.utils import cycle as cyc


__all__ = ["connected", "strongly_connected"]


def connected(
    size: int,
    edge_count: int,
    *args: Any,
    directed: bool = False,
    self_loop: bool = False,
    cycle: bool = False,
    multiedge: bool = False,
    weight_gener: Optional[Callable[[int, int], Any]] = None,
    **kwargs: Any,
) -> "Graph":
    """
    Returns a connected graph with `size` vertices and `edge_count` edges.

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
        RuntimeWarning: If extra arguments are provided.

    Returns:
        Graph: A connected graph with `size` vertices and `edge_count` edges.
    """
    if args or kwargs:
        warnings.warn("Extra arguments are ignored", RuntimeWarning, 2)
    if cycle and not directed:
        warnings.warn(
            "Argument `cycle` is ignored for undirected graphs", RuntimeWarning, 2
        )

    if edge_count < size - 1:
        raise ValueError(f"Too few edges: {edge_count} < {size - 1}")
    if not multiedge:
        max_edge = Graph.calc_max_edge(size, directed and cycle, self_loop)
        if edge_count > max_edge:
            raise ValueError(f"Too many edges: {edge_count} > {max_edge}")

    if weight_gener is None:
        weight_gener = lambda u, v: None

    graph = Graph(size, directed)
    for u, v, _ in spanning_tree(size).get_edges():
        u, v = sorted((u, v))
        graph.add_edge(u, v, weight_gener(u, v))
    while graph.edges < edge_count:
        u, v = (
            random.sample(range(size), 2)
            if not self_loop
            else random.choices(range(size), k=2)
        )
        if not cycle:
            u, v = sorted((u, v))
        if multiedge or graph.count_edge(u, v) == 0:
            graph.add_edge(u, v, weight_gener(u, v))
    return graph


def strongly_connected(
    size: int,
    edge_count: int,
    *args: Any,
    self_loop: bool = False,
    multiedge: bool = False,
    weight_gener: Optional[Callable[[int, int], Any]] = None,
    **kwargs: Any,
) -> "Graph":
    """
    Returns a strongly connected graph with `size` vertices and `edge_count` edges.

    Args:
        size (int): The number of vertices.
        edge_count (int): The number of edges.
        self_loop (bool, optional): Specifies whether self-loops are allowed. Defaults to False.
        multiedge (bool, optional): Specifies whether multiple edges are allowed. Defaults to False.
        weight_gener (Callable[[int, int], Any], optional): A function to generate edge weights. Defaults to None.

    Raises:
        ValueError: If the number of edges is invalid.

    Warnings:
        RuntimeWarning: If extra arguments are provided.

    Returns:
        Graph: A strongly connected graph with `size` vertices and `edge_count` edges.
    """
    if args or kwargs:
        warnings.warn("Extra arguments are ignored", RuntimeWarning, 2)
    if weight_gener is None:
        weight_gener = lambda u, v: None

    if edge_count < size:
        raise ValueError(f"Too few edges: {edge_count} < {size}")
    if not multiedge:
        max_edge = Graph.calc_max_edge(size, True, self_loop)
        if edge_count > max_edge:
            raise ValueError(f"Too many edges: {edge_count} > {max_edge}")

    graph = cyc(size, directed=True, weight_gener=weight_gener)

    while graph.edges < edge_count:
        u, v = (
            random.sample(range(size), 2)
            if not self_loop
            else random.choices(range(size), k=2)
        )
        if multiedge or graph.count_edge(u, v) == 0:
            graph.add_edge(u, v, weight_gener(u, v))

    return graph
