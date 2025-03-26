"""
This module provides utilities for creating and manipulating graphs.
"""

from typing import Any

from rdgc.graph.graph import Graph
from rdgc.graph.utils import null, complete, tournament, random, cycle, wheel
from rdgc.graph.tree import chain, star, tree, binary_tree, spanning_tree
from rdgc.graph.degree import from_degree_sequence, k_regular
from rdgc.graph.connected import connected, strongly_connected

__all__ = [
    "Graph",
    "graph",
    "GRAPH_GENERS",
    "null",
    "complete",
    "tournament",
    "random",
    "chain",
    "star",
    "tree",
    "binary_tree",
    "spanning_tree",
    "cycle",
    "wheel",
    "connected",
    "strongly_connected",
    "from_degree_sequence",
    "k_regular",
]

GRAPH_GENERS = tuple(__all__[3:])


def graph(_type: str, *args: Any, **kwargs: Any) -> Graph:
    """
    Returns a graph of the specified type.

    Args:
        name (str): The type of graph.
        *args: The arguments to pass to the graph constructor.
        **kwargs: The keyword arguments to pass to the graph constructor.

    Raises:
        ValueError: If the graph type is unknown.

    Warnings:
        RuntimeWarning: If extra arguments are provided

    Returns:
        Graph: A graph of the specified type.
    """
    if _type not in GRAPH_GENERS:
        raise ValueError(f"Unknown graph type: {_type}")
    return getattr(Graph, _type)(*args, **kwargs)
