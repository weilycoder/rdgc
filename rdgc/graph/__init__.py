"""
This module provides utilities for creating and manipulating graphs.
"""

from typing import Any, Callable, Dict

from rdgc.graph.base import Graph
from rdgc.graph.utils import null, complete, tournament, random_graph, cycle, wheel
from rdgc.graph.tree import chain, star, tree, binary_tree, spanning_tree
from rdgc.graph.degree import from_degree_sequence, k_regular
from rdgc.graph.connected import connected, strongly_connected

__all__ = [
    "Graph",
    "make_graph",
    "GRAPH_GENERS",
    "null",
    "complete",
    "tournament",
    "random_graph",
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

GRAPH_GENERS: Dict[str, Callable[..., Graph]] = {
    "null": null,
    "complete": complete,
    "tournament": tournament,
    "random": random_graph,
    "random_graph": random_graph,
    "chain": chain,
    "star": star,
    "tree": tree,
    "binary_tree": binary_tree,
    "spanning_tree": spanning_tree,
    "cycle": cycle,
    "wheel": wheel,
    "connected": connected,
    "strongly_connected": strongly_connected,
    "from_degree_sequence": from_degree_sequence,
    "k_regular": k_regular,
}


def make_graph(_type: str, *args: Any, **kwargs: Any) -> Graph:
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
    return GRAPH_GENERS[_type](*args, **kwargs)
