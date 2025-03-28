"""
This module provides functions to generate tree graphs.
"""

import random
import warnings

from typing import Any, Callable, Optional, cast

from rdgc.utils import Dsu
from rdgc.graph.base import Graph


__all__ = ["chain", "star", "tree", "binary_tree", "spanning_tree"]


def chain(
    size: int,
    *args: Any,
    directed: bool = False,
    weight_gener: Optional[Callable[[int, int], Any]] = None,
    **kwargs: Any,
) -> "Graph":
    """
    Returns a chain graph with `size` vertices.

    Args:
        size (int): The number of vertices.
        directed (bool, optional): Specifies whether the graph is directed. Defaults to False.
        weight_gener (Callable[[int, int], Any], optional): A function to generate edge weights. Defaults to None.

    Warnings:
        RuntimeWarning: If extra arguments are provided.

    Returns:
        Graph: A chain graph with `size` vertices.
    """
    if args or kwargs:
        warnings.warn("Extra arguments are ignored", RuntimeWarning, 2)
    return tree(size, 1.0, 0.0, directed=directed, weight_gener=weight_gener)


def star(
    size: int,
    *args: Any,
    directed: bool = False,
    weight_gener: Optional[Callable[[int, int], Any]] = None,
    **kwargs: Any,
) -> "Graph":
    """
    Returns a star graph with `size` vertices.

    Args:
        size (int): The number of vertices.
        directed (bool, optional): Specifies whether the graph is directed. Defaults to False.
        weight_gener (Callable[[int, int], Any], optional): A function to generate edge weights. Defaults to None.

    Warnings:
        RuntimeWarning: If extra arguments are provided.

    Returns:
        Graph: A star graph with `size` vertices.
    """
    if args or kwargs:
        warnings.warn("Extra arguments are ignored", RuntimeWarning, 2)
    return tree(size, 0.0, 1.0, directed=directed, weight_gener=weight_gener)


def tree(
    size: int,
    chain_ratio: float = 0.0,
    star_ratio: float = 0.0,
    *args: Any,
    directed: bool = False,
    rank: bool = True,
    root_rank: Any = 0,
    chain_rank: Any = 1,
    star_rank: Any = 2,
    rand_rank: Any = -1,
    weight_gener: Optional[Callable[[int, int], Any]] = None,
    father_gener: Optional[Callable[[int], int]] = None,
    **kwargs: Any,
) -> "Graph":
    """
    Returns a tree graph with `size` vertices.

    Args:
        size (int): The number of vertices.
        chain (float, optional): The proportion of chain edges. Defaults to 0.0.
        star (float, optional): The proportion of star edges. Defaults to 0.0.
        directed (bool, optional): Specifies whether the graph is directed. Defaults to False.
        weight_gener (Callable[[int, int], Any], optional): A function to generate edge weights. Defaults to None.
        father_gener (Callable[[int], int], optional): A function to generate parent vertices. Defaults to None.

    Raises:
        ValueError: If the parameters are invalid.

    Warnings:
        RuntimeWarning: If extra arguments are provided.

    Returns:
        Graph: A tree graph with `size` vertices.
    """
    if args or kwargs:
        warnings.warn("Extra arguments are ignored", RuntimeWarning, 2)
    if chain_ratio + star_ratio > 1.0 or chain_ratio < 0.0 or star_ratio < 0.0:
        raise ValueError("Invalid parameters")

    if weight_gener is None:
        weight_gener = lambda u, v: None
    if father_gener is None:
        father_gener = lambda u: random.randint(0, u - 1)
    if not rank:
        root_rank = chain_rank = star_rank = rand_rank = None

    chain_cnt = int((size - 1) * chain_ratio)
    star_cnt = int((size - 1) * star_ratio)
    tree_graph = Graph(size, directed)
    tree_graph.set_rank(0, root_rank)
    for i in range(1, chain_cnt + 1):
        tree_graph.set_rank(i, chain_rank)
        tree_graph.add_edge(i, i - 1, weight_gener(i, i - 1))
    for i in range(chain_cnt + 1, chain_cnt + star_cnt + 1):
        tree_graph.set_rank(i, star_rank)
        tree_graph.add_edge(i, chain_cnt, weight_gener(i, chain_cnt))
    for i in range(chain_cnt + star_cnt + 1, size):
        father = father_gener(i)
        tree_graph.set_rank(i, rand_rank)
        tree_graph.add_edge(i, father, weight_gener(i, father))
    return tree_graph


def binary_tree(
    size: int,
    left: Optional[float] = None,
    right: Optional[float] = None,
    *args: Any,
    root: int = 0,
    rank: bool = True,
    root_rank: Any = 0,
    left_rank: Any = -1,
    right_rank: Any = 1,
    directed: bool = False,
    weight_gener: Optional[Callable[[int, int], Any]] = None,
    **kwargs: Any,
) -> "Graph":
    """
    Returns a binary tree graph with `size` vertices.

    Args:
        size (int): The number of vertices.
        left (float, optional): The proportion of left edges. Defaults to None.
        right (float, optional): The proportion of right edges. Defaults to None.
        root (int, optional): The root vertex. Defaults to 0.
        root_rank (Any, optional): The rank of the root vertex. Defaults to 0.
        left_rank (Any, optional): The rank of the left child. Defaults to -1.
        right_rank (Any, optional): The rank of the right child. Defaults to 1.
        directed (bool, optional): Specifies whether the graph is directed. Defaults to False.
        weight_gener (Callable[[int, int], Any], optional): A function to generate edge weights. Defaults to None.

    Raises:
        ValueError: If the root vertex is invalid.

    Warnings:
        RuntimeWarning: If extra arguments are provided.

    Returns:
        Graph: A binary tree graph with `size` vertices.
    """
    if args or kwargs:
        warnings.warn("Extra arguments are ignored", RuntimeWarning, 2)
    if root < 0 or root >= size:
        raise ValueError("Invalid root")

    if weight_gener is None:
        weight_gener = lambda u, v: None
    if not rank:
        root_rank = left_rank = right_rank = None
    if left is None and right is None:
        rnk = 0.5
    elif left is None:
        rnk = 1.0 - cast(float, right)
    elif right is None:
        rnk = left
    else:
        rnk = left / (left + right)

    btree = Graph(size, directed)
    left_rt, right_rt = [root], [root]
    for i in range(size):
        if i == root:
            btree.set_rank(i, root_rank)
            continue
        if random.random() < rnk:
            btree.set_rank(i, left_rank)
            rt = random.randrange(len(left_rt))
            btree.add_edge(i, left_rt[rt], weight_gener(i, left_rt[rt]))
            left_rt[rt] = i
            right_rt.append(i)
        else:
            btree.set_rank(i, right_rank)
            rt = random.randrange(len(right_rt))
            btree.add_edge(i, right_rt[rt], weight_gener(i, right_rt[rt]))
            right_rt[rt] = i
            left_rt.append(i)
    return btree


def spanning_tree(
    size: int,
    *args: Any,
    weight_gener: Optional[Callable[[int, int], Any]] = None,
    **kwargs: Any,
) -> "Graph":
    """
    Returns a tree graph with `size` vertices, which is generated by union-find.

    Args:
        size (int): The number of vertices.
        weight_gener (Callable[[int, int], Any], optional): A function to generate edge weights. Defaults to None.

    Warnings:
        RuntimeWarning: If extra arguments are provided.

    Returns:
        Graph: A tree graph with `size` vertices.
    """
    if args or kwargs:
        warnings.warn("Extra arguments are ignored", RuntimeWarning, 2)
    if weight_gener is None:
        weight_gener = lambda u, v: None

    tree_graph = Graph(size)
    dsu_instance = Dsu(size)
    while tree_graph.edges < size - 1:
        u, v = random.sample(range(size), 2)
        if dsu_instance.union(u, v):
            tree_graph.add_edge(u, v, weight_gener(u, v))
    return tree_graph
