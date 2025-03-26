"""
A module to represent a graph.
"""

import itertools
import math
import random
from collections import defaultdict

from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

from rdgc.utils import filter_none


__all__ = ["Graph"]


class Graph:
    """A class to represent a graph."""

    __directed: bool
    __total_edge_count: int
    __vertex_ranks: List[Any]
    __adjacency_list: List[List[Any]]
    __indeg_counter: List[int]
    __outdeg_counter: List[int]
    __edge_counter: Dict[Tuple[int, int], int]

    def __init__(self, vertices: int, directed: bool = False, rnk: Iterable[Any] = ()):
        """
        Initializes a graph object.

        Args:
            vertices (int): The number of vertices in the graph. Must be non-negative.
            directed (bool, optional): Specifies whether the graph is directed. Defaults to False.
            edge_count (bool, optional): If True, enables tracking of edge counts. Defaults to False.

        Raises:
            ValueError: If the number of vertices is negative.

        Attributes:
            __directed (bool): Indicates if the graph is directed.
            __edge_cnt (int): Tracks the number of edges in the graph.
            __edges (list): Adjacency list representation of the graph.
            __edge_set (defaultdict or None): Tracks edge counts if `edge_count` is True, otherwise None.
        """
        if vertices < 0:
            raise ValueError("Number of vertices must be non-negative")
        self.__directed = directed
        self.__total_edge_count = 0
        self.__vertex_ranks = [None for _ in range(vertices)]
        self.__adjacency_list = [[] for _ in range(vertices)]
        self.__indeg_counter = [0] * vertices if directed else []
        self.__outdeg_counter = [0] * vertices
        self.__edge_counter = defaultdict(int)
        self.set_ranks(rnk)

    @property
    def edges(self) -> int:
        """Returns the total number of edges in the graph."""
        return self.__total_edge_count

    @property
    def vertices(self) -> int:
        """Returns the total number of vertices in the graph."""
        return len(self.__adjacency_list)

    @property
    def directed(self) -> bool:
        """Returns True if the graph is directed, otherwise False."""
        return self.__directed

    def __add_edge(self, u: int, v: int) -> None:
        """Increments the edge count between vertices `u` and `v`."""
        self.__edge_counter[(u, v)] += 1
        if not self.__directed and u != v:
            self.__edge_counter[(v, u)] += 1

    def set_rank(self, u: int, rnk: Any = None):
        """
        Sets the rank of vertex `u`.

        Args:
            u (int): The vertex.
            rnk (Any, optional): The rank of the vertex. Defaults to None.

        Raises:
            IndexError: If the vertex is invalid.
        """
        self.__vertex_ranks[u] = rnk

    def get_rank(self, u: int) -> Any:
        """
        Returns the rank of vertex `u`.

        Args:
            u (int): The vertex.

        Returns:
            Any: The rank of the vertex.

        Raises:
            IndexError: If the vertex is invalid.
        """
        return self.__vertex_ranks[u]

    def set_ranks(self, rnk: Iterable[Any], default: Any = None) -> None:
        """
        Sets the ranks of vertices `u`.

        Args:
            rnk (Iterable[Any]): The ranks of the vertices.
            default (Any, optional): The default rank. Defaults to None.

        Raises:
            IndexError: If the vertex is invalid.
        """
        rnk = itertools.chain(rnk, itertools.repeat(default))
        rnk = itertools.islice(rnk, self.vertices)
        for i, r in enumerate(rnk):
            self.set_rank(i, r)

    def get_ranks(self) -> Tuple[Any]:
        """
        Returns the ranks of all vertices.

        Returns:
            List[Any]: The ranks of all vertices.
        """
        return tuple(self.__vertex_ranks)

    def indegree(self, u: int) -> int:
        """
        Returns the indegree of vertex `u`.

        If the graph is undirected, please use the `degree` method instead.

        Args:
            u (int): The vertex.

        Raises:
            IndexError: If the vertex is invalid.
            ValueError: If the graph is undirected.

        Returns:
            int: The indegree of vertex `u`.
        """
        if not self.__directed:
            raise ValueError("Cannot calculate indegree of an undirected graph")
        return self.__indeg_counter[u]

    def outdegree(self, u: int) -> int:
        """
        Returns the outdegree of vertex `u`.

        If the graph is undirected, please use the `degree` method instead.

        Args:
            u (int): The vertex.

        Raises:
            IndexError: If the vertex is invalid.
            ValueError: If the graph is undirected.

        Returns:
            int: The outdegree of vertex `u`.
        """
        if not self.__directed:
            raise ValueError("Cannot calculate outdegree of an undirected graph")
        return len(self.__adjacency_list[u])

    def degree(self, u: int) -> int:
        """
        Returns the degree of vertex `u`.

        If the graph is directed, please use the `indegree` and `outdegree` methods instead.

        Args:
            u (int): The vertex.

        Raises:
            IndexError: If the vertex is invalid.
            ValueError: If the graph is directed.

        Returns:
            int: The degree of vertex `u`.
        """
        if self.__directed:
            raise ValueError("Cannot calculate degree of a directed graph")
        return self.__outdeg_counter[u]

    def get_edges(
        self, u: Optional[int] = None
    ) -> Generator[Tuple[int, int, Any], None, None]:
        """
        Returns an iterator over the edges of the graph.

        Args:
            u (int, optional): If specified, returns edges incident to vertex `u`. Defaults to None.

        Yields:
            Tuple[int, int, Any]: A tuple containing the vertices and weight of an edge.
        """
        if u is None:
            for uu in range(self.vertices):
                for vv, weight in self.__adjacency_list[uu]:
                    if self.directed or uu <= vv:
                        yield uu, vv, weight
        else:
            for v, weight in self.__adjacency_list[u]:
                yield u, v, weight

    def add_edge(self, u: int, v: int, weight: Any = None) -> None:
        """
        Adds an edge between vertices `u` and `v` with an optional weight.

        Args:
            u (int): The source vertex.
            v (int): The destination vertex.
            weight (Any, optional): The weight of the edge. Defaults to None.

        Raises:
            ValueError: If the vertices are invalid.
        """
        if u not in range(self.vertices) or v not in range(self.vertices):
            raise ValueError("Invalid vertex")
        self.__adjacency_list[u].append((v, weight))
        if not self.__directed and u != v:
            self.__adjacency_list[v].append((u, weight))
        self.__add_edge(u, v)
        self.__outdeg_counter[u] += 1
        if not self.__directed:
            self.__outdeg_counter[v] += 1
        else:
            self.__indeg_counter[v] += 1
        self.__total_edge_count += 1

    def add_rand_edge(
        self,
        edge_count: int,
        *,
        self_loop: bool = False,
        multiedge: bool = False,
        weight_gener: Optional[Callable[[int, int], Any]] = None,
    ) -> None:
        """
        Adds `edge_count` random edges to the graph.

        Args:
            edge_count (int): The number of edges to add.
            self_loop (bool, optional): Specifies whether self-loops are allowed. Defaults to False.
            multiedge (bool, optional): Specifies whether multiple edges are allowed. Defaults to False.
            weight_gener (Callable[[int, int], Any], optional): A function to generate edge weights. Defaults to None.

        Raises:
            ValueError: If the number of edges is invalid.
        """
        if weight_gener is None:
            weight_gener = lambda u, v: None
        if not multiedge:
            max_edge = Graph.calc_max_edge(self.vertices, self.directed, self_loop)
            if self.edges + edge_count > max_edge:
                raise ValueError(
                    f"Too many edges: {self.edges} + {edge_count} > {max_edge}"
                )

        while self.edges < edge_count:
            u, v = (
                random.sample(range(self.vertices), 2)
                if not self_loop
                else random.choices(range(self.vertices), k=2)
            )
            if multiedge or self.count_edge(u, v) == 0:
                self.add_edge(u, v, weight_gener(u, v))

    def count_edges(self, u: int) -> int:
        """
        Returns the number of edges incident to vertex `u`.

        Args:
            u (int): The vertex.

        Returns:
            int: The number of edges incident to vertex `u`.

        Raises:
            ValueError: If the vertex is invalid.
        """
        if u not in range(self.vertices):
            raise ValueError("Invalid vertex")
        return len(self.__adjacency_list[u])

    def count_edge(self, u: int, v: int) -> int:
        """
        Returns the number of edges between vertices `u` and `v`.

        Args:
            u (int): The source vertex.
            v (int): The destination vertex.

        Returns:
            int: The number of edges between vertices `u` and `v`.

        Raises:
            ValueError: If the vertices are invalid or edge count is not enabled.
        """
        if u not in range(self.vertices) or v not in range(self.vertices):
            raise ValueError("Invalid vertex")
        return 0 if (u, v) not in self.__edge_counter else self.__edge_counter[(u, v)]

    def shuffle_nodes(
        self, shuffle: Optional[Callable[[int], Sequence[int]]] = None
    ) -> "Graph":
        """
        Shuffles the vertices of the graph.

        Args:
            shuffle (Callable[[int], Sequence[int]], optional): A function to shuffle the vertices. Defaults to None.

        Returns:
            Graph: A new graph with shuffled vertices.
        """
        mapping = (
            random.sample(range(self.vertices), self.vertices)
            if shuffle is None
            else shuffle(self.vertices)
        )
        new_graph = Graph(self.vertices, self.directed)
        for u in range(self.vertices):
            new_graph.set_rank(mapping[u], self.get_rank(u))
        for u, v, w in self.get_edges():
            new_graph.add_edge(mapping[u], mapping[v], w)
        return new_graph

    def copy(self) -> "Graph":
        """
        Returns a deep copy of the graph.

        Returns:
            Graph: A deep copy of the graph.
        """
        new_graph = Graph(self.vertices, self.directed)
        for u in range(self.vertices):
            new_graph.set_rank(u, self.get_rank(u))
        for u, v, w in self.get_edges():
            new_graph.add_edge(u, v, w)
        return new_graph

    def transpose(self) -> "Graph":
        """
        Returns the transpose of the graph.

        Returns:
            Graph: The transpose of the graph.
        """
        if not self.directed:
            raise ValueError("Cannot transpose an undirected graph")
        new_graph = Graph(self.vertices, self.directed)
        for u, v, w in self.get_edges():
            new_graph.add_edge(v, u, w)
        return new_graph

    def output_nodes(
        self,
        *,
        sep: str = " ",
        printer: Optional[Callable[[int], str]] = None,
    ) -> str:
        """
        Returns a string representation of the vertices.

        Returns:
            str: A string representation of the vertices.
        """
        return sep.join(map(str if printer is None else printer, self.get_ranks()))

    def output_edges(
        self,
        *,
        sep: str = "\n",
        shuffle: bool = False,
        printer: Optional[Callable[[int, int, Any], Optional[str]]] = None,
    ) -> str:
        """
        Returns a string representation of the graph.

        Args:
            shuffle (bool, optional): If True, shuffles the edges. Defaults to False.
            printer (Callable[[int, int, Any], str], optional): A function to format the edges. Defaults to `f"{u} {v} {w}"`.

        Returns:
            str: A string representation of the graph.
        """
        if printer is None:
            printer = lambda u, v, w: f"{u} {v}" if w is None else f"{u} {v} {w}"

        edge_output = [printer(u, v, w) for u, v, w in self.get_edges()]
        if shuffle:
            random.shuffle(edge_output)

        return sep.join(filter_none(edge_output))

    @staticmethod
    def calc_max_edge(size: int, directed: bool, self_loop: bool):
        """
        Calculates the maximum number of edges in a graph with `size` vertices.

        Args:
            size (int): The number of vertices.
            directed (bool): Specifies whether the graph is directed.
            self_loop (bool): Specifies whether self-loops are allowed.

        Returns:
            int: The maximum number of edges in the graph
        """
        max_edge = size * (size - 1)
        if not directed:
            max_edge //= 2
        if self_loop:
            max_edge += size
        return max_edge

    @staticmethod
    def _estimate_comb(n: int, k: int) -> float:
        try:
            return float(sum(math.log(n - i) - math.log(i + 1) for i in range(k)))
        except ValueError:
            return 0.0

    @staticmethod
    def edge_estimate(
        size: int,
        edge_count: int,
        directed: bool,
        self_loop: bool,
        multiedge: bool,
    ) -> float:
        """
        Estimates the number of graphs with `size` vertices and `edge_count` edges.

        Args:
            size (int): The number of vertices.
            edge_count (int): The number of edges.
            directed (bool): Specifies whether the graph is directed.
            self_loop (bool): Specifies whether self-loops are allowed.
            multiedge (bool): Specifies whether multiple edges are allowed.

        Returns:
            float: The estimated number of graphs.
        """
        tot_edge = Graph.calc_max_edge(size, directed, self_loop)
        if multiedge:
            return Graph._estimate_comb(edge_count + tot_edge - 1, edge_count)
        return Graph._estimate_comb(tot_edge, edge_count)
