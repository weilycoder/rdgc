import random
from collections import defaultdict

from .utils import dsu

from typing import *  # type: ignore


__all__ = ["Graph"]


class Graph:
    __directed: bool
    __edge_cnt: int
    __edges: List[List[Any]]
    __edge_set: Dict[Tuple[int, int], int]

    def __init__(self, vertices: int, directed: bool = False):
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
        self.__edge_cnt = 0
        self.__edges = [[] for _ in range(vertices)]
        self.__edge_set = defaultdict(int)

    @property
    def edges(self) -> int:
        """Returns the total number of edges in the graph."""
        return self.__edge_cnt

    @property
    def vertices(self) -> int:
        """Returns the total number of vertices in the graph."""
        return len(self.__edges)

    @property
    def directed(self) -> bool:
        """Returns True if the graph is directed, otherwise False."""
        return self.__directed

    def __add_edge(self, u: int, v: int) -> None:
        """Increments the edge count between vertices `u` and `v`."""
        self.__edge_set[(u, v)] += 1
        if not self.__directed and u != v:
            self.__edge_set[(v, u)] += 1

    def get_edges(
        self, u: Union[int, None] = None
    ) -> Generator[Tuple[int, int, Any], None, None]:
        """
        Returns an iterator over the edges of the graph.

        Args:
            u (int, optional): If specified, returns edges incident to vertex `u`. Defaults to None.

        Yields:
            Tuple[int, int, Any]: A tuple containing the vertices and weight of an edge.
        """
        if u is None:
            for u in range(self.vertices):
                for v, weight in self.__edges[u]:
                    if self.directed or u <= v:
                        yield u, v, weight
        else:
            for v, weight in self.__edges[u]:
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
        self.__edges[u].append((v, weight))
        if not self.__directed and u != v:
            self.__edges[v].append((u, weight))
        self.__add_edge(u, v)
        self.__edge_cnt += 1

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
        return len(self.__edges[u])

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
        return 0 if (u, v) not in self.__edge_set else self.__edge_set[(u, v)]

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

    def output(
        self,
        *,
        shuffle: bool = False,
        printer: Optional[Callable[[int, int, Any], str]] = None,
    ) -> str:
        """
        Returns a string representation of the graph.

        Args:
            shuffle (bool, optional): If True, shuffles the edges. Defaults to False.
            printer (Callable[[int, int, Any], str], optional): A function to format the edges. Defaults to `f"{u} {v} {w}"`.
        """
        if printer is None:
            printer = lambda u, v, w: f"{u} {v}" if w is None else f"{u} {v} {w}"
        output = [printer(u, v, w) for u, v, w in self.get_edges()]
        if shuffle:
            random.shuffle(output)
        return "\n".join(output)

    @staticmethod
    def null(size: int, *, directed: bool = False) -> "Graph":
        """
        Returns a null graph with `size` vertices.

        Args:
            size (int): The number of vertices.
            directed (bool, optional): Specifies whether the graph is directed. Defaults to False.
        """
        return Graph(size, directed)

    @staticmethod
    def complete(
        size: int,
        *,
        directed: bool = False,
        self_loop: bool = False,
        weight_gener: Optional[Callable[[int, int], Any]] = None,
    ) -> "Graph":
        """
        Returns a complete graph with `size` vertices.

        Args:
            size (int): The number of vertices.
            directed (bool, optional): Specifies whether the graph is directed. Defaults to False.
            self_loop (bool, optional): Specifies whether self-loops are allowed. Defaults to False.
            weight_gener (Callable[[int, int], Any], optional): A function to generate edge weights. Defaults to None.
        """
        if weight_gener is None:
            weight_gener = lambda u, v: None
        graph = Graph(size, directed)
        for u in range(size):
            for v in range(size) if directed else range(u, size):
                if u != v or self_loop:
                    graph.add_edge(u, v, weight_gener(u, v))
        return graph

    @staticmethod
    def tournament(
        size: int,
        weight_gener: Optional[Callable[[int, int], Any]] = None,
    ) -> "Graph":
        """
        Returns a tournament graph with `size` vertices.

        Args:
            size (int): The number of vertices.
            weight_gener (Callable[[int, int], Any], optional): A function to generate edge weights. Defaults to None.
        """
        if weight_gener is None:
            weight_gener = lambda u, v: None
        graph = Graph(size, True)
        for u in range(size):
            for v in range(u + 1, size):
                i, j = random.sample((u, v), 2)
                graph.add_edge(i, j, weight_gener(i, j))
        return graph

    @staticmethod
    def random(
        size: int,
        edge_count: int,
        *,
        directed: bool = False,
        self_loop: bool = False,
        multiedge: bool = False,
        weight_gener: Optional[Callable[[int, int], Any]] = None,
    ) -> "Graph":
        graph = Graph(size, directed)
        if weight_gener is None:
            weight_gener = lambda u, v: None
        if not multiedge:
            max_edge = Graph._calc_max_edge(size, directed, self_loop)
            if edge_count > max_edge:
                raise ValueError(f"Too many edges: {edge_count} > {max_edge}")
        while graph.edges < edge_count:
            u, v = (
                random.sample(range(size), 2)
                if not self_loop
                else random.choices(range(size), k=2)
            )
            if multiedge or graph.count_edge(u, v) == 0:
                graph.add_edge(u, v, weight_gener(u, v))
        return graph

    @staticmethod
    def chain(
        size: int,
        *,
        directed: bool = False,
        weight_gener: Optional[Callable[[int, int], Any]] = None,
    ) -> "Graph":
        """
        Returns a chain graph with `size` vertices.

        Args:
            size (int): The number of vertices.
            directed (bool, optional): Specifies whether the graph is directed. Defaults to False.
            weight_gener (Callable[[int, int], Any], optional): A function to generate edge weights. Defaults to None.
        """
        return Graph.tree(size, 1.0, 0.0, directed=directed, weight_gener=weight_gener)

    @staticmethod
    def star(
        size: int,
        *,
        directed: bool = False,
        weight_gener: Optional[Callable[[int, int], Any]] = None,
    ) -> "Graph":
        """
        Returns a star graph with `size` vertices.

        Args:
            size (int): The number of vertices.
            directed (bool, optional): Specifies whether the graph is directed. Defaults to False.
            weight_gener (Callable[[int, int], Any], optional): A function to generate edge weights. Defaults to None.
        """
        return Graph.tree(size, 0.0, 1.0, directed=directed, weight_gener=weight_gener)

    @staticmethod
    def tree(
        size: int,
        chain: float = 0.0,
        star: float = 0.0,
        *,
        directed: bool = False,
        weight_gener: Optional[Callable[[int, int], Any]] = None,
        father_gener: Optional[Callable[[int], int]] = None,
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
        """
        if chain + star > 1.0 or chain < 0.0 or star < 0.0:
            raise ValueError("Invalid parameters")
        if weight_gener is None:
            weight_gener = lambda u, v: None
        if father_gener is None:
            father_gener = lambda u: random.randint(0, u - 1)
        chain_cnt = int((size - 1) * chain)
        star_cnt = int((size - 1) * star)
        tree = Graph(size, directed)
        for i in range(1, chain_cnt + 1):
            tree.add_edge(i, i - 1, weight_gener(i, i - 1))
        for i in range(chain_cnt + 1, chain_cnt + star_cnt + 1):
            tree.add_edge(i, chain_cnt, weight_gener(i, chain_cnt))
        for i in range(chain_cnt + star_cnt + 1, size):
            father = father_gener(i)
            tree.add_edge(i, father, weight_gener(i, father))
        return tree

    @staticmethod
    def binary_tree(
        size: int,
        left: Optional[float] = None,
        right: Optional[float] = None,
        *,
        directed: bool = False,
        weight_gener: Optional[Callable[[int, int], Any]] = None,
    ) -> "Graph":
        """
        Returns a binary tree graph with `size` vertices.

        Args:
            size (int): The number of vertices.
            left (float, optional): The proportion of left edges. Defaults to None.
            right (float, optional): The proportion of right edges. Defaults to None.
            directed (bool, optional): Specifies whether the graph is directed. Defaults to False.
            weight_gener (Callable[[int, int], Any], optional): A function to generate edge weights. Defaults to None.
        """
        if weight_gener is None:
            weight_gener = lambda u, v: None
        if left is None and right is None:
            rnk = 0.5
        elif left is None:
            rnk = 1.0 - cast(float, right)
        elif right is None:
            rnk = left
        else:
            rnk = left / (left + right)
        tree = Graph(size, directed)
        left_rt, right_rt = [0], [0]
        for i in range(1, size):
            if random.random() < rnk:
                rt = random.randrange(len(left_rt))
                tree.add_edge(i, left_rt[rt], weight_gener(i, left_rt[rt]))
                left_rt[rt] = i
                right_rt.append(i)
            else:
                rt = random.randrange(len(right_rt))
                tree.add_edge(i, right_rt[rt], weight_gener(i, right_rt[rt]))
                right_rt[rt] = i
                left_rt.append(i)
        return tree

    @staticmethod
    def union_tree(
        size: int,
        *,
        weight_gener: Optional[Callable[[int, int], Any]] = None,
    ) -> "Graph":
        """
        Returns a union tree graph with `size` vertices.

        Args:
            size (int): The number of vertices.
            weight_gener (Callable[[int, int], Any], optional): A function to generate edge weights. Defaults to None.
        """
        if weight_gener is None:
            weight_gener = lambda u, v: None
        tree = Graph(size)
        dsu_instance = dsu(size)
        while tree.edges < size - 1:
            u, v = random.sample(range(size), 2)
            if dsu_instance.union(u, v):
                tree.add_edge(u, v, weight_gener(u, v))
        return tree

    @staticmethod
    def connected(
        size: int,
        edge_count: int,
        *,
        directed: bool = False,
        self_loop: bool = False,
        multiedge: bool = False,
        weight_gener: Optional[Callable[[int, int], Any]] = None,
    ) -> "Graph":
        if weight_gener is None:
            weight_gener = lambda u, v: None
        if edge_count < size - 1:
            raise ValueError(f"Too few edges: {edge_count} < {size - 1}")
        if not multiedge:
            max_edge = Graph._calc_max_edge(size, directed, self_loop)
            if edge_count > max_edge:
                raise ValueError(f"Too many edges: {edge_count} > {max_edge}")
        graph = Graph(size, directed)
        for u, v, _ in Graph.union_tree(size).get_edges():
            u, v = sorted((u, v))
            graph.add_edge(u, v, weight_gener(u, v))
        while graph.edges < edge_count:
            u, v = (
                random.sample(range(size), 2)
                if not self_loop
                else random.choices(range(size), k=2)
            )
            if multiedge or graph.count_edge(u, v) == 0:
                graph.add_edge(u, v, weight_gener(u, v))
        return graph

    @staticmethod
    def _calc_max_edge(size: int, directed: bool, self_loop: bool):
        max_edge = size * (size - 1)
        if not directed:
            max_edge //= 2
        if self_loop:
            max_edge += size
        return max_edge
