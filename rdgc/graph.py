import itertools
import random
from collections import defaultdict
import warnings

from .utils import dsu, filter_none

from typing import *  # type: ignore


__all__ = ["Graph"]


class Graph:
    """A class to represent a graph."""

    GRAPH_TYPES = (
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
    )

    __directed: bool
    __edge_cnt: int
    __ver_rnk: List[Any]
    __edges: List[List[Any]]
    __edge_set: Dict[Tuple[int, int], int]

    def __init__(self, vertices: int, directed: bool = False, rnk: Iterable[Any] = []):
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
        self.__ver_rnk = [None for _ in range(vertices)]
        self.__edges = [[] for _ in range(vertices)]
        self.__edge_set = defaultdict(int)
        self.set_ranks(rnk)

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

    def set_rank(self, u: int, rnk: Any = None):
        """
        Sets the rank of vertex `u`.

        Args:
            u (int): The vertex.
            rnk (Any, optional): The rank of the vertex. Defaults to None.

        Raises:
            IndexError: If the vertex is invalid.
        """
        self.__ver_rnk[u] = rnk

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
        return self.__ver_rnk[u]

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
        return tuple(self.__ver_rnk)

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
    def null(
        size: int,
        *args: Any,
        directed: bool = False,
        **kwargs: Any,
    ) -> "Graph":
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

    @staticmethod
    def complete(
        size: int,
        *args: Any,
        directed: bool = False,
        self_loop: bool = False,
        weight_gener: Optional[Callable[[int, int], Any]] = None,
        **kwargs: Any,
    ) -> "Graph":
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

    @staticmethod
    def tournament(
        size: int,
        *args: Any,
        weight_gener: Optional[Callable[[int, int], Any]] = None,
        **kwargs: Any,
    ) -> "Graph":
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
                i, j = random.sample((u, v), 2)
                graph.add_edge(i, j, weight_gener(i, j))
        return graph

    @staticmethod
    def random(
        size: int,
        edge_count: int,
        *args: Any,
        directed: bool = False,
        self_loop: bool = False,
        multiedge: bool = False,
        weight_gener: Optional[Callable[[int, int], Any]] = None,
        **kwargs: Any,
    ) -> "Graph":
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
            max_edge = Graph._calc_max_edge(size, directed, self_loop)
            if edge_count > max_edge:
                raise ValueError(f"Too many edges: {edge_count} > {max_edge}")

        graph = Graph(size, directed)
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
        return Graph.tree(size, 1.0, 0.0, directed=directed, weight_gener=weight_gener)

    @staticmethod
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
        return Graph.tree(size, 0.0, 1.0, directed=directed, weight_gener=weight_gener)

    @staticmethod
    def tree(
        size: int,
        chain: float = 0.0,
        star: float = 0.0,
        *args: Any,
        directed: bool = False,
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
        *args: Any,
        root: int = 0,
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
        if left is None and right is None:
            rnk = 0.5
        elif left is None:
            rnk = 1.0 - cast(float, right)
        elif right is None:
            rnk = left
        else:
            rnk = left / (left + right)

        tree = Graph(size, directed)
        left_rt, right_rt = [root], [root]
        for i in range(size):
            if i == root:
                tree.set_rank(i, root_rank)
                continue
            if random.random() < rnk:
                tree.set_rank(i, left_rank)
                rt = random.randrange(len(left_rt))
                tree.add_edge(i, left_rt[rt], weight_gener(i, left_rt[rt]))
                left_rt[rt] = i
                right_rt.append(i)
            else:
                tree.set_rank(i, right_rank)
                rt = random.randrange(len(right_rt))
                tree.add_edge(i, right_rt[rt], weight_gener(i, right_rt[rt]))
                right_rt[rt] = i
                left_rt.append(i)
        return tree

    @staticmethod
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

        tree = Graph(size)
        dsu_instance = dsu(size)
        while tree.edges < size - 1:
            u, v = random.sample(range(size), 2)
            if dsu_instance.union(u, v):
                tree.add_edge(u, v, weight_gener(u, v))
        return tree

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
            max_edge = Graph._calc_max_edge(size, directed and cycle, self_loop)
            if edge_count > max_edge:
                raise ValueError(f"Too many edges: {edge_count} > {max_edge}")

        if weight_gener is None:
            weight_gener = lambda u, v: None

        graph = Graph(size, directed)
        for u, v, _ in Graph.spanning_tree(size).get_edges():
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

    @staticmethod
    def graph(_type: str, *args: Any, **kwargs: Any) -> "Graph":
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
        if _type not in Graph.GRAPH_TYPES:
            raise ValueError(f"Unknown graph type: {_type}")
        return getattr(Graph, _type)(*args, **kwargs)

    @staticmethod
    def _calc_max_edge(size: int, directed: bool, self_loop: bool):
        max_edge = size * (size - 1)
        if not directed:
            max_edge //= 2
        if self_loop:
            max_edge += size
        return max_edge
