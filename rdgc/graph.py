import itertools
import math
import random
import warnings
from collections import defaultdict

from .utils import dsu, filter_none

from typing import *  # type: ignore


__all__ = ["Graph"]


class SwitchGraph:
    """A graph which can switch edges quickly"""

    __directed: bool
    __edges: List[Tuple[int, int]]
    __edges_set: Counter[Tuple[int, int]]

    def __init__(
        self,
        directed: bool = False,
        edge_seq: Sequence[Tuple[int, int]] = [],
    ):
        self.__directed = directed
        self.__edges = []
        self.__edges_set = Counter()
        for u, v in edge_seq:
            self.insert(u, v)

    def count(self, u: int, v: int) -> bool:
        """
        Returns the number of edges between vertices `u` and `v`.
        """
        if not self.__directed and u > v:
            u, v = v, u
        return (u, v) in self.__edges_set

    def insert(self, u: int, v: int) -> None:
        """
        Inserts an edge between vertices `u` and `v`.
        """
        if not self.__directed and u > v:
            u, v = v, u
        self.__edges.append((u, v))
        self.__edges_set[(u, v)] += 1

    def remove(self, ind: Sequence[int]) -> None:
        """
        Removes edges at the given indices by swapping them to the end of the list and then removing.

        Args:
            ind (Sequence[int]): The indices of the edges to remove.
        """
        for t, i in enumerate(sorted(ind, reverse=True), start=1):
            assert 0 <= i < len(self.__edges)
            self.__edges[i], self.__edges[-t] = self.__edges[-t], self.__edges[i]
        for _ in range(len(ind)):
            u, v = self.__edges.pop()
            self.__edges_set[(u, v)] -= 1
            if self.__edges_set[(u, v)] == 0:
                self.__edges_set.pop((u, v))

    def _switch3(
        self,
        first: int,
        second: int,
        *,
        self_loop: bool = False,
        multiedge: bool = False,
    ) -> bool:
        """
        Switches three random edges in the graph.

        Returns:
            bool: True if the edges are switched, otherwise False.
        """
        if len(self.__edges) < 3:
            return False
        u0, u1 = self.__edges[first]
        v0, v1 = self.__edges[second]
        assert u0 == u1 and v0 == v1

        third = random.choice(range(len(self.__edges)))
        while third == first or third == second:
            third = random.choice(range(len(self.__edges)))
        x, y = self.__edges[third]

        # u -> v v -> y x -> u
        if not self_loop:
            if u0 == v0 or v0 == y or x == u0:
                return False
        if {(u0, u1), (v0, v1), (x, y)} == {(u0, v0), (v0, y), (x, u0)}:
            return False
        if not multiedge and len({(u0, v0), (v0, y), (x, u0)}) < 3:
            return False

        self.remove([first, second, third])

        if self.count(u0, v0) or self.count(v0, y) or self.count(x, u0):
            self.insert(u0, u1)
            self.insert(v0, v1)
            self.insert(x, y)
            return False

        self.insert(u0, v0)
        self.insert(v0, y)
        self.insert(x, u0)
        return True

    def switch(self, *, self_loop: bool = False, multiedge: bool = False) -> bool:
        """
        Switches two or three random edges in the graph.

        Returns:
            bool: True if the edges are switched, otherwise False.
        """
        if len(self.__edges) < 2:
            return False

        first, second = random.choices(range(len(self.__edges)), k=2)

        x1, y1 = self.__edges[first]
        x2, y2 = self.__edges[second]

        if self_loop:
            if x1 == x2 or y1 == y2:
                return False
        else:
            if {x1, y1} & {x2, y2} != set():
                return False

        if not multiedge and (self.count(x1, y2) or self.count(x2, y1)):
            return False

        if not multiedge and (x1 == y1 and x2 == y2):
            return self._switch3(
                first, second, self_loop=self_loop, multiedge=multiedge
            )

        self.remove([first, second])

        self.insert(x1, y2)
        self.insert(x2, y1)

        return True

    @staticmethod
    def from_directed_degree_sequence(
        degree_sequence: Sequence[Tuple[int, int]],
        *,
        self_loop: bool = False,
        multiedge: bool = False,
    ) -> "SwitchGraph":
        """
        Returns a graph with the given directed degree sequence.

        Note the time complexity of this function is O(n*m*log(n)),
        where n is the number of vertices and m is the number of edges.

        Args:
            degree_sequence (Sequence[Tuple[int, int]]): The directed degree sequence.
            self_loop (bool, optional): Specifies whether self-loops are allowed. Defaults to False.
            multiedge (bool, optional): Specifies whether multiple edges are allowed. Defaults to False.

        Raises:
            ValueError: If the degree sequence is invalid.

        Returns:
            SwitchGraph: A graph with the given directed degree sequence.
        """
        if any(x < 0 or y < 0 for (x, y) in degree_sequence):
            raise ValueError("Degree sequence must be non-negative")

        x, y = zip(*degree_sequence)
        if sum(x) != sum(y):
            raise ValueError("Degree sequence is not graphical")

        ret = SwitchGraph(True)

        if len(degree_sequence) == 0:
            return ret

        degseq = [[sout, sin, vn] for vn, (sin, sout) in enumerate(degree_sequence)]
        degseq.sort(reverse=True)

        try:
            while max(s[1] for s in degseq) > 0:
                kk = [i for i in range(len(degseq)) if degseq[i][1] > 0]
                _, in_d, vto = degseq[kk[0]]
                degseq[kk[0]][1] = 0
                j = 0
                while in_d:
                    _, _, vfrom = degseq[j]
                    if vto == vfrom and not self_loop:
                        j += 1
                        _, _, vfrom = degseq[j]
                    while in_d and degseq[j][0]:
                        in_d -= 1
                        degseq[j][0] -= 1
                        ret.insert(vfrom, vto)
                        if not multiedge:
                            break
                    j += 1
                degseq.sort(reverse=True)
        except IndexError as err:
            raise ValueError("Degree sequence is not graphical") from err

        return ret

    @staticmethod
    def from_undirected_degree_sequence(
        degree_sequence: Sequence[int],
        *,
        self_loop: bool = False,
        multiedge: bool = False,
    ) -> "SwitchGraph":
        """
        Returns a graph with the given undirected degree sequence.

        Note the time complexity of this function is O(n*m*log(n)),
        where n is the number of vertices and m is the number of edges.

        Args:
            degree_sequence (Sequence[int]): The undirected degree sequence.
            self_loop (bool, optional): Specifies whether self-loops are allowed. Defaults to False.
            multiedge (bool, optional): Specifies whether multiple edges are allowed. Defaults to False.

        Raises:
            ValueError: If the degree sequence is invalid.

        Returns:
            SwitchGraph: A graph with the given undirected degree sequence.
        """
        if any(x < 0 for x in degree_sequence):
            raise ValueError("Degree sequence must be non-negative")

        if sum(degree_sequence) % 2 != 0:
            raise ValueError("Degree sequence is not graphical")

        if len(degree_sequence) == 0:
            return SwitchGraph(False)

        degseq = [[d, vn] for vn, d in enumerate(degree_sequence)]
        degseq.sort(reverse=True)

        edges: List[Tuple[int, int]] = []
        try:
            while len(edges) * 2 < sum(degree_sequence):
                deg, x = degseq[0]
                degseq[0][0] = 0
                if self_loop:
                    while deg > 1:
                        deg -= 2
                        edges.append((x, x))
                        if not multiedge:
                            break
                y = 1
                while deg:
                    while deg and degseq[y][0]:
                        deg -= 1
                        degseq[y][0] -= 1
                        edges.append((x, degseq[y][1]))
                        if not multiedge:
                            break
                    y += 1
                degseq.sort(reverse=True)
        except IndexError as err:
            raise ValueError("Degree sequence is not graphical") from err

        return SwitchGraph(False, edges)

    @staticmethod
    def k_regular(
        size: int,
        k: int,
        *,
        self_loop: bool = False,
        multiedge: bool = False,
    ) -> "SwitchGraph":
        """
        Returns a k-regular graph with `size` vertices and degree `k`.

        Args:
            size (int): The number of vertices.
            k (int): The degree of each vertex.
            self_loop (bool, optional): Specifies whether self-loops are allowed. Defaults to False.
            multiedge (bool, optional): Specifies whether multiple edges are allowed. Defaults to False.

        Raises:
            ValueError: If the degree is invalid.

        Returns:
            SwitchGraph: A k-regular graph with `size` vertices and degree `k`.
        """
        if k < 0:
            raise ValueError("Degree must be non-negative")
        if size * k % 2 != 0:
            raise ValueError("Degree sequence is not graphical")
        if not multiedge:
            max_deg = size if self_loop else size - 1
            if k > max_deg:
                raise ValueError(f"Degree is too large: {k} > {max_deg}")
        if size == 0 or k == 0:
            return SwitchGraph()
        sg = SwitchGraph()
        if k % 2 == 1:
            # size % 2 == 0
            half = size // 2
            for i in range(half):
                sg.insert(i, (i + half) % size)
        k //= 2
        start = 0 if self_loop else 1
        for i in range(size):
            for j in range(start, start + k):
                sg.insert(i, (i + j) % size)
        return sg

    def edge_count(self) -> int:
        return len(self.__edges)

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        return iter(self.__edges)


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
        "from_degree_sequence",
        "k_regular",
    )

    __directed: bool
    __total_edge_count: int
    __vertex_ranks: List[Any]
    __adjacency_list: List[List[Any]]
    __indeg_counter: List[int]
    __outdeg_counter: List[int]
    __edge_counter: Dict[Tuple[int, int], int]

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
            for u in range(self.vertices):
                for v, weight in self.__adjacency_list[u]:
                    if self.directed or u <= v:
                        yield u, v, weight
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
            max_edge = Graph._calc_max_edge(self.vertices, self.directed, self_loop)
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
    def from_degree_sequence(
        degree_sequence: Union[Sequence[int], Sequence[Tuple[int, int]]],
        iter_times: Optional[int] = None,
        *args: Any,
        self_loop: bool = False,
        multiedge: bool = False,
        weight_gener: Optional[Callable[[int, int], Any]] = None,
        iter_limit: int = int(1e6),
        **kwargs: Any,
    ) -> "Graph":
        """
        Returns a graph with the given degree sequence.

        Note the time complexity of this function is O(n*m*log(n)),
        where n is the number of vertices and m is the number of edges.

        Args:
            degree_sequence (Union[Sequence[int], Sequence[Tuple[int, int]]]): The degree sequence.
            iter_times (int, optional): The number of iterations. Defaults to None.
            self_loop (bool, optional): Specifies whether self-loops are allowed. Defaults to False.
            multiedge (bool, optional): Specifies whether multiple edges are allowed. Defaults to False.
            weight_gener (Callable[[int, int], Any], optional): A function to generate edge weights. Defaults to None.
            iter_limit (int, optional): The maximum number of iterations. Defaults to 1e6.

        Raises:
            ValueError: If the degree sequence is invalid.

        Warnings:
            RuntimeWarning: If extra arguments are provided.

        Returns:
            Graph: A graph with the given degree sequence.
        """
        if args or kwargs:
            warnings.warn("Extra arguments are ignored", RuntimeWarning, 2)
        if weight_gener is None:
            weight_gener = lambda u, v: None

        if len(degree_sequence) == 0:
            return Graph.null(0)

        directed = not isinstance(degree_sequence[0], int)
        if directed:
            sg = SwitchGraph.from_directed_degree_sequence(
                cast(Sequence[Tuple[int, int]], degree_sequence),
                self_loop=self_loop,
                multiedge=multiedge,
            )
        else:
            sg = SwitchGraph.from_undirected_degree_sequence(
                cast(Sequence[int], degree_sequence),
                self_loop=self_loop,
                multiedge=multiedge,
            )

        size = len(degree_sequence)
        edge_count = sg.edge_count()
        if iter_times is None:
            iter_times = int(
                Graph._estimate_upperbound(
                    size, edge_count, directed, self_loop, multiedge
                )
                / math.log(2)
            )
        iter_times = min(iter_times + 1, iter_limit)

        for _ in range(iter_times):
            sg.switch(self_loop=self_loop, multiedge=multiedge)
        graph = Graph(size, directed)
        for u, v in sg:
            graph.add_edge(u, v, weight_gener(u, v))
        return graph

    @staticmethod
    def k_regular(
        size: int,
        k: int,
        iter_times: Optional[int] = None,
        *args: Any,
        self_loop: bool = False,
        multiedge: bool = False,
        weight_gener: Optional[Callable[[int, int], Any]] = None,
        iter_limit: int = int(1e6),
        **kwargs: Any,
    ) -> "Graph":
        if args or kwargs:
            warnings.warn("Extra arguments are ignored", RuntimeWarning, 2)
        if weight_gener is None:
            weight_gener = lambda u, v: None

        sg = SwitchGraph.k_regular(size, k, self_loop=self_loop, multiedge=multiedge)
        edge_count = sg.edge_count()
        if iter_times is None:
            iter_times = int(
                Graph._estimate_upperbound(
                    size, edge_count, False, self_loop, multiedge
                )
                / math.log(2)
            )
        iter_times = min(iter_times + 1, iter_limit)

        for _ in range(iter_times):
            sg.switch(self_loop=self_loop, multiedge=multiedge)
        graph = Graph(size, False)
        for u, v in sg:
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

    @staticmethod
    def _estimate_comb(n: int, k: int) -> float:
        try:
            return float(sum(math.log(n - i) - math.log(i + 1) for i in range(k)))
        except ValueError:
            return 0.0

    @staticmethod
    def _estimate_upperbound(
        size: int,
        edge_count: int,
        directed: bool,
        self_loop: bool,
        multiedge: bool,
    ) -> float:
        tot_edge = Graph._calc_max_edge(size, directed, self_loop)
        if multiedge:
            return Graph._estimate_comb(edge_count + tot_edge - 1, edge_count)
        else:
            return Graph._estimate_comb(tot_edge, edge_count)
