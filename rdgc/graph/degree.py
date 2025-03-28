"""
This module provides functions to generate graphs with a given degree sequence.
"""

import math
import random
import warnings

from typing import (
    Any,
    Callable,
    Counter,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

from rdgc.graph.base import Graph


__all__ = ["from_degree_sequence", "k_regular"]


class SwitchGraph:
    """A graph which can switch edges quickly"""

    __directed: bool
    __edges: List[Tuple[int, int]]
    __edges_set: Counter[Tuple[int, int]]

    def __init__(
        self,
        directed: bool = False,
        edge_seq: Sequence[Tuple[int, int]] = (),
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
        while third in (first, second):
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
        """Returns the total number of edges in the graph."""
        return len(self.__edges)

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        return iter(self.__edges)


def from_degree_sequence(
    degree_sequence: Union[Sequence[int], Sequence[Tuple[int, int]]],
    iter_times: Optional[int] = None,
    *args: Any,
    self_loop: bool = False,
    multiedge: bool = False,
    weight_gener: Optional[Callable[[int, int], Any]] = None,
    iter_limit: int = int(1e6),
    **kwargs: Any,
) -> Graph:
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
        return Graph(0, False)

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
            Graph.edge_estimate(size, edge_count, directed, self_loop, multiedge)
            / math.log(2)
        )
    iter_times = min(iter_times + 1, iter_limit)

    for _ in range(iter_times):
        sg.switch(self_loop=self_loop, multiedge=multiedge)
    graph = Graph(size, directed)
    for u, v in sg:
        graph.add_edge(u, v, weight_gener(u, v))
    return graph


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
) -> Graph:
    """
    Returns a k-regular graph with `size` vertices and degree `k`.

    Args:
        size (int): The number of vertices.
        k (int): The degree of each vertex.
        iter_times (int, optional): The number of iterations. Defaults to None.
        self_loop (bool, optional): Specifies whether self-loops are allowed. Defaults to False.
        multiedge (bool, optional): Specifies whether multiple edges are allowed. Defaults to False.
        weight_gener (Callable[[int, int], Any], optional): A function to generate edge weights. Defaults to None.
        iter_limit (int, optional): The maximum number of iterations. Defaults to 1e6.

    Raises:
        ValueError: If the degree is invalid.

    Warnings:
        RuntimeWarning: If extra arguments are provided.

    Returns:
        Graph: A k-regular graph with `size` vertices and degree `k`.
    """
    if args or kwargs:
        warnings.warn("Extra arguments are ignored", RuntimeWarning, 2)
    if weight_gener is None:
        weight_gener = lambda u, v: None

    sg = SwitchGraph.k_regular(size, k, self_loop=self_loop, multiedge=multiedge)
    edge_count = sg.edge_count()
    if iter_times is None:
        iter_times = int(
            Graph.edge_estimate(size, edge_count, False, self_loop, multiedge)
            / math.log(2)
        )
    iter_times = min(iter_times + 1, iter_limit)

    for _ in range(iter_times):
        sg.switch(self_loop=self_loop, multiedge=multiedge)
    graph = Graph(size, False)
    for u, v in sg:
        graph.add_edge(u, v, weight_gener(u, v))
    return graph
