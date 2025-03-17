import random
from collections import defaultdict
from typing import *  # type: ignore


__all__ = ["Graph"]


class Graph:
    __directed: bool
    __edge_cnt: int
    __edges: List[List[Any]]
    __edge_set: Union[Dict[Tuple[int, int], int], None]

    def __init__(
        self, vertices: int, directed: bool = False, *, enable_set: bool = False
    ):
        if vertices < 0:
            raise ValueError("Number of vertices must be non-negative")
        self.__directed = directed
        self.__edge_cnt = 0
        self.__edges = [[] for _ in range(vertices)]
        self.__edge_set = defaultdict(int) if enable_set else None

    @property
    def edges(self) -> int:
        return self.__edge_cnt

    @property
    def vertices(self) -> int:
        return len(self.__edges)

    @property
    def directed(self) -> bool:
        return self.__directed

    @property
    def enable_set(self) -> bool:
        return self.__edge_set is not None

    def get_edges(
        self, u: Union[int, None] = None
    ) -> Generator[Tuple[int, int, Any], None, None]:
        if u is None:
            for u in range(self.vertices):
                for v, weight in self.__edges[u]:
                    if self.directed or u <= v:
                        yield u, v, weight
        else:
            for v, weight in self.__edges[u]:
                yield u, v, weight

    def add_edge(self, u: int, v: int, weight: Any = None) -> None:
        if u not in range(self.vertices) or v not in range(self.vertices):
            raise ValueError("Invalid vertex")
        self.__edges[u].append((v, weight))
        if not self.__directed and u != v:
            self.__edges[v].append((u, weight))
        if self.__edge_set is not None:
            self.__edge_set[(u, v)] += 1
            if not self.__directed and u != v:
                self.__edge_set[(v, u)] += 1
        self.__edge_cnt += 1

    def count_edges(self, u: int) -> int:
        if u not in range(self.vertices):
            raise ValueError("Invalid vertex")
        return len(self.__edges[u])

    def count_edge(self, u: int, v: int) -> int:
        if self.__edge_set is None:
            raise ValueError("Edge set is not enabled")
        return self.__edge_set[(u, v)]

    def shuffle_edges(
        self,
        shuffle: Callable[[int], Sequence[int]] = lambda x: random.sample(range(x), x),
    ) -> "Graph":
        mapping = shuffle(self.vertices)
        new_graph = Graph(self.vertices, self.directed, enable_set=self.enable_set)
        for u, v, w in self.get_edges():
            new_graph.add_edge(mapping[u], mapping[v], w)
        return new_graph

    def output(
        self,
        *,
        shuffle: bool = False,
        printer: Union[Callable[[int, int, Any], str], None] = None,
    ) -> str:
        if printer is None:
            printer = lambda u, v, w: f"{u} {v}" if w is None else f"{u} {v} {w}"
        output = [printer(u, v, w) for u, v, w in self.get_edges()]
        if shuffle:
            random.shuffle(output)
        return "\n".join(output)
