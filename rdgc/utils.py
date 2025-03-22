import sys
import random

from typing import *  # type: ignore

__all__ = ["set_randseed_from_shell"]


T = TypeVar("T")


def set_randseed_from_shell() -> Tuple[Union[None, int, Tuple[int, ...]]]:
    random.seed(" ".join(sys.argv[1:]), version=2)
    return random.getstate()


def filter_none(iterable: Iterable[Optional[T]]) -> Iterable[T]:
    return (x for x in iterable if x is not None)


class dsu:
    def __init__(self, n: int):
        self.par = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]

    def union(self, x: int, y: int) -> bool:
        x, y = self.find(x), self.find(y)
        if x == y:
            return False
        if self.rank[x] < self.rank[y]:
            x, y = y, x
        if self.rank[x] == self.rank[y]:
            self.rank[x] += 1
        self.par[y] = x
        return True
