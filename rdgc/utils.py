import os
import sys
import random

from typing import *  # type: ignore

__all__ = ["set_randseed_from_shell", "dos2unix", "dos2unix_file", "dos2unix_dir"]


T = TypeVar("T")


def set_randseed_from_shell() -> Tuple[Union[None, int, Tuple[int, ...]]]:
    random.seed(" ".join(sys.argv[1:]), version=2)
    return random.getstate()


def dos2unix(s: str) -> str:
    return s.replace("\r\n", "\n")


def dos2unix_file(file: str) -> None:
    data: List[str] = []

    with open(file, "r", encoding="ascii") as f:
        for line in f:
            data.append(line.strip())

    while data and not data[-1]:
        data.pop()

    with open(file, "w", encoding="ascii", newline="\n") as f:
        for line in data:
            print(line, file=f)


def dos2unix_dir(
    directory: str = ".",
    recursive: bool = False,
    *,
    suffixs: List[str] = ["in", "out"],
    echo: bool = False,
) -> None:
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(suffix) for suffix in suffixs):
                dos2unix_file(os.path.join(root, file))
                if echo:
                    print(f"Converted {ascii(file)} to Unix format.")
        if not recursive:
            break


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
