"""Utility functions."""

import os
import sys
import random

from typing import *  # pylint: disable=W0401,W0614

__all__ = ["set_randseed_from_shell", "dos2unix", "dos2unix_file", "dos2unix_dir"]


T = TypeVar("T")


def set_randseed_from_shell() -> Tuple[Union[None, int, Tuple[int, ...]]]:
    """
    Set the random seed from the command line arguments.

    The random seed is set by hashing the command line arguments and using the hash as the seed.

    Returns:
        The random state.
    """
    random.seed(" ".join(sys.argv[1:]), version=2)
    return random.getstate()


def dos2unix(s: str) -> str:
    """
    Convert DOS line endings to Unix line endings.

    Args:
        s(str): The string to convert.

    Returns:
        The string with DOS line endings converted to Unix line endings.
    """
    return s.replace("\r\n", "\n")


def dos2unix_file(file: str) -> None:
    """
    Convert DOS line endings to Unix line endings in a file.

    Args:
        file(str): The file to convert.
    """
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
    suffixs: Tuple[str, ...] = ("in", "out"),
    echo: bool = False,
) -> None:
    """
    Convert DOS line endings to Unix line endings in a directory.

    Args:
        directory(str): The directory to convert (default: ".").
        recursive(bool): Whether to convert recursively (default: False).
        suffixs(Tuple[str]): The suffixes of the files to convert (default: ("in", "out")).
        echo(bool): Whether to print the files that are converted (default: False).
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(suffix) for suffix in suffixs):
                dos2unix_file(os.path.join(root, file))
                if echo:
                    print(f"Converted {ascii(file)} to Unix format.")
        if not recursive:
            break


def filter_none(iterable: Iterable[Optional[T]]) -> Iterable[T]:
    """Filter out the None values in an iterable."""
    return (x for x in iterable if x is not None)


class Dsu:
    """Disjoint Set Union data structure."""

    def __init__(self, n: int):
        self.par = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Finds the representative (root) of the set containing the element `x`."""
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]

    def union(self, x: int, y: int) -> bool:
        """Unions the sets containing the elements `x` and `y`."""
        x, y = self.find(x), self.find(y)
        if x == y:
            return False
        if self.rank[x] < self.rank[y]:
            x, y = y, x
        if self.rank[x] == self.rank[y]:
            self.rank[x] += 1
        self.par[y] = x
        return True
