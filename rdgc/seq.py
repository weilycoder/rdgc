"""
This module provides two classes, `Seq` and `SeqIter`, for working with mathematical sequences.
"""

import sys
import warnings

from typing import Any, Callable, Dict, Generator, Iterable, Optional, Union, overload


__all__ = ["Seq", "SeqIter", "Number"]


Number = Any


class SeqIter:
    """
    An iterator class for iterating over a sequence (`Seq`) with customizable
    start, stop, and step parameters.
    Attributes:
        _seq (Seq): The sequence to iterate over.
        start (Optional[int]): The starting index of the iteration. Defaults to 0.
        stop (Optional[int]): The stopping index of the iteration. Defaults to infinity.
        step (Optional[int]): The step size for the iteration. Defaults to 1.
    Methods:
        __iter__(): Returns the iterator object itself.
        __next__(): Returns the next value in the sequence based on the current
                    position and step size. Raises StopIteration when the end
                    of the iteration is reached.
    """

    def __init__(
        self,
        _seq: "Seq",
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
    ):
        self.__seq = _seq
        self.__start = start if start is not None else 0
        self.__stop = stop if stop is not None else float("inf")
        self.__step = step if step is not None else 1
        self.__pos = self.__start

    def __iter__(self):
        return self

    def __next__(self) -> Number:
        if self.__step >= 0 and self.__pos >= self.__stop:
            raise StopIteration
        if self.__step < 0 and self.__pos <= self.__stop:
            raise StopIteration
        value = self.__seq[self.__pos]
        self.__pos += self.__step
        return value


class Seq:
    """
    Seq is a class that represents a mathematical sequence, allowing for the calculation of sequence values
    based on a provided formula or a generator function. The generator function is used to avoid recursion.
    """

    __type: bool
    __iter_limit: Optional[int]
    __formula: Callable[[int, Callable[[int], Number]], Number]
    __yield_formula: Callable[[int], Generator[int, Number, Number]]
    __values: Dict[int, Number]

    def __init__(
        self,
        formula: Optional[Callable[[int, Callable[[int], Number]], Number]] = None,
        *values0: Number,
        values1: Optional[Dict[int, Number]] = None,
        yield_formula: Optional[Callable[[int], Generator[int, Number, Number]]] = None,
        iteration_limit: Optional[int] = int(1e6),
    ):
        """
        Create a new sequence object.

        Args:
            formula (Callable[[int, Callable[[int], Number]], Number]): The formula to calculate the sequence.
                The first argument is the index of the value to calculate.
                The second argument is a function to get the value of the sequence at a given index.
            yield_formula (Callable[[int], Generator[int, Number, Number]]): The formula to calculate the sequence.
                The argument is the index of the value to calculate.
                The generator should yield the indices that need to be calculated.
            *values0 (Number): The initial values of the sequence.
            values1 (Dict[int, Number]): The values of the sequence at the indices specified in this dictionary.
                These values will overwrite the values in *values0.

        Raises:
            ValueError: If neither formula nor yield_formula is provided.
            ValueError: If both formula and yield_formula are provided.

        Warnings:
            RuntimeWarning: If a value is overwritten in the values1 dictionary.

        Examples:
            >>> s = Seq(lambda i, f: f(i - 1) + f(i - 2), 0, 1)
            >>> s[0]
            0
            >>> s[1]
            1
            >>> s[2]
            1
            >>> list(s[:10])
            [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
            >>> def fib_gen(i):
            ...     if i < 2:
            ...         return i
            ...     x = yield i - 1
            ...     y = yield i - 2
            ...     return (x + y) % 998244353
            ...
            >>> s = Seq(yield_formula=fib_gen)
            >>> s[100000]
            10519474
        """
        if formula is not None:
            self.__type = False
            self.__formula = formula
        if yield_formula is not None:
            self.__type = True
            self.__yield_formula = yield_formula
        if formula is None and yield_formula is None:
            raise ValueError("Either formula or yield_formula must be provided")
        if formula is not None and yield_formula is not None:
            raise ValueError("Only one of formula and yield_formula can be provided")
        self.__iter_limit = iteration_limit

        self.__values = {} if values1 is None else values1.copy()
        for i, v in enumerate(values0):
            if i in self.__values:
                warnings.warn(
                    f"Duplicate value for index {i}, overwriting with new value",
                    RuntimeWarning,
                    2,
                )
            else:
                self.__values[i] = v

    @property
    def has_yield_formula(self) -> bool:
        """Whether the sequence has a yield formula."""
        return self.__type

    @property
    def iter_limit(self) -> Optional[int]:
        """The iteration limit of the sequence."""
        return self.__iter_limit

    def __calc_with_yield(self, i: int, gen: Generator[int, Number, Number]) -> Number:
        stack = [(i, gen)]
        ret: Optional[Number] = None
        while True:
            try:
                if ret is not None:
                    cur = stack[-1][1].send(ret)
                    ret = None
                else:
                    cur = next(stack[-1][1])
                if cur in self.__values:
                    ret = self.__values[cur]
                else:
                    stack.append((cur, self.__yield_formula(cur)))
                    if self.__iter_limit is not None and len(stack) > self.__iter_limit:
                        raise RecursionError("Iteration limit exceeded")
            except StopIteration as stop:
                d, _ = stack.pop()
                ret = self.__values[d] = stop.value
                if not stack:
                    return stop.value

    def __call__(self, i: int) -> Number:
        """
        Get the value of the sequence at index i.

        Args:
            i: The index of the value to get.

        Returns:
            The value of the sequence at index i.
        """
        if i not in self.__values:
            if not self.has_yield_formula:
                self.__values[i] = self.__formula(i, self)
            else:
                self.__calc_with_yield(i, self.__yield_formula(i))
        return self.__values[i]

    @overload
    def __getitem__(self, ind: int) -> Number: ...

    @overload
    def __getitem__(self, ind: slice) -> Iterable[Number]: ...

    def __getitem__(self, ind: Union[int, slice]) -> Union[Number, Iterable[Number]]:
        """
        Get the value of the sequence at index i.

        Args:
            i: The index of the value to get.

        Returns:
            The value of the sequence at index i.
        """
        if isinstance(ind, int):
            return self.__call__(ind)
        return SeqIter(self, ind.start, ind.stop, ind.step)

    def __setitem__(self, i: int, value: Number):
        """
        Set the value of the sequence at index i.

        Args:
            i: The index of the value to set.
            value: The value to set the sequence at index i to.
        """
        self.__values[i] = value

    def __iter__(self):
        return iter(self.__values.items())

    def setiterationlimit(self, limit: Optional[int]):
        """
        Set the iteration limit of the sequence.

        Args:
            limit (Optional[int]): The new iteration limit. None for no limit.
        """
        self.__iter_limit = limit

    @staticmethod
    def getrecursionlimit() -> int:
        """
        Get the recursion limit of the sequence.

        Returns:
            The recursion limit of the sequence.
        """
        return sys.getrecursionlimit()

    @staticmethod
    def setrecursionlimit(limit: int):
        """
        Set the recursion limit of the sequence.

        Args:
            limit (int): The new recursion limit.
        """
        sys.setrecursionlimit(limit)
