import sys
import warnings

from typing import *  # type: ignore


__all__ = ["Seq"]


Number = Union[int, float, complex]


class Seq:
    __formula: Callable[[int, Callable[[int], Number]], Number]
    __values: Dict[int, Number]

    class __SeqIter:
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

    def __init__(
        self,
        formula: Callable[[int, Callable[[int], Number]], Number],
        *values0: Number,
        values1: Dict[int, Number] = {},
    ):
        """
        Create a new sequence object.

        Args:
            formula: A function that takes an index and a function that returns the value of the sequence at that index and
                     returns the value of the sequence at that index.
            *values0: The initial values of the sequence.
            values1: The values of the sequence at the indices specified in this dictionary.
                     These values will overwrite the values in *values0.
        """
        self.__formula = formula
        self.__values = values1.copy()
        for i, v in enumerate(values0):
            if i in self.__values:
                warnings.warn(
                    f"Duplicate value for index {i}, overwriting with new value",
                    RuntimeWarning,
                    2,
                )
            else:
                self.__values[i] = v

    def __call__(self, i: int) -> Number:
        """
        Get the value of the sequence at index i.

        Args:
            i: The index of the value to get.

        Returns:
            The value of the sequence at index i.
        """
        if i not in self.__values:
            self.__values[i] = self.__formula(i, self)
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
        else:
            return self.__SeqIter(self, ind.start, ind.stop, ind.step)

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

    @staticmethod
    def setrecursionlimit(limit: int):
        sys.setrecursionlimit(limit)
