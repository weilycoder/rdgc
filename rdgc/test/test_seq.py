from typing import Callable
import unittest

from rdgc import Seq


class TestSeq(unittest.TestCase):
    def test_general(self):
        s1 = Seq(lambda x, f: x + 1)
        s2 = Seq(lambda x, f: x**2)
        for i in range(10):
            self.assertEqual(s1[i], i + 1)
            self.assertEqual(s2[i], i * i)

    def test_recursion(self):
        fib = [0, 1]
        for i in range(2, 10):
            fib.append(fib[i - 1] + fib[i - 2])
        s1 = Seq(lambda x, f: f(x - 1) + 1 if x > 0 else 0)
        s2 = Seq(lambda x, f: f(x - 1) + f(x - 2), 0, 1)
        for i in range(10):
            self.assertEqual(s1[i], i)
            self.assertEqual(s2[i], fib[i])

    def test_slice(self):
        formula: Callable[[int], int] = lambda x: x**3 - 2 * x**2 + 3 * x - 4
        seq = Seq(lambda x, f: formula(x))
        self.assertEqual(list(seq[0:5]), [formula(x) for x in range(5)])
        self.assertEqual(list(seq[2:10:3]), [formula(x) for x in range(2, 10, 3)])
        self.assertEqual(list(seq[10:2:-2]), [formula(x) for x in range(10, 2, -2)])
