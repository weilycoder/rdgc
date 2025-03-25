# pylint: disable=C0114, C0115, C0116, W0401, W0614

import unittest

from rdgc.nt import *


__all__ = ["TestNt"]


class TestNt(unittest.TestCase):
    def test_prime_sieve(self):
        self.assertEqual(prime_sieve(10), [2, 3, 5, 7])
        self.assertEqual(prime_sieve(11), [2, 3, 5, 7, 11])
        self.assertEqual(prime_sieve(1000), prime_sieve2(1000))

    def test_is_prime(self):
        ps = set(prime_sieve(1000))
        for n in range(1001):
            self.assertEqual(is_prime(n), n in ps)

    def test_miller_rabin(self):
        ps = set(prime_sieve(1000))
        for n in range(1001):
            self.assertEqual(miller_rabin(n), n in ps)

    def test_randprime(self):
        for _ in range(100):
            p = randprime(0, 1000)
            self.assertTrue(is_prime(p))
            self.assertLessEqual(p, 1000)
