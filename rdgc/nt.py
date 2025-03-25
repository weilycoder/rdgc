"""
This module provides some commonly used number theory methods.
"""

import random

from typing import List, Tuple


__all__ = [
    "prime_sieve",
    "prime_sieve2",
    "is_prime",
    "miller_rabin_test",
    "miller_rabin",
    "fast_isPrime",
    "nextprime",
    "prevprime",
    "randprime",
]


def prime_sieve(n: int) -> List[int]:
    """
    Return a list of prime numbers less than or equal to n.

    The time complexity is O(n * log(log(n))).

    However, it is usually faster than the O(n) version.

    If n is float, it will be converted to int.

    Args:
        n(int): The upper bound.

    Returns:
        List[int]: A list of prime numbers less than or equal to n.
    """
    n = int(n)
    vis = [True] * (n + 1)
    for i in range(2, int(n**0.5) + 1):
        if vis[i]:
            for j in range(i * i, n + 1, i):
                vis[j] = False
    return [i for i in range(2, n + 1) if vis[i]]


def prime_sieve2(n: int) -> List[int]:
    """
    Return a list of prime numbers less than or equal to n.

    The time complexity is O(n).

    If n is float, it will be converted to int.

    Args:
        n(int): The upper bound.

    Returns:
        List[int]: A list of prime numbers less than or equal to n.
    """
    n = int(n)
    vis: List[bool] = [True] * (n + 1)
    prime: List[int] = []
    for i in range(2, n + 1):
        if vis[i]:
            prime.append(i)
        for p in prime:
            if p * i > n:
                break
            vis[p * i] = False
            if i % p == 0:
                break
    return prime


def is_prime(n: int) -> bool:
    """
    Check if n is a prime number.

    The time complexity is O(sqrt(n)).

    Args:
        n(int): The number to be checked.

    Returns:
        bool: True if n is a prime number, False otherwise.
    """
    if not isinstance(n, int):
        raise ValueError("n must be an integer")
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def miller_rabin_test(n: int, r: int, d: int, a: int) -> bool:
    """
    Perform the Miller-Rabin primality test for a given base.
    The Miller-Rabin test is a probabilistic algorithm used to determine
    whether a number is a probable prime. This function tests the primality
    of a number `n` using a specific base `a`.

    Args:
        n (int): The number to test for primality.
        r (int): The number of times `n - 1` can be divided by 2 (exponent of 2 in `n - 1`).
        d (int): The odd part of `n - 1` such that `n - 1 = 2^r * d`.
        a (int): The base to use for the test.

    Returns:
        bool: True if `n` passes the Miller-Rabin test for the given base `a`,
              indicating that `n` is a probable prime. False otherwise.
    """
    x = pow(a, d, n)
    if x in (0, 1, n - 1):
        return True
    for _ in range(r - 1):
        x = x * x % n
        if x == n - 1:
            return True
    return False


def miller_rabin(n: int, k: int = 16, *, bases: Tuple[int, ...] = ()) -> bool:
    """
    Check if a number is a prime using the Miller-Rabin primality test.

    The Miller-Rabin algorithm is a probabilistic test to determine if a number is a prime.
    It is efficient and widely used for large numbers.

    Note that the algorithm is probabilistic, meaning that it can return false positives;
    however, the probability of a false positive can be reduced by increasing the number of iterations `k`.

    The algorithm has a time complexity of O(k * log(n)). For most applications, `k = 16` is sufficient.

    Args:
        n (int): The number to be tested for primality.
        k (int, optional): The number of iterations to perform. Higher values of `k`
            increase the accuracy of the test. Defaults to 16.

    Returns:
        bool: Returns `True` if the number `n` is likely a prime, and `False` otherwise.

    Raises:
        ValueError: If `n` is not an integer.
    """
    if not isinstance(n, int):
        raise ValueError("n must be an integer")
    if n < 2:
        return False
    if n in [2, 3, 5, 7, 11, 13, 17, 19]:
        return True
    if any(n % p == 0 for p in [2, 3, 5, 7, 11, 13, 17, 19]):
        return False
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for b in bases:
        if not miller_rabin_test(n, r, d, b):
            return False
    for _ in range(k - len(bases)):
        if not miller_rabin_test(n, r, d, random.randint(2, n - 2)):
            return False
    return True


fast_isPrime = miller_rabin


def nextprime(n: int) -> int:
    """
    Find the next prime number greater than or equal to the given integer `n`.

    Notes:
        - If `n` is less than 2, the function returns 2 as the smallest prime number.
        - The function uses the Miller-Rabin primality test to check for prime numbers.
        - The algorithm optimizes the search by leveraging properties of numbers
          in the 6k ± 1 form, where k is an integer.

    Args:
        n (int): The starting integer to find the next prime number.

    Returns:
        int: The next prime number greater than or equal to `n`.
    """
    n = int(n)
    if n < 2:
        return 2
    if n == 2:
        return 3
    if n % 6 == 0:
        if miller_rabin(n + 1):
            return n + 1
        n += 6
    elif 1 <= n % 6 < 5:
        n += 6 - n % 6
    else:
        if miller_rabin(n + 2):
            return n + 2
        n += 7
    while True:
        if miller_rabin(n - 1):
            return n - 1
        if miller_rabin(n + 1):
            return n + 1
        n += 6


def prevprime(n: int, raise_error: bool = True) -> int:
    """
    Find the largest prime number less than the given integer `n`.

    Notes:
        - The function uses the Miller-Rabin primality test to check for prime numbers.
        - The algorithm optimizes the search by leveraging properties of numbers
          in the 6k ± 1 form, where k is an integer.

    Args:
        n (int): The integer to find the previous prime number for.
        raise_error (bool, optional): If True, raises a ValueError when no prime
            number exists below `n`. Defaults to True.

    Returns:
        int: The largest prime number less than `n`. If no such prime exists
            and `raise_error` is False, returns 0.

    Raises:
        ValueError: If `n` is less than or equal to 2 and `raise_error` is True.
    """
    n = int(n)
    if n <= 2:
        if raise_error:
            raise ValueError(f"No prime number less than {n}")
        return 0
    if n == 3:
        return 2
    if n <= 5:
        return 3
    if n % 6 == 0:
        if miller_rabin(n - 1):
            return n - 1
        n -= 6
    elif n % 6 == 1:
        if miller_rabin(n - 2):
            return n - 2
        n -= 7
    else:
        n -= n % 6
    while True:
        if miller_rabin(n + 1):
            return n + 1
        if miller_rabin(n - 1):
            return n - 1
        n -= 6


def randprime(a: int, b: int, raise_error: bool = True) -> int:
    """
    Generate a random prime number within the specified range [a, b].

    This function attempts to find a random prime number between the integers
    `a` and `b` (inclusive). If no prime number exists in the range, it either
    raises a ValueError or returns 0, depending on the `raise_error` flag.

    Note: This function cannot uniformly randomly select from all primes in the range.

    Args:
        a(int): The lower bound of the range (inclusive).
        b(int): The upper bound of the range (inclusive).
        raise_error (bool, optional): If True, raises a ValueError when no prime
            number is found in the range. If False, returns 0 in such cases.
            Defaults to True.

    Returns:
        int: A prime number within the range [a, b], or 0 if no prime is found
            and `raise_error` is set to False.

    Raises:
        ValueError: If no prime number exists in the range [a, b] and
            `raise_error` is True.
    """
    st = random.randint(a, b)
    if miller_rabin(st):
        return st
    nxt = nextprime(st)
    if nxt <= b:
        return nxt
    pre = prevprime(st)
    if pre >= a:
        return pre
    if raise_error:
        raise ValueError(f"No prime number in the range [{a}, {b}]")
    return 0
