"""RDGC Data Generator for Contests"""

from rdgc.nt import (
    prime_sieve,
    prime_sieve2,
    is_prime,
    miller_rabin_test,
    miller_rabin,
    fast_isPrime,
    nextprime,
    prevprime,
    randprime,
)
from rdgc.seq import Seq, SeqIter, Number
from rdgc.graph import Graph, GRAPH_GENERS, make_graph
from rdgc.utils import set_randseed_from_shell, dos2unix, dos2unix_file, dos2unix_dir


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
    "Seq",
    "SeqIter",
    "Number",
    "Graph",
    "make_graph",
    "GRAPH_GENERS",
    "set_randseed_from_shell",
    "dos2unix",
    "dos2unix_file",
    "dos2unix_dir",
]
