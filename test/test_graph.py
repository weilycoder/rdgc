# pylint: disable=all

import itertools
import random as rd
import unittest
from rdgc import Graph
from rdgc.graph import *
from rdgc.utils import Dsu


__all__ = ["TestGraph"]


class CallCounter:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)


class TestGraph(unittest.TestCase):
    def assert_connected(self, graph: Graph):
        n = graph.vertices
        dsu_instance = Dsu(n)
        for u, v, _ in graph.get_edges():
            dsu_instance.union(u, v)
        self.assertEqual(len(set(dsu_instance.find(i) for i in range(n))), 1)

    def test_output_node(self):
        graph = Graph(3, rnk=[3, 4, 5])
        self.assertEqual(
            graph.output_nodes(sep=", "),
            "3, 4, 5",
        )

    def test_output_edge(self):
        graph = Graph(3)
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 0)
        self.assertEqual(
            graph.output_edges(),
            "0 1\n" "0 2\n" "1 2",
        )

    def test_shuffle(self):
        graph = Graph(10)
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 0)
        graph.add_edge(3, 4)
        graph.add_edge(9, 5)
        shuffled = graph.shuffle_nodes()
        self.assertNotEqual(graph.output_edges(), shuffled.output_edges())

    def test_count(self):
        graph = Graph(3)
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 0)
        self.assertEqual(graph.count_edges(0), 2)
        self.assertEqual(graph.count_edges(1), 2)
        self.assertEqual(graph.count_edge(0, 1), 1)
        self.assertEqual(graph.edges, 3)
        self.assertEqual(graph.vertices, 3)

    def test_null(self):
        graph = null(10)
        self.assertEqual(graph.output_edges(), "")
        self.assertEqual(graph.edges, 0)
        self.assertEqual(graph.vertices, 10)

    def test_complete(self):
        graph0 = complete(4)
        self.assertEqual(
            graph0.output_edges(),
            "0 1\n" "0 2\n" "0 3\n" "1 2\n" "1 3\n" "2 3",
        )
        graph1 = complete(3, directed=True)
        self.assertEqual(
            graph1.output_edges(),
            "0 1\n" "0 2\n" "1 0\n" "1 2\n" "2 0\n" "2 1",
        )
        graph2 = complete(3, directed=True, self_loop=True)
        self.assertEqual(
            graph2.output_edges(),
            "0 0\n" "0 1\n" "0 2\n" "1 0\n" "1 1\n" "1 2\n" "2 0\n" "2 1\n" "2 2",
        )

    def test_tournament(self):
        for _ in range(10):
            N = rd.randint(10, 50)
            graph = tournament(N)
            self.assertNotEqual(
                graph.output_edges(),
                complete(N, directed=True).output_edges(),
            )
            self.assertEqual(
                [[u, v] for u in range(N) for v in range(u + 1, N)],
                sorted(sorted((u, v)) for u, v, _ in graph.get_edges()),
            )

    def test_tree(self):
        N = 100
        for _ in range(10):
            r1, r2 = sorted((rd.uniform(0.0, 1.0), rd.uniform(0.0, 1.0)))
            r2 -= r1
            tg = tree(N, r1, r2)
            self.assertEqual(tg.vertices, N)
            self.assertEqual(tg.edges, N - 1)
            self.assert_connected(tg)

    def test_chain(self):
        N = 20
        graph = chain(N)
        self.assertEqual(graph.vertices, N)
        self.assertEqual(graph.edges, N - 1)
        for u, v, _ in graph.get_edges():
            self.assertEqual(u + 1, v)

    def test_star(self):
        N = 20
        graph = star(N)
        self.assertEqual(graph.vertices, N)
        self.assertEqual(graph.edges, N - 1)
        for u, v, _ in graph.get_edges():
            self.assertEqual(u, 0)
            self.assertNotEqual(v, 0)
        self.assertEqual(graph.count_edges(0), N - 1)

    def test_union_tree(self):
        N = 20
        graph = spanning_tree(N)
        self.assertEqual(graph.vertices, N)
        self.assertEqual(graph.edges, N - 1)
        self.assert_connected(graph)

    def test_self_loop(self):
        N = 20
        for _ in range(4):
            graph = random_graph(N, N * N // 4, self_loop=True)
            flag = False
            for u in range(N):
                if graph.count_edge(u, u) > 0:
                    flag = True
            if flag:
                break
        else:
            self.fail("No loops")

    def test_mutiedges(self):
        N = 20
        for _ in range(4):
            graph = random_graph(N, N * N, multiedge=True)
            self.assertEqual(graph.vertices, N)
            self.assertEqual(graph.edges, N * N)
            self.assertTrue(len(set(graph.get_edges())) < graph.edges)

    def test_random_too_many_edges(self):
        N = 20

        def _test_call_count(e: int):
            gener = CallCounter(lambda u, v: None)
            random_graph(
                N,
                e,
                directed=True,
                self_loop=False,
                weight_gener=gener,
            )
            self.assertEqual(gener.count, e)

        M = Graph.calc_max_edge(N, True, False)
        for ee in [M, M - 1, M // 2, M // 3, M // 3 * 2]:
            _test_call_count(ee)
        with self.assertRaises(ValueError):
            _test_call_count(M + 1)

    def test_connected(self):
        N = 20
        graph = connected(N, N * N // 4)
        self.assertEqual(graph.vertices, N)
        self.assertEqual(graph.edges, N * N // 4)
        self.assert_connected(graph)
        for _ in range(10):
            graph = connected(N, N + 1, directed=True)
            self.assertEqual(graph.vertices, N)
            self.assertEqual(graph.edges, N + 1)
            self.assert_connected(graph)
        # Test graph with too many edges
        graph = connected(
            N,
            Graph.calc_max_edge(N, False, False),  # type: ignore
            directed=True,
            self_loop=False,
        )
        for u, v, _ in graph.get_edges():
            self.assertTrue(u < v)
        with self.assertRaises(ValueError):
            connected(N, N * N, directed=True)
        # Test graph with too few edges
        connected(N, N - 1)
        with self.assertRaises(ValueError):
            connected(N, N - 2)

    def test_cycle(self):
        N = 20
        for _ in range(10):
            graph = cycle(N)
            self.assertEqual(graph.vertices, N)
            self.assertEqual(graph.edges, N)
            for u in range(N):
                self.assertEqual(graph.count_edge(u, (u + 1) % N), 1)

    def test_wheel(self):
        N = 20
        for _ in range(10):
            graph = wheel(N)
            self.assertEqual(graph.vertices, N)
            self.assertEqual(graph.edges, 2 * N - 2)
            self.assertEqual(graph.count_edges(0), N - 1)
            for u in range(1, N):
                self.assertEqual(graph.count_edge(0, u), 1)
                self.assertEqual(graph.count_edge(u, u % (N - 1) + 1), 1)

    def test_lattice_2D(self):
        for _ in range(10):
            n = rd.randint(2, 10)
            graph = lattice([n, n], circular=False)
            self.assertEqual(graph.vertices, n * n)
            self.assertEqual(graph.edges, 2 * n * (n - 1))
            for u, v in itertools.product(range(n), repeat=2):
                if u < n - 1:
                    self.assertEqual(graph.count_edge(u * n + v, (u + 1) * n + v), 1)
                if v < n - 1:
                    self.assertEqual(graph.count_edge(u * n + v, u * n + v + 1), 1)

        for _ in range(10):
            n = rd.randint(3, 10)
            graph = lattice([n, n], circular=True)
            self.assertEqual(graph.vertices, n * n)
            self.assertEqual(graph.edges, 2 * n * n)
            for u, v in itertools.product(range(n), repeat=2):
                if u < n - 1:
                    self.assertEqual(graph.count_edge(u * n + v, (u + 1) * n + v), 1)
                else:
                    self.assertEqual(graph.count_edge(u * n + v, v), 1)
                if v < n - 1:
                    self.assertEqual(graph.count_edge(u * n + v, u * n + v + 1), 1)
                else:
                    self.assertEqual(graph.count_edge(u * n + v, u * n), 1)

    def test_lattice_3D(self):
        for _ in range(10):
            n = rd.randint(2, 5)
            graph = lattice([n, n, n], circular=False)
            self.assertEqual(graph.vertices, n * n * n)
            self.assertEqual(graph.edges, 3 * n * n * (n - 1))
            for u, v, w in itertools.product(range(n), repeat=3):
                if u < n - 1:
                    self.assertEqual(graph.count_edge(u * n * n + v * n + w, (u + 1) * n * n + v * n + w), 1)
                if v < n - 1:
                    self.assertEqual(graph.count_edge(u * n * n + v * n + w, u * n * n + (v + 1) * n + w), 1)
                if w < n - 1:
                    self.assertEqual(graph.count_edge(u * n * n + v * n + w, u * n * n + v * n + (w + 1)), 1)

    def test_union(self):
        N = 4
        graph1 = complete(N)
        graph2 = chain(N)
        graph = union(graph1, graph2, default_mapping="connect")
        self.assertEqual(graph.vertices, 2 * N - 1)
        self.assertEqual(graph.edges, graph1.edges + graph2.edges)
        self.assertEqual(
            graph.output_edges(),
            "0 1\n" "0 2\n" "0 3\n" "1 2\n" "1 3\n" "2 3\n" "3 4\n" "4 5\n" "5 6",
        )

    def test_complement_undirected(self):
        N = 6
        for _ in range(10):
            graph = random_graph(N, 7)
            comp = complement(graph)

            self.assertEqual(comp.vertices, N)
            for u in range(N):
                self.assertEqual(comp.count_edge(u, u), 0)
                for v in range(u + 1, N):
                    self.assertEqual(graph.count_edge(u, v) + comp.count_edge(u, v), 1)

    def test_complement_directed(self):
        N = 5
        for _ in range(10):
            graph = random_graph(N, 8, directed=True)
            comp = complement(graph)

            self.assertEqual(comp.vertices, N)
            for u in range(N):
                self.assertEqual(comp.count_edge(u, u), 0)
                for v in range(N):
                    if u != v:
                        self.assertEqual(graph.count_edge(u, v) + comp.count_edge(u, v), 1)

    def test_complement_self_loop_and_rank(self):
        graph = Graph(4, directed=True, rnk=[10, 20, 30, 40])
        graph.add_edge(0, 1)
        graph.add_edge(2, 2)

        comp = complement(graph, self_loop=True)

        self.assertEqual(comp.get_ranks(), graph.get_ranks())
        for u in range(graph.vertices):
            for v in range(graph.vertices):
                self.assertEqual(graph.count_edge(u, v) + comp.count_edge(u, v), 1)

    def test_degree_unmutiedge(self):
        N = 20
        for _ in range(40):
            graph = from_degree_sequence([2] * N, multiedge=False)
            self.assertEqual(graph.vertices, N)
            self.assertEqual(graph.edges, N)
            for u in range(N):
                self.assertEqual(graph.count_edges(u), 2)
            self.assertEqual(len(set(graph.get_edges())), N)

    def test_degree_mutiedge(self):
        N = 20
        muti = False
        for _ in range(40):
            graph = from_degree_sequence([2] * N, multiedge=True)
            self.assertEqual(graph.vertices, N)
            self.assertEqual(graph.edges, N)
            for u in range(N):
                self.assertEqual(graph.count_edges(u), 2)
            if len(set(graph.get_edges())) < N:
                muti = True
        self.assertTrue(muti)

    def test_degree_loop(self):
        N = 20
        for _ in range(20):
            graph = from_degree_sequence([2] * N, self_loop=True)
            self.assertEqual(graph.vertices, N)
            self.assertEqual(graph.edges, N)
            for u in range(N):
                self.assertEqual(graph.degree(u), 2)
        for _ in range(20):
            graph = from_degree_sequence([20] * N, self_loop=True)
            self.assertEqual(graph.vertices, N)
            self.assertEqual(graph.edges, N * 10)
            for u in range(N):
                self.assertEqual(graph.degree(u), 20)
