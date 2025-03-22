import random
import unittest
from rdgc import Graph
from rdgc.utils import dsu


__all__ = ["TestGraph"]


class TestGraph(unittest.TestCase):
    def assert_connected(self, graph: Graph):
        n = graph.vertices
        dsu_instance = dsu(n)
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
        graph = Graph.null(10)
        self.assertEqual(graph.output_edges(), "")
        self.assertEqual(graph.edges, 0)
        self.assertEqual(graph.vertices, 10)

    def test_complete(self):
        graph0 = Graph.complete(4)
        self.assertEqual(
            graph0.output_edges(),
            "0 1\n" "0 2\n" "0 3\n" "1 2\n" "1 3\n" "2 3",
        )
        graph1 = Graph.complete(3, directed=True)
        self.assertEqual(
            graph1.output_edges(),
            "0 1\n" "0 2\n" "1 0\n" "1 2\n" "2 0\n" "2 1",
        )
        graph2 = Graph.complete(3, directed=True, self_loop=True)
        self.assertEqual(
            graph2.output_edges(),
            "0 0\n" "0 1\n" "0 2\n" "1 0\n" "1 1\n" "1 2\n" "2 0\n" "2 1\n" "2 2",
        )

    def test_tournament(self):
        for _ in range(10):
            N = random.randint(10, 50)
            graph = Graph.tournament(N)
            self.assertNotEqual(
                graph.output_edges(),
                Graph.complete(N, directed=True).output_edges(),
            )
            self.assertEqual(
                [[u, v] for u in range(N) for v in range(u + 1, N)],
                sorted(sorted((u, v)) for u, v, _ in graph.get_edges()),
            )

    def test_tree(self):
        N = 100
        for _ in range(10):
            r1, r2 = sorted((random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)))
            r2 -= r1
            tree = Graph.tree(N, r1, r2)
            self.assertEqual(tree.vertices, N)
            self.assertEqual(tree.edges, N - 1)
            self.assert_connected(tree)

    def test_chain(self):
        N = 20
        graph = Graph.chain(N)
        self.assertEqual(graph.vertices, N)
        self.assertEqual(graph.edges, N - 1)
        for u, v, _ in graph.get_edges():
            self.assertEqual(u + 1, v)

    def test_star(self):
        N = 20
        graph = Graph.star(N)
        self.assertEqual(graph.vertices, N)
        self.assertEqual(graph.edges, N - 1)
        for u, v, _ in graph.get_edges():
            self.assertEqual(u, 0)
            self.assertNotEqual(v, 0)
        self.assertEqual(graph.count_edges(0), N - 1)

    def test_union_tree(self):
        N = 20
        graph = Graph.spanning_tree(N)
        self.assertEqual(graph.vertices, N)
        self.assertEqual(graph.edges, N - 1)
        self.assert_connected(graph)

    def test_self_loop(self):
        N = 20
        for _ in range(4):
            graph = Graph.random(N, N * N // 4, self_loop=True)
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
            graph = Graph.random(N, N * N, multiedge=True)
            self.assertEqual(graph.vertices, N)
            self.assertEqual(graph.edges, N * N)
            self.assertTrue(len(set(graph.get_edges())) < graph.edges)

    def test_random(self):
        N = 20
        # Test graph with too many edges
        Graph.random(
            N,
            Graph._calc_max_edge(N, True, False),  # type: ignore
            directed=True,
            self_loop=False,
        )
        with self.assertRaises(ValueError):
            Graph.random(N, N * N, directed=True)

    def test_connected(self):
        N = 20
        graph = Graph.connected(N, N * N // 4)
        self.assertEqual(graph.vertices, N)
        self.assertEqual(graph.edges, N * N // 4)
        self.assert_connected(graph)
        for _ in range(10):
            graph = Graph.connected(N, N + 1, directed=True)
            self.assertEqual(graph.vertices, N)
            self.assertEqual(graph.edges, N + 1)
            self.assert_connected(graph)
        # Test graph with too many edges
        graph = Graph.connected(
            N,
            Graph._calc_max_edge(N, False, False),  # type: ignore
            directed=True,
            self_loop=False,
        )
        for u, v, _ in graph.get_edges():
            self.assertTrue(u < v)
        with self.assertRaises(ValueError):
            Graph.connected(N, N * N, directed=True)
        # Test graph with too few edges
        Graph.connected(N, N - 1)
        with self.assertRaises(ValueError):
            Graph.connected(N, N - 2)

    def test_cycle(self):
        N = 20
        for _ in range(10):
            graph = Graph.cycle(N)
            self.assertEqual(graph.vertices, N)
            self.assertEqual(graph.edges, N)
            for u in range(N):
                self.assertEqual(graph.count_edge(u, (u + 1) % N), 1)

    def test_wheel(self):
        N = 20
        for _ in range(10):
            graph = Graph.wheel(N)
            self.assertEqual(graph.vertices, N)
            self.assertEqual(graph.edges, 2 * N - 2)
            self.assertEqual(graph.count_edges(0), N - 1)
            for u in range(1, N):
                self.assertEqual(graph.count_edge(0, u), 1)
                self.assertEqual(graph.count_edge(u, u % (N - 1) + 1), 1)

    def test_degree_unmutiedge(self):
        N = 20
        for _ in range(40):
            graph = Graph.from_degree_sequence([2] * N, multiedge=False)
            self.assertEqual(graph.vertices, N)
            self.assertEqual(graph.edges, N)
            for u in range(N):
                self.assertEqual(graph.count_edges(u), 2)
            self.assertEqual(len(set(graph.get_edges())), N)

    def test_degree_mutiedge(self):
        N = 20
        muti = False
        for _ in range(40):
            graph = Graph.from_degree_sequence([2] * N, multiedge=True)
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
            graph = Graph.from_degree_sequence([2] * N, self_loop=True)
            self.assertEqual(graph.vertices, N)
            self.assertEqual(graph.edges, N)
            for u in range(N):
                self.assertEqual(graph.degree(u), 2)
        for _ in range(20):
            graph = Graph.from_degree_sequence([20] * N, self_loop=True)
            self.assertEqual(graph.vertices, N)
            self.assertEqual(graph.edges, N * 10)
            for u in range(N):
                self.assertEqual(graph.degree(u), 20)
