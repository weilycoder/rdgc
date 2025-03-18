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

    def test_output(self):
        graph = Graph(3)
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 0)
        self.assertEqual(
            graph.output(),
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
        self.assertNotEqual(graph.output(), shuffled.output())

    def test_count(self):
        graph = Graph(3, edge_count=True)
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
        self.assertEqual(graph.output(), "")
        self.assertEqual(graph.edges, 0)
        self.assertEqual(graph.vertices, 10)

    def test_complete(self):
        graph0 = Graph.complete(4)
        self.assertEqual(
            graph0.output(),
            "0 1\n" "0 2\n" "0 3\n" "1 2\n" "1 3\n" "2 3",
        )
        graph1 = Graph.complete(3, directed=True)
        self.assertEqual(
            graph1.output(),
            "0 1\n" "0 2\n" "1 0\n" "1 2\n" "2 0\n" "2 1",
        )
        graph2 = Graph.complete(3, directed=True, self_loop=True)
        self.assertEqual(
            graph2.output(),
            "0 0\n" "0 1\n" "0 2\n" "1 0\n" "1 1\n" "1 2\n" "2 0\n" "2 1\n" "2 2",
        )

    def test_tournament(self):
        for _ in range(10):
            N = random.randint(10, 50)
            graph = Graph.tournament(N)
            self.assertNotEqual(
                graph.output(),
                Graph.complete(N, directed=True).output(),
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
        graph = Graph.union_tree(N)
        self.assertEqual(graph.vertices, N)
        self.assertEqual(graph.edges, N - 1)
