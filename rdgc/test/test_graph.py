import unittest
from rdgc import Graph


__all__ = ["TestGraph"]


class TestGraph(unittest.TestCase):
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
        graph = Graph.tournament(5)
        self.assertEqual(
            [[u, v] for u in range(5) for v in range(u + 1, 5)],
            sorted(sorted((u, v)) for u, v, _ in graph.get_edges()),
        )
