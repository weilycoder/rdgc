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
