# -*- coding: utf-8 -*-

import dsp
import dsph
import time
import codecs
import unittest


class TestDijkstraShortestPath(unittest.TestCase):

    def setUp(self):
        self.alists = [[2, 3], [1, 3], [1, 2]]

    def reader_matrix(self):
        n = 200
        matrix = [[1000000 for _ in range(n + 1)] for _ in range(n + 1)]
        for i in range(1, n + 1):
            matrix[i][i] = 0
        file = codecs.open('data.txt', 'r', encoding='utf-8')
        for line in file:
            line = line.strip().split('\t')
            i = int(line[0])
            for data in line[1:]:
                j, weight = [int(_) for _ in data.split(',')]
                matrix[i][j] = weight
                matrix[j][i] = weight
        file.close()
        return n, matrix

    def reader_graph(self):
        n = 200
        file = codecs.open('data.txt', 'r', encoding='utf-8')
        graph = dsph.Graph()
        for index in range(1, n + 1):
            vertex = dsph.Vertex(index)
            graph.add_vertex(vertex)
        for line in file:
            line = line.strip().split('\t')
            i = int(line[0])
            source = graph.lookup[i]
            for data in line[1:]:
                j, weight = [int(_) for _ in data.split(',')]
                target = graph.lookup[j]
                source.edges.append(dsph.Edge(source, target, weight))
                target.edges.append(dsph.Edge(target, source, weight))
        file.close()
        return graph

    def test_coursera_naive(self):
        n, matrix = self.reader_matrix()
        graph = dsp.Graph(n, matrix)
        start = time.time()
        graph.dijkstra(1)
        end = time.time()
        print("\nnaive: %.4f seconds\n" % (end - start))
        targets = [7, 37, 59, 82, 99, 115, 133, 165, 188, 197]
        answer = ','.join([str(graph.shortest[target]) for target in targets])
        self.assertEqual('2599,2610,2947,2052,2367,2399,2029,2442,2505,3068', answer)

    def test_coursera_heap(self):
        graph = self.reader_graph()
        start = time.time()
        graph.dijkstra(1)
        end = time.time()
        print("\nheap: %.4f seconds\n" % (end - start))
        targets = [7, 37, 59, 82, 99, 115, 133, 165, 188, 197]
        answer = ','.join([str(graph.lookup[target].score) for target in targets])
        self.assertEqual('2599,2610,2947,2052,2367,2399,2029,2442,2505,3068', answer)


if __name__ == '__main__':
    unittest.main()
