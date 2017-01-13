# -*- coding: utf-8 -*-

import codecs
import unittest

from dsp import Graph


class TestDijkstraShortestPath(unittest.TestCase):

    def setUp(self):
        self.alists = [[2, 3], [1, 3], [1, 2]]

    def reader(self):
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

    def test_coursera(self):
        n, matrix = self.reader()
        graph = Graph(n, matrix)
        graph.dijkstra(1)
        targets = [7, 37, 59, 82, 99, 115, 133, 165, 188, 197]
        answer = ','.join([str(graph.shortest[target]) for target in targets])
        self.assertEqual('2599,2610,2947,2052,2367,2399,2029,2442,2505,3068', answer)


if __name__ == '__main__':
    unittest.main()
