# -*- coding: utf-8 -*-

import codecs
import unittest

from sccs import Graph


class TestSCC(unittest.TestCase):

    def setUp(self):
        self.answers = ['3,3,3', '3,3,2', '3,3,1,1', '7,1', '6,3,2,1']

    def reader(self, filename):
        file = codecs.open(filename, 'r', encoding='utf-8')
        raw = file.read().strip().replace('\r', '')
        data = [[int(v) for v in row.strip().split(' ')] for row in raw.split('\n')]
        file.close()
        return max([max(row) for row in data]), data

    def sizer(self, graph):
        sizes = [len(graph.components[leader]) for leader in graph.components]
        sizes = sorted(sizes, reverse=True)
        return ','.join([str(size) for size in sizes[:5]])

    def test_scc(self):
        for i in range(len(self.answers)):
            answer = self.answers[i]
            filename = 'cases/{}.txt'.format(i + 1)
            n, edges = self.reader(filename)
            graph = Graph(n=n, edges=edges)
            graph.scc()
            self.assertEqual(answer, self.sizer(graph))

    def test_coursera(self):
        n, edges = self.reader('data.txt')
        graph = Graph(n=n, edges=edges)
        graph.scc()
        self.assertEqual('434821,968,459,313,211', self.sizer(graph))


if __name__ == '__main__':
    unittest.main()
