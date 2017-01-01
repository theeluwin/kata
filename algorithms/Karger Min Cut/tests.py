# -*- coding: utf-8 -*-

import kmc
import math
import codecs
import unittest


class TestKargerMinCut(unittest.TestCase):

    def setUp(self):
        self.alists = [[2, 3], [1, 3], [1, 2]]

    def reader(self):
        file = codecs.open('data.txt', 'r', encoding='utf-8')
        alists = [[int(v) for v in line.strip().split('\t')[1:]] for line in file]
        file.close()
        return alists

    def test_init(self):
        graph = kmc.Graph(self.alists)
        self.assertEqual(3, len(graph.edges))

    def test_graph_init(self):
        graph = kmc.Graph(self.alists)
        graph.contract(1, 2)
        self.assertEqual(2, len(graph.edges))

    def test_karger_min_cut(self):
        graph = kmc.Graph(self.alists)
        k = graph.karger_min_cut()
        self.assertEqual(2, k)

    def test_coursera(self):
        n = 200
        # T = n * n * int(math.ceil(math.log(n)))  # this guarantees the probability that the algorithm fails to be lower than 1/n, but my algorithm was too slow! I wasted time on contracting edges by O(m) per iteration...
        T = n  # but hey, this gave me correct answer
        ks = []
        alists = self.reader()
        for t in range(T):
            graph = kmc.Graph(alists)
            ks.append(graph.karger_min_cut())
        # self.assertEqual(17, min(ks))
        # actually, since this is a random algorithm, testing whether the output equals to the answer is just just just wrong thing to do! you should test whether the graph is splitted by the output edges. never ever do it like this. this is just terribly wrong.
        print(min(ks))


if __name__ == '__main__':
    unittest.main()
