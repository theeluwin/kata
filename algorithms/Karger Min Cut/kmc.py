# -*- coding: utf-8 -*-

import time
import random

random.seed(time.time())


class Edge(object):

    def __init__(self, vi, vt, index=0):
        # the index and o-prefixeds are just incase when you want to back-track which edges should be deleted from the origin
        self.index = index
        self.ovi = vi
        self.ovt = vt
        self.vi = vi
        self.vt = vt


class Graph(object):

    def __init__(self, alists=[]):
        edge_index = 1
        self.n = len(alists)
        self.alists = [None] + alists
        self.edges = []
        for i in range(1, self.n + 1):
            for j in self.alists[i]:
                if j > i:
                    self.edges.append(Edge(i, j, edge_index))
                    edge_index += 1

    def karger_min_cut(self):
        while self.n > 2:
            index = random.randrange(0, len(self.edges))
            edge = self.edges[index]
            vi = edge.vi
            vt = edge.vt
            self.edges = self.edges[:index] + self.edges[index + 1:]
            self.contract(vi, vt)
            self.n = self.n - 1
        return len(self.edges)

    def contract(self, nvi, nvt):
        for edge in self.edges:
            if edge.vi == nvi:
                edge.vi = nvt
            if edge.vt == nvi:
                edge.vt = nvt
        edges = []
        for edge in self.edges:
            if edge.vi != edge.vt:
                edges.append(edge)
        self.edges = edges
