# -*- coding: utf-8 -*-

import sys
import math
import functools

sys.path.append('../Heap')

from heap import Heap  # tested to be valid


@functools.total_ordering
class Vertex(object):

    def __init__(self, index=1, score=math.inf):
        self.index = index
        self.score = score
        self.edges = []

    def __eq__(self, other):
        return self.score == other.score

    def __lt__(self, other):
        return self.score < other.score

    def __hash__(self):
        return self.index


class Edge(object):

    def __init__(self, source, target, weight=0):
        self.source = source
        self.target = target
        self.weight = weight


class Graph(object):

    def __init__(self):
        self.lookup = {}
        self.vertices = set()

    def add_vertex(self, vertex):
        self.vertices.add(vertex)
        self.lookup[vertex.index] = vertex

    def dijkstra(self, s=1):
        heap = Heap()
        starter = self.lookup[s]
        starter.score = 0
        cloud = set([starter])

        def update(vertex):
            for edge in vertex.edges:
                target = edge.target
                if target in cloud:
                    continue

                def manipulator(v):
                    v.score = min(v.score, vertex.score + edge.weight)

                if target in heap.tree:
                    heap.manipulate(target, manipulator)
                else:
                    manipulator(target)
                    heap.push(target)

        update(starter)
        while not heap.is_empty():
            vertex = heap.pop()
            cloud.add(vertex)
            update(vertex)
