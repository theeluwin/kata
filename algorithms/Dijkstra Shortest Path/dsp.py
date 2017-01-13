# -*- coding: utf-8 -*-


class Graph(object):

    def __init__(self, n=0, matrix=[]):
        self.n = n
        self.matrix = matrix

    def dijkstra(self, s=1):
        self.shortest = [1000000 for _ in range(self.n + 1)]
        self.shortest[s] = 0
        cloud = set([s])
        while True:
            greed = 1000000
            darker = None
            for dark in cloud:
                for i in range(1, self.n + 1):
                    if i not in cloud and self.shortest[dark] + self.matrix[dark][i] < greed:
                        greed = self.shortest[dark] + self.matrix[dark][i]
                        darker = i
            if darker:
                cloud.add(darker)
                self.shortest[darker] = greed
            else:
                break
