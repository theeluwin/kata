# -*- coding: utf-8 -*-

from collections import defaultdict


class Graph(object):

    def __init__(self, n=1, edges=[]):
        self.n = n
        self.alists = defaultdict(lambda: [])
        self.rlists = defaultdict(lambda: [])
        self.avisited = defaultdict(lambda: False)
        self.rvisited = defaultdict(lambda: False)
        self.leader = defaultdict(lambda: None)
        self.components = defaultdict(lambda: [])
        self.f = {}
        self.r = {}
        for u, v in edges:
            self.alists[u].append(v)
            self.rlists[v].append(u)

    def scc(self):
        self.t = 0
        for i in range(self.n, 0, -1):
            if not self.rvisited[i]:
                self.rdfs(i)
        for key in self.f:
            self.r[self.f[key]] = key
        self.s = None
        for i in range(self.n, 0, -1):
            if not self.avisited[self.r[i]]:
                self.s = self.r[i]
                self.adfs(self.r[i])

    def rdfs(self, i):
        self.rvisited[i] = True
        for j in self.rlists[i]:
            if not self.rvisited[j]:
                self.rdfs(j)
        self.t += 1
        self.f[i] = self.t

    def adfs(self, i):
        self.avisited[i] = True
        self.leader[i] = self.s
        self.components[self.s].append(i)
        for j in self.alists[i]:
            if not self.avisited[j]:
                self.adfs(j)
