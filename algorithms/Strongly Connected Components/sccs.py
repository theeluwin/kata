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
        for s in range(self.n, 0, -1):
            if self.rvisited[s]:
                continue
            self.rvisited[s] = True
            stack = [s]
            while len(stack):
                i = stack[-1]
                sink = True
                for j in self.rlists[i]:
                    if not self.rvisited[j]:
                        self.rvisited[j] = True
                        stack.append(j)
                        sink = False
                        break
                if sink:
                    self.t += 1
                    self.f[i] = self.t
                    stack.pop()
        for key in self.f:
            self.r[self.f[key]] = key
        for t in range(self.n, 0, -1):
            s = self.r[t]
            if self.avisited[s]:
                continue
            self.avisited[s] = True
            stack = [s]
            while len(stack):
                i = stack.pop()
                self.leader[i] = s
                self.components[s].append(i)
                for j in self.alists[i]:
                    if not self.avisited[j]:
                        self.avisited[j] = True
                        stack.append(j)
