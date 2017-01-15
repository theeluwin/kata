# -*- coding: utf-8 -*-


class Heap(object):

    def __init__(self, initial=[]):
        self.tree = []
        self.lookup = {}
        for value in initial:
            self.push(value)

    def is_empty(self):
        return not bool(len(self.tree))

    def left(self, index):
        return 2 * index + 1

    def right(self, index):
        return 2 * index + 2

    def parent(self, index):
        return int((index - 1) / 2)

    def swap(self, index1, index2):
        value1 = self.tree[index1]
        value2 = self.tree[index2]
        self.lookup[value1] = index2
        self.lookup[value2] = index1
        self.tree[index1] = value2
        self.tree[index2] = value1

    def up(self, index):
        parent = self.parent(index)
        if self.tree[index] < self.tree[parent]:
            self.swap(parent, index)
            return self.up(parent)

    def down(self, index):
        size = len(self.tree)
        left = self.left(index)
        right = self.right(index)
        if left < size:
            if right < size:
                if self.tree[left] < self.tree[index] or self.tree[right] < self.tree[index]:
                    if self.tree[left] < self.tree[right]:
                        self.swap(left, index)
                        return self.down(left)
                    else:
                        self.swap(right, index)
                        return self.down(right)
            else:
                if self.tree[left] < self.tree[index]:
                    self.swap(left, index)
                    return self.down(left)

    def manipulate(self, value, manipulator):
        index = self.lookup[value]
        manipulator(self.tree[index])
        self.down(index)
        self.up(index)

    def delete(self, index):
        value = self.tree[index]
        del self.lookup[value]
        last = len(self.tree) - 1
        self.swap(last, index)
        self.tree.pop()
        self.down(index)
        return value

    def push(self, value):
        self.tree.append(value)
        index = len(self.tree) - 1
        self.lookup[value] = index
        while True:
            if index == 0:
                return
            parent = self.parent(index)
            if self.tree[parent] > self.tree[index]:
                self.swap(parent, index)
                index = parent
            else:
                break

    def pop(self):
        return self.delete(0)
