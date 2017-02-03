# -*- coding: utf-8 -*-

import codecs


def reader(filename):
    data = []
    flag = {}
    with codecs.open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            value = int(line)
            if value not in flag:
                flag[value] = True
                data.append(value)
    return data


def search(a, left, right, value):
    if left >= right:
        return right
    pivot = int((right - left) / 2) + left
    if value < a[pivot]:
        return search(a, left, pivot - 1, value)
    elif a[pivot] < value:
        return search(a, pivot + 1, right, value)
    else:
        return pivot


def solve():
    m = 10000
    flag = {t: 0 for t in range(-m, m + 1)}
    a = reader('data.txt')
    a.sort()
    n = len(a)
    for i in range(n):
        left = max(search(a, 0, n - 1, -m - a[i]) - 1, 0)
        right = min(search(a, 0, n - 1, m - a[i]) + 1, n)
        if left < right:
            for j in range(left, right):
                t = a[i] + a[j]
                if i != j and t >= -m and t <= m:
                    flag[t] = 1
    print(sum(flag.values()))  # 427


if __name__ == '__main__':
    solve()
