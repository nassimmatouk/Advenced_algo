import numpy as np
from heapq import heappop, heappush

from helper.useful import generate_random_matrix


def prim_mst(adj):
    N = len(adj)
    selected = [False] * N
    mst_edges = []
    min_heap = [(0, 0, -1)]  # (cost, current_node, parent_node)

    while min_heap:
        cost, u, parent = heappop(min_heap)
        if selected[u]:
            continue
        selected[u] = True
        if parent != -1:
            mst_edges.append((parent, u, cost))

        for v in range(N):
            if not selected[v] and adj[u][v] != 0:
                heappush(min_heap, (adj[u][v], v, u))

    return mst_edges

def preorder_traversal(mst, start):
    from collections import defaultdict, deque

    graph = defaultdict(list)
    for u, v, cost in mst:
        graph[u].append(v)
        graph[v].append(u)

    visited = set()
    path = []

    def dfs(node):
        visited.add(node)
        path.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)

    dfs(start)
    return path

def tsp_mst_approximation(adj):
    mst = prim_mst(adj)
    path = preorder_traversal(mst, 0)
    path.append(path[0])  # return to the starting point
    return path

# Code principal
if __name__ == "__main__":

    adj = generate_random_matrix(num_cities=10, symmetric=True)

    path = tsp_mst_approximation(adj)
    print("Chemin approximatif du TSP (MST) :", path)