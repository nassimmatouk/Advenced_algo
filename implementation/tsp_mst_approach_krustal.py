import numpy as np

from helper.useful import generate_random_matrix


class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

def kruskal_mst(adj):
    N = len(adj)
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i][j] != 0:
                edges.append((adj[i][j], i, j))
    edges.sort()

    ds = DisjointSet(N)
    mst_edges = []

    for edge in edges:
        weight, u, v = edge
        if ds.find(u) != ds.find(v):
            ds.union(u, v)
            mst_edges.append((u, v, weight))

    return mst_edges

def preorder_traversal(mst, start):
    from collections import defaultdict

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
    mst = kruskal_mst(adj)
    path = preorder_traversal(mst, 0)
    path.append(path[0])  # return to the starting point
    return path

# Code principal
if __name__ == "__main__":

    adj = generate_random_matrix(num_cities=10, symmetric=True)
    path = tsp_mst_approximation(adj)
    print("Chemin approximatif du TSP (MST avec Kruskal) :", path)