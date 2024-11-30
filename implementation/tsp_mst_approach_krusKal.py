import time
import numpy as np
from helper.useful import generate_random_matrix

# Class to implement Disjoint Sets (Union-Find)
class DisjointSet:
    def __init__(self, n):
        # Initialization: each element is its own parent
        self.parent = list(range(n))
        # Rank array to optimize union operations
        self.rank = [0] * n

    def find(self, u):
        # Find the representative of the set containing u (with path compression)
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])  # Optimization by attaching directly to the root parent
        return self.parent[u]

    def union(self, u, v):
        # Union the sets containing u and v
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            # Attach the tree with the smaller rank to the tree with the larger rank
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

# Kruskal's Algorithm to construct the MST
def kruskal_mst(adj):
    N = len(adj)
    edges = []
    # Collect all edges with their weights
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i][j] != 0:
                edges.append((adj[i][j], i, j))
    edges.sort()  # Sort edges by ascending weight

    ds = DisjointSet(N)  # Initialize Union-Find
    mst_edges = []

    # Iterate through sorted edges
    for edge in edges:
        weight, u, v = edge
        if ds.find(u) != ds.find(v):  # If u and v are not in the same set
            ds.union(u, v)  # Union the sets
            mst_edges.append((u, v, weight))  # Add edge to MST

    return mst_edges

# Preorder traversal of the MST to get a path
def preorder_traversal(mst, start):
    from collections import defaultdict

    # Build an undirected graph from the MST
    graph = defaultdict(list)
    for u, v, cost in mst:
        graph[u].append(v)
        graph[v].append(u)

    visited = set()  # Set of visited nodes
    path = []

    def dfs(node):
        # Depth First Search (DFS) to build the path in preorder
        visited.add(node)
        path.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)

    dfs(start)
    return path

# Approximation of TSP using Minimum Spanning Tree
def tsp_mst_kruskal(adj):
    mst = kruskal_mst(adj)  # Construct the MST with Kruskal
    path = preorder_traversal(mst, 0)  # Preorder traversal of the MST
    path.append(path[0])  # Return to the start point to form a cycle
    cost = sum(adj[path[i]][path[i + 1]] for i in range(len(path) - 1))  # Calculate the total cost
    return cost, path

# Function to measure execution time of the TSP approximation
def measure_execution_time_mst_kruskal(distances):
    start_time = time.time()
    cost, path = tsp_mst_kruskal(distances)  # Run TSP approximation
    end_time = time.time()
    execution_time = end_time - start_time
    return cost, path, execution_time

# Main code specific to only the kruskal algorithm
if __name__ == "__main__":
    # Loop to test with different graph sizes (number of cities)
    for number_cities in range(3, 100):
        print("\n******************** Number of cities:", number_cities, "********************")
        adj = generate_random_matrix(num_cities=number_cities, symmetric=True)  # Generate an adjacency matrix
        cost, path, execution_time = measure_execution_time_mst_kruskal(adj)  # Measure execution time
        print("Approximate TSP Path (MST with Kruskal):", path)
        print("Number of cities:", number_cities)
        print("Minimal distance:", cost)
