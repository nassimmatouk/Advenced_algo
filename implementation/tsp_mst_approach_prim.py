import time  # To measure execution time
import numpy as np  # For matrix manipulation
from heapq import heappop, heappush  # To use a priority queue (min-heap)

from helper.useful import generate_random_matrix  # Generate a random adjacency matrix (defined elsewhere)

# Function to calculate the Minimum Spanning Tree (MST) using Prim's algorithm
def prim_mst(adj):
    N = len(adj)  # Number of nodes in the graph
    selected = [False] * N  # Track nodes already added to the MST
    mst_edges = []  # Store edges of the MST
    min_heap = [(0, 0, -1)]  # (cost, current node, parent node)

    while min_heap:
        cost, u, parent = heappop(min_heap)  # Extract the edge with minimum cost
        if selected[u]:
            continue  # Skip if the node is already in the MST
        selected[u] = True  # Mark the node as selected
        if parent != -1:
            mst_edges.append((parent, u, cost))  # Add edge to MST

        for v in range(N):  # Explore neighbors
            if not selected[v] and adj[u][v] != 0:  # Check unvisited neighbors
                heappush(min_heap, (adj[u][v], v, u))  # Add edge to the priority queue

    return mst_edges  # Return edges of the MST

# Function for preorder traversal of the MST (depth-first search)
def preorder_traversal(mst, start):
    from collections import defaultdict

    graph = defaultdict(list)  # Create a graph from MST edges
    for u, v, cost in mst:
        graph[u].append(v)
        graph[v].append(u)

    visited = set()  # Keep track of visited nodes
    path = []  # Store the path

    def dfs(node):
        visited.add(node)  # Mark node as visited
        path.append(node)  # Add node to the path
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)  # Recursively visit neighbors

    dfs(start)
    return path  # Return the path in preorder

# Function to approximate TSP using MST and preorder traversal
def tsp_mst_prim(adj):
    mst = prim_mst(adj)  # Calculate MST using Prim's algorithm
    path = preorder_traversal(mst, 0)  # Perform preorder traversal of MST
    path.append(path[0])  # Return to the starting point
    cost = sum(adj[path[i]][path[i + 1]] for i in range(len(path) - 1))  # Calculate total cost
    return cost, path  # Return total cost and the path

# Function to measure the execution time of the TSP approximation
def measure_execution_time_mst_prim(distances):
    start_time = time.time()  # Start the timer
    cost, path = tsp_mst_prim(distances)  # Run the algorithm
    end_time = time.time()  # Stop the timer
    execution_time = end_time - start_time  # Calculate elapsed time
    return cost, path, execution_time  # Return cost, path, and execution time

# Main code specific to only the prim algorithm
if __name__ == "__main__":
    for number_cities in range(3, 100):  # Test with graphs from 3 to 99 cities
        print("\n******************** Number of cities:", number_cities, "********************")
        adj = generate_random_matrix(num_cities=number_cities, symmetric=True)  # Generate a random graph
        cost, path, execution_time = measure_execution_time_mst_prim(adj)  # Run the algorithm and measure time
        print("Approximate TSP path (MST with Prim):", path)
        print("Number of cities:", number_cities)
        print("Minimum distance:", cost)
