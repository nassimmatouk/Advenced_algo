import sys
import os

import tsplib95
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from itertools import permutations

# Add the parent directory to PYTHONPATH (to execute the file directly)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper.useful import generate_random_matrix, generate_erdos_renyi_graph, generate_watts_strogatz_graph

### Greedy Algorithm for TSP ###
def tsp_greedy(distances):
    """
    Solves the Traveling Salesman Problem (TSP) using a greedy approach.
    :param distances: NxN distance matrix.
    :return: The found path and the total distance.
    """
    num_cities = len(distances)
    visited = [False] * num_cities  # Tracks whether a city has been visited
    path = []  # Stores the final path
    total_distance = 0

    # Start with the first city
    current_city = 0
    path.append(current_city)
    visited[current_city] = True

    # Repeat until all cities are visited
    for _ in range(num_cities - 1):
        nearest_city = None
        min_distance = float('inf')

        # Find the nearest unvisited city
        for city in range(num_cities):
            if not visited[city] and distances[current_city][city] < min_distance:
                nearest_city = city
                min_distance = distances[current_city][city]

        # Move to the nearest city
        path.append(nearest_city)
        visited[nearest_city] = True
        total_distance += min_distance
        current_city = nearest_city

    # Return to the starting city to complete the circuit
    total_distance += distances[current_city][path[0]]
    path.append(path[0])

    return path, total_distance

# Create a distance matrix from a TSPLIB problem
def create_distance_matrix(problem, max_nodes=8):
    """
    Generates a distance matrix from a TSPLIB problem instance.
    :param problem: A TSPLIB problem instance.
    :param max_nodes: Maximum number of nodes to consider.
    :return: Distance matrix and the selected nodes.
    """
    nodes = list(problem.get_nodes())[:max_nodes]
    num_nodes = len(nodes)
    distances = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distances[i, j] = problem.get_weight(nodes[i], nodes[j])
    return distances, nodes

# Measure the execution time of the greedy TSP algorithm
def measure_execution_time_greedy(distances):
    start_time = time.time()
    best_route, min_distance = tsp_greedy(distances)
    end_time = time.time()
    execution_time = end_time - start_time
    return min_distance, best_route, execution_time

# Plot the graph with cities and optional distances
def plot_graph(cities, distances=None, labels=True):
    """
    Plots the graph of cities and edges.
    :param cities: List of city coordinates.
    :param distances: NxN distance matrix (optional).
    :param labels: Whether to display labels on nodes.
    """
    G = nx.Graph()

    # Add nodes (cities) to the graph
    for i, city in enumerate(cities):
        G.add_node(i, pos=city)

    # Add edges if distances are provided
    if distances is not None and isinstance(distances, np.ndarray) and distances.size > 0:
        for i in range(len(cities)):
            for j in range(len(cities)):
                if i != j and distances[i, j] < float('inf'):
                    G.add_edge(i, j, weight=distances[i, j])

    # Retrieve node positions for display
    pos = {i: city for i, city in enumerate(cities)}
    nx.draw(G, pos, with_labels=labels, node_color='lightblue', node_size=500, font_size=10, font_color='black')
    plt.title("City Graph")
    plt.show()

# Visualize the TSP solution path
def plot_solution(cities, best_route):
    """
    Plots the TSP solution path.
    :param cities: List of city coordinates.
    :param best_route: Ordered list of cities in the solution.
    """
    x = [cities[i][0] for i in best_route] + [cities[best_route[0]][0]]
    y = [cities[i][1] for i in best_route] + [cities[best_route[0]][1]]
    plt.plot(x, y, 'bo-', label="Optimal Path")
    plt.plot(x[0], y[0], 'go', label="Start/End")
    plt.title("TSP Solution (Greedy Algorithm)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

# Analyze and visualize the time complexity for the Greedy TSP algorithm
def analyze_complexity_greedy(max_cities=10):
    """
    Analyzes the time complexity of the greedy TSP algorithm.
    :param max_cities: Maximum number of cities to test.
    """
    times = []
    city_counts = list(range(3, max_cities + 1))

    for num_cities in city_counts:
        # Generate a random symmetric distance matrix
        distances = generate_random_matrix(num_cities, symmetric=True)

        # Measure execution time of the greedy algorithm
        start_time = time.time()
        tsp_greedy(distances)
        end_time = time.time()

        execution_time = end_time - start_time
        times.append(execution_time)

        print(f"{num_cities} cities: execution time = {execution_time:.6f} seconds")

    # Visualize the time complexity
    plt.plot(city_counts, times, 'o-', color='g', label="Greedy TSP")
    plt.xlabel("Number of Cities")
    plt.ylabel("Execution Time (s)")
    plt.title("Time Complexity of Greedy TSP Algorithm")
    plt.grid()
    plt.legend()
    plt.show()

### Test cases for the Greedy TSP approach ###

# Test Case 1: Simple Distance Matrix
def test_simple_matrix_greedy():
    print("\n=== Test with a Simple Distance Matrix (Greedy) ===")
    distances = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    cities = [(i * 10, i * 5) for i in range(len(distances))]  # Mock city positions

    plot_graph(cities, distances)
    best_route, min_distance = tsp_greedy(distances)
    print("Best Route (Greedy):", best_route)
    print("Minimal Distance (Greedy):", min_distance)
    plot_solution(cities, best_route)

    print("\n=== Complexity Analysis for a Simple Matrix ===")
    analyze_complexity_greedy(max_cities=5)

# Test Case 2: Randomly Generated Distance Matrix
def test_random_matrix_greedy(num_cities=5):
    print(f"\n=== Test with a Random Distance Matrix ({num_cities} cities) (Greedy) ===")
    distances = generate_random_matrix(num_cities)
    cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_cities)]

    plot_graph(cities, distances)
    best_route, min_distance = tsp_greedy(distances)
    print("Best Route (Greedy):", best_route)
    print("Minimal Distance (Greedy):", min_distance)
    plot_solution(cities, best_route)

    print("\n=== Complexity Analysis for a Random Matrix ===")
    analyze_complexity_greedy(max_cities=num_cities)

# Test Case 3: TSPLIB Instance
def test_tsplib_instance_greedy(tsp_file, max_nodes=8):
    print(f"\n=== Test with a TSPLIB Instance ({max_nodes} cities) (Greedy) ===")
    problem = tsplib95.load(tsp_file)
    distances, selected_nodes = create_distance_matrix(problem, max_nodes)

    cities = [problem.node_coords[node] for node in selected_nodes]
    plot_graph(cities, distances)

    best_route, min_distance = tsp_greedy(distances)
    print("Best Route (Greedy):", best_route)
    print("Minimal Distance (Greedy):", min_distance)
    plot_solution(cities, best_route)

    print("\n=== Complexity Analysis for a TSPLIB Instance ===")
    analyze_complexity_greedy(max_cities=max_nodes)

### Main Section ###
if __name__ == "__main__":
    # Test 1: Simple Matrix
    #test_simple_matrix_greedy()

    # Test 2: Random Matrix
    test_random_matrix_greedy(num_cities=150)

    # Test 3: TSPLIB Instance
    #tsp_file = "datasets/data/a280.tsp" # Or
    #tsp_file = "datasets/data/berlin52.tsp"
    #test_tsplib_instance_greedy(tsp_file, max_nodes=100)
