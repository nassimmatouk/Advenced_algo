import sys
import os

import tsplib95
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import numpy as np
import random
import time
from itertools import permutations

# Add the parent directory to PYTHONPATH (to execute the file directly)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper.useful import generate_random_matrix, generate_erdos_renyi_graph, generate_watts_strogatz_graph

### Randomized Algorithm for TSP ###
def tsp_randomized(distances, iterations=1000):
    """
    Solves the Traveling Salesman Problem (TSP) using a randomized approach.
    :param distances: NxN distance matrix.
    :param iterations: Number of iterations to generate random routes.
    :return: Best route found and total distance.
    """
    num_cities = len(distances)
    best_route = None
    min_distance = float('inf')

    for _ in range(iterations):
        # Generate a random permutation of cities
        route = list(range(num_cities))
        random.shuffle(route)

        # Calculate the total distance for this permutation
        current_distance = sum(distances[route[i]][route[i + 1]] for i in range(num_cities - 1))
        current_distance += distances[route[-1]][route[0]]  # Return to the starting city

        # Update the best solution if necessary
        if current_distance < min_distance:
            min_distance = current_distance
            best_route = route

    # Add the starting city at the end to complete the loop
    if best_route:
        best_route.append(best_route[0])

    return best_route, min_distance


### Distance Matrix Creation from TSPLIB ###
def create_distance_matrix(problem, max_nodes=8):
    """
    Creates a distance matrix from a TSPLIB problem instance.
    :param problem: TSPLIB problem instance.
    :param max_nodes: Maximum number of cities (nodes) to consider.
    :return: Distance matrix and list of nodes.
    """
    nodes = list(problem.get_nodes())[:max_nodes]
    num_nodes = len(nodes)
    distances = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distances[i, j] = problem.get_weight(nodes[i], nodes[j])
    return distances, nodes


### Measure Execution Time ###
def measure_execution_time_randomized(distances):
    """
    Measures execution time of the randomized TSP algorithm.
    :param distances: NxN distance matrix.
    :return: Minimum distance, best route, and execution time.
    """
    start_time = time.time()
    best_route, min_distance = tsp_randomized(distances)  # Run TSP approximation
    end_time = time.time()
    execution_time = end_time - start_time
    return min_distance, best_route, execution_time


### Graph Plotting Functions ###
def plot_graph(cities, distances=None, labels=True):
    """
    Plots a graph of cities and their connections.
    :param cities: List of city positions (coordinates).
    :param distances: Optional NxN distance matrix to add edges.
    :param labels: Whether to display node labels.
    """
    G = nx.Graph()

    # Add nodes to the graph
    for i, city in enumerate(cities):
        G.add_node(i, pos=city)

    # Add edges to the graph (if distances are provided)
    if distances is not None and isinstance(distances, np.ndarray) and distances.size > 0:
        for i in range(len(cities)):
            for j in range(len(cities)):
                if i != j and distances[i, j] < float('inf'):
                    G.add_edge(i, j, weight=distances[i, j])

    # Extract positions for visualization
    pos = {i: city for i, city in enumerate(cities)}
    nx.draw(G, pos, with_labels=labels, node_color='lightblue', node_size=500, font_size=10, font_color='black')
    plt.title("City Graph")
    plt.show()


def plot_solution(cities, best_route):
    """
    Visualizes the solution to the TSP problem.
    :param cities: List of city positions (coordinates).
    :param best_route: List of cities representing the best route.
    """
    x = [cities[i][0] for i in best_route] + [cities[best_route[0]][0]]
    y = [cities[i][1] for i in best_route] + [cities[best_route[0]][1]]
    plt.plot(x, y, 'bo-', label="Optimal Path")
    plt.plot(x[0], y[0], 'go', label="Start/End")
    plt.title("TSP Solution (Randomized)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()


### Complexity Analysis for TSP Randomized ###
def analyze_complexity_randomized(max_cities=10, iterations=1000):
    """
    Analyzes the time complexity of the randomized TSP algorithm.
    :param max_cities: Maximum number of cities for testing.
    :param iterations: Number of random iterations per test.
    """
    times = []
    city_counts = list(range(3, max_cities + 1))

    for num_cities in city_counts:
        # Generate a random distance matrix
        distances = generate_random_matrix(num_cities, symmetric=True)

        # Measure execution time of the randomized algorithm
        start_time = time.time()
        tsp_randomized(distances, iterations=iterations)
        end_time = time.time()

        execution_time = end_time - start_time
        times.append(execution_time)

        print(f"{num_cities} cities: execution time = {execution_time:.6f} seconds")

    # Visualize complexity
    plt.plot(city_counts, times, 'o-', color='purple', label="TSP Randomized")
    plt.xlabel("Number of Cities")
    plt.ylabel("Execution Time (s)")
    plt.title("Time Complexity of TSP Algorithm (Randomized)")
    plt.grid()
    plt.legend()
    plt.show()


### Test Cases for Randomized TSP ###
def test_simple_matrix_randomized():
    """
    Test case: TSP with a simple predefined distance matrix.
    """
    print("\n=== Test with a Simple Distance Matrix (Randomized) ===")
    distances = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    cities = [(i * 10, i * 5) for i in range(len(distances))]  # Dummy city positions

    best_route, min_distance = tsp_randomized(distances, iterations=1000)
    print("Best Route (Randomized):", best_route)
    print("Minimum Distance (Randomized):", min_distance)
    plot_solution(cities, best_route)

    # Analyze complexity
    print("\n=== Complexity Analysis for Simple Matrix ===")
    analyze_complexity_randomized(max_cities=5, iterations=1000)


def test_random_matrix_randomized(num_cities=5, iterations=1000):
    """
    Test case: TSP with a randomly generated distance matrix.
    """
    print(f"\n=== Test with a Random Distance Matrix ({num_cities} Cities) (Randomized) ===")
    distances = generate_random_matrix(num_cities)
    cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_cities)]

    best_route, min_distance = tsp_randomized(distances, iterations=iterations)
    print("Best Route (Randomized):", best_route)
    print("Minimum Distance (Randomized):", min_distance)
    plot_solution(cities, best_route)

    print("\n=== Complexity Analysis for Random Matrix ===")
    analyze_complexity_randomized(max_cities=num_cities, iterations=iterations)


def test_tsplib_instance_randomized(tsp_file, max_nodes=8, iterations=1000):
    """
    Test case: TSP with a TSPLIB instance.
    :param tsp_file: Path to the TSPLIB file.
    :param max_nodes: Maximum number of cities to consider.
    :param iterations: Number of iterations for randomized search.
    """
    print(f"\n=== Test with TSPLIB Instance ({max_nodes} Cities) (Randomized) ===")
    problem = tsplib95.load(tsp_file)
    distances, selected_nodes = create_distance_matrix(problem, max_nodes)

    cities = [problem.node_coords[node] for node in selected_nodes]

    best_route, min_distance = tsp_randomized(distances, iterations=iterations)
    print("Best Route (Randomized):", best_route)
    print("Minimum Distance (Randomized):", min_distance)
    plot_solution(cities, best_route)

    print("\n=== Complexity Analysis for TSPLIB Instance ===")
    analyze_complexity_randomized(max_cities=max_nodes, iterations=iterations)


### Main Execution ###
if __name__ == "__main__":
    # Test 1: Simple distance matrix
    #test_simple_matrix_randomized()

    # Test 2: Randomly generated distance matrix
    test_random_matrix_randomized(num_cities=100, iterations=5000)

    # Test 3: TSPLIB instance
    #tsp_file = "datasets/data/a280.tsp" # Or
    #tsp_file = "datasets/data/berlin52.tsp"
    #test_tsplib_instance_randomized(tsp_file, max_nodes=100, iterations=5000)
