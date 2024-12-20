import sys
import os
import tsplib95
import networkx as nx
from itertools import permutations
import matplotlib.pyplot as plt
import numpy as np
import random
import time

# Add the parent directory to PYTHONPATH (to allow direct execution of the script)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper.useful import generate_random_matrix, generate_erdos_renyi_graph, generate_watts_strogatz_graph

# Function to calculate the total distance of a given route
def calculate_total_distance(route, distances):
    """
    Calculate the total distance of a route using a distance matrix.

    :param route: List of city indices representing the route.
    :param distances: 2D distance matrix.
    :return: Total distance of the route.
    """
    total_distance = 0
    num_cities = len(route)
    for i in range(num_cities - 1):
        total_distance += distances[route[i]][route[i + 1]]
    total_distance += distances[route[-1]][route[0]]  # Return to the starting city
    return total_distance

# Brute force approach for solving the TSP
def tsp_brute_force(distances):
    """
    Solve the Traveling Salesman Problem (TSP) using brute force.

    :param distances: 2D distance matrix.
    :return: Best route and its minimum distance.
    """
    num_cities = len(distances)
    cities = list(range(num_cities))
    
    min_distance = float('inf')
    best_route = None
    
    # Generate all possible permutations of cities and calculate their distances
    for perm in permutations(cities):
        current_distance = calculate_total_distance(perm, distances)
        if current_distance < min_distance:
            min_distance = current_distance
            best_route = perm
    
    # Return the route including a return to the starting city
    return list(best_route) + [best_route[0]], min_distance

# Create a distance matrix from a TSPLIB problem instance
def create_distance_matrix(problem, max_nodes=8):
    """
    Create a distance matrix from a TSPLIB problem.

    :param problem: TSPLIB problem instance.
    :param max_nodes: Maximum number of nodes to include in the matrix.
    :return: Distance matrix and list of selected nodes.
    """
    nodes = list(problem.get_nodes())[:max_nodes]
    num_nodes = len(nodes)
    distances = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distances[i, j] = problem.get_weight(nodes[i], nodes[j])
    return distances, nodes

# Plot the graph of cities and their connections
def plot_graph(cities, distances=None, labels=True):
    """
    Visualize the graph of cities and their potential connections.

    :param cities: List of tuples representing city coordinates (x, y).
    :param distances: Distance matrix (optional).
    :param labels: Boolean indicating whether to display node labels.
    """
    G = nx.Graph()

    # Add nodes to the graph
    for i, city in enumerate(cities):
        G.add_node(i, pos=city)

    # Add edges to the graph if distances are provided
    if distances is not None and isinstance(distances, np.ndarray) and distances.size > 0:
        for i in range(len(cities)):
            for j in range(len(cities)):
                if i != j and distances[i, j] < float('inf'):
                    G.add_edge(i, j, weight=distances[i, j])

    # Get positions for the nodes and plot the graph
    pos = {i: city for i, city in enumerate(cities)}
    nx.draw(G, pos, with_labels=labels, node_color='lightblue', node_size=500, font_size=10, font_color='black')
    plt.title("City Graph")
    plt.show()

# Plot the solution of the TSP
def plot_solution(cities, best_route):
    """
    Visualize the optimal route found by the TSP algorithm.

    :param cities: List of tuples representing city coordinates (x, y).
    :param best_route: List of city indices representing the optimal route.
    """
    x = [cities[i][0] for i in best_route] + [cities[best_route[0]][0]]
    y = [cities[i][1] for i in best_route] + [cities[best_route[0]][1]]
    plt.plot(x, y, 'bo-', label="Optimal Route")
    plt.plot(x[0], y[0], 'go', label="Start/End Point")
    plt.title("TSP Solution (Brute Force)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to measure the execution time of the TSP approximation
def measure_execution_time_brute_force(distances):
    """
    Measure the execution time of the brute force TSP algorithm.

    :param distances: 2D distance matrix.
    :return: Minimum distance, best route, and execution time.
    """
    start_time = time.time()
    best_route, min_distance = tsp_brute_force(distances)
    end_time = time.time()
    execution_time = end_time - start_time
    return min_distance, best_route, execution_time

# Analyze the time complexity of the brute force algorithm
def analyze_complexity(max_cities=10):
    """
    Analyze the time complexity of the brute force TSP algorithm.

    :param max_cities: Maximum number of cities to analyze.
    """
    times = []
    city_counts = list(range(3, max_cities + 1))

    for num_cities in city_counts:
        distances = generate_random_matrix(num_cities, symmetric=True)
        _, _, execution_time = measure_execution_time_brute_force(distances)
        times.append(execution_time)
        print(f"{num_cities} cities: execution time = {execution_time:.4f} seconds")
    
    plt.figure(figsize=(10, 6))
    plt.plot(city_counts, times, 'o-', color='b')
    plt.xlabel("Number of Cities")
    plt.ylabel("Execution Time (s)")
    plt.title("Time Complexity of Brute Force TSP Algorithm")
    plt.grid(True)
    plt.show()

# Test case 1: Simple distance matrix
def test_simple_matrix():
    print("\n=== Test with a Simple Distance Matrix ===")
    distances = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    cities = [(i * 10, i * 5) for i in range(len(distances))]  # Fictitious city positions

    #plot_graph(cities, distances)
    best_route, min_distance = tsp_brute_force(distances)
    print("Best route:", best_route)
    print("Minimal distance:", min_distance)
    plot_solution(cities, best_route)
    analyze_complexity(max_cities=5)

# Test case 2: Randomly generated distance matrix
def test_random_matrix(num_cities=5):
    print(f"\n=== Test with a Random Distance Matrix ({num_cities} cities) ===")
    distances = generate_random_matrix(num_cities)
    cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_cities)]

    #plot_graph(cities, distances)
    best_route, min_distance = tsp_brute_force(distances)
    print("Best route:", best_route)
    print("Minimal distance:", min_distance)
    plot_solution(cities, best_route)

# Test case 3: TSPLIB instance
def test_tsplib_instance(tsp_file, max_nodes=8):
    print(f"\n=== Test with a TSPLIB Instance ({max_nodes} cities) ===")
    problem = tsplib95.load(tsp_file)
    distances, selected_nodes = create_distance_matrix(problem, max_nodes)
    cities = [problem.node_coords[node] for node in selected_nodes]

    #plot_graph(cities, distances)
    best_route, min_distance = tsp_brute_force(distances)
    print("Best route:", best_route)
    print("Minimal distance:", min_distance)
    plot_solution(cities, best_route)

### Main ###
if __name__ == "__main__":
    print("\n*************************************************************")
    print("*                  TSP BRUTE FORCE ANALYSIS                 *")
    print("*************************************************************")
    for number_cities in range(3, 10):
        print("\n-------------------- Number of cities:", number_cities, "--------------------")
        adj = generate_random_matrix(num_cities=number_cities, symmetric=True)
        cost, path, execution_time = measure_execution_time_brute_force(adj)
        print("TSP Path:", path)
        print("Minimal distance:", cost)

    # Other tests like :
    #test_simple_matrix();

    test_random_matrix();

    # Test 3 : Instance TSPLIB
    #tsp_file = "datasets/data/a280.tsp" # Or
    #tsp_file = "datasets/data/berlin52.tsp"
    #test_tsplib_instance(tsp_file, max_nodes=8)