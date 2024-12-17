import sys
import os
import tsplib95
import networkx as nx
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from heapq import heappop, heappush

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

# Function to calculate the lower bound for a given partial solution
def calculate_lower_bound(partial_route, distances, visited):
    """
    Calculate the lower bound for the current partial solution using a minimum spanning tree (MST) heuristic.

    :param partial_route: List of cities visited so far.
    :param distances: 2D distance matrix.
    :param visited: List indicating which cities have been visited.
    :return: Lower bound value.
    """
    num_cities = len(distances)
    unvisited = [i for i in range(num_cities) if not visited[i]]
    
    # If only one city is left to visit, return the distance to return to the starting city
    if len(unvisited) == 1:
        return distances[partial_route[-1]][unvisited[0]] + distances[unvisited[0]][partial_route[0]]
    
    # Compute the MST lower bound
    mst_cost = 0
    for i in unvisited:
        min_edge = float('inf')
        for j in unvisited:
            if i != j:
                min_edge = min(min_edge, distances[i][j])
        mst_cost += min_edge
    
    return mst_cost

# Branch and Bound approach for solving the TSP
def tsp_branch_and_bound(distances):
    """
    Solve the Traveling Salesman Problem (TSP) using Branch and Bound.

    :param distances: 2D distance matrix.
    :return: Best route and its minimum distance.
    """
    num_cities = len(distances)
    
    # Priority queue for exploring the partial routes
    pq = []
    
    # Initial route: start with city 0
    initial_route = [0]
    visited = [False] * num_cities
    visited[0] = True
    
    # Calculate lower bound for the initial route (MST heuristic)
    lower_bound = calculate_lower_bound(initial_route, distances, visited)
    
    # Push the initial state into the priority queue
    heappush(pq, (lower_bound, initial_route, visited))
    
    min_distance = float('inf')
    best_route = None
    
    while pq:
        # Get the route with the smallest lower bound
        bound, partial_route, visited = heappop(pq)
        
        # If the partial route is a complete tour, check its total distance
        if len(partial_route) == num_cities:
            current_distance = calculate_total_distance(partial_route, distances)
            if current_distance < min_distance:
                min_distance = current_distance
                best_route = partial_route
        else:
            # Branch: explore all unvisited cities
            for i in range(num_cities):
                if not visited[i]:
                    new_route = partial_route + [i]
                    new_visited = visited[:]
                    new_visited[i] = True
                    
                    # Calculate the new lower bound
                    new_bound = calculate_lower_bound(new_route, distances, new_visited)
                    
                    # Only push the new route if its lower bound is promising
                    if new_bound < min_distance:
                        heappush(pq, (new_bound, new_route, new_visited))
    
    # Return the route including a return to the starting city
    return best_route + [best_route[0]], min_distance

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
    plt.title("TSP Solution (Branch and Bound)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to measure the execution time of the Branch and Bound algorithm
def measure_execution_time_bnb(distances):
    """
    Measure the execution time of the Branch and Bound TSP algorithm.

    :param distances: 2D distance matrix.
    :return: Minimum distance, best route, and execution time.
    """
    start_time = time.time()
    best_route, min_distance = tsp_branch_and_bound(distances)
    end_time = time.time()
    execution_time = end_time - start_time
    return min_distance, best_route, execution_time

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
    best_route, min_distance = tsp_branch_and_bound(distances)
    print("Best route:", best_route)
    print("Minimal distance:", min_distance)
    plot_solution(cities, best_route)

# Test case 2: Randomly generated distance matrix
def test_random_matrix(num_cities=5):
    print(f"\n=== Test with a Random Distance Matrix ({num_cities} cities) ===")
    distances = generate_random_matrix(num_cities)
    cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_cities)]

    #plot_graph(cities, distances)
    best_route, min_distance = tsp_branch_and_bound(distances)
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
    best_route, min_distance = tsp_branch_and_bound(distances)
    print("Best route:", best_route)
    print("Minimal distance:", min_distance)
    plot_solution(cities, best_route)

### Main ###
if __name__ == "__main__":
    print("\n*************************************************************")
    print("*                  TSP BRANCH AND BOUND ANALYSIS            *")
    print("*************************************************************")
    for number_cities in range(3, 10):
        print("\n-------------------- Number of cities:", number_cities, "--------------------")
        adj = generate_random_matrix(num_cities=number_cities, symmetric=True)
        cost, path, execution_time = measure_execution_time_bnb(adj)
        print("TSP Path:", path)
        print("Minimal distance:", cost)

    # Other tests like :
    #test_simple_matrix();
    test_random_matrix();
    # Test 3 : Instance TSPLIB
    #tsp_file = "datasets/data/a280.tsp" # Or
    #tsp_file = "datasets/data/berlin52.tsp"
    #test_tsplib_instance(tsp_file, max_nodes=8)
