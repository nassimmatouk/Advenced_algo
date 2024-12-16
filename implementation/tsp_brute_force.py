import sys
import os

import tsplib95
import networkx as nx
from itertools import permutations
import matplotlib.pyplot as plt
import numpy as np
import random
import time

# Ajouter le répertoire parent au PYTHONPATH (pour pouvoir exécuté directement le fichier)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper.useful import generate_random_matrix, generate_erdos_renyi_graph, generate_watts_strogatz_graph

def calculate_total_distance(route, distances):
    total_distance = 0
    num_cities = len(route)
    for i in range(num_cities - 1):
        total_distance += distances[route[i]][route[i + 1]]
    total_distance += distances[route[-1]][route[0]]  # Retour à la ville de départ
    return total_distance

def tsp_brute_force(distances):
    num_cities = len(distances)
    cities = list(range(num_cities))
    
    min_distance = float('inf')
    best_route = None
    
    for perm in permutations(cities):
        current_distance = calculate_total_distance(perm, distances)
        if current_distance < min_distance:
            min_distance = current_distance
            best_route = perm
    
    #return best_route, min_distance
    return list(best_route) + [best_route[0]], min_distance

# Création d'une matrice de distances depuis TSPLIB
def create_distance_matrix(problem, max_nodes=8):
    nodes = list(problem.get_nodes())[:max_nodes]
    num_nodes = len(nodes)
    distances = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distances[i, j] = problem.get_weight(nodes[i], nodes[j])
    return distances, nodes

def plot_graph(cities, distances=None, labels=True):
    G = nx.Graph()

    # Ajout des nœuds au graphe
    for i, city in enumerate(cities):
        G.add_node(i, pos=city)

    # Ajout des arêtes au graphe (si les distances sont fournies)
    if distances is not None and isinstance(distances, np.ndarray) and distances.size > 0:
        for i in range(len(cities)):
            for j in range(len(cities)):
                if i != j and distances[i, j] < float('inf'):
                    G.add_edge(i, j, weight=distances[i, j])

    # Récupération des positions pour l'affichage
    pos = {i: city for i, city in enumerate(cities)}
    nx.draw(G, pos, with_labels=labels, node_color='lightblue', node_size=500, font_size=10, font_color='black')
    plt.title("Graphe des villes")
    plt.show()


# Visualisation de la solution du TSP
def plot_solution(cities, best_route):
    x = [cities[i][0] for i in best_route] + [cities[best_route[0]][0]]
    y = [cities[i][1] for i in best_route] + [cities[best_route[0]][1]]
    plt.plot(x, y, 'bo-', label="Chemin optimal")
    plt.plot(x[0], y[0], 'go', label="Départ/Arrivée")
    plt.title("Solution TSP (Brute Force)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

# Function to measure execution time of the TSP approximation
def measure_execution_time(distances):
    start_time = time.time()
    best_route, min_distance = tsp_brute_force(distances)  # Run TSP approximation
    end_time = time.time()
    execution_time = end_time - start_time
    return min_distance, best_route, execution_time

def measure_execution_time_brute_force(distances):
    start_time = time.time()
    best_route, min_distance = tsp_brute_force(distances)  # Run TSP approximation
    end_time = time.time()
    execution_time = end_time - start_time
    return min_distance, best_route, execution_time

def analyze_complexity(max_cities=10):
    times = []
    city_counts = list(range(3, max_cities + 1))

    for num_cities in city_counts:
        distances = generate_random_matrix(num_cities, symmetric=True)
        
        execution_time = measure_execution_time(distances)
        times.append(execution_time)
        print(f"{num_cities} villes: temps d'exécution = {execution_time:.4f} secondes")
    
    plt.figure(figsize=(10, 6))
    plt.plot(city_counts, times, 'o-', color='b')
    plt.xlabel("Nombre de villes")
    plt.ylabel("Temps d'exécution (s)")
    plt.title("Complexité en temps de l'algorithme TSP (force brute)")
    plt.show()
    
# Cas 1 : Matrice de distances simple
def test_simple_matrix():
    print("\n=== Test avec une matrice simple ===")
    distances = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    cities = [(i * 10, i * 5) for i in range(len(distances))]  # Positions fictives des villes

    plot_graph(cities, distances)
    best_route, min_distance = tsp_brute_force(distances)
    print("Meilleur chemin :", best_route)
    print("Distance minimale :", min_distance)
    plot_solution(cities, best_route)
    analyze_complexity(max_cities=5)


# Cas 2 : Matrice de distances générée aléatoirement
def test_random_matrix(num_cities=5):
    print(f"\n=== Test avec une matrice aléatoire ({num_cities} villes) ===")
    distances = generate_random_matrix(num_cities)
    cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_cities)]  # Générer des positions aléatoires

    plot_graph(cities, distances)
    best_route, min_distance = tsp_brute_force(distances)
    print("Meilleur chemin :", best_route)
    print("Distance minimale :", min_distance)
    plot_solution(cities, best_route)
    analyze_complexity(max_cities=num_cities)


# Cas 3 : Instance TSPLIB
def test_tsplib_instance(tsp_file, max_nodes=8):
    print(f"\n=== Test avec une instance TSPLIB ({max_nodes} villes) ===")
    problem = tsplib95.load(tsp_file)
    distances, selected_nodes = create_distance_matrix(problem, max_nodes)

    # Extraire les coordonnées des villes
    cities = [problem.node_coords[node] for node in selected_nodes]
    plot_graph(cities, distances)

    best_route, min_distance = tsp_brute_force(distances)
    print("Meilleur chemin :", best_route)
    print("Distance minimale :", min_distance)
    plot_solution(cities, best_route)
    analyze_complexity(max_cities=max_nodes)


### Main ###
if __name__ == "__main__":
    # Test 1 : Matrice simple
    #test_simple_matrix()
    
    # Test 2 : Matrice aléatoire
    #test_random_matrix(num_cities=6)

    # Test 3 : Instance TSPLIB
    #tsp_file = "C:/Users/userlocal/Desktop/M1/AA/theProject/berlin52.tsp"
    #test_tsplib_instance(tsp_file, max_nodes=8)
    print("\n*************************************************************")
    print("*                      TPS BRUTE FORCE                      *")
    print("*************************************************************")
    for number_cities in range(3, 10):
        print("\n-------------------- Number of cities:", number_cities, "--------------------")
        adj = generate_random_matrix(num_cities=number_cities, symmetric=True)  # Generate an adjacency matrix
        cost, path, execution_time = measure_execution_time_brute_force(adj)  # Measure execution time
        print("Approximate TSP Path :", path)
        #print("Number of cities:", number_cities)
        print("Minimal distance:", cost)

