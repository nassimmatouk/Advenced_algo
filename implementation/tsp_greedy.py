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

# Ajouter le répertoire parent au PYTHONPATH (pour pouvoir exécuté directement le fichier)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helper.useful import generate_random_matrix, generate_erdos_renyi_graph, generate_watts_strogatz_graph

### Algorithme glouton pour le TSP ###
def tsp_greedy(distances):
    """
    Résout le problème TSP en utilisant une approche gloutonne.
    :param distances: Matrice de distances NxN.
    :return: Chemin trouvé et distance totale.
    """
    num_cities = len(distances)
    visited = [False] * num_cities
    path = []
    total_distance = 0

    # Commencer par la première ville
    current_city = 0
    path.append(current_city)
    visited[current_city] = True

    # Répéter jusqu'à ce que toutes les villes soient visitées
    for _ in range(num_cities - 1):
        nearest_city = None
        min_distance = float('inf')

        # Trouver la ville la plus proche non visitée
        for city in range(num_cities):
            if not visited[city] and distances[current_city][city] < min_distance:
                nearest_city = city
                min_distance = distances[current_city][city]

        # Ajouter la ville la plus proche au chemin
        path.append(nearest_city)
        visited[nearest_city] = True
        total_distance += min_distance
        current_city = nearest_city

    # Retourner à la ville de départ
    total_distance += distances[current_city][path[0]]
    path.append(path[0])

    return path, total_distance

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

def measure_execution_time_greedy(distances):
    start_time = time.time()
    best_route, min_distance = tsp_greedy(distances)  # Run TSP approximation
    end_time = time.time()
    execution_time = end_time - start_time
    return min_distance, best_route, execution_time

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

# Analyse et visualisation de la complexité pour TSP Greedy
def analyze_complexity_greedy(max_cities=10):
    """
    Analyse la complexité en temps de l'algorithme glouton TSP.
    :param max_cities: Nombre maximum de villes pour les tests.
    """
    times = []
    city_counts = list(range(3, max_cities + 1))

    for num_cities in city_counts:
        # Générer une matrice de distances aléatoire
        distances = generate_random_matrix(num_cities, symmetric=True)

        # Mesurer le temps d'exécution de l'algorithme glouton
        start_time = time.time()
        tsp_greedy(distances)
        end_time = time.time()

        execution_time = end_time - start_time
        times.append(execution_time)

        print(f"{num_cities} villes : temps d'exécution = {execution_time:.6f} secondes")

    # Visualiser la complexité
    plt.plot(city_counts, times, 'o-', color='g', label="TSP Greedy")
    plt.xlabel("Nombre de villes")
    plt.ylabel("Temps d'exécution (s)")
    plt.title("Complexité en temps de l'algorithme TSP (Greedy)")
    plt.grid()
    plt.legend()
    plt.show()

### Tests par cas pour l'approche gloutonne ###

# Cas 1 : Matrice de distances simple
def test_simple_matrix_greedy():
    print("\n=== Test avec une matrice simple (Greedy) ===")
    distances = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    cities = [(i * 10, i * 5) for i in range(len(distances))]  # Positions fictives des villes

    plot_graph(cities, distances)
    best_route, min_distance = tsp_greedy(distances)
    print("Meilleur chemin (Greedy) :", best_route)
    print("Distance minimale (Greedy) :", min_distance)
    plot_solution(cities, best_route)

    # Analyse de la complexité pour un maximum de 5 villes
    print("\n=== Analyse de la complexité pour une matrice simple ===")
    analyze_complexity_greedy(max_cities=5)


# Cas 2 : Matrice de distances générée aléatoirement
def test_random_matrix_greedy(num_cities=5):
    print(f"\n=== Test avec une matrice aléatoire ({num_cities} villes) (Greedy) ===")
    distances = generate_random_matrix(num_cities)
    cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_cities)]  # Générer des positions aléatoires

    plot_graph(cities, distances)
    best_route, min_distance = tsp_greedy(distances)
    print("Meilleur chemin (Greedy) :", best_route)
    print("Distance minimale (Greedy) :", min_distance)
    plot_solution(cities, best_route)

    # Analyse de la complexité pour le nombre de villes spécifié
    print("\n=== Analyse de la complexité pour une matrice aléatoire ===")
    analyze_complexity_greedy(max_cities=num_cities)


# Cas 3 : Instance TSPLIB
def test_tsplib_instance_greedy(tsp_file, max_nodes=8):
    print(f"\n=== Test avec une instance TSPLIB ({max_nodes} villes) (Greedy) ===")
    problem = tsplib95.load(tsp_file)
    distances, selected_nodes = create_distance_matrix(problem, max_nodes)

    # Extraire les coordonnées des villes
    cities = [problem.node_coords[node] for node in selected_nodes]
    plot_graph(cities, distances)

    best_route, min_distance = tsp_greedy(distances)
    print("Meilleur chemin (Greedy) :", best_route)
    print("Distance minimale (Greedy) :", min_distance)
    plot_solution(cities, best_route)

    # Analyse de la complexité pour les nœuds sélectionnés
    print("\n=== Analyse de la complexité pour une instance TSPLIB ===")
    analyze_complexity_greedy(max_cities=max_nodes)


### Main avec analyse de la complexité ###
if __name__ == "__main__":
    # Test 1 : Matrice simple
    test_simple_matrix_greedy()

    # Test 2 : Matrice aléatoire
    # test_random_matrix_greedy(num_cities=6)

    # Test 3 : Instance TSPLIB
    # tsp_file = "C:/Users/userlocal/Desktop/M1/AA/theProject/berlin52.tsp"
    # test_tsplib_instance_greedy(tsp_file, max_nodes=8)
